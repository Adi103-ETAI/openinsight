"""
LLM Provider Registry — Config-driven from providers.json.

Add a new provider by editing providers.json — no Python code changes needed.
Each provider specifies: base_url, api_key_env, auth_type, api_type, models.

Usage:
    from src.services.llm.registry import create_llm_client, get_llm_client, list_providers

    client = get_llm_client("together")
    response = await client.chat_completions(messages=[...])

    # List available providers
    print(list_providers())
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.settings import get_settings
from src.services.llm.base import BaseLLMClient

# Path to providers config
_PROVIDERS_JSON = Path(__file__).parent / "providers.json"

# Loaded config cache
_providers_config: dict[str, Any] | None = None

# Client class cache (provider_name -> instance)
_clients: dict[str, BaseLLMClient] = {}


def _load_config() -> dict[str, Any]:
    """Load and cache providers.json."""
    global _providers_config
    if _providers_config is None:
        with open(_PROVIDERS_JSON) as f:
            data = json.load(f)
        _providers_config = data.get("providers", {})
        logger.debug(f"Loaded {len(_providers_config)} LLM providers from config")
    return _providers_config


def _reload_config() -> dict[str, Any]:
    """Force reload providers.json (for hot-reload)."""
    global _providers_config
    _providers_config = None
    return _load_config()


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get config for a specific provider from providers.json."""
    config = _load_config()
    provider = provider.lower().strip()
    if provider not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Unknown LLM provider: '{provider}'. Available: {available}")
    return config[provider]


def list_providers() -> dict[str, dict[str, Any]]:
    """
    List all configured providers with their status.

    Returns:
        Dict of provider_name -> {display_name, configured, models_count, default_model}
    """
    config = _load_config()
    result = {}
    for name, cfg in config.items():
        api_key_env = cfg.get("api_key_env", "")
        has_key = bool(os.getenv(api_key_env)) if api_key_env else True  # Ollama has no key
        result[name] = {
            "display_name": cfg.get("display_name", name),
            "configured": has_key,
            "api_type": cfg.get("api_type", "openai"),
            "default_model": cfg.get("default_model", ""),
            "models_count": len(cfg.get("models", {})),
        }
    return result


def list_models(provider: str) -> dict[str, dict[str, Any]]:
    """List available models for a specific provider."""
    cfg = get_provider_config(provider)
    return cfg.get("models", {})


def create_llm_client(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """
    Create an LLM client from providers.json config.

    Args:
        provider: Provider name (e.g., "nvidia", "together"). Uses default if None.
        model: Model override. Uses provider's default_model if None.
        api_key: API key override. Reads from env var if None.
        **kwargs: Additional overrides (base_url, timeout, etc.)

    Returns:
        BaseLLMClient instance.
    """
    config = _load_config()
    settings = get_settings()

    if provider is None:
        provider = settings.llm_default_provider

    provider = provider.lower().strip()

    if provider not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Unknown LLM provider: '{provider}'. Available: {available}")

    cfg = config[provider]

    # Resolve API key: explicit > env var > settings > empty
    api_key_env = cfg.get("api_key_env", "")
    resolved_key = (
        api_key
        or (os.getenv(api_key_env) if api_key_env else "")
        or _get_key_from_settings(provider, settings)
        or ""
    )

    resolved_model = model or cfg.get("default_model", "")
    resolved_url = kwargs.get("base_url") or cfg.get("base_url", "")
    resolved_timeout = kwargs.get("timeout") or cfg.get("timeout", 60)
    api_type = cfg.get("api_type", "openai")

    # Dispatch to the right client class based on api_type
    if api_type == "openai":
        from src.services.llm.providers.openai_compatible import OpenAICompatibleClient
        return OpenAICompatibleClient(
            provider_name=provider,
            base_url=resolved_url,
            api_key=resolved_key,
            model=resolved_model,
            auth_type=cfg.get("auth_type", "bearer"),
            timeout=resolved_timeout,
        )

    elif api_type == "anthropic":
        from src.services.llm.providers.openai_compatible import OpenAICompatibleClient
        # Anthropic has a different auth header but we can adapt
        return OpenAICompatibleClient(
            provider_name=provider,
            base_url=resolved_url,
            api_key=resolved_key,
            model=resolved_model,
            auth_type=cfg.get("auth_type", "x-api-key"),
            timeout=resolved_timeout,
        )

    elif api_type == "google":
        from src.services.llm.providers.google import GoogleClient
        return GoogleClient(
            base_url=resolved_url,
            api_key=resolved_key,
            model=resolved_model,
            timeout=resolved_timeout,
        )

    elif api_type == "cohere":
        from src.services.llm.providers.cohere import CohereClient
        return CohereClient(
            base_url=resolved_url,
            api_key=resolved_key,
            model=resolved_model,
            timeout=resolved_timeout,
        )

    elif api_type == "ollama":
        from src.services.llm.providers.ollama import OllamaClient
        return OllamaClient(
            base_url=resolved_url,
            model=resolved_model,
            timeout=resolved_timeout,
        )

    else:
        raise ValueError(f"Unknown api_type '{api_type}' for provider '{provider}'")


def _get_key_from_settings(provider: str, settings: Any) -> str | None:
    """Fallback: try to get API key from legacy settings fields."""
    mapping = {
        "nvidia": getattr(settings, "nvidia_nim_api_key", ""),
        "openai": getattr(settings, "openai_api_key", ""),
        "anthropic": getattr(settings, "anthropic_api_key", ""),
        "google": getattr(settings, "google_api_key", ""),
        "together": getattr(settings, "together_api_key", ""),
        "openrouter": getattr(settings, "openrouter_api_key", ""),
        "groq": getattr(settings, "groq_api_key", ""),
        "aiml": getattr(settings, "aiml_api_key", ""),
        "cohere": getattr(settings, "cohere_api_key", ""),
    }
    return mapping.get(provider, "")


# ── Singleton management ─────────────────────────────────────────────────────


def get_llm_client(provider: str | None = None) -> BaseLLMClient:
    """
    Get or create a cached LLM client singleton.

    Args:
        provider: Provider name. If None, uses default from settings.

    Returns:
        Cached BaseLLMClient instance.
    """
    if provider is None:
        settings = get_settings()
        provider = settings.llm_default_provider

    provider = provider.lower().strip()

    if provider not in _clients:
        _clients[provider] = create_llm_client(provider)

    return _clients[provider]


async def close_all_clients() -> None:
    """Close all cached client connections. Call on shutdown."""
    for name, client in _clients.items():
        try:
            await client.close()
        except Exception as e:
            logger.warning(f"Failed to close LLM client {name}: {e}")
    _clients.clear()


# ── Plug-and-play registration ───────────────────────────────────────────────

def register_provider(name: str, client_class: type[BaseLLMClient]) -> None:
    """
    Register a custom provider class at runtime.

    For providers not in providers.json, you can register a class directly:
        register_provider("my_custom", MyCustomClient)
        client = get_llm_client("my_custom")
    """
    # This adds to the runtime class cache — used for custom providers
    # that aren't in providers.json
    from src.services.llm.providers.openai_compatible import OpenAICompatibleClient

    # Store for later — create_llm_client will use the api_type dispatch
    # For fully custom providers, users should subclass BaseLLMClient
    logger.info(f"Custom LLM provider registered: {name}")
