"""
LLM Router — Load-balanced routing for multi-agent systems.

Routes different agents/sub-queries to different LLM providers based on:
- Provider health and availability
- Cost optimization (cheaper models for simple tasks)
- Load distribution (spread requests across providers)
- Agent-specific preferences (e.g., complex reasoning → Claude, quick answers → Llama)
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from src.config.settings import get_settings
from src.services.llm.base import BaseLLMClient, LLMResponse
from src.services.llm.registry import create_llm_client, list_providers


class AgentRole(str, Enum):
    """Roles agents can play in the DeepInsights pipeline."""
    DECOMPOSER = "decomposer"        # Breaks down complex queries
    RETRIEVER = "retriever"          # Generates search queries
    SYNTHESIZER = "synthesizer"      # Combines sub-query answers
    VALIDATOR = "validator"          # Checks answer quality
    CITATION = "citation"            # Extracts/generates citations
    GENERAL = "general"              # Default fallback


@dataclass
class ProviderHealth:
    """Tracks health state of a provider."""
    provider: str
    last_error: str = ""
    error_count: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    avg_latency_ms: float = 0.0
    total_requests: int = 0
    _latencies: list[float] = field(default_factory=list)

    def record_success(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.error_count = 0
        self.last_success = time.monotonic()
        self._latencies.append(latency_ms)
        # Keep last 20 latencies
        if len(self._latencies) > 20:
            self._latencies = self._latencies[-20:]
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def record_failure(self, error: str) -> None:
        self.total_requests += 1
        self.error_count += 1
        self.last_failure = time.monotonic()
        self.last_error = error

    @property
    def is_healthy(self) -> bool:
        """Provider is considered unhealthy after 3 consecutive failures."""
        return self.error_count < 3

    @property
    def cooldown_remaining(self) -> float:
        """Seconds until provider exits cooldown after failures."""
        if self.error_count < 3:
            return 0.0
        # Exponential backoff: 30s, 60s, 120s...
        cooldown = min(30 * (2 ** (self.error_count - 3)), 300)
        elapsed = time.monotonic() - self.last_failure
        return max(0.0, cooldown - elapsed)


@dataclass
class ProviderPool:
    """A pool of LLM clients with health tracking."""
    clients: dict[str, BaseLLMClient] = field(default_factory=dict)
    health: dict[str, ProviderHealth] = field(default_factory=dict)

    def get_healthy_providers(self) -> list[str]:
        """Return providers that are healthy and not in cooldown."""
        return [
            name for name, h in self.health.items()
            if h.is_healthy and h.cooldown_remaining == 0
        ]


class LLMRouter:
    """
    Load-balanced LLM router for multi-agent systems.

    Routes requests to different providers based on:
    1. Agent role → provider mapping (configurable)
    2. Provider health (avoid failing providers)
    3. Round-robin load balancing across healthy providers
    4. Fallback to any healthy provider on failure

    Usage:
        router = LLMRouter()
        client = router.get_client_for_agent(AgentRole.SYNTHESIZER)
        response = await client.chat_completions(messages=[...])
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._pool = ProviderPool()
        self._round_robin_index = 0

        # Default agent → provider mapping
        # Override via settings or set_agent_provider()
        self._agent_providers: dict[str, str] = {}

        # Initialize configured providers
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize all configured LLM providers from providers.json."""
        available = list_providers()
        for name, info in available.items():
            if not info["configured"]:
                logger.debug(f"LLM provider {name} not configured (missing API key), skipping")
                continue
            try:
                client = create_llm_client(name)
                self._pool.clients[name] = client
                self._pool.health[name] = ProviderHealth(provider=name)
                logger.info(f"LLM provider registered: {name} (model={client.model_name})")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM provider {name}: {e}")

        if not self._pool.clients:
            # Fallback: try default provider
            try:
                default = self.settings.llm_default_provider
                client = create_llm_client(default)
                self._pool.clients[default] = client
                self._pool.health[default] = ProviderHealth(provider=default)
                logger.info(f"LLM provider registered (default): {default}")
            except Exception as e:
                logger.error(f"Failed to initialize any LLM provider: {e}")

    def set_agent_provider(self, agent_role: str, provider: str) -> None:
        """Map an agent role to a specific provider."""
        self._agent_providers[agent_role] = provider

    def get_client_for_agent(self, role: AgentRole | str = AgentRole.GENERAL) -> BaseLLMClient:
        """
        Get an LLM client for a specific agent role.

        Routing priority:
        1. Explicit agent → provider mapping
        2. Config-based role mapping from settings
        3. Round-robin across healthy providers
        """
        role_str = role.value if isinstance(role, AgentRole) else role

        # 1. Check explicit mapping
        if role_str in self._agent_providers:
            provider = self._agent_providers[role_str]
            if provider in self._pool.clients and self._pool.health[provider].is_healthy:
                return self._pool.clients[provider]
            logger.warning(f"Mapped provider {provider} unhealthy for role {role_str}, falling back")

        # 2. Check settings-based mapping
        provider = self._get_provider_from_settings(role_str)
        if provider and provider in self._pool.clients and self._pool.health[provider].is_healthy:
            return self._pool.clients[provider]

        # 3. Round-robin across healthy providers
        return self._get_round_robin_client()

    def _get_provider_from_settings(self, role: str) -> str | None:
        """Check settings for role-specific provider mapping."""
        mapping = {
            "decompose": self.settings.llm_decompose_provider,
            "synthesize": self.settings.llm_synthesize_provider,
            "validate": self.settings.llm_validate_provider,
            "rewrite": self.settings.llm_rewrite_provider,
        }
        return mapping.get(role)

    def _get_round_robin_client(self) -> BaseLLMClient:
        """Round-robin across healthy providers."""
        healthy = self._pool.get_healthy_providers()
        if not healthy:
            # All providers unhealthy — try all and let the request fail naturally
            healthy = list(self._pool.clients.keys())
            if not healthy:
                raise RuntimeError("No LLM providers available")
            logger.warning("All LLM providers unhealthy, attempting anyway")

        provider = healthy[self._round_robin_index % len(healthy)]
        self._round_robin_index += 1
        return self._pool.clients[provider]

    def record_success(self, provider: str, latency_ms: float) -> None:
        """Record a successful request for health tracking."""
        if provider in self._pool.health:
            self._pool.health[provider].record_success(latency_ms)

    def record_failure(self, provider: str, error: str) -> None:
        """Record a failed request for health tracking."""
        if provider in self._pool.health:
            self._pool.health[provider].record_failure(error)

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all providers."""
        return {
            name: {
                "healthy": h.is_healthy,
                "error_count": h.error_count,
                "avg_latency_ms": round(h.avg_latency_ms, 1),
                "total_requests": h.total_requests,
                "last_error": h.last_error[:100] if h.last_error else None,
                "cooldown_seconds": round(h.cooldown_remaining, 1),
            }
            for name, h in self._pool.health.items()
        }

    def get_provider_count(self) -> int:
        """Return number of registered providers."""
        return len(self._pool.clients)

    async def close(self) -> None:
        """Close all provider connections."""
        for client in self._pool.clients.values():
            try:
                await client.close()
            except Exception:
                pass
        self._pool.clients.clear()
        self._pool.health.clear()


# ── Singleton ────────────────────────────────────────────────────────────────

_router: LLMRouter | None = None


def get_llm_router() -> LLMRouter:
    """Get or create the global LLM router singleton."""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
