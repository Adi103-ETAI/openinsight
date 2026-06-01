"""
Generic OpenAI-compatible LLM client.
Works with any provider that implements the OpenAI API format:
  NVIDIA, OpenAI, Together, OpenRouter, Groq, AIML, and more.

Config-driven from providers.json — no hardcoded provider lists.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from loguru import logger

from src.services.llm.base import BaseLLMClient

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
    _HAS_TENACITY = True
except ImportError:
    _HAS_TENACITY = False


def _retry(func):
    if not _HAS_TENACITY:
        return func
    return retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )(func)


class OpenAICompatibleClient(BaseLLMClient):
    """
    Generic client for any OpenAI-compatible API.

    Config example (from providers.json):
    {
        "name": "together",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "auth_type": "bearer",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    }
    """

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str = "",
        model: str = "",
        auth_type: str = "bearer",
        timeout: float = 60.0,
    ):
        self._provider_name = provider_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._auth_type = auth_type
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model_name(self) -> str:
        return self._model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _build_auth_headers(self) -> dict[str, str]:
        """Build auth headers based on auth_type."""
        if self._auth_type == "bearer":
            return {"Authorization": f"Bearer {self._api_key}"}
        elif self._auth_type == "x-api-key":
            return {"x-api-key": self._api_key}
        elif self._auth_type == "header":
            return {"api-key": self._api_key}
        return {}

    @_retry
    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if not self._api_key:
            raise ValueError(f"{self._provider_name}: API key not configured")

        client = await self._get_client()
        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json", **self._build_auth_headers()}

        body = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(kwargs)

        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    @_retry
    async def completions(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if not self._api_key:
            raise ValueError(f"{self._provider_name}: API key not configured")

        client = await self._get_client()
        url = f"{self._base_url}/completions"
        headers = {"Content-Type": "application/json", **self._build_auth_headers()}

        body = {
            "model": self._model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(kwargs)

        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("text", "")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
