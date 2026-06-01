"""
Ollama LLM provider.
Uses Ollama's local API for self-hosted models (Llama, Mistral, etc.).
"""
from __future__ import annotations

from typing import Any, Optional

import httpx
from loguru import logger

from src.services.llm.base import BaseLLMClient

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
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


class OllamaClient(BaseLLMClient):
    """Ollama local API client."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1:70b",
        timeout: float = 120.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    @_retry
    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        client = await self._get_client()
        url = f"{self._base_url}/api/chat"

        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")

    @_retry
    async def completions(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        client = await self._get_client()
        url = f"{self._base_url}/api/generate"

        body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
