"""
OpenAI LLM provider.
Uses OpenAI's API for GPT-4, GPT-4o, GPT-3.5, etc.
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


class OpenAIClient(BaseLLMClient):
    """OpenAI API client (GPT-4, GPT-4o, GPT-3.5, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "openai"

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
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY not configured")

        client = await self._get_client()
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
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
            raise ValueError("OPENAI_API_KEY not configured")

        client = await self._get_client()
        url = f"{self._base_url}/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
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
