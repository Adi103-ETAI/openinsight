"""
Cohere LLM provider.
Uses Cohere's v2 Chat API.
"""
from __future__ import annotations

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


class CohereClient(BaseLLMClient):
    """
    Cohere API client (v2 Chat).

    Config example (from providers.json):
    {
        "name": "cohere",
        "base_url": "https://api.cohere.com/v2",
        "api_key_env": "COHERE_API_KEY",
        "api_type": "cohere",
        "default_model": "command-r-plus"
    }
    """

    def __init__(
        self,
        base_url: str = "https://api.cohere.com/v2",
        api_key: str = "",
        model: str = "command-r-plus",
        timeout: float = 60.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    @property
    def provider_name(self) -> str:
        return "cohere"

    @property
    def model_name(self) -> str:
        return self._model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[str, list[dict]]:
        """Convert OpenAI-style messages to Cohere v2 format."""
        system_message = ""
        chat_history = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                chat_history.append({"role": "user", "message": msg["content"]})
            elif msg["role"] == "assistant":
                chat_history.append({"role": "chatbot", "message": msg["content"]})

        return system_message, chat_history

    @_retry
    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if not self._api_key:
            raise ValueError("COHERE_API_KEY not configured")

        client = await self._get_client()
        system_message, chat_history = self._convert_messages(messages)

        url = f"{self._base_url}/chat"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # The last user message becomes the 'message' field
        message = chat_history.pop(-1)["message"] if chat_history else ""

        body: dict[str, Any] = {
            "model": self._model,
            "message": message,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if chat_history:
            body["chat_history"] = chat_history
        if system_message:
            body["preamble"] = system_message

        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

        # Extract text from Cohere v2 response
        message_block = data.get("message", {})
        content = message_block.get("content", [])
        return "".join(block.get("text", "") for block in content if block.get("type") == "text")

    @_retry
    async def completions(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """Cohere doesn't have a separate completions endpoint; use chat."""
        return await self.chat_completions(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
