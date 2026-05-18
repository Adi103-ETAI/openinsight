from __future__ import annotations

from typing import Optional

import httpx
from loguru import logger

from src.config.settings import get_settings


class NVIDIAClient:
    """
    Direct NVIDIA NIM client using httpx.
    Replaces OpenAI SDK for NVIDIA endpoints.
    """

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.nvidia_nim_base_url.rstrip("/")
        self.api_key = settings.nvidia_nim_api_key
        self.model = settings.nim_model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Call NVIDIA NIM chat completions API."""
        if not self.api_key:
            raise ValueError("NVIDIA_NIM_API_KEY not configured")

        client = await self._get_client()
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def completions(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """Call NVIDIA NIM completions API (non-chat)."""
        if not self.api_key:
            raise ValueError("NVIDIA_NIM_API_KEY not configured")

        client = await self._get_client()
        
        url = f"{self.base_url}/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        
        data = response.json()
        return data.get("choices", [{}])[0].get("text", "")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# Global singleton
_nim_client: Optional[NVIDIAClient] = None


def get_nim_client() -> NVIDIAClient:
    global _nim_client
    if _nim_client is None:
        _nim_client = NVIDIAClient()
    return _nim_client