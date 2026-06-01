"""
Abstract base class for LLM providers.
All providers implement the same interface so callers are provider-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Full response from an LLM provider."""
    content: str
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    latency_ms: float = 0.0


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider must implement chat_completions() and completions().
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g., 'nvidia', 'openai')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model (e.g., 'gpt-4o')."""
        ...

    @abstractmethod
    async def chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """Chat completions API. Returns assistant response text."""
        ...

    @abstractmethod
    async def completions(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        """Completions API (non-chat). Returns generated text."""
        ...

    async def chat_completions_detailed(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Chat completions with full response details."""
        import time
        start = time.monotonic()
        content = await self.chat_completions(messages, temperature, max_tokens, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000
        return LLMResponse(content=content, model=self.model_name, provider=self.provider_name, latency_ms=latency_ms)

    async def close(self) -> None:
        """Clean up resources."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name} model={self.model_name}>"
