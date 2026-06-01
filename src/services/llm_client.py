"""
LLM Client — Backward-compatible wrapper around the dynamic provider system.

All code that previously used `get_nim_client()` continues to work.
New code should use `src.services.llm.registry.get_llm_client()` directly.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from src.services.llm.base import BaseLLMClient

# Cached default client
_default_client: Optional[BaseLLMClient] = None


def get_nim_client() -> BaseLLMClient:
    """
    Get the default LLM client (backward-compatible).

    Returns the configured default provider client.
    Old code calling `get_nim_client()` gets the same behavior,
    now backed by the dynamic provider system.
    """
    global _default_client
    if _default_client is None:
        from src.services.llm.registry import get_llm_client
        _default_client = get_llm_client()
    return _default_client
