"""truncate_text — limit text length with smart ellipsis."""
from __future__ import annotations


def truncate(text: str, max_len: int = 1000, suffix: str = "...") -> str:
    """Truncate text to max_len, breaking on a word boundary if possible."""
    if len(text) <= max_len:
        return text
    if max_len <= len(suffix):
        return text[:max_len]
    truncated = text[: max_len - len(suffix)]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return truncated + suffix


def truncate_to_tokens_approx(text: str, max_tokens: int = 500) -> str:
    """Rough token-aware truncation: ~4 chars per token."""
    return truncate(text, max_len=max_tokens * 4)
