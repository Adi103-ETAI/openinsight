"""
Query Rewriter
Rewrites raw doctor queries into expanded medical queries before vector search.
Uses a fast NIM call with a dedicated rewrite prompt.
"""

from loguru import logger
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pathlib import Path

from src.core.config import get_settings

settings = get_settings()


def _load_rewrite_prompt() -> str:
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return (prompts_dir / "query_rewrite.md").read_text(encoding="utf-8")


REWRITE_PROMPT = _load_rewrite_prompt()


def _is_bad_rewrite(candidate: str) -> bool:
    lowered = candidate.lower().strip()
    if not lowered:
        return True
    meta_phrases = [
        "what is the query",
        "please provide",
        "could you provide",
        "i can help",
        "normalise",
        "normalize",
    ]
    if any(p in lowered for p in meta_phrases):
        return True
    if lowered.endswith("?"):
        return True
    return False


async def rewrite_query(query: str) -> str:
    """
    Rewrite a raw doctor query into an expanded medical query.
    Returns the original query unchanged if rewriting fails.
    """
    if len(query.strip()) < 4:
        return query

    try:
        client = AsyncOpenAI(
            api_key=settings.nvidia_nim_api_key,
            base_url=settings.nvidia_nim_base_url,
        )
        response = await client.chat.completions.create(
            model=settings.nim_model,
            messages=[
                {"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": query.strip()},
            ],
            temperature=0.0,
            max_tokens=64,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten and len(rewritten) > 3 and not _is_bad_rewrite(rewritten):
            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        return query
    except (
        APIConnectionError,
        APITimeoutError,
        RateLimitError,
        RuntimeError,
        ValueError,
    ) as exc:
        logger.warning(f"Query rewrite failed, using original: {exc}")
        return query
