from pathlib import Path


def _load_prompt(filename: str) -> str:
    """Load a prompt from the prompts/ directory at project root."""
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    return (prompts_dir / filename).read_text(encoding="utf-8")


SYSTEM_PROMPT = _load_prompt("system.md")


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown Source")
        source_type = chunk.get("source_type", "unknown")
        score = float(chunk.get("score", 0.0))
        chunk_text = chunk.get("chunk_text", "").strip()
        context_blocks.append(
            f"[{idx}] Source: {title} ({source_type}, relevance: {score:.2f})\n"
            f'"{chunk_text}"'
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."
    return (
        "Clinical context passages:\n\n"
        f"{context_text}\n\n"
        f"Doctor's question:\n{query}\n\n"
        "Provide a concise, clinically actionable answer with citations."
    )
