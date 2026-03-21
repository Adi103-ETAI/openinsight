SYSTEM_PROMPT = """You are an AI clinical decision support assistant for Indian physicians.
Follow these rules strictly:
1) Answer only using the provided context.
2) Always cite supporting statements using bracketed source indices like [1], [2].
3) Use ICMR guidelines as the primary authority when they are present in context.
4) Clearly mention whether key recommendations are India-specific or general clinical evidence.
5) Be concise, clinically actionable, and structured for point-of-care use.
6) Do not invent, infer, or hallucinate facts that are not in the context.
7) If the query is outside available knowledge, clearly state the limitation at the end."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown Source")
        source_type = chunk.get("source_type", "unknown")
        score = float(chunk.get("score", 0.0))
        chunk_text = chunk.get("chunk_text", "").strip()
        context_blocks.append(
            f"[{idx}] Source: {title} ({source_type}, relevance: {score:.2f})\n"
            f"\"{chunk_text}\""
        )

    context_text = "\n\n".join(context_blocks) if context_blocks else "No context available."
    return (
        "Clinical context passages:\n\n"
        f"{context_text}\n\n"
        f"Doctor's question:\n{query}\n\n"
        "Provide a concise, clinically actionable answer with citations."
    )
