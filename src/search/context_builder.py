from __future__ import annotations

from src.search.retriever import RetrievedChunk


EVIDENCE_LEVEL_LABELS = {
    "1a": "Systematic Review / Meta-Analysis",
    "1b": "Randomised Controlled Trial",
    "2a": "Systematic Review of Cohort Studies",
    "2b": "Cohort Study",
    "3": "Case-Control Study",
    "4": "Case Series",
    "5": "Expert Opinion / Guideline",
    "unknown": "Not Classified",
}


def assemble_context(chunks: list[RetrievedChunk], max_tokens: int = 3000) -> str:
    parts: list[str] = []
    total_chars = 0
    char_limit = max_tokens * 4

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata or {}
        title = metadata.get("title", "Untitled")
        year = metadata.get("year", "")
        journal = metadata.get("journal", "")
        evidence_level = str(metadata.get("evidence_level", "unknown")).lower()
        doc_type = str(metadata.get("doc_type", "unknown")).replace("_", " ").title()
        india_flag = (
            " 🇮🇳 India-relevant" if bool(metadata.get("india_relevant", False)) else ""
        )

        block = (
            f"[{i}] {title} ({year}, {journal})\n"
            f"Evidence: {EVIDENCE_LEVEL_LABELS.get(evidence_level, 'Not Classified')} | {doc_type}{india_flag}\n"
            f"{chunk.text}"
        )

        if total_chars + len(block) > char_limit:
            break

        parts.append(block)
        total_chars += len(block)

    return "\n---\n".join(parts)


def build_citation_list(chunks: list[RetrievedChunk]) -> list[dict]:
    citations: list[dict] = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.metadata or {}
        pmid = metadata.get("pmid") or ""
        citations.append(
            {
                "number": i,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", []),
                "year": metadata.get("year", 0),
                "journal": metadata.get("journal", ""),
                "doi": metadata.get("doi", ""),
                "pmid": pmid,
                "evidence_level": metadata.get("evidence_level", "unknown"),
                "doc_type": metadata.get("doc_type", ""),
                "india_relevant": bool(metadata.get("india_relevant", False)),
                "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}" if pmid else "",
            }
        )

    return citations
