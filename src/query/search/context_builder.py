from __future__ import annotations

from .retriever import RetrievedChunk, RetrievedParentChunk


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


def assemble_context_with_parent(
    child_chunks: list[RetrievedChunk],
    parent_chunks: list[RetrievedParentChunk],
    max_tokens: int = 4000,
    include_parent_context: bool = True,
) -> str:
    """
    Assemble context from both child and parent chunks for enhanced RAG.
    
    This function creates a context that includes:
    1. Parent chunks with full section context (for comprehensive understanding)
    2. Child chunks with precise matches (for specific details)
    
    The child chunks are included as references within their parent context,
    allowing the LLM to have both broad and detailed information.
    
    Args:
        child_chunks: Retrieved child chunks for precise retrieval
        parent_chunks: Retrieved parent chunks with full section context
        max_tokens: Maximum tokens for the assembled context
        include_parent_context: Whether to include parent context (True) or just children (False)
        
    Returns:
        Assembled context string with source citations
    """
    if not include_parent_context:
        # Fall back to standard child-only context
        return assemble_context(child_chunks, max_tokens)
    
    parts: list[str] = []
    total_chars = 0
    char_limit = max_tokens * 4  # Approximate chars per token
    
    # Create a mapping of parent chunks by their child references
    parent_by_child_id: dict[str, RetrievedParentChunk] = {}
    for parent in parent_chunks:
        for child in parent.child_chunks:
            parent_by_child_id[child.chunk_id] = parent
    
    # Process parent chunks first (they have full context)
    for i, parent in enumerate(parent_chunks, 1):
        metadata = parent.metadata or {}
        title = metadata.get("title", "Untitled")
        year = metadata.get("year", "")
        journal = metadata.get("journal", "")
        evidence_level = str(metadata.get("evidence_level", "unknown")).lower()
        doc_type = str(metadata.get("doc_type", "unknown")).replace("_", " ").title()
        section_title = metadata.get("section_title", "Unknown Section")
        india_flag = (
            " 🇮🇳 India-relevant" if bool(metadata.get("india_relevant", False)) else ""
        )
        
        # Include child chunk references if available
        child_references = ""
        if parent.child_chunks:
            child_refs = [f"[Child-{j+1}]" for j in range(len(parent.child_chunks))]
            child_references = f" (Matched sections: {', '.join(child_refs)})"
        
        block = (
            f"[Parent-{i}] {title} ({year}, {journal})\n"
            f"Section: {section_title}\n"
            f"Evidence: {EVIDENCE_LEVEL_LABELS.get(evidence_level, 'Not Classified')} | {doc_type}{india_flag}{child_references}\n"
            f"{parent.text}"
        )
        
        if total_chars + len(block) > char_limit:
            break
            
        parts.append(block)
        total_chars += len(block)
    
    # If we have room, also include some direct child chunks that weren't fully covered
    if total_chars < char_limit * 0.8:
        remaining_chars = char_limit - total_chars
        uncovered_children = [
            c for c in child_chunks 
            if c.chunk_id not in parent_by_child_id
        ]
        
        if uncovered_children:
            parts.append("\n--- Additional precise matches ---\n")
            for j, child in enumerate(uncovered_children[:5], 1):  # Limit additional children
                metadata = child.metadata or {}
                title = metadata.get("title", "Untitled")
                year = metadata.get("year", "")
                section_title = metadata.get("section_title", "")
                
                block = (
                    f"[Child-{j}] {title} ({year}) - {section_title}\n"
                    f"{child.text}"
                )
                
                if total_chars + len(block) > remaining_chars:
                    break
                    
                parts.append(block)
                total_chars += len(block)
    
    return "\n---\n".join(parts)


def build_citation_list_with_parent(
    child_chunks: list[RetrievedChunk],
    parent_chunks: list[RetrievedParentChunk],
) -> list[dict]:
    """
    Build citation list including both child and parent chunks.
    
    Args:
        child_chunks: Retrieved child chunks
        parent_chunks: Retrieved parent chunks
        
    Returns:
        List of citation dictionaries
    """
    citations: list[dict] = []
    seen_doc_ids: set[str] = set()
    
    # First add parent chunks (they have full context)
    for i, parent in enumerate(parent_chunks, 1):
        metadata = parent.metadata or {}
        pmid = metadata.get("pmid") or ""
        doc_id = parent.doc_id
        
        # Avoid duplicate citations for the same document
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        
        citations.append(
            {
                "number": i,
                "doc_id": doc_id,
                "chunk_id": parent.chunk_id,
                "chunk_type": "parent",
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
                "section_title": metadata.get("section_title", ""),
                "child_chunk_count": len(parent.child_chunks),
            }
        )
    
    # Then add uncovered child chunks
    parent_doc_ids = {p.doc_id for p in parent_chunks}
    current_num = len(citations) + 1
    
    for child in child_chunks:
        if child.doc_id in parent_doc_ids:
            continue  # Already cited via parent
            
        if child.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(child.doc_id)
        
        metadata = child.metadata or {}
        pmid = metadata.get("pmid") or ""
        
        citations.append(
            {
                "number": current_num,
                "doc_id": child.doc_id,
                "chunk_id": child.chunk_id,
                "chunk_type": "child",
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
                "section_title": metadata.get("section_title", ""),
                "child_chunk_count": 1,
            }
        )
        current_num += 1
    
    return citations
