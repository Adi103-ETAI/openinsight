"""tools — flat registry exposing every tool function under one namespace."""
from __future__ import annotations

# Filesystem tools
from src.tools.filesystemtools import (
    write_text, write_json, write_bytes,
    read_text, read_json, read_bytes,
    append_to_file, replace_in_file, insert_at_line,
    delete_file, delete_directory, cleanup_temp_files,
    list_files, list_by_extension, get_file_size, get_file_info,
    make_dir, make_temp_dir, make_reports_dir,
    hash_string, hash_file, cache_key,
    truncate, truncate_to_tokens_approx,
    save_chunk, load_chunk,
)

# Web search tools
from src.tools.websearchtools import (
    extract_domain, is_same_domain,
    extract_snippet, extract_text_blocks,
    is_medical_domain, filter_medical, list_medical_domains,
    rank_by_keywords, top_n,
    group_by_domain, count_per_domain,
    deduplicate_by_url, deduplicate_by_title,
)

# Citation tools
from src.tools.citationtools import (
    extract_chunk_ids, extract_web_ids,
    extract_all_citations, extract_citation_markers,
    claim_supported_by_source, is_supported,
    build_citation_schema, find_best_source,
)

# Document tools
from src.tools.doctools import (
    generate_pdf, generate_docx, generate_filename,
    get_pdf_metadata, split_sections, build_doc_sections,
    format_citations_inline, count_citations,
)

# Registry: name -> function (for dynamic lookups by string name)
TOOL_REGISTRY: dict[str, callable] = {
    # filesystem
    "write_text": write_text, "write_json": write_json, "write_bytes": write_bytes,
    "read_text": read_text, "read_json": read_json, "read_bytes": read_bytes,
    "append_to_file": append_to_file, "replace_in_file": replace_in_file, "insert_at_line": insert_at_line,
    "delete_file": delete_file, "delete_directory": delete_directory, "cleanup_temp_files": cleanup_temp_files,
    "list_files": list_files, "list_by_extension": list_by_extension,
    "get_file_size": get_file_size, "get_file_info": get_file_info,
    "make_dir": make_dir, "make_temp_dir": make_temp_dir, "make_reports_dir": make_reports_dir,
    "hash_string": hash_string, "hash_file": hash_file, "cache_key": cache_key,
    "truncate": truncate, "truncate_to_tokens_approx": truncate_to_tokens_approx,
    "save_chunk": save_chunk, "load_chunk": load_chunk,
    # websearch
    "extract_domain": extract_domain, "is_same_domain": is_same_domain,
    "extract_snippet": extract_snippet, "extract_text_blocks": extract_text_blocks,
    "is_medical_domain": is_medical_domain, "filter_medical": filter_medical,
    "list_medical_domains": list_medical_domains,
    "rank_by_keywords": rank_by_keywords, "top_n": top_n,
    "group_by_domain": group_by_domain, "count_per_domain": count_per_domain,
    "deduplicate_by_url": deduplicate_by_url, "deduplicate_by_title": deduplicate_by_title,
    # citation
    "extract_chunk_ids": extract_chunk_ids, "extract_web_ids": extract_web_ids,
    "extract_all_citations": extract_all_citations, "extract_citation_markers": extract_citation_markers,
    "claim_supported_by_source": claim_supported_by_source, "is_supported": is_supported,
    "build_citation_schema": build_citation_schema, "find_best_source": find_best_source,
    # doc
    "generate_pdf": generate_pdf, "generate_docx": generate_docx,
    "generate_filename": generate_filename, "get_pdf_metadata": get_pdf_metadata,
    "split_sections": split_sections, "build_doc_sections": build_doc_sections,
    "format_citations_inline": format_citations_inline, "count_citations": count_citations,
}


def get_tool(name: str):
    """Look up a tool function by name. Raises KeyError if unknown."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Unknown tool: {name}")
    return fn


def list_tools() -> list[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(TOOL_REGISTRY.keys())


__all__ = [
    "TOOL_REGISTRY", "get_tool", "list_tools",
    # All tool functions re-exported for convenience
    "write_text", "write_json", "write_bytes",
    "read_text", "read_json", "read_bytes",
    "append_to_file", "replace_in_file", "insert_at_line",
    "delete_file", "delete_directory", "cleanup_temp_files",
    "list_files", "list_by_extension", "get_file_size", "get_file_info",
    "make_dir", "make_temp_dir", "make_reports_dir",
    "hash_string", "hash_file", "cache_key",
    "truncate", "truncate_to_tokens_approx",
    "save_chunk", "load_chunk",
    "extract_domain", "is_same_domain",
    "extract_snippet", "extract_text_blocks",
    "is_medical_domain", "filter_medical", "list_medical_domains",
    "rank_by_keywords", "top_n",
    "group_by_domain", "count_per_domain",
    "deduplicate_by_url", "deduplicate_by_title",
    "extract_chunk_ids", "extract_web_ids",
    "extract_all_citations", "extract_citation_markers",
    "claim_supported_by_source", "is_supported",
    "build_citation_schema", "find_best_source",
    "generate_pdf", "generate_docx", "generate_filename",
    "get_pdf_metadata", "split_sections", "build_doc_sections",
    "format_citations_inline", "count_citations",
]
