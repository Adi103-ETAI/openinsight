"""tools — flat registry exposing every tool function under one namespace.

Each registered tool carries metadata so callers can detect whether it is a
coroutine without having to call it. Use `get_tool(name)` to fetch the
function, or `get_tool_meta(name)` to get the full metadata dict.
"""
from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Union

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


# Each entry: (callable, short_description)
_TOOL_ENTRIES: list[tuple[Callable, str]] = [
    # filesystem (async)
    (write_text, "Write text content to a file in the temp dir."),
    (write_json, "Write JSON-serializable data to a file."),
    (write_bytes, "Write raw bytes to a file with a given extension."),
    (read_text, "Read text content from a file. Returns None on failure."),
    (read_json, "Read and parse JSON from a file. Returns None on failure."),
    (read_bytes, "Read raw bytes from a file. Returns None on failure."),
    (append_to_file, "Append text to the end of a file."),
    (replace_in_file, "Replace all occurrences of a substring in a file."),
    (insert_at_line, "Insert content at a specific 1-indexed line number."),
    (delete_file, "Delete a single file (restricted to allowed roots)."),
    (delete_directory, "Recursively delete a directory (restricted to allowed roots)."),
    (cleanup_temp_files, "Remove temp files older than max_age_hours."),
    (list_files, "List files in a directory matching a glob pattern."),
    (list_by_extension, "List files with a specific extension."),
    (get_file_size, "Return file size in bytes (0 if not found)."),
    (get_file_info, "Return dict with size/mtime/ctime/is_file/is_dir."),
    (make_dir, "Create a directory (and parents) under a parent path."),
    (make_temp_dir, "Create a subdirectory under the standard temp dir."),
    (make_reports_dir, "Create the standard reports output directory."),
    (save_chunk, "Save a chunk dict as a JSON file."),
    (load_chunk, "Load a chunk dict from a JSON file."),
    # filesystem (sync)
    (hash_string, "Return hex digest of a string."),
    (hash_file, "Return hex digest of a file."),
    (cache_key, "Generate a short cache key from a query string."),
    (truncate, "Truncate text to max_len, breaking on word boundary."),
    (truncate_to_tokens_approx, "Rough token-aware truncation (~4 chars/token)."),
    # websearch (all sync)
    (extract_domain, "Parse the netloc from a URL."),
    (is_same_domain, "True if both URLs share the same domain."),
    (extract_snippet, "Strip HTML tags and create a plain-text snippet."),
    (extract_text_blocks, "Return plain text blocks from HTML."),
    (is_medical_domain, "True if URL is from a known medical authority."),
    (filter_medical, "Filter results to keep only medical domains."),
    (list_medical_domains, "List the trusted medical domains."),
    (rank_by_keywords, "Rank results by keyword overlap with title/snippet."),
    (top_n, "Return the first n results."),
    (group_by_domain, "Bucket results by domain."),
    (count_per_domain, "Count results per domain."),
    (deduplicate_by_url, "Remove duplicate URLs (keeps first)."),
    (deduplicate_by_title, "Remove duplicate case-insensitive titles."),
    # citation (all sync)
    (extract_chunk_ids, "Return CHUNK citation IDs found in text."),
    (extract_web_ids, "Return WEB citation IDs found in text."),
    (extract_all_citations, "Return dict with chunk + web citation IDs."),
    (extract_citation_markers, "Return ordered citation marker dicts."),
    (claim_supported_by_source, "Score overlap between claim and source."),
    (is_supported, "Boolean wrapper around claim_supported_by_source."),
    (build_citation_schema, "Build UI-friendly citation schema."),
    (find_best_source, "Find the source that best supports a claim."),
    # doc (all sync)
    (generate_pdf, "Render sections into a PDF file (reportlab)."),
    (generate_docx, "Render sections into a DOCX file (python-docx)."),
    (generate_filename, "Generate a safe timestamped filename."),
    (get_pdf_metadata, "Read metadata from a PDF file (PyPDF2)."),
    (split_sections, "Split a clinical answer into labeled sections."),
    (build_doc_sections, "Build a complete sections dict for doc rendering."),
    (format_citations_inline, "Normalize citation marker formatting."),
    (count_citations, "Count CHUNK and WEB citations in text."),
]


def _is_coroutine(fn: Callable) -> bool:
    """True if `fn` is a coroutine function (defined with `async def`)."""
    return inspect.iscoroutinefunction(fn)


def _make_entry(fn: Callable, desc: str) -> dict:
    return {
        "fn": fn,
        "async": _is_coroutine(fn),
        "desc": desc,
        "name": fn.__name__,
    }


# Build the metadata registry: name -> {fn, async, desc, name}
TOOL_REGISTRY: dict[str, dict] = {
    entry["name"]: entry for entry in (_make_entry(fn, desc) for fn, desc in _TOOL_ENTRIES)
}


def get_tool(name: str) -> Callable:
    """Look up a tool function by name. Raises KeyError if unknown."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Unknown tool: {name}")
    return entry["fn"]


def get_tool_meta(name: str) -> dict:
    """Return the full metadata dict for a tool: {fn, async, desc, name}."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Unknown tool: {name}")
    return dict(entry)


def is_async_tool(name: str) -> bool:
    """Return True if the tool is a coroutine function (must be awaited)."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise KeyError(f"Unknown tool: {name}")
    return entry["async"]


async def call_tool(name: str, *args, **kwargs) -> Any:
    """
    Call a tool by name, awaiting the result if it's async.
    Returns whatever the tool returns. Raises KeyError if unknown.
    """
    fn = get_tool(name)
    if is_async_tool(name):
        return await fn(*args, **kwargs)
    return fn(*args, **kwargs)


def list_tools() -> list[str]:
    """Return a sorted list of all registered tool names."""
    return sorted(TOOL_REGISTRY.keys())


def list_async_tools() -> list[str]:
    """Return a sorted list of all async (coroutine) tool names."""
    return sorted(n for n, e in TOOL_REGISTRY.items() if e["async"])


def list_sync_tools() -> list[str]:
    """Return a sorted list of all sync tool names."""
    return sorted(n for n, e in TOOL_REGISTRY.items() if not e["async"])


# Backward-compat alias: a flat callable map for old consumers (e.g. orchestrator).
TOOL_FUNCTIONS: dict[str, Callable] = {n: e["fn"] for n, e in TOOL_REGISTRY.items()}


__all__ = [
    "TOOL_REGISTRY", "TOOL_FUNCTIONS",
    "get_tool", "get_tool_meta", "is_async_tool", "call_tool",
    "list_tools", "list_async_tools", "list_sync_tools",
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
