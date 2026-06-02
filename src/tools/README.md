# Tools — `src/tools/`

A flat registry of 55 standalone functions used by agents in the
DeepInsights pipeline and by the `/search/document` route. Each tool lives
in its own file with explicit parameters (no hidden globals, no wrapper
classes).

## Layout

```
src/tools/
├── __init__.py                  # TOOL_REGISTRY (with async/desc metadata),
│                                #   get_tool(), call_tool(), list_tools()
├── safety.py                    # path validation, filename sanitization
│
├── filesystemtools/             # 27 tools / 9 files
│   ├── write_file.py            # write_text, write_json, write_bytes
│   ├── read_file.py             # read_text, read_json, read_bytes
│   ├── edit_file.py             # append_to_file, replace_in_file, insert_at_line
│   ├── delete_file.py           # delete_file, delete_directory, cleanup_temp_files
│   ├── list_files.py            # list_files, list_by_extension, get_file_size, get_file_info
│   ├── make_directory.py        # make_dir, make_temp_dir, make_reports_dir
│   ├── hash_text.py             # hash_string, hash_file, cache_key
│   ├── truncate_text.py         # truncate, truncate_to_tokens_approx
│   └── save_chunk.py            # save_chunk, load_chunk
│
├── websearchtools/              # 13 tools / 6 files
│   ├── extract_domain.py        # extract_domain, is_same_domain
│   ├── extract_snippet.py       # extract_snippet, extract_text_blocks
│   ├── filter_medical.py        # is_medical_domain, filter_medical, list_medical_domains
│   ├── rank_results.py          # rank_by_keywords, top_n
│   ├── group_by_domain.py       # group_by_domain, count_per_domain
│   └── deduplicate.py           # deduplicate_by_url, deduplicate_by_title
│
├── citationtools/               # 8 tools / 4 files
│   ├── extract_citations.py     # extract_chunk_ids, extract_web_ids, extract_all_citations, extract_citation_markers
│   ├── validate_claim.py        # claim_supported_by_source, is_supported
│   ├── build_citation_schema.py # build_citation_schema
│   └── find_best_source.py      # find_best_source
│
└── doctools/                    # 8 tools / 7 files
    ├── generate_pdf.py          # generate_pdf
    ├── generate_docx.py         # generate_docx
    ├── generate_filename.py     # generate_filename
    ├── get_pdf_metadata.py      # get_pdf_metadata
    ├── split_sections.py        # split_sections
    ├── build_doc_sections.py    # build_doc_sections
    └── format_citations.py      # format_citations_inline, count_citations
```

## Usage

### Direct import (preferred — explicit and greppable)

```python
from src.tools.filesystemtools.write_file import write_text
from src.tools.doctools.generate_pdf import generate_pdf
from src.tools.citationtools.build_citation_schema import build_citation_schema

# Sync tool — just call
schema = build_citation_schema(claims, sources)

# Async tool — await
path = await write_text("hello", prefix="greeting")
```

### Registry lookup

```python
from src.tools import get_tool, get_tool_meta, is_async_tool, call_tool

# Get the function
fn = get_tool("generate_pdf")
path = fn(sections, title="Report")

# Inspect metadata
meta = get_tool_meta("write_text")
# {"fn": <coroutine>, "async": True, "desc": "...", "name": "write_text"}

# Detect async without calling
is_async_tool("write_text")  # True
is_async_tool("hash_string")  # False

# Call by name with auto-await
result = await call_tool("read_json", "/tmp/openinsight_temp/x.json")
```

### Wiring points

- `src/query/deepinsight/orchestrator.py` binds the registry: `self.tools = TOOL_REGISTRY`
- `src/api/routes/search.py` uses `get_tool("build_doc_sections")` and `get_tool("generate_pdf"|"generate_docx")` in `POST /search/document`
- All 5 agents (RAG, Web Search, Synthesis, Citation Validator, DocGen) import tools directly

## Async vs sync

The registry records each tool's kind:

```python
from src.tools import list_async_tools, list_sync_tools, list_tools

list_async_tools()
# ['append_to_file', 'cleanup_temp_files', 'delete_directory', 'delete_file',
#  'get_file_info', 'get_file_size', 'insert_at_line', 'list_by_extension',
#  'list_files', 'load_chunk', 'make_dir', 'make_reports_dir', 'make_temp_dir',
#  'read_bytes', 'read_json', 'read_text', 'replace_in_file', 'save_chunk',
#  'write_bytes', 'write_json', 'write_text']

list_sync_tools()
# ['build_citation_schema', 'build_doc_sections', 'cache_key', ...]
```

Use `is_async_tool(name)` to branch without invoking, or `await call_tool(...)` to
dispatch transparently.

## Safety

All filesystem tools that accept a path check it against
`ALLOWED_ROOTS` in `src/tools/safety.py` (currently `/tmp/openinsight_temp`,
`/tmp/openinsight_reports`, and `/tmp` itself). Operations on paths outside
those roots:

- **Read / list / stat**: silently return `None` / `[]` / `0` and log a warning.
- **Write / edit**: raise `ValueError`.
- **Delete**: require `confirm=True` and log a warning.

Filenames are sanitized to strip path separators, NUL bytes, and control
characters. Traversal sequences like `..` are rejected in directory names.

```python
from src.tools import call_tool

# This will be rejected:
await call_tool("make_dir", "../../etc/passwd")
# ValueError: name must be a relative, non-traversal string, got: '../../etc/passwd'

# Delete outside allowed roots requires confirm:
await call_tool("delete_directory", "/var/log/something")
# PermissionError: refusing to delete directory /var/log/something
#   outside allowed roots. Pass confirm=True to override.
```

## Optional dependencies

| Tool | Optional dep | Fallback |
|------|-------------|----------|
| All filesystem I/O | `aiofiles` | stdlib `open()` (sync) |
| `get_pdf_metadata` | `PyPDF2` | returns "Unknown" / 1 page |
| `generate_pdf` | `reportlab` | returns `""` and logs warning |
| `generate_docx` | `python-docx` | returns `""` and logs warning |

No new pip dependencies are required for the tools package to load. Tools
that need an optional dep will return a sensible empty/None result if it is
not installed.

## Citation heuristics

`claim_supported_by_source` and `is_supported` use **token overlap**, not
semantic similarity. See the module docstring of
`src/tools/citationtools/validate_claim.py` for the full list of
limitations.

To plug in a semantic check (e.g. embedding similarity):

```python
from src.tools.citationtools.validate_claim import register_semantic_check

def my_semantic_check(claim: str, source_text: str):
    score = compute_embedding_similarity(claim, source_text)
    return {"supported": score > 0.75, "score": score, "method": "embedding"}

register_semantic_check(my_semantic_check)
```

The semantic check is invoked first; if it returns `None`, the function
falls back to token overlap. Pass `None` to `register_semantic_check` to
clear it.
