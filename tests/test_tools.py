"""
Smoke tests for src/tools/.

Covers:
- Registry metadata (async/sync classification, list_tools, etc.)
- Safety helpers (filename sanitization, path validation)
- Filesystem tools (read/write/edit/list/delete with safety)
- Web search tools (extraction, filtering, ranking, dedup)
- Citation tools (extraction, validation, schema)
- Doc tools (filename, sections, citations, PDF metadata coercion)

These are smoke tests — they exercise happy paths and a few common
failure modes. They intentionally avoid network and LLM calls.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Make src importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Registry / metadata
# ---------------------------------------------------------------------------

def test_registry_has_tools():
    from src.tools import TOOL_REGISTRY, list_tools
    assert len(TOOL_REGISTRY) >= 50, f"expected >= 50 tools, got {len(TOOL_REGISTRY)}"
    assert len(list_tools()) == len(TOOL_REGISTRY)


def test_registry_metadata_shape():
    from src.tools import TOOL_REGISTRY, get_tool_meta
    for name, entry in TOOL_REGISTRY.items():
        assert entry["name"] == name
        assert "fn" in entry and callable(entry["fn"])
        assert isinstance(entry["async"], bool)
        assert isinstance(entry["desc"], str) and entry["desc"]


def test_async_sync_classification_is_correct():
    from src.tools import is_async_tool
    # Spot-check a few known values
    assert is_async_tool("write_text") is True
    assert is_async_tool("read_text") is True
    assert is_async_tool("hash_string") is False
    assert is_async_tool("generate_pdf") is False
    assert is_async_tool("extract_domain") is False


def test_get_tool_raises_for_unknown():
    from src.tools import get_tool
    with pytest.raises(KeyError):
        get_tool("nonexistent_tool")


def test_list_async_and_sync_partition():
    from src.tools import list_async_tools, list_sync_tools, list_tools
    a = set(list_async_tools())
    s = set(list_sync_tools())
    assert a.isdisjoint(s)
    assert a | s == set(list_tools())


@pytest.mark.asyncio
async def test_call_tool_awaits_when_async():
    from src.tools import call_tool
    path = await call_tool("write_text", "hello world", prefix="smoke")
    assert path and Path(path).exists()
    content = await call_tool("read_text", path)
    assert content == "hello world"
    # cleanup
    Path(path).unlink(missing_ok=True)


def test_call_tool_invokes_sync_directly():
    from src.tools import call_tool
    result = asyncio.run(call_tool("hash_string", "abc"))
    assert isinstance(result, str) and len(result) == 64


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

def test_sanitize_filename_strips_separators():
    from src.tools.safety import sanitize_filename
    assert "/" not in sanitize_filename("a/b/c")
    assert "\\" not in sanitize_filename("a\\b\\c")
    assert "\x00" not in sanitize_filename("a\x00b")
    assert sanitize_filename("") == "unnamed"
    assert sanitize_filename("...") == "unnamed"


def test_sanitize_filename_keeps_safe_chars():
    from src.tools.safety import sanitize_filename
    assert sanitize_filename("report_2024.pdf") == "report_2024.pdf"


def test_sanitize_directory_name_strips_dots():
    from src.tools.safety import sanitize_directory_name
    assert "." not in sanitize_directory_name(".hidden")
    assert sanitize_directory_name("") == "unnamed"


def test_is_absolute_or_traversal():
    from src.tools.safety import is_absolute_or_traversal
    assert is_absolute_or_traversal("/etc/passwd") is True
    assert is_absolute_or_traversal("~/foo") is True
    assert is_absolute_or_traversal("../escape") is True
    assert is_absolute_or_traversal("a/../../etc") is True
    assert is_absolute_or_traversal("normal_name") is False
    assert is_absolute_or_traversal("") is True


def test_is_path_safe_accepts_temp_dir():
    from src.tools.safety import is_path_safe
    assert is_path_safe(Path("/tmp/openinsight_temp/x")) is True
    assert is_path_safe(Path("/tmp/something_but_still_in_tmp")) is True


def test_is_path_safe_rejects_system_dirs():
    from src.tools.safety import is_path_safe
    assert is_path_safe(Path("/etc/passwd")) is False
    assert is_path_safe(Path("/var/log/foo")) is False
    assert is_path_safe(Path("/root/.ssh/id_rsa")) is False


# ---------------------------------------------------------------------------
# Filesystem tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_write_text_and_read_text():
    from src.tools.filesystemtools.write_file import write_text
    from src.tools.filesystemtools.read_file import read_text
    path = await write_text("hello", prefix="smoke")
    try:
        assert await read_text(path) == "hello"
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_write_text_rejects_absolute_prefix():
    from src.tools.filesystemtools.write_file import write_text
    with pytest.raises(ValueError):
        await write_text("x", prefix="/etc/passwd")


@pytest.mark.asyncio
async def test_write_text_rejects_traversal_prefix():
    from src.tools.filesystemtools.write_file import write_text
    with pytest.raises(ValueError):
        await write_text("x", prefix="../escape")


@pytest.mark.asyncio
async def test_write_json_and_read_json_roundtrip():
    from src.tools.filesystemtools.write_file import write_json
    from src.tools.filesystemtools.read_file import read_json
    path = await write_json({"a": 1, "b": [2, 3]}, prefix="smoke")
    try:
        assert await read_json(path) == {"a": 1, "b": [2, 3]}
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_read_text_rejects_unsafe_path():
    from src.tools.filesystemtools.read_file import read_text
    assert await read_text("/etc/passwd") is None


@pytest.mark.asyncio
async def test_make_dir_sanitizes_name():
    from src.tools.filesystemtools.make_directory import make_dir
    # Traversal attempts are rejected outright (security)
    with pytest.raises(ValueError):
        await make_dir("../../../etc/passwd")


@pytest.mark.asyncio
async def test_make_dir_accepts_normal_name():
    from src.tools.filesystemtools.make_directory import make_dir
    path = await make_dir("normal-subdir")
    assert Path(path).exists()
    # cleanup
    Path(path).rmdir()


@pytest.mark.asyncio
async def test_make_dir_rejects_absolute_name():
    from src.tools.filesystemtools.make_directory import make_dir
    with pytest.raises(ValueError):
        await make_dir("/absolute/path")


@pytest.mark.asyncio
async def test_delete_file_outside_allowed_roots_requires_confirm():
    from src.tools.filesystemtools.delete_file import delete_file
    # /etc/passwd exists on linux; without confirm we must refuse
    result = await delete_file("/etc/passwd")
    assert result is False


@pytest.mark.asyncio
async def test_delete_file_inside_allowed_roots_works():
    from src.tools.filesystemtools.write_file import write_text
    from src.tools.filesystemtools.delete_file import delete_file
    path = await write_text("to be deleted", prefix="smoke")
    assert Path(path).exists()
    assert await delete_file(path) is True
    assert not Path(path).exists()


@pytest.mark.asyncio
async def test_save_chunk_and_load_chunk():
    from src.tools.filesystemtools.save_chunk import save_chunk, load_chunk
    data = {"id": "C001", "text": "metformin is used for diabetes"}
    path = await save_chunk(data, "smoke_test")
    try:
        loaded = await load_chunk(path)
        assert loaded == data
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_save_chunk_sanitizes_id():
    from src.tools.filesystemtools.save_chunk import save_chunk
    data = {"id": "../../etc/passwd", "text": "x"}
    path = await save_chunk(data, "smoke_test")
    assert ".." not in path
    assert "/" not in Path(path).name.replace(".json", "")
    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Web search tools
# ---------------------------------------------------------------------------

def test_extract_domain_strips_www():
    from src.tools.websearchtools.extract_domain import extract_domain
    assert extract_domain("https://www.cdc.gov/diabetes") == "cdc.gov"
    assert extract_domain("https://cdc.gov") == "cdc.gov"
    assert extract_domain("") == "unknown"
    assert extract_domain("not a url") == "unknown"


def test_is_medical_domain():
    from src.tools.websearchtools.filter_medical import is_medical_domain
    assert is_medical_domain("https://www.cdc.gov/x") is True
    assert is_medical_domain("https://www.mayoclinic.org/x") is True
    assert is_medical_domain("https://random-blog.example/x") is False


def test_filter_medical_keeps_only_medical():
    from src.tools.websearchtools.filter_medical import filter_medical
    results = [
        {"url": "https://www.cdc.gov/x", "title": "CDC"},
        {"url": "https://random-blog.com/y", "title": "Blog"},
    ]
    out = filter_medical(results)
    assert len(out) == 1
    assert out[0]["is_medical"] is True
    assert out[0]["domain"] == "cdc.gov"


def test_rank_by_keywords_scores_by_overlap():
    from src.tools.websearchtools.rank_results import rank_by_keywords
    results = [
        {"title": "Diabetes treatment options", "snippet": "metformin info"},
        {"title": "Random article", "snippet": "cooking tips"},
    ]
    ranked = rank_by_keywords("diabetes treatment", results)
    assert ranked[0]["title"].startswith("Diabetes")
    assert ranked[0]["relevance_score"] >= ranked[1]["relevance_score"]


def test_deduplicate_by_url_normalizes_trailing_slash():
    from src.tools.websearchtools.deduplicate import deduplicate_by_url
    results = [
        {"url": "https://cdc.gov/a"},
        {"url": "https://cdc.gov/a/"},
        {"url": "https://cdc.gov/b"},
    ]
    deduped = deduplicate_by_url(results)
    assert len(deduped) == 2


def test_group_by_domain():
    from src.tools.websearchtools.group_by_domain import group_by_domain, count_per_domain
    results = [
        {"url": "https://cdc.gov/a"},
        {"url": "https://cdc.gov/b"},
        {"url": "https://nih.gov/c"},
    ]
    groups = group_by_domain(results)
    assert len(groups["cdc.gov"]) == 2
    assert len(groups["nih.gov"]) == 1
    assert count_per_domain(results) == {"cdc.gov": 2, "nih.gov": 1}


# ---------------------------------------------------------------------------
# Citation tools
# ---------------------------------------------------------------------------

def test_extract_citations_chunk_and_web():
    from src.tools.citationtools.extract_citations import (
        extract_chunk_ids, extract_web_ids, extract_all_citations, extract_citation_markers,
    )
    text = "See [CHUNK_001] and [WEB_007] for details."
    assert extract_chunk_ids(text) == ["CHUNK_001"]
    assert extract_web_ids(text) == ["WEB_007"]
    assert extract_all_citations(text) == {
        "chunk_citations": ["CHUNK_001"],
        "web_citations": ["WEB_007"],
    }
    markers = extract_citation_markers(text)
    assert len(markers) == 2
    assert markers[0]["type"] == "corpus"
    assert markers[1]["type"] == "web"


def test_extract_citations_zero_pads():
    from src.tools.citationtools.extract_citations import extract_chunk_ids
    assert extract_chunk_ids("[CHUNK_5]") == ["CHUNK_005"]
    assert extract_chunk_ids("[CHUNK_123]") == ["CHUNK_123"]


def test_claim_supported_by_source_token_overlap():
    from src.tools.citationtools.validate_claim import claim_supported_by_source, is_supported
    r = claim_supported_by_source("metformin treats diabetes", "metformin is used to treat type 2 diabetes")
    assert r["supported"] is True
    assert is_supported("metformin treats diabetes", "metformin is used to treat type 2 diabetes") is True


def test_claim_supported_by_source_flags_negation():
    from src.tools.citationtools.validate_claim import claim_supported_by_source
    # Negation words should be flagged even when claim is otherwise supported
    r = claim_supported_by_source(
        "metformin does not treat diabetes",
        "metformin is used to treat type 2 diabetes",
    )
    assert r["has_negation"] is True
    # Note: token overlap still says "supported" — caller must inspect has_negation


def test_claim_unsupported_when_no_overlap():
    from src.tools.citationtools.validate_claim import is_supported
    assert is_supported("completely unrelated claim about cats", "metformin is used to treat type 2 diabetes") is False


def test_register_semantic_check_overrides():
    from src.tools.citationtools.validate_claim import (
        register_semantic_check, claim_supported_by_source,
    )

    def my_check(claim, source):
        return {"supported": True, "score": 0.99, "method": "test_plugin"}

    register_semantic_check(my_check)
    try:
        r = claim_supported_by_source("anything", "anything")
        assert r["method"] == "test_plugin"
        assert r["supported"] is True
        assert r["fallback"]["method"] == "token_overlap"
    finally:
        register_semantic_check(None)


def test_register_semantic_check_none_falls_back():
    from src.tools.citationtools.validate_claim import (
        register_semantic_check, claim_supported_by_source,
    )
    register_semantic_check(None)
    r = claim_supported_by_source("metformin treats diabetes", "metformin treats diabetes patients")
    assert r["method"] == "token_overlap"


def test_build_citation_schema_skips_missing_sources():
    from src.tools.citationtools.build_citation_schema import build_citation_schema
    claims = [
        {"claim_id": "C1", "claim_text": "x", "source_id": "CHUNK_001", "confidence": 0.9},
        {"claim_id": "C2", "claim_text": "y", "source_id": "MISSING_ID", "confidence": 0.5},
    ]
    sources = [{"id": "CHUNK_001", "title": "Doc", "text": "..."}]
    schema = build_citation_schema(claims, sources)
    assert len(schema) == 1
    assert schema[0]["claim_id"] == "C1"
    assert schema[0]["source_type"] == "corpus"


def test_find_best_source_returns_highest_overlap():
    from src.tools.citationtools.find_best_source import find_best_source
    sources = [
        {"id": "A", "text": "random unrelated content"},
        {"id": "B", "text": "metformin is the first-line therapy for type 2 diabetes"},
    ]
    best = find_best_source("metformin treats diabetes", sources)
    assert best is not None
    assert best["id"] == "B"


# ---------------------------------------------------------------------------
# Doc tools
# ---------------------------------------------------------------------------

def test_generate_filename_is_safe_and_stamped():
    from src.tools.doctools.generate_filename import generate_filename
    name = generate_filename("My Report!", ext="pdf")
    assert name.startswith("openinsight_My_Report_")
    assert name.endswith(".pdf")
    assert "!" not in name
    assert " " not in name


def test_generate_filename_empty_title_falls_back():
    from src.tools.doctools.generate_filename import generate_filename
    name = generate_filename("!!!", ext="docx")
    assert "openinsight_report_" in name
    assert name.endswith(".docx")


def test_split_sections_marks_known_headers():
    from src.tools.doctools.split_sections import split_sections
    text = "Diagnosis: diabetes\ndetail\n\nTreatment: metformin\ndetail"
    out = split_sections(text)
    assert "diagnosis" in out
    assert "treatment" in out


def test_split_sections_fallback_to_paragraphs():
    from src.tools.doctools.split_sections import split_sections
    text = "First paragraph here.\n\nSecond paragraph here."
    out = split_sections(text)
    assert "paragraph_1" in out or "full_answer" in out


def test_build_doc_sections_includes_header_and_disclaimer():
    from src.tools.doctools.build_doc_sections import build_doc_sections
    out = build_doc_sections("Diagnosis: x", [{"claim_id": "C1", "source_title": "Doc"}], "Title")
    assert "header" in out
    assert "disclaimer" in out
    assert "sources_and_citations" in out


def test_count_citations():
    from src.tools.doctools.format_citations import count_citations
    text = "See [CHUNK_001], [CHUNK_002] and [WEB_001]."
    assert count_citations(text) == {"chunk_count": 2, "web_count": 1}


def test_format_citations_inline_is_identity_for_now():
    from src.tools.doctools.format_citations import format_citations_inline
    assert format_citations_inline("see [CHUNK_001]") == "see [CHUNK_001]"


# ---------------------------------------------------------------------------
# PDF metadata coercion
# ---------------------------------------------------------------------------

def test_coerce_date_handles_datetime():
    from src.tools.doctools.get_pdf_metadata import _coerce_date
    from datetime import datetime
    assert _coerce_date(datetime(2024, 1, 15, 10, 30, 0)) == "2024-01-15 10:30:00"


def test_coerce_date_handles_pdf_string_format():
    from src.tools.doctools.get_pdf_metadata import _coerce_date
    assert _coerce_date("D:20240115103000") == "2024-01-15 10:30:00"
    # Note: the strict regex requires full YYYYMMDDhhmmss; partial dates
    # like "D:20240115" fall through to generic handling, which currently
    # returns the original string (the PyPDF2 library typically fills in
    # missing fields, so this is mostly a defensive case).
    out = _coerce_date("D:20240115")
    assert out  # should be non-empty


def test_coerce_date_handles_none_and_empty():
    from src.tools.doctools.get_pdf_metadata import _coerce_date
    assert _coerce_date(None) == "Unknown"
    assert _coerce_date("") == "Unknown"
    assert _coerce_date("not a date") == "not a date"  # falls back to original


def test_coerce_str_handles_none_and_empty():
    from src.tools.doctools.get_pdf_metadata import _coerce_str
    assert _coerce_str(None) == "Unknown"
    assert _coerce_str("") == "Unknown"
    assert _coerce_str("Author Name") == "Author Name"


def test_get_pdf_metadata_returns_defaults_for_missing_file(tmp_path):
    from src.tools.doctools.get_pdf_metadata import get_pdf_metadata
    out = get_pdf_metadata(str(tmp_path / "nonexistent.pdf"))
    assert out["page_count"] == 1
    assert out["title"] == "Unknown"
    assert out["file_size"] == 0
