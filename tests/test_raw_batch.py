"""Raw-lake ingest batch tests — stubs only, no torch/network/DB.

Covers routing (pubmed XML bytes vs HTML parser dispatch vs fallback) and
the ingest_raw_batch summary shape + delegation to the shared downstream
(_ingest_parsed_tuples, the same routine ingest_scraped_documents uses,
which owns the change-detection skip).

Sync tests drive coroutines via asyncio.run() so no pytest-asyncio plugin
is needed. Skipped entirely where pipeline deps are unavailable.
"""
from __future__ import annotations

import asyncio
import sys
import types

import pytest

pipeline = pytest.importorskip("src.ingestion.pipeline")
from src.ingestion.deduplication import new_scraped_summary  # noqa: E402

pytestmark = pytest.mark.unit

EFETCH_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Rifampicin for tuberculosis treatment</ArticleTitle>
        <Abstract><AbstractText>Rifampicin induces CYP450 enzymes and is a first-line TB drug. This abstract is deliberately long enough to pass the minimal content threshold for chunking in tests.</AbstractText></Abstract>
        <Journal><Title>Tuberculosis Journal</Title><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def _bare_pipeline():
    """Instance without __init__ (no embedder/mongo/network)."""
    return object.__new__(pipeline.IngestionPipeline)


def _meta(**over):
    base = {
        "doc_id": "pmid_12345",
        "source": "pubmed",
        "url": "https://example.test/12345",
        "content_type": "application/xml",
        "title": "hint title",
        "error": None,
    }
    base.update(over)
    return base


class TestParserDispatchMap:
    def test_covers_notebook_sources(self):
        assert set(pipeline._SCRAPED_PARSER_PATHS) == {
            "indmed", "medknow", "pmc_india", "statpearls",
            "ncbi_bookshelf", "cdsco", "ctri", "nfi",
        }

    def test_pubmed_and_unknown_have_no_scraped_parser(self):
        assert pipeline.get_scraped_parser("pubmed") is None
        assert pipeline.get_scraped_parser("nope") is None


class TestRawItemRouting:
    def test_pubmed_xml_bytes_routed_to_efetch_parser(self):
        pipe = _bare_pipeline()
        record, chunks = pipe._raw_item_to_parsed(
            "pubmed", EFETCH_XML, _meta()
        )
        assert record.source_type == "pubmed"
        assert "Rifampicin" in record.title
        assert len(chunks) == 1
        assert "Rifampicin" in chunks[0].chunk_text

    def test_pubmed_xml_with_book_article(self):
        book_xml = (
            b'<?xml version="1.0"?><PubmedArticleSet><PubmedBookArticle>'
            b"<BookDocument><PMID>NBK999</PMID>"
            b"<ArticleTitle>StatPearls Test Chapter Title</ArticleTitle>"
            b"<Abstract><AbstractText>Abstract text that is long enough to clear the eighty character minimum for a chunk record.</AbstractText></Abstract>"
            b"</BookDocument></PubmedBookArticle></PubmedArticleSet>"
        )
        pipe = _bare_pipeline()
        record, chunks = pipe._raw_item_to_parsed(
            "pubmed", book_xml, _meta()
        )
        assert "StatPearls Test Chapter" in record.title
        assert len(chunks) == 1

    def test_invalid_xml_returns_none(self):
        pipe = _bare_pipeline()
        assert pipe._raw_item_to_parsed("pubmed", b"not xml", _meta()) is None

    def test_tiny_pubmed_content_returns_none(self):
        tiny = (
            b'<?xml version="1.0"?><PubmedArticleSet><PubmedArticle><MedlineCitation>'
            b"<PMID>1</PMID><Article><ArticleTitle>T</ArticleTitle>"
            b"<Abstract><AbstractText>short</AbstractText></Abstract></Article>"
            b"</MedlineCitation></PubmedArticle></PubmedArticleSet>"
        )
        pipe = _bare_pipeline()
        assert pipe._raw_item_to_parsed("pubmed", tiny, _meta()) is None

    def test_html_routing_delegates_to_source_parser(self, monkeypatch):
        seen = {}

        class FakeParser:
            def parse(self, scraped):
                seen["scraped"] = scraped
                return ("REC", ["CHK"])

        monkeypatch.setattr(
            pipeline, "get_scraped_parser", lambda name: FakeParser()
        )
        pipe = _bare_pipeline()
        html = b"<html><body>StatPearls content</body></html>"
        out = pipe._raw_item_to_parsed(
            "statpearls", html,
            _meta(source="statpearls", content_type="text/html"),
        )
        assert out == ("REC", ["CHK"])
        scraped = seen["scraped"]
        assert scraped.content == html
        assert scraped.source == "statpearls"
        assert scraped.url == "https://example.test/12345"

    def test_unknown_source_uses_minimal_fallback(self, monkeypatch):
        monkeypatch.setattr(pipeline, "get_scraped_parser", lambda name: None)
        pipe = _bare_pipeline()
        body = b"<html><body>" + b"clinical text " * 100 + b"</body></html>"
        record, chunks = pipe._raw_item_to_parsed(
            "weird", body, _meta(source="weird", content_type="text/html"),
        )
        assert record.source_type == "weird"
        assert len(chunks) == 1

    def test_tiny_fallback_content_returns_none(self, monkeypatch):
        monkeypatch.setattr(pipeline, "get_scraped_parser", lambda name: None)
        pipe = _bare_pipeline()
        assert pipe._raw_item_to_parsed(
            "weird", b"tiny", _meta(source="weird", content_type="text/html"),
        ) is None


def _install_fake_raw_lake(items, monkeypatch):
    """Stub src.ingestion.raw_lake.iter_raw with an in-memory item list."""
    mod = types.ModuleType("src.ingestion.raw_lake")

    def iter_raw(source, limit=0):
        count = 0
        for meta, content in items:
            yield meta, content
            count += 1
            if limit and count >= limit:
                return

    mod.iter_raw = iter_raw
    monkeypatch.setitem(sys.modules, "src.ingestion.raw_lake", mod)
    return mod


class TestIngestRawBatch:
    def _pipe_with_fake_downstream(self, monkeypatch):
        pipe = _bare_pipeline()
        seen = {}

        async def fake_downstream(documents, source, batch_size=10,
                                  recreate_index=False):
            seen["documents"] = list(documents)
            seen["kwargs"] = {
                "source": source, "batch_size": batch_size,
                "recreate_index": recreate_index,
            }
            return new_scraped_summary(len(documents))

        monkeypatch.setattr(
            pipeline.IngestionPipeline, "_ingest_parsed_tuples", fake_downstream
        )
        return pipe, seen

    def test_summary_shape_routing_and_limit(self, monkeypatch):
        class FakeParser:
            def parse(self, scraped):
                return ("REC2", ["CHK2"])

        monkeypatch.setattr(
            pipeline, "get_scraped_parser", lambda name: FakeParser()
        )
        items = [
            (_meta(), EFETCH_XML),  # pubmed XML → parsed
            (_meta(source="statpearls", content_type="text/html",
                   doc_id="ch1"), b"<html>StatPearls</html>"),  # HTML → parser
            (_meta(doc_id="bad", error="fetch failed"), b""),  # manifest error
            (_meta(doc_id="broken"), b"not xml"),  # invalid XML
        ]
        _install_fake_raw_lake(items, monkeypatch)
        pipe, seen = self._pipe_with_fake_downstream(monkeypatch)

        summary = asyncio.run(pipe.ingest_raw_batch("pubmed"))

        assert set(summary) == set(new_scraped_summary(0))
        assert summary["documents_total"] == 4
        assert len(seen["documents"]) == 2
        assert summary["files_failed"] == 2
        assert seen["kwargs"]["source"] == "pubmed"

    def test_limit_passes_through_to_iter_raw(self, monkeypatch):
        monkeypatch.setattr(pipeline, "get_scraped_parser", lambda name: None)
        items = [(_meta(doc_id=f"d{i}"), EFETCH_XML) for i in range(5)]
        _install_fake_raw_lake(items, monkeypatch)
        pipe, seen = self._pipe_with_fake_downstream(monkeypatch)

        summary = asyncio.run(pipe.ingest_raw_batch("pubmed", limit=2))

        assert summary["documents_total"] == 2
        assert len(seen["documents"]) == 2

    def test_scraped_entry_point_shares_downstream(self, monkeypatch):
        """ingest_scraped_documents must delegate to _ingest_parsed_tuples."""
        pipe, seen = self._pipe_with_fake_downstream(monkeypatch)
        docs = [("R", ["C"])]

        summary = asyncio.run(
            pipe.ingest_scraped_documents(documents=docs, source="indmed")
        )

        assert seen["documents"] == docs
        assert set(summary) == set(new_scraped_summary(0))
