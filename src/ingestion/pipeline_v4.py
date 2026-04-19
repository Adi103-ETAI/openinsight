from __future__ import annotations

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from src.core.config import get_settings
from src.ingestion.chunker_v3 import HierarchicalChunkerV3
from src.ingestion.embedder_v2 import DualEmbedderV2
from src.ingestion.metadata_v2 import MetadataEnricherV2
from src.ingestion.mongo_store_v2 import MongoDocStoreV2
from src.ingestion.parsers.grobid import GROBIDParser
from src.ingestion.parsers.icmr import ICMRParser
from src.ingestion.parsers.ocr import OCRParser
from src.ingestion.qdrant_indexer_v2 import QdrantIndexerV2

logger = logging.getLogger(__name__)


class IngestionPipelineV4:
    """
    v2 ingestion orchestrator for local directories.

    Flow:
    parse -> enrich metadata -> chunk -> embed -> qdrant upsert -> mongo store
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunker = HierarchicalChunkerV3()
        self.metadata = MetadataEnricherV2()
        self.embedder = DualEmbedderV2(
            dense_model_name=self.settings.dense_model_name
            or self.settings.embedding_model
        )
        self.indexer = QdrantIndexerV2(
            qdrant_url=self.settings.qdrant_url,
            qdrant_api_key=self.settings.qdrant_api_key,
        )
        self.mongo = MongoDocStoreV2(
            mongo_url=self.settings.mongodb_url,
            db_name=self.settings.mongodb_db,
        )

    async def ingest_directory(
        self,
        directory: str,
        source: str,
        recreate_index: bool = False,
        batch_size: int = 10,
    ) -> dict[str, int]:
        input_dir = Path(directory)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        files = sorted(
            [
                p
                for p in input_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in {".pdf", ".xml"}
            ]
        )

        logger.info("[v4] Found %s candidate files in %s", len(files), directory)

        self.indexer.create_collection(
            recreate=recreate_index,
            collection_name=self.settings.qdrant_collection_v2,
        )

        summary = {
            "files_total": len(files),
            "files_parsed": 0,
            "documents_stored": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "files_failed": 0,
        }

        if not files:
            return summary

        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            logger.info(
                "[v4] Processing batch %s (%s files)",
                (start // batch_size) + 1,
                len(batch_files),
            )

            parsed_docs = await self._parse_batch(batch_files, source)

            docs_for_batch: list[dict[str, Any]] = []
            chunks_for_batch: list[Any] = []

            for doc in parsed_docs:
                normalized = self._normalize_document(doc, source)
                enriched = self.metadata.enrich_document(normalized, source)

                chunks = self.chunker.chunk_document(normalized, enriched)
                if not chunks:
                    continue

                total_chunks = len(chunks)
                for idx, chunk in enumerate(chunks):
                    chunk_meta = self.metadata.build_chunk_metadata(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        chunk_type=chunk.chunk_type,
                        section_title=chunk.section_title,
                        chunk_index=idx,
                        total_chunks=total_chunks,
                        source=source,
                        doc_metadata=enriched,
                        has_table=chunk.chunk_type == "table",
                    )
                    chunk.chunk_index = idx
                    chunk.total_chunks = total_chunks
                    chunk.metadata = chunk_meta.to_qdrant_payload()

                docs_for_batch.append({"doc": normalized, "enriched": enriched})
                chunks_for_batch.extend(chunks)

            if not chunks_for_batch:
                continue

            contextual_texts = [c.contextual_text for c in chunks_for_batch]
            dense_embeddings = await self._run_cpu(
                self.embedder.embed_batch,
                contextual_texts,
                32,
            )
            sparse_vectors = [
                self.embedder.compute_sparse_vector(text) for text in contextual_texts
            ]

            indexed = self.indexer.upsert_chunks(
                chunks=chunks_for_batch,
                dense_embeddings=dense_embeddings,
                sparse_vectors=sparse_vectors,
                collection_name=self.settings.qdrant_collection_v2,
            )

            for entry in docs_for_batch:
                await self.mongo.store_document(entry["doc"], entry["enriched"])
            await self.mongo.store_chunks(chunks_for_batch)

            summary["files_parsed"] += len(parsed_docs)
            summary["documents_stored"] += len(docs_for_batch)
            summary["chunks_created"] += len(chunks_for_batch)
            summary["chunks_indexed"] += indexed

            logger.info(
                "[v4] Batch complete: docs=%s chunks=%s indexed=%s",
                len(docs_for_batch),
                len(chunks_for_batch),
                indexed,
            )

        summary["files_failed"] = summary["files_total"] - summary["files_parsed"]
        logger.info("[v4] Ingestion complete: %s", summary)
        return summary

    async def _parse_batch(self, file_paths: list[Path], source: str) -> list[Any]:
        loop = asyncio.get_running_loop()
        parsed: list[Any] = []

        with ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [
                loop.run_in_executor(pool, self._parse_file, file_path, source)
                for file_path in file_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("[v4] Parse worker failed: %s", result)
                continue
            if not result:
                continue
            parsed.extend(result)

        return parsed

    def _parse_file(self, file_path: Path, source: str) -> list[Any]:
        suffix = file_path.suffix.lower()
        if suffix == ".xml":
            return self._parse_pubmed_xml_file(file_path)

        if suffix != ".pdf":
            return []

        try:
            if source in {"icmr", "nmc_guideline", "rssdi"}:
                docs = ICMRParser(file_path).parse()
            else:
                docs = GROBIDParser(file_path, source_type=source).parse()

            if docs:
                return docs

            if OCRParser.is_scanned(file_path):
                return OCRParser(file_path, source_type=source).parse()
            return []
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error("[v4] Parser failed for %s: %s", file_path, exc)
            return []

    def _parse_pubmed_xml_file(self, file_path: Path) -> list[dict[str, Any]]:
        try:
            tree = ET.parse(file_path)
        except ET.ParseError as exc:
            logger.error("[v4] Invalid XML %s: %s", file_path, exc)
            return []

        root = tree.getroot()
        articles = root.findall(".//PubmedArticle")
        if not articles and root.tag.endswith("PubmedArticle"):
            articles = [root]

        docs: list[dict[str, Any]] = []
        for article in articles:
            citation = article.find("MedlineCitation")
            if citation is None:
                continue
            article_data = citation.find("Article")
            if article_data is None:
                continue

            title = (article_data.findtext("ArticleTitle") or "").strip()
            abstract_nodes = article_data.findall(".//AbstractText")
            abstract = " ".join(
                (node.text or "").strip()
                for node in abstract_nodes
                if (node.text or "").strip()
            )
            if not title and not abstract:
                continue

            pmid = (citation.findtext("PMID") or "").strip()
            year_text = (
                article_data.findtext(".//Journal/JournalIssue/PubDate/Year") or ""
            ).strip()
            year = int(year_text) if year_text.isdigit() else 0
            journal = (
                article_data.findtext(".//Journal/Title")
                or article_data.findtext(".//Journal/ISOAbbreviation")
                or ""
            ).strip()

            doi = None
            for id_node in article.findall(".//ArticleId"):
                if id_node.attrib.get("IdType") == "doi":
                    doi = (id_node.text or "").strip() or None
                    break

            mesh_terms = [
                (n.text or "").strip()
                for n in citation.findall(".//MeshHeading/DescriptorName")
                if (n.text or "").strip()
            ]
            keywords = [
                (n.text or "").strip()
                for n in citation.findall(".//Keyword")
                if (n.text or "").strip()
            ]

            authors: list[str] = []
            for author in article_data.findall(".//Author"):
                last = (author.findtext("LastName") or "").strip()
                fore = (author.findtext("ForeName") or "").strip()
                full_name = f"{last} {fore}".strip()
                if full_name:
                    authors.append(full_name)

            content = f"{title}\n\n{abstract}".strip()
            doc_id = (
                f"pmid_{pmid}"
                if pmid
                else self._hash_doc_id(str(file_path), title, abstract)
            )

            docs.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "abstract": abstract,
                    "content": content,
                    "authors": authors,
                    "year": year,
                    "journal": journal,
                    "doi": doi,
                    "pmid": pmid or None,
                    "mesh_terms": mesh_terms,
                    "keywords": keywords,
                    "sections": [],
                    "url": str(file_path.resolve()),
                    "source_type": "pubmed",
                }
            )

        return docs

    def _normalize_document(self, doc: Any, source: str) -> dict[str, Any]:
        if isinstance(doc, dict):
            out = dict(doc)
        else:
            out = {
                "title": getattr(doc, "title", ""),
                "abstract": getattr(doc, "abstract", ""),
                "content": getattr(doc, "content", ""),
                "authors": getattr(doc, "authors", []),
                "year": getattr(doc, "year", 0),
                "journal": getattr(doc, "journal", ""),
                "doi": getattr(doc, "doi", None),
                "pmid": getattr(doc, "pmid", None),
                "mesh_terms": getattr(doc, "mesh_terms", []),
                "keywords": getattr(doc, "keywords", []),
                "sections": getattr(doc, "sections", []),
                "url": getattr(doc, "url", ""),
                "source_type": getattr(doc, "source_type", source),
            }

        out.setdefault("abstract", "")
        out.setdefault("content", "")
        out.setdefault("authors", [])
        out.setdefault("mesh_terms", [])
        out.setdefault("keywords", [])
        out.setdefault("sections", [])
        out.setdefault("source_type", source)

        year = out.get("year", 0)
        try:
            out["year"] = int(year) if year is not None else 0
        except (TypeError, ValueError):
            out["year"] = 0

        pmid = out.get("pmid")
        doi = out.get("doi")
        doc_id = out.get("doc_id")
        if not doc_id:
            if isinstance(pmid, str) and pmid.strip():
                doc_id = f"pmid_{pmid.strip()}"
            elif isinstance(doi, str) and doi.strip():
                doc_id = f"doi_{doi.strip().replace('/', '_')}"
            else:
                doc_id = self._hash_doc_id(
                    out.get("url", ""), out.get("title", ""), out.get("content", "")
                )
        out["doc_id"] = doc_id

        return out

    def _hash_doc_id(self, *parts: str) -> str:
        joined = "|".join(part or "" for part in parts)
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
        return f"doc_{digest}"

    async def _run_cpu(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args))
