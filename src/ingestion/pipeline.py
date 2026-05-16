from __future__ import annotations

import asyncio
import hashlib
import logging
import xml.etree.ElementTree as ET

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config.settings import get_settings
from src.ml.chunking.chunker import HierarchicalChunkerV3
from src.ingestion.checkpoint import CheckpointManager
from src.ingestion.dedupe import DocumentDeduplicator
from src.ml.embedding.embedder import BaseEmbedder, create_embedder
from src.ingestion.metadata import MetadataEnricherV2
from src.data.mongo.doc_store import MongoDocStoreV2
from src.ingestion.monitoring import IngestionMonitor, RunMetrics
from src.ingestion.parsers.grobid import GROBIDParser
from src.ingestion.parsers.icmr import ICMRParser
from src.ingestion.parsers.ocr import OCRParser
from src.ingestion.quality import score_chunks
from src.ingestion.vector_indexer import VectorIndexer

logger = logging.getLogger(__name__)

# Error types for dead letter queue
ERROR_TYPE_PARSE = "parse_error"
ERROR_TYPE_EMBED = "embed_error"
ERROR_TYPE_INDEX = "index_error"
ERROR_TYPE_OCR = "ocr_error"

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:

    def retry(*_args, **_kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    def stop_after_attempt(_attempts):
        return None

    def wait_exponential(**_kwargs):
        return None


class IngestionPipeline:
    """
    Ingestion orchestrator for local directories.

    Flow:
    parse -> dedup -> enrich metadata -> chunk -> quality score -> embed -> vector upsert -> mongo store -> monitor

    Features:
    - Retry logic for embedding failures
    - Quality scoring and filtering
    - Run monitoring and metrics
    - Document deduplication
    - Configurable thread pool based on CPU cores (default: 75% of cores)
    - Shared MongoDB connection pool for connection reuse
    - Dead letter queue for failed documents
    - OCR fallback for PDF parsing failures
    - Retry logic with exponential backoff
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunker = HierarchicalChunkerV3()
        self.metadata = MetadataEnricherV2()
        # Use config-driven embedder (local for Kaggle/Colab GPU, remote for CPU server)
        self.embedder: BaseEmbedder = create_embedder()
        self.indexer = VectorIndexer()
        self.mongo = MongoDocStoreV2(
            mongo_url=self.settings.mongodb_url,
            db_name=self.settings.mongodb_db,
        )
        self.deduplicator = DocumentDeduplicator(self.mongo)
        self.monitor = IngestionMonitor(self.mongo.db)

        # Thread pool configuration - use settings or fall back to CPU-based default
        self._parse_workers = self.settings.parsing_thread_workers
        self._embed_workers = self.settings.ingestion_thread_workers

        # Dead letter queue configuration
        self._dead_letter_enabled = self.settings.dead_letter_enabled
        self._dead_letter_collection = self.settings.dead_letter_collection
        self._dead_letter_db = self.mongo.db[self._dead_letter_collection]

        logger.info(
            "[pipeline] Pipeline initialized: parse_workers=%d, embed_workers=%d, dead_letter=%s",
            self._parse_workers,
            self._embed_workers,
            "enabled" if self._dead_letter_enabled else "disabled",
        )
        self.checkpoint = CheckpointManager(
            mongo_url=self.settings.mongodb_url,
            db_name=self.settings.mongodb_db,
        )

    async def _store_to_dead_letter(
        self,
        file_path: Path,
        error_type: str,
        error_message: str,
        retry_count: int = 0,
    ) -> None:
        """Store failed document to dead letter queue."""
        if not self._dead_letter_enabled:
            return

        try:
            failed_doc = {
                "file_path": str(file_path.resolve()),
                "error_type": error_type,
                "error_message": str(error_message)[:1000],  # Limit message length
                "timestamp": datetime.utcnow().isoformat(),
                "retry_count": retry_count,
                "source": file_path.parent.name,
            }
            await self._dead_letter_db.insert_one(failed_doc)
            logger.warning(
                "[pipeline] Stored to dead letter: %s - %s (retry: %d)",
                file_path.name,
                error_type,
                retry_count,
            )
        except Exception as e:
            logger.error("[pipeline] Failed to store to dead letter queue: %s", e)

    async def _parse_with_ocr_fallback(
        self, file_path: Path, source: str, max_retries: int = 3
    ) -> list[Any]:
        """
        Parse file with primary parser, falling back to OCR on failure.

        Tries primary parser first (GROBID/ICMR), then falls back to OCR parser
        if PDF parsing fails. This handles scanned PDFs that standard parsing
        cannot process.

        Args:
            file_path: Path to the file to parse
            source: Source type (icmr, pubmed, etc.)
            max_retries: Maximum retry attempts for primary parser

        Returns:
            List of parsed documents, or empty list on failure
        """
        suffix = file_path.suffix.lower()

        # XML files: use primary parser directly without OCR fallback
        if suffix == ".xml":
            result = await self._try_primary_parser(file_path, source)
            if result:
                return result
            # XML parsing failed - store to dead letter
            await self._store_to_dead_letter(
                file_path,
                ERROR_TYPE_PARSE,
                "XML parsing failed",
                max_retries,
            )
            return []

        # PDF files: try primary parser, then OCR fallback
        if suffix == ".pdf":
            # Try primary parser first
            primary_result = await self._try_primary_parser(file_path, source)

            if primary_result:
                return primary_result

            # Primary parser failed, try OCR fallback
            logger.info(
                "[pipeline] Primary parser failed, trying OCR fallback for: %s",
                file_path.name,
            )
            ocr_result = await self._try_ocr_parser(file_path, source)

            if ocr_result:
                logger.info("[pipeline] OCR fallback succeeded for: %s", file_path.name)
                return ocr_result

            # Both parsers failed - store to dead letter
            await self._store_to_dead_letter(
                file_path,
                ERROR_TYPE_PARSE,
                "Both primary parser and OCR fallback failed",
                max_retries,
            )
            return []

        return []

    async def _try_primary_parser(self, file_path: Path, source: str) -> list[Any]:
        """Try primary parser (GROBID or ICMR) for PDF."""
        try:
            if source in {"icmr", "nmc_guideline", "rssdi"}:
                docs = ICMRParser(file_path).parse()
            else:
                docs = GROBIDParser(file_path, source_type=source).parse()

            if docs:
                return docs
        except Exception as e:
            logger.debug(
                "[pipeline] Primary parser failed for %s: %s", file_path.name, e
            )

        return None

    async def _try_ocr_parser(self, file_path: Path, source: str) -> list[Any]:
        """Try OCR parser as fallback for scanned PDFs."""
        try:
            # Check if it's a scanned PDF
            if not OCRParser.is_scanned(file_path):
                logger.debug(
                    "[pipeline] PDF not detected as scanned: %s", file_path.name
                )
                return None

            docs = OCRParser(file_path, source_type=source).parse()
            if docs:
                return docs
        except Exception as e:
            logger.warning("[pipeline] OCR parser failed for %s: %s", file_path.name, e)
            await self._store_to_dead_letter(file_path, ERROR_TYPE_OCR, str(e), 1)

        return None

    async def _parse_with_retry(
        self, file_path: Path, source: str, error_type: str, max_retries: int = 3
    ) -> list[Any]:
        """
        Parse file with retry logic and dead letter queue.

        Retries parsing up to max_retries times with exponential backoff.
        Stores to dead letter queue after all retries exhausted.
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                if error_type == ERROR_TYPE_PARSE:
                    # For parsing, use OCR fallback logic
                    result = await self._parse_with_ocr_fallback(file_path, source, 1)
                    if result:
                        return result
                else:
                    # For other errors, return empty (handled elsewhere)
                    return []

            except Exception as e:
                last_error = e
                logger.warning(
                    "[pipeline] Parse attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    max_retries,
                    file_path.name,
                    e,
                )

                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2**attempt)

        # All retries exhausted - store to dead letter
        await self._store_to_dead_letter(
            file_path,
            error_type,
            str(last_error) if last_error else "Unknown error",
            max_retries,
        )

        return []

    async def ingest_directory(
        self,
        directory: str,
        source: str,
        recreate_index: bool = False,
        batch_size: int = 10,
        resume: bool = True,
        reset: bool = False,
        single_file: str | None = None,
    ) -> dict[str, int]:
        """
        Ingest documents from a directory.

        Args:
            directory: Path to directory containing PDF/XML files
            source: Source label (pubmed, icmr, cochrane, etc.)
            recreate_index: Whether to recreate the vector collection
            batch_size: Number of files to process per batch
            resume: Whether to resume from last checkpoint (default: True)
            reset: Whether to reset checkpoint and start fresh (default: False)
            single_file: If provided, process only this specific file instead of directory
        """
        input_dir = Path(directory)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        # Handle single file mode - process only the specified file
        if single_file:
            single_path = Path(single_file)
            if single_path.exists() and single_path.is_file():
                files = [single_path]
            else:
                logger.warning(f"[pipeline] Single file not found or not a file: {single_file}")
                files = []
        else:
            files = sorted(
                [
                    p
                    for p in input_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".pdf", ".xml"}
                ]
            )

        logger.info("[pipeline] Found %s candidate files in %s", len(files), directory)

        self.indexer.create_collection(
            recreate=recreate_index,
            collection_name=self.settings.vector_collection_v2,
        )

        run_started_at = datetime.utcnow()
        summary = {
            "files_total": len(files),
            "files_parsed": 0,
            "documents_stored": 0,
            "chunks_created": 0,
            "chunks_indexed": 0,
            "files_failed": 0,
            "chunks_deduped": 0,
            "chunks_quality_filtered": 0,
        }

        if not files:
            return summary

        # Handle checkpoint reset
        if reset:
            await self.checkpoint.clear_checkpoint(source, directory)
            logger.info("[pipeline] Checkpoint reset for %s", source)

        # Determine starting batch index
        start_batch_index = 0
        if resume:
            resume_index = await self.checkpoint.get_resume_batch_index(
                source, directory
            )
            if resume_index == -2:
                logger.info("[pipeline] Already completed: source=%s", source)
                summary["status"] = "already_completed"
                return summary
            elif resume_index >= 0:
                start_batch_index = resume_index
                logger.info("[pipeline] Resuming from batch %d", start_batch_index)

        # Create checkpoint to track progress
        await self.checkpoint.create_checkpoint(
            source=source,
            directory=directory,
            batch_size=batch_size,
            total_files=len(files),
        )

        # Track processed files for checkpoint
        processed_files: list[str] = []

        for start in range(0, len(files), batch_size):
            batch_index = start // batch_size

            # Skip batches already completed (when resuming)
            if batch_index < start_batch_index:
                logger.info(
                    "[pipeline] Skipping batch %d (already processed)",
                    batch_index,
                )
                continue

            batch_files = files[start : start + batch_size]
            logger.info(
                "[pipeline] Processing batch %d (%s files)",
                batch_index,
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
                    chunk.metadata = chunk_meta.to_vector_payload()

                docs_for_batch.append({"doc": normalized, "enriched": enriched})
                chunks_for_batch.extend(chunks)

            if not chunks_for_batch:
                continue

            # Deduplication check
            original_count = len(chunks_for_batch)
            chunks_to_process = []
            seen_ids = set()
            for chunk in chunks_for_batch:
                should_skip, _ = await self.deduplicator.check_document(
                    {
                        "doc_id": chunk.doc_id,
                        "title": getattr(chunk, "title", ""),
                        "content": getattr(chunk, "text", ""),
                    },
                    force_reindex=False,
                )
                if not should_skip and chunk.chunk_id not in seen_ids:
                    chunks_to_process.append(chunk)
                    seen_ids.add(chunk.chunk_id)

            chunks_for_batch = chunks_to_process
            summary["chunks_deduped"] = original_count - len(chunks_for_batch)

            # Quality scoring (from v3)
            if chunks_for_batch:
                score_chunks(chunks_for_batch)
                before_quality = len(chunks_for_batch)
                chunks_for_batch = [
                    c
                    for c in chunks_for_batch
                    if c.quality_score >= self.settings.quality_score_threshold
                ]
                summary["chunks_quality_filtered"] = before_quality - len(
                    chunks_for_batch
                )

            if not chunks_for_batch:
                continue

            contextual_texts = [c.contextual_text for c in chunks_for_batch]

            # Retry logic from v3 with dead letter queue for failures
            dense_embeddings = None
            embed_error = None
            max_embed_retries = 2

            for attempt in range(max_embed_retries):
                try:
                    batch_size = 32 if attempt == 0 else 16
                    dense_embeddings = await self._run_cpu(
                        self.embedder.embed_batch,
                        contextual_texts,
                        batch_size,
                    )
                    break
                except Exception as e:
                    embed_error = e
                    logger.warning(
                        f"[pipeline] Embedding attempt {attempt + 1} failed: {e}"
                    )

            if dense_embeddings is None:
                # Embedding failed after all retries - store docs to dead letter
                logger.error(
                    "[pipeline] Embedding failed after %d retries, storing to dead letter",
                    max_embed_retries,
                )
                for entry in docs_for_batch:
                    doc_id = entry["doc"].get("doc_id", "unknown")
                    await self._store_to_dead_letter(
                        Path(entry["doc"].get("url", "/")),
                        ERROR_TYPE_EMBED,
                        f"Embedding failed after {max_embed_retries} retries: {embed_error}",
                        retry_count=max_embed_retries,
                    )
                continue  # Skip this batch

            # Validate embeddings: detect if API returned all-zero vectors
            zero_count = int(np.all(dense_embeddings == 0, axis=1).sum())
            if zero_count == len(dense_embeddings):
                logger.error(
                    "[pipeline] All %d embeddings are zero vectors — API likely failed. "
                    "Storing to dead letter. If using HuggingFace provider, switch to "
                    "EMBED_PROVIDER=local for GPU ingestion.",
                    len(dense_embeddings),
                )
                for entry in docs_for_batch:
                    await self._store_to_dead_letter(
                        Path(entry["doc"].get("url", "/")),
                        ERROR_TYPE_EMBED,
                        "All embeddings are zero vectors (API failure suspected)",
                        retry_count=max_embed_retries,
                    )
                continue
            elif zero_count > 0:
                logger.warning(
                    "[pipeline] %d/%d embeddings are zero vectors — partial API failure",
                    zero_count,
                    len(dense_embeddings),
                )

            sparse_vectors = [
                self.embedder.compute_sparse_vector(text) for text in contextual_texts
            ]

            # Vector indexing with error handling
            indexed = 0
            index_error = None
            try:
                indexed = self.indexer.upsert_chunks(
                    chunks=chunks_for_batch,
                    dense_embeddings=dense_embeddings,
                    sparse_vectors=sparse_vectors,
                    collection_name=self.settings.vector_collection_v2,
                )
            except Exception as e:
                index_error = e
                logger.error("[pipeline] Vector indexing failed: %s", e)
                # Store to dead letter if indexing fails
                for entry in docs_for_batch:
                    doc_id = entry["doc"].get("doc_id", "unknown")
                    await self._store_to_dead_letter(
                        Path(entry["doc"].get("url", "/")),
                        ERROR_TYPE_INDEX,
                        f"Vector indexing failed: {index_error}",
                        retry_count=1,
                    )
                continue  # Skip this batch

            for entry in docs_for_batch:
                await self.mongo.store_document(entry["doc"], entry["enriched"])
            await self.mongo.store_chunks(chunks_for_batch)

            summary["files_parsed"] += len(parsed_docs)
            summary["documents_stored"] += len(docs_for_batch)
            summary["chunks_created"] += len(chunks_for_batch)
            summary["chunks_indexed"] += indexed

            # Track processed files for checkpoint
            processed_files.extend(str(f) for f in batch_files)

            # Save checkpoint after successful batch completion
            await self.checkpoint.save_batch_complete(
                source=source,
                directory=directory,
                batch_index=batch_index,
                processed_files=processed_files,
            )

            logger.info(
                "[pipeline] Batch %d complete: docs=%s chunks=%s indexed=%s checkpoint saved",
                batch_index,
                len(docs_for_batch),
                len(chunks_for_batch),
                indexed,
            )

        summary["files_failed"] = summary["files_total"] - summary["files_parsed"]

        # Save metrics (from v3)
        metrics = RunMetrics(
            run_id=str(uuid4()),
            started_at=run_started_at,
            source_type=source,
            documents_fetched=summary["files_parsed"],
            documents_stored=summary["documents_stored"],
            chunks_created=summary["chunks_created"],
            chunks_embedded=summary["chunks_indexed"],
            chunks_skipped_quality=summary.get("chunks_quality_filtered", 0),
        )
        metrics.finish(status="completed")
        await self.monitor.save_run(metrics)

        # Mark checkpoint as complete
        await self.checkpoint.mark_complete(source, directory)

        logger.info("[pipeline] Ingestion complete: %s", summary)
        return summary

    async def _parse_batch(self, file_paths: list[Path], source: str) -> list[Any]:
        """
        Parse a batch of files using a configurable thread pool.

        Uses parsing_thread_workers from settings (default: 75% of CPU cores)
        to parallelize file parsing. Falls back to min of file count or
        configured workers.

        Tracks failed files and stores them to dead letter queue after batch.
        """
        loop = asyncio.get_running_loop()
        parsed: list[Any] = []
        failed_files: list[tuple[Path, str]] = []  # Track (file_path, error_message)

        # Use configured workers, but don't exceed file count
        worker_count = min(self._parse_workers, len(file_paths))

        logger.debug("[pipeline] Parsing batch with %d workers", worker_count)

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            tasks = [
                loop.run_in_executor(pool, self._parse_file, file_path, source)
                for file_path in file_paths
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results and track failures
        for file_path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error(
                    "[pipeline] Parse worker failed: %s - %s", file_path.name, result
                )
                failed_files.append((file_path, str(result)))
                continue
            if not result:
                # Empty result means parser failed to extract content
                logger.warning(
                    "[pipeline] No content extracted from: %s", file_path.name
                )
                failed_files.append((file_path, "No content extracted"))
                continue
            parsed.extend(result)

        # Store failed files to dead letter queue
        if failed_files and self._dead_letter_enabled:
            for file_path, error_msg in failed_files:
                await self._store_to_dead_letter(
                    file_path,
                    ERROR_TYPE_PARSE,
                    error_msg,
                    retry_count=1,
                )

        return parsed

    def _parse_file(self, file_path: Path, source: str) -> list[Any]:
        """
        Parse a file with OCR fallback support.

        For PDF files: tries primary parser first (GROBID/ICMR), then falls back
        to OCR parser if primary fails or returns empty results.
        For XML files: routes to the appropriate parser based on source type.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".xml":
            # Route to the correct XML parser based on source type
            if source == "medquad":
                from src.ingestion.parsers.medquad import MedQuADParser
                try:
                    docs = MedQuADParser(file_path).parse()
                    if docs:
                        return docs
                except (RuntimeError, ValueError, TypeError, OSError) as exc:
                    logger.warning("[pipeline] MedQuAD parser failed for %s: %s", file_path.name, exc)
                return []
            return self._parse_pubmed_xml_file(file_path)

        if suffix != ".pdf":
            return []

        # Try primary parser first
        try:
            if source in {"icmr", "nmc_guideline", "rssdi"}:
                docs = ICMRParser(file_path).parse()
            else:
                docs = GROBIDParser(file_path, source_type=source).parse()

            if docs:
                return docs
        except (RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.warning(
                "[pipeline] Primary parser failed for %s: %s", file_path.name, exc
            )
            # Continue to OCR fallback

        # Primary parser failed or returned empty - try OCR fallback
        try:
            if OCRParser.is_scanned(file_path):
                logger.info("[pipeline] Trying OCR fallback for: %s", file_path.name)
                ocr_docs = OCRParser(file_path, source_type=source).parse()
                if ocr_docs:
                    logger.info(
                        "[pipeline] OCR fallback succeeded for: %s", file_path.name
                    )
                    return ocr_docs
        except (RuntimeError, ValueError, TypeError, OSError) as ocr_exc:
            logger.error(
                "[pipeline] OCR fallback failed for %s: %s", file_path.name, ocr_exc
            )

        # Both parsers failed - will be tracked for dead letter queue
        logger.error("[pipeline] All parsers failed for: %s", file_path.name)
        return []

    def _parse_pubmed_xml_file(self, file_path: Path) -> list[dict[str, Any]]:
        try:
            tree = ET.parse(file_path)
        except ET.ParseError as exc:
            logger.error("[pipeline] Invalid XML %s: %s", file_path, exc)
            return []

        root = tree.getroot()

        # Collect regular PubMed articles
        articles = root.findall(".//PubmedArticle")
        if not articles and root.tag.endswith("PubmedArticle"):
            articles = [root]

        # Also collect book articles (StatPearls use PubmedBookArticle format)
        book_articles = root.findall(".//PubmedBookArticle")

        docs: list[dict[str, Any]] = []

        # Parse regular PubMed articles
        for article in articles:
            doc = self._parse_pubmed_article(article, file_path)
            if doc:
                docs.append(doc)

        # Parse book articles (StatPearls, books, etc.)
        for book_article in book_articles:
            doc = self._parse_pubmed_book_article(book_article, file_path)
            if doc:
                docs.append(doc)

        return docs

    def _parse_pubmed_article(
        self, article: ET.Element, file_path: Path
    ) -> dict[str, Any] | None:
        """Parse a standard PubmedArticle XML element."""
        citation = article.find("MedlineCitation")
        if citation is None:
            return None
        article_data = citation.find("Article")
        if article_data is None:
            return None

        title = (article_data.findtext("ArticleTitle") or "").strip()
        abstract_nodes = article_data.findall(".//AbstractText")
        abstract = " ".join(
            (node.text or "").strip()
            for node in abstract_nodes
            if (node.text or "").strip()
        )
        if not title and not abstract:
            return None

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

        return {
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

    def _parse_pubmed_book_article(
        self, book_article: ET.Element, file_path: Path
    ) -> dict[str, Any] | None:
        """Parse a PubmedBookArticle XML element (StatPearls, books, etc.)."""
        book_document = book_article.find("BookDocument")
        if book_document is None:
            return None

        # Title from ArticleTitle
        title = (book_document.findtext("ArticleTitle") or "").strip()

        # Abstract from Abstract/AbstractText
        abstract_nodes = book_document.findall(".//AbstractText")
        abstract = " ".join(
            (node.text or "").strip()
            for node in abstract_nodes
            if (node.text or "").strip()
        )

        if not title and not abstract:
            return None

        # PMID from BookDocument/PMID
        pmid = (book_document.findtext("PMID") or "").strip()

        # Book info
        book = book_document.find("Book")
        journal = ""
        year = 0
        if book is not None:
            book_title = (book.findtext("BookTitle") or "").strip()
            if book_title:
                journal = book_title
            pub_date = book.find("PubDate")
            if pub_date is not None:
                year_text = (pub_date.findtext("Year") or "").strip()
                year = int(year_text) if year_text.isdigit() else 0

        # DOI
        doi = None
        for id_node in book_document.findall(".//ArticleId"):
            if id_node.attrib.get("IdType") == "doi":
                doi = (id_node.text or "").strip() or None
                break
            elif id_node.attrib.get("IdType") == "bookaccession":
                # Use book accession as fallback identifier
                if not doi:
                    doi = (id_node.text or "").strip() or None

        # MeSH terms from BookDocument
        mesh_terms = [
            (n.text or "").strip()
            for n in book_document.findall(".//MeshHeading/DescriptorName")
            if (n.text or "").strip()
        ]

        # Keywords
        keywords = [
            (n.text or "").strip()
            for n in book_document.findall(".//Keyword")
            if (n.text or "").strip()
        ]

        # Authors
        authors: list[str] = []
        for author in book_document.findall(".//Author"):
            last = (author.findtext("LastName") or "").strip()
            fore = (author.findtext("ForeName") or "").strip()
            full_name = f"{last} {fore}".strip()
            if full_name:
                authors.append(full_name)

        # Sections from BookDocument/Sections
        sections: list[dict[str, str]] = []
        for section in book_document.findall(".//Section"):
            section_title = (section.findtext("SectionTitle") or "").strip()
            section_content_nodes = section.findall(".//AbstractText")
            section_content = " ".join(
                (n.text or "").strip()
                for n in section_content_nodes
                if (n.text or "").strip()
            )
            if section_title or section_content:
                sections.append({"title": section_title, "content": section_content})

        content = f"{title}\n\n{abstract}".strip()
        doc_id = (
            f"pmid_{pmid}"
            if pmid
            else self._hash_doc_id(str(file_path), title, abstract)
        )

        return {
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
            "sections": sections,
            "url": str(file_path.resolve()),
            "source_type": "pubmed",
        }

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

        content_parts = [out.get("title", ""), out.get("abstract", ""), out.get("content", "")]
        content_hash = hashlib.sha256("|".join(p.strip() for p in content_parts if p.strip()).encode("utf-8")).hexdigest()
        out["content_hash"] = content_hash

        return out

    def _hash_doc_id(self, *parts: str) -> str:
        joined = "|".join(part or "" for part in parts)
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
        return f"doc_{digest}"

    async def _run_cpu(self, func, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args))
