"""
Re-ingestion Script v2
Clears old v1 data, re-ingests all ICMR PDFs using the v2 pipeline.
Run this once after Phase 2 is complete.
"""

import asyncio
import importlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _import_attr(module_name: str, attr_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


async def clear_v1_data():
    """Remove all v1-ingested ICMR documents and their chunks from MongoDB and Qdrant."""
    get_db = _import_attr("src.ingestion.document_db", "get_db")
    get_qdrant = _import_attr("src.ingestion.vector_db", "get_qdrant")
    get_settings = _import_attr("src.core.config", "get_settings")

    settings = get_settings()
    db = get_db()

    logger.info("Clearing v1 ICMR data from MongoDB...")
    result = await db["documents"].delete_many(
        {
            "source_type": "icmr",
            "$or": [
                {"parser_version": "v1"},
                {"parser_version": {"$exists": False}},
            ],
        }
    )
    logger.info("Deleted %s v1 documents", result.deleted_count)

    chunk_result = await db["chunks"].delete_many(
        {
            "source_type": "icmr",
            "$or": [
                {"parser_version": "v1"},
                {"parser_version": {"$exists": False}},
            ],
        }
    )
    logger.info("Deleted %s v1 chunks", chunk_result.deleted_count)

    # Clear Qdrant collection entirely and recreate
    client = get_qdrant()
    client.delete_collection(settings.qdrant_collection)
    logger.info("Qdrant collection cleared")


async def main():
    icmr_parser_cls = _import_attr("src.ingestion.parsers.icmr", "ICMRParser")
    ocr_parser_cls = _import_attr("src.ingestion.parsers.ocr", "OCRParser")
    run_pipeline_v2 = _import_attr("src.ingestion.pipeline_v2", "run_pipeline_v2")

    icmr_dir = Path("data/raw/icmr")
    if not icmr_dir.exists():
        print("data/raw/icmr/ not found. Add ICMR PDFs first.")
        return

    pdf_files = sorted(icmr_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw/icmr/")
        return

    print(f"Found {len(pdf_files)} PDFs")
    print("Clearing v1 data...")
    await clear_v1_data()

    all_documents = []
    parsed = 0
    ocr_used = 0
    failed = 0

    for pdf_path in pdf_files:
        # Try pdfplumber first, fall back to OCR for scanned PDFs
        parser = icmr_parser_cls(pdf_path)
        docs = parser.parse()

        if not docs or not docs[0].content.strip():
            logger.warning("pdfplumber got no text, trying OCR: %s", pdf_path.name)
            parser = ocr_parser_cls(pdf_path, source_type="icmr")
            docs = parser.parse()
            if docs:
                ocr_used += 1

        if docs:
            # Mark as India-specific for ICMR sources
            for doc in docs:
                doc.is_india_specific = True
                doc.parser_version = "v2"
            all_documents.extend(docs)
            parsed += 1
        else:
            logger.error("Failed to parse: %s", pdf_path.name)
            failed += 1

    print("\nParsing complete:")
    print(f"  Parsed: {parsed}")
    print(f"  OCR used: {ocr_used}")
    print(f"  Failed: {failed}")
    print(f"\nStarting v2 pipeline for {len(all_documents)} documents...")

    summary = await run_pipeline_v2(all_documents)

    print("\nRe-ingestion complete:")
    print(f"  Documents stored: {summary['documents_stored']}")
    print(f"  Chunks created:   {summary['chunks_created']}")
    print(f"  Chunks embedded:  {summary['chunks_embedded']}")
    print(f"  Noise skipped:    {summary['chunks_skipped_noise']}")


if __name__ == "__main__":
    asyncio.run(main())
