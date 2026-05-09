"""
Re-ingestion Script v4
Re-ingests all ICMR PDFs using the v4 pipeline with deduplication and quality scoring.
Run this to re-ingest all documents.
"""

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


async def main():
    from src.ingestion.pipeline_v4 import IngestionPipelineV4

    icmr_dir = Path("data/raw/icmr")
    if not icmr_dir.exists():
        print("data/raw/icmr/ not found. Add ICMR PDFs first.")
        return

    pdf_files = sorted(icmr_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/raw/icmr/")
        return

    print(f"Found {len(pdf_files)} PDFs")

    pipeline = IngestionPipelineV4()

    summary = await pipeline.ingest_directory(
        directory=str(icmr_dir),
        source="icmr",
        recreate_index=False,
        batch_size=10,
    )

    print("\n=== Re-ingestion Complete ===")
    print(f"Files total: {summary['files_total']}")
    print(f"Files parsed: {summary['files_parsed']}")
    print(f"Documents stored: {summary['documents_stored']}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Chunks indexed: {summary['chunks_indexed']}")
    print(f"Chunks deduped: {summary.get('chunks_deduped', 0)}")
    print(f"Chunks quality filtered: {summary.get('chunks_quality_filtered', 0)}")
    print(f"Files failed: {summary['files_failed']}")


if __name__ == "__main__":
    asyncio.run(main())