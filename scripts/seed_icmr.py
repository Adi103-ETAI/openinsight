import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.parsers.icmr import ICMRParser
from src.ingestion.pipeline_v4 import IngestionPipelineV4


def main() -> None:
    icmr_dir = Path("data/raw/icmr")
    if not icmr_dir.exists():
        print("Directory not found: data/raw/icmr/. Create it and add ICMR PDF files first.")
        return

    pdf_files = sorted(icmr_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/raw/icmr/. Add at least one .pdf and rerun.")
        return

    print(f"Found {len(pdf_files)} PDF files")

    pipeline = IngestionPipelineV4()

    summary = asyncio.run(
        pipeline.ingest_directory(
            directory=str(icmr_dir),
            source="icmr",
            recreate_index=False,
            batch_size=10,
        )
    )

    print("\n=== ICMR Ingestion Complete ===")
    print(f"Files total: {summary['files_total']}")
    print(f"Files parsed: {summary['files_parsed']}")
    print(f"Documents stored: {summary['documents_stored']}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Chunks indexed: {summary['chunks_indexed']}")
    print(f"Chunks deduped: {summary.get('chunks_deduped', 0)}")
    print(f"Chunks quality filtered: {summary.get('chunks_quality_filtered', 0)}")
    print(f"Files failed: {summary['files_failed']}")


if __name__ == "__main__":
    main()