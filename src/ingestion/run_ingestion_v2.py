from __future__ import annotations

import argparse
import asyncio

from src.ingestion.pipeline_v4 import IngestionPipelineV4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenInsight v2 ingestion pipeline"
    )
    parser.add_argument(
        "--dir", required=True, help="Directory containing PDF/XML files"
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["pubmed", "icmr", "cochrane", "nmc_guideline", "rssdi", "who"],
        help="Source label for metadata and parser routing",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the v2 Qdrant collection before ingestion",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process per batch (default: 10)",
    )
    return parser


async def _main_async() -> None:
    args = _build_parser().parse_args()
    pipeline = IngestionPipelineV4()
    summary = await pipeline.ingest_directory(
        directory=args.dir,
        source=args.source,
        recreate_index=args.recreate,
        batch_size=max(1, args.batch_size),
    )
    print("v2 ingestion complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(_main_async())
