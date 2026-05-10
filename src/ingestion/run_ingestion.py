from __future__ import annotations

import argparse
import asyncio

from src.ingestion.pipeline import IngestionPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenInsight ingestion pipeline"
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
        help="Recreate the vector collection before ingestion",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process per batch (default: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint/resume - start from beginning (default: resume enabled)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start fresh",
    )
    return parser


async def _main_async() -> None:
    args = _build_parser().parse_args()
    pipeline = IngestionPipeline()
    summary = await pipeline.ingest_directory(
        directory=args.dir,
        source=args.source,
        recreate_index=args.recreate,
        batch_size=max(1, args.batch_size),
        resume=not args.no_resume,
        reset=args.reset,
    )
    print("Ingestion complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(_main_async())