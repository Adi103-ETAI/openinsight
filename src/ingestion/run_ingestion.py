from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


SOURCES = ["pubmed", "icmr", "cochrane", "nmc_guideline", "rssdi", "who", "cdc", "statpearls"]


# Handle source list before importing pipeline
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--source-list":
    print("Available sources:")
    for s in SOURCES:
        print(f"  - {s}")
    sys.exit(0)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenInsight ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s --dir ./data/pdfs --source pubmed
  %(prog)s --dir ./data/pdfs --source icmr --workers 8 --recreate
  %(prog)s --dir ./data/pdfs --source cochrane --dry-run
  %(prog)s --single ./data/pdfs/sample.pdf --source pubmed
  %(prog)s --source-list

Available sources: {', '.join(SOURCES)}
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dir", 
        help="Directory containing PDF/XML files"
    )
    input_group.add_argument(
        "--single",
        help="Process a single file instead of directory"
    )
    input_group.add_argument(
        "--source-list",
        action="store_true",
        help="List available sources and exit"
    )
    
    # Source
    parser.add_argument(
        "--source",
        choices=SOURCES,
        help="Source label for metadata and parser routing"
    )
    
    # Processing options
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=6,
        help="Number of parallel workers (default: 6)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=10,
        help="Number of files to process per batch (default: 10)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of files to process"
    )
    
    # Pipeline control
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse & chunk without embedding/indexing"
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding (just parse & store in Mongo)"
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip vector indexing (just store in Mongo)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the vector collection before ingestion"
    )
    
    # Checkpoint options
    parser.add_argument(
        "--resume/--no-resume",
        default=True,
        help="Enable/disable checkpoint resume (default: enabled)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset checkpoint and start fresh"
    )
    
    # Output options
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed statistics"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


async def _main_async() -> args:
    args = _build_parser().parse_args()
    
    # Handle source list
    if args.source_list:
        print("Available sources:")
        for s in SOURCES:
            print(f"  - {s}")
        sys.exit(0)
    
    # Validate source is provided
    if not args.source:
        print("Error: --source is required", file=sys.stderr)
        sys.exit(1)
    
    # Handle single file
    if args.single:
        args.dir = str(Path(args.single).parent)
        args.single = True
    else:
        args.single = False
    
    # Build pipeline kwargs
    pipeline_kwargs = {
        "source": args.source,
        "recreate_index": args.recreate,
        "batch_size": max(1, args.batch_size),
        "resume": args.resume,
        "reset": args.reset,
    }
    
    if args.limit:
        pipeline_kwargs["limit"] = args.limit
    
    # Run pipeline
    pipeline = IngestionPipeline()
    
    if args.dry_run:
        print("🔍 Dry run mode - parsing & chunking only")
        summary = await pipeline.ingest_directory(
            directory=args.dir,
            skip_embed=True,
            skip_index=True,
            **pipeline_kwargs
        )
    elif args.skip_embed:
        print("📝 Parse & store mode - skipping embedding")
        summary = await pipeline.ingest_directory(
            directory=args.dir,
            skip_embed=True,
            **pipeline_kwargs
        )
    elif args.skip_index:
        print("💾 Parse & embed mode - skipping vector indexing")
        summary = await pipeline.ingest_directory(
            directory=args.dir,
            skip_index=True,
            **pipeline_kwargs
        )
    else:
        print("🚀 Full pipeline - parse, chunk, embed, index")
        summary = await pipeline.ingest_directory(
            directory=args.dir,
            **pipeline_kwargs
        )
    
    # Output results
    print("\n" + "="*50)
    print("✅ Ingestion complete")
    print("="*50)
    
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    if args.stats and hasattr(pipeline, 'get_stats'):
        stats = pipeline.get_stats()
        print("\n📊 Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(_main_async())