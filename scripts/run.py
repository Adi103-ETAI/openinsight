#!/usr/bin/env python3
"""
OpenInsight Unified Ingestion Runner

A simple wrapper to run the ingestion pipeline with sensible defaults.

Usage:
    python run.py --source pubmed --dir ./data/pdfs
    python run.py --source icmr --dir ./data/pdfs --workers 8
    python run.py --source cochrane --dry-run
    python run.py --help

Quick Examples:
    python run.py pubmed ./data/pdfs                    # Basic
    python run.py icmr ./data/pdfs -w 8 --recreate      # Parallel, recreate index
    python run.py who ./pdfs --dry-run                  # Test run
    python run.py pubmed ./pdfs --limit 100              # Limit files
    python run.py                                        # Interactive mode
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SOURCES = ["pubmed", "icmr", "cochrane", "nmc_guideline", "rssdi", "who", "cdc", "statpearls"]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_ingestion(args: argparse.Namespace) -> int:
    """Run the ingestion pipeline."""
    cmd = [
        sys.executable,
        "-m", "src.ingestion.run_ingestion",
    ]
    
    # Add source
    if args.source:
        cmd.extend(["--source", args.source])
    
    # Add directory
    if args.dir:
        cmd.extend(["--dir", args.dir])
    
    # Add single file
    if args.single:
        cmd.extend(["--single", args.single])
    
    # Add workers
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
    
    # Add batch size
    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    
    # Add limit
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    # Add flags
    if args.recreate:
        cmd.append("--recreate")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_embed:
        cmd.append("--skip-embed")
    if args.skip_index:
        cmd.append("--skip-index")
    if args.no_resume:
        cmd.append("--no-resume")
    if args.reset:
        cmd.append("--reset")
    if args.stats:
        cmd.append("--stats")
    if args.verbose:
        cmd.append("--verbose")
    
    # Change to project root and run
    os.chdir(get_project_root())
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    return subprocess.call(cmd)


def interactive_mode() -> None:
    """Run in interactive mode."""
    print("🩺 OpenInsight Ingestion - Interactive Mode")
    print("=" * 50)
    
    # Get source
    print("\nAvailable sources:")
    for i, s in enumerate(SOURCES, 1):
        print(f"  {i}. {s}")
    print(f"  0. List all sources")
    
    try:
        choice = input("\nSelect source (number or name): ").strip()
    except EOFError:
        sys.exit(1)
    
    if choice == "0":
        print("\nAvailable sources:", ", ".join(SOURCES))
        sys.exit(0)
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(SOURCES):
            source = SOURCES[idx]
        else:
            print("Invalid selection")
            sys.exit(1)
    else:
        source = choice.lower()
        if source not in SOURCES:
            print(f"Unknown source: {source}")
            sys.exit(1)
    
    # Get directory
    default_dir = "./data/pdfs"
    dir_input = input(f"Directory (default: {default_dir}): ").strip() or default_dir
    
    if not Path(dir_input).exists():
        print(f"Error: Directory '{dir_input}' does not exist")
        sys.exit(1)
    
    # Get workers
    workers = input("Workers (default: 6): ").strip() or "6"
    
    # Get extra options
    print("\nOptions (press Enter for default):")
    options = []
    if input("  Recreate index? [y/N]: ").strip().lower() == "y":
        options.append("--recreate")
    if input("  Dry run? [y/N]: ").strip().lower() == "y":
        options.append("--dry-run")
    if input("  Show stats? [y/N]: ").strip().lower() == "y":
        options.append("--stats")
    
    # Build args
    class Args:
        source = source
        dir = dir_input
        single = None
        workers = int(workers)
        batch_size = 10
        limit = None
        recreate = "--recreate" in options
        dry_run = "--dry-run" in options
        skip_embed = False
        skip_index = False
        no_resume = False
        reset = False
        stats = "--stats" in options
        verbose = False
    
    run_ingestion(Args())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenInsight Unified Ingestion Runner",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quick Examples:
  python run.py pubmed ./data/pdfs                    # Basic
  python run.py icmr ./data/pdfs -w 8 --recreate    # Parallel
  python run.py who ./pdfs --dry-run                 # Test run
  python run.py pubmed ./pdfs --limit 100            # Limit files
  python run.py                                      # Interactive mode

Available sources: pubmed, icmr, cochrane, nmc_guideline, rssdi, who, cdc, statpearls
        """
    )
    
    # Allow positional source and directory
    parser.add_argument(
        "source",
        nargs="?",
        help="Source (pubmed, icmr, cochrane, etc.)"
    )
    parser.add_argument(
        "dir",
        nargs="?",
        help="Directory containing files"
    )
    
    # Flags
    parser.add_argument("-w", "--workers", type=int, help="Number of workers")
    parser.add_argument("-b", "--batch-size", type=int, help="Batch size")
    parser.add_argument("-l", "--limit", type=int, help="Limit files")
    parser.add_argument("--recreate", action="store_true", help="Recreate index")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing")
    parser.add_argument("--no-resume", action="store_true", help="No resume")
    parser.add_argument("--reset", action="store_true", help="Reset checkpoint")
    parser.add_argument("--stats", action="store_true", help="Show stats")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument("--single", help="Single file")
    parser.add_argument("--help", action="store_true", help="Show help")
    
    args = parser.parse_args()
    
    # Show help if requested or no args
    if args.help or (not args.source and not args.dir and sys.stdin.isatty()):
        parser.print_help()
        print("\n" + "="*50)
        print("Or run in interactive mode:")
        print("  python run.py")
        print("="*50)
        sys.exit(0)
    
    # Run interactive if no args
    if not args.source and not args.dir:
        interactive_mode()
        sys.exit(0)
    
    # Validate source
    if args.source and args.source not in SOURCES:
        print(f"Unknown source: {args.source}")
        print(f"Available: {', '.join(SOURCES)}")
        sys.exit(1)
    
    # Validate directory
    if args.dir and not Path(args.dir).exists():
        print(f"Directory does not exist: {args.dir}")
        sys.exit(1)
    
    sys.exit(run_ingestion(args))


if __name__ == "__main__":
    main()