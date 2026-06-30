"""Maintenance helpers for the OpenInsight backend repository.

Run this script to remove Python __pycache__ directories from the project tree
when you need a clean working copy.

Examples:
    python scripts/maintainence.py
    python scripts/maintainence.py --root .
    python scripts/maintainence.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build"}


def remove_pycache(root: Path, dry_run: bool = False) -> list[Path]:
    """Delete all __pycache__ directories below root.

    Returns the list of directories removed, or the directories that would be
    removed in dry-run mode.
    """

    removed: list[Path] = []

    for current_root, dirnames, _ in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]

        current_path = Path(current_root)
        for directory_name in list(dirnames):
            if directory_name != "__pycache__":
                continue

            path = current_path / directory_name
            removed.append(path)
            if dry_run:
                continue

            shutil.rmtree(path)

            dirnames.remove(directory_name)

    return removed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove Python __pycache__ directories from the repository."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to clean (default: project root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the directories that would be removed without deleting them.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    removed = remove_pycache(root, dry_run=args.dry_run)

    if not removed:
        print(f"No __pycache__ directories found under {root}")
        return

    action = "Would remove" if args.dry_run else "Removed"
    for path in removed:
        print(f"{action}: {path}")

    print(
        f"{action.lower()} {len(removed)} __pycache__ director{'y' if len(removed) == 1 else 'ies'}."
    )


if __name__ == "__main__":
    main()
