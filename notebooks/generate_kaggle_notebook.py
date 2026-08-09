#!/usr/bin/env python3
"""Generate kaggle_ingestion_v2.ipynb from the .py script.

The .py file uses '# === CELL N ===' markers to delimit cells.
The module-level docstring (between the first pair of triple-quotes) is
excluded from the notebook — it's documentation for the .py file, not
notebook content.
"""
import json
from pathlib import Path

py_path = Path(__file__).parent / "kaggle_ingestion_v2.py"
py_content = py_path.read_text()

# --- Parse the .py file into cells -----------------------------------------

cells_raw: list[tuple[str | None, str]] = []  # (cell_marker, source)
current_lines: list[str] = []
current_marker: str | None = None

# Skip the module docstring (everything between the first pair of """)
lines = py_content.splitlines()
i = 0
# Skip leading blank lines
while i < len(lines) and not lines[i].strip():
        i += 1
# Skip module docstring if present
if i < len(lines) and lines[i].strip().startswith('"""'):
        # Find the closing """
        if lines[i].count('"""') >= 2:
                # Single-line docstring
                i += 1
        else:
                # Multi-line docstring — find closing
                i += 1
                while i < len(lines):
                        if '"""' in lines[i]:
                                i += 1
                                break
                        i += 1

# Now parse cells
for line in lines[i:]:
        if line.startswith("# === CELL"):
                if current_lines:
                        cells_raw.append((current_marker, "\n".join(current_lines)))
                current_marker = line
                current_lines = []
        else:
                current_lines.append(line)

if current_lines:
        cells_raw.append((current_marker, "\n".join(current_lines)))

# --- Build the notebook ----------------------------------------------------

notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
                "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3",
                },
                "language_info": {
                        "name": "python",
                        "version": "3.10.0",
                },
                "kaggle": {
                        "accelerator": "gpu-t4x2",
                        "dataSources": [],
                        "isGpuEnabled": True,
                        "isInternetEnabled": True,
                        "language": "python",
                },
        },
        "cells": [],
}

# Intro markdown cell
intro_md = """# OpenInsight — Kaggle Ingestion

**Purpose**: Run the OpenInsight ingestion pipeline on Kaggle's free GPU to populate Milvus + MongoDB with indexed medical content.

## Setup (one-time)

1. **MongoDB Atlas** (free, 512MB) — create M0 cluster in **Mumbai region**, get connection string
2. **Zilliz Cloud** (free, 2M vectors) — create cluster in **Mumbai region**, get URI + token
3. **NCBI API Key** (free) — sign up at ncbi.nlm.nih.gov/account, raises rate from 3→10 req/sec
4. **Kaggle Secrets** — add: `MONGODB_URL`, `ZILLIZ_URI`, `ZILLIZ_TOKEN`, `NCBI_API_KEY`
5. **Kaggle Settings** — GPU T4 × 2, Internet ON, Persistence ON

## Usage

Edit the configuration cell below to pick sources + limits, then Run All.

See `notebooks/KAGGLE_INGESTION_README.md` for full documentation.
"""

notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": intro_md,
})

# Add code cells from the .py file
for marker, source in cells_raw:
        if not source.strip():
                continue  # skip empty cells
        notebook["cells"].append({
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source,
        })

# Write the notebook
ipynb_path = py_path.with_suffix(".ipynb")
ipynb_path.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"✓ Generated {ipynb_path} ({len(notebook['cells'])} cells)")
