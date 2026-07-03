#!/usr/bin/env python3
"""Generate kaggle_ingestion_v2.ipynb from the .py script."""
import json
import sys
from pathlib import Path

# Read the .py file
py_path = Path(__file__).parent / "kaggle_ingestion_v2.py"
py_content = py_path.read_text()

# Split into cells by the === CELL N === markers
cells_raw = []
current_cell_lines = []
current_cell_num = None
current_cell_type = "code"

for line in py_content.splitlines():
    if line.startswith("# === CELL"):
        if current_cell_lines:
            cells_raw.append({
                "num": current_cell_num,
                "type": current_cell_type,
                "source": "\n".join(current_cell_lines),
            })
        current_cell_num = line
        current_cell_lines = []
        # Extract cell type from the header
        if "Configuration" in line or "Configuration" in line:
            current_cell_type = "code"
        else:
            current_cell_type = "code"
    elif line.startswith('"""') and current_cell_num is None:
        # Module docstring — skip
        continue
    else:
        current_cell_lines.append(line)

if current_cell_lines:
    cells_raw.append({
        "num": current_cell_num,
        "type": current_cell_type,
        "source": "\n".join(current_cell_lines),
    })

# Build the notebook
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

# Add a markdown intro cell
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

# Add cells from the .py file
for cell in cells_raw:
    if cell["source"].strip():
        notebook["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": cell["source"],
        })

# Write the notebook
ipynb_path = py_path.with_suffix(".ipynb")
ipynb_path.write_text(json.dumps(notebook, indent=1))
print(f"✓ Generated {ipynb_path} ({len(notebook['cells'])} cells)")
