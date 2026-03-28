# OpenInsight — Agent Build Context
## Table Parsing Fix — ICMR Parser

You are fixing a specific bug in the OpenInsight ingestion pipeline. Clinical tables in ICMR PDFs are being extracted as flat garbled text — numbers and drug names concatenated without structure. Doctors cannot use this data.

**Example of the problem:**
```
"Amikacin 27 49 38 21 35 Aztreonam 62 55 30 48 Cefepime 52 57 20 41 sulbactam 39 41 30 38 Ceftazidime 64 51 51 23 47 Imipenem 17 54 48 25 37..."
```

This is an antimicrobial resistance table that should look like:

```
Antimicrobial Resistance Data (ICMR AMR 2014):
| Drug | PGIMER | AIIMS | JIPMER | Chandigarh | New Delhi |
|------|--------|-------|--------|------------|-----------|
| Amikacin | 27% | 49% | 38% | 21% | 35% |
| Aztreonam | 62% | 55% | 30% | 48% | - |
```

---

## Files To Modify

Only touch these files:
- `src/ingestion/parsers/icmr.py` — main fix goes here
- `src/utils/chunker_v2.py` — add table-aware chunking

Do not touch any other files.

---

## Current `src/ingestion/parsers/icmr.py` Structure

The current parser uses `pdfplumber` and calls `page.extract_text()` for every page. This loses all table structure. Tables need to be extracted separately using `page.extract_tables()` and formatted before being merged with the text.

---

## The Fix

### Step 1 — Update `src/ingestion/parsers/icmr.py`

Replace the page extraction logic inside the `parse()` method. The current logic does:

```python
for page in pdf.pages:
    page_text = page.extract_text() or ""
    pages_text.append(page_text)
```

Replace the entire `parse()` method body with this improved version that handles tables separately:

```python
def _format_table(self, table: list[list]) -> str:
    """
    Convert a pdfplumber table (list of rows, each row is list of cells)
    into a readable markdown-style text table.
    Skips empty tables and single-cell tables.
    """
    if not table or len(table) < 2:
        return ""

    # Clean cells — replace None with empty string
    cleaned = []
    for row in table:
        cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
        cleaned.append(cleaned_row)

    # Skip if table is mostly empty
    non_empty = sum(1 for row in cleaned for cell in row if cell)
    if non_empty < 4:
        return ""

    # Calculate column widths
    col_widths = []
    num_cols = max(len(row) for row in cleaned)
    for col_idx in range(num_cols):
        width = max(
            (len(row[col_idx]) if col_idx < len(row) else 0)
            for row in cleaned
        )
        col_widths.append(max(width, 4))

    # Format as aligned text table
    lines = []
    for row_idx, row in enumerate(cleaned):
        # Pad row to num_cols
        padded = row + [""] * (num_cols - len(row))
        line = " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(padded))
        lines.append(line)
        # Add separator after header row
        if row_idx == 0:
            separator = "-+-".join("-" * col_widths[i] for i in range(num_cols))
            lines.append(separator)

    return "\n".join(lines)


def _extract_page_content(self, page) -> str:
    """
    Extract text and tables from a single page.
    Tables are extracted first with structure preserved,
    then remaining text is extracted with table bounding boxes masked.
    """
    import pdfplumber

    page_parts = []

    # Extract tables with structure
    tables = page.find_tables()
    table_bboxes = []

    for table in tables:
        table_data = table.extract()
        if table_data:
            formatted = self._format_table(table_data)
            if formatted:
                page_parts.append(f"\n[TABLE]\n{formatted}\n[/TABLE]\n")
            table_bboxes.append(table.bbox)

    # Extract text, masking out table areas to avoid duplication
    if table_bboxes:
        # Crop page to exclude table areas and extract remaining text
        remaining_text_parts = []
        try:
            # Get text outside table bounding boxes
            filtered_page = page
            for bbox in table_bboxes:
                # Extract text excluding this table region
                pass
            # Simpler approach: extract all text then remove table content
            full_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            remaining_text_parts.append(full_text)
        except Exception:
            full_text = page.extract_text() or ""
            remaining_text_parts.append(full_text)

        # Combine text parts
        text = "\n".join(remaining_text_parts).strip()
    else:
        text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

    if text:
        page_parts.insert(0, text)

    return "\n".join(page_parts)


def parse(self) -> list[DocumentRecord]:
    if not self.file_path.exists():
        logger.error(f"ICMR file not found: {self.file_path}")
        return []

    try:
        import pdfplumber
        with pdfplumber.open(self.file_path) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"Parsing ICMR PDF: {self.file_path.name} ({page_count} pages)")

            pages_content = []
            table_count = 0

            for page in pdf.pages:
                try:
                    # Count tables on this page
                    page_tables = page.find_tables()
                    table_count += len(page_tables)

                    content = self._extract_page_content(page)
                    if content.strip():
                        pages_content.append(content)
                except Exception as e:
                    logger.debug(f"Page extraction error: {e}")
                    # Fallback to simple text extraction for this page
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_content.append(text)

            logger.info(f"Found {table_count} tables in {self.file_path.name}")

    except Exception as exc:
        logger.error(f"Failed to read ICMR PDF {self.file_path}: {exc}")
        return []

    raw_content = "\n\n".join(pages_content)
    cleaned_content = self._clean_text(raw_content)

    if not cleaned_content:
        logger.error(f"No parseable text extracted from: {self.file_path}")
        return []

    title = self.file_path.stem.replace("_", " ")
    document = DocumentRecord(
        source_type=self.source_type,
        title=title,
        content=cleaned_content,
        url=str(self.file_path.resolve()),
        condition_tags=self._extract_condition_tags(),
        is_india_specific=True,
        parser_version="v2",
    )
    logger.info(f"Parsed {self.file_path.name}: {len(cleaned_content)} chars, {table_count} tables")
    return [document]
```

Also update `_clean_text()` to preserve table markers. Find the current `_clean_text` method and add this — do not strip lines inside `[TABLE]...[/TABLE]` blocks:

```python
def _clean_text(self, text: str) -> str:
    """Clean text while preserving table structure markers."""
    lines = text.splitlines()
    cleaned_lines = []
    inside_table = False

    for raw_line in lines:
        # Track table blocks — preserve them as-is
        if raw_line.strip() == "[TABLE]":
            inside_table = True
            cleaned_lines.append(raw_line)
            continue
        if raw_line.strip() == "[/TABLE]":
            inside_table = False
            cleaned_lines.append(raw_line)
            continue
        if inside_table:
            # Preserve table content without cleaning
            cleaned_lines.append(raw_line)
            continue

        # Normal text cleaning
        import re
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if len(line) < 20 and not re.search(r"[.!?;:]$", line):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
```

---

### Step 2 — Update chunker to handle table blocks

In `src/utils/chunker_v2.py`, update `_split_by_paragraphs()` to keep table blocks intact — never split a table across chunks:

Find the `_split_by_paragraphs` function and replace it with:

```python
def _split_by_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs.
    Table blocks ([TABLE]...[/TABLE]) are kept as single units — never split.
    """
    import re
    segments = []
    
    # Split on table markers first
    table_pattern = re.compile(r'(\[TABLE\].*?\[/TABLE\])', re.DOTALL)
    parts = table_pattern.split(text)
    
    for part in parts:
        if part.startswith('[TABLE]'):
            # Keep entire table as one segment
            segments.append(part.strip())
        else:
            # Split non-table text by paragraphs
            paragraphs = re.split(r'\n\s*\n', part)
            segments.extend([p.strip() for p in paragraphs if p.strip()])
    
    return segments
```

Also update `_merge_into_chunks()` — when a segment starts with `[TABLE]`, always keep it as its own chunk regardless of size. Find the for loop in `_merge_into_chunks` and add this check at the start of the loop:

```python
for segment in segments:
    # Table blocks always get their own chunk — never split or merge with other content
    if segment.startswith('[TABLE]'):
        if current_words:
            # Flush current chunk first
            chunk_text_str = " ".join(current_words)
            if len(chunk_text_str) >= MIN_CHUNK_CHARS:
                chunks.append(TextChunk(
                    text=chunk_text_str,
                    chunk_index=idx,
                    char_count=len(chunk_text_str),
                    token_count=current_tokens,
                    section=section,
                    level="paragraph",
                ))
                idx += 1
            current_words = []
            current_tokens = 0
        
        # Add table as its own chunk
        table_text = segment.replace('[TABLE]\n', '').replace('\n[/TABLE]', '').strip()
        if table_text:
            chunks.append(TextChunk(
                text=table_text,
                chunk_index=idx,
                char_count=len(table_text),
                token_count=_estimate_tokens(table_text),
                section=section,
                level="table",
            ))
            idx += 1
        continue
    
    # ... rest of existing loop logic unchanged
```

---

## Verification — Run After Fix

```bash
# 1. Test table extraction on one PDF
python -c "
from src.ingestion.parsers.icmr import ICMRParser
from pathlib import Path
import glob

pdfs = glob.glob('data/raw/icmr/*.pdf')
if not pdfs:
    print('No PDFs found in data/raw/icmr/')
else:
    parser = ICMRParser(pdfs[0])
    docs = parser.parse()
    if docs:
        content = docs[0].content
        # Check if tables were extracted
        table_count = content.count('[TABLE]')
        print(f'Tables found: {table_count}')
        # Show first table if any
        if '[TABLE]' in content:
            start = content.index('[TABLE]')
            end = content.index('[/TABLE]', start) + len('[/TABLE]')
            print('First table:')
            print(content[start:end][:500])
        else:
            print('No tables found — PDF may not have extractable tables')
            print('First 300 chars:', content[:300])
"

# 2. Re-ingest with fixed parser
python scripts/reingest_v2.py

# 3. Test query — drug resistance query should now return structured table data
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "antimicrobial resistance data acinetobacter India", "top_k": 5}'
```

The citation passages should now show structured table rows instead of `"Amikacin 27 49 38 21 35 Aztreonam 62 55..."`.

---

## Important Notes

- If a PDF has no extractable tables (`find_tables()` returns empty list), the parser falls back cleanly to normal text extraction — no crash
- The `[TABLE]` markers are internal processing markers — they get stripped when the chunk is stored, replaced with clean formatted table text
- Do not change `pipeline_v2.py`, `ner.py`, `vector_db.py`, or any other file
- After re-ingestion, table chunks will have `level="table"` in the TextChunk — this can be used later for weighted retrieval

---

*OpenInsight Table Fix — SentArc Labs | Director: Aditya Singh*
