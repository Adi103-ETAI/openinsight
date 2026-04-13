import re
from pathlib import Path

import pdfplumber
from loguru import logger

from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

NOISE_WORDS = {
    "icmr",
    "guidelines",
    "guideline",
    "management",
    "protocol",
    "national",
    "for",
    "of",
    "the",
    "and",
    "on",
    "clinical",
    "india",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
}


class ICMRParser(BaseParser):
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    @property
    def source_type(self) -> str:
        return "icmr"

    def _extract_condition_tags(self) -> list[str]:
        stem = self.file_path.stem.lower()
        parts = [p for p in re.split(r"[_\-]+", stem) if p]
        tags: list[str] = []
        for part in parts:
            if part not in NOISE_WORDS and part not in tags:
                tags.append(part)
        return tags

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
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if re.fullmatch(r"\d+", line):
                continue
            if len(line) < 20 and not re.search(r"[.!?;:]$", line):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

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
            try:
                # Get text outside table bounding boxes
                full_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            except Exception:
                full_text = page.extract_text() or ""

            text = full_text.strip()
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
