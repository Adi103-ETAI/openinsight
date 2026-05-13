import re
import importlib
import logging
from pathlib import Path

from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser

logger = logging.getLogger(__name__)

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
            cleaned_row = [
                str(cell).strip() if cell is not None else "" for cell in row
            ]
            cleaned.append(cleaned_row)

        # Skip if table is mostly empty - require at least 2 rows with content
        # A 2x2 table has 4 cells minimum, so require >= 50% non-empty cells for small tables
        total_cells = sum(len(row) for row in cleaned)
        non_empty = sum(1 for row in cleaned for cell in row if cell)
        # Require at least 4 non-empty cells (a small meaningful table) OR at least 30% filled
        if total_cells > 0 and non_empty < 4 and (non_empty / total_cells) < 0.3:
            return ""

        # Calculate column widths
        col_widths = []
        num_cols = max(len(row) for row in cleaned)
        for col_idx in range(num_cols):
            width = max(
                (len(row[col_idx]) if col_idx < len(row) else 0) for row in cleaned
            )
            col_widths.append(max(width, 4))

        # Format as aligned text table
        lines = []
        for row_idx, row in enumerate(cleaned):
            # Pad row to num_cols
            padded = row + [""] * (num_cols - len(row))
            line = " | ".join(
                cell.ljust(col_widths[i]) for i, cell in enumerate(padded)
            )
            lines.append(line)
            # Add separator after header row - use same format as row (space-pipe-space)
            if row_idx == 0:
                separator = " | ".join("-" * col_widths[i] for i in range(num_cols))
                lines.append(separator)

        return "\n".join(lines)

    def _extract_page_content(self, page) -> str:
        """
        Extract text and tables from a single page.
        Tables are extracted first with structure preserved,
        then remaining text is extracted with table bounding boxes masked
        to avoid extracting duplicate text from table areas.
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
        # We'll extract the full text and filter out content that overlaps with tables
        if table_bboxes:
            try:
                full_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            except (RuntimeError, ValueError, TypeError, OSError):
                full_text = page.extract_text() or ""

            # Mask out text that overlaps with table bounding boxes
            # by extracting text from areas outside the tables
            text_parts = []
            page_height = page.height

            # Sort bboxes by top position
            sorted_bboxes = sorted(table_bboxes, key=lambda b: b[1] if b else 0)

            # Extract text between tables (before first table, between tables, after last table)
            prev_bottom = 0
            for bbox in sorted_bboxes:
                if bbox is None:
                    continue
                top = bbox[1]
                # Extract text from region before this table
                if top > prev_bottom:
                    try:
                        region_text = page.crop((0, prev_bottom, page.width, top)).extract_text()
                        if region_text:
                            text_parts.append(region_text)
                    except (RuntimeError, ValueError, TypeError, OSError):
                        pass
                prev_bottom = bbox[3]

            # Extract text after the last table
            if prev_bottom < page_height:
                try:
                    region_text = page.crop((0, prev_bottom, page.width, page_height)).extract_text()
                    if region_text:
                        text_parts.append(region_text)
                except (RuntimeError, ValueError, TypeError, OSError):
                    pass

            text = "\n".join(text_parts).strip()
        else:
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""

        if text:
            page_parts.insert(0, text)

        return "\n".join(page_parts)

    def parse(self) -> list[DocumentRecord]:
        if not self.file_path.exists():
            logger.error("ICMR file not found: %s", self.file_path)
            return []

        try:
            pdfplumber = importlib.import_module("pdfplumber")

            with pdfplumber.open(self.file_path) as pdf:
                page_count = len(pdf.pages)
                logger.info(
                    "Parsing ICMR PDF: %s (%s pages)",
                    self.file_path.name,
                    page_count,
                )

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
                    except (RuntimeError, ValueError, TypeError, OSError) as e:
                        logger.debug("Page extraction error: %s", e)
                        # Fallback to simple text extraction for this page
                        text = page.extract_text() or ""
                        if text.strip():
                            pages_content.append(text)

                logger.info("Found %s tables in %s", table_count, self.file_path.name)

        except (ImportError, RuntimeError, ValueError, TypeError, OSError) as exc:
            logger.error("Failed to read ICMR PDF %s: %s", self.file_path, exc)
            return []

        raw_content = "\n\n".join(pages_content)
        cleaned_content = self._clean_text(raw_content)

        if not cleaned_content:
            logger.error("No parseable text extracted from: %s", self.file_path)
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
        logger.info(
            "Parsed %s: %s chars, %s tables",
            self.file_path.name,
            len(cleaned_content),
            table_count,
        )
        return [document]
