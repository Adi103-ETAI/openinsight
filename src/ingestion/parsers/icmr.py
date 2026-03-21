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
        cleaned_lines: list[str] = []
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            if re.fullmatch(r"\d+", line):
                continue
            if len(line) < 20 and not re.search(r"[.!?;:]$", line):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    def parse(self) -> list[DocumentRecord]:
        if not self.file_path.exists():
            logger.error(f"ICMR file not found: {self.file_path}")
            return []

        try:
            with pdfplumber.open(self.file_path) as pdf:
                page_count = len(pdf.pages)
                logger.info(f"Parsing ICMR PDF: {self.file_path.name} ({page_count} pages)")
                pages_text: list[str] = []
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages_text.append(page_text)
        except Exception as exc:
            logger.error(f"Failed to read ICMR PDF {self.file_path}: {exc}")
            return []

        raw_text = "\n".join(pages_text)
        cleaned_text = self._clean_text(raw_text)
        if not cleaned_text:
            logger.error(f"No parseable text extracted from ICMR PDF: {self.file_path}")
            return []

        title = self.file_path.stem.replace("_", " ")
        document = DocumentRecord(
            source_type=self.source_type,
            title=title,
            content=cleaned_text,
            url=str(self.file_path.resolve()),
            condition_tags=self._extract_condition_tags(),
        )
        return [document]
