"""
OCR Parser
Uses Tesseract to extract text from scanned PDFs that pdfplumber cannot read.
Auto-detects scanned PDFs by checking if pdfplumber returns empty text.
"""
import re
from pathlib import Path
from loguru import logger

from src.ingestion.document_db import DocumentRecord
from src.ingestion.parsers.base import BaseParser
from src.ingestion.ner import infer_study_type


class OCRParser(BaseParser):
    """
    OCR fallback parser for scanned PDFs.
    Converts each page to image, runs Tesseract OCR, combines results.
    """

    def __init__(self, file_path: str | Path, source_type: str = "icmr"):
        self.file_path = Path(file_path)
        self._source_type = source_type

    @property
    def source_type(self) -> str:
        return self._source_type

    @staticmethod
    def is_scanned(file_path: Path) -> bool:
        """Check if a PDF is scanned (pdfplumber returns empty text on first pages)."""
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                text_found = 0
                for page in pdf.pages[:5]:
                    text = page.extract_text() or ""
                    if len(text.strip()) > 50:
                        text_found += 1
                return text_found == 0
        except Exception:
            return False

    def parse(self) -> list[DocumentRecord]:
        if not self.file_path.exists():
            logger.error(f"File not found: {self.file_path}")
            return []

        try:
            import pytesseract
            from PIL import Image
            import pdfplumber
        except ImportError as e:
            logger.error(f"OCR dependencies not installed: {e}")
            return []

        logger.info(f"Running OCR on scanned PDF: {self.file_path.name}")

        try:
            with pdfplumber.open(self.file_path) as pdf:
                page_texts = []
                for i, page in enumerate(pdf.pages):
                    try:
                        # Convert page to image
                        img = page.to_image(resolution=200).original
                        text = pytesseract.image_to_string(img, lang="eng")
                        text = re.sub(r"\s+", " ", text).strip()
                        if len(text) > 30:
                            page_texts.append(text)
                    except Exception as e:
                        logger.debug(f"OCR failed on page {i}: {e}")

            if not page_texts:
                logger.error(f"OCR extracted no text from {self.file_path.name}")
                return []

            full_text = "\n\n".join(page_texts)
            title = self.file_path.stem.replace("_", " ")
            study_type, evidence_level = infer_study_type(full_text[:500], title)

            doc = DocumentRecord(
                source_type=self.source_type,
                title=title,
                content=full_text,
                url=str(self.file_path.resolve()),
                study_type=study_type,
                evidence_level=evidence_level,
                is_india_specific=True,
                parser_version="v2",
            )
            logger.info(f"OCR complete: {title[:60]} — {len(full_text)} chars from {len(page_texts)} pages")
            return [doc]

        except Exception as e:
            logger.error(f"OCR parsing failed for {self.file_path.name}: {e}")
            return []


def is_scanned_check(file_path: str | Path) -> bool:
    return OCRParser.is_scanned(Path(file_path))
