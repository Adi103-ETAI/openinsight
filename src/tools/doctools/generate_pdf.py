"""generate_pdf — render a structured dict of sections into a PDF file."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def generate_pdf(sections: Dict[str, str], title: str, output_dir: Path = None) -> str:
    """
    Render sections into a PDF using reportlab. Returns the file path.
    Returns "" if reportlab is not installed or an error occurs.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

        from src.tools.doctools.generate_filename import generate_filename

        if output_dir is None:
            output_dir = Path("/tmp") / "openinsight_reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / generate_filename(title, "pdf")

        doc = SimpleDocTemplate(
            str(path), pagesize=A4,
            rightMargin=inch, leftMargin=inch,
            topMargin=inch, bottomMargin=inch,
        )
        styles = getSampleStyleSheet()
        story = []

        for key, text in sections.items():
            if key == "header":
                story.append(Paragraph(text.replace("\n", "<br/>"), styles["Title"]))
                story.append(Spacer(1, 12))
            elif key == "disclaimer":
                story.append(Paragraph(text, styles["Normal"]))
            else:
                heading = key.replace("_", " ").title()
                story.append(Paragraph(heading, styles["Heading2"]))
                story.append(Paragraph(text.replace("\n", "<br/>"), styles["Normal"]))
                story.append(Spacer(1, 8))

        doc.build(story)
        return str(path)
    except ImportError:
        logger.warning("reportlab not installed; cannot generate PDF")
        return ""
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return ""
