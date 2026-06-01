"""
PDF Renderer — Generate formatted PDF reports using reportlab.
Falls back to plain text if reportlab is not installed.
"""
from __future__ import annotations

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.reports.models import (
    ClinicalSummaryReport,
    EvidenceReviewReport,
)


def _check_reportlab() -> bool:
    """Check if reportlab is available."""
    try:
        import reportlab  # noqa: F401
        return True
    except ImportError:
        return False


def _draw_header(canvas: Any, doc: Any, title: str, subtitle: str) -> None:
    """Draw report header with title and metadata."""
    from reportlab.lib.units import inch

    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 16)
    canvas.drawString(72, 750, title)
    canvas.setFont("Helvetica", 10)
    canvas.drawString(72, 735, subtitle)
    canvas.line(72, 730, 540, 730)
    canvas.restoreState()


def _draw_footer(canvas: Any, doc: Any) -> None:
    """Draw page footer with disclaimer and page number."""
    canvas.saveState()
    canvas.setFont("Helvetica-Oblique", 7)
    canvas.drawString(
        72, 36,
        "OpenInsight AI Clinical Decision Support — Generated reports should be verified by a clinician."
    )
    canvas.drawRightString(540, 36, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def render_clinical_summary_pdf(report: ClinicalSummaryReport) -> bytes:
    """Render a Clinical Summary Report as PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=72,
        rightMargin=72,
        topMargin=90,
        bottomMargin=60,
    )

    styles = getSampleStyleSheet()
    story = []

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"], fontSize=14, spaceAfter=6
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=11, spaceAfter=4, spaceBefore=10
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=4
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, leading=10, spaceAfter=2
    )

    # Title
    story.append(Paragraph("Clinical Summary Report", title_style))
    story.append(Spacer(1, 6))

    # Metadata table
    meta_data = [
        ["Query", report.query[:100]],
        ["Generated", report.metadata.generated_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Confidence", f"{report.confidence_score:.0%}"],
        ["Recommendation", report.recommendation],
        ["Citations", str(len(report.citations))],
    ]
    meta_table = Table(meta_data, colWidths=[90, 380])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Clinical Answer
    story.append(Paragraph("Clinical Answer", section_style))
    for para in report.clinical_answer.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, body_style))

    # Key Findings
    if report.key_findings:
        story.append(Paragraph("Key Findings", section_style))
        for i, finding in enumerate(report.key_findings, 1):
            story.append(Paragraph(f"{i}. {finding}", body_style))

    # Evidence Summary
    story.append(Paragraph("Evidence Summary", section_style))
    story.append(Paragraph(report.evidence_summary, body_style))

    # Citations
    if report.citations:
        story.append(Paragraph("Sources", section_style))
        for c in report.citations:
            text = f"[{c.index}] {c.title}"
            if c.source_type:
                text += f" ({c.source_type})"
            if c.year:
                text += f", {c.year}"
            story.append(Paragraph(text, small_style))

    # Safety Warnings
    if report.safety_warnings:
        story.append(Paragraph("Safety Warnings", section_style))
        for w in report.safety_warnings:
            text = f"[{w.severity}] {w.warning_type}: {w.message}"
            story.append(Paragraph(text, small_style))

    # Disclaimer
    story.append(Spacer(1, 20))
    story.append(Paragraph("Disclaimer", section_style))
    story.append(Paragraph(report.disclaimer, small_style))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buffer.getvalue()


def render_evidence_review_pdf(report: EvidenceReviewReport) -> bytes:
    """Render an Evidence Review Report as PDF."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=72,
        rightMargin=72,
        topMargin=90,
        bottomMargin=60,
    )

    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Heading1"], fontSize=14, spaceAfter=6
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], fontSize=11, spaceAfter=4, spaceBefore=10
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9, leading=12, spaceAfter=4
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, leading=10, spaceAfter=2
    )

    # Title
    story.append(Paragraph("Evidence Review Report", title_style))
    story.append(Spacer(1, 6))

    # Metadata
    meta_data = [
        ["Query", report.query[:100]],
        ["Generated", report.metadata.generated_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Confidence", f"{report.confidence_score:.0%}"],
        ["Recommendation", report.recommendation],
    ]
    meta_table = Table(meta_data, colWidths=[90, 380])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Evidence Analysis
    story.append(Paragraph("Evidence Analysis", section_style))
    story.append(Paragraph(report.evidence_analysis, body_style))

    # Source Breakdown
    if report.source_breakdown:
        story.append(Paragraph("Source Breakdown", section_style))
        for source, count in report.source_breakdown.items():
            story.append(Paragraph(f"  {source}: {count} source(s)", body_style))

    # Evidence Levels
    if report.evidence_levels:
        story.append(Paragraph("Evidence Level Distribution", section_style))
        for level, count in report.evidence_levels.items():
            story.append(Paragraph(f"  {level}: {count}", body_style))

    # Answer
    story.append(Paragraph("Generated Answer", section_style))
    for para in report.answer.split("\n\n"):
        para = para.strip()
        if para:
            story.append(Paragraph(para, body_style))

    # Citations
    if report.citations:
        story.append(Paragraph("All Sources", section_style))
        for c in report.citations:
            text = f"[{c.index}] {c.title} ({c.source_type})"
            if c.year:
                text += f", {c.year}"
            story.append(Paragraph(text, small_style))

    # Hallucination Flags
    if report.hallucination_flags:
        story.append(Paragraph("Coverage Analysis", section_style))
        for flag in report.hallucination_flags:
            text = f"  {flag.get('chunk_title', 'Unknown')}: {flag.get('note', '')} ({flag.get('coverage', 0):.0%} overlap)"
            story.append(Paragraph(text, small_style))

    # Quality Notes
    if report.quality_notes:
        story.append(Paragraph("Quality Notes", section_style))
        story.append(Paragraph(report.quality_notes, body_style))

    # Safety Warnings
    if report.safety_warnings:
        story.append(Paragraph("Safety Warnings", section_style))
        for w in report.safety_warnings:
            text = f"[{w.severity}] {w.warning_type}: {w.message}"
            story.append(Paragraph(text, small_style))

    # Disclaimer
    story.append(Spacer(1, 20))
    story.append(Paragraph("Disclaimer", section_style))
    story.append(Paragraph(report.disclaimer, small_style))

    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return buffer.getvalue()


def render_report(report: ClinicalSummaryReport | EvidenceReviewReport, fmt: str = "pdf") -> bytes | dict:
    """
    Render a report to the specified format.
    Falls back to JSON dict if reportlab is not installed for PDF.
    """
    if fmt == "pdf":
        if not _check_reportlab():
            logger.warning("reportlab not installed, falling back to JSON")
            return report.model_dump()

        if isinstance(report, ClinicalSummaryReport):
            return render_clinical_summary_pdf(report)
        elif isinstance(report, EvidenceReviewReport):
            return render_evidence_review_pdf(report)
    elif fmt == "json":
        return report.model_dump()

    return report.model_dump()
