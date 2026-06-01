"""
Report API — Generate clinical summary and evidence review reports.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from loguru import logger

from src.reports.generators import generate_clinical_summary, generate_evidence_review
from src.reports.models import (
    Citation,
    ConfidenceBreakdown,
    ReportFormat,
    ReportRequest,
    SafetyWarning,
)
from src.reports.pdf_renderer import render_report

router = APIRouter()


@router.post("/generate")
async def generate_report(payload: ReportRequest, request: Request):
    """
    Generate a clinical report from search results.

    Supports JSON and PDF output formats.
    """
    try:
        # Convert citation models
        citations = payload.citations
        safety_warnings = payload.safety_warnings
        breakdown = payload.confidence_breakdown

        if payload.report_type == "clinical_summary":
            report = generate_clinical_summary(
                query=payload.query,
                answer=payload.answer,
                citations=citations,
                safety_warnings=safety_warnings,
                confidence_score=payload.confidence_score,
                confidence_breakdown=breakdown,
                recommendation=payload.recommendation,
            )
        elif payload.report_type == "evidence_review":
            report = generate_evidence_review(
                query=payload.query,
                answer=payload.answer,
                citations=citations,
                safety_warnings=safety_warnings,
                confidence_score=payload.confidence_score,
                confidence_breakdown=breakdown,
                recommendation=payload.recommendation,
                source_chunks=payload.source_chunks,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown report type: {payload.report_type}")

        # Render to requested format
        if payload.format == ReportFormat.PDF:
            pdf_bytes = render_report(report, fmt="pdf")
            if isinstance(pdf_bytes, bytes):
                report_id = report.metadata.report_id
                return Response(
                    content=pdf_bytes,
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f'attachment; filename="{payload.report_type}_{report_id[:8]}.pdf"'
                    },
                )
            # reportlab not installed, fall through to JSON

        # JSON format
        return render_report(report, fmt="json")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/types")
async def list_report_types():
    """List available report types."""
    return {
        "types": [
            {
                "id": "clinical_summary",
                "name": "Clinical Summary",
                "description": "Quick actionable summary with key findings, citations, and safety warnings",
            },
            {
                "id": "evidence_review",
                "name": "Evidence Review",
                "description": "Detailed evidence analysis with source quality breakdown and confidence scoring",
            },
        ],
        "formats": ["json", "pdf"],
    }
