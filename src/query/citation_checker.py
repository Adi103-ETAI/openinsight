"""
Citation Checker
Verifies that citations exist in MongoDB, are from trusted sources,
have valid evidence levels, and are not outdated.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from bson import ObjectId
from loguru import logger

from src.ingestion.document_db import get_db


# Trusted source types (ordered by trust level)
TRUSTED_SOURCES = {
    "cochrane": 5,  # Systematic reviews, highest evidence
    "who": 5,  # WHO guidelines
    "nice": 5,  # NICE guidelines
    "pubmed": 4,  # Peer-reviewed
    "icmr": 4,  # ICMR guidelines (India)
    "cdc": 4,  # CDC guidelines
    "statpearls": 3,  # Medical reference
    "medline": 3,  # Medical literature
    "uptodate": 3,  # Clinical decision support
}

# Maximum age for citations before flagging as outdated (years)
MAX_CITATION_AGE_YEARS = 5


@dataclass
class CitationIssue:
    """An issue found with a citation."""

    citation_index: int
    issue_type: str  # "NOT_FOUND" | "UNTRUSTED_SOURCE" | "LOW_EVIDENCE" | "OUTDATED"
    severity: str  # "HIGH" | "MEDIUM" | "LOW"
    details: str


@dataclass
class CitationCheckResult:
    """Result of citation validation."""

    valid_citations: int
    total_citations: int
    issues: list[CitationIssue] = field(default_factory=list)
    evidence_distribution: dict = field(default_factory=dict)
    avg_evidence_level: float = 3.0  # Default to middle level
    trusted_source_ratio: float = 1.0


async def check_citations(
    citations: list[dict],
    verify_in_db: bool = True,
) -> CitationCheckResult:
    """
    Validate citations for existence, trust level, evidence, and freshness.

    Args:
        citations: List of citation dicts with mongo_id, source_type, title, etc.
        verify_in_db: Whether to verify citations exist in MongoDB

    Returns:
        CitationCheckResult with validation details
    """
    if not citations:
        return CitationCheckResult(
            valid_citations=0,
            total_citations=0,
            issues=[],
            evidence_distribution={},
            avg_evidence_level=0.0,
            trusted_source_ratio=0.0,
        )

    issues: list[CitationIssue] = []
    evidence_levels = []
    trusted_count = 0
    valid_count = 0

    db = get_db() if verify_in_db else None

    for idx, citation in enumerate(citations):
        citation_idx = citation.get("index", idx + 1)
        mongo_id = citation.get("mongo_id", "")
        source_type = citation.get("source_type", "unknown").lower()
        title = citation.get("title", "Unknown")

        # 1. Verify citation exists in MongoDB
        if verify_in_db and db and mongo_id:
            try:
                chunk = await db["chunks"].find_one({"_id": ObjectId(mongo_id)})
                if not chunk:
                    issues.append(
                        CitationIssue(
                            citation_index=citation_idx,
                            issue_type="NOT_FOUND",
                            severity="HIGH",
                            details=f"Citation '{title[:50]}' not found in database",
                        )
                    )
                    continue

                # Get evidence level from chunk if available
                chunk_evidence = chunk.get("evidence_level", 3)
                evidence_levels.append(chunk_evidence)

                # Check publication date
                published_date = chunk.get("published_date")
                if published_date:
                    try:
                        if isinstance(published_date, str):
                            pub_year = int(published_date[:4])
                        else:
                            pub_year = published_date.year

                        current_year = datetime.now().year
                        age = current_year - pub_year

                        if age > MAX_CITATION_AGE_YEARS:
                            issues.append(
                                CitationIssue(
                                    citation_index=citation_idx,
                                    issue_type="OUTDATED",
                                    severity="LOW",
                                    details=f"Citation from {pub_year} ({age} years old)",
                                )
                            )
                    except (ValueError, AttributeError):
                        pass

                valid_count += 1

            except Exception as e:
                logger.warning(f"Citation check failed for {mongo_id}: {e}")
                issues.append(
                    CitationIssue(
                        citation_index=citation_idx,
                        issue_type="NOT_FOUND",
                        severity="MEDIUM",
                        details=f"Could not verify citation: {str(e)[:50]}",
                    )
                )
                continue
        else:
            # If not verifying in DB, assume valid
            valid_count += 1
            evidence_levels.append(3)  # Default evidence level

        # 2. Check source trust level
        if source_type in TRUSTED_SOURCES:
            trusted_count += 1
        else:
            issues.append(
                CitationIssue(
                    citation_index=citation_idx,
                    issue_type="UNTRUSTED_SOURCE",
                    severity="MEDIUM",
                    details=f"Source '{source_type}' not in trusted sources list",
                )
            )

        # 3. Check evidence level
        evidence_level = citation.get("evidence_level", 3)
        if evidence_level > 3:  # Evidence levels: 1=highest, 5=lowest
            issues.append(
                CitationIssue(
                    citation_index=citation_idx,
                    issue_type="LOW_EVIDENCE",
                    severity="LOW",
                    details=f"Evidence level {evidence_level} (case report/expert opinion)",
                )
            )

    # Calculate evidence distribution
    evidence_distribution = {
        "grade_i_count": sum(1 for e in evidence_levels if e == 1),
        "grade_ii_count": sum(1 for e in evidence_levels if e == 2),
        "grade_iii_count": sum(1 for e in evidence_levels if e == 3),
        "grade_iv_count": sum(1 for e in evidence_levels if e == 4),
        "grade_v_count": sum(1 for e in evidence_levels if e == 5),
    }

    avg_evidence = (
        sum(evidence_levels) / len(evidence_levels) if evidence_levels else 3.0
    )
    trusted_ratio = trusted_count / len(citations) if citations else 0.0

    return CitationCheckResult(
        valid_citations=valid_count,
        total_citations=len(citations),
        issues=issues,
        evidence_distribution=evidence_distribution,
        avg_evidence_level=avg_evidence,
        trusted_source_ratio=trusted_ratio,
    )
