"""
Medical Safety Checker
Detects treatment recommendations, drug interactions, contraindications,
and other potentially dangerous medical content that needs disclaimers.
"""

import re
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class SafetyWarning:
    """A safety warning for the answer."""

    warning_type: str  # e.g., "TREATMENT_RECOMMENDATION", "DRUG_INTERACTION", etc.
    severity: str  # "HIGH" | "MEDIUM" | "LOW"
    message: str
    matched_text: Optional[str] = None


@dataclass
class SafetyCheckResult:
    """Result of safety checking."""

    is_safe: bool
    warnings: list[SafetyWarning] = field(default_factory=list)
    needs_disclaimer: bool = False
    has_treatment_advice: bool = False
    has_dosage_info: bool = False


# Patterns for detecting treatment recommendations
TREATMENT_PATTERNS = [
    (
        r"\bshould\s+(be\s+)?(?:treat|give|administer|prescribe)",
        "TREATMENT_RECOMMENDATION",
    ),
    (
        r"\brecommend(?:ed|s)?\s+(?:to\s+)?(?:treat|give|administer|use)",
        "TREATMENT_RECOMMENDATION",
    ),
    (r"\btreat(?:ed|ment)?\s+with\b", "TREATMENT_RECOMMENDATION"),
    (r"\badminister(?:ed|ing)?\s+", "TREATMENT_RECOMMENDATION"),
    (r"\bprescribe(?:d|s)?\s+", "TREATMENT_RECOMMENDATION"),
    (r"\bfirst[- ]?line\s+(?:treatment|therapy|drug)", "TREATMENT_RECOMMENDATION"),
    (r"\bgiven?\s+(?:the\s+)?patient", "TREATMENT_RECOMMENDATION"),
]

# Patterns for dosage information
DOSAGE_PATTERNS = [
    (
        r"\b(\d+(?:\.\d+)?)\s*(?:mg|g|kg|mcg|µg|ml|mL|IU|units?)/(?:kg|day|dose)",
        "DOSAGE_INFO",
    ),
    (
        r"\b(\d+(?:\.\d+)?)\s*(?:mg|g|kg|mcg|µg|ml|mL|IU|units?)\s+(?:once|twice|three times|daily|weekly|monthly)",
        "DOSAGE_INFO",
    ),
    (r"dose\s+of\s+(\d+(?:\.\d+)?)\s*(?:mg|g|kg|mcg|µg)", "DOSAGE_INFO"),
    (
        r"\b(\d+)\s*(?:tablets?|capsules?|pills?)\s+(?:daily|per day|twice daily)",
        "DOSAGE_INFO",
    ),
]

# Patterns for contraindications
CONTRAINDICATION_PATTERNS = [
    (r"\bcontraindicated\s+(?:in|for|with)", "CONTRAINDICATION"),
    (
        r"\bshould\s+(?:not|never)\s+(?:be\s+)?(?:used|given|administered)",
        "CONTRAINDICATION",
    ),
    (r"\bavoid(?:ed|ing)?\s+(?:in|during|with)", "CONTRAINDICATION"),
    (r"\bnot\s+recommended\s+(?:for|in|during)", "CONTRAINDICATION"),
    (r"\bdo\s+not\s+(?:use|give|administer)", "CONTRAINDICATION"),
]

# Patterns for drug interactions
INTERACTION_PATTERNS = [
    (r"\binteracts?\s+with\b", "DRUG_INTERACTION"),
    (r"\bdrug[- ]?drug\s+interaction", "DRUG_INTERACTION"),
    (r"\bwhen\s+(?:taken|used|combined)\s+with\b", "DRUG_INTERACTION"),
    (
        r"\bconcomitant(?:ly)?\s+(?:use|administration)\s+(?:of|with)",
        "DRUG_INTERACTION",
    ),
    (r"\bshould\s+not\s+be\s+(?:taken|used|combined)\s+with", "DRUG_INTERACTION"),
    (r"\bavoid\s+(?:concurrent|concomitant|simultaneous)", "DRUG_INTERACTION"),
]

# Patterns for monitoring requirements
MONITORING_PATTERNS = [
    (r"\bmonitor(?:ing|ed)?\s+(?:for|of|closely|regularly)", "MONITORING_REQUIRED"),
    (r"\brequires?\s+(?:regular|close|careful)\s+monitoring", "MONITORING_REQUIRED"),
    (
        r"\bcheck\s+(?:liver|kidney|renal|hepatic|blood|cardiac|ECG)",
        "MONITORING_REQUIRED",
    ),
    (r"\bperiodic\s+(?:monitoring|testing|checks)", "MONITORING_REQUIRED"),
    (
        r"\bbaseline\s+(?:and\s+)?(?:periodic|regular)\s+(?:tests?|monitoring)",
        "MONITORING_REQUIRED",
    ),
]

# Patterns for off-label use
OFFLABEL_PATTERNS = [
    (r"\boff[- ]?label\s+(?:use|indication)", "OFF_LABEL_USE"),
    (r"\bnot\s+(?:FDA[- ]?)?approved\s+(?:for|in)", "OFF_LABEL_USE"),
    (r"\bunapproved\s+(?:use|indication)", "OFF_LABEL_USE"),
]

# Known dangerous drugs/combinations (simplified - would need a proper database)
DANGEROUS_COMBINATIONS = [
    (["warfarin", "aspirin"], "Increased bleeding risk"),
    (["methotrexate", "nsaid"], "Increased methotrexate toxicity"),
    (["lithium", "nsaid"], "Increased lithium toxicity"),
    (["digoxin", "amiodarone"], "Increased digoxin levels"),
    (["simvastatin", "gemfibrozil"], "Increased risk of rhabdomyolysis"),
    (["maoi", "ssri"], "Risk of serotonin syndrome"),
    (["linezolid", "ssri"], "Risk of serotonin syndrome"),
]

# Drugs requiring specific monitoring
MONITORING_DRUGS = {
    "linezolid": "Monitor for peripheral neuropathy and myelosuppression",
    "bedaquiline": "Monitor QTc interval (ECG) for cardiac arrhythmias",
    "delamanid": "Monitor QTc interval (ECG) for cardiac arrhythmias",
    "clofazimine": "Monitor QTc interval and skin discoloration",
    "aminoglycosides": "Monitor renal function and hearing",
    "amikacin": "Monitor renal function and ototoxicity",
    "kanamycin": "Monitor renal function and ototoxicity",
    "streptomycin": "Monitor renal function and ototoxicity",
    "capreomycin": "Monitor renal function, electrolytes, and hearing",
    "cycloserine": "Monitor for CNS toxicity (seizures, psychosis)",
    "ethionamide": "Monitor hepatic function",
    "isoniazid": "Monitor hepatic function, check B6 supplementation",
    "rifampicin": "Monitor hepatic function, drug interactions",
    "pyrazinamide": "Monitor hepatic function and uric acid",
    "fluoroquinolones": "Monitor for tendon issues and QTc prolongation",
    "methotrexate": "Monitor CBC, hepatic and renal function",
    "warfarin": "Monitor INR regularly",
    "lithium": "Monitor serum lithium levels",
    "digoxin": "Monitor serum digoxin levels",
    "vancomycin": "Monitor trough levels and renal function",
}


def _check_patterns(
    text: str, patterns: list[tuple[str, str]], severity: str = "MEDIUM"
) -> list[SafetyWarning]:
    """Check text against a list of regex patterns."""
    warnings = []
    text_lower = text.lower()

    for pattern, warning_type in patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            # Get context around match
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 50)
            context = text[start:end].strip()

            warnings.append(
                SafetyWarning(
                    warning_type=warning_type,
                    severity=severity,
                    message=f"Detected: {warning_type.replace('_', ' ').lower()}",
                    matched_text=context[:100],
                )
            )
    return warnings


def _check_drug_interactions(text: str) -> list[SafetyWarning]:
    """Check for known dangerous drug combinations."""
    warnings = []
    text_lower = text.lower()

    for drugs, risk in DANGEROUS_COMBINATIONS:
        # Check if all drugs in the combination are mentioned
        if all(drug in text_lower for drug in drugs):
            warnings.append(
                SafetyWarning(
                    warning_type="DRUG_INTERACTION",
                    severity="HIGH",
                    message=f"Potential interaction: {' + '.join(drugs)} - {risk}",
                    matched_text=None,
                )
            )

    return warnings


def _check_monitoring_requirements(text: str) -> list[SafetyWarning]:
    """Check if mentioned drugs require specific monitoring."""
    warnings = []
    text_lower = text.lower()

    for drug, monitoring_note in MONITORING_DRUGS.items():
        if drug in text_lower:
            warnings.append(
                SafetyWarning(
                    warning_type="MONITORING_REQUIRED",
                    severity="MEDIUM",
                    message=f"{drug.title()}: {monitoring_note}",
                    matched_text=None,
                )
            )

    return warnings


def check_safety(answer: str, chunks: list[dict] = None) -> SafetyCheckResult:
    """
    Check answer for medical safety concerns.

    Args:
        answer: The LLM-generated answer text
        chunks: Source chunks (optional, for context verification)

    Returns:
        SafetyCheckResult with warnings and safety status
    """
    if not answer:
        return SafetyCheckResult(is_safe=True, warnings=[], needs_disclaimer=False)

    all_warnings: list[SafetyWarning] = []

    # Check for treatment recommendations (HIGH severity - needs disclaimer)
    treatment_warnings = _check_patterns(answer, TREATMENT_PATTERNS, severity="MEDIUM")
    if treatment_warnings:
        all_warnings.extend(treatment_warnings)

    # Check for dosage information (MEDIUM severity - needs verification)
    dosage_warnings = _check_patterns(answer, DOSAGE_PATTERNS, severity="MEDIUM")
    if dosage_warnings:
        all_warnings.extend(dosage_warnings)

    # Check for contraindications (LOW severity - informational)
    contraindication_warnings = _check_patterns(
        answer, CONTRAINDICATION_PATTERNS, severity="LOW"
    )
    all_warnings.extend(contraindication_warnings)

    # Check for drug interaction mentions (MEDIUM severity)
    interaction_warnings = _check_patterns(
        answer, INTERACTION_PATTERNS, severity="MEDIUM"
    )
    all_warnings.extend(interaction_warnings)

    # Check for known dangerous combinations (HIGH severity)
    combination_warnings = _check_drug_interactions(answer)
    all_warnings.extend(combination_warnings)

    # Check for drugs requiring monitoring (MEDIUM severity)
    monitoring_warnings = _check_monitoring_requirements(answer)
    all_warnings.extend(monitoring_warnings)

    # Check monitoring pattern mentions
    monitoring_pattern_warnings = _check_patterns(
        answer, MONITORING_PATTERNS, severity="LOW"
    )
    all_warnings.extend(monitoring_pattern_warnings)

    # Check for off-label use mentions (MEDIUM severity)
    offlabel_warnings = _check_patterns(answer, OFFLABEL_PATTERNS, severity="MEDIUM")
    all_warnings.extend(offlabel_warnings)

    # Deduplicate warnings by type and message
    seen = set()
    unique_warnings = []
    for w in all_warnings:
        key = (w.warning_type, w.message[:50])
        if key not in seen:
            seen.add(key)
            unique_warnings.append(w)

    # Determine overall safety status
    high_severity_count = sum(1 for w in unique_warnings if w.severity == "HIGH")
    has_treatment_advice = any(
        w.warning_type == "TREATMENT_RECOMMENDATION" for w in unique_warnings
    )
    has_dosage_info = any(w.warning_type == "DOSAGE_INFO" for w in unique_warnings)

    # Consider unsafe if there are HIGH severity warnings
    is_safe = high_severity_count == 0
    # Needs disclaimer if there's treatment advice or dosage info
    needs_disclaimer = has_treatment_advice or has_dosage_info

    return SafetyCheckResult(
        is_safe=is_safe,
        warnings=unique_warnings,
        needs_disclaimer=needs_disclaimer,
        has_treatment_advice=has_treatment_advice,
        has_dosage_info=has_dosage_info,
    )
