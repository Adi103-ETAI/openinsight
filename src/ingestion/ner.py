"""
Medical Named Entity Recognition
Extracts diseases, drugs, symptoms from chunk text using scispaCy.
Falls back to rule-based extraction if model unavailable.
"""
from functools import lru_cache
from loguru import logger
from typing import Optional
import re


DRUG_PATTERNS = [
    r"\b(?:doxycycline|azithromycin|rifampicin|isoniazid|pyrazinamide|ethambutol|"
    r"streptomycin|amoxicillin|ciprofloxacin|metronidazole|fluconazole|amphotericin|"
    r"artemisinin|chloroquine|primaquine|oseltamivir|acyclovir|cotrimoxazole|"
    r"paracetamol|ibuprofen|aspirin|metformin|insulin|amlodipine|atenolol|"
    r"enalapril|losartan|furosemide|spironolactone|digoxin|warfarin|heparin)\b",
]

DISEASE_PATTERNS = [
    r"\b(?:tuberculosis|TB|malaria|dengue|typhoid|leptospirosis|scrub typhus|"
    r"rickettsial|COVID-19|SARS-CoV-2|pneumonia|sepsis|meningitis|encephalitis|"
    r"hepatitis|cirrhosis|diabetes|hypertension|heart failure|myocardial infarction|"
    r"stroke|asthma|COPD|chronic kidney disease|CKD|anaemia|anemia|cholera|"
    r"chikungunya|Japanese encephalitis|rabies|snakebite|mucormycosis)\b",
]


@lru_cache(maxsize=1)
def _load_scispacy():
    try:
        import spacy

        nlp = spacy.load("en_core_sci_sm")
        logger.info("scispaCy model loaded successfully")
        return nlp
    except Exception as e:
        logger.warning(f"scispaCy model not available, using rule-based NER: {e}")
        return None


def extract_entities(text: str) -> dict:
    """
    Extract medical entities from text.
    Returns dict with diseases, drugs, symptoms lists.
    """
    diseases = []
    drugs = []
    symptoms = []

    # Rule-based extraction (always runs)
    text_lower = text.lower()
    for pattern in DRUG_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        drugs.extend([m.lower() for m in matches])

    for pattern in DISEASE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        diseases.extend([m.lower() for m in matches])

    # scispaCy NER (runs if model available)
    nlp = _load_scispacy()
    if nlp:
        try:
            doc = nlp(text[:1000])  # limit to 1000 chars for speed
            for ent in doc.ents:
                label = ent.label_.upper()
                val = ent.text.lower().strip()
                if len(val) < 3:
                    continue
                if label in ("DISEASE", "DISORDER", "SYNDROME"):
                    diseases.append(val)
                elif label in ("CHEMICAL", "DRUG", "MEDICATION"):
                    drugs.append(val)
                elif label in ("SIGN_SYMPTOM", "SYMPTOM"):
                    symptoms.append(val)
        except Exception as e:
            logger.debug(f"scispaCy extraction error: {e}")

    # Deduplicate
    return {
        "diseases": list(set(diseases))[:10],
        "drugs": list(set(drugs))[:10],
        "symptoms": list(set(symptoms))[:10],
    }


def classify_content_type(text: str, section: Optional[str] = None) -> tuple[str, float]:
    """
    Classify chunk content type and return (content_type, weight).

    Returns:
        content_type: "clinical" | "preclinical" | "background" | "noise"
        weight: 1.5 for clinical, 1.0 for preclinical, 0.7 for background, 0.1 for noise
    """
    text_lower = text.lower()

    # Noise patterns — administrative boilerplate
    noise_patterns = [
        r"\breferences?\b",
        r"\backnowledgements?\b",
        r"\bforeword\b",
        r"\bcommittee members?\b",
        r"\btable of contents\b",
        r"\bindex\b",
        r"\bcopyright\b",
        r"\ball rights reserved\b",
        r"\bissn\b",
        r"\bdoi:\s*10\.",
        r"\bfigure \d+\b",
        r"\btable \d+\b",
        r"\bappendix\b",
    ]
    noise_count = sum(1 for p in noise_patterns if re.search(p, text_lower))
    if noise_count >= 2:
        return "noise", 0.1

    # Check section label
    if section:
        section_lower = section.lower()
        if any(
            w in section_lower
            for w in ["reference", "acknowledgement", "foreword", "appendix", "index"]
        ):
            return "noise", 0.1
        if any(
            w in section_lower
            for w in ["abstract", "introduction", "background", "history", "epidemiology"]
        ):
            return "background", 0.7
        if any(w in section_lower for w in ["method", "animal", "in vitro", "mouse", "rat model"]):
            return "preclinical", 1.0
        if any(
            w in section_lower
            for w in [
                "treatment",
                "management",
                "dosage",
                "dose",
                "therapy",
                "diagnosis",
                "protocol",
                "guideline",
                "recommendation",
                "drug",
                "antibiotic",
                "clinical",
                "patient",
            ]
        ):
            return "clinical", 1.5

    # Content-based classification
    clinical_signals = [
        r"\b(?:dosage|dose|mg|mcg|kg|treatment|therapy|antibiotic|drug|prescri)\b",
        r"\b(?:patient|physician|doctor|hospital|clinic|ward)\b",
        r"\b(?:diagnosis|diagnostic|symptom|sign|fever|pain|infection)\b",
        r"\b(?:guideline|protocol|recommendation|management|should|must)\b",
        r"\b(?:oral|intravenous|IV|IM|SC|subcutaneous|intramuscular)\b",
    ]
    clinical_count = sum(1 for p in clinical_signals if re.search(p, text_lower))

    preclinical_signals = [
        r"\b(?:mouse|rat|animal|in vitro|cell line|assay|mechanism|pathway)\b",
    ]
    preclinical_count = sum(1 for p in preclinical_signals if re.search(p, text_lower))

    background_signals = [
        r"\b(?:history|historical|was first|reported in|century|epidemic|endemic)\b",
        r"\b(?:introduction|overview|background|etiology|epidemiology|prevalence)\b",
    ]
    background_count = sum(1 for p in background_signals if re.search(p, text_lower))

    if preclinical_count >= 2:
        return "preclinical", 1.0
    if clinical_count >= 3:
        return "clinical", 1.5
    if clinical_count >= 1:
        return "clinical", 1.5
    if background_count >= 2:
        return "background", 0.7

    return "background", 0.7


def infer_study_type(text: str, title: str = "") -> tuple[str, int]:
    """
    Infer study type and evidence level from text/title.
    Returns (study_type, evidence_level) where level 1=highest, 5=lowest.
    """
    combined = (title + " " + text[:500]).lower()

    if re.search(r"\b(?:meta.analysis|systematic review|cochrane)\b", combined):
        return "meta_analysis", 1
    if re.search(r"\b(?:randomized|randomised|RCT|clinical trial|controlled trial)\b", combined):
        return "rct", 1
    if re.search(r"\b(?:cohort study|case.control|prospective|retrospective study)\b", combined):
        return "observational", 2
    if re.search(r"\b(?:guideline|recommendation|protocol|ICMR|WHO|NMC|MoHFW)\b", combined):
        return "guideline", 2
    if re.search(r"\b(?:review article|narrative review|literature review)\b", combined):
        return "review", 3
    if re.search(r"\b(?:case report|case series|case study)\b", combined):
        return "case_report", 4

    return "unknown", 5
