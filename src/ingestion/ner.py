"""
Medical Named Entity Recognition
Extracts diseases, drugs, symptoms, dosages, contraindications, patient
populations, and outcomes from chunk text using scispaCy.
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
    r"enalapril|losartan|furosemide|spironolactone|digoxin|warfarin|heparin|"
    r"vancomycin|linezolid|meropenem|ceftriaxone|piperacillin|tazobactam|"
    r"amikacin|gentamicin|colistin|tigecycline|daptomycin|clindamycin|"
    r"hydroxychloroquine|remdesivir|dexamethasone|methylprednisolone|prednisolone|"
    r"omeprazole|pantoprazole|ranitidine|ondansetron|metoclopramide|"
    r"clopidogrel|atorvastatin|rosuvastatin|simvastatin|amlodipine|nifedipine|"
    r"propranolol|metoprolol|bisoprolol|ramipril|lisinopril|valsartan|"
    r"levodopa|carbidopa|donepezil|memantine|haloperidol|risperidone|olanzapine|"
    r"sertraline|fluoxetine|paroxetine|escitalopram|amitriptyline|nortriptyline|"
    r"carbamazepine|phenytoin|valproate|lamotrigine|levetiracetam|"
    r"morphine|fentanyl|tramadol|codeine|oxycodone|naloxone|buprenorphine|"
    r"salbutamol|ipratropium|tiotropium|budesonide|fluticasone|montelukast|"
    r"methotrexate|hydroxychloroquine|sulfasalazine|adalimumab|infliximab|"
    r"rituximab|bevacizumab|trastuzumab|imatinib|erlotinib|sorafenib)\b",
]

DISEASE_PATTERNS = [
    r"\b(?:tuberculosis|TB|malaria|dengue|typhoid|leptospirosis|scrub typhus|"
    r"rickettsial|COVID-19|SARS-CoV-2|pneumonia|sepsis|meningitis|encephalitis|"
    r"hepatitis|cirrhosis|diabetes|hypertension|heart failure|myocardial infarction|"
    r"stroke|asthma|COPD|chronic kidney disease|CKD|anaemia|anemia|cholera|"
    r"chikungunya|Japanese encephalitis|rabies|snakebite|mucormycosis|"
    r"influenza|HIV|AIDS|malignancy|cancer|carcinoma|lymphoma|leukemia|"
    r"Alzheimer|Parkinson|epilepsy|schizophrenia|bipolar disorder|depression|"
    r"anxiety disorder|PTSD|autism|ADHD|hypothyroidism|hyperthyroidism|"
    r"rheumatoid arthritis|lupus|psoriasis|Crohn|ulcerative colitis|"
    r"osteoporosis|osteoarthritis|gout|ankylosing spondylitis|"
    r"acute kidney injury|AKI|chronic liver disease|fatty liver|NAFLD|"
    r"pulmonary embolism|deep vein thrombosis|DVT|atrial fibrillation|"
    r"acute coronary syndrome|ACS|unstable angina|STEMI|NSTEMI)\b",
]

# Dosage patterns: e.g. "500 mg", "2.5 mg/kg", "10 mg twice daily"
DOSAGE_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|mEq|mmol|IU|units?)\b(?:\s*/\s*(?:kg|m2|day|dose|hour|week))?",
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g)\s*(?:twice|thrice|once|three times?|four times?)\s*(?:daily|a day|per day)\b",
    r"\b(?:once|twice|thrice|three times?)\s*(?:daily|a day|weekly|per week)\b",
    r"\b(?:BD|TDS|QID|OD|QDS|PRN|SOS|nocte|mane)\b",
]

# Contraindication patterns
CONTRAINDICATION_PATTERNS = [
    r"\b(?:contraindicated?|avoid(?:ed|ing)?\s+in|not\s+(?:recommended|used?|given?)\s+in|"
    r"should\s+not\s+(?:be\s+)?used?|do\s+not\s+(?:use|administer)|"
    r"caution\s+in|use\s+with\s+caution)\b",
]

# Safety / warning patterns
SAFETY_PATTERNS = [
    r"\b(?:warning|caution|alert|black.?box|serious\s+adverse|"
    r"life.?threatening|fatal|death|toxicity|overdose|"
    r"anaphylaxis|anaphylactic|hypersensitivity|severe\s+reaction|"
    r"QT\s*prolongation|nephrotoxic|hepatotoxic|cardiotoxic|neurotoxic|"
    r"teratogenic|pregnancy\s*category\s*[DX]|fetal\s+(?:harm|risk)|"
    r"renal\s+failure|liver\s+failure|bone\s+marrow\s+suppression)\b",
]

# Patient population patterns
POPULATION_PATTERNS = [
    r"\b(?:paediatric|pediatric|child(?:ren)?|infant|neonat(?:e|al)|"
    r"adult|elderly|geriatric|older\s+adults?|"
    r"pregnant|pregnancy|lactating|breastfeeding|"
    r"immunocompromised|immunodeficient|HIV.positive|"
    r"renal(?:\s+impairment)?|hepatic(?:\s+impairment)?|"
    r"diabetic|hypertensive|obese|underweight)\b",
]

# Clinical outcome patterns
OUTCOME_PATTERNS = [
    r"\b(?:mortality|morbidity|survival|cure\s+rate|response\s+rate|"
    r"remission|relapse|recurrence|hospitalization|readmission|"
    r"adverse\s+(?:event|effect|reaction)|side\s+effect|"
    r"quality\s+of\s+life|QoL|clinical\s+outcome|"
    r"resolution|recovery|improvement|deterioration|progression)\b",
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
    Returns dict with diseases, drugs, symptoms, dosages, contraindications,
    patient_populations, outcomes, and has_safety_flag.
    """
    diseases: list[str] = []
    drugs: list[str] = []
    symptoms: list[str] = []
    dosages: list[str] = []
    contraindications: list[str] = []
    patient_populations: list[str] = []
    outcomes: list[str] = []
    has_safety_flag = False

    # Rule-based extraction (always runs)
    for pattern in DRUG_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        drugs.extend([m.lower() for m in matches])

    for pattern in DISEASE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        diseases.extend([m.lower() for m in matches])

    for pattern in DOSAGE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dosages.extend([m.strip() for m in matches if m.strip()])

    for pattern in CONTRAINDICATION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            # Extract surrounding context (up to 80 chars) as the contraindication note
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start = max(0, m.start() - 10)
                end = min(len(text), m.end() + 80)
                snippet = text[start:end].strip()
                contraindications.append(snippet[:120])
            break  # one pass is enough

    for pattern in SAFETY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            has_safety_flag = True
            break

    for pattern in POPULATION_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        patient_populations.extend([m.lower().strip() for m in matches])

    for pattern in OUTCOME_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        outcomes.extend([m.lower().strip() for m in matches])

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

    # Deduplicate and limit lists
    return {
        "diseases": list(dict.fromkeys(diseases))[:10],
        "drugs": list(dict.fromkeys(drugs))[:10],
        "symptoms": list(dict.fromkeys(symptoms))[:10],
        "dosages": list(dict.fromkeys(dosages))[:10],
        "contraindications": list(dict.fromkeys(contraindications))[:5],
        "patient_populations": list(dict.fromkeys(patient_populations))[:10],
        "outcomes": list(dict.fromkeys(outcomes))[:10],
        "has_safety_flag": has_safety_flag,
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

    # Each pattern group covers a distinct biological context; a chunk must match
    # at least 2 groups to be classified as preclinical (reduces false positives
    # from texts that mention only one animal/mechanism term incidentally).
    preclinical_signals = [
        r"\b(?:mouse|rat|animal model|in vivo)\b",
        r"\b(?:in vitro|cell line|cell culture|primary cells)\b",
        r"\b(?:assay|mechanism|pathway|molecular target)\b",
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
    if re.search(r"\b(?:guideline|recommendation|protocol|ICMR|WHO|NMC|MoHFW|CDC|NIH)\b", combined):
        return "guideline", 2
    if re.search(r"\b(?:review article|narrative review|literature review)\b", combined):
        return "review", 3
    if re.search(r"\b(?:case report|case series|case study)\b", combined):
        return "case_report", 4

    return "unknown", 5
