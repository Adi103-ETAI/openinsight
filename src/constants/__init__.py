from dataclasses import dataclass
from typing import Final


@dataclass
class EvidenceBoost:
    """Evidence level boost scores for ranking."""
    
    # From fusion.py and metadata_v2.py - consolidated
    LEVEL_1A: Final[float] = 1.35  # RCT meta-analysis
    LEVEL_1B: Final[float] = 1.25  # Single RCT
    LEVEL_2A: Final[float] = 1.15  # Cohort study
    LEVEL_2B: Final[float] = 1.10  # Case control
    LEVEL_3: Final[float] = 1.05   # Case series
    LEVEL_4: Final[float] = 1.00   # Expert consensus
    LEVEL_5: Final[float] = 1.10   # Expert opinion
    UNKNOWN: Final[float] = 1.00

    @classmethod
    def get_boost(cls, level: str) -> float:
        level_lower = str(level).lower().strip()
        if not level_lower or level_lower == "unknown":
            return cls.UNKNOWN
        if level_lower.startswith("1a"):
            return cls.LEVEL_1A
        if level_lower.startswith("1b"):
            return cls.LEVEL_1B
        if level_lower.startswith("2a"):
            return cls.LEVEL_2A
        if level_lower.startswith("2b"):
            return cls.LEVEL_2B
        if level_lower.startswith("3"):
            return cls.LEVEL_3
        if level_lower.startswith("4"):
            return cls.LEVEL_4
        if level_lower.startswith("5"):
            return cls.LEVEL_5
        return cls.UNKNOWN


@dataclass
class RecencyBoost:
    """Recency boost scores for ranking."""
    
    YEAR_2026: Final[float] = 1.12
    YEAR_2025: Final[float] = 1.10
    YEAR_2024: Final[float] = 1.08
    YEAR_2023: Final[float] = 1.05
    YEAR_2022: Final[float] = 1.03
    PRE_2022: Final[float] = 1.00

    @classmethod
    def get_boost(cls, year: int) -> float:
        if year >= 2026:
            return cls.YEAR_2026
        if year == 2025:
            return cls.YEAR_2025
        if year == 2024:
            return cls.YEAR_2024
        if year == 2023:
            return cls.YEAR_2023
        if year == 2022:
            return cls.YEAR_2022
        return cls.PRE_2022


# RRF (Reciprocal Rank Fusion) constants
RRF_K: Final[int] = 60

# Cache key settings
CACHE_KEY_PREFIX_LENGTH: Final[int] = 16

# Thread pool defaults
DEFAULT_MAX_WORKERS: Final[int] = 4

# Chunk text limits
RERANKER_MAX_CHARS: Final[int] = 512
DEEP_INSIGHTS_CONTEXT_CHARS: Final[int] = 300
DEEP_INSIGHTS_MAX_SUB_QUERIES: Final[int] = 6

# Timeout defaults
LLM_TIMEOUT: Final[float] = 60.0
HYDE_TIMEOUT: Final[float] = 15.0

# Quality scoring
QUALITY_MIN_TOKENS: Final[int] = 50
QUALITY_TARGET_TOKENS: Final[int] = 300
QUALITY_HIGH_VALUE_BONUS: Final[float] = 0.15
QUALITY_LOW_VALUE_PENALTY: Final[float] = 0.5


# Evidence level mapping for quality scoring
EVIDENCE_LEVELS = {
    "1a": "RCT Meta-analysis",
    "1b": "Single RCT",
    "2a": "Cohort Study",
    "2b": "Case Control",
    "3": "Case Series",
    "4": "Expert Consensus",
    "5": "Expert Opinion",
}