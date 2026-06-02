"""websearchtools — separate file per tool, all standalone functions."""
from src.tools.websearchtools.extract_domain import extract_domain, is_same_domain
from src.tools.websearchtools.extract_snippet import extract_snippet, extract_text_blocks
from src.tools.websearchtools.filter_medical import (
    is_medical_domain, filter_medical, list_medical_domains,
)
from src.tools.websearchtools.rank_results import rank_by_keywords, top_n
from src.tools.websearchtools.group_by_domain import group_by_domain, count_per_domain
from src.tools.websearchtools.deduplicate import deduplicate_by_url, deduplicate_by_title

__all__ = [
    "extract_domain", "is_same_domain",
    "extract_snippet", "extract_text_blocks",
    "is_medical_domain", "filter_medical", "list_medical_domains",
    "rank_by_keywords", "top_n",
    "group_by_domain", "count_per_domain",
    "deduplicate_by_url", "deduplicate_by_title",
]
