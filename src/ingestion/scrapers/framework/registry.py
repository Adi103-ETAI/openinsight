"""Source registry — get_scraper(), list_sources().

Sources register themselves via the @register_source decorator in their
source file. The first call to get_scraper() imports all source modules
to populate the registry.
"""
from __future__ import annotations

from typing import Any

from src.ingestion.scrapers.framework.base import _SOURCE_REGISTRY, BaseScraper, SourceConfig

_SOURCES_IMPORTED = False


def _import_all_sources() -> None:
    """Import all source modules so their @register_source decorators run."""
    global _SOURCES_IMPORTED
    if _SOURCES_IMPORTED:
        return
    _SOURCES_IMPORTED = True
    # Import all source modules — each registers itself via @register_source
    # Phase 0.5: PubMed
    # Phase 1: IndMED, Medknow, PMC India
    # Phase 2: StatPearls, NCBI Bookshelf, NMC curriculum, Govt manuals (NTEP/NVBDCP/NHM/NPCDS)
    # Phase 3: CDSCO, CTRI, PvPI (NFI parser exists but no scraper — needs PDF access)
    # Phase 4: Specialty societies (RSSDI, CSI, ISCCM, IAP, FOGSI, AIOS, ISN)
    for module_name in [
        "src.ingestion.scrapers.sources.pubmed",
        "src.ingestion.scrapers.sources.indmed",
        "src.ingestion.scrapers.sources.medknow",
        "src.ingestion.scrapers.sources.pmc_india",
        "src.ingestion.scrapers.sources.statpearls",
        "src.ingestion.scrapers.sources.ncbi_bookshelf",
        "src.ingestion.scrapers.sources.nmc_curriculum",
        "src.ingestion.scrapers.sources.govt_manuals",
        "src.ingestion.scrapers.sources.cdsco",
        "src.ingestion.scrapers.sources.ctri",
        "src.ingestion.scrapers.sources.pvpi",
        "src.ingestion.scrapers.sources.specialty_societies",
    ]:
        try:
            __import__(module_name)
        except Exception as e:
            # Source may have optional deps; log and continue
            import loguru
            loguru.logger.debug(f"[registry] could not import {module_name}: {e}")


def list_sources() -> list[str]:
    """List all registered source names."""
    _import_all_sources()
    return sorted(_SOURCE_REGISTRY.keys())


def get_scraper(
    name: str,
    http_client: Any | None = None,
    cache: Any | None = None,
    rate_limiter: Any | None = None,
    dedup_index: Any | None = None,
) -> BaseScraper:
    """Instantiate a scraper by name.

    Args:
        name: source name (e.g. "pubmed", "indmed")
        http_client, cache, rate_limiter, dedup_index: optional DI overrides

    Raises:
        KeyError: if source is not registered
    """
    _import_all_sources()
    if name not in _SOURCE_REGISTRY:
        raise KeyError(f"Source '{name}' not registered. Available: {list_sources()}")
    cls = _SOURCE_REGISTRY[name]
    return cls(
        http_client=http_client,
        cache=cache,
        rate_limiter=rate_limiter,
        dedup_index=dedup_index,
    )


def register_source(name: str) -> Any:
    """Public re-export of the register_source decorator."""
    from src.ingestion.scrapers.framework.base import register_source as _register
    return _register(name)
