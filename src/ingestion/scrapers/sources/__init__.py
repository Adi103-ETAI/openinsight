"""Source-specific scraper implementations.

Each source file registers a BaseScraper subclass via @register_source.
Sources added per phase:
    Phase 0.5: pubmed.py (refactored — moves Entrez fetching out of PubMedParser)
    Phase 1:   indmed.py, medknow.py, pmc_india.py
    Phase 2:   ncbi_bookshelf.py, nmc.py, govt_manuals.py
    Phase 3:   nfi.py, cdsco.py, ctri.py, pvpi.py
    Phase 4:   ntep.py, nvbdcp.py, csi.py, specialty_society.py
    Phase 5:   nfhs.py, ncdir.py
"""
from __future__ import annotations
