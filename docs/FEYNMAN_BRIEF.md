# Feynman Collection Brief — OpenInsight (standing context)

Read this before every collection task. It defines what we need and why.

## What OpenInsight is

OpenInsight (SentArc Labs) is an AI clinical-intelligence platform for
**Indian physicians**. A doctor asks a clinical question; the system retrieves
evidence from trusted medical sources and returns a cited answer. Every claim
must trace to a verifiable source — so the corpus IS the product.

## Retrieval stack (why source quality matters)

Hybrid dense (PubMedBERT-768) + sparse retrieval over Milvus, cross-encoder
rerank, citation-verified generation. The pipeline ingests: discover → fetch →
parse → chunk → embed → upsert (Milvus `openinsight_v2` + MongoDB). A publish
gate (count reconciliation, dim check, golden-QA smoke test) accepts each
batch. Bad sources in = wrong answers out; there is no downstream fix for a
bad corpus.

## What to collect (priority order)

1. **Indian authority documents**: ICMR treatment guidelines, NMC curricula,
   NTEP/RNTCP TB guidelines, CDSCO drug labels, CTRI trial records,
   specialty-society guidelines (RSSDI, CSI, ISCCM, IAP, FOGSI, AIOS, ISN).
   Prefer official PDFs/portal pages with stable URLs.
2. **PubMed-indexed Indian research**: landmark PMIDs per specialty, with
   preference for Indian journals (Indian J Med Res, JAPI, Neurology India…).
3. **Global references**: WHO/CDC guidelines, Cochrane reviews, StatPearls /
   NCBI Bookshelf overviews for background coverage.
4. **Golden QA material**: real clinical questions doctors ask per topic, with
   the guideline answer + citation — used to grow our retrieval smoke tests.

## What NOT to collect

Predatory journals, pharma marketing brochures, patient-blog anecdotes,
paywalled content without an accessible abstract, non-English sources,
veterinary or purely Western-epidemiology material with no Indian relevance.

## Output contract (every task)

Return structured lists only — no essays:
`- title | source authority | stable URL or ID (PMID/NBK/DOI) | 1-line relevance`
Group by the priority tiers above. Flag anything uncertain rather than
inventing URLs — a wrong URL wastes a GPU ingestion run; a missing one just
waits for the next pack.
