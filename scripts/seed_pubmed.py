import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.parsers.pubmed import PubMedParser
from src.ingestion.pipeline import run_pipeline

INDIA_QUERIES = [
    "tuberculosis drug resistant India treatment 2020 2021 2022 2023",
    "dengue management India clinical guidelines",
    "leptospirosis India diagnosis treatment",
    "snakebite envenomation India management",
    "typhoid enteric fever India antibiotic treatment",
    "malaria India Plasmodium treatment guidelines",
    "COVID-19 India clinical management",
    "sepsis India ICU management",
    "acute respiratory infection India pediatric",
    "diabetic ketoacidosis India management",
]


def main() -> None:
    all_documents = []

    for idx, query in enumerate(INDIA_QUERIES, start=1):
        print(f"[{idx}/{len(INDIA_QUERIES)}] Fetching PubMed query: {query}")
        parser = PubMedParser(query=query, max_results=200)
        docs = parser.parse()
        print(f"  Parsed documents: {len(docs)}")
        all_documents.extend(docs)

        if idx < len(INDIA_QUERIES):
            time.sleep(2)

    summary = asyncio.run(run_pipeline(all_documents))

    print("PubMed ingestion complete")
    print(f"Queries run: {len(INDIA_QUERIES)}")
    print(f"Documents created: {summary['documents_stored']}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Chunks embedded: {summary['chunks_embedded']}")


if __name__ == "__main__":
    main()
