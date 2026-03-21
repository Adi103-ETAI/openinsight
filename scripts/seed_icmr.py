import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.parsers.icmr import ICMRParser
from src.ingestion.pipeline import run_pipeline


def main() -> None:
    icmr_dir = Path("data/raw/icmr")
    if not icmr_dir.exists():
        print("Directory not found: data/raw/icmr/. Create it and add ICMR PDF files first.")
        return

    pdf_files = sorted(icmr_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/raw/icmr/. Add at least one .pdf and rerun.")
        return

    all_documents = []
    files_parsed = 0

    for pdf_path in pdf_files:
        parser = ICMRParser(pdf_path)
        docs = parser.parse()
        if docs:
            files_parsed += 1
            all_documents.extend(docs)

    summary = asyncio.run(run_pipeline(all_documents))

    print("ICMR ingestion complete")
    print(f"Files found: {len(pdf_files)}")
    print(f"Files parsed: {files_parsed}")
    print(f"Documents created: {summary['documents_stored']}")
    print(f"Chunks created: {summary['chunks_created']}")
    print(f"Chunks embedded: {summary['chunks_embedded']}")


if __name__ == "__main__":
    main()
