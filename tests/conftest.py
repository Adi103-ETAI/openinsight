"""
OpenInsight Test Suite - Shared Fixtures and Configuration

Provides reusable fixtures, mock helpers, and test utilities
for unit and integration tests across the OpenInsight codebase.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_settings(monkeypatch):
    """Isolate settings from environment variables to ensure test reproducibility."""
    # Override all env-driven settings with safe test defaults
    test_env = {
        "MONGODB_URL": "mongodb://localhost:27017/test_db",
        "MONGODB_DB": "test_openinsight",
        "VECTOR_STORE_PROVIDER": "memory",
        "EMBED_PROVIDER": "local",
        "RERANK_PROVIDER": "local",
        "GROBID_URL": "http://localhost:8070",
        "DEAD_LETTER_ENABLED": "false",
        "QUALITY_SCORE_THRESHOLD": "0.0",
        "HYDE_ENABLED": "false",
        "SPACY_MODEL": "en_core_web_sm",
        "EMBEDDING_DIM": "768",
        "VECTOR_DIM": "768",
        "SPARSE_VOCAB_SIZE": "50000",
        "DENSE_MODEL_NAME": "pritamdeka/S-PubMedBert-MS-MARCO",
        "HF_API_TOKEN": "",
        "HF_EMBED_MODEL": "",
        "HF_RERANK_MODEL": "",
        "COHERE_API_KEY": "",
        "COHERE_EMBED_MODEL": "embed-english-v3.0",
        "COHERE_RERANK_MODEL": "rerank-english-v3.0",
        "NVIDIA_NIM_BASE_URL": "",
        "NVIDIA_NIM_API_KEY": "",
        "NIM_MODEL": "",
        "RERANKER_MODEL_NAME": "BAAI/bge-reranker-v2-m3",
        "RERANKER_MAX_LENGTH": "512",
        "RERANKER_MAX_CHARS": "500",
        "PARSING_THREAD_WORKERS": "2",
        "INGESTION_THREAD_WORKERS": "2",
        "DEAD_LETTER_COLLECTION": "dead_letter",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    yield


# ---------------------------------------------------------------------------
# Mock vector store
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    """Create a mock vector store that simulates Zilliz/Qdrant behavior."""
    store = MagicMock()
    store.client = MagicMock()
    store.client.has_collection.return_value = True
    store.client.load_collection = MagicMock()
    store.search_dense.return_value = []
    store.search_sparse.return_value = []
    store.upsert.return_value = 0
    return store


@pytest.fixture
def mock_vector_store_with_results():
    """Create a mock vector store with predefined search results."""
    from src.vectorstore.types import ScoredPoint, SparseVector

    store = MagicMock()
    store.client = MagicMock()
    store.client.has_collection.return_value = True
    store.client.load_collection = MagicMock()
    store.search_dense.return_value = []
    store.search_sparse.return_value = []
    store.upsert.return_value = 0

    def make_scored_point(
        point_id: str,
        score: float,
        chunk_id: str = "",
        doc_id: str = "",
        chunk_text: str = "",
        contextual_text: str = "",
        metadata: dict | None = None,
    ) -> ScoredPoint:
        payload = {
            "chunk_id": chunk_id or point_id,
            "doc_id": doc_id,
            "chunk_text": chunk_text,
            "contextual_text": contextual_text,
            **(metadata or {}),
        }
        return ScoredPoint(
            point_id=point_id,
            score=score,
            payload=payload,
        )

    store.make_scored_point = make_scored_point
    return store


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic embeddings."""

    class MockEmbedder:
        """Mock embedder returning simple deterministic vectors."""

        def __init__(self, dim: int = 8):
            self._dim = dim

        def embed_batch(
            self, texts: list[str], batch_size: int = 32
        ) -> tuple[np.ndarray, list[int]]:
            embeddings = np.zeros((len(texts), self._dim), dtype=np.float32)
            for i, text in enumerate(texts):
                # Simple hash-based embedding for determinism
                for j, char in enumerate(text[: self._dim]):
                    embeddings[i, j % self._dim] = ord(char) / 255.0
            return embeddings, []

        def embed_query(self, query_text: str) -> np.ndarray:
            embedding = np.zeros(self._dim, dtype=np.float32)
            for j, char in enumerate(query_text[: self._dim]):
                embedding[j % self._dim] = ord(char) / 255.0
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

        def dimension(self) -> int:
            return self._dim

        def compute_sparse_vector(self, text: str) -> dict[str, list]:
            return {"indices": [0, 1, 2], "values": [0.1, 0.2, 0.3]}

    return MockEmbedder(dim=8)


# ---------------------------------------------------------------------------
# Mock MongoDB
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mongo_client():
    """Create a mock MongoDB client."""
    client = MagicMock()
    client.db = MagicMock()
    client.db.__getitem__ = MagicMock(return_value=MagicMock())
    return client


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document():
    """Create a sample document dict for testing."""
    return {
        "doc_id": "pmid_12345",
        "title": "Treatment of Multidrug-Resistant Tuberculosis",
        "abstract": "This study examines treatment protocols for MDR-TB patients.",
        "content": "Bedaquiline is recommended for MDR-TB treatment regimens. "
        "The standard dose is 200mg daily for 2 weeks, then 100mg three times weekly.",
        "authors": ["Smith J", "Doe A"],
        "year": 2024,
        "journal": "Journal of Infectious Diseases",
        "doi": "10.1234/test.2024",
        "pmid": "12345",
        "mesh_terms": ["Tuberculosis", "Drug Resistance"],
        "keywords": ["MDR-TB", "bedaquiline"],
        "sections": [],
        "url": "/test/path/doc.xml",
        "source_type": "pubmed",
    }


@pytest.fixture
def sample_chunk():
    """Create a sample ChunkV3 for testing."""
    from src.ml.chunking.chunker import ChunkV3

    return ChunkV3(
        chunk_id="doc_001_chunk_0",
        doc_id="doc_001",
        chunk_type="text",
        section_title="Introduction",
        text="Bedaquiline is recommended for MDR-TB treatment.",
        contextual_text="Introduction\nBedaquiline is recommended for MDR-TB treatment.",
        char_count=50,
        token_estimate=10,
        chunk_index=0,
        total_chunks=3,
        metadata={"source": "pubmed", "year": 2024},
    )


@pytest.fixture
def sample_retrieved_chunks():
    """Create a list of sample RetrievedChunk objects."""
    from src.query.search.retriever import RetrievedChunk

    return [
        RetrievedChunk(
            chunk_id=f"chunk_{i}",
            doc_id=f"doc_{i}",
            score=0.9 - i * 0.1,
            text=f"Medical content about treatment {i}.",
            contextual_text=f"Section {i}\nMedical content about treatment {i}.",
            metadata={"evidence_level": 1, "year": 2024 - i, "source_type": "pubmed"},
            retrieval_source="dense",
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_chunk_record():
    """Create a sample ChunkRecord for testing."""
    from src.ingestion.document_db import ChunkRecord

    return ChunkRecord(
        document_id="doc1",
        source_type="pubmed",
        title="Test Document",
        chunk_text="The patient should receive doxycycline 100 mg twice daily.",
        chunk_index=0,
        content_type="clinical",
        content_weight=1.5,
        evidence_level=1,
        token_count=50,
        diseases=["tuberculosis"],
        drugs=["doxycycline"],
        dosages=["100 mg"],
        symptoms=[],
        contraindications=[],
        patient_populations=[],
        outcomes=[],
        has_safety_flag=False,
        quality_score=0.0,
    )


@pytest.fixture
def sample_document_record():
    """Create a sample DocumentRecord for testing."""
    from src.ingestion.document_db import DocumentRecord

    return DocumentRecord(
        source_type="pubmed",
        title="Efficacy of Doxycycline in Treating Malaria",
        content="A systematic review of doxycycline efficacy in malaria treatment protocols.",
        year=2024,
        journal="Tropical Medicine Journal",
        study_type="systematic_review",
        evidence_level=1,
    )


# ---------------------------------------------------------------------------
# Temporary directory helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_xml_file(tmp_path: Path) -> Path:
    """Create a temporary PubMed XML file for testing."""
    xml_content = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>Test Article on MDR-TB Treatment</ArticleTitle>
        <Abstract>
          <AbstractText>This is a test abstract about tuberculosis treatment.</AbstractText>
        </Abstract>
        <Journal>
          <Title>Test Journal</Title>
          <JournalIssue>
            <PubDate><Year>2024</Year></PubDate>
          </JournalIssue>
        </Journal>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Tuberculosis</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
    xml_file = tmp_path / "test_article.xml"
    xml_file.write_text(xml_content, encoding="utf-8")
    return xml_file


@pytest.fixture
def temp_invalid_xml_file(tmp_path: Path) -> Path:
    """Create a temporary invalid XML file for testing."""
    xml_file = tmp_path / "invalid.xml"
    xml_file.write_text("This is not valid XML content <unclosed>", encoding="utf-8")
    return xml_file


@pytest.fixture
def temp_pubmed_book_xml_file(tmp_path: Path) -> Path:
    """Create a temporary PubmedBookArticle XML file for testing."""
    xml_content = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedBookArticle>
    <BookDocument>
      <PMID>67890</PMID>
      <ArticleTitle>StatPearls: Tuberculosis Management</ArticleTitle>
      <Abstract>
        <AbstractText>Comprehensive guide to TB management.</AbstractText>
      </Abstract>
      <Book>
        <BookTitle>StatPearls</BookTitle>
        <PubDate><Year>2024</Year></PubDate>
      </Book>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Tuberculosis</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </BookDocument>
  </PubmedBookArticle>
</PubmedArticleSet>"""
    xml_file = tmp_path / "test_book.xml"
    xml_file.write_text(xml_content, encoding="utf-8")
    return xml_file


# ---------------------------------------------------------------------------
# Async test helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def async_mock():
    """Create an AsyncMock for async function patching."""
    return AsyncMock()


# ---------------------------------------------------------------------------
# Parametrization data
# ---------------------------------------------------------------------------

MEDICAL_QUERY_INTENT_PAIRS = [
    ("what causes tuberculosis", "diagnostic"),
    ("treatment for diabetes", "therapeutic"),
    ("prognosis of heart failure", "prognostic"),
    ("side effects of metformin", "drug_info"),
    ("WHO guideline for TB", "guideline"),
    ("what is pneumonia", "diagnostic"),
    ("how to treat hypertension", "therapeutic"),
    ("mortality rate of cancer", "prognostic"),
    ("drug interactions with warfarin", "drug_info"),
    ("ICMR recommendation for malaria", "guideline"),
    ("general medical question", "general"),
]

CONTENT_TYPE_CLASSIFICATION_PAIRS = [
    ("Patient should receive doxycycline 100 mg oral twice daily.", "clinical"),
    ("References\nAcknowledgements\nCopyright 2023", "noise"),
    ("Mouse models were used. In vitro cell line assay.", "preclinical"),
    ("Background information about the disease.", "background"),
    ("Standard dosing applies.", "clinical"),
]

EVIDENCE_LEVEL_BOOST_PAIRS = [
    (1, 1.5),  # Grade I: highest boost
    (2, 1.3),  # Grade II
    (3, 1.0),  # Grade III: neutral
    (4, 0.8),  # Grade IV: slight penalty
    (5, 0.5),  # Grade V: significant penalty
]
