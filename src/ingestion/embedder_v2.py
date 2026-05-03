from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class DualEmbedderV2:
    """
    Dense + sparse embedder for OpenInsight v2 ingestion/search.

    Dense embeddings are generated from contextual_text so chunk semantics
    include source, type, title, and section context.
    """

    VOCAB_SIZE = 50000

    MEDICAL_COMPOUNDS = [
        "type 2 diabetes",
        "type 1 diabetes",
        "heart failure",
        "blood pressure",
        "myocardial infarction",
        "atrial fibrillation",
        "blood glucose",
        "hemoglobin a1c",
        "hba1c",
        "randomized controlled trial",
        "systematic review",
        "meta analysis",
        "meta-analysis",
        "insulin resistance",
        "glycemic control",
        "renal failure",
        "coronary artery disease",
        "coronary heart disease",
    ]

    STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "we",
        "our",
        "they",
        "their",
        "as",
        "if",
        "than",
        "more",
        "most",
        "such",
    }

    def __init__(self, dense_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"):
        self.dense_model = SentenceTransformer(dense_model_name)
        self.dense_model.eval()
        if torch.cuda.is_available():
            self.dense_model = self.dense_model.cuda()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        with torch.inference_mode():
            embeddings = self.dense_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embeddings

    def embed_query(self, query_text: str) -> np.ndarray:
        with torch.inference_mode():
            embedding = self.dense_model.encode(
                query_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embedding

    def compute_sparse_vector(self, text: str) -> dict[str, list[float] | list[int]]:
        tokens = self._medical_tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        tf = Counter(tokens)
        total_tokens = len(tokens)
        # Sparse vector backends require unique indices; hash collisions can
        # produce duplicates, so merge weights by index.
        weight_by_index: dict[int, float] = defaultdict(float)

        for term, count in tf.items():
            term_idx = self._term_to_index(term)
            tf_norm = count / max(1, total_tokens)
            idf_weight = self._get_idf_weight(term)
            weight = tf_norm * idf_weight
            if weight > 0.001:
                weight_by_index[term_idx] += float(weight)

        if not weight_by_index:
            return {"indices": [], "values": []}

        sorted_indices = sorted(weight_by_index.keys())
        sorted_values = [weight_by_index[idx] for idx in sorted_indices]

        return {"indices": sorted_indices, "values": sorted_values}

    def _medical_tokenize(self, text: str) -> list[str]:
        text_lower = text.lower()
        compound_tokens: list[str] = []

        for compound in self.MEDICAL_COMPOUNDS:
            token_version = compound.replace(" ", "_")
            if compound in text_lower:
                text_lower = text_lower.replace(compound, f" {token_version} ")
                compound_tokens.append(token_version)

        words = re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", text_lower)
        words = [w for w in words if w not in self.STOPWORDS]
        return compound_tokens + words

    def _term_to_index(self, term: str) -> int:
        return abs(hash(term)) % self.VOCAB_SIZE

    def _get_idf_weight(self, term: str) -> float:
        if "_" in term:
            return 3.5
        if len(term) > 10:
            return 3.0
        if len(term) > 6:
            return 2.0
        return 1.0
