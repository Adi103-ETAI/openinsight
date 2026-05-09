from __future__ import annotations

import numpy as np

from .retriever import RetrievedChunk


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def maximal_marginal_relevance(
    chunks: list[RetrievedChunk],
    embedder,
    lambda_param: float = 0.7,
    n_select: int = 6,
) -> list[RetrievedChunk]:
    if len(chunks) <= n_select:
        return chunks

    texts = [chunk.text[:400] for chunk in chunks]
    embeddings = embedder.embed_batch(texts, batch_size=16)

    selected_indices: list[int] = []
    candidate_indices = list(range(len(chunks)))

    first_idx = max(candidate_indices, key=lambda i: chunks[i].score)
    selected_indices.append(first_idx)
    candidate_indices.remove(first_idx)

    while len(selected_indices) < n_select and candidate_indices:
        best_idx = None
        best_mmr_score = -float("inf")

        for idx in candidate_indices:
            relevance = float(chunks[idx].score)
            max_sim = max(
                cosine_similarity(embeddings[idx], embeddings[selected_idx])
                for selected_idx in selected_indices
            )
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return [chunks[i] for i in selected_indices]
