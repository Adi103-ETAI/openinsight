from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.settings import get_settings
from .retriever import RetrievedChunk


class CrossEncoderReranker:
    MODEL_NAME = "BAAI/bge-reranker-base"

    def __init__(self) -> None:
        settings = get_settings()
        model_name = settings.reranker_model_name or self.MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 8,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [[query, chunk.text[:512]] for chunk in chunks]

        try:
            with torch.inference_mode():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits.squeeze(-1)
                scores: np.ndarray = logits.detach().cpu().numpy()

            for chunk, score in zip(chunks, scores):
                chunk.score = float(score)

            ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
            return ranked[:top_k]
        except (RuntimeError, ValueError, TypeError):
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_k]
