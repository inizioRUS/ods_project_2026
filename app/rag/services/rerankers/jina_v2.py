from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config import RerankerConfig
from models.schemas import ChunkInput
from services.rerankers.base import BaseReranker


@dataclass
class JinaV2Reranker(BaseReranker):
    config: RerankerConfig
    _tokenizer: AutoTokenizer | None = None
    _model: AutoModelForSequenceClassification | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            self._model.to(self.config.device)
            self._model.eval()
        return self._model

    def score(self, query: str, chunks: list[ChunkInput]) -> list[float]:
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]

        # Jina rerankers expose compute_score in trust_remote_code models.
        if hasattr(self.model, "compute_score"):
            scores = self.model.compute_score(
                pairs,
                max_length=self.config.max_length,
                batch_size=self.config.batch_size,
            )
            if isinstance(scores, float):
                return [scores]
            return [float(s) for s in scores]

        all_scores: list[float] = []
        for start in range(0, len(pairs), self.config.batch_size):
            batch = pairs[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.config.device)
            with torch.no_grad():
                output = self.model(**encoded)
                logits = output.logits
                if logits.ndim == 2 and logits.shape[1] > 1:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.reshape(-1)
            all_scores.extend([float(x) for x in batch_scores.detach().cpu().tolist()])
        return all_scores
