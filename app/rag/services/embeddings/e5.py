from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EmbeddingConfig


@dataclass
class E5Embedder:
    config: EmbeddingConfig
    _model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
        return self._model

    def encode(self, texts: list[str], input_type: Literal["query", "passage"] = "passage") -> np.ndarray:
        prefix = self.config.query_prefix if input_type == "query" else self.config.passage_prefix
        prepared = [prefix + t for t in texts]
        vectors = self.model.encode(
            prepared,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        print(vectors)
        return np.asarray(vectors, dtype="float32")

    @property
    def model_name(self) -> str:
        return self.config.model_name
