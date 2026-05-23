from __future__ import annotations

from models.schemas import ChunkInput
from services.rerankers.base import BaseReranker


class NoopReranker(BaseReranker):
    def score(self, query: str, chunks: list[ChunkInput]) -> list[float]:
        return [0.0 for _ in chunks]
