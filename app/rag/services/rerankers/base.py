from __future__ import annotations

from abc import ABC, abstractmethod

from models.schemas import ChunkInput


class BaseReranker(ABC):
    @abstractmethod
    def score(self, query: str, chunks: list[ChunkInput]) -> list[float]:
        raise NotImplementedError
