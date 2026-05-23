from __future__ import annotations

from core.config import RerankerConfig
from services.rerankers.base import BaseReranker
from services.rerankers.jina_v2 import JinaV2Reranker
from services.rerankers.noop import NoopReranker


def build_reranker(config: RerankerConfig) -> BaseReranker:
    if config.provider == "noop":
        return NoopReranker()
    if config.provider == "jina_v2":
        return JinaV2Reranker(config)
    raise ValueError(f"Unknown reranker provider: {config.provider}")
