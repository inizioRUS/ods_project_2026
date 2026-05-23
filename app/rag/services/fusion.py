from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal

import numpy as np

from models.schemas import ChunkInput, ScoreBreakdown, SearchResult


class CandidatePool:
    def __init__(self) -> None:
        self.items: dict[str, ChunkInput] = {}
        self.scores: dict[str, ScoreBreakdown] = defaultdict(ScoreBreakdown)
        self.ranks: dict[str, dict[str, int]] = defaultdict(dict)

    def add(self, source: Literal["semantic", "bm25"], rows: list[dict]) -> None:
        for row in rows:
            chunk: ChunkInput = row["chunk"]
            self.items[chunk.id] = chunk
            setattr(self.scores[chunk.id], source, float(row["score"]))
            self.ranks[chunk.id][source] = int(row.get("rank", 10**9))


def _zscore(values: dict[str, float | None]) -> dict[str, float]:
    present = {k: float(v) for k, v in values.items() if v is not None}
    if not present:
        return {}
    arr = np.array(list(present.values()), dtype="float32")
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return {k: 0.0 for k in present}
    return {k: (v - mean) / std for k, v in present.items()}


def fuse_candidates(
    pool: CandidatePool,
    method: Literal["rrf", "weighted_zscore", "weighted_sum"] = "rrf",
    semantic_weight: float = 0.55,
    bm25_weight: float = 0.45,
    rrf_k: int = 60,
) -> list[SearchResult]:
    if method == "rrf":
        fused: dict[str, float] = {}
        for chunk_id in pool.items:
            score = 0.0
            if "semantic" in pool.ranks[chunk_id]:
                score += semantic_weight / (rrf_k + pool.ranks[chunk_id]["semantic"])
            if "bm25" in pool.ranks[chunk_id]:
                score += bm25_weight / (rrf_k + pool.ranks[chunk_id]["bm25"])
            fused[chunk_id] = score
    elif method == "weighted_zscore":
        semantic_z = _zscore({k: v.semantic for k, v in pool.scores.items()})
        bm25_z = _zscore({k: v.bm25 for k, v in pool.scores.items()})
        fused = {}
        for chunk_id in pool.items:
            fused[chunk_id] = semantic_weight * semantic_z.get(chunk_id, 0.0) + bm25_weight * bm25_z.get(chunk_id, 0.0)
    elif method == "weighted_sum":
        fused = {}
        for chunk_id, scores in pool.scores.items():
            fused[chunk_id] = semantic_weight * (scores.semantic or 0.0) + bm25_weight * math.log1p(scores.bm25 or 0.0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    results: list[SearchResult] = []
    for chunk_id, chunk in pool.items.items():
        scores = pool.scores[chunk_id]
        scores.fused = fused.get(chunk_id, 0.0)
        scores.final = scores.fused
        results.append(
            SearchResult(
                id=chunk.id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index,
                scores=scores,
            )
        )
    return sorted(results, key=lambda r: r.scores.final or 0.0, reverse=True)


def apply_rerank(results: list[SearchResult], rerank_scores: list[float], rerank_weight: float = 1.0) -> list[SearchResult]:
    if len(results) != len(rerank_scores):
        raise ValueError("results and rerank_scores length mismatch")

    z = _zscore({r.id: s for r, s in zip(results, rerank_scores, strict=False)})
    for result, raw in zip(results, rerank_scores, strict=False):
        result.scores.rerank = float(raw)
        # Keep fused signal, then add normalized reranker signal.
        result.scores.final = (result.scores.fused or 0.0) + rerank_weight * z.get(result.id, 0.0)
    return sorted(results, key=lambda r: r.scores.final or 0.0, reverse=True)
