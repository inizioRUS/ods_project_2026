from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


JsonDict = dict[str, Any]


class DocumentInput(BaseModel):
    id: str | None = None
    text: str = Field(min_length=1)
    metadata: JsonDict = Field(default_factory=dict)


class ChunkConfigInput(BaseModel):
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    min_chunk_size: int | None = None


class ChunkInput(BaseModel):
    id: str
    doc_id: str
    text: str
    metadata: JsonDict = Field(default_factory=dict)
    chunk_index: int


class ChunkRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    ids: list[str] | None = None
    metadata: list[JsonDict] | None = None
    config: ChunkConfigInput | None = None


class ChunkResponse(BaseModel):
    chunks: list[ChunkInput]


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    input_type: Literal["query", "passage"] = "passage"


class EmbeddingResponse(BaseModel):
    model: str
    dim: int
    embeddings: list[list[float]]


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)
    reset: bool = False
    chunk: bool = True
    chunk_config: ChunkConfigInput | None = None


class IngestResponse(BaseModel):
    documents_count: int
    chunks_count: int
    index_size: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    semantic_top_k: int | None = None
    bm25_top_k: int | None = None
    final_top_k: int | None = None
    fusion_method: Literal["rrf", "weighted_zscore", "weighted_sum"] | None = None
    rerank: bool = True
    filters: JsonDict | None = None


class ScoreBreakdown(BaseModel):
    semantic: float | None = None
    bm25: float | None = None
    fused: float | None = None
    rerank: float | None = None
    final: float | None = None


class SearchResult(BaseModel):
    id: str
    doc_id: str
    text: str
    metadata: JsonDict = Field(default_factory=dict)
    chunk_index: int
    scores: ScoreBreakdown


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class GenerateRequest(SearchRequest):
    max_context_chars: int = 8000


class Citation(BaseModel):
    marker: str
    chunk_id: str
    doc_id: str
    text: str
    metadata: JsonDict = Field(default_factory=dict)
    scores: ScoreBreakdown


class GenerateResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]


class ResetResponse(BaseModel):
    ok: bool
    index_size: int
