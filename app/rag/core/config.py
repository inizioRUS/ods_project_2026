from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class AppConfig(BaseModel):
    title: str = "Basic Modular RAG API"
    version: str = "0.1.0"
    data_dir: str = "data/indexes/default"


class EmbeddingConfig(BaseModel):
    provider: Literal["e5"] = "e5"
    model_name: str = "intfloat/multilingual-e5-large-instruct"
    device: str = Field(default_factory=lambda: os.getenv("RAG_DEVICE", "cpu"))
    normalize: bool = True
    batch_size: int = 16
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


class ChunkingConfig(BaseModel):
    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_size: int = 80
    separators: list[str] = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


class RetrievalConfig(BaseModel):
    semantic_top_k: int = 30
    bm25_top_k: int = 30
    final_top_k: int = 8
    fusion_method: Literal["rrf", "weighted_zscore", "weighted_sum"] = "rrf"
    semantic_weight: float = 0.55
    bm25_weight: float = 0.45
    rrf_k: int = 60


class RerankerConfig(BaseModel):
    provider: Literal["jina_v2", "noop"] = "jina_v2"
    model_name: str = "jinaai/jina-reranker-v2-base-multilingual"
    device: str = Field(default_factory=lambda: os.getenv("RAG_DEVICE", "cpu"))
    batch_size: int = 8
    max_length: int = 1024
    weight: float = 1.0


class LLMConfig(BaseModel):
    provider: Literal["mistral", "noop"] = "mistral"
    model: str = Field(default_factory=lambda: os.getenv("MISTRAL_MODEL", "mistral-large-latest"))
    temperature: float = 0.1
    max_tokens: int = 900
    timeout_seconds: int = 60


class PromptConfig(BaseModel):
    system: str
    user_template: str


class Settings(BaseModel):
    app: AppConfig = AppConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    llm: LLMConfig = LLMConfig()
    prompts: PromptConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_path = Path(os.getenv("RAG_CONFIG_PATH", "config.yaml"))
    raw = _read_yaml(config_path)

    # Env overrides for deployment convenience.
    raw.setdefault("app", {})["data_dir"] = os.getenv("RAG_DATA_DIR", raw.get("app", {}).get("data_dir", "data/indexes/default"))
    raw.setdefault("embedding", {})["device"] = os.getenv("RAG_DEVICE", raw.get("embedding", {}).get("device", "cpu"))
    raw.setdefault("reranker", {})["device"] = os.getenv("RAG_DEVICE", raw.get("reranker", {}).get("device", "cpu"))
    raw.setdefault("llm", {})["model"] = os.getenv("MISTRAL_MODEL", raw.get("llm", {}).get("model", "mistral-large-latest"))

    return Settings(**raw)
