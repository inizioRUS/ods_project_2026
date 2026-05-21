from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from models.schemas import ChunkInput


class FaissVectorStore:
    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "index.faiss"
        self.docs_path = self.persist_dir / "chunks.jsonl"
        self.index: faiss.Index | None = None
        self.chunks: list[ChunkInput] = []
        self._load_if_exists()

    def reset(self) -> None:
        self.index = None
        self.chunks = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.docs_path.exists():
            self.docs_path.unlink()

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[ChunkInput], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings must have shape [len(chunks), dim]")

        vectors = np.asarray(embeddings, dtype="float32")
        dim = vectors.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        if self.index.d != dim:
            raise ValueError(f"FAISS dim mismatch: index={self.index.d}, new={dim}")

        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.persist()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[dict[str, Any]]:
        if self.index is None or self.size == 0:
            return []
        q = np.asarray(query_embedding, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        scores, indices = self.index.search(q, min(top_k, self.size))
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append({"chunk": chunk, "score": float(score), "rank": len(results) + 1})
        return results

    def persist(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with self.docs_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + "\n")

    def _load_if_exists(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.docs_path.exists():
            with self.docs_path.open("r", encoding="utf-8") as f:
                self.chunks = [ChunkInput(**json.loads(line)) for line in f if line.strip()]
