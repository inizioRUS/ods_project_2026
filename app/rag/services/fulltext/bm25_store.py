from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from models.schemas import ChunkInput

_TOKEN_RE = re.compile(r"[\wа-яА-ЯёЁ]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self, persist_dir: str | Path):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.docs_path = self.persist_dir / "bm25_chunks.jsonl"

        self.chunks: list[ChunkInput] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

        self._load_if_exists()

    @property
    def size(self) -> int:
        return len(self.chunks)

    def reset(self) -> None:
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None

        if self.docs_path.exists():
            self.docs_path.unlink()

    def add(self, chunks: list[ChunkInput]) -> None:
        if not chunks:
            return

        self._append_to_disk(chunks)

        self.chunks.extend(chunks)
        self.tokenized_corpus.extend(tokenize(c.text) for c in chunks)
        self._rebuild()

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []

        tokenized = tokenize(query)
        scores = self.bm25.get_scores(tokenized)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:top_k]

        results = []
        for idx, score in ranked:
            score = float(score)
            if score <= 0:
                continue

            results.append({
                "chunk": self.chunks[idx],
                "score": score,
                "rank": len(results) + 1,
            })

        return results

    def persist(self) -> None:
        """
        Полная перезапись. Можно оставить для compact/rebuild сценариев,
        но не вызывать при обычном add().
        """
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        with self.docs_path.open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(chunk.model_dump_json() + "\n")

    def _append_to_disk(self, chunks: list[ChunkInput]) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        with self.docs_path.open("a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

    def _load_if_exists(self) -> None:
        if not self.docs_path.exists():
            return

        with self.docs_path.open("r", encoding="utf-8") as f:
            self.chunks = [
                ChunkInput(**json.loads(line))
                for line in f
                if line.strip()
            ]

        self.tokenized_corpus = [tokenize(c.text) for c in self.chunks]
        self._rebuild()

    def _rebuild(self) -> None:
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None