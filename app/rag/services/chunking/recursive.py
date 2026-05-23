from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Iterable

from core.config import ChunkingConfig
from models.schemas import ChunkConfigInput, ChunkInput, DocumentInput

_SPACE_RE = re.compile(r"\s+")


@dataclass
class RecursiveTextChunker:
    config: ChunkingConfig

    def chunk_documents(
        self,
        documents: Iterable[DocumentInput],
        override: ChunkConfigInput | None = None,
    ) -> list[ChunkInput]:
        chunks: list[ChunkInput] = []
        for doc_pos, doc in enumerate(documents):
            doc_id = doc.id or f"doc_{doc_pos}_{uuid.uuid4().hex[:8]}"
            texts = self.split_text(doc.text, override)
            for i, chunk_text in enumerate(texts):
                chunks.append(
                    ChunkInput(
                        id=f"{doc_id}::chunk_{i}",
                        doc_id=doc_id,
                        text=chunk_text,
                        metadata=doc.metadata.copy(),
                        chunk_index=i,
                    )
                )
        return chunks

    def split_text(self, text: str, override: ChunkConfigInput | None = None) -> list[str]:
        chunk_size = override.chunk_size if override and override.chunk_size else self.config.chunk_size
        chunk_overlap = override.chunk_overlap if override and override.chunk_overlap is not None else self.config.chunk_overlap
        min_chunk_size = override.min_chunk_size if override and override.min_chunk_size else self.config.min_chunk_size

        text = self._normalize(text)
        if len(text) <= chunk_size:
            return [text] if text else []

        pieces = self._recursive_split(text, chunk_size, self.config.separators)
        return self._merge_pieces(pieces, chunk_size, chunk_overlap, min_chunk_size)

    def _recursive_split(self, text: str, chunk_size: int, separators: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        if not separators:
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        sep = separators[0]
        parts = text.split(sep)
        if len(parts) == 1:
            return self._recursive_split(text, chunk_size, separators[1:])

        result: list[str] = []
        for i, part in enumerate(parts):
            if not part:
                continue
            # Restore separator except when it is whitespace-only.
            restored = part + (sep if i < len(parts) - 1 and sep.strip() else "")
            if len(restored) > chunk_size:
                result.extend(self._recursive_split(restored, chunk_size, separators[1:]))
            else:
                result.append(restored)
        return result

    def _merge_pieces(self, pieces: list[str], chunk_size: int, chunk_overlap: int, min_chunk_size: int) -> list[str]:
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if len(current.strip()) >= min_chunk_size:
                    chunks.append(current.strip())
                current = self._tail(current, chunk_overlap) + piece
                if len(current) > chunk_size * 1.2:
                    chunks.append(current[:chunk_size].strip())
                    current = self._tail(current, chunk_overlap)

        if current.strip() and len(current.strip()) >= min_chunk_size:
            chunks.append(current.strip())
        return chunks

    @staticmethod
    def _tail(text: str, n: int) -> str:
        if n <= 0:
            return ""
        return text[-n:]

    @staticmethod
    def _normalize(text: str) -> str:
        return _SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()
