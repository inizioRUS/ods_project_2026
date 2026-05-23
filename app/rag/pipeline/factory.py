from __future__ import annotations

from functools import lru_cache

from core.config import get_settings
from pipeline.rag import RAGPipeline
from services.chunking.recursive import RecursiveTextChunker
from services.embeddings.e5 import E5Embedder
from services.fulltext.bm25_store import BM25Store
from services.generation.factory import build_generator
from services.rerankers.factory import build_reranker
from services.vectorstores.faiss_store import FaissVectorStore


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    data_dir = settings.app.data_dir
    return RAGPipeline(
        settings=settings,
        chunker=RecursiveTextChunker(settings.chunking),
        embedder=E5Embedder(settings.embedding),
        vector_store=FaissVectorStore(data_dir),
        bm25_store=BM25Store(data_dir),
        reranker=build_reranker(settings.reranker),
        generator=build_generator(settings.llm),
    )
