from __future__ import annotations

from dataclasses import dataclass

from core.config import Settings
from models.schemas import (
    Citation,
    GenerateRequest,
    GenerateResponse,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from services.chunking.recursive import RecursiveTextChunker
from services.embeddings.e5 import E5Embedder
from services.fulltext.bm25_store import BM25Store
from services.fusion import CandidatePool, apply_rerank, fuse_candidates
from services.generation.base import BaseGenerator
from services.rerankers.base import BaseReranker
from services.vectorstores.faiss_store import FaissVectorStore


@dataclass
class RAGPipeline:
    settings: Settings
    chunker: RecursiveTextChunker
    embedder: E5Embedder
    vector_store: FaissVectorStore
    bm25_store: BM25Store
    reranker: BaseReranker
    generator: BaseGenerator

    def ingest(self, request: IngestRequest) -> IngestResponse:
        if request.reset:
            self.reset()

        if request.chunk:
            chunks = self.chunker.chunk_documents(request.documents, request.chunk_config)
        else:
            chunks = []
            for i, doc in enumerate(request.documents):
                doc_id = doc.id or f"doc_{i}"
                chunks.append(
                    self._document_as_chunk(doc_id=doc_id, text=doc.text, metadata=doc.metadata, chunk_index=0)
                )

        embeddings = self.embedder.encode([c.text for c in chunks], input_type="passage")
        self.vector_store.add(chunks, embeddings)
        self.bm25_store.add(chunks)

        return IngestResponse(
            documents_count=len(request.documents),
            chunks_count=len(chunks),
            index_size=self.vector_store.size,
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        retrieval_cfg = self.settings.retrieval
        semantic_top_k = request.semantic_top_k or retrieval_cfg.semantic_top_k
        bm25_top_k = request.bm25_top_k or retrieval_cfg.bm25_top_k
        final_top_k = request.final_top_k or retrieval_cfg.final_top_k
        fusion_method = request.fusion_method or retrieval_cfg.fusion_method

        query_embedding = self.embedder.encode([request.query], input_type="query")
        semantic_rows = self.vector_store.search(query_embedding[0], semantic_top_k)
        bm25_rows = self.bm25_store.search(request.query, bm25_top_k)

        pool = CandidatePool()
        pool.add("semantic", semantic_rows)
        pool.add("bm25", bm25_rows)

        fused = fuse_candidates(
            pool,
            method=fusion_method,
            semantic_weight=retrieval_cfg.semantic_weight,
            bm25_weight=retrieval_cfg.bm25_weight,
            rrf_k=retrieval_cfg.rrf_k,
        )

        # Rerank a wider candidate set, then crop.
        rerank_candidates = fused[: max(final_top_k * 3, final_top_k)]
        if request.rerank and rerank_candidates:
            chunks_for_rerank = [
                self._document_as_chunk(
                    doc_id=r.doc_id,
                    text=r.text,
                    metadata=r.metadata,
                    chunk_index=r.chunk_index,
                    chunk_id=r.id,
                )
                for r in rerank_candidates
            ]
            rerank_scores = self.reranker.score(request.query, chunks_for_rerank)
            rerank_candidates = apply_rerank(
                rerank_candidates,
                rerank_scores,
                rerank_weight=self.settings.reranker.weight,
            )

        return SearchResponse(query=request.query, results=rerank_candidates[:final_top_k])

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        search_response = self.search(request)
        context, citations = self._build_cited_context(search_response.results, request.max_context_chars)
        user_prompt = self.settings.prompts.user_template.format(query=request.query, context=context)
        answer = await self.generator.generate(self.settings.prompts.system, user_prompt)
        return GenerateResponse(query=request.query, answer=answer, citations=citations)

    def reset(self) -> None:
        self.vector_store.reset()
        self.bm25_store.reset()

    def _build_cited_context(self, results, max_chars: int) -> tuple[str, list[Citation]]:
        parts: list[str] = []
        citations: list[Citation] = []
        total = 0
        for i, result in enumerate(results, start=1):
            marker = f"C{i}"
            block = f"[{marker}] doc_id={result.doc_id}; chunk_id={result.id}; metadata={result.metadata}\n{result.text}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
            citations.append(
                Citation(
                    marker=f"[{marker}]",
                    chunk_id=result.id,
                    doc_id=result.doc_id,
                    text=result.text,
                    metadata=result.metadata,
                    scores=result.scores,
                )
            )
        return "\n\n".join(parts), citations

    @staticmethod
    def _document_as_chunk(doc_id: str, text: str, metadata: dict, chunk_index: int, chunk_id: str | None = None):
        from app.rag.models.schemas import ChunkInput

        return ChunkInput(
            id=chunk_id or f"{doc_id}::chunk_{chunk_index}",
            doc_id=doc_id,
            text=text,
            metadata=metadata,
            chunk_index=chunk_index,
        )
