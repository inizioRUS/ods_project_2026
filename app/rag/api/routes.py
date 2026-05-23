from __future__ import annotations

from fastapi import APIRouter, Depends

from models.schemas import (
    ChunkRequest,
    ChunkResponse,
    DocumentInput,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerateRequest,
    GenerateResponse,
    IngestRequest,
    IngestResponse,
    ResetResponse,
    SearchRequest,
    SearchResponse,
)
from pipeline.factory import get_pipeline
from pipeline.rag import RAGPipeline

router = APIRouter(prefix="/v1")


@router.get("/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    return {
        "ok": True,
        "index_size": pipeline.vector_store.size,
        "embedding_model": pipeline.embedder.model_name,
        "reranker": pipeline.settings.reranker.provider,
        "llm": pipeline.settings.llm.provider,
    }


@router.post("/chunk", response_model=ChunkResponse)
def chunk(request: ChunkRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> ChunkResponse:
    documents = []
    for i, text in enumerate(request.texts):
        doc_id = request.ids[i] if request.ids and i < len(request.ids) else f"text_{i}"
        metadata = request.metadata[i] if request.metadata and i < len(request.metadata) else {}
        documents.append(DocumentInput(id=doc_id, text=text, metadata=metadata))
    return ChunkResponse(chunks=pipeline.chunker.chunk_documents(documents, request.config))


@router.post("/embed", response_model=EmbeddingResponse)
def embed(request: EmbeddingRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> EmbeddingResponse:
    vectors = pipeline.embedder.encode(request.texts, input_type=request.input_type)
    return EmbeddingResponse(
        model=pipeline.embedder.model_name,
        dim=int(vectors.shape[1]),
        embeddings=vectors.tolist(),
    )


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> IngestResponse:
    return pipeline.ingest(request)


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> SearchResponse:
    return pipeline.search(request)


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> GenerateResponse:
    return await pipeline.generate(request)


@router.post("/index/reset", response_model=ResetResponse)
def reset_index(pipeline: RAGPipeline = Depends(get_pipeline)) -> ResetResponse:
    pipeline.reset()
    return ResetResponse(ok=True, index_size=pipeline.vector_store.size)
