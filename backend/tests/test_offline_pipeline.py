from uuid import uuid4

import pymupdf
import pytest
from qdrant_client import QdrantClient

from app.embeddings import EmbeddingService
from app.generation import AnswerGenerator, RetrievedContext
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf import extract_pdf_pages
from app.models import DocumentChunk
from app.reranking import Reranker
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.vectorstore import QdrantVectorStore


def _make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=16)
        return document.tobytes()
    finally:
        document.close()


@pytest.mark.parametrize("question", ["What was FY2025 revenue?", "Tell me the revenue amount."])
def test_offline_pipeline_returns_grounded_answer_and_page_citation(question: str) -> None:
    organization_id = uuid4()
    document_id = uuid4()
    extracted_pages = extract_pdf_pages(
        _make_pdf("FY2025 revenue was 1250 crore. Enterprise sales were stronger.")
    )
    chunks = chunk_pages(
        [{"page_number": page.page_number, "text": page.text} for page in extracted_pages],
        chunk_size=800,
        chunk_overlap=120,
    )
    records = [
        DocumentChunk(
            id=uuid4(),
            document_id=document_id,
            organization_id=organization_id,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            token_count=chunk.token_count,
        )
        for chunk in chunks
    ]

    embedding_service = EmbeddingService()
    vector_store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        embedding_service=embedding_service,
    )
    vector_store.upsert_chunks(records, "annual-report.pdf")
    vector_results = vector_store.search(
        embedding_service.embed_query(question),
        organization_id,
        limit=5,
    )
    keyword_results = BM25Index([(record, "annual-report.pdf") for record in records]).search(
        question,
        limit=5,
    )
    fused = reciprocal_rank_fusion(vector_results, keyword_results, limit=5, rrf_k=60)
    reranked = Reranker(provider="lexical").rerank(question, fused, limit=1)
    payload = reranked[0].payload
    answer = AnswerGenerator().generate(
        question,
        [
            RetrievedContext(
                chunk_id=str(payload["chunk_id"]),
                document_id=str(payload["document_id"]),
                filename=str(payload["filename"]),
                page_number=int(payload["page_number"]),
                content=str(payload["content"]),
                score=reranked[0].score,
            )
        ],
    )

    assert "1250 crore" in answer
    assert payload["filename"] == "annual-report.pdf"
    assert payload["page_number"] == 1
