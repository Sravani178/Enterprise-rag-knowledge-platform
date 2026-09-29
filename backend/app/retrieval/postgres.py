from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentChunk
from app.retrieval.types import RankedChunk, chunk_to_payload


async def search_keyword_chunks(
    db: AsyncSession,
    query: str,
    organization_id: UUID,
    limit: int,
) -> list[RankedChunk]:
    """Search the persisted PostgreSQL tsvector instead of rebuilding BM25 per request."""

    query_text = func.websearch_to_tsquery("simple", query)
    score = func.ts_rank_cd(DocumentChunk.search_vector, query_text)
    statement = (
        select(DocumentChunk, Document.filename, score.label("keyword_score"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.organization_id == organization_id,
            Document.organization_id == organization_id,
            Document.status == "COMPLETED",
            DocumentChunk.search_vector.op("@@")(query_text),
        )
        .order_by(desc(score), DocumentChunk.chunk_index)
        .limit(limit)
    )
    result = await db.execute(statement)
    return [
        RankedChunk(
            chunk_id=str(chunk.id),
            score=float(keyword_score),
            payload=chunk_to_payload(chunk, filename),
        )
        for chunk, filename, keyword_score in result.all()
    ]
