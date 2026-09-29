from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocumentStatus
from app.retrieval.types import RankedChunk


async def keep_completed_document_results(
    db: AsyncSession,
    results: list[RankedChunk],
    organization_id: UUID,
) -> list[RankedChunk]:
    """Hide stale Qdrant vectors for deleted, processing, or failed documents."""

    document_ids = {
        UUID(str(result.payload["document_id"]))
        for result in results
        if result.payload.get("document_id")
    }
    if not document_ids:
        return []
    result = await db.execute(
        select(Document.id).where(
            Document.id.in_(document_ids),
            Document.organization_id == organization_id,
            Document.status == DocumentStatus.COMPLETED,
        )
    )
    visible = {row[0] for row in result.all()}
    return [
        result
        for result in results
        if result.payload.get("document_id")
        and UUID(str(result.payload["document_id"])) in visible
    ]
