from dataclasses import dataclass
from typing import Any

from app.models import DocumentChunk


@dataclass(frozen=True, slots=True)
class RankedChunk:
    chunk_id: str
    score: float
    payload: dict[str, Any]
    retrieval_score: float | None = None


def chunk_to_payload(chunk: DocumentChunk, filename: str) -> dict[str, Any]:
    return {
        "chunk_id": str(chunk.id),
        "document_id": str(chunk.document_id),
        "organization_id": str(chunk.organization_id),
        "filename": filename,
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "token_count": chunk.token_count,
        "content": chunk.content,
    }
