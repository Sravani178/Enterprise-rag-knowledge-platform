from uuid import UUID

from sqlalchemy import ForeignKey, Index, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.models.document import Base


class BM25Document(Base):
    """Persistent BM25 statistics for one indexed document chunk."""

    __tablename__ = "bm25_documents"
    __table_args__ = (
        Index("ix_bm25_documents_organization", "organization_id"),
        Index("ix_bm25_documents_organization_document", "organization_id", "document_id"),
    )

    chunk_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    organization_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    document_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_length: Mapped[int] = mapped_column(Integer, nullable=False)


class BM25Posting(Base):
    """Term-frequency posting used by the persistent BM25 query path."""

    __tablename__ = "bm25_postings"
    __table_args__ = (
        Index("ix_bm25_postings_organization_term", "organization_id", "term"),
        Index("ix_bm25_postings_chunk", "chunk_id"),
    )

    organization_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True)
    term: Mapped[str] = mapped_column(String(255), primary_key=True)
    chunk_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("bm25_documents.chunk_id", ondelete="CASCADE"),
        primary_key=True,
    )
    term_frequency: Mapped[int] = mapped_column(Integer, nullable=False)
