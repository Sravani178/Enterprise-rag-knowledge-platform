from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy import Enum as SqlEnum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentStatus(StrEnum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_organization_created", "organization_id", "created_at"),
        UniqueConstraint(
            "organization_id",
            "idempotency_key",
            name="uq_documents_organization_idempotency",
        ),
        UniqueConstraint(
            "organization_id",
            "content_sha256",
            name="uq_documents_organization_content_hash",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    organization_id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), nullable=False)
    uploaded_by: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    extracted_text_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        SqlEnum(DocumentStatus, name="document_status", native_enum=False, length=20),
        nullable=False,
        default=DocumentStatus.UPLOADED,
    )
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
