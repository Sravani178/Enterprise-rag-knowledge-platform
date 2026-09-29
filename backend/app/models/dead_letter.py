from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.document import Base


class TaskDeadLetter(Base):
    """Durable record of a Celery task that needs operator attention."""

    __tablename__ = "task_dead_letters"
    __table_args__ = (
        Index("ix_task_dead_letters_status_created", "status", "created_at"),
        Index("ix_task_dead_letters_document", "document_id"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    task_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    task_name: Mapped[str] = mapped_column(String(255), nullable=False)
    document_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False, default=dict)
    exception_type: Mapped[str] = mapped_column(String(255), nullable=False)
    error_message: Mapped[str] = mapped_column(Text, nullable=False)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="PENDING")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
