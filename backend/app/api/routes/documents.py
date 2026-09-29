import hashlib
from datetime import datetime
from pathlib import PurePath
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict
from redis.exceptions import RedisError
from sqlalchemy import delete, desc, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_principal, get_organization_id, require_roles
from app.api.rate_limit import enforce_rate_limit
from app.auth import Principal
from app.cache import QuestionCache
from app.core.config import get_settings
from app.db.session import get_db
from app.models import (
    BM25Document,
    BM25Posting,
    Document,
    DocumentChunk,
    DocumentStatus,
    MembershipRole,
)
from app.storage import S3Storage, StorageError
from app.vectorstore import VectorStoreError
from app.worker.tasks import _delete_external_document_resources, process_document

router = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    id: UUID
    organization_id: UUID
    filename: str
    mime_type: str
    file_size: int
    status: DocumentStatus
    page_count: int | None
    chunk_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class DocumentAcceptedResponse(DocumentResponse):
    task_id: str


@router.post("", response_model=DocumentAcceptedResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: AsyncSession = Depends(get_db),
    organization_id: UUID = Depends(get_organization_id),
    principal: Principal | None = Depends(get_current_principal),
    _role: Principal | None = Depends(
        require_roles(MembershipRole.OWNER, MembershipRole.ADMIN, MembershipRole.MEMBER)
    ),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> DocumentAcceptedResponse:
    raw_filename = (file.filename or "document.pdf").replace("\\", "/")
    filename = PurePath(raw_filename).name
    if PurePath(filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF files are supported")
    if file.content_type not in {None, "application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF files are supported")
    if idempotency_key is not None:
        idempotency_key = idempotency_key.strip()
        if not idempotency_key or len(idempotency_key) > 255:
            raise HTTPException(status_code=400, detail="Invalid Idempotency-Key")
        existing = await db.scalar(
            select(Document).where(
                Document.organization_id == organization_id,
                Document.idempotency_key == idempotency_key,
                Document.status != DocumentStatus.DELETED,
            )
        )
        if existing is not None:
            if existing.status == DocumentStatus.DELETING:
                raise HTTPException(status_code=409, detail="Document cleanup is in progress")
            return DocumentAcceptedResponse.model_validate(
                {**existing.__dict__, "task_id": existing.processing_task_id or "already-submitted"}
            )

    pdf_bytes = await file.read(settings.max_upload_size_bytes + 1)
    if len(pdf_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="The PDF exceeds the 25 MB upload limit")
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF")
    content_sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    existing_content = await db.scalar(
        select(Document).where(
            Document.organization_id == organization_id,
            Document.content_sha256 == content_sha256,
            Document.status != DocumentStatus.DELETED,
        )
    )
    if existing_content is not None:
        if existing_content.status == DocumentStatus.DELETING:
            raise HTTPException(status_code=409, detail="Document cleanup is in progress")
        return DocumentAcceptedResponse.model_validate(
            {
                **existing_content.__dict__,
                "task_id": existing_content.processing_task_id or "already-submitted",
            }
        )

    document_id = uuid4()
    storage_key = f"{organization_id}/original/{document_id}/{filename}"
    storage = S3Storage()

    try:
        await run_in_threadpool(storage.ensure_bucket)
        await run_in_threadpool(storage.put_bytes, storage_key, pdf_bytes, "application/pdf")
    except StorageError as exc:
        raise HTTPException(status_code=503, detail="Document storage is unavailable") from exc

    document = Document(
        id=document_id,
        organization_id=organization_id,
        filename=filename,
        storage_key=storage_key,
        mime_type="application/pdf",
        file_size=len(pdf_bytes),
        content_sha256=content_sha256,
        status=DocumentStatus.UPLOADED,
        idempotency_key=idempotency_key,
        uploaded_by=principal.user_id if principal is not None else None,
    )
    try:
        db.add(document)
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        try:
            await run_in_threadpool(storage.delete_object, storage_key)
        except StorageError:
            pass
        if idempotency_key is not None or content_sha256:
            existing = await db.scalar(
                    select(Document).where(
                        Document.organization_id == organization_id,
                        Document.status != DocumentStatus.DELETED,
                        (
                        (Document.idempotency_key == idempotency_key)
                        if idempotency_key is not None
                        else (Document.content_sha256 == content_sha256)
                    ),
                )
            )
            if existing is not None:
                return DocumentAcceptedResponse.model_validate(
                    {
                        **existing.__dict__,
                        "task_id": existing.processing_task_id or "already-submitted",
                    }
                )
        raise HTTPException(status_code=409, detail="Duplicate document request") from exc
    except SQLAlchemyError as exc:
        await db.rollback()
        try:
            await run_in_threadpool(storage.delete_object, storage_key)
        except StorageError:
            pass
        raise HTTPException(status_code=503, detail="Document metadata is unavailable") from exc
    await db.refresh(document)
    await _invalidate_question_cache(organization_id)

    try:
        task = process_document.delay(str(document.id))
    except Exception as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = "Unable to enqueue document processing"
        await db.commit()
        raise HTTPException(status_code=503, detail="Document processing is unavailable") from exc

    return DocumentAcceptedResponse.model_validate({**document.__dict__, "task_id": task.id})


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    organization_id: UUID = Depends(get_organization_id),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> list[Document]:
    result = await db.execute(
        select(Document)
        .where(
            Document.organization_id == organization_id,
            Document.status.not_in([DocumentStatus.DELETED, DocumentStatus.DELETING]),
        )
        .order_by(desc(Document.created_at))
    )
    return list(result.scalars().all())


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    organization_id: UUID = Depends(get_organization_id),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> Document:
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.organization_id == organization_id,
            Document.status.not_in([DocumentStatus.DELETED, DocumentStatus.DELETING]),
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.post("/{document_id}/reprocess", response_model=DocumentAcceptedResponse, status_code=202)
async def reprocess_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    organization_id: UUID = Depends(get_organization_id),
    _role: Principal | None = Depends(
        require_roles(MembershipRole.OWNER, MembershipRole.ADMIN, MembershipRole.MEMBER)
    ),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> DocumentAcceptedResponse:
    result = await db.execute(
        select(Document).where(
            Document.id == document_id,
            Document.organization_id == organization_id,
        )
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status in {DocumentStatus.DELETED, DocumentStatus.DELETING}:
        raise HTTPException(status_code=409, detail="Deleted documents cannot be reprocessed")

    document.status = DocumentStatus.UPLOADED
    document.error_message = None
    document.processing_task_id = None
    document.processing_started_at = None
    await db.commit()
    await db.refresh(document)
    await _invalidate_question_cache(organization_id)

    try:
        task = process_document.delay(str(document.id))
    except Exception as exc:
        document.status = DocumentStatus.FAILED
        document.error_message = "Unable to enqueue document processing"
        await db.commit()
        raise HTTPException(status_code=503, detail="Document processing is unavailable") from exc

    return DocumentAcceptedResponse.model_validate({**document.__dict__, "task_id": task.id})


@router.delete("/{document_id}", response_model=DocumentResponse)
async def delete_document(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
    organization_id: UUID = Depends(get_organization_id),
    _role: Principal | None = Depends(
        require_roles(MembershipRole.OWNER, MembershipRole.ADMIN)
    ),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> Document:
    result = await db.execute(
        select(Document)
        .where(
            Document.id == document_id,
            Document.organization_id == organization_id,
        )
        .with_for_update()
    )
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.status == DocumentStatus.DELETED:
        return document

    if document.status != DocumentStatus.DELETING:
        document.status = DocumentStatus.DELETING
        document.processing_task_id = None
        document.processing_started_at = None
        await db.commit()
        await db.refresh(document)
        await _invalidate_question_cache(organization_id)

    try:
        await run_in_threadpool(
            _delete_external_document_resources,
            document.id,
            document.organization_id,
            document.storage_key,
            document.extracted_text_key,
        )
    except (StorageError, VectorStoreError) as exc:
        raise HTTPException(status_code=503, detail="Document cleanup is unavailable") from exc

    bm25_chunk_ids = select(BM25Document.chunk_id).where(
        BM25Document.document_id == document.id
    )
    await db.execute(delete(BM25Posting).where(BM25Posting.chunk_id.in_(bm25_chunk_ids)))
    await db.execute(delete(BM25Document).where(BM25Document.document_id == document.id))
    await db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document.id,
            DocumentChunk.organization_id == document.organization_id,
        )
    )
    document.status = DocumentStatus.DELETED
    document.processing_task_id = None
    document.processing_started_at = None
    document.error_message = None
    # Release deduplication keys once the record is tombstoned. This allows a
    # later upload of the same content while retaining the deleted row for
    # audit/history.
    document.idempotency_key = None
    document.content_sha256 = None
    await db.commit()
    await db.refresh(document)
    await _invalidate_question_cache(organization_id)
    return document


async def _invalidate_question_cache(organization_id: UUID) -> None:
    if not settings.question_cache_enabled:
        return
    try:
        await run_in_threadpool(QuestionCache().bump_version, organization_id)
    except RedisError:
        # Cache invalidation is best-effort; database state remains authoritative.
        pass
