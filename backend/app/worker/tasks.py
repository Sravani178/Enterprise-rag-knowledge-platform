from datetime import UTC, datetime, timedelta
from threading import Event, Thread
from uuid import UUID, uuid4

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from redis.exceptions import RedisError
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from app.cache import QuestionCache
from app.core.config import get_settings
from app.core.observability import logger, metrics
from app.db.session import SyncSessionFactory
from app.embeddings import EmbeddingError, EmbeddingService
from app.ingestion.chunker import chunk_pages
from app.ingestion.pdf import PdfExtractionError, extract_pdf_pages
from app.models import Document, DocumentChunk, DocumentStatus, TaskDeadLetter
from app.retrieval.bm25 import (
    build_persistent_bm25_records,
    delete_persistent_bm25_for_document,
)
from app.storage import S3Storage, StorageError
from app.vectorstore import QdrantVectorStore, VectorStoreError
from app.worker.celery_app import celery_app


def _record_dead_letter(
    task_id: str,
    task_name: str,
    args: tuple[object, ...] | list[object],
    kwargs: dict[str, object],
    exc: BaseException,
    retry_count: int,
) -> None:
    """Persist terminal task failures without masking the original failure."""

    if not task_id:
        return
    document_id: UUID | None = None
    if args:
        try:
            document_id = UUID(str(args[0]))
        except (AttributeError, ValueError):
            document_id = None
    payload = {
        "args": [str(value) for value in args],
        "kwargs": {str(key): str(value) for key, value in kwargs.items()},
    }
    try:
        with SyncSessionFactory() as session:
            dead_letter = session.execute(
                select(TaskDeadLetter).where(TaskDeadLetter.task_id == task_id)
            ).scalar_one_or_none()
            if dead_letter is None:
                dead_letter = TaskDeadLetter(
                    task_id=task_id,
                    task_name=task_name,
                    document_id=document_id,
                    payload=payload,
                    exception_type=type(exc).__name__,
                    error_message=str(exc)[:4000],
                    retry_count=retry_count,
                    status="PENDING",
                )
                session.add(dead_letter)
            else:
                dead_letter.error_message = str(exc)[:4000]
                dead_letter.exception_type = type(exc).__name__
                dead_letter.retry_count = retry_count
                dead_letter.status = "PENDING"
            session.commit()
        metrics.increment(
            "celery_dead_letters_total",
            labels={"task": task_name},
        )
        logger.error(
            "Celery task moved to dead letter storage",
            extra={
                "event": "celery_dead_letter",
                "fields": {
                    "task_id": task_id,
                    "task": task_name,
                    "retry_count": retry_count,
                    "exception_type": type(exc).__name__,
                },
            },
        )
    except SQLAlchemyError:
        # A database outage must not hide the task's actual failure. The
        # normal task retry/lease recovery remains the fallback mechanism.
        return


class DocumentTask(Task):
    """Celery base task that records terminal failures in the durable DLQ."""

    def on_failure(
        self,
        exc: BaseException,
        task_id: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        einfo: object,
    ) -> None:
        retry_count = int(getattr(self.request, "retries", 0))
        max_retries = int(getattr(self, "max_retries", get_settings().celery_max_retries))
        retryable = isinstance(
            exc,
            (
                SQLAlchemyError,
                StorageError,
                VectorStoreError,
                EmbeddingError,
                SoftTimeLimitExceeded,
            ),
        )
        if not retryable or retry_count >= max_retries or isinstance(exc, SoftTimeLimitExceeded):
            _record_dead_letter(task_id, self.name, args, kwargs, exc, retry_count)
        super().on_failure(exc, task_id, args, kwargs, einfo)


def _claim_document(
    document_id: UUID,
    task_id: str,
    *,
    redelivered: bool = False,
) -> tuple[str, UUID, str] | str | None:
    """Claim a document with a row lock and a stale-processing lease."""

    settings = get_settings()
    now = datetime.now(UTC)
    with SyncSessionFactory() as session:
        document = session.execute(
            select(Document).where(Document.id == document_id).with_for_update()
        ).scalar_one_or_none()
        if document is None:
            return None
        if document.status in {
            DocumentStatus.COMPLETED,
            DocumentStatus.DELETING,
            DocumentStatus.DELETED,
        }:
            return document.status.value

        if document.status == DocumentStatus.PROCESSING:
            started_at = document.processing_started_at
            if started_at is not None and started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            is_stale = (
                started_at is None
                or (now - started_at).total_seconds() >= settings.processing_stale_after_seconds
            )
            # A late-ack task can be delivered again after its worker dies.
            # Celery marks that delivery as redelivered; allow the same task
            # id to reclaim its lease in that case. A normal duplicate delivery
            # still remains blocked while the original worker is active.
            same_redelivered_task = (
                redelivered
                and document.processing_task_id == task_id
                and is_stale
            )
            if not is_stale and not same_redelivered_task:
                return "ALREADY_PROCESSING"

        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        document.processing_task_id = task_id
        document.processing_started_at = now
        session.commit()
        return document.storage_key, document.organization_id, document.filename


def _set_document_state(
    document_id: UUID,
    status: DocumentStatus,
    *,
    page_count: int | None = None,
    chunk_count: int | None = None,
    extracted_text_key: str | None = None,
    error_message: str | None = None,
    expected_status: DocumentStatus | None = DocumentStatus.PROCESSING,
) -> bool:
    with SyncSessionFactory() as session:
        statement = select(Document).where(Document.id == document_id).with_for_update()
        document = session.execute(statement).scalar_one_or_none()
        if document is None:
            return False
        if expected_status is not None and document.status != expected_status:
            return False
        document.status = status
        document.error_message = error_message
        if page_count is not None:
            document.page_count = page_count
        if chunk_count is not None:
            document.chunk_count = chunk_count
        if extracted_text_key is not None:
            document.extracted_text_key = extracted_text_key
        # A retryable task failure must release the processing lease. Celery
        # will redeliver the same task and _claim_document must be able to
        # claim the document again. Terminal states also release the lease.
        if status in {
            DocumentStatus.UPLOADED,
            DocumentStatus.COMPLETED,
            DocumentStatus.FAILED,
        }:
            document.processing_task_id = None
            document.processing_started_at = None
        session.commit()
        return True


def _renew_processing_lease(document_id: UUID, task_id: str) -> None:
    """Move the lease deadline forward while the worker is still alive."""

    with SyncSessionFactory() as session:
        document = session.execute(
            select(Document).where(
                Document.id == document_id,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_task_id == task_id,
            )
        ).scalar_one_or_none()
        if document is None:
            return
        document.processing_started_at = datetime.now(UTC)
        session.commit()


def _start_lease_heartbeat(document_id: UUID, task_id: str) -> tuple[Event, Thread]:
    interval = max(1.0, min(get_settings().processing_stale_after_seconds / 3, 60.0))
    stop = Event()

    def heartbeat() -> None:
        while not stop.wait(interval):
            try:
                _renew_processing_lease(document_id, task_id)
            except SQLAlchemyError:
                # The task's normal database operation will retry/fail. A
                # temporary heartbeat failure must not kill the worker thread.
                continue

    thread = Thread(target=heartbeat, name=f"document-lease-{document_id}", daemon=True)
    thread.start()
    return stop, thread


class DocumentProcessingCancelled(RuntimeError):
    """Raised when deletion wins the race with document indexing."""


def _delete_external_document_resources(
    document_id: UUID,
    organization_id: UUID,
    storage_key: str,
    extracted_text_key: str | None,
) -> None:
    """Delete Qdrant and object-storage state in an idempotent order."""

    QdrantVectorStore().delete_document(document_id, organization_id)
    storage = S3Storage()
    storage.delete_object(storage_key)
    if extracted_text_key:
        storage.delete_object(extracted_text_key)


def _index_chunks(
    document_id: UUID,
    organization_id: UUID,
    filename: str,
    pages: list[dict[str, object]],
) -> int:
    settings = get_settings()
    embedding_service = EmbeddingService()
    text_chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_provider=(
            embedding_service if settings.semantic_chunking_enabled else None
        ),
        semantic_similarity_threshold=settings.semantic_chunk_similarity_threshold,
        semantic_min_size=settings.semantic_chunk_min_size,
    )
    chunk_records = [
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
        for chunk in text_chunks
    ]

    vector_store = QdrantVectorStore(embedding_service=embedding_service)
    vector_store.delete_document(document_id, organization_id)
    vector_store.upsert_chunks(chunk_records, filename)
    bm25_documents, bm25_postings = build_persistent_bm25_records(chunk_records)

    # Hold the document row lock while replacing SQL chunks. Deletion also
    # takes this lock before cleanup, so it cannot delete the SQL rows and
    # then have a worker insert them after the cleanup completed.
    with SyncSessionFactory() as session:
        document = session.execute(
            select(Document).where(Document.id == document_id).with_for_update()
        ).scalar_one_or_none()
        if document is None or document.status != DocumentStatus.PROCESSING:
            vector_store.delete_document(document_id, organization_id)
            raise DocumentProcessingCancelled
        delete_persistent_bm25_for_document(session, document_id)
        session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.organization_id == organization_id,
            )
        )
        if chunk_records:
            session.add_all(chunk_records)
            session.add_all(bm25_documents)
            session.add_all(bm25_postings)
        session.commit()

    return len(chunk_records)


@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="documents.process",
    autoretry_for=(
        SQLAlchemyError,
        StorageError,
        VectorStoreError,
        EmbeddingError,
        SoftTimeLimitExceeded,
    ),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=get_settings().celery_max_retries,
)
def process_document(self: Task, document_id: str) -> dict[str, str | int]:
    """Extract page text into object storage and finalize document status."""

    metrics.increment("document_processing_started_total")
    document_uuid = UUID(document_id)
    task_id = str(self.request.id or "")
    delivery_info = getattr(self.request, "delivery_info", {}) or {}
    claim = _claim_document(
        document_uuid,
        task_id,
        redelivered=bool(delivery_info.get("redelivered", False)),
    )
    if claim is None:
        metrics.increment("document_processing_total", labels={"outcome": "missing"})
        return {"document_id": document_id, "status": "MISSING"}
    if isinstance(claim, str):
        metrics.increment("document_processing_total", labels={"outcome": claim.lower()})
        return {"document_id": document_id, "status": claim}
    storage_key, organization_id, filename = claim
    heartbeat_stop, heartbeat_thread = _start_lease_heartbeat(document_uuid, task_id)

    try:
        storage = S3Storage()
        storage.ensure_bucket()
        pdf_bytes = storage.get_bytes(storage_key)
        pages = extract_pdf_pages(pdf_bytes)
        extracted_key = f"{organization_id}/extracted/{document_uuid}.json"
        storage.put_json(
            extracted_key,
            [{"page_number": page.page_number, "text": page.text} for page in pages],
        )
        chunk_count = _index_chunks(
            document_uuid,
            organization_id,
            filename,
            [{"page_number": page.page_number, "text": page.text} for page in pages],
        )
    except PdfExtractionError as exc:
        message = str(exc)
        _set_document_state(document_uuid, DocumentStatus.FAILED, error_message=message)
        _record_dead_letter(
            task_id,
            self.name,
            [document_id],
            {},
            exc,
            int(getattr(self.request, "retries", 0)),
        )
        metrics.increment("document_processing_total", labels={"outcome": "failed"})
        return {"document_id": document_id, "status": "FAILED"}
    except DocumentProcessingCancelled:
        metrics.increment("document_processing_total", labels={"outcome": "cancelled"})
        return {"document_id": document_id, "status": "CANCELLED"}
    except Exception as exc:
        retryable = isinstance(
            exc,
            (
                SQLAlchemyError,
                StorageError,
                VectorStoreError,
                EmbeddingError,
                SoftTimeLimitExceeded,
            ),
        )
        retry_count = int(getattr(self.request, "retries", 0))
        max_retries = int(getattr(self, "max_retries", 3))
        try:
            _set_document_state(
                document_uuid,
                DocumentStatus.UPLOADED if retryable and retry_count < max_retries
                else DocumentStatus.FAILED,
                error_message=(
                    f"Retry scheduled: {str(exc)[:1900]}"
                    if retryable and retry_count < max_retries
                    else str(exc)[:2000]
                ),
            )
        except SQLAlchemyError:
            # Preserve the original failure; the lease timeout can reclaim it.
            pass
        metrics.increment(
            "document_processing_total",
            labels={
                "outcome": "retry" if retryable and retry_count < max_retries else "failed"
            },
        )
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)

    completed = _set_document_state(
        document_uuid,
        DocumentStatus.COMPLETED,
        page_count=len(pages),
        chunk_count=chunk_count,
        extracted_text_key=extracted_key,
    )
    if not completed:
        # Deletion may have claimed the row after indexing completed. Its
        # cleanup path removes the vectors and object data; do not resurrect
        # the document or populate the cache.
        metrics.increment("document_processing_total", labels={"outcome": "cancelled"})
        return {"document_id": document_id, "status": "CANCELLED"}
    if get_settings().question_cache_enabled:
        try:
            QuestionCache().bump_version(organization_id)
        except RedisError:
            pass
    metrics.increment("document_processing_total", labels={"outcome": "completed"})
    return {
        "document_id": document_id,
        "status": "COMPLETED",
        "page_count": len(pages),
        "chunk_count": chunk_count,
    }


@celery_app.task(
    bind=True,
    base=DocumentTask,
    name="documents.reconcile",
    autoretry_for=(SQLAlchemyError, StorageError, VectorStoreError),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=get_settings().celery_max_retries,
)
def reconcile_documents(self: Task) -> dict[str, int]:
    """Repair drift between PostgreSQL metadata and Qdrant vectors.

    The normal indexing order prevents a document from becoming COMPLETED
    before vectors exist. This periodic task handles failures that happen
    after completion (for example, a lost Qdrant volume) and removes vectors
    for deleted/unknown documents. It is intentionally idempotent and safe to
    run concurrently with document processing.
    """

    vector_store = QdrantVectorStore()
    references = vector_store.list_document_references()
    repaired_orphans = 0
    requeued = 0

    with SyncSessionFactory() as session:
        documents = list(session.execute(select(Document)).scalars().all())
        known = {(document.organization_id, document.id): document for document in documents}

    storage = S3Storage()
    known_storage_keys = {
        key
        for document in documents
        if document.status != DocumentStatus.DELETED
        for key in (document.storage_key, document.extracted_text_key)
        if key
    }
    storage_cutoff = datetime.now(UTC) - timedelta(
        seconds=get_settings().storage_orphan_grace_seconds
    )
    repaired_storage = 0
    for key, modified in storage.list_objects():
        modified_at = modified if modified.tzinfo is not None else modified.replace(tzinfo=UTC)
        if key in known_storage_keys or modified_at > storage_cutoff:
            continue
        storage.delete_object(key)
        repaired_storage += 1

    for organization_id, document_id in references:
        document = known.get((organization_id, document_id))
        if document is None or document.status != DocumentStatus.COMPLETED:
            vector_store.delete_document(document_id, organization_id)
            repaired_orphans += 1

    finalized_deletions = 0
    for document in documents:
        if document.status != DocumentStatus.DELETING:
            continue

        _delete_external_document_resources(
            document.id,
            document.organization_id,
            document.storage_key,
            document.extracted_text_key,
        )
        with SyncSessionFactory() as session:
            current = session.execute(
                select(Document).where(Document.id == document.id).with_for_update()
            ).scalar_one_or_none()
            if current is None or current.status != DocumentStatus.DELETING:
                continue
            delete_persistent_bm25_for_document(session, current.id)
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == current.id,
                    DocumentChunk.organization_id == current.organization_id,
                )
            )
            current.status = DocumentStatus.DELETED
            current.processing_task_id = None
            current.processing_started_at = None
            current.error_message = None
            current.idempotency_key = None
            current.content_sha256 = None
            session.commit()
            finalized_deletions += 1

    now = datetime.now(UTC)
    processing_timeout = get_settings().processing_stale_after_seconds
    for document in documents:
        if document.status == DocumentStatus.PROCESSING:
            started_at = document.processing_started_at
            if started_at is not None and started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            if (
                started_at is not None
                and (now - started_at).total_seconds() < processing_timeout
            ):
                continue
        if document.status == DocumentStatus.COMPLETED:
            vector_count = vector_store.count_document(document.id, document.organization_id)
            if vector_count == document.chunk_count:
                continue
        elif document.status not in {
            DocumentStatus.PROCESSING,
            DocumentStatus.UPLOADED,
        }:
            continue

        with SyncSessionFactory() as session:
            current = session.execute(
                select(Document).where(Document.id == document.id).with_for_update()
            ).scalar_one_or_none()
            if current is None or current.status not in {
                DocumentStatus.COMPLETED,
                DocumentStatus.PROCESSING,
                DocumentStatus.UPLOADED,
            }:
                continue
            if current.status == DocumentStatus.PROCESSING:
                current_started_at = current.processing_started_at
                if current_started_at is not None and current_started_at.tzinfo is None:
                    current_started_at = current_started_at.replace(tzinfo=UTC)
                if (
                    current_started_at is not None
                    and (datetime.now(UTC) - current_started_at).total_seconds()
                    < processing_timeout
                ):
                    continue
            current.status = DocumentStatus.UPLOADED
            current.error_message = "Requeued after indexing consistency check"
            current.processing_task_id = None
            current.processing_started_at = None
            session.commit()

        process_document.delay(str(document.id))
        requeued += 1

    return {
        "repaired_orphans": repaired_orphans,
        "repaired_storage": repaired_storage,
        "finalized_deletions": finalized_deletions,
        "requeued": requeued,
    }
