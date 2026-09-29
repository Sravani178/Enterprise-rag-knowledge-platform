from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from botocore.exceptions import EndpointConnectionError
from celery.exceptions import SoftTimeLimitExceeded

from app.core.config import Settings
from app.core.openai import post_with_retry
from app.core.redis import create_redis_client
from app.db.session import engine, sync_engine
from app.embeddings import EmbeddingError
from app.models import Document, DocumentChunk, DocumentStatus, TaskDeadLetter
from app.retrieval.bm25 import build_persistent_bm25_records
from app.storage import S3Storage, StorageError
from app.vectorstore import QdrantVectorStore, VectorStoreError
from app.worker.tasks import (
    _delete_external_document_resources,
    _record_dead_letter,
    _set_document_state,
    process_document,
    reconcile_documents,
)


def test_openai_retries_rate_limit_and_honors_retry_after() -> None:
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(200, json={"ok": True}),
    ]
    settings = Settings(openai_max_retries=1, openai_retry_backoff_seconds=0.001)

    with (
        patch("app.core.openai.get_settings", return_value=settings),
        patch("app.core.openai.httpx.post", side_effect=responses) as post,
        patch("app.core.openai.time.sleep") as sleep,
    ):
        response = post_with_retry(
            "https://api.openai.com/v1/test",
            headers={},
            payload={},
        )

    assert response.status_code == 200
    assert post.call_count == 2
    sleep.assert_called_once()


def test_database_engines_use_bounded_configured_pools() -> None:
    settings = Settings()

    assert engine.pool.size() == settings.db_pool_size
    assert engine.pool._max_overflow == settings.db_max_overflow
    assert engine.pool._timeout == settings.db_pool_timeout_seconds
    assert engine.pool._recycle == settings.db_pool_recycle_seconds
    assert sync_engine.pool.size() == settings.db_pool_size
    assert sync_engine.pool._max_overflow == settings.db_max_overflow


def test_redis_factory_uses_bounded_blocking_pool() -> None:
    settings = Settings(
        redis_max_connections=7,
        redis_pool_timeout_seconds=0.25,
    )
    fake_pool = MagicMock()

    with (
        patch("app.core.redis.get_settings", return_value=settings),
        patch(
            "app.core.redis.BlockingConnectionPool.from_url",
            return_value=fake_pool,
        ) as from_url,
    ):
        client = create_redis_client("redis://localhost:6379/2", decode_responses=True)

    assert client.connection_pool is fake_pool
    from_url.assert_called_once_with(
        "redis://localhost:6379/2",
        decode_responses=True,
        max_connections=7,
        timeout=0.25,
        socket_connect_timeout=settings.redis_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )


def test_settings_reject_unbounded_or_invalid_pool_limits() -> None:
    with pytest.raises(ValueError, match="DB_POOL_SIZE"):
        Settings(db_pool_size=0)
    with pytest.raises(ValueError, match="DB_MAX_OVERFLOW"):
        Settings(db_max_overflow=-1)
    with pytest.raises(ValueError, match="REDIS_MAX_CONNECTIONS"):
        Settings(redis_max_connections=0)


def test_celery_time_limits_and_retries_are_configured() -> None:
    from app.worker.celery_app import celery_app

    assert celery_app.conf.task_soft_time_limit == Settings().celery_soft_time_limit_seconds
    assert celery_app.conf.task_time_limit == Settings().celery_time_limit_seconds
    assert SoftTimeLimitExceeded in tuple(process_document.autoretry_for)
    assert process_document.max_retries == Settings().celery_max_retries


def test_settings_require_hard_celery_limit_above_soft_limit() -> None:
    with pytest.raises(ValueError, match="greater than"):
        Settings(celery_soft_time_limit_seconds=60, celery_time_limit_seconds=60)


def test_dead_letter_persistence_is_idempotent() -> None:
    existing = TaskDeadLetter(
        task_id="task-1",
        task_name="documents.process",
        exception_type="OldError",
        error_message="old",
        retry_count=1,
        status="PENDING",
        payload={},
    )
    first_result = MagicMock()
    first_result.scalar_one_or_none.return_value = None
    second_result = MagicMock()
    second_result.scalar_one_or_none.return_value = existing
    fake_session = MagicMock()
    fake_session.execute.side_effect = [first_result, second_result]
    fake_factory = MagicMock()
    fake_factory.return_value.__enter__.return_value = fake_session
    fake_factory.return_value.__exit__.return_value = False
    error = RuntimeError("worker permanently failed")

    with patch("app.worker.tasks.SyncSessionFactory", return_value=fake_factory()):
        _record_dead_letter("task-1", "documents.process", [str(uuid4())], {}, error, 3)
        _record_dead_letter("task-1", "documents.process", [str(uuid4())], {}, error, 3)

    fake_session.add.assert_called_once()
    assert existing.exception_type == "RuntimeError"
    assert existing.error_message == "worker permanently failed"
    assert existing.retry_count == 3
    assert existing.status == "PENDING"




def test_persistent_bm25_records_store_tenant_scoped_term_frequencies() -> None:
    organization_id = uuid4()
    chunk = DocumentChunk(
        id=uuid4(),
        document_id=uuid4(),
        organization_id=organization_id,
        chunk_index=0,
        content="Revenue revenue increased.",
        page_number=1,
        start_char=0,
        end_char=26,
        token_count=3,
    )

    documents, postings = build_persistent_bm25_records([chunk])

    assert documents[0].organization_id == organization_id
    assert documents[0].document_length == 3
    revenue = next(posting for posting in postings if posting.term == "revenue")
    assert revenue.organization_id == organization_id
    assert revenue.chunk_id == chunk.id
    assert revenue.term_frequency == 2


def test_qdrant_outage_is_translated_to_vector_store_error() -> None:
    class FailingClient:
        def collection_exists(self, **kwargs: object) -> bool:
            return True

        def search(self, **kwargs: object) -> None:
            raise RuntimeError("qdrant unavailable")

    class FakeEmbeddingService:
        dimension = 2

    with pytest.raises(VectorStoreError, match="Unable to search Qdrant"):
        QdrantVectorStore(
            client=FailingClient(),
            embedding_service=FakeEmbeddingService(),
        ).search([1.0, 0.0], uuid4(), 5)


def test_object_storage_outage_is_translated_to_storage_error() -> None:
    failing_client = MagicMock()
    failing_client.list_buckets.side_effect = EndpointConnectionError(
        endpoint_url="http://minio:9000"
    )
    settings = Settings(
        storage_connect_timeout_seconds=0.1,
        storage_read_timeout_seconds=0.2,
    )

    with (
        patch("app.storage.s3.get_settings", return_value=settings),
        patch("app.storage.s3.boto3.client", return_value=failing_client),
    ):
        storage = S3Storage()
        with pytest.raises(StorageError, match="Unable to reach object storage"):
            storage.check_connection()


def test_retryable_worker_failures_are_configured_for_celery_retry() -> None:
    autoretry_for = tuple(process_document.autoretry_for)

    assert EmbeddingError in autoretry_for
    assert StorageError in autoretry_for
    assert VectorStoreError in autoretry_for


def test_retryable_worker_state_releases_processing_lease() -> None:
    document = Document(
        id=uuid4(),
        organization_id=uuid4(),
        filename="report.pdf",
        storage_key="original/report.pdf",
        mime_type="application/pdf",
        file_size=10,
        status=DocumentStatus.PROCESSING,
        processing_task_id="task-1",
        processing_started_at=datetime.now(UTC),
    )
    fake_session = MagicMock()
    fake_session.execute.return_value.scalar_one_or_none.return_value = document
    fake_factory = MagicMock()
    fake_factory.return_value.__enter__.return_value = fake_session
    fake_factory.return_value.__exit__.return_value = False

    with patch("app.worker.tasks.SyncSessionFactory", fake_factory):
        updated = _set_document_state(
            document.id,
            DocumentStatus.UPLOADED,
            error_message="Retry scheduled: OpenAI embeddings are unavailable",
        )

    assert updated is True
    assert document.processing_task_id is None
    assert document.processing_started_at is None
    assert document.error_message.startswith("Retry scheduled:")


def test_external_document_cleanup_removes_vectors_and_both_objects() -> None:
    vector_store = MagicMock()
    storage = MagicMock()

    with (
        patch("app.worker.tasks.QdrantVectorStore", return_value=vector_store),
        patch("app.worker.tasks.S3Storage", return_value=storage),
    ):
        _delete_external_document_resources(
            uuid4(),
            uuid4(),
            "tenant/original/report.pdf",
            "tenant/extracted/document.json",
        )

    vector_store.delete_document.assert_called_once()
    assert storage.delete_object.call_count == 2


def test_external_document_cleanup_skips_missing_extracted_object_key() -> None:
    vector_store = MagicMock()
    storage = MagicMock()

    with (
        patch("app.worker.tasks.QdrantVectorStore", return_value=vector_store),
        patch("app.worker.tasks.S3Storage", return_value=storage),
    ):
        _delete_external_document_resources(
            uuid4(),
            uuid4(),
            "tenant/original/report.pdf",
            None,
        )

    storage.delete_object.assert_called_once_with("tenant/original/report.pdf")


def test_reconciliation_finalizes_deleting_document_after_external_cleanup() -> None:
    document = Document(
        id=uuid4(),
        organization_id=uuid4(),
        filename="report.pdf",
        storage_key="tenant/original/report.pdf",
        extracted_text_key="tenant/extracted/report.json",
        mime_type="application/pdf",
        file_size=10,
        status=DocumentStatus.DELETING,
    )
    initial_result = MagicMock()
    initial_result.scalars.return_value.all.return_value = [document]
    locked_result = MagicMock()
    locked_result.scalar_one_or_none.return_value = document
    fake_session = MagicMock()
    fake_session.execute.side_effect = [
        initial_result,
        locked_result,
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]
    fake_factory = MagicMock()
    fake_factory.return_value.__enter__.return_value = fake_session
    fake_factory.return_value.__exit__.return_value = False
    vector_store = MagicMock()
    vector_store.list_document_references.return_value = set()
    storage = MagicMock()
    storage.list_objects.return_value = []

    with (
        patch("app.worker.tasks.SyncSessionFactory", return_value=fake_factory()),
        patch("app.worker.tasks.QdrantVectorStore", return_value=vector_store),
        patch("app.worker.tasks.S3Storage", return_value=storage),
        patch("app.worker.tasks.get_settings", return_value=Settings()),
    ):
        result = reconcile_documents.run()

    assert result["finalized_deletions"] == 1
    assert document.status == DocumentStatus.DELETED
    assert document.content_sha256 is None
    assert vector_store.delete_document.called
    assert storage.delete_object.call_count == 2
