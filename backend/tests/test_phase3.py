import asyncio
import sys
import types
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

import app.api.concurrency as query_concurrency
from app.api.routes.query import (
    QueryRequest,
    _execute_uncached_query,
    _generate_and_evaluate_answer,
)
from app.auth import Principal, create_access_token, hash_password, verify_password
from app.cache import QuestionCache, cosine_similarity
from app.core.config import Settings
from app.db.session import get_db
from app.embeddings import EmbeddingError, EmbeddingService
from app.generation import (
    AnswerEvaluation,
    AnswerEvaluator,
    AnswerGenerator,
    GenerationError,
    RetrievedContext,
)
from app.ingestion.chunker import chunk_pages
from app.main import app
from app.models import Document, DocumentChunk, DocumentStatus, MembershipRole
from app.query import QueryAnalysis, QueryUnderstandingService
from app.reranking import Reranker
from app.retrieval.bm25 import BM25Index, search_bm25_chunks
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.types import RankedChunk
from app.vectorstore import QdrantVectorStore, VectorSearchResult
from app.worker.celery_app import celery_app
from app.worker.tasks import _claim_document


def test_chunker_preserves_page_and_overlap_metadata() -> None:
    chunks = chunk_pages(
        [{"page_number": 7, "text": "Revenue grew steadily. " * 80}],
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert all(chunk.start_char < chunk.end_char for chunk in chunks)
    assert chunks[0].chunk_index == 0


def test_hybrid_chunker_creates_boundary_when_adjacent_topics_diverge() -> None:
    class FakeEmbeddingProvider:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]][: len(texts)]

    chunks = chunk_pages(
        [
            {
                "page_number": 1,
                "text": (
                    "Revenue increased in FY2025. Enterprise sales drove the growth. "
                    "The office moved to a new building."
                ),
            }
        ],
        chunk_size=200,
        chunk_overlap=0,
        embedding_provider=FakeEmbeddingProvider(),
        semantic_similarity_threshold=0.8,
        semantic_min_size=10,
    )

    assert len(chunks) == 2
    assert "Enterprise sales" in chunks[0].content
    assert "office moved" in chunks[1].content


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sorted_sets: dict[str, dict[str, float]] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value

    def incr(self, key: str) -> int:
        value = int(self.values.get(key, "0")) + 1
        self.values[key] = str(value)
        return value

    def zadd(self, key: str, values: dict[str, float]) -> None:
        self.sorted_sets.setdefault(key, {}).update(values)

    def zrevrange(self, key: str, start: int, end: int) -> list[str]:
        ranked = sorted(
            self.sorted_sets.get(key, {}).items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [member for member, _ in ranked[start : end + 1]]

    def zincrby(self, key: str, amount: float, member: str) -> float:
        self.sorted_sets.setdefault(key, {})[member] = (
            self.sorted_sets.setdefault(key, {}).get(member, 0.0) + amount
        )
        return self.sorted_sets[key][member]

    def zrem(self, key: str, member: str) -> None:
        self.sorted_sets.get(key, {}).pop(member, None)


def test_semantic_question_cache_matches_and_invalidates_tenant_answers() -> None:
    organization_id = uuid4()
    response = {"answer": "Revenue was 1250 crore."}
    cache = QuestionCache(client=_FakeRedis())
    cache.put(organization_id, "What was revenue?", [1.0, 0.0], response)

    exact = cache.get_exact(organization_id, "  WHAT was revenue? ")
    semantic = cache.get_semantic(organization_id, [0.99, 0.1])
    assert exact is not None and exact.response == response
    assert semantic is not None and semantic.response == response
    assert cosine_similarity([1.0, 0.0], [0.99, 0.1]) > 0.94

    cache.bump_version(organization_id)
    assert cache.get_exact(organization_id, "What was revenue?") is None


def test_qdrant_search_filters_by_organization() -> None:
    organization_a = uuid4()
    organization_b = uuid4()
    document_a = uuid4()
    document_b = uuid4()
    chunks = [
        DocumentChunk(
            id=uuid4(),
            document_id=document_a,
            organization_id=organization_a,
            chunk_index=0,
            content="FY2025 revenue was 1250 crore.",
            page_number=42,
            start_char=0,
            end_char=32,
            token_count=5,
        ),
        DocumentChunk(
            id=uuid4(),
            document_id=document_b,
            organization_id=organization_b,
            chunk_index=0,
            content="FY2025 revenue was 1250 crore.",
            page_number=42,
            start_char=0,
            end_char=32,
            token_count=5,
        ),
    ]
    store = QdrantVectorStore(client=QdrantClient(location=":memory:"))
    store.upsert_chunks(chunks, "annual_report.pdf")

    results = store.search(
        EmbeddingService().embed_query("What was FY2025 revenue?"),
        organization_a,
        limit=5,
    )

    assert len(results) == 1
    assert results[0].payload["organization_id"] == str(organization_a)
    assert store.count_document(document_a, organization_a) == 1
    assert (organization_a, document_a) in store.list_document_references()


def test_hashing_embeddings_are_deterministic_and_normalized() -> None:
    service = EmbeddingService()
    first = service.embed_query("What was FY2025 revenue?")
    second = service.embed_query("What was FY2025 revenue?")

    assert first == second
    assert len(first) == 384
    assert round(sum(value * value for value in first), 6) == 1.0


def test_openai_embeddings_are_requested_with_configured_model_and_dimension() -> None:
    settings = Settings(
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        embedding_dimension=3,
        openai_api_key="test-key",
    )
    response = MagicMock()
    response.json.return_value = {
        "data": [
            {"index": 1, "embedding": [0.2, 0.3, 0.4]},
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
        ]
    }

    with (
        patch("app.embeddings.service.get_settings", return_value=settings),
        patch("app.embeddings.service.post_with_retry", return_value=response) as post,
    ):
        embeddings = EmbeddingService().embed_texts(["first", "second"])

    assert embeddings == [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
    assert post.call_args.kwargs["payload"] == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
    }


def test_openai_embedding_failure_is_translated_to_domain_error() -> None:
    settings = Settings(
        embedding_provider="openai",
        embedding_dimension=3,
        openai_api_key="test-key",
    )
    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "unauthorized",
        request=MagicMock(),
        response=MagicMock(),
    )

    with (
        patch("app.embeddings.service.get_settings", return_value=settings),
        patch("app.embeddings.service.post_with_retry", return_value=response),
        pytest.raises(EmbeddingError, match="embeddings are unavailable"),
    ):
        EmbeddingService().embed_query("test")


def test_openai_generation_uses_chat_completion_and_translates_failures() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    context = RetrievedContext(
        chunk_id="chunk-1",
        document_id="document-1",
        filename="report.pdf",
        page_number=1,
        content="Revenue was 1250 crore.",
        score=0.9,
    )
    response = MagicMock()
    response.json.return_value = {
        "choices": [{"message": {"content": "Revenue was 1250 crore. [1]"}}]
    }

    with (
        patch("app.generation.basic.get_settings", return_value=settings),
        patch("app.generation.basic.post_with_retry", return_value=response) as post,
    ):
        answer = AnswerGenerator().generate("What was revenue?", [context])

    assert "1250 crore" in answer
    assert post.call_args.args[0].endswith("/chat/completions")

    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error",
        request=MagicMock(),
        response=MagicMock(),
    )
    with (
        patch("app.generation.basic.get_settings", return_value=settings),
        patch("app.generation.basic.post_with_retry", return_value=response),
        pytest.raises(GenerationError, match="generation is unavailable"),
    ):
        AnswerGenerator().generate("What was revenue?", [context])


def test_query_understanding_corrects_typo_and_preserves_original_query() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        query_understanding_enabled=True,
    )
    response = MagicMock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"corrected_query":"how to reset password", '
                        '"retrieval_query":"password reset instructions"}'
                    )
                }
            }
        ]
    }

    with (
        patch("app.query.understanding.get_settings", return_value=settings),
        patch("app.query.understanding.post_with_retry", return_value=response),
    ):
        analysis = QueryUnderstandingService().analyze("how to reset passwrod")

    assert analysis.original_query == "how to reset passwrod"
    assert analysis.corrected_query == "how to reset password"
    assert analysis.retrieval_query == "password reset instructions"


def test_query_understanding_rejects_rewrite_that_changes_product_code() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        query_understanding_enabled=True,
    )
    response = MagicMock()
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"corrected_query":"install API v2", '
                        '"retrieval_query":"API v2 installation"}'
                    )
                }
            }
        ]
    }

    with (
        patch("app.query.understanding.get_settings", return_value=settings),
        patch("app.query.understanding.post_with_retry", return_value=response),
    ):
        analysis = QueryUnderstandingService().analyze("install API-v2")

    assert analysis.corrected_query == "install API-v2"
    assert analysis.retrieval_query == "install API-v2"


def test_query_understanding_fails_open_to_original_query() -> None:
    settings = Settings(
        llm_provider="openai",
        openai_api_key="test-key",
        query_understanding_enabled=True,
    )

    with (
        patch("app.query.understanding.get_settings", return_value=settings),
        patch(
            "app.query.understanding.post_with_retry",
            side_effect=httpx.TimeoutException("down"),
        ),
    ):
        analysis = QueryUnderstandingService().analyze("how to reset passwrod")

    assert analysis.corrected_query == analysis.original_query
    assert analysis.retrieval_query == analysis.original_query


@pytest.mark.asyncio
async def test_rewritten_query_is_used_for_retrieval_but_original_is_used_for_answer() -> None:
    organization_id = uuid4()
    document_id = uuid4()
    candidate = RankedChunk(
        "chunk-1",
        0.4,
        {
            "chunk_id": "chunk-1",
            "document_id": str(document_id),
            "filename": "report.pdf",
            "page_number": 1,
            "content": "Password reset instructions.",
        },
    )
    retrieval_queries: list[str] = []
    answer_queries: list[str] = []

    class FakeReranker:
        def rerank(self, question: str, candidates: list[RankedChunk], limit: int):
            retrieval_queries.append(question)
            return [candidate]

    def fake_answer(question: str, contexts: list[RetrievedContext]):
        answer_queries.append(question)
        from app.generation import EvaluatedAnswer

        return EvaluatedAnswer("Password reset instructions. [1]", 1, 1.0, True)

    with (
        patch(
            "app.api.routes.query.QueryUnderstandingService.analyze",
            return_value=QueryAnalysis(
                "how to reset passwrod",
                "how to reset password",
                "password reset instructions",
            ),
        ),
        patch("app.api.routes.query.EmbeddingService.embed_query", return_value=[1.0]),
        patch("app.api.routes.query.QdrantVectorStore.search", return_value=[candidate]),
        patch(
            "app.api.routes.query.keep_completed_document_results",
            return_value=[candidate],
        ),
        patch(
            "app.api.routes.query.search_bm25_chunks",
            new=AsyncMock(return_value=[]),
        ) as bm25,
        patch("app.api.routes.query.get_reranker", return_value=FakeReranker()),
        patch("app.api.routes.query._generate_and_evaluate_answer", side_effect=fake_answer),
    ):
        response = await _execute_uncached_query(
            QueryRequest(query="how to reset passwrod", top_k=1),
            MagicMock(),
            organization_id,
            "how to reset passwrod",
            None,
            None,
            0.0,
        )

    assert retrieval_queries == ["password reset instructions"]
    assert answer_queries == ["how to reset passwrod"]
    assert bm25.await_args.args[1] == "password reset instructions"
    assert response.metadata.original_query == "how to reset passwrod"
    assert response.metadata.corrected_query == "how to reset password"
    assert response.metadata.retrieval_query == "password reset instructions"


def test_answer_generator_refuses_to_invent_without_context() -> None:
    answer = AnswerGenerator().generate("What was FY2025 revenue?", [])

    assert "couldn't find enough information" in answer


def test_local_answer_evaluator_accepts_supported_citation_and_rejects_unsupported_claim() -> None:
    context = RetrievedContext(
        chunk_id="chunk-1",
        document_id="document-1",
        filename="report.pdf",
        page_number=1,
        content="FY2025 revenue was 1250 crore.",
        score=0.9,
    )
    evaluator = AnswerEvaluator()

    supported = evaluator.evaluate(
        "What was revenue?",
        "FY2025 revenue was 1250 crore. [1]",
        [context],
    )
    unsupported = evaluator.evaluate(
        "What was revenue?",
        "The revenue was 9999 crore. [1]",
        [context],
    )

    assert supported.passed is True
    assert unsupported.passed is False


def test_answer_quality_loop_revises_failed_draft_before_returning() -> None:
    context = RetrievedContext(
        chunk_id="chunk-1",
        document_id="document-1",
        filename="report.pdf",
        page_number=1,
        content="FY2025 revenue was 1250 crore.",
        score=0.9,
    )

    class FakeGenerator:
        def generate(self, question: str, contexts: list[RetrievedContext]) -> str:
            return "unsupported draft"

        def generate_revision(
            self,
            question: str,
            contexts: list[RetrievedContext],
            draft: str,
            feedback: str,
        ) -> str:
            return "FY2025 revenue was 1250 crore. [1]"

    class FakeEvaluator:
        calls = 0

        def evaluate(
            self,
            question: str,
            draft: str,
            contexts: list[RetrievedContext],
        ) -> AnswerEvaluation:
            self.calls += 1
            if self.calls == 1:
                return AnswerEvaluation(False, 0.2, "Add supported evidence.")
            return AnswerEvaluation(True, 0.95, "Supported.")

    with (
        patch("app.api.routes.query.AnswerGenerator", return_value=FakeGenerator()),
        patch("app.api.routes.query.AnswerEvaluator", return_value=FakeEvaluator()),
    ):
        result = _generate_and_evaluate_answer("What was revenue?", [context])

    assert result.answer == "FY2025 revenue was 1250 crore. [1]"
    assert result.attempts == 2
    assert result.passed is True


def test_bm25_prefers_chunks_sharing_query_terms() -> None:
    first = DocumentChunk(
        id=uuid4(),
        document_id=uuid4(),
        organization_id=uuid4(),
        chunk_index=0,
        content="FY2025 revenue increased to 1250 crore.",
        page_number=1,
        start_char=0,
        end_char=43,
        token_count=6,
    )
    second = DocumentChunk(
        id=uuid4(),
        document_id=first.document_id,
        organization_id=first.organization_id,
        chunk_index=1,
        content="The office moved to a new building.",
        page_number=2,
        start_char=0,
        end_char=36,
        token_count=7,
    )

    results = BM25Index([(first, "report.pdf"), (second, "report.pdf")]).search(
        "What was FY2025 revenue?",
        limit=2,
    )

    assert results[0].chunk_id == str(first.id)


@pytest.mark.asyncio
async def test_bm25_loader_ranks_database_chunks_without_database_full_text_search() -> None:
    organization_id = uuid4()
    first = DocumentChunk(
        id=uuid4(),
        document_id=uuid4(),
        organization_id=organization_id,
        chunk_index=0,
        content="Contract renewal amount is 1250 crore.",
        page_number=1,
        start_char=0,
        end_char=38,
        token_count=6,
    )
    second = DocumentChunk(
        id=uuid4(),
        document_id=first.document_id,
        organization_id=organization_id,
        chunk_index=1,
        content="The office moved to a new building.",
        page_number=2,
        start_char=0,
        end_char=36,
        token_count=7,
    )

    class StatsResult:
        def one(self) -> tuple[int, float]:
            return (2, 6.5)

    class PostingResult:
        def all(self) -> list[tuple[object, ...]]:
            return [
                ("contract", first.id, 1, 6, first, "report.pdf"),
                ("renewal", first.id, 1, 6, first, "report.pdf"),
                ("amount", first.id, 1, 6, first, "report.pdf"),
                ("office", second.id, 1, 7, second, "report.pdf"),
            ]

    class FakeSession:
        calls = 0

        async def execute(self, statement: object) -> object:
            self.calls += 1
            return StatsResult() if self.calls == 1 else PostingResult()

    results = await search_bm25_chunks(
        FakeSession(),
        "What is the contract renewal amount?",
        organization_id,
        limit=2,
    )

    assert results[0].chunk_id == str(first.id)
    assert results[0].payload["filename"] == "report.pdf"


def test_rrf_rewards_results_found_by_both_retrievers() -> None:
    shared = RankedChunk("shared", 0.4, {"content": "shared"})
    vector_only = RankedChunk("vector", 0.9, {"content": "vector"})
    keyword_only = RankedChunk("keyword", 12.0, {"content": "keyword"})

    results = reciprocal_rank_fusion(
        [shared, vector_only],
        [shared, keyword_only],
        limit=3,
        rrf_k=60,
    )

    assert results[0].chunk_id == "shared"


def test_document_chunks_require_unique_indexes_per_document() -> None:
    constraint_names = {
        constraint.name for constraint in DocumentChunk.__table__.constraints
    }

    assert "uq_document_chunks_document_index" in constraint_names


def test_lexical_reranker_promotes_the_most_relevant_candidate() -> None:
    candidates = [
        RankedChunk(
            "weak",
            0.9,
            {"content": "The office moved to a new building."},
        ),
        RankedChunk(
            "strong",
            0.7,
            {"content": "FY2025 revenue was 1250 crore."},
        ),
    ]

    results = Reranker(provider="lexical").rerank(
        "What was FY2025 revenue?",
        candidates,
        limit=1,
    )

    assert results[0].chunk_id == "strong"
    assert results[0].retrieval_score == 0.7


def test_cross_encoder_reranker_scores_candidates_and_preserves_retrieval_score() -> None:
    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            assert model_name == "test-cross-encoder"
            assert kwargs["max_length"] == 512

        def predict(
            self,
            pairs: list[tuple[str, str]],
            **kwargs: object,
        ) -> list[float]:
            assert pairs[0][0] == "What was revenue?"
            assert kwargs["batch_size"] == 16
            return [0.1, 0.95]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    candidates = [
        RankedChunk("weak", 0.9, {"content": "The office moved."}),
        RankedChunk("strong", 0.7, {"content": "Revenue was 1250 crore."}),
    ]

    with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
        results = Reranker(
            provider="cross_encoder",
            model_name="test-cross-encoder",
        ).rerank("What was revenue?", candidates, limit=1)

    assert results[0].chunk_id == "strong"
    assert results[0].retrieval_score == 0.7


def test_cross_encoder_truncates_candidate_text_and_normalizes_scores() -> None:
    seen_pairs: list[tuple[str, str]] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            pass

        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            seen_pairs.extend(pairs)
            return [-2.0, 2.0]

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FakeCrossEncoder
    settings = Settings(
        reranker_provider="cross_encoder",
        reranker_max_text_chars=10,
        reranker_fallback_provider="none",
    )
    candidates = [
        RankedChunk("first", 0.2, {"content": "abcdefghijklmno"}),
        RankedChunk("second", 0.1, {"content": "short"}),
    ]

    with (
        patch("app.reranking.service.get_settings", return_value=settings),
        patch.dict(sys.modules, {"sentence_transformers": fake_module}),
    ):
        results = Reranker(provider="cross_encoder").rerank("query", candidates, limit=2)

    assert seen_pairs[0][1] == "abcdefghij"
    assert results[0].score == 1.0
    assert results[1].score == 0.0


def test_cross_encoder_uses_configured_lexical_fallback_when_model_fails() -> None:
    class FailingCrossEncoder:
        def __init__(self, model_name: str, **kwargs: object) -> None:
            pass

        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            raise RuntimeError("model unavailable")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = FailingCrossEncoder
    settings = Settings(
        reranker_provider="cross_encoder",
        reranker_fallback_provider="lexical",
    )
    candidates = [
        RankedChunk("weak", 0.9, {"content": "The office moved."}),
        RankedChunk("strong", 0.7, {"content": "Revenue was 1250 crore."}),
    ]

    with (
        patch("app.reranking.service.get_settings", return_value=settings),
        patch.dict(sys.modules, {"sentence_transformers": fake_module}),
    ):
        reranker = Reranker(provider="cross_encoder")
        results = reranker.rerank("What was revenue?", candidates, limit=1)

    assert reranker.last_used_provider == "lexical_fallback"
    assert results[0].chunk_id == "strong"


@pytest.mark.asyncio
async def test_cross_encoder_runs_before_answer_evaluation_loop() -> None:
    organization_id = uuid4()
    document_id = uuid4()
    candidate = RankedChunk(
        "chunk-1",
        0.4,
        {
            "chunk_id": "chunk-1",
            "document_id": str(document_id),
            "filename": "report.pdf",
            "page_number": 1,
            "content": "Revenue was 1250 crore.",
        },
    )
    events: list[str] = []

    class FakeReranker:
        def rerank(self, question: str, candidates: list[RankedChunk], limit: int):
            events.append("cross_encoder")
            return [candidate]

    def fake_answer(question: str, contexts: list[RetrievedContext]):
        events.append("answer_evaluation_loop")
        from app.generation import EvaluatedAnswer

        return EvaluatedAnswer("Revenue was 1250 crore. [1]", 1, 1.0, True)

    with (
        patch("app.api.routes.query.EmbeddingService.embed_query", return_value=[1.0]),
        patch("app.api.routes.query.QdrantVectorStore.search", return_value=[candidate]),
        patch(
            "app.api.routes.query.keep_completed_document_results",
            return_value=[candidate],
        ),
        patch("app.api.routes.query.search_bm25_chunks", return_value=[]),
        patch("app.api.routes.query.get_reranker", return_value=FakeReranker()),
        patch("app.api.routes.query._generate_and_evaluate_answer", side_effect=fake_answer),
    ):
        response = await _execute_uncached_query(
            QueryRequest(query="What was revenue?", top_k=1),
            MagicMock(),
            organization_id,
            "What was revenue?",
            None,
            None,
            0.0,
        )

    assert events == ["cross_encoder", "answer_evaluation_loop"]
    assert response.citations[0].chunk_id == "chunk-1"


def test_redelivered_worker_task_reclaims_its_own_processing_lease() -> None:
    organization_id = uuid4()
    document = Document(
        id=uuid4(),
        organization_id=organization_id,
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
        active_duplicate = _claim_document(document.id, "task-1")
        document.processing_started_at = datetime.now(UTC) - timedelta(hours=1)
        redelivery = _claim_document(document.id, "task-1", redelivered=True)
        document.processing_started_at = datetime.now(UTC)
        fresh_redelivery = _claim_document(document.id, "task-1", redelivered=True)

    assert active_duplicate == "ALREADY_PROCESSING"
    assert isinstance(redelivery, tuple)
    assert redelivery[0] == "original/report.pdf"
    assert fresh_redelivery == "ALREADY_PROCESSING"


def test_worker_requeues_lost_tasks_instead_of_acknowledging_early() -> None:
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1


def test_settings_reject_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
        Settings(chunk_size=100, chunk_overlap=100)


def test_query_request_rejects_whitespace_only_input() -> None:
    with pytest.raises(ValueError, match="non-whitespace"):
        QueryRequest(query="   ")


@pytest.mark.asyncio
async def test_query_concurrency_rejects_requests_after_local_capacity_wait() -> None:
    original_settings = query_concurrency.settings
    original_slots = query_concurrency._query_slots
    query_concurrency.settings = Settings(
        query_concurrency_limit=1,
        query_concurrency_wait_seconds=0.01,
        query_concurrency_distributed_enabled=False,
    )
    query_concurrency._query_slots = asyncio.Semaphore(1)
    first = query_concurrency.enforce_query_concurrency()
    second = query_concurrency.enforce_query_concurrency()
    try:
        await anext(first)
        with pytest.raises(HTTPException, match="capacity"):
            await anext(second)
    finally:
        await first.aclose()
        query_concurrency.settings = original_settings
        query_concurrency._query_slots = original_slots


@pytest.mark.asyncio
async def test_query_concurrency_fails_open_to_local_limit_when_redis_is_down() -> None:
    original_settings = query_concurrency.settings
    original_slots = query_concurrency._query_slots
    query_concurrency.settings = Settings(
        query_concurrency_limit=1,
        query_concurrency_wait_seconds=0.01,
        query_concurrency_distributed_enabled=True,
    )
    query_concurrency._query_slots = asyncio.Semaphore(1)
    first = query_concurrency.enforce_query_concurrency()
    try:
        with patch(
            "app.api.concurrency._acquire_distributed_slot",
            side_effect=query_concurrency.RedisError("redis down"),
        ):
            await anext(first)
    finally:
        await first.aclose()
        query_concurrency.settings = original_settings
        query_concurrency._query_slots = original_slots


def test_password_hashing_and_jwt_token_creation() -> None:
    password = "correct horse battery staple"
    encoded = hash_password(password)
    principal = Principal(uuid4(), uuid4(), MembershipRole.MEMBER, "user@example.com")

    assert verify_password(password, encoded)
    assert not verify_password("wrong password", encoded)
    assert isinstance(create_access_token(principal), str)


def test_query_endpoint_returns_grounded_citations() -> None:
    fake_result = VectorSearchResult(
        chunk_id=str(uuid4()),
        score=0.91,
        payload={
            "chunk_id": str(uuid4()),
            "document_id": str(uuid4()),
            "organization_id": "00000000-0000-0000-0000-000000000001",
            "filename": "annual_report.pdf",
            "page_number": 42,
            "content": "FY2025 revenue was 1250 crore.",
        },
    )

    class FakeResult:
        def __init__(self, rows: list[tuple[object, ...]]) -> None:
            self.rows = rows

        def all(self) -> list[tuple[object, ...]]:
            return self.rows

    class FakeSession:
        calls = 0

        async def execute(self, statement: object) -> FakeResult:
            self.calls += 1
            if self.calls == 1:
                return FakeResult([(UUID(fake_result.payload["document_id"]),)])
            return FakeResult([])

    async def override_get_db():
        yield FakeSession()

    app.dependency_overrides[get_db] = override_get_db
    try:
        with (
            patch("app.api.routes.query.QdrantVectorStore") as store_class,
            patch(
                "app.api.routes.query.search_bm25_chunks",
                new=AsyncMock(return_value=[]),
            ),
        ):
            store_class.return_value.search.return_value = [fake_result]
            with TestClient(app) as client:
                response = client.post(
                    "/api/v1/query",
                    json={"query": "What was FY2025 revenue?", "top_k": 5},
                )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200
    assert response.json()["citations"][0]["document_name"] == "annual_report.pdf"
    assert response.json()["citations"][0]["page"] == 42
    assert response.json()["metadata"]["reranker"] == "lexical"
    assert response.json()["metadata"]["reranked_count"] == 1
    assert response.json()["metadata"]["answer_evaluation_passed"] is True
