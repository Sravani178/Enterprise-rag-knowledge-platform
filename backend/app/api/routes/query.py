import asyncio
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.concurrency import enforce_query_concurrency
from app.api.deps import get_organization_id, require_roles
from app.api.rate_limit import enforce_rate_limit
from app.auth import Principal
from app.cache import CachedQuestionAnswer, QuestionCache
from app.core.config import get_settings
from app.db.session import get_db
from app.embeddings import EmbeddingError, EmbeddingService
from app.generation import (
    AnswerEvaluationError,
    AnswerEvaluator,
    AnswerGenerator,
    EvaluatedAnswer,
    GenerationError,
    RetrievedContext,
)
from app.models import MembershipRole
from app.query import QueryUnderstandingService
from app.reranking import RerankerError, get_reranker
from app.retrieval.bm25 import search_bm25_chunks
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.types import RankedChunk
from app.retrieval.visibility import keep_completed_document_results
from app.vectorstore import QdrantVectorStore, VectorStoreError

router = APIRouter(prefix="/query", tags=["query"])
settings = get_settings()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class QueryCitation(BaseModel):
    citation_id: int
    chunk_id: str
    document_id: str
    document_name: str
    page: int
    score: float


class QueryMetadata(BaseModel):
    retrieval_count: int
    vector_count: int
    keyword_count: int
    candidate_count: int
    reranked_count: int
    reranker: str
    returned_count: int
    rerank_latency_ms: int
    answer_evaluation_attempts: int
    answer_evaluation_score: float
    answer_evaluation_passed: bool
    cache_hit: bool = False
    cache_similarity: float | None = None
    latency_ms: int
    original_query: str | None = None
    corrected_query: str | None = None
    retrieval_query: str | None = None
    reranker_model: str | None = None
    reranker_model_revision: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[QueryCitation]
    metadata: QueryMetadata


def _to_context(result: RankedChunk) -> RetrievedContext | None:
    payload = result.payload
    required = {"chunk_id", "document_id", "filename", "page_number", "content"}
    if not required.issubset(payload):
        return None
    return RetrievedContext(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        filename=str(payload["filename"]),
        page_number=int(payload["page_number"]),
        content=str(payload["content"]),
        score=result.score,
    )


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    organization_id: UUID = Depends(get_organization_id),
    _role: Principal | None = Depends(
        require_roles(
            MembershipRole.OWNER,
            MembershipRole.ADMIN,
            MembershipRole.MEMBER,
            MembershipRole.VIEWER,
        )
    ),
    _rate_limit: None = Depends(enforce_rate_limit),
    _concurrency: None = Depends(enforce_query_concurrency),
) -> QueryResponse:
    started_at = perf_counter()
    question = request.query.strip()
    question_cache = QuestionCache() if settings.question_cache_enabled else None
    if question_cache is not None:
        try:
            cached = await run_in_threadpool(
                question_cache.get_exact,
                organization_id,
                question,
            )
        except RedisError:
            cached = None
        if cached is not None:
            return _cached_query_response(cached, original_query=question)

    lock_token: str | None = None
    lock_acquired = False
    if question_cache is not None:
        lock_token = uuid4().hex
        try:
            lock_acquired = await run_in_threadpool(
                question_cache.acquire_lock,
                organization_id,
                question,
                lock_token,
            )
            if not lock_acquired:
                cached = await run_in_threadpool(
                    question_cache.wait_for_exact,
                    organization_id,
                    question,
                )
                if cached is not None:
                    return _cached_query_response(cached, original_query=question)
        except RedisError:
            question_cache = None

    cache_version: int | None = None
    if question_cache is not None:
        try:
            cache_version = await run_in_threadpool(
                question_cache.current_version,
                organization_id,
            )
        except RedisError:
            question_cache = None

    try:
        return await _execute_uncached_query(
            request,
            db,
            organization_id,
            question,
            question_cache,
            cache_version,
            started_at,
        )
    finally:
        if question_cache is not None and lock_acquired and lock_token is not None:
            try:
                await run_in_threadpool(
                    question_cache.release_lock,
                    organization_id,
                    question,
                    lock_token,
                )
            except RedisError:
                pass


async def _execute_uncached_query(
    request: QueryRequest,
    db: AsyncSession,
    organization_id: UUID,
    question: str,
    question_cache: QuestionCache | None,
    cache_version: int | None,
    started_at: float,
) -> QueryResponse:
    analysis = QueryUnderstandingService().analyze(question)
    retrieval_query = analysis.retrieval_query
    try:
        embedding_service = EmbeddingService()
        query_vector = await run_in_threadpool(embedding_service.embed_query, retrieval_query)
    except EmbeddingError as exc:
        raise HTTPException(status_code=503, detail="Embedding service is unavailable") from exc

    if question_cache is not None:
        try:
            cached = await run_in_threadpool(
                question_cache.get_semantic,
                organization_id,
                query_vector,
            )
        except RedisError:
            cached = None
        if cached is not None:
            return _cached_query_response(cached, original_query=question)

    try:
        vector_results = await run_in_threadpool(
            QdrantVectorStore().search,
            query_vector,
            organization_id,
            settings.vector_candidate_k,
        )
    except VectorStoreError as exc:
        raise HTTPException(status_code=503, detail="Vector search is unavailable") from exc

    try:
        vector_results = await keep_completed_document_results(
            db,
            vector_results,
            organization_id,
        )
    except (SQLAlchemyError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Document visibility is unavailable") from exc
    vector_results = [
        result for result in vector_results if result.score >= settings.min_retrieval_score
    ]
    try:
        keyword_results = await search_bm25_chunks(
            db,
            retrieval_query,
            organization_id,
            settings.bm25_candidate_k,
        )
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail="Keyword search is unavailable") from exc
    fused_results = reciprocal_rank_fusion(
        vector_results,
        keyword_results,
        limit=settings.reranker_candidate_k,
        rrf_k=settings.rrf_k,
    )
    rerank_started_at = perf_counter()
    reranker = get_reranker()
    try:
        reranked_results = await asyncio.wait_for(
            run_in_threadpool(
                reranker.rerank,
                retrieval_query,
                fused_results,
                request.top_k,
            ),
            timeout=settings.reranker_timeout_seconds,
        )
    except (RerankerError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Reranking is unavailable") from exc
    rerank_latency_ms = int((perf_counter() - rerank_started_at) * 1000)
    contexts = [
        context
        for result in reranked_results
        for context in [_to_context(result)]
        if context is not None
    ]
    try:
        answer_result = await run_in_threadpool(
            _generate_and_evaluate_answer,
            question,
            contexts,
        )
    except (AnswerEvaluationError, GenerationError) as exc:
        raise HTTPException(status_code=503, detail="Answer generation is unavailable") from exc
    citations = [
        QueryCitation(
            citation_id=index,
            chunk_id=context.chunk_id,
            document_id=context.document_id,
            document_name=context.filename,
            page=context.page_number,
            score=round(context.score, 4),
        )
        for index, context in enumerate(contexts, start=1)
    ]
    elapsed_ms = int((perf_counter() - started_at) * 1000)

    response = QueryResponse(
        answer=answer_result.answer,
        citations=citations,
        metadata=QueryMetadata(
            retrieval_count=len(vector_results) + len(keyword_results),
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            candidate_count=len(fused_results),
            reranked_count=len(reranked_results),
            reranker=getattr(reranker, "last_used_provider", settings.reranker_provider),
            reranker_model=settings.reranker_model,
            reranker_model_revision=settings.reranker_model_revision,
            returned_count=len(contexts),
            rerank_latency_ms=rerank_latency_ms,
            answer_evaluation_attempts=answer_result.attempts,
            answer_evaluation_score=answer_result.score,
            answer_evaluation_passed=answer_result.passed,
            latency_ms=elapsed_ms,
            original_query=analysis.original_query,
            corrected_query=analysis.corrected_query,
            retrieval_query=analysis.retrieval_query,
        ),
    )
    # Cache both grounded answers and the explicit no-context refusal. The
    # tenant version is bumped on every document upload/delete/reprocess, so a
    # refusal cannot survive a later knowledge-base change.
    if question_cache is not None and answer_result.passed:
        try:
            await run_in_threadpool(
                question_cache.put,
                organization_id,
            question,
            query_vector,
            response.model_dump(mode="json"),
            cache_version,
        )
        except RedisError:
            pass
    return response


def _cached_query_response(
    cached: CachedQuestionAnswer,
    *,
    original_query: str | None = None,
) -> QueryResponse:
    response = QueryResponse.model_validate(cached.response)
    response.metadata.cache_hit = True
    response.metadata.cache_similarity = round(cached.similarity, 4)
    if original_query is not None:
        response.metadata.original_query = original_query
        response.metadata.corrected_query = None
        response.metadata.retrieval_query = None
    return response


def _generate_and_evaluate_answer(
    question: str,
    contexts: list[RetrievedContext],
) -> EvaluatedAnswer:
    generator = AnswerGenerator()
    draft = generator.generate(question, contexts)
    if not settings.answer_evaluation_enabled or not contexts:
        return EvaluatedAnswer(draft, 0, 1.0, True)

    evaluator = AnswerEvaluator()
    best_answer = draft
    best_score = 0.0
    for attempt in range(1, settings.answer_evaluation_max_attempts + 1):
        evaluation = evaluator.evaluate(question, best_answer, contexts)
        best_score = max(best_score, evaluation.score)
        if evaluation.passed:
            return EvaluatedAnswer(best_answer, attempt, evaluation.score, True)
        if attempt < settings.answer_evaluation_max_attempts:
            best_answer = generator.generate_revision(
                question,
                contexts,
                best_answer,
                evaluation.feedback,
            )

    return EvaluatedAnswer(
        "I couldn't verify a sufficiently grounded answer in the retrieved documents.",
        settings.answer_evaluation_max_attempts,
        best_score,
        False,
    )
