import re
from collections.abc import Sequence
from functools import lru_cache
from threading import BoundedSemaphore, Lock
from typing import Any

from app.core.config import get_settings
from app.retrieval.types import RankedChunk


class RerankerError(RuntimeError):
    """Raised when the configured reranker cannot score candidates."""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


class Reranker:
    """Rank retrieved chunks with a local fallback or optional cross-encoder."""

    def __init__(self, provider: str | None = None, model_name: str | None = None) -> None:
        settings = get_settings()
        self.provider = provider or settings.reranker_provider
        self.model_name = model_name or settings.reranker_model
        self.model_revision = settings.reranker_model_revision
        self.device = settings.reranker_device
        self.max_length = settings.reranker_max_length
        self.max_text_chars = settings.reranker_max_text_chars
        self.batch_size = settings.reranker_batch_size
        self.fallback_provider = settings.reranker_fallback_provider
        self._model: Any | None = None
        self._model_lock = Lock()
        self._inference_slots = BoundedSemaphore(settings.reranker_max_concurrency)
        self.last_used_provider = self.provider
        if self.provider not in {"lexical", "cross_encoder"}:
            raise RerankerError(f"Unsupported reranker provider: {self.provider}")
        if self.fallback_provider not in {"none", "lexical"}:
            raise RerankerError(f"Unsupported reranker fallback: {self.fallback_provider}")

    def rerank(
        self,
        query: str,
        candidates: Sequence[RankedChunk],
        limit: int,
    ) -> list[RankedChunk]:
        if limit <= 0 or not candidates:
            return []

        if self.provider == "cross_encoder":
            try:
                scores = self._cross_encoder_scores(query, candidates)
                self.last_used_provider = "cross_encoder"
            except RerankerError:
                if self.fallback_provider != "lexical":
                    raise
                scores = [self._lexical_score(query, candidate) for candidate in candidates]
                self.last_used_provider = "lexical_fallback"
        else:
            scores = [self._lexical_score(query, candidate) for candidate in candidates]
            self.last_used_provider = "lexical"

        scores = self._normalize_scores(scores)

        reranked = [
            RankedChunk(
                chunk_id=candidate.chunk_id,
                score=float(score),
                payload=dict(candidate.payload),
                retrieval_score=candidate.score,
            )
            for candidate, score in zip(candidates, scores, strict=True)
        ]
        reranked.sort(key=lambda result: (-result.score, result.chunk_id))
        return reranked[:limit]

    def _cross_encoder_scores(
        self,
        query: str,
        candidates: Sequence[RankedChunk],
    ) -> list[float]:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is required when RERANKER_PROVIDER=cross_encoder"
            ) from exc

        try:
            with self._model_lock:
                if self._model is None:
                    kwargs: dict[str, Any] = {"max_length": self.max_length}
                    if self.device != "auto":
                        kwargs["device"] = self.device
                    if self.model_revision:
                        kwargs["revision"] = self.model_revision
                    self._model = CrossEncoder(self.model_name, **kwargs)
            pairs = [
                (
                    query,
                    str(candidate.payload.get("content", ""))[: self.max_text_chars],
                )
                for candidate in candidates
            ]
            if not self._inference_slots.acquire(timeout=1.0):
                raise RerankerError("Cross-encoder concurrency capacity is full")
            try:
                scores = self._model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
            finally:
                self._inference_slots.release()
            return [float(score) for score in scores]
        except Exception as exc:
            if isinstance(exc, RerankerError):
                raise
            raise RerankerError("The cross-encoder could not score retrieval candidates") from exc

    @staticmethod
    def _normalize_scores(scores: Sequence[float]) -> list[float]:
        if not scores:
            return []
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return [1.0 for _ in scores]
        return [(float(score) - minimum) / (maximum - minimum) for score in scores]

    @staticmethod
    def _lexical_score(query: str, candidate: RankedChunk) -> float:
        query_tokens = _tokens(query)
        content = str(candidate.payload.get("content", ""))
        content_tokens = _tokens(content)
        if not query_tokens or not content_tokens:
            return 0.0

        query_set = set(query_tokens)
        content_set = set(content_tokens)
        overlap = len(query_set & content_set) / len(query_set)
        phrase_bonus = 1.0 if query.lower().strip() in content.lower() else 0.0
        repeated_term_bonus = sum(content_tokens.count(token) for token in query_set) / len(
            content_tokens
        )
        return overlap + phrase_bonus + repeated_term_bonus


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    """Reuse one reranker service, including its optional loaded model, per process."""

    return Reranker()
