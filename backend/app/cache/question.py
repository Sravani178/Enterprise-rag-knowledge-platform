import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from uuid import UUID, uuid4

from redis import Redis

from app.core.config import get_settings
from app.core.redis import create_redis_client


@dataclass(frozen=True, slots=True)
class CachedQuestionAnswer:
    response: dict[str, Any]
    similarity: float


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().casefold())


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first or not second:
        return 0.0
    numerator = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if not first_norm or not second_norm:
        return 0.0
    return numerator / (first_norm * second_norm)


@lru_cache(maxsize=1)
def get_question_cache_client() -> Redis:
    settings = get_settings()
    return create_redis_client(
        settings.question_cache_redis_url,
        decode_responses=True,
    )


class QuestionCache:
    """Tenant-scoped semantic answer cache backed by Redis.

    Redis stores question embeddings and response payloads. Similarity is
    calculated in the API process because the local Redis image does not
    require Redis Stack or a vector-search module. The sorted set score is the
    question hit count, so frequently asked questions are examined first.
    """

    def __init__(self, client: Redis | None = None) -> None:
        self.settings = get_settings()
        self.client = client or get_question_cache_client()

    @staticmethod
    def _organization_key(organization_id: UUID) -> str:
        return str(organization_id)

    def _version_key(self, organization_id: UUID) -> str:
        return f"faq:version:{self._organization_key(organization_id)}"

    def _index_key(self, organization_id: UUID) -> str:
        return f"faq:index:{self._organization_key(organization_id)}"

    def _exact_key(self, organization_id: UUID, question: str) -> str:
        digest = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
        return f"faq:exact:{self._organization_key(organization_id)}:{digest}"

    def _lock_key(self, organization_id: UUID, question: str) -> str:
        digest = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
        return f"faq:lock:{self._organization_key(organization_id)}:{digest}"

    def _entry_key(self, organization_id: UUID, entry_id: str) -> str:
        return f"faq:entry:{self._organization_key(organization_id)}:{entry_id}"

    def _version(self, organization_id: UUID) -> int:
        value = self.client.get(self._version_key(organization_id))
        return int(value or 0)

    def bump_version(self, organization_id: UUID) -> None:
        """Invalidate older answers after the tenant's knowledge changes."""

        self.client.incr(self._version_key(organization_id))

    def current_version(self, organization_id: UUID) -> int:
        return self._version(organization_id)

    def acquire_lock(self, organization_id: UUID, question: str, token: str) -> bool:
        return bool(
            self.client.set(
                self._lock_key(organization_id, question),
                token,
                ex=self.settings.question_cache_lock_seconds,
                nx=True,
            )
        )

    def release_lock(self, organization_id: UUID, question: str, token: str) -> None:
        self.client.eval(
            "if redis.call('get', KEYS[1]) == ARGV[1] then "
            "return redis.call('del', KEYS[1]) else return 0 end",
            1,
            self._lock_key(organization_id, question),
            token,
        )

    def wait_for_exact(
        self,
        organization_id: UUID,
        question: str,
        timeout_seconds: float = 15.0,
    ) -> CachedQuestionAnswer | None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            cached = self.get_exact(organization_id, question)
            if cached is not None:
                return cached
            time.sleep(0.05)
        return None

    def get_exact(
        self,
        organization_id: UUID,
        question: str,
    ) -> CachedQuestionAnswer | None:
        entry_id = self.client.get(self._exact_key(organization_id, question))
        if not entry_id:
            return None
        entry = self._read_entry(organization_id, str(entry_id))
        if entry is None:
            return None
        self.client.zincrby(self._index_key(organization_id), 1.0, str(entry_id))
        return CachedQuestionAnswer(entry["response"], 1.0)

    def get_semantic(
        self,
        organization_id: UUID,
        embedding: list[float],
    ) -> CachedQuestionAnswer | None:
        best: tuple[float, dict[str, Any], str] | None = None
        members = self.client.zrevrange(
            self._index_key(organization_id),
            0,
            self.settings.question_cache_max_candidates - 1,
        )
        for member in members:
            entry_id = str(member)
            entry = self._read_entry(organization_id, entry_id)
            if entry is None:
                self.client.zrem(self._index_key(organization_id), entry_id)
                continue
            similarity = cosine_similarity(embedding, entry["embedding"])
            if best is None or similarity > best[0]:
                best = (similarity, entry, entry_id)

        if best is None or best[0] < self.settings.question_cache_similarity_threshold:
            return None
        self.client.zincrby(self._index_key(organization_id), 1.0, best[2])
        return CachedQuestionAnswer(best[1]["response"], best[0])

    def put(
        self,
        organization_id: UUID,
        question: str,
        embedding: list[float],
        response: dict[str, Any],
        version: int | None = None,
    ) -> None:
        entry_id = uuid4().hex
        record = {
            "question": normalize_question(question),
            "embedding": embedding,
            "response": response,
            "version": self._version(organization_id) if version is None else version,
        }
        ttl = self.settings.question_cache_ttl_seconds
        self.client.setex(
            self._entry_key(organization_id, entry_id),
            ttl,
            json.dumps(record, ensure_ascii=False),
        )
        self.client.setex(self._exact_key(organization_id, question), ttl, entry_id)
        self.client.zadd(self._index_key(organization_id), {entry_id: 1.0})

    def _read_entry(self, organization_id: UUID, entry_id: str) -> dict[str, Any] | None:
        raw = self.client.get(self._entry_key(organization_id, entry_id))
        if not raw:
            return None
        try:
            entry = json.loads(raw)
            if int(entry["version"]) != self._version(organization_id):
                return None
            if not isinstance(entry["response"], dict) or not isinstance(
                entry["embedding"], list
            ):
                return None
            return entry
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
