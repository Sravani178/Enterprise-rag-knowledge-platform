import hashlib
import math
import re

import httpx

from app.core.config import get_settings
from app.core.openai import post_with_retry


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot create vectors."""


class EmbeddingService:
    """Deterministic local embeddings for a dependency-light Phase 3 MVP."""

    def __init__(self) -> None:
        settings = get_settings()
        if settings.embedding_provider not in {"hashing", "openai"}:
            raise EmbeddingError(
                f"Unsupported embedding provider: {settings.embedding_provider}. "
                "Phase 3 supports 'hashing' and 'openai'."
            )
        if settings.embedding_provider == "openai" and not settings.openai_api_key:
            raise EmbeddingError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self.provider = settings.embedding_provider
        self.model = settings.embedding_model
        self.api_key = settings.openai_api_key
        self.base_url = settings.openai_base_url
        self.dimension = settings.embedding_dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "openai":
            return self._embed_with_openai(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.provider == "openai":
            return self.embed_texts([text])[0]
        return self._embed(text)

    def _embed_with_openai(self, texts: list[str]) -> list[list[float]]:
        try:
            response = post_with_retry(
                f"{self.base_url.rstrip('/')}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                payload={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            embeddings = [
                item["embedding"]
                for item in sorted(response.json()["data"], key=lambda item: item["index"])
            ]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError("OpenAI embeddings are unavailable") from exc
        if embeddings and len(embeddings[0]) != self.dimension:
            raise EmbeddingError(
                f"EMBEDDING_DIMENSION={self.dimension} does not match the provider dimension "
                f"{len(embeddings[0])}"
            )
        return embeddings

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        vector = [0.0] * self.dimension

        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            return [value / norm for value in vector]
        return vector
