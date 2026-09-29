import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.openai import post_with_retry


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """User-facing and retrieval-facing forms of one question."""

    original_query: str
    corrected_query: str
    retrieval_query: str


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _protected_terms(value: str) -> set[str]:
    """Return tokens that a rewrite must preserve exactly."""

    return {
        token
        for token in re.findall(r"[^\s]+", value)
        if any(character.isdigit() for character in token)
        or "-" in token
        or "_" in token
        or any(character.isupper() for character in token[1:])
    }


def _preserves_protected_terms(original: str, candidate: str) -> bool:
    return _protected_terms(original).issubset(set(re.findall(r"[^\s]+", candidate)))


class QueryUnderstandingService:
    """Conservative typo correction and retrieval query rewriting."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def analyze(self, original_query: str) -> QueryAnalysis:
        original = _normalize(original_query)
        if not self.settings.query_understanding_enabled:
            return QueryAnalysis(original, original, original)
        if self.settings.llm_provider != "openai" or not self.settings.openai_api_key:
            return QueryAnalysis(original, original, original)

        try:
            result = self._analyze_with_openai(original)
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return QueryAnalysis(original, original, original)

        corrected = self._safe_candidate(original, result.get("corrected_query"))
        rewritten = self._safe_candidate(corrected, result.get("retrieval_query"))
        return QueryAnalysis(original, corrected, rewritten)

    def _analyze_with_openai(self, original: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "temperature": 0,
            "max_tokens": self.settings.query_understanding_max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You prepare a search query for an enterprise document system. "
                        "Correct obvious spelling mistakes only when confidence is high. "
                        "Preserve IDs, numbers, product codes, filenames, acronyms, names, "
                        "and technical terms exactly. Then produce one concise retrieval "
                        "query using the same meaning. Do not answer the question. Return "
                        "JSON only with corrected_query and retrieval_query."
                    ),
                },
                {"role": "user", "content": original},
            ],
        }
        response = post_with_retry(
            f"{self.settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
        )
        response.raise_for_status()
        raw = str(response.json()["choices"][0]["message"]["content"]).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise TypeError("query-understanding response must be an object")
        return result

    @staticmethod
    def _safe_candidate(original: str, candidate: object) -> str:
        if not isinstance(candidate, str):
            return original
        normalized = _normalize(candidate)
        if not normalized or len(normalized) > 2000:
            return original
        if not _preserves_protected_terms(original, normalized):
            return original
        return normalized
