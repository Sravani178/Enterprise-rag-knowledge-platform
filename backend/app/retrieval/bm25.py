import math
import re
from collections import Counter
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import BM25Document, BM25Posting, Document, DocumentChunk
from app.retrieval.types import RankedChunk, chunk_to_payload


def tokenize(text: str) -> list[str]:
    # PostgreSQL VARCHAR(255) stores postings safely even for an unusually
    # long unbroken token from a hostile or malformed document.
    return [token[:255] for token in re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)]


class BM25Index:
    """In-memory BM25 index retained for deterministic unit-level testing."""

    def __init__(self, records: list[tuple[DocumentChunk, str]]) -> None:
        self.records = records
        self.document_tokens = [tokenize(chunk.content) for chunk, _ in records]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths) if self.document_lengths else 0
        )
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.document_tokens:
            self.document_frequency.update(set(tokens))

    def search(self, query: str, limit: int) -> list[RankedChunk]:
        query_terms = tokenize(query)
        if not query_terms or not self.records or not self.average_document_length:
            return []

        document_count = len(self.records)
        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, int]] = []

        for index, tokens in enumerate(self.document_tokens):
            term_frequency = Counter(tokens)
            document_length = self.document_lengths[index]
            score = 0.0
            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                normalization = k1 * (
                    1 - b + b * document_length / self.average_document_length
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (k1 + 1)
                    / (frequency + normalization)
                )

            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], self.records[item[1]][0].chunk_index))
        return [
            RankedChunk(
                chunk_id=str(self.records[index][0].id),
                score=score,
                payload=chunk_to_payload(*self.records[index]),
            )
            for score, index in scored[:limit]
        ]


def build_persistent_bm25_records(
    chunks: list[DocumentChunk],
) -> tuple[list[BM25Document], list[BM25Posting]]:
    documents: list[BM25Document] = []
    postings: list[BM25Posting] = []
    for chunk in chunks:
        tokens = tokenize(chunk.content)
        documents.append(
            BM25Document(
                chunk_id=chunk.id,
                organization_id=chunk.organization_id,
                document_id=chunk.document_id,
                document_length=len(tokens),
            )
        )
        for term, frequency in Counter(tokens).items():
            postings.append(
                BM25Posting(
                    organization_id=chunk.organization_id,
                    term=term,
                    chunk_id=chunk.id,
                    term_frequency=frequency,
                )
            )
    return documents, postings


@dataclass
class _PersistentCandidate:
    chunk: DocumentChunk
    filename: str
    document_length: int
    term_frequencies: dict[str, int] = field(default_factory=dict)


async def search_bm25_chunks(
    db: AsyncSession,
    query: str,
    organization_id: UUID,
    limit: int,
) -> list[RankedChunk]:
    """Search persisted BM25 postings without rebuilding an index per query."""

    query_terms = sorted(set(tokenize(query)))
    if not query_terms or limit <= 0:
        return []

    stats_result = await db.execute(
        select(
            func.count(BM25Document.chunk_id),
            func.avg(BM25Document.document_length),
        )
        .join(Document, Document.id == BM25Document.document_id)
        .where(
            BM25Document.organization_id == organization_id,
            Document.organization_id == organization_id,
            Document.status == "COMPLETED",
        )
    )
    document_count, average_document_length = stats_result.one()
    document_count = int(document_count or 0)
    average_document_length = float(average_document_length or 0.0)
    if not document_count or not average_document_length:
        return []

    result = await db.execute(
        select(
            BM25Posting.term,
            BM25Posting.chunk_id,
            BM25Posting.term_frequency,
            BM25Document.document_length,
            DocumentChunk,
            Document.filename,
        )
        .join(BM25Document, BM25Document.chunk_id == BM25Posting.chunk_id)
        .join(DocumentChunk, DocumentChunk.id == BM25Posting.chunk_id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            BM25Posting.organization_id == organization_id,
            BM25Posting.term.in_(query_terms),
            DocumentChunk.organization_id == organization_id,
            Document.organization_id == organization_id,
            Document.status == "COMPLETED",
        )
    )

    candidates: dict[UUID, _PersistentCandidate] = {}
    document_frequency: Counter[str] = Counter()
    for term, chunk_id, term_frequency, document_length, chunk, filename in result.all():
        candidate = candidates.setdefault(
            chunk_id,
            _PersistentCandidate(
                chunk=chunk,
                filename=filename,
                document_length=int(document_length),
            ),
        )
        candidate.term_frequencies[str(term)] = int(term_frequency)
        document_frequency[str(term)] += 1

    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, _PersistentCandidate]] = []
    for candidate in candidates.values():
        score = 0.0
        for term, frequency in candidate.term_frequencies.items():
            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            normalization = k1 * (
                1 - b + b * candidate.document_length / average_document_length
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1)
                / (frequency + normalization)
            )
        if score > 0:
            scored.append((score, candidate))

    scored.sort(key=lambda item: (-item[0], item[1].chunk.chunk_index))
    return [
        RankedChunk(
            chunk_id=str(candidate.chunk.id),
            score=score,
            payload=chunk_to_payload(candidate.chunk, candidate.filename),
        )
        for score, candidate in scored[:limit]
    ]


def delete_persistent_bm25_for_document(session: object, document_id: UUID) -> None:
    """Delete a document's postings before replacing/deleting its chunks."""

    chunk_ids = select(BM25Document.chunk_id).where(BM25Document.document_id == document_id)
    session.execute(delete(BM25Posting).where(BM25Posting.chunk_id.in_(chunk_ids)))
    session.execute(delete(BM25Document).where(BM25Document.document_id == document_id))
