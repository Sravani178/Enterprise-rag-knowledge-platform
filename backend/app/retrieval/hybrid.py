from collections.abc import Sequence

from app.retrieval.types import RankedChunk


def reciprocal_rank_fusion(
    vector_results: Sequence[RankedChunk],
    keyword_results: Sequence[RankedChunk],
    *,
    limit: int,
    rrf_k: int,
) -> list[RankedChunk]:
    """Merge ranked lists without assuming their raw scores share a scale."""

    fused: dict[str, RankedChunk] = {}
    for ranked_results in (vector_results, keyword_results):
        for rank, result in enumerate(ranked_results, start=1):
            contribution = 1.0 / (rrf_k + rank)
            existing = fused.get(result.chunk_id)
            if existing is None:
                fused[result.chunk_id] = RankedChunk(
                    chunk_id=result.chunk_id,
                    score=contribution,
                    payload=dict(result.payload),
                )
            else:
                fused[result.chunk_id] = RankedChunk(
                    chunk_id=result.chunk_id,
                    score=existing.score + contribution,
                    payload={**existing.payload, **result.payload},
                )

    return sorted(fused.values(), key=lambda result: (-result.score, result.chunk_id))[:limit]
