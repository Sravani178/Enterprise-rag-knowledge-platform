import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_index: int
    content: str
    page_number: int
    start_char: int
    end_char: int
    token_count: int


@dataclass(frozen=True, slots=True)
class _TextUnit:
    start_char: int
    end_char: int
    content: str


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def _token_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    numerator = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = sum(value * value for value in first) ** 0.5
    second_norm = sum(value * value for value in second) ** 0.5
    if not first_norm or not second_norm:
        return 0.0
    return numerator / (first_norm * second_norm)


def _split_into_units(text: str) -> list[_TextUnit]:
    """Split text into paragraph/sentence units while preserving offsets."""

    units: list[_TextUnit] = []
    paragraph_pattern = re.compile(r"\S.*?(?=\n\s*\n|\Z)", flags=re.DOTALL)
    sentence_pattern = re.compile(r".+?(?:[.!?](?=\s|$)|\Z)", flags=re.DOTALL)
    for paragraph in paragraph_pattern.finditer(text):
        paragraph_text = paragraph.group()
        for sentence in sentence_pattern.finditer(paragraph_text):
            raw = sentence.group()
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw.rstrip())
            start = paragraph.start() + sentence.start() + leading
            end = paragraph.start() + sentence.start() + trailing
            if start < end:
                units.append(_TextUnit(start, end, text[start:end]))
    return units


def _split_long_unit(unit: _TextUnit, text: str, chunk_size: int) -> list[_TextUnit]:
    if len(unit.content) <= chunk_size:
        return [unit]
    pieces: list[_TextUnit] = []
    start = unit.start_char
    while start < unit.end_char:
        end = min(start + chunk_size, unit.end_char)
        boundary = text.rfind(" ", start + (chunk_size // 2), end)
        if boundary > start:
            end = boundary
        content = text[start:end].strip()
        if content:
            leading = len(text[start:end]) - len(text[start:end].lstrip())
            trailing = len(text[start:end].rstrip())
            content_start = start + leading
            content_end = start + trailing
            pieces.append(_TextUnit(content_start, content_end, text[content_start:content_end]))
        if end >= unit.end_char:
            break
        start = max(end, start + 1)
    return pieces


def _emit_unit_chunk(
    chunks: list[TextChunk],
    text: str,
    units: list[_TextUnit],
    page_number: int,
) -> None:
    if not units:
        return
    start_char = units[0].start_char
    end_char = units[-1].end_char
    content = text[start_char:end_char].strip()
    if not content:
        return
    content_start = start_char + len(text[start_char:end_char]) - len(
        text[start_char:end_char].lstrip()
    )
    content_end = start_char + len(text[start_char:end_char].rstrip())
    chunks.append(
        TextChunk(
            chunk_index=len(chunks),
            content=content,
            page_number=page_number,
            start_char=content_start,
            end_char=content_end,
            token_count=_token_count(content),
        )
    )


def chunk_pages(
    pages: list[dict[str, Any]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_provider: EmbeddingProvider | None = None,
    semantic_similarity_threshold: float = 0.78,
    semantic_min_size: int = 0,
) -> list[TextChunk]:
    """Create hybrid semantic chunks with hard size and overlap limits.

    Pages are first split at paragraph/sentence boundaries. When an embedding
    provider is supplied, a topic-shift boundary is added when adjacent units
    fall below ``semantic_similarity_threshold``. The hard size limit remains
    the safety guard for very long paragraphs or sentences.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
    if not 0.0 <= semantic_similarity_threshold <= 1.0:
        raise ValueError("semantic_similarity_threshold must be between 0 and 1")
    if semantic_min_size < 0 or semantic_min_size >= chunk_size:
        raise ValueError("semantic_min_size must be non-negative and smaller than chunk_size")

    chunks: list[TextChunk] = []

    for page in pages:
        page_number = int(page["page_number"])
        text = str(page.get("text", "")).strip()
        if not text:
            continue

        units = _split_into_units(text)
        if not units:
            units = [_TextUnit(0, len(text), text)]
        expanded_units = [
            piece
            for unit in units
            for piece in _split_long_unit(unit, text, chunk_size)
        ]
        embeddings = (
            embedding_provider.embed_texts([unit.content for unit in expanded_units])
            if embedding_provider is not None
            else None
        )
        current: list[_TextUnit] = []
        for index, unit in enumerate(expanded_units):
            if current:
                current_length = unit.end_char - current[0].start_char
                similarity = (
                    _cosine_similarity(embeddings[index - 1], embeddings[index])
                    if embeddings is not None
                    else 1.0
                )
                semantic_boundary = (
                    embeddings is not None
                    and similarity < semantic_similarity_threshold
                    and current_length >= semantic_min_size
                )
                size_boundary = current_length > chunk_size
                if semantic_boundary or size_boundary:
                    _emit_unit_chunk(chunks, text, current, page_number)
                    overlap: list[_TextUnit] = []
                    overlap_length = 0
                    for previous in reversed(current):
                        if overlap_length >= chunk_overlap:
                            break
                        overlap.insert(0, previous)
                        overlap_length += previous.end_char - previous.start_char
                    current = overlap
            current.append(unit)
        _emit_unit_chunk(chunks, text, current, page_number)

    return [
        TextChunk(
            chunk_index=index,
            content=chunk.content,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            token_count=chunk.token_count,
        )
        for index, chunk in enumerate(chunks)
    ]
