from uuid import UUID

from qdrant_client import QdrantClient, models

from app.core.config import get_settings
from app.embeddings import EmbeddingError, EmbeddingService
from app.models import DocumentChunk
from app.retrieval.types import RankedChunk, chunk_to_payload


class VectorStoreError(RuntimeError):
    """Raised when Qdrant cannot complete a vector operation."""


class QdrantVectorStore:
    def __init__(
        self,
        client: QdrantClient | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        settings = get_settings()
        self.collection = settings.qdrant_collection
        self.embedding_service = embedding_service or EmbeddingService()
        self.client = client or QdrantClient(url=settings.qdrant_url, timeout=30)

    def check_connection(self) -> None:
        try:
            self.client.get_collections()
        except Exception as exc:
            raise VectorStoreError("Unable to reach Qdrant") from exc

    def ensure_collection(self) -> None:
        try:
            exists = self.client.collection_exists(collection_name=self.collection)
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.embedding_service.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as exc:
            raise VectorStoreError("Unable to initialize the Qdrant collection") from exc

    def upsert_chunks(self, chunks: list[DocumentChunk], filename: str) -> None:
        if not chunks:
            return

        self.ensure_collection()
        try:
            vectors = self.embedding_service.embed_texts([chunk.content for chunk in chunks])
        except EmbeddingError as exc:
            raise VectorStoreError("Unable to create document chunk embeddings") from exc
        points = [
            models.PointStruct(
                id=str(chunk.id),
                vector=vector,
                payload=chunk_to_payload(chunk, filename),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        except Exception as exc:
            raise VectorStoreError("Unable to index document chunks in Qdrant") from exc

    def delete_document(self, document_id: UUID, organization_id: UUID) -> None:
        self.ensure_collection()
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=self._document_filter(document_id, organization_id)
                ),
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Unable to remove document vectors from Qdrant") from exc

    def count_document(self, document_id: UUID, organization_id: UUID) -> int:
        """Return the number of vectors belonging to one document."""

        self.ensure_collection()
        try:
            result = self.client.count(
                collection_name=self.collection,
                count_filter=self._document_filter(document_id, organization_id),
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            raise VectorStoreError("Unable to count document vectors in Qdrant") from exc

    def list_document_references(self) -> set[tuple[UUID, UUID]]:
        """List tenant/document pairs present in Qdrant for orphan cleanup."""

        self.ensure_collection()
        references: set[tuple[UUID, UUID]] = set()
        offset = None
        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=256,
                    offset=offset,
                    with_payload=["document_id", "organization_id"],
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    if not payload.get("document_id") or not payload.get("organization_id"):
                        continue
                    try:
                        organization_id = UUID(str(payload["organization_id"]))
                        document_id = UUID(str(payload["document_id"]))
                        references.add(
                            (organization_id, document_id)
                        )
                    except ValueError:
                        # Malformed legacy payloads are not allowed to break
                        # reconciliation of otherwise valid tenants.
                        continue
                if offset is None:
                    break
        except Exception as exc:
            raise VectorStoreError("Unable to enumerate Qdrant document vectors") from exc
        return references

    def search(
        self,
        query_vector: list[float],
        organization_id: UUID,
        limit: int,
    ) -> list[RankedChunk]:
        self.ensure_collection()
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="organization_id",
                    match=models.MatchValue(value=str(organization_id)),
                )
            ]
        )

        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
                points = response.points
            else:
                points = self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
        except Exception as exc:
            raise VectorStoreError("Unable to search Qdrant") from exc

        return [
            RankedChunk(
                chunk_id=str(point.id),
                score=float(point.score),
                payload=dict(point.payload or {}),
            )
            for point in points
        ]

    @staticmethod
    def _document_filter(document_id: UUID, organization_id: UUID) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="organization_id",
                    match=models.MatchValue(value=str(organization_id)),
                ),
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                ),
            ]
        )
