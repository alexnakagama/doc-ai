import numpy as np
import numpy.typing as npt

from doc_ai.models.chunk import Chunk
from doc_ai.models.embedding import ChunkEmbedding
from doc_ai.models.search import SearchResult


class InMemoryVectorStore:
    """Exact cosine-similarity search over precomputed chunk embeddings.

    Vectors are normalized once when added, so a search is a single
    matrix-vector product. Row i of the matrix belongs to chunk i.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._chunk_ids: set[str] = set()
        self._document_ids: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._vectors: npt.NDArray[np.float32] = np.empty((0, 0), dtype=np.float32)
        self._model: str | None = None

    async def add(self, chunks: list[Chunk], embeddings: list[ChunkEmbedding]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Expected exactly one embedding per chunk")

        if not chunks:
            return

        self._check_chunk_ids(chunks, embeddings)
        model = self._check_model(embeddings)
        vectors = _unit_vectors([embedding.vector for embedding in embeddings])
        self._check_dimension(vectors.shape[1])

        self._vectors = np.vstack((self._vectors, vectors)) if self._chunks else vectors
        self._document_ids = np.concatenate(
            (self._document_ids, [chunk.document_id for chunk in chunks])
        )
        self._chunks.extend(chunks)
        self._chunk_ids.update(chunk.id for chunk in chunks)
        self._model = model

    async def search(
        self,
        query_vector: list[float],
        k: int,
        document_id: int | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be greater than 0")

        query = _unit_vectors([query_vector])[0]

        if not self._chunks:
            return []

        self._check_dimension(query.shape[0])

        if document_id is None:
            candidates = np.arange(len(self._chunks))
        else:
            candidates = np.flatnonzero(self._document_ids == document_id)

        # Rounding can push float32 cosine values just outside [-1, 1].
        scores = np.clip(self._vectors[candidates] @ query, -1.0, 1.0)
        top = np.argsort(-scores, kind="stable")[:k]

        return [
            SearchResult(
                chunk=self._chunks[candidates[index]],
                score=float(scores[index]),
            )
            for index in top
        ]

    def _check_chunk_ids(
        self, chunks: list[Chunk], embeddings: list[ChunkEmbedding]
    ) -> None:
        seen: set[str] = set()

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.id != embedding.chunk_id:
                raise ValueError(
                    f"Embedding for chunk {embedding.chunk_id} "
                    f"does not match chunk {chunk.id}"
                )

            if chunk.id in self._chunk_ids or chunk.id in seen:
                raise ValueError(f"Chunk {chunk.id} is already stored")

            seen.add(chunk.id)

    def _check_model(self, embeddings: list[ChunkEmbedding]) -> str:
        models = {embedding.model for embedding in embeddings}

        if len(models) > 1:
            raise ValueError("All embeddings must come from the same model")

        [model] = models

        if self._model is not None and model != self._model:
            raise ValueError(
                f"Embedding model {model} does not match stored model {self._model}"
            )

        return model

    def _check_dimension(self, dimension: int) -> None:
        if self._chunks and dimension != self._vectors.shape[1]:
            raise ValueError(
                f"Vector dimension {dimension} does not match "
                f"stored dimension {self._vectors.shape[1]}"
            )


def _unit_vectors(vectors: list[list[float]]) -> npt.NDArray[np.float32]:
    dimensions = {len(vector) for vector in vectors}

    if len(dimensions) > 1:
        raise ValueError("All vectors must have the same dimension")

    if dimensions == {0}:
        raise ValueError("Vectors must not be empty")

    # Validate and normalize in float64 so large values do not overflow.
    matrix = np.asarray(vectors, dtype=np.float64)

    if not np.isfinite(matrix).all():
        raise ValueError("Vectors must contain only finite values")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)

    if not (np.isfinite(norms) & (norms > 0)).all():
        raise ValueError("Vectors must have a finite, non-zero length")

    return (matrix / norms).astype(np.float32)
