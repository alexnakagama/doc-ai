from typing import Protocol

from doc_ai.models.chunk import Chunk
from doc_ai.models.embedding import ChunkEmbedding, QueryEmbedding
from doc_ai.models.search import SearchResult


class VectorStoreInterface(Protocol):
    async def add(
        self, chunks: list[Chunk], embeddings: list[ChunkEmbedding]
    ) -> None: ...

    async def search(
        self,
        query: QueryEmbedding,
        k: int,
        document_id: int | None = None,
    ) -> list[SearchResult]: ...
