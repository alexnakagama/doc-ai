from typing import Protocol

from doc_ai.models.chunk import Chunk
from doc_ai.models.embedding import ChunkEmbedding, QueryEmbedding


class EmbeddingServiceInterface(Protocol):
    async def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]: ...
    async def embed_query(self, text: str) -> QueryEmbedding: ...
