from typing import Protocol

from doc_ai.models.chunk import Chunk
from doc_ai.models.embedding import ChunkEmbedding


class EmbeddingServiceInterface(Protocol):
    async def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]: ...
