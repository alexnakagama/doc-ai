from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from doc_ai.core.config import Settings
from doc_ai.exceptions.embedding import EmbeddingError
from doc_ai.models.chunk import Chunk
from doc_ai.models.embedding import ChunkEmbedding, QueryEmbedding


class EmbeddingService:
    def __init__(self, embeddings: Embeddings, model: str):
        self.embeddings = embeddings
        self.model = model

    async def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        if not chunks:
            return []

        try:
            vectors = await self.embeddings.aembed_documents(
                [chunk.content for chunk in chunks]
            )
        except Exception as error:
            raise EmbeddingError("Embedding provider failed") from error

        if len(vectors) != len(chunks):
            raise EmbeddingError(
                "Embedding provider returned the wrong number of vectors"
            )

        return [
            ChunkEmbedding(chunk_id=chunk.id, vector=vector, model=self.model)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    async def embed_query(self, text: str) -> QueryEmbedding:
        try:
            vector = await self.embeddings.aembed_query(text)
        except Exception as error:
            raise EmbeddingError("Embedding provider failed") from error

        if not vector:
            raise EmbeddingError("Embedding provider returned an empty vector")

        return QueryEmbedding(vector=vector, model=self.model)


def create_openai_embedding_service(settings: Settings) -> EmbeddingService:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    return EmbeddingService(embeddings, model=settings.embedding_model)
