import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from doc_ai.core.config import Settings
from doc_ai.exceptions.embedding import EmbeddingError
from doc_ai.models.chunk import Chunk
from doc_ai.services.embedding_service import (
    EmbeddingService,
    create_openai_embedding_service,
)

pytestmark = pytest.mark.anyio


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider is down")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("provider is down")


class SingleVectorEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0]]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


class EmptyVectorEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return []


def make_chunk(chunk_index: int, content: str) -> Chunk:
    return Chunk(
        id=f"1-{chunk_index}",
        document_id=1,
        chunk_index=chunk_index,
        content=content,
        filename="notes.txt",
        page_number=None,
        start_index=0,
    )


def make_service() -> EmbeddingService:
    return EmbeddingService(DeterministicFakeEmbedding(size=8), model="fake-model")


async def test_embeds_each_chunk_in_order():
    chunks = [make_chunk(0, "first"), make_chunk(1, "second")]

    embeddings = await make_service().embed_chunks(chunks)

    assert [embedding.chunk_id for embedding in embeddings] == ["1-0", "1-1"]
    assert all(embedding.model == "fake-model" for embedding in embeddings)
    assert all(len(embedding.vector) == 8 for embedding in embeddings)


async def test_same_text_gets_same_vector():
    chunks = [make_chunk(0, "same text"), make_chunk(1, "same text")]

    first, second = await make_service().embed_chunks(chunks)

    assert first.vector == second.vector


async def test_no_chunks_does_not_call_the_provider():
    service = EmbeddingService(FailingEmbeddings(), model="fake-model")

    assert await service.embed_chunks([]) == []


async def test_provider_error_raises_embedding_error():
    service = EmbeddingService(FailingEmbeddings(), model="fake-model")

    with pytest.raises(EmbeddingError):
        await service.embed_chunks([make_chunk(0, "text")])


async def test_wrong_number_of_vectors_raises_embedding_error():
    service = EmbeddingService(SingleVectorEmbeddings(), model="fake-model")
    chunks = [make_chunk(0, "first"), make_chunk(1, "second")]

    with pytest.raises(EmbeddingError):
        await service.embed_chunks(chunks)


async def test_query_gets_the_same_vector_as_a_chunk_with_the_same_text():
    service = make_service()

    query = await service.embed_query("same text")
    [chunk_embedding] = await service.embed_chunks([make_chunk(0, "same text")])

    assert query.vector == chunk_embedding.vector
    assert query.model == "fake-model"


async def test_query_provider_error_raises_embedding_error_with_cause():
    service = EmbeddingService(FailingEmbeddings(), model="fake-model")

    with pytest.raises(EmbeddingError) as error:
        await service.embed_query("question")

    assert isinstance(error.value.__cause__, RuntimeError)


async def test_empty_query_vector_raises_embedding_error():
    service = EmbeddingService(EmptyVectorEmbeddings(), model="fake-model")

    with pytest.raises(EmbeddingError):
        await service.embed_query("question")


def test_openai_factory_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_openai_embedding_service(Settings())


def test_openai_factory_builds_service_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    service = create_openai_embedding_service(Settings())

    assert isinstance(service, EmbeddingService)
