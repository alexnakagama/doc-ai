from io import BytesIO

import pymupdf
import pytest
from fastapi import UploadFile
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from doc_ai.exceptions.embedding import EmbeddingError
from doc_ai.exceptions.question import InvalidQuestionError
from doc_ai.models.chunk import Chunk
from doc_ai.services.chunking_service import ChunkingService
from doc_ai.services.document_service import DocumentService
from doc_ai.services.embedding_service import EmbeddingService
from doc_ai.services.pdf_service import PDFService
from doc_ai.services.retrieval_service import RetrievalService
from doc_ai.services.text_service import TextService
from doc_ai.services.vector_store import InMemoryVectorStore

pytestmark = pytest.mark.anyio

MAX_QUESTION_LENGTH = 50


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider is down")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("provider is down")


def make_embedding_service(
    embeddings: Embeddings | None = None, model: str = "fake-model"
) -> EmbeddingService:
    return EmbeddingService(embeddings or DeterministicFakeEmbedding(size=8), model)


def make_service(
    vector_store: InMemoryVectorStore,
    embedding_service: EmbeddingService | None = None,
    top_k: int = 4,
) -> RetrievalService:
    return RetrievalService(
        embedding_service or make_embedding_service(),
        vector_store,
        top_k=top_k,
        max_question_length=MAX_QUESTION_LENGTH,
    )


async def add_chunks(
    vector_store: InMemoryVectorStore, document_id: int, contents: list[str]
) -> list[Chunk]:
    chunks = [
        Chunk(
            id=f"{document_id}-{index}",
            document_id=document_id,
            chunk_index=index,
            content=content,
            filename=f"doc{document_id}.txt",
            page_number=None,
            start_index=0,
        )
        for index, content in enumerate(contents)
    ]
    embeddings = await make_embedding_service().embed_chunks(chunks)
    await vector_store.add(chunks, embeddings)
    return chunks


async def test_question_matching_a_chunk_returns_that_chunk_first():
    vector_store = InMemoryVectorStore()
    chunks = await add_chunks(vector_store, 1, ["alpha", "beta", "gamma"])

    results = await make_service(vector_store).retrieve("beta")

    assert results[0].chunk == chunks[1]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


async def test_returns_top_k_results_in_vector_store_order():
    vector_store = InMemoryVectorStore()
    await add_chunks(vector_store, 1, [f"text {index}" for index in range(6)])

    results = await make_service(vector_store, top_k=3).retrieve("question")

    query = await make_embedding_service().embed_query("question")
    assert results == await vector_store.search(query, k=3)
    assert len(results) == 3
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


async def test_returns_every_chunk_when_fewer_than_top_k_are_stored():
    vector_store = InMemoryVectorStore()
    await add_chunks(vector_store, 1, ["alpha", "beta"])

    results = await make_service(vector_store, top_k=4).retrieve("alpha")

    assert len(results) == 2


async def test_document_id_limits_results_to_that_document():
    vector_store = InMemoryVectorStore()
    await add_chunks(vector_store, 1, ["alpha", "beta"])
    second = await add_chunks(vector_store, 2, ["gamma", "delta"])

    results = await make_service(vector_store).retrieve("alpha", document_id=2)

    assert {result.chunk.id for result in results} == {chunk.id for chunk in second}


async def test_unknown_document_returns_nothing():
    vector_store = InMemoryVectorStore()
    await add_chunks(vector_store, 1, ["alpha"])

    assert await make_service(vector_store).retrieve("alpha", document_id=99) == []


async def test_empty_store_returns_nothing():
    assert await make_service(InMemoryVectorStore()).retrieve("alpha") == []


async def test_question_is_stripped_before_embedding():
    vector_store = InMemoryVectorStore()
    [chunk] = await add_chunks(vector_store, 1, ["alpha"])

    results = await make_service(vector_store).retrieve("  alpha \n")

    assert results[0].chunk == chunk
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("question", ["", "   ", "\n\t"], ids=repr)
async def test_blank_question_is_rejected_without_calling_the_provider(
    question: str,
):
    service = make_service(
        InMemoryVectorStore(), make_embedding_service(FailingEmbeddings())
    )

    with pytest.raises(InvalidQuestionError, match="empty"):
        await service.retrieve(question)


async def test_too_long_question_is_rejected_without_calling_the_provider():
    service = make_service(
        InMemoryVectorStore(), make_embedding_service(FailingEmbeddings())
    )

    with pytest.raises(InvalidQuestionError, match=str(MAX_QUESTION_LENGTH)):
        await service.retrieve("a" * (MAX_QUESTION_LENGTH + 1))


async def test_question_at_the_length_limit_is_accepted_after_stripping():
    service = make_service(InMemoryVectorStore())

    assert await service.retrieve(f"  {'a' * MAX_QUESTION_LENGTH}  ") == []


async def test_provider_failure_raises_embedding_error():
    service = make_service(
        InMemoryVectorStore(), make_embedding_service(FailingEmbeddings())
    )

    with pytest.raises(EmbeddingError):
        await service.retrieve("alpha")


@pytest.mark.parametrize(("top_k", "max_question_length"), [(0, 10), (-1, 10), (4, 0)])
def test_invalid_limits_are_rejected(top_k: int, max_question_length: int):
    with pytest.raises(ValueError):
        RetrievalService(
            make_embedding_service(),
            InMemoryVectorStore(),
            top_k=top_k,
            max_question_length=max_question_length,
        )


# Service-level integration: upload through DocumentService, then retrieve with
# the same EmbeddingService and vector store, wired as in core/dependencies.py.


def make_document_service(
    embedding_service: EmbeddingService, vector_store: InMemoryVectorStore
) -> DocumentService:
    return DocumentService(
        PDFService(),
        TextService(),
        ChunkingService(chunk_size=50, chunk_overlap=10),
        embedding_service,
        vector_store,
    )


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


def make_pdf_bytes(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        document.new_page().insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


async def test_uploaded_chunks_can_be_retrieved():
    embedding_service = make_embedding_service()
    vector_store = InMemoryVectorStore()
    documents = make_document_service(embedding_service, vector_store)
    retrieval = make_service(vector_store, embedding_service)
    text = " ".join(f"word{i}" for i in range(40))

    document = await documents.create_document(
        make_upload("notes.txt", text.encode("utf-8"))
    )
    chunks = await documents.get_chunks(document.id)

    results = await retrieval.retrieve(chunks[2].content)

    assert results[0].chunk == chunks[2]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


async def test_retrieval_is_limited_to_the_requested_upload():
    embedding_service = make_embedding_service()
    vector_store = InMemoryVectorStore()
    documents = make_document_service(embedding_service, vector_store)
    retrieval = make_service(vector_store, embedding_service)

    first = await documents.create_document(make_upload("a.txt", b"first file"))
    second = await documents.create_document(make_upload("b.txt", b"second file"))

    results = await retrieval.retrieve("first file", document_id=second.id)

    assert [result.chunk for result in results] == await documents.get_chunks(second.id)
    assert first.id != second.id


async def test_retrieved_pdf_chunks_carry_citation_metadata():
    embedding_service = make_embedding_service()
    vector_store = InMemoryVectorStore()
    documents = make_document_service(embedding_service, vector_store)
    retrieval = make_service(vector_store, embedding_service)

    document = await documents.create_document(
        make_upload("report.pdf", make_pdf_bytes(["first page", "second page"]))
    )

    [top, *_] = await retrieval.retrieve("second page", document_id=document.id)

    assert (top.chunk.filename, top.chunk.page_number) == ("report.pdf", 2)


async def test_retrieval_with_a_different_embedding_model_is_rejected():
    vector_store = InMemoryVectorStore()
    documents = make_document_service(make_embedding_service(), vector_store)
    retrieval = make_service(vector_store, make_embedding_service(model="other-model"))
    await documents.create_document(make_upload("notes.txt", b"some text"))

    with pytest.raises(ValueError, match="model"):
        await retrieval.retrieve("some text")
