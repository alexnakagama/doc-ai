import asyncio
from io import BytesIO

import pymupdf
import pytest
from fastapi import UploadFile
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from doc_ai.core.config import settings
from doc_ai.exceptions.document import (
    DocumentNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from doc_ai.exceptions.embedding import EmbeddingError
from doc_ai.models.chunk import Chunk
from doc_ai.models.embedding import ChunkEmbedding, QueryEmbedding
from doc_ai.services.chunking_service import ChunkingService
from doc_ai.services.document_service import DocumentService
from doc_ai.services.embedding_service import EmbeddingService
from doc_ai.services.pdf_service import PDFService
from doc_ai.services.text_service import TextService
from doc_ai.services.vector_store import InMemoryVectorStore

pytestmark = pytest.mark.anyio


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider is down")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("provider is down")


class SlowEmbeddings(DeterministicFakeEmbedding):
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        await asyncio.sleep(0.05)
        return self.embed_documents(texts)


def make_service(
    embeddings: Embeddings | None = None,
    vector_store: InMemoryVectorStore | None = None,
) -> DocumentService:
    return DocumentService(
        PDFService(),
        TextService(),
        ChunkingService(chunk_size=50, chunk_overlap=10),
        EmbeddingService(
            embeddings or DeterministicFakeEmbedding(size=8), model="fake-model"
        ),
        vector_store or InMemoryVectorStore(),
    )


def make_query(vector: list[float]) -> QueryEmbedding:
    return QueryEmbedding(vector=vector, model="fake-model")


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


async def test_txt_upload_creates_document_with_its_text():
    service = make_service()

    document = await service.create_document(
        make_upload("notes.txt", "café".encode("cp1252"))
    )

    assert document.filename == "notes.txt"
    assert document.content == "café"
    assert await service.get_documents() == [document]


async def test_empty_file_raises_empty_file_error():
    with pytest.raises(EmptyFileError):
        await make_service().create_document(make_upload("notes.txt", b""))


async def test_oversized_file_raises_file_too_large(monkeypatch):
    monkeypatch.setattr(settings, "max_file_size", 4)

    with pytest.raises(FileTooLargeError):
        await make_service().create_document(make_upload("notes.txt", b"12345"))


async def test_disallowed_extension_raises_unsupported_file_type():
    with pytest.raises(UnsupportedFileTypeError):
        await make_service().create_document(make_upload("report.docx", b"data"))


def make_pdf_bytes(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        document.new_page().insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


async def test_txt_upload_stores_chunks_for_the_document():
    service = make_service()
    text = " ".join(f"word{i}" for i in range(40))

    document = await service.create_document(
        make_upload("notes.txt", text.encode("utf-8"))
    )
    chunks = await service.get_chunks(document.id)

    assert len(chunks) > 1
    assert all(chunk.document_id == document.id for chunk in chunks)
    assert all(chunk.page_number is None for chunk in chunks)


async def test_pdf_upload_stores_page_numbered_chunks():
    service = make_service()

    document = await service.create_document(
        make_upload("report.pdf", make_pdf_bytes(["first page", "second page"]))
    )
    chunks = await service.get_chunks(document.id)

    assert [(chunk.page_number, chunk.content) for chunk in chunks] == [
        (1, "first page"),
        (2, "second page"),
    ]


async def test_get_chunks_for_unknown_document_raises_not_found():
    with pytest.raises(DocumentNotFoundError):
        await make_service().get_chunks(999)


async def test_upload_stores_an_embedding_for_each_chunk():
    service = make_service()
    text = " ".join(f"word{i}" for i in range(40))

    document = await service.create_document(
        make_upload("notes.txt", text.encode("utf-8"))
    )
    chunks = await service.get_chunks(document.id)
    embeddings = await service.get_embeddings(document.id)

    assert [embedding.chunk_id for embedding in embeddings] == [
        chunk.id for chunk in chunks
    ]
    assert all(len(embedding.vector) == 8 for embedding in embeddings)


async def test_upload_makes_chunks_searchable():
    vector_store = InMemoryVectorStore()
    service = make_service(vector_store=vector_store)
    text = " ".join(f"word{i}" for i in range(40))

    document = await service.create_document(
        make_upload("notes.txt", text.encode("utf-8"))
    )
    chunks = await service.get_chunks(document.id)
    embeddings = await service.get_embeddings(document.id)

    results = await vector_store.search(
        make_query(embeddings[1].vector), k=len(chunks), document_id=document.id
    )

    assert results[0].chunk == chunks[1]
    assert results[0].score == pytest.approx(1.0, abs=1e-6)
    assert {result.chunk.id for result in results} == {chunk.id for chunk in chunks}


async def test_embedding_failure_stores_nothing():
    vector_store = InMemoryVectorStore()
    service = make_service(FailingEmbeddings(), vector_store)

    with pytest.raises(EmbeddingError):
        await service.create_document(make_upload("notes.txt", b"some text"))

    assert await service.get_documents() == []
    with pytest.raises(DocumentNotFoundError):
        await service.get_chunks(1)
    assert await vector_store.search(make_query([1.0] * 8), k=1) == []


async def test_vector_store_rejection_stores_nothing():
    vector_store = InMemoryVectorStore()
    existing = Chunk(
        id="existing",
        document_id=0,
        chunk_index=0,
        content="stored by another model",
        filename="old.txt",
        page_number=None,
        start_index=0,
    )
    await vector_store.add(
        [existing],
        [ChunkEmbedding(chunk_id="existing", vector=[1.0] * 8, model="other-model")],
    )
    service = make_service(vector_store=vector_store)

    with pytest.raises(ValueError, match="model"):
        await service.create_document(make_upload("notes.txt", b"some text"))

    assert await service.get_documents() == []
    with pytest.raises(DocumentNotFoundError):
        await service.get_chunks(1)
    results = await vector_store.search(
        QueryEmbedding(vector=[1.0] * 8, model="other-model"), k=5
    )
    assert [result.chunk for result in results] == [existing]


async def test_get_embeddings_for_unknown_document_raises_not_found():
    with pytest.raises(DocumentNotFoundError):
        await make_service().get_embeddings(999)


async def test_concurrent_uploads_get_distinct_ids():
    vector_store = InMemoryVectorStore()
    service = make_service(SlowEmbeddings(size=8), vector_store)

    first, second = await asyncio.gather(
        service.create_document(make_upload("a.txt", b"first file")),
        service.create_document(make_upload("b.txt", b"second file")),
    )

    assert first.id != second.id
    assert (await service.get_chunks(first.id))[0].content == "first file"
    assert (await service.get_chunks(second.id))[0].content == "second file"
    for document in (first, second):
        [chunk] = await service.get_chunks(document.id)
        results = await vector_store.search(
            make_query([1.0] * 8), k=5, document_id=document.id
        )
        assert [result.chunk for result in results] == [chunk]
