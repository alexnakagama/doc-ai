from io import BytesIO

import pymupdf
import pytest
from fastapi import UploadFile
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    ParrotFakeChatModel,
)

from doc_ai.exceptions.document import DocumentNotFoundError
from doc_ai.exceptions.embedding import EmbeddingError
from doc_ai.exceptions.llm import LLMError
from doc_ai.exceptions.question import InvalidQuestionError
from doc_ai.services.chunking_service import ChunkingService
from doc_ai.services.document_service import DocumentService
from doc_ai.services.embedding_service import EmbeddingService
from doc_ai.services.llm_service import LLMService
from doc_ai.services.pdf_service import PDFService
from doc_ai.services.question_service import NO_CONTENT_ANSWER, QuestionService
from doc_ai.services.retrieval_service import RetrievalService
from doc_ai.services.text_service import TextService
from doc_ai.services.vector_store import InMemoryVectorStore

pytestmark = pytest.mark.anyio


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding provider is down")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embedding provider is down")


class FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("llm provider is down")


class App:
    """The services wired as in core/dependencies.py, with fake providers."""

    def __init__(
        self,
        chat_model: BaseChatModel | None = None,
        embeddings: Embeddings | None = None,
    ):
        embedding_service = EmbeddingService(
            embeddings or DeterministicFakeEmbedding(size=8), model="fake-model"
        )
        vector_store = InMemoryVectorStore()
        self.documents = DocumentService(
            PDFService(),
            TextService(),
            ChunkingService(chunk_size=200, chunk_overlap=20),
            embedding_service,
            vector_store,
        )
        self.retrieval = RetrievalService(
            embedding_service, vector_store, top_k=4, max_question_length=200
        )
        self.questions = QuestionService(
            self.documents,
            self.retrieval,
            LLMService(chat_model or FailingChatModel()),
        )

    async def upload(self, filename: str, content: bytes) -> int:
        upload = UploadFile(file=BytesIO(content), filename=filename)
        return (await self.documents.create_document(upload)).id


def make_pdf_bytes(page_texts: list[str]) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


async def test_answer_returns_the_llm_reply_and_the_retrieved_sources():
    app = App(FakeListChatModel(responses=["Refunds take 14 days."]))
    document_id = await app.upload("policy.txt", b"Refunds take 14 days.")

    answer = await app.questions.answer("How long do refunds take?", document_id)

    assert answer.text == "Refunds take 14 days."
    assert answer.sources == await app.retrieval.retrieve(
        "How long do refunds take?", document_id
    )
    assert [source.chunk.content for source in answer.sources] == [
        "Refunds take 14 days."
    ]


async def test_llm_receives_the_retrieved_chunks_and_the_question():
    app = App(ParrotFakeChatModel())
    document_id = await app.upload("policy.txt", b"Refunds take 14 days.")

    answer = await app.questions.answer("  How long do refunds take? ", document_id)

    assert answer.text == (
        "Sources:\n\n"
        '<source id="1" file="policy.txt">\n'
        "Refunds take 14 days.\n"
        "</source>\n\n"
        "Question: How long do refunds take?"
    )


async def test_question_across_all_documents_uses_every_document():
    app = App(FakeListChatModel(responses=["answer"]))
    first = await app.upload("a.txt", b"first file")
    second = await app.upload("b.txt", b"second file")

    answer = await app.questions.answer("file")

    assert {source.chunk.document_id for source in answer.sources} == {first, second}


async def test_document_id_limits_the_sources_to_that_document():
    app = App(FakeListChatModel(responses=["answer"]))
    await app.upload("a.txt", b"first file")
    second = await app.upload("b.txt", b"second file")

    answer = await app.questions.answer("first file", second)

    assert [source.chunk.document_id for source in answer.sources] == [second]


async def test_unknown_document_raises_before_calling_any_provider():
    app = App(FailingChatModel(), FailingEmbeddings())

    with pytest.raises(DocumentNotFoundError):
        await app.questions.answer("Question?", document_id=999)


async def test_document_without_text_returns_fixed_answer_without_calling_llm():
    app = App(FailingChatModel())
    document_id = await app.upload("scan.pdf", make_pdf_bytes([""]))

    answer = await app.questions.answer("What does it say?", document_id)

    assert answer.text == NO_CONTENT_ANSWER
    assert answer.sources == []


async def test_empty_store_returns_fixed_answer_without_calling_llm():
    answer = await App(FailingChatModel()).questions.answer("Question?")

    assert answer.text == NO_CONTENT_ANSWER
    assert answer.sources == []


@pytest.mark.parametrize("question", ["", "   "], ids=repr)
async def test_blank_question_raises_invalid_question(question: str):
    app = App(FailingChatModel(), FailingEmbeddings())

    with pytest.raises(InvalidQuestionError):
        await app.questions.answer(question)


async def test_embedding_failure_raises_embedding_error():
    app = App(FailingChatModel(), FailingEmbeddings())

    with pytest.raises(EmbeddingError):
        await app.questions.answer("Question?")


async def test_llm_failure_raises_llm_error():
    app = App(FailingChatModel())
    document_id = await app.upload("policy.txt", b"Refunds take 14 days.")

    with pytest.raises(LLMError):
        await app.questions.answer("How long do refunds take?", document_id)
