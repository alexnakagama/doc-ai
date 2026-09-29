import importlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from doc_ai.core.config import settings
from doc_ai.services.chunking_service import ChunkingService
from doc_ai.services.document_service import DocumentService
from doc_ai.services.embedding_service import EmbeddingService
from doc_ai.services.llm_service import LLMService
from doc_ai.services.pdf_service import PDFService
from doc_ai.services.question_service import NO_CONTENT_ANSWER, QuestionService
from doc_ai.services.retrieval_service import RetrievalService
from doc_ai.services.text_service import TextService
from doc_ai.services.vector_store import InMemoryVectorStore


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("secret embedding provider detail")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("secret embedding provider detail")


class FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("secret llm provider detail")


def make_client(
    chat_model: BaseChatModel, embeddings: Embeddings | None = None
) -> Iterator[TestClient]:
    # Importing the app builds the real OpenAI-backed services, which only needs
    # a key to be set; they are never called because both dependencies below are
    # overridden with services that use fake providers.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(settings, "openai_api_key", "test-key-not-used")
        dependencies = importlib.import_module("doc_ai.core.dependencies")
        app = importlib.import_module("doc_ai.main").app

    embedding_service = EmbeddingService(
        embeddings or DeterministicFakeEmbedding(size=8), model="fake-model"
    )
    vector_store = InMemoryVectorStore()
    document_service = DocumentService(
        PDFService(),
        TextService(),
        ChunkingService(chunk_size=200, chunk_overlap=20),
        embedding_service,
        vector_store,
    )
    question_service = QuestionService(
        document_service,
        RetrievalService(
            embedding_service, vector_store, top_k=4, max_question_length=200
        ),
        LLMService(chat_model),
    )

    app.dependency_overrides[dependencies.get_document_service] = lambda: (
        document_service
    )
    app.dependency_overrides[dependencies.get_question_service] = lambda: (
        question_service
    )
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield from make_client(FakeListChatModel(responses=["Refunds take 14 days."]))


@pytest.fixture
def failing_llm_client() -> Iterator[TestClient]:
    yield from make_client(FailingChatModel())


@pytest.fixture
def failing_embedding_client() -> Iterator[TestClient]:
    yield from make_client(FailingChatModel(), FailingEmbeddings())


def upload(client: TestClient, content: bytes) -> int:
    response = client.post(
        "/documents", files={"file": ("policy.txt", content, "text/plain")}
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_question_about_a_document_returns_the_answer(client):
    document_id = upload(client, b"Refunds take 14 days.")

    response = client.post(
        "/questions",
        json={"question": "How long do refunds take?", "document_id": document_id},
    )

    assert response.status_code == 200
    assert response.json() == {"answer": "Refunds take 14 days."}


def test_document_id_is_optional(client):
    upload(client, b"Refunds take 14 days.")

    response = client.post("/questions", json={"question": "Refunds?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Refunds take 14 days."}


def test_question_without_any_documents_returns_the_fixed_answer(client):
    response = client.post("/questions", json={"question": "Refunds?"})

    assert response.status_code == 200
    assert response.json() == {"answer": NO_CONTENT_ANSWER}


@pytest.mark.parametrize(
    "question", ["", "   ", "a" * 201], ids=["empty", "blank", "long"]
)
def test_invalid_question_returns_400(client, question):
    response = client.post("/questions", json={"question": question})

    assert response.status_code == 400
    assert "Question must" in response.json()["detail"]


def test_unknown_document_returns_404(client):
    response = client.post(
        "/questions", json={"question": "Refunds?", "document_id": 999}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": None},
        {"question": ["not", "a", "string"]},
        {"question": "Refunds?", "document_id": "one"},
    ],
    ids=["missing", "null", "list", "bad-document-id"],
)
def test_malformed_body_returns_422(client, body):
    assert client.post("/questions", json=body).status_code == 422


def test_non_json_body_returns_422(client):
    response = client.post(
        "/questions", content=b"not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_llm_failure_returns_502_without_provider_details(failing_llm_client):
    document_id = upload(failing_llm_client, b"Refunds take 14 days.")

    response = failing_llm_client.post(
        "/questions", json={"question": "Refunds?", "document_id": document_id}
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Language model provider failed"}


def test_embedding_failure_returns_502_without_provider_details(
    failing_embedding_client,
):
    response = failing_embedding_client.post(
        "/questions", json={"question": "Refunds?"}
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "Embedding provider failed"}
