import pytest

from doc_ai.core.config import Settings


def test_chunk_settings_have_defaults(monkeypatch):
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)

    settings = Settings()

    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200


def test_chunk_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")

    settings = Settings()

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [("0", "0"), ("100", "-1"), ("100", "100"), ("100", "150")],
)
def test_invalid_chunk_settings_raise(monkeypatch, chunk_size, chunk_overlap):
    monkeypatch.setenv("CHUNK_SIZE", chunk_size)
    monkeypatch.setenv("CHUNK_OVERLAP", chunk_overlap)

    with pytest.raises(ValueError):
        Settings()


def test_embedding_settings_have_defaults(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    settings = Settings()

    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.openai_api_key is None


def test_embedding_settings_read_from_env(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    settings = Settings()

    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.openai_api_key == "sk-test"


def test_retrieval_settings_have_defaults(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)

    settings = Settings()

    assert settings.retrieval_top_k == 4
    assert settings.max_question_length == 2000


def test_retrieval_top_k_reads_from_env(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "8")

    assert Settings().retrieval_top_k == 8


@pytest.mark.parametrize("top_k", ["0", "-1"])
def test_non_positive_retrieval_top_k_raises(monkeypatch, top_k):
    monkeypatch.setenv("RETRIEVAL_TOP_K", top_k)

    with pytest.raises(ValueError, match="RETRIEVAL_TOP_K"):
        Settings()
