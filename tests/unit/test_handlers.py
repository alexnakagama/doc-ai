import logging

import pytest

from doc_ai.exceptions.embedding import EmbeddingError
from doc_ai.exceptions.handlers import embedding_error_handler, invalid_question_handler
from doc_ai.exceptions.question import InvalidQuestionError

pytestmark = pytest.mark.anyio


async def test_embedding_error_is_logged_with_its_cause_and_returns_502(caplog):
    try:
        raise EmbeddingError("Embedding provider failed") from RuntimeError(
            "invalid api key"
        )
    except EmbeddingError as caught:
        error = caught

    with caplog.at_level(logging.ERROR):
        response = await embedding_error_handler(None, error)

    assert response.status_code == 502
    assert b"invalid api key" not in response.body
    assert "invalid api key" in caplog.text


async def test_invalid_question_returns_400_with_its_message():
    response = await invalid_question_handler(
        None, InvalidQuestionError("Question must not be empty")
    )

    assert response.status_code == 400
    assert response.body == b'{"detail":"Question must not be empty"}'
