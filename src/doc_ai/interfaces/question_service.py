from typing import Protocol

from doc_ai.models.answer import Answer


class QuestionServiceInterface(Protocol):
    async def answer(self, question: str, document_id: int | None = None) -> Answer: ...
