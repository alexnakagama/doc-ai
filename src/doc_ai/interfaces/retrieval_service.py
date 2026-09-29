from typing import Protocol

from doc_ai.models.search import SearchResult


class RetrievalServiceInterface(Protocol):
    async def retrieve(
        self, question: str, document_id: int | None = None
    ) -> list[SearchResult]: ...
