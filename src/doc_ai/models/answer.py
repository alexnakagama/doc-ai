from pydantic import BaseModel

from doc_ai.models.search import SearchResult


class Answer(BaseModel):
    text: str
    sources: list[SearchResult]
