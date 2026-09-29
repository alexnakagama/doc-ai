from pydantic import BaseModel

from doc_ai.models.chunk import Chunk


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
