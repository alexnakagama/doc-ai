from typing import Protocol

from doc_ai.models.chunk import Chunk
from doc_ai.models.document import Document
from doc_ai.models.page import PageText


class ChunkingServiceInterface(Protocol):
    async def chunk(self, document: Document, pages: list[PageText]) -> list[Chunk]: ...
