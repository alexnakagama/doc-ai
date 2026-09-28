from typing import Protocol

from doc_ai.models.page import PageText


class PDFServiceInterface(Protocol):
    async def extract_pages(self, file_path: str) -> list[PageText]: ...
