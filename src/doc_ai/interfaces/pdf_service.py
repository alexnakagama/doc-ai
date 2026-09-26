from typing import Protocol


class PDFServiceInterface(Protocol):
    async def extract_text(self, file_path: str) -> str: ...
