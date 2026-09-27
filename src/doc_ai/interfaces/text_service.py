from typing import Protocol


class TextServiceInterface(Protocol):
    async def extract_text(self, file_path: str) -> str: ...
