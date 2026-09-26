from typing import Protocol

from doc_ai.models.document import Document


class DocumentServiceInterface(Protocol):
    def __init__(self):
        self.documents: list[Document] = []

    async def get_documents(self) -> list[Document]: ...
