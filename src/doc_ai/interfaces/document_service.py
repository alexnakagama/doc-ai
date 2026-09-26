from typing import Protocol

from doc_ai.models.document import Document


class DocumentServiceInterface(Protocol):
    async def get_documents(self) -> list[Document]: ...
