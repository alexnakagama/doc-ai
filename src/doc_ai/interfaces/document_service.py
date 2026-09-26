from typing import Protocol

from fastapi import UploadFile

from doc_ai.models.document import Document


class DocumentServiceInterface(Protocol):
    async def get_documents(self) -> list[Document]: ...
    async def create_document(self, file: UploadFile) -> Document: ...
