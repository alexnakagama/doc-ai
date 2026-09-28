from typing import Protocol

from fastapi import UploadFile

from doc_ai.models.chunk import Chunk
from doc_ai.models.document import Document
from doc_ai.models.embedding import ChunkEmbedding


class DocumentServiceInterface(Protocol):
    async def get_documents(self) -> list[Document]: ...
    async def create_document(self, file: UploadFile) -> Document: ...
    async def get_chunks(self, document_id: int) -> list[Chunk]: ...
    async def get_embeddings(self, document_id: int) -> list[ChunkEmbedding]: ...
