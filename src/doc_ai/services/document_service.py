import itertools
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from doc_ai.core.config import settings
from doc_ai.exceptions.document import (
    DocumentNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from doc_ai.interfaces.chunking_service import ChunkingServiceInterface
from doc_ai.interfaces.embedding_service import EmbeddingServiceInterface
from doc_ai.interfaces.pdf_service import PDFServiceInterface
from doc_ai.interfaces.text_service import TextServiceInterface
from doc_ai.interfaces.vector_store import VectorStoreInterface
from doc_ai.models.chunk import Chunk
from doc_ai.models.document import Document
from doc_ai.models.embedding import ChunkEmbedding
from doc_ai.models.page import PageText


class DocumentService:
    def __init__(
        self,
        pdf_service: PDFServiceInterface,
        text_service: TextServiceInterface,
        chunking_service: ChunkingServiceInterface,
        embedding_service: EmbeddingServiceInterface,
        vector_store: VectorStoreInterface,
    ):
        self.documents: list[Document] = []
        self.chunks: dict[int, list[Chunk]] = {}
        self.embeddings: dict[int, list[ChunkEmbedding]] = {}
        self.document_ids = itertools.count(1)
        self.pdf_service = pdf_service
        self.text_service = text_service
        self.chunking_service = chunking_service
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def get_documents(self) -> list[Document]:
        return self.documents

    async def get_document(self, document_id: int) -> Document:
        for document in self.documents:
            if document.id == document_id:
                return document

        raise DocumentNotFoundError("Document not found")

    async def get_chunks(self, document_id: int) -> list[Chunk]:
        if document_id not in self.chunks:
            raise DocumentNotFoundError("Document not found")

        return self.chunks[document_id]

    async def get_embeddings(self, document_id: int) -> list[ChunkEmbedding]:
        if document_id not in self.embeddings:
            raise DocumentNotFoundError("Document not found")

        return self.embeddings[document_id]

    async def create_document(self, file: UploadFile) -> Document:
        content = await file.read()

        if not content:
            raise EmptyFileError("File is empty")

        if len(content) > settings.max_file_size:
            raise FileTooLargeError("File is too large")

        filename = file.filename or "unknown file"
        extension = Path(filename).suffix.lower()

        if extension not in settings.allowed_file_types:
            raise UnsupportedFileTypeError("File type is not allowed")

        file_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=extension,
            ) as temp_file:
                temp_file.write(content)
                file_path = Path(temp_file.name)

            if extension == ".pdf":
                pages = await self.pdf_service.extract_pages(str(file_path))
            else:
                encoding = await self.text_service.detect_encoding(str(file_path))
                text = await self.text_service.extract_text(str(file_path), encoding)
                pages = [PageText(page_number=None, text=text)]

            document = Document(
                id=next(self.document_ids),
                filename=filename,
                content="\n\n".join(page.text for page in pages),
                created_at=datetime.now(UTC),
            )

            chunks = await self.chunking_service.chunk(document, pages)
            embeddings = await self.embedding_service.embed_chunks(chunks)

            await self.vector_store.add(chunks, embeddings)
            self.documents.append(document)
            self.chunks[document.id] = chunks
            self.embeddings[document.id] = embeddings

            return document

        finally:
            if file_path is not None:
                file_path.unlink(missing_ok=True)
