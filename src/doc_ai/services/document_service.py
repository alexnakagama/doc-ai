import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from doc_ai.core.config import settings
from doc_ai.exceptions.document import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from doc_ai.interfaces.pdf_service import PDFServiceInterface
from doc_ai.interfaces.text_service import TextServiceInterface
from doc_ai.models.document import Document


class DocumentService:
    def __init__(
        self,
        pdf_service: PDFServiceInterface,
        text_service: TextServiceInterface,
    ):
        self.documents: list[Document] = []
        self.pdf_service = pdf_service
        self.text_service = text_service

    async def get_documents(self) -> list[Document]:
        return self.documents

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
                text = await self.pdf_service.extract_text(str(file_path))
            else:
                encoding = await self.text_service.detect_encoding(str(file_path))
                text = await self.text_service.extract_text(str(file_path), encoding)

            document = Document(
                id=len(self.documents) + 1,
                filename=filename,
                content=text,
                created_at=datetime.now(UTC),
            )

            self.documents.append(document)

            return document

        finally:
            if file_path is not None:
                file_path.unlink(missing_ok=True)
