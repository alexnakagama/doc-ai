from io import BytesIO

import pytest
from fastapi import UploadFile

from doc_ai.core.config import settings
from doc_ai.exceptions.document import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from doc_ai.services.document_service import DocumentService
from doc_ai.services.pdf_service import PDFService
from doc_ai.services.text_service import TextService

pytestmark = pytest.mark.anyio


def make_service() -> DocumentService:
    return DocumentService(PDFService(), TextService())


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


async def test_txt_upload_creates_document_with_its_text():
    service = make_service()

    document = await service.create_document(
        make_upload("notes.txt", "café".encode("cp1252"))
    )

    assert document.filename == "notes.txt"
    assert document.content == "café"
    assert await service.get_documents() == [document]


async def test_empty_file_raises_empty_file_error():
    with pytest.raises(EmptyFileError):
        await make_service().create_document(make_upload("notes.txt", b""))


async def test_oversized_file_raises_file_too_large(monkeypatch):
    monkeypatch.setattr(settings, "max_file_size", 4)

    with pytest.raises(FileTooLargeError):
        await make_service().create_document(make_upload("notes.txt", b"12345"))


async def test_disallowed_extension_raises_unsupported_file_type():
    with pytest.raises(UnsupportedFileTypeError):
        await make_service().create_document(make_upload("report.docx", b"data"))
