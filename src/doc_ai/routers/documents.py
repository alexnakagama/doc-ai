from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from doc_ai.core.config import settings
from doc_ai.interfaces.document_service import DocumentServiceInterface
from doc_ai.schemas.chunk import ChunkResponse
from doc_ai.schemas.document import DocumentResponse
from doc_ai.services.chunking_service import ChunkingService
from doc_ai.services.document_service import DocumentService
from doc_ai.services.pdf_service import PDFService
from doc_ai.services.text_service import TextService

router = APIRouter()


pdf_service = PDFService()
text_service = TextService()
chunking_service = ChunkingService(settings.chunk_size, settings.chunk_overlap)
document_service = DocumentService(pdf_service, text_service, chunking_service)


def get_document_service() -> DocumentServiceInterface:
    return document_service


DocumentServiceDependency = Annotated[
    DocumentServiceInterface,
    Depends(get_document_service),
]


@router.get(
    "/documents",
    response_model=list[DocumentResponse],
)
async def get_documents(service: DocumentServiceDependency):
    return await service.get_documents()


@router.post(
    "/documents",
    response_model=DocumentResponse,
)
async def upload_document(
    file: UploadFile,
    service: DocumentServiceDependency,
):
    return await service.create_document(file)


@router.get(
    "/documents/{document_id}/chunks",
    response_model=list[ChunkResponse],
)
async def get_document_chunks(
    document_id: int,
    service: DocumentServiceDependency,
):
    return await service.get_chunks(document_id)
