from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from doc_ai.interfaces.document_service import DocumentServiceInterface
from doc_ai.interfaces.pdf_service import PDFServiceInterface
from doc_ai.schemas.document import DocumentResponse
from doc_ai.services.document_service import DocumentService
from doc_ai.services.pdf_service import PDFService

router = APIRouter()


def get_pdf_service() -> PDFServiceInterface:
    return PDFService()


def get_document_service(
    pdf_service: PDFServiceInterface,
) -> DocumentServiceInterface:
    return DocumentService(pdf_service)


DocumentServiceDependency = Annotated[
    DocumentServiceInterface,
    Depends(get_document_service),
]


@router.get("/documents")
async def get_documents(service: DocumentServiceDependency):
    return await service.get_documents()


@router.post("/documents", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile,
    service: DocumentServiceDependency,
):
    return await service.create_document(file)
