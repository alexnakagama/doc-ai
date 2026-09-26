from typing import Annotated

from fastapi import APIRouter, Depends, UploadFile

from doc_ai.interfaces.document_service import DocumentServiceInterface
from doc_ai.schemas.document import DocumentResponse
from doc_ai.services.document_service import DocumentService
from doc_ai.services.pdf_service import PDFService

router = APIRouter()


pdf_service = PDFService()
document_service = DocumentService(pdf_service)


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
