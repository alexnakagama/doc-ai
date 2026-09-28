from fastapi import APIRouter, UploadFile

from doc_ai.core.dependencies import DocumentServiceDependency
from doc_ai.schemas.chunk import ChunkResponse
from doc_ai.schemas.document import DocumentResponse

router = APIRouter()


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
