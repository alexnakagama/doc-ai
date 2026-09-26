from fastapi import APIRouter

from doc_ai.services.document_service import DocumentService

router = APIRouter()

service = DocumentService()


@router.get("/documents")
async def get_documents():
    return await service.get_docuemnts()
