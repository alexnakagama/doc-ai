from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    filename: str


class DocumentDetailResponse(DocumentResponse):
    content: str
    created_at: str
