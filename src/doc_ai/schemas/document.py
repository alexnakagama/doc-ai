from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: int
    filename: str


class DocumentDetailResponse(DocumentResponse):
    id: int
    filename: str
    content: str
