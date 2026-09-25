from pydantic import BaseModel


class Document(BaseModel):
    id: int
    filename: str
    content: str