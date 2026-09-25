from pydantic import BaseModel


class Chunk(BaseModel):
    id: int
    document_id: int
    content: str
