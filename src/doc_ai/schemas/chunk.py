from pydantic import BaseModel


class ChunkResponse(BaseModel):
    id: str
    document_id: int
    chunk_index: int
    content: str
    filename: str
    page_number: int | None
    start_index: int
