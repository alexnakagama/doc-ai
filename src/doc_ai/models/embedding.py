from pydantic import BaseModel


class ChunkEmbedding(BaseModel):
    chunk_id: str
    vector: list[float]
    model: str
