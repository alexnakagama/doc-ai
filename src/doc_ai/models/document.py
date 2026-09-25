from pydantic import BaseModel
from time import datetime


class Document(BaseModel):
    id: int
    filename: str
    content: str
    created_at: datetime