from time import datetime

from pydantic import BaseModel


class Document(BaseModel):
    id: int
    filename: str
    content: str
    created_at: datetime
