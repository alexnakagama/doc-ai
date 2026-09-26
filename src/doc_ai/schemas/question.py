from pydantic import BaseModel


class QuestionRequest(BaseModel):
    document_id: int
    question: str


class QuestionResponse(BaseModel):
    answer: str
