from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str
    document_id: int | None = None


class SourceResponse(BaseModel):
    filename: str
    page_number: int | None


class QuestionResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
