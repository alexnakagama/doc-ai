from fastapi import APIRouter

from doc_ai.core.dependencies import QuestionServiceDependency
from doc_ai.schemas.question import QuestionRequest, QuestionResponse

router = APIRouter()


@router.post(
    "/questions",
    response_model=QuestionResponse,
)
async def ask_question(
    request: QuestionRequest,
    service: QuestionServiceDependency,
):
    answer = await service.answer(request.question, request.document_id)
    return QuestionResponse(answer=answer.text)
