from fastapi import APIRouter

from doc_ai.core.dependencies import QuestionServiceDependency
from doc_ai.models.search import SearchResult
from doc_ai.schemas.question import QuestionRequest, QuestionResponse, SourceResponse

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
    return QuestionResponse(
        answer=answer.text,
        sources=[_to_source_response(result) for result in answer.sources],
    )


def _to_source_response(result: SearchResult) -> SourceResponse:
    return SourceResponse(
        filename=result.chunk.filename,
        page_number=result.chunk.page_number,
    )
