from doc_ai.interfaces.document_service import DocumentServiceInterface
from doc_ai.interfaces.llm_service import LLMServiceInterface
from doc_ai.interfaces.retrieval_service import RetrievalServiceInterface
from doc_ai.models.answer import Answer
from doc_ai.services.question_prompt import build_prompt

NO_CONTENT_ANSWER = "No document content is available to answer this question."


class QuestionService:
    def __init__(
        self,
        document_service: DocumentServiceInterface,
        retrieval_service: RetrievalServiceInterface,
        llm_service: LLMServiceInterface,
    ):
        self.document_service = document_service
        self.retrieval_service = retrieval_service
        self.llm_service = llm_service

    async def answer(self, question: str, document_id: int | None = None) -> Answer:
        # Checked first so an unknown document costs no provider calls.
        if document_id is not None:
            await self.document_service.get_document(document_id)

        results = await self.retrieval_service.retrieve(question, document_id)

        if not results:
            return Answer(text=NO_CONTENT_ANSWER, sources=[])

        text = await self.llm_service.generate(build_prompt(question, results))

        return Answer(text=text, sources=results)
