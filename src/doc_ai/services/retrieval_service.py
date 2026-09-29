from doc_ai.exceptions.question import InvalidQuestionError
from doc_ai.interfaces.embedding_service import EmbeddingServiceInterface
from doc_ai.interfaces.vector_store import VectorStoreInterface
from doc_ai.models.search import SearchResult


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingServiceInterface,
        vector_store: VectorStoreInterface,
        top_k: int,
        max_question_length: int,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if max_question_length <= 0:
            raise ValueError("max_question_length must be greater than 0")

        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.top_k = top_k
        self.max_question_length = max_question_length

    async def retrieve(
        self, question: str, document_id: int | None = None
    ) -> list[SearchResult]:
        question = question.strip()

        if not question:
            raise InvalidQuestionError("Question must not be empty")

        if len(question) > self.max_question_length:
            raise InvalidQuestionError(
                f"Question must be at most {self.max_question_length} characters"
            )

        query = await self.embedding_service.embed_query(question)

        return await self.vector_store.search(
            query, k=self.top_k, document_id=document_id
        )
