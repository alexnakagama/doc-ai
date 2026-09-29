from typing import Annotated

from fastapi import Depends

from doc_ai.core.config import settings
from doc_ai.interfaces.document_service import DocumentServiceInterface
from doc_ai.interfaces.question_service import QuestionServiceInterface
from doc_ai.services.chunking_service import ChunkingService
from doc_ai.services.document_service import DocumentService
from doc_ai.services.embedding_service import create_openai_embedding_service
from doc_ai.services.llm_service import create_openai_llm_service
from doc_ai.services.pdf_service import PDFService
from doc_ai.services.question_service import QuestionService
from doc_ai.services.retrieval_service import RetrievalService
from doc_ai.services.text_service import TextService
from doc_ai.services.vector_store import InMemoryVectorStore

pdf_service = PDFService()
text_service = TextService()
chunking_service = ChunkingService(settings.chunk_size, settings.chunk_overlap)
embedding_service = create_openai_embedding_service(settings)
vector_store = InMemoryVectorStore()
document_service = DocumentService(
    pdf_service,
    text_service,
    chunking_service,
    embedding_service,
    vector_store,
)
# Must share embedding_service and vector_store with document_service, so questions
# are embedded with the same model as the stored chunks.
retrieval_service = RetrievalService(
    embedding_service,
    vector_store,
    top_k=settings.retrieval_top_k,
    max_question_length=settings.max_question_length,
)
llm_service = create_openai_llm_service(settings)
question_service = QuestionService(document_service, retrieval_service, llm_service)


def get_document_service() -> DocumentServiceInterface:
    return document_service


def get_question_service() -> QuestionServiceInterface:
    return question_service


DocumentServiceDependency = Annotated[
    DocumentServiceInterface,
    Depends(get_document_service),
]

QuestionServiceDependency = Annotated[
    QuestionServiceInterface,
    Depends(get_question_service),
]
