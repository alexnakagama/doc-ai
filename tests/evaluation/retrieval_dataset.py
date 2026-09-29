"""Synthetic corpus and questions with hand-picked relevant chunk ids.

The chunks are written for this dataset and added straight to the vector store,
so their ids are fixed and don't depend on extraction or chunking.
"""

import re
import zlib

from langchain_core.embeddings import Embeddings
from retrieval_metrics import KS, EvaluationCase

from doc_ai.models.chunk import Chunk
from doc_ai.services.embedding_service import EmbeddingService
from doc_ai.services.retrieval_service import RetrievalService
from doc_ai.services.vector_store import InMemoryVectorStore

DOCUMENTS = {
    1: (
        "refund-policy.txt",
        None,
        [
            (
                "You can request a refund within 30 days of delivery. Refund requests "
                "made after 30 days are declined."
            ),
            (
                "Approved refunds are paid back to the original payment method within "
                "5 business days."
            ),
        ],
    ),
    2: (
        "shipping-policy.pdf",
        1,
        [
            (
                "Standard shipping takes 3 to 5 business days. Express shipping takes "
                "1 to 2 business days."
            ),
            "Shipping costs are not refunded when an order is returned.",
        ],
    ),
    3: (
        "account-cancellation.txt",
        None,
        [
            "To cancel your account, open Settings and choose Cancel account.",
            (
                "Cancellation takes effect at the end of the current billing period. "
                "Your data is deleted 30 days after cancellation."
            ),
        ],
    ),
    4: (
        "payment-methods.pdf",
        1,
        [
            "We accept Visa, Mastercard, American Express and PayPal.",
            "Bank transfers and cash on delivery are not accepted.",
        ],
    ),
}

CORPUS = [
    Chunk(
        id=f"{document_id}-{index}",
        document_id=document_id,
        chunk_index=index,
        content=content,
        filename=filename,
        page_number=page_number,
        start_index=0,
    )
    for document_id, (filename, page_number, contents) in DOCUMENTS.items()
    for index, content in enumerate(contents)
]

CASES = [
    EvaluationCase(
        question="How long do I have to request a refund?",
        relevant_chunk_ids={"1-0"},
    ),
    EvaluationCase(
        question="When will I get my money back?",
        relevant_chunk_ids={"1-1"},
    ),
    EvaluationCase(
        question="How long does shipping take?",
        relevant_chunk_ids={"2-0"},
    ),
    EvaluationCase(
        question="How can I cancel my account?",
        relevant_chunk_ids={"3-0"},
    ),
    # Paraphrase with no word in common with the relevant chunk.
    EvaluationCase(
        question="How do I close my profile?",
        relevant_chunk_ids={"3-0"},
    ),
    EvaluationCase(
        question="Which payment methods are accepted?",
        relevant_chunk_ids={"4-0", "4-1"},
    ),
    EvaluationCase(
        question="Is cash on delivery accepted?",
        relevant_chunk_ids={"4-1"},
    ),
    # Nothing in the corpus answers these.
    EvaluationCase(question="What are the store's opening hours?", answerable=False),
    EvaluationCase(
        question="Does the company offer international phone support?",
        answerable=False,
    ),
    # Shares "days" and "paid" with the refund chunks.
    EvaluationCase(
        question="How many days of paid vacation do employees get?",
        answerable=False,
    ),
]

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "have", "how", "i", "in", "is", "it", "my", "of", "on", "or", "the",
    "to", "we", "what", "when", "which", "will", "with", "you", "your",
}  # fmt: skip


class BagOfWordsEmbeddings(Embeddings):
    """Word counts hashed into a fixed-size vector.

    Similarity comes only from shared words (no stemming or synonyms), which is
    crude but deterministic, offline and, unlike DeterministicFakeEmbedding,
    related to the text's content.
    """

    def __init__(self, size: int = 4096):
        self.size = size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word not in STOP_WORDS:
                vector[zlib.crc32(word.encode()) % self.size] += 1.0
        return vector


async def build_retrieval_service(embeddings: Embeddings) -> RetrievalService:
    embedding_service = EmbeddingService(embeddings, model="evaluation-fake")
    vector_store = InMemoryVectorStore()
    await vector_store.add(CORPUS, await embedding_service.embed_chunks(CORPUS))

    return RetrievalService(
        embedding_service, vector_store, top_k=max(KS), max_question_length=200
    )
