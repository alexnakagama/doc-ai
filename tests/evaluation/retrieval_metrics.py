"""Retrieval evaluation: compares retrieved chunk ids with the expected ones.

Kept out of the production package on purpose; it only consumes what
`RetrievalService.retrieve` returns.
"""

from pydantic import BaseModel, Field

from doc_ai.interfaces.retrieval_service import RetrievalServiceInterface

KS = (1, 2, 4)


class EvaluationCase(BaseModel):
    question: str
    relevant_chunk_ids: set[str] = Field(min_length=1)


class CaseResult(BaseModel):
    question: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: set[str]


class MetricsAtK(BaseModel):
    k: int
    recall: float
    precision: float
    hit_rate: float


def calculate_recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    if not relevant_ids:
        raise ValueError("A case needs at least one relevant chunk id")

    return _relevant_in_top_k(retrieved_ids, relevant_ids, k) / len(relevant_ids)


def calculate_precision_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    returned = min(k, len(retrieved_ids))
    hits = _relevant_in_top_k(retrieved_ids, relevant_ids, k)

    return hits / returned if returned else 0.0


def calculate_hit_rate_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    return 1.0 if _relevant_in_top_k(retrieved_ids, relevant_ids, k) else 0.0


def _relevant_in_top_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> int:
    if k < 1:
        raise ValueError("k must be at least 1")

    return len(set(retrieved_ids[:k]) & relevant_ids)


async def run_evaluation(
    retrieval_service: RetrievalServiceInterface, cases: list[EvaluationCase]
) -> list[CaseResult]:
    results = []

    for case in cases:
        search_results = await retrieval_service.retrieve(case.question)
        results.append(
            CaseResult(
                question=case.question,
                retrieved_chunk_ids=[result.chunk.id for result in search_results],
                relevant_chunk_ids=case.relevant_chunk_ids,
            )
        )

    return results


def aggregate_results(
    results: list[CaseResult], ks: tuple[int, ...] | list[int] = KS
) -> list[MetricsAtK]:
    if not results:
        raise ValueError("Evaluation needs at least one case")

    def average(metric, k: int) -> float:
        return sum(
            metric(result.retrieved_chunk_ids, result.relevant_chunk_ids, k)
            for result in results
        ) / len(results)

    return [
        MetricsAtK(
            k=k,
            recall=average(calculate_recall_at_k, k),
            precision=average(calculate_precision_at_k, k),
            hit_rate=average(calculate_hit_rate_at_k, k),
        )
        for k in ks
    ]


def format_report(metrics: list[MetricsAtK]) -> str:
    lines = ["  K  Recall  Precision  Hit rate"]
    lines.extend(
        f"{m.k:>3}  {m.recall:>6.3f}  {m.precision:>9.3f}  {m.hit_rate:>8.3f}"
        for m in metrics
    )

    return "\n".join(lines)
