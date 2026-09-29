"""Retrieval evaluation: compares retrieved chunk ids with the expected ones.

Kept out of the production package on purpose; it only consumes what
`RetrievalService.retrieve` returns.
"""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from doc_ai.interfaces.retrieval_service import RetrievalServiceInterface

KS = (1, 2, 4)


class EvaluationCase(BaseModel):
    """A question and the chunks that answer it.

    An unanswerable question (nothing in the corpus answers it) is marked with
    `answerable=False` and has no relevant chunk ids.
    """

    question: str
    answerable: bool = True
    relevant_chunk_ids: set[str] = Field(default_factory=set)

    @model_validator(mode="after")
    def check_relevant_chunk_ids(self) -> Self:
        if self.answerable and not self.relevant_chunk_ids:
            raise ValueError("An answerable case needs at least one relevant chunk id")

        if not self.answerable and self.relevant_chunk_ids:
            raise ValueError("An unanswerable case must not have relevant chunk ids")

        return self


class CaseResult(EvaluationCase):
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float]

    @model_validator(mode="after")
    def check_scores(self) -> Self:
        if len(self.retrieved_scores) != len(self.retrieved_chunk_ids):
            raise ValueError("Expected one score per retrieved chunk")

        return self


class MetricsAtK(BaseModel):
    k: int
    recall: float
    precision: float
    hit_rate: float


class EvaluationSummary(BaseModel):
    answerable_cases: int
    unanswerable_cases: int
    # Answerable cases only; empty when there are none.
    metrics_at_k: list[MetricsAtK]
    # Unanswerable cases only; None when there are none.
    empty_retrieval_rate: float | None
    # Lowest score of a relevant chunk retrieved for an answerable question, and
    # highest score retrieved for an unanswerable one. None when nothing counts.
    lowest_relevant_score: float | None
    highest_unanswerable_score: float | None


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


def calculate_empty_retrieval(retrieved_ids: list[str]) -> float:
    return 0.0 if retrieved_ids else 1.0


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
                **case.model_dump(),
                retrieved_chunk_ids=[result.chunk.id for result in search_results],
                retrieved_scores=[result.score for result in search_results],
            )
        )

    return results


def aggregate_results(
    results: list[CaseResult], ks: tuple[int, ...] | list[int] = KS
) -> EvaluationSummary:
    if not results:
        raise ValueError("Evaluation needs at least one case")

    answerable = [result for result in results if result.answerable]
    unanswerable = [result for result in results if not result.answerable]

    def average(metric, k: int) -> float:
        return sum(
            metric(result.retrieved_chunk_ids, result.relevant_chunk_ids, k)
            for result in answerable
        ) / len(answerable)

    relevant_scores = [
        score
        for result in answerable
        for chunk_id, score in zip(
            result.retrieved_chunk_ids, result.retrieved_scores, strict=True
        )
        if chunk_id in result.relevant_chunk_ids
    ]
    unanswerable_scores = [
        score for result in unanswerable for score in result.retrieved_scores
    ]

    metrics_at_k = []
    if answerable:
        metrics_at_k = [
            MetricsAtK(
                k=k,
                recall=average(calculate_recall_at_k, k),
                precision=average(calculate_precision_at_k, k),
                hit_rate=average(calculate_hit_rate_at_k, k),
            )
            for k in ks
        ]

    empty_retrieval_rate = None
    if unanswerable:
        empty_retrieval_rate = sum(
            calculate_empty_retrieval(result.retrieved_chunk_ids)
            for result in unanswerable
        ) / len(unanswerable)

    return EvaluationSummary(
        answerable_cases=len(answerable),
        unanswerable_cases=len(unanswerable),
        metrics_at_k=metrics_at_k,
        empty_retrieval_rate=empty_retrieval_rate,
        lowest_relevant_score=min(relevant_scores, default=None),
        highest_unanswerable_score=max(unanswerable_scores, default=None),
    )


def format_report(summary: EvaluationSummary) -> str:
    lines = [
        f"Answerable cases: {summary.answerable_cases}",
        f"Unanswerable cases: {summary.unanswerable_cases}",
        "",
        "Answerable questions",
        "  K  Recall  Precision  Hit rate",
    ]
    lines.extend(
        f"{m.k:>3}  {m.recall:>6.3f}  {m.precision:>9.3f}  {m.hit_rate:>8.3f}"
        for m in summary.metrics_at_k
    )
    lines += [
        "",
        "Unanswerable questions",
        f"  Empty retrieval rate: {_format_optional(summary.empty_retrieval_rate)}",
        "",
        "Scores",
        "  Lowest score of a retrieved relevant chunk: "
        + _format_optional(summary.lowest_relevant_score),
        "  Highest score retrieved for an unanswerable question: "
        + _format_optional(summary.highest_unanswerable_score),
    ]

    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
