import pytest
from pydantic import ValidationError
from retrieval_metrics import (
    CaseResult,
    EvaluationCase,
    EvaluationSummary,
    MetricsAtK,
    aggregate_results,
    calculate_empty_retrieval,
    calculate_hit_rate_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    format_report,
)


def test_recall_is_one_when_every_relevant_chunk_is_retrieved():
    assert calculate_recall_at_k(["a", "b", "x"], {"a", "b"}, k=3) == 1.0


def test_recall_counts_only_the_relevant_chunks_retrieved():
    assert calculate_recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_is_zero_when_no_relevant_chunk_is_retrieved():
    assert calculate_recall_at_k(["x", "y"], {"a"}, k=2) == 0.0


def test_recall_ignores_relevant_chunks_below_k():
    assert calculate_recall_at_k(["x", "a"], {"a"}, k=1) == 0.0


def test_recall_with_several_relevant_chunks_grows_with_k():
    retrieved = ["a", "x", "b", "c"]
    relevant = {"a", "b", "c"}

    assert [calculate_recall_at_k(retrieved, relevant, k) for k in (1, 2, 4)] == [
        pytest.approx(1 / 3),
        pytest.approx(1 / 3),
        1.0,
    ]


def test_precision_with_mixed_results():
    assert calculate_precision_at_k(["a", "x", "b", "y"], {"a", "b"}, k=4) == 0.5


def test_precision_is_one_when_every_result_is_relevant():
    assert calculate_precision_at_k(["a", "b"], {"a", "b", "c"}, k=2) == 1.0


def test_precision_divides_by_the_results_returned_when_fewer_than_k():
    assert calculate_precision_at_k(["a", "x"], {"a"}, k=4) == 0.5


def test_precision_is_zero_without_results():
    assert calculate_precision_at_k([], {"a"}, k=4) == 0.0


def test_hit_rate_is_one_when_any_relevant_chunk_is_in_the_top_k():
    assert calculate_hit_rate_at_k(["x", "b"], {"a", "b"}, k=2) == 1.0


def test_hit_rate_is_zero_when_no_relevant_chunk_is_in_the_top_k():
    assert calculate_hit_rate_at_k(["x", "b"], {"b"}, k=1) == 0.0


def test_k_larger_than_the_results_uses_every_result():
    retrieved = ["a", "x"]
    relevant = {"a", "b"}

    assert calculate_recall_at_k(retrieved, relevant, k=10) == 0.5
    assert calculate_precision_at_k(retrieved, relevant, k=10) == 0.5
    assert calculate_hit_rate_at_k(retrieved, relevant, k=10) == 1.0


@pytest.mark.parametrize(
    "metric",
    [calculate_recall_at_k, calculate_precision_at_k, calculate_hit_rate_at_k],
)
def test_metrics_reject_k_below_one(metric):
    with pytest.raises(ValueError, match="k must be"):
        metric(["a"], {"a"}, k=0)


def test_recall_rejects_a_case_without_relevant_chunks():
    with pytest.raises(ValueError, match="relevant"):
        calculate_recall_at_k(["a"], set(), k=1)


def result(
    retrieved: list[str],
    relevant: set[str] | None = None,
    scores: list[float] | None = None,
) -> CaseResult:
    return CaseResult(
        question="q",
        answerable=bool(relevant),
        relevant_chunk_ids=relevant or set(),
        retrieved_chunk_ids=retrieved,
        retrieved_scores=scores or [0.5] * len(retrieved),
    )


def test_answerable_case_is_the_default():
    case = EvaluationCase(question="q", relevant_chunk_ids={"a"})

    assert case.answerable


def test_unanswerable_case_is_accepted_without_relevant_ids():
    case = EvaluationCase(question="q", answerable=False)

    assert case.relevant_chunk_ids == set()


def test_answerable_case_requires_relevant_ids():
    with pytest.raises(ValidationError, match="at least one relevant"):
        EvaluationCase(question="q", relevant_chunk_ids=set())


def test_unanswerable_case_rejects_relevant_ids():
    with pytest.raises(ValidationError, match="must not have relevant"):
        EvaluationCase(question="q", answerable=False, relevant_chunk_ids={"a"})


def test_case_result_needs_one_score_per_retrieved_chunk():
    with pytest.raises(ValidationError, match="one score per"):
        CaseResult(
            question="q",
            relevant_chunk_ids={"a"},
            retrieved_chunk_ids=["a", "b"],
            retrieved_scores=[0.9],
        )


def test_empty_retrieval_is_one_only_without_results():
    assert calculate_empty_retrieval([]) == 1.0
    assert calculate_empty_retrieval(["a"]) == 0.0


def test_aggregate_averages_each_metric_over_the_cases():
    results = [
        result(["a", "x"], {"a"}),
        result(["y", "b"], {"b", "c"}),
    ]

    assert aggregate_results(results, ks=[1, 2]).metrics_at_k == [
        MetricsAtK(k=1, recall=0.5, precision=0.5, hit_rate=0.5),
        MetricsAtK(k=2, recall=0.75, precision=0.5, hit_rate=1.0),
    ]


def test_unanswerable_question_with_nearest_neighbours_is_not_an_empty_retrieval():
    summary = aggregate_results([result(["a", "b"], scores=[0.4, 0.3])], ks=[1])

    assert summary.unanswerable_cases == 1
    assert summary.empty_retrieval_rate == 0.0
    assert summary.highest_unanswerable_score == 0.4


def test_mixed_dataset_keeps_unanswerable_cases_out_of_answerable_metrics():
    answerable = [
        result(["a", "x"], {"a"}, scores=[0.9, 0.3]),
        result(["y", "b"], {"b", "c"}, scores=[0.5, 0.2]),
    ]
    unanswerable = [
        result([]),
        result(["a", "b"], scores=[0.6, 0.1]),
    ]

    summary = aggregate_results(answerable + unanswerable, ks=[1, 2])

    assert summary == EvaluationSummary(
        answerable_cases=2,
        unanswerable_cases=2,
        metrics_at_k=aggregate_results(answerable, ks=[1, 2]).metrics_at_k,
        empty_retrieval_rate=0.5,
        lowest_relevant_score=0.2,
        highest_unanswerable_score=0.6,
    )


def test_summary_of_answerable_cases_only_has_no_unanswerable_figures():
    summary = aggregate_results([result(["x"], {"a"})], ks=[1])

    assert summary.unanswerable_cases == 0
    assert summary.empty_retrieval_rate is None
    assert summary.highest_unanswerable_score is None
    assert summary.lowest_relevant_score is None


def test_summary_of_unanswerable_cases_only_has_no_answerable_metrics():
    summary = aggregate_results([result([])], ks=[1])

    assert summary.answerable_cases == 0
    assert summary.metrics_at_k == []
    assert summary.empty_retrieval_rate == 1.0
    assert summary.highest_unanswerable_score is None


def test_aggregate_is_deterministic():
    results = [result(["a", "b", "x"], {"b"}), result(["x"])]

    assert aggregate_results(results, ks=[1, 2, 4]) == aggregate_results(
        results, ks=[1, 2, 4]
    )


def test_aggregate_rejects_an_empty_dataset():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_results([], ks=[1])


def test_report_separates_answerable_and_unanswerable_cases():
    report = format_report(
        EvaluationSummary(
            answerable_cases=2,
            unanswerable_cases=1,
            metrics_at_k=[
                MetricsAtK(k=1, recall=0.5, precision=0.5, hit_rate=0.5),
                MetricsAtK(k=4, recall=1.0, precision=0.25, hit_rate=1.0),
            ],
            empty_retrieval_rate=0.0,
            lowest_relevant_score=0.25,
            highest_unanswerable_score=0.375,
        )
    )

    assert report.splitlines() == [
        "Answerable cases: 2",
        "Unanswerable cases: 1",
        "",
        "Answerable questions",
        "  K  Recall  Precision  Hit rate",
        "  1   0.500      0.500     0.500",
        "  4   1.000      0.250     1.000",
        "",
        "Unanswerable questions",
        "  Empty retrieval rate: 0.000",
        "",
        "Scores",
        "  Lowest score of a retrieved relevant chunk: 0.250",
        "  Highest score retrieved for an unanswerable question: 0.375",
    ]


def test_report_shows_missing_figures_as_not_available():
    report = format_report(
        EvaluationSummary(
            answerable_cases=0,
            unanswerable_cases=1,
            metrics_at_k=[],
            empty_retrieval_rate=1.0,
            lowest_relevant_score=None,
            highest_unanswerable_score=None,
        )
    )

    assert "  Lowest score of a retrieved relevant chunk: n/a" in report.splitlines()
    assert (
        "  Highest score retrieved for an unanswerable question: n/a"
        in report.splitlines()
    )
