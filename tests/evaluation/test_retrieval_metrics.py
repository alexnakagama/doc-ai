import pytest
from retrieval_metrics import (
    CaseResult,
    MetricsAtK,
    aggregate_results,
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


def test_aggregate_averages_each_metric_over_the_cases():
    results = [
        CaseResult(
            question="q1", retrieved_chunk_ids=["a", "x"], relevant_chunk_ids={"a"}
        ),
        CaseResult(
            question="q2", retrieved_chunk_ids=["y", "b"], relevant_chunk_ids={"b", "c"}
        ),
    ]

    assert aggregate_results(results, ks=[1, 2]) == [
        MetricsAtK(k=1, recall=0.5, precision=0.5, hit_rate=0.5),
        MetricsAtK(k=2, recall=0.75, precision=0.5, hit_rate=1.0),
    ]


def test_aggregate_is_deterministic():
    results = [
        CaseResult(
            question="q", retrieved_chunk_ids=["a", "b", "x"], relevant_chunk_ids={"b"}
        )
    ]

    assert aggregate_results(results, ks=[1, 2, 4]) == aggregate_results(
        results, ks=[1, 2, 4]
    )


def test_aggregate_rejects_an_empty_dataset():
    with pytest.raises(ValueError, match="at least one"):
        aggregate_results([], ks=[1])


def test_report_lists_every_metric_for_every_k():
    report = format_report(
        [
            MetricsAtK(k=1, recall=0.5, precision=0.5, hit_rate=0.5),
            MetricsAtK(k=4, recall=1.0, precision=0.25, hit_rate=1.0),
        ]
    )

    assert report.splitlines() == [
        "  K  Recall  Precision  Hit rate",
        "  1   0.500      0.500     0.500",
        "  4   1.000      0.250     1.000",
    ]
