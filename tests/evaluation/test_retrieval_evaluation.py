import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from retrieval_dataset import (
    CASES,
    CORPUS,
    BagOfWordsEmbeddings,
    build_retrieval_service,
)
from retrieval_metrics import KS, aggregate_results, run_evaluation

pytestmark = pytest.mark.anyio


def test_every_expected_chunk_id_exists_in_the_corpus():
    corpus_ids = {chunk.id for chunk in CORPUS}

    for case in CASES:
        assert case.relevant_chunk_ids <= corpus_ids, case.question


def test_corpus_chunk_ids_and_questions_are_unique():
    assert len({chunk.id for chunk in CORPUS}) == len(CORPUS)
    assert len({case.question for case in CASES}) == len(CASES)


def test_dataset_has_answerable_and_unanswerable_questions():
    assert sum(case.answerable for case in CASES) == 7
    assert [case.question for case in CASES if not case.answerable] == [
        "What are the store's opening hours?",
        "Does the company offer international phone support?",
        "How many days of paid vacation do employees get?",
    ]


def test_bag_of_words_embeddings_score_shared_words_only():
    embeddings = BagOfWordsEmbeddings()

    refund, shipping = embeddings.embed_documents(
        ["Request a refund.", "Shipping is free."]
    )
    query = embeddings.embed_query("How do I request a refund?")

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, refund) > 0
    assert dot(query, shipping) == 0


async def test_synthetic_dataset_produces_the_expected_metrics():
    service = await build_retrieval_service(BagOfWordsEmbeddings())

    results = await run_evaluation(service, CASES)
    summary = aggregate_results(results, KS)

    # Misses: "close my profile" shares no word with any chunk, and 4-0 says
    # "accept", not "accepted", so it falls outside the top 4 for its question.
    assert [result.retrieved_chunk_ids[0] for result in results] == [
        "1-0",
        "1-1",
        "2-0",
        "3-0",
        "1-0",
        "4-1",
        "4-1",
        "1-0",
        "1-0",
        "1-1",
    ]
    assert summary.answerable_cases == 7
    assert summary.unanswerable_cases == 3
    assert [metrics.model_dump() for metrics in summary.metrics_at_k] == [
        pytest.approx(
            {"k": 1, "recall": 5.5 / 7, "precision": 6 / 7, "hit_rate": 6 / 7}
        ),
        pytest.approx(
            {"k": 2, "recall": 5.5 / 7, "precision": 3 / 7, "hit_rate": 6 / 7}
        ),
        pytest.approx(
            {"k": 4, "recall": 5.5 / 7, "precision": 1.5 / 7, "hit_rate": 6 / 7}
        ),
    ]
    # Vector search always returns its nearest neighbours.
    assert summary.empty_retrieval_rate == 0.0
    # Lowest relevant: 1-1 for "money back" (1 shared word of 3 and 11).
    # Highest unanswerable: 1-1 for "paid vacation" (shares "paid" and "days"),
    # so with this fake no score cut-off separates the two groups.
    assert summary.lowest_relevant_score == pytest.approx(1 / (11 * 3) ** 0.5, abs=1e-6)
    assert summary.highest_unanswerable_score == pytest.approx(
        2 / (11 * 6) ** 0.5, abs=1e-6
    )


async def test_unanswerable_questions_without_shared_words_score_zero():
    service = await build_retrieval_service(BagOfWordsEmbeddings())
    cases = [case for case in CASES if not case.answerable][:2]

    for result in await run_evaluation(service, cases):
        assert result.relevant_chunk_ids == set()
        assert result.retrieved_scores == [0.0] * max(KS)


async def test_evaluation_runs_on_the_production_retrieval_components():
    # DeterministicFakeEmbedding vectors carry no meaning, so this only checks
    # that the evaluation runs end to end and is repeatable, not the scores.
    first = await run_evaluation(
        await build_retrieval_service(DeterministicFakeEmbedding(size=16)), CASES
    )
    second = await run_evaluation(
        await build_retrieval_service(DeterministicFakeEmbedding(size=16)), CASES
    )

    corpus_ids = {chunk.id for chunk in CORPUS}
    assert first == second
    for result in first:
        assert len(result.retrieved_chunk_ids) == max(KS)
        assert set(result.retrieved_chunk_ids) <= corpus_ids
        assert result.retrieved_scores == sorted(result.retrieved_scores, reverse=True)

    summary = aggregate_results(first, KS)
    assert summary.answerable_cases == 7
    assert summary.unanswerable_cases == 3
    assert summary.empty_retrieval_rate == 0.0
    for metrics in summary.metrics_at_k:
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.hit_rate <= 1.0
