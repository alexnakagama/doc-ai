"""Print retrieval metrics and scores for the synthetic dataset.

Run with: uv run python tests/evaluation/run_retrieval_evaluation.py
"""

import asyncio

from retrieval_dataset import CASES, BagOfWordsEmbeddings, build_retrieval_service
from retrieval_metrics import (
    KS,
    CaseResult,
    aggregate_results,
    format_report,
    run_evaluation,
)


def format_case(result: CaseResult) -> list[str]:
    if not result.answerable:
        label = "unanswerable"
    elif result.relevant_chunk_ids & set(result.retrieved_chunk_ids):
        label = "hit"
    else:
        label = "miss"

    lines = [f"[{label}] {result.question}"]
    lines.extend(
        f"    {chunk_id}  {score:.3f}"
        + ("  relevant" if chunk_id in result.relevant_chunk_ids else "")
        for chunk_id, score in zip(
            result.retrieved_chunk_ids, result.retrieved_scores, strict=True
        )
    )
    return lines


async def main() -> None:
    service = await build_retrieval_service(BagOfWordsEmbeddings())
    results = await run_evaluation(service, CASES)

    for result in results:
        print("\n".join(format_case(result)))

    print()
    print(format_report(aggregate_results(results, KS)))
    print()
    print("Scores come from a fake embedding; do not use them to pick a threshold.")


if __name__ == "__main__":
    asyncio.run(main())
