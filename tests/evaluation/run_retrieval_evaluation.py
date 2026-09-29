"""Print retrieval metrics for the synthetic dataset.

Run with: uv run python tests/evaluation/run_retrieval_evaluation.py
"""

import asyncio

from retrieval_dataset import CASES, BagOfWordsEmbeddings, build_retrieval_service
from retrieval_metrics import KS, aggregate_results, format_report, run_evaluation


async def main() -> None:
    service = await build_retrieval_service(BagOfWordsEmbeddings())
    results = await run_evaluation(service, CASES)

    for result in results:
        mark = (
            "hit "
            if result.relevant_chunk_ids & set(result.retrieved_chunk_ids)
            else "miss"
        )
        print(
            f"{mark}  {result.question:<42} expected {sorted(result.relevant_chunk_ids)}"
            f"  retrieved {result.retrieved_chunk_ids}"
        )

    print()
    print(format_report(aggregate_results(results, KS)))


if __name__ == "__main__":
    asyncio.run(main())
