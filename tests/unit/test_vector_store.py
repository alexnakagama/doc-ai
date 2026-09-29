import math

import pytest

from doc_ai.models.chunk import Chunk
from doc_ai.models.embedding import ChunkEmbedding
from doc_ai.services.vector_store import InMemoryVectorStore

pytestmark = pytest.mark.anyio


def make_chunk(document_id: int, chunk_index: int) -> Chunk:
    return Chunk(
        id=f"{document_id}-{chunk_index}",
        document_id=document_id,
        chunk_index=chunk_index,
        content=f"chunk {chunk_index} of document {document_id}",
        filename=f"doc{document_id}.txt",
        page_number=None,
        start_index=0,
    )


def make_embedding(
    chunk: Chunk, vector: list[float], model: str = "fake-model"
) -> ChunkEmbedding:
    return ChunkEmbedding(chunk_id=chunk.id, vector=vector, model=model)


async def add_vectors(
    store: InMemoryVectorStore, document_id: int, vectors: list[list[float]]
) -> list[Chunk]:
    chunks = [make_chunk(document_id, index) for index in range(len(vectors))]
    await store.add(
        chunks,
        [make_embedding(chunk, vector) for chunk, vector in zip(chunks, vectors)],
    )
    return chunks


async def test_search_on_empty_store_returns_nothing():
    assert await InMemoryVectorStore().search([1.0, 0.0], k=3) == []


async def test_search_ranks_chunks_by_cosine_similarity():
    store = InMemoryVectorStore()
    far, near, middle = await add_vectors(
        store, 1, [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )

    results = await store.search([1.0, 0.0], k=3)

    assert [result.chunk for result in results] == [near, middle, far]
    assert [result.score for result in results] == pytest.approx(
        [1.0, 1 / math.sqrt(2), 0.0], abs=1e-6
    )


async def test_scores_do_not_depend_on_vector_length():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[10.0, 0.0]])

    results = await store.search([0.5, 0.0], k=1)

    assert results[0].score == pytest.approx(1.0, abs=1e-6)


async def test_opposite_vectors_score_minus_one():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[-1.0, 0.0]])

    results = await store.search([1.0, 0.0], k=1)

    assert results[0].score == pytest.approx(-1.0, abs=1e-6)


async def test_search_returns_at_most_k_results():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    assert len(await store.search([1.0, 0.0], k=2)) == 2


async def test_search_with_k_larger_than_store_returns_everything():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0], [0.0, 1.0]])

    assert len(await store.search([1.0, 0.0], k=10)) == 2


async def test_equal_scores_keep_insertion_order():
    store = InMemoryVectorStore()
    first = await add_vectors(store, 1, [[1.0, 0.0], [2.0, 0.0]])
    second = await add_vectors(store, 2, [[3.0, 0.0]])

    results = await store.search([1.0, 0.0], k=3)

    assert [result.chunk for result in results] == first + second


async def test_search_can_be_limited_to_one_document():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0]])
    second = await add_vectors(store, 2, [[0.0, 1.0], [1.0, 1.0]])

    results = await store.search([1.0, 0.0], k=5, document_id=2)

    assert [result.chunk for result in results] == [second[1], second[0]]


async def test_search_for_unknown_document_returns_nothing():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0]])

    assert await store.search([1.0, 0.0], k=5, document_id=99) == []


async def test_adding_an_empty_batch_is_a_no_op():
    store = InMemoryVectorStore()

    await store.add([], [])

    assert await store.search([1.0, 0.0], k=1) == []


@pytest.mark.parametrize("k", [0, -1])
async def test_non_positive_k_is_rejected(k: int):
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0]])

    with pytest.raises(ValueError, match="k"):
        await store.search([1.0, 0.0], k=k)


async def test_query_with_wrong_dimension_is_rejected():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0]])

    with pytest.raises(ValueError, match="dimension"):
        await store.search([1.0, 0.0, 0.0], k=1)


@pytest.mark.parametrize(
    "query", [[0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]], ids=str
)
async def test_invalid_query_vector_is_rejected(query: list[float]):
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0]])

    with pytest.raises(ValueError):
        await store.search(query, k=1)


async def test_mismatched_batch_lengths_are_rejected():
    chunks = [make_chunk(1, 0), make_chunk(1, 1)]

    with pytest.raises(ValueError, match="embedding"):
        await InMemoryVectorStore().add(chunks, [make_embedding(chunks[0], [1.0, 0.0])])


async def test_embeddings_must_match_chunks_in_order():
    chunks = [make_chunk(1, 0), make_chunk(1, 1)]
    embeddings = [
        make_embedding(chunks[1], [1.0, 0.0]),
        make_embedding(chunks[0], [0.0, 1.0]),
    ]

    with pytest.raises(ValueError, match="chunk"):
        await InMemoryVectorStore().add(chunks, embeddings)


async def test_duplicate_chunk_ids_in_one_batch_are_rejected():
    chunk = make_chunk(1, 0)

    with pytest.raises(ValueError, match="already"):
        await InMemoryVectorStore().add(
            [chunk, chunk],
            [make_embedding(chunk, [1.0, 0.0]), make_embedding(chunk, [0.0, 1.0])],
        )


async def test_adding_a_stored_chunk_again_is_rejected():
    store = InMemoryVectorStore()
    [chunk] = await add_vectors(store, 1, [[1.0, 0.0]])

    with pytest.raises(ValueError, match="already"):
        await store.add([chunk], [make_embedding(chunk, [0.0, 1.0])])


async def test_vectors_with_different_dimensions_in_one_batch_are_rejected():
    chunks = [make_chunk(1, 0), make_chunk(1, 1)]

    with pytest.raises(ValueError, match="dimension"):
        await InMemoryVectorStore().add(
            chunks,
            [
                make_embedding(chunks[0], [1.0, 0.0]),
                make_embedding(chunks[1], [1.0, 0.0, 0.0]),
            ],
        )


async def test_vectors_must_match_the_stored_dimension():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0]])
    chunk = make_chunk(2, 0)

    with pytest.raises(ValueError, match="dimension"):
        await store.add([chunk], [make_embedding(chunk, [1.0, 0.0, 0.0])])


async def test_embeddings_from_a_different_model_are_rejected():
    store = InMemoryVectorStore()
    await add_vectors(store, 1, [[1.0, 0.0]])
    chunk = make_chunk(2, 0)

    with pytest.raises(ValueError, match="model"):
        await store.add([chunk], [make_embedding(chunk, [1.0, 0.0], "other-model")])


@pytest.mark.parametrize(
    "vector", [[], [0.0, 0.0], [math.nan, 1.0], [math.inf, 1.0]], ids=str
)
async def test_invalid_stored_vectors_are_rejected(vector: list[float]):
    chunk = make_chunk(1, 0)

    with pytest.raises(ValueError):
        await InMemoryVectorStore().add([chunk], [make_embedding(chunk, vector)])


async def test_rejected_batch_leaves_the_store_unchanged():
    store = InMemoryVectorStore()
    stored = await add_vectors(store, 1, [[1.0, 0.0]])
    chunks = [make_chunk(2, 0), make_chunk(2, 1)]

    with pytest.raises(ValueError):
        await store.add(
            chunks,
            [
                make_embedding(chunks[0], [1.0, 0.0]),
                make_embedding(chunks[1], [0.0, 0.0]),
            ],
        )

    results = await store.search([1.0, 0.0], k=10)
    assert [result.chunk for result in results] == stored
    chunk = make_chunk(2, 0)
    await store.add([chunk], [make_embedding(chunk, [0.0, 1.0])])


async def test_first_rejected_batch_does_not_fix_dimension_or_model():
    store = InMemoryVectorStore()
    chunk = make_chunk(1, 0)

    with pytest.raises(ValueError):
        await store.add([chunk], [make_embedding(chunk, [0.0, 0.0, 0.0], "old")])

    await store.add([chunk], [make_embedding(chunk, [1.0, 0.0])])
    assert len(await store.search([1.0, 0.0], k=1)) == 1
