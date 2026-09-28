from datetime import UTC, datetime
from itertools import pairwise

import pytest

from doc_ai.models.document import Document
from doc_ai.models.page import PageText
from doc_ai.services.chunking_service import ChunkingService

pytestmark = pytest.mark.anyio

LONG_TEXT = " ".join(f"word{i}" for i in range(60))


def make_document(document_id: int = 7) -> Document:
    return Document(
        id=document_id,
        filename="report.pdf",
        content="",
        created_at=datetime.now(UTC),
    )


async def test_long_text_splits_into_overlapping_chunks():
    service = ChunkingService(chunk_size=50, chunk_overlap=20)

    chunks = await service.chunk(
        make_document(), [PageText(page_number=None, text=LONG_TEXT)]
    )

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 50 for chunk in chunks)
    for previous, current in pairwise(chunks):
        assert current.start_index < previous.start_index + len(previous.content)


async def test_start_index_points_at_chunk_text_in_its_page():
    service = ChunkingService(chunk_size=50, chunk_overlap=20)

    chunks = await service.chunk(
        make_document(), [PageText(page_number=None, text=LONG_TEXT)]
    )

    assert chunks
    for chunk in chunks:
        start = chunk.start_index
        assert LONG_TEXT[start : start + len(chunk.content)] == chunk.content


async def test_chunk_index_continues_across_pages_and_keeps_page_numbers():
    service = ChunkingService(chunk_size=50, chunk_overlap=20)
    pages = [
        PageText(page_number=1, text=LONG_TEXT),
        PageText(page_number=2, text=LONG_TEXT),
    ]

    chunks = await service.chunk(make_document(), pages)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    page_numbers = [chunk.page_number for chunk in chunks]
    assert page_numbers == sorted(page_numbers)
    assert set(page_numbers) == {1, 2}


async def test_chunks_carry_document_metadata_and_stable_ids():
    service = ChunkingService(chunk_size=50, chunk_overlap=20)

    chunks = await service.chunk(
        make_document(document_id=7), [PageText(page_number=3, text="short text")]
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == "7-0"
    assert chunk.document_id == 7
    assert chunk.filename == "report.pdf"
    assert chunk.page_number == 3
    assert chunk.start_index == 0
    assert chunk.content == "short text"


async def test_blank_pages_produce_no_chunks():
    service = ChunkingService(chunk_size=50, chunk_overlap=20)
    pages = [
        PageText(page_number=1, text="   \n  "),
        PageText(page_number=2, text="real text"),
    ]

    chunks = await service.chunk(make_document(), pages)

    assert [(chunk.page_number, chunk.chunk_index) for chunk in chunks] == [(2, 0)]
