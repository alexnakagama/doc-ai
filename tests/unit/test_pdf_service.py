from pathlib import Path

import pymupdf
import pytest

from doc_ai.exceptions.pdf import InvalidPDFError
from doc_ai.services.pdf_service import PDFService

pytestmark = pytest.mark.anyio


def make_pdf(path: Path, page_texts: list[str]) -> None:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


async def test_extract_pages_returns_numbered_pages(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    make_pdf(path, ["first page", "second page"])

    pages = await PDFService().extract_pages(str(path))

    assert [page.page_number for page in pages] == [1, 2]
    assert [page.text.strip() for page in pages] == ["first page", "second page"]


async def test_extract_pages_raises_for_invalid_pdf(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(InvalidPDFError):
        await PDFService().extract_pages(str(path))
