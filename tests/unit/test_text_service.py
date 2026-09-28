from pathlib import Path

import pytest

from doc_ai.exceptions.text import UnsupportedEncodingError
from doc_ai.services.text_service import TextService

pytestmark = pytest.mark.anyio


async def read_with_detected_encoding(path: Path) -> str:
    service = TextService()
    encoding = await service.detect_encoding(str(path))
    return await service.extract_text(str(path), encoding)


async def test_reads_utf8_file(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_bytes("héllo wörld".encode("utf-8"))  # noqa: UP012

    assert await read_with_detected_encoding(path) == "héllo wörld"


async def test_reads_utf8_file_with_bom_without_the_bom(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_bytes("héllo".encode("utf-8-sig"))

    assert await read_with_detected_encoding(path) == "héllo"


async def test_reads_utf16_file_with_bom(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_bytes("héllo".encode("utf-16"))

    assert await read_with_detected_encoding(path) == "héllo"


async def test_reads_cp1252_file(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_bytes("café – 5€".encode("cp1252"))

    assert await read_with_detected_encoding(path) == "café – 5€"


async def test_undecodable_file_raises_unsupported_encoding(tmp_path: Path):
    path = tmp_path / "doc.txt"
    # Invalid as UTF-8 and undefined in cp1252.
    path.write_bytes(b"\x81\x8d\x8f\x90\x9d")

    with pytest.raises(UnsupportedEncodingError):
        await TextService().detect_encoding(str(path))
