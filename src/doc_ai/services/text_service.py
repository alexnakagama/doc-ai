import codecs
from pathlib import Path

from doc_ai.exceptions.text import UnsupportedEncodingError


class TextService:
    async def detect_encoding(self, file_path: str) -> str:
        content = Path(file_path).read_bytes()

        if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            candidates = ["utf-16"]
        else:
            candidates = ["utf-8-sig", "cp1252"]

        for encoding in candidates:
            if self._decodes(content, encoding):
                return encoding

        raise UnsupportedEncodingError("Could not detect the file encoding")

    async def validate_encoding(self, file_path: str, encoding: str) -> bool:
        return self._decodes(Path(file_path).read_bytes(), encoding)

    async def extract_text(self, file_path: str, encoding: str = "utf-8") -> str:
        return Path(file_path).read_text(encoding=encoding)

    def _decodes(self, content: bytes, encoding: str) -> bool:
        try:
            content.decode(encoding)
            return True
        except UnicodeDecodeError:
            return False
