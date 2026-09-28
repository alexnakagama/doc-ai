from pathlib import Path


class TextService:
    async def detect_encoding(self, file_path: str) -> str:
        return "utf-8"

    async def validate_encoding(self, file_path: str, encoding: str) -> bool:
        try:
            Path(file_path).read_text(encoding=encoding)
            return True
        except UnicodeDecodeError:
            return False

    async def extract_text(self, file_path: str, encoding: str = "utf-8") -> str:
        return Path(file_path).read_text(encoding=encoding)
