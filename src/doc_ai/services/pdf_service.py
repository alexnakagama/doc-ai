import pymupdf

from doc_ai.exceptions.pdf import InvalidPDFError


class PDFService:
    async def extract_text(
        self,
        file_path: str,
    ) -> str:
        try:
            document = pymupdf.open(file_path)

            text = ""

            for page in document:
                text += page.get_text()

            document.close()

            return text

        except pymupdf.FileDataError as error:
            raise InvalidPDFError("Invalid PDF file") from error
