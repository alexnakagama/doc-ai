import pymupdf

from doc_ai.exceptions.pdf import InvalidPDFError
from doc_ai.models.page import PageText


class PDFService:
    async def extract_pages(
        self,
        file_path: str,
    ) -> list[PageText]:
        try:
            document = pymupdf.open(file_path)

            pages = [
                PageText(page_number=number, text=page.get_text())
                for number, page in enumerate(document, start=1)
            ]

            document.close()

            return pages

        except pymupdf.FileDataError as error:
            raise InvalidPDFError("Invalid PDF file") from error
