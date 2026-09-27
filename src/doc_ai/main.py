from fastapi import FastAPI

from doc_ai.exceptions.document import EmptyFileError
from doc_ai.exceptions.handlers import empty_file_handler, invalid_pdf_handler
from doc_ai.exceptions.pdf import InvalidPDFError
from doc_ai.routers.documents import router as documents_router

app = FastAPI()

app.add_exception_handler(EmptyFileError, empty_file_handler)

app.add_exception_handler(InvalidPDFError, invalid_pdf_handler)

app.include_router(documents_router)
