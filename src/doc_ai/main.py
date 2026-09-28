from fastapi import FastAPI

from doc_ai.exceptions.document import (
    DocumentNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from doc_ai.exceptions.handlers import (
    document_not_found_handler,
    empty_file_handler,
    file_too_large_handler,
    invalid_pdf_handler,
    unsupported_encoding_handler,
    unsupported_file_type_handler,
)
from doc_ai.exceptions.pdf import InvalidPDFError
from doc_ai.exceptions.text import UnsupportedEncodingError
from doc_ai.routers.documents import router as documents_router

app = FastAPI()

app.add_exception_handler(EmptyFileError, empty_file_handler)

app.add_exception_handler(InvalidPDFError, invalid_pdf_handler)

app.add_exception_handler(FileTooLargeError, file_too_large_handler)

app.add_exception_handler(UnsupportedFileTypeError, unsupported_file_type_handler)

app.add_exception_handler(UnsupportedEncodingError, unsupported_encoding_handler)

app.add_exception_handler(DocumentNotFoundError, document_not_found_handler)

app.include_router(documents_router)
