import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from doc_ai.exceptions.document import (
    DocumentNotFoundError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from doc_ai.exceptions.embedding import EmbeddingError
from doc_ai.exceptions.pdf import InvalidPDFError
from doc_ai.exceptions.text import UnsupportedEncodingError

logger = logging.getLogger(__name__)


async def empty_file_handler(request: Request, error: EmptyFileError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


async def invalid_pdf_handler(request: Request, error: InvalidPDFError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


async def file_too_large_handler(
    request: Request, error: FileTooLargeError
) -> JSONResponse:
    return JSONResponse(status_code=413, content={"detail": str(error)})


async def unsupported_file_type_handler(
    request: Request, error: UnsupportedFileTypeError
) -> JSONResponse:
    return JSONResponse(status_code=415, content={"detail": str(error)})


async def unsupported_encoding_handler(
    request: Request, error: UnsupportedEncodingError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


async def document_not_found_handler(
    request: Request, error: DocumentNotFoundError
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(error)})


async def embedding_error_handler(
    request: Request, error: EmbeddingError
) -> JSONResponse:
    logger.error("Embedding request failed", exc_info=error)
    return JSONResponse(status_code=502, content={"detail": str(error)})
