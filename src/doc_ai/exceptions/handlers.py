from fastapi import Request
from fastapi.responses import JSONResponse

from doc_ai.exceptions.document import EmptyFileError


async def empty_file_handler(request: Request, error: EmptyFileError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})
