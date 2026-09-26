from fastapi import FastAPI

from doc_ai.routers.documents import router as documents_router

app = FastAPI()

app.include_router(documents_router)
