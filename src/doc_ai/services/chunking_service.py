from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_ai.models.chunk import Chunk
from doc_ai.models.document import Document
from doc_ai.models.page import PageText


class ChunkingService:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    async def chunk(self, document: Document, pages: list[PageText]) -> list[Chunk]:
        chunks: list[Chunk] = []

        for page in pages:
            for split in self.splitter.create_documents([page.text]):
                chunk_index = len(chunks)
                chunks.append(
                    Chunk(
                        id=f"{document.id}-{chunk_index}",
                        document_id=document.id,
                        chunk_index=chunk_index,
                        content=split.page_content,
                        filename=document.filename,
                        page_number=page.page_number,
                        start_index=split.metadata["start_index"],
                    )
                )

        return chunks
