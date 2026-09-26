from doc_ai.models.document import Document


class DocumentService:
    def __init__(self):
        self.documents: list[Document] = []

    async def get_documents(self):
        return self.documents
