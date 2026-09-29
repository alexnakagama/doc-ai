import os


class Settings:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024
        self.allowed_file_types = {".pdf", ".txt"}
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "4"))
        self.max_question_length = 2000
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_timeout_seconds = 60

        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0")

        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE")

        if self.retrieval_top_k <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be greater than 0")


settings = Settings()
