class Settings:
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024
        self.allowed_file_types = {".pdf", ".txt"}


settings = Settings()
