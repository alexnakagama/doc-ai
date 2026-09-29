from typing import Protocol

from doc_ai.models.prompt import Prompt


class LLMServiceInterface(Protocol):
    async def generate(self, prompt: Prompt) -> str: ...
