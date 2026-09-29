from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from doc_ai.core.config import Settings
from doc_ai.exceptions.llm import LLMError
from doc_ai.models.prompt import Prompt


class LLMService:
    def __init__(self, chat_model: BaseChatModel):
        self.chat_model = chat_model

    async def generate(self, prompt: Prompt) -> str:
        messages = [SystemMessage(prompt.system), HumanMessage(prompt.user)]

        try:
            response = await self.chat_model.ainvoke(messages)
        except Exception as error:
            raise LLMError("Language model provider failed") from error

        text = response.text.strip()

        if not text:
            raise LLMError("Language model returned an empty answer")

        return text


def create_openai_llm_service(settings: Settings) -> LLMService:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    chat_model = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        timeout=settings.llm_timeout_seconds,
    )

    return LLMService(chat_model)
