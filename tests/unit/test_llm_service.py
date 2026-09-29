import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from doc_ai.core.config import Settings
from doc_ai.exceptions.llm import LLMError
from doc_ai.models.prompt import Prompt
from doc_ai.services.llm_service import LLMService, create_openai_llm_service

pytestmark = pytest.mark.anyio

PROMPT = Prompt(system="Follow the rules.", user="What is the answer?")


class RecordingChatModel(BaseChatModel):
    received: list[BaseMessage] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received = list(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage("ok"))])


class FailingChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "failing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("insufficient_quota")


async def test_returns_the_model_reply():
    service = LLMService(FakeListChatModel(responses=["The answer."]))

    assert await service.generate(PROMPT) == "The answer."


async def test_sends_system_then_user_message():
    chat_model = RecordingChatModel()

    await LLMService(chat_model).generate(PROMPT)

    assert [(message.type, message.content) for message in chat_model.received] == [
        ("system", "Follow the rules."),
        ("human", "What is the answer?"),
    ]


async def test_reply_is_stripped():
    service = LLMService(FakeListChatModel(responses=["  The answer.\n"]))

    assert await service.generate(PROMPT) == "The answer."


async def test_provider_error_raises_llm_error_with_cause():
    with pytest.raises(LLMError) as error:
        await LLMService(FailingChatModel()).generate(PROMPT)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert "insufficient_quota" not in str(error.value)


@pytest.mark.parametrize("reply", ["", "  \n"], ids=repr)
async def test_empty_reply_raises_llm_error(reply: str):
    with pytest.raises(LLMError):
        await LLMService(FakeListChatModel(responses=[reply])).generate(PROMPT)


def test_openai_factory_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_openai_llm_service(Settings())


def test_openai_factory_builds_service_with_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    service = create_openai_llm_service(Settings())

    assert isinstance(service, LLMService)
