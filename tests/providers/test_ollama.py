from typing import List

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from legion.errors import ProviderError
from legion.interface.schemas import Message, ModelResponse, ProviderConfig, Role
from legion.interface.tools import BaseTool
from legion.providers.ollama import OllamaFactory, OllamaProvider

# Load environment variables
load_dotenv()

MODEL = "llama3.2"

class TestSchema(BaseModel):
    name: str
    age: int
    hobbies: List[str]

class SimpleToolParams(BaseModel):
    message: str

class SimpleTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simple_tool",
            description="A simple test tool",
            parameters=SimpleToolParams
        )

    def run(self, message: str) -> str:
        """Implement the sync run method"""
        return f"Tool response: {message}"

    async def arun(self, message: str) -> str:
        """Implement the async run method"""
        return self.run(message)

@pytest.fixture
def provider():
    config = ProviderConfig(
        # api_key=os.getenv("OLLAMA_API_KEY", "test-key")
    )
    return OllamaProvider(config=config)

@pytest.fixture
def factory():
    return OllamaFactory()

def test_provider_initialization(provider):
    assert isinstance(provider, OllamaProvider)
    assert provider.client is not None

def test_factory_creation(factory):
    config = ProviderConfig(
        # api_key=os.getenv("OLLAMA_API_KEY", "test-key")
    )
    provider = factory.create_provider(config=config)
    assert isinstance(provider, OllamaProvider)

@pytest.mark.asyncio
async def test_basic_completion(provider):
    messages = [
        Message(role=Role.USER, content="Say 'Hello, World!'")
    ]
    response = await provider.complete(
        messages=messages,
        model=MODEL,
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.usage is not None
    assert response.tool_calls is None

@pytest.mark.asyncio
async def test_tool_completion(provider):
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools when appropriate."),
        Message(role=Role.USER, content="Use the simple tool to say hello")
    ]
    response = await provider.complete(
        messages=messages,
        model=MODEL,
        tools=[tool],
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "simple_tool"

@pytest.mark.asyncio
async def test_json_completion(provider):
    messages = [
        Message(
            role=Role.USER,
            content="Give me information about a person named John who is 25 and likes reading and gaming"
        )
    ]
    response = await provider.complete(
        messages=messages,
        model=MODEL,
        temperature=0,
        response_schema=TestSchema
    )
    assert isinstance(response, ModelResponse)
    data = TestSchema.model_validate_json(response.content)
    assert isinstance(data.name, str)
    assert isinstance(data.age, int)
    assert isinstance(data.hobbies, list)

@pytest.mark.asyncio
async def test_tool_and_json_completion(provider):
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools and returns structured data."),
        Message(role=Role.USER, content="Use the simple tool to say hello, then format the response as a person's info")
    ]
    response = await provider.complete(
        messages=messages,
        model=MODEL,
        tools=[tool],
        temperature=0,
        response_schema=TestSchema
    )
    assert isinstance(response, ModelResponse)
    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "simple_tool"

    # Verify JSON response
    data = TestSchema.model_validate_json(response.content)
    assert isinstance(data.name, str)
    assert isinstance(data.age, int)
    assert isinstance(data.hobbies, list)

@pytest.mark.asyncio
async def test_async_completion(provider):
    messages = [
        Message(role=Role.USER, content="Say 'Hello, World!'")
    ]
    response = await provider.acomplete(
        messages=messages,
        model=MODEL,
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_invalid_model(provider):
    messages = [Message(role=Role.USER, content="test")]

    with pytest.raises(ProviderError):
        await provider.complete(
            messages=messages,
            model="invalid-model"
        )

@pytest.mark.asyncio
async def test_system_message_handling(provider):
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant"),
        Message(role=Role.USER, content="Who are you?")
    ]
    response = await provider.complete(
        messages=messages,
        model=MODEL,
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    # Check that the response is a proper self-introduction
    assert any(word in response.content.lower() for word in ["assist", "help", "support"])

if __name__ == "__main__":
    pytest.main()
