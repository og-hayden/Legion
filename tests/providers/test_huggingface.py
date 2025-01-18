import os
from typing import List

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from legion.errors import ProviderError
from legion.interface.schemas import Message, ModelResponse, ProviderConfig, Role
from legion.interface.tools import BaseTool
from legion.providers.huggingface import HuggingFaceFactory, HuggingFaceProvider

# Load environment variables
load_dotenv()

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
    """Create a HuggingFace provider for testing"""
    config = ProviderConfig(
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        api_secret=None,
        organization_id=None
    )
    return HuggingFaceProvider(config=config, debug=True)  # Enable debug mode to see responses

@pytest.fixture
def factory():
    return HuggingFaceFactory()

def test_provider_initialization(provider):
    assert isinstance(provider, HuggingFaceProvider)
    assert provider.client is not None

def test_factory_creation(factory):
    config = ProviderConfig(
        api_key=os.getenv("HUGGINGFACE_API_KEY", "test-key")
    )
    provider = factory.create_provider(config=config)
    assert isinstance(provider, HuggingFaceProvider)

@pytest.mark.asyncio
async def test_basic_completion(provider):
    messages = [
        Message(role=Role.USER, content="Say 'Hello, World!'")
    ]
    response = await provider.complete(
        messages=messages,
        model="google/gemma-2-2b-it",
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert response.usage is not None
    assert response.tool_calls is None

@pytest.mark.asyncio
async def test_tool_completion(provider):
    import asyncio
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools when appropriate. When asked to use a tool, respond with a tool call in this format: {\"name\": \"tool_name\", \"arguments\": {\"message\": \"hello world\"}}"),
        Message(role=Role.USER, content="Please use the simple_tool to say hello. Respond with a tool call only.")
    ]
    try:
        response = await asyncio.wait_for(
            provider.complete(
                messages=messages,
                model="google/gemma-2-2b-it",
                tools=[tool],
                temperature=0
            ),
            timeout=30.0
        )
        print("\nModel response in test_tool_completion:")
        print(f"Content: {response.content}")
        print(f"Raw response: {response.raw_response}")
        assert isinstance(response, ModelResponse)
        assert response.tool_calls is not None
    except asyncio.TimeoutError:
        print("\nTest timed out after 30 seconds")
        pytest.skip("Test timed out")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_json_completion(provider):
    import asyncio
    messages = [
        Message(
            role=Role.USER,
            content="Give me information about a person named John who is 25 and likes reading and gaming"
        )
    ]
    try:
        response = await asyncio.wait_for(
            provider.complete(
                messages=messages,
                model="google/gemma-2-2b-it",
                temperature=0,
                response_schema=TestSchema
            ),
            timeout=30.0
        )
        assert isinstance(response, ModelResponse)
        data = TestSchema.model_validate_json(response.content)
        assert isinstance(data.name, str)
        assert isinstance(data.age, int)
        assert isinstance(data.hobbies, list)
    except asyncio.TimeoutError:
        print("\nTest timed out after 30 seconds")
        pytest.skip("Test timed out")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_tool_and_json_completion(provider):
    import asyncio
    tool = SimpleTool()
    messages = [
        Message(role=Role.SYSTEM, content="You are a helpful assistant that uses tools and returns structured data. When asked to use a tool, respond with a tool call in this format: {\"name\": \"tool_name\", \"arguments\": {\"message\": \"hello world\"}}"),
        Message(role=Role.USER, content="First use the simple_tool to say hello (respond with a tool call), then after getting the tool response, format it as a person's info")
    ]
    try:
        # Set a 30 second timeout
        response = await asyncio.wait_for(
            provider.complete(
                messages=messages,
                model="google/gemma-2-2b-it",
                tools=[tool],
                temperature=0,
                response_schema=TestSchema
            ),
            timeout=30.0
        )
        print("\nModel response in test_tool_and_json_completion:")
        print(f"Content: {response.content}")
        print(f"Raw response: {response.raw_response}")
        assert isinstance(response, ModelResponse)
        assert response.tool_calls is not None
    except asyncio.TimeoutError:
        print("\nTest timed out after 30 seconds")
        pytest.skip("Test timed out")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_async_completion(provider):
    messages = [
        Message(role=Role.USER, content="Say 'Hello, World!'")
    ]
    response = await provider.acomplete(
        messages=messages,
        model="google/gemma-2-2b-it",
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    assert isinstance(response.content, str)
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_invalid_api_key():
    config = ProviderConfig(api_key="invalid_key")
    provider = HuggingFaceProvider(config=config)
    messages = [Message(role=Role.USER, content="test")]

    with pytest.raises(ProviderError):
        await provider.complete(
            messages=messages,
            model="google/gemma-2-2b-it"
        )

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
        model="google/gemma-2-2b-it",
        temperature=0
    )
    assert isinstance(response, ModelResponse)
    # Check that the response is a proper self-introduction
    assert any(word in response.content.lower() for word in ["assist", "help", "support"])

if __name__ == "__main__":
    pytest.main()
