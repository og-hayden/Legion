"""Ollama-specific implementation of the LLM interface"""

import json
from typing import Any, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
from ..interface.schemas import (
    ChatParameters,
    Message,
    ModelResponse,
    ProviderConfig,
    Role,
    TokenUsage,
)
from ..interface.tools import BaseTool
from . import ProviderFactory


class OllamaFactory(ProviderFactory):
    """Factory for creating Ollama providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new Ollama provider instance"""
        return OllamaProvider(config=config, **kwargs)

class OllamaProvider(LLMInterface):
    """Ollama-specific provider implementation"""

    def __init__(self, config: ProviderConfig, debug: bool = False):
        """Initialize provider with both sync and async clients"""
        super().__init__(config, debug)
        self._async_client = None  # Initialize async client lazily

    def _setup_client(self) -> None:
        """Initialize Ollama client"""
        try:
            from ollama import Client
            self.client = Client(host=self.config.base_url or "http://localhost:11434")
        except Exception as e:
            raise ProviderError(f"Failed to initialize Ollama client: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert messages to Ollama format"""
        ollama_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Ollama handles system messages as special user messages
                ollama_messages.append({
                    "role": "system",
                    "content": msg.content
                })
                continue

            if msg.role == Role.TOOL:
                # Format tool results
                ollama_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "name": msg.name
                })
            else:
                # Format regular messages
                ollama_messages.append({
                    "role": "user" if msg.role == Role.USER else "assistant",
                    "content": msg.content
                })

        return ollama_messages

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        # max_tokens: Optional[int] = None,
        stream: Optional[bool] = False
        # params: ChatParameters
    ) -> ModelResponse:
        """Get a basic chat completion"""
        try:
            # Build options dictionary
            options = {
                # "temperature": params.temperature
                "temperature": temperature
            }

            response = self.client.chat(
                model=model,
                messages=self._format_messages(messages),
                options=options,
                # stream=params.stream
                stream=stream
            )

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Ollama completion failed: {str(e)}")

    def _get_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        json_temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get completion with tool usage"""
        try:
            # Build options dictionary
            options = {
                "temperature": temperature
            }

            response = self.client.chat(
                model=model,
                messages=self._format_messages(messages),
                tools=[tool.get_schema() for tool in tools],
                options=options,
                stream=False
            )

            # Process tool calls if any
            tool_calls = self._extract_tool_calls(response)

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=tool_calls
            )
        except Exception as e:
            raise ProviderError(f"Ollama tool completion failed: {str(e)}")

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Optional[Type[BaseModel]],
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON"""
        try:
            # Format schema for system prompt
            schema_json = schema.model_json_schema()
            schema_prompt = (
                "You must respond with valid JSON that matches this schema:\n"
                f"{json.dumps(schema_json, indent=2)}\n\n"
                "Respond ONLY with valid JSON. No other text."
            )

            # Create new messages list with modified system message
            formatted_messages = []
            system_content = schema_prompt

            for msg in messages:
                if msg.role == Role.SYSTEM:
                    # Combine existing system message with schema prompt
                    system_content = f"{msg.content}\n\n{schema_prompt}"
                else:
                    formatted_messages.append(msg)

            # Add system message at the start
            formatted_messages.insert(0, Message(
                role=Role.SYSTEM,
                content=system_content
            ))

            # Build options dictionary
            options = {
                "temperature": temperature,
                "format": "json"  # Enable JSON mode
            }

            response = self.client.chat(
                model=model,
                messages=self._format_messages(formatted_messages),
                options=options,
                stream=False
            )

            # Validate response against schema
            try:
                content = self._extract_content(response)
                data = json.loads(content)
                schema.model_validate(data)
            except Exception as e:
                raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=response,
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Ollama JSON completion failed: {str(e)}")

    def _extract_usage(self, response: Any) -> TokenUsage:
        """Extract token usage from Ollama response"""
        # Ollama might not provide token counts
        return TokenUsage(
            prompt_tokens=getattr(response, "prompt_tokens", 0),
            completion_tokens=getattr(response, "completion_tokens", 0),
            total_tokens=getattr(response, "total_tokens", 0)
        )

    def _extract_content(self, response: Any) -> str:
        """Extract content from Ollama response"""
        if not hasattr(response, "message"):
            return ""

        if hasattr(response.message, "content"):
            return response.message.content.strip()

        return ""

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from Ollama response"""
        if not hasattr(response, "message") or not hasattr(response.message, "tool_calls"):
            return None

        tool_calls = []
        for tool_call in response.message.tool_calls:
            # Generate a unique ID if none provided
            call_id = getattr(tool_call, "id", f"call_{len(tool_calls)}")

            tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": json.dumps(tool_call.function.arguments)
                }
            })

            if self.debug:
                print(f"\nExtracted tool call: {json.dumps(tool_calls[-1], indent=2)}")

        return tool_calls if tool_calls else None

    async def _asetup_client(self) -> None:
        """Initialize async OpenAI client"""
        from ollama import AsyncClient
        try:
            self._async_client = AsyncClient(
                host=self.config.base_url or "http://localhost:11434",
                # timeout=self.config.timeout,
                # max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize async OpenAI client: {str(e)}")
        
    async def _ensure_async_client(self) -> None:
        """Ensure async client is initialized"""
        if self._async_client is None:
            await self._asetup_client()
    
    async def _aget_chat_completion(self, messages, model, temperature, max_tokens = None):
        """Get a basic chat completion asynchronously"""
        try:
            await self._ensure_async_client()
            response = await self._async_client.chat(
                model=model,
                messages=[msg.model_dump() for msg in messages],
                options={"temperature": temperature}
            )

            return ModelResponse(
                content=self._extract_content(response),
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"Ollama async completion failed: {str(e)}")
    
    def _aget_json_completion(self, messages, model, schema, temperature, max_tokens = None):
        return super()._aget_json_completion(messages, model, schema, temperature, max_tokens)
    
    def _aget_tool_completion(self, messages, model, tools, temperature, max_tokens = None, format_json = False, json_schema = None):
        return super()._aget_tool_completion(messages, model, tools, temperature, max_tokens, format_json, json_schema)
    
    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert Ollama response to dictionary"""
        return {
            "message": {
                "role": response.message.role,
                "content": response.message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in (response.message.tool_calls or [])
                ] if response.message.tool_calls else None
            },
            # "usage": {
            #     "prompt_tokens": response.usage.prompt_tokens,
            #     "completion_tokens": response.usage.completion_tokens,
            #     "total_tokens": response.usage.total_tokens
            # },
            "model": response.model,
            "created_at": response.created_at
        }