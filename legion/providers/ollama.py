"""Ollama-specific implementation of the LLM interface"""

import ast
import json
from typing import Any, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel

from ..errors import ProviderError
from ..interface.base import LLMInterface
from ..interface.schemas import (
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

    def _format_arguments(
            self,
            arguments: Any
    ):
        """Convert Tools Arguments into JSON"""
        if isinstance(arguments, dict):
            try:
                args = {k:ast.literal_eval(v) for k,v in arguments.items()}
            except Exception:
                args = arguments
        else:
                args = arguments
        return json.dumps(args)


    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        stream: Optional[bool] = False
    ) -> ModelResponse:
        """Get a basic chat completion"""
        try:
            # Build options dictionary
            options = {
                "temperature": temperature
            }

            response = self.client.chat(
                model=model,
                messages=self._format_messages(messages),
                options=options,
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

    async def _aget_json_completion(
            self,
            messages,
            model,
            schema,
            temperature,
            max_tokens = None,
            preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
        ):
        """Get a chat completion formatted as JSON asynchronously"""
        try:
            await self._ensure_async_client()

            # Get generic JSON formatting prompt
            formatting_prompt = self._get_json_formatting_prompt(schema, messages[-1].content)

            # Create messages for Ollama
            messages = [
                {"role": "system", "content": formatting_prompt}
            ]

            # Add remaining messages, skipping system
            messages.extend([
                msg for msg in messages
                if msg["role"] != Role.SYSTEM
            ])

            response = await self._async_client.chat(
                model=model,
                messages=messages,
                format="json",
                options={"temperature": temperature}
            )

            # Validate against schema
            content = response.message.content
            try:
                data = json.loads(content)
                schema.model_validate(data)
            except Exception as e:
                raise ProviderError(f"Invalid JSON response: {str(e)}")

            return ModelResponse(
                content=content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=preserve_tool_calls
            )
        except Exception as e:
            raise ProviderError(f"Ollama async JSON completion failed: {str(e)}")

    async def _aget_tool_completion(
            self,
            messages,
            model,
            tools,
            temperature,
            max_tokens = None,
            format_json = False,
            json_schema = None
        ):
        "Get completion with tool usage asynchronously"""
        await self._ensure_async_client()
        current_messages = list(messages)
        all_tool_calls = []

        try:
            # First phase: Use tools
            while True:
                if self.debug:
                    print("\n🔄 Making async Ollama API call:")
                    print(f"Messages count: {len(current_messages)}")
                    print("Tools:", [t.name for t in tools])

                # Convert messages and tools to dict format
                message_dicts = self._format_messages(current_messages)
                tool_dicts = [t.model_dump() for t in tools]

                try:
                    response = await self._async_client.chat(
                        model=model,
                        messages=message_dicts,
                        tools=tool_dicts,
                        options={"temperature": temperature}
                    )
                except Exception as api_error:
                    if self.debug:
                        print(f"\n❌ Async API call failed: {str(api_error)}")
                    raise

                content = response.message.content or ""

                # Process tool calls if any
                if response.message.tool_calls:
                    # First add the assistant's message with tool calls
                    tool_call_data = []
                    for id, tool_call in enumerate(response.message.tool_calls):
                        call_data = {
                            "id": str(id),
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": self._format_arguments(tool_call.function.arguments)
                            }
                        }
                        tool_call_data.append(call_data)

                    # Add the assistant's message with tool calls
                    current_messages.append(Message(
                        role=Role.ASSISTANT,
                        content=content,
                        tool_calls=tool_call_data
                    ))

                    # Process each tool call
                    for id, tool_call in enumerate(response.message.tool_calls):
                        tool = next(
                            (t for t in tools if t.name == tool_call.function.name),
                            None
                        )

                        if tool:
                            args = json.loads(self._format_arguments(tool_call.function.arguments))
                            result = await tool.arun(**args)  # Use async tool call

                            if self.debug:
                                print(f"Tool {tool.name} returned: {result}")

                            # Add the tool's response
                            current_messages.append(Message(
                                role=Role.TOOL,
                                content=json.dumps(result) if isinstance(result, dict) else str(result),
                                tool_call_id=str(id),
                                name=tool_call.function.name
                            ))

                            # Store tool call for final response
                            call_data = next(
                                c for c in tool_call_data
                                if c["id"] == str(id)
                            )
                            call_data["result"] = json.dumps(result) if isinstance(result, dict) else str(result)
                            all_tool_calls.append(call_data)
                    continue

                # No more tool calls - get final response
                if format_json and json_schema:
                    json_response = await self._aget_json_completion(
                        messages=current_messages,
                        model=model,
                        schema=json_schema,
                        temperature=0.0,
                        preserve_tool_calls=all_tool_calls if all_tool_calls else None
                    )
                    return json_response

                return ModelResponse(
                    content=content,
                    raw_response=self._response_to_dict(response),
                    tool_calls=all_tool_calls if all_tool_calls else None,
                    usage=self._extract_usage(response)
                )

        except Exception as e:
            raise ProviderError(f"Ollama async tool completion failed: {str(e)}")

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
            "model": response.model,
            "created_at": response.created_at
        }
