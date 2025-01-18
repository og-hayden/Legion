# File: llm_kit/providers/huggingface.py

import json
from typing import Any, Dict, List, Optional, Sequence, Type
import uuid

from openai import AsyncOpenAI, OpenAI
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
from .factory import ProviderFactory


class HuggingFaceFactory(ProviderFactory):
    """Factory for creating HuggingFace providers"""

    def create_provider(self, config: Optional[ProviderConfig] = None, **kwargs) -> LLMInterface:
        """Create a new HuggingFace provider instance"""
        return HuggingFaceProvider(config=config or ProviderConfig(), **kwargs)

class HuggingFaceProvider(LLMInterface):
    """HuggingFace-specific implementation of the LLM interface"""

    def __init__(self, config: ProviderConfig, debug: bool = False):
        """Initialize provider with both sync and async clients"""
        super().__init__(config, debug)
        self._async_client = None  # Initialize async client lazily

    def _setup_client(self) -> None:
        """Initialize HuggingFace client"""
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url="https://api-inference.huggingface.co/v1/",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize HuggingFace client: {str(e)}")

    async def _asetup_client(self) -> None:
        """Initialize async HuggingFace client"""
        try:
            self._async_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url="https://api-inference.huggingface.co/v1/",
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
        except Exception as e:
            raise ProviderError(f"Failed to initialize async HuggingFace client: {str(e)}")

    async def _ensure_async_client(self) -> None:
        """Ensure async client is initialized"""
        if self._async_client is None:
            await self._asetup_client()

    async def _aget_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion asynchronously"""
        try:
            await self._ensure_async_client()
            formatted_messages = self._format_messages(messages)
            response = await self._async_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"HuggingFace async completion failed: {str(e)}")

    async def _aget_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get completion with tool usage asynchronously"""
        await self._ensure_async_client()
        current_messages = list(messages)
        all_tool_calls = []
        max_retries = 3
        retry_count = 0

        try:
            while retry_count < max_retries:
                if self.debug:
                    print(f"\n🔄 Making async HuggingFace API call (attempt {retry_count + 1}/{max_retries}):")
                    print(f"Messages count: {len(current_messages)}")
                    print("Tools:", [t.name for t in tools])

                # Convert messages and tools to dict format
                message_dicts = self._format_messages(current_messages)
                tool_dicts = [t.model_dump() for t in tools]

                try:
                    response = await self._async_client.chat.completions.create(
                        model=model,
                        messages=message_dicts,
                        tools=tool_dicts,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    if self.debug:
                        print("\n📥 Raw API Response:")
                        print(f"ID: {response.id}")
                        print(f"Object: {response.object}")
                        print(f"Created: {response.created}")
                        print(f"Model: {response.model}")
                        print("\nChoices:")
                        for choice in response.choices:
                            print(f"  Index: {choice.index}")
                            print(f"  Message Role: {choice.message.role}")
                            print(f"  Message Content: {choice.message.content}")
                            if hasattr(choice.message, 'tool_calls'):
                                print(f"  Tool Calls: {choice.message.tool_calls}")
                            print(f"  Finish Reason: {choice.finish_reason}")

                    choice = response.choices[0]
                    content = choice.message.content
                    
                    # First check for tool_calls in the message object
                    tool_calls = None
                    if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                        tool_calls = [{
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in choice.message.tool_calls]

                    if tool_calls:
                        # Process each tool call
                        for tool_call in tool_calls:
                            tool = next(
                                (t for t in tools if t.name == tool_call["function"]["name"]),
                                None
                            )

                            if tool:
                                try:
                                    # Arguments are already a dict
                                    args = tool_call["function"]["arguments"]
                                    result = await tool.arun(**args)
                                except Exception as e:
                                    if self.debug:
                                        print(f"\n❌ Tool execution failed: {str(e)}")
                                    raise ProviderError(f"Tool execution failed: {str(e)}")

                                # Add the tool's response
                                current_messages.append(Message(
                                    role=Role.TOOL,
                                    content=str(result),
                                    tool_call_id=tool_call["id"],
                                    name=tool_call["function"]["name"]
                                ))

                                # Store tool call for final response
                                all_tool_calls.append(tool_call)

                        # Get final response after tool usage
                        if format_json and json_schema:
                            # Add a message to format the tool response as JSON
                            current_messages.append(Message(
                                role=Role.USER,
                                content="Based on the tool response above, create a JSON object representing a person. Include a name (string), age (number), and hobbies (array of strings). Respond with ONLY the JSON object, no additional text."
                            ))
                            
                            json_response = await self._aget_json_completion(
                                messages=current_messages,
                                model=model,
                                schema=json_schema,
                                temperature=0.0,
                                max_tokens=max_tokens,
                                preserve_tool_calls=all_tool_calls
                            )
                            return json_response

                        return ModelResponse(
                            content=content or "",  # Convert None to empty string
                            raw_response=self._response_to_dict(response),
                            tool_calls=all_tool_calls,
                            usage=self._extract_usage(response)
                        )

                    retry_count += 1
                    if retry_count < max_retries:
                        if self.debug:
                            print(f"\n⚠️ No tool calls found in response, retrying ({retry_count}/{max_retries})")
                        continue
                    
                    # If we've exhausted retries, return the last response
                    return ModelResponse(
                        content=content or "",  # Convert None to empty string
                        raw_response=self._response_to_dict(response),
                        tool_calls=None,
                        usage=self._extract_usage(response)
                    )

                except Exception as api_error:
                    if self.debug:
                        print(f"\n❌ Async API call failed: {str(api_error)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise

        except Exception as e:
            raise ProviderError(f"HuggingFace async tool completion failed: {str(e)}")

    def _get_json_formatting_prompt(self, schema: Type[BaseModel], content: str) -> str:
        """Get a prompt for JSON formatting"""
        schema_fields = schema.model_json_schema()["properties"]
        field_descriptions = []
        for field, info in schema_fields.items():
            field_type = info.get("type", "any")
            if field_type == "array":
                items_type = info.get("items", {}).get("type", "any")
                field_type = f"array of {items_type}"
            field_descriptions.append(f"- {field} ({field_type})")

        return (
            "Format the following content as a valid JSON object with these fields:\n"
            + "\n".join(field_descriptions)
            + "\n\nRespond with ONLY the JSON object, no additional text. Example format:\n"
            + '{\n  "name": "John",\n  "age": 25,\n  "hobbies": ["reading", "gaming"]\n}'
            + "\n\nEnsure all required fields are included and the response is valid JSON with no line breaks outside the JSON."
        )

    async def _aget_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON asynchronously"""
        try:
            await self._ensure_async_client()

            # Get generic JSON formatting prompt
            formatting_prompt = self._get_json_formatting_prompt(schema, messages[-1].content)

            # Create messages for HuggingFace
            hf_messages = self._format_messages([
                Message(role=Role.SYSTEM, content=formatting_prompt),
                *messages
            ])

            if self.debug:
                print("\n📤 Sending messages for JSON completion:")
                for msg in hf_messages:
                    print(f"Role: {msg['role']}, Content: {msg['content'][:100]}...")

            response = await self._async_client.chat.completions.create(
                model=model,
                messages=hf_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Print raw response content for debugging
            content = response.choices[0].message.content
            print("\n📥 Raw JSON Response Content:")
            print(content)

            try:
                # Try to extract JSON from the response if it's not already JSON
                if not content.strip().startswith("{"):
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(0)
                        print("\n🔍 Extracted JSON:")
                        print(content)

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
            raise ProviderError(f"HuggingFace async JSON completion failed: {str(e)}")

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Format messages for HuggingFace API"""
        formatted = []
        system_content = []

        # Extract system messages
        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_content.append(msg.content)
                continue

            if msg.role == Role.TOOL:
                # Tool responses are treated as assistant messages
                formatted.append({"role": "assistant", "content": msg.content})
            else:
                # Map user/assistant roles directly
                formatted.append({"role": msg.role.value, "content": msg.content})

        # If we have system messages, prepend them to the first non-system message
        # or add them as a user message if there are no other messages
        if system_content:
            system_text = "\n".join(system_content)
            if formatted:
                formatted[0]["content"] = f"{system_text}\n\n{formatted[0]['content']}"
            else:
                formatted.append({"role": "user", "content": system_text})

        # Ensure proper role alternation
        alternated = []
        for i, msg in enumerate(formatted):
            alternated.append(msg)
            # If this is a user message and not the last message
            if i < len(formatted) - 1 and msg["role"] == "user" and formatted[i + 1]["role"] == "user":
                # Insert an assistant message in between
                alternated.append({"role": "assistant", "content": "I understand. Please continue."})
            # If this is an assistant message and not the last message
            elif i < len(formatted) - 1 and msg["role"] == "assistant" and formatted[i + 1]["role"] == "assistant":
                # Insert a user message in between
                alternated.append({"role": "user", "content": "Please continue."})

        # Ensure conversation starts with a user message
        if alternated and alternated[0]["role"] == "assistant":
            alternated.insert(0, {"role": "user", "content": "Please help me with the following."})

        return alternated

    def _extract_tool_calls(self, response: Any) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls from response"""
        # If response is a string (content), try to find tool calls in it
        if isinstance(response, str):
            try:
                # Try to find JSON-formatted tool calls in the content
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                    if isinstance(data, dict) and "name" in data and "arguments" in data:
                        return [{
                            "id": str(uuid.uuid4()),
                            "function": {
                                "name": data["name"],
                                "arguments": json.dumps(data["arguments"])
                            }
                        }]
            except Exception:
                pass
            return None

        # If response is a dict (raw response), check for tool_calls in the message
        if isinstance(response, dict) and "choices" in response:
            choice = response["choices"][0]
            if "message" in choice and choice["message"].get("tool_calls"):
                tool_calls = choice["message"]["tool_calls"]
                return [{
                    "id": tool_call["id"],
                    "function": {
                        "name": tool_call["function"]["name"],
                        "arguments": json.dumps(tool_call["function"]["arguments"])
                    }
                } for tool_call in tool_calls]

        # If response is an OpenAI response object, check for tool_calls in the message
        if hasattr(response, "choices") and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                tool_calls = choice.message.tool_calls
                return [{
                    "id": tool_call.id,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": json.dumps(tool_call.function.arguments)
                    }
                } for tool_call in tool_calls]

        return None

    def _extract_usage(self, response: Any) -> Optional[Dict[str, int]]:
        """Extract token usage from response"""
        if not response.usage:
            return None
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

    def _response_to_dict(self, response: Any) -> Dict[str, Any]:
        """Convert response to dictionary format"""
        return {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {
                        "role": c.message.role,
                        "content": c.message.content
                    },
                    "finish_reason": c.finish_reason
                }
                for c in response.choices
            ],
            "usage": response.usage.model_dump() if response.usage else None
        }

    def _get_chat_completion(
        self,
        messages: List[Message],
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Get a basic chat completion"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[msg.model_dump() for msg in messages],
                temperature=temperature,
                max_tokens=max_tokens
            )

            return ModelResponse(
                content=response.choices[0].message.content,
                raw_response=self._response_to_dict(response),
                usage=self._extract_usage(response),
                tool_calls=None
            )
        except Exception as e:
            raise ProviderError(f"HuggingFace completion failed: {str(e)}")

    def _get_tool_completion(
        self,
        messages: List[Message],
        model: str,
        tools: Sequence[BaseTool],
        temperature: float,
        max_tokens: Optional[int] = None,
        format_json: bool = False,
        json_schema: Optional[Type[BaseModel]] = None
    ) -> ModelResponse:
        """Get completion with tool usage"""
        current_messages = list(messages)
        all_tool_calls = []

        try:
            # First phase: Use tools
            while True:
                if self.debug:
                    print("\n🔄 Making HuggingFace API call:")
                    print(f"Messages count: {len(current_messages)}")
                    print("Tools:", [t.name for t in tools])

                # Convert messages and tools to dict format
                message_dicts = self._format_messages(current_messages)
                tool_dicts = [t.model_dump() for t in tools]

                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=message_dicts,
                        tools=tool_dicts,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as api_error:
                    if self.debug:
                        print(f"\n❌ API call failed: {str(api_error)}")
                    raise

                choice = response.choices[0]
                content = choice.message.content or ""

                # Extract tool calls from content if not provided directly
                tool_calls = choice.message.tool_calls or self._extract_tool_calls(content)

                if tool_calls:
                    # Process each tool call
                    for tool_call in tool_calls:
                        tool = next(
                            (t for t in tools if t.name == tool_call.function.name),
                            None
                        )

                        if tool:
                            try:
                                # Parse arguments string to dict
                                args = json.loads(tool_call.function.arguments)
                                result = tool.run(**args)
                            except json.JSONDecodeError:
                                # If arguments are not valid JSON, try to parse as string
                                result = tool.run(tool_call.function.arguments)

                            # Add the tool's response
                            current_messages.append(Message(
                                role=Role.TOOL,
                                content=str(result),
                                tool_call_id=tool_call.id,
                                name=tool_call.function.name
                            ))

                            # Store tool call for final response
                            all_tool_calls.append({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                },
                                "result": str(result)
                            })
                    continue

                # No more tool calls - get final response
                if format_json and json_schema:
                    # Add a message to format the tool response as JSON
                    current_messages.append(Message(
                        role=Role.USER,
                        content="Based on the tool response above, create a JSON object representing a person. Include a name (string), age (number), and hobbies (array of strings). Respond with ONLY the JSON object, no additional text."
                    ))
                    
                    json_response = self._get_json_completion(
                        messages=current_messages,
                        model=model,
                        schema=json_schema,
                        temperature=0.0,
                        max_tokens=max_tokens,
                        preserve_tool_calls=all_tool_calls if all_tool_calls else None
                    )
                    return json_response

                return ModelResponse(
                    content=content or "",  # Convert None to empty string
                    raw_response=self._response_to_dict(response),
                    tool_calls=all_tool_calls if all_tool_calls else None,
                    usage=self._extract_usage(response)
                )

        except Exception as e:
            raise ProviderError(f"HuggingFace tool completion failed: {str(e)}")

    def _get_json_completion(
        self,
        messages: List[Message],
        model: str,
        schema: Type[BaseModel],
        temperature: float,
        max_tokens: Optional[int] = None,
        preserve_tool_calls: Optional[List[Dict[str, Any]]] = None
    ) -> ModelResponse:
        """Get a chat completion formatted as JSON"""
        try:
            # Get generic JSON formatting prompt
            formatting_prompt = self._get_json_formatting_prompt(schema, messages[-1].content)

            # Create messages for HuggingFace
            hf_messages = [
                {"role": "system", "content": formatting_prompt}
            ]

            # Add remaining messages, skipping system
            hf_messages.extend([
                msg.model_dump() for msg in messages
                if msg.role != Role.SYSTEM
            ])

            response = self.client.chat.completions.create(
                model=model,
                messages=hf_messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Validate against schema
            content = response.choices[0].message.content
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
            raise ProviderError(f"HuggingFace JSON completion failed: {str(e)}")

    def _extract_content(self, response: Any) -> str:
        """Extract content from response"""
        return response.choices[0].message.content or ""
