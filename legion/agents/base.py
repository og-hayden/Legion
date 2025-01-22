import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from pydantic import BaseModel
from rich import print as rprint
from rich.console import Console

from ..interface.base import LLMInterface
from ..interface.schemas import Message, ModelResponse, ProviderConfig, Role, SystemPrompt
from ..interface.tools import BaseTool
from ..memory.base import MemoryProvider
from ..memory.providers.memory import ConversationMemory
from ..providers import get_provider

# Set up rich console and logging
console = Console()
log = logging.getLogger("legion")

# Role mapping for convenience
ROLE_MAPPING = {
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "system": Role.SYSTEM,
    "tool": Role.TOOL,
    # Allow both cases
    "USER": Role.USER,
    "ASSISTANT": Role.ASSISTANT,
    "SYSTEM": Role.SYSTEM,
    "TOOL": Role.TOOL
}

class Agent:
    """Base agent class with LLM capabilities"""

    def __init__(
        self,
        name: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[Union[str, SystemPrompt]] = None,
        debug: bool = False,
        **kwargs
    ):
        """Initialize agent with configuration"""
        self.name = name
        # Handle provider prefix in model name
        if ":" in model:
            provider, model_name = model.split(":", 1)
            self._provider_name = provider
            self.model = model_name
        else:
            self._provider_name = "openai"  # Default provider
            self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug = debug

        # Handle system prompt
        if isinstance(system_prompt, SystemPrompt):
            self.system_prompt = system_prompt
        else:
            self.system_prompt = SystemPrompt(static_prompt=system_prompt or "")

        self._provider = None
        self._tools = []
        self._memory = ConversationMemory()
        self._memory_provider = None
        self._kwargs = kwargs

        # Initialize LLM provider
        self.llm = self._setup_provider(self._provider_name)

        # Add system prompt to memory - but don't render it yet
        # It will be rendered with dynamic values during process/aprocess
        self._memory.add_message(Message(
            role=Role.SYSTEM,
            content=""  # Empty initially, will be updated during process
        ))

    @property
    def full_model_name(self) -> str:
        """Get full model name including provider prefix"""
        return f"{self._provider_name}:{self.model}"

    @property
    def tools(self) -> List[BaseTool]:
        """Get agent's tools"""
        return self._tools

    @tools.setter
    def tools(self, value: Sequence[BaseTool]):
        """Set agent's tools"""
        self._tools = list(value)

    def _setup_provider(self, provider: Union[str, LLMInterface], api_key: Optional[str] = None) -> LLMInterface:
        """Set up the LLM provider"""
        # If provider is already an instance, return it
        if isinstance(provider, LLMInterface):
            return provider

        # Otherwise, treat it as a provider name
        provider_config = ProviderConfig(
            api_key=api_key,
            model=self.model,
            **self._kwargs
        )
        return get_provider(provider, provider_config)

    def _build_enhanced_prompt(self, dynamic_values: Optional[Dict[str, str]] = None) -> str:
        """Build enhanced system prompt with tools"""
        # Get base prompt with dynamic values
        base_prompt = self.system_prompt.render(dynamic_values)

        # Add tools section
        if self._tools:
            tools_text = []

            # Add tools
            for tool in self._tools:
                tools_text.append(f"- {tool.name}: {tool.description}")

            base_prompt += "\n\nAvailable Tools:\n" + "\n".join(tools_text)

        return base_prompt
    def _create_message(self, message: Union[str, Dict[str, Any], Message]) -> Message:
        """Convert various message formats to Message object"""
        if isinstance(message, Message):
            return message
        elif isinstance(message, str):
            return Message(role=Role.USER, content=message)
        elif isinstance(message, dict):
            # Map string role to Role enum
            role = message.get("role", "user")
            if isinstance(role, str):
                role = ROLE_MAPPING.get(role, Role.USER)

            return Message(
                role=role,
                content=message.get("content", ""),
                name=message.get("name"),
                tool_call_id=message.get("tool_call_id"),
                tool_calls=message.get("tool_calls")
            )
        else:
            raise ValueError(f"Unsupported message format: {type(message)}")

    def _log_message(self, message: str, verbose: bool = False, color: str = None) -> None:
        """Internal method for consistent logging"""
        if verbose:
            if color:
                rprint(f"\n[{color}]{message}[/{color}]")
            else:
                rprint(f"\n{message}")

    def _log_response(self, response: ModelResponse, verbose: bool = False) -> None:
        """Log response details when in verbose mode"""

        self._log_message("Agent Response:", verbose, "bold green")
        self._log_message(response.content, verbose)

        if response.usage:
            self._log_message("Token Usage:", verbose, "bold blue")
            self._log_message(f"Input tokens: {response.usage.prompt_tokens}", verbose)
            self._log_message(f"Output tokens: {response.usage.completion_tokens}", verbose)
            self._log_message(f"Total tokens: {response.usage.total_tokens}", verbose)

        if response.tool_calls:
            self._log_message("Tool Calls:", verbose, "bold blue")
            for tool_call in response.tool_calls:
                self._log_message(f"Tool: {tool_call['function']['name']}", verbose, "bold yellow")
                self._log_message(f"Arguments: {tool_call['function']['arguments']}", verbose)
                if "result" in tool_call:
                    self._log_message(f"Result: {tool_call['result']}", verbose)
                self._log_message("---", verbose)

    async def _aprocess(
        self,
        message: Union[str, Dict[str, Any], Message],
        response_schema: Optional[Type[BaseModel]] = None,
        dynamic_values: Optional[Dict[str, str]] = None,
        injected_parameters: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = False
    ) -> ModelResponse:
        """Process a message asynchronously and return a response"""

        if verbose:
            self.print_hierarchy()

        self._log_message(f"\n🤖 Agent {self.name} processing:", verbose, "bold blue")
        self._log_message(f"Temperature: {self.temperature}", verbose)
        self._log_message(f"Model: {self.model}", verbose)
        self._log_message(f"Tools: {[t.name for t in self._tools]}", verbose)
        self._log_message(f"Response Schema: {response_schema.__name__ if response_schema else 'None'}", verbose)
        if dynamic_values:
            self._log_message(f"Dynamic Values: {dynamic_values}", verbose)
        if injected_parameters:
            self._log_message(f"Injected Parameters: {injected_parameters}", verbose)

        # Convert message to proper format
        message_obj = self._create_message(message)

        # Update system prompt with current dynamic values and tools
        enhanced_prompt = self._build_enhanced_prompt(dynamic_values)
        if self._memory.messages and self._memory.messages[0].role == Role.SYSTEM:
            self._memory.messages[0].content = enhanced_prompt
        else:
            # Insert system prompt at the beginning if not present
            self._memory.messages.insert(0, Message(
                role=Role.SYSTEM,
                content=enhanced_prompt
            ))

        # Add user message to memory
        self.memory.add_message(message_obj)

        self._log_message("\n📨 System Prompt:", verbose, "bold blue")
        self._log_message(enhanced_prompt, verbose)
        self._log_message("\n📨 User Message:", verbose, "bold blue")
        self._log_message(f"Content: {message_obj.content}", verbose)

        try:
            # Get response from provider
            self._log_message("\n🔄 Getting response from provider...", verbose, "bold yellow")

            response = await self.llm.acomplete(
                messages=self.memory.messages,  # Use full conversation history
                model=self.model,
                tools=self._tools,
                temperature=self.temperature,
                response_schema=response_schema
            )

            self._log_response(response, verbose)

            # Add response to memory
            self.memory.add_message(Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # Handle tool calls if any
            if response.tool_calls:
                tool_results = []
                for tool_call in response.tool_calls:
                    tool = next(
                        (t for t in self._tools if t.name == tool_call["function"]["name"]),
                        None
                    )
                    if tool:
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            # Add injected parameters to tool call
                            if injected_parameters:
                                args["__injected_parameters__"] = injected_parameters
                            result = await tool(**args)
                            # Convert result to string if it's a dict
                            if isinstance(result, dict):
                                result = json.dumps(result, indent=2)

                            # Add tool result to memory
                            self.memory.add_message(Message(
                                role=Role.TOOL,
                                content=str(result),
                                name=tool_call["function"]["name"],
                                tool_call_id=tool_call["id"]
                            ))

                            tool_results.append(str(result))
                        except Exception as e:
                            self._log_message(f"\n❌ Tool execution failed: {str(e)}", verbose, "bold red")
                            raise

                # Combine tool results into final response
                combined_content = response.content
                response = ModelResponse(
                    content=combined_content,
                    raw_response=response.raw_response,
                    usage=response.usage,
                    tool_calls=response.tool_calls
                )

            self._log_message("\n✅ Got response from provider:", verbose, "bold green")
            if len(response.content) > 200:
                self._log_message(f"Content: {response.content[:200]}...", verbose, "bold blue")
            else:
                self._log_message(f"Content: {response.content}", verbose, "bold blue")

            return response

        except Exception as e:
            self._log_message(f"\n❌ Error in agent processing: {str(e)}", verbose, "bold red")
            raise

    def process(
        self,
        message: Union[str, Dict[str, Any], Message],
        response_schema: Optional[Type[BaseModel]] = None,
        thread_id: Optional[str] = None,
        dynamic_values: Optional[Dict[str, str]] = None,
        injected_parameters: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = False
    ) -> ModelResponse:
        """Process a message and return a response (sync version)"""
        return asyncio.run(self.aprocess(
            message,
            response_schema=response_schema,
            thread_id=thread_id,
            dynamic_values=dynamic_values,
            injected_parameters=injected_parameters,
            verbose=verbose
        ))

    async def aprocess(
        self,
        message: Union[str, Dict[str, Any], Message],
        response_schema: Optional[Type[BaseModel]] = None,
        thread_id: Optional[str] = None,
        dynamic_values: Optional[Dict[str, str]] = None,
        injected_parameters: Optional[List[Dict[str, Any]]] = None,
        verbose: bool = False
    ) -> ModelResponse:
        """Process a message asynchronously and return a response

        Args:
        ----
            message: The message to process
            response_schema: Optional schema for response validation
            thread_id: Optional thread ID for memory persistence
            dynamic_values: Optional dynamic values for system prompt
            injected_parameters: Optional list of parameter injections for tools
                Format: [
                    {
                        "tool": tool_function,  # The tool function to inject into
                        "parameters": {  # Parameters to inject
                            "param_name": "value"
                        }
                    }
                ]
            verbose: Whether to print verbose output

        """
        if self.memory_provider:
            # If no thread specified but we have a memory provider,
            # get or create a default thread
            if thread_id is None:
                thread_id = await self.memory_provider.get_or_create_thread(self.name)

            # Load thread state
            self._current_thread = thread_id
            await self._load_thread_state(thread_id)

        try:
            # Process message
            response = await self._aprocess(
                message,
                response_schema=response_schema,
                dynamic_values=dynamic_values,
                injected_parameters=injected_parameters,
                verbose=verbose
            )

            # Save state if using memory
            if self.memory_provider:
                await self._save_thread_state()

            return response
        finally:
            self._current_thread = None

    # Just here for backward compatibility
    def generate(self, *args, **kwargs) -> ModelResponse:
        """Deprecated: Use process() instead"""
        import warnings
        warnings.warn(
            "generate() is deprecated and will be removed in a future version. "
            "Use process() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.process(*args, **kwargs)

    def wipe_memory(self) -> None:
        """Wipes the conversation memory but preserves the system prompt.
        This is useful for starting a new conversation while keeping the agent's core instructions.
        """
        # Get the system prompt from the current memory
        system_messages = [msg for msg in self.memory.messages if msg.role == Role.SYSTEM]
        system_prompt = system_messages[0] if system_messages else None

        # Create new memory instance
        self.memory = ConversationMemory()

        # Restore system prompt if it exists
        if system_prompt:
            self.memory.add_message(system_prompt)
        else:
            # If no system prompt found, rebuild it from config
            enhanced_prompt = self._build_enhanced_prompt()
            self.memory.add_message(Message(
                role=Role.SYSTEM,
                content=enhanced_prompt
            ))

    def print_hierarchy(self, indent: str = "") -> None:
        """Print agent in hierarchy"""
        rprint(f"{indent}[cyan]\n└──[/cyan] [bold]{self.name}[/bold] ([yellow]Agent[/yellow])")

        # Print tools if any
        if self.tools:
            tool_indent = indent + "    "
            for tool in self.tools:
                rprint(f"{tool_indent}[dim]• Tool: {tool.name}[/dim]")

    @staticmethod
    def _parse_model_string(model_str: str) -> tuple[str, str]:
        """Parse a model string in format 'provider:model'

        Examples
        --------
            'openai:gpt-4o-mini' -> ('openai', 'gpt-4o-mini')
            'anthropic:claude-3-opus' -> ('anthropic', 'claude-3-opus')
            'groq:mixtral-8x7b' -> ('groq', 'mixtral-8x7b')

        """
        if ":" not in model_str:
            # Default to OpenAI if no provider specified
            return "openai", model_str

        provider, model = model_str.split(":", 1)
        return provider.lower(), model

    async def _load_thread_state(self, thread_id: str) -> None:
        """Load state for current thread"""
        if not self.memory_provider:
            return

        # First, wipe current memory but preserve system prompt
        system_messages = [msg for msg in self.memory.messages if msg.role == Role.SYSTEM]
        self.memory.messages = system_messages.copy()

        # Then load state
        state = await self.memory_provider.load_state(self.name, thread_id)
        if state and "messages" in state:
            # Convert dict messages back to Message objects and append to existing system messages
            loaded_messages = [Message(**msg) for msg in state["messages"]]
            # Filter out system messages from loaded state to avoid duplicates
            loaded_messages = [msg for msg in loaded_messages if msg.role != Role.SYSTEM]
            self.memory.messages.extend(loaded_messages)

    async def _save_thread_state(self) -> None:
        """Save current state to thread"""
        if not self.memory_provider or not self._current_thread:
            return

        state = {
            "messages": [msg.dict() for msg in self.memory.messages],
            "last_updated": datetime.now().isoformat()
        }
        await self.memory_provider.save_state(
            self.name,
            self._current_thread,
            state
        )

    async def add_thread_metadata(self, key: str, value: Any) -> None:
        """Add metadata to current thread"""
        if not self.memory_provider or not self._current_thread:
            raise ValueError("No active thread or memory provider")

        thread = await self.memory_provider.get_thread(self._current_thread)
        if not thread:
            raise ValueError(f"Thread {self._current_thread} not found")

        thread.metadata[key] = value

        # Save updated state
        state = {
            "messages": [msg.dict() for msg in self.memory.messages],
            "metadata": thread.metadata,
            "last_updated": datetime.now().isoformat()
        }
        await self.memory_provider.save_state(
            self.name,
            self._current_thread,
            state
        )

    @property
    def memory(self) -> ConversationMemory:
        """Get agent's memory"""
        return self._memory

    @property
    def memory_provider(self) -> Optional[MemoryProvider]:
        """Get agent's memory provider"""
        return self._memory_provider
