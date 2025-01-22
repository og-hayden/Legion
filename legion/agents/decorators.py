import inspect
import logging
from typing import List, Optional, Type

from rich import print as rprint
from rich.console import Console

from legion.agents.base import Agent
from legion.interface.schemas import SystemPrompt, SystemPromptSection
from legion.interface.tools import BaseTool

# Set up rich console and logging
console = Console()
logger = logging.getLogger(__name__)

def agent(
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[List[BaseTool]] = None,  # Allow tools to be passed directly
    debug: Optional[bool] = False,
    **kwargs
):
    """Decorator for creating agent classes"""
    # Validate temperature
    if temperature < 0 or temperature > 1:
        raise ValueError("Temperature must be between 0 and 1")

    def _log_message(message: str, color: str = None) -> None:
        """Internal method for consistent logging"""
        if debug:
            if color:
                rprint(f"\n[{color}]{message}[/{color}]")
            else:
                rprint(f"\n{message}")

    def decorator(cls: Type) -> Type:

        _log_message(f"Decorating class {cls.__name__}", color="bold blue")
        _log_message(f"Original class bases: {cls.__bases__}")

        # Get system prompt from decorator or fallback to docstring
        if system_prompt is not None:  # Check decorator param first
            prompt_obj = SystemPrompt(static_prompt=system_prompt) if isinstance(system_prompt, str) else system_prompt
        elif hasattr(cls, "_system_prompt"):  # Then check for dynamic system prompt
            prompt_obj = cls._system_prompt
        elif cls.__doc__:  # Finally fallback to docstring
            prompt_obj = SystemPrompt(sections=[SystemPromptSection(
                content=cls.__doc__,
                is_dynamic=False
            )])
        else:
            prompt_obj = SystemPrompt(sections=[])  # Empty prompt if nothing provided

        # Create configuration
        config = {
            "model": model,
            "temperature": temperature,
            "system_prompt": prompt_obj,
            "debug": debug,
            **kwargs
        }
        if max_tokens is not None:
            config["max_tokens"] = max_tokens

        # Store original __init__ if it exists
        original_init = getattr(cls, "__init__", None)
        if original_init is object.__init__:
            original_init = None

        def __init__(self, *args, **kwargs):

            _log_message(f"Initializing {cls.__name__} instance", color="bold blue")
            _log_message(f"Instance type: {type(self)}")
            _log_message(f"Instance bases: {type(self).__bases__}")

            # Initialize Agent with config and proper name
            agent_config = {
                **config,
                "name": cls.__name__  # Always use the class name
            }

            _log_message("Calling Agent.__init__", color="bold blue")
            Agent.__init__(self, **agent_config)
            _log_message("✅Agent.__init__ completed", color="bold green")

            # Initialize tools list
            self._tools = []

            # Get tools from class attributes with @tool decorator
            for attr_name, attr in inspect.getmembers(cls):
                if hasattr(attr, "__tool__"):
                    _log_message(f"Found tool attribute: {attr_name}")
                    tool = attr.__tool_instance__
                    if tool:
                        _log_message(f"Binding tool {tool.name} to instance")
                        self._tools.append(tool.bind_to(self))
                elif isinstance(attr, BaseTool):
                    _log_message(f"Found BaseTool instance: {attr_name}")
                    self._tools.append(attr.bind_to(self))

            # Add tools passed to decorator
            if tools:
                _log_message("Adding tools from decorator", color="bold blue")
                for tool in tools:
                    _log_message(f"Binding external tool {tool.name} to instance")
                    self._tools.append(tool.bind_to(self))

            # Get tools from constructor kwargs
            constructor_tools = kwargs.pop("tools", [])
            if constructor_tools:
                _log_message("Adding tools from constructor kwargs")
                for tool in constructor_tools:
                    _log_message(f"Binding constructor tool {tool.name} to instance")
                    self._tools.append(tool.bind_to(self))

            # Call the original class's __init__ if it exists
            if original_init:
                _log_message("Calling original __init__", color="bold yellow")
                original_init(self, *args, **kwargs)

            _log_message(f"✅Registered tools: {[t.name for t in self._tools]}", color="bold green")

        # Create new class attributes
        attrs = {
            "__init__": __init__,
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "__doc__": cls.__doc__,
            "_tools": [],  # Initialize class-level tools list
            "__agent_decorator__": True  # Mark as agent class
        }

        # Copy over class attributes and methods
        for attr_name, attr in cls.__dict__.items():
            if not attr_name.startswith("__"):
                attrs[attr_name] = attr

        # Create the new class with proper inheritance
        bases = (Agent,)
        if cls.__bases__ != (object,):
            bases = bases + tuple(b for b in cls.__bases__ if b != object)

        _log_message(f"Creating new class with bases: {bases}")
        AgentClass = type(cls.__name__, bases, attrs)
        _log_message(f"✅Created new class {AgentClass.__name__} with MRO: {AgentClass.__mro__}", color="bold green")

        # Copy over any class-level tools
        if hasattr(cls, "_tools"):
            _log_message("Copying class-level tools")
            AgentClass._tools = cls._tools.copy()

        return AgentClass

    return decorator
