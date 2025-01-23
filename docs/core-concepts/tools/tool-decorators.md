# The tool decorator

The `@tool` decorator is a mechanism for defining specialized functions (tools) that can be called by an agent during runtime. Tools extend an agent's capabilities by allowing it to execute specific, structured tasks that might require external inputs or perform operations outside the agent's core model reasoning. These tasks can include API calls, pre-defined and specific computations, data processing, and more.

## Basic Structure of a Tool

Here's a simple example of how a tool is defined and used:

```python
from typing import Annotated
from pydantic import Field
from legion.interface.decorators import tool

@tool
def add_numbers(
    a: Annotated[int, Field(description="The first number")],
    b: Annotated[int, Field(description="The second number")]
) -> int:
    """Adds two numbers and returns the result."""
    return a + b
```
Note that the tool has all the parameters type annotated. This is non-optional in Legion, since the framework uses these annotations to give more information to the the agent of the type of data that the tool expects in the middle of the computation cycle.

## Usage of the Tool in an Agent

```python
from legion.agents import agent

@agent(
    model="openai:gpt-4o-mini",
    tools=[add_numbers],  # Bind the tool
    temperature=0.5
)
class MathAgent:
    """An agent that can perform mathematical operations using tools."""
```

## Calling the Tool

```python
async def main():
    agent = MathAgent()
    response = await agent.aprocess("Add 7 and 3 using the tool.")
    print(response.content)  # Output: The result of adding 7 and 3 is 10.
```

After this, the LLM will automatically try to understand what tool is meant to be used and how. It will invoke the function `add_numbers` with `a=7` and `b=3`. It's response will again be given back to the LLM for it to decide how to present the final answer of the computation to the user.

The full code -

```python
from typing import Annotated
from pydantic import Field
from legion.interface.decorators import tool
from legion.agents import agent

# Define the tool
@tool
def add_numbers(
    a: Annotated[int, Field(description="The first number")],
    b: Annotated[int, Field(description="The second number")]
) -> int:
    """Adds two numbers and returns the result."""
    return a + b

# Define the agent
@agent(
    model="openai:gpt-4o-mini",  # Specify the model
    tools=[add_numbers],         # Bind the tool to the agent
    temperature=0.5              # Set the temperature for response generation
)
class MathAgent:
    """An agent that can perform mathematical operations using tools."""

# Async main function to run the agent
async def main():
    agent = MathAgent()  # Initialize the agent
    response = await agent.aprocess("Add 7 and 3 using the tool.")  # Use the agent to process a request
    print(response.content)  # Print the result

# Run the main function in an async environment
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```
