"""Ollama Agent Example

This example demonstrates how to create a simple agent using Legion's decorator syntax.
The agent can use both internal (nested) tools and external tools.
"""

from typing import Annotated, List

from dotenv import load_dotenv
from pydantic import Field

from legion.agents import agent
from legion.interface.decorators import tool

load_dotenv()


@tool
def add_numbers(
    numbers: Annotated[List[float], Field(description="List of numbers to add together")]
) -> float:
    """Add a list of numbers together and return the sum."""
    return sum(numbers)


@tool
def multiply(
    a: Annotated[float, Field(description="First number to multiply")],
    b: Annotated[float, Field(description="Second number to multiply")]
) -> float:
    """Multiply two numbers together."""
    return a * b


@agent(
    model="ollama:llama3.2",
    temperature=0.3,
    tools=[add_numbers, multiply],  # Bind external tools
)
class MathHelper:
    """An agent that helps with basic arithmetic and string operations.

    I can perform calculations and manipulate text based on your requests.
    I have both external tools for math operations and internal tools for formatting.
    I format arguments in JSON format and pay attention to decimal places.
    My answers are succinct and only contain the input arguments and the answer.
    """

    @tool
    def format_result(
        self,
        number: Annotated[float, Field(description="Number to format")],
        prefix: Annotated[str, Field(description="Text to add before the number")] = (
            "Result: "
        )
    ) -> str:
        """Format a number with a custom prefix."""
        return f"{prefix}{number:.2f}"


async def main():
    # Create an instance of our agent
    agent = MathHelper()

    # Example 1: Using external add_numbers tool with internal format_result
    response = await agent.aprocess(
        "I have the numbers 1.5, 2.5, and 3.5. Can you add them together and format "
        "the result nicely?"
    )
    print("Example 1 Response:")
    print(response.content)
    print()

    # Example 2: Using external multiply tool with internal format_result
    response = await agent.aprocess(
        "I have the numbers 4.2 and 2.0. Can you multiply them and then format "
        "the result nicely?"
    )
    print("Example 2 Response:")
    print(response.content)
    print()

    # Example 3: Complex operation using both external and internal tools
    response = await agent.aprocess(
        "I need to add the numbers 10.5 and 20.5, then multiply the result by 2"
    )
    print("Example 3 Response:")
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
