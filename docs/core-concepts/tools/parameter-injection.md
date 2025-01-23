# Parameter Injection

A function can be made into a tool that the agent can use using the `@tool` decorator. It can have the following:

- A name
- A description
- Parameters with type hints and descriptions
- Injectable parameters for dynamic configuration

For example, we can define a tool called `process_api_query` that is supposed to ping some endpoint with a query authenticated by a given credential. We can define it like so -

```python
from typing import Annotated
from pydantic import Field
from legion.interface.decorators import tool

@tool(
    inject=["api_key", "endpoint"],
    description="Process a query using an external API",
    defaults={"api_key": "sk_test_default", "endpoint": "https://api.example.com/v1"}
)
def process_api_query(
    query: Annotated[str, Field(description="The query to process")],
    api_key: Annotated[str, Field(description="API key for the service")],
    endpoint: Annotated[str, Field(description="API endpoint")]
) -> str:
    """Process a query using an external API"""
    # API call logic here
    return f"Processed '{query}' using API at {endpoint} with key {api_key[:4]}..."
```

This has three parameters, `query`, `api_key` and `endpoint`. In simpler examples of how tools are meant to be used, we say the LLM deciding the values of these parameters. However, in this case, `api_key` and `endpoint` needs to come from the user and we do not want to expose these details to the LLM.

We can define parameter injection in this manner to pass on those details from the application side without exposing them to the LLM.

During the invocation of the agent, we can inject these parameters like so -

```python
response = await agent.aprocess(
    "Process the query 'test message' with production credentials.",
    injected_parameters=[
        {
            "tool": process_api_query,
            "parameters": {
                "api_key": "sk_prod_key_123",
                "endpoint": "https://api.prod.example.com/v1"
            }
        }
    ]
)
```
Putting it all together,

```python
from typing import Annotated

from colorama import Fore, Style
from dotenv import load_dotenv
from pydantic import Field

from legion.agents import agent
from legion.interface.decorators import tool  # noqa: F401

load_dotenv()


@tool(
    inject=["api_key", "endpoint"],
    description="Process a query using an external API",
    defaults={"api_key": "sk_test_default", "endpoint": "https://api.example.com/v1"}
)
def process_api_query(
    query: Annotated[str, Field(description="The query to process")],
    api_key: Annotated[str, Field(description="API key for the service")],
    endpoint: Annotated[str, Field(description="API endpoint")]
) -> str:
    """Process a query using an external API"""
    # API call logic here
    response = f"Processed '{query}' using API at {endpoint} with key {api_key[:4]}..."
    print(response)
    return response


@agent(
    model="openai:gpt-4o-mini",
    temperature=0.2,
    tools=[process_api_query],  # Bind the tool
)
class APIAgent:
    """An agent that demonstrates using tools with injected parameters.

    I can process queries using an external API without exposing sensitive credentials.
    Parameters are injected per-message with optional defaults. Simply execute the given
    process_api_query tool with the given parameters.
    """


async def main():
    print(f"{Fore.GREEN}[MAIN]{Style.RESET_ALL} Creating agent instance")
    # Create an instance of our agent
    agent = APIAgent()

    response = await agent.aprocess(
        "Process the query 'test message' with production credentials.",
        injected_parameters=[
            {
                "tool": process_api_query,
                "parameters": {
                    "api_key": "sk_prod_key_123",
                    "endpoint": "https://api.prod.example.com/v1"
                }
            }
        ]
    )

    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

This creates a tool with injected parameters determined at runtime.
