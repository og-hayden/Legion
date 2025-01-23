# Quick Start Guide

This guide will help you get started with Legion quickly. We'll cover the basic setup and show you how to create your first agent.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

## Installation

```bash
pip install legion
```

## Basic Usage

Save your OpenAI API key in a .env file.

```bash
OPENAI_API_KEY='<API-KEY>'
```

Here's a simple example of creating an agent:

```python
from legion.agents import agent
from legion.interface.decorators import tool
from dotenv import load_dotenv

load_dotenv()

@agent(model="openai:gpt-4-turbo", temperature=0.2)
class SimpleAgent:
    """A simple agent that can perform basic tasks"""

    @tool
    def greet(self, name: str) -> str:
        """Greet someone by name"""
        return f"Hello, {name}!"

async def main():
    agent = SimpleAgent()
    response = await agent.aprocess("Greet someone named Alice")
    print(response.content)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Next Steps

- Read the [Installation Guide](installation.md) for detailed setup instructions
- Learn about [Basic Concepts](basic-concepts.md)
- Try the [First Agent Example](first-agent.md)
