# Building a simple chatbot agent with Legion

## Simple conversational agent

This guide will explain how to set up an chatbot agent with Legion.

```python
from legion.agents import agent
from dotenv import load_dotenv

load_dotenv()

@agent(model="openai:gpt-4o-mini", temperature=0.2)
class LifeStoryInterviewer:
    """You are an expert interviewer that conducts interview of people about thier life."""

async def main():
    agent = LifeStoryInterviewer()
    while True:
        message = input("You: ")
        response = await agent.aprocess(message=message, thread_id="abc")
        print(f"Agent:{response.content}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

> Note: Assumes that you have the API key set in a .env file. See quick start for details.

This script sets up an interactive agent designed to engage in meaningful conversations about your life.

The `thread_id` works as a tag for the conversation, letting the agent remember everything said during the session. This makes the agent’s responses more relevant and ensures the conversation feels natural and connected. Memory handling is built right into Legion, so it’s simple to create smooth, context-aware interactions without extra effort.
