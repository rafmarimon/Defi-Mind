#!/usr/bin/env python3
"""
A simple test script for the browser-use package with the OpenAI API key
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("ERROR: OPENAI_API_KEY not set in .env file")
    exit(1)

print(f"API key found with length: {len(openai_api_key)}")
print(f"First 4 characters: {openai_api_key[:4]}...")
print(f"Last 4 characters: ...{openai_api_key[-4:]}")

async def run_browser_test():
    """Run a simple test with browser-use"""
    try:
        from browser_use import Agent
        from langchain_openai import ChatOpenAI
        
        # Initialize the ChatOpenAI model
        model_name = os.getenv("LLM_MODEL", "gpt-4o")
        print(f"Using model: {model_name}")
        llm = ChatOpenAI(model=model_name)
        
        # Initialize the browser agent
        print("Initializing browser agent...")
        agent = Agent(
            llm=llm,
            headless=False,  # Set to False to see the browser
        )
        
        # Run a simple task
        print("Running a simple web search task...")
        query = "What is the current Bitcoin price? Check CoinMarketCap and return just the price value."
        
        result = await agent.run(task=query)
        
        print("\nAgent Result:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
        return True
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("SIMPLE BROWSER-USE TEST")
    print("=" * 50)
    success = asyncio.run(run_browser_test())
    if success:
        print("Test completed successfully!")
    else:
        print("Test failed. See error details above.") 