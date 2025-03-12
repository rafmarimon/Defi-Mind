# DEFIMIND Browser Automation Instructions

## Introduction

DEFIMIND includes powerful browser automation capabilities for collecting real-time yield data from DeFi platforms. This document provides instructions for using these capabilities in your own projects.

## Available Scripts

DEFIMIND includes several scripts for browser automation:

1. **direct_browser_demo.py** - Direct browser automation without requiring API keys
2. **browser_use_demo.py** - AI-powered browsing with browser-use (requires OpenAI API key)
3. **test_browser_agent.py** - Comprehensive test suite for the DefiBrowserAgent
4. **browser_simple_test.py** - Simple test for verifying API key functionality

## Prerequisites

Before using the browser automation capabilities, ensure you have:

1. **Python 3.11+** installed
2. **Playwright** installed and configured
3. **Required packages** installed from requirements.txt
4. **OpenAI API key** (for AI-powered browsing only)

## Setup Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m playwright install
   ```

2. **Configure API key** (for AI-powered browsing):
   Create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your-api-key-here
   LLM_MODEL=gpt-4o
   BROWSER_HEADLESS=false
   ```

3. **Ensure cache directory exists**:
   ```bash
   mkdir -p cache
   ```

## Using Direct Browser Automation

Direct browser automation works without requiring an API key:

```bash
# Run the direct browser demo to scrape DefiLlama
python direct_browser_demo.py --target defillama

# Run the direct browser demo to scrape Aave
python direct_browser_demo.py --target aave

# Run the direct browser demo to scrape both
python direct_browser_demo.py --target both
```

Results will be saved to the `browser_results/` directory as CSV and JSON files.

## Using AI-Powered Browsing

AI-powered browsing requires an OpenAI API key:

```bash
# Run the browser-use demo to scrape DefiLlama
python browser_use_demo.py --mode defillama

# Run the browser-use demo to analyze a specific protocol
python browser_use_demo.py --mode protocol --protocol aave

# Run the browser-use demo to compare protocols
python browser_use_demo.py --mode compare --protocols aave compound curve
```

Results will be saved to the `browser_use_results/` directory.

## Using DefiBrowserAgent in Your Code

You can use the DefiBrowserAgent in your own code:

```python
from core.defi_browser_agent import DefiBrowserAgent
import asyncio

async def collect_defi_data():
    # Initialize agent (with or without API key)
    agent = DefiBrowserAgent(llm_api_key="your-api-key-here", headless=False)
    
    # Collect yield data from DefiLlama
    # This will use API-based method if API key is provided, otherwise direct automation
    yields = await agent.collect_defi_yields()
    
    # Collect data from specific protocols
    aave_data = await agent.collect_from_protocol("aave")
    compound_data = await agent.collect_from_protocol("compound")
    
    # Analyze a specific opportunity (requires API key)
    analysis = await agent.analyze_defi_opportunity("aave", "USDC")
    
    # Get agent stats
    stats = agent.get_stats()
    
    return yields, aave_data, compound_data, analysis, stats

# Run the async function
results = asyncio.run(collect_defi_data())
```

## Testing the DefiBrowserAgent

You can run the test suite to verify functionality:

```bash
# Test direct automation only
python test_browser_agent.py --test direct

# Test API-based automation (requires API key)
python test_browser_agent.py --test api

# Test fallback mechanism
python test_browser_agent.py --test fallback

# Run all tests
python test_browser_agent.py --test all
```

## Troubleshooting

If you encounter issues:

1. **Browser doesn't launch**:
   - Ensure Playwright is installed: `python -m playwright install`
   - Try running in visible mode: Set `BROWSER_HEADLESS=false` in `.env`

2. **API-based browsing fails**:
   - Check your API key is valid and has sufficient credits
   - Verify the model specified in `.env` is available to your account
   - Try a different model like `gpt-3.5-turbo` if `gpt-4o` is unavailable

3. **Data extraction fails**:
   - Websites may change their structure over time
   - Try increasing timeouts in the code
   - Update selectors if website layouts have changed

4. **Performance issues**:
   - Run in headless mode for better performance: Set `BROWSER_HEADLESS=true`
   - Limit the number of data points collected
   - Use direct automation for critical, time-sensitive operations

## Security Considerations

When using browser automation:

1. **Never connect to real wallets** - The browser automation is for data collection only
2. **Keep API keys secure** - Store in `.env` file and never commit to version control
3. **Run in isolated environments** - Consider using containers for production deployments
4. **Monitor usage** - Keep track of API usage to avoid unexpected costs

## Getting Help

If you need assistance:

1. Check the documentation in the `docs/` directory
2. Review the code comments and docstrings
3. Look at the example scripts for usage patterns
4. Consult the Playwright and browser-use documentation for specific issues 