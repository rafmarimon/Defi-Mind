# DEFIMIND Browser Automation Guide

## Introduction

DEFIMIND includes two browser automation approaches for collecting DeFi yield data:

1. **Direct Browser Automation** - Using Playwright directly without requiring API keys
2. **Browser-Use Integration** - Using the browser-use package with OpenAI API for AI-powered browsing

This guide explains how to use both approaches, their pros and cons, and when to use each one.

## Direct Browser Automation

The direct browser automation approach uses Playwright directly to automate Chrome/Chromium to scrape DeFi websites. This approach:

- Does not require any API keys
- Works without requiring LLMs
- Is faster and more lightweight
- Is more predictable for specific sites
- Requires updating scripts if website structures change

### Example: Scraping DefiLlama Yields

```python
# Run the Direct Browser Demo
python direct_browser_demo.py --target defillama
```

This will:
1. Launch a Chrome browser
2. Navigate to DefiLlama's yield page
3. Extract yield data from the table
4. Save results to CSV and JSON files in the `browser_results` directory

### When to Use Direct Automation

Use direct browser automation when:
- You don't have an OpenAI API key
- You need to scrape specific, well-defined data from known websites
- Performance and reliability are more important than flexibility
- You want full control over the browser automation process

## Browser-Use Integration

The Browser-Use integration uses the [browser-use](https://github.com/Sinaptik-AI/browser-use) package to create an AI-powered browser agent that can:

- Navigate websites autonomously
- Understand and extract data using LLMs
- Adapt to website changes
- Follow complex instructions
- Handle unfamiliar websites

### Requirements

To use Browser-Use integration, you need:
- An OpenAI API key (set in `.env` file as `OPENAI_API_KEY`)
- The browser-use package (`pip install browser-use`)
- Playwright installed (`pip install playwright && python -m playwright install`)

### Example: Using Browser-Use Demo

```python
# Run the Browser-Use Demo
python browser_use_demo.py --mode defillama
```

This will:
1. Create an AI agent using your OpenAI API key
2. Launch a browser controlled by the AI
3. Follow instructions to navigate to DefiLlama and extract yield data
4. Process and save the results

### When to Use Browser-Use Integration

Use Browser-Use integration when:
- You have an OpenAI API key
- You need to scrape data from unfamiliar or changing websites
- You want to handle complex, multi-step browsing tasks
- You need the flexibility of natural language instructions
- The website structure might change frequently

## Using DefiBrowserAgent in DEFIMIND

The `DefiBrowserAgent` class in DEFIMIND can use both approaches:

```python
from core.defi_browser_agent import DefiBrowserAgent

# Using Browser-Use (requires API key)
agent = DefiBrowserAgent(llm_api_key="your-openai-api-key")
yields = await agent.collect_defi_yields()

# Using direct automation (no API key required)
agent = DefiBrowserAgent()
yields = await agent.collect_from_defillama_direct()
```

## Performance Comparison

| Feature | Direct Automation | Browser-Use |
|---------|-------------------|-------------|
| Speed | Faster | Slower |
| Flexibility | Lower | Higher |
| Requires API Key | No | Yes |
| Handles Site Changes | No | Yes |
| Development Effort | Higher | Lower |
| Maintenance Effort | Higher | Lower |

## Recommended Approach

For the best results, we recommend:

1. **Use Direct Automation** for:
   - Critical, frequently-used data sources
   - Production environments with high reliability requirements
   - Environments without API keys

2. **Use Browser-Use** for:
   - Exploratory analysis
   - Handling unfamiliar websites
   - Complex multi-step browsing tasks
   - When website structures change frequently

The DEFIMIND platform is designed to gracefully fall back to direct automation when API keys are not available, ensuring you can always collect the data you need. 