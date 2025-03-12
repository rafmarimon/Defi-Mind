# DEFIMIND Browser Agent

## Overview

The DEFIMIND Browser Agent is a powerful tool for autonomous DeFi yield data collection and analysis. It enables DEFIMIND to interact with DeFi websites, gather real-time yield information, and make data-driven decisions.

This module provides two implementations:

1. **Custom Browser Controller** - Our own implementation using Playwright
2. **Browser-Use Integration** - Leveraging the `browser-use` package for AI-powered browsing

## Features

- **Automated Yield Data Collection**: Scrape yield data from DeFi aggregators and individual protocols
- **Protocol Analysis**: Analyze specific protocols and compare opportunities
- **Smart Navigation**: AI-powered browsing to extract structured data
- **Cross-Platform Support**: Works on Windows, macOS, and Linux
- **Headless Operation**: Can run in both visible and headless modes

## Setup

### Prerequisites

- Python 3.11 or higher
- Playwright browser automation framework
- OpenAI API key (for AI-powered browsing)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   playwright install
   ```

2. Set up environment variables:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   echo "BROWSER_HEADLESS=true" >> .env
   echo "LLM_MODEL=gpt-4o" >> .env
   ```

## Usage

### Custom Browser Controller

Our custom implementation provides direct control over browser operations:

```python
from core.browser_controller import BrowserController

# Initialize browser
browser = BrowserController(headless=True)

# Navigate to URL
await browser.navigate("https://defillama.com/yields")

# Extract data
data = await browser.extract_table_data(".yields-table")

# Take screenshot
await browser.take_screenshot("defi_yields.png")

# Close browser
await browser.close()
```

### Browser-Use Integration

For AI-powered browsing, we integrate with the `browser-use` package:

```python
from core.defi_browser_agent import DefiBrowserAgent

# Initialize agent
agent = DefiBrowserAgent(headless=True)

# Collect yield data from DefiLlama
yields_df = await agent.collect_from_defillama()

# Analyze specific protocol
aave_data = await agent.collect_from_protocol("aave")

# Analyze specific opportunity
analysis = await agent.analyze_defi_opportunity("curve", "3pool")

# Get agent stats
stats = agent.get_stats()
```

### Demo Scripts

We provide two demo scripts to help you get started:

1. **test_browser_agent.py** - Tests our custom browser agent
   ```bash
   python test_browser_agent.py --test all --visible
   ```

2. **browser_use_demo.py** - Demonstrates the browser-use integration
   ```bash
   python browser_use_demo.py --mode all --protocol aave
   ```

## Command-Line Options

### test_browser_agent.py

```
--test [all|defillama|protocol|analyze]  Test to run
--protocol [aave|compound|curve|etc]     Protocol to test with
--pool [USDC|3pool|etc]                  Pool to analyze
--visible                                Run browser in visible mode
```

### browser_use_demo.py

```
--mode [all|defillama|protocol|compare]  Mode to run
--protocol [aave|compound|curve|etc]     Protocol to analyze
--protocols [aave compound curve]        Protocols to compare
--url                                    Custom URL for protocol analysis
```

## Integration with DEFIMIND

The browser agent is integrated with the `AutonomousAgent` class to enhance its data collection capabilities:

```python
from core.autonomous_agent import AutonomousAgent

# Initialize agent with browser capabilities
agent = AutonomousAgent(
    config={
        "browser_headless": True,
        "llm_model": "gpt-4o",
        # other config options...
    }
)

# Run a cycle with enhanced data collection
await agent.run_cycle()
```

## Advanced Usage

### Custom Protocol Adapters

You can extend the browser agent to support additional protocols:

```python
# In core/defi_browser_agent.py
async def collect_from_custom_protocol(self):
    """Collect yield data from a custom protocol"""
    # Implementation...
    return results
```

### Debugging

For troubleshooting, you can:

1. Run in visible mode to see the browser's actions
2. Check screenshots saved in the `screenshots` directory
3. Review logs in the `logs` directory
4. Set logging level to DEBUG for more detailed information

## Security Considerations

- The browser agent runs locally and doesn't have access to your wallet or private keys
- Always verify opportunities identified by the agent before investing
- The browser agent operates in read-only mode by default
- For connecting to wallets, additional security setup is required

## License

This project is licensed under the MIT License - see the LICENSE file for details. 