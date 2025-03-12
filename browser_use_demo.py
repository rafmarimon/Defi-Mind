#!/usr/bin/env python3
"""
Demo script for using browser-use to scrape DeFi yields from DEX and lending platforms.

This script demonstrates how to use the browser-use package to create
an AI agent that can scrape yield data from DeFi platforms.
"""

import os
import asyncio
import json
import pandas as pd
import re
from datetime import datetime
from dotenv import load_dotenv

# Make sure required packages are installed
try:
    # Try importing with available browser-use version
    from browser_use import Agent
    import openai
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure to install required packages:")
    print("pip install browser-use langchain-openai openai pandas python-dotenv")
    print("playwright install")
    exit(1)

# Configure environment
load_dotenv()

# Configure output directories
RESULTS_DIR = "browser_use_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configure OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

openai.api_key = openai_api_key

# Configure LangChain
llm_model = os.getenv("LLM_MODEL", "gpt-4o")
llm = ChatOpenAI(model=llm_model, temperature=0)


# Custom function to extract tables from markdown text since utils module may vary
def extract_tables_from_markdown(text):
    """Extract markdown tables from text"""
    table_pattern = r'\|(.+)\|\n\|(?:-+\|)+\n((?:\|.+\|\n)+)'
    tables = []
    
    for match in re.finditer(table_pattern, text, re.MULTILINE):
        header_row = match.group(1).strip()
        data_rows = match.group(2).strip()
        
        headers = [h.strip() for h in header_row.split('|') if h.strip()]
        rows = []
        
        for line in data_rows.split('\n'):
            if not line.strip():
                continue
            row_data = [cell.strip() for cell in line.split('|') if cell.strip()]
            if len(row_data) == len(headers):
                rows.append(row_data)
        
        if headers and rows:
            tables.append({
                'header': headers,
                'data': rows
            })
    
    return tables


def create_defi_yield_agent():
    """Create a browser agent that can scrape DeFi yield data"""
    headless = os.getenv("BROWSER_HEADLESS", "true").lower() == "true"
    
    system_prompt = """You are a DeFi yield expert agent that helps users find the best yield opportunities.
    Your goal is to search and analyze yield farming and lending options across different DeFi platforms.
    Follow these guidelines:
    1. Navigate to DeFi aggregator sites like DefiLlama or individual protocols
    2. Interpret APY/APR data correctly, distinguishing between variable and fixed rates
    3. Extract key information including: protocol name, pool/asset name, APY/APR, TVL, risk factors
    4. For lending protocols, note both supply and borrow rates when available
    5. Format results in a structured way that can be saved as CSV/JSON
    6. Provide insights about the yield opportunities, including potential risks

    When analyzing yield opportunities, pay attention to:
    - Impermanent loss risk for AMM pools
    - Protocol security history and audits
    - Lock-up periods or withdrawal limitations
    - Whether yields are sustainable or temporary (e.g., boosted by incentives)
    
    Be thorough in your data collection and precise with your analysis.
    Format all data tables using markdown tables with headers.
    """
    
    # Create agent with updated API
    agent = Agent(
        llm=llm,
        headless=headless,
        system_prompt=system_prompt
    )
    return agent


async def scrape_defillama_yields():
    """Scrape yield data from DeFi Llama"""
    print("Creating DeFi yield agent...")
    agent = create_defi_yield_agent()
    
    # Define the query for top yield opportunities
    query = """Go to DefiLlama's yield page (https://defillama.com/yields) and collect data on the top 20 yield opportunities.
    For each opportunity, extract:
    1. Protocol name
    2. Pool name
    3. APY percentage
    4. TVL (Total Value Locked)
    5. Available tokens
    6. Any risk indicators

    Filter for opportunities with at least $100,000 TVL to avoid low-liquidity pools.
    Organize the data in a markdown table format suitable for saving to a CSV file.
    After collecting the data, provide a brief analysis of the top 3 opportunities based on risk-adjusted returns.
    """
    
    print("Querying DefiLlama for yield data...")
    start_time = datetime.now()
    
    try:
        # Run the agent
        result = await agent.run(task=query)
        
        print(f"Agent completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        print("\nAgent Response:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Save raw result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{RESULTS_DIR}/defillama_yields_raw_{timestamp}.txt", "w") as f:
            f.write(result)
        
        # Try to parse and structure the data
        # This is a basic parser - in production you would want something more robust
        try:
            # Parse tables from the result
            tables = extract_tables_from_markdown(result)
            if tables:
                # Convert the first table to DataFrame
                df = pd.DataFrame(tables[0]['data'], columns=tables[0]['header'])
                
                # Save to CSV
                csv_path = f"{RESULTS_DIR}/defillama_yields_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                print(f"Saved structured data to {csv_path}")
                
                # Save to JSON
                json_path = f"{RESULTS_DIR}/defillama_yields_{timestamp}.json"
                df.to_json(json_path, orient="records", indent=2)
                print(f"Saved JSON data to {json_path}")
                
                # Display top 5 yields
                print("\nTop 5 Yield Opportunities:")
                print(df.head(5).to_string())
            else:
                print("No tables found in the response.")
        except Exception as e:
            print(f"Error parsing data: {str(e)}")
    
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")


async def analyze_protocol(protocol_name, url=None):
    """Analyze a specific DeFi protocol"""
    print(f"Creating agent to analyze {protocol_name}...")
    agent = create_defi_yield_agent()
    
    if not url:
        # Default URLs for common protocols
        protocol_urls = {
            "aave": "https://app.aave.com/markets/",
            "compound": "https://app.compound.finance/",
            "curve": "https://curve.fi/#/ethereum/pools",
            "convex": "https://www.convexfinance.com/stake",
            "balancer": "https://app.balancer.fi/#/ethereum/pools",
            "uniswap": "https://app.uniswap.org/#/pools",
        }
        url = protocol_urls.get(protocol_name.lower(), f"https://defillama.com/protocol/{protocol_name}")
    
    # Define the query for protocol analysis
    query = f"""Go to {url} and analyze the yield opportunities available on {protocol_name}.
    
    Extract the following information for each pool or asset:
    1. Asset/Pool name
    2. Current APY/APR
    3. TVL or liquidity
    4. Any additional rewards or incentives
    5. Risk factors or special conditions
    
    For lending protocols like Aave or Compound, include both supply and borrow rates.
    For AMM protocols like Curve or Uniswap, include information about the token composition.
    
    After collecting the data, provide:
    1. A summary of the protocol's current yield offerings
    2. The top 3 opportunities based on risk-adjusted returns
    3. Any notable risks or considerations specific to this protocol
    
    Present the data in a structured markdown table format.
    """
    
    print(f"Querying {protocol_name} for yield data...")
    start_time = datetime.now()
    
    try:
        # Run the agent
        result = await agent.run(task=query)
        
        print(f"Agent completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        print("\nAgent Response:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Save raw result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{RESULTS_DIR}/{protocol_name}_analysis_raw_{timestamp}.txt", "w") as f:
            f.write(result)
        
        # Try to parse and structure the data
        try:
            # Parse tables from the result
            tables = extract_tables_from_markdown(result)
            if tables:
                # Save each table
                for i, table in enumerate(tables):
                    df = pd.DataFrame(table['data'], columns=table['header'])
                    
                    # Save to CSV
                    csv_path = f"{RESULTS_DIR}/{protocol_name}_table{i+1}_{timestamp}.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"Saved table {i+1} to {csv_path}")
                
                # Save first table to JSON as main results
                json_path = f"{RESULTS_DIR}/{protocol_name}_yields_{timestamp}.json"
                pd.DataFrame(tables[0]['data'], columns=tables[0]['header']).to_json(
                    json_path, orient="records", indent=2
                )
                print(f"Saved JSON data to {json_path}")
            else:
                print("No tables found in the response.")
        except Exception as e:
            print(f"Error parsing data: {str(e)}")
    
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")


async def compare_protocols(protocols=None):
    """Compare yields across multiple protocols"""
    if protocols is None:
        protocols = ["aave", "compound", "curve"]
    
    print(f"Creating agent to compare protocols: {', '.join(protocols)}...")
    agent = create_defi_yield_agent()
    
    # Define the query for protocol comparison
    query = f"""Compare the yield opportunities across the following protocols: {', '.join(protocols)}.
    
    For each protocol:
    1. Visit the protocol's main application (you can search for "{protocols[0]} app" if needed)
    2. Identify the top 3 yield opportunities by APY
    3. Note the APY, TVL, and any risks or special conditions
    
    Focus on these specific assets across all protocols if available:
    - Stablecoins (USDC, USDT, DAI)
    - Major cryptocurrencies (ETH, BTC)
    
    After collecting data from all protocols, create a comparison table showing:
    1. The best places to earn yield on each asset type
    2. The risk-adjusted returns across protocols
    3. Any protocol-specific advantages or disadvantages
    
    Conclude with recommendations for:
    - Best stablecoin yields
    - Best ETH/BTC yields
    - Safest options for conservative investors
    - Highest yields for risk-tolerant investors
    
    Format all data in markdown tables.
    """
    
    print("Comparing yields across protocols...")
    start_time = datetime.now()
    
    try:
        # Run the agent with updated API
        result = await agent.run(task=query)
        
        print(f"Agent completed in {(datetime.now() - start_time).total_seconds():.2f} seconds")
        print("\nAgent Response:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        # Save raw result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        protocols_str = "_".join(protocols)
        with open(f"{RESULTS_DIR}/comparison_{protocols_str}_{timestamp}.txt", "w") as f:
            f.write(result)
        
        # Try to parse and structure the data
        try:
            # Parse tables from the result
            tables = extract_tables_from_markdown(result)
            if tables:
                # Save each table
                for i, table in enumerate(tables):
                    df = pd.DataFrame(table['data'], columns=table['header'])
                    
                    # Save to CSV
                    csv_path = f"{RESULTS_DIR}/comparison_{protocols_str}_table{i+1}_{timestamp}.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"Saved comparison table {i+1} to {csv_path}")
                
                # Save first table to JSON as main results
                json_path = f"{RESULTS_DIR}/comparison_{protocols_str}_{timestamp}.json"
                pd.DataFrame(tables[0]['data'], columns=tables[0]['header']).to_json(
                    json_path, orient="records", indent=2
                )
                print(f"Saved JSON data to {json_path}")
            else:
                print("No tables found in the response.")
        except Exception as e:
            print(f"Error parsing data: {str(e)}")
    
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")


async def main():
    """Main function to run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="BrowserUse DeFi Yield Demo")
    parser.add_argument("--mode", choices=["defillama", "protocol", "compare", "all"], 
                      default="defillama", help="Mode to run")
    parser.add_argument("--protocol", default="aave", help="Protocol to analyze")
    parser.add_argument("--protocols", nargs="+", default=["aave", "compound", "curve"],
                      help="Protocols to compare")
    parser.add_argument("--url", default=None, help="Custom URL for protocol analysis")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BROWSER-USE DEFI YIELD DEMO")
    print("=" * 80)
    print("This demo uses browser-use and LLMs to scrape DeFi yield data")
    print(f"Results will be saved to {RESULTS_DIR}/")
    print("=" * 80)
    
    if args.mode == "defillama" or args.mode == "all":
        print("\n=== SCRAPING DEFILLAMA YIELDS ===")
        await scrape_defillama_yields()
    
    if args.mode == "protocol" or args.mode == "all":
        print(f"\n=== ANALYZING {args.protocol.upper()} ===")
        await analyze_protocol(args.protocol, args.url)
    
    if args.mode == "compare" or args.mode == "all":
        print(f"\n=== COMPARING PROTOCOLS: {', '.join(args.protocols).upper()} ===")
        await compare_protocols(args.protocols)
    
    print("\n=== DEMO COMPLETE ===")
    print(f"All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main()) 