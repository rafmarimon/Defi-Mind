#!/usr/bin/env python3
"""
Test script for the DefiBrowserAgent

This script demonstrates the capabilities of the DefiBrowserAgent,
including both direct browser automation and API-based methods.
"""

import os
import asyncio
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from core.defi_browser_agent import DefiBrowserAgent

# Load environment variables
load_dotenv()

async def test_direct_automation():
    """Test direct browser automation (no API key required)"""
    print("\n" + "=" * 80)
    print("TESTING DIRECT BROWSER AUTOMATION")
    print("=" * 80)
    print("This test uses Playwright directly without requiring an API key")
    
    # Initialize agent with headless=False to see the browser
    agent = DefiBrowserAgent(headless=False)
    
    # Collect yield data from DefiLlama
    print("\nCollecting yield data from DefiLlama...")
    yields = await agent.collect_from_defillama_direct()
    
    if not yields.empty:
        print(f"Successfully collected {len(yields)} yield opportunities")
        print("\nTop 5 yields:")
        print(yields.head(5))
        
        # Save to CSV for reference
        yields.to_csv("defillama_yields_direct.csv", index=False)
        print("\nSaved results to defillama_yields_direct.csv")
    else:
        print("Failed to collect yield data from DefiLlama")
    
    # Collect data from Aave
    print("\nCollecting data from Aave...")
    aave_data = await agent.collect_from_aave_direct()
    
    if not aave_data.empty:
        print(f"Successfully collected {len(aave_data)} markets from Aave")
        print("\nTop 5 Aave markets:")
        print(aave_data.head(5))
        
        # Save to CSV for reference
        aave_data.to_csv("aave_markets_direct.csv", index=False)
        print("\nSaved results to aave_markets_direct.csv")
    else:
        print("Failed to collect data from Aave")
    
    # Print agent stats
    print("\nAgent Stats:")
    print(json.dumps(agent.get_stats(), indent=2))
    
    return yields, aave_data

async def test_api_based_automation():
    """Test API-based browser automation (requires OpenAI API key)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n" + "=" * 80)
        print("SKIPPING API-BASED AUTOMATION TEST")
        print("=" * 80)
        print("No OpenAI API key found in .env file")
        print("To run this test, add your API key to the .env file as OPENAI_API_KEY")
        return None, None
    
    print("\n" + "=" * 80)
    print("TESTING API-BASED BROWSER AUTOMATION")
    print("=" * 80)
    print("This test uses browser-use with your OpenAI API key")
    
    # Initialize agent with API key and headless=False to see the browser
    agent = DefiBrowserAgent(llm_api_key=api_key, headless=False)
    
    # Collect yield data from DefiLlama
    print("\nCollecting yield data from DefiLlama using API-based automation...")
    try:
        yields = await agent.collect_from_defillama()
        
        if not yields.empty:
            print(f"Successfully collected {len(yields)} yield opportunities")
            print("\nTop 5 yields:")
            print(yields.head(5))
            
            # Save to CSV for reference
            yields.to_csv("defillama_yields_api.csv", index=False)
            print("\nSaved results to defillama_yields_api.csv")
        else:
            print("Failed to collect yield data from DefiLlama")
    except Exception as e:
        print(f"Error collecting data from DefiLlama: {str(e)}")
        yields = None
    
    # Analyze a specific opportunity
    print("\nAnalyzing a specific opportunity...")
    try:
        analysis = await agent.analyze_defi_opportunity("aave", "USDC")
        print(json.dumps(analysis, indent=2))
        
        # Save analysis to JSON
        with open("aave_usdc_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        print("\nSaved analysis to aave_usdc_analysis.json")
    except Exception as e:
        print(f"Error analyzing opportunity: {str(e)}")
        analysis = None
    
    # Print agent stats
    print("\nAgent Stats:")
    print(json.dumps(agent.get_stats(), indent=2))
    
    return yields, analysis

async def test_fallback_mechanism():
    """Test the fallback mechanism from API to direct automation"""
    print("\n" + "=" * 80)
    print("TESTING FALLBACK MECHANISM")
    print("=" * 80)
    print("This test demonstrates the fallback from API to direct automation")
    
    # First, save the real API key if it exists
    real_api_key = os.getenv("OPENAI_API_KEY")
    
    # Set an invalid API key to force fallback
    os.environ["OPENAI_API_KEY"] = "sk-invalid-key"
    
    # Initialize agent with the invalid key
    agent = DefiBrowserAgent(headless=False)
    
    # Try to collect data, which should fall back to direct automation
    print("\nAttempting to collect data with invalid API key...")
    print("This should automatically fall back to direct automation")
    
    yields = await agent.collect_defi_yields()
    
    if not yields.empty:
        print(f"Successfully collected {len(yields)} yield opportunities using fallback")
        print("\nTop 5 yields:")
        print(yields.head(5))
    else:
        print("Failed to collect data even with fallback")
    
    # Restore the real API key if it existed
    if real_api_key:
        os.environ["OPENAI_API_KEY"] = real_api_key
    
    # Print agent stats
    print("\nAgent Stats:")
    print(json.dumps(agent.get_stats(), indent=2))
    
    return yields

async def compare_results(direct_results, api_results):
    """Compare results from direct and API-based automation"""
    if direct_results is None or api_results is None:
        print("\nCannot compare results - one or both methods failed")
        return
    
    print("\n" + "=" * 80)
    print("COMPARING DIRECT VS API-BASED RESULTS")
    print("=" * 80)
    
    # Basic comparison
    print(f"Direct automation collected {len(direct_results)} records")
    print(f"API-based automation collected {len(api_results)} records")
    
    # Compare columns
    direct_cols = set(direct_results.columns)
    api_cols = set(api_results.columns)
    
    print("\nColumns in direct results:", direct_cols)
    print("Columns in API results:", api_cols)
    
    # Common columns
    common_cols = direct_cols.intersection(api_cols)
    print("\nCommon columns:", common_cols)
    
    # If there are common columns, compare some values
    if common_cols and 'project' in common_cols:
        print("\nProjects in direct results:", direct_results['project'].unique())
        print("Projects in API results:", api_results['project'].unique())
    
    # Save comparison to CSV
    if 'project' in common_cols and 'pool' in common_cols:
        # Try to merge on project and pool
        merged = pd.merge(
            direct_results, 
            api_results,
            on=['project', 'pool'],
            suffixes=('_direct', '_api'),
            how='outer'
        )
        
        merged.to_csv("comparison_results.csv", index=False)
        print("\nSaved comparison to comparison_results.csv")

async def main():
    """Main function to run the tests"""
    parser = argparse.ArgumentParser(description="Test the DefiBrowserAgent")
    parser.add_argument("--test", choices=["direct", "api", "fallback", "all"], 
                      default="direct", help="Test to run")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DEFIBROWSERAGENT TEST SCRIPT")
    print("=" * 80)
    print("This script tests the DefiBrowserAgent's capabilities")
    print("Results will be saved to CSV files for reference")
    
    direct_results = None
    api_results = None
    
    if args.test in ["direct", "all"]:
        direct_results, _ = await test_direct_automation()
    
    if args.test in ["api", "all"]:
        api_results, _ = await test_api_based_automation()
    
    if args.test in ["fallback", "all"]:
        await test_fallback_mechanism()
    
    if args.test == "all" and direct_results is not None and api_results is not None:
        await compare_results(direct_results, api_results)
    
    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 