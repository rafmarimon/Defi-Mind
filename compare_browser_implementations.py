#!/usr/bin/env python3
"""
Comparison of Browser Automation Approaches for DEFIMIND

This script compares our custom BrowserController with the browser-use integration.
It runs the same data collection tasks with both implementations and compares:
- Speed
- Accuracy
- Completeness of data
- Robustness
"""

import os
import time
import json
import asyncio
import argparse
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("browser_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrowserComparison")

# Create results directory
RESULTS_DIR = "comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

# Check for required modules
CUSTOM_AVAILABLE = False
BROWSER_USE_AVAILABLE = False

try:
    from core.browser_controller import BrowserController
    CUSTOM_AVAILABLE = True
    logger.info("✓ Custom BrowserController is available")
except ImportError:
    logger.warning("✗ Custom BrowserController is not available")

try:
    from core.defi_browser_agent import DefiBrowserAgent
    BROWSER_USE_AVAILABLE = True
    logger.info("✓ DefiBrowserAgent (browser-use) is available")
except ImportError:
    logger.warning("✗ DefiBrowserAgent is not available")

try:
    from browser_use import BrowserUse
    from langchain.chat_models import ChatOpenAI
    BROWSER_USE_RAW_AVAILABLE = True
    logger.info("✓ browser-use package is available")
except ImportError:
    logger.warning("✗ browser-use package is not available")
    BROWSER_USE_RAW_AVAILABLE = False


class ComparisonResult:
    """Class to store and analyze comparison results"""
    
    def __init__(self):
        self.results = {
            "custom": {},
            "browser_use": {},
            "browser_use_raw": {}
        }
    
    def add_result(self, implementation, task, start_time, end_time, success, data=None, error=None):
        """Add a result for a specific implementation and task"""
        duration = end_time - start_time
        
        self.results[implementation][task] = {
            "duration": duration,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "error": str(error) if error else None
        }
        
        if data is not None:
            if isinstance(data, pd.DataFrame):
                self.results[implementation][task]["row_count"] = len(data)
                self.results[implementation][task]["column_count"] = len(data.columns) if len(data) > 0 else 0
                
                # Store column names
                if len(data) > 0:
                    self.results[implementation][task]["columns"] = list(data.columns)
            elif isinstance(data, list):
                self.results[implementation][task]["row_count"] = len(data)
            elif isinstance(data, dict):
                self.results[implementation][task]["entry_count"] = len(data)
    
    def save_results(self):
        """Save the comparison results to a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{RESULTS_DIR}/comparison_results_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print a summary of the comparison results"""
        print("\n" + "=" * 80)
        print(" BROWSER IMPLEMENTATION COMPARISON RESULTS ")
        print("=" * 80)
        
        tasks = set()
        for impl in self.results:
            tasks.update(self.results[impl].keys())
        
        # Print header
        implementations = [impl for impl in ["custom", "browser_use", "browser_use_raw"] 
                          if self.results[impl]]
        print(f"{'Task':<25} | " + " | ".join(f"{impl.upper():<15}" for impl in implementations))
        print("-" * 25 + "+" + "+".join(["-" * 17] * len(implementations)))
        
        # Print results for each task
        for task in sorted(tasks):
            row = f"{task:<25} | "
            
            for impl in implementations:
                if task in self.results[impl]:
                    result = self.results[impl][task]
                    success = "✓" if result["success"] else "✗"
                    duration = f"{result['duration']:.2f}s"
                    
                    row_count = result.get("row_count", "N/A")
                    row += f"{success} {duration:<6} {row_count:<5} | "
                else:
                    row += f"{'N/A':<15} | "
            
            print(row)
        
        print("\nLegend: ✓ Success | ✗ Failure | Duration | Row Count")
        print("=" * 80)


async def run_custom_browser(task, headless=True):
    """Run a task with our custom BrowserController"""
    if not CUSTOM_AVAILABLE:
        return False, None, "Custom BrowserController not available"
    
    logger.info(f"Running custom browser for task: {task}")
    browser = BrowserController(headless=headless)
    
    try:
        if task == "defillama_yields":
            # Navigate to DefiLlama yields page
            await browser.navigate("https://defillama.com/yields")
            await asyncio.sleep(5)  # Wait for page to load
            
            # Extract the yield table data
            selector = ".sc-beqWaB.bpXRKw"  # This is the DefiLlama yields table class
            data = await browser.extract_table_data(selector)
            
            # Take a screenshot
            os.makedirs(f"{RESULTS_DIR}/screenshots", exist_ok=True)
            await browser.take_screenshot(f"{RESULTS_DIR}/screenshots/custom_defillama.png")
            
            # Convert to DataFrame
            if data and len(data) > 0:
                df = pd.DataFrame(data[1:], columns=data[0])
                return True, df, None
            else:
                return False, None, "No data extracted"
            
        elif task == "aave_markets":
            # Navigate to Aave markets page
            await browser.navigate("https://app.aave.com/markets/")
            await asyncio.sleep(10)  # Aave takes longer to load
            
            # Extract market data
            selector = ".MuiTableBody-root"  # Aave's table body
            data = await browser.extract_table_data(selector)
            
            # Take a screenshot
            os.makedirs(f"{RESULTS_DIR}/screenshots", exist_ok=True)
            await browser.take_screenshot(f"{RESULTS_DIR}/screenshots/custom_aave.png")
            
            # Convert to DataFrame
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                return True, df, None
            else:
                return False, None, "No data extracted"
            
        else:
            return False, None, f"Unknown task: {task}"
    
    except Exception as e:
        logger.error(f"Error in custom browser for task {task}: {str(e)}")
        # Take error screenshot
        try:
            os.makedirs(f"{RESULTS_DIR}/screenshots", exist_ok=True)
            await browser.take_screenshot(f"{RESULTS_DIR}/screenshots/custom_{task}_error.png")
        except:
            pass
        return False, None, str(e)
    
    finally:
        await browser.close()


async def run_browser_use_agent(task, headless=True):
    """Run a task with our DefiBrowserAgent (browser-use integration)"""
    if not BROWSER_USE_AVAILABLE:
        return False, None, "DefiBrowserAgent not available"
    
    logger.info(f"Running DefiBrowserAgent for task: {task}")
    agent = DefiBrowserAgent(headless=headless)
    
    try:
        if task == "defillama_yields":
            # Use our agent to collect DefiLlama yields
            df = await agent.collect_from_defillama()
            return True, df, None
            
        elif task == "aave_markets":
            # Use our agent to collect Aave markets
            results = await agent.collect_from_protocol("aave")
            
            # Convert to DataFrame if it's a list
            if isinstance(results, list):
                df = pd.DataFrame(results)
                return True, df, None
            else:
                return True, results, None
            
        else:
            return False, None, f"Unknown task: {task}"
    
    except Exception as e:
        logger.error(f"Error in DefiBrowserAgent for task {task}: {str(e)}")
        return False, None, str(e)


async def run_browser_use_raw(task, headless=True):
    """Run a task with raw browser-use package"""
    if not BROWSER_USE_RAW_AVAILABLE:
        return False, None, "browser-use package not available"
    
    if not os.getenv("OPENAI_API_KEY"):
        return False, None, "OPENAI_API_KEY not set"
    
    logger.info(f"Running raw browser-use for task: {task}")
    browser = BrowserUse(headless=headless)
    
    try:
        # Set up the LLM
        llm_model = os.getenv("LLM_MODEL", "gpt-4o")
        llm = ChatOpenAI(model=llm_model, temperature=0)
        
        # Create the agent with a task-specific prompt
        if task == "defillama_yields":
            system_prompt = """You are a DeFi yield expert that helps users collect data from DeFi platforms.
            Your task is to navigate to DefiLlama's yields page and extract the top 20 yield opportunities.
            For each opportunity, collect:
            - Protocol name
            - Pool name
            - APY percentage
            - TVL (Total Value Locked)
            - Any risk indicators
            
            Format the data in a markdown table for easy parsing.
            """
            
            agent = browser.create_agent(system_prompt=system_prompt, llm=llm)
            
            # Define the query
            query = """Go to https://defillama.com/yields and extract the top 20 yield opportunities.
            Make sure to include the protocol name, pool name, APY percentage, and TVL for each opportunity.
            Present the results in a markdown table with headers.
            """
            
        elif task == "aave_markets":
            system_prompt = """You are a DeFi lending market expert that helps users collect data from lending platforms.
            Your task is to navigate to Aave's markets page and extract information about all available markets.
            For each market, collect:
            - Asset name
            - Supply APY
            - Borrow APY (variable)
            - Borrow APY (stable) if available
            - Total supplied
            - Total borrowed
            
            Format the data in a markdown table for easy parsing.
            """
            
            agent = browser.create_agent(system_prompt=system_prompt, llm=llm)
            
            # Define the query
            query = """Go to https://app.aave.com/markets/ and extract information about all available markets.
            Make sure to include the asset name, supply APY, borrow APY (variable and stable), total supplied, and total borrowed.
            Present the results in a markdown table with headers.
            """
            
        else:
            return False, None, f"Unknown task: {task}"
        
        # Run the agent
        response = await agent.ainvoke({"input": query})
        result = response["output"]
        
        # Parse tables from the result
        tables = browser.utils.extract_tables_from_markdown(result)
        if tables and len(tables) > 0:
            # Convert the first table to DataFrame
            df = pd.DataFrame(tables[0]['data'], columns=tables[0]['header'])
            return True, df, None
        else:
            return False, result, "No tables found in response"
    
    except Exception as e:
        logger.error(f"Error in raw browser-use for task {task}: {str(e)}")
        return False, None, str(e)


async def compare_implementations(tasks, headless=True):
    """Compare different browser automation implementations"""
    comparison = ComparisonResult()
    
    for task in tasks:
        logger.info(f"Starting comparison for task: {task}")
        
        # Run custom browser
        if CUSTOM_AVAILABLE:
            start_time = time.time()
            success, data, error = await run_custom_browser(task, headless)
            end_time = time.time()
            
            comparison.add_result("custom", task, start_time, end_time, success, data, error)
            
            if success and isinstance(data, pd.DataFrame):
                os.makedirs(f"{RESULTS_DIR}/data", exist_ok=True)
                data.to_csv(f"{RESULTS_DIR}/data/custom_{task}.csv", index=False)
        
        # Run DefiBrowserAgent
        if BROWSER_USE_AVAILABLE:
            start_time = time.time()
            success, data, error = await run_browser_use_agent(task, headless)
            end_time = time.time()
            
            comparison.add_result("browser_use", task, start_time, end_time, success, data, error)
            
            if success and isinstance(data, pd.DataFrame):
                os.makedirs(f"{RESULTS_DIR}/data", exist_ok=True)
                data.to_csv(f"{RESULTS_DIR}/data/agent_{task}.csv", index=False)
        
        # Run raw browser-use
        if BROWSER_USE_RAW_AVAILABLE:
            start_time = time.time()
            success, data, error = await run_browser_use_raw(task, headless)
            end_time = time.time()
            
            comparison.add_result("browser_use_raw", task, start_time, end_time, success, data, error)
            
            if success and isinstance(data, pd.DataFrame):
                os.makedirs(f"{RESULTS_DIR}/data", exist_ok=True)
                data.to_csv(f"{RESULTS_DIR}/data/raw_{task}.csv", index=False)
    
    # Save and print results
    comparison.save_results()
    comparison.print_summary()
    
    return comparison


async def main():
    """Main function for running the comparison"""
    parser = argparse.ArgumentParser(description="Compare browser automation implementations")
    parser.add_argument("--tasks", nargs="+", default=["defillama_yields", "aave_markets"], 
                      help="Tasks to run")
    parser.add_argument("--visible", action="store_true", help="Run browser in visible mode")
    
    args = parser.parse_args()
    headless = not args.visible
    
    print("=" * 80)
    print(" BROWSER IMPLEMENTATION COMPARISON ")
    print("=" * 80)
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Browser mode: {'Visible' if not headless else 'Headless'}")
    print("=" * 80)
    
    # Check if we have the necessary implementations
    if not any([CUSTOM_AVAILABLE, BROWSER_USE_AVAILABLE, BROWSER_USE_RAW_AVAILABLE]):
        logger.error("No browser implementations available. Please install the required packages.")
        return
    
    # Run the comparison
    await compare_implementations(args.tasks, headless)


if __name__ == "__main__":
    asyncio.run(main()) 