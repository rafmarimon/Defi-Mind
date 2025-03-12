#!/usr/bin/env python3
"""
Simple Browser Controller Demo

This script demonstrates how to use our custom BrowserController
to scrape yield data from DefiLlama without requiring an OpenAI API key.
"""

import os
import asyncio
import json
import pandas as pd
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BrowserDemo")

# Create results directory
RESULTS_DIR = "browser_demo_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Import the browser controller
try:
    from core.browser_controller import BrowserController
    CONTROLLER_AVAILABLE = True
    logger.info("Browser controller is available")
except ImportError:
    CONTROLLER_AVAILABLE = False
    logger.error("Failed to import BrowserController")


async def scrape_defillama(headless=False):
    """
    Scrape yield data from DefiLlama using the custom BrowserController
    
    Args:
        headless: Whether to run the browser in headless mode
    """
    if not CONTROLLER_AVAILABLE:
        logger.error("BrowserController not available. Exiting.")
        return
    
    logger.info("Initializing browser controller")
    browser = BrowserController(headless=headless)
    
    try:
        # Navigate to DefiLlama yields page
        logger.info("Navigating to DefiLlama yields page")
        await browser.navigate("https://defillama.com/yields")
        
        # Wait for the page to load
        logger.info("Waiting for page to load...")
        await asyncio.sleep(5)
        
        # Take a screenshot
        logger.info("Taking screenshot")
        screenshot_path = f"{RESULTS_DIR}/defillama_screenshot.png"
        await browser.take_screenshot(screenshot_path)
        logger.info(f"Screenshot saved to {screenshot_path}")
        
        # Try to extract the yields table
        logger.info("Extracting yield data")
        # This selector might need to be adjusted based on the current DefiLlama structure
        selector = "table"
        
        # Extract table data
        table_data = await browser.extract_table_data(selector)
        
        if table_data and len(table_data) > 0:
            logger.info(f"Successfully extracted data with {len(table_data)} rows")
            
            # Assuming the first row contains headers
            headers = table_data[0]
            rows = table_data[1:]
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"{RESULTS_DIR}/defillama_yields_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            # Also save as JSON
            json_path = f"{RESULTS_DIR}/defillama_yields_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(df.to_dict(orient="records"), f, indent=2)
            
            logger.info(f"Results saved to CSV: {csv_path}")
            logger.info(f"Results saved to JSON: {json_path}")
            
            # Display top 5 rows
            print("\n=== Top 5 DeFi Yield Opportunities ===")
            print(df.head(5).to_string())
            
            return df
        else:
            logger.warning("No table data found or extraction failed")
            
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
    finally:
        # Close the browser
        logger.info("Closing browser")
        await browser.close()


async def scrape_aave(headless=False):
    """
    Scrape market data from Aave using the custom BrowserController
    
    Args:
        headless: Whether to run the browser in headless mode
    """
    if not CONTROLLER_AVAILABLE:
        logger.error("BrowserController not available. Exiting.")
        return
    
    logger.info("Initializing browser controller")
    browser = BrowserController(headless=headless)
    
    try:
        # Navigate to Aave markets page
        logger.info("Navigating to Aave markets page")
        await browser.navigate("https://app.aave.com/markets/")
        
        # Wait for the page to load (Aave takes longer)
        logger.info("Waiting for page to load...")
        await asyncio.sleep(10)
        
        # Take a screenshot
        logger.info("Taking screenshot")
        screenshot_path = f"{RESULTS_DIR}/aave_screenshot.png"
        await browser.take_screenshot(screenshot_path)
        logger.info(f"Screenshot saved to {screenshot_path}")
        
        # Try to extract the markets table
        logger.info("Extracting market data")
        # This selector might need adjustment based on Aave's current structure
        selector = "table"
        
        # Extract table data
        table_data = await browser.extract_table_data(selector)
        
        if table_data and len(table_data) > 0:
            logger.info(f"Successfully extracted data with {len(table_data)} rows")
            
            # Assuming the first row contains headers
            headers = table_data[0] if len(table_data) > 0 else []
            rows = table_data[1:] if len(table_data) > 1 else []
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=headers) if headers and rows else pd.DataFrame()
            
            if not df.empty:
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f"{RESULTS_DIR}/aave_markets_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                
                # Also save as JSON
                json_path = f"{RESULTS_DIR}/aave_markets_{timestamp}.json"
                with open(json_path, 'w') as f:
                    json.dump(df.to_dict(orient="records"), f, indent=2)
                
                logger.info(f"Results saved to CSV: {csv_path}")
                logger.info(f"Results saved to JSON: {json_path}")
                
                # Display top 5 rows
                print("\n=== Top 5 Aave Markets ===")
                print(df.head(5).to_string())
                
                return df
            else:
                logger.warning("Failed to create DataFrame from extracted data")
        else:
            logger.warning("No table data found or extraction failed")
            
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
    finally:
        # Close the browser
        logger.info("Closing browser")
        await browser.close()


async def main():
    """Main function to run the browser demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Browser Controller Demo")
    parser.add_argument("--target", choices=["defillama", "aave", "all"], 
                      default="defillama", help="Target to scrape")
    parser.add_argument("--headless", action="store_true", 
                      help="Run in headless mode (no browser window)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" BROWSER CONTROLLER DEMO ")
    print("=" * 80)
    print(f"Target: {args.target}")
    print(f"Headless mode: {args.headless}")
    print("Results will be saved to:", RESULTS_DIR)
    print("=" * 80)
    
    if args.target == "defillama" or args.target == "all":
        print("\n--- SCRAPING DEFILLAMA YIELDS ---")
        await scrape_defillama(headless=args.headless)
    
    if args.target == "aave" or args.target == "all":
        print("\n--- SCRAPING AAVE MARKETS ---")
        await scrape_aave(headless=args.headless)
    
    print("\n--- DEMO COMPLETE ---")
    print(f"All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main()) 