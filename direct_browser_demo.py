#!/usr/bin/env python3
"""
Direct Browser Automation Demo

This script demonstrates browser automation capabilities using Playwright directly,
without requiring the browser-use package or an OpenAI API key.
It scrapes DeFi yield data from DefiLlama's yield page.
"""

import asyncio
import os
import json
import pandas as pd
from datetime import datetime
from playwright.async_api import async_playwright
import re

# Configure output directory
RESULTS_DIR = "browser_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


async def scrape_defillama_yields():
    """Scrape yield data from DefiLlama using direct Playwright automation"""
    print("Starting DefiLlama yield scraping...")
    
    async with async_playwright() as p:
        # Launch browser
        print("Launching browser...")
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Navigate to DefiLlama yields page
        print("Navigating to DefiLlama yields page...")
        await page.goto("https://defillama.com/yields")
        
        # Wait for the table to load
        print("Waiting for yield table to load...")
        await page.wait_for_selector("table", timeout=30000)
        
        # Extract table data
        print("Extracting yield data...")
        table_data = await page.evaluate("""() => {
            const table = document.querySelector('table');
            if (!table) return [];
            
            // Get headers
            const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
            
            // Get rows
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            
            // Get data from each row
            return rows.slice(0, 20).map(row => {
                const cells = Array.from(row.querySelectorAll('td'));
                const rowData = {};
                
                cells.forEach((cell, index) => {
                    if (headers[index]) {
                        rowData[headers[index]] = cell.textContent.trim();
                    }
                });
                
                return rowData;
            });
        }""")
        
        # Close browser
        print("Closing browser...")
        await browser.close()
        
        # Process the data
        if table_data:
            print(f"Successfully extracted {len(table_data)} yield opportunities")
            
            # Convert to DataFrame
            df = pd.DataFrame(table_data)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save raw data as JSON
            json_path = f"{RESULTS_DIR}/defillama_yields_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(table_data, f, indent=2)
            print(f"Saved raw data to {json_path}")
            
            # Save as CSV
            csv_path = f"{RESULTS_DIR}/defillama_yields_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved data to {csv_path}")
            
            # Display top 5 opportunities
            if len(df) >= 5:
                print("\nTop 5 Yield Opportunities:")
                print(df.head(5))
            else:
                print("\nYield Opportunities:")
                print(df)
            
            return df
        else:
            print("No data extracted from DefiLlama")
            return None


async def scrape_aave_markets():
    """Scrape lending markets data from Aave using direct Playwright automation"""
    print("\nStarting Aave markets scraping...")
    
    async with async_playwright() as p:
        # Launch browser
        print("Launching browser...")
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        # Navigate to Aave markets page
        print("Navigating to Aave markets page...")
        await page.goto("https://app.aave.com/markets/")
        
        # Wait for the markets to load
        print("Waiting for markets data to load...")
        await page.wait_for_selector('[data-cy="markets"]', timeout=60000)
        
        # Extract market data
        print("Extracting market data...")
        markets_data = await page.evaluate("""() => {
            // Helper function to extract numeric values
            const extractNumber = (text) => {
                if (!text) return null;
                const match = text.match(/([\\d,.]+)%?/);
                return match ? match[1].replace(/,/g, '') : null;
            };
            
            // Get all market rows
            const marketRows = document.querySelectorAll('[data-cy="marketListItemLink"]');
            if (!marketRows.length) return [];
            
            return Array.from(marketRows).map(row => {
                // Extract asset name
                const assetElement = row.querySelector('[data-cy="marketListItemName"]');
                const asset = assetElement ? assetElement.textContent.trim() : 'Unknown';
                
                // Extract total supplied
                const totalSuppliedElement = row.querySelector('[data-cy="marketListTotalSupplied"]');
                const totalSupplied = totalSuppliedElement ? totalSuppliedElement.textContent.trim() : 'N/A';
                
                // Extract supply APY
                const supplyApyElement = row.querySelector('[data-cy="marketListSupplyAPY"]');
                const supplyApy = supplyApyElement ? supplyApyElement.textContent.trim() : 'N/A';
                
                // Extract total borrowed
                const totalBorrowedElement = row.querySelector('[data-cy="marketListTotalBorrowed"]');
                const totalBorrowed = totalBorrowedElement ? totalBorrowedElement.textContent.trim() : 'N/A';
                
                // Extract borrow APY
                const borrowApyElement = row.querySelector('[data-cy="marketListBorrowAPY"]');
                const borrowApy = borrowApyElement ? borrowApyElement.textContent.trim() : 'N/A';
                
                return {
                    'Asset': asset,
                    'Total Supplied': totalSupplied,
                    'Supply APY': supplyApy,
                    'Total Borrowed': totalBorrowed,
                    'Borrow APY': borrowApy
                };
            });
        }""")
        
        # Close browser
        print("Closing browser...")
        await browser.close()
        
        # Process the data
        if markets_data:
            print(f"Successfully extracted {len(markets_data)} Aave markets")
            
            # Convert to DataFrame
            df = pd.DataFrame(markets_data)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save raw data as JSON
            json_path = f"{RESULTS_DIR}/aave_markets_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(markets_data, f, indent=2)
            print(f"Saved raw data to {json_path}")
            
            # Save as CSV
            csv_path = f"{RESULTS_DIR}/aave_markets_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved data to {csv_path}")
            
            # Display top 5 markets
            if len(df) >= 5:
                print("\nTop 5 Aave Markets:")
                print(df.head(5))
            else:
                print("\nAave Markets:")
                print(df)
            
            return df
        else:
            print("No data extracted from Aave")
            return None


async def main():
    """Main function to run the demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct Browser Automation Demo")
    parser.add_argument("--target", choices=["defillama", "aave", "both"], default="both", 
                        help="Target platform to scrape")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DIRECT BROWSER AUTOMATION DEMO")
    print("=" * 80)
    print("This demo uses Playwright to scrape DeFi yield data without requiring browser-use or an API key")
    print(f"Results will be saved to {RESULTS_DIR}/")
    print("=" * 80)
    
    if args.target in ["defillama", "both"]:
        await scrape_defillama_yields()
    
    if args.target in ["aave", "both"]:
        await scrape_aave_markets()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print(f"All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main()) 