#!/usr/bin/env python3
"""
DeFi Browser Agent module for DEFIMIND

This module uses browser-use to enable DEFIMIND to browse DeFi platforms,
collect yield data, and interact with web3 interfaces.
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re
import pandas as pd
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/defi_browser_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DefiBrowserAgent")

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class DefiBrowserAgent:
    """
    DefiBrowserAgent uses browser-use to enable DEFIMIND to interact with
    DeFi platforms through browser automation.
    """
    
    def __init__(self, llm_api_key: Optional[str] = None, headless: bool = True):
        """
        Initialize the DefiBrowserAgent.
        
        Args:
            llm_api_key: API key for the LLM provider (defaults to env variable)
            headless: Whether to run the browser in headless mode
        """
        # Load environment variables
        load_dotenv()
        
        # Set API key
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        if not self.llm_api_key:
            logger.warning("No LLM API key provided. Agent functionality will be limited to direct automation.")
        
        self.headless = headless
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Track stats
        self.stats = {
            "platforms_visited": 0,
            "data_points_collected": 0,
            "errors": 0,
            "successful_interactions": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Initialize dependencies
        self._initialize_dependencies()
        
        logger.info("DefiBrowserAgent initialized")
    
    def _initialize_dependencies(self):
        """Check for and install necessary dependencies"""
        try:
            # Try importing required modules
            import browser_use
            from langchain_openai import ChatOpenAI
            logger.info("browser-use already installed")
        except ImportError:
            # Install browser-use if not available
            logger.info("Installing browser-use and dependencies")
            import subprocess
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "browser-use", "langchain-openai", "python-dotenv"], 
                               check=True, capture_output=True)
                subprocess.run([sys.executable, "-m", "playwright", "install"], 
                               check=True, capture_output=True)
                logger.info("browser-use installed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install browser-use: {e.stderr.decode()}")
                raise RuntimeError("Failed to install browser automation dependencies")
    
    async def setup_agent(self, model: str = "gpt-4o"):
        """
        Set up the browser-use agent with specified model.
        
        Args:
            model: The LLM model to use for the agent
        
        Returns:
            The configured agent
        """
        from browser_use import Agent
        from langchain_openai import ChatOpenAI
        
        try:
            llm = ChatOpenAI(model=model)
            
            # Configure the agent with browser settings
            agent = Agent(
                llm=llm,
                headless=self.headless,
            )
            
            logger.info(f"Set up browser agent using {model}")
            return agent
        except Exception as e:
            logger.error(f"Error setting up browser agent: {str(e)}")
            self.stats["errors"] += 1
            raise
    
    async def collect_defi_yields(self, protocols: List[str] = None) -> pd.DataFrame:
        """
        Collect yield data from DeFi protocols.
        
        Args:
            protocols: List of DeFi protocols to check (if None, uses defillama)
        
        Returns:
            DataFrame containing protocol, token, APY, TVL and other relevant data
        """
        # Try API-based method first if API key is available
        if self.llm_api_key:
            try:
                if not protocols:
                    # Default to using DefiLlama yield aggregator
                    return await self.collect_from_defillama()
                
                all_results = []
                for protocol in protocols:
                    try:
                        results = await self.collect_from_protocol(protocol)
                        all_results.extend(results)
                        self.stats["platforms_visited"] += 1
                    except Exception as e:
                        logger.error(f"Error collecting data from {protocol}: {str(e)}")
                        self.stats["errors"] += 1
                
                if all_results:
                    # Convert to DataFrame
                    df = pd.DataFrame(all_results)
                    
                    # Save to cache
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cache_path = os.path.join(self.cache_dir, f"yield_data_{timestamp}.csv")
                    df.to_csv(cache_path, index=False)
                    
                    logger.info(f"Collected {len(df)} yield opportunities from {len(protocols)} protocols")
                    self.stats["data_points_collected"] += len(df)
                    self.stats["successful_interactions"] += len(protocols)
                    
                    return df
                else:
                    return pd.DataFrame()
            except Exception as e:
                logger.warning(f"API-based collection failed: {str(e)}. Falling back to direct methods.")
        
        # Fallback to direct browser automation if API collection failed or no API key
        logger.info("Using direct browser automation (no API key required)")
        if not protocols or "defillama" in [p.lower() for p in protocols]:
            return await self.collect_from_defillama_direct()
        elif "aave" in [p.lower() for p in protocols]:
            return await self.collect_from_aave_direct()
        else:
            logger.error(f"No direct automation method available for specified protocols: {protocols}")
            return pd.DataFrame()
    
    async def collect_from_defillama(self) -> pd.DataFrame:
        """
        Collect yield data from DefiLlama's yield aggregator.
        
        Returns:
            DataFrame containing yield data from multiple protocols
        """
        logger.info("Collecting yield data from DefiLlama")
        
        try:
            agent = await self.setup_agent()
            
            # Define the task for the agent
            task = """
            Visit https://defillama.com/yields and extract data from the yield table.
            For each row, collect:
            1. Pool name
            2. Project name
            3. APY percentage
            4. TVL value
            5. Reward tokens
            
            Filter for stablecoin pools (USDC, USDT, DAI) and the top 10 non-stablecoin pools by TVL.
            Format the data as a JSON array of objects with the fields: pool, project, apy, tvl, reward_tokens
            """
            
            # Run the agent with the defined task
            result = await agent.run(task=task)
            
            # Extract data from the agent's response
            # The response likely contains a mix of text and JSON data
            # So we need to extract the JSON part
            json_data = self._extract_json_from_text(result)
            
            if json_data:
                # Convert to DataFrame
                df = pd.DataFrame(json_data)
                
                # Process numeric fields
                if 'apy' in df.columns:
                    df['apy'] = df['apy'].apply(self._convert_percentage_to_float)
                if 'tvl' in df.columns:
                    df['tvl'] = df['tvl'].apply(self._convert_tvl_to_float)
                
                # Save to cache
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cache_path = os.path.join(self.cache_dir, f"defillama_yields_{timestamp}.csv")
                df.to_csv(cache_path, index=False)
                
                logger.info(f"Collected {len(df)} yield opportunities from DefiLlama")
                self.stats["data_points_collected"] += len(df)
                self.stats["successful_interactions"] += 1
                
                return df
            else:
                logger.warning("No JSON data found in DefiLlama response")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error collecting data from DefiLlama: {str(e)}")
            self.stats["errors"] += 1
            # If we have no API key or it fails, try the direct automation approach
            if not self.llm_api_key:
                logger.info("Falling back to direct browser automation")
                return await self.collect_from_defillama_direct()
            raise
    
    async def collect_from_defillama_direct(self) -> pd.DataFrame:
        """
        Collect yield data from DefiLlama using direct Playwright automation (no API key required).
        
        Returns:
            DataFrame containing yield data from DefiLlama
        """
        logger.info("Collecting yield data from DefiLlama using direct browser automation")
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Navigate to DefiLlama yields page
                logger.info("Navigating to DefiLlama yields page")
                await page.goto("https://defillama.com/yields")
                
                # Wait for the table to load
                logger.info("Waiting for yield table to load")
                await page.wait_for_selector("table", timeout=30000)
                
                # Extract table data
                logger.info("Extracting yield data")
                table_data = await page.evaluate("""() => {
                    const table = document.querySelector('table');
                    if (!table) return [];
                    
                    // Get headers
                    const headers = Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim());
                    
                    // Get rows
                    const rows = Array.from(table.querySelectorAll('tbody tr'));
                    
                    // Get data from each row
                    return rows.slice(0, 30).map(row => {
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
                await browser.close()
                
                # Process the data
                if table_data:
                    logger.info(f"Successfully extracted {len(table_data)} yield opportunities")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(table_data)
                    
                    # Rename columns to match the expected format
                    column_mapping = {
                        'Pool': 'pool',
                        'Project': 'project',
                        'Chain': 'chain',
                        'APY': 'apy',
                        'TVL': 'tvl',
                        'Rewards': 'reward_tokens'
                    }
                    
                    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                    
                    # Save to cache
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cache_path = os.path.join(self.cache_dir, f"defillama_yields_direct_{timestamp}.csv")
                    df.to_csv(cache_path, index=False)
                    
                    logger.info(f"Collected {len(df)} yield opportunities from DefiLlama using direct automation")
                    self.stats["data_points_collected"] += len(df)
                    self.stats["successful_interactions"] += 1
                    
                    return df
                else:
                    logger.warning("No data extracted from DefiLlama")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error collecting data from DefiLlama using direct automation: {str(e)}")
            self.stats["errors"] += 1
            raise
    
    async def collect_from_aave_direct(self) -> pd.DataFrame:
        """
        Collect yield data from Aave using direct Playwright automation (no API key required).
        
        Returns:
            DataFrame containing supply and borrow rates from Aave
        """
        logger.info("Collecting yield data from Aave using direct browser automation")
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                # Launch browser
                browser = await p.chromium.launch(headless=self.headless)
                context = await browser.new_context()
                page = await context.new_page()
                
                # Navigate to Aave markets page
                logger.info("Navigating to Aave markets page")
                await page.goto("https://app.aave.com/markets/")
                
                # Wait for the markets to load
                logger.info("Waiting for markets data to load")
                await page.wait_for_selector('[data-cy="markets"]', timeout=60000)
                
                # Extract market data
                logger.info("Extracting market data")
                markets_data = await page.evaluate("""() => {
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
                            'pool': `Aave V3 ${asset}`,
                            'project': 'Aave',
                            'asset': asset,
                            'supply_apy': supplyApy,
                            'borrow_apy': borrowApy,
                            'total_supplied': totalSupplied,
                            'total_borrowed': totalBorrowed,
                            'chain': 'Ethereum'
                        };
                    });
                }""")
                
                # Close browser
                await browser.close()
                
                # Process the data
                if markets_data:
                    logger.info(f"Successfully extracted {len(markets_data)} Aave markets")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(markets_data)
                    
                    # Save to cache
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cache_path = os.path.join(self.cache_dir, f"aave_markets_direct_{timestamp}.csv")
                    df.to_csv(cache_path, index=False)
                    
                    logger.info(f"Collected {len(df)} markets from Aave using direct automation")
                    self.stats["data_points_collected"] += len(df)
                    self.stats["successful_interactions"] += 1
                    
                    return df
                else:
                    logger.warning("No data extracted from Aave")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error collecting data from Aave using direct automation: {str(e)}")
            self.stats["errors"] += 1
            raise
    
    async def collect_from_protocol(self, protocol: str) -> List[Dict[str, Any]]:
        """
        Collect yield data from a specific protocol.
        
        Args:
            protocol: The protocol name to collect data from
        
        Returns:
            List of dictionaries containing yield data
        """
        logger.info(f"Collecting yield data from {protocol}")
        
        try:
            agent = await self.setup_agent()
            
            # Define the task for the agent
            protocol_url = self._get_protocol_url(protocol)
            task = f"""
            Visit {protocol_url} and extract yield data.
            For each pool or asset, collect:
            1. Pool/asset name
            2. APY/APR percentage
            3. TVL (Total Value Locked)
            4. Any reward tokens or incentives
            
            For lending protocols, include both supply and borrow rates.
            Format the data as a JSON array of objects with the fields: 
            pool, project, apy, tvl, reward_tokens, chain (if available)
            
            Make sure all yield data is properly extracted from the page.
            If there are multiple chains supported, try to collect data from Ethereum mainnet.
            """
            
            # Run the agent with the defined task
            result = await agent.run(task=task)
            
            # Extract data from the agent's response
            json_data = self._extract_json_from_text(result)
            
            if json_data:
                # Add protocol name if not present
                for item in json_data:
                    if 'project' not in item:
                        item['project'] = protocol
                
                logger.info(f"Collected {len(json_data)} yield opportunities from {protocol}")
                self.stats["data_points_collected"] += len(json_data)
                self.stats["successful_interactions"] += 1
                
                return json_data
            else:
                logger.warning(f"No JSON data found in {protocol} response")
                return []
        except Exception as e:
            logger.error(f"Error collecting data from {protocol}: {str(e)}")
            self.stats["errors"] += 1
            raise
    
    def _get_protocol_url(self, protocol: str) -> str:
        """Get the URL for a specific protocol"""
        # Define URLs for popular protocols
        protocol_urls = {
            "aave": "https://app.aave.com/markets/",
            "compound": "https://app.compound.finance/",
            "curve": "https://curve.fi/#/ethereum/pools",
            "convex": "https://www.convexfinance.com/stake",
            "balancer": "https://app.balancer.fi/#/ethereum/pools",
            "uniswap": "https://app.uniswap.org/#/pools",
            "sushiswap": "https://www.sushi.com/pool",
            "yearn": "https://yearn.finance/vaults",
            "defillama": "https://defillama.com/yields"
        }
        
        # Return URL if found, otherwise use defillama search
        return protocol_urls.get(protocol.lower(), f"https://defillama.com/protocol/{protocol}")
    
    async def connect_wallet(self, wallet_type: str = "metamask") -> bool:
        """
        Simulate connecting to a wallet.
        
        Args:
            wallet_type: Type of wallet to connect to
        
        Returns:
            True if connected successfully, False otherwise
        """
        if not self.llm_api_key:
            logger.warning("Cannot connect wallet: No LLM API key provided")
            return False
        
        logger.info(f"Attempting to connect to {wallet_type} wallet")
        
        try:
            agent = await self.setup_agent()
            
            # Define the task for the agent
            task = f"""
            Do not actually connect to any real wallet.
            Instead, simulate the process of connecting to a {wallet_type} wallet:
            
            1. Describe what pages or interfaces would need to be navigated
            2. What buttons would be clicked
            3. What popups would appear
            4. How address selection would work
            
            Return a JSON object with:
            {{
                "success": true,
                "wallet_type": "{wallet_type}",
                "simulated_address": "0x...[random valid ETH address]",
                "steps_performed": [list of steps]
            }}
            """
            
            # Run the agent with the defined task
            result = await agent.run(task=task)
            
            # Extract data from the agent's response
            json_data = self._extract_json_from_text(result)
            
            if json_data and isinstance(json_data, list) and len(json_data) > 0:
                data = json_data[0]
            elif json_data and isinstance(json_data, dict):
                data = json_data
            else:
                logger.warning("Invalid JSON data format for wallet connection")
                return False
            
            success = data.get('success', False)
            
            if success:
                logger.info(f"Successfully simulated connection to {wallet_type} wallet")
                self.stats["successful_interactions"] += 1
            else:
                logger.warning(f"Failed to simulate connection to {wallet_type} wallet")
                self.stats["errors"] += 1
            
            return success
        except Exception as e:
            logger.error(f"Error connecting to wallet: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    async def analyze_defi_opportunity(self, protocol: str, pool: str) -> Dict[str, Any]:
        """
        Analyze a specific DeFi opportunity and provide insights.
        
        Args:
            protocol: The protocol name
            pool: The pool or asset name
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.llm_api_key:
            logger.warning("Cannot analyze opportunity: No LLM API key provided")
            return {"error": "No LLM API key provided for analysis"}
        
        logger.info(f"Analyzing opportunity: {protocol} - {pool}")
        
        try:
            agent = await self.setup_agent()
            
            # Define the task for the agent
            protocol_url = self._get_protocol_url(protocol)
            task = f"""
            Visit {protocol_url} and find information about the {pool} pool.
            
            Analyze this opportunity with regard to:
            1. Current APY/APR (supply and borrow rates if applicable)
            2. TVL (Total Value Locked)
            3. Risks associated with this pool
            4. Historical performance if available
            5. Reward tokens or incentives
            6. Impermanent loss risk (for AMM pools)
            7. Protocol security and audits
            
            Return your analysis as a JSON object with:
            {{
                "protocol": "{protocol}",
                "pool": "{pool}",
                "apy": [value],
                "tvl": [value],
                "risks": [list of risks],
                "rewards": [list of rewards],
                "impermanent_loss_risk": [low/medium/high],
                "security_rating": [1-5 scale],
                "analysis": [detailed text analysis]
            }}
            """
            
            # Run the agent with the defined task
            result = await agent.run(task=task)
            
            # Extract data from the agent's response
            json_data = self._extract_json_from_text(result)
            
            if json_data and isinstance(json_data, list) and len(json_data) > 0:
                data = json_data[0]
            elif json_data and isinstance(json_data, dict):
                data = json_data
            else:
                logger.warning(f"Invalid JSON data format for {protocol} - {pool} analysis")
                return {"error": "Failed to parse analysis results"}
            
            logger.info(f"Successfully analyzed {protocol} - {pool}")
            self.stats["successful_interactions"] += 1
            
            return data
        except Exception as e:
            logger.error(f"Error analyzing {protocol} - {pool}: {str(e)}")
            self.stats["errors"] += 1
            return {"error": str(e)}
    
    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON data from text response.
        
        Args:
            text: Text potentially containing JSON data
        
        Returns:
            List of dictionaries extracted from JSON
        """
        # Look for JSON objects in the text
        json_pattern = r'```(?:json)?\s*(\[.*?\]|\{.*?\})```'
        json_matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in json_matches:
            try:
                json_str = match.group(1)
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return [data]  # Return as a list for consistency
                return data
            except json.JSONDecodeError:
                continue
        
        # Try to find JSON without markdown code blocks
        json_pattern = r'\[[\s\S]*?\]|\{[\s\S]*?\}'
        json_matches = re.finditer(json_pattern, text, re.DOTALL)
        
        for match in json_matches:
            try:
                json_str = match.group(0)
                data = json.loads(json_str)
                if isinstance(data, dict):
                    return [data]  # Return as a list for consistency
                return data
            except json.JSONDecodeError:
                continue
        
        return []
    
    def _convert_percentage_to_float(self, percentage: str) -> float:
        """Convert a percentage string to a float"""
        if not percentage or not isinstance(percentage, str):
            return 0.0
        
        # Remove % and convert to float
        percentage = percentage.replace('%', '').replace(',', '')
        try:
            return float(percentage)
        except ValueError:
            return 0.0
    
    def _convert_tvl_to_float(self, tvl: str) -> float:
        """Convert a TVL string (e.g. $1.2M) to a float"""
        if not tvl or not isinstance(tvl, str):
            return 0.0
        
        # Remove $ and convert to float
        tvl = tvl.replace('$', '').replace(',', '')
        
        # Handle suffixes
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
        for suffix, multiplier in multipliers.items():
            if tvl.endswith(suffix):
                try:
                    return float(tvl[:-1]) * multiplier
                except ValueError:
                    return 0.0
        
        try:
            return float(tvl)
        except ValueError:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        # Update end time
        self.stats["end_time"] = datetime.now().isoformat()
        
        # Calculate duration
        start_time = datetime.fromisoformat(self.stats["start_time"])
        end_time = datetime.fromisoformat(self.stats["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.stats["duration_seconds"] = duration
        
        return self.stats


async def test_defi_browser_agent():
    """Test function for the DefiBrowserAgent"""
    # Initialize agent
    agent = DefiBrowserAgent(headless=False)
    
    # Collect yield data from DefiLlama using direct automation
    print("Collecting yield data from DefiLlama using direct automation...")
    yields = await agent.collect_from_defillama_direct()
    
    if not yields.empty:
        print(f"Successfully collected {len(yields)} yield opportunities")
        print("\nTop 5 yields:")
        print(yields.head(5))
    else:
        print("Failed to collect yield data from DefiLlama")
    
    # Print agent stats
    print("\nAgent Stats:")
    print(json.dumps(agent.get_stats(), indent=2))


if __name__ == "__main__":
    asyncio.run(test_defi_browser_agent()) 