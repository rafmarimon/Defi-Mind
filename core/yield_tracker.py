#!/usr/bin/env python3
"""
DEFIMIND Yield Tracker

This module tracks and compares yields across various DeFi platforms,
helping the agent identify the best opportunities for yield optimization.
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("defimind.yield_tracker")

class YieldTracker:
    """Tracks and compares yields across DeFi platforms"""
    
    def __init__(self, db_connection=None):
        """
        Initialize the yield tracker
        
        Args:
            db_connection: Optional database connection for persisting yield data
        """
        self.db = db_connection
        self.platforms = [
            "aave", "compound", "curve", "uniswap", 
            "balancer", "yearn", "convex", "lido"
        ]
        
        # API keys and endpoints
        self.defillama_endpoint = "https://yields.llama.fi/pools"
        self.coingecko_endpoint = "https://api.coingecko.com/api/v3"
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY", "")
        
        # Cache for yield data
        self.cache = {
            "all_yields": None,
            "cache_time": None
        }
        
        logger.info(f"YieldTracker initialized with {len(self.platforms)} platforms")
    
    def fetch_current_yields(self, force_refresh=False):
        """
        Fetch current APY data across all tracked platforms
        
        Args:
            force_refresh: Whether to force a refresh of cached data
            
        Returns:
            Dictionary of yield data by platform
        """
        # Check cache first
        if not force_refresh and self.cache["all_yields"] and self.cache["cache_time"]:
            cache_age = datetime.now() - self.cache["cache_time"]
            if cache_age < timedelta(hours=1):  # Cache for 1 hour
                logger.info(f"Using cached yield data ({cache_age.seconds // 60} minutes old)")
                return self.cache["all_yields"]
        
        try:
            # Use DefiLlama Yields API as primary source
            yields_data = self._fetch_defillama_yields()
            
            # Organize by platform
            platform_yields = {}
            for platform in self.platforms:
                platform_yields[platform] = [
                    pool for pool in yields_data 
                    if pool.get("project", "").lower() == platform.lower()
                ]
            
            # Update cache
            self.cache["all_yields"] = platform_yields
            self.cache["cache_time"] = datetime.now()
            
            logger.info(f"Successfully fetched yield data for {len(yields_data)} pools across {len(platform_yields)} platforms")
            return platform_yields
        
        except Exception as e:
            logger.error(f"Error fetching current yields: {e}")
            
            # Fall back to mock data if API fails
            return self._generate_mock_yield_data()
    
    def _fetch_defillama_yields(self):
        """Fetch yield data from DefiLlama Yields API"""
        try:
            response = requests.get(self.defillama_endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            pools = data.get("data", [])
            
            # Process and clean the data
            processed_pools = []
            for pool in pools:
                processed_pool = {
                    "platform": pool.get("project", "unknown"),
                    "pool": pool.get("symbol", "unknown"),
                    "token": pool.get("symbol", "unknown").split("-")[0],  # Extract main token
                    "apy": pool.get("apy", 0),
                    "tvl_usd": pool.get("tvlUsd", 0),
                    "liquidity_usd": pool.get("tvlUsd", 0),
                    "il_risk": "high" if "-" in pool.get("symbol", "") else "low",  # Simple heuristic
                    "updated_at": datetime.now().isoformat()
                }
                processed_pools.append(processed_pool)
            
            return processed_pools
            
        except Exception as e:
            logger.error(f"Error fetching DefiLlama yield data: {e}")
            raise
    
    def _fetch_platform_yields(self, platform):
        """Platform-specific yield data fetchers"""
        if platform == "aave":
            return self._fetch_aave_yields()
        elif platform == "compound":
            return self._fetch_compound_yields()
        elif platform == "curve":
            return self._fetch_curve_yields()
        else:
            # Default to DefiLlama filtered data
            all_yields = self.fetch_current_yields()
            return all_yields.get(platform, [])
    
    def _fetch_aave_yields(self):
        """Fetch Aave-specific yield data"""
        try:
            # In a real implementation, this would use Aave's API directly
            # For now, filter from DefiLlama data
            all_yields = self.fetch_current_yields()
            return all_yields.get("aave", [])
        except Exception as e:
            logger.error(f"Error fetching Aave yields: {e}")
            return []
    
    def _fetch_compound_yields(self):
        """Fetch Compound-specific yield data"""
        try:
            # In a real implementation, this would use Compound's API directly
            all_yields = self.fetch_current_yields()
            return all_yields.get("compound", [])
        except Exception as e:
            logger.error(f"Error fetching Compound yields: {e}")
            return []
    
    def _fetch_curve_yields(self):
        """Fetch Curve-specific yield data"""
        try:
            # In a real implementation, this would use Curve's API directly
            all_yields = self.fetch_current_yields()
            return all_yields.get("curve", [])
        except Exception as e:
            logger.error(f"Error fetching Curve yields: {e}")
            return []
    
    def get_best_yields_by_token(self, min_liquidity=100000, token_filter=None):
        """
        Find the best yield for each token across platforms
        
        Args:
            min_liquidity: Minimum liquidity in USD to consider
            token_filter: Optional list of tokens to filter by
            
        Returns:
            DataFrame of best yields by token
        """
        try:
            # Get all yields
            all_yields = self.fetch_current_yields()
            
            # Flatten the data
            flattened = []
            for platform, pools in all_yields.items():
                for pool in pools:
                    pool_copy = pool.copy()
                    pool_copy['platform'] = platform.lower()
                    flattened.append(pool_copy)
            
            # Convert to DataFrame
            df = pd.DataFrame(flattened)
            
            # Apply filters
            if not df.empty:
                # Filter by minimum liquidity
                df = df[df['liquidity_usd'] >= min_liquidity]
                
                # Filter by tokens if specified
                if token_filter:
                    df = df[df['token'].isin(token_filter)]
                
                # Group by token and find max APY
                if not df.empty and 'token' in df.columns and 'apy' in df.columns:
                    best_yields = df.loc[df.groupby('token')['apy'].idxmax()]
                    return best_yields.sort_values('apy', ascending=False)
            
            # If empty or error, return empty DataFrame
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting best yields by token: {e}")
            return pd.DataFrame()
    
    def get_historical_yield_trends(self, days=30, tokens=None, platforms=None):
        """
        Get historical yield trends for analysis
        
        Args:
            days: Number of days of historical data
            tokens: Optional list of tokens to filter
            platforms: Optional list of platforms to filter
            
        Returns:
            DataFrame of historical yield data
        """
        # In a real implementation, this would query a database of historical yield data
        # For now, generate mock historical data
        try:
            if self.db:
                # Query historical data from database
                # Implementation would depend on your database schema
                pass
            
            # Fall back to mock data
            return self._generate_mock_historical_data(days, tokens, platforms)
            
        except Exception as e:
            logger.error(f"Error getting historical yield trends: {e}")
            return pd.DataFrame()
    
    def _generate_mock_yield_data(self):
        """Generate mock yield data for testing"""
        mock_data = {}
        
        # Standard tokens across platforms
        tokens = ["USDC", "ETH", "BTC", "DAI", "USDT"]
        
        for platform in self.platforms:
            pools = []
            
            # Generate basic yields for each token
            for token in tokens:
                # Base APY varies by platform and token
                if platform == "aave":
                    base_apy = 3.2 if token == "USDC" else 1.8 if token == "ETH" else 0.9
                elif platform == "compound":
                    base_apy = 3.4 if token == "USDC" else 1.5 if token == "ETH" else 0.8
                elif platform == "curve":
                    base_apy = 4.1 if token == "USDC" else 0.7 if token == "ETH" else 2.2
                else:
                    base_apy = np.random.uniform(0.5, 5.0)
                
                # Add some randomness
                apy = base_apy * np.random.uniform(0.9, 1.1)
                
                # Liquidity also varies by token and platform
                liquidity_base = 1000000 if token in ["USDC", "USDT", "DAI"] else 500000
                liquidity = liquidity_base * np.random.uniform(0.7, 1.3)
                
                pool = {
                    "pool": f"{token}-{platform}",
                    "token": token,
                    "apy": apy,
                    "tvl_usd": liquidity,
                    "liquidity_usd": liquidity,
                    "il_risk": "low",
                    "updated_at": datetime.now().isoformat()
                }
                pools.append(pool)
            
            # Add some platform-specific pools
            if platform == "curve":
                # Add some curve-specific pools (stablecoin pools)
                curve_pools = [
                    {
                        "pool": "3pool",
                        "token": "3CRV",
                        "apy": 4.5 * np.random.uniform(0.9, 1.1),
                        "tvl_usd": 300000000 * np.random.uniform(0.9, 1.1),
                        "liquidity_usd": 300000000 * np.random.uniform(0.9, 1.1),
                        "il_risk": "low",
                        "updated_at": datetime.now().isoformat()
                    },
                    {
                        "pool": "stETH",
                        "token": "stETH",
                        "apy": 3.2 * np.random.uniform(0.9, 1.1),
                        "tvl_usd": 500000000 * np.random.uniform(0.9, 1.1),
                        "liquidity_usd": 500000000 * np.random.uniform(0.9, 1.1),
                        "il_risk": "medium",
                        "updated_at": datetime.now().isoformat()
                    }
                ]
                pools.extend(curve_pools)
            
            mock_data[platform] = pools
        
        return mock_data
    
    def _generate_mock_historical_data(self, days=30, tokens=None, platforms=None):
        """Generate mock historical yield data"""
        # Filter tokens and platforms
        if tokens is None:
            tokens = ["USDC", "ETH", "BTC", "DAI", "USDT"]
        
        if platforms is None:
            platforms = self.platforms[:4]  # Limit to first 4 for simplicity
        
        # Generate dates
        end_date = datetime.now()
        dates = [end_date - timedelta(days=i) for i in range(days)]
        dates.reverse()  # Chronological order
        
        # Generate data
        data = []
        
        for platform in platforms:
            for token in tokens:
                # Base APY varies by platform and token
                if platform == "aave":
                    base_apy = 3.2 if token == "USDC" else 1.8 if token == "ETH" else 0.9
                elif platform == "compound":
                    base_apy = 3.4 if token == "USDC" else 1.5 if token == "ETH" else 0.8
                elif platform == "curve":
                    base_apy = 4.1 if token == "USDC" else 0.7 if token == "ETH" else 2.2
                else:
                    base_apy = np.random.uniform(0.5, 5.0)
                
                # Generate a trend with some randomness
                trend = np.linspace(-0.5, 0.5, days) * base_apy * 0.3  # Trend component
                noise = np.random.normal(0, base_apy * 0.05, days)  # Noise component
                
                for i, date in enumerate(dates):
                    apy = max(0.1, base_apy + trend[i] + noise[i])  # Ensure non-negative
                    
                    entry = {
                        "date": date.strftime("%Y-%m-%d"),
                        "platform": platform,
                        "token": token,
                        "apy": apy,
                        "timestamp": date.isoformat()
                    }
                    data.append(entry)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    
    def persist_yield_data(self, yield_data):
        """
        Persist yield data to database
        
        Args:
            yield_data: Yield data to persist
            
        Returns:
            Boolean indicating success
        """
        if not self.db:
            logger.warning("No database connection available for persisting yield data")
            return False
        
        try:
            # Implementation would depend on your database schema
            # self.db.store_yield_data(yield_data)
            return True
        except Exception as e:
            logger.error(f"Error persisting yield data: {e}")
            return False


# For testing
if __name__ == "__main__":
    tracker = YieldTracker()
    
    # Test fetching current yields
    print("\n=== CURRENT YIELDS ===")
    yields = tracker.fetch_current_yields()
    for platform, pools in yields.items():
        print(f"\n{platform.upper()} ({len(pools)} pools):")
        for i, pool in enumerate(pools[:3]):  # Show first 3 pools
            print(f"  {i+1}. {pool['pool']}: {pool['apy']:.2f}% APY (${pool['liquidity_usd']:,.0f} liquidity)")
        if len(pools) > 3:
            print(f"  ... and {len(pools) - 3} more pools")
    
    # Test getting best yields by token
    print("\n=== BEST YIELDS BY TOKEN ===")
    best_yields = tracker.get_best_yields_by_token()
    if not best_yields.empty:
        for _, row in best_yields.iterrows():
            print(f"{row['token']}: {row['apy']:.2f}% APY on {row['platform']} (${row['liquidity_usd']:,.0f} liquidity)")
    
    # Test historical trends
    print("\n=== HISTORICAL YIELD TRENDS (USDC) ===")
    hist_data = tracker.get_historical_yield_trends(days=7, tokens=["USDC"])
    if not hist_data.empty:
        usdc_data = hist_data[hist_data['token'] == 'USDC']
        platforms = usdc_data['platform'].unique()
        
        for platform in platforms:
            platform_data = usdc_data[usdc_data['platform'] == platform]
            print(f"\n{platform.upper()} USDC Yield Trend:")
            for _, row in platform_data.iterrows():
                print(f"  {row['date']}: {row['apy']:.2f}% APY") 