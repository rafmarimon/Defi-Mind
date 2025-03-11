import os
import json
import asyncio
import logging
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("yield_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_trading_bot")

# Load environment variables
load_dotenv()

# Check if we're in simulation mode
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "false").lower() == "true"

# Constants
DEFAULT_TIMEOUT = 15  # seconds
DEFAULT_CACHE_DURATION = timedelta(minutes=10)
DATA_DIR = os.path.expanduser("~/defi_scanner_data")
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class YieldPool:
    """Data class to store yield farming pool information"""
    pool_id: str
    project: str
    chain: str
    symbol: str
    tvl: float  # Total Value Locked in USD
    apy: float  # Annual Percentage Yield as decimal
    apr: Optional[float] = None  # Annual Percentage Rate
    reward_tokens: List[str] = field(default_factory=list)
    il_risk: float = 0.0  # Impermanent Loss Risk (0-1)
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_llama(cls, data: Dict[str, any]) -> Optional['YieldPool']:
        """Create a YieldPool from DefiLlama API response"""
        try:
            return cls(
                pool_id=data.get('pool', ''),
                project=data.get('project', '').lower(),
                chain=data.get('chain', '').lower(),
                symbol=data.get('symbol', ''),
                tvl=float(data.get('tvlUsd', 0)),
                apy=float(data.get('apy', 0)) / 100,  # Convert % to decimal
                apr=float(data.get('apr', 0)) / 100 if 'apr' in data else None,
                reward_tokens=data.get('rewardTokens', []),
                il_risk=cls._calculate_il_risk(data.get('symbol', ''))
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to convert data to YieldPool: {data}. Error: {e}")
            return None

    @staticmethod
    def _calculate_il_risk(symbol: str) -> float:
        """Calculate impermanent loss risk based on token symbols"""
        if '-' not in symbol:
            return 0.1  # Single asset staking
        
        tokens = symbol.split('-')
        stablecoins = ['usdc', 'usdt', 'dai', 'busd', 'usd']
        
        if all(any(stable in token.lower() for stable in stablecoins) for token in tokens):
            return 0.05  # Low risk for stablecoin pairs
        
        if any(stable in token.lower() for token in tokens for stable in stablecoins) and \
           any(major in symbol.lower() for major in ['btc', 'eth', 'wbtc', 'weth']):
            return 0.4  # Medium risk
            
        return 0.7  # Higher risk for other pairs


class PoolData:
    """Data class for pool information"""
    
    def __init__(self, name, protocol, apy, tvl, risk=0.5):
        self.name = name
        self.protocol = protocol
        self.apy = apy  # Annual percentage yield (as a decimal, e.g. 0.10 for 10%)
        self.tvl = tvl  # Total value locked in USD
        self.risk = risk  # Risk score from 0.0 to 1.0


class YieldScanner:
    """Enhanced yield scanner with async capabilities"""
    def __init__(self):
        self.cache = {}
        self.session = None
        self.supported_protocols = {
            'pancakeswap': ['bsc', 'ethereum'],
            'uniswap': ['ethereum', 'polygon', 'arbitrum', 'optimism'],
            'aave': ['ethereum', 'polygon', 'avalanche'],
            'curve': ['ethereum', 'polygon', 'avalanche', 'fantom'],
            'traderjoe': ['avalanche', 'arbitrum'],
            'sushi': ['ethereum', 'polygon', 'arbitrum', 'avalanche'],
            'balancer': ['ethereum', 'polygon', 'arbitrum'],
            'compound': ['ethereum'],
            'gmx': ['arbitrum', 'avalanche'],
            'velodrome': ['optimism'],
            'quickswap': ['polygon']
        }
        
        # For simulation mode
        self.simulated_pools = {}
        self.simulated_portfolio_value = 1000  # Start with $1000 in simulation
        
        if SIMULATION_MODE:
            # Create simulated data for testing
            self._initialize_simulated_data()
            logger.info("🧪 YieldScanner initialized in SIMULATION MODE with mock data")
    
    def _initialize_simulated_data(self):
        """Initialize simulated pool data for testing"""
        # PancakeSwap simulated pools
        self.simulated_pools["pancakeswap"] = [
            PoolData("CAKE-BNB", "pancakeswap", 0.215, 12500000, 0.4),
            PoolData("BUSD-USDT", "pancakeswap", 0.045, 25000000, 0.2),
            PoolData("ETH-BNB", "pancakeswap", 0.18, 8700000, 0.5)
        ]
        
        # TraderJoe simulated pools
        self.simulated_pools["traderjoe"] = [
            PoolData("JOE-AVAX", "traderjoe", 0.23, 5800000, 0.65),
            PoolData("USDC.e-USDT", "traderjoe", 0.035, 18000000, 0.15),
            PoolData("ETH-AVAX", "traderjoe", 0.19, 7200000, 0.55)
        ]
        
        # QuickSwap simulated pools
        self.simulated_pools["quickswap"] = [
            PoolData("QUICK-MATIC", "quickswap", 0.28, 4500000, 0.7),
            PoolData("USDC-DAI", "quickswap", 0.042, 15000000, 0.18),
            PoolData("ETH-MATIC", "quickswap", 0.21, 6300000, 0.6)
        ]
        
        # Add more protocols
        self.simulated_pools["uniswap"] = [
            PoolData("UNI-ETH", "uniswap", 0.185, 22000000, 0.5),
            PoolData("USDC-ETH", "uniswap", 0.06, 35000000, 0.3),
            PoolData("WBTC-ETH", "uniswap", 0.14, 18000000, 0.45)
        ]
        
        self.simulated_pools["curve"] = [
            PoolData("3pool", "curve", 0.052, 42000000, 0.2),
            PoolData("stETH", "curve", 0.087, 25000000, 0.35),
            PoolData("tricrypto", "curve", 0.11, 15000000, 0.5)
        ]
        
        self.simulated_pools["aave"] = [
            PoolData("USDC", "aave", 0.038, 58000000, 0.15),
            PoolData("ETH", "aave", 0.022, 45000000, 0.25),
            PoolData("MATIC", "aave", 0.051, 12000000, 0.4)
        ]
        
        self.last_gas_price = 40  # Simulated gas price in Gwei
        
    async def initialize(self):
        """Initialize async session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT))
            logger.info("✅ YieldScanner initialized with live API connections")
        else:
            logger.info("✅ YieldScanner initialized in simulation mode")
        return True

    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()
        return True

    async def get_defi_llama_pools(self) -> List[YieldPool]:
        """Fetch all pools from DefiLlama"""
        async def fetch_data():
            url = "https://yields.llama.fi/pools"
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

                if not isinstance(data, dict) or "data" not in data:
                    logger.error("❌ Unexpected response format from DefiLlama")
                    return []

                pools = []
                for entry in data["data"]:
                    pool = YieldPool.from_llama(entry)
                    if pool:
                        pools.append(pool)

                return pools

        return await fetch_data()

    async def get_protocol_pools(self, protocol: str) -> List[YieldPool]:
        """Get all pools for a specific protocol"""
        all_pools = await self.get_defi_llama_pools()
        return [p for p in all_pools if protocol in p.project]

    async def get_best_apy_for_protocol(self, protocol: str, min_tvl: float = 100000) -> Optional[YieldPool]:
        """Get highest APY pool for a protocol with minimum TVL"""
        pools = await self.get_protocol_pools(protocol)
        valid_pools = [p for p in pools if p.tvl >= min_tvl]
        
        if not valid_pools:
            logger.warning(f"No valid pools found for {protocol}")
            return None
        
        if SIMULATION_MODE:
            # Return simulated data
            if protocol in self.simulated_pools:
                # Add some randomness to simulate market changes
                for pool in self.simulated_pools[protocol]:
                    # Random fluctuation of ±15%
                    fluctuation = random.uniform(0.85, 1.15)
                    pool.apy = pool.apy * fluctuation
                    
                # Sort by APY and return the best
                return sorted(self.simulated_pools[protocol], key=lambda x: x.apy, reverse=True)[0]
            else:
                logger.warning(f"⚠️ Protocol {protocol} not found in simulated data")
                return None
        
        # Real implementation would connect to APIs here
        # For now we'll return None for any real requests
        logger.warning(f"⚠️ Live data for {protocol} not implemented yet")
        return None

    async def scan_all_protocols(self, min_tvl: float = 500000) -> Dict[str, YieldPool]:
        """Scan all supported protocols for their best yield opportunities"""
        result = {}
        tasks = {protocol: self.get_best_apy_for_protocol(protocol, min_tvl) for protocol in self.supported_protocols}

        for protocol, task in tasks.items():
            try:
                best_pool = await task
                if best_pool:
                    result[protocol] = best_pool
            except Exception as e:
                logger.error(f"Error scanning {protocol}: {e}")

        return result

    def save_opportunity_report(self, opportunities: Dict[str, YieldPool]) -> str:
        """Save yield opportunities to a CSV report"""
        if not opportunities:
            return "No opportunities found"
        
        data = [{
            "protocol": protocol,
            "chain": pool.chain,
            "symbol": pool.symbol,
            "apy": f"{pool.apy*100:.2f}%",
            "tvl": f"${pool.tvl:,.2f}",
            "il_risk": f"{pool.il_risk*100:.1f}%",
            "pool_id": pool.pool_id
        } for protocol, pool in opportunities.items()]
        
        df = pd.DataFrame(data)
        filename = f"yield_opportunities_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        return filepath

    async def update_gas_prices(self):
        """Update current gas prices from the network"""
        if SIMULATION_MODE:
            # Simulate gas price changes
            base_gas = 40  # Base gas price in Gwei
            fluctuation = random.uniform(0.7, 1.5)  # Random fluctuation
            self.last_gas_price = base_gas * fluctuation
            return self.last_gas_price
            
        # Real implementation would fetch from blockchain here
        # For now just return a placeholder
        return 50
        
    def get_portfolio_value(self):
        """Get the current portfolio value (for simulation)"""
        if SIMULATION_MODE:
            # Add random daily fluctuation of ±5%
            fluctuation = random.uniform(0.95, 1.05)
            self.simulated_portfolio_value *= fluctuation
            return self.simulated_portfolio_value
            
        # Real implementation would calculate from wallet balances
        return 1000  # Placeholder


async def main():
    """Main function"""
    scanner = YieldScanner()
    try:
        await scanner.initialize()
        
        logger.info("Scanning all protocols for yield opportunities...")
        opportunities = await scanner.scan_all_protocols()
        
        logger.info(f"Found {len(opportunities)} opportunities")
        for protocol, pool in opportunities.items():
            logger.info(f"{protocol}: {pool.symbol} - APY: {pool.apy*100:.2f}% - TVL: ${pool.tvl:,.2f}")
        
        report_path = scanner.save_opportunity_report(opportunities)
        logger.info(f"Report saved to {report_path}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await scanner.close()


if __name__ == "__main__":
    asyncio.run(main())
