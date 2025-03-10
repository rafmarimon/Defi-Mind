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

    async def initialize(self):
        """Initialize async session"""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT))

    async def close(self):
        """Close session"""
        if self.session:
            await self.session.close()

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
        
        return max(valid_pools, key=lambda p: p.apy)

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
