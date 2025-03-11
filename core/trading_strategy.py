#!/usr/bin/env python3
"""
DEFIMIND Trading Strategy Module

This module contains trading strategies that analyze market data
and blockchain insights to generate investment signals.
"""

import os
import time
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from core.live_data_fetcher import LiveDataFetcher
from core.defimind_persistence import MarketDataStore, MemoryDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trading_strategy")

# Load environment variables
load_dotenv()

# Constants
TOTAL_INVESTMENT = float(os.getenv("TOTAL_INVESTMENT", "100"))
MAX_SINGLE_PROTOCOL_ALLOCATION = float(os.getenv("MAX_SINGLE_PROTOCOL_ALLOCATION", "0.5"))
RISK_TOLERANCE = float(os.getenv("RISK_TOLERANCE", "0.6"))  # 0-1 scale, higher means more risk
MIN_TVL_THRESHOLD = float(os.getenv("MIN_TVL_THRESHOLD", "1000000"))  # Minimum TVL in USD
MIN_LIQUIDITY_THRESHOLD = float(os.getenv("MIN_LIQUIDITY_THRESHOLD", "500000"))  # Minimum liquidity


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, market_store=None, memory_db=None):
        """Initialize the trading strategy"""
        self.market_store = market_store or MarketDataStore()
        self.memory_db = memory_db or MemoryDatabase()
        self.data_fetcher = None
        
    async def initialize(self):
        """Initialize connections and resources"""
        if not self.data_fetcher:
            self.data_fetcher = LiveDataFetcher()
            await self.data_fetcher.initialize()
        return True
        
    async def close(self):
        """Close connections and resources"""
        if self.data_fetcher:
            await self.data_fetcher.close()
            
    async def analyze_market(self):
        """Analyze current market conditions"""
        raise NotImplementedError("Subclasses must implement analyze_market()")
        
    async def generate_signals(self):
        """Generate trading signals"""
        raise NotImplementedError("Subclasses must implement generate_signals()")
        
    async def decide_allocations(self):
        """Decide investment allocations"""
        raise NotImplementedError("Subclasses must implement decide_allocations()")
        
    async def run_strategy(self):
        """Run the complete strategy workflow"""
        await self.initialize()
        try:
            analysis = await self.analyze_market()
            signals = await self.generate_signals()
            allocations = await self.decide_allocations()
            return {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "signals": signals,
                "allocations": allocations
            }
        finally:
            await self.close()


class YieldOptimizer(TradingStrategy):
    """Strategy focused on optimizing yield while managing risk"""
    
    async def analyze_market(self):
        """Analyze current market conditions with focus on yield opportunities"""
        logger.info("Analyzing market for yield opportunities...")
        
        # Get best pools from database
        best_pools = self.market_store.get_best_pools(min_tvl=MIN_TVL_THRESHOLD)
        
        # Get recent gas prices to estimate transaction costs
        gas_prices = self.market_store.get_historical_gas_prices(hours=24)
        avg_gas_price = np.mean([price for _, price in gas_prices]) if gas_prices else None
        
        # Calculate market metrics
        pool_count = len(best_pools)
        avg_apy = np.mean([pool[5] for pool in best_pools]) if pool_count > 0 else 0
        avg_tvl = np.mean([pool[3] for pool in best_pools]) if pool_count > 0 else 0
        avg_risk = np.mean([pool[5] for pool in best_pools]) if pool_count > 0 else 0
        
        # Check gas costs as factor in yield optimization
        gas_cost_factor = None
        if avg_gas_price:
            # Adjust this calculation based on your transaction frequency
            estimated_gas_units_per_month = 1000000  # Example: 1M gas units monthly for all transactions
            monthly_gas_cost = avg_gas_price * 1e-9 * estimated_gas_units_per_month * 30  # in ETH
            gas_cost_factor = monthly_gas_cost / TOTAL_INVESTMENT  # As percentage of investment
            logger.info(f"Estimated gas cost factor: {gas_cost_factor:.4f}")
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "pool_count": pool_count,
            "avg_apy": avg_apy,
            "avg_tvl": avg_tvl,
            "avg_risk": avg_risk,
            "gas_price_gwei": avg_gas_price,
            "gas_cost_factor": gas_cost_factor,
            "market_temperature": self._calculate_market_temperature()
        }
        
        logger.info(f"Market analysis complete: {json.dumps(analysis)}")
        return analysis
    
    def _calculate_market_temperature(self):
        """Calculate overall market temperature (hot/cold) based on metrics"""
        # This is a simplified placeholder implementation
        # In a real system, you'd want to look at more factors and historical trends
        
        # Get latest token prices
        token_prices = self.market_store.get_latest_token_prices(limit=10)
        
        # Get recent blocks to analyze transaction counts
        recent_blocks = self.market_store.get_recent_blocks(limit=20)
        
        # Calculate average transactions per block as activity indicator
        avg_tx_count = np.mean([block[3] for block in recent_blocks]) if recent_blocks else 0
        
        # Simple temperature scale: 0 (very cold) to 1 (very hot)
        # Placeholder logic - should be enhanced with real metrics
        if avg_tx_count > 200:
            return 0.8  # Hot market
        elif avg_tx_count > 100:
            return 0.6  # Warm market
        else:
            return 0.4  # Cool market
    
    async def generate_signals(self):
        """Generate trading signals based on yield and risk analysis"""
        logger.info("Generating trading signals...")
        
        signals = []
        
        # Get best pools
        pools = self.market_store.get_best_pools(min_tvl=MIN_TVL_THRESHOLD, limit=20)
        
        # Get recent gas prices
        gas_prices = self.market_store.get_historical_gas_prices(hours=24)
        avg_gas_price = np.mean([price for _, price in gas_prices]) if gas_prices else 0
        
        # Analyze each pool for signals
        for pool in pools:
            protocol, chain, pool_id, tvl, apy, risk_level = pool
            
            # Calculate risk-adjusted return
            risk_adjusted_return = apy / (risk_level + 0.1)  # Avoid division by zero
            
            # Check if gas costs would eat too much of the yield
            # This is a simplified calculation - actual implementation would be more complex
            estimated_annual_yield = TOTAL_INVESTMENT * (apy / 100)
            min_yield_threshold = 50  # $50 minimum annual yield to be worthwhile
            
            signal = {
                "protocol": protocol,
                "chain": chain,
                "pool_id": pool_id,
                "tvl_usd": tvl,
                "apy": apy,
                "risk_level": risk_level,
                "risk_adjusted_return": risk_adjusted_return,
                "estimated_annual_yield": estimated_annual_yield,
                "signal": None,
                "strength": 0,
                "reasoning": []
            }
            
            # Generate signal based on various factors
            if risk_level > RISK_TOLERANCE:
                signal["signal"] = "AVOID"
                signal["strength"] = 0.8
                signal["reasoning"].append(f"Risk level {risk_level} exceeds tolerance {RISK_TOLERANCE}")
            elif estimated_annual_yield < min_yield_threshold:
                signal["signal"] = "AVOID"
                signal["strength"] = 0.6
                signal["reasoning"].append(f"Annual yield ${estimated_annual_yield:.2f} below minimum threshold ${min_yield_threshold}")
            elif risk_adjusted_return > 0.1 and tvl > MIN_TVL_THRESHOLD:
                signal["signal"] = "BUY"
                signal["strength"] = min(0.9, risk_adjusted_return / 0.2)  # Scale strength but cap at 0.9
                signal["reasoning"].append(f"Strong risk-adjusted return ({risk_adjusted_return:.4f})")
                
                # Check if this is a top performer
                if risk_adjusted_return > 0.2:
                    signal["strength"] = 0.9
                    signal["reasoning"].append("Top performing pool in risk-adjusted terms")
            else:
                signal["signal"] = "NEUTRAL"
                signal["strength"] = 0.5
                signal["reasoning"].append("Average performance metrics")
            
            # Incorporate gas price considerations for low-value investments
            if avg_gas_price > 100 and estimated_annual_yield < 500:  # If gas is high and yield is modest
                if signal["signal"] == "BUY":
                    signal["signal"] = "NEUTRAL"
                    signal["strength"] = max(0.3, signal["strength"] - 0.3)
                    signal["reasoning"].append(f"High gas price ({avg_gas_price:.2f} Gwei) reduces viability for smaller allocations")
            
            signals.append(signal)
        
        # Sort by signal strength (strongest first) then by signal type (BUY > NEUTRAL > AVOID)
        signal_priority = {"BUY": 3, "NEUTRAL": 2, "AVOID": 1}
        signals.sort(key=lambda x: (signal_priority.get(x["signal"], 0), x["strength"]), reverse=True)
        
        # Log signals
        logger.info(f"Generated {len(signals)} trading signals")
        return signals
    
    async def decide_allocations(self):
        """Decide investment allocations based on signals"""
        logger.info("Deciding investment allocations...")
        
        # Get signals
        signals = await self.generate_signals()
        
        # Filter to only BUY signals
        buy_signals = [s for s in signals if s["signal"] == "BUY"]
        
        total_strength = sum(s["strength"] for s in buy_signals)
        allocations = []
        remaining_investment = TOTAL_INVESTMENT
        
        if total_strength > 0:
            # First pass: Calculate proportional allocations
            for signal in buy_signals:
                # Calculate proportional allocation
                allocation_percentage = (signal["strength"] / total_strength)
                
                # Apply maximum single protocol allocation limit
                allocation_percentage = min(allocation_percentage, MAX_SINGLE_PROTOCOL_ALLOCATION)
                
                # Calculate allocation amount
                allocation_amount = TOTAL_INVESTMENT * allocation_percentage
                
                # Save allocation
                allocations.append({
                    "protocol": signal["protocol"],
                    "chain": signal["chain"],
                    "pool_id": signal["pool_id"],
                    "percent": allocation_percentage,
                    "amount": allocation_amount,
                    "apy": signal["apy"],
                    "risk_level": signal["risk_level"],
                    "estimated_annual_yield": signal["estimated_annual_yield"],
                    "reasoning": signal["reasoning"]
                })
                
                # Update remaining investment
                remaining_investment -= allocation_amount
            
            # Second pass: Redistribute any remaining investment
            if remaining_investment > 0 and allocations:
                # Simple redistribution: add equally to all allocations
                extra_per_allocation = remaining_investment / len(allocations)
                for allocation in allocations:
                    allocation["amount"] += extra_per_allocation
                    allocation["percent"] = allocation["amount"] / TOTAL_INVESTMENT
        
        # If no buy signals, recommend holding cash
        if not allocations:
            allocations.append({
                "protocol": "CASH",
                "chain": None,
                "pool_id": None,
                "percent": 1.0,
                "amount": TOTAL_INVESTMENT,
                "apy": 0,
                "risk_level": 0,
                "estimated_annual_yield": 0,
                "reasoning": ["No attractive investment opportunities identified", "Holding as cash until market conditions improve"]
            })
        
        # Save investment decisions to database
        timestamp = int(time.time())
        for allocation in allocations:
            try:
                self.market_store.save_investment_decision({
                    "timestamp": timestamp,
                    "protocol": allocation["protocol"],
                    "chain": allocation["chain"] or "none",
                    "allocation_percentage": allocation["percent"],
                    "reason": json.dumps(allocation["reasoning"]),
                    "status": "pending"
                })
            except Exception as e:
                logger.error(f"Error saving investment decision: {e}")
        
        logger.info(f"Decided on {len(allocations)} investment allocations")
        return allocations


class SmartLiquidityProvider(TradingStrategy):
    """Strategy focused on providing liquidity to DEXes based on market conditions"""
    
    async def analyze_market(self):
        """Analyze market with focus on DEX liquidity opportunities"""
        logger.info("Analyzing DEX liquidity opportunities...")
        
        # This would be implemented with specific DEX analytics
        # For now, we'll use a simplified placeholder
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "dex_opportunities": await self._identify_dex_opportunities(),
            "gas_conditions": await self._analyze_gas_conditions(),
            "market_volatility": await self._estimate_market_volatility()
        }
        
        logger.info(f"Market analysis complete: {json.dumps(analysis)}")
        return analysis
    
    async def _identify_dex_opportunities(self):
        """Identify opportunities in DEX pools"""
        # In a real implementation, we would analyze:
        # - Trading volumes
        # - Fee generation
        # - Impermanent loss risks
        # - Pool compositions
        
        # For demonstration, we'll create placeholder data
        return {
            "uniswap_v3": {
                "opportunity_score": 0.75,
                "top_pools": ["ETH-USDC", "WBTC-ETH"]
            },
            "curve": {
                "opportunity_score": 0.65,
                "top_pools": ["3pool", "stETH"]
            }
        }
    
    async def _analyze_gas_conditions(self):
        """Analyze current gas conditions for optimal entry/exit"""
        try:
            gas_price = await self.data_fetcher.get_eth_gas_price()
            historical_gas = self.market_store.get_historical_gas_prices(hours=72)
            
            gas_percentile = 0.5  # Default middle
            if historical_gas and gas_price:
                historical_prices = [price for _, price in historical_gas]
                gas_percentile = sum(1 for p in historical_prices if p < gas_price) / len(historical_prices)
            
            return {
                "current_gas_gwei": gas_price,
                "gas_percentile": gas_percentile,
                "is_favorable": gas_percentile < 0.3  # Favorable if in bottom 30%
            }
        except Exception as e:
            logger.error(f"Error analyzing gas conditions: {e}")
            return {"error": str(e)}
    
    async def _estimate_market_volatility(self):
        """Estimate market volatility to assess impermanent loss risk"""
        # Placeholder function - in a real implementation, this would:
        # - Analyze price movements of key assets
        # - Calculate historical volatility
        # - Estimate correlation between paired assets
        return {
            "eth_volatility": 0.45,
            "btc_volatility": 0.38,
            "overall_market": "medium"
        }
    
    async def generate_signals(self):
        """Generate trading signals for liquidity provision"""
        # Implementation for liquidity provision strategy
        # This would be more complex in a real system
        return []
    
    async def decide_allocations(self):
        """Decide LP allocations based on signals"""
        # Implementation for liquidity provision allocations
        # This would be more complex in a real system
        return []


class MultiStrategyAllocator:
    """Combines multiple strategies for optimal allocation"""
    
    def __init__(self):
        """Initialize multiple strategies"""
        self.market_store = MarketDataStore()
        self.memory_db = MemoryDatabase()
        self.strategies = {
            "yield": YieldOptimizer(self.market_store, self.memory_db),
            "liquidity": SmartLiquidityProvider(self.market_store, self.memory_db)
        }
        self.strategy_weights = {
            "yield": 0.7,
            "liquidity": 0.3
        }
    
    async def run_allocator(self):
        """Run all strategies and combine results"""
        logger.info("Running multi-strategy allocator...")
        
        results = {}
        allocations = []
        
        # Run each strategy
        for name, strategy in self.strategies.items():
            try:
                result = await strategy.run_strategy()
                results[name] = result
                
                # Weight the allocations
                weight = self.strategy_weights.get(name, 0)
                if "allocations" in result:
                    for allocation in result["allocations"]:
                        allocation["amount"] *= weight
                        allocation["percent"] *= weight
                        allocations.append(allocation)
            except Exception as e:
                logger.error(f"Error running strategy {name}: {e}")
        
        # Combine and normalize allocations
        combined_allocations = self._combine_allocations(allocations)
        
        # Save combined allocations to database
        timestamp = int(time.time())
        for allocation in combined_allocations:
            try:
                self.market_store.save_investment_decision({
                    "timestamp": timestamp,
                    "protocol": allocation["protocol"],
                    "chain": allocation["chain"] or "none",
                    "allocation_percentage": allocation["percent"],
                    "reason": json.dumps(allocation.get("reasoning", [])),
                    "status": "pending"
                })
            except Exception as e:
                logger.error(f"Error saving combined investment decision: {e}")
        
        logger.info(f"Multi-strategy allocation complete: {len(combined_allocations)} allocations")
        return {
            "timestamp": datetime.now().isoformat(),
            "strategy_results": results,
            "combined_allocations": combined_allocations
        }
    
    def _combine_allocations(self, allocations):
        """Combine allocations from multiple strategies"""
        # Group by protocol and pool
        allocation_map = {}
        
        for allocation in allocations:
            key = f"{allocation['protocol']}:{allocation['chain']}:{allocation['pool_id']}"
            if key not in allocation_map:
                allocation_map[key] = {
                    "protocol": allocation["protocol"],
                    "chain": allocation["chain"],
                    "pool_id": allocation["pool_id"],
                    "amount": 0,
                    "percent": 0,
                    "apy": allocation["apy"],
                    "risk_level": allocation["risk_level"],
                    "reasoning": []
                }
            
            # Combine amounts and percentages
            allocation_map[key]["amount"] += allocation["amount"]
            allocation_map[key]["percent"] += allocation["percent"]
            
            # Combine reasoning
            if "reasoning" in allocation and allocation["reasoning"]:
                allocation_map[key]["reasoning"].extend(allocation["reasoning"])
        
        # Convert back to list and normalize if needed
        combined = list(allocation_map.values())
        total_percent = sum(a["percent"] for a in combined)
        
        if total_percent > 0:
            # Normalize percentages
            for allocation in combined:
                allocation["percent"] = allocation["percent"] / total_percent
                allocation["amount"] = TOTAL_INVESTMENT * allocation["percent"]
        
        return combined


async def run_trading_strategy():
    """Run the trading strategy as a standalone process"""
    logging.info("Starting trading strategy process...")
    
    # Choose which strategy implementation to use
    strategy = YieldOptimizer()
    
    try:
        # Run the strategy
        result = await strategy.run_strategy()
        
        # Output the results
        print(json.dumps(result, indent=2))
        
        return result
    except Exception as e:
        logger.error(f"Error running trading strategy: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the trading strategy
    asyncio.run(run_trading_strategy()) 