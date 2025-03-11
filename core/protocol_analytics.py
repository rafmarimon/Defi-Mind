#!/usr/bin/env python3
"""
DEFIMIND Protocol Analytics Module

Provides specialized analytics for major DeFi protocols.
"""

import os
import json
import time
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from web3 import Web3
from core.live_data_fetcher import LiveDataFetcher
from core.defimind_persistence import MarketDataStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("protocol_analytics")

# Load environment variables
load_dotenv()

# Load protocol configurations
PROTOCOL_CONFIG_PATH = os.getenv("PROTOCOL_CONFIG_PATH", "protocol_configs.json")
with open(PROTOCOL_CONFIG_PATH, "r") as f:
    PROTOCOL_CONFIGS = json.load(f)


class ProtocolAnalytics:
    """Base class for protocol-specific analytics"""
    
    def __init__(self, protocol_id, market_store=None):
        """Initialize protocol analytics"""
        self.protocol_id = protocol_id
        self.protocol_config = self._get_protocol_config(protocol_id)
        self.market_store = market_store or MarketDataStore()
        self.data_fetcher = None
        
    def _get_protocol_config(self, protocol_id):
        """Get protocol configuration"""
        protocols = PROTOCOL_CONFIGS.get("protocols", {})
        if protocol_id in protocols:
            return protocols[protocol_id]
        logger.warning(f"Protocol config not found for: {protocol_id}")
        return {}
        
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
            
    async def analyze(self):
        """Analyze protocol data - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement analyze()")


class AaveAnalytics(ProtocolAnalytics):
    """Analytics for the Aave lending protocol"""
    
    def __init__(self, market_store=None):
        """Initialize Aave analytics"""
        super().__init__("aave_v2", market_store)
        
    async def analyze(self):
        """Analyze Aave lending protocol"""
        logger.info("Analyzing Aave protocol...")
        
        await self.initialize()
        try:
            # Get Aave lending pool address from config
            lending_pool_address = self.protocol_config.get("addresses", {}).get("ethereum", {}).get("lending_pool")
            if not lending_pool_address:
                return {"error": "Aave lending pool address not found in config"}
            
            # Get data from blockchain
            wallet_data = await self.data_fetcher.analyze_wallet(lending_pool_address)
            
            # Get recent events (deposits, borrows, repayments, etc.)
            recent_events = await self._get_recent_events(lending_pool_address)
            
            # Get historical protocol data from database
            historical_data = self.market_store.get_protocol_data(lending_pool_address, limit=10)
            
            # Analyze health and risk metrics
            health_metrics = await self._analyze_health_metrics(lending_pool_address)
            
            # Compile results
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "protocol": "Aave V2",
                "address": lending_pool_address,
                "assets": self._analyze_assets(wallet_data),
                "recent_activity": self._analyze_activity(recent_events),
                "health_metrics": health_metrics,
                "trend_analysis": self._analyze_trends(historical_data),
                "recommendation": self._generate_recommendation(health_metrics)
            }
            
            # Save analysis to database
            self.market_store.save_market_analysis({
                "timestamp": int(time.time()),
                "analysis_type": "protocol_analysis",
                "data": analysis
            })
            
            logger.info(f"Aave analysis complete: {len(analysis['assets'])} assets analyzed")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing Aave: {e}")
            return {"error": str(e)}
        finally:
            await self.close()
            
    async def _get_recent_events(self, lending_pool_address):
        """Get recent events for Aave lending pool"""
        # Event signatures for Aave
        deposit_event_sig = "0xde6857219544bb5b7746f48ed30be6386fefc61b2f864cacf559893bf50fd951"  # Deposit event
        borrow_event_sig = "0xc6a898309e823ee50bac7c9c5d5c476290e5a0b77d38c1e53fcb73a68e0c3644"  # Borrow event
        
        # Get deposit events
        deposit_logs = await self.data_fetcher.get_contract_logs(
            lending_pool_address, 
            deposit_event_sig, 
            max_results=20
        )
        
        # Get borrow events
        borrow_logs = await self.data_fetcher.get_contract_logs(
            lending_pool_address, 
            borrow_event_sig, 
            max_results=20
        )
        
        # Combine and sort events by block number (descending)
        combined_logs = []
        if deposit_logs:
            for log in deposit_logs:
                log["event_type"] = "Deposit"
                combined_logs.append(log)
                
        if borrow_logs:
            for log in borrow_logs:
                log["event_type"] = "Borrow"
                combined_logs.append(log)
                
        combined_logs.sort(key=lambda x: int(x.get("blockNumber", "0x0"), 16), reverse=True)
        
        return combined_logs[:30]  # Return top 30 most recent events
    
    def _analyze_assets(self, wallet_data):
        """Analyze assets in Aave lending pool"""
        # In a real implementation, we would parse the token data to identify:
        # - Reserve assets (tokens available for borrowing)
        # - Utilization rates
        # - Interest rates
        
        assets = []
        if wallet_data and "tokens" in wallet_data:
            for token in wallet_data["tokens"]:
                # Simple placeholder analysis
                assets.append({
                    "token": token.get("name", "Unknown"),
                    "symbol": token.get("symbol", "???"),
                    "balance": token.get("balance", 0),
                    "estimated_liquidity_usd": token.get("balance", 0) * 1.0  # Would multiply by token price in real impl
                })
                
        # Sort by balance (highest first)
        assets.sort(key=lambda x: x["balance"], reverse=True)
        return assets
    
    def _analyze_activity(self, events):
        """Analyze recent activity in Aave"""
        if not events:
            return {"activity_level": "low", "recent_events": []}
            
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event.get("event_type", "Unknown")
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
        # Determine activity level
        total_events = len(events)
        if total_events > 20:
            activity_level = "high"
        elif total_events > 10:
            activity_level = "medium"
        else:
            activity_level = "low"
            
        return {
            "activity_level": activity_level,
            "event_counts": event_counts,
            "total_events": total_events,
            "recent_events": events[:5]  # Include 5 most recent events
        }
    
    async def _analyze_health_metrics(self, lending_pool_address):
        """Analyze health metrics for Aave"""
        # In a real implementation, we would:
        # - Calculate utilization rates
        # - Analyze liquidation risks
        # - Check collateralization ratios
        
        # For demonstration, using placeholder metrics
        return {
            "total_value_locked_usd": 8500000000,  # $8.5B TVL
            "utilization_rate": 0.65,  # 65% utilization
            "average_health_factor": 1.8,  # Average health factor (>1 is good)
            "liquidation_risk": "low",
            "market_risk_assessment": "medium"
        }
    
    def _analyze_trends(self, historical_data):
        """Analyze historical trends"""
        if not historical_data:
            return {"trend": "unknown", "data_points": 0}
            
        # Extract TVL from historical data
        tvl_values = []
        timestamps = []
        
        for data_point in historical_data:
            timestamp = data_point.get("timestamp")
            data = data_point.get("data", {})
            
            if "health_metrics" in data and "total_value_locked_usd" in data["health_metrics"]:
                tvl = data["health_metrics"]["total_value_locked_usd"]
                tvl_values.append(tvl)
                timestamps.append(timestamp)
        
        if len(tvl_values) < 2:
            return {"trend": "unknown", "data_points": len(tvl_values)}
            
        # Calculate trend
        first_tvl = tvl_values[-1]  # Oldest
        last_tvl = tvl_values[0]    # Most recent
        
        if last_tvl > first_tvl * 1.05:  # 5% increase
            trend = "increasing"
        elif last_tvl < first_tvl * 0.95:  # 5% decrease
            trend = "decreasing"
        else:
            trend = "stable"
            
        return {
            "trend": trend,
            "data_points": len(tvl_values),
            "percent_change": ((last_tvl / first_tvl) - 1) * 100 if first_tvl > 0 else 0
        }
    
    def _generate_recommendation(self, health_metrics):
        """Generate recommendation based on health metrics"""
        # Simple logic for demonstration
        utilization = health_metrics.get("utilization_rate", 0)
        health_factor = health_metrics.get("average_health_factor", 0)
        liquidation_risk = health_metrics.get("liquidation_risk", "medium")
        
        if utilization > 0.8 and health_factor < 1.5:
            action = "AVOID"
            reasoning = "High utilization combined with low health factor increases protocol risk"
        elif utilization > 0.7 and liquidation_risk == "high":
            action = "AVOID"
            reasoning = "Elevated liquidation risk in current market conditions"
        elif 0.5 <= utilization <= 0.75 and health_factor >= 1.5:
            action = "INVEST"
            reasoning = "Good utilization with healthy collateralization metrics"
        elif utilization < 0.4:
            action = "NEUTRAL"
            reasoning = "Low utilization may indicate reduced yield opportunities"
        else:
            action = "NEUTRAL"
            reasoning = "Average metrics across the board"
            
        return {
            "action": action,
            "reasoning": reasoning,
            "utilization_factor": utilization,
            "health_factor": health_factor
        }


class UniswapAnalytics(ProtocolAnalytics):
    """Analytics for Uniswap V3 DEX"""
    
    def __init__(self, market_store=None):
        """Initialize Uniswap analytics"""
        super().__init__("uniswap_v3", market_store)
        
    async def analyze(self):
        """Analyze Uniswap V3 DEX"""
        logger.info("Analyzing Uniswap V3 protocol...")
        
        await self.initialize()
        try:
            # Get Uniswap V3 factory address from config
            factory_address = self.protocol_config.get("addresses", {}).get("ethereum", {}).get("factory")
            if not factory_address:
                return {"error": "Uniswap factory address not found in config"}
            
            # Get data from blockchain
            wallet_data = await self.data_fetcher.analyze_wallet(factory_address)
            
            # Get recent pool creation events
            pool_creation_events = await self._get_pool_creation_events(factory_address)
            
            # Analyze trading activity
            trading_activity = await self._analyze_trading_activity(factory_address)
            
            # Get gas price to estimate swap costs
            gas_price = await self.data_fetcher.get_eth_gas_price()
            
            # Compile results
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "protocol": "Uniswap V3",
                "address": factory_address,
                "recent_pools": self._analyze_pools(pool_creation_events),
                "trading_activity": trading_activity,
                "fee_analysis": self._analyze_fees(trading_activity),
                "gas_cost_analysis": self._analyze_gas_costs(gas_price),
                "recommendation": self._generate_recommendation(trading_activity)
            }
            
            # Save analysis to database
            self.market_store.save_market_analysis({
                "timestamp": int(time.time()),
                "analysis_type": "protocol_analysis",
                "data": analysis
            })
            
            logger.info("Uniswap V3 analysis complete")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing Uniswap V3: {e}")
            return {"error": str(e)}
        finally:
            await self.close()
    
    async def _get_pool_creation_events(self, factory_address):
        """Get pool creation events from Uniswap factory"""
        # PoolCreated event signature
        pool_created_sig = "0x783cca1c0412dd0d695e784568c96da2e9c22ff989357a2e8b1d9b2b4e6b7118"
        
        # Get recent pool creation events
        pool_logs = await self.data_fetcher.get_contract_logs(
            factory_address, 
            pool_created_sig, 
            max_results=10
        )
        
        return pool_logs
    
    async def _analyze_trading_activity(self, factory_address):
        """Analyze trading activity across Uniswap"""
        # In a real implementation, we would:
        # - Query multiple popular pools
        # - Analyze swap volumes
        # - Track price impacts
        
        # For demonstration, using placeholder data
        return {
            "daily_volume_usd": 1200000000,  # $1.2B daily volume
            "active_pools_count": 3500,
            "top_tokens_by_volume": ["ETH", "USDC", "USDT", "WBTC"],
            "avg_trade_size_usd": 5200,
            "fee_tiers": {
                "0.01%": 0.15,  # 15% of volume
                "0.05%": 0.25,  # 25% of volume
                "0.3%": 0.45,   # 45% of volume
                "1%": 0.15      # 15% of volume
            }
        }
    
    def _analyze_pools(self, pool_events):
        """Analyze recently created pools"""
        pools = []
        for event in pool_events:
            # In a real implementation, we would decode the event data
            # For now, using simple placeholder data
            pools.append({
                "pool_address": event.get("address", "unknown"),
                "block_number": int(event.get("blockNumber", "0x0"), 16),
                "transaction_hash": event.get("transactionHash", "unknown")
            })
        
        return {
            "recent_pool_count": len(pools),
            "pools": pools
        }
    
    def _analyze_fees(self, trading_activity):
        """Analyze fee generation in Uniswap"""
        daily_volume = trading_activity.get("daily_volume_usd", 0)
        fee_tiers = trading_activity.get("fee_tiers", {})
        
        # Calculate fee generation by tier
        fee_generation = {}
        total_daily_fees = 0
        
        for tier, volume_percentage in fee_tiers.items():
            fee_rate = float(tier.strip("%")) / 100
            tier_volume = daily_volume * volume_percentage
            tier_fees = tier_volume * fee_rate
            
            fee_generation[tier] = {
                "daily_volume_usd": tier_volume,
                "daily_fees_usd": tier_fees
            }
            
            total_daily_fees += tier_fees
        
        return {
            "total_daily_fees_usd": total_daily_fees,
            "fee_generation_by_tier": fee_generation,
            "annualized_fee_yield": (total_daily_fees * 365) / daily_volume if daily_volume > 0 else 0
        }
    
    def _analyze_gas_costs(self, gas_price):
        """Analyze gas costs for Uniswap operations"""
        if not gas_price:
            return {"error": "Gas price not available"}
            
        # Estimate gas costs for common operations
        # Gas usage estimates
        swap_gas = 180000  # Simple swap gas usage
        add_liquidity_gas = 350000  # Add liquidity gas usage
        remove_liquidity_gas = 250000  # Remove liquidity gas usage
        
        # Calculate costs in ETH
        eth_price_usd = 2800  # Placeholder ETH price - would get from API in real implementation
        gas_price_eth = gas_price * 1e-9  # Convert Gwei to ETH
        
        swap_cost_eth = swap_gas * gas_price_eth
        swap_cost_usd = swap_cost_eth * eth_price_usd
        
        add_liquidity_cost_eth = add_liquidity_gas * gas_price_eth
        add_liquidity_cost_usd = add_liquidity_cost_eth * eth_price_usd
        
        remove_liquidity_cost_eth = remove_liquidity_gas * gas_price_eth
        remove_liquidity_cost_usd = remove_liquidity_cost_eth * eth_price_usd
        
        return {
            "gas_price_gwei": gas_price,
            "eth_price_usd": eth_price_usd,
            "operations": {
                "swap": {
                    "gas_units": swap_gas,
                    "cost_eth": swap_cost_eth,
                    "cost_usd": swap_cost_usd
                },
                "add_liquidity": {
                    "gas_units": add_liquidity_gas,
                    "cost_eth": add_liquidity_cost_eth,
                    "cost_usd": add_liquidity_cost_usd
                },
                "remove_liquidity": {
                    "gas_units": remove_liquidity_gas,
                    "cost_eth": remove_liquidity_cost_eth,
                    "cost_usd": remove_liquidity_cost_usd
                }
            },
            "minimum_profitable_trade": swap_cost_usd * 10  # 10x gas cost as rough estimate
        }
    
    def _generate_recommendation(self, trading_activity):
        """Generate recommendation for Uniswap"""
        daily_volume = trading_activity.get("daily_volume_usd", 0)
        active_pools = trading_activity.get("active_pools_count", 0)
        
        # Simple logic for demonstration
        if daily_volume > 1000000000 and active_pools > 3000:  # $1B+ volume, 3000+ pools
            action = "INVEST"
            reasoning = "High trading volume and active pool count indicate strong protocol activity"
        elif daily_volume > 500000000:  # $500M+ volume
            action = "NEUTRAL"
            reasoning = "Moderate trading volume suggests stable protocol activity"
        else:
            action = "AVOID"
            reasoning = "Low trading volume may indicate reduced fee generation potential"
            
        return {
            "action": action,
            "reasoning": reasoning,
            "volume_factor": daily_volume / 1000000000,  # Normalized to $1B
            "active_pools_factor": active_pools / 4000  # Normalized to 4000 pools
        }


class CompoundAnalytics(ProtocolAnalytics):
    """Analytics for Compound lending protocol"""
    
    def __init__(self, market_store=None):
        """Initialize Compound analytics"""
        super().__init__("compound", market_store)
        
    async def analyze(self):
        """Analyze Compound lending protocol"""
        logger.info("Analyzing Compound protocol...")
        
        await self.initialize()
        try:
            # Get Compound Comptroller address
            comptroller_address = self.protocol_config.get("addresses", {}).get("ethereum", {}).get("comptroller")
            if not comptroller_address:
                return {"error": "Compound comptroller address not found in config"}
                
            # Placeholder implementation - similar to Aave analytics
            # In a real implementation, would use Compound's specific contract interfaces
            
            # Get data from blockchain
            wallet_data = await self.data_fetcher.analyze_wallet(comptroller_address)
            
            # Get recent events
            recent_events = await self._get_recent_events(comptroller_address)
            
            # Placeholder health metrics (would calculate from on-chain data in real implementation)
            health_metrics = {
                "total_supply_usd": 7200000000,  # $7.2B supply
                "total_borrow_usd": 3800000000,  # $3.8B borrow
                "utilization_rate": 0.53,        # 53% utilization
                "comp_distribution_daily": 2300, # 2300 COMP distributed daily
                "risk_assessment": "medium-low"
            }
            
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "protocol": "Compound",
                "address": comptroller_address,
                "tokens": self._analyze_tokens(wallet_data),
                "recent_activity": self._analyze_activity(recent_events),
                "health_metrics": health_metrics,
                "recommendations": self._generate_recommendations(health_metrics)
            }
            
            # Save analysis to database
            self.market_store.save_market_analysis({
                "timestamp": int(time.time()),
                "analysis_type": "protocol_analysis",
                "data": analysis
            })
            
            logger.info("Compound analysis complete")
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing Compound: {e}")
            return {"error": str(e)}
        finally:
            await self.close()
    
    async def _get_recent_events(self, comptroller_address):
        """Get recent Compound events"""
        # MarketEntered event signature (user enters a market)
        market_entered_sig = "0x3ab23ab0d51cccc0c3085aec51f99f24f830f18b6661aa6b771a7644e1f4d4e9"
        
        # Get recent market entry events
        market_logs = await self.data_fetcher.get_contract_logs(
            comptroller_address, 
            market_entered_sig, 
            max_results=15
        )
        
        return market_logs
    
    def _analyze_tokens(self, wallet_data):
        """Analyze tokens in Compound"""
        tokens = []
        if wallet_data and "tokens" in wallet_data:
            for token in wallet_data["tokens"]:
                tokens.append({
                    "name": token.get("name", "Unknown"),
                    "symbol": token.get("symbol", "???"),
                    "balance": token.get("balance", 0)
                })
                
        return tokens
    
    def _analyze_activity(self, events):
        """Analyze Compound activity from events"""
        return {
            "event_count": len(events),
            "recent_events": events[:5] if events else []
        }
    
    def _generate_recommendations(self, health_metrics):
        """Generate recommendations for Compound"""
        utilization = health_metrics.get("utilization_rate", 0)
        
        if utilization > 0.7:
            action = "AVOID"
            reasoning = "High utilization may indicate supply-side yield compression"
        elif 0.4 <= utilization <= 0.65:
            action = "INVEST"
            reasoning = "Balanced utilization suggests optimal yield generation"
        else:
            action = "NEUTRAL"
            reasoning = "Low utilization may indicate reduced borrowing demand"
            
        return {
            "action": action,
            "reasoning": reasoning,
            "preferred_assets": ["USDC", "ETH"] if action == "INVEST" else []
        }


class ProtocolAnalyzer:
    """Main class to analyze multiple protocols"""
    
    def __init__(self):
        """Initialize the protocol analyzer"""
        self.market_store = MarketDataStore()
        self.analyzers = {
            "aave_v2": AaveAnalytics(self.market_store),
            "uniswap_v3": UniswapAnalytics(self.market_store),
            "compound": CompoundAnalytics(self.market_store)
        }
    
    async def analyze_all(self):
        """Analyze all configured protocols"""
        logger.info("Starting analysis of all protocols...")
        
        results = {}
        for protocol_id, analyzer in self.analyzers.items():
            try:
                logger.info(f"Analyzing protocol: {protocol_id}")
                results[protocol_id] = await analyzer.analyze()
            except Exception as e:
                logger.error(f"Error analyzing {protocol_id}: {e}")
                results[protocol_id] = {"error": str(e)}
                
        # Calculate overall market assessment
        market_assessment = self._assess_market(results)
        
        # Compile final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "protocols_analyzed": len(results),
            "protocol_results": results,
            "market_assessment": market_assessment
        }
        
        # Save report
        self.market_store.save_market_analysis({
            "timestamp": int(time.time()),
            "analysis_type": "market_report",
            "data": report
        })
        
        logger.info(f"Protocol analysis complete: {len(results)} protocols analyzed")
        return report
    
    def _assess_market(self, results):
        """Assess overall market conditions from protocol results"""
        # Count recommendations by type
        recommendations = {
            "INVEST": 0,
            "NEUTRAL": 0,
            "AVOID": 0
        }
        
        # Extract health metrics
        tvl_values = []
        utilization_rates = []
        
        for protocol_id, analysis in results.items():
            # Skip protocols with errors
            if "error" in analysis:
                continue
                
            # Count recommendations
            if "recommendation" in analysis:
                action = analysis["recommendation"].get("action")
                if action in recommendations:
                    recommendations[action] += 1
            
            # Extract health metrics
            if "health_metrics" in analysis:
                health = analysis["health_metrics"]
                
                # TVL (may be called different things in different protocols)
                if "total_value_locked_usd" in health:
                    tvl_values.append(health["total_value_locked_usd"])
                elif "total_supply_usd" in health:
                    tvl_values.append(health["total_supply_usd"])
                    
                # Utilization
                if "utilization_rate" in health:
                    utilization_rates.append(health["utilization_rate"])
        
        # Calculate overall metrics
        total_tvl = sum(tvl_values)
        avg_utilization = np.mean(utilization_rates) if utilization_rates else None
        
        # Determine market sentiment
        if recommendations["INVEST"] > recommendations["AVOID"]:
            sentiment = "Bullish"
        elif recommendations["INVEST"] < recommendations["AVOID"]:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
            
        return {
            "total_tvl_usd": total_tvl,
            "average_utilization": avg_utilization,
            "protocol_recommendations": recommendations,
            "market_sentiment": sentiment,
            "top_protocols": self._get_top_protocols(results)
        }
    
    def _get_top_protocols(self, results):
        """Get top protocols based on analysis"""
        top_protocols = []
        
        for protocol_id, analysis in results.items():
            # Skip protocols with errors
            if "error" in analysis:
                continue
                
            # Only include protocols with INVEST recommendation
            if "recommendation" in analysis and analysis["recommendation"].get("action") == "INVEST":
                top_protocols.append({
                    "protocol_id": protocol_id,
                    "name": analysis.get("protocol", protocol_id),
                    "reasoning": analysis["recommendation"].get("reasoning", "")
                })
                
        return top_protocols


async def run_protocol_analysis():
    """Run protocol analysis as a standalone process"""
    logger.info("Starting protocol analysis process...")
    
    analyzer = ProtocolAnalyzer()
    
    try:
        result = await analyzer.analyze_all()
        
        # Output the results
        print(json.dumps(result, indent=2))
        
        return result
    except Exception as e:
        logger.error(f"Error running protocol analysis: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the protocol analysis
    asyncio.run(run_protocol_analysis()) 