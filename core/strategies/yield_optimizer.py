#!/usr/bin/env python3
"""
DEFIMIND Yield Optimizer Strategy

This module implements a yield optimization strategy that identifies the best
yield opportunities across DeFi platforms and generates allocation plans to
maximize returns while managing risk.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("defimind.yield_optimizer")

class YieldOptimizer:
    """Strategy for optimizing yield across DeFi platforms"""
    
    def __init__(self, config=None):
        """
        Initialize the yield optimizer
        
        Args:
            config: Configuration dictionary with strategy parameters
        """
        try:
            from core.yield_tracker import YieldTracker
            self.yield_tracker = YieldTracker()
            logger.info("YieldTracker initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import YieldTracker: {e}")
            self.yield_tracker = None
        
        # Default configuration
        default_config = {
            'min_liquidity': 500000,  # $500k minimum pool size
            'max_position_size': 10000,  # Max $10k per position
            'min_position_size': 100,  # Min $100 per position
            'risk_tolerance': 'medium',  # low, medium, high
            'rebalance_threshold': 0.5,  # Rebalance if APY diff > 0.5%
            'min_position_time': 7,  # Min days to hold a position
            'max_positions': 10,  # Maximum number of positions
            'gas_threshold': 30,  # Gwei threshold for transactions
            'diversification': {
                'max_per_platform': 0.4,  # Max 40% in one platform
                'max_per_token': 0.5  # Max 50% in one token
            }
        }
        
        # Override defaults with provided config
        self.config = default_config
        if config:
            self.config.update(config)
        
        logger.info(f"YieldOptimizer initialized with risk tolerance: {self.config['risk_tolerance']}")
    
    def analyze_opportunities(self):
        """
        Find the best yield opportunities given constraints
        
        Returns:
            DataFrame with the best opportunities sorted by expected return
        """
        if not self.yield_tracker:
            logger.error("YieldTracker not available")
            return pd.DataFrame()
        
        try:
            # Get best yields across platforms
            best_yields = self.yield_tracker.get_best_yields_by_token(
                min_liquidity=self.config['min_liquidity']
            )
            
            if best_yields.empty:
                logger.warning("No yield opportunities found meeting criteria")
                return pd.DataFrame()
            
            # Filter based on risk profile
            filtered = self._apply_risk_filters(best_yields)
            
            # Calculate potential returns
            with_returns = self._calculate_expected_returns(filtered)
            
            if not with_returns.empty:
                return with_returns.sort_values('expected_return', ascending=False)
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error analyzing opportunities: {e}")
            return pd.DataFrame()
    
    def _apply_risk_filters(self, yields_df):
        """
        Apply risk filters based on configuration
        
        Args:
            yields_df: DataFrame of yield opportunities
            
        Returns:
            Filtered DataFrame
        """
        if yields_df.empty:
            return yields_df
        
        risk_level = self.config['risk_tolerance']
        
        try:
            # Make a copy to avoid modifying the original
            filtered_df = yields_df.copy()
            
            # Add a risk score column (lower is less risky)
            filtered_df['risk_score'] = 1.0
            
            # Adjust risk score based on platform (subjective rankings)
            platform_risk = {
                'aave': 1.0,      # Well-established, low risk
                'compound': 1.0,   # Well-established, low risk
                'curve': 1.2,      # Well-established, slightly higher complexity
                'uniswap': 1.4,    # Higher IL risk
                'balancer': 1.4,   # Higher complexity
                'yearn': 1.5,      # Higher complexity, more moving parts
                'convex': 1.6,     # Higher complexity
                'lido': 1.3        # Moderate risk
            }
            
            # Apply platform risk adjustments
            for platform, risk in platform_risk.items():
                mask = filtered_df['platform'].str.lower() == platform.lower()
                filtered_df.loc[mask, 'risk_score'] *= risk
            
            # Adjust risk score based on liquidity (more liquidity = lower risk)
            filtered_df['risk_score'] *= (1 + (1 / np.log10(filtered_df['liquidity_usd'] + 1)))
            
            # Apply risk level filters
            if risk_level == 'low':
                # Very conservative approach
                return filtered_df[
                    (filtered_df['liquidity_usd'] >= 1000000) &  # $1M+ liquidity
                    (filtered_df['risk_score'] <= 1.3)          # Low risk score
                ]
            elif risk_level == 'medium':
                # Balanced approach
                return filtered_df[
                    (filtered_df['liquidity_usd'] >= 500000) &   # $500k+ liquidity
                    (filtered_df['risk_score'] <= 1.6)           # Medium risk score
                ]
            else:  # high
                # Aggressive approach, still with some basic safety
                return filtered_df[filtered_df['liquidity_usd'] >= 100000]  # $100k+ liquidity
                
        except Exception as e:
            logger.error(f"Error applying risk filters: {e}")
            return yields_df  # Return original on error
    
    def _calculate_expected_returns(self, yields_df):
        """
        Calculate expected returns considering gas costs and other factors
        
        Args:
            yields_df: DataFrame of yield opportunities
            
        Returns:
            DataFrame with added expected return calculations
        """
        if yields_df.empty:
            return yields_df
        
        try:
            # Make a copy to avoid modifying the original
            df = yields_df.copy()
            
            # Calculate expected returns (annualized)
            # APY is already in percent, so direct use is fine
            df['expected_return'] = df['apy']
            
            # Adjust for pool size (slight penalty for very large pools as they often have lower returns)
            df['size_factor'] = 1.0 - (0.1 * np.log10(df['liquidity_usd'] / 1000000).clip(0, 1))
            df['expected_return'] *= df['size_factor']
            
            # Calculate a risk-adjusted return
            df['risk_adjusted_return'] = df['expected_return'] / df['risk_score']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating expected returns: {e}")
            return yields_df  # Return original on error
    
    def generate_allocation_plan(self, current_portfolio, available_funds):
        """
        Generate an allocation plan for available funds
        
        Args:
            current_portfolio: Dictionary with current portfolio data
            available_funds: Amount of funds available to allocate
            
        Returns:
            List of allocation actions
        """
        if available_funds <= 0:
            logger.info("No funds available for allocation")
            return []
        
        try:
            # Get current opportunities
            opportunities = self.analyze_opportunities()
            
            if opportunities.empty:
                logger.warning("No suitable opportunities found for allocation")
                return []
            
            # Check current gas costs
            # In a real implementation, this would fetch real gas prices
            current_gas_gwei = 25  # Example value
            
            # Skip allocation if gas is too high
            if current_gas_gwei > self.config['gas_threshold']:
                logger.warning(f"Gas price too high for allocation: {current_gas_gwei} gwei > {self.config['gas_threshold']} gwei threshold")
                return []
            
            # Generate allocation plan
            allocation_plan = []
            remaining_funds = available_funds
            
            # Get current allocation to check diversification
            current_by_platform = self._get_current_by_platform(current_portfolio)
            current_by_token = self._get_current_by_token(current_portfolio)
            
            # Calculate total portfolio value including available funds
            total_value = current_portfolio.get('total_value_usd', 0) + available_funds
            
            # Sort opportunities by risk-adjusted return
            sorted_opps = opportunities.sort_values('risk_adjusted_return', ascending=False)
            
            for _, opp in sorted_opps.iterrows():
                if remaining_funds < self.config['min_position_size']:
                    break
                
                platform = opp['platform']
                token = opp['token']
                
                # Check platform diversification limit
                platform_allocation = current_by_platform.get(platform, 0)
                platform_limit = total_value * self.config['diversification']['max_per_platform']
                
                # Check token diversification limit
                token_allocation = current_by_token.get(token, 0)
                token_limit = total_value * self.config['diversification']['max_per_token']
                
                # Calculate maximum allocation respecting limits
                max_by_platform = max(0, platform_limit - platform_allocation)
                max_by_token = max(0, token_limit - token_allocation)
                max_by_config = self.config['max_position_size']
                
                # Allocation is minimum of all constraints
                allocation = min(
                    remaining_funds,
                    max_by_platform,
                    max_by_token,
                    max_by_config,
                    opp['liquidity_usd'] * 0.01  # Don't exceed 1% of pool liquidity
                )
                
                # Only create an allocation if it's meaningful
                if allocation >= self.config['min_position_size']:
                    # Create allocation action
                    action = {
                        'type': 'allocation',
                        'platform': platform,
                        'pool': opp['pool'],
                        'token': token,
                        'amount': allocation,
                        'expected_apy': opp['apy'],
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    allocation_plan.append(action)
                    
                    # Update remaining funds and current allocations
                    remaining_funds -= allocation
                    current_by_platform[platform] = current_by_platform.get(platform, 0) + allocation
                    current_by_token[token] = current_by_token.get(token, 0) + allocation
                
                # Stop if we've reached max positions or used all funds
                if len(allocation_plan) >= self.config['max_positions'] or remaining_funds < self.config['min_position_size']:
                    break
            
            logger.info(f"Generated allocation plan with {len(allocation_plan)} actions, allocating ${available_funds - remaining_funds:,.2f}")
            return allocation_plan
            
        except Exception as e:
            logger.error(f"Error generating allocation plan: {e}")
            return []
    
    def generate_rebalance_plan(self, current_portfolio):
        """
        Generate a plan to rebalance the portfolio for better yields
        
        Args:
            current_portfolio: Dictionary with current portfolio data
            
        Returns:
            List of rebalance actions
        """
        try:
            # Get current opportunities
            opportunities = self.analyze_opportunities()
            
            if opportunities.empty:
                logger.warning("No suitable opportunities found for rebalancing")
                return []
            
            # Current portfolio assets
            assets = current_portfolio.get('assets', [])
            if not assets:
                logger.info("No assets to rebalance")
                return []
            
            # Check current gas costs
            # In a real implementation, this would fetch real gas prices
            current_gas_gwei = 25  # Example value
            
            # Skip rebalancing if gas is too high
            if current_gas_gwei > self.config['gas_threshold']:
                logger.warning(f"Gas price too high for rebalancing: {current_gas_gwei} gwei > {self.config['gas_threshold']} gwei threshold")
                return []
            
            # Generate rebalance plan
            rebalance_plan = []
            
            # Create a lookup for current APYs
            current_apys = {}
            for asset in assets:
                platform = asset.get('platform', '')
                token = asset.get('token', '')
                key = f"{token}_{platform}"
                
                # If we don't know the APY, estimate it from asset data
                # In a real implementation, this would come from portfolio tracking
                if 'apy' in asset:
                    current_apys[key] = asset['apy']
            
            # Find opportunities for each asset
            for asset in assets:
                platform = asset.get('platform', '')
                token = asset.get('token', '')
                amount = asset.get('amount', 0)
                value_usd = asset.get('value_usd', 0)
                
                # Skip small positions
                if value_usd < self.config['min_position_size']:
                    continue
                
                # Get current APY
                key = f"{token}_{platform}"
                current_apy = current_apys.get(key, 0)
                
                # Find best opportunity for this token
                token_opps = opportunities[opportunities['token'] == token]
                if token_opps.empty:
                    continue
                
                best_opp = token_opps.iloc[0]  # Already sorted by expected return
                best_platform = best_opp['platform']
                best_apy = best_opp['apy']
                
                # Only rebalance if the APY difference is significant
                apy_diff = best_apy - current_apy
                if (apy_diff > self.config['rebalance_threshold'] and 
                    best_platform != platform):
                    
                    # Check if the new platform has enough liquidity
                    if best_opp['liquidity_usd'] >= value_usd * 10:  # 10x safety factor
                        # Create rebalance action
                        action = {
                            'type': 'rebalance',
                            'from_platform': platform,
                            'to_platform': best_platform,
                            'token': token,
                            'amount': amount,
                            'value_usd': value_usd,
                            'current_apy': current_apy,
                            'new_apy': best_apy,
                            'apy_diff': apy_diff,
                            'timestamp': datetime.now().isoformat(),
                            'reason': f"better yield ({best_apy:.2f}% vs {current_apy:.2f}%)"
                        }
                        
                        rebalance_plan.append(action)
            
            logger.info(f"Generated rebalance plan with {len(rebalance_plan)} actions")
            return rebalance_plan
            
        except Exception as e:
            logger.error(f"Error generating rebalance plan: {e}")
            return []
    
    def _get_current_by_platform(self, portfolio):
        """Calculate current allocation by platform"""
        result = {}
        for asset in portfolio.get('assets', []):
            platform = asset.get('platform', 'unknown')
            value = asset.get('value_usd', 0)
            result[platform] = result.get(platform, 0) + value
        return result
    
    def _get_current_by_token(self, portfolio):
        """Calculate current allocation by token"""
        result = {}
        for asset in portfolio.get('assets', []):
            token = asset.get('token', 'unknown')
            value = asset.get('value_usd', 0)
            result[token] = result.get(token, 0) + value
        return result


# For testing
if __name__ == "__main__":
    optimizer = YieldOptimizer()
    
    # Mock portfolio
    portfolio = {
        "total_value_usd": 10450.23,
        "assets": [
            {"token": "ETH", "amount": 2.5, "value_usd": 8201.13, "platform": "aave", "apy": 1.8},
            {"token": "USDC", "amount": 2249.1, "value_usd": 2249.1, "platform": "compound", "apy": 3.0}
        ],
        "available_funds": 1000.00,
        "total_yield_annual": 2.8,
        "timestamp": datetime.now().isoformat()
    }
    
    # Test opportunity analysis
    print("\n=== YIELD OPPORTUNITIES ===")
    opportunities = optimizer.analyze_opportunities()
    if not opportunities.empty:
        for i, (_, opp) in enumerate(opportunities.head(5).iterrows()):
            print(f"{i+1}. {opp['token']} on {opp['platform']}: {opp['apy']:.2f}% APY (Risk score: {opp['risk_score']:.2f})")
    
    # Test allocation plan
    print("\n=== ALLOCATION PLAN ===")
    plan = optimizer.generate_allocation_plan(portfolio, portfolio['available_funds'])
    for i, action in enumerate(plan):
        print(f"{i+1}. Allocate ${action['amount']:.2f} to {action['token']} on {action['platform']} ({action['expected_apy']:.2f}% APY)")
    
    # Test rebalance plan
    print("\n=== REBALANCE PLAN ===")
    rebalance = optimizer.generate_rebalance_plan(portfolio)
    for i, action in enumerate(rebalance):
        print(f"{i+1}. Move {action['token']} (${action['value_usd']:.2f}) from {action['from_platform']} to {action['to_platform']}")
        print(f"   Reason: {action['reason']}") 