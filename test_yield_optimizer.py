#!/usr/bin/env python3
"""
Test script for the YieldOptimizer component

This script tests the functionality of the YieldOptimizer class, ensuring
it correctly identifies yield opportunities and generates allocation plans.
"""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime

try:
    from core.strategies.yield_optimizer import YieldOptimizer
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

@unittest.skipIf(not MODULE_AVAILABLE, "YieldOptimizer module not available")
class TestYieldOptimizer(unittest.TestCase):
    """Test cases for the YieldOptimizer component"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('core.strategies.yield_optimizer.YieldTracker'):
            self.optimizer = YieldOptimizer()
            
            # Mock the yield tracker
            self.optimizer.yield_tracker = MagicMock()
            
            # Sample yield data
            yield_data = pd.DataFrame([
                {
                    'platform': 'aave',
                    'pool': 'USDC-aave',
                    'token': 'USDC',
                    'apy': 3.2,
                    'tvl_usd': 1000000,
                    'liquidity_usd': 1000000,
                    'il_risk': 'low',
                    'updated_at': datetime.now().isoformat()
                },
                {
                    'platform': 'compound',
                    'pool': 'USDC-compound',
                    'token': 'USDC',
                    'apy': 3.0,
                    'tvl_usd': 1200000,
                    'liquidity_usd': 1200000,
                    'il_risk': 'low',
                    'updated_at': datetime.now().isoformat()
                },
                {
                    'platform': 'curve',
                    'pool': 'USDC-curve',
                    'token': 'USDC',
                    'apy': 3.5,
                    'tvl_usd': 800000,
                    'liquidity_usd': 800000,
                    'il_risk': 'low',
                    'updated_at': datetime.now().isoformat()
                },
                {
                    'platform': 'aave',
                    'pool': 'ETH-aave',
                    'token': 'ETH',
                    'apy': 1.8,
                    'tvl_usd': 500000,
                    'liquidity_usd': 500000,
                    'il_risk': 'low',
                    'updated_at': datetime.now().isoformat()
                }
            ])
            
            # Set up mock returns
            self.optimizer.yield_tracker.get_best_yields_by_token.return_value = yield_data
    
    def test_initialization(self):
        """Test that the YieldOptimizer initializes correctly"""
        self.assertIsNotNone(self.optimizer)
        self.assertIsNotNone(self.optimizer.yield_tracker)
        self.assertIsNotNone(self.optimizer.config)
        self.assertEqual(self.optimizer.config['risk_tolerance'], 'medium')
    
    def test_analyze_opportunities(self):
        """Test analyzing yield opportunities"""
        # Test the function
        opportunities = self.optimizer.analyze_opportunities()
        
        # Assertions
        self.assertIsNotNone(opportunities)
        self.assertIsInstance(opportunities, pd.DataFrame)
        self.assertTrue(self.optimizer.yield_tracker.get_best_yields_by_token.called)
        
        # Check that risk score and expected return were calculated
        if not opportunities.empty:
            self.assertIn('risk_score', opportunities.columns)
            self.assertIn('expected_return', opportunities.columns)
    
    def test_apply_risk_filters(self):
        """Test applying risk filters to opportunities"""
        # Setup test data
        test_df = pd.DataFrame([
            {
                'platform': 'aave',
                'token': 'USDC',
                'apy': 3.2,
                'liquidity_usd': 1000000,
            },
            {
                'platform': 'uniswap',
                'token': 'ETH',
                'apy': 4.5,
                'liquidity_usd': 200000,
            },
            {
                'platform': 'yearn',
                'token': 'DAI',
                'apy': 5.2,
                'liquidity_usd': 50000,
            }
        ])
        
        # Test with different risk levels
        self.optimizer.config['risk_tolerance'] = 'low'
        low_risk = self.optimizer._apply_risk_filters(test_df)
        
        self.optimizer.config['risk_tolerance'] = 'medium'
        medium_risk = self.optimizer._apply_risk_filters(test_df)
        
        self.optimizer.config['risk_tolerance'] = 'high'
        high_risk = self.optimizer._apply_risk_filters(test_df)
        
        # Assertions (low risk should filter more aggressively)
        self.assertLessEqual(len(low_risk), len(medium_risk))
        self.assertLessEqual(len(medium_risk), len(high_risk))
    
    def test_calculate_expected_returns(self):
        """Test calculating expected returns"""
        # Setup test data
        test_df = pd.DataFrame([
            {
                'platform': 'aave',
                'token': 'USDC',
                'apy': 3.2,
                'liquidity_usd': 1000000,
                'risk_score': 1.0
            },
            {
                'platform': 'compound',
                'token': 'ETH',
                'apy': 1.8,
                'liquidity_usd': 500000,
                'risk_score': 1.1
            }
        ])
        
        # Test function
        result = self.optimizer._calculate_expected_returns(test_df)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn('expected_return', result.columns)
        self.assertIn('risk_adjusted_return', result.columns)
    
    def test_generate_allocation_plan(self):
        """Test generating an allocation plan"""
        # Mock portfolio
        portfolio = {
            "total_value_usd": 10000.00,
            "assets": [
                {"token": "ETH", "amount": 2.5, "value_usd": 8000.00, "platform": "aave", "apy": 1.8},
                {"token": "USDC", "amount": 2000.00, "value_usd": 2000.00, "platform": "compound", "apy": 3.0}
            ],
            "available_funds": 1000.00,
            "total_yield_annual": 2.8,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test function with real portfolio and funds
        plan = self.optimizer.generate_allocation_plan(portfolio, portfolio['available_funds'])
        
        # Assertions
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, list)
        self.assertTrue(self.optimizer.yield_tracker.get_best_yields_by_token.called)
        
        # Check plan details if actions were generated
        if plan:
            self.assertIn('type', plan[0])
            self.assertEqual(plan[0]['type'], 'allocation')
            self.assertIn('platform', plan[0])
            self.assertIn('token', plan[0])
            self.assertIn('amount', plan[0])
            
            # Total allocated should be <= available funds
            total_allocated = sum(action['amount'] for action in plan)
            self.assertLessEqual(total_allocated, portfolio['available_funds'])
    
    def test_generate_rebalance_plan(self):
        """Test generating a rebalance plan"""
        # Mock portfolio with suboptimal allocations
        portfolio = {
            "total_value_usd": 10000.00,
            "assets": [
                {"token": "ETH", "amount": 2.5, "value_usd": 8000.00, "platform": "compound", "apy": 1.5},
                {"token": "USDC", "amount": 2000.00, "value_usd": 2000.00, "platform": "compound", "apy": 2.8}
            ],
            "available_funds": 0.00,
            "total_yield_annual": 2.8,
            "timestamp": datetime.now().isoformat()
        }
        
        # Test function
        plan = self.optimizer.generate_rebalance_plan(portfolio)
        
        # Assertions
        self.assertIsNotNone(plan)
        self.assertIsInstance(plan, list)
        self.assertTrue(self.optimizer.yield_tracker.get_best_yields_by_token.called)
        
        # Check plan details if actions were generated
        if plan:
            self.assertIn('type', plan[0])
            self.assertEqual(plan[0]['type'], 'rebalance')
            self.assertIn('from_platform', plan[0])
            self.assertIn('to_platform', plan[0])
            self.assertIn('token', plan[0])
            self.assertIn('apy_diff', plan[0])
            self.assertTrue(plan[0]['apy_diff'] > self.optimizer.config['rebalance_threshold'])
    
    def test_get_current_by_platform(self):
        """Test calculating current allocation by platform"""
        # Mock portfolio
        portfolio = {
            "assets": [
                {"token": "ETH", "value_usd": 8000.00, "platform": "aave"},
                {"token": "USDC", "value_usd": 1500.00, "platform": "aave"},
                {"token": "USDC", "value_usd": 500.00, "platform": "compound"}
            ]
        }
        
        # Test function
        result = self.optimizer._get_current_by_platform(portfolio)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn('aave', result)
        self.assertIn('compound', result)
        self.assertEqual(result['aave'], 9500.00)
        self.assertEqual(result['compound'], 500.00)
    
    def test_get_current_by_token(self):
        """Test calculating current allocation by token"""
        # Mock portfolio
        portfolio = {
            "assets": [
                {"token": "ETH", "value_usd": 8000.00, "platform": "aave"},
                {"token": "USDC", "value_usd": 1500.00, "platform": "aave"},
                {"token": "USDC", "value_usd": 500.00, "platform": "compound"}
            ]
        }
        
        # Test function
        result = self.optimizer._get_current_by_token(portfolio)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn('ETH', result)
        self.assertIn('USDC', result)
        self.assertEqual(result['ETH'], 8000.00)
        self.assertEqual(result['USDC'], 2000.00)


if __name__ == '__main__':
    unittest.main() 