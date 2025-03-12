#!/usr/bin/env python3
"""
Test script for the YieldTracker component

This script tests the functionality of the YieldTracker class, ensuring
it correctly tracks and compares yields across DeFi platforms.
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

try:
    from core.yield_tracker import YieldTracker
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

@unittest.skipIf(not MODULE_AVAILABLE, "YieldTracker module not available")
class TestYieldTracker(unittest.TestCase):
    """Test cases for the YieldTracker component"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = YieldTracker()
    
    def test_initialization(self):
        """Test that the YieldTracker initializes correctly"""
        self.assertIsNotNone(self.tracker)
        self.assertIsInstance(self.tracker.platforms, list)
        self.assertTrue(len(self.tracker.platforms) > 0)
        self.assertIsNotNone(self.tracker.cache)
    
    @patch('core.yield_tracker.YieldTracker._fetch_defillama_yields')
    def test_fetch_current_yields(self, mock_fetch):
        """Test fetching current yields"""
        # Setup mock data
        mock_data = [
            {
                "platform": "aave",
                "pool": "USDC-aave",
                "token": "USDC",
                "apy": 3.2,
                "tvl_usd": 1000000,
                "liquidity_usd": 1000000,
                "il_risk": "low",
                "updated_at": datetime.now().isoformat()
            },
            {
                "platform": "compound",
                "pool": "ETH-compound",
                "token": "ETH",
                "apy": 1.5,
                "tvl_usd": 500000,
                "liquidity_usd": 500000,
                "il_risk": "low",
                "updated_at": datetime.now().isoformat()
            }
        ]
        mock_fetch.return_value = mock_data
        
        # Test fetch function
        yields = self.tracker.fetch_current_yields(force_refresh=True)
        
        # Assertions
        self.assertIsNotNone(yields)
        self.assertIsInstance(yields, dict)
        self.assertTrue(mock_fetch.called)
    
    def test_get_best_yields_by_token(self):
        """Test finding the best yields by token"""
        # Use mock data from the cache
        self.tracker.cache = {
            "all_yields": self.tracker._generate_mock_yield_data(),
            "cache_time": datetime.now()
        }
        
        # Test function
        best_yields = self.tracker.get_best_yields_by_token(min_liquidity=100000)
        
        # Assertions
        self.assertIsInstance(best_yields, pd.DataFrame)
        if not best_yields.empty:
            self.assertIn('token', best_yields.columns)
            self.assertIn('apy', best_yields.columns)
            self.assertIn('platform', best_yields.columns)
    
    def test_get_historical_yield_trends(self):
        """Test retrieving historical yield trends"""
        # Test function
        trends = self.tracker.get_historical_yield_trends(days=7, tokens=["USDC"])
        
        # Assertions
        self.assertIsInstance(trends, pd.DataFrame)
        if not trends.empty:
            self.assertIn('date', trends.columns)
            self.assertIn('token', trends.columns)
            self.assertIn('apy', trends.columns)
    
    def test_mock_data_generation(self):
        """Test mock data generation functionality"""
        # Test mock yield data
        mock_yield = self.tracker._generate_mock_yield_data()
        self.assertIsInstance(mock_yield, dict)
        self.assertTrue(len(mock_yield) > 0)
        
        # Test mock historical data
        mock_hist = self.tracker._generate_mock_historical_data(days=7)
        self.assertIsInstance(mock_hist, pd.DataFrame)
        self.assertFalse(mock_hist.empty)


if __name__ == '__main__':
    unittest.main() 