#!/usr/bin/env python3
"""
Test script for the AgentCommunicator component

This script tests the functionality of the AgentCommunicator class, ensuring
it correctly communicates with users through market updates, activity reports,
and answering questions.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

try:
    from core.agent_communication import AgentCommunicator
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

@unittest.skipIf(not MODULE_AVAILABLE, "AgentCommunicator module not available")
class TestAgentCommunicator(unittest.TestCase):
    """Test cases for the AgentCommunicator component"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a communicator with mocked dependencies
        with patch('core.agent_communication.LangChainAgent'), \
             patch('core.agent_communication.MarketDataStore'):
            self.communicator = AgentCommunicator()
            
            # Mock the language agent
            self.communicator.lang_agent = MagicMock()
            self.communicator.lang_agent.process_message.return_value = "This is a test response"
            
            # Mock the data store
            self.communicator.data_store = MagicMock()
    
    def test_initialization(self):
        """Test that the AgentCommunicator initializes correctly"""
        self.assertIsNotNone(self.communicator)
        self.assertIsNotNone(self.communicator.lang_agent)
        self.assertIsNotNone(self.communicator.data_store)
    
    def test_generate_market_update(self):
        """Test generating a market update"""
        # Mock internal methods
        self.communicator._get_market_data = MagicMock(return_value={})
        self.communicator._get_portfolio_changes = MagicMock(return_value={})
        
        # Test the function
        result = self.communicator.generate_market_update()
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result, "This is a test response")
        self.assertTrue(self.communicator.lang_agent.process_message.called)
        self.assertTrue(self.communicator._get_market_data.called)
        self.assertTrue(self.communicator._get_portfolio_changes.called)
    
    def test_generate_activity_report(self):
        """Test generating an activity report"""
        # Mock internal methods
        self.communicator._get_portfolio_data = MagicMock(return_value={})
        
        # Test with provided actions
        actions = [
            {
                "type": "allocation",
                "platform": "aave",
                "token": "ETH",
                "amount": 1.0,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        result = self.communicator.generate_activity_report(actions)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result, "This is a test response")
        self.assertTrue(self.communicator.lang_agent.process_message.called)
        self.assertTrue(self.communicator._get_portfolio_data.called)
    
    def test_answer_user_question(self):
        """Test answering a user question"""
        # Mock internal methods
        self.communicator._get_market_data = MagicMock(return_value={})
        self.communicator._get_portfolio_data = MagicMock(return_value={})
        self.communicator._get_recent_actions = MagicMock(return_value=[])
        
        # Test the function
        question = "What's the best yield for USDC?"
        result = self.communicator.answer_user_question(question)
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result, "This is a test response")
        self.assertTrue(self.communicator.lang_agent.process_message.called)
        self.assertTrue(self.communicator._get_market_data.called)
        self.assertTrue(self.communicator._get_portfolio_data.called)
        self.assertTrue(self.communicator._get_recent_actions.called)
    
    def test_fallback_data_methods(self):
        """Test fallback data methods when data store is unavailable"""
        # Remove data store to force fallback
        self.communicator.data_store = None
        
        # Test market data fallback
        market_data = self.communicator._get_market_data()
        self.assertIsInstance(market_data, dict)
        self.assertIn("tokens", market_data)
        self.assertIn("yields", market_data)
        
        # Test portfolio data fallback
        portfolio = self.communicator._get_portfolio_data()
        self.assertIsInstance(portfolio, dict)
        self.assertIn("total_value_usd", portfolio)
        self.assertIn("assets", portfolio)
        
        # Test portfolio changes fallback
        changes = self.communicator._get_portfolio_changes()
        self.assertIsInstance(changes, dict)
        self.assertIn("value_change_usd", changes)
        
        # Test recent actions fallback
        actions = self.communicator._get_recent_actions()
        self.assertIsInstance(actions, list)
        self.assertTrue(len(actions) > 0)
    
    def test_error_handling(self):
        """Test error handling in methods"""
        # Make the language agent raise an exception
        self.communicator.lang_agent.process_message.side_effect = Exception("Test error")
        
        # Test market update with error
        result = self.communicator.generate_market_update()
        self.assertIn("Unable to generate market update", result)
        self.assertIn("Test error", result)
        
        # Test activity report with error
        result = self.communicator.generate_activity_report()
        self.assertIn("Unable to generate activity report", result)
        self.assertIn("Test error", result)
        
        # Test question answering with error
        result = self.communicator.answer_user_question("test question")
        self.assertIn("Unable to answer your question", result)
        self.assertIn("Test error", result)


if __name__ == '__main__':
    unittest.main() 