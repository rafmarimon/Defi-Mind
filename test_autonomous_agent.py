#!/usr/bin/env python3
"""
Test script for the AutonomousAgent component

This script tests the functionality of the AutonomousAgent class, ensuring
it correctly coordinates the various components of the DEFIMIND system.
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

try:
    from core.autonomous_agent import AutonomousAgent
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False

@unittest.skipIf(not MODULE_AVAILABLE, "AutonomousAgent module not available")
class TestAutonomousAgent(unittest.TestCase):
    """Test cases for the AutonomousAgent component"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create agent with mocked dependencies
        with patch('core.autonomous_agent.AgentCommunicator'), \
             patch('core.autonomous_agent.YieldOptimizer'), \
             patch('core.autonomous_agent.MarketDataStore'):
            self.agent = AutonomousAgent()
            
            # Force simulation mode
            self.agent.config['simulation_mode'] = True
            
            # Mock components
            self.agent.communicator = MagicMock()
            self.agent.yield_optimizer = MagicMock()
            self.agent.data_store = MagicMock()
            
            # Mock communicator responses
            self.agent.communicator.generate_market_update.return_value = "Market update"
            self.agent.communicator.generate_activity_report.return_value = "Activity report"
            self.agent.communicator.answer_user_question.return_value = "Question answer"
            
            # Mock optimizer responses
            self.agent.yield_optimizer.generate_allocation_plan.return_value = [
                {
                    "type": "allocation",
                    "platform": "aave",
                    "token": "USDC",
                    "amount": 500.00,
                    "expected_apy": 3.2,
                    "timestamp": datetime.now().isoformat()
                }
            ]
            self.agent.yield_optimizer.generate_rebalance_plan.return_value = [
                {
                    "type": "rebalance",
                    "from_platform": "compound",
                    "to_platform": "aave",
                    "token": "USDC",
                    "amount": 1000.00,
                    "value_usd": 1000.00,
                    "current_apy": 2.8,
                    "new_apy": 3.2,
                    "apy_diff": 0.4,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "better yield (3.2% vs 2.8%)"
                }
            ]
    
    def test_initialization(self):
        """Test that the AutonomousAgent initializes correctly"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.communicator)
        self.assertIsNotNone(self.agent.yield_optimizer)
        self.assertIsNotNone(self.agent.data_store)
        self.assertTrue(self.agent.config['simulation_mode'])
    
    def test_run_cycle(self):
        """Test running a complete agent cycle"""
        # Mock internal methods
        self.agent._fetch_market_data = MagicMock(return_value={})
        self.agent._get_portfolio = MagicMock(return_value={
            "total_value_usd": 10000.00,
            "assets": [
                {"token": "ETH", "amount": 2.5, "value_usd": 8000.00, "platform": "aave", "apy": 1.8},
                {"token": "USDC", "amount": 2000.00, "value_usd": 2000.00, "platform": "compound", "apy": 3.0}
            ],
            "available_funds": 500.00,
            "total_yield_annual": 2.4
        })
        self.agent._execute_plan = MagicMock(side_effect=lambda x: [dict(a, status="simulated") for a in x])
        self.agent._should_rebalance = MagicMock(return_value=True)
        self.agent._should_communicate = MagicMock(return_value=True)
        self.agent._store_cycle_results = MagicMock()
        
        # Run a cycle
        result = self.agent.run_cycle()
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertIn('cycle_start', result)
        self.assertIn('cycle_end', result)
        self.assertIn('actions_taken', result)
        self.assertIn('user_update', result)
        
        # Check that key methods were called
        self.assertTrue(self.agent._fetch_market_data.called)
        self.assertTrue(self.agent._get_portfolio.called)
        self.assertTrue(self.agent.yield_optimizer.generate_allocation_plan.called)
        self.assertTrue(self.agent.yield_optimizer.generate_rebalance_plan.called)
        self.assertTrue(self.agent._execute_plan.called)
        self.assertTrue(self.agent.communicator.generate_activity_report.called)
        self.assertTrue(self.agent._store_cycle_results.called)
    
    def test_execute_plan_simulation(self):
        """Test executing a plan in simulation mode"""
        # Setup test plan
        plan = [
            {
                "type": "allocation",
                "platform": "aave",
                "token": "USDC",
                "amount": 500.00,
                "expected_apy": 3.2,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Execute in simulation mode
        self.agent.config['simulation_mode'] = True
        result = self.agent._execute_plan(plan)
        
        # Assertions
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['status'], 'simulated')
        self.assertEqual(result[0]['type'], 'allocation')
        self.assertEqual(result[0]['platform'], 'aave')
    
    def test_should_rebalance(self):
        """Test rebalance decision logic"""
        # Test when last action time is None (should return True)
        self.agent.last_action_time = None
        self.assertTrue(self.agent._should_rebalance())
        
        # Test when last action was just now (should return False)
        self.agent.last_action_time = datetime.now()
        self.assertFalse(self.agent._should_rebalance())
        
        # Test when last action was long ago (should return True)
        self.agent.last_action_time = datetime.now() - timedelta(days=self.agent.config['rebalance_days'] + 1)
        self.assertTrue(self.agent._should_rebalance())
    
    def test_should_communicate(self):
        """Test communication decision logic"""
        # Test when last communication time is None (should return True)
        self.agent.last_communication_time = None
        self.assertTrue(self.agent._should_communicate())
        
        # Test when last communication was just now (should return False)
        self.agent.last_communication_time = datetime.now()
        self.assertFalse(self.agent._should_communicate())
        
        # Test when last communication was long ago (should return True)
        self.agent.last_communication_time = datetime.now() - timedelta(hours=self.agent.config['communication_interval_hours'] + 1)
        self.assertTrue(self.agent._should_communicate())
    
    def test_answer_question(self):
        """Test answering a user question"""
        # Test the function
        question = "What's the best yield for USDC?"
        result = self.agent.answer_question(question)
        
        # Assertions
        self.assertEqual(result, "Question answer")
        self.assertTrue(self.agent.communicator.answer_user_question.called)
        self.agent.communicator.answer_user_question.assert_called_with(question)
    
    def test_get_status(self):
        """Test getting agent status"""
        # Set some state for the test
        self.agent.last_action_time = datetime.now() - timedelta(hours=5)
        self.agent.last_communication_time = datetime.now() - timedelta(hours=2)
        self.agent.actions_taken = [{}, {}]  # Two dummy actions
        
        # Get status
        status = self.agent.get_status()
        
        # Assertions
        self.assertIsNotNone(status)
        self.assertIn('initialized', status)
        self.assertIn('simulation_mode', status)
        self.assertIn('risk_tolerance', status)
        self.assertIn('last_action_time', status)
        self.assertIn('last_communication_time', status)
        self.assertIn('uptime_seconds', status)
        self.assertEqual(status['actions_taken_count'], 2)
        self.assertTrue(status['initialized'])
        self.assertTrue(status['simulation_mode'])


if __name__ == '__main__':
    unittest.main() 