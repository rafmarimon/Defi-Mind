#!/usr/bin/env python3
"""
DEFIMIND Agent Communication Module

This module enables the DEFIMIND agent to communicate with users through
natural language, providing updates on market conditions, agent activities,
and answering user questions.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("defimind.communication")

class AgentCommunicator:
    """Handles natural language communication between the DEFIMIND agent and users"""
    
    def __init__(self):
        """Initialize the agent communicator"""
        try:
            from core.langchain_agent import LangChainAgent
            from core.defimind_persistence import MarketDataStore
            
            self.lang_agent = LangChainAgent()
            self.data_store = MarketDataStore()
            self.last_update = None
            logger.info("AgentCommunicator initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to initialize AgentCommunicator: {e}")
            self.lang_agent = None
            self.data_store = None
    
    def generate_market_update(self):
        """Generate a natural language update on market conditions"""
        if not self.lang_agent:
            return "Communication system is unavailable. Please check LangChain integration."
        
        try:
            # Get recent market data
            market_data = self._get_market_data()
            
            context = {
                "market_data": market_data,
                "timestamp": datetime.now().isoformat(),
                "portfolio_changes": self._get_portfolio_changes()
            }
            
            prompt = """
            Generate a concise update on the current DeFi market conditions, focusing on:
            1. Current yield opportunities across platforms (Aave, Compound, Curve, etc.)
            2. Recent market trends and significant changes
            3. Risk factors to be aware of
            4. Actionable insights or recommendations
            
            Keep it professional but conversational, with clear sections.
            """
            
            response = self.lang_agent.process_message(prompt, context)
            self.last_update = datetime.now()
            
            return response
        except Exception as e:
            logger.error(f"Error generating market update: {e}")
            return f"Unable to generate market update at this time. Technical details: {str(e)}"
    
    def generate_activity_report(self, actions_taken=None):
        """Generate a report of what the agent has done recently"""
        if not self.lang_agent:
            return "Communication system is unavailable. Please check LangChain integration."
        
        try:
            # Get recent actions if none provided
            if actions_taken is None:
                actions_taken = self._get_recent_actions()
            
            # Get portfolio data
            portfolio = self._get_portfolio_data()
            
            context = {
                "actions": actions_taken,
                "portfolio": portfolio,
                "timestamp": datetime.now().isoformat()
            }
            
            prompt = """
            Generate a summary of recent actions taken by the DEFIMIND agent and their outcomes.
            Include:
            1. What changes were made to the portfolio
            2. Why these actions were taken (market conditions, yield opportunities, etc.)
            3. The current portfolio composition
            4. Performance metrics (if available)
            
            Format as a clear report with sections. Be specific about yields, tokens, and platforms.
            """
            
            response = self.lang_agent.process_message(prompt, context)
            return response
        except Exception as e:
            logger.error(f"Error generating activity report: {e}")
            return f"Unable to generate activity report at this time. Technical details: {str(e)}"
    
    def answer_user_question(self, question):
        """Answer a specific user question with context"""
        if not self.lang_agent:
            return "Communication system is unavailable. Please check LangChain integration."
        
        try:
            # Get context data
            market_data = self._get_market_data()
            portfolio = self._get_portfolio_data()
            recent_actions = self._get_recent_actions()
            
            context = {
                "current_market": market_data,
                "portfolio": portfolio,
                "recent_actions": recent_actions,
                "timestamp": datetime.now().isoformat()
            }
            
            # Process the user's question
            response = self.lang_agent.process_message(question, context)
            return response
        except Exception as e:
            logger.error(f"Error answering user question: {e}")
            return f"Unable to answer your question at this time. Technical details: {str(e)}"
    
    def _get_market_data(self):
        """Get recent market data for context"""
        try:
            if self.data_store:
                return self.data_store.get_recent_data(hours=24)
            
            # Fallback to placeholder data if data store not available
            return {
                "tokens": {
                    "ETH": {"price": 3280.45, "24h_change": 2.3},
                    "USDC": {"price": 1.0, "24h_change": 0.01},
                    "WBTC": {"price": 54320.15, "24h_change": 1.2}
                },
                "yields": {
                    "aave": {
                        "USDC": 3.2,
                        "ETH": 1.8,
                        "WBTC": 0.9
                    },
                    "compound": {
                        "USDC": 3.4,
                        "ETH": 1.5,
                        "WBTC": 0.8
                    }
                },
                "gas_price_gwei": 25,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"error": str(e)}
    
    def _get_portfolio_data(self):
        """Get current portfolio data"""
        try:
            if self.data_store:
                return self.data_store.get_current_portfolio()
            
            # Fallback to placeholder data
            return {
                "total_value_usd": 10450.23,
                "assets": [
                    {"token": "ETH", "amount": 2.5, "value_usd": 8201.13, "platform": "aave"},
                    {"token": "USDC", "amount": 2249.1, "value_usd": 2249.1, "platform": "compound"}
                ],
                "available_funds": 100.00,
                "total_yield_annual": 2.8,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio data: {e}")
            return {"error": str(e)}
    
    def _get_portfolio_changes(self):
        """Get portfolio changes over time"""
        try:
            if self.data_store:
                return self.data_store.get_portfolio_changes(days=1)
            
            # Fallback to placeholder data
            return {
                "value_change_usd": 120.45,
                "value_change_percent": 1.2,
                "yield_change": 0.1,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio changes: {e}")
            return {"error": str(e)}
    
    def _get_recent_actions(self):
        """Get recent actions taken by the agent"""
        try:
            if self.data_store:
                return self.data_store.get_recent_actions(count=5)
            
            # Fallback to placeholder data
            return [
                {
                    "type": "allocation",
                    "platform": "aave",
                    "token": "ETH",
                    "amount": 0.5,
                    "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
                    "apy": 1.8
                },
                {
                    "type": "rebalance",
                    "from_platform": "compound",
                    "to_platform": "aave",
                    "token": "USDC",
                    "amount": 500,
                    "timestamp": (datetime.now() - timedelta(hours=12)).isoformat(),
                    "reason": "better yield (3.2% vs 2.9%)"
                }
            ]
        except Exception as e:
            logger.error(f"Error getting recent actions: {e}")
            return []


# For testing
if __name__ == "__main__":
    communicator = AgentCommunicator()
    
    # Test market update
    print("\n=== MARKET UPDATE ===")
    update = communicator.generate_market_update()
    print(update)
    
    # Test activity report
    print("\n=== ACTIVITY REPORT ===")
    report = communicator.generate_activity_report()
    print(report)
    
    # Test user question
    print("\n=== USER QUESTION ===")
    answer = communicator.answer_user_question("What's the best yield for USDC right now?")
    print(answer) 