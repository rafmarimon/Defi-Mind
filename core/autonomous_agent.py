#!/usr/bin/env python3
"""
DEFIMIND Autonomous Agent

This module provides the autonomous agent functionality for DEFIMIND,
coordinating the various components and strategies to make decisions,
optimize yield, and communicate with users.
"""

import os
import json
import time
import logging
import traceback
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
from dotenv import load_dotenv

# Try to import our components
try:
    from core.agent_communication import AgentCommunicator
    from core.strategies.yield_optimizer import YieldOptimizer
    from core.defi_browser_agent import DefiBrowserAgent
    from core.self_improvement import SelfImprovement
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    print("Warning: Some DEFIMIND components are not available. Running in simulation mode.")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/autonomous_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutonomousAgent")

class AutonomousAgent:
    """
    AutonomousAgent class that coordinates the operation of DEFIMIND
    components to provide automated DeFi yield optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AutonomousAgent with optional custom configuration"""
        # Load environment variables
        load_dotenv()
        
        # Set default configuration
        self.config = {
            "simulation_mode": True,  # Default to simulation mode for safety
            "run_interval_seconds": 3600,  # Run cycle every hour
            "communication_interval_hours": 6,  # Send updates every 6 hours
            "gas_threshold_gwei": 50,  # Max gas price for transactions
            "risk_tolerance": "medium",  # Default risk tolerance
            "max_position_size_percent": 25,  # Max % of assets in one position
            "rebalance_days": 7,  # Check for rebalance weekly
            "browser_headless": True,  # Run browser in headless mode
            "llm_model": "gpt-4o",  # Default LLM model to use
        }
        
        # Override defaults with custom config if provided
        if config:
            self.config.update(config)
        
        # State tracking
        self.initialized = False
        self.start_time = datetime.now()
        self.last_action_time = None
        self.last_communication_time = None
        self.actions_taken = []
        
        # Initialize components
        self._init_components()
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        logger.info("AutonomousAgent initialized with config: " + json.dumps(self.config))
    
    def _init_components(self):
        """Initialize the required components"""
        try:
            if not COMPONENTS_AVAILABLE:
                logger.warning("Running with limited functionality due to missing components")
                self.communicator = None
                self.yield_optimizer = None
                self.browser_agent = None
                self.improver = None
                self.data_store = None
                return
            
            # Initialize the agent communicator
            self.communicator = AgentCommunicator()
            logger.info("AgentCommunicator initialized")
            
            # Initialize the yield optimizer
            optimizer_config = {
                "risk_tolerance": self.config["risk_tolerance"],
                "max_position_size_percent": self.config["max_position_size_percent"],
                "rebalance_threshold": 0.5  # Minimum APY difference to trigger rebalance
            }
            self.yield_optimizer = YieldOptimizer(config=optimizer_config)
            logger.info("YieldOptimizer initialized")
            
            # Initialize the browser agent
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("No OpenAI API key found. Browser agent will be limited.")
            
            self.browser_agent = DefiBrowserAgent(
                llm_api_key=openai_api_key,
                headless=self.config["browser_headless"]
            )
            logger.info("DefiBrowserAgent initialized")
            
            # Initialize the self-improvement module
            self.improver = SelfImprovement()
            logger.info("SelfImprovement module initialized")
            
            # Try to initialize a data store if available
            try:
                from core.market_data_store import MarketDataStore
                self.data_store = MarketDataStore()
                logger.info("MarketDataStore initialized")
            except ImportError:
                logger.warning("MarketDataStore not available, using fallback data methods")
                self.data_store = None
            
            self.initialized = True
        
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    async def run_cycle(self) -> Dict[str, Any]:
        """
        Run a complete agent cycle:
        1. Fetch latest market data
        2. Analyze portfolio
        3. Generate allocation plan
        4. Execute plan
        5. Generate user update
        6. Store cycle results
        
        Returns a summary of the cycle results
        """
        cycle_start = datetime.now()
        results = {
            "cycle_start": cycle_start.isoformat(),
            "cycle_end": None,
            "actions_taken": [],
            "user_update": None,
        }
        
        logger.info("Starting agent cycle")
        
        try:
            # Step 1: Fetch the latest market data using browser agent
            logger.info("Fetching market data")
            market_data = await self._fetch_market_data()
            
            # Step 2: Analyze the current portfolio
            logger.info("Analyzing portfolio")
            portfolio = self._get_portfolio()
            
            # Step 3: Generate allocation plan for available funds
            if portfolio.get("available_funds", 0) > 0:
                logger.info(f"Generating allocation plan for {portfolio['available_funds']} USD")
                allocation_plan = self.yield_optimizer.generate_allocation_plan(
                    portfolio, portfolio["available_funds"]
                )
                
                if allocation_plan:
                    # Step 4: Execute the allocation plan
                    logger.info(f"Executing allocation plan with {len(allocation_plan)} actions")
                    executed_allocations = await self._execute_plan(allocation_plan)
                    results["actions_taken"].extend(executed_allocations)
                    self.actions_taken.extend(executed_allocations)
            
            # Step 5: Check if rebalancing is needed
            if self._should_rebalance():
                logger.info("Checking for rebalance opportunities")
                rebalance_plan = self.yield_optimizer.generate_rebalance_plan(portfolio)
                
                if rebalance_plan:
                    # Execute the rebalance plan
                    logger.info(f"Executing rebalance plan with {len(rebalance_plan)} actions")
                    executed_rebalances = await self._execute_plan(rebalance_plan)
                    results["actions_taken"].extend(executed_rebalances)
                    self.actions_taken.extend(executed_rebalances)
                    self.last_action_time = datetime.now()
            
            # Step 6: Generate user update if needed
            if self._should_communicate() or results["actions_taken"]:
                if results["actions_taken"]:
                    logger.info("Generating activity report")
                    user_update = self.communicator.generate_activity_report(results["actions_taken"])
                else:
                    logger.info("Generating market update")
                    user_update = self.communicator.generate_market_update()
                
                results["user_update"] = user_update
                self.last_communication_time = datetime.now()
            
            # Step 7: Run improvement cycle
            if hasattr(self, 'improver') and self.improver:
                logger.info("Running self-improvement cycle")
                improvement_result = self.improver.run_improvement_cycle()
                results["improvement_result"] = improvement_result
            
            # Step 8: Store the results of this cycle
            self._store_cycle_results(results)
            
        except Exception as e:
            logger.error(f"Error in agent cycle: {str(e)}")
            logger.error(traceback.format_exc())
            results["error"] = str(e)
            results["traceback"] = traceback.format_exc()
        
        # Finalize results
        cycle_end = datetime.now()
        results["cycle_end"] = cycle_end.isoformat()
        results["duration_seconds"] = (cycle_end - cycle_start).total_seconds()
        
        logger.info(f"Agent cycle completed in {results['duration_seconds']:.2f} seconds")
        return results
    
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """
        Fetch the latest market data using the browser agent
        """
        market_data = {}
        
        try:
            if self.browser_agent:
                # Use the browser agent to collect yield data
                protocols = ["aave", "compound", "curve"]
                yields_df = await self.browser_agent.collect_defi_yields(protocols)
                
                if not yields_df.empty:
                    # Convert DataFrame to dictionary for storage
                    market_data["defi_yields"] = yields_df.to_dict(orient="records")
                    logger.info(f"Collected yield data for {len(market_data['defi_yields'])} opportunities")
                else:
                    # Fallback to DefiLlama if specific protocols failed
                    yields_df = await self.browser_agent.collect_from_defillama()
                    if not yields_df.empty:
                        market_data["defi_yields"] = yields_df.to_dict(orient="records")
                        logger.info(f"Collected yield data from DefiLlama: {len(market_data['defi_yields'])} opportunities")
            
            if not market_data and self.data_store:
                # Fallback to data store if browser agent failed
                market_data = self.data_store.get_current_market_data()
                logger.info("Using data from MarketDataStore")
            
            if not market_data:
                # Last resort fallback to mock data
                logger.warning("Using mock market data")
                market_data = self._generate_mock_market_data()
            
            # Store the market data if we have a data store
            if self.data_store:
                self.data_store.update_market_data(market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_mock_market_data()
    
    def _get_portfolio(self) -> Dict[str, Any]:
        """
        Get the current portfolio data
        In a real implementation, this would interact with wallets/exchanges
        """
        if self.data_store:
            try:
                portfolio = self.data_store.get_portfolio()
                if portfolio:
                    return portfolio
            except Exception as e:
                logger.error(f"Error getting portfolio from data store: {str(e)}")
        
        # Mock portfolio data for simulation
        return {
            "total_value_usd": 10000.00,
            "assets": [
                {
                    "token": "ETH",
                    "amount": 2.5,
                    "value_usd": 8000.00,
                    "platform": "aave",
                    "apy": 1.8,
                    "position_start": (datetime.now() - timedelta(days=30)).isoformat()
                },
                {
                    "token": "USDC",
                    "amount": 2000.00,
                    "value_usd": 2000.00,
                    "platform": "compound",
                    "apy": 3.0,
                    "position_start": (datetime.now() - timedelta(days=15)).isoformat()
                }
            ],
            "available_funds": 500.00,
            "total_yield_annual": 2.04,  # Weighted average APY
            "timestamp": datetime.now().isoformat()
        }
    
    async def _execute_plan(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute a plan (allocation or rebalance)
        In simulation mode, just return the plan with status
        In real mode, would execute transactions
        
        Returns the executed actions with status updates
        """
        executed_actions = []
        
        if self.config["simulation_mode"]:
            logger.info("Simulation mode: not executing real transactions")
            
            # Simulate execution
            for action in plan:
                # Copy the action and add execution status
                executed = action.copy()
                executed["status"] = "simulated"
                executed["execution_time"] = datetime.now().isoformat()
                executed_actions.append(executed)
                
                if action["type"] == "allocation":
                    logger.info(f"Simulated allocation: {action['amount']} {action['token']} on {action['platform']}")
                elif action["type"] == "rebalance":
                    logger.info(f"Simulated rebalance: {action['amount']} {action['token']} from {action['from_platform']} to {action['to_platform']}")
        else:
            logger.info("Executing real transactions (not implemented yet)")
            # Here we would implement actual transaction execution
            # This would likely involve wallet integrations, DEX interactions, etc.
            # For now, just mark as pending
            for action in plan:
                executed = action.copy()
                executed["status"] = "pending"
                executed["execution_time"] = datetime.now().isoformat()
                executed_actions.append(executed)
        
        return executed_actions
    
    def _should_rebalance(self) -> bool:
        """
        Determine if it's time to check for rebalancing opportunities
        based on the last action time and rebalance frequency
        """
        if not self.last_action_time:
            return True
            
        days_since_last_action = (datetime.now() - self.last_action_time).days
        return days_since_last_action >= self.config["rebalance_days"]
    
    def _should_communicate(self) -> bool:
        """
        Determine if it's time to send an update to the user
        based on the communication interval
        """
        if not self.last_communication_time:
            return True
            
        hours_since_last_comm = (datetime.now() - self.last_communication_time).total_seconds() / 3600
        return hours_since_last_comm >= self.config["communication_interval_hours"]
    
    def _store_cycle_results(self, results: Dict[str, Any]) -> None:
        """Store the results of the agent cycle"""
        # Ensure we have a directory to store results
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the results to a timestamped file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(results_dir, f"cycle_results_{timestamp}.json")
        
        try:
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Stored cycle results to {result_path}")
        except Exception as e:
            logger.error(f"Error storing cycle results: {str(e)}")
    
    def _generate_mock_market_data(self) -> Dict[str, Any]:
        """Generate mock market data for simulation"""
        return {
            "tokens": {
                "ETH": {"price_usd": 3200.00, "24h_change": 2.5},
                "BTC": {"price_usd": 65000.00, "24h_change": 1.8},
                "USDC": {"price_usd": 1.00, "24h_change": 0.0},
                "USDT": {"price_usd": 1.00, "24h_change": 0.01},
                "DAI": {"price_usd": 1.00, "24h_change": -0.02}
            },
            "yields": [
                {"platform": "aave", "token": "ETH", "apy": 1.8, "tvl": 2500000000},
                {"platform": "aave", "token": "USDC", "apy": 3.2, "tvl": 5000000000},
                {"platform": "compound", "token": "ETH", "apy": 1.5, "tvl": 1800000000},
                {"platform": "compound", "token": "USDC", "apy": 3.0, "tvl": 4500000000},
                {"platform": "curve", "token": "USDC/USDT/DAI", "apy": 3.5, "tvl": 3200000000}
            ],
            "gas": {
                "slow": 25,  # gwei
                "average": 35,
                "fast": 50
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def answer_question(self, question: str) -> str:
        """
        Answer a user question about market conditions or the agent's actions
        """
        if not self.communicator:
            return "Sorry, the communication module is not available."
        
        try:
            answer = self.communicator.answer_user_question(question)
            return answer
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    async def analyze_opportunity(self, protocol: str, pool: str) -> Dict[str, Any]:
        """
        Analyze a specific DeFi opportunity using the browser agent
        """
        if not self.browser_agent:
            return {"error": "Browser agent not available"}
        
        try:
            analysis = await self.browser_agent.analyze_defi_opportunity(protocol, pool)
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {str(e)}")
            return {
                "protocol": protocol,
                "pool": pool,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent
        """
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        status = {
            "initialized": self.initialized,
            "simulation_mode": self.config["simulation_mode"],
            "risk_tolerance": self.config["risk_tolerance"],
            "uptime_seconds": uptime_seconds,
            "actions_taken_count": len(self.actions_taken),
            "last_action_time": self.last_action_time.isoformat() if self.last_action_time else None,
            "last_communication_time": self.last_communication_time.isoformat() if self.last_communication_time else None,
        }
        
        # Add component status
        status["components"] = {
            "communicator": self.communicator is not None,
            "yield_optimizer": self.yield_optimizer is not None,
            "browser_agent": self.browser_agent is not None,
            "data_store": self.data_store is not None,
            "improver": hasattr(self, 'improver') and self.improver is not None
        }
        
        # Add browser agent stats if available
        if self.browser_agent:
            status["browser_agent_stats"] = self.browser_agent.get_stats()
        
        # Add improvement status if available
        if hasattr(self, 'improver') and self.improver:
            status["improvement_status"] = self.improver.get_improvement_status()
        
        return status


# For testing
async def test_autonomous_agent():
    """Test the AutonomousAgent functionality"""
    # Initialize the agent
    agent = AutonomousAgent(config={"simulation_mode": True, "browser_headless": False})
    
    # Run a single cycle
    print("Running agent cycle...")
    result = await agent.run_cycle()
    
    # Check for user update
    if result.get("user_update"):
        print("\nUser Update:")
        print(result["user_update"])
    
    # Show actions taken
    if result.get("actions_taken"):
        print("\nActions Taken:")
        for action in result["actions_taken"]:
            print(f"- {action['type']} action: {action.get('amount')} {action.get('token')} on {action.get('platform')}")
    
    # Answer a sample question
    print("\nAnswering question about yields:")
    answer = agent.answer_question("What are the best yields for USDC right now?")
    print(answer)
    
    # Show agent status
    print("\nAgent Status:")
    status = agent.get_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    asyncio.run(test_autonomous_agent()) 