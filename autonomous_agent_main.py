import os
import logging
import asyncio
import json
from datetime import datetime
from dotenv import load_dotenv

# Import our agent components
from agent_brain import AutonomousAgent
from ai_agent import AITradingAgent, TradingBot
from yield_scanner import YieldScanner

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("autonomous_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("autonomous_agent")

class AutonomousTradingSystem:
    """System that integrates the cognitive agent with the trading functionality"""
    
    def __init__(self, use_llm=True, model_path="models/latest_model.h5"):
        # Initialize the cognitive agent
        self.agent = AutonomousAgent(name="TradingMind", use_llm=use_llm)
        
        # Initialize the trading bot
        self.trading_bot = TradingBot()
        
        # Set initial goals
        self.agent.set_goals([
            "Maximize investment returns across DeFi protocols",
            "Minimize risk during high volatility periods",
            "Identify optimal yield opportunities",
            "Maintain balanced portfolio exposure"
        ])
        
        # Store market state
        self.market_state = {}
        
    async def initialize(self):
        """Initialize all system components"""
        # Start the cognitive agent
        self.agent.start()
        
        # Initialize trading components
        await self.trading_bot.scanner.initialize()
        logger.info("✅ Trading components initialized")
        
    async def collect_observations(self):
        """Gather data from various sources to create a comprehensive observation"""
        # Get market data from the yield scanner
        try:
            logger.info("📊 Collecting market observations...")
            
            # Get APY data for different protocols
            pancake_pool = await self.trading_bot.scanner.get_best_apy_for_protocol("pancakeswap")
            joe_pool = await self.trading_bot.scanner.get_best_apy_for_protocol("traderjoe")
            quickswap_pool = await self.trading_bot.scanner.get_best_apy_for_protocol("quickswap")
            
            # Format the data
            observation = {
                "market_data": {
                    "protocols": {
                        "pancakeswap": {
                            "apy": pancake_pool.apy if pancake_pool else 0.0,
                            "pool": pancake_pool.name if pancake_pool else "unknown",
                            "tvl": pancake_pool.tvl if pancake_pool else 0.0
                        },
                        "traderjoe": {
                            "apy": joe_pool.apy if joe_pool else 0.0,
                            "pool": joe_pool.name if joe_pool else "unknown",
                            "tvl": joe_pool.tvl if joe_pool else 0.0
                        },
                        "quickswap": {
                            "apy": quickswap_pool.apy if quickswap_pool else 0.0,
                            "pool": quickswap_pool.name if quickswap_pool else "unknown",
                            "tvl": quickswap_pool.tvl if quickswap_pool else 0.0
                        }
                    },
                    "timestamp": datetime.now().isoformat(),
                    "gas_price": self.trading_bot.scanner.last_gas_price if hasattr(self.trading_bot.scanner, "last_gas_price") else 0,
                    # We would add more data here in a complete implementation
                    "volatility": 0.45  # This would be calculated from real data
                }
            }
            
            # Store the current market state
            self.market_state = observation
            
            return observation
            
        except Exception as e:
            logger.error(f"❌ Error collecting observations: {e}")
            return {"error": str(e)}
    
    async def execute_recommendation(self, action_data):
        """Execute a recommendation from the cognitive agent"""
        try:
            # Check what kind of action is recommended
            action = action_data.get("action", "")
            
            if action == "reallocate_portfolio" and "allocations" in action_data:
                # Execute the rebalance
                allocations = action_data["allocations"]
                await self.trading_bot.execute_rebalance(allocations)
                return {"status": "success", "message": "Portfolio reallocated"}
                
            elif action == "reduce_risk":
                # Create safe allocations - more in stablecoin pools 
                safe_allocations = {
                    "pancakeswap": 0.2,
                    "traderjoe": 0.1,
                    "quickswap": 0.1
                    # Remaining 0.6 would go to stables in a real implementation
                }
                await self.trading_bot.execute_rebalance(safe_allocations)
                return {"status": "success", "message": "Risk reduced"}
                
            elif action == "gather_more_information":
                # Just return that we'll continue monitoring
                return {"status": "pending", "message": "Continuing to monitor market conditions"}
                
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"❌ Error executing recommendation: {e}")
            return {"status": "error", "message": str(e)}
            
    async def cognitive_decision_cycle(self):
        """Run one full autonomous decision cycle"""
        # 1. Collect observations from the environment
        observation = await self.collect_observations()
        
        # 2. Process through the cognitive agent's sense-think-act cycle
        cycle_result = self.agent.sense_think_act_cycle(observation)
        
        # 3. Execute the recommended action
        if cycle_result.get("thought") and "action" in cycle_result["thought"]:
            action = cycle_result["thought"]["action"]
            reasoning = cycle_result["thought"].get("reasoning", "No reasoning provided")
            
            logger.info(f"🧠 Agent reasoning: {reasoning}")
            logger.info(f"🚀 Executing recommended action: {action}")
            
            # Execute the action
            execution_result = await self.execute_recommendation(cycle_result["thought"])
            
            # Add the execution result to our memory
            self.agent.memory.add_short_term(f"Action execution result: {execution_result}")
            
            return {
                "observation": observation,
                "reasoning": reasoning,
                "action": action,
                "execution_result": execution_result
            }
        
        return {
            "observation": observation,
            "error": "No action recommended by cognitive process"
        }
    
    async def run_autonomous_loop(self, cycles=5, interval_seconds=30):
        """Run the autonomous agent for a specified number of cycles"""
        logger.info(f"🤖 Starting autonomous operation for {cycles} cycles...")
        
        try:
            # Initialize components
            await self.initialize()
            
            # Run the specified number of cycles
            for cycle in range(1, cycles + 1):
                logger.info(f"🔄 Starting decision cycle {cycle}/{cycles}")
                
                # Run one cognitive cycle
                result = await self.cognitive_decision_cycle()
                
                # Log the result
                if "error" in result:
                    logger.error(f"❌ Cycle {cycle} error: {result['error']}")
                else:
                    logger.info(f"✅ Cycle {cycle} completed successfully")
                    
                # Wait before the next cycle
                if cycle < cycles:
                    logger.info(f"⏳ Waiting {interval_seconds} seconds before next cycle...")
                    await asyncio.sleep(interval_seconds)
                    
            logger.info("🏁 Autonomous operation completed")
            
        except Exception as e:
            logger.error(f"❌ Error in autonomous loop: {e}")
            
        finally:
            # Cleanup
            self.agent.stop()
            await self.trading_bot.scanner.close()
            logger.info("🛑 System shutdown complete")

async def main():
    """Main entry point"""
    # Check for OpenAI API key for LLM reasoning
    use_llm = os.getenv("OPENAI_API_KEY") is not None
    
    # Initialize the autonomous trading system
    system = AutonomousTradingSystem(use_llm=use_llm)
    
    # Run the autonomous loop
    await system.run_autonomous_loop(cycles=3, interval_seconds=20)
    
if __name__ == "__main__":
    asyncio.run(main()) 