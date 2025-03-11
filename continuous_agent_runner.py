import os
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Import our agent components
from agent_brain import AutonomousAgent, Memory
from ai_agent import TradingBot
from agent_communication import AgentCommunication
from autonomous_agent_main import AutonomousTradingSystem

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("continuous_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("continuous_agent")

class ContinuousAgent:
    """A continuous running agent that interacts with the user"""
    
    def __init__(self, name="TradingMind", use_llm=True):
        # Initialize the trading system
        self.trading_system = AutonomousTradingSystem(use_llm=use_llm)
        
        # Initialize the communication interface
        self.communication = AgentCommunication(name=name)
        
        # Control flags
        self.running = False
        self.paused = False
        
        # Interval for autonomous cycles (in seconds)
        self.cycle_interval = 60  # Default to 1 minute
        
        # Track state
        self.last_cycle_time = None
        self.cycle_count = 0
        self.last_user_interaction = None
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("🚀 Initializing continuous agent...")
        
        # Initialize trading system components
        await self.trading_system.initialize()
        
        # Start communication interface
        self.communication.start_communication_loop()
        
        # Set control flags
        self.running = True
        self.paused = False
        
        # Record initialization time
        self.last_cycle_time = datetime.now()
        self.last_user_interaction = datetime.now()
        
        logger.info("✅ Continuous agent initialized and ready")
        
    async def run_forever(self):
        """Run the agent continuously until stopped"""
        try:
            # Initialize components
            await self.initialize()
            
            # Welcome message
            self.communication.send_message_to_user(
                f"Hello! I'm {self.communication.name}, your autonomous trading assistant. I'm now running and monitoring the markets."
            )
            self.communication.send_message_to_user(
                "You can interact with me at any time. Type 'help' to see available commands."
            )
            
            # Main loop
            while self.running:
                try:
                    # Check if it's time to run an autonomous cycle
                    current_time = datetime.now()
                    time_since_last_cycle = (current_time - self.last_cycle_time).total_seconds()
                    
                    # Check for user input (non-blocking)
                    await self.check_for_user_input()
                    
                    # If not paused and it's time for an autonomous cycle
                    if not self.paused and time_since_last_cycle >= self.cycle_interval:
                        # Run one autonomous cycle
                        result = await self.run_autonomous_cycle()
                        
                        # Update counters
                        self.cycle_count += 1
                        self.last_cycle_time = current_time
                        
                        # Notify user about what the agent just did
                        self.notify_user_about_cycle_result(result)
                    
                    # Pause briefly to avoid consuming too much CPU
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    self.communication.send_message_to_user(
                        f"I encountered an error in my processing cycle: {str(e)}. I'll continue running."
                    )
            
            # Goodbye message
            self.communication.send_message_to_user(
                "I'm shutting down now. Thank you for working with me!"
            )
            
        except Exception as e:
            logger.error(f"❌ Fatal error in continuous agent: {e}")
            self.communication.send_message_to_user(
                f"I encountered a critical error and need to shut down: {str(e)}"
            )
            
        finally:
            # Cleanup
            await self.shutdown()
    
    async def check_for_user_input(self):
        """Check for and process user input in a non-blocking way"""
        # Check if there's any input available
        if not sys.stdin.isatty():
            return
            
        # Use asyncio to check for input without blocking
        try:
            # Set stdin to non-blocking mode
            import fcntl
            import os
            
            old_flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
            
            # Try to read input
            user_input = sys.stdin.readline().strip()
            
            # Reset to blocking mode
            fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, old_flags)
            
            # If we got input, process it
            if user_input:
                await self.process_user_input(user_input)
                self.last_user_interaction = datetime.now()
                
        except (BlockingIOError, Exception):
            # No input available or error reading input
            pass
    
    async def process_user_input(self, user_input):
        """Process input from the user"""
        # Process the command
        command_result = self.communication.process_command(user_input)
        
        if command_result["action"] == "stop":
            # Stop the agent
            self.running = False
            self.communication.send_message_to_user("I'm shutting down as requested.")
            
        elif command_result["action"] == "help":
            # Display help
            self.communication.send_message_to_user(command_result["message"])
            
        elif command_result["action"] == "status":
            # Return agent status
            status_data = {
                "running": self.running,
                "paused": self.paused,
                "cycles_completed": self.cycle_count,
                "last_cycle": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
                "cycle_interval": f"{self.cycle_interval} seconds",
                "time_to_next_cycle": f"{self.cycle_interval - (datetime.now() - self.last_cycle_time).total_seconds():.1f} seconds" if self.last_cycle_time else "Now"
            }
            
            self.communication.send_message_to_user(
                "Here's my current status:",
                data=status_data
            )
            
        elif command_result["action"] == "goals":
            # Return agent goals
            goals = self.trading_system.agent.goals
            self.communication.send_message_to_user(
                "These are my current investment goals:",
                data={"goals": goals}
            )
            
        elif command_result["action"] == "report":
            # Generate a report on demand
            self.communication.send_message_to_user(
                "Generating an investment report for you..."
            )
            
            report = await self.trading_system.generate_daily_report()
            if report:
                self.communication.send_message_to_user(
                    "📊 INVESTMENT REPORT 📊",
                    data={"report": report}
                )
            else:
                self.communication.send_message_to_user(
                    "I wasn't able to generate a report at this time."
                )
            
        elif command_result["action"] == "memory":
            # Show recent memories
            short_term = list(self.trading_system.agent.memory.short_term)[-5:]
            
            memory_data = {
                "short_term_memory": short_term,
                "total_memories": {
                    "short_term": len(self.trading_system.agent.memory.short_term),
                    "long_term": len(self.trading_system.agent.memory.long_term),
                    "episodic": len(self.trading_system.agent.memory.episodic)
                }
            }
            
            self.communication.send_message_to_user(
                "Here are my recent memories:",
                data=memory_data
            )
            
        elif command_result["action"] == "think":
            # Perform thinking about a topic
            topic = command_result.get("topic", "")
            
            self.communication.send_message_to_user(f"I'll think about '{topic}' and get back to you...")
            
            # Get agent to think about this topic
            thought_result = await self.think_about_topic(topic)
            
            self.communication.send_message_to_user(
                f"Here are my thoughts about '{topic}':",
                data={"thoughts": thought_result}
            )
            
        elif command_result["action"] == "natural_language":
            # Generate a response based on the agent's state
            agent_state = self.get_agent_state()
            
            response = self.communication.generate_response(
                user_input,
                agent_state
            )
            
            self.communication.send_message_to_user(response)
        
        elif command_result["action"] == "analyze":
            # Analyze a specific protocol
            protocol = command_result.get("protocol", "").lower()
            
            self.communication.send_message_to_user(
                f"Analyzing {protocol}... Please wait while I gather data."
            )
            
            # Analyze the requested protocol
            analysis = await self.analyze_protocol(protocol)
            
            self.communication.send_message_to_user(
                f"Here's my analysis of {protocol}:",
                data={"protocol_analysis": analysis}
            )
    
    async def run_autonomous_cycle(self):
        """Run one cognitive cycle of the agent"""
        logger.info(f"🔄 Running autonomous cycle #{self.cycle_count + 1}")
        
        # Notify user that we're starting a cycle
        self.communication.send_message_to_user(
            f"I'm starting autonomous cycle #{self.cycle_count + 1} to analyze market conditions..."
        )
        
        # Run the cognitive cycle
        result = await self.trading_system.cognitive_decision_cycle()
        
        logger.info(f"✅ Completed autonomous cycle #{self.cycle_count + 1}")
        return result
    
    def notify_user_about_cycle_result(self, result):
        """Notify the user about the result of an autonomous cycle"""
        # First, check if there's a daily report to display
        if "daily_report" in result and result["daily_report"]:
            self.communication.send_message_to_user(
                "📊 DAILY INVESTMENT REPORT 📊",
                data={"daily_report": result["daily_report"]}
            )
        
        # Check if the cycle produced an action
        if "action" in result and result["action"]:
            action = result["action"]
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Format the message based on the action type
            if action == "reallocate_portfolio":
                allocations = result.get("execution_result", {}).get("allocations", {})
                allocation_text = ", ".join([f"{k}: {v*100:.1f}%" for k, v in allocations.items()])
                
                self.communication.send_message_to_user(
                    f"I've analyzed the market and decided to reallocate the portfolio as follows: {allocation_text}",
                    data={
                        "reasoning": reasoning,
                        "allocations": allocations,
                        "execution_result": result.get("execution_result", {})
                    }
                )
                
            elif action == "take_profits":
                self.communication.send_message_to_user(
                    f"I've identified a profit-taking opportunity and executed the trade.",
                    data={
                        "reasoning": reasoning,
                        "execution_result": result.get("execution_result", {})
                    }
                )
                
            elif action == "reduce_risk":
                self.communication.send_message_to_user(
                    f"Market volatility has increased beyond my threshold. I've reduced our risk exposure by moving to safer positions.",
                    data={
                        "reasoning": reasoning,
                        "execution_result": result.get("execution_result", {})
                    }
                )
                
            elif action == "maintain_positions":
                self.communication.send_message_to_user(
                    f"After analyzing current market conditions, I've determined our current positions remain optimal.",
                    data={
                        "reasoning": reasoning,
                        "execution_result": result.get("execution_result", {})
                    }
                )
                
            elif action == "increase_exposure":
                self.communication.send_message_to_user(
                    f"I've identified particularly strong opportunities and increased our exposure to capitalize on them.",
                    data={
                        "reasoning": reasoning,
                        "execution_result": result.get("execution_result", {})
                    }
                )
                
            else:
                self.communication.send_message_to_user(
                    f"I've completed my analysis and decided to: {action}",
                    data={
                        "reasoning": reasoning,
                        "execution_result": result.get("execution_result", {})
                    }
                )
                
        elif "error" in result:
            # There was an error
            self.communication.send_message_to_user(
                f"I encountered an issue during my market analysis: {result['error']}",
                data={"observation": result.get("observation", {})}
            )
        else:
            # No specific action was taken
            self.communication.send_message_to_user(
                "I've completed my market analysis but no action was required at this time."
            )
    
    def get_agent_state(self):
        """Get the current state of the agent for communication purposes"""
        # Get agent goals
        goals = self.trading_system.agent.goals
        
        # Get recent memories
        recent_memories = []
        for memory in list(self.trading_system.agent.memory.short_term)[-5:]:
            if isinstance(memory["content"], str) and len(memory["content"]) < 500:
                recent_memories.append(memory["content"])
            elif isinstance(memory["content"], dict):
                recent_memories.append(str(memory["content"]))
        
        # Get last observation
        last_observation = self.trading_system.market_state
        
        # Get last action
        last_action = "No recent action"
        for memory in reversed(list(self.trading_system.agent.memory.short_term)):
            if isinstance(memory["content"], str) and "Executed action:" in memory["content"]:
                last_action = memory["content"].replace("Executed action:", "").strip()
                break
        
        return {
            "goals": goals,
            "recent_memories": recent_memories,
            "last_observation": last_observation,
            "last_action": last_action,
            "paused": self.paused,
            "cycles_completed": self.cycle_count
        }
    
    async def think_about_topic(self, topic):
        """Ask the agent to think about a specific topic"""
        # Create a thinking prompt
        prompt = f"""
As a trading agent with my current knowledge and capabilities, 
I need to think about: {topic}

Consider:
1. How does this relate to my goals of maximizing returns and minimizing risk?
2. What historical data would be relevant to this topic?
3. What actions might be appropriate given this topic?
4. What might be the potential risks and opportunities?
"""
        
        # Use the reasoning engine to think about this
        if hasattr(self.trading_system.agent.reasoning, "llm_reasoning") and self.trading_system.agent.reasoning.use_llm:
            # Use LLM if available
            thought_result = self.trading_system.agent.reasoning.llm_reasoning(
                prompt,
                system="You are an autonomous trading agent thinking through a problem. Provide your analysis."
            )
            
            # Extract the reasoning
            if isinstance(thought_result, dict) and "reasoning" in thought_result:
                return thought_result["reasoning"]
            else:
                return str(thought_result)
        else:
            # Simple fallback if LLM not available
            return f"I've considered {topic} from multiple angles, but without my advanced reasoning capabilities, I can only provide limited insights. This topic appears relevant to market analysis and trading strategies."
    
    async def analyze_protocol(self, protocol_name):
        """Perform a detailed analysis of a specific DeFi protocol"""
        try:
            # First, check if we have data on this protocol
            market_data = self.trading_system.market_state.get("market_data", {})
            protocols_data = market_data.get("protocols", {})
            
            protocol_data = protocols_data.get(protocol_name, {})
            
            # If we don't have data, try to fetch it
            if not protocol_data:
                self.communication.send_message_to_user(
                    f"I don't have data on {protocol_name} in my current state. Attempting to fetch it..."
                )
                
                # Try to fetch from the scanner if available
                if hasattr(self.trading_system.trading_bot, "scanner") and hasattr(self.trading_system.trading_bot.scanner, "get_best_apy_for_protocol"):
                    protocol_pool = await self.trading_system.trading_bot.scanner.get_best_apy_for_protocol(protocol_name)
                    if protocol_pool:
                        protocol_data = {
                            "apy": protocol_pool.apy,
                            "pool": protocol_pool.name,
                            "tvl": protocol_pool.tvl
                        }
            
            # Build the analysis, either with data or as a theoretical analysis
            if self.trading_system.agent.reasoning.use_llm:
                # Use LLM for a comprehensive analysis
                prompt = f"""
Generate a detailed investment analysis for the DeFi protocol: {protocol_name.title()}

Available Data:
{json.dumps(protocol_data, indent=2) if protocol_data else "Limited data available on this protocol."}

Market Context:
{json.dumps(market_data, indent=2) if market_data else "Limited market context available."}

Your analysis should include:
1. Overview of the protocol and its key features
2. Current yield opportunities and their sustainability
3. Risk assessment (protocol risk, smart contract risk, impermanent loss risk)
4. Liquidity analysis and TVL trends
5. Comparative analysis with similar protocols
6. Investment recommendation (whether to allocate capital and at what percentage)
7. Outlook for next 30 days
"""
                response = self.trading_system.agent.reasoning.llm_reasoning(
                    prompt,
                    system="You are an expert DeFi analyst and investment advisor. Provide detailed, data-driven analysis of DeFi protocols with specific recommendations. Be precise about numbers and percentages."
                )
                
                if "error" not in response:
                    return response["reasoning"]
                    
            # Fallback to simple analysis if LLM unavailable or errored
            if protocol_data:
                analysis = f"Analysis of {protocol_name.title()}:\n\n"
                analysis += f"Current APY: {protocol_data.get('apy', 'Unknown')*100:.2f}%\n"
                analysis += f"Pool: {protocol_data.get('pool', 'Unknown')}\n"
                analysis += f"TVL: ${protocol_data.get('tvl', 'Unknown'):,}\n\n"
                
                # Simple recommendation based on APY
                apy = protocol_data.get('apy', 0)
                if apy > 0.3:  # 30% APY
                    analysis += "Recommendation: Consider a significant allocation due to high APY."
                elif apy > 0.15:  # 15% APY
                    analysis += "Recommendation: Consider a moderate allocation."
                elif apy > 0.05:  # 5% APY
                    analysis += "Recommendation: Consider a small allocation."
                else:
                    analysis += "Recommendation: Monitor but consider alternatives with higher yield."
                
                return analysis
            else:
                return f"I don't have sufficient data on {protocol_name.title()} to provide a detailed analysis. Consider using 'think about {protocol_name}' for a more theoretical assessment."
                
        except Exception as e:
            logger.error(f"❌ Error analyzing protocol {protocol_name}: {e}")
            return f"Error analyzing {protocol_name}: {str(e)}"
    
    async def shutdown(self):
        """Shutdown all components"""
        logger.info("🛑 Shutting down continuous agent...")
        
        # Stop the agent
        self.running = False
        
        # Stop communication
        self.communication.stop_communication()
        
        # Stop the trading system
        if hasattr(self.trading_system, "agent") and self.trading_system.agent:
            self.trading_system.agent.stop()
            
        # Close any connections
        if hasattr(self.trading_system, "trading_bot") and hasattr(self.trading_system.trading_bot, "scanner"):
            await self.trading_system.trading_bot.scanner.close()
            
        logger.info("✅ Continuous agent shutdown complete")

async def main():
    """Main entry point"""
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()
    
    # Check for OpenAI API key
    use_llm = os.getenv("OPENAI_API_KEY") is not None
    
    # Create the continuous agent
    agent = ContinuousAgent(name="TradingMind", use_llm=use_llm)
    
    # Handle keyboard interrupt
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(agent.shutdown()))
    
    # Print welcome message
    print("\n" + "=" * 50)
    print("🤖 Continuous Autonomous Agent".center(50))
    print("=" * 50 + "\n")
    print("Starting the agent. You can interact with it by typing commands or questions.")
    print("Type 'help' to see available commands or 'exit' to quit.\n")
    
    # Run the agent forever
    await agent.run_forever()
    
if __name__ == "__main__":
    asyncio.run(main()) 