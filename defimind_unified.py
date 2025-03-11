#!/usr/bin/env python3
"""
DEFIMIND Unified Interface

This file provides a single entry point to all DEFIMIND capabilities,
combining the agent brain, communication, and DeFi analysis features.
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import all DEFIMIND components
from agent_brain import AutonomousAgent, Memory, Reasoning
from agent_communication import AgentCommunication
from ai_agent import TradingBot, AITradingAgent
from yield_scanner import YieldScanner
from autonomous_agent_main import AutonomousTradingSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("defimind_unified.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("defimind_unified")

# Load environment variables
load_dotenv()

class DefiMindUnified:
    """Unified interface for all DEFIMIND capabilities"""
    
    def __init__(self):
        """Initialize the unified DEFIMIND system"""
        # Get configuration from environment
        self.agent_name = os.getenv("AGENT_NAME", "Safe AI")
        self.agent_home = os.getenv("AGENT_HOME", "DEFIMIND")
        self.cycle_interval = int(os.getenv("AGENT_CYCLE_INTERVAL", "20"))
        self.simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"
        self.use_llm = os.getenv("AGENT_USE_LLM", "true").lower() == "true"
        
        # Components (initialized on demand)
        self.trading_system = None
        self.communication = None
        self.agent = None
        self.scanner = None
        
        logger.info(f"🚀 DEFIMIND Unified Interface initialized")
        logger.info(f"Agent: {self.agent_name}, Home: {self.agent_home}")
        logger.info(f"Simulation Mode: {self.simulation_mode}, LLM: {self.use_llm}")
        
    async def initialize(self):
        """Initialize all DEFIMIND components"""
        logger.info("Initializing all DEFIMIND components...")
        
        # Initialize the trading system
        self.trading_system = AutonomousTradingSystem(use_llm=self.use_llm)
        await self.trading_system.initialize()
        
        # Initialize communication
        self.communication = AgentCommunication(name=self.agent_name)
        self.communication.start_communication_loop()
        
        # Get references to other components for convenience
        self.agent = self.trading_system.agent
        self.scanner = self.trading_system.trading_bot.scanner
        
        logger.info("All DEFIMIND components initialized successfully")
        return True
        
    async def shutdown(self):
        """Shut down all DEFIMIND components"""
        logger.info("Shutting down DEFIMIND...")
        
        # Stop the agent
        if self.agent:
            self.agent.stop()
            
        # Stop communication
        if self.communication:
            self.communication.stop_communication()
            
        # Close connections
        if self.scanner:
            await self.scanner.close()
            
        logger.info("DEFIMIND shutdown complete")
        return True
        
    async def run_autonomous_cycle(self):
        """Run a single autonomous decision cycle"""
        if not self.trading_system:
            await self.initialize()
            
        result = await self.trading_system.cognitive_decision_cycle()
        return result
        
    async def analyze_protocol(self, protocol_name):
        """Analyze a specific DeFi protocol"""
        if not self.trading_system:
            await self.initialize()
            
        # First, check if we have data on this protocol
        market_data = self.trading_system.market_state.get("market_data", {})
        protocols_data = market_data.get("protocols", {})
        
        protocol_data = protocols_data.get(protocol_name, {})
        
        # If we don't have data, try to fetch it
        if not protocol_data:
            if self.communication:
                self.communication.send_message_to_user(
                    f"I don't have data on {protocol_name} in my current state. Attempting to fetch it..."
                )
                
            # Try to fetch from the scanner if available
            if self.scanner:
                protocol_pool = await self.scanner.get_best_apy_for_protocol(protocol_name)
                if protocol_pool:
                    protocol_data = {
                        "apy": protocol_pool.apy,
                        "pool": protocol_pool.name,
                        "tvl": protocol_pool.tvl
                    }
        
        # Build the analysis
        if self.agent.reasoning.use_llm:
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
            response = self.agent.reasoning.llm_reasoning(
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
            return f"I don't have sufficient data on {protocol_name.title()} to provide a detailed analysis."
    
    async def generate_report(self):
        """Generate an investment report"""
        if not self.trading_system:
            await self.initialize()
            
        return await self.trading_system.generate_daily_report()
    
    async def process_user_command(self, command_text):
        """Process a user command and return the result"""
        if not self.communication:
            await self.initialize()
            
        # Process the command
        command_result = self.communication.process_command(command_text)
        
        if command_result["action"] == "help":
            return {"type": "help", "content": command_result["message"]}
            
        elif command_result["action"] == "status":
            status_data = {
                "agent_name": self.agent_name,
                "agent_home": self.agent_home,
                "simulation_mode": self.simulation_mode,
                "use_llm": self.use_llm,
                "initialized": self.trading_system is not None
            }
            return {"type": "status", "content": status_data}
            
        elif command_result["action"] == "goals":
            if not self.agent:
                await self.initialize()
            goals = self.agent.goals
            return {"type": "goals", "content": goals}
            
        elif command_result["action"] == "report":
            report = await self.generate_report()
            return {"type": "report", "content": report}
            
        elif command_result["action"] == "analyze":
            protocol = command_result.get("protocol", "").lower()
            analysis = await self.analyze_protocol(protocol)
            return {"type": "analysis", "content": analysis, "protocol": protocol}
            
        elif command_result["action"] == "think":
            topic = command_result.get("topic", "")
            # Simple thought for now, could be enhanced with LLM
            return {
                "type": "thinking", 
                "content": f"I'm thinking about {topic} from the perspective of a DeFi investment strategy",
                "topic": topic
            }
            
        elif command_result["action"] == "natural_language":
            # Generate response to natural language
            if not self.trading_system:
                await self.initialize()
                
            agent_state = {
                "goals": self.agent.goals if self.agent else ["No goals set"],
                "last_observation": self.trading_system.market_state if self.trading_system else {},
                "recent_memories": self._get_recent_memories()
            }
            
            response = self.communication.generate_response(command_text, agent_state)
            return {"type": "conversation", "content": response}
        
        return {"type": "unknown", "content": "Command not recognized"}
    
    def _get_recent_memories(self):
        """Get recent memories from the agent for context"""
        if not self.agent or not hasattr(self.agent, "memory"):
            return []
            
        memories = []
        for memory in list(self.agent.memory.short_term)[-5:]:
            if isinstance(memory.get("content"), str):
                memories.append(memory["content"])
        return memories

    async def run_continuous(self, cycles=5, interval_seconds=None):
        """Run the agent continuously for a specified number of cycles"""
        if interval_seconds is None:
            interval_seconds = self.cycle_interval
            
        if not self.trading_system:
            await self.initialize()
            
        logger.info(f"Starting continuous operation for {cycles} cycles")
        
        for cycle in range(1, cycles + 1):
            logger.info(f"Running cycle {cycle}/{cycles}")
            
            # Run one cycle
            result = await self.run_autonomous_cycle()
            
            # Wait between cycles
            if cycle < cycles:
                await asyncio.sleep(interval_seconds)
                
        logger.info("Continuous operation completed")
        return True

    async def run_interactive_session(self):
        """Run an interactive session with the user"""
        if not self.communication:
            await self.initialize()
            
        self.communication.send_message_to_user(
            f"Hello! I am {self.agent_name} from {self.agent_home}. How can I assist you today?"
        )
        
        try:
            while True:
                user_input = self.communication.get_user_input()
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    self.communication.send_message_to_user("Goodbye! Have a great day.")
                    break
                    
                # Process the command
                result = await self.process_user_command(user_input)
                
                # Format and display the result
                if result["type"] == "help":
                    self.communication.send_message_to_user(result["content"])
                elif result["type"] == "status":
                    self.communication.send_message_to_user(
                        "Here's my current status:",
                        data=result["content"]
                    )
                elif result["type"] == "goals":
                    self.communication.send_message_to_user(
                        "These are my investment goals:",
                        data={"goals": result["content"]}
                    )
                elif result["type"] == "report":
                    self.communication.send_message_to_user(
                        "📊 INVESTMENT REPORT 📊",
                        data={"report": result["content"]}
                    )
                elif result["type"] == "analysis":
                    self.communication.send_message_to_user(
                        f"Here's my analysis of {result['protocol']}:",
                        data={"analysis": result["content"]}
                    )
                elif result["type"] == "thinking":
                    self.communication.send_message_to_user(result["content"])
                else:
                    self.communication.send_message_to_user(result["content"])
                    
        except KeyboardInterrupt:
            self.communication.send_message_to_user("Session interrupted. Goodbye!")
        finally:
            await self.shutdown()


async def run_defimind(mode="interactive", cycles=5, interval=None, protocol=None):
    """
    Single unified function to run DEFIMIND with all its capabilities
    
    Parameters:
        mode (str): Operating mode - "interactive", "continuous", "analyze", "report"
        cycles (int): Number of cycles for continuous mode
        interval (int): Seconds between cycles (defaults to env setting)
        protocol (str): Protocol name for analysis mode
        
    Returns:
        dict: Results of the operation
    """
    # Create the unified DEFIMIND instance
    defimind = DefiMindUnified()
    
    try:
        # Run in the selected mode
        if mode == "interactive":
            # Interactive session with the user
            await defimind.run_interactive_session()
            return {"status": "success", "mode": "interactive"}
            
        elif mode == "continuous":
            # Run autonomous cycles
            await defimind.run_continuous(cycles=cycles, interval_seconds=interval)
            return {"status": "success", "mode": "continuous", "cycles": cycles}
            
        elif mode == "analyze" and protocol:
            # Analyze a specific protocol
            await defimind.initialize()
            analysis = await defimind.analyze_protocol(protocol)
            return {
                "status": "success", 
                "mode": "analyze", 
                "protocol": protocol,
                "analysis": analysis
            }
            
        elif mode == "report":
            # Generate a report
            await defimind.initialize()
            report = await defimind.generate_report()
            return {"status": "success", "mode": "report", "report": report}
            
        else:
            return {"status": "error", "message": "Invalid mode or missing parameters"}
    
    except Exception as e:
        logger.error(f"Error in unified DEFIMIND function: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    
    finally:
        await defimind.shutdown()


# Command line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEFIMIND Unified Interface")
    parser.add_argument(
        "--mode",
        choices=["interactive", "continuous", "analyze", "report"],
        default="interactive",
        help="Operation mode"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Number of cycles for continuous mode"
    )
    parser.add_argument(
        "--interval",
        type=int,
        help="Interval between cycles in seconds"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        help="Protocol to analyze in analyze mode"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "analyze" and not args.protocol:
        print("Error: Protocol name required for analyze mode")
        sys.exit(1)
    
    # Run the unified function
    try:
        result = asyncio.run(run_defimind(
            mode=args.mode,
            cycles=args.cycles,
            interval=args.interval,
            protocol=args.protocol
        ))
        
        # Print result
        if result["status"] == "success":
            print(f"DEFIMIND {args.mode} operation completed successfully")
            if "analysis" in result:
                print("\n" + result["analysis"])
            elif "report" in result:
                print("\n" + result["report"])
        else:
            print(f"Error: {result['message']}")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1) 