#!/usr/bin/env python3
"""
Autonomous Agent Launcher

This script provides a simple way to launch the autonomous agent with different options.
It handles command line arguments and environment setup.
"""

import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# Ensure we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Launch the autonomous agent")
    
    parser.add_argument(
        "--mode", 
        choices=["continuous", "single", "communication"],
        default="continuous",
        help="Agent operation mode: continuous (default), single cycle, or communication only"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        help="Name for the agent (default from .env or 'TradingMind')"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        help="Interval between cycles in seconds (default from .env or 60)"
    )
    
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based reasoning even if API key is available"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional logging"
    )
    
    return parser.parse_args()

async def main():
    """Main entry point for the launcher"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = setup_args()
    
    # Set up configuration
    agent_name = args.name or os.getenv("AGENT_NAME", "TradingMind")
    interval = args.interval or int(os.getenv("AGENT_CYCLE_INTERVAL", "60"))
    use_llm = not args.no_llm and (os.getenv("AGENT_USE_LLM", "true").lower() == "true")
    
    # Handle debug mode
    if args.debug:
        os.environ["LOG_LEVEL"] = "DEBUG"
        print("🐛 Debug mode enabled")
    
    # Display startup information
    print("\n" + "=" * 60)
    print(f"🤖 Autonomous Agent: {agent_name}".center(60))
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Interval between cycles: {interval} seconds")
    print(f"LLM reasoning: {'Enabled' if use_llm else 'Disabled'}")
    
    if not os.getenv("OPENAI_API_KEY") and use_llm:
        print("\n⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("⚠️  The agent will run with limited reasoning capabilities.")
        use_llm = False
    
    print("\nStarting agent, please wait...")
    print("=" * 60 + "\n")
    
    # Import and launch based on selected mode
    if args.mode == "continuous":
        from continuous_agent_runner import ContinuousAgent
        
        agent = ContinuousAgent(name=agent_name, use_llm=use_llm)
        agent.cycle_interval = interval
        await agent.run_forever()
        
    elif args.mode == "single":
        from autonomous_agent_main import AutonomousTradingSystem
        
        system = AutonomousTradingSystem(use_llm=use_llm)
        await system.initialize()
        result = await system.cognitive_decision_cycle()
        
        print("\n" + "=" * 60)
        print("Single Cycle Result:".center(60))
        print("=" * 60)
        
        if "action" in result:
            print(f"Action: {result['action']}")
            print(f"Reasoning: {result.get('reasoning', 'No reasoning provided')}")
        elif "error" in result:
            print(f"Error: {result['error']}")
        
        # Cleanup
        if hasattr(system.agent, "stop"):
            system.agent.stop()
        if hasattr(system.trading_bot, "scanner") and hasattr(system.trading_bot.scanner, "close"):
            await system.trading_bot.scanner.close()
        
    elif args.mode == "communication":
        from agent_communication import AgentCommunication
        
        comm = AgentCommunication(name=agent_name)
        comm.start_communication_loop()
        
        print("Starting communication-only mode...")
        comm.send_message_to_user(f"Hello! I'm {agent_name}. This is communication-only mode.")
        comm.send_message_to_user("I won't be making autonomous decisions in this mode.")
        
        try:
            while True:
                user_input = comm.get_user_input()
                
                if user_input.lower() in ["exit", "quit", "bye"]:
                    comm.send_message_to_user("Goodbye! Have a great day.")
                    break
                    
                result = comm.process_command(user_input)
                
                if result["action"] == "help":
                    comm.send_message_to_user(result["message"])
                elif result["action"] == "stop":
                    comm.send_message_to_user("Shutting down, goodbye!")
                    break
                else:
                    # Generate a response
                    response = comm.generate_response(user_input, {
                        "goals": ["Maximize investment returns", "Minimize risk"],
                        "last_observation": {},
                        "recent_memories": []
                    })
                    comm.send_message_to_user(response)
                
        except KeyboardInterrupt:
            pass
        finally:
            comm.stop_communication()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Agent operation interrupted by user. Shutting down...")
        sys.exit(0) 