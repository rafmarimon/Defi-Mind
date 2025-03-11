import os
import json
import logging
import numpy as np
import time
import threading
from datetime import datetime
from collections import deque
import tensorflow as tf
import openai
import requests
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Get agent configuration
AGENT_HOME = os.getenv("AGENT_HOME", "DEFIMIND")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_brain.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent_brain")

# Set up OpenAI API key - for LLM reasoning capabilities
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    logger.info("✅ OpenAI API key loaded")
else:
    logger.warning("⚠️ OpenAI API key not found. LLM reasoning disabled.")

# Set up LLM API configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER == "openai" and os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    logger.info("✅ OpenAI API initialized for reasoning")
    USE_LLM = True
elif LLM_PROVIDER == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    logger.info("✅ Openrouter API initialized for reasoning")
    USE_LLM = True
else:
    logger.warning("⚠️ No LLM API configured. LLM reasoning disabled.")
    USE_LLM = False

class Memory:
    """Short and long-term memory for the agent"""
    
    def __init__(self, short_term_capacity=100):
        self.short_term = deque(maxlen=short_term_capacity)
        self.long_term = []
        self.episodic = []
        
    def add_short_term(self, memory_item):
        """Add an item to short-term memory"""
        self.short_term.append({
            "timestamp": datetime.now().isoformat(),
            "content": memory_item
        })
        
    def add_long_term(self, memory_item, importance=0.5):
        """Add an important memory to long-term storage"""
        self.long_term.append({
            "timestamp": datetime.now().isoformat(),
            "content": memory_item,
            "importance": importance
        })
        
    def remember_episode(self, actions, observations, outcome, metadata=None):
        """Record a complete episode of interaction"""
        self.episodic.append({
            "timestamp": datetime.now().isoformat(),
            "actions": actions,
            "observations": observations,
            "outcome": outcome,
            "metadata": metadata or {}
        })
        
    def retrieve_relevant(self, query, k=5):
        """Retrieve most relevant memories to a query - for more advanced implementations, 
        this would use embeddings and semantic search"""
        # Simple keyword-based retrieval for now
        results = []
        
        # Search through all memory types
        for memory in list(self.short_term) + self.long_term:
            if isinstance(memory["content"], str) and query.lower() in memory["content"].lower():
                results.append(memory)
            
            if len(results) >= k:
                break
                
        return results
        
    def reflect(self):
        """Periodically consolidate short-term to long-term memory
        and identify patterns in episodic memory"""
        # This would be more sophisticated in a full implementation
        important_memories = list(self.short_term)[-10:]
        for memory in important_memories:
            if memory not in self.long_term:
                self.add_long_term(memory["content"], 0.7)
                
        return {
            "consolidated_memories": len(important_memories),
            "total_long_term": len(self.long_term),
            "total_episodic": len(self.episodic)
        }

class Perception:
    """Processes observations from environment into meaningful representations"""
    
    def __init__(self):
        self.current_state = {}
        self.attention_focus = None
        
    def process_observation(self, raw_observation):
        """Process raw data into structured perception"""
        processed = {
            "timestamp": datetime.now().isoformat(),
            "raw": raw_observation,
            "structured": self._structure_data(raw_observation)
        }
        
        self.current_state = processed
        return processed
        
    def _structure_data(self, data):
        """Convert raw data into structured representation"""
        # Implementation depends on the type of data
        if isinstance(data, dict):
            return data
        elif isinstance(data, str):
            # Try to parse as JSON
            try:
                return json.loads(data)
            except:
                # Return as text observation
                return {"text": data}
        elif isinstance(data, (list, tuple)):
            return {"sequence": data}
        else:
            return {"value": data}
            
    def focus_attention(self, aspect):
        """Direct attention to specific aspect of environment"""
        self.attention_focus = aspect
        
        if isinstance(self.current_state.get("structured"), dict):
            if aspect in self.current_state["structured"]:
                return {
                    "focus": aspect,
                    "value": self.current_state["structured"][aspect]
                }
        
        return {
            "focus": aspect,
            "value": None
        }

class Reasoning:
    """Decision-making module of the agent"""
    
    def __init__(self, use_llm=False):
        self.use_llm = use_llm and USE_LLM
        self.model = None
        self.llm_provider = LLM_PROVIDER
        
    def load_model(self, model_path=None):
        """Load a TensorFlow model for decision-making"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"✅ Reasoning model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load reasoning model: {e}")
            return False
            
    def llm_reasoning(self, prompt, system="You are an autonomous agent that helps make decisions."):
        """Use LLM for complex reasoning"""
        if not self.use_llm:
            return {"error": "LLM reasoning not available"}
            
        try:
            if self.llm_provider == "openai":
                # OpenAI API
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": system + f" You operate within {AGENT_HOME}, a sophisticated DeFi investment platform."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                return {
                    "reasoning": response.choices[0].message.content,
                    "model": "gpt-3.5-turbo-0125",
                    "tokens": response.usage.total_tokens
                }
            elif self.llm_provider == "openrouter":
                # Openrouter API
                headers = {
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/rafmarimon/DefiMind",
                }
                
                data = {
                    "model": "deepseek-ai/deepseek-coder",  # Can be configured based on preference
                    "messages": [
                        {"role": "system", "content": system + f" You operate within {AGENT_HOME}, a sophisticated DeFi investment platform."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    resp_json = response.json()
                    return {
                        "reasoning": resp_json["choices"][0]["message"]["content"],
                        "model": resp_json["model"],
                        "tokens": resp_json.get("usage", {}).get("total_tokens", 0)
                    }
                else:
                    logger.error(f"❌ Openrouter API error: {response.text}")
                    return {"error": f"Openrouter API error: {response.status_code}"}
            else:
                return {"error": "Unknown LLM provider"}
                
        except Exception as e:
            logger.error(f"❌ LLM reasoning error: {e}")
            return {"error": str(e)}
            
    def decide_action(self, perception, memory, goals, use_llm=None):
        """Make a decision based on current perception, memory and goals"""
        # Determine whether to use LLM (override possible)
        should_use_llm = self.use_llm if use_llm is None else use_llm
        
        # Complex situation requiring LLM reasoning
        if should_use_llm:
            prompt = f"""
# Investment Analysis Task

## Current Market Data:
{json.dumps(perception, indent=2)}

## Recent Portfolio History:
{json.dumps(list(memory.short_term)[-3:], indent=2)}

## Investment Goals:
{json.dumps(goals, indent=2)}

## Analysis Instructions:
1. Analyze the current APY rates across all protocols.
2. Identify the highest-performing opportunities with consideration for risk.
3. Determine if any existing positions should be exited (for profit-taking or loss mitigation).
4. Calculate optimal allocation across protocols to maximize returns while managing risk.
5. Consider gas costs for any rebalancing actions.

## Required Decision:
Based on this analysis, what specific action should be taken? Consider these options:
- MAINTAIN_POSITIONS: Keep current allocations if they remain optimal
- REBALANCE: Provide specific new allocation percentages across protocols
- TAKE_PROFITS: Exit specific positions that have reached profit targets
- REDUCE_RISK: Move capital to safer positions due to market conditions
- INCREASE_EXPOSURE: Allocate more capital to high-performing opportunities

Provide your investment banking analysis and a clear recommendation with allocation percentages.
"""
            llm_response = self.llm_reasoning(prompt, 
                system="You are an elite DeFi investment banker AI that analyzes market data and makes optimal allocation decisions. Your recommendations must be specific, data-driven, and focused on maximizing returns while managing risk.")
            
            if "error" not in llm_response:
                action_decision = self._extract_investment_decision(llm_response["reasoning"])
                return {
                    "action": action_decision["action"],
                    "allocations": action_decision.get("allocations", {}),
                    "reasoning": llm_response["reasoning"],
                    "method": "llm"
                }
        
        # Fallback to rule-based reasoning
        return self._rule_based_decision(perception, memory, goals)
        
    def _extract_investment_decision(self, llm_text):
        """Extract structured investment decision from LLM text response"""
        # Default response if parsing fails
        default_response = {
            "action": "gather_more_information",
            "allocations": {}
        }
        
        # Look for specific action recommendations
        action_mapping = {
            "MAINTAIN_POSITIONS": "maintain_positions",
            "REBALANCE": "reallocate_portfolio",
            "TAKE_PROFITS": "take_profits",
            "REDUCE_RISK": "reduce_risk",
            "INCREASE_EXPOSURE": "increase_exposure"
        }
        
        # Try to find an action recommendation
        for action_key, action_value in action_mapping.items():
            if action_key in llm_text:
                default_response["action"] = action_value
                break
        
        # Try to extract allocation percentages
        allocations = {}
        for protocol in ["pancakeswap", "traderjoe", "quickswap", "uniswap", "curve", "aave", "compound"]:
            # Look for allocation patterns like "PancakeSwap: 25%" or "Allocate 30% to QuickSwap"
            protocol_lower = protocol.lower()
            allocation_patterns = [
                f"{protocol}[:\s]?\s*(\d+)%",
                f"{protocol.title()}[:\s]?\s*(\d+)%",
                f"{protocol_lower}[:\s]?\s*(\d+)%",
                f"(\d+)%\s*(?:to|for|in)?\s*{protocol}",
                f"(\d+)%\s*(?:to|for|in)?\s*{protocol.title()}",
                f"(\d+)%\s*(?:to|for|in)?\s*{protocol_lower}"
            ]
            
            for pattern in allocation_patterns:
                match = re.search(pattern, llm_text, re.IGNORECASE)
                if match:
                    allocations[protocol_lower] = float(match.group(1)) / 100
                    break
        
        # Only include allocations if we found some
        if allocations:
            default_response["allocations"] = allocations
            
        return default_response
        
    def _rule_based_decision(self, perception, memory, goals):
        """Simple rule-based decision making as fallback"""
        # Enhanced rules for investment decisions
        
        # Extract market data if available
        market_data = perception.get("market_data", {})
        protocols_data = market_data.get("protocols", {})
        
        # Default allocations in case we need them
        default_allocations = {
            "pancakeswap": 0.33,
            "traderjoe": 0.33,
            "quickswap": 0.34
        }
        
        # Check for high volatility market conditions
        if market_data.get("volatility", 0) > 0.6:
            return {
                "action": "reduce_risk",
                "confidence": 0.8,
                "reasoning": "High market volatility detected. Reducing risk by moving to more stable positions.",
                "method": "rule_based"
            }
        
        # Check for significant APY differential (opportunity to rebalance)
        if protocols_data:
            # Calculate average APY
            apy_values = [p.get("apy", 0) for p in protocols_data.values()]
            if apy_values:
                avg_apy = sum(apy_values) / len(apy_values)
                
                # Find protocols with much higher APY than average
                high_apy_protocols = {
                    name: data.get("apy", 0) 
                    for name, data in protocols_data.items() 
                    if data.get("apy", 0) > avg_apy * 1.5
                }
                
                if high_apy_protocols:
                    # Calculate allocations based on relative APY values
                    total_high_apy = sum(high_apy_protocols.values())
                    allocations = {
                        name: (apy / total_high_apy) if total_high_apy > 0 else 0.5
                        for name, apy in high_apy_protocols.items()
                    }
                    
                    return {
                        "action": "reallocate_portfolio",
                        "allocations": allocations,
                        "confidence": 0.7,
                        "reasoning": f"Identified protocols with significantly higher APY: {', '.join(high_apy_protocols.keys())}",
                        "method": "rule_based"
                    }
        
        # If we have very little data, request more data
        if not protocols_data:
            return {
                "action": "gather_more_information",
                "confidence": 0.9,
                "reasoning": "Insufficient protocol data for making an informed investment decision",
                "method": "rule_based"
            }
                
        # Default: maintain current positions if no clear action emerges
        return {
            "action": "maintain_positions",
            "allocations": default_allocations,
            "confidence": 0.5,
            "reasoning": "Current positions appear optimal based on available data",
            "method": "rule_based"
        }

class AutonomousAgent:
    """Main agent class that integrates all cognitive components"""
    
    def __init__(self, name="Agent", use_llm=True):
        self.name = name
        self.memory = Memory()
        self.perception = Perception()
        self.reasoning = Reasoning(use_llm=use_llm)
        self.goals = []
        self.running = False
        self.reflection_interval = 60  # seconds
        self.thread = None
        
    def set_goals(self, goals):
        """Set the agent's current goals"""
        self.goals = goals
        self.memory.add_long_term(f"Goals set: {goals}", importance=0.9)
        logger.info(f"🎯 Goals set: {goals}")
        
    def observe(self, observation):
        """Process a new observation from the environment"""
        processed = self.perception.process_observation(observation)
        self.memory.add_short_term(f"Observed: {processed}")
        return processed
        
    def think(self, observation=None):
        """Reason about current state and decide on action"""
        if observation:
            self.observe(observation)
            
        action_decision = self.reasoning.decide_action(
            self.perception.current_state,
            self.memory,
            self.goals
        )
        
        # Record this thought process
        self.memory.add_short_term(f"Thought process: {action_decision}")
        
        return action_decision
        
    def act(self, action):
        """Execute an action and return the result"""
        # This would connect to actual action execution systems
        # For now it just logs the action
        logger.info(f"🤖 {self.name} is executing action: {action}")
        
        # Record the action in memory
        self.memory.add_short_term(f"Executed action: {action}")
        
        # Return a simulated result
        return {
            "status": "completed",
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
        
    def reflection_loop(self):
        """Background thread for periodic reflection"""
        while self.running:
            time.sleep(self.reflection_interval)
            if not self.running:
                break
                
            logger.info(f"🧠 {self.name} is reflecting on experiences...")
            reflection_results = self.memory.reflect()
            logger.info(f"✨ Reflection complete: {reflection_results}")
            
    def start(self):
        """Start the agent's autonomous processes"""
        if self.running:
            return False
            
        self.running = True
        logger.info(f"🚀 {self.name} is starting up!")
        
        # Start reflection thread
        self.thread = threading.Thread(target=self.reflection_loop)
        self.thread.daemon = True
        self.thread.start()
        
        return True
        
    def stop(self):
        """Stop the agent's autonomous processes"""
        if not self.running:
            return False
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            
        logger.info(f"🛑 {self.name} is shutting down")
        return True
        
    def sense_think_act_cycle(self, observation):
        """Run one complete cognitive cycle"""
        # Sense (perceive)
        processed_obs = self.observe(observation)
        
        # Think (reason)
        action_decision = self.think()
        
        # Act (execute)
        if "action" in action_decision:
            result = self.act(action_decision["action"])
            
            # Learn from experience
            self.memory.remember_episode(
                actions=action_decision,
                observations=processed_obs,
                outcome=result
            )
            
            return {
                "observation": processed_obs,
                "thought": action_decision,
                "action": result
            }
        
        return {
            "observation": processed_obs,
            "thought": action_decision,
            "action": None
        }

# Usage example
if __name__ == "__main__":
    # Initialize the agent
    agent = AutonomousAgent(name="TradingAssistant")
    
    # Set goals
    agent.set_goals([
        "Maximize investment returns",
        "Minimize risk exposure",
        "Adapt to changing market conditions"
    ])
    
    # Start agent
    agent.start()
    
    # Example observation
    market_data = {
        "market_data": {
            "volatility": 0.65,
            "trend": "bearish",
            "volume": "high",
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Process one cycle
    result = agent.sense_think_act_cycle(market_data)
    print(f"Cycle result: {json.dumps(result, indent=2)}")
    
    # Run for a while
    try:
        time.sleep(10)  # Let the agent run for 10 seconds
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the agent
        agent.stop() 