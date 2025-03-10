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
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        self.use_llm = use_llm and OPENAI_API_KEY is not None
        self.model = None
        
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
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": system},
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
Current Perception:
{json.dumps(perception, indent=2)}

Recent Memory Items:
{json.dumps(list(memory.short_term)[-3:], indent=2)}

Agent's Goals:
{json.dumps(goals, indent=2)}

Based on the information above, what action should the agent take next?
Provide your reasoning and specific action recommendation.
"""
            llm_response = self.llm_reasoning(prompt)
            if "error" not in llm_response:
                return {
                    "action": self._extract_action(llm_response["reasoning"]),
                    "reasoning": llm_response["reasoning"],
                    "method": "llm"
                }
        
        # Fallback to rule-based reasoning
        return self._rule_based_decision(perception, memory, goals)
        
    def _rule_based_decision(self, perception, memory, goals):
        """Simple rule-based decision making as fallback"""
        # This would be expanded with actual domain-specific rules
        
        # Example for trading domain
        if "market_data" in perception:
            if perception["market_data"].get("volatility", 0) > 0.5:
                return {
                    "action": "reduce_risk",
                    "confidence": 0.8,
                    "reasoning": "High market volatility detected",
                    "method": "rule_based"
                }
                
        # Default response if no rules match
        return {
            "action": "gather_more_information",
            "confidence": 0.3,
            "reasoning": "Insufficient information for confident decision",
            "method": "rule_based"
        }
        
    def _extract_action(self, llm_text):
        """Extract structured action from LLM text response"""
        # This is a simplified implementation
        # A more sophisticated version would do better parsing
        
        if "recommend" in llm_text.lower() and ":" in llm_text:
            parts = llm_text.split(":")
            for part in parts:
                if "recommend" in part.lower():
                    return parts[parts.index(part) + 1].strip()
                    
        # Default extraction - last paragraph, first sentence
        paragraphs = [p for p in llm_text.split("\n\n") if p.strip()]
        if paragraphs:
            sentences = [s.strip() for s in paragraphs[-1].split(".") if s.strip()]
            if sentences:
                return sentences[0]
                
        return "no_clear_action"

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