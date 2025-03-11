#!/usr/bin/env python3
"""
DEFIMIND LangChain Integration

This module integrates LangChain with the DEFIMIND agent's reasoning system.
It provides LLM-based processing for chat interactions and decision making.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langsmith import Client

# Configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("defimind_langchain")

# LangSmith configuration
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "DeFiMind")

# LLM configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

# Set environment variables programmatically if not set
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_TRACING"] = "true"

if OPENAI_API_KEY and LLM_PROVIDER == "openai":
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if OPENROUTER_API_KEY and LLM_PROVIDER == "openrouter":
    os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY

class LangChainAgent:
    """LangChain integration for DEFIMIND's reasoning and chat capabilities"""
    
    def __init__(self, 
                 model_name: str = None, 
                 temperature: float = 0.7,
                 trace_enabled: bool = True):
        """
        Initialize the LangChain agent
        
        Args:
            model_name: The name of the LLM model to use, defaults to provider-specific default
            temperature: LLM temperature parameter (0-1)
            trace_enabled: Whether to enable LangSmith tracing
        """
        self.temperature = temperature
        self.trace_enabled = trace_enabled and LANGSMITH_API_KEY is not None
        
        # Set default model based on provider
        if model_name is None:
            if LLM_PROVIDER == "openrouter":
                # OpenRouter default model
                self.model_name = "anthropic/claude-3-opus"
            else:
                # OpenAI default model
                self.model_name = "gpt-3.5-turbo"
        else:
            self.model_name = model_name
        
        # Initialize the LLM based on provider
        try:
            if LLM_PROVIDER == "openrouter":
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=temperature,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/defimind",
                        "X-Title": "DEFIMIND Trading Agent"
                    }
                )
                logger.info(f"Initialized OpenRouter with model {self.model_name}")
            else:
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=temperature
                )
                logger.info(f"Initialized ChatOpenAI with model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM with provider {LLM_PROVIDER}: {e}")
            self.llm = None
        
        # Initialize tracing if enabled
        self.tracer = None
        if self.trace_enabled:
            try:
                self.tracer = LangChainTracer(
                    project_name=LANGSMITH_PROJECT
                )
                logger.info(f"LangSmith tracing enabled for project '{LANGSMITH_PROJECT}'")
            except Exception as e:
                logger.error(f"Failed to initialize LangSmith tracing: {e}")
                self.trace_enabled = False
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Set up the system prompt
        self.system_prompt = self._get_system_prompt()
        
        # Set up the chains
        self._setup_chains()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        agent_name = os.getenv("AGENT_NAME", "DEFIMIND")
        return f"""You are {agent_name}, an autonomous trading agent for DeFi markets. You analyze blockchain data, 
        market conditions, and protocol performance to generate investment recommendations.
        
        Your capabilities include:
        1. Analyzing real-time blockchain data from Alchemy API and other sources
        2. Evaluating DeFi protocol performance and risks
        3. Generating trading recommendations optimized for yield and risk balance
        4. Explaining your reasoning process and market analysis
        
        When responding to queries:
        - Be factual and precise about market conditions
        - Explain your reasoning process for any recommendation
        - Provide context about relevant market factors
        - Be helpful and educational about DeFi concepts when needed
        
        Current date: {current_date}
        """
    
    def _setup_chains(self):
        """Set up the conversation and reasoning chains"""
        if not self.llm:
            logger.error("Cannot set up chains: LLM not initialized")
            return
        
        # Create the conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        # Create the reasoning chain with a structured prompt
        agent_name = os.getenv("AGENT_NAME", "DEFIMIND")
        current_date = datetime.now().strftime("%Y-%m-%d")
        reasoning_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt.format(current_date=current_date)),
            ("human", "{input}"),
        ])
        
        self.reasoning_chain = LLMChain(
            llm=self.llm, 
            prompt=reasoning_prompt,
            verbose=True
        )
        
        logger.info("LangChain chains initialized successfully")
    
    def process_message(self, user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process a user message and generate a response using LangChain
        
        Args:
            user_message: The message from the user
            context: Additional context to provide to the model
            
        Returns:
            The agent's response
        """
        if not self.llm:
            return "I'm sorry, the LLM-based reasoning system is not available right now."
        
        try:
            # Format context for the model
            formatted_context = ""
            if context:
                formatted_context = "\n\nCurrent context:\n"
                for key, value in context.items():
                    if isinstance(value, dict):
                        formatted_context += f"- {key}: {json.dumps(value, indent=2)}\n"
                    else:
                        formatted_context += f"- {key}: {value}\n"
            
            # Combine user message with context
            full_input = f"{user_message}\n{formatted_context}"
            
            # Use the reasoning chain with tracing if enabled
            if self.trace_enabled and self.tracer:
                with self.tracer.start_trace(project_name=LANGSMITH_PROJECT):
                    response = self.reasoning_chain.run(input=full_input)
            else:
                response = self.reasoning_chain.run(input=full_input)
            
            # Add to memory
            self.memory.save_context({"input": user_message}, {"output": response})
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message with LangChain: {e}")
            return f"I encountered an error while processing your request. Technical details: {str(e)}"
    
    def get_chain_of_thought(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed chain-of-thought reasoning for a specific question
        
        Args:
            question: The analysis question
            context: Market data and other context
            
        Returns:
            A dictionary with the reasoning steps and conclusion
        """
        if not self.llm:
            return {"error": "LLM not available"}
        
        try:
            # Create a prompt specifically for reasoning
            cot_prompt = f"""
            Question: {question}
            
            Context:
            {json.dumps(context, indent=2)}
            
            Thought process:
            1. Let me analyze the available data
            2. Consider relevant market factors
            3. Evaluate possible outcomes
            4. Reach a conclusion based on evidence
            
            Detailed reasoning:
            """
            
            # Run with tracing if enabled
            if self.trace_enabled and self.tracer:
                with self.tracer.start_trace(project_name=LANGSMITH_PROJECT):
                    reasoning = self.llm.invoke(cot_prompt)
            else:
                reasoning = self.llm.invoke(cot_prompt)
            
            result = {
                "question": question,
                "reasoning": reasoning.content if hasattr(reasoning, 'content') else str(reasoning),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error in chain-of-thought reasoning: {e}")
            return {"error": str(e)}
    
    def analyze_market_condition(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market conditions based on provided data
        
        Args:
            market_data: Dictionary containing market metrics
            
        Returns:
            Analysis results with reasoning
        """
        question = "What is the current market sentiment based on the provided data, and what trading strategy would be most appropriate?"
        return self.get_chain_of_thought(question, market_data)
    
    def evaluate_investment_opportunity(self, protocol_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a specific investment opportunity
        
        Args:
            protocol_data: Data about the protocol and opportunity
            
        Returns:
            Evaluation with risk assessment and recommendation
        """
        question = "Is this protocol a good investment opportunity considering the current market conditions, risks, and potential returns?"
        return self.get_chain_of_thought(question, protocol_data)
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history
        
        Returns:
            List of messages in the conversation
        """
        if not hasattr(self.memory, 'chat_memory') or not hasattr(self.memory.chat_memory, 'messages'):
            return []
        
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "agent", "content": message.content})
            elif isinstance(message, SystemMessage):
                history.append({"role": "system", "content": message.content})
        
        return history
    
    def clear_memory(self):
        """Clear the conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")


# Testing the LangChain integration
if __name__ == "__main__":
    agent = LangChainAgent()
    
    # Test a simple query
    response = agent.process_message("What's the current market sentiment for Ethereum?")
    print(f"Response: {response}")
    
    # Test with context
    context = {
        "market_data": {
            "eth_price": 3240.50,
            "eth_24h_change": 2.3,
            "gas_price_gwei": 25,
            "total_tvl_usd": 15000000000
        }
    }
    
    response = agent.process_message("Should I invest in Aave right now?", context)
    print(f"Response with context: {response}") 