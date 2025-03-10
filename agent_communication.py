import os
import asyncio
import logging
import json
import time
import threading
from datetime import datetime
from queue import Queue
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("agent_communication.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent_communication")

class AgentCommunication:
    """Handles communication between the autonomous agent and the user"""
    
    def __init__(self, name="Agent"):
        self.name = name
        self.message_history = []
        self.outgoing_messages = Queue()
        self.incoming_messages = Queue()
        self.running = False
        self.thread = None
        self.use_llm = os.getenv("OPENAI_API_KEY") is not None
        
        # Set up OpenAI client if API key is available
        if self.use_llm:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            logger.info("✅ OpenAI API initialized for communication")
        
    def start_communication_loop(self):
        """Start the communication thread"""
        if self.running:
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._communication_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"📢 Communication interface for {self.name} started")
        return True
        
    def stop_communication(self):
        """Stop the communication thread"""
        if not self.running:
            return False
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        
        logger.info(f"🛑 Communication interface for {self.name} stopped")
        return True
        
    def _communication_loop(self):
        """Background thread for handling message processing"""
        while self.running:
            # Process outgoing messages (agent to user)
            self._process_outgoing_messages()
            
            # Brief pause to prevent high CPU usage
            time.sleep(0.1)
        
    def _process_outgoing_messages(self):
        """Process messages from agent to user"""
        if not self.outgoing_messages.empty():
            message = self.outgoing_messages.get()
            
            # Record in history
            self.message_history.append({
                "role": "agent",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display the message to the user
            self._display_agent_message(message)
            
            # Mark as done
            self.outgoing_messages.task_done()
    
    def _display_agent_message(self, message):
        """Format and display agent message to the user"""
        # You can customize this for different display methods
        time_str = datetime.now().strftime("%H:%M:%S")
        
        if isinstance(message, dict):
            # If it's a structured message, format it nicely
            if "message" in message:
                print(f"\n🤖 [{time_str}] {self.name}: {message['message']}")
                
                # If there's additional data, show it in a structured way
                if "data" in message and message["data"]:
                    print(f"📊 Additional data:")
                    for key, value in message["data"].items():
                        if isinstance(value, (dict, list)):
                            print(f"  • {key}: {json.dumps(value, indent=2)}")
                        else:
                            print(f"  • {key}: {value}")
            else:
                # Just print the JSON
                print(f"\n🤖 [{time_str}] {self.name}: {json.dumps(message, indent=2)}")
        else:
            # Simple text message
            print(f"\n🤖 [{time_str}] {self.name}: {message}")
    
    def send_message_to_user(self, message, data=None):
        """Agent sends a message to the user"""
        if data:
            msg_obj = {"message": message, "data": data}
        else:
            msg_obj = message
            
        self.outgoing_messages.put(msg_obj)
        return True
    
    def get_user_input(self, prompt=None):
        """Synchronously get input from the user"""
        if prompt:
            print(f"\n❓ {self.name}: {prompt}")
        
        try:
            user_input = input("\n💬 You: ")
            
            # Record in history
            self.message_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            })
            
            return user_input
        except (KeyboardInterrupt, EOFError):
            return "exit"
            
    async def async_get_user_input(self, prompt=None):
        """Asynchronously get input from the user"""
        if prompt:
            print(f"\n❓ {self.name}: {prompt}")
            
        # Create a future that will store the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        # Run the input function in a separate thread to avoid blocking
        def get_input():
            try:
                user_input = input("\n💬 You: ")
                loop.call_soon_threadsafe(future.set_result, user_input)
                
                # Record in history
                self.message_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
            except (KeyboardInterrupt, EOFError):
                loop.call_soon_threadsafe(future.set_result, "exit")
        
        # Run the input function in a separate thread
        threading.Thread(target=get_input).start()
        
        # Wait for the input
        return await future
        
    def process_command(self, command_text):
        """Process a command from the user"""
        # Basic command processing
        command = command_text.strip().lower()
        
        if command in ["quit", "exit", "stop"]:
            return {"action": "stop", "success": True, "message": "Stopping the agent..."}
        
        elif command in ["help", "?"]:
            help_text = """
Available commands:
- help, ? - Show this help message
- status - Get the current status of the agent
- goals - Show the agent's current goals
- stop, quit, exit - Stop the agent
- memory - Show agent's recent memories
- think about [topic] - Ask the agent to think about a specific topic
            """
            return {"action": "help", "success": True, "message": help_text}
        
        elif command == "status":
            return {"action": "status", "success": True}
        
        elif command == "goals":
            return {"action": "goals", "success": True}
            
        elif command == "memory":
            return {"action": "memory", "success": True}
            
        elif command.startswith("think about "):
            topic = command[12:].strip()
            return {"action": "think", "topic": topic, "success": True}
        
        else:
            # If it's not a specific command, treat it as natural language
            return {"action": "natural_language", "text": command_text, "success": True}
            
    def generate_response(self, user_input, agent_state):
        """Generate a natural language response based on user input and agent state"""
        if not self.use_llm:
            # Fallback to template responses if LLM not available
            return self._template_response(user_input, agent_state)
            
        try:
            # Create a prompt for the LLM
            system_prompt = f"""You are {self.name}, an autonomous AI agent with these capabilities:
1. Making trading decisions for DeFi protocols
2. Analyzing market data and yield opportunities
3. Learning from past experiences
4. Adapting to changing market conditions

Respond as if you are the agent directly communicating with the user.
Keep responses concise but informative.
"""
            
            # Include relevant agent state in the prompt
            context = f"""
Current agent state:
- Goals: {agent_state.get('goals', ['No goals set'])}
- Last observation: {json.dumps(agent_state.get('last_observation', {}), indent=2)}
- Last action: {agent_state.get('last_action', 'No recent actions')}

Recent memory: {json.dumps(agent_state.get('recent_memories', []), indent=2)}
"""
            
            user_message = f"User message: {user_input}"
            
            # Call the OpenAI API to generate a response
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context + "\n\n" + user_message}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"❌ Error generating LLM response: {e}")
            return self._template_response(user_input, agent_state)
    
    def _template_response(self, user_input, agent_state):
        """Generate template responses when LLM is not available"""
        user_input_lower = user_input.lower()
        
        # Simple pattern matching
        if "hello" in user_input_lower or "hi" in user_input_lower:
            return f"Hello! I am {self.name}. How can I assist you today?"
            
        elif "status" in user_input_lower or "how are you" in user_input_lower:
            return "I'm operating normally and monitoring the markets for optimal yield opportunities."
            
        elif "market" in user_input_lower:
            return "I'm currently analyzing market conditions across multiple DeFi protocols."
            
        elif any(word in user_input_lower for word in ["recommendation", "suggest", "advise"]):
            return "Based on my latest analysis, I would recommend maintaining a diversified position across protocols."
            
        else:
            return "I understand you're asking about something, but I'm not sure how to respond. Could you try phrasing it differently?"

# For direct testing
if __name__ == "__main__":
    comm = AgentCommunication("TestAgent")
    comm.start_communication_loop()
    
    try:
        comm.send_message_to_user("Hello! I'm your agent assistant. How can I help you today?")
        
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
                    "goals": ["Test goal 1", "Test goal 2"],
                    "last_observation": {"test": "data"},
                    "recent_memories": ["Memory 1", "Memory 2"]
                })
                comm.send_message_to_user(response)
            
    except KeyboardInterrupt:
        pass
    finally:
        comm.stop_communication() 