# Autonomous Agent with Cognitive Architecture

This project implements an autonomous agent with a cognitive architecture, designed to think and act on its own. The initial implementation focuses on DeFi yield analysis and trading decisions, but the architecture can be adapted for various autonomous applications.

## 🧠 Architecture Overview

The agent consists of several cognitive components that work together to create an intelligent, autonomous system:

1. **Memory System**: Multi-layered memory including short-term, long-term, and episodic memory
2. **Perception Module**: Processes raw observations into structured representations
3. **Reasoning Engine**: Makes decisions using rule-based logic or LLM assistance
4. **Action System**: Executes decisions and interacts with the environment
5. **Reflection Loop**: Periodically reviews experiences and consolidates memories

![Agent Architecture](https://mermaid.ink/img/pako:eNp1kc1OwzAQhF9l5XMr9Q04IKCqkBAqpygnywkm8TZ2ZduVlIK8O-tQUH9Evcz8O17PXtkJK5oxG7JLrAK1j7WzkrxjF9kCIYwxcIgRrwRd0A38u24gZYvOpTQk1kMnFnN2uFu0HTz2uLVYwLUv4IZcFQK1iR0YpXwrELVTgMXoDkY4PYYyumIIRF7OoL6Y0e4zdWgxQcUBTw2_4UR-wBvDp78_-JKRZXOGq8RqJ6ug-0ZTLCr4PHzL7H8Z0aZ0MCrCE44Jd6lXVHkUb-Z1V0_2WrfrtrcFbXZ1UzXr5q669_tVs51W93W1aTfr5dP6dlPOhR1kkvPCFuLTaGUHzwYoaR4qS1l4OYyf49Cx4XwKCkvOtx6zrQQ15p91B9Kh1V0xIlMWL0XO3yyp2XCqbF4qmzObLXOq-OkBo8zGSvTsbHjH3m8dOY13)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- MacOS (tested on MacBook Pro)
- Optional: OpenAI API key for LLM-powered reasoning

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autonomous-agent.git
cd autonomous-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

5. Edit the `.env` file and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
INFURA_RPC_URL=your_infura_rpc_url
```

### Running the Agent

To run the autonomous agent:

```bash
python autonomous_agent_main.py
```

### Configuration

You can configure the agent's behavior by modifying these files:

- `autonomous_agent_main.py`: Main entry point and configuration
- `agent_brain.py`: Cognitive architecture components
- `.env`: API keys and environment variables

## 🛠️ Core Components

### AutonomousAgent

The main agent class that integrates all cognitive components. Key methods:

- `set_goals()`: Define the agent's objectives
- `observe()`: Process observations from the environment
- `think()`: Reason about the current state and decide on actions
- `act()`: Execute actions based on decisions
- `sense_think_act_cycle()`: Run one complete cognitive cycle
- `start()/stop()`: Control the agent's operation

### Memory System

Multi-layered memory structure:

- **Short-term memory**: Recent observations and thoughts
- **Long-term memory**: Important knowledge and experiences
- **Episodic memory**: Complete sequences of agent interactions

### Reasoning Engine

Decision-making system with multiple reasoning methods:

- **LLM-powered reasoning**: Uses OpenAI's models for complex decisions
- **Rule-based reasoning**: Fallback for when LLM is unavailable
- **Model-based reasoning**: Can use a TensorFlow model for predictions

## 🤝 Integration with Existing Systems

The agent is designed to integrate with your existing AI trading components:

- Works with your `YieldScanner` and `TradingBot` classes
- Augments decision-making with cognitive capabilities
- Maintains memory of past interactions and learning

## 📊 Extending the Agent

To extend the agent for different domains:

1. Modify the `Perception` class to handle your specific data types
2. Update the rule-based decision logic in `Reasoning._rule_based_decision()`
3. Add new action types in the `execute_recommendation()` method
4. Customize the goals in `set_goals()` to match your objectives

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔮 Future Enhancements

- Vectorized memory with semantic search capabilities
- Reinforcement learning for adaptive decision-making
- Integration with more data sources
- Multi-agent collaboration capabilities
- Natural language command interface