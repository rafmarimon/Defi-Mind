# DEFIMIND

DEFIMIND is an autonomous AI agent for DeFi yield optimization. It leverages machine learning, browser automation, and self-improvement capabilities to find, analyze, and allocate funds to the best DeFi yield opportunities.

## 🌟 Features

- **Yield Optimization**: Automatically scans and analyzes DeFi platforms for the best yield opportunities
- **Autonomous Operation**: Self-running agent that makes decisions based on market conditions
- **Web Browser Automation**: Utilizes browser-use to interact with DeFi platforms in real-time
- **Self-Improvement**: Monitors its own performance and implements improvements automatically
- **Risk Management**: Configurable risk tolerance and position sizing
- **Communication**: Natural language updates and responses to user questions
- **Visualization**: Interactive dashboard for monitoring opportunities and agent status

## 📊 Components

DEFIMIND consists of several key components:

- **AutonomousAgent**: The main controller that orchestrates all components
- **YieldTracker**: Tracks and compares yields across DeFi platforms
- **YieldOptimizer**: Generates allocation and rebalancing plans based on yield data
- **DefiBrowserAgent**: Automates browser interactions with DeFi platforms
- **AgentCommunicator**: Generates natural language updates and answers user questions
- **SelfImprovement**: Monitors system performance and implements improvements

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/defimind.git
cd defimind
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install browser automation dependencies:
```bash
playwright install
```

5. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ETH_RPC_URL=your_ethereum_rpc_url
```

### Running DEFIMIND

#### Running the Dashboard

```bash
streamlit run core/dashboard.py
```

#### Running the Autonomous Agent

```bash
python -m core.autonomous_agent
```

## 🛠️ Configuration

DEFIMIND is highly configurable. Main settings:

```python
config = {
    "simulation_mode": True,  # Set to False for real transactions
    "risk_tolerance": "medium",  # Options: low, medium, high
    "max_position_size_percent": 25,  # Maximum % in a single position
    "browser_headless": True,  # Run browser in background
    "rebalance_days": 7,  # Check for rebalance weekly
}
```

## 🔒 Security

By default, DEFIMIND runs in simulation mode with no real transactions. To run with real funds:

1. Set `simulation_mode` to `False` in the configuration
2. Provide wallet connection details (requires additional setup)
3. Start with small amounts until you're comfortable with the system

## 📝 Testing

Run the test suite:

```bash
python run_all_tests.py
```

Individual component tests:

```bash
python test_yield_tracker.py
python test_agent_communication.py
python test_autonomous_agent.py
python test_yield_optimizer.py
```

## 📖 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

DEFIMIND is an experimental project. Use at your own risk. The developers are not responsible for any losses incurred through its use. Always start with small amounts and test thoroughly.