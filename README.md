# DEFIMIND Autonomous Trading Agent

DEFIMIND is an AI-powered autonomous trading agent for DeFi markets, providing data collection, analysis, and investment recommendations based on real-time blockchain data.

## Overview

DEFIMIND combines blockchain data from Alchemy API with protocol-specific analytics, machine learning models, and trading strategies to make informed DeFi investment decisions. The system learns from past performance and market conditions to continuously improve its recommendations.

### Key Features

- **Live Blockchain Data**: Real-time data from Ethereum, Polygon, and other chains
- **Protocol Analytics**: Deep analysis of major DeFi protocols like Aave, Uniswap, and Compound
- **Machine Learning**: Predictive models for yield and risk assessment
- **Multi-Strategy Trading**: Combines different trading approaches for optimal allocation
- **Web Dashboard**: Visual monitoring of data, analysis, and recommendations
- **Continuous Learning**: Models that evolve with market conditions
- **Pyth SVM Searcher**: Integration with Pyth network for limit order opportunities

## Components

The system consists of several key components (now organized in the `core/` directory):

1. **Live Data Fetcher** (`core/live_data_fetcher.py`): Collects data from blockchain nodes, DeFi protocols, and APIs
2. **Trading Strategy** (`core/trading_strategy.py`): Implements various investment strategies and portfolio allocation
3. **Protocol Analytics** (`core/protocol_analytics.py`): Analyzes specific DeFi protocols in depth
4. **Machine Learning** (`core/machine_learning.py`): Trains and runs predictive models
5. **Persistence Layer** (`core/defimind_persistence.py`): Stores data, models, and decisions
6. **Dashboard** (`core/dashboard.py`): Web interface for monitoring and control
7. **Runner** (`core/defimind_runner.py`): Orchestrates all components
8. **Pyth Searcher** (`core/pyth_searcher.py`): Finds and executes Pyth limit order opportunities
9. **LangChain Agent** (`core/langchain_agent.py`): AI agent integration for natural language interactions

## Project Structure

```
defimind/
├── core/                    # Core functionality
│   ├── dashboard.py         # Web interface
│   ├── pyth_searcher.py     # Pyth SVM searcher
│   ├── langchain_agent.py   # LangChain integration
│   └── ...                  # Other core modules
│
├── config/                  # Configuration
│   ├── .env                 # Environment variables (not in Git)
│   └── .env.*               # Environment file variants
│
├── data/                    # Data storage
│   ├── models/              # Trained ML models
│   ├── market/              # Market data
│   └── memory/              # Agent memory
│
├── logs/                    # Log files directory
│
├── .env.template            # Template for environment variables
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

For a more detailed breakdown of the project structure, see `PROJECT_STRUCTURE.md`.

## Installation

### Prerequisites

- Python 3.10+
- Alchemy API key (for blockchain data)
- CoinGecko API key (optional, for market data)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/rafmarimon/DeFiMind.git
   cd DeFiMind
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Copy the environment template and configure it:
   ```
   cp .env.template config/.env
   ```
   
   Then edit `config/.env` with your configuration:
   ```
   # API Keys
   ALCHEMY_API_KEY=your_alchemy_api_key
   
   # Configuration
   SIMULATION_MODE=false
   AGENT_CYCLE_INTERVAL=20
   TOTAL_INVESTMENT=100
   MAX_SINGLE_PROTOCOL_ALLOCATION=0.5
   RISK_TOLERANCE=0.6
   
   # LLM Configuration (optional)
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

### Running the Dashboard

```bash
# Run the dashboard
streamlit run core/dashboard.py
```

### Running the Full Agent

```bash
# Run once
python -m core.defimind_runner run_once

# Run continuously
python -m core.defimind_runner run

# Run only the dashboard
python -m core.defimind_runner dashboard
```

### Running Individual Components

```bash
# Run data collection only
python -m core.defimind_runner data

# Run protocol analytics only
python -m core.defimind_runner analytics

# Run trading strategy only
python -m core.defimind_runner strategy

# Run model training only
python -m core.defimind_runner models
```

### Accessing the Dashboard

Once started, the dashboard is available at:
```
http://localhost:8501
```

## System Architecture

```
┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │
│  Blockchain     │         │  Market Data    │
│  (Alchemy API)  │◄────────┤  APIs           │
│                 │         │                 │
└────────┬────────┘         └────────┬────────┘
         │                           │
         ▼                           ▼
┌─────────────────────────────────────────────┐
│                                             │
│            Live Data Fetcher                │
│                                             │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│                                             │
│               Persistence Layer             │
│                                             │
└───┬─────────────────┬──────────────────┬────┘
    │                 │                  │
    ▼                 ▼                  ▼
┌────────────┐   ┌────────────┐    ┌────────────┐
│            │   │            │    │            │
│ Protocol   │   │ Machine    │    │ Trading    │
│ Analytics  │   │ Learning   │    │ Strategy   │
│            │   │            │    │            │
└────┬───────┘   └────┬───────┘    └────┬───────┘
     │                │                 │
     └────────────────┼─────────────────┘
                      │
                      ▼
             ┌────────────────┐
             │                │
             │   Dashboard    │
             │                │
             └────────────────┘
```

## Configuration

### Environment Variables

| Variable                        | Description                                  | Default |
|---------------------------------|----------------------------------------------|---------|
| `ALCHEMY_API_KEY`               | Alchemy API key                              | -       |
| `SIMULATION_MODE`               | Run in simulation mode without transactions  | true    |
| `AGENT_CYCLE_INTERVAL`          | Time between agent cycles (minutes)          | 20      |
| `TOTAL_INVESTMENT`              | Total investment amount                      | 100     |
| `MAX_SINGLE_PROTOCOL_ALLOCATION`| Maximum allocation to a single protocol      | 0.5     |
| `RISK_TOLERANCE`                | Risk tolerance (0-1)                         | 0.6     |
| `DATA_COLLECTION_INTERVAL_MINUTES` | Data collection frequency                | 15      |
| `MODEL_TRAINING_INTERVAL_HOURS` | Model retraining frequency                   | 24      |

For a complete list of configuration options, see `.env.template`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alchemy API for blockchain data access
- DeFi Llama for yield data
- CoinGecko for market data
- Pyth Network for price feed data