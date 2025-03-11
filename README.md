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

## Components

The system consists of several key components:

1. **Live Data Fetcher** (`live_data_fetcher.py`): Collects data from blockchain nodes, DeFi protocols, and APIs
2. **Trading Strategy** (`trading_strategy.py`): Implements various investment strategies and portfolio allocation
3. **Protocol Analytics** (`protocol_analytics.py`): Analyzes specific DeFi protocols in depth
4. **Machine Learning** (`machine_learning.py`): Trains and runs predictive models
5. **Persistence Layer** (`defimind_persistence.py`): Stores data, models, and decisions
6. **Dashboard** (`dashboard.py`): Web interface for monitoring and control
7. **Runner** (`defimind_runner.py`): Orchestrates all components

## Installation

### Prerequisites

- Python 3.10+
- Alchemy API key (for blockchain data)
- CoinGecko API key (optional, for market data)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/defimind.git
   cd defimind
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

4. Create a `.env` file with your configuration:
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
   OPENROUTER_API_KEY=your_openrouter_api_key
   LLM_PROVIDER=openrouter
   ```

## Usage

### Running the Full Agent

```
# Run once
python defimind_runner.py run_once

# Run continuously
python defimind_runner.py run

# Run only the dashboard
python defimind_runner.py dashboard
```

### Running Individual Components

```
# Run data collection only
python defimind_runner.py data

# Run protocol analytics only
python defimind_runner.py analytics

# Run trading strategy only
python defimind_runner.py strategy

# Run model training only
python defimind_runner.py models
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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Alchemy API for blockchain data access
- DeFi Llama for yield data
- CoinGecko for market data