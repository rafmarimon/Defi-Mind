# DEFIMIND Project Structure

This document outlines the organization of the DEFIMIND project files after cleanup.

## Core Components

The core components are now organized in the `core/` directory:

| File | Description |
|------|-------------|
| `dashboard.py` | Web interface for monitoring and control |
| `pyth_searcher.py` | Pyth SVM searcher for limit order opportunities |
| `langchain_agent.py` | LangChain integration for AI agent capabilities |
| `defimind_persistence.py` | Data persistence and storage layer |
| `defimind_runner.py` | Main orchestration module for running the system |
| `machine_learning.py` | Machine learning models and analysis |
| `trading_strategy.py` | Trading strategy implementation |
| `protocol_analytics.py` | Protocol analysis and data processing |
| `live_data_fetcher.py` | Fetches live data from blockchain APIs |
| `yield_scanner.py` | Scans for yield opportunities |

## Configuration

Environment and configuration files are stored in the `config/` directory:

| File | Description |
|------|-------------|
| `.env.template` | (Root dir) Template for environment variables |
| `.env` | (Config dir) Actual environment variables (not committed to Git) |
| `.env.backup` | Backup of environment settings |
| `.env.testnet` | Testnet-specific environment settings |
| `.env.example` | Example environment file |

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
├── README.md                # Project documentation
├── PROJECT_STRUCTURE.md     # This file
├── backup_project.py        # Backup utility
└── cleanup_project.py       # Cleanup utility
```

## Archived Components

Less frequently used or deprecated components are moved to the `archive/` directory:

- `archive/logs/` - Log files
- `archive/tests/` - Test scripts
- `archive/deployment/` - Deployment-related files
- `archive/ethereum/` - Ethereum/Hardhat related files
- `archive/scripts/` - Utility scripts
- `archive/agent_variants/` - Alternative agent implementations

## Running the Project

After cleanup, you should run the project from the `core` directory:

```bash
# Run the dashboard
python -m core.dashboard

# Run the full agent
python -m core.defimind_runner run

# Run in dashboard-only mode
python -m core.defimind_runner dashboard
```

## Import Updates

If you had import statements referencing the original file structure, you'll need to update them to reflect the new structure.

Example:
```python
# Old import
from live_data_fetcher import fetch_price

# New import
from core.live_data_fetcher import fetch_price
```

## Git Setup

The `.gitignore` file has been updated to exclude the `archive/` directory from version control, keeping your repository clean. 