#!/usr/bin/env python3
"""
DEFIMIND Runner Script

Orchestrates the various components of the DEFIMIND autonomous trading agent,
running them in the correct sequence and managing their interactions.
"""

import os
import sys
import json
import time
import logging
import asyncio
import argparse
import subprocess
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import DEFIMIND components
from live_data_fetcher import LiveDataFetcher, run_continuous_data_collection
from trading_strategy import YieldOptimizer, MultiStrategyAllocator
from protocol_analytics import ProtocolAnalyzer, run_protocol_analysis
from machine_learning import ModelTrainer, PortfolioOptimizer, run_model_training
from defimind_persistence import MemoryDatabase, MarketDataStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("defimind.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("defimind_runner")

# Load environment variables
load_dotenv()

# Configuration
DATA_COLLECTION_INTERVAL = int(os.getenv("DATA_COLLECTION_INTERVAL_MINUTES", "15")) * 60  # Convert to seconds
ANALYTICS_INTERVAL = int(os.getenv("ANALYTICS_INTERVAL_MINUTES", "60")) * 60
STRATEGY_INTERVAL = int(os.getenv("STRATEGY_INTERVAL_MINUTES", "120")) * 60
MODEL_TRAINING_INTERVAL = int(os.getenv("MODEL_TRAINING_INTERVAL_HOURS", "24")) * 3600
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))


class DefiMindRunner:
    """
    Main runner class for the DEFIMIND autonomous trading agent.
    Orchestrates all components and manages their execution.
    """
    
    def __init__(self):
        """Initialize the runner"""
        logger.info("Initializing DEFIMIND Runner")
        self.market_store = MarketDataStore()
        self.memory_db = MemoryDatabase()
        self.data_fetcher = None
        self.dashboard_process = None
        
        # Initialize common lock for ensuring only one process runs at a time
        self.lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.defimind_lock')
        
        # Last run timestamps
        self.last_data_collection = 0
        self.last_analytics_run = 0
        self.last_strategy_run = 0
        self.last_model_training = 0
        
        # Status tracking
        self.status = {
            "data_collection": {"status": "idle", "last_run": None, "last_success": None},
            "protocol_analytics": {"status": "idle", "last_run": None, "last_success": None},
            "trading_strategy": {"status": "idle", "last_run": None, "last_success": None},
            "model_training": {"status": "idle", "last_run": None, "last_success": None},
            "dashboard": {"status": "stopped", "pid": None}
        }
    
    async def initialize(self):
        """Initialize the runner and its components"""
        logger.info("Starting DEFIMIND initialization")
        
        # Initialize data fetcher if needed
        if not self.data_fetcher:
            self.data_fetcher = LiveDataFetcher()
            await self.data_fetcher.initialize()
        
        # Create necessary directories
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/memory", exist_ok=True)
        os.makedirs("data/market", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info("DEFIMIND initialization complete")
        return True
    
    async def run_data_collection(self):
        """Run data collection cycle"""
        logger.info("Running data collection cycle")
        self.status["data_collection"]["status"] = "running"
        self.status["data_collection"]["last_run"] = datetime.now().isoformat()
        
        try:
            if not self.data_fetcher:
                await self.initialize()
                
            # Run data collection cycle
            results = await self.data_fetcher.run_data_collection_cycle()
            
            # Update status
            self.status["data_collection"]["status"] = "idle"
            self.status["data_collection"]["last_success"] = datetime.now().isoformat()
            self.last_data_collection = time.time()
            
            logger.info("Data collection cycle completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in data collection cycle: {e}")
            self.status["data_collection"]["status"] = "error"
            return {"error": str(e)}
    
    async def run_protocol_analytics(self):
        """Run protocol analytics"""
        logger.info("Running protocol analytics")
        self.status["protocol_analytics"]["status"] = "running"
        self.status["protocol_analytics"]["last_run"] = datetime.now().isoformat()
        
        try:
            # Create protocol analyzer
            analyzer = ProtocolAnalyzer()
            
            # Run analysis
            results = await analyzer.analyze_all()
            
            # Update status
            self.status["protocol_analytics"]["status"] = "idle"
            self.status["protocol_analytics"]["last_success"] = datetime.now().isoformat()
            self.last_analytics_run = time.time()
            
            logger.info("Protocol analytics completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in protocol analytics: {e}")
            self.status["protocol_analytics"]["status"] = "error"
            return {"error": str(e)}
    
    async def run_trading_strategy(self):
        """Run trading strategy"""
        logger.info("Running trading strategy")
        self.status["trading_strategy"]["status"] = "running"
        self.status["trading_strategy"]["last_run"] = datetime.now().isoformat()
        
        try:
            # Create multi-strategy allocator
            allocator = MultiStrategyAllocator()
            
            # Run allocator
            results = await allocator.run_allocator()
            
            # Update status
            self.status["trading_strategy"]["status"] = "idle"
            self.status["trading_strategy"]["last_success"] = datetime.now().isoformat()
            self.last_strategy_run = time.time()
            
            logger.info("Trading strategy completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in trading strategy: {e}")
            self.status["trading_strategy"]["status"] = "error"
            return {"error": str(e)}
    
    def run_model_training(self):
        """Run model training"""
        logger.info("Running model training")
        self.status["model_training"]["status"] = "running"
        self.status["model_training"]["last_run"] = datetime.now().isoformat()
        
        try:
            # Create model trainer
            trainer = ModelTrainer()
            
            # Train models
            results = trainer.train_all_models()
            
            # Update status
            self.status["model_training"]["status"] = "idle"
            self.status["model_training"]["last_success"] = datetime.now().isoformat()
            self.last_model_training = time.time()
            
            logger.info("Model training completed successfully")
            return results
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            self.status["model_training"]["status"] = "error"
            return {"error": str(e)}
    
    def start_dashboard(self):
        """Start the dashboard if it's not running"""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            logger.info("Dashboard is already running")
            return
            
        try:
            # Start dashboard as a separate process
            cmd = [sys.executable, "dashboard.py"]
            self.dashboard_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Update status
            self.status["dashboard"]["status"] = "running"
            self.status["dashboard"]["pid"] = self.dashboard_process.pid
            
            logger.info(f"Dashboard started on port {DASHBOARD_PORT}")
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            self.status["dashboard"]["status"] = "error"
    
    def stop_dashboard(self):
        """Stop the dashboard if it's running"""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                logger.info("Dashboard stopped")
            except subprocess.TimeoutExpired:
                self.dashboard_process.kill()
                logger.warning("Dashboard killed after timeout")
            
            self.status["dashboard"]["status"] = "stopped"
            self.status["dashboard"]["pid"] = None
    
    def acquire_lock(self):
        """Acquire a lock to ensure only one instance runs at a time"""
        if os.path.exists(self.lock_file):
            # Check if process is still running
            try:
                with open(self.lock_file, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process is running
                try:
                    os.kill(pid, 0)  # This will raise an error if process is not running
                    logger.error(f"Another instance is already running (PID {pid})")
                    return False
                except OSError:
                    # Process not running, remove stale lock file
                    logger.warning("Removing stale lock file")
                    os.remove(self.lock_file)
            except (ValueError, IOError):
                # Invalid lock file, remove it
                os.remove(self.lock_file)
        
        # Create lock file with current PID
        with open(self.lock_file, 'w') as f:
            f.write(str(os.getpid()))
            
        return True
    
    def release_lock(self):
        """Release the lock"""
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)
    
    async def run_sequence(self):
        """Run a single sequence of operations"""
        current_time = time.time()
        
        # Check if enough time has passed since last run
        if (current_time - self.last_data_collection) >= DATA_COLLECTION_INTERVAL:
            await self.run_data_collection()
        
        if (current_time - self.last_analytics_run) >= ANALYTICS_INTERVAL:
            await self.run_protocol_analytics()
        
        if (current_time - self.last_strategy_run) >= STRATEGY_INTERVAL:
            await self.run_trading_strategy()
        
        if (current_time - self.last_model_training) >= MODEL_TRAINING_INTERVAL:
            self.run_model_training()
    
    def save_status(self):
        """Save status to file for external monitoring"""
        status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'status.json')
        
        # Add timestamp
        status_copy = self.status.copy()
        status_copy["last_updated"] = datetime.now().isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status_copy, f, indent=2)
    
    async def run_continuous(self):
        """Run the agent continuously"""
        if not self.acquire_lock():
            logger.error("Failed to acquire lock. Another instance may be running.")
            return
        
        try:
            # Initialize
            await self.initialize()
            
            # Start dashboard
            self.start_dashboard()
            
            # Run initial sequence
            await self.run_sequence()
            
            # Run continuous loop
            while True:
                # Save status
                self.save_status()
                
                # Run sequence
                await self.run_sequence()
                
                # Sleep to avoid high CPU usage
                await asyncio.sleep(60)  # Check every minute
        finally:
            # Stop dashboard
            self.stop_dashboard()
            
            # Release lock
            self.release_lock()
    
    async def close(self):
        """Clean up resources"""
        # Stop dashboard
        self.stop_dashboard()
        
        # Close data fetcher
        if self.data_fetcher:
            await self.data_fetcher.close()
        
        # Release lock
        self.release_lock()
        
        logger.info("DEFIMIND runner stopped")


async def run_once():
    """Run a single cycle of the DEFIMIND agent"""
    runner = DefiMindRunner()
    try:
        await runner.initialize()
        await runner.run_data_collection()
        await runner.run_protocol_analytics()
        await runner.run_trading_strategy()
        runner.run_model_training()
    finally:
        await runner.close()


async def run_continuous():
    """Run the DEFIMIND agent continuously"""
    runner = DefiMindRunner()
    try:
        await runner.run_continuous()
    except KeyboardInterrupt:
        logger.info("DEFIMIND runner interrupted")
    finally:
        await runner.close()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DEFIMIND Autonomous Trading Agent")
    
    # Add commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run once
    run_once_parser = subparsers.add_parser("run_once", help="Run a single cycle")
    
    # Run continuous
    run_continuous_parser = subparsers.add_parser("run", help="Run continuously")
    
    # Dashboard only
    dashboard_parser = subparsers.add_parser("dashboard", help="Start only the dashboard")
    
    # Data collection only
    data_parser = subparsers.add_parser("data", help="Run data collection only")
    
    # Analytics only
    analytics_parser = subparsers.add_parser("analytics", help="Run protocol analytics only")
    
    # Strategy only
    strategy_parser = subparsers.add_parser("strategy", help="Run trading strategy only")
    
    # Models only
    models_parser = subparsers.add_parser("models", help="Run model training only")
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    runner = DefiMindRunner()
    
    try:
        await runner.initialize()
        
        if args.command == "run_once":
            await run_once()
        elif args.command == "run":
            await run_continuous()
        elif args.command == "dashboard":
            runner.start_dashboard()
            print(f"Dashboard started on port {DASHBOARD_PORT}. Press Ctrl+C to stop.")
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("Stopping dashboard...")
        elif args.command == "data":
            await runner.run_data_collection()
        elif args.command == "analytics":
            await runner.run_protocol_analytics()
        elif args.command == "strategy":
            await runner.run_trading_strategy()
        elif args.command == "models":
            runner.run_model_training()
        else:
            # Default to run once
            await run_once()
    finally:
        await runner.close()


if __name__ == "__main__":
    asyncio.run(main()) 