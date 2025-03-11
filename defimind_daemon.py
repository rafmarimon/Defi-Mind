#!/usr/bin/env python3
"""
DEFIMIND Daemon

This script runs DEFIMIND as a daemon process that stays alive continuously,
automatically learning from market data and making investment decisions.
"""

import os
import sys
import time
import signal
import logging
import argparse
import asyncio
import threading
import subprocess
import json
from datetime import datetime, timedelta
import schedule
from dotenv import load_dotenv
import pandas as pd

# Import DEFIMIND components
from defimind_persistence import MemoryDatabase, ModelManager, MarketDataStore
from live_data_fetcher import LiveDataFetcher, run_continuous_data_collection
from defimind_unified import DefiMindUnified

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("defimind_daemon.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("defimind_daemon")

# Process management
DAEMON_PID_FILE = os.getenv("DAEMON_PID_FILE", "defimind_daemon.pid")
DATA_COLLECTOR_PID_FILE = os.getenv("DATA_COLLECTOR_PID_FILE", "data_collector.pid")
AGENT_PID_FILE = os.getenv("AGENT_PID_FILE", "agent.pid")

class DaemonProcess:
    """Base class for daemon processes"""
    
    def __init__(self, pid_file, name="Daemon"):
        self.pid_file = pid_file
        self.name = name
        self.pid = None
        self.running = False
        self.process = None
        
    def is_running(self):
        """Check if the process is running"""
        if not os.path.exists(self.pid_file):
            return False
            
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
                
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            # Process not running or PID file is invalid
            return False
            
    def write_pid(self):
        """Write current PID to file"""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
            
    def remove_pid(self):
        """Remove PID file"""
        if os.path.exists(self.pid_file):
            os.remove(self.pid_file)
            
    def start_subprocess(self, cmd, env=None):
        """Start a subprocess and monitor it"""
        try:
            process = subprocess.Popen(
                cmd,
                env=env if env else os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.process = process
            
            # Write the subprocess PID to file
            with open(self.pid_file, 'w') as f:
                f.write(str(process.pid))
                
            logger.info(f"{self.name} started with PID {process.pid}")
            return process
        except Exception as e:
            logger.error(f"Error starting {self.name}: {e}")
            return None
            
    def stop_subprocess(self):
        """Stop the subprocess"""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.info(f"{self.name} process terminated")
            except subprocess.TimeoutExpired:
                logger.warning(f"{self.name} process did not terminate, forcing...")
                self.process.kill()
            finally:
                self.process = None
                self.remove_pid()

class DefiMindDaemon:
    """Main daemon that keeps DEFIMIND running autonomously"""
    
    def __init__(self):
        self.db = MemoryDatabase()
        self.model_manager = ModelManager()
        self.market_store = MarketDataStore()
        self.fetcher = None
        self.agent = None
        
        # Process controllers
        self.data_collector = DaemonProcess(DATA_COLLECTOR_PID_FILE, "Data Collector")
        self.agent_process = DaemonProcess(AGENT_PID_FILE, "Agent Process")
        
        # Daemon state
        self.running = False
        self.next_training_time = None
        self.next_report_time = None
        
        # Get configuration
        self.data_interval = int(os.getenv("DATA_COLLECTION_INTERVAL_MINUTES", "15"))
        self.agent_cycle_interval = int(os.getenv("AGENT_CYCLE_INTERVAL", "20"))
        self.training_interval_hours = int(os.getenv("MODEL_TRAINING_INTERVAL_HOURS", "24"))
        self.report_interval_hours = int(os.getenv("REPORT_INTERVAL_HOURS", "6"))
        
    async def initialize(self):
        """Initialize daemon"""
        # Create data directories if they don't exist
        os.makedirs("logs", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
        # Initialize fetcher for data validation
        self.fetcher = LiveDataFetcher()
        await self.fetcher.initialize()
        
        # Schedule training and reporting
        self.next_training_time = datetime.now() + timedelta(hours=self.training_interval_hours)
        self.next_report_time = datetime.now() + timedelta(hours=self.report_interval_hours)
        
        logger.info("🚀 DEFIMIND Daemon initialized")
        return True
        
    def start_data_collector(self):
        """Start the data collection process"""
        if self.data_collector.is_running():
            logger.info("Data collector is already running")
            return True
            
        cmd = [sys.executable, "live_data_fetcher.py", "--interval", str(self.data_interval)]
        process = self.data_collector.start_subprocess(cmd)
        
        if process:
            logger.info(f"Data collector started with PID {process.pid}")
            
            # Start a thread to monitor its output
            threading.Thread(target=self._monitor_output, 
                             args=(process, "data_collector")).start()
            return True
        else:
            logger.error("Failed to start data collector")
            return False
            
    def start_agent(self):
        """Start the agent process"""
        if self.agent_process.is_running():
            logger.info("Agent is already running")
            return True
            
        cmd = [sys.executable, "defimind_unified.py", "--mode", "continuous", 
               "--cycles", "0", "--interval", str(self.agent_cycle_interval)]
        process = self.agent_process.start_subprocess(cmd)
        
        if process:
            logger.info(f"Agent started with PID {process.pid}")
            
            # Start a thread to monitor its output
            threading.Thread(target=self._monitor_output, 
                             args=(process, "agent")).start()
            return True
        else:
            logger.error("Failed to start agent")
            return False
    
    def _monitor_output(self, process, name):
        """Monitor the output of a subprocess"""
        log_file = open(f"logs/{name}_output.log", "a")
        
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                break
                
            if output:
                log_file.write(output.decode('utf-8'))
                log_file.flush()
                
        # Process has ended, check return code
        return_code = process.poll()
        log_file.write(f"Process exited with code {return_code}\n")
        log_file.close()
        
        # If not explicitly stopped, restart it
        if self.running:
            logger.warning(f"{name} process ended unexpectedly, restarting...")
            if name == "data_collector":
                self.start_data_collector()
            elif name == "agent":
                self.start_agent()
    
    async def train_models(self):
        """Train and update agent models with collected data"""
        logger.info("Starting model training cycle")
        
        try:
            # Load market data for training
            market_df = self.market_store.create_market_dataframe(days=30)
            
            if market_df.empty:
                logger.warning("No market data available for training")
                return False
                
            # Load performance data
            performance_data = self._get_performance_records()
            
            if not performance_data:
                logger.warning("No performance data available for training")
                
            # Training logic would go here - example only
            # For demonstration, we'll just create a dummy model
            
            # In a real implementation, you would:
            # 1. Prepare features and labels from market_df and performance_data
            # 2. Train a machine learning model (e.g., using scikit-learn)
            # 3. Evaluate the model's performance
            # 4. Save the model if it performs better than previous versions
            
            # For now, we'll just create a dummy model
            import sklearn.ensemble
            model = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
            
            # Dummy training data
            if not market_df.empty:
                # Extract some features from market data
                features = []
                labels = []
                
                # Very simplified example - in reality, you'd have proper feature engineering
                for protocol in market_df['protocol'].unique():
                    protocol_data = market_df[market_df['protocol'] == protocol]
                    if len(protocol_data) > 0:
                        avg_apy = protocol_data['apy'].mean() if 'apy' in protocol_data.columns else 0
                        avg_tvl = protocol_data['tvl'].mean() if 'tvl' in protocol_data.columns else 0
                        features.append([avg_apy, avg_tvl])
                        
                        # Simplified label: 1 if APY > 5%, else 0
                        labels.append(1 if avg_apy > 0.05 else 0)
                
                if features and labels:
                    # Train the model
                    model.fit(features, labels)
                    
                    # Save the model
                    metadata = {
                        "trained_on": datetime.now().isoformat(),
                        "data_points": len(features),
                        "protocols": list(market_df['protocol'].unique()),
                        "feature_columns": ["avg_apy", "avg_tvl"],
                        "training_days": 30
                    }
                    
                    model_path = self.model_manager.save_model(model, "investment_decision_model", metadata)
                    logger.info(f"Model trained and saved to {model_path}")
                    
                    return True
            
            logger.warning("Insufficient data for model training")
            return False
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
            
    def _get_performance_records(self):
        """Get performance records from database"""
        conn = self.db.db_path.replace('memory.db', '')
        try:
            query = "SELECT * FROM performance ORDER BY timestamp DESC LIMIT 1000"
            return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Error fetching performance records: {e}")
            return None
    
    async def generate_system_report(self):
        """Generate a comprehensive system report"""
        logger.info("Generating system report")
        
        try:
            # Get memory statistics
            short_term_count = len(self.db.get_recent_memories(limit=1000))
            long_term_memories = self.db.get_important_long_term_memories(limit=1000)
            
            # Get model statistics
            investment_models = self.model_manager.get_model_history("investment_decision_model")
            
            # Get market data statistics
            market_snapshots = self.market_store.load_recent_snapshots(days=7)
            
            # Create report
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_status": {
                    "data_collector_running": self.data_collector.is_running(),
                    "agent_running": self.agent_process.is_running(),
                    "next_training": self.next_training_time.isoformat() if self.next_training_time else None,
                    "next_report": self.next_report_time.isoformat() if self.next_report_time else None
                },
                "memory_stats": {
                    "short_term_memories": short_term_count,
                    "long_term_memories": len(long_term_memories),
                    "important_memories_sample": [m["content"] for m in long_term_memories[:5]]
                },
                "model_stats": {
                    "total_models": len(investment_models),
                    "latest_model": investment_models[0] if investment_models else None
                },
                "market_stats": {
                    "snapshots_collected": len(market_snapshots),
                    "latest_snapshot": market_snapshots[0]["timestamp"] if market_snapshots else None
                }
            }
            
            # Save report to file
            report_file = f"logs/system_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"System report generated and saved to {report_file}")
            
            # If agent is running, also generate an investment report
            if self.agent_process.is_running():
                # Could use defimind_unified to generate a report
                pass
                
            return report
            
        except Exception as e:
            logger.error(f"Error generating system report: {e}")
            return None
    
    async def consolidate_memory(self):
        """Consolidate short-term to long-term memory"""
        logger.info("Consolidating memory")
        
        try:
            result = self.db.consolidate_memories()
            logger.info(f"Memory consolidation: {result['consolidated_count']} memories consolidated, {result['pruned_count']} pruned")
            return True
        except Exception as e:
            logger.error(f"Error consolidating memory: {e}")
            return False
    
    async def run_maintenance_tasks(self):
        """Run periodic maintenance tasks"""
        # Check if it's time for model training
        if self.next_training_time and datetime.now() >= self.next_training_time:
            await self.train_models()
            self.next_training_time = datetime.now() + timedelta(hours=self.training_interval_hours)
            
        # Check if it's time for system report
        if self.next_report_time and datetime.now() >= self.next_report_time:
            await self.generate_system_report()
            self.next_report_time = datetime.now() + timedelta(hours=self.report_interval_hours)
            
        # Consolidate memory daily
        await self.consolidate_memory()
        
        # Check if processes are running and restart if necessary
        if not self.data_collector.is_running():
            logger.warning("Data collector not running, restarting...")
            self.start_data_collector()
            
        if not self.agent_process.is_running():
            logger.warning("Agent not running, restarting...")
            self.start_agent()
    
    async def run(self):
        """Run the daemon main loop"""
        # Initialize
        await self.initialize()
        self.running = True
        
        # Start components
        self.start_data_collector()
        self.start_agent()
        
        logger.info("🤖 DEFIMIND Daemon is now running")
        
        try:
            # Main loop
            while self.running:
                # Run maintenance tasks
                await self.run_maintenance_tasks()
                
                # Wait a bit
                await asyncio.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("Daemon interrupted by user")
        except Exception as e:
            logger.error(f"Error in daemon main loop: {e}")
        finally:
            # Cleanup
            self.running = False
            self.data_collector.stop_subprocess()
            self.agent_process.stop_subprocess()
            
            if self.fetcher:
                await self.fetcher.close()
                
            logger.info("DEFIMIND Daemon stopped")

# Function to run as a system service
def run_daemon():
    """Run the DEFIMIND daemon"""
    daemon = DefiMindDaemon()
    asyncio.run(daemon.run())

# For direct execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEFIMIND Daemon")
    parser.add_argument("--action", choices=["start", "stop", "restart", "status"], default="start",
                      help="Action to perform")
    
    args = parser.parse_args()
    
    daemon_process = DaemonProcess(DAEMON_PID_FILE, "DEFIMIND Daemon")
    
    if args.action == "start":
        if daemon_process.is_running():
            print("DEFIMIND Daemon is already running.")
            sys.exit(0)
            
        print("Starting DEFIMIND Daemon...")
        run_daemon()
        
    elif args.action == "stop":
        if not daemon_process.is_running():
            print("DEFIMIND Daemon is not running.")
            sys.exit(0)
            
        print("Stopping DEFIMIND Daemon...")
        with open(DAEMON_PID_FILE, 'r') as f:
            pid = int(f.read().strip())
            
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(2)
            print("DEFIMIND Daemon stopped.")
        except OSError as e:
            print(f"Error stopping daemon: {e}")
            
    elif args.action == "restart":
        # Stop if running
        if daemon_process.is_running():
            print("Stopping DEFIMIND Daemon...")
            with open(DAEMON_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
                
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
            except OSError as e:
                print(f"Error stopping daemon: {e}")
                
        # Start
        print("Starting DEFIMIND Daemon...")
        run_daemon()
        
    elif args.action == "status":
        if daemon_process.is_running():
            with open(DAEMON_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
                
            print(f"DEFIMIND Daemon is running with PID {pid}.")
            
            # Check component processes
            data_collector = DaemonProcess(DATA_COLLECTOR_PID_FILE, "Data Collector")
            agent_process = DaemonProcess(AGENT_PID_FILE, "Agent Process")
            
            if data_collector.is_running():
                with open(DATA_COLLECTOR_PID_FILE, 'r') as f:
                    dc_pid = int(f.read().strip())
                print(f"Data Collector is running with PID {dc_pid}.")
            else:
                print("Data Collector is not running.")
                
            if agent_process.is_running():
                with open(AGENT_PID_FILE, 'r') as f:
                    agent_pid = int(f.read().strip())
                print(f"Agent Process is running with PID {agent_pid}.")
            else:
                print("Agent Process is not running.")
        else:
            print("DEFIMIND Daemon is not running.") 