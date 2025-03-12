#!/usr/bin/env python3
"""
DEFIMIND Component Tester

This script tests the functionality of individual DEFIMIND components
to verify they work correctly with the new project structure.
"""

import os
import sys
import time
import json
import importlib
from datetime import datetime
from pathlib import Path

# Add the project root to Python path if not already there
if str(Path('.').absolute()) not in sys.path:
    sys.path.insert(0, str(Path('.').absolute()))

# Create a directory for test outputs
os.makedirs('test_outputs', exist_ok=True)

class ComponentTester:
    """Test runner for DEFIMIND components"""
    
    def __init__(self):
        self.results = {}
        self.start_time = datetime.now()
    
    def log(self, component, message, success=True):
        """Log a test message"""
        status = "✅" if success else "❌"
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {status} {component}: {message}")
        
        if component not in self.results:
            self.results[component] = {"success": True, "messages": []}
        
        if not success:
            self.results[component]["success"] = False
        
        self.results[component]["messages"].append({
            "time": timestamp,
            "message": message,
            "success": success
        })
    
    def import_component(self, component_path):
        """Safely import a component and report status"""
        component_name = component_path.split('.')[-1]
        
        try:
            component = importlib.import_module(component_path)
            self.log(component_name, f"Successfully imported module")
            return component
        except ImportError as e:
            self.log(component_name, f"Failed to import: {str(e)}", success=False)
            return None
    
    def test_runner(self):
        """Test the defimind runner module"""
        component_name = "defimind_runner"
        runner = self.import_component("core.defimind_runner")
        if not runner:
            return
        
        # Check if the module has the expected functions
        if hasattr(runner, 'main'):
            self.log(component_name, "Found main function")
        else:
            self.log(component_name, "Missing main function", success=False)
        
        # Check if parse_args function exists
        if hasattr(runner, 'parse_args'):
            self.log(component_name, "Found parse_args function")
        else:
            self.log(component_name, "Missing parse_args function", success=False)
    
    def test_data_fetcher(self):
        """Test the live data fetcher module"""
        component_name = "live_data_fetcher"
        fetcher = self.import_component("core.live_data_fetcher")
        if not fetcher:
            return
        
        # Check if the module has the expected functions
        for func_name in ['fetch_price', 'fetch_protocol_data']:
            if hasattr(fetcher, func_name):
                self.log(component_name, f"Found {func_name} function")
            else:
                self.log(component_name, f"Missing {func_name} function", success=False)
    
    def test_persistence(self):
        """Test the persistence layer module"""
        component_name = "defimind_persistence"
        persistence = self.import_component("core.defimind_persistence")
        if not persistence:
            return
        
        # Check if the main class exists
        if hasattr(persistence, 'DefimindPersistence'):
            self.log(component_name, "Found DefimindPersistence class")
            
            # Try to initialize the persistence layer
            try:
                # Initialize with test mode to avoid creating real files
                persist = persistence.DefimindPersistence(test_mode=True)
                self.log(component_name, "Successfully initialized persistence layer")
            except Exception as e:
                self.log(component_name, f"Failed to initialize: {str(e)}", success=False)
        else:
            self.log(component_name, "Missing DefimindPersistence class", success=False)
    
    def test_machine_learning(self):
        """Test the machine learning module"""
        component_name = "machine_learning"
        ml = self.import_component("core.machine_learning")
        if not ml:
            return
        
        # Check if key classes/functions exist
        if hasattr(ml, 'train_model'):
            self.log(component_name, "Found train_model function")
        else:
            self.log(component_name, "Missing train_model function", success=False)
    
    def test_trading_strategy(self):
        """Test the trading strategy module"""
        component_name = "trading_strategy"
        strategy = self.import_component("core.trading_strategy")
        if not strategy:
            return
        
        # Check if key classes exist
        for class_name in ['Strategy', 'TradingStrategy']:
            if hasattr(strategy, class_name):
                self.log(component_name, f"Found {class_name} class")
            else:
                self.log(component_name, f"Missing {class_name} class", success=False)
    
    def test_protocol_analytics(self):
        """Test the protocol analytics module"""
        component_name = "protocol_analytics"
        analytics = self.import_component("core.protocol_analytics")
        if not analytics:
            return
        
        # Check if key functions exist
        for func_name in ['analyze_protocol', 'get_protocol_data']:
            if hasattr(analytics, func_name):
                self.log(component_name, f"Found {func_name} function")
            else:
                self.log(component_name, f"Missing {func_name} function", success=False)
    
    def test_pyth_searcher(self):
        """Test the Pyth SVM searcher module"""
        component_name = "pyth_searcher"
        searcher = self.import_component("core.pyth_searcher")
        if not searcher:
            return
        
        # Check if key classes/functions exist
        if hasattr(searcher, 'PythSearcher'):
            self.log(component_name, "Found PythSearcher class")
        else:
            self.log(component_name, "Missing PythSearcher class", success=False)
        
        if hasattr(searcher, 'Opportunity'):
            self.log(component_name, "Found Opportunity class")
        else:
            self.log(component_name, "Missing Opportunity class", success=False)
    
    def test_langchain_agent(self):
        """Test the LangChain agent module"""
        component_name = "langchain_agent"
        agent = self.import_component("core.langchain_agent")
        if not agent:
            return
        
        # Check if key classes exist
        if hasattr(agent, 'LangChainAgent'):
            self.log(component_name, "Found LangChainAgent class")
        else:
            self.log(component_name, "Missing LangChainAgent class", success=False)
    
    def run_all_tests(self):
        """Run all component tests"""
        print(f"\n{'='*50}")
        print(f"DEFIMIND Component Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")
        
        # List of test methods to run
        tests = [
            self.test_runner,
            self.test_data_fetcher,
            self.test_persistence,
            self.test_machine_learning,
            self.test_trading_strategy,
            self.test_protocol_analytics,
            self.test_pyth_searcher,
            self.test_langchain_agent
        ]
        
        # Run each test
        for test in tests:
            test()
            print("")  # Add a blank line between test components
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of all test results"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        success_count = sum(1 for result in self.results.values() if result["success"])
        total_count = len(self.results)
        
        print(f"\n{'='*50}")
        print(f"Test Summary")
        print(f"{'='*50}")
        print(f"Total components tested: {total_count}")
        print(f"Successful components: {success_count}")
        print(f"Failed components: {total_count - success_count}")
        print(f"Total duration: {duration:.2f} seconds")
        print(f"{'='*50}\n")
        
        # Save results to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"test_outputs/component_test_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration": f"{duration:.2f} seconds",
                "total_components": total_count,
                "successful_components": success_count,
                "failed_components": total_count - success_count,
                "results": self.results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {result_file}")


if __name__ == "__main__":
    tester = ComponentTester()
    tester.run_all_tests() 