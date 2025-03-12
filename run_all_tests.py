#!/usr/bin/env python3
"""
Master test script for DEFIMIND
Runs all component tests and generates a summary report
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

# List of test scripts to run
TEST_SCRIPTS = [
    "test_components.py",
    "test_runner.py",
    "test_pyth_searcher.py",
    "test_langchain_agent.py"
]

class TestRunner:
    """Manages the execution of all DEFIMIND component tests"""
    
    def __init__(self):
        """Initialize the test runner"""
        self.results = {}
        self.start_time = datetime.now()
        
        # Create output directory
        os.makedirs('test_outputs', exist_ok=True)
    
    def run_test(self, test_script):
        """Run a single test script and capture its output"""
        print(f"\n\n{'='*70}")
        print(f"Running Test: {test_script}")
        print(f"{'='*70}\n")
        
        component_name = test_script.replace('.py', '').replace('test_', '')
        
        try:
            start_time = time.time()
            
            # Run the test script as a subprocess
            result = subprocess.run(
                [sys.executable, test_script],
                capture_output=True,
                text=True,
                check=False
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Print the output from the test
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            
            # Determine if the test passed
            success = result.returncode == 0
            
            # Store the result
            self.results[component_name] = {
                "success": success,
                "duration": duration,
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr if result.stderr else None
            }
            
            return success
            
        except Exception as e:
            print(f"Error running test {test_script}: {e}")
            self.results[component_name] = {
                "success": False,
                "duration": 0,
                "return_code": -1,
                "output": "",
                "error": str(e)
            }
            return False
    
    def run_all_tests(self):
        """Run all test scripts"""
        print(f"\n{'='*70}")
        print(f"DEFIMIND Master Test Suite - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        all_success = True
        test_summary = []
        
        for test_script in TEST_SCRIPTS:
            if os.path.exists(test_script):
                success = self.run_test(test_script)
                component_name = test_script.replace('.py', '').replace('test_', '')
                status = "✅ PASSED" if success else "❌ FAILED"
                test_summary.append((component_name, status))
                
                if not success:
                    all_success = False
            else:
                print(f"Test script not found: {test_script}")
                test_summary.append((test_script, "⚠️ NOT FOUND"))
                all_success = False
        
        # Generate summary report
        self.print_summary(test_summary)
        
        return all_success
    
    def print_summary(self, test_summary):
        """Print a summary of all test results"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Count successes
        success_count = sum(1 for _, status in test_summary if status == "✅ PASSED")
        total_count = len(test_summary)
        
        print(f"\n\n{'='*70}")
        print(f"Test Summary")
        print(f"{'='*70}")
        print(f"Total tests: {total_count}")
        print(f"Successful tests: {success_count}")
        print(f"Failed tests: {total_count - success_count}")
        print(f"Total duration: {duration:.2f} seconds")
        print(f"{'='*70}\n")
        
        # Print individual test results
        print("Test Results:")
        for component, status in test_summary:
            print(f"  {status} - {component}")
        
        # Save results to a JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"test_outputs/master_test_report_{timestamp}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration": f"{duration:.2f} seconds",
                "total_tests": total_count,
                "successful_tests": success_count,
                "failed_tests": total_count - success_count,
                "test_summary": {component: status for component, status in test_summary},
                "detailed_results": self.results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to: {result_file}")

if __name__ == "__main__":
    # Create and run the test runner
    runner = TestRunner()
    success = runner.run_all_tests()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1) 