#!/usr/bin/env python3
"""
Test script for the DEFIMIND Runner component.
This script tests various functionalities of the defimind_runner.py module.
"""

import os
import sys
import importlib
from pathlib import Path

# Add the project root to Python path if not already there
if str(Path('.').absolute()) not in sys.path:
    sys.path.insert(0, str(Path('.').absolute()))

def test_runner_imports():
    """Test importing the runner module and its dependencies"""
    print("\n==== Testing Runner Imports ====")
    
    try:
        # Import the runner module
        from core import defimind_runner
        print("✅ Successfully imported defimind_runner")
        
        # Check for critical attributes
        if hasattr(defimind_runner, 'main'):
            print("✅ Found 'main' function")
        else:
            print("❌ Missing 'main' function")
        
        if hasattr(defimind_runner, 'parse_arguments'):
            print("✅ Found 'parse_arguments' function")
        else:
            print("❌ Missing 'parse_arguments' function")
        
        # Check version if available
        if hasattr(defimind_runner, '__version__'):
            print(f"✅ Runner version: {defimind_runner.__version__}")
        else:
            print("ℹ️ No version information available")
        
        return True, defimind_runner
    
    except ImportError as e:
        print(f"❌ Failed to import defimind_runner: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None

def test_argument_parser(runner_module):
    """Test the argument parser functionality"""
    print("\n==== Testing Argument Parser ====")
    
    if not hasattr(runner_module, 'parse_arguments'):
        print("❌ Cannot test argument parser: function not found")
        return False
    
    try:
        # Test the argument parser function exists
        print("✅ Found parse_arguments function")
        
        # We can't easily test the actual parsing without modifying sys.argv
        # So we just check it exists for now
        return True
    except Exception as e:
        print(f"❌ Unexpected error in argument parser: {e}")
        return False

def test_component_loading(runner_module):
    """Test the component loading functionality"""
    print("\n==== Testing Component Loading ====")
    
    # Check for DefiMindRunner class
    if not hasattr(runner_module, 'DefiMindRunner'):
        print("❌ Missing DefiMindRunner class")
        return False
    
    try:
        # Get the class
        DefiMindRunner = runner_module.DefiMindRunner
        print("✅ Found DefiMindRunner class")
        
        # Check for component initialization methods
        component_methods = [
            ("initialize", "Initialization"),
            ("run_data_collection", "Live Data Fetcher"),
            ("run_protocol_analytics", "Protocol Analytics"),
            ("run_trading_strategy", "Trading Strategy"),
            ("run_model_training", "Machine Learning"),
        ]
        
        success = True
        for method_name, component_name in component_methods:
            if hasattr(DefiMindRunner, method_name):
                print(f"✅ Found method for {component_name}")
            else:
                print(f"❌ Missing method for {component_name}")
                success = False
        
        return success
    except Exception as e:
        print(f"❌ Error checking DefiMindRunner class: {e}")
        return False

def run_tests():
    """Run all tests for the runner module"""
    print("\n=========================================")
    print("DEFIMIND Runner Component Test")
    print("=========================================")
    
    # Test imports
    import_success, runner_module = test_runner_imports()
    if not import_success:
        print("\n❌ Import test failed. Stopping further tests.")
        return False
    
    # Test argument parser
    parser_success = test_argument_parser(runner_module)
    
    # Test component loading
    loading_success = test_component_loading(runner_module)
    
    # Overall test result
    overall_success = import_success and parser_success and loading_success
    
    print("\n=========================================")
    print(f"Overall Test Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    print("=========================================")
    
    return overall_success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 