#!/usr/bin/env python3
"""
Test script for the DEFIMIND Pyth Searcher component.
This script tests the Pyth SVM searcher for limit order opportunities.
"""

import os
import sys
import time
import importlib
from pathlib import Path

# Add the project root to Python path if not already there
if str(Path('.').absolute()) not in sys.path:
    sys.path.insert(0, str(Path('.').absolute()))

def test_pyth_searcher_imports():
    """Test importing the Pyth searcher module and its classes"""
    print("\n==== Testing Pyth Searcher Imports ====")
    
    try:
        # Import the Pyth searcher module
        from core import pyth_searcher
        print("✅ Successfully imported pyth_searcher module")
        
        # Check for key classes
        classes_to_check = [
            "PythSearcher",
            "Opportunity"
        ]
        
        all_classes_found = True
        for class_name in classes_to_check:
            if hasattr(pyth_searcher, class_name):
                print(f"✅ Found '{class_name}' class")
            else:
                print(f"❌ Missing '{class_name}' class")
                all_classes_found = False
        
        # Check for key functions
        functions_to_check = [
            "run_pyth_searcher_demo"
        ]
        
        all_functions_found = True
        for func_name in functions_to_check:
            if hasattr(pyth_searcher, func_name):
                print(f"✅ Found '{func_name}' function")
            else:
                print(f"❌ Missing '{func_name}' function")
                all_functions_found = False
        
        return all_classes_found and all_functions_found, pyth_searcher
    
    except ImportError as e:
        print(f"❌ Failed to import pyth_searcher: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False, None

def test_opportunity_class(pyth_module):
    """Test the Opportunity class"""
    print("\n==== Testing Opportunity Class ====")
    
    if not hasattr(pyth_module, 'Opportunity'):
        print("❌ Cannot test Opportunity class: class not found")
        return False
    
    try:
        # Create a test opportunity with sample data
        Opportunity = pyth_module.Opportunity
        
        sample_data = {
            "order": "c2FtcGxlX29yZGVyX2RhdGE=",  # base64 encoded "sample_order_data"
            "order_address": "SampleAddress123",
            "program": "limo",
            "chain_id": "solana",
            "version": "v1"
        }
        
        test_opportunity = Opportunity(sample_data)
        
        print(f"✅ Successfully created Opportunity instance")
        
        # Test attributes
        if hasattr(test_opportunity, 'order_address'):
            print(f"✅ Opportunity.order_address = {test_opportunity.order_address}")
        else:
            print(f"❌ Missing 'order_address' attribute")
        
        if hasattr(test_opportunity, 'program'):
            print(f"✅ Opportunity.program = {test_opportunity.program}")
        else:
            print(f"❌ Missing 'program' attribute")
        
        return True
    except Exception as e:
        print(f"❌ Error testing Opportunity class: {e}")
        return False

def test_pyth_searcher_class(pyth_module):
    """Test the PythSearcher class"""
    print("\n==== Testing PythSearcher Class ====")
    
    if not hasattr(pyth_module, 'PythSearcher'):
        print("❌ Cannot test PythSearcher class: class not found")
        return False
    
    try:
        # Create a test searcher in simulation mode
        PythSearcher = pyth_module.PythSearcher
        
        test_searcher = PythSearcher(
            wallet_address="SimulatedWalletAddress",
            chains=["solana"],
            simulation_mode=True
        )
        
        print(f"✅ Successfully created PythSearcher instance")
        
        # Test attributes
        if hasattr(test_searcher, 'simulation_mode'):
            print(f"✅ PythSearcher.simulation_mode = {test_searcher.simulation_mode}")
        else:
            print(f"❌ Missing 'simulation_mode' attribute")
        
        # Test methods
        methods_to_check = [
            "initialize",
            "close",
            "subscribe_to_opportunities",
            "get_statistics"
        ]
        
        all_methods_found = True
        for method_name in methods_to_check:
            if hasattr(test_searcher, method_name):
                print(f"✅ Found '{method_name}' method")
            else:
                print(f"❌ Missing '{method_name}' method")
                all_methods_found = False
        
        return all_methods_found
    except Exception as e:
        print(f"❌ Error testing PythSearcher class: {e}")
        return False

def test_demo_function(pyth_module):
    """Test the run_pyth_searcher_demo function"""
    print("\n==== Testing run_pyth_searcher_demo Function ====")
    
    if not hasattr(pyth_module, 'run_pyth_searcher_demo'):
        print("❌ Cannot test demo function: function not found")
        return False
    
    try:
        # Check if the function exists but don't run it
        # since it might make network calls or run for a long time
        print(f"✅ Found run_pyth_searcher_demo function")
        
        # Check the function signature
        import inspect
        sig = inspect.signature(pyth_module.run_pyth_searcher_demo)
        print(f"✅ Function signature: {sig}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing demo function: {e}")
        return False

def run_tests():
    """Run all tests for the Pyth searcher module"""
    print("\n=========================================")
    print("DEFIMIND Pyth Searcher Component Test")
    print("=========================================")
    
    # Test imports
    import_success, pyth_module = test_pyth_searcher_imports()
    if not import_success:
        print("\n❌ Import test failed. Stopping further tests.")
        return False
    
    # Test Opportunity class
    opportunity_success = test_opportunity_class(pyth_module)
    
    # Test PythSearcher class
    searcher_success = test_pyth_searcher_class(pyth_module)
    
    # Test demo function
    demo_success = test_demo_function(pyth_module)
    
    # Overall test result
    overall_success = import_success and opportunity_success and searcher_success and demo_success
    
    print("\n=========================================")
    print(f"Overall Test Result: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    print("=========================================")
    
    return overall_success

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 