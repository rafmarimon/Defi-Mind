#!/usr/bin/env python3
"""
Test script for demonstrating the autonomous agent's use of browser capabilities

This script shows how the AutonomousAgent integrates with the browser controller
to collect yield data for decision making.
"""

import os
import asyncio
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutonomousBrowserTest")

# Ensure results directory exists
RESULTS_DIR = "autonomous_browser_test"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load environment variables
load_dotenv()

# Check for required modules
try:
    from core.autonomous_agent import AutonomousAgent
    from core.browser_controller import BrowserController
    MODULES_AVAILABLE = True
    logger.info("Required modules are available")
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.error(f"Failed to import required modules: {str(e)}")


async def test_fetch_market_data():
    """Test the autonomous agent's ability to fetch market data using the browser"""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Exiting.")
        return
    
    logger.info("Initializing AutonomousAgent")
    
    # Use custom config to ensure we use the browser
    config = {
        "simulation_mode": True,
        "browser_headless": False,  # Set to False to see the browser in action
        "run_interval_seconds": 3600,
        "communication_interval_hours": 6,
        "gas_threshold_gwei": 50,
        "risk_tolerance": "medium",
        "max_position_size_percent": 25,
        "rebalance_days": 7
    }
    
    agent = AutonomousAgent(config=config)
    
    try:
        logger.info("Fetching market data using browser")
        market_data = await agent._fetch_market_data()
        
        if market_data:
            logger.info(f"Successfully fetched market data with {len(market_data.get('defi_yields', []))} yield opportunities")
            
            # Save the market data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(RESULTS_DIR, f"market_data_{timestamp}.json")
            
            with open(result_path, 'w') as f:
                json.dump(market_data, f, indent=2)
            logger.info(f"Market data saved to {result_path}")
            
            # Display some yield opportunities
            if 'defi_yields' in market_data:
                yields = market_data['defi_yields']
                print("\n=== Top DeFi Yield Opportunities ===")
                for i, opportunity in enumerate(yields[:5], 1):
                    project = opportunity.get('project', 'Unknown')
                    pool = opportunity.get('pool', 'Unknown')
                    apy = opportunity.get('apy', 0)
                    tvl = opportunity.get('tvl', 0)
                    
                    if isinstance(apy, str):
                        apy = apy.replace('%', '')
                        try:
                            apy = float(apy)
                        except ValueError:
                            apy = 0
                    
                    print(f"{i}. {pool} on {project}: {apy:.2f}% APY with ${tvl:,} TVL")
        else:
            logger.warning("No market data returned")
    
    except Exception as e:
        logger.error(f"Error in test_fetch_market_data: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


async def test_analyze_opportunity():
    """Test the autonomous agent's ability to analyze a specific opportunity"""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Exiting.")
        return
    
    logger.info("Initializing AutonomousAgent")
    
    # Use custom config
    config = {
        "simulation_mode": True,
        "browser_headless": False,  # Set to False to see the browser in action
    }
    
    agent = AutonomousAgent(config=config)
    
    try:
        # Pick a protocol and pool to analyze
        protocol = "curve"
        pool = "3pool"
        
        logger.info(f"Analyzing opportunity: {protocol}/{pool}")
        analysis = await agent.analyze_opportunity(protocol, pool)
        
        if analysis and 'error' not in analysis:
            logger.info(f"Successfully analyzed {protocol}/{pool}")
            
            # Save the analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(RESULTS_DIR, f"analysis_{protocol}_{pool}_{timestamp}.json")
            
            with open(result_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Analysis saved to {result_path}")
            
            # Display the analysis
            print(f"\n=== Analysis of {protocol}/{pool} ===")
            print(analysis.get('analysis', 'No analysis provided'))
        else:
            error = analysis.get('error', 'Unknown error') if analysis else 'No analysis returned'
            logger.warning(f"Failed to analyze {protocol}/{pool}: {error}")
    
    except Exception as e:
        logger.error(f"Error in test_analyze_opportunity: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


async def test_agent_cycle():
    """Test a complete autonomous agent cycle, which includes using the browser"""
    if not MODULES_AVAILABLE:
        logger.error("Required modules not available. Exiting.")
        return
    
    logger.info("Initializing AutonomousAgent")
    
    # Use custom config
    config = {
        "simulation_mode": True,
        "browser_headless": False,  # Set to False to see the browser in action
        "run_interval_seconds": 3600,
    }
    
    agent = AutonomousAgent(config=config)
    
    try:
        logger.info("Running agent cycle")
        result = await agent.run_cycle()
        
        if result:
            logger.info("Agent cycle completed successfully")
            
            # Save the cycle results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(RESULTS_DIR, f"cycle_result_{timestamp}.json")
            
            with open(result_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Cycle result saved to {result_path}")
            
            # Display user update if available
            if result.get("user_update"):
                print("\n=== User Update ===")
                print(result["user_update"])
            
            # Display actions taken
            if result.get("actions_taken"):
                print("\n=== Actions Taken ===")
                for action in result["actions_taken"]:
                    print(f"- {action}")
        else:
            logger.warning("No result returned from agent cycle")
    
    except Exception as e:
        logger.error(f"Error in test_agent_cycle: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


async def main():
    """Main function to run the tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Autonomous Browser Test")
    parser.add_argument("--test", choices=["market_data", "analyze", "cycle", "all"], 
                      default="market_data", help="Test to run")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(" AUTONOMOUS BROWSER TEST ")
    print("=" * 80)
    print(f"Test: {args.test}")
    print("Results will be saved to:", RESULTS_DIR)
    print("=" * 80)
    
    if args.test == "market_data" or args.test == "all":
        print("\n--- TESTING MARKET DATA FETCHING ---")
        await test_fetch_market_data()
    
    if args.test == "analyze" or args.test == "all":
        print("\n--- TESTING OPPORTUNITY ANALYSIS ---")
        await test_analyze_opportunity()
    
    if args.test == "cycle" or args.test == "all":
        print("\n--- TESTING COMPLETE AGENT CYCLE ---")
        await test_agent_cycle()
    
    print("\n--- TEST COMPLETE ---")
    print(f"All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    asyncio.run(main()) 