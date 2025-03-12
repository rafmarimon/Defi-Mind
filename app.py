#!/usr/bin/env python3
"""
DEFIMIND Application Entry Point

This file serves as the main entry point for Digital Ocean App Platform.
It launches the DEFIMIND dashboard by default.
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure we can import from core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Main entry point
async def main():
    """
    Main entry point for the DEFIMIND application
    Starts the dashboard by default
    """
    from core.defimind_runner import DefiMindRunner
    
    # Load environment variables
    load_dotenv()
    
    # Get port from environment (Digital Ocean expects 8080)
    port = int(os.environ.get("PORT", 8080))
    os.environ["DASHBOARD_PORT"] = str(port)
    
    print(f"Starting DEFIMIND on port {port}...")
    
    # Initialize the runner
    runner = DefiMindRunner()
    
    try:
        await runner.initialize()
        runner.start_dashboard()
        print(f"DEFIMIND Dashboard started on port {port}. Press Ctrl+C to stop.")
        
        # Keep the application running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down DEFIMIND...")
    finally:
        await runner.close()

# Run the application
if __name__ == "__main__":
    asyncio.run(main()) 