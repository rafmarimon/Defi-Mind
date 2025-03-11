#!/usr/bin/env python3
"""
Test script for Alchemy API integration in DEFIMIND
"""

import asyncio
import os
import json
from dotenv import load_dotenv
from live_data_fetcher import LiveDataFetcher

# Load environment variables
load_dotenv()

# VITALIK's ADDRESS - used in the example
TEST_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"

async def main():
    print("DEFIMIND - Alchemy API Test")
    print("-" * 50)
    
    # Initialize the data fetcher
    fetcher = LiveDataFetcher()
    await fetcher.initialize()
    
    try:
        print(f"Testing token balances for address: {TEST_ADDRESS}")
        token_balances = await fetcher.get_token_balances(TEST_ADDRESS)
        
        if token_balances and "tokenBalances" in token_balances:
            print(f"✅ Successfully retrieved {len(token_balances['tokenBalances'])} token balances")
            
            # Print the first 3 tokens
            print("\nSample Token Balances:")
            for i, token in enumerate(token_balances["tokenBalances"][:3]):
                print(f"{i+1}. Contract: {token.get('contractAddress')}")
                print(f"   Balance: {token.get('tokenBalance')}")
                print()
                
            # Now test the full wallet analysis
            print("\nRunning full wallet analysis...")
            wallet_data = await fetcher.analyze_wallet(TEST_ADDRESS)
            
            if wallet_data and "error" not in wallet_data:
                print(f"✅ Successfully analyzed wallet with {len(wallet_data['tokens'])} tokens")
                
                # Print a few tokens with proper balances
                print("\nTop tokens by balance:")
                for i, token in enumerate(sorted(wallet_data["tokens"], 
                                               key=lambda x: x["balance"], 
                                               reverse=True)[:5]):
                    print(f"{i+1}. {token['name']} ({token['symbol']})")
                    print(f"   Balance: {token['balance']}")
                    print()
                    
                # Print recent activity summary
                if "recent_activity" in wallet_data and wallet_data["recent_activity"]:
                    print(f"\nRecent activity: {len(wallet_data['recent_activity'])} transfers")
            else:
                print(f"❌ Failed to analyze wallet: {wallet_data.get('error', 'Unknown error')}")
        else:
            print("❌ Failed to retrieve token balances")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        # Clean up
        await fetcher.close()
        
if __name__ == "__main__":
    asyncio.run(main()) 