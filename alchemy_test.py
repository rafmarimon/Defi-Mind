#!/usr/bin/env python3
"""
Simplified test for Alchemy API integration in DEFIMIND
This standalone script tests the Alchemy API directly without dependencies on other parts of the codebase.
"""

import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# VITALIK's ADDRESS - used in the example
TEST_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
ALCHEMY_ETH_API = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

async def get_token_balances(address):
    """Get all ERC20 token balances for an address using Alchemy API"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address, "erc20"]
        }
        
        try:
            async with session.post(ALCHEMY_ETH_API, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Successfully fetched token balances")
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    print(f"❌ Failed to get token balances: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"❌ Error fetching token balances: {e}")
            return None

async def get_token_metadata(token_addresses):
    """Get metadata for tokens using Alchemy API"""
    async with aiohttp.ClientSession() as session:
        results = {}
        
        for address in token_addresses:
            payload = {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "alchemy_getTokenMetadata",
                "params": [address]
            }
            
            try:
                async with session.post(ALCHEMY_ETH_API, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        results[address] = result.get("result", {})
                    else:
                        error_text = await response.text()
                        print(f"❌ Failed to get token metadata: {response.status} - {error_text}")
            except Exception as e:
                print(f"❌ Error fetching token metadata: {e}")
                
        return results

async def main():
    print("DEFIMIND - Alchemy API Direct Test")
    print("-" * 50)
    print(f"API Key (first 4 chars): {ALCHEMY_API_KEY[:4]}...")
    print(f"Testing address: {TEST_ADDRESS}")
    
    # Step 1: Get token balances
    token_balances = await get_token_balances(TEST_ADDRESS)
    
    if token_balances and "tokenBalances" in token_balances:
        print(f"✅ Retrieved {len(token_balances['tokenBalances'])} token balances")
        
        # Get the first few token balances
        print("\nSample Token Balances:")
        for i, token in enumerate(token_balances["tokenBalances"][:3]):
            print(f"{i+1}. Contract: {token.get('contractAddress')}")
            print(f"   Balance: {token.get('tokenBalance')}")
            
        # Step 2: Get token metadata for the first few tokens
        token_addresses = [tb.get("contractAddress") for tb in token_balances.get("tokenBalances", [])[:5]]
        print(f"\nFetching metadata for {len(token_addresses)} tokens...")
        
        token_metadata = await get_token_metadata(token_addresses)
        
        print("\nToken Metadata Results:")
        for address, metadata in token_metadata.items():
            print(f"Token: {address}")
            print(f"  Name: {metadata.get('name', 'Unknown')}")
            print(f"  Symbol: {metadata.get('symbol', 'Unknown')}")
            print(f"  Decimals: {metadata.get('decimals', 'Unknown')}")
            print()
    else:
        print("❌ Failed to retrieve token balances")

if __name__ == "__main__":
    asyncio.run(main()) 