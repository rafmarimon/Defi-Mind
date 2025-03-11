#!/usr/bin/env python3
"""
DEFIMIND - Alchemy API Showcase
Demonstrates the key API features available in the Alchemy Growth plan
"""

import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# VITALIK's ADDRESS - used for examples
TEST_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
# UNISWAP V3 - Major protocol example
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"

# Alchemy API configuration
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
ALCHEMY_ETH_API = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

# Divider function for prettier output
def print_section(title):
    print("\n" + "=" * 50)
    print(f" {title} ".center(50, "="))
    print("=" * 50)

async def get_token_balances(address):
    """Get all ERC20 token balances for an address using Alchemy API"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address, "erc20"]
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", {})
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def get_token_metadata(contract_address):
    """Get metadata for a token contract"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenMetadata",
            "params": [contract_address]
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", {})
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def get_asset_transfers(address, category=["external", "internal", "erc20"], 
                             max_count=5, from_block="0x0", to_block="latest"):
    """Get historical transfers for an address"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "fromAddress": address,
                    "maxCount": max_count,
                    "category": category
                }
            ]
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", {})
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def get_eth_gas_price():
    """Get current gas price using Eth API"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_gasPrice",
            "params": [],
            "id": 1
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                gas_price_hex = result.get("result", "0x0")
                gas_price_wei = int(gas_price_hex, 16)
                gas_price_gwei = gas_price_wei / 1e9
                return gas_price_gwei
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def get_latest_block():
    """Get information about the latest block"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": ["latest", False],
            "id": 1
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", {})
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def get_nft_balance(address):
    """Get NFT balance for an address"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address, "erc721"]
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", {})
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def get_contract_logs(contract_address, event_signature, max_results=5):
    """Get logs for a specific contract and event"""
    async with aiohttp.ClientSession() as session:
        # Example: Transfer event signature
        if not event_signature:
            event_signature = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
            
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getLogs",
            "params": [
                {
                    "address": contract_address,
                    "topics": [event_signature],
                    "fromBlock": "0x" + hex(int(await get_latest_block_number(), 16) - 10000)[2:],
                    "toBlock": "latest"
                }
            ]
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                logs = result.get("result", [])
                return logs[:max_results]  # Limit the number of results
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def get_latest_block_number():
    """Get the latest block number"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", "0x0")
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return "0x0"

async def get_transaction_receipts_by_block(block_number):
    """Get all transaction receipts for a block"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTransactionReceipts",
            "params": [{"blockNumber": block_number}]
        }
        
        async with session.post(ALCHEMY_ETH_API, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("result", {})
            else:
                error_text = await response.text()
                print(f"❌ Error: {response.status} - {error_text}")
                return None

async def main():
    print_section("DEFIMIND - ALCHEMY API SHOWCASE")
    print(f"API Key: {ALCHEMY_API_KEY[:4]}...{ALCHEMY_API_KEY[-4:]}")
    print(f"Test Address: {TEST_ADDRESS}")
    
    # 1. Get Token Balances - Core API for any DeFi application
    print_section("1. TOKEN BALANCES (ERC-20)")
    token_balances = await get_token_balances(TEST_ADDRESS)
    if token_balances and "tokenBalances" in token_balances:
        print(f"Found {len(token_balances['tokenBalances'])} ERC-20 tokens")
        print("\nTop 3 tokens:")
        for i, token in enumerate(token_balances["tokenBalances"][:3]):
            metadata = await get_token_metadata(token["contractAddress"])
            name = metadata.get("name", "Unknown")
            symbol = metadata.get("symbol", "???")
            decimals = int(metadata.get("decimals", "0"))
            balance_raw = int(token["tokenBalance"], 16)
            balance = balance_raw / (10 ** decimals) if decimals > 0 else balance_raw
            
            print(f"{i+1}. {name} ({symbol})")
            print(f"   Address: {token['contractAddress']}")
            print(f"   Balance: {balance:.6f}")
            print()
    
    # 2. Gas Price Monitoring - Essential for transaction optimization
    print_section("2. GAS PRICE MONITORING")
    gas_price = await get_eth_gas_price()
    print(f"Current Ethereum Gas Price: {gas_price:.2f} Gwei")
    
    # 3. NFT Holdings - Growing area of interest for financial apps
    print_section("3. NFT HOLDINGS (ERC-721)")
    nft_balances = await get_nft_balance(TEST_ADDRESS)
    if nft_balances and "tokenBalances" in nft_balances:
        print(f"Found {len(nft_balances['tokenBalances'])} NFT collections")
        if len(nft_balances["tokenBalances"]) > 0:
            print("\nSample NFT collections:")
            for i, nft in enumerate(nft_balances["tokenBalances"][:3]):
                metadata = await get_token_metadata(nft["contractAddress"])
                name = metadata.get("name", "Unknown NFT Collection")
                print(f"{i+1}. {name}")
                print(f"   Contract: {nft['contractAddress']}")
                print()
    
    # 4. Transaction History - Critical for analytics
    print_section("4. TRANSACTION HISTORY")
    transfers = await get_asset_transfers(TEST_ADDRESS, max_count=3)
    if transfers and "transfers" in transfers:
        print(f"Recent transfers (showing {len(transfers['transfers'])} of {transfers.get('pageKey', 'many')}):")
        for i, tx in enumerate(transfers["transfers"]):
            print(f"{i+1}. Block {int(tx['blockNum'], 16)}")
            print(f"   From: {tx.get('from')}")
            print(f"   To: {tx.get('to')}")
            print(f"   Asset: {tx.get('asset', 'ETH')}")
            print(f"   Value: {tx.get('value')}")
            print()
    
    # 5. Blockchain Data - For protocol analysis
    print_section("5. BLOCKCHAIN DATA")
    latest_block = await get_latest_block()
    if latest_block:
        print(f"Latest Block: {int(latest_block['number'], 16)}")
        print(f"Timestamp: {int(latest_block['timestamp'], 16)}")
        print(f"Gas Used: {int(latest_block['gasUsed'], 16)}")
        print(f"Transactions: {len(latest_block['transactions'])}")
    
    # 6. Contract Logs - Important for tracking protocol activity
    print_section("6. CONTRACT EVENT LOGS")
    # Using Transfer event for a token contract from the balances
    if token_balances and "tokenBalances" in token_balances and len(token_balances["tokenBalances"]) > 0:
        sample_token = token_balances["tokenBalances"][0]["contractAddress"]
        print(f"Checking Transfer events for token: {sample_token}")
        
        # Transfer event signature
        transfer_sig = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
        logs = await get_contract_logs(sample_token, transfer_sig, max_results=3)
        
        if logs:
            print(f"Found {len(logs)} recent Transfer events")
            for i, log in enumerate(logs):
                print(f"{i+1}. Transaction: {log['transactionHash']}")
                print(f"   Block: {int(log['blockNumber'], 16)}")
                print(f"   Log Index: {int(log['logIndex'], 16)}")
                print()
    
    # 7. Transaction Receipts - For detailed transaction analysis
    print_section("7. TRANSACTION RECEIPTS")
    latest_block_num = await get_latest_block_number()
    receipts = await get_transaction_receipts_by_block(latest_block_num)
    
    if receipts and "receipts" in receipts:
        print(f"Transactions in block {int(latest_block_num, 16)}: {len(receipts['receipts'])}")
        if len(receipts['receipts']) > 0:
            sample_tx = receipts['receipts'][0]
            print("\nSample transaction receipt:")
            print(f"Transaction Hash: {sample_tx.get('transactionHash')}")
            print(f"From: {sample_tx.get('from')}")
            print(f"To: {sample_tx.get('to')}")
            print(f"Status: {'Success' if sample_tx.get('status') == '0x1' else 'Failed'}")
            print(f"Gas Used: {int(sample_tx.get('gasUsed', '0x0'), 16)}")
    
    print_section("SHOWCASE COMPLETE")
    print("These APIs form the foundation of your DEFIMIND trading agent")
    print("With your $200/month budget, you can perform extensive blockchain analysis")
    print("and make informed trading decisions based on real-time data.")

if __name__ == "__main__":
    asyncio.run(main()) 