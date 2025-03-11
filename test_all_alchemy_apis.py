#!/usr/bin/env python3
"""
DEFIMIND - Comprehensive Alchemy API Test Suite
Tests all implemented Alchemy API features in live_data_fetcher.py
"""

import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from live_data_fetcher import LiveDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("alchemy_test")

# Test addresses
VITALIK_ADDRESS = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
UNISWAP_V3_FACTORY = "0x1F98431c8aD98523631AE4a59f267346ea31F984"
AAVE_LENDING_POOL = "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9"

# Load environment variables
load_dotenv()

async def test_token_balances(fetcher):
    print("\n=== Testing Token Balances API ===")
    token_balances = await fetcher.get_token_balances(VITALIK_ADDRESS)
    
    if token_balances and "tokenBalances" in token_balances:
        print(f"✅ Successfully retrieved {len(token_balances['tokenBalances'])} token balances")
        # Show first few tokens
        for i, token in enumerate(token_balances["tokenBalances"][:3]):
            print(f"{i+1}. Contract: {token.get('contractAddress')}")
            print(f"   Balance: {token.get('tokenBalance')}")
    else:
        print("❌ Failed to retrieve token balances")
    
    return token_balances

async def test_token_metadata(fetcher, token_balances):
    print("\n=== Testing Token Metadata API ===")
    if not token_balances or "tokenBalances" not in token_balances:
        print("❌ Cannot test token metadata - no token balances available")
        return None
        
    # Get the first 3 token addresses
    token_addresses = [tb.get("contractAddress") for tb in token_balances.get("tokenBalances", [])[:3]]
    
    if not token_addresses:
        print("❌ No token addresses found")
        return None
        
    print(f"Testing metadata for {len(token_addresses)} tokens...")
    token_metadata = await fetcher.get_token_metadata(token_addresses)
    
    if token_metadata:
        print(f"✅ Successfully retrieved metadata for {len(token_metadata)} tokens")
        # Show metadata
        for address, metadata in token_metadata.items():
            print(f"Token: {address}")
            print(f"  Name: {metadata.get('name', 'Unknown')}")
            print(f"  Symbol: {metadata.get('symbol', 'Unknown')}")
            print(f"  Decimals: {metadata.get('decimals', 'Unknown')}")
    else:
        print("❌ Failed to retrieve token metadata")
    
    return token_metadata

async def test_asset_transfers(fetcher):
    print("\n=== Testing Asset Transfers API ===")
    # Use the correct parameter names
    from_block = "0x0"
    to_block = "latest"
    transfers = await fetcher.get_asset_transfers(VITALIK_ADDRESS, from_block=from_block, to_block=to_block)
    
    if transfers and "transfers" in transfers:
        print(f"✅ Successfully retrieved {len(transfers['transfers'])} transfers")
        # Show transfers
        for i, tx in enumerate(transfers["transfers"][:3]):
            print(f"{i+1}. Transfer at block {int(tx['blockNum'], 16)}")
            print(f"   From: {tx.get('from')}")
            print(f"   To: {tx.get('to')}")
            print(f"   Asset: {tx.get('asset', 'ETH')}")
            print(f"   Value: {tx.get('value')}")
    else:
        print("❌ Failed to retrieve asset transfers")
    
    return transfers

async def test_gas_price(fetcher):
    print("\n=== Testing Gas Price API ===")
    gas_price = await fetcher.get_eth_gas_price()
    
    if gas_price is not None:
        print(f"✅ Current Ethereum gas price: {gas_price:.2f} Gwei")
    else:
        print("❌ Failed to retrieve gas price")
    
    return gas_price

async def test_latest_block(fetcher):
    print("\n=== Testing Latest Block API ===")
    latest_block = await fetcher.get_latest_block()
    
    if latest_block:
        block_number = int(latest_block.get("number", "0x0"), 16)
        timestamp = int(latest_block.get("timestamp", "0x0"), 16)
        gas_used = int(latest_block.get("gasUsed", "0x0"), 16)
        tx_count = len(latest_block.get("transactions", []))
        
        print(f"✅ Latest block: {block_number}")
        print(f"   Timestamp: {timestamp}")
        print(f"   Gas used: {gas_used}")
        print(f"   Transaction count: {tx_count}")
    else:
        print("❌ Failed to retrieve latest block")
    
    return latest_block

async def test_nft_balance(fetcher):
    print("\n=== Testing NFT Balance API ===")
    nft_balances = await fetcher.get_nft_balance(VITALIK_ADDRESS)
    
    if nft_balances and "tokenBalances" in nft_balances:
        print(f"✅ Successfully retrieved {len(nft_balances['tokenBalances'])} NFT balances")
        # Show first few NFTs
        for i, nft in enumerate(nft_balances["tokenBalances"][:3]):
            print(f"{i+1}. Contract: {nft.get('contractAddress')}")
            print(f"   Balance: {nft.get('tokenBalance')}")
    else:
        print("❌ Failed to retrieve NFT balances")
    
    return nft_balances

async def test_contract_logs(fetcher):
    print("\n=== Testing Contract Logs API ===")
    # Transfer event signature (common in ERC-20 tokens)
    transfer_event_sig = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    logs = await fetcher.get_contract_logs(UNISWAP_V3_FACTORY, transfer_event_sig, max_results=3)
    
    if logs:
        print(f"✅ Successfully retrieved {len(logs)} logs")
        # Show logs
        for i, log in enumerate(logs):
            print(f"{i+1}. Transaction: {log.get('transactionHash')}")
            print(f"   Block: {int(log.get('blockNumber', '0x0'), 16)}")
            print(f"   Log Index: {int(log.get('logIndex', '0x0'), 16)}")
    else:
        print("❌ Failed to retrieve contract logs or no logs found")
    
    return logs

async def test_transaction_receipts(fetcher):
    print("\n=== Testing Transaction Receipts API ===")
    latest_block_num = await fetcher.get_latest_block_number()
    receipts = await fetcher.get_transaction_receipts_by_block(latest_block_num)
    
    if receipts and "receipts" in receipts:
        print(f"✅ Successfully retrieved {len(receipts['receipts'])} receipts for block {int(latest_block_num, 16)}")
        # Show first receipt
        if receipts["receipts"]:
            receipt = receipts["receipts"][0]
            print(f"Sample receipt:")
            print(f"   Transaction: {receipt.get('transactionHash')}")
            print(f"   From: {receipt.get('from')}")
            print(f"   To: {receipt.get('to')}")
            print(f"   Status: {'Success' if receipt.get('status') == '0x1' else 'Failed'}")
    else:
        print("❌ Failed to retrieve transaction receipts")
    
    return receipts

async def test_analyze_wallet(fetcher):
    print("\n=== Testing Wallet Analysis API ===")
    wallet_data = await fetcher.analyze_wallet(VITALIK_ADDRESS)
    
    if wallet_data and "error" not in wallet_data:
        print(f"✅ Successfully analyzed wallet with {len(wallet_data['tokens'])} tokens")
        # Show top tokens
        sorted_tokens = sorted(wallet_data["tokens"], key=lambda x: x["balance"], reverse=True)
        for i, token in enumerate(sorted_tokens[:3]):
            print(f"{i+1}. {token.get('name', 'Unknown')} ({token.get('symbol', '???')})")
            print(f"   Balance: {token.get('balance')}")
    else:
        error = wallet_data.get("error", "Unknown error") if wallet_data else "No data returned"
        print(f"❌ Failed to analyze wallet: {error}")
    
    return wallet_data

async def test_data_collection_cycle(fetcher):
    print("\n=== Testing Full Data Collection Cycle ===")
    print("Running data collection cycle (this may take a minute)...")
    await fetcher.run_data_collection_cycle()
    print("✅ Data collection cycle completed")

async def main():
    print("DEFIMIND - Comprehensive Alchemy API Test Suite")
    print("=" * 50)
    
    # Initialize the data fetcher
    fetcher = LiveDataFetcher()
    await fetcher.initialize()
    
    try:
        # Test token balances API
        token_balances = await test_token_balances(fetcher)
        
        # Test token metadata API
        await test_token_metadata(fetcher, token_balances)
        
        # Test asset transfers API
        await test_asset_transfers(fetcher)
        
        # Test gas price API
        await test_gas_price(fetcher)
        
        # Test latest block API
        await test_latest_block(fetcher)
        
        # Test NFT balance API
        await test_nft_balance(fetcher)
        
        # Test contract logs API
        await test_contract_logs(fetcher)
        
        # Test transaction receipts API
        await test_transaction_receipts(fetcher)
        
        # Test wallet analysis API
        await test_analyze_wallet(fetcher)
        
        # Test full data collection cycle
        await test_data_collection_cycle(fetcher)
        
        print("\n" + "=" * 50)
        print("✅ All Alchemy API tests completed")
        print("=" * 50)
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    finally:
        # Clean up
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(main()) 