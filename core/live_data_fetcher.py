#!/usr/bin/env python3
"""
DEFIMIND Live Data Fetcher

Fetches real-time market data from various DeFi protocols and APIs.
Connects to real blockchain data instead of using simulated data.
"""

import os
import json
import asyncio
import logging
import aiohttp
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from web3 import Web3
import ccxt
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Import persistence layer for storing the data
from core.defimind_persistence import MemoryDatabase, MarketDataStore

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("live_data_fetcher.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_data_fetcher")

# API configurations
DEFI_LLAMA_API = "https://yields.llama.fi/pools"
COINGECKO_API = "https://api.coingecko.com/api/v3"
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
POLYGONSCAN_API_KEY = os.getenv("POLYGONSCAN_API_KEY")
BSCSCAN_API_KEY = os.getenv("BSCSCAN_API_KEY")

# Alchemy API configurations
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")
ALCHEMY_ETH_API = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
ALCHEMY_POLYGON_API = f"https://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
ALCHEMY_ARBITRUM_API = f"https://arb-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
ALCHEMY_OPTIMISM_API = f"https://opt-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
ALCHEMY_ETH_WS = f"wss://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
ALCHEMY_POLYGON_WS = f"wss://polygon-mainnet.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

# RPC URLs
ETHEREUM_RPC_URL = os.getenv("ETHEREUM_RPC_URL", ALCHEMY_ETH_API)
POLYGON_RPC_URL = os.getenv("POLYGON_RPC_URL", ALCHEMY_POLYGON_API)
BSC_RPC_URL = os.getenv("BSC_RPC_URL")
AVALANCHE_RPC_URL = os.getenv("AVALANCHE_RPC_URL")

# Protocol addresses and ABIs
PROTOCOL_CONFIG_PATH = os.getenv("PROTOCOL_CONFIG_PATH", "protocol_configs.json")

# Use PostgreSQL for better scalability
DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/memory.db")
engine = create_engine(DB_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Simulation mode (for testing without real blockchain interactions)
SIMULATION_MODE = os.getenv("SIMULATION_MODE", "false").lower() == "true"

class LiveDataFetcher:
    """Fetches live data from DeFi platforms and blockchain networks"""
    
    def __init__(self):
        """Initialize the data fetcher with required connections and configurations."""
        self.session = None
        self.web3_connections = {}
        self.web3_ws_connections = {}
        self.protocol_configs = {}
        self.memory_db = MemoryDatabase()
        self.market_store = MarketDataStore()
        self.alchemy_endpoints = {
            "ethereum": ALCHEMY_ETH_API,
            "polygon": ALCHEMY_POLYGON_API,
            "arbitrum": ALCHEMY_ARBITRUM_API,
            "optimism": ALCHEMY_OPTIMISM_API
        }
        self.websocket_connections = {}
        
        # Load protocol configurations if available
        self._load_protocol_configs()
        
        # Initialize crypto exchange API (for reference prices)
        self.exchange = ccxt.binance({
            'rateLimit': 1200,  # Adjust based on API limits
            'enableRateLimit': True
        })
        
    def _load_protocol_configs(self):
        """Load protocol configuration data"""
        try:
            if os.path.exists(PROTOCOL_CONFIG_PATH):
                with open(PROTOCOL_CONFIG_PATH, 'r') as f:
                    self.protocol_configs = json.load(f)
                logger.info(f"Loaded protocol configurations from {PROTOCOL_CONFIG_PATH}")
            else:
                logger.warning(f"Protocol config file not found: {PROTOCOL_CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error loading protocol configurations: {e}")
            
    async def initialize(self):
        """Initialize connections and sessions"""
        # Create HTTP session
        self.session = aiohttp.ClientSession()
        
        # Initialize Web3 connections if RPC URLs are available
        if ETHEREUM_RPC_URL:
            try:
                self.web3_connections["ethereum"] = Web3(Web3.HTTPProvider(ETHEREUM_RPC_URL))
                if self.web3_connections["ethereum"].is_connected():
                    logger.info("✅ Connected to Ethereum network")
            except Exception as e:
                logger.error(f"Failed to connect to Ethereum network: {e}")
                
        if POLYGON_RPC_URL:
            try:
                self.web3_connections["polygon"] = Web3(Web3.HTTPProvider(POLYGON_RPC_URL))
                if self.web3_connections["polygon"].is_connected():
                    logger.info("✅ Connected to Polygon network")
            except Exception as e:
                logger.error(f"Failed to connect to Polygon network: {e}")
                
        if BSC_RPC_URL:
            try:
                self.web3_connections["bsc"] = Web3(Web3.HTTPProvider(BSC_RPC_URL))
                if self.web3_connections["bsc"].is_connected():
                    logger.info("✅ Connected to BSC network")
            except Exception as e:
                logger.error(f"Failed to connect to BSC network: {e}")
                
        if AVALANCHE_RPC_URL:
            try:
                self.web3_connections["avalanche"] = Web3(Web3.HTTPProvider(AVALANCHE_RPC_URL))
                if self.web3_connections["avalanche"].is_connected():
                    logger.info("✅ Connected to Avalanche network")
            except Exception as e:
                logger.error(f"Failed to connect to Avalanche network: {e}")
                
        # Initialize WebSocket connections if needed
        if ALCHEMY_API_KEY:
            await self._init_websocket_connections()
        
        return True
        
    async def _init_websocket_connections(self):
        """Initialize WebSocket connections to Alchemy endpoints for real-time data"""
        try:
            # Check if websocket provider is available
            try:
                from websockets.legacy.client import connect as ws_connect
                from web3.providers import WebsocketProvider
                websocket_available = True
            except ImportError:
                logger.warning("WebSocket provider not available. Install with: pip install websockets")
                websocket_available = False

            if not websocket_available:
                return
                
            # Initialize for Ethereum
            if ALCHEMY_ETH_WS:
                web3_ws = Web3(WebsocketProvider(ALCHEMY_ETH_WS))
                self.websocket_connections["ethereum"] = web3_ws
                logger.info("✅ Connected to Ethereum WebSocket via Alchemy")
            
            # Initialize for Polygon
            if ALCHEMY_POLYGON_WS:
                web3_ws_polygon = Web3(WebsocketProvider(ALCHEMY_POLYGON_WS))
                self.websocket_connections["polygon"] = web3_ws_polygon
                logger.info("✅ Connected to Polygon WebSocket via Alchemy")
                
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket connections: {e}")

    async def close(self):
        """Close all connections and resources."""
        # Close HTTP session
        if self.session:
            await self.session.close()
            
        # Close WebSocket connections
        for network, ws_connection in self.websocket_connections.items():
            try:
                # Close WebSocket provider
                provider = ws_connection.provider
                if hasattr(provider, "disconnect"):
                    await provider.disconnect()
                logger.info(f"Closed {network} WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing {network} WebSocket: {e}")
            
    async def fetch_defi_llama_data(self):
        """Fetch yield data from DeFi Llama API"""
        if not self.session:
            await self.initialize()
            
        try:
            async with self.session.get(DEFI_LLAMA_API) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Process and organize the data
                    processed_data = {
                        "timestamp": datetime.now().isoformat(),
                        "source": "defillama",
                        "protocols": {}
                    }
                    
                    # Map project names to our protocol identifiers
                    project_mapping = {
                        "Pancakeswap": "pancakeswap",
                        "Trader Joe": "traderjoe",
                        "QuickSwap": "quickswap",
                        "Uniswap": "uniswap",
                        "Curve": "curve",
                        "Aave": "aave",
                        "Compound": "compound"
                    }
                    
                    # Process pools data
                    if "data" in data and isinstance(data["data"], list):
                        for pool in data["data"]:
                            project = pool.get("project")
                            
                            # Map to our protocol name if it exists
                            protocol = project_mapping.get(project, project.lower() if project else "unknown")
                            
                            # Extract pool data
                            pool_data = {
                                "name": pool.get("pool", ""),
                                "symbol": pool.get("symbol", ""),
                                "apy": pool.get("apy") / 100 if pool.get("apy") else 0,  # Convert from percentage
                                "tvl": pool.get("tvlUsd", 0),
                                "chain": pool.get("chain", ""),
                                "risk_level": self._calculate_risk_level(pool)
                            }
                            
                            # Add to the protocol data, keeping highest APY pools
                            if protocol not in processed_data["protocols"]:
                                processed_data["protocols"][protocol] = []
                                
                            processed_data["protocols"][protocol].append(pool_data)
                    
                    # Sort pools by APY and keep only top N per protocol
                    for protocol in processed_data["protocols"]:
                        processed_data["protocols"][protocol].sort(key=lambda x: x["apy"], reverse=True)
                        processed_data["protocols"][protocol] = processed_data["protocols"][protocol][:5]  # Keep top 5
                        
                    # Save to database and file
                    self._save_protocol_data(processed_data)
                    
                    return processed_data
                else:
                    logger.error(f"DeFi Llama API error: {response.status} - {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching DeFi Llama data: {e}")
            return None
            
    def _calculate_risk_level(self, pool_data):
        """Calculate risk level for a pool based on various factors"""
        risk_score = 0.5  # Default mid-risk
        
        # Factor 1: TVL - higher TVL = lower risk
        tvl = pool_data.get("tvlUsd", 0)
        if tvl > 100000000:  # $100M+
            risk_score -= 0.2
        elif tvl > 10000000:  # $10M+
            risk_score -= 0.1
        elif tvl < 1000000:  # <$1M
            risk_score += 0.2
            
        # Factor 2: APY - higher APY = higher risk
        apy = pool_data.get("apy", 0)
        if apy > 100:  # 100%+
            risk_score += 0.3
        elif apy > 50:  # 50%+
            risk_score += 0.2
        elif apy > 20:  # 20%+
            risk_score += 0.1
            
        # Factor 3: Chain - some chains might be deemed higher risk
        chain = pool_data.get("chain", "").lower()
        if chain in ["ethereum", "avalanche"]:
            risk_score -= 0.05
        elif chain in ["optimism", "arbitrum"]:
            risk_score += 0.05
        elif chain not in ["ethereum", "bsc", "polygon", "avalanche", "arbitrum", "optimism"]:
            risk_score += 0.1  # Less established chains
            
        # Factor 4: Protocol age and audit status (would need additional data)
        
        # Ensure score is between 0 and 1
        return max(0.1, min(0.9, risk_score))
    
    def _save_protocol_data(self, processed_data):
        """Save protocol data to database and file storage"""
        try:
            # Save to market data store (file-based)
            self.market_store.save_market_snapshot(processed_data, source="defillama")
            
            # Save to database for each protocol
            for protocol_name, pools in processed_data["protocols"].items():
                for pool in pools:
                    self.memory_db.add_market_data(
                        protocol=protocol_name,
                        pool=pool["name"],
                        apy=pool["apy"],
                        tvl=pool["tvl"],
                        raw_data=json.dumps(pool)
                    )
                    
            logger.info(f"Saved data for {len(processed_data['protocols'])} protocols")
            
        except Exception as e:
            logger.error(f"Error saving protocol data: {e}")
    
    async def fetch_token_prices(self, tokens=None):
        """Fetch current token prices from CoinGecko"""
        if not tokens:
            # Default tokens to track
            tokens = ["bitcoin", "ethereum", "matic-network", "binancecoin", "avalanche-2"]
            
        if not self.session:
            await self.initialize()
            
        try:
            url = f"{COINGECKO_API}/simple/price"
            params = {
                "ids": ",".join(tokens),
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Format the data
                    price_data = {
                        "timestamp": datetime.now().isoformat(),
                        "source": "coingecko",
                        "prices": data
                    }
                    
                    # Save to file
                    self.market_store.save_market_snapshot(price_data, source="coingecko")
                    
                    return price_data
                else:
                    logger.error(f"CoinGecko API error: {response.status} - {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching token prices: {e}")
            return None
    
    async def fetch_gas_prices(self):
        """Fetch current gas prices from multiple chains"""
        if not self.session:
            await self.initialize()
            
        gas_data = {
            "timestamp": datetime.now().isoformat(),
            "gas_prices": {}
        }
        
        # Fetch from Ethereum
        try:
            if "ethereum" in self.web3_connections:
                eth_gas_price = self.web3_connections["ethereum"].eth.gas_price
                gas_data["gas_prices"]["ethereum"] = eth_gas_price / 1e9  # Convert to Gwei
        except Exception as e:
            logger.error(f"Error fetching Ethereum gas price: {e}")
            
        # Fetch from Polygon
        try:
            if "polygon" in self.web3_connections:
                polygon_gas_price = self.web3_connections["polygon"].eth.gas_price
                gas_data["gas_prices"]["polygon"] = polygon_gas_price / 1e9  # Convert to Gwei
        except Exception as e:
            logger.error(f"Error fetching Polygon gas price: {e}")
            
        # Fetch from BSC
        try:
            if "bsc" in self.web3_connections:
                bsc_gas_price = self.web3_connections["bsc"].eth.gas_price
                gas_data["gas_prices"]["bsc"] = bsc_gas_price / 1e9  # Convert to Gwei
        except Exception as e:
            logger.error(f"Error fetching BSC gas price: {e}")
            
        # Save gas price data
        if gas_data["gas_prices"]:
            self.market_store.save_market_snapshot(gas_data, source="gas_prices")
            
        return gas_data
    
    async def get_protocol_tvl(self, protocol_address, chain="ethereum"):
        """Get TVL from a protocol by checking its contracts"""
        if chain not in self.web3_connections:
            logger.error(f"No Web3 connection for chain: {chain}")
            return None
            
        web3 = self.web3_connections[chain]
        
        try:
            # This is a simplified example and would need to be customized
            # based on the specific protocol's contract structure
            if protocol_address in self.protocol_configs.get(chain, {}):
                config = self.protocol_configs[chain][protocol_address]
                
                # Load ABI
                with open(config["abi_path"], 'r') as f:
                    abi = json.load(f)
                    
                # Create contract instance
                contract = web3.eth.contract(address=protocol_address, abi=abi)
                
                # Call TVL function (varies by protocol)
                if hasattr(contract.functions, config.get("tvl_function", "getTotalValueLocked")):
                    tvl_function = getattr(contract.functions, config.get("tvl_function"))
                    tvl_wei = tvl_function().call()
                    
                    # Convert to USD using protocol's token price
                    # This would need price data from an oracle or API
                    return tvl_wei / 1e18  # Simplified conversion
                    
            return None
        except Exception as e:
            logger.error(f"Error getting protocol TVL: {e}")
            return None
    
    async def get_token_balances(self, address, network="ethereum"):
        """
        Use Alchemy API to get all ERC20 token balances for an address
        
        Args:
            address (str): The Ethereum address to check
            network (str): The network to query (ethereum, polygon, etc.)
            
        Returns:
            dict: Token balance information
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address, "erc20"]
        }
        
        try:
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully fetched token balances for {address}")
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get token balances: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching token balances: {e}")
            return None
    
    async def get_token_metadata(self, token_addresses, network="ethereum"):
        """
        Get metadata for a list of token addresses using Alchemy API
        
        Args:
            token_addresses (list): List of token contract addresses
            network (str): The network to query
            
        Returns:
            dict: Token metadata information
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        results = {}
        
        for address in token_addresses:
            payload = {
                "id": 1,
                "jsonrpc": "2.0",
                "method": "alchemy_getTokenMetadata",
                "params": [address]
            }
            
            try:
                async with self.session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        results[address] = result.get("result", {})
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to get token metadata: {response.status} - {error_text}")
            except Exception as e:
                logger.error(f"Error fetching token metadata: {e}")
                
        return results
    
    async def get_asset_transfers(self, address, category=["external", "internal", "erc20"], 
                                  from_block="0x0", to_block="latest", network="ethereum"):
        """
        Get historical transfers for an address using Alchemy API
        
        Args:
            address (str): The address to check transfers for
            category (list): Categories of transfers to include
            from_block (str): The starting block (hex)
            to_block (str): The ending block (hex)
            network (str): The network to query
            
        Returns:
            dict: Historical transfer data
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": from_block,
                    "toBlock": to_block,
                    "fromAddress": address,
                    "category": category
                }
            ]
        }
        
        try:
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully fetched asset transfers for {address}")
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get asset transfers: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching asset transfers: {e}")
            return None

    async def analyze_wallet(self, address):
        """
        Analyze a wallet address to get comprehensive information using Alchemy API
        
        Args:
            address (str): The wallet address to analyze
            
        Returns:
            dict: Wallet analysis data
        """
        # Fetch token balances
        token_balances = await self.get_token_balances(address)
        
        if not token_balances or not token_balances.get("tokenBalances"):
            logger.warning(f"No token balances found for {address}")
            return {"error": "No token data available"}
        
        # Extract token addresses
        token_addresses = [tb.get("contractAddress") for tb in token_balances.get("tokenBalances", [])]
        
        # Get token metadata
        token_metadata = await self.get_token_metadata(token_addresses)
        
        # Get recent transfers
        # Use a fixed block range for recent activity
        from_block = "0x" + hex(18000000)[2:]  # Start from a reasonable recent block
        to_block = "latest"
        
        recent_transfers = await self.get_asset_transfers(
            address=address, 
            from_block=from_block,
            to_block=to_block
        )
        
        # Combine the data
        wallet_analysis = {
            "address": address,
            "tokens": [],
            "recent_activity": recent_transfers.get("transfers", []) if recent_transfers else []
        }
        
        # Process token data
        for balance in token_balances.get("tokenBalances", []):
            contract_address = balance.get("contractAddress")
            token_data = token_metadata.get(contract_address, {})
            
            raw_balance = int(balance.get("tokenBalance", "0x0"), 16)
            decimals = int(token_data.get("decimals", "0"))
            
            try:
                actual_balance = raw_balance / (10 ** decimals) if decimals > 0 else raw_balance
            except Exception:
                actual_balance = raw_balance
                
            wallet_analysis["tokens"].append({
                "contract_address": contract_address,
                "name": token_data.get("name", "Unknown Token"),
                "symbol": token_data.get("symbol", "???"),
                "logo": token_data.get("logo", None),
                "balance": actual_balance,
                "raw_balance": raw_balance,
                "decimals": decimals
            })
        
        return wallet_analysis

    async def get_eth_gas_price(self, network="ethereum"):
        """
        Get current gas price using Eth API
        
        Args:
            network (str): The network to query (ethereum, polygon, etc.)
            
        Returns:
            float: Gas price in Gwei
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_gasPrice",
            "params": [],
            "id": 1
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    gas_price_hex = result.get("result", "0x0")
                    gas_price_wei = int(gas_price_hex, 16)
                    gas_price_gwei = gas_price_wei / 1e9
                    return gas_price_gwei
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get gas price: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching gas price: {e}")
            return None
            
    async def get_latest_block(self, network="ethereum"):
        """
        Get information about the latest block
        
        Args:
            network (str): The network to query
            
        Returns:
            dict: Block information
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": ["latest", False],
            "id": 1
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get latest block: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching latest block: {e}")
            return None
    
    async def get_latest_block_number(self, network="ethereum"):
        """
        Get the latest block number
        
        Args:
            network (str): The network to query
            
        Returns:
            str: Block number in hex
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return "0x0"
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_blockNumber",
            "params": [],
            "id": 1
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", "0x0")
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get latest block number: {response.status} - {error_text}")
                    return "0x0"
        except Exception as e:
            logger.error(f"Error fetching latest block number: {e}")
            return "0x0"
    
    async def get_nft_balance(self, address, network="ethereum"):
        """
        Get NFT balance for an address
        
        Args:
            address (str): The address to check
            network (str): The network to query
            
        Returns:
            dict: NFT balance information
        """
        # For this implementation, we'll use the standard getTokenBalances method
        # with erc721 parameter, which works for ERC-721 tokens (NFTs)
        
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTokenBalances",
            "params": [address, "erc721"]
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Successfully fetched NFT balances for {address}")
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get NFT balances: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching NFT balances: {e}")
            return None
    
    async def get_contract_logs(self, contract_address, event_signature, 
                               max_results=10, from_block=None, to_block="latest", network="ethereum"):
        """
        Get logs for a specific contract and event
        
        Args:
            contract_address (str): The contract address
            event_signature (str): The event signature hash
            max_results (int): Maximum number of results to return
            from_block (str): The starting block (hex)
            to_block (str): The ending block (hex)
            network (str): The network to query
            
        Returns:
            list: Contract logs
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        
        if from_block is None:
            # Default to 10000 blocks back if not specified
            latest_block_hex = await self.get_latest_block_number(network)
            latest_block = int(latest_block_hex, 16)
            from_block = "0x" + hex(max(0, latest_block - 10000))[2:]
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_getLogs",
            "params": [
                {
                    "address": contract_address,
                    "topics": [event_signature],
                    "fromBlock": from_block,
                    "toBlock": to_block
                }
            ]
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logs = result.get("result", [])
                    return logs[:max_results]  # Limit the number of results
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get contract logs: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching contract logs: {e}")
            return None
    
    async def get_transaction_receipts_by_block(self, block_number, network="ethereum"):
        """
        Get all transaction receipts for a block
        
        Args:
            block_number (str): The block number in hex
            network (str): The network to query
            
        Returns:
            dict: Transaction receipts
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "alchemy_getTransactionReceipts",
            "params": [{"blockNumber": block_number}]
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get transaction receipts: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching transaction receipts: {e}")
            return None
    
    async def get_transaction_trace(self, tx_hash, network="ethereum"):
        """
        Get the trace for a transaction using Alchemy Trace API
        
        Args:
            tx_hash (str): The transaction hash
            network (str): The network to query
            
        Returns:
            dict: Transaction trace
        """
        if network not in self.alchemy_endpoints:
            logger.error(f"Network {network} not supported by Alchemy endpoints")
            return None
            
        endpoint = self.alchemy_endpoints[network]
        payload = {
            "id": 1,
            "jsonrpc": "2.0",
            "method": "trace_transaction",
            "params": [tx_hash]
        }
        
        try:
            if not self.session:
                await self.initialize()
                
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("result", {})
                else:
                    error_text = await response.text()
                    logger.error(f"Failed to get transaction trace: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching transaction trace: {e}")
            return None
    
    async def run_data_collection_cycle(self):
        """Run a complete data collection cycle, fetching data from all sources."""
        logger.info("Starting data collection cycle...")
        
        try:
            # Fetch yield data from DeFi Llama
            yield_data = await self.fetch_defi_llama_data()
            
            # Store yield data
            if yield_data and "data" in yield_data:
                for pool in yield_data.get("data", []):
                    try:
                        self.market_store.save_pool_data({
                            "timestamp": int(time.time()),
                            "pool_id": pool.get("pool", ""),
                            "protocol": pool.get("project", ""),
                            "chain": pool.get("chain", ""),
                            "tvl_usd": pool.get("tvlUsd", 0.0),
                            "apy": pool.get("apy", 0.0),
                            "risk_level": self._calculate_risk_level(pool),
                            "source": "defi_llama"
                        })
                    except Exception as e:
                        logger.error(f"Error saving pool data: {e}")
            
            # Fetch token prices
            token_prices = await self.fetch_token_prices()
            
            # Store token prices
            if token_prices and "prices" in token_prices:
                for token, price_data in token_prices.get("prices", {}).items():
                    try:
                        self.market_store.save_token_price({
                            "timestamp": int(time.time()),
                            "token_symbol": token,
                            "price_usd": price_data.get("usd", 0.0),
                            "market_cap": price_data.get("market_cap", 0.0),
                            "volume_24h": price_data.get("volume_24h", 0.0),
                            "source": "coingecko"
                        })
                    except Exception as e:
                        logger.error(f"Error saving token price: {e}")
            
            # Fetch gas prices
            gas_data = await self.fetch_gas_prices()
            
            # Store gas prices
            if gas_data and "gas_prices" in gas_data:
                for network, gas_price in gas_data.get("gas_prices", {}).items():
                    try:
                        self.market_store.save_gas_price_data({
                            "timestamp": int(time.time()),
                            "network": network,
                            "gas_price_gwei": gas_price,
                            "source": "etherscan"
                        })
                    except Exception as e:
                        logger.error(f"Error saving gas price: {e}")
            
            # Adding Alchemy-specific data collection
            if ALCHEMY_API_KEY:
                try:
                    # Get current gas prices from Alchemy (potentially more accurate)
                    eth_gas_price = await self.get_eth_gas_price()
                    if eth_gas_price:
                        logger.info(f"Ethereum gas price from Alchemy: {eth_gas_price:.2f} Gwei")
                        # Store gas price data
                        self.market_store.save_gas_price_data({
                            "timestamp": int(time.time()),
                            "network": "ethereum",
                            "gas_price_gwei": eth_gas_price,
                            "source": "alchemy_api"
                        })
                    
                    # Get latest block data (useful for timing strategies)
                    latest_block = await self.get_latest_block()
                    if latest_block:
                        block_number = int(latest_block.get("number", "0x0"), 16)
                        block_timestamp = int(latest_block.get("timestamp", "0x0"), 16)
                        logger.info(f"Latest Ethereum block: {block_number} (timestamp: {block_timestamp})")
                        # Store block data
                        self.market_store.save_block_data({
                            "timestamp": int(time.time()),
                            "network": "ethereum",
                            "block_number": block_number,
                            "block_timestamp": block_timestamp,
                            "gas_used": int(latest_block.get("gasUsed", "0x0"), 16),
                            "transaction_count": len(latest_block.get("transactions", [])),
                            "source": "alchemy_api"
                        })
                    
                    # Example: Collect data for important DeFi protocols
                    important_addresses = [
                        # Aave lending pool on Ethereum
                        "0x7d2768dE32b0b80b7a3454c06BdAc94A69DDc7A9",
                        # Uniswap V3 factory
                        "0x1F98431c8aD98523631AE4a59f267346ea31F984",
                        # Compound Comptroller
                        "0x3d9819210A31b4961b30EF54bE2aeD79B9c9Cd3B"
                    ]
                    
                    for address in important_addresses:
                        try:
                            # Analyze protocol wallet
                            wallet_data = await self.analyze_wallet(address)
                            if wallet_data and "error" not in wallet_data:
                                # Store the protocol data
                                self.market_store.save_protocol_data({
                                    "timestamp": int(time.time()),
                                    "protocol_address": address,
                                    "protocol_name": self._get_protocol_name(address),
                                    "data": wallet_data,
                                    "source": "alchemy_api"
                                })
                            
                            # Get protocol activity (Transfer events, which are common in DeFi)
                            transfer_event_sig = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
                            logs = await self.get_contract_logs(address, transfer_event_sig, max_results=20)
                            if logs:
                                logger.info(f"Found {len(logs)} recent events for protocol {address}")
                                # Store protocol activity data
                                self.market_store.save_protocol_data({
                                    "timestamp": int(time.time()),
                                    "protocol_address": address,
                                    "protocol_name": self._get_protocol_name(address),
                                    "data": {"activity_logs": logs, "event_type": "Transfer"},
                                    "source": "alchemy_api"
                                })
                        except Exception as e:
                            logger.error(f"Error analyzing protocol address {address}: {e}")
                except Exception as e:
                    logger.error(f"Error in Alchemy data collection: {e}")
            
            # Create a summary
            summary = {
                "timestamp": datetime.now().isoformat(),
                "protocols_collected": len(yield_data.get("data", [])) if yield_data else 0,
                "token_prices_collected": len(token_prices.get("prices", {})) if token_prices else 0,
                "gas_prices_collected": len(gas_data.get("gas_prices", {})) if gas_data else 0
            }
            
            logger.info(f"Data collection cycle completed: {json.dumps(summary)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error in data collection cycle: {e}")
            return {"error": str(e)}
            
    def _get_protocol_name(self, address):
        """Get protocol name from address"""
        address = address.lower()
        
        # Check if we have the protocol in our configs
        for protocol_id, protocol_data in self.protocol_configs.get("protocols", {}).items():
            for chain, addresses in protocol_data.get("addresses", {}).items():
                for key, addr in addresses.items():
                    if addr.lower() == address:
                        return protocol_data.get("name", protocol_id)
        
        return "Unknown Protocol"

# Function to run the data fetcher as a daemon
async def run_continuous_data_collection(interval_minutes=15):
    """Run continuous data collection at specified intervals"""
    # Create data fetcher
    fetcher = LiveDataFetcher()
    
    try:
        # Initialize connections
        await fetcher.initialize()
        
        logger.info(f"Starting continuous data collection every {interval_minutes} minutes")
        
        while True:
            try:
                # Run a complete data collection cycle
                await fetcher.run_data_collection_cycle()
                
                # Find best opportunities
                opportunities = await fetcher.get_best_opportunities()
                logger.info(f"Found {len(opportunities)} promising opportunities")
                
                # Create market summary
                summary = await fetcher.create_market_summary()
                if summary:
                    logger.info(f"Market summary: Avg APY: {summary['market_overview']['average_apy']:.2%}, "
                               f"Total TVL: ${summary['market_overview']['total_tvl']:,.2f}")
                
                # Pause for the specified interval
                logger.info(f"Waiting {interval_minutes} minutes until next data collection cycle")
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in data collection cycle: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
                
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in data collection: {e}")
    finally:
        # Close connections
        await fetcher.close()

# For direct testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DEFIMIND Live Data Fetcher")
    parser.add_argument("--once", action="store_true", help="Run a single data collection cycle")
    parser.add_argument("--interval", type=int, default=15, help="Minutes between data collection cycles")
    
    args = parser.parse_args()
    
    if args.once:
        async def run_once():
            fetcher = LiveDataFetcher()
            await fetcher.initialize()
            await fetcher.run_data_collection_cycle()
            opportunities = await fetcher.get_best_opportunities()
            print(f"Top opportunities:")
            for i, opp in enumerate(opportunities[:5], 1):
                print(f"{i}. {opp['protocol']} - {opp['pool']}: {opp['apy']:.2%} APY")
            await fetcher.close()
            
        asyncio.run(run_once())
    else:
        # Run continuous data collection
        asyncio.run(run_continuous_data_collection(interval_minutes=args.interval)) 