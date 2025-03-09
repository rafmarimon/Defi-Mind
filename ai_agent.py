import os
import json
import time
import requests
import logging
import numpy as np
import tensorflow as tf
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_trading_bot")

# Load environment variables from .env file
load_dotenv()

#####################################
# 1) AI MODEL (Training & Decision) #
#####################################

class AITradingAgent:
    def __init__(self, model_path=None):
        """
        Initialize the AI Trading Agent with optional pre-trained model loading.
        
        Args:
            model_path (str, optional): Path to load a pre-trained model from.
        """
        self.model = self.create_model()
        self.history = None
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

    def create_model(self):
        """
        Create a neural network model for yield farming decision making.
        
        Returns:
            tf.keras.Model: A compiled TensorFlow model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(5,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # Output for each protocol
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self, data, labels, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the model with historical yield and market data.
        
        Args:
            data (np.array): Features matrix including APYs, market conditions, etc.
            labels (np.array): Target labels (one-hot encoded for each protocol)
            validation_split (float): Portion of data to use for validation
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            history: Training history object
        """
        # Implement early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduce learning rate when training plateaus
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        self.history = self.model.fit(
            np.array(data), 
            np.array(labels),
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save the trained model
        self.save_model("models/latest_model.h5")
        return self.history

    def save_model(self, path):
        """Save the model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_weights(path)
        logger.info(f"Model saved to {path}")
        
    def predict_action(self, inputs):
        """
        Predict which protocol to allocate funds to.
        
        Args:
            inputs (list/array): Input features 
                                 e.g. [pancake_apy, joe_apy, quickswap_apy, market_volatility, gas_price]
                                                
        Returns:
            np.array: Probability distribution across protocols
        """
        try:
            predictions = self.model.predict(np.array([inputs]))
            return predictions[0]  # Returns probabilities for each protocol
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Default to roughly equal distribution if error
            return np.array([0.33, 0.33, 0.34])


##############################
# 2) YIELD SCANNING SERVICES #
##############################

class YieldScanner:
    def __init__(self):
        """Initialize the yield scanner with API keys and settings"""
        self.api_keys = {
            "defi_llama": os.getenv("DEFI_LLAMA_API_KEY", ""),
            "covalent": os.getenv("COVALENT_API_KEY", "")
        }
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(minutes=10)  # Cache APY data for 10 minutes
    
    def _get_cached_or_fetch(self, key, fetch_func):
        """Get data from cache or fetch fresh data if expired"""
        now = datetime.now()
        
        if key in self.cache and self.cache_expiry[key] > now:
            logger.debug(f"Using cached data for {key}")
            return self.cache[key]
            
        logger.info(f"Fetching fresh data for {key}")
        data = fetch_func()
        self.cache[key] = data
        self.cache_expiry[key] = now + self.cache_duration
        return data

    def get_pancakeswap_apy(self):
        """
        Get PancakeSwap APY from DefiLlama API or fallback.
        
        Returns:
            float: Current APY as a decimal (e.g., 0.30 for 30%)
        """
        def fetch_data():
            try:
                # Replace with actual API endpoint
                url = "https://yields.llama.fi/pools"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Filter for PancakeSwap pools
                    pancake_pools = [p for p in data.get('data', []) if 'pancake' in p.get('project', '').lower()]
                    if pancake_pools:
                        # Return highest APY from PancakeSwap pools
                        return max([p.get('apy', 0) / 100 for p in pancake_pools])
                return 0.30  # Fallback
            except Exception as e:
                logger.error(f"Error fetching PancakeSwap APY: {e}")
                return 0.30  # Fallback
        
        return self._get_cached_or_fetch("pancakeswap", fetch_data)

    def get_traderjoe_apy(self):
        """
        Get Trader Joe APY from DefiLlama or fallback.
        
        Returns:
            float: Current APY as a decimal
        """
        def fetch_data():
            try:
                url = "https://yields.llama.fi/pools"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    # Filter for Trader Joe pools
                    joe_pools = [p for p in data.get('data', []) if 'joe' in p.get('project', '').lower()]
                    if joe_pools:
                        return max([p.get('apy', 0) / 100 for p in joe_pools])
                return 0.25
            except Exception as e:
                logger.error(f"Error fetching Trader Joe APY: {e}")
                return 0.25
        
        return self._get_cached_or_fetch("traderjoe", fetch_data)

    def get_quickswap_apy(self):
        """
        Get QuickSwap APY from DefiLlama or fallback.
        
        Returns:
            float: Current APY as a decimal
        """
        def fetch_data():
            try:
                url = "https://yields.llama.fi/pools"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    quick_pools = [p for p in data.get('data', []) if 'quick' in p.get('project', '').lower()]
                    if quick_pools:
                        return max([p.get('apy', 0) / 100 for p in quick_pools])
                return 0.20
            except Exception as e:
                logger.error(f"Error fetching QuickSwap APY: {e}")
                return 0.20
        
        return self._get_cached_or_fetch("quickswap", fetch_data)
    
    def get_market_volatility(self):
        """
        Get market volatility index as an additional feature.
        
        Returns:
            float: Volatility index normalized to 0-1 range
        """
        try:
            # This would typically come from a market data API
            return 0.4
        except Exception as e:
            logger.error(f"Error fetching market volatility: {e}")
            return 0.5
    
    def get_gas_prices(self, chain="bsc"):
        """
        Get current gas prices for a blockchain (normalized 0-1).
        
        Args:
            chain (str): Chain identifier (bsc, avalanche, polygon)
            
        Returns:
            float: Normalized gas price
        """
        chains = {
            "bsc": "https://api.bscscan.com/api?module=proxy&action=eth_gasPrice&apikey=",
            "avalanche": "https://api.snowtrace.io/api?module=proxy&action=eth_gasPrice&apikey=",
            "polygon": "https://api.polygonscan.com/api?module=proxy&action=eth_gasPrice&apikey="
        }
        
        try:
            if chain in chains:
                # Example simulated values for demonstration
                gas_prices = {"bsc": 5, "avalanche": 25, "polygon": 50}
                return min(1.0, gas_prices.get(chain, 5) / 100)  # normalize
            return 0.05
        except Exception as e:
            logger.error(f"Error fetching gas price for {chain}: {e}")
            return 0.05


###################################
# 3) BLOCKCHAIN INTERACTION CLASS #
###################################

class BlockchainManager:
    def __init__(self, chain="bsc"):
        """
        Initialize blockchain connection for a specific chain.
        
        Args:
            chain (str): Chain to connect to (bsc, avalanche, polygon)
        """
        self.chain = chain
        self.rpc_urls = {
            "bsc": os.getenv("BSC_RPC_URL", "https://data-seed-prebsc-1-s1.binance.org:8545"),
            "avalanche": os.getenv("AVAX_RPC_URL", "https://api.avax-test.network/ext/bc/C/rpc"),
            "polygon": os.getenv("POLYGON_RPC_URL", "https://rpc-mumbai.maticvigil.com")
        }
        
        # Connect to the blockchain
        self.web3 = Web3(Web3.HTTPProvider(self.rpc_urls.get(chain)))
        if not self.web3.is_connected():
            logger.error(f"Failed to connect to {chain} network")
            raise ConnectionError(f"Could not connect to {chain} RPC")
            
        # Load wallet info from environment variables
        self.private_key = os.getenv("PRIVATE_KEY")
        self.wallet_address = os.getenv("WALLET_ADDRESS")
        
        if not self.private_key or not self.wallet_address:
            logger.error("Missing wallet credentials in environment variables")
            raise ValueError("Private key and wallet address must be configured")
            
        # Contract addresses for each protocol on each chain
        self.contracts = {
            "bsc": {
                "pancakeswap": {
                    "router": "0x10ED43C718714eb63d5aA57B78B54704E256024E",
                    "masterchef": "0xa5f8C5Dbd5F286960b9d90548680aE8290B43e99",
                    "token": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82"
                }
            },
            "avalanche": {
                "traderjoe": {
                    "router": "0x60aE616a2155Ee3d9A68541Ba4544862310933d4",
                    "masterchef": "0x188bED1968b795d5c9022F6a0bb5931Ac4c18F00",
                    "token": "0x6e84a6216eA6dACC71eE8E6b0a5B7322EEbC0fDd"
                }
            },
            "polygon": {
                "quickswap": {
                    "router": "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff",
                    "masterchef": "0x20ec0d06F447d550fC6edee42121bc8C1817b97D",
                    "token": "0x831753DD7087CaC61aB5644b308642cc1c33Dc13"
                }
            }
        }
        
        # Load ABIs from files or embed them
        self.abis = {
            "masterchef": self._load_abi("MasterChef_ABI.json", self._default_masterchef_abi()),
            "router": self._load_abi("Router_ABI.json", self._default_router_abi()),
            "token": self._load_abi("ERC20_ABI.json", self._default_erc20_abi())
        }
    
    def _load_abi(self, filename, default_abi):
        """Load ABI from file or use default"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
            return default_abi
        except Exception as e:
            logger.warning(f"Failed to load ABI from {filename}: {e}")
            return default_abi
            
    def _default_masterchef_abi(self):
        """Return a simplified MasterChef ABI with essential functions"""
        return [
            {"inputs":[{"internalType":"uint256","name":"_pid","type":"uint256"},{"internalType":"uint256","name":"_amount","type":"uint256"}],"name":"deposit","outputs":[],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[{"internalType":"uint256","name":"_pid","type":"uint256"},{"internalType":"uint256","name":"_amount","type":"uint256"}],"name":"withdraw","outputs":[],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[{"internalType":"uint256","name":"_pid","type":"uint256"},{"internalType":"address","name":"_user","type":"address"}],"name":"pendingReward","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"internalType":"uint256","name":"","type":"uint256"}],"name":"poolInfo","outputs":[{"internalType":"address","name":"lpToken","type":"address"},{"internalType":"uint256","name":"allocPoint","type":"uint256"},{"internalType":"uint256","name":"lastRewardBlock","type":"uint256"},{"internalType":"uint256","name":"accRewardPerShare","type":"uint256"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"internalType":"uint256","name":"","type":"uint256"},{"internalType":"address","name":"","type":"address"}],"name":"userInfo","outputs":[{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"uint256","name":"rewardDebt","type":"uint256"}],"stateMutability":"view","type":"function"}
        ]
        
    def _default_router_abi(self):
        """Return a simplified Router ABI with essential functions"""
        return [
            {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"},{"internalType":"uint256","name":"amountInMax","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapTokensForExactTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
        ]
        
    def _default_erc20_abi(self):
        """Return a simplified ERC20 ABI with essential functions"""
        return [
            {"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
            {"constant":False,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"nonpayable","type":"function"},
            {"constant":True,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
            {"constant":False,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"payable":False,"stateMutability":"nonpayable","type":"function"}
        ]
    
    def load_contract(self, protocol, contract_type):
        """
        Load a contract instance based on protocol and type.
        
        Args:
            protocol (str): Protocol name (pancakeswap, traderjoe, quickswap)
            contract_type (str): Type of contract (masterchef, router, token)
            
        Returns:
            Contract: Web3 contract instance
        """
        try:
            contract_address = self.contracts[self.chain][protocol][contract_type]
            contract_abi = self.abis[contract_type]
            return self.web3.eth.contract(address=contract_address, abi=contract_abi)
        except KeyError:
            logger.error(f"Contract not found: {protocol} {contract_type} on {self.chain}")
            raise ValueError(f"Contract configuration not available for {protocol} {contract_type}")
    
    def stake_tokens(self, protocol, pool_id, amount_wei):
        """
        Stake tokens in a yield farm.
        
        Args:
            protocol (str): Protocol to stake in (pancakeswap, traderjoe, quickswap)
            pool_id (int): Pool ID to stake in
            amount_wei (int): Amount to stake in wei
            
        Returns:
            str: Transaction hash
        """
        try:
            masterchef = self.load_contract(protocol, "masterchef")
            pool_info = masterchef.functions.poolInfo(pool_id).call()
            lp_token_address = pool_info[0]
            
            lp_token = self.web3.eth.contract(address=lp_token_address, abi=self.abis["token"])
            balance = lp_token.functions.balanceOf(self.wallet_address).call()
            
            if balance < amount_wei:
                logger.error(f"Insufficient balance: {balance} < {amount_wei}")
                raise ValueError(f"Not enough tokens to stake. Have: {balance/10**18}, Need: {amount_wei/10**18}")
            
            # Approve if needed
            allowance = lp_token.functions.allowance(self.wallet_address, masterchef.address).call()
            if allowance < amount_wei:
                logger.info(f"Approving token spend: {amount_wei}")
                approve_txn = lp_token.functions.approve(
                    masterchef.address,
                    amount_wei
                ).build_transaction({
                    "from": self.wallet_address,
                    "nonce": self.web3.eth.get_transaction_count(self.wallet_address),
                    "gas": 100000,
                    "gasPrice": self.web3.eth.gas_price
                })
                
                signed_approve = self.web3.eth.account.sign_transaction(approve_txn, self.private_key)
                approve_hash = self.web3.eth.send_raw_transaction(signed_approve.rawTransaction)
                logger.info(f"Waiting for approval: {self.web3.toHex(approve_hash)}")
                self.web3.eth.wait_for_transaction_receipt(approve_hash)
            
            # Deposit
            nonce = self.web3.eth.get_transaction_count(self.wallet_address)
            stake_txn = masterchef.functions.deposit(
                pool_id, 
                amount_wei
            ).build_transaction({
                "from": self.wallet_address,
                "nonce": nonce,
                "gas": 300000,
                "gasPrice": self.web3.eth.gas_price
            })
            
            signed_txn = self.web3.eth.account.sign_transaction(stake_txn, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                logger.info(f"Staking successful! TX: {self.web3.toHex(tx_hash)}")
            else:
                logger.error(f"Staking failed. TX: {self.web3.toHex(tx_hash)}")
            
            return self.web3.toHex(tx_hash)
        except Exception as e:
            logger.error(f"Error in stake_tokens: {e}")
            raise

    def unstake_tokens(self, protocol, pool_id, amount_wei):
        """
        Unstake tokens from a yield farm.
        
        Args:
            protocol (str): Protocol to unstake from
            pool_id (int): Pool ID
            amount_wei (int): Amount to unstake in wei
            
        Returns:
            str: Transaction hash
        """
        try:
            masterchef = self.load_contract(protocol, "masterchef")
            user_info = masterchef.functions.userInfo(pool_id, self.wallet_address).call()
            staked_amount = user_info[0]
            
            if staked_amount < amount_wei:
                logger.warning(f"Not enough staked tokens. Have: {staked_amount/10**18}, Want: {amount_wei/10**18}")
                amount_wei = staked_amount
            
            nonce = self.web3.eth.get_transaction_count(self.wallet_address)
            unstake_txn = masterchef.functions.withdraw(
                pool_id, 
                amount_wei
            ).build_transaction({
                "from": self.wallet_address,
                "nonce": nonce,
                "gas": 300000,
                "gasPrice": self.web3.eth.gas_price
            })
            
            signed_txn = self.web3.eth.account.sign_transaction(unstake_txn, self.private_key)
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            if receipt.status == 1:
                logger.info(f"Unstaking successful! TX: {self.web3.toHex(tx_hash)}")
            else:
                logger.error(f"Unstaking failed. TX: {self.web3.toHex(tx_hash)}")
            
            return self.web3.toHex(tx_hash)
        except Exception as e:
            logger.error(f"Error in unstake_tokens: {e}")
            raise

    def check_pending_rewards(self, protocol, pool_id):
        """
        Check pending reward tokens for a staked position.
        
        Args:
            protocol (str): Protocol name
            pool_id (int): Pool ID
            
        Returns:
            float: Pending rewards in token units (not wei)
        """
        try:
            masterchef = self.load_contract(protocol, "masterchef")
            pending = masterchef.functions.pendingReward(pool_id, self.wallet_address).call()
            return pending / 10**18
        except Exception as e:
            logger.error(f"Error checking rewards: {e}")
            return 0


#########################
# 4) STRATEGY MANAGER   #
#########################

class YieldFarmingStrategy:
    def __init__(self, chain="bsc"):
        """Initialize the yield farming strategy controller"""
        self.scanner = YieldScanner()
        self.model_path = "models/latest_model.h5"
        self.ai_agent = AITradingAgent(self.model_path)
        
        # This manager handles on-chain interactions for the chosen chain
        self.bc_manager = BlockchainManager(chain=chain)
        
        # Track current allocations
        self.allocations = {
            "pancakeswap": 0,
            "traderjoe": 0,
            "quickswap": 0
        }
        
        # Track historical performance
        self.performance_history = []
        
        # We assume each protocol has a single "pool_id" for demonstration
        self.protocol_pool_ids = {
            "pancakeswap": 1, 
            "traderjoe": 0,
            "quickswap": 2
        }
        
        # How many tokens do we have available? 
        # For simplicity, let's assume we have 1 LP token for each protocol
        # In real life, you'd handle actual token addresses, bridging, etc.
        self.total_tokens = 1.0
        
    def collect_features(self):
        """
        Collect all features needed for the AI decision.
        
        Returns:
            list: [pancake_apy, joe_apy, quickswap_apy, volatility, gas_price]
        """
        pancake_apy = self.scanner.get_pancakeswap_apy()
        joe_apy = self.scanner.get_traderjoe_apy()
        quickswap_apy = self.scanner.get_quickswap_apy()
        
        volatility = self.scanner.get_market_volatility()
        gas_price = self.scanner.get_gas_prices(chain=self.bc_manager.chain)
        
        logger.info(f"APYs => Pancake: {pancake_apy:.2%}, Joe: {joe_apy:.2%}, QuickSwap: {quickswap_apy:.2%}")
        logger.info(f"Volatility: {volatility:.2f}, Gas Price: {gas_price:.2f}")
        
        return [pancake_apy, joe_apy, quickswap_apy, volatility, gas_price]
        
    def decide_allocation(self):
        """
        Decide how to allocate funds based on AI predictions.
        
        Returns:
            dict: Allocation percentages for each protocol
        """
        features = self.collect_features()
        prediction = self.ai_agent.predict_action(features)  # 3-element array, sum ~ 1.0
        
        allocations = {
            "pancakeswap": float(prediction[0]),
            "traderjoe": float(prediction[1]),
            "quickswap": float(prediction[2])
        }
        logger.info(f"AI recommended allocations: {allocations}")
        return allocations
    
    def execute_rebalance(self, allocations, total_tokens=None):
        """
        Execute a portfolio rebalance based on the allocations.
        
        Args:
            allocations (dict): Target allocations for each protocol
            total_tokens (float): total tokens we have to allocate
            
        Returns:
            bool: True if successful
        """
        if total_tokens is None:
            total_tokens = self.total_tokens
        
        # Each protocol's token amount = total_tokens * allocation
        # In real life, you might need to handle bridging, swapping, etc.
        for protocol, allocation_fraction in allocations.items():
            target_amount = total_tokens * allocation_fraction
            current_amount = self.allocations.get(protocol, 0)
            pool_id = self.protocol_pool_ids.get(protocol, 0)
            
            # Example: if we have 0.5 staked but AI wants 0.2 => unstake 0.3
            # If we have 0.2 but AI wants 0.5 => stake 0.3 more
            diff = target_amount - current_amount
            
            if diff > 0:
                # Need to stake more
                logger.info(f"Staking {diff} tokens in {protocol}")
                # Convert diff to Wei - assume 1 token = 1e18 Wei
                amount_wei = int(diff * 1e18)
                try:
                    self.bc_manager.stake_tokens(protocol, pool_id, amount_wei)
                except Exception as e:
                    logger.error(f"Failed to stake in {protocol}: {e}")
            elif diff < 0:
                # Need to unstake
                logger.info(f"Unstaking {-diff} tokens from {protocol}")
                amount_wei = int((-diff) * 1e18)
                try:
                    self.bc_manager.unstake_tokens(protocol, pool_id, amount_wei)
                except Exception as e:
                    logger.error(f"Failed to unstake from {protocol}: {e}")
            
            # Update local record
            self.allocations[protocol] = target_amount
        
        # Save performance snapshot
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "allocations": self.allocations.copy(),
            "apy_snapshot": {
                "pancakeswap": self.scanner.get_pancakeswap_apy(),
                "traderjoe": self.scanner.get_traderjoe_apy(),
                "quickswap": self.scanner.get_quickswap_apy()
            }
        })
        logger.info(f"Rebalance complete. New allocations: {self.allocations}")
        return True
    
    def run_strategy(self):
        """
        High-level function: 
        1) Get AI's recommended allocations
        2) Rebalance to match those allocations
        """
        allocations = self.decide_allocation()
        self.execute_rebalance(allocations)


########################################
# 5) MAIN EXECUTION (ENTRY POINT)      #
########################################

def main():
    """
    1. Initialize strategy
    2. Optionally train the model if you want new data
    3. Run the yield-farming strategy
    """
    try:
        # Example: use BSC testnet
        strategy = YieldFarmingStrategy(chain="bsc")
        
        # (Optional) If you want to quickly re-train:
        # data = np.random.rand(100, 5)   # 5 features
        # labels = np.random.randint(0, 3, size=(100,))
        # one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=3)
        # strategy.ai_agent.train_model(data, one_hot_labels, epochs=5)
        
        # Run the strategy (scan yields -> AI decides -> stake/unstake)
        strategy.run_strategy()
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
