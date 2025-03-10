import os
import json
import asyncio
import logging
import numpy as np
import tensorflow as tf
from web3 import Web3
from dotenv import load_dotenv
from datetime import datetime, timedelta
from yield_scanner import YieldScanner  # Import yield scanner for live APY data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_trading_bot")

# Load environment variables
load_dotenv()

# Connect to Polygon Mainnet using Infura RPC from .env
INFURA_RPC_URL = os.getenv("INFURA_RPC_URL")
if not INFURA_RPC_URL:
    logger.error("❌ Missing INFURA_RPC_URL in environment variables!")
    exit(1)

try:
    web3 = Web3(Web3.HTTPProvider(INFURA_RPC_URL))
    if web3.is_connected():
        logger.info("✅ Successfully connected to Polygon Mainnet via Infura!")
    else:
        raise ConnectionError("Failed to connect to Polygon RPC.")
except Exception as e:
    logger.error(f"❌ RPC Connection Error: {e}")
    exit(1)

class AITradingAgent:
    def __init__(self, model_path=None):
        self.model = self.create_model()
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"✅ Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def predict_action(self, inputs):
        try:
            normalized_inputs = np.clip(inputs, 0, 1)  # Ensure values are between 0-1
            predictions = self.model.predict(np.array([normalized_inputs]))
            return predictions[0]
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return np.array([0.33, 0.33, 0.34])

class TradingBot:
    def __init__(self):
        self.scanner = YieldScanner()
        self.ai_agent = AITradingAgent("models/latest_model.h5")
        self.allocations = {"pancakeswap": 0, "traderjoe": 0, "quickswap": 0}

    async def collect_features(self):
        pancake_pool = await self.scanner.get_best_apy_for_protocol("pancakeswap")
        joe_pool = await self.scanner.get_best_apy_for_protocol("traderjoe")
        quickswap_pool = await self.scanner.get_best_apy_for_protocol("quickswap")

        pancake_apy = min(pancake_pool.apy if pancake_pool else 0.30, 1.0)
        joe_apy = min(joe_pool.apy if joe_pool else 0.25, 1.0)
        quickswap_apy = min(quickswap_pool.apy if quickswap_pool else 0.20, 1.0)

        gas_price = web3.eth.gas_price / 10**9
        volatility = 0.4

        logger.info(f"📊 APYs => Pancake: {pancake_apy*100:.2f}%, Joe: {joe_apy*100:.2f}%, QuickSwap: {quickswap_apy*100:.2f}%")
        logger.info(f"📉 Market Conditions => Volatility: {volatility:.2f}, Gas Price: {gas_price:.2f}")

        return [pancake_apy, joe_apy, quickswap_apy, volatility, gas_price]

    async def decide_allocation(self):
        features = await self.collect_features()
        prediction = self.ai_agent.predict_action(features)

        allocations = {
            "pancakeswap": float(prediction[0]),
            "traderjoe": float(prediction[1]),
            "quickswap": float(prediction[2])
        }
        logger.info(f"🤖 AI recommended allocations: {allocations}")
        return allocations

    async def execute_rebalance(self, allocations, total_amount_usd=1000):
        try:
            logger.info(f"🔄 Rebalancing portfolio with ${total_amount_usd} total investment")
            amounts = {protocol: total_amount_usd * percentage for protocol, percentage in allocations.items()}
            logger.info(f"💰 Allocating: {amounts}")
            self.allocations = allocations
            logger.info(f"✅ Final Allocations: {self.allocations}")
            return True
        except Exception as e:
            logger.error(f"❌ Error in rebalance: {e}")
            return False

async def main():
    bot = TradingBot()
    try:
        await bot.scanner.initialize()
        allocations = await bot.decide_allocation()
        await bot.execute_rebalance(allocations)
    except Exception as e:
        logger.error(f"❌ Error in main execution: {e}")
    finally:
        await bot.scanner.close()

if __name__ == "__main__":
    asyncio.run(main())
