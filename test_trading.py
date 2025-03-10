import logging
from ai_agent import BlockchainManager  # or wherever BlockchainManager is defined
from dotenv import load_dotenv
import os

logger = logging.getLogger("ai_trading_bot")

def main():
    load_dotenv()
    bc_manager = BlockchainManager(chain="bsc")  # Example using BSC

    # Attempt to stake tokens:
    try:
        pool_id = 1  # example
        amount_wei = int(0.1 * 1e18)  # stake 0.1 tokens
        tx_hash = bc_manager.stake_tokens(pool_id, amount_wei)
        logger.info(f"Staking transaction: {tx_hash}")
    except Exception as e:
        logger.error(f"Error in test stake: {e}")

if __name__ == "__main__":
    main()

