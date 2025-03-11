#!/usr/bin/env python3
"""
DEFIMIND Persistence Layer

This module provides database storage for persistent learning and memory.
Enables the agent to evolve over time and remember past experiences.
"""

import os
import json
import time
import logging
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import pickle
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("defimind_persistence.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("defimind_persistence")

# Default paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, "data", "memory")
MARKET_DATA_DIR = os.path.join(BASE_DIR, "data", "market")

# Create directories if they don't exist
Path(MEMORY_DIR).mkdir(parents=True, exist_ok=True)
Path(MARKET_DATA_DIR).mkdir(parents=True, exist_ok=True)

class MemoryDatabase:
    """Persistent storage for agent memory and knowledge"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(MEMORY_DIR, "memory.db")
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize the database if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memory table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            type TEXT,
            content TEXT,
            metadata TEXT
        )
        ''')
        
        # Create learning table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS learnings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            concept TEXT,
            understanding TEXT,
            confidence REAL,
            source TEXT
        )
        ''')
        
        # Create decision table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            context TEXT,
            decision TEXT,
            reasoning TEXT,
            outcome TEXT,
            feedback REAL
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_short_term_memory(self, content, content_type="text", importance=0.5, context=None):
        """Add an item to short-term memory"""
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO short_term_memory (timestamp, content, content_type, importance, context) VALUES (?, ?, ?, ?, ?)",
            (timestamp, str(content), content_type, importance, json.dumps(context) if context else None)
        )
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id
        
    def add_long_term_memory(self, content, content_type="text", importance=0.7, embedding_file=None, context=None):
        """Add an item to long-term memory"""
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO long_term_memory (timestamp, content, content_type, importance, embedding_file, context) VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, str(content), content_type, importance, embedding_file, json.dumps(context) if context else None)
        )
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id
        
    def add_episodic_memory(self, actions, observations, outcome, metadata=None, importance=0.6):
        """Add an episode to episodic memory"""
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO episodic_memory (timestamp, actions, observations, outcome, metadata, importance) VALUES (?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                json.dumps(actions),
                json.dumps(observations),
                json.dumps(outcome),
                json.dumps(metadata) if metadata else None,
                importance
            )
        )
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id
        
    def add_market_data(self, protocol, pool, apy, tvl, volume_24h=None, price=None, raw_data=None):
        """Store market data in the database"""
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO market_data (timestamp, protocol, pool, apy, tvl, volume_24h, price, raw_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                protocol.lower(),
                pool,
                apy,
                tvl,
                volume_24h,
                price,
                json.dumps(raw_data) if raw_data else None
            )
        )
        
        data_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return data_id
        
    def add_performance_record(self, action_type, protocol=None, allocation=None, expected_outcome=None, actual_outcome=None, success_rating=None, profit_loss=None):
        """Record performance of an action for learning"""
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO performance (timestamp, action_type, protocol, allocation, expected_outcome, actual_outcome, success_rating, profit_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                action_type,
                protocol,
                allocation,
                json.dumps(expected_outcome) if expected_outcome else None,
                json.dumps(actual_outcome) if actual_outcome else None,
                success_rating,
                profit_loss
            )
        )
        
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return record_id
        
    def get_recent_memories(self, limit=10):
        """Get recent memories from short-term memory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM short_term_memory ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        
        memories = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return memories
        
    def get_important_long_term_memories(self, importance_threshold=0.7, limit=20):
        """Get important long-term memories"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM long_term_memory WHERE importance >= ? ORDER BY importance DESC, last_accessed DESC LIMIT ?",
            (importance_threshold, limit)
        )
        
        memories = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return memories
        
    def search_memories(self, query, limit=10):
        """Search memories using simple keyword matching"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Simple keyword search across both memory types
        cursor.execute(
            """
            SELECT 'short_term' as memory_type, * FROM short_term_memory 
            WHERE content LIKE ? 
            UNION ALL
            SELECT 'long_term' as memory_type, * FROM long_term_memory 
            WHERE content LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit)
        )
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
        
    def get_protocol_performance(self, protocol, days=30):
        """Get historical performance data for a protocol"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT timestamp, apy, tvl, volume_24h, price
            FROM market_data
            WHERE protocol = ? AND timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (protocol.lower(), cutoff_date)
        )
        
        data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return data
        
    def consolidate_memories(self):
        """Move important short-term memories to long-term memory"""
        # Get short-term memories that are important enough to keep
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Select memories older than 1 day with importance >= 0.6
        one_day_ago = (datetime.now() - timedelta(days=1)).isoformat()
        
        cursor.execute(
            """
            SELECT * FROM short_term_memory
            WHERE timestamp < ? AND importance >= 0.6
            """,
            (one_day_ago,)
        )
        
        important_memories = [dict(row) for row in cursor.fetchall()]
        
        # Move these to long-term memory
        for memory in important_memories:
            cursor.execute(
                """
                INSERT INTO long_term_memory
                (timestamp, content, content_type, importance, context)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory['timestamp'],
                    memory['content'],
                    memory['content_type'],
                    memory['importance'],
                    memory['context']
                )
            )
            
            # Remove from short-term memory
            cursor.execute("DELETE FROM short_term_memory WHERE id = ?", (memory['id'],))
        
        # Clean up old, unimportant short-term memories
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute("DELETE FROM short_term_memory WHERE timestamp < ?", (thirty_days_ago,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Memory consolidation: {len(important_memories)} memories moved to long-term storage, {deleted_count} old memories pruned")
        
        return {
            "consolidated_count": len(important_memories),
            "pruned_count": deleted_count
        }

class ModelManager:
    """Manages ML models for learning from experiences"""
    
    def __init__(self, model_dir=MEMORY_DIR):
        self.model_dir = model_dir
        Path(model_dir).mkdir(exist_ok=True)
        
    def save_model(self, model, name, metadata=None):
        """Save a trained model to disk"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        filepath = os.path.join(self.model_dir, filename)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
        # Save metadata if provided
        if metadata:
            metadata_file = f"{filepath}.meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        logger.info(f"Model saved to {filepath}")
        return filepath
        
    def load_latest_model(self, name_prefix):
        """Load the latest model with the given name prefix"""
        files = [f for f in os.listdir(self.model_dir) if f.startswith(name_prefix) and f.endswith('.pkl')]
        
        if not files:
            logger.warning(f"No models found with prefix {name_prefix}")
            return None
            
        # Get the most recent file
        latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(self.model_dir, f)))
        filepath = os.path.join(self.model_dir, latest_file)
        
        # Load the model
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        logger.info(f"Loaded model from {filepath}")
        return model
        
    def get_model_history(self, name_prefix):
        """Get history of all models with the given prefix"""
        files = [f for f in os.listdir(self.model_dir) if f.startswith(name_prefix) and f.endswith('.pkl')]
        
        history = []
        for file in files:
            filepath = os.path.join(self.model_dir, file)
            
            # Try to load metadata
            metadata_file = f"{filepath}.meta.json"
            metadata = None
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    
            history.append({
                "filename": file,
                "created": datetime.fromtimestamp(os.path.getctime(filepath)).isoformat(),
                "size_bytes": os.path.getsize(filepath),
                "metadata": metadata
            })
            
        # Sort by creation time (newest first)
        history.sort(key=lambda x: x["created"], reverse=True)
        return history

class MarketDataStore:
    """Storage for market data and analysis results"""
    
    def __init__(self, db_path=None):
        """Initialize the market data store"""
        if db_path is None:
            db_path = os.path.join(MARKET_DATA_DIR, "market_data.db")
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Token prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS token_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            token_symbol TEXT,
            price_usd REAL,
            market_cap REAL,
            volume_24h REAL,
            source TEXT
        )
        ''')
        
        # Pool data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pool_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            pool_id TEXT,
            protocol TEXT,
            chain TEXT,
            tvl_usd REAL,
            apy REAL,
            risk_level REAL,
            source TEXT
        )
        ''')
        
        # Gas prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS gas_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            network TEXT,
            gas_price_gwei REAL,
            source TEXT
        )
        ''')
        
        # Block data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS block_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            network TEXT,
            block_number INTEGER,
            block_timestamp INTEGER,
            gas_used INTEGER,
            transaction_count INTEGER,
            source TEXT
        )
        ''')
        
        # Protocol historical data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS protocol_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            protocol_address TEXT,
            protocol_name TEXT,
            data_json TEXT,
            source TEXT
        )
        ''')
        
        # Investment decisions
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS investment_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            protocol TEXT,
            chain TEXT,
            allocation_percentage REAL,
            reason TEXT,
            status TEXT
        )
        ''')
        
        # Market analysis
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            analysis_type TEXT,
            data_json TEXT
        )
        ''')
        
        # Analytics logs (for debugging)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            log_type TEXT,
            message TEXT,
            data_json TEXT
        )
        ''')
        
        self.conn.commit()
        
    def save_token_price(self, data):
        """Save token price data"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO token_prices (timestamp, token_symbol, price_usd, market_cap, volume_24h, source) VALUES (?, ?, ?, ?, ?, ?)',
            (
                data.get('timestamp', int(time.time())),
                data.get('token_symbol', ''),
                data.get('price_usd', 0.0),
                data.get('market_cap', 0.0),
                data.get('volume_24h', 0.0),
                data.get('source', 'default')
            )
        )
        self.conn.commit()
        
    def save_pool_data(self, data):
        """Save pool data"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO pool_data (timestamp, pool_id, protocol, chain, tvl_usd, apy, risk_level, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                data.get('timestamp', int(time.time())),
                data.get('pool_id', ''),
                data.get('protocol', ''),
                data.get('chain', ''),
                data.get('tvl_usd', 0.0),
                data.get('apy', 0.0),
                data.get('risk_level', 0.0),
                data.get('source', 'default')
            )
        )
        self.conn.commit()
        
    def save_gas_price_data(self, data):
        """Save gas price data"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO gas_prices (timestamp, network, gas_price_gwei, source) VALUES (?, ?, ?, ?)',
            (
                data.get('timestamp', int(time.time())),
                data.get('network', 'ethereum'),
                data.get('gas_price_gwei', 0.0),
                data.get('source', 'default')
            )
        )
        self.conn.commit()
        
    def save_block_data(self, data):
        """Save blockchain block data"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO block_data (timestamp, network, block_number, block_timestamp, gas_used, transaction_count, source) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                data.get('timestamp', int(time.time())),
                data.get('network', 'ethereum'),
                data.get('block_number', 0),
                data.get('block_timestamp', 0),
                data.get('gas_used', 0),
                data.get('transaction_count', 0),
                data.get('source', 'default')
            )
        )
        self.conn.commit()
        
    def save_protocol_data(self, data):
        """Save protocol data"""
        cursor = self.conn.cursor()
        
        # Convert complex data to JSON
        data_json = json.dumps(data.get('data', {}))
        
        cursor.execute(
            'INSERT INTO protocol_data (timestamp, protocol_address, protocol_name, data_json, source) VALUES (?, ?, ?, ?, ?)',
            (
                data.get('timestamp', int(time.time())),
                data.get('protocol_address', ''),
                data.get('protocol_name', ''),
                data_json,
                data.get('source', 'default')
            )
        )
        self.conn.commit()
        
    def save_investment_decision(self, data):
        """Save investment decision"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO investment_decisions (timestamp, protocol, chain, allocation_percentage, reason, status) VALUES (?, ?, ?, ?, ?, ?)',
            (
                data.get('timestamp', int(time.time())),
                data.get('protocol', ''),
                data.get('chain', ''),
                data.get('allocation_percentage', 0.0),
                data.get('reason', ''),
                data.get('status', 'pending')
            )
        )
        self.conn.commit()
        return cursor.lastrowid
        
    def save_market_analysis(self, data):
        """Save market analysis"""
        cursor = self.conn.cursor()
        
        # Convert complex data to JSON
        data_json = json.dumps(data.get('data', {}))
        
        cursor.execute(
            'INSERT INTO market_analysis (timestamp, analysis_type, data_json) VALUES (?, ?, ?)',
            (
                data.get('timestamp', int(time.time())),
                data.get('analysis_type', ''),
                data_json
            )
        )
        self.conn.commit()
        
    def log_analytics(self, log_type, message, data=None):
        """Log analytics data for debugging"""
        cursor = self.conn.cursor()
        
        # Convert data to JSON if present
        data_json = json.dumps(data) if data else None
        
        cursor.execute(
            'INSERT INTO analytics_logs (timestamp, log_type, message, data_json) VALUES (?, ?, ?, ?)',
            (
                int(time.time()),
                log_type,
                message,
                data_json
            )
        )
        self.conn.commit()
        
    def get_latest_token_prices(self, limit=50):
        """Get latest token prices"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT token_symbol, price_usd, timestamp FROM token_prices ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        )
        return cursor.fetchall()
        
    def get_best_pools(self, min_tvl=1000000, limit=10):
        """Get best pools by APY with minimum TVL"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT protocol, chain, pool_id, tvl_usd, apy, risk_level FROM pool_data WHERE tvl_usd >= ? ORDER BY apy DESC LIMIT ?',
            (min_tvl, limit)
        )
        return cursor.fetchall()
        
    def get_historical_gas_prices(self, network='ethereum', hours=24):
        """Get historical gas prices for a network"""
        cursor = self.conn.cursor()
        timestamp_cutoff = int(time.time()) - (hours * 3600)  # Convert hours to seconds
        cursor.execute(
            'SELECT timestamp, gas_price_gwei FROM gas_prices WHERE network = ? AND timestamp > ? ORDER BY timestamp ASC',
            (network, timestamp_cutoff)
        )
        return cursor.fetchall()
        
    def get_recent_blocks(self, network='ethereum', limit=100):
        """Get recent block data for a network"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT block_number, block_timestamp, gas_used, transaction_count FROM block_data WHERE network = ? ORDER BY block_number DESC LIMIT ?',
            (network, limit)
        )
        return cursor.fetchall()
        
    def get_protocol_data(self, protocol_address, limit=10):
        """Get historical data for a protocol"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT timestamp, data_json FROM protocol_data WHERE protocol_address = ? ORDER BY timestamp DESC LIMIT ?',
            (protocol_address, limit)
        )
        results = cursor.fetchall()
        
        # Parse JSON in the results
        parsed_results = []
        for timestamp, data_json in results:
            try:
                data = json.loads(data_json)
                parsed_results.append({
                    'timestamp': timestamp,
                    'data': data
                })
            except json.JSONDecodeError:
                continue
                
        return parsed_results
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

    def get_historical_pools(self, days=90):
        """Get historical pool data for the last N days"""
        cursor = self.conn.cursor()
        timestamp_cutoff = int(time.time()) - (days * 24 * 3600)  # Convert days to seconds
        cursor.execute(
            'SELECT timestamp, protocol, chain, pool_id, tvl_usd, apy, risk_level FROM pool_data WHERE timestamp > ? ORDER BY timestamp DESC',
            (timestamp_cutoff,)
        )
        
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        result = []
        for row in rows:
            result.append({
                "timestamp": row[0],
                "protocol": row[1],
                "chain": row[2],
                "pool_id": row[3],
                "tvl_usd": row[4],
                "apy": row[5],
                "risk_level": row[6]
            })
            
        return result
    
    def get_gas_price_at_time(self, timestamp, network="ethereum"):
        """Get gas price at specific time for a network"""
        cursor = self.conn.cursor()
        # Find closest gas price entry to the timestamp
        cursor.execute(
            'SELECT gas_price_gwei FROM gas_prices WHERE network = ? AND ABS(timestamp - ?) < 7200 ORDER BY ABS(timestamp - ?) LIMIT 1',
            (network, timestamp, timestamp)  # 2-hour window
        )
        
        row = cursor.fetchone()
        if row:
            return row[0]
        return None
    
    def get_latest_market_analyses(self, analysis_type=None, limit=1):
        """Get latest market analyses of a specific type"""
        cursor = self.conn.cursor()
        
        if analysis_type:
            cursor.execute(
                'SELECT timestamp, analysis_type, data_json FROM market_analysis WHERE analysis_type = ? ORDER BY timestamp DESC LIMIT ?',
                (analysis_type, limit)
            )
        else:
            cursor.execute(
                'SELECT timestamp, analysis_type, data_json FROM market_analysis ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            )
            
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        result = []
        for row in rows:
            try:
                data = json.loads(row[2])
                result.append({
                    "timestamp": row[0],
                    "analysis_type": row[1],
                    "data": data
                })
            except (json.JSONDecodeError, TypeError):
                # Skip entries with invalid JSON
                continue
                
        return result
    
    def get_historical_market_analyses(self, analysis_type=None, days=30):
        """Get historical market analyses for the last N days"""
        cursor = self.conn.cursor()
        timestamp_cutoff = int(time.time()) - (days * 24 * 3600)  # Convert days to seconds
        
        if analysis_type:
            cursor.execute(
                'SELECT timestamp, analysis_type, data_json FROM market_analysis WHERE analysis_type = ? AND timestamp > ? ORDER BY timestamp DESC',
                (analysis_type, timestamp_cutoff)
            )
        else:
            cursor.execute(
                'SELECT timestamp, analysis_type, data_json FROM market_analysis WHERE timestamp > ? ORDER BY timestamp DESC',
                (timestamp_cutoff,)
            )
            
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        result = []
        for row in rows:
            try:
                data = json.loads(row[2])
                result.append({
                    "timestamp": row[0],
                    "analysis_type": row[1],
                    "data": data
                })
            except (json.JSONDecodeError, TypeError):
                # Skip entries with invalid JSON
                continue
                
        return result
    
    def get_latest_investment_decisions(self, limit=5):
        """Get latest investment decisions"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT timestamp, protocol, chain, allocation_percentage, reason, status FROM investment_decisions ORDER BY timestamp DESC LIMIT ?',
            (limit,)
        )
        
        rows = cursor.fetchall()
        
        # Convert to list of dicts
        result = []
        for row in rows:
            result.append({
                "timestamp": row[0],
                "protocol": row[1],
                "chain": row[2],
                "allocation_percentage": row[3],
                "reason": row[4],
                "status": row[5]
            })
            
        return result

# Simple usage example
if __name__ == "__main__":
    db = MemoryDatabase()
    
    # Add some test memories
    db.add_short_term_memory("Observed high APY on PancakeSwap CAKE-BNB pool", importance=0.7)
    db.add_short_term_memory("Gas prices increasing, might delay transactions", importance=0.6)
    
    db.add_long_term_memory("PancakeSwap historically provides the best returns during bull markets", importance=0.8)
    
    # Add market data
    db.add_market_data("pancakeswap", "CAKE-BNB", 0.215, 12500000)
    db.add_market_data("traderjoe", "JOE-AVAX", 0.23, 5800000)
    
    # Test retrieval
    recent_memories = db.get_recent_memories(5)
    print(f"Retrieved {len(recent_memories)} recent memories")
    
    # Test model manager
    model_mgr = ModelManager()
    
    # Create a simple test model
    import sklearn.ensemble
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
    model.fit([[0, 1], [1, 1], [2, 2]], [0, 1, 2])  # Dummy training
    
    # Save the model
    model_path = model_mgr.save_model(model, "test_model", {"accuracy": 0.95})
    print(f"Model saved to {model_path}")
    
    # Market data store
    market_store = MarketDataStore()
    snapshot_path = market_store.save_market_snapshot({
        "protocols": {
            "pancakeswap": {"apy": 0.215, "tvl": 12500000},
            "traderjoe": {"apy": 0.23, "tvl": 5800000}
        }
    })
    print(f"Market snapshot saved to {snapshot_path}")
    
    print("Persistence layer test complete") 