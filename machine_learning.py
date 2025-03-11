#!/usr/bin/env python3
"""
DEFIMIND Machine Learning Module

Provides continuous learning and prediction capabilities for the DEFIMIND trading agent.
"""

import os
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from defimind_persistence import MemoryDatabase, MarketDataStore, ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("machine_learning")

# Load environment variables
load_dotenv()

# Constants
MIN_TRAINING_SAMPLES = int(os.getenv("MIN_TRAINING_SAMPLES", "100"))
RETRAINING_INTERVAL_DAYS = int(os.getenv("RETRAINING_INTERVAL_DAYS", "7"))
MODELS_DIR = os.getenv("MODELS_DIR", "data/models")
USE_AUTO_ML = os.getenv("USE_AUTO_ML", "false").lower() == "true"


class BaseModel:
    """Base class for all machine learning models"""
    
    def __init__(self, name, model_type="regression"):
        """Initialize the base model"""
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.target_name = ""
        self.model_manager = ModelManager()
        self.metadata = {
            "created_at": None,
            "last_trained": None,
            "training_samples": 0,
            "performance": {},
            "features": []
        }
    
    def preprocess(self, X):
        """Preprocess input data"""
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X)
            return X_scaled
        else:
            return self.scaler.transform(X)
    
    def fit(self, X, y):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement fit()")
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model is not trained yet")
            
        X_scaled = self.preprocess(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model is not trained yet")
            
        # Make predictions
        predictions = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        # Update metadata
        self.metadata["performance"] = {
            "mse": mse,
            "mae": mae,
            "rmse": np.sqrt(mse),
            "r2": r2
        }
        
        return self.metadata["performance"]
    
    def save(self):
        """Save the model"""
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "metadata": self.metadata
        }
        
        self.model_manager.save_model(model_data, self.name, self.metadata)
        logger.info(f"Model {self.name} saved successfully")
    
    def load(self):
        """Load the latest version of the model"""
        model_data = self.model_manager.load_latest_model(self.name)
        
        if model_data:
            self.model = model_data.get("model")
            self.scaler = model_data.get("scaler")
            self.feature_names = model_data.get("feature_names", [])
            self.target_name = model_data.get("target_name", "")
            self.metadata = model_data.get("metadata", {})
            logger.info(f"Model {self.name} loaded successfully")
            return True
        
        logger.warning(f"No saved model found for {self.name}")
        return False


class YieldPredictor(BaseModel):
    """Model for predicting yield returns based on market conditions"""
    
    def __init__(self):
        """Initialize the yield predictor model"""
        super().__init__("yield_predictor")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    
    def prepare_training_data(self, market_store):
        """Prepare training data from market store"""
        # In a real implementation, this would query historical pool data
        # and prepare it for training. For demonstration purposes, using
        # placeholder method.
        
        # Get historical pool data
        pools = market_store.get_historical_pools(days=90)
        
        if not pools or len(pools) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient data for training. Need at least {MIN_TRAINING_SAMPLES} samples.")
            return None, None
        
        # Convert to DataFrame
        data = []
        for pool in pools:
            # Extract features and target
            timestamp = pool.get("timestamp")
            protocol = pool.get("protocol", "")
            chain = pool.get("chain", "")
            pool_id = pool.get("pool_id", "")
            tvl_usd = pool.get("tvl_usd", 0)
            apy = pool.get("apy", 0)  # Target variable
            risk_level = pool.get("risk_level", 0.5)
            
            # Add additional features based on timestamp
            dt = datetime.fromtimestamp(timestamp)
            day_of_week = dt.weekday()
            hour_of_day = dt.hour
            
            # Get gas price at that time
            gas_price = market_store.get_gas_price_at_time(timestamp, network=chain) or 20.0  # Default 20 Gwei
            
            # Create feature vector
            data.append({
                "timestamp": timestamp,
                "protocol": protocol,
                "chain": chain,
                "pool_id": pool_id,
                "tvl_usd": tvl_usd,
                "risk_level": risk_level,
                "gas_price_gwei": gas_price,
                "day_of_week": day_of_week,
                "hour_of_day": hour_of_day,
                "apy": apy  # Target
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create dummy variables for categorical features
        df_encoded = pd.get_dummies(df, columns=["protocol", "chain"])
        
        # Extract features and target
        feature_cols = [c for c in df_encoded.columns if c != "apy" and c != "timestamp" and c != "pool_id"]
        X = df_encoded[feature_cols]
        y = df_encoded["apy"]
        
        self.feature_names = feature_cols
        self.target_name = "apy"
        
        return X, y
    
    def fit(self, X, y):
        """Train the yield prediction model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        train_preds = self.model.predict(X_train_scaled)
        val_preds = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_preds)
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        
        # Update metadata
        self.metadata["created_at"] = self.metadata.get("created_at") or datetime.now().isoformat()
        self.metadata["last_trained"] = datetime.now().isoformat()
        self.metadata["training_samples"] = len(X)
        self.metadata["performance"] = {
            "train_mse": train_mse,
            "train_rmse": np.sqrt(train_mse),
            "val_mse": val_mse,
            "val_rmse": np.sqrt(val_mse),
            "val_mae": val_mae,
            "val_r2": val_r2
        }
        self.metadata["features"] = self.feature_names
        
        logger.info(f"Model trained on {len(X)} samples. Validation RMSE: {np.sqrt(val_mse):.4f}")
        return self.metadata["performance"]
    
    def predict_yield(self, protocol, chain, tvl_usd, risk_level, gas_price_gwei=None):
        """Predict yield for a protocol/chain combination"""
        if self.model is None:
            success = self.load()
            if not success:
                logger.error("Failed to load model for prediction")
                return None
        
        # Create feature vector
        features = {}
        
        # Add continuous features
        features["tvl_usd"] = tvl_usd
        features["risk_level"] = risk_level
        features["gas_price_gwei"] = gas_price_gwei or 20.0  # Default gas price
        
        # Add time-based features
        now = datetime.now()
        features["day_of_week"] = now.weekday()
        features["hour_of_day"] = now.hour
        
        # Create dummy variables for categorical features
        # For protocol
        for feature in self.feature_names:
            if feature.startswith("protocol_"):
                protocol_name = feature.split("protocol_")[1]
                features[feature] = 1 if protocol == protocol_name else 0
        
        # For chain
        for feature in self.feature_names:
            if feature.startswith("chain_"):
                chain_name = feature.split("chain_")[1]
                features[feature] = 1 if chain == chain_name else 0
        
        # Fill any missing features with 0
        for feature in self.feature_names:
            if feature not in features:
                features[feature] = 0
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure correct columns in correct order
        X = X[self.feature_names]
        
        # Make prediction
        predicted_apy = self.predict(X)[0]
        
        return max(0, predicted_apy)  # Ensure non-negative APY


class RiskAssessor(BaseModel):
    """Model for assessing risk levels based on protocol and market data"""
    
    def __init__(self):
        """Initialize the risk assessor model"""
        super().__init__("risk_assessor")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
    
    def prepare_training_data(self, market_store, memory_db):
        """Prepare training data for risk assessment"""
        # In a real implementation, this would combine market data
        # with historical performance data to train a risk model
        
        # Use protocol analytics results and historical volatility
        protocol_analyses = market_store.get_historical_market_analyses(
            analysis_type="protocol_analysis", 
            days=90
        )
        
        if not protocol_analyses or len(protocol_analyses) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Insufficient data for risk model training. Need at least {MIN_TRAINING_SAMPLES} samples.")
            return None, None
        
        # Prepare data
        data = []
        for analysis in protocol_analyses:
            timestamp = analysis.get("timestamp")
            protocol_data = analysis.get("data", {})
            protocol = protocol_data.get("protocol", "")
            health_metrics = protocol_data.get("health_metrics", {})
            
            # Skip entries without proper health metrics
            if not health_metrics:
                continue
            
            # Extract risk-related features
            tvl = health_metrics.get("total_value_locked_usd") or health_metrics.get("total_supply_usd", 0)
            utilization = health_metrics.get("utilization_rate", 0.5)
            health_factor = health_metrics.get("average_health_factor", 1.5)
            liquidation_risk = 0.7 if health_metrics.get("liquidation_risk") == "high" else 0.3 if health_metrics.get("liquidation_risk") == "low" else 0.5
            
            # Get market conditions at that time
            gas_price = market_store.get_gas_price_at_time(timestamp) or 20.0
            
            # Target variable: risk level
            # In a real system, this would come from historical performance data
            # For demo, deriving it from metrics
            risk_level = 0.2 + (0.3 * utilization) + (0.3 * (1.0 / max(1.0, health_factor))) + (0.2 * liquidation_risk)
            risk_level = min(1.0, max(0.0, risk_level))  # Ensure between 0 and 1
            
            data.append({
                "timestamp": timestamp,
                "protocol": protocol,
                "tvl_usd": tvl,
                "utilization_rate": utilization,
                "health_factor": health_factor,
                "liquidation_risk": liquidation_risk,
                "gas_price_gwei": gas_price,
                "risk_level": risk_level  # Target
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create dummy variables for categorical features
        df_encoded = pd.get_dummies(df, columns=["protocol"])
        
        # Extract features and target
        feature_cols = [c for c in df_encoded.columns if c != "risk_level" and c != "timestamp"]
        X = df_encoded[feature_cols]
        y = df_encoded["risk_level"]
        
        self.feature_names = feature_cols
        self.target_name = "risk_level"
        
        return X, y
    
    def fit(self, X, y):
        """Train the risk assessment model"""
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_preds = self.model.predict(X_train_scaled)
        val_preds = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_preds)
        val_mse = mean_squared_error(y_val, val_preds)
        val_mae = mean_absolute_error(y_val, val_preds)
        val_r2 = r2_score(y_val, val_preds)
        
        # Update metadata
        self.metadata["created_at"] = self.metadata.get("created_at") or datetime.now().isoformat()
        self.metadata["last_trained"] = datetime.now().isoformat()
        self.metadata["training_samples"] = len(X)
        self.metadata["performance"] = {
            "train_mse": train_mse,
            "train_rmse": np.sqrt(train_mse),
            "val_mse": val_mse,
            "val_rmse": np.sqrt(val_mse),
            "val_mae": val_mae,
            "val_r2": val_r2
        }
        self.metadata["features"] = self.feature_names
        
        logger.info(f"Risk model trained on {len(X)} samples. Validation RMSE: {np.sqrt(val_mse):.4f}")
        return self.metadata["performance"]
    
    def assess_risk(self, protocol, tvl_usd, utilization_rate, health_factor=1.5, gas_price_gwei=None):
        """Assess risk for a protocol with given metrics"""
        if self.model is None:
            success = self.load()
            if not success:
                logger.error("Failed to load risk model for prediction")
                return 0.5  # Default moderate risk
        
        # Create feature vector
        features = {}
        
        # Add continuous features
        features["tvl_usd"] = tvl_usd
        features["utilization_rate"] = utilization_rate
        features["health_factor"] = health_factor
        features["liquidation_risk"] = 0.5  # Default moderate risk
        features["gas_price_gwei"] = gas_price_gwei or 20.0
        
        # Add protocol one-hot encoding
        for feature in self.feature_names:
            if feature.startswith("protocol_"):
                protocol_name = feature.split("protocol_")[1]
                features[feature] = 1 if protocol == protocol_name else 0
        
        # Fill any missing features with 0
        for feature in self.feature_names:
            if feature not in features:
                features[feature] = 0
        
        # Convert to DataFrame
        X = pd.DataFrame([features])
        
        # Ensure correct columns in correct order
        X = X[self.feature_names]
        
        # Make prediction and ensure it's between 0 and 1
        predicted_risk = self.predict(X)[0]
        return min(1.0, max(0.0, predicted_risk))


class PortfolioOptimizer(BaseModel):
    """Model for optimizing portfolio allocations based on predictions"""
    
    def __init__(self):
        """Initialize the portfolio optimizer"""
        super().__init__("portfolio_optimizer")
        self.yield_predictor = YieldPredictor()
        self.risk_assessor = RiskAssessor()
        self.risk_tolerance = float(os.getenv("RISK_TOLERANCE", "0.6"))  # Default moderate risk tolerance
        self.min_allocation = float(os.getenv("MIN_ALLOCATION_PERCENT", "0.05"))  # Minimum 5% allocation
        self.max_allocation = float(os.getenv("MAX_ALLOCATION_PERCENT", "0.4"))  # Maximum 40% allocation
    
    def optimize_portfolio(self, opportunities, total_investment=100.0):
        """Optimize portfolio allocation based on opportunities"""
        # Load models if needed
        if not self.yield_predictor.model:
            self.yield_predictor.load()
        
        if not self.risk_assessor.model:
            self.risk_assessor.load()
        
        # Enhanced opportunities with predictions
        enhanced_opportunities = []
        
        for opportunity in opportunities:
            protocol = opportunity.get("protocol", "")
            chain = opportunity.get("chain", "ethereum")
            pool = opportunity.get("pool", "")
            tvl = opportunity.get("tvl", 1000000)
            reported_apy = opportunity.get("apy", 0)
            reported_risk = opportunity.get("risk_level", 0.5)
            
            # Predict APY and risk using our models
            predicted_apy = self.yield_predictor.predict_yield(
                protocol, chain, tvl, reported_risk
            )
            
            # If prediction failed, use reported APY
            if predicted_apy is None:
                predicted_apy = reported_apy
            
            # Blend reported and predicted APY (giving more weight to our prediction)
            blended_apy = 0.3 * reported_apy + 0.7 * predicted_apy
            
            # Assess risk
            utilization = opportunity.get("utilization_rate", 0.5)
            health_factor = opportunity.get("health_factor", 1.5)
            predicted_risk = self.risk_assessor.assess_risk(
                protocol, tvl, utilization, health_factor
            )
            
            # Blend reported and predicted risk
            blended_risk = 0.3 * reported_risk + 0.7 * predicted_risk
            
            # Calculate risk-adjusted return
            risk_adjusted_return = blended_apy / (blended_risk + 0.1)  # Avoid division by zero
            
            # Add to enhanced opportunities
            enhanced_opportunities.append({
                "protocol": protocol,
                "chain": chain,
                "pool": pool,
                "tvl": tvl,
                "reported_apy": reported_apy,
                "predicted_apy": predicted_apy,
                "blended_apy": blended_apy,
                "reported_risk": reported_risk,
                "predicted_risk": predicted_risk,
                "blended_risk": blended_risk,
                "risk_adjusted_return": risk_adjusted_return
            })
        
        # Filter opportunities based on risk tolerance
        filtered_opportunities = [
            opp for opp in enhanced_opportunities 
            if opp["blended_risk"] <= self.risk_tolerance
        ]
        
        # If no opportunities meet risk criteria, return nothing or cash allocation
        if not filtered_opportunities:
            return [{
                "protocol": "CASH",
                "chain": "none",
                "allocation_percentage": 1.0,
                "amount": total_investment,
                "reasoning": ["All opportunities exceed risk tolerance", "Holding as cash"]
            }]
        
        # Sort by risk-adjusted return (descending)
        filtered_opportunities.sort(key=lambda x: x["risk_adjusted_return"], reverse=True)
        
        # Limit to top opportunities
        top_opportunities = filtered_opportunities[:5]
        
        # Calculate allocation based on risk-adjusted return
        total_rar = sum(opp["risk_adjusted_return"] for opp in top_opportunities)
        
        # Initialize allocations
        allocations = []
        remaining_allocation = 1.0
        
        # First pass: Calculate proportional allocations
        for opp in top_opportunities:
            allocation = (opp["risk_adjusted_return"] / total_rar) if total_rar > 0 else 0
            
            # Apply min/max constraints
            allocation = max(self.min_allocation, min(self.max_allocation, allocation))
            remaining_allocation -= allocation
            
            allocations.append({
                "protocol": opp["protocol"],
                "chain": opp["chain"],
                "pool": opp["pool"],
                "allocation_percentage": allocation,
                "amount": allocation * total_investment,
                "expected_apy": opp["blended_apy"],
                "risk_level": opp["blended_risk"],
                "reasoning": [
                    f"Strong risk-adjusted return: {opp['risk_adjusted_return']:.4f}",
                    f"Blended APY: {opp['blended_apy']:.2f}%",
                    f"Risk assessment: {opp['blended_risk']:.2f}"
                ]
            })
        
        # Check if we need to allocate remaining to cash
        if remaining_allocation > 0.05:  # Only add cash if significant amount remains
            allocations.append({
                "protocol": "CASH",
                "chain": "none",
                "allocation_percentage": remaining_allocation,
                "amount": remaining_allocation * total_investment,
                "expected_apy": 0,
                "risk_level": 0,
                "reasoning": ["Strategic cash reserve", "Available for future opportunities"]
            })
        else:
            # Normalize allocations to sum to 1.0
            total_allocation = sum(alloc["allocation_percentage"] for alloc in allocations)
            for alloc in allocations:
                alloc["allocation_percentage"] /= total_allocation
                alloc["amount"] = alloc["allocation_percentage"] * total_investment
        
        return allocations


class ModelTrainer:
    """Manages training and updating of all models"""
    
    def __init__(self):
        """Initialize the model trainer"""
        self.market_store = MarketDataStore()
        self.memory_db = MemoryDatabase()
        self.models = {
            "yield_predictor": YieldPredictor(),
            "risk_assessor": RiskAssessor()
        }
    
    def check_if_retraining_needed(self, model_name):
        """Check if a model needs retraining"""
        model = self.models.get(model_name)
        if not model:
            logger.error(f"Model {model_name} not found")
            return True
        
        # Try to load the model
        loaded = model.load()
        if not loaded:
            # No saved model found, needs training
            logger.info(f"No saved model found for {model_name}, will train new model")
            return True
        
        # Check when the model was last trained
        last_trained = model.metadata.get("last_trained")
        if not last_trained:
            return True
            
        # Parse the timestamp
        try:
            last_trained_dt = datetime.fromisoformat(last_trained)
            days_since_training = (datetime.now() - last_trained_dt).days
            
            if days_since_training >= RETRAINING_INTERVAL_DAYS:
                logger.info(f"Model {model_name} was trained {days_since_training} days ago, will retrain")
                return True
        except Exception as e:
            logger.error(f"Error parsing last trained date: {e}")
            return True
        
        # Model is recent enough
        return False
    
    def train_all_models(self):
        """Train or update all models"""
        results = {}
        
        # Train yield predictor if needed
        if self.check_if_retraining_needed("yield_predictor"):
            try:
                model = self.models["yield_predictor"]
                X, y = model.prepare_training_data(self.market_store)
                
                if X is not None and y is not None:
                    performance = model.fit(X, y)
                    model.save()
                    results["yield_predictor"] = {
                        "status": "trained",
                        "samples": len(X),
                        "performance": performance
                    }
                else:
                    results["yield_predictor"] = {
                        "status": "skipped",
                        "reason": "insufficient data"
                    }
            except Exception as e:
                logger.error(f"Error training yield predictor: {e}")
                results["yield_predictor"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            results["yield_predictor"] = {
                "status": "skipped",
                "reason": "recent model available"
            }
        
        # Train risk assessor if needed
        if self.check_if_retraining_needed("risk_assessor"):
            try:
                model = self.models["risk_assessor"]
                X, y = model.prepare_training_data(self.market_store, self.memory_db)
                
                if X is not None and y is not None:
                    performance = model.fit(X, y)
                    model.save()
                    results["risk_assessor"] = {
                        "status": "trained",
                        "samples": len(X),
                        "performance": performance
                    }
                else:
                    results["risk_assessor"] = {
                        "status": "skipped",
                        "reason": "insufficient data"
                    }
            except Exception as e:
                logger.error(f"Error training risk assessor: {e}")
                results["risk_assessor"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            results["risk_assessor"] = {
                "status": "skipped",
                "reason": "recent model available"
            }
        
        # Save training results to memory for tracking
        self.memory_db.add_learning(
            concept="model_training",
            understanding=json.dumps(results),
            confidence=1.0,
            source="ModelTrainer"
        )
        
        return results


def run_model_training():
    """Run model training as a standalone process"""
    logger.info("Starting model training process...")
    
    trainer = ModelTrainer()
    results = trainer.train_all_models()
    
    # Print results
    print(json.dumps(results, indent=2))
    
    return results


def run_portfolio_optimization(opportunities, total_investment=100.0):
    """Run portfolio optimization as a standalone process"""
    logger.info("Starting portfolio optimization process...")
    
    optimizer = PortfolioOptimizer()
    allocations = optimizer.optimize_portfolio(opportunities, total_investment)
    
    # Print results
    print(json.dumps(allocations, indent=2))
    
    return allocations


if __name__ == "__main__":
    # Run model training
    run_model_training() 