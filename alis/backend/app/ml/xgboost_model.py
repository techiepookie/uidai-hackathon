"""
XGBoost-based time series forecasting model for ALIS.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
from datetime import datetime, timedelta

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed, XGBoostForecaster will use fallback")


class XGBoostForecaster:
    """
    XGBoost model for time series forecasting.
    
    Uses feature engineering to convert time series into supervised learning:
    - Lagged values (t-1, t-2, ..., t-n)
    - Rolling statistics (mean, std)
    - Day of week features
    - Trend features
    """
    
    def __init__(
        self,
        n_lags: int = 14,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1
    ):
        self.n_lags = n_lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.last_values = None
        self.metadata = {}
    
    def _create_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features from time series data.
        
        Returns:
            Tuple of (X, y) arrays
        """
        n = len(data)
        if n <= self.n_lags:
            raise ValueError(f"Need at least {self.n_lags + 1} data points")
        
        features = []
        targets = []
        
        for i in range(self.n_lags, n):
            # Lagged values
            lags = data[i-self.n_lags:i]
            
            # Rolling statistics
            rolling_mean_7 = np.mean(data[max(0, i-7):i])
            rolling_std_7 = np.std(data[max(0, i-7):i])
            rolling_mean_14 = np.mean(data[max(0, i-14):i])
            
            # Trend (difference from n days ago)
            trend_7 = data[i-1] - data[max(0, i-7)] if i >= 7 else 0
            
            # Day indicator (position in week, assuming daily data)
            day_of_week = i % 7
            
            # Combine features
            feature_row = np.concatenate([
                lags,
                [rolling_mean_7, rolling_std_7, rolling_mean_14],
                [trend_7, day_of_week]
            ])
            
            features.append(feature_row)
            targets.append(data[i])
        
        return np.array(features), np.array(targets)
    
    def _create_prediction_features(
        self, 
        last_values: np.ndarray,
        predictions: List[float],
        step: int
    ) -> np.ndarray:
        """
        Create features for prediction step.
        """
        # Combine historical values with predictions made so far
        combined = np.concatenate([last_values, predictions])
        
        # Take the last n_lags values
        lags = combined[-self.n_lags:]
        
        # Rolling statistics from combined data
        last_14 = combined[-14:] if len(combined) >= 14 else combined
        last_7 = combined[-7:] if len(combined) >= 7 else combined
        
        rolling_mean_7 = np.mean(last_7)
        rolling_std_7 = np.std(last_7) if len(last_7) > 1 else 0
        rolling_mean_14 = np.mean(last_14)
        
        # Trend
        trend_7 = combined[-1] - combined[-7] if len(combined) >= 7 else 0
        
        # Day of week
        day_of_week = (len(last_values) + step) % 7
        
        feature_row = np.concatenate([
            lags,
            [rolling_mean_7, rolling_std_7, rolling_mean_14],
            [trend_7, day_of_week]
        ])
        
        return feature_row.reshape(1, -1)
    
    def fit(self, data: np.ndarray) -> 'XGBoostForecaster':
        """
        Fit XGBoost model on time series data.
        
        Args:
            data: Time series values (1D array)
            
        Returns:
            Self for chaining
        """
        if not XGB_AVAILABLE:
            logger.warning("XGBoost not available, fit will be no-op")
            self.last_values = data[-self.n_lags:] if len(data) >= self.n_lags else data
            return self
        
        if len(data) <= self.n_lags:
            logger.warning(f"Insufficient data (n={len(data)}, need > {self.n_lags})")
            self.last_values = data
            return self
        
        try:
            # Create features
            X, y = self._create_features(data)
            
            # Train XGBoost
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                verbosity=0
            )
            
            self.model.fit(X, y)
            
            # Store last values for prediction
            self.last_values = data[-self.n_lags:]
            
            # Calculate in-sample metrics
            y_pred = self.model.predict(X)
            mae = np.mean(np.abs(y - y_pred))
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            
            self.metadata = {
                'n_samples': len(data),
                'n_features': X.shape[1],
                'in_sample_mae': float(mae),
                'in_sample_rmse': float(rmse),
                'fit_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"XGBoost fitted: MAE={mae:.2f}, RMSE={rmse:.2f}")
            return self
            
        except Exception as e:
            logger.error(f"XGBoost fit failed: {e}")
            self.last_values = data[-self.n_lags:] if len(data) >= self.n_lags else data
            return self
    
    def predict(self, horizon: int = 30) -> np.ndarray:
        """
        Generate predictions for future steps.
        
        Args:
            horizon: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        if self.model is None or not XGB_AVAILABLE:
            # Fallback: simple moving average
            if self.last_values is not None and len(self.last_values) > 0:
                return np.full(horizon, np.mean(self.last_values))
            return np.zeros(horizon)
        
        try:
            predictions = []
            
            for step in range(horizon):
                # Create features for this step
                X = self._create_prediction_features(
                    self.last_values,
                    predictions,
                    step
                )
                
                # Predict
                pred = self.model.predict(X)[0]
                pred = max(0, pred)  # Ensure non-negative
                predictions.append(pred)
            
            return np.array(predictions)
            
        except Exception as e:
            logger.error(f"XGBoost predict failed: {e}")
            if self.last_values is not None and len(self.last_values) > 0:
                return np.full(horizon, np.mean(self.last_values))
            return np.zeros(horizon)
    
    def evaluate(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        """
        if len(test_data) == 0:
            return {}
        
        # Fit on training data
        self.fit(train_data)
        
        # Predict
        predictions = self.predict(len(test_data))
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - test_data))
        rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
        mape = np.mean(np.abs((predictions - test_data) / (test_data + 1e-10))) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        import joblib
        
        model_data = {
            'model': self.model,
            'last_values': self.last_values,
            'n_lags': self.n_lags,
            'metadata': self.metadata
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"XGBoost model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBoostForecaster':
        """Load model from file."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        instance = cls(n_lags=model_data['n_lags'])
        instance.model = model_data['model']
        instance.last_values = model_data['last_values']
        instance.metadata = model_data['metadata']
        
        logger.info(f"XGBoost model loaded from {filepath}")
        return instance


# Test
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 7)
    trend = 0.1 * t
    noise = np.random.normal(0, 2, n)
    data = 50 + trend + seasonal + noise
    
    # Create and fit model
    model = XGBoostForecaster(n_lags=14)
    model.fit(data)
    
    # Generate predictions
    predictions = model.predict(14)
    print(f"XGBoost Predictions (14 days): {predictions}")
    print(f"Metadata: {model.metadata}")
    
    # Evaluate
    train, test = data[:80], data[80:]
    metrics = model.evaluate(train, test)
    print(f"Evaluation: {metrics}")
