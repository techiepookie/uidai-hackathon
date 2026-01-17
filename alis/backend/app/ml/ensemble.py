"""
Ensemble forecaster combining SARIMA and XGBoost models.
"""

import numpy as np
from typing import Optional, Dict, Any, List
from loguru import logger

from app.ml.sarima_model import SARIMAModel


class EnsembleForecaster:
    """
    Ensemble model that combines SARIMA and XGBoost predictions.
    
    Uses weighted averaging with automatic weight adjustment based on
    recent prediction accuracy.
    
    Fallback Strategy:
    - If XGBoost fails: Use SARIMA only
    - If SARIMA fails: Use simple moving average
    """
    
    def __init__(
        self,
        sarima_weight: float = 0.4,
        xgboost_weight: float = 0.3,
        lstm_weight: float = 0.3
    ):
        self.sarima_weight = sarima_weight
        self.xgboost_weight = xgboost_weight
        self.lstm_weight = lstm_weight
        self.sarima_model = None
        self.xgboost_model = None
        self.lstm_model = None
        self.is_fitted = False
        self.metadata = {}
    
    def fit(self, data: np.ndarray) -> 'EnsembleForecaster':
        """
        Fit all models on the data.
        """
        if len(data) < 10:
            logger.warning("Insufficient data for ensemble, using simple model")
            self.is_fitted = False
            return self
        
        # Fit SARIMA model
        try:
            self.sarima_model = SARIMAModel()
            self.sarima_model.fit(data, auto_order=True)
            logger.info("SARIMA model fitted successfully")
        except Exception as e:
            logger.error(f"SARIMA fitting failed: {e}")
            self.sarima_model = None
        
        # Fit XGBoost model
        try:
            from app.ml.xgboost_model import XGBoostForecaster
            self.xgboost_model = XGBoostForecaster()
            self.xgboost_model.fit(data)
            logger.info("XGBoost model fitted successfully")
        except Exception as e:
            logger.warning(f"XGBoost model failed: {e}")
            self.xgboost_model = None
            
        # Fit LSTM model
        try:
            from app.ml.lstm_model import LSTMForecaster
            self.lstm_model = LSTMForecaster()
            self.lstm_model.fit(data)
            logger.info("LSTM model fitted successfully")
        except Exception as e:
            logger.warning(f"LSTM model failed: {e}")
            self.lstm_model = None
        
        # Adjust weights if models failed
        models_status = {
            'sarima': self.sarima_model is not None,
            'xgboost': self.xgboost_model is not None,
            'lstm': self.lstm_model is not None
        }
        
        # Recalculate weights if some failed
        active_weights = 0
        if models_status['sarima']: active_weights += self.sarima_weight
        if models_status['xgboost']: active_weights += self.xgboost_weight
        if models_status['lstm']: active_weights += self.lstm_weight
        
        if active_weights > 0:
            if models_status['sarima']: self.sarima_weight /= active_weights
            else: self.sarima_weight = 0
            
            if models_status['xgboost']: self.xgboost_weight /= active_weights
            else: self.xgboost_weight = 0
            
            if models_status['lstm']: self.lstm_weight /= active_weights
            else: self.lstm_weight = 0
        else:
            # Fallback
            self.sarima_weight = 0
            self.xgboost_weight = 0
            self.lstm_weight = 0
        
        self.is_fitted = any(models_status.values())
        
        self.metadata = {
            'sarima_fitted': models_status['sarima'],
            'xgboost_fitted': models_status['xgboost'],
            'lstm_fitted': models_status['lstm'],
            'sarima_weight': self.sarima_weight,
            'xgboost_weight': self.xgboost_weight,
            'lstm_weight': self.lstm_weight,
            'n_samples': len(data)
        }
        
        return self
    
    def predict(
        self, 
        data: np.ndarray = None,
        horizon: int = 30
    ) -> np.ndarray:
        """Generate ensemble predictions."""
        # Fit if not already fitted
        if not self.is_fitted and data is not None:
            self.fit(data)
        
        if not self.is_fitted:
            logger.warning("No models fitted, using fallback")
            if data is not None and len(data) > 0:
                return np.full(horizon, np.mean(data[-7:]))
            return np.zeros(horizon)
        
        predictions = []
        weights = []
        
        # Get SARIMA predictions
        if self.sarima_model is not None:
            try:
                sarima_preds, _ = self.sarima_model.predict(horizon, return_conf_int=False)
                predictions.append(sarima_preds)
                weights.append(self.sarima_weight)
            except Exception as e:
                logger.error(f"SARIMA prediction failed: {e}")
        
        # Get XGBoost predictions
        if self.xgboost_model is not None:
            try:
                xgb_preds = self.xgboost_model.predict(horizon)
                predictions.append(xgb_preds)
                weights.append(self.xgboost_weight)
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
                
        # Get LSTM predictions
        if self.lstm_model is not None:
            try:
                lstm_preds = self.lstm_model.predict(horizon)
                predictions.append(lstm_preds)
                weights.append(self.lstm_weight)
            except Exception as e:
                logger.error(f"LSTM prediction failed: {e}")
        
        if len(predictions) == 0:
            # Complete fallback
            if data is not None and len(data) > 0:
                return np.full(horizon, np.mean(data[-7:]))
            return np.zeros(horizon)
        
        # Weighted average
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        ensemble_pred = np.zeros(horizon)
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred

    def get_model_contributions(self) -> Dict[str, float]:
        """Get the contribution weights of each model."""
        return {
            'sarima': self.sarima_weight,
            'xgboost': self.xgboost_weight,
            'lstm': self.lstm_weight
        }


# Simple test
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 7)
    trend = 0.1 * t
    noise = np.random.normal(0, 2, n)
    data = 50 + trend + seasonal + noise
    
    # Create ensemble
    ensemble = EnsembleForecaster()
    
    # Generate forecast
    result = ensemble.forecast(data, horizon=14)
    
    print("Ensemble Forecast Results:")
    print(f"Predictions (first 7 days): {result['predictions'][:7]}")
    print(f"Metadata: {result['metadata']}")
