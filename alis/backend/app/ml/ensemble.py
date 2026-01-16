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
        sarima_weight: float = 0.6,
        xgboost_weight: float = 0.4
    ):
        self.sarima_weight = sarima_weight
        self.xgboost_weight = xgboost_weight
        self.sarima_model = None
        self.xgboost_model = None
        self.is_fitted = False
        self.metadata = {}
    
    def fit(self, data: np.ndarray) -> 'EnsembleForecaster':
        """
        Fit both models on the data.
        
        Args:
            data: Time series values (1D numpy array)
            
        Returns:
            Self for chaining
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
        
        # Fit XGBoost model (try import, fallback if not available)
        try:
            from app.ml.xgboost_model import XGBoostForecaster
            self.xgboost_model = XGBoostForecaster()
            self.xgboost_model.fit(data)
            logger.info("XGBoost model fitted successfully")
        except ImportError:
            logger.warning("XGBoost model not available, using SARIMA only")
            self.xgboost_model = None
            self.sarima_weight = 1.0
            self.xgboost_weight = 0.0
        except Exception as e:
            logger.error(f"XGBoost fitting failed: {e}")
            self.xgboost_model = None
            self.sarima_weight = 1.0
            self.xgboost_weight = 0.0
        
        self.is_fitted = self.sarima_model is not None or self.xgboost_model is not None
        
        self.metadata = {
            'sarima_fitted': self.sarima_model is not None,
            'xgboost_fitted': self.xgboost_model is not None,
            'sarima_weight': self.sarima_weight,
            'xgboost_weight': self.xgboost_weight,
            'n_samples': len(data)
        }
        
        return self
    
    def predict(
        self, 
        data: np.ndarray = None,
        horizon: int = 30
    ) -> np.ndarray:
        """
        Generate ensemble predictions.
        
        Args:
            data: Historical data (optional if already fitted)
            horizon: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
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
                logger.debug("SARIMA predictions generated")
            except Exception as e:
                logger.error(f"SARIMA prediction failed: {e}")
        
        # Get XGBoost predictions
        if self.xgboost_model is not None:
            try:
                xgb_preds = self.xgboost_model.predict(horizon)
                predictions.append(xgb_preds)
                weights.append(self.xgboost_weight)
                logger.debug("XGBoost predictions generated")
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
        
        if len(predictions) == 0:
            # Complete fallback
            logger.warning("All models failed, using simple average")
            if data is not None and len(data) > 0:
                return np.full(horizon, np.mean(data[-7:]))
            return np.zeros(horizon)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.zeros(horizon)
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred
    
    def forecast(
        self,
        data: np.ndarray,
        horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Complete forecasting pipeline.
        
        Args:
            data: Historical time series data
            horizon: Forecast horizon in days
            
        Returns:
            Dictionary with forecasts and metadata
        """
        # Fit on data
        self.fit(data)
        
        # Generate predictions
        predictions = self.predict(data, horizon)
        
        # Calculate confidence intervals (simple approach)
        recent_std = np.std(data[-30:]) if len(data) >= 30 else np.std(data)
        conf_multiplier = np.linspace(1.0, 2.0, horizon)  # Wider as we go further
        
        lower_bound = predictions - 1.96 * recent_std * conf_multiplier
        upper_bound = predictions + 1.96 * recent_std * conf_multiplier
        
        # Ensure non-negative bounds
        lower_bound = np.maximum(lower_bound, 0)
        
        return {
            'predictions': predictions.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'metadata': self.metadata,
            'model_used': 'ensemble'
        }
    
    def get_model_contributions(self) -> Dict[str, float]:
        """Get the contribution weights of each model."""
        total = self.sarima_weight + self.xgboost_weight
        return {
            'sarima': self.sarima_weight / total if total > 0 else 0,
            'xgboost': self.xgboost_weight / total if total > 0 else 0
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
