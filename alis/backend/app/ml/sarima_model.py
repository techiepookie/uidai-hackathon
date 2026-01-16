"""
SARIMA time series forecasting model for ALIS.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
from loguru import logger
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')


class SARIMAModel:
    """
    Seasonal ARIMA model for time series forecasting.
    
    Best for:
    - Capturing seasonality (weekly/monthly patterns)
    - Short-term forecasting (7-30 days)
    - Univariate time series
    """
    
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        trend: str = 'n'
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.model = None
        self.fitted = None
        self.metadata = {}
    
    def auto_order(self, data: np.ndarray) -> Tuple[Tuple, Tuple]:
        """
        Automatically determine SARIMA order based on data characteristics.
        
        Args:
            data: Time series data (numpy array)
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        n = len(data)
        
        # Check stationarity using Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(data, autolag='AIC')
            is_stationary = adf_result[1] <= 0.05
        except:
            is_stationary = False
        
        # Determine differencing order (d)
        d = 0 if is_stationary else 1
        
        # Check for seasonality
        seasonal_strength = 0
        if n >= 14:  # Minimum for weekly seasonality detection
            try:
                # Use shorter period for computational efficiency
                period = min(7, n // 2) if n >= 14 else 7
                decomp = seasonal_decompose(data, model='additive', period=period)
                seasonal_strength = np.std(decomp.seasonal) / (np.std(data) + 1e-10)
            except Exception as e:
                logger.warning(f"Seasonality decomposition failed: {e}")
                seasonal_strength = 0
        
        # Determine orders based on data characteristics
        if seasonal_strength > 0.15 and n >= 21:  # Strong weekly seasonality
            order = (1, d, 1)
            seasonal_order = (1, 1, 1, 7)
            logger.info(f"Strong seasonality detected (strength: {seasonal_strength:.2f})")
        elif seasonal_strength > 0.05 and n >= 14:  # Weak seasonality
            order = (1, d, 1)
            seasonal_order = (0, 1, 1, 7)
            logger.info(f"Weak seasonality detected (strength: {seasonal_strength:.2f})")
        else:  # No seasonality
            order = (1, d, 1)
            seasonal_order = (0, 0, 0, 0)
            logger.info(f"No significant seasonality (strength: {seasonal_strength:.2f})")
        
        # Adjust for very short series
        if n < 10:
            order = (0, d, 0)
            seasonal_order = (0, 0, 0, 0)
            logger.warning(f"Short series (n={n}), using simple model")
        
        self.metadata.update({
            'auto_order': True,
            'stationary': is_stationary,
            'seasonal_strength': seasonal_strength,
            'differencing_order': d,
            'n_samples': n
        })
        
        return order, seasonal_order
    
    def fit(
        self,
        data: np.ndarray,
        auto_order: bool = True,
        maxiter: int = 50
    ) -> 'SARIMAModel':
        """
        Fit SARIMA model to data.
        
        Args:
            data: Time series values (1D array)
            auto_order: Whether to auto-determine order
            maxiter: Maximum iterations for optimization
            
        Returns:
            Self for chaining
        """
        if len(data) < 5:
            logger.error("Insufficient data for SARIMA modeling (need at least 5 points)")
            raise ValueError("Data must have at least 5 observations")
        
        if auto_order:
            self.order, self.seasonal_order = self.auto_order(data)
        
        try:
            logger.info(f"Fitting SARIMA model with order={self.order}, seasonal_order={self.seasonal_order}")
            
            # Fit SARIMAX model
            self.model = SARIMAX(
                data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted = self.model.fit(
                maxiter=maxiter,
                disp=False,
                method='lbfgs'
            )
            
            # Store metadata
            self.metadata.update({
                'fit_timestamp': datetime.now().isoformat(),
                'aic': float(self.fitted.aic),
                'bic': float(self.fitted.bic),
                'hqic': float(self.fitted.hqic),
                'nobs': int(self.fitted.nobs),
                'converged': self.fitted.mle_retvals['converged'] if hasattr(self.fitted, 'mle_retvals') else True
            })
            
            logger.info(f"Model fitted successfully. AIC: {self.fitted.aic:.2f}")
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit SARIMA model: {e}")
            # Fallback to simple moving average
            logger.info("Falling back to simple moving average model")
            self.order = (0, 0, 0)
            self.seasonal_order = (0, 0, 0, 0)
            self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted = self.model.fit(disp=False)
            return self
    
    def predict(
        self,
        steps: int,
        alpha: float = 0.05,
        return_conf_int: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate predictions with confidence intervals.
        
        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals
            return_conf_int: Whether to return confidence intervals
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        if self.fitted is None:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Get forecast
            forecast = self.fitted.get_forecast(steps=steps)
            
            # Extract predictions
            predictions = forecast.predicted_mean
            
            # Extract confidence intervals if requested
            conf_int = None
            if return_conf_int:
                conf_int = forecast.conf_int(alpha=alpha)
                # Convert to numpy array for consistency
                conf_int = np.column_stack([
                    conf_int.iloc[:, 0].values,
                    conf_int.iloc[:, 1].values
                ])
            
            logger.info(f"Generated {steps}-step forecast")
            return predictions.values, conf_int
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback: Simple exponential smoothing
            logger.info("Using fallback prediction method")
            predictions = np.full(steps, np.mean(self.fitted.data.endog[-7:]))
            conf_int = None
            if return_conf_int:
                std = np.std(self.fitted.data.endog[-7:])
                conf_int = np.column_stack([
                    predictions - 1.96 * std,
                    predictions + 1.96 * std
                ])
            return predictions, conf_int
    
    def forecast(
        self,
        data: np.ndarray,
        forecast_horizon: int = 30
    ) -> Dict[str, Any]:
        """
        Complete forecasting pipeline with evaluation.
        
        Args:
            data: Historical time series data
            forecast_horizon: Number of days to forecast
            
        Returns:
            Dictionary with forecast results and metrics
        """
        if len(data) < 10:
            logger.warning("Insufficient data for reliable forecast")
            return self._simple_forecast(data, forecast_horizon)
        
        # Fit model
        self.fit(data, auto_order=True)
        
        # Generate forecast
        predictions, conf_int = self.predict(forecast_horizon, return_conf_int=True)
        
        # Calculate forecast dates
        last_date = pd.Timestamp.now() if len(data) == 0 else pd.Timestamp.now() - pd.Timedelta(days=len(data))
        forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_horizon)]
        
        # Prepare results
        result = {
            'forecast': predictions.tolist(),
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'confidence_intervals': conf_int.tolist() if conf_int is not None else None,
            'metadata': self.metadata,
            'model_params': {
                'order': self.order,
                'seasonal_order': self.seasonal_order
            },
            'metrics': self._calculate_metrics(data)
        }
        
        logger.info(f"Forecast generated for {forecast_horizon} days")
        return result
    
    def evaluate(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            train_data: Training data
            test_data: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(test_data) == 0:
            return {}
        
        # Fit on training data
        self.fit(train_data, auto_order=True)
        
        # Predict on test data
        predictions, _ = self.predict(len(test_data), return_conf_int=False)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - test_data))
        rmse = np.sqrt(np.mean((predictions - test_data) ** 2))
        mape = np.mean(np.abs((predictions - test_data) / (test_data + 1e-10))) * 100
        
        # Directional accuracy
        actual_diff = np.diff(test_data)
        pred_diff = np.diff(predictions)
        directional_accuracy = np.mean((actual_diff * pred_diff) > 0)
        
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'directional_accuracy': float(directional_accuracy)
        }
        
        logger.info(f"Evaluation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
        return metrics
    
    def _calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate in-sample metrics."""
        if self.fitted is None:
            return {}
        
        try:
            # Get fitted values
            fitted_values = self.fitted.fittedvalues
            
            # Calculate metrics
            residuals = data[len(data)-len(fitted_values):] - fitted_values
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            return {
                'in_sample_mae': float(mae),
                'in_sample_rmse': float(rmse),
                'aic': float(self.fitted.aic),
                'bic': float(self.fitted.bic)
            }
        except:
            return {}
    
    def _simple_forecast(
        self,
        data: np.ndarray,
        forecast_horizon: int
    ) -> Dict[str, Any]:
        """Fallback forecasting method for very short series."""
        if len(data) == 0:
            forecast = np.zeros(forecast_horizon)
        else:
            # Use last 7 days average
            recent = data[-7:] if len(data) >= 7 else data
            forecast = np.full(forecast_horizon, np.mean(recent))
        
        # Simple confidence intervals
        std = np.std(recent) if len(recent) > 1 else 1.0
        conf_int = np.column_stack([
            forecast - 1.96 * std,
            forecast + 1.96 * std
        ])
        
        # Calculate dates
        forecast_dates = [datetime.now() + timedelta(days=i+1) for i in range(forecast_horizon)]
        
        return {
            'forecast': forecast.tolist(),
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'confidence_intervals': conf_int.tolist(),
            'metadata': {'method': 'fallback', 'n_samples': len(data)},
            'model_params': {'order': (0, 0, 0), 'seasonal_order': (0, 0, 0, 0)},
            'metrics': {'mae': float(std), 'rmse': float(std)}
        }
    
    def save(self, filepath: str):
        """Save model to file."""
        if self.fitted is None:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        
        model_data = {
            'model': self.model,
            'fitted': self.fitted,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'metadata': self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SARIMAModel':
        """Load model from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(
            order=model_data['order'],
            seasonal_order=model_data['seasonal_order']
        )
        instance.model = model_data['model']
        instance.fitted = model_data['fitted']
        instance.metadata = model_data['metadata']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data with seasonality
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / 7)
    trend = 0.1 * t
    noise = np.random.normal(0, 2, n)
    data = 50 + trend + seasonal + noise
    
    # Create and fit model
    model = SARIMAModel()
    
    # Test auto_order
    order, seasonal_order = model.auto_order(data)
    print(f"Auto-detected orders: order={order}, seasonal_order={seasonal_order}")
    
    # Fit model
    model.fit(data, auto_order=True)
    
    # Generate forecast
    forecast_result = model.forecast(data, forecast_horizon=14)
    
    print("\nForecast Results:")
    print(f"Next 7 days: {forecast_result['forecast'][:7]}")
    print(f"Metrics: {forecast_result['metrics']}")
    
    # Test evaluation
    train = data[:80]
    test = data[80:]
    eval_metrics = model.evaluate(train, test)
    print(f"\nEvaluation Metrics: {eval_metrics}")
    
    # Test save/load
    model.save("test_sarima_model.pkl")
    loaded_model = SARIMAModel.load("test_sarima_model.pkl")
    print(f"\nModel loaded successfully: {loaded_model.metadata['fit_timestamp']}")