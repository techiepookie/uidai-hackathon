"""
LSTM (Long Short-Term Memory) Forecaster.
Deep learning model for time series forecasting using TensorFlow/Keras.
"""

import numpy as np
import warnings
from loguru import logger
from typing import Tuple, List, Optional, Dict, Any
import os

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. LSTM model will not work.")
    TF_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler


class LSTMForecaster:
    """
    LSTM-based time series forecaster.
    
    Attributes:
        look_back (int): Number of previous time steps to use as input features.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        units (int): Number of LSTM units (neurons).
    """
    
    def __init__(
        self, 
        look_back: int = 14, 
        epochs: int = 50, 
        batch_size: int = 16,
        units: int = 50
    ):
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.last_sequence = None
    
    def _create_dataset(self, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert an array of values into a dataset matrix."""
        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back):
            a = dataset[i:(i + self.look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.look_back, 0])
        return np.array(dataX), np.array(dataY)
    
    def fit(self, data: np.ndarray):
        """
        Fit the LSTM model to the data.
        
        Args:
            data: 1D numpy array of time series values.
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow is not installed. Cannot fit LSTM model.")
            return self
            
        if len(data) < self.look_back + 10:
            logger.warning(f"Insufficient data for LSTM (min {self.look_back + 10} required).")
            return self
            
        # Reshape & Scale
        data = data.reshape(-1, 1).astype('float32')
        self.scaler.fit(data)
        dataset = self.scaler.transform(data)
        
        # Create dataset
        X, y = self._create_dataset(dataset)
        
        # Reshape input to be [samples, time steps, features]
        # X shape: (samples, look_back) -> (samples, look_back, 1)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build Model
        self.model = Sequential()
        self.model.add(Input(shape=(self.look_back, 1)))
        self.model.add(LSTM(self.units, activation='relu'))
        self.model.add(Dense(1))
        
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        
        # Train
        early_stop = EarlyStopping(
            monitor='loss', 
            patience=5,
            restore_best_weights=True,
            verbose=0
        )
        
        self.model.fit(
            X, y, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=0,
            callbacks=[early_stop],
            shuffle=False
        )
        
        # Calculate in-sample metrics
        y_pred = self.model.predict(X, verbose=0)
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        y_true_inv = self.scaler.inverse_transform(y.reshape(-1, 1))
        
        residuals = y_true_inv - y_pred_inv
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        # AIC calculation
        n = len(y)
        mse = rmse ** 2
        k = self.model.count_params()
        aic = n * np.log(mse) + 2 * k if mse > 0 else float('inf')
        
        self.metadata = {
            'in_sample_mae': float(mae),
            'in_sample_rmse': float(rmse),
            'aic': float(aic),
            'epochs': self.epochs,
            'look_back': self.look_back
        }
        
        self.is_fitted = True
        self.last_sequence = dataset[-self.look_back:]
        logger.info(f"LSTM model fitted. MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        return self
    
    def predict(self, horizon: int = 30) -> np.ndarray:
        """
        Generate future predictions.
        
        Args:
            horizon: Number of days to predict.
            
        Returns:
            1D numpy array of predicted values.
        """
        if not self.is_fitted or self.model is None:
            logger.warning("LSTM model not fitted. returning zeros.")
            return np.zeros(horizon)
        
        predictions = []
        current_sequence = self.last_sequence.copy() # Shape (look_back, 1)
        
        for _ in range(horizon):
            # Reshape for prediction: (1, look_back, 1)
            input_seq = current_sequence.reshape(1, self.look_back, 1)
            
            # Predict next step
            pred_scaled = self.model.predict(input_seq, verbose=0) # Shape (1, 1)
            pred_value = pred_scaled[0, 0]
            
            predictions.append(pred_value)
            
            # Update sequence: remove first, add prediction
            # current_sequence shape is (look_back, 1)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = pred_value
            
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_actual = self.scaler.inverse_transform(predictions)
        
    def save(self, path: str):
        """Save the model to disk."""
        if self.model is None:
            logger.warning("No model to save")
            return
            
        try:
            # Save Keras model
            model_path = path.replace('.pkl', '.keras')
            self.model.save(model_path)
            
            # Save scaler and attributes
            attr_path = path.replace('.pkl', '_attr.pkl')
            import joblib
            joblib.dump({
                'scaler': self.scaler,
                'look_back': self.look_back,
                'units': self.units,
                'is_fitted': self.is_fitted,
                'last_sequence': self.last_sequence
            }, attr_path)
            
            logger.info(f"LSTM model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save LSTM model: {e}")

    def load(self, path: str):
        """Load the model from disk."""
        if not TF_AVAILABLE:
            return
            
        try:
            # Load attributes
            attr_path = path.replace('.pkl', '_attr.pkl')
            import joblib
            if os.path.exists(attr_path):
                attrs = joblib.load(attr_path)
                self.scaler = attrs['scaler']
                self.look_back = attrs['look_back']
                self.units = attrs['units']
                self.is_fitted = attrs['is_fitted']
                self.last_sequence = attrs['last_sequence']
            
            # Load Keras model
            model_path = path.replace('.pkl', '.keras')
            from tensorflow.keras.models import load_model
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logger.info(f"LSTM model loaded from {model_path}")
            else:
                logger.warning(f"LSTM model file not found: {model_path}")
                self.is_fitted = False
                
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            self.is_fitted = False

    def evaluate(self, train: np.ndarray, test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        Returns metrics dictionary.
        """
        try:
            self.fit(train)
            predictions = self.predict(len(test))
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - test))
            rmse = np.sqrt(np.mean((predictions - test) ** 2))
            
            # Calculate AIC (approximate for LSTM)
            # AIC = n * log(MSE) + 2 * k
            # where n = number of samples, k = number of parameters
            n = len(test)
            mse = rmse ** 2
            k = self.model.count_params() if self.model else 0
            
            if mse > 0:
                aic = n * np.log(mse) + 2 * k
            else:
                aic = float('inf')
                
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'aic': float(aic),
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"LSTM evaluation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
