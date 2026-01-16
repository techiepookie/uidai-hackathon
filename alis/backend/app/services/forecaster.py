"""
Forecasting service using ensemble of models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
import joblib
from pathlib import Path
from loguru import logger

from app.models.db_models import RawUpdate, PincodeMetric, Prediction
from app.config import settings
from app.ml.ensemble import EnsembleForecaster


class ForecasterService:
    """
    Service for generating forecasts using ensemble methods.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.ensemble = EnsembleForecaster()
        self.horizon_days = settings.FORECAST_HORIZON_DAYS
        self.min_history_days = 30
    
    def get_historical_data(
        self,
        pincode: str,
        metric_type: str = 'bio',
        days: int = 90
    ) -> pd.DataFrame:
        """
        Get historical time series data for a pincode.
        """
        cutoff_date = date.today() - timedelta(days=days)
        
        column_map = {
            'bio': RawUpdate.bio_total,
            'demo': RawUpdate.demo_total,
            'mobile': RawUpdate.mobile_updates
        }
        
        data = self.db.query(
            RawUpdate.date,
            column_map[metric_type].label('value')
        ).filter(
            RawUpdate.pincode == pincode,
            RawUpdate.date >= cutoff_date
        ).order_by(RawUpdate.date).all()
        
        df = pd.DataFrame([{'date': d.date, 'value': d.value or 0} for d in data])
        
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Resample to fill gaps
            df = df.resample('D').mean()
            df['value'] = df['value'].fillna(method='ffill').fillna(0)
        
        return df
    
    def generate_forecast(
        self,
        pincode: str,
        metric_type: str = 'bio',
        horizon: int = None
    ) -> Dict[str, Any]:
        """
        Generate forecast for a specific pincode and metric.
        
        Returns:
            Dictionary with predictions and metadata
        """
        horizon = horizon or self.horizon_days
        
        # Get historical data
        history = self.get_historical_data(pincode, metric_type)
        
        if len(history) < self.min_history_days:
            logger.warning(f"Insufficient data for {pincode}/{metric_type}")
            return self._generate_fallback_forecast(pincode, metric_type, horizon)
        
        try:
            # Generate predictions
            predictions = self.ensemble.predict(
                history['value'].values,
                horizon=horizon
            )
            
            # Generate dates
            last_date = history.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=horizon,
                freq='D'
            )
            
            # Calculate confidence intervals
            std = np.std(history['value'].values[-30:])  # Use recent volatility
            
            results = {
                'pincode': pincode,
                'metric_type': metric_type,
                'model_used': 'ensemble',
                'forecasts': []
            }
            
            for i, (dt, pred) in enumerate(zip(forecast_dates, predictions)):
                # Wider intervals further into future
                confidence_factor = 1 + (i * 0.05)
                lower = max(0, pred - std * 1.96 * confidence_factor)
                upper = pred + std * 1.96 * confidence_factor
                
                # Confidence decreases over time
                confidence = max(0.5, 0.95 - (i * 0.01))
                
                results['forecasts'].append({
                    'date': dt.date(),
                    'predicted_value': round(pred, 2),
                    'lower_bound': round(lower, 2),
                    'upper_bound': round(upper, 2),
                    'confidence': round(confidence, 2)
                })
            
            # Calculate summary statistics
            pred_values = [f['predicted_value'] for f in results['forecasts']]
            results['summary'] = {
                'mean_prediction': round(np.mean(pred_values), 2),
                'max_prediction': round(np.max(pred_values), 2),
                'min_prediction': round(np.min(pred_values), 2),
                'peak_date': results['forecasts'][np.argmax(pred_values)]['date'],
                'total_predicted': round(sum(pred_values), 2)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Forecast error for {pincode}/{metric_type}: {e}")
            return self._generate_fallback_forecast(pincode, metric_type, horizon)
    
    def _generate_fallback_forecast(
        self,
        pincode: str,
        metric_type: str,
        horizon: int
    ) -> Dict[str, Any]:
        """Generate simple fallback forecast when models fail."""
        
        # Use simple average-based forecast
        history = self.get_historical_data(pincode, metric_type, days=30)
        
        if len(history) > 0:
            base_value = history['value'].mean()
        else:
            base_value = 10  # Default fallback
        
        forecast_dates = pd.date_range(
            start=date.today() + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        results = {
            'pincode': pincode,
            'metric_type': metric_type,
            'model_used': 'fallback_average',
            'forecasts': []
        }
        
        for dt in forecast_dates:
            results['forecasts'].append({
                'date': dt.date(),
                'predicted_value': round(base_value, 2),
                'lower_bound': round(base_value * 0.7, 2),
                'upper_bound': round(base_value * 1.3, 2),
                'confidence': 0.5
            })
        
        results['summary'] = {
            'mean_prediction': round(base_value, 2),
            'max_prediction': round(base_value, 2),
            'min_prediction': round(base_value, 2),
            'peak_date': forecast_dates[0].date(),
            'total_predicted': round(base_value * horizon, 2)
        }
        
        return results
    
    def save_predictions(
        self,
        pincode: str,
        forecast_result: Dict[str, Any]
    ):
        """Save predictions to database."""
        
        pincode_metric = self.db.query(PincodeMetric).filter(
            PincodeMetric.pincode == pincode
        ).first()
        
        if not pincode_metric:
            return
        
        for forecast in forecast_result['forecasts']:
            # Check for existing prediction
            existing = self.db.query(Prediction).filter(
                Prediction.pincode_metric_id == pincode_metric.id,
                Prediction.prediction_date == forecast['date'],
                Prediction.prediction_type == forecast_result['metric_type']
            ).first()
            
            if existing:
                existing.predicted_value = forecast['predicted_value']
                existing.lower_bound = forecast['lower_bound']
                existing.upper_bound = forecast['upper_bound']
                existing.confidence = forecast['confidence']
                existing.model_used = forecast_result['model_used']
            else:
                prediction = Prediction(
                    pincode_metric_id=pincode_metric.id,
                    prediction_date=forecast['date'],
                    prediction_type=forecast_result['metric_type'],
                    predicted_value=forecast['predicted_value'],
                    lower_bound=forecast['lower_bound'],
                    upper_bound=forecast['upper_bound'],
                    confidence=forecast['confidence'],
                    model_used=forecast_result['model_used']
                )
                self.db.add(prediction)
    
    def generate_all_forecasts(
        self,
        limit_pincodes: int = None
    ) -> int:
        """
        Generate forecasts for all pincodes.
        Returns count of forecasts generated.
        """
        logger.info("Starting forecast generation")
        
        query = self.db.query(PincodeMetric.pincode)
        if limit_pincodes:
            query = query.limit(limit_pincodes)
        
        pincodes = [p.pincode for p in query.all()]
        
        total_forecasts = 0
        
        for pincode in pincodes:
            for metric_type in ['bio', 'demo', 'mobile']:
                try:
                    result = self.generate_forecast(pincode, metric_type)
                    self.save_predictions(pincode, result)
                    total_forecasts += len(result['forecasts'])
                except Exception as e:
                    logger.error(f"Error forecasting {pincode}/{metric_type}: {e}")
        
        self.db.commit()
        logger.info(f"Generated {total_forecasts} forecasts")
        
        return total_forecasts
    
    def validate_predictions(
        self,
        days_back: int = 7
    ) -> Dict[str, float]:
        """
        Validate past predictions against actual values.
        Returns validation metrics.
        """
        cutoff_date = date.today() - timedelta(days=days_back)
        
        predictions = self.db.query(Prediction).filter(
            Prediction.prediction_date >= cutoff_date,
            Prediction.prediction_date < date.today()
        ).all()
        
        errors = []
        abs_errors = []
        
        for pred in predictions:
            # Get actual value
            pincode_metric = pred.pincode_metric
            if not pincode_metric:
                continue
            
            column_map = {
                'bio': RawUpdate.bio_total,
                'demo': RawUpdate.demo_total,
                'mobile': RawUpdate.mobile_updates
            }
            
            actual = self.db.query(
                column_map[pred.prediction_type]
            ).filter(
                RawUpdate.pincode == pincode_metric.pincode,
                RawUpdate.date == pred.prediction_date
            ).scalar()
            
            if actual is not None:
                pred.actual_value = actual
                error = pred.predicted_value - actual
                pred.prediction_error = error
                
                errors.append(error)
                abs_errors.append(abs(error))
        
        self.db.commit()
        
        if not errors:
            return {'mae': 0, 'rmse': 0, 'mape': 0}
        
        mae = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(np.array(errors) ** 2))
        
        # Calculate MAPE (avoiding division by zero)
        actuals = [p.actual_value for p in predictions if p.actual_value and p.actual_value > 0]
        preds = [p.predicted_value for p in predictions if p.actual_value and p.actual_value > 0]
        
        if actuals:
            mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / np.array(actuals))) * 100
        else:
            mape = 0
        
        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2),
            'predictions_validated': len(errors)
        }