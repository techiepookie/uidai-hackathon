"""
Model training pipeline for ALIS.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
import joblib

from app.ml.sarima_model import SARIMAModel
from app.ml.xgboost_model import XGBoostForecaster
from app.ml.ensemble import EnsembleForecaster
from app.config import settings
from app.database import get_db_context
from app.models.db_models import RawUpdate


class ModelTrainer:
    """
    Orchestrates training of all ML models.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_results = {}
    
    def get_training_data(self, days: int = 90) -> pd.DataFrame:
        """
        Fetch training data from database.
        """
        with get_db_context() as db:
            from sqlalchemy import func
            from datetime import date, timedelta
            
            cutoff = date.today() - timedelta(days=days)
            
            # Aggregate by date
            data = db.query(
                RawUpdate.date,
                func.sum(RawUpdate.bio_total).label('bio_total'),
                func.sum(RawUpdate.demo_total).label('demo_total'),
                func.sum(RawUpdate.mobile_updates).label('mobile_updates')
            ).filter(
                RawUpdate.date >= cutoff
            ).group_by(
                RawUpdate.date
            ).order_by(RawUpdate.date).all()
            
            df = pd.DataFrame([
                {
                    'date': d.date,
                    'bio_total': d.bio_total or 0,
                    'demo_total': d.demo_total or 0,
                    'mobile_updates': d.mobile_updates or 0
                }
                for d in data
            ])
            
            return df
    
    def train_sarima(
        self,
        data: np.ndarray,
        metric_name: str = 'bio'
    ) -> Dict[str, Any]:
        """
        Train SARIMA model.
        """
        logger.info(f"Training SARIMA model for {metric_name}")
        
        try:
            model = SARIMAModel()
            model.fit(data, auto_order=True)
            
            # Save model
            model_path = self.models_dir / f"sarima_{metric_name}.pkl"
            model.save(str(model_path))
            
            result = {
                'status': 'success',
                'model_path': str(model_path),
                'metadata': model.metadata,
                'order': model.order,
                'seasonal_order': model.seasonal_order
            }
            
            logger.info(f"SARIMA model saved to {model_path}")
            return result
            
        except Exception as e:
            logger.error(f"SARIMA training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def train_xgboost(
        self,
        data: np.ndarray,
        metric_name: str = 'bio'
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.
        """
        logger.info(f"Training XGBoost model for {metric_name}")
        
        try:
            model = XGBoostForecaster()
            model.fit(data)
            
            # Save model
            model_path = self.models_dir / f"xgboost_{metric_name}.pkl"
            model.save(str(model_path))
            
            result = {
                'status': 'success',
                'model_path': str(model_path),
                'metadata': model.metadata
            }
            
            logger.info(f"XGBoost model saved to {model_path}")
            return result
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def train_all(self) -> Dict[str, Any]:
        """
        Train all models for all metrics.
        """
        logger.info("Starting full model training pipeline")
        
        # Get training data
        df = self.get_training_data(days=90)
        
        if len(df) < 30:
            logger.warning("Insufficient training data")
            return {
                'status': 'insufficient_data',
                'available_samples': len(df)
            }
        
        results = {
            'training_timestamp': datetime.now().isoformat(),
            'samples': len(df),
            'models': {}
        }
        
        # Train for each metric
        metrics = ['bio_total', 'demo_total', 'mobile_updates']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            data = df[metric].values
            metric_name = metric.replace('_total', '').replace('_updates', '')
            
            results['models'][metric_name] = {
                'sarima': self.train_sarima(data, metric_name),
                'xgboost': self.train_xgboost(data, metric_name)
            }
        
        # Save training results
        results_path = self.models_dir / "training_results.json"
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Training complete. Results saved to {results_path}")
        return results
    
    def evaluate_models(self, test_days: int = 7) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        """
        logger.info("Evaluating models")
        
        df = self.get_training_data(days=90)
        
        if len(df) < 30:
            return {'error': 'Insufficient data'}
        
        train_size = len(df) - test_days
        
        results = {}
        metrics = ['bio_total', 'demo_total', 'mobile_updates']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            data = df[metric].values
            train, test = data[:train_size], data[train_size:]
            metric_name = metric.replace('_total', '').replace('_updates', '')
            
            results[metric_name] = {}
            
            # Evaluate SARIMA
            try:
                sarima = SARIMAModel()
                results[metric_name]['sarima'] = sarima.evaluate(train, test)
            except Exception as e:
                results[metric_name]['sarima'] = {'error': str(e)}
            
            # Evaluate XGBoost
            try:
                xgb = XGBoostForecaster()
                results[metric_name]['xgboost'] = xgb.evaluate(train, test)
            except Exception as e:
                results[metric_name]['xgboost'] = {'error': str(e)}
            
            # Evaluate Ensemble
            try:
                ensemble = EnsembleForecaster()
                ensemble.fit(train)
                preds = ensemble.predict(len(test))
                mae = np.mean(np.abs(preds - test))
                rmse = np.sqrt(np.mean((preds - test) ** 2))
                results[metric_name]['ensemble'] = {
                    'mae': float(mae),
                    'rmse': float(rmse)
                }
            except Exception as e:
                results[metric_name]['ensemble'] = {'error': str(e)}
        
        return results


def train_all_models():
    """Convenience function to train all models."""
    trainer = ModelTrainer()
    return trainer.train_all()


if __name__ == "__main__":
    # Run training
    results = train_all_models()
    print("Training Results:")
    import json
    print(json.dumps(results, indent=2, default=str))
