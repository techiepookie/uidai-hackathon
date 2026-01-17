"""
Anomaly detection service using multiple methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional, Any
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from app.models.db_models import RawUpdate, PincodeMetric, Anomaly
from app.config import settings


class AnomalyDetectorService:
    """
    Multi-method anomaly detection service.
    
    Methods:
    1. Z-Score (Statistical)
    2. IQR (Interquartile Range)
    3. Isolation Forest (ML-based)
    4. Seasonal Decomposition
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.z_score_threshold = 3.0
        self.iqr_multiplier = 1.5
        self.isolation_contamination = 0.05
        self.min_data_points = 14
    
    def detect_z_score_anomalies(
        self, 
        values: np.ndarray, 
        threshold: float = None
    ) -> np.ndarray:
        """
        Detect anomalies using Z-Score method.
        Points more than threshold std devs from mean are anomalies.
        """
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool)
        
        threshold = threshold or self.z_score_threshold
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return np.zeros(len(values), dtype=bool)
        
        z_scores = np.abs((values - mean) / std)
        
        return z_scores > threshold
    
    def detect_iqr_anomalies(
        self, 
        values: np.ndarray,
        multiplier: float = None
    ) -> np.ndarray:
        """
        Detect anomalies using IQR method.
        Points outside Q1-1.5*IQR to Q3+1.5*IQR are anomalies.
        """
        if len(values) < 4:
            return np.zeros(len(values), dtype=bool)
        
        multiplier = multiplier or self.iqr_multiplier
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        return (values < lower_bound) | (values > upper_bound)
    
    def detect_isolation_forest_anomalies(
        self, 
        values: np.ndarray,
        contamination: float = None
    ) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest.
        Unsupervised ML method for anomaly detection.
        """
        if len(values) < 10:
            return np.zeros(len(values), dtype=bool)
        
        contamination = contamination or self.isolation_contamination
        
        try:
            # Reshape for sklearn
            X = values.reshape(-1, 1)
            
            # Fit Isolation Forest
            clf = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                n_jobs=1  # Single thread to avoid parallel issues
            )
            
            predictions = clf.fit_predict(X)
            
            # -1 indicates anomaly
            return predictions == -1
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
            return np.zeros(len(values), dtype=bool)
    
    def detect_rolling_anomalies(
        self,
        values: np.ndarray,
        window: int = 7,
        std_multiplier: float = 2.5
    ) -> np.ndarray:
        """
        Detect anomalies using rolling statistics.
        Compares each point to rolling mean/std.
        """
        if len(values) < window + 1:
            return np.zeros(len(values), dtype=bool)
        
        series = pd.Series(values)
        
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        
        # Calculate deviation from rolling mean
        deviation = np.abs(series - rolling_mean)
        threshold = std_multiplier * rolling_std
        
        # Avoid division by zero
        threshold = threshold.replace(0, np.inf)
        
        return (deviation > threshold).values
    
    def calculate_anomaly_confidence(
        self,
        methods_flagged: List[str],
        deviation_percent: float
    ) -> float:
        """
        Calculate confidence score for anomaly detection.
        
        Based on:
        - Number of methods that agree
        - Magnitude of deviation
        """
        # Method agreement score (0-0.6)
        method_score = len(methods_flagged) / 4 * 0.6
        
        # Deviation magnitude score (0-0.4)
        deviation_score = min(abs(deviation_percent) / 200, 1.0) * 0.4
        
        return round(method_score + deviation_score, 2)
    
    def classify_anomaly_severity(
        self,
        confidence: float,
        deviation_percent: float
    ) -> str:
        """Classify anomaly severity based on confidence and deviation."""
        if confidence >= 0.8 and abs(deviation_percent) > 100:
            return 'CRITICAL'
        elif confidence >= 0.6 and abs(deviation_percent) > 50:
            return 'HIGH'
        elif confidence >= 0.4:
            return 'MEDIUM'
        return 'LOW'
    
    def detect_anomalies_for_pincode(
        self,
        pincode: str,
        metric_type: str = 'bio',
        days_lookback: int = 90
    ) -> List[Dict]:
        """
        Detect anomalies for a specific pincode.
        
        Args:
            pincode: The pincode to analyze
            metric_type: 'bio', 'demo', or 'mobile'
            days_lookback: Number of days of historical data
        
        Returns:
            List of detected anomalies with details
        """
        cutoff_date = date.today() - timedelta(days=days_lookback)
        
        # Get historical data
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
        
        if len(data) < self.min_data_points:
            return []
        
        dates = [d.date for d in data]
        values = np.array([d.value or 0 for d in data], dtype=float)
        
        # Run all detection methods
        z_anomalies = self.detect_z_score_anomalies(values)
        iqr_anomalies = self.detect_iqr_anomalies(values)
        iso_anomalies = self.detect_isolation_forest_anomalies(values)
        rolling_anomalies = self.detect_rolling_anomalies(values)
        
        # Find consensus anomalies
        detected = []
        mean_value = np.mean(values)
        
        for i, (dt, val) in enumerate(zip(dates, values)):
            methods_flagged = []
            
            if z_anomalies[i]:
                methods_flagged.append('z_score')
            if iqr_anomalies[i]:
                methods_flagged.append('iqr')
            if iso_anomalies[i]:
                methods_flagged.append('isolation_forest')
            if rolling_anomalies[i]:
                methods_flagged.append('rolling')
            
            # Require at least 2 methods to agree
            if len(methods_flagged) >= 2:
                deviation_percent = ((val - mean_value) / mean_value * 100) if mean_value > 0 else 0
                confidence = self.calculate_anomaly_confidence(methods_flagged, deviation_percent)
                
                if confidence >= settings.ANOMALY_CONFIDENCE_THRESHOLD:
                    anomaly_type = 'SPIKE' if val > mean_value else 'DROP'
                    
                    detected.append({
                        'date': dt,
                        'value': val,
                        'expected': mean_value,
                        'deviation_percent': deviation_percent,
                        'anomaly_type': anomaly_type,
                        'methods': methods_flagged,
                        'confidence': confidence,
                        'severity': self.classify_anomaly_severity(confidence, deviation_percent)
                    })
        
        return detected
    
    def run_anomaly_detection(
        self,
        days_lookback: int = 90,
        limit_pincodes: int = None
    ) -> int:
        """
        Run anomaly detection for all pincodes.
        Returns count of anomalies detected.
        """
        logger.info("Starting anomaly detection run")
        
        # Get pincodes to analyze
        query = self.db.query(PincodeMetric.pincode)
        if limit_pincodes:
            query = query.limit(limit_pincodes)
        
        pincodes = [p.pincode for p in query.all()]
        
        total_anomalies = 0
        
        for pincode in pincodes:
            for metric_type in ['bio', 'demo', 'mobile']:
                try:
                    anomalies = self.detect_anomalies_for_pincode(
                        pincode, metric_type, days_lookback
                    )
                    
                    for anomaly_data in anomalies:
                        self._save_anomaly(pincode, metric_type, anomaly_data)
                        total_anomalies += 1
                        
                except Exception as e:
                    logger.error(f"Error detecting anomalies for {pincode}/{metric_type}: {e}")
        
        self.db.commit()
        logger.info(f"Detected {total_anomalies} anomalies")
        
        return total_anomalies
    
    def _save_anomaly(
        self,
        pincode: str,
        metric_type: str,
        anomaly_data: Dict
    ):
        """Save detected anomaly to database."""
        
        # Get pincode metric ID
        pincode_metric = self.db.query(PincodeMetric).filter(
            PincodeMetric.pincode == pincode
        ).first()
        
        if not pincode_metric:
            return
        
        # Check for existing anomaly on same date
        existing = self.db.query(Anomaly).filter(
            Anomaly.pincode_metric_id == pincode_metric.id,
            Anomaly.detected_date == anomaly_data['date'],
            Anomaly.metric_affected == metric_type
        ).first()
        
        if existing:
            return
        
        anomaly = Anomaly(
            pincode_metric_id=pincode_metric.id,
            detected_date=anomaly_data['date'],
            anomaly_type=anomaly_data['anomaly_type'],
            metric_affected=metric_type,
            expected_value=anomaly_data['expected'],
            actual_value=anomaly_data['value'],
            deviation_percent=anomaly_data['deviation_percent'],
            detection_methods=anomaly_data['methods'],
            confidence_score=anomaly_data['confidence'],
            severity=anomaly_data['severity'],
            is_seasonal=False,
            is_investigated=False
        )
        
        self.db.add(anomaly)
    
    def get_recent_anomalies(
        self,
        days: int = 7,
        severity: Optional[str] = None,
        limit: int = 50
    ) -> List[Anomaly]:
        """Get recently detected anomalies."""
        
        cutoff_date = date.today() - timedelta(days=days)
        
        query = self.db.query(Anomaly).filter(
            Anomaly.detected_date >= cutoff_date
        )
        
        if severity:
            query = query.filter(Anomaly.severity == severity)
        
        return query.order_by(
            Anomaly.confidence_score.desc()
        ).limit(limit).all()