"""
Risk calculation service for ALIS.
Implements 8 risk metrics as specified in PRD.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from loguru import logger

from app.models.db_models import RawUpdate, PincodeMetric, Recommendation
from app.config import settings


class RiskCalculatorService:
    """
    Service for calculating risk scores and metrics.
    
    Risk Metrics:
    1. Child Bio Update Rate (5-17 age group)
    2. Biometric Update Intensity
    3. Mobile Linkage Gap
    4. Demographic Update Rate
    5. Update Volatility
    6. Trend Analysis
    7. Migration Score
    8. Overall Risk Score
    """
    
    # Weights for overall risk calculation
    RISK_WEIGHTS = {
        'bio_risk': 0.35,
        'demo_risk': 0.25,
        'mobile_risk': 0.20,
        'migration_risk': 0.10,
        'volatility_risk': 0.10
    }
    
    # Risk thresholds
    RISK_THRESHOLDS = {
        'CRITICAL': 80,
        'HIGH': 60,
        'MEDIUM': 40,
        'LOW': 0
    }
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_child_bio_rate(self, bio_5_17: int, enrol_5_17: int) -> float:
        """
        Calculate bio update rate for 5-17 age group.
        Higher rate indicates potential authentication issues.
        
        Formula: (bio_5_17 / enrol_5_17) * 100
        Normalized to 0-100 score.
        """
        if enrol_5_17 <= 0:
            return 0.0
        
        rate = (bio_5_17 / enrol_5_17) * 100
        
        # Normalize: Rate above 10% is concerning
        # Score = min(rate * 10, 100)
        score = min(rate * 10, 100)
        return round(score, 2)
    
    def calculate_bio_risk_score(
        self, 
        bio_total: int, 
        enrol_total: int,
        bio_5_17: int,
        enrol_5_17: int
    ) -> float:
        """
        Calculate overall biometric risk score.
        Combines adult and child bio update rates.
        
        Components:
        - Overall bio update rate
        - Child-specific bio update rate (weighted higher)
        """
        if enrol_total <= 0:
            return 0.0
        
        # Overall bio rate (0-100)
        overall_rate = (bio_total / enrol_total) * 100
        overall_score = min(overall_rate * 10, 100)
        
        # Child bio rate (0-100)
        child_score = self.calculate_child_bio_rate(bio_5_17, enrol_5_17)
        
        # Weighted combination (child updates are more critical)
        score = overall_score * 0.4 + child_score * 0.6
        
        return round(score, 2)
    
    def calculate_demo_risk_score(
        self,
        demo_total: int,
        enrol_total: int
    ) -> float:
        """
        Calculate demographic update risk score.
        High demo updates might indicate migration or data quality issues.
        
        Formula: (demo_total / enrol_total) * 100
        """
        if enrol_total <= 0:
            return 0.0
        
        rate = (demo_total / enrol_total) * 100
        
        # Demo updates above 20% of population is notable
        score = min(rate * 5, 100)
        
        return round(score, 2)
    
    def calculate_mobile_linkage_gap(
        self,
        mobile_linked: int,
        enrol_total: int
    ) -> float:
        """
        Calculate mobile linkage gap score.
        Gap indicates potential OTP authentication issues.
        
        Formula: (1 - mobile_linked/enrol_total) * 100
        """
        if enrol_total <= 0:
            return 0.0
        
        linkage_rate = mobile_linked / enrol_total
        gap = (1 - linkage_rate) * 100
        
        return round(min(gap, 100), 2)
    
    def calculate_volatility(self, values: List[float]) -> float:
        """
        Calculate coefficient of variation as volatility measure.
        High volatility indicates inconsistent patterns.
        
        Formula: (std_dev / mean) * 100
        """
        if not values or len(values) < 2:
            return 0.0
        
        arr = np.array(values)
        mean = np.mean(arr)
        
        if mean == 0:
            return 0.0
        
        cv = (np.std(arr) / mean) * 100
        
        # Normalize: CV above 50% is high volatility
        score = min(cv * 2, 100)
        
        return round(score, 2)
    
    def calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """
        Calculate trend direction and strength.
        Uses simple linear regression.
        
        Returns:
            Tuple of (trend_direction, trend_strength)
            trend_direction: 'INCREASING', 'DECREASING', 'STABLE'
            trend_strength: 0-100
        """
        if not values or len(values) < 3:
            return ('STABLE', 0.0)
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine direction
        if slope > 0.5:
            direction = 'INCREASING'
        elif slope < -0.5:
            direction = 'DECREASING'
        else:
            direction = 'STABLE'
        
        # Strength based on R-squared and slope magnitude
        strength = abs(r_squared) * min(abs(slope) * 10, 100)
        
        return (direction, round(strength, 2))
    
    def calculate_migration_score(
        self,
        demo_total: int,
        prev_demo_total: int,
        enrol_total: int
    ) -> float:
        """
        Calculate migration indicator score.
        Sudden demo update surges often indicate migration.
        
        Formula: Based on demographic update growth rate
        """
        if prev_demo_total <= 0 or enrol_total <= 0:
            return 0.0
        
        growth_rate = ((demo_total - prev_demo_total) / prev_demo_total) * 100
        population_rate = (demo_total / enrol_total) * 100
        
        # Combine growth rate and population rate
        # High growth + high rate = likely migration
        score = (growth_rate * 0.6 + population_rate * 0.4)
        score = min(max(score, 0), 100)
        
        return round(score, 2)
    
    def calculate_overall_risk(
        self,
        bio_risk: float,
        demo_risk: float,
        mobile_risk: float,
        migration_risk: float = 0.0,
        volatility_risk: float = 0.0
    ) -> float:
        """
        Calculate weighted overall risk score.
        """
        score = (
            bio_risk * self.RISK_WEIGHTS['bio_risk'] +
            demo_risk * self.RISK_WEIGHTS['demo_risk'] +
            mobile_risk * self.RISK_WEIGHTS['mobile_risk'] +
            migration_risk * self.RISK_WEIGHTS['migration_risk'] +
            volatility_risk * self.RISK_WEIGHTS['volatility_risk']
        )
        
        return round(min(score, 100), 2)
    
    def get_risk_category(self, score: float) -> str:
        """Determine risk category from score."""
        for category, threshold in self.RISK_THRESHOLDS.items():
            if score >= threshold:
                return category
        return 'LOW'
    
    def calculate_data_confidence(
        self,
        days_of_data: int,
        completeness: float,
        recency_days: int
    ) -> float:
        """
        Calculate confidence score for the data.
        
        Factors:
        - Days of historical data (more = better)
        - Data completeness (no gaps)
        - Data recency (fresh = better)
        """
        # Days factor: 90 days is ideal
        days_factor = min(days_of_data / 90, 1.0)
        
        # Completeness factor
        completeness_factor = completeness
        
        # Recency factor: Data older than 7 days loses confidence
        recency_factor = max(1 - (recency_days / 30), 0)
        
        confidence = (
            days_factor * 0.4 +
            completeness_factor * 0.4 +
            recency_factor * 0.2
        )
        
        return round(confidence, 2)
    
    def calculate_all_metrics(self, days_lookback: int = 90) -> int:
        """
        Calculate metrics for all pincodes.
        Returns count of pincodes processed.
        """
        logger.info(f"Calculating metrics for all pincodes (lookback: {days_lookback} days)")
        
        cutoff_date = date.today() - timedelta(days=days_lookback)
        
        # Get all unique pincodes with aggregated data
        # Group only by pincode to avoid duplicates (same pincode in different state/district records)
        pincode_data = self.db.query(
            RawUpdate.pincode,
            func.max(RawUpdate.state).label('state'),
            func.max(RawUpdate.district).label('district'),
            func.sum(RawUpdate.bio_5_17).label('bio_5_17'),
            func.sum(RawUpdate.bio_total).label('bio_total'),
            func.sum(RawUpdate.demo_total).label('demo_total'),
            func.sum(RawUpdate.mobile_updates).label('mobile_updates'),
            func.sum(RawUpdate.enrol_5_17).label('enrol_5_17'),
            func.sum(RawUpdate.enrol_total).label('enrol_total'),
            func.count(RawUpdate.id).label('record_count'),
            func.max(RawUpdate.date).label('last_date')
        ).filter(
            RawUpdate.date >= cutoff_date
        ).group_by(
            RawUpdate.pincode
        ).all()
        
        processed = 0
        
        for row in pincode_data:
            try:
                self._calculate_pincode_metrics(row, days_lookback)
                processed += 1
                if processed % 1000 == 0:
                    self.db.commit()  # Commit periodically
                    logger.info(f"Processed {processed} pincodes...")
            except Exception as e:
                logger.error(f"Error processing pincode {row.pincode}: {e}")
        
        self.db.commit()
        logger.info(f"Processed {processed} pincodes")
        
        return processed
    
    def _calculate_pincode_metrics(self, row, days_lookback: int):
        """Calculate and store metrics for a single pincode."""
        
        # Calculate risk scores
        bio_risk = self.calculate_bio_risk_score(
            row.bio_total or 0,
            row.enrol_total or 0,
            row.bio_5_17 or 0,
            row.enrol_5_17 or 0
        )
        
        demo_risk = self.calculate_demo_risk_score(
            row.demo_total or 0,
            row.enrol_total or 0
        )
        
        mobile_risk = self.calculate_mobile_linkage_gap(
            row.mobile_updates or 0,
            row.enrol_total or 0
        )
        
        # Get historical data for trend and volatility
        historical = self.db.query(
            RawUpdate.bio_total,
            RawUpdate.demo_total
        ).filter(
            RawUpdate.pincode == row.pincode
        ).order_by(RawUpdate.date).all()
        
        bio_values = [h.bio_total for h in historical if h.bio_total]
        demo_values = [h.demo_total for h in historical if h.demo_total]
        
        # Calculate volatility
        bio_volatility = self.calculate_volatility(bio_values)
        demo_volatility = self.calculate_volatility(demo_values)
        
        # Calculate trends
        bio_trend, bio_trend_strength = self.calculate_trend(bio_values)
        demo_trend, demo_trend_strength = self.calculate_trend(demo_values)
        
        # Calculate migration score (compare recent to previous period)
        mid_point = len(demo_values) // 2
        if mid_point > 0:
            recent_demo = sum(demo_values[mid_point:])
            prev_demo = sum(demo_values[:mid_point])
            migration_score = self.calculate_migration_score(
                recent_demo,
                prev_demo,
                row.enrol_total or 1
            )
        else:
            migration_score = 0.0
        
        # Calculate overall risk
        overall_risk = self.calculate_overall_risk(
            bio_risk,
            demo_risk,
            mobile_risk,
            migration_score,
            (bio_volatility + demo_volatility) / 2
        )
        
        # Calculate data confidence
        days_of_data = row.record_count
        completeness = min(row.record_count / days_lookback, 1.0)
        recency = (date.today() - row.last_date).days if row.last_date else 30
        data_confidence = self.calculate_data_confidence(days_of_data, completeness, recency)
        
        # Priority score (risk adjusted by confidence)
        priority_score = overall_risk * data_confidence
        
        # Get risk category
        risk_category = self.get_risk_category(overall_risk)
        
        # Calculate child bio coverage
        child_coverage = 0.0
        if row.enrol_5_17 and row.enrol_5_17 > 0:
            child_coverage = ((row.enrol_5_17 - row.bio_5_17) / row.enrol_5_17) * 100
            child_coverage = max(0, min(child_coverage, 100))
        
        # Update or create metric
        existing = self.db.query(PincodeMetric).filter(
            PincodeMetric.pincode == row.pincode
        ).first()
        
        metric_data = {
            'pincode': row.pincode,
            'state': row.state,
            'district': row.district,
            'bio_risk_score': bio_risk,
            'demo_risk_score': demo_risk,
            'mobile_risk_score': mobile_risk,
            'overall_risk_score': overall_risk,
            'priority_score': priority_score,
            'risk_category': risk_category,
            'total_bio_updates': row.bio_total or 0,
            'total_demo_updates': row.demo_total or 0,
            'total_mobile_updates': row.mobile_updates or 0,
            'total_enrollments': row.enrol_total or 0,
            'bio_update_rate': (row.bio_total / row.enrol_total * 100) if row.enrol_total else 0,
            'demo_update_rate': (row.demo_total / row.enrol_total * 100) if row.enrol_total else 0,
            'mobile_linkage_rate': (row.mobile_updates / row.enrol_total * 100) if row.enrol_total else 0,
            'bio_trend': bio_trend,
            'demo_trend': demo_trend,
            'bio_volatility': bio_volatility,
            'demo_volatility': demo_volatility,
            'child_bio_coverage': child_coverage,
            'child_update_rate': self.calculate_child_bio_rate(row.bio_5_17 or 0, row.enrol_5_17 or 0),
            'migration_score': migration_score,
            'data_confidence': data_confidence,
            'days_of_data': days_of_data,
            'last_update_date': row.last_date,
            'calculated_at': datetime.utcnow()
        }
        
        if existing:
            for key, value in metric_data.items():
                setattr(existing, key, value)
        else:
            metric = PincodeMetric(**metric_data)
            self.db.add(metric)
        
        # Generate recommendation if high risk
        if risk_category in ['CRITICAL', 'HIGH']:
            self._generate_recommendation(row.pincode, metric_data)
    
    def _generate_recommendation(self, pincode: str, metrics: Dict):
        """Generate operational recommendation for high-risk pincode."""
        
        # Check if recent recommendation exists
        existing = self.db.query(Recommendation).filter(
            Recommendation.pincode_metric_id == self.db.query(PincodeMetric.id).filter(
                PincodeMetric.pincode == pincode
            ).scalar()
        ).filter(
            Recommendation.status.in_(['PENDING', 'IN_PROGRESS'])
        ).first()
        
        if existing:
            return
        
        pincode_metric = self.db.query(PincodeMetric).filter(
            PincodeMetric.pincode == pincode
        ).first()
        
        if not pincode_metric:
            return
        
        # Calculate resource needs
        bio_updates = metrics['total_bio_updates']
        operators_needed = max(1, bio_updates // 500)  # 1 operator per 500 updates
        estimated_budget = operators_needed * 0.25  # 25K per operator in Lakhs
        
        # Calculate deadline (within 2 weeks for critical, 4 weeks for high)
        days_to_deadline = 14 if metrics['risk_category'] == 'CRITICAL' else 28
        deadline = date.today() + timedelta(days=days_to_deadline)
        
        recommendation = Recommendation(
            pincode_metric_id=pincode_metric.id,
            recommendation_type='DEPLOY_CAMP',
            title=f"Deploy biometric update camp in {pincode}",
            description=f"High biometric update demand detected. Risk Score: {metrics['overall_risk_score']:.0f}. "
                       f"Estimated {bio_updates} updates pending.",
            priority='URGENT' if metrics['risk_category'] == 'CRITICAL' else 'HIGH',
            urgency_score=metrics['priority_score'],
            operators_needed=operators_needed,
            estimated_budget=estimated_budget,
            deadline=deadline,
            expected_impact=f"Prevent {bio_updates * 10} potential authentication failures",
            status='PENDING'
        )
        
        self.db.add(recommendation)