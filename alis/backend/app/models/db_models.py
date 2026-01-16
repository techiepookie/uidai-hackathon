"""
SQLAlchemy database models for ALIS.
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, 
    Boolean, Text, ForeignKey, Index, JSON
)
from sqlalchemy.orm import relationship
from app.database import Base


class RawUpdate(Base):
    """Raw Aadhaar update data from UIDAI."""
    
    __tablename__ = "raw_updates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=True)
    pincode = Column(String(6), nullable=False, index=True)
    
    # Age-wise biometric updates
    bio_0_5 = Column(Integer, default=0)
    bio_5_17 = Column(Integer, default=0)
    bio_18_30 = Column(Integer, default=0)
    bio_30_60 = Column(Integer, default=0)
    bio_60_plus = Column(Integer, default=0)
    bio_total = Column(Integer, default=0)
    
    # Age-wise demographic updates
    demo_0_5 = Column(Integer, default=0)
    demo_5_17 = Column(Integer, default=0)
    demo_18_30 = Column(Integer, default=0)
    demo_30_60 = Column(Integer, default=0)
    demo_60_plus = Column(Integer, default=0)
    demo_total = Column(Integer, default=0)
    
    # Mobile updates
    mobile_updates = Column(Integer, default=0)
    
    # Enrollments (total population reference)
    enrol_0_5 = Column(Integer, default=0)
    enrol_5_17 = Column(Integer, default=0)
    enrol_18_30 = Column(Integer, default=0)
    enrol_30_60 = Column(Integer, default=0)
    enrol_60_plus = Column(Integer, default=0)
    enrol_total = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    is_interpolated = Column(Boolean, default=False)
    data_quality_score = Column(Float, default=1.0)
    
    __table_args__ = (
        Index('idx_pincode_date', 'pincode', 'date'),
        Index('idx_state_date', 'state', 'date'),
    )


class PincodeMetric(Base):
    """Calculated metrics for each pincode."""
    
    __tablename__ = "pincode_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pincode = Column(String(6), nullable=False, unique=True, index=True)
    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=True)
    
    # Risk scores (0-100)
    bio_risk_score = Column(Float, default=0.0)
    demo_risk_score = Column(Float, default=0.0)
    mobile_risk_score = Column(Float, default=0.0)
    overall_risk_score = Column(Float, default=0.0)
    priority_score = Column(Float, default=0.0, index=True)
    
    # Risk categories
    risk_category = Column(String(20), default="LOW")  # LOW, MEDIUM, HIGH, CRITICAL
    
    # Aggregated statistics
    total_bio_updates = Column(Integer, default=0)
    total_demo_updates = Column(Integer, default=0)
    total_mobile_updates = Column(Integer, default=0)
    total_enrollments = Column(Integer, default=0)
    
    # Rates
    bio_update_rate = Column(Float, default=0.0)
    demo_update_rate = Column(Float, default=0.0)
    mobile_linkage_rate = Column(Float, default=0.0)
    
    # Trends
    bio_trend = Column(String(20), default="STABLE")  # INCREASING, DECREASING, STABLE
    demo_trend = Column(String(20), default="STABLE")
    
    # Volatility
    bio_volatility = Column(Float, default=0.0)
    demo_volatility = Column(Float, default=0.0)
    
    # Child-specific metrics (5-17 age group)
    child_bio_coverage = Column(Float, default=0.0)
    child_update_rate = Column(Float, default=0.0)
    
    # Migration indicator
    migration_score = Column(Float, default=0.0)
    
    # Cluster assignment
    cluster_id = Column(Integer, ForeignKey("pincode_clusters.id"), nullable=True)
    
    # Data quality
    data_confidence = Column(Float, default=1.0)
    days_of_data = Column(Integer, default=0)
    last_update_date = Column(Date, nullable=True)
    
    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    cluster = relationship("PincodeCluster", back_populates="pincodes")
    predictions = relationship("Prediction", back_populates="pincode_metric")
    anomalies = relationship("Anomaly", back_populates="pincode_metric")
    recommendations = relationship("Recommendation", back_populates="pincode_metric")


class PincodeCluster(Base):
    """Cluster definitions for pincode grouping."""
    
    __tablename__ = "pincode_clusters"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    
    # Cluster characteristics
    avg_bio_updates = Column(Float, default=0.0)
    avg_demo_updates = Column(Float, default=0.0)
    avg_mobile_updates = Column(Float, default=0.0)
    avg_enrollments = Column(Float, default=0.0)
    
    # Cluster profile
    profile = Column(String(50), nullable=True)  # e.g., "HIGH_MIGRATION", "STABLE_URBAN"
    
    # Statistics
    pincode_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pincodes = relationship("PincodeMetric", back_populates="cluster")


class Prediction(Base):
    """Forecasted values for pincodes."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pincode_metric_id = Column(Integer, ForeignKey("pincode_metrics.id"), nullable=False)
    
    prediction_date = Column(Date, nullable=False, index=True)
    prediction_type = Column(String(20), nullable=False)  # bio, demo, mobile
    
    # Predictions
    predicted_value = Column(Float, nullable=False)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    confidence = Column(Float, default=0.8)
    
    # Model info
    model_used = Column(String(50), default="ensemble")
    
    # Validation (filled when actual data available)
    actual_value = Column(Float, nullable=True)
    prediction_error = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pincode_metric = relationship("PincodeMetric", back_populates="predictions")


class Anomaly(Base):
    """Detected anomalies in update patterns."""
    
    __tablename__ = "anomalies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pincode_metric_id = Column(Integer, ForeignKey("pincode_metrics.id"), nullable=False)
    
    detected_date = Column(Date, nullable=False, index=True)
    anomaly_type = Column(String(50), nullable=False)  # SPIKE, DROP, PATTERN_BREAK
    
    # Anomaly details
    metric_affected = Column(String(20), nullable=False)  # bio, demo, mobile
    expected_value = Column(Float, nullable=True)
    actual_value = Column(Float, nullable=False)
    deviation_percent = Column(Float, nullable=True)
    
    # Detection method
    detection_methods = Column(JSON, default=list)  # List of methods that flagged this
    confidence_score = Column(Float, nullable=False)
    
    # Classification
    severity = Column(String(20), default="MEDIUM")  # LOW, MEDIUM, HIGH, CRITICAL
    is_seasonal = Column(Boolean, default=False)
    is_investigated = Column(Boolean, default=False)
    
    # Investigation notes
    investigation_notes = Column(Text, nullable=True)
    root_cause = Column(String(200), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pincode_metric = relationship("PincodeMetric", back_populates="anomalies")


class Recommendation(Base):
    """Generated recommendations for operational actions."""
    
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    pincode_metric_id = Column(Integer, ForeignKey("pincode_metrics.id"), nullable=False)
    
    # Recommendation details
    recommendation_type = Column(String(50), nullable=False)  # DEPLOY_CAMP, INVESTIGATE, MONITOR
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Priority
    priority = Column(String(20), default="MEDIUM")  # LOW, MEDIUM, HIGH, URGENT
    urgency_score = Column(Float, default=0.0)
    
    # Resource estimates
    operators_needed = Column(Integer, default=0)
    estimated_budget = Column(Float, default=0.0)  # in Lakhs
    
    # Timeline
    deadline = Column(Date, nullable=True)
    expected_impact = Column(Text, nullable=True)
    
    # Status tracking
    status = Column(String(20), default="PENDING")  # PENDING, IN_PROGRESS, COMPLETED, CANCELLED
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    pincode_metric = relationship("PincodeMetric", back_populates="recommendations")