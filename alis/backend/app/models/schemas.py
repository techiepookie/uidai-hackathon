"""
Pydantic schemas for API request/response validation.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# Base schemas
class PincodeBase(BaseModel):
    """Base schema for pincode data."""
    pincode: str = Field(..., min_length=6, max_length=6)
    state: str
    district: Optional[str] = None


# Response schemas
class PincodeMetricResponse(BaseModel):
    """Response schema for pincode metrics."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    pincode: str
    state: str
    district: Optional[str]
    
    # Risk scores
    bio_risk_score: float
    demo_risk_score: float
    mobile_risk_score: float
    overall_risk_score: float
    priority_score: float
    risk_category: str
    
    # Statistics
    total_bio_updates: int
    total_demo_updates: int
    total_mobile_updates: int
    total_enrollments: int
    
    # Rates
    bio_update_rate: float
    demo_update_rate: float
    mobile_linkage_rate: float
    
    # Trends
    bio_trend: str
    demo_trend: str
    
    # Child metrics
    child_bio_coverage: float
    child_update_rate: float
    
    # Migration
    migration_score: float
    
    # Data quality
    data_confidence: float
    days_of_data: int
    last_update_date: Optional[date]
    
    # Cluster
    cluster_id: Optional[int]
    
    calculated_at: datetime


class PincodeListResponse(BaseModel):
    """Response schema for list of pincodes."""
    total: int
    pincodes: List[PincodeMetricResponse]


class PincodeSummary(BaseModel):
    """Summary schema for map display."""
    model_config = ConfigDict(from_attributes=True)
    
    pincode: str
    state: str
    overall_risk_score: float
    risk_category: str
    priority_score: float
    total_bio_updates: int
    total_demo_updates: int


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    pincode: str
    prediction_date: date
    prediction_type: str
    predicted_value: float
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    confidence: float
    model_used: str


class ForecastResponse(BaseModel):
    """Response schema for forecast data."""
    pincode: str
    forecasts: List[PredictionResponse]
    summary: Dict[str, Any]


class AnomalyResponse(BaseModel):
    """Response schema for anomalies."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    pincode: str
    state: str
    detected_date: date
    anomaly_type: str
    metric_affected: str
    expected_value: Optional[float]
    actual_value: float
    deviation_percent: Optional[float]
    detection_methods: List[str]
    confidence_score: float
    severity: str
    is_seasonal: bool
    is_investigated: bool
    investigation_notes: Optional[str]
    root_cause: Optional[str]
    created_at: datetime


class AnomalyListResponse(BaseModel):
    """Response schema for list of anomalies."""
    total: int
    anomalies: List[AnomalyResponse]


class RecommendationResponse(BaseModel):
    """Response schema for recommendations."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    pincode: str
    state: str
    recommendation_type: str
    title: str
    description: Optional[str]
    priority: str
    urgency_score: float
    operators_needed: int
    estimated_budget: float
    deadline: Optional[date]
    expected_impact: Optional[str]
    status: str
    created_at: datetime


class RecommendationListResponse(BaseModel):
    """Response schema for list of recommendations."""
    total: int
    recommendations: List[RecommendationResponse]


class ClusterResponse(BaseModel):
    """Response schema for clusters."""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    name: str
    description: Optional[str]
    profile: Optional[str]
    avg_bio_updates: float
    avg_demo_updates: float
    avg_mobile_updates: float
    pincode_count: int


class StateAnalytics(BaseModel):
    """Analytics summary for a state."""
    state: str
    total_pincodes: int
    critical_pincodes: int
    high_risk_pincodes: int
    medium_risk_pincodes: int
    low_risk_pincodes: int
    total_bio_updates: int
    total_demo_updates: int
    total_mobile_updates: int
    total_enrollments: int
    avg_bio_risk: float
    avg_demo_risk: float
    avg_mobile_linkage: float
    top_priority_pincodes: List[PincodeSummary]


class NationalAnalytics(BaseModel):
    """National level analytics."""
    total_pincodes: int
    total_states: int
    total_bio_updates: int
    total_demo_updates: int
    total_mobile_updates: int
    total_enrollments: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    avg_bio_risk: float
    avg_demo_risk: float
    anomalies_last_7_days: int
    state_summaries: List[StateAnalytics]


class TimeSeriesData(BaseModel):
    """Time series data point."""
    date: date
    bio_updates: int
    demo_updates: int
    mobile_updates: int


class TimeSeriesResponse(BaseModel):
    """Response for time series data."""
    pincode: str
    data: List[TimeSeriesData]


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    checks: Dict[str, bool]


# Request schemas
class InvestigateAnomalyRequest(BaseModel):
    """Request to update anomaly investigation."""
    is_investigated: bool = True
    investigation_notes: Optional[str] = None
    root_cause: Optional[str] = None


class UpdateRecommendationStatusRequest(BaseModel):
    """Request to update recommendation status."""
    status: str = Field(..., pattern="^(PENDING|IN_PROGRESS|COMPLETED|CANCELLED)$")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)


class FilterParams(BaseModel):
    """Filter parameters for queries."""
    state: Optional[str] = None
    district: Optional[str] = None
    risk_category: Optional[str] = None
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None