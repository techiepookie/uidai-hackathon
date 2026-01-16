# backend/app/routers/pincodes.py
"""
Pincode API routes - Core endpoints for pincode data and metrics.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import date, timedelta

from app.database import get_db
from app.models import schemas
from app.models.db_models import RawUpdate, PincodeMetric, Prediction, Anomaly, Recommendation
from app.services.risk_calculator import RiskCalculatorService
from app.services.forecaster import ForecasterService
from app.services.clustering import ClusteringService

router = APIRouter(prefix="/pincodes", tags=["Pincodes"])


@router.get("/", response_model=schemas.PincodeListResponse)
def get_all_pincodes(
    state: Optional[str] = None,
    district: Optional[str] = None,
    risk_category: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    min_risk: Optional[float] = Query(None, ge=0, le=100),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Get list of all pincodes with metrics.
    
    Supports filtering by state, district, risk category, and minimum risk score.
    """
    query = db.query(PincodeMetric)
    
    if state:
        query = query.filter(PincodeMetric.state == state)
    
    if district:
        query = query.filter(PincodeMetric.district == district)
    
    if risk_category:
        query = query.filter(PincodeMetric.risk_category == risk_category)
    
    if min_risk is not None:
        query = query.filter(PincodeMetric.overall_risk_score >= min_risk)
    
    total = query.count()
    
    pincodes = query.order_by(
        PincodeMetric.priority_score.desc()
    ).offset(skip).limit(limit).all()
    
    return schemas.PincodeListResponse(
        total=total,
        pincodes=[schemas.PincodeMetricResponse.model_validate(p) for p in pincodes]
    )


@router.get("/map-data", response_model=List[schemas.PincodeSummary])
def get_map_data(
    state: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get pincode data optimized for map display.
    Returns lightweight summary with coordinates and risk scores.
    """
    query = db.query(
        PincodeMetric.pincode,
        PincodeMetric.state,
        PincodeMetric.overall_risk_score,
        PincodeMetric.risk_category,
        PincodeMetric.priority_score,
        PincodeMetric.total_bio_updates,
        PincodeMetric.total_demo_updates
    )
    
    if state:
        query = query.filter(PincodeMetric.state == state)
    
    results = query.all()
    
    return [
        schemas.PincodeSummary(
            pincode=r.pincode,
            state=r.state,
            overall_risk_score=r.overall_risk_score or 0,
            risk_category=r.risk_category or "LOW",
            priority_score=r.priority_score or 0,
            total_bio_updates=r.total_bio_updates or 0,
            total_demo_updates=r.total_demo_updates or 0
        )
        for r in results
    ]


@router.get("/priority", response_model=List[schemas.PincodeMetricResponse])
def get_priority_pincodes(
    limit: int = Query(20, ge=1, le=100),
    urgency: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    db: Session = Depends(get_db)
):
    """
    Get top priority pincodes sorted by priority score.
    Used for the main dashboard priority list.
    """
    query = db.query(PincodeMetric)
    
    if urgency:
        query = query.filter(PincodeMetric.risk_category == urgency)
    
    pincodes = query.order_by(
        PincodeMetric.priority_score.desc()
    ).limit(limit).all()
    
    return [schemas.PincodeMetricResponse.model_validate(p) for p in pincodes]


@router.get("/states")
def get_states(db: Session = Depends(get_db)):
    """Get list of all unique states."""
    states = db.query(PincodeMetric.state).distinct().order_by(PincodeMetric.state).all()
    return {"states": [s.state for s in states if s.state]}


@router.get("/{pincode}", response_model=schemas.PincodeMetricResponse)
def get_pincode_detail(
    pincode: str,
    db: Session = Depends(get_db)
):
    """Get detailed metrics for a specific pincode."""
    metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    if not metric:
        raise HTTPException(status_code=404, detail="Pincode not found")
    
    return schemas.PincodeMetricResponse.model_validate(metric)


@router.get("/{pincode}/history", response_model=schemas.TimeSeriesResponse)
def get_pincode_history(
    pincode: str,
    days: int = Query(90, ge=7, le=365),
    db: Session = Depends(get_db)
):
    """Get historical update data for a pincode."""
    cutoff_date = date.today() - timedelta(days=days)
    
    data = db.query(RawUpdate).filter(
        RawUpdate.pincode == pincode,
        RawUpdate.date >= cutoff_date
    ).order_by(RawUpdate.date).all()
    
    if not data:
        raise HTTPException(status_code=404, detail="No data found for pincode")
    
    return schemas.TimeSeriesResponse(
        pincode=pincode,
        data=[
            schemas.TimeSeriesData(
                date=d.date,
                bio_updates=d.bio_total or 0,
                demo_updates=d.demo_total or 0,
                mobile_updates=d.mobile_updates or 0
            )
            for d in data
        ]
    )


@router.get("/{pincode}/forecast", response_model=schemas.ForecastResponse)
def get_pincode_forecast(
    pincode: str,
    metric_type: str = Query("bio", pattern="^(bio|demo|mobile)$"),
    horizon: int = Query(30, ge=7, le=90),
    db: Session = Depends(get_db)
):
    """Get forecast predictions for a pincode."""
    # Check pincode exists
    metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    if not metric:
        raise HTTPException(status_code=404, detail="Pincode not found")
    
    # Generate forecast
    forecaster = ForecasterService(db)
    result = forecaster.generate_forecast(pincode, metric_type, horizon)
    
    # Build response
    forecasts = []
    for f in result['forecasts']:
        forecasts.append(schemas.PredictionResponse(
            id=0,
            pincode=pincode,
            prediction_date=f['date'],
            prediction_type=metric_type,
            predicted_value=f['predicted_value'],
            lower_bound=f.get('lower_bound'),
            upper_bound=f.get('upper_bound'),
            confidence=f.get('confidence', 0.8),
            model_used=result['model_used']
        ))
    
    return schemas.ForecastResponse(
        pincode=pincode,
        forecasts=forecasts,
        summary=result.get('summary', {})
    )


@router.get("/{pincode}/similar", response_model=List[schemas.PincodeSummary])
def get_similar_pincodes(
    pincode: str,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Find pincodes similar to the given one (same cluster)."""
    clustering = ClusteringService(db)
    similar = clustering.get_similar_pincodes(pincode, limit)
    
    return [
        schemas.PincodeSummary(
            pincode=p.pincode,
            state=p.state,
            overall_risk_score=p.overall_risk_score or 0,
            risk_category=p.risk_category or "LOW",
            priority_score=p.priority_score or 0,
            total_bio_updates=p.total_bio_updates or 0,
            total_demo_updates=p.total_demo_updates or 0
        )
        for p in similar
    ]


@router.get("/{pincode}/recommendations", response_model=schemas.RecommendationListResponse)
def get_pincode_recommendations(
    pincode: str,
    status: Optional[str] = Query(None, pattern="^(PENDING|IN_PROGRESS|COMPLETED|CANCELLED)$"),
    db: Session = Depends(get_db)
):
    """Get recommendations for a specific pincode."""
    metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    if not metric:
        raise HTTPException(status_code=404, detail="Pincode not found")
    
    query = db.query(Recommendation).filter(
        Recommendation.pincode_metric_id == metric.id
    )
    
    if status:
        query = query.filter(Recommendation.status == status)
    
    recommendations = query.order_by(
        Recommendation.urgency_score.desc()
    ).all()
    
    rec_responses = []
    for r in recommendations:
        rec_responses.append(schemas.RecommendationResponse(
            id=r.id,
            pincode=pincode,
            state=metric.state,
            recommendation_type=r.recommendation_type,
            title=r.title,
            description=r.description,
            priority=r.priority,
            urgency_score=r.urgency_score,
            operators_needed=r.operators_needed,
            estimated_budget=r.estimated_budget,
            deadline=r.deadline,
            expected_impact=r.expected_impact,
            status=r.status,
            created_at=r.created_at
        ))
    
    return schemas.RecommendationListResponse(
        total=len(rec_responses),
        recommendations=rec_responses
    )


@router.post("/{pincode}/recalculate")
def recalculate_pincode_metrics(
    pincode: str,
    db: Session = Depends(get_db)
):
    """Recalculate all metrics for a specific pincode."""
    # Check pincode has data
    exists = db.query(RawUpdate).filter(
        RawUpdate.pincode == pincode
    ).first()
    
    if not exists:
        raise HTTPException(status_code=404, detail="Pincode not found in raw data")
    
    calculator = RiskCalculatorService(db)
    count = calculator.calculate_all_metrics(days_lookback=90)
    
    # Get updated metric
    metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    return {
        "status": "success",
        "pincode": pincode,
        "priority_score": metric.priority_score if metric else 0,
        "risk_category": metric.risk_category if metric else "LOW"
    }