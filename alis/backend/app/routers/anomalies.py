"""
Anomaly detection API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date, timedelta

from app.database import get_db
from app.models import schemas
from app.models.db_models import Anomaly, PincodeMetric
from app.services.anamoly_detector import AnomalyDetectorService

router = APIRouter(prefix="/anomalies", tags=["Anomalies"])


@router.get("/", response_model=schemas.AnomalyListResponse)
def get_anomalies(
    days: int = Query(7, ge=1, le=90),
    severity: Optional[str] = Query(None, pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$"),
    metric_type: Optional[str] = Query(None, pattern="^(bio|demo|mobile)$"),
    investigated: Optional[bool] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """
    Get list of detected anomalies.
    
    Filters:
    - days: Number of days to look back (default: 7)
    - severity: Filter by severity level (LOW, MEDIUM, HIGH, CRITICAL)
    - metric_type: Filter by metric (bio, demo, mobile)
    - investigated: Filter by investigation status
    """
    cutoff_date = date.today() - timedelta(days=days)
    
    query = db.query(Anomaly).join(PincodeMetric).filter(
        Anomaly.detected_date >= cutoff_date
    )
    
    if severity:
        query = query.filter(Anomaly.severity == severity)
    
    if metric_type:
        query = query.filter(Anomaly.metric_affected == metric_type)
    
    if investigated is not None:
        query = query.filter(Anomaly.is_investigated == investigated)
    
    total = query.count()
    
    anomalies = query.order_by(
        Anomaly.confidence_score.desc()
    ).offset(skip).limit(limit).all()
    
    # Build response
    anomaly_responses = []
    for a in anomalies:
        pincode_metric = a.pincode_metric
        anomaly_responses.append(schemas.AnomalyResponse(
            id=a.id,
            pincode=pincode_metric.pincode if pincode_metric else "Unknown",
            state=pincode_metric.state if pincode_metric else "Unknown",
            detected_date=a.detected_date,
            anomaly_type=a.anomaly_type,
            metric_affected=a.metric_affected,
            expected_value=a.expected_value,
            actual_value=a.actual_value,
            deviation_percent=a.deviation_percent,
            detection_methods=a.detection_methods or [],
            confidence_score=a.confidence_score,
            severity=a.severity,
            is_seasonal=a.is_seasonal,
            is_investigated=a.is_investigated,
            investigation_notes=a.investigation_notes,
            root_cause=a.root_cause,
            created_at=a.created_at
        ))
    
    return schemas.AnomalyListResponse(
        total=total,
        anomalies=anomaly_responses
    )


@router.get("/recent", response_model=List[schemas.AnomalyResponse])
def get_recent_anomalies(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get most recent anomalies for dashboard alerts."""
    detector = AnomalyDetectorService(db)
    anomalies = detector.get_recent_anomalies(days=7, limit=limit)
    
    responses = []
    for a in anomalies:
        pincode_metric = a.pincode_metric
        responses.append(schemas.AnomalyResponse(
            id=a.id,
            pincode=pincode_metric.pincode if pincode_metric else "Unknown",
            state=pincode_metric.state if pincode_metric else "Unknown",
            detected_date=a.detected_date,
            anomaly_type=a.anomaly_type,
            metric_affected=a.metric_affected,
            expected_value=a.expected_value,
            actual_value=a.actual_value,
            deviation_percent=a.deviation_percent,
            detection_methods=a.detection_methods or [],
            confidence_score=a.confidence_score,
            severity=a.severity,
            is_seasonal=a.is_seasonal,
            is_investigated=a.is_investigated,
            investigation_notes=a.investigation_notes,
            root_cause=a.root_cause,
            created_at=a.created_at
        ))
    
    return responses


@router.get("/{anomaly_id}", response_model=schemas.AnomalyResponse)
def get_anomaly_detail(
    anomaly_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed information about a specific anomaly."""
    anomaly = db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()
    
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    
    pincode_metric = anomaly.pincode_metric
    
    return schemas.AnomalyResponse(
        id=anomaly.id,
        pincode=pincode_metric.pincode if pincode_metric else "Unknown",
        state=pincode_metric.state if pincode_metric else "Unknown",
        detected_date=anomaly.detected_date,
        anomaly_type=anomaly.anomaly_type,
        metric_affected=anomaly.metric_affected,
        expected_value=anomaly.expected_value,
        actual_value=anomaly.actual_value,
        deviation_percent=anomaly.deviation_percent,
        detection_methods=anomaly.detection_methods or [],
        confidence_score=anomaly.confidence_score,
        severity=anomaly.severity,
        is_seasonal=anomaly.is_seasonal,
        is_investigated=anomaly.is_investigated,
        investigation_notes=anomaly.investigation_notes,
        root_cause=anomaly.root_cause,
        created_at=anomaly.created_at
    )


@router.put("/{anomaly_id}/investigate", response_model=schemas.AnomalyResponse)
def investigate_anomaly(
    anomaly_id: int,
    request: schemas.InvestigateAnomalyRequest,
    db: Session = Depends(get_db)
):
    """Update investigation status and notes for an anomaly."""
    anomaly = db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()
    
    if not anomaly:
        raise HTTPException(status_code=404, detail="Anomaly not found")
    
    anomaly.is_investigated = request.is_investigated
    
    if request.investigation_notes:
        anomaly.investigation_notes = request.investigation_notes
    
    if request.root_cause:
        anomaly.root_cause = request.root_cause
    
    db.commit()
    db.refresh(anomaly)
    
    pincode_metric = anomaly.pincode_metric
    
    return schemas.AnomalyResponse(
        id=anomaly.id,
        pincode=pincode_metric.pincode if pincode_metric else "Unknown",
        state=pincode_metric.state if pincode_metric else "Unknown",
        detected_date=anomaly.detected_date,
        anomaly_type=anomaly.anomaly_type,
        metric_affected=anomaly.metric_affected,
        expected_value=anomaly.expected_value,
        actual_value=anomaly.actual_value,
        deviation_percent=anomaly.deviation_percent,
        detection_methods=anomaly.detection_methods or [],
        confidence_score=anomaly.confidence_score,
        severity=anomaly.severity,
        is_seasonal=anomaly.is_seasonal,
        is_investigated=anomaly.is_investigated,
        investigation_notes=anomaly.investigation_notes,
        root_cause=anomaly.root_cause,
        created_at=anomaly.created_at
    )


@router.post("/detect")
def run_anomaly_detection(
    days_lookback: int = Query(90, ge=30, le=365),
    limit_pincodes: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Trigger anomaly detection for all pincodes.
    Admin endpoint for batch processing.
    """
    detector = AnomalyDetectorService(db)
    count = detector.run_anomaly_detection(
        days_lookback=days_lookback,
        limit_pincodes=limit_pincodes
    )
    
    return {
        "status": "success",
        "anomalies_detected": count,
        "days_lookback": days_lookback
    }


@router.get("/stats/summary")
def get_anomaly_stats(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db)
):
    """Get summary statistics for anomalies."""
    from sqlalchemy import func
    
    cutoff_date = date.today() - timedelta(days=days)
    
    # Count by severity
    severity_counts = db.query(
        Anomaly.severity,
        func.count(Anomaly.id).label('count')
    ).filter(
        Anomaly.detected_date >= cutoff_date
    ).group_by(Anomaly.severity).all()
    
    # Count by metric
    metric_counts = db.query(
        Anomaly.metric_affected,
        func.count(Anomaly.id).label('count')
    ).filter(
        Anomaly.detected_date >= cutoff_date
    ).group_by(Anomaly.metric_affected).all()
    
    # Investigation status
    investigated = db.query(func.count(Anomaly.id)).filter(
        Anomaly.detected_date >= cutoff_date,
        Anomaly.is_investigated == True
    ).scalar()
    
    not_investigated = db.query(func.count(Anomaly.id)).filter(
        Anomaly.detected_date >= cutoff_date,
        Anomaly.is_investigated == False
    ).scalar()
    
    return {
        "period_days": days,
        "by_severity": {s.severity: s.count for s in severity_counts},
        "by_metric": {m.metric_affected: m.count for m in metric_counts},
        "investigation_status": {
            "investigated": investigated,
            "pending": not_investigated
        }
    }
