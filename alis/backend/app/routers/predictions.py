# backend/app/routers/predictions.py
"""
Predictions API routes - Forecasting and model validation.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date, datetime, timedelta

from app.database import get_db
from app.models import schemas
from app.models.db_models import PincodeMetric, Prediction
from app.services.forecaster import ForecasterService

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/{pincode}", response_model=List[schemas.PredictionResponse])
def get_predictions(
    pincode: str,
    days: int = Query(30, ge=1, le=90),
    metric_type: Optional[str] = Query(None, pattern="^(bio|demo|mobile)$"),
    db: Session = Depends(get_db)
):
    """
    Get existing predictions for a specific pincode.
    """
    pincode_metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    if not pincode_metric:
        raise HTTPException(status_code=404, detail="Pincode not found")
    
    query = db.query(Prediction).filter(
        Prediction.pincode_metric_id == pincode_metric.id,
        Prediction.prediction_date >= date.today()
    )
    
    if metric_type:
        query = query.filter(Prediction.prediction_type == metric_type)
    
    predictions = query.order_by(
        Prediction.prediction_date
    ).limit(days).all()
    
    return [
        schemas.PredictionResponse(
            id=p.id,
            pincode=pincode,
            prediction_date=p.prediction_date,
            prediction_type=p.prediction_type,
            predicted_value=p.predicted_value,
            lower_bound=p.lower_bound,
            upper_bound=p.upper_bound,
            confidence=p.confidence,
            model_used=p.model_used
        )
        for p in predictions
    ]


@router.post("/{pincode}/generate")
def generate_predictions(
    pincode: str,
    days: int = Query(30, ge=7, le=90),
    metric_type: str = Query("bio", pattern="^(bio|demo|mobile)$"),
    db: Session = Depends(get_db)
):
    """
    Generate new predictions for a pincode.
    """
    pincode_metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    if not pincode_metric:
        raise HTTPException(status_code=404, detail="Pincode not found")
    
    forecaster = ForecasterService(db)
    result = forecaster.generate_forecast(pincode, metric_type, days)
    forecaster.save_predictions(pincode, result)
    
    return {
        "status": "success",
        "pincode": pincode,
        "metric_type": metric_type,
        "predictions_generated": len(result['forecasts']),
        "model_used": result['model_used'],
        "summary": result.get('summary', {})
    }


@router.post("/generate-all")
def generate_all_predictions(
    limit_pincodes: Optional[int] = Query(None, ge=1),
    db: Session = Depends(get_db)
):
    """
    Generate predictions for all pincodes.
    Admin endpoint for batch processing.
    """
    forecaster = ForecasterService(db)
    count = forecaster.generate_all_forecasts(limit_pincodes=limit_pincodes)
    
    return {
        "status": "success",
        "total_forecasts_generated": count
    }


@router.get("/validate/all")
def validate_all_predictions(
    days_back: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    Validate prediction accuracy against actual values.
    """
    forecaster = ForecasterService(db)
    metrics = forecaster.validate_predictions(days_back=days_back)
    
    return {
        "status": "success",
        "days_back": days_back,
        "metrics": metrics
    }


@router.get("/{pincode}/validate")
def validate_pincode_predictions(
    pincode: str,
    db: Session = Depends(get_db)
):
    """
    Validate prediction accuracy for a specific pincode.
    """
    pincode_metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    if not pincode_metric:
        raise HTTPException(status_code=404, detail="Pincode not found")
    
    # Get predictions with actual values
    predictions = db.query(Prediction).filter(
        Prediction.pincode_metric_id == pincode_metric.id,
        Prediction.actual_value.isnot(None)
    ).order_by(Prediction.prediction_date.desc()).limit(30).all()
    
    if not predictions:
        return {
            "message": "No validated predictions available",
            "predictions_count": 0
        }
    
    # Calculate metrics
    import numpy as np
    
    actuals = [p.actual_value for p in predictions]
    preds = [p.predicted_value for p in predictions]
    
    mae = np.mean(np.abs(np.array(preds) - np.array(actuals)))
    rmse = np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))
    
    # Avoid division by zero
    non_zero = [a for a in actuals if a > 0]
    if non_zero:
        mape = np.mean(np.abs((np.array(actuals) - np.array(preds)) / (np.array(actuals) + 1e-10))) * 100
    else:
        mape = 0
    
    return {
        "pincode": pincode,
        "predictions_count": len(predictions),
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2)
        }
    }


@router.get("/peak-detection/{pincode}")
def detect_peaks(
    pincode: str,
    days: int = Query(30, ge=7, le=90),
    metric_type: str = Query("bio", pattern="^(bio|demo|mobile)$"),
    db: Session = Depends(get_db)
):
    """
    Detect predicted peak days for resource planning.
    """
    pincode_metric = db.query(PincodeMetric).filter(
        PincodeMetric.pincode == pincode
    ).first()
    
    if not pincode_metric:
        raise HTTPException(status_code=404, detail="Pincode not found")
    
    forecaster = ForecasterService(db)
    result = forecaster.generate_forecast(pincode, metric_type, days)
    
    if not result['forecasts']:
        return {"message": "No forecast data available"}
    
    forecasts = result['forecasts']
    values = [f['predicted_value'] for f in forecasts]
    
    import numpy as np
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    threshold = mean_val + 1.5 * std_val
    
    peaks = []
    for f in forecasts:
        if f['predicted_value'] > threshold:
            peaks.append({
                "date": f['date'].isoformat() if isinstance(f['date'], date) else f['date'],
                "predicted_value": f['predicted_value'],
                "severity": "HIGH" if f['predicted_value'] > mean_val + 2 * std_val else "ELEVATED"
            })
    
    return {
        "pincode": pincode,
        "metric_type": metric_type,
        "forecast_days": days,
        "mean_predicted": round(mean_val, 2),
        "threshold": round(threshold, 2),
        "peak_days": peaks,
        "peak_count": len(peaks)
    }