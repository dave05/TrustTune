"""FastAPI application for calibration service."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import numpy as np
from typing import List, Optional
from trusttune.core.factory import create_calibrator
from trusttune.monitoring.metrics import MetricsCollector

app = FastAPI(title="TrustTune Calibration Service")
metrics = MetricsCollector()

class CalibrationRequest(BaseModel):
    scores: List[float] = Field(..., description="Uncalibrated probability scores")
    labels: List[int] = Field(..., description="Binary ground truth labels")
    calibrator_type: str = Field(..., description="Type of calibrator to use")
    
    @field_validator("scores")
    def validate_scores(cls, v):
        if not all(0 <= x <= 1 for x in v):
            raise ValueError("Scores must be between 0 and 1")
        return v
    
    @field_validator("labels")
    def validate_labels(cls, v):
        if not all(x in [0, 1] for x in v):
            raise ValueError("Labels must be binary (0 or 1)")
        return v

class CalibrationResponse(BaseModel):
    calibrated_scores: List[float]
    metrics: dict

@app.post("/calibrate", response_model=CalibrationResponse)
async def calibrate(request: CalibrationRequest):
    """Calibrate probability scores."""
    try:
        with metrics.measure_latency():
            calibrator = create_calibrator(request.calibrator_type)
            scores = np.array(request.scores)
            labels = np.array(request.labels)
            
            calibrator.fit(scores, labels)
            calibrated_scores = calibrator.predict_proba(scores)
            
            metrics.record_request(
                calibrator_type=request.calibrator_type,
                input_size=len(scores)
            )
            
            return {
                "calibrated_scores": calibrated_scores.tolist(),
                "metrics": {
                    "ece": float(compute_ece(scores, labels)),
                    "brier_score": float(compute_brier_score(scores, labels))
                }
            }
    except ValueError as e:
        metrics.record_error(str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        metrics.record_error(str(e))
        raise HTTPException(status_code=500, detail="Internal server error") 