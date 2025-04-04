from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, field_validator
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime, UTC
import asyncio
import json
import uuid
import logging
from ..exceptions import TrustTuneError
from ..utils.logging_config import setup_logger

from ..streaming.online_calibrator import OnlineCalibrator
from ..monitoring.metrics import MetricsCollector
from .models import BatchUpdateRequest, DriftMonitoringConfig

logger = setup_logger(__name__)
router = APIRouter(prefix="/streaming")

# In-memory storage for streaming calibrators
streaming_calibrators = {}

# Add storage for saved states
saved_states = {}

class StreamingCalibratorRequest(BaseModel):
    calibrator_type: str
    window_size: int
    update_threshold: float
    initial_scores: List[float]
    initial_labels: List[int]

    @field_validator('window_size')
    @classmethod
    def validate_window_size(cls, v):
        if v <= 0:
            raise ValueError("Window size must be positive")
        return v

    @field_validator('update_threshold')
    @classmethod
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Update threshold must be between 0 and 1")
        return v

class StreamingUpdateRequest(BaseModel):
    scores: List[float]
    labels: List[int]

class StreamingScoresRequest(BaseModel):
    scores: List[float]

class RollbackRequest(BaseModel):
    version: int

class LoadStateRequest(BaseModel):
    state_id: str

@router.post("/calibrators/")
async def create_streaming_calibrator(request: StreamingCalibratorRequest):
    try:
        # Convert initial data to numpy arrays
        initial_scores = np.array(request.initial_scores, dtype=np.float64)
        initial_labels = np.array(request.initial_labels, dtype=np.int32)
        
        calibrator = OnlineCalibrator(
            base_calibrator=request.calibrator_type,
            window_size=request.window_size,
            update_threshold=request.update_threshold
        )
        
        calibrator.fit(initial_scores, initial_labels)
        calibrator_id = str(uuid.uuid4())
        streaming_calibrators[calibrator_id] = calibrator
        
        return {
            "calibrator_id": calibrator_id,
            "initial_ece": float(calibrator.get_metrics()["ece"])
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/calibrators/{calibrator_id}/update")
async def update_streaming_calibrator(
    calibrator_id: str,
    request: StreamingUpdateRequest
):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(
            status_code=404,
            detail={"error": "Calibrator not found"}
        )
    
    try:
        calibrator = streaming_calibrators[calibrator_id]
        updated = calibrator.update(
            np.array(request.scores),
            np.array(request.labels)
        )
        
        return {
            "updated": updated,
            "current_metrics": calibrator.get_metrics()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)}
        )

@router.get("/calibrators/{calibrator_id}/metrics")
async def get_streaming_calibrator_metrics(calibrator_id: str):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(
            status_code=404,
            detail={"error": "Calibrator not found"}
        )
    
    calibrator = streaming_calibrators[calibrator_id]
    metrics = calibrator.get_metrics()
    
    return {
        "ece": metrics["ece"],
        "version": calibrator.version,
        "update_history": calibrator.history
    }

@router.post("/calibrators/{calibrator_id}/calibrate")
async def streaming_calibrate_scores(
    calibrator_id: str,
    request: StreamingScoresRequest
):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(
            status_code=404,
            detail={"error": "Calibrator not found"}
        )
    
    try:
        calibrator = streaming_calibrators[calibrator_id]
        scores = np.array(request.scores)
        calibrated_scores = calibrator.predict_proba(scores)
        
        return {
            "calibrated_scores": calibrated_scores.tolist()
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)}
        )

@router.get("/calibrators/{calibrator_id}/versions")
async def get_streaming_calibrator_versions(calibrator_id: str):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(
            status_code=404,
            detail={"error": "Calibrator not found"}
        )
    
    calibrator = streaming_calibrators[calibrator_id]
    return {
        "versions": calibrator.history
    }

@router.post("/calibrators/{calibrator_id}/batch-update")
async def batch_update_streaming_calibrator(
    calibrator_id: str,
    request: BatchUpdateRequest
) -> Dict:
    """Update streaming calibrator with new batch of data."""
    try:
        if calibrator_id not in streaming_calibrators:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Calibrator {calibrator_id} not found"
            )
        
        logger.info(
            "Processing batch update for calibrator %s with %d samples",
            calibrator_id, len(request.scores)
        )
        
        calibrator = streaming_calibrators[calibrator_id]
        scores = np.array(request.scores, dtype=np.float64)
        labels = np.array(request.labels, dtype=np.int32)
        
        updated = calibrator.update(scores, labels)
        metrics = calibrator.get_metrics()
        
        response = {
            "updated": updated,
            "version": calibrator.version,
            "metrics": metrics
        }
        
        # Handle drift monitoring if provided
        if request.drift_monitoring:
            drift_metrics = _check_drift(calibrator, request.drift_monitoring)
            response.update(drift_metrics)
            
            if drift_metrics["detected"]:
                logger.warning(
                    "Drift detected for calibrator %s: %s",
                    calibrator_id, drift_metrics
                )
        
        logger.info(
            "Successfully processed batch update for calibrator %s (version=%d)",
            calibrator_id, calibrator.version
        )
        return response
        
    except TrustTuneError as e:
        logger.error(
            "Failed to process batch update for calibrator %s: %s",
            calibrator_id, str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Unexpected error during batch update for calibrator %s: %s",
            calibrator_id, str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/calibrators/{calibrator_id}/rollback")
async def rollback_calibrator(calibrator_id: str, request: RollbackRequest):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(status_code=404, detail="Calibrator not found")
    
    calibrator = streaming_calibrators[calibrator_id]
    
    if not hasattr(calibrator, "history") or not calibrator.history:
        raise HTTPException(status_code=400, detail="No version history available")
    
    # Find requested version in history
    target_state = None
    for state in calibrator.history:
        if state["version"] == request.version:
            target_state = state
            break
    
    if target_state is None:
        raise HTTPException(status_code=400, detail=f"Version {request.version} not found")
    
    try:
        calibrator.set_state(target_state)
        return {
            "version": request.version,
            "metrics": calibrator.get_metrics()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/calibrators/{calibrator_id}/monitoring")
async def get_streaming_monitoring(calibrator_id: str):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(status_code=404, detail={"error": "Calibrator not found"})
    
    calibrator = streaming_calibrators[calibrator_id]
    
    # Get current metrics
    current_metrics = calibrator.get_metrics()
    
    # Get historical metrics
    historical_metrics = []
    if hasattr(calibrator, "history"):
        for state in calibrator.history:
            metrics_entry = {
                "version": state["version"],
                "metrics": state["metrics"],
                "timestamp": state.get("timestamp", datetime.now(UTC).isoformat())
            }
            historical_metrics.append(metrics_entry)
    
    return {
        "current_metrics": current_metrics,
        "historical_metrics": historical_metrics,
        "drift_status": _get_drift_status(calibrator),
        "update_frequency": _calculate_update_frequency(historical_metrics),
        "performance_stats": _get_performance_stats(calibrator)
    }

@router.post("/calibrators/{calibrator_id}/save")
async def save_calibrator_state(calibrator_id: str):
    if calibrator_id not in streaming_calibrators:
        raise HTTPException(status_code=404, detail={"error": "Calibrator not found"})
    
    calibrator = streaming_calibrators[calibrator_id]
    
    # Generate state ID
    state_id = f"state_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{calibrator.version}"
    
    # Save state to storage (in-memory for now)
    saved_states[state_id] = calibrator.get_state()
    
    return {"state_id": state_id}

@router.post("/calibrators/load")
async def load_calibrator_state(request: LoadStateRequest):
    if request.state_id not in saved_states:
        raise HTTPException(status_code=404, detail={"error": "State not found"})
    
    try:
        # Load state
        state = saved_states[request.state_id]
        
        # Create new calibrator
        calibrator = OnlineCalibrator(
            base_calibrator=state["base_calibrator_type"],
            window_size=state["window_size"],
            update_threshold=state.get("update_threshold", 0.1)
        )
        
        # Restore state including fitting status
        calibrator.set_state(state)
        
        # Generate new ID and store
        calibrator_id = str(uuid.uuid4())
        streaming_calibrators[calibrator_id] = calibrator
        
        return {"calibrator_id": calibrator_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

# Helper functions
def _check_drift(calibrator: OnlineCalibrator, config: Dict) -> Dict:
    """Check for different types of drift."""
    metrics = calibrator.get_metrics()
    
    drift_metrics = {
        "ece": metrics["ece"] > config.get("ece_threshold", 0.1),
        "reliability": _check_reliability_drift(
            metrics,
            config.get("reliability_threshold", 0.15)
        ),
        "distribution": _check_distribution_drift(
            calibrator,
            config.get("distribution_threshold", 0.2)
        )
    }
    
    return {
        "detected": any(drift_metrics.values()),
        "metrics": drift_metrics,
        "drift_metrics": {
            "ece_drift": metrics["ece"],
            "reliability_drift": max(abs(np.array(metrics["reliability_scores"]) - np.array(metrics["reliability_labels"]))),
            "distribution_drift": _calculate_distribution_drift(calibrator)
        }
    }

def _check_reliability_drift(metrics: Dict, threshold: float) -> bool:
    """Check for drift in reliability curve."""
    if not metrics["reliability_scores"]:
        return False
    
    # Calculate maximum deviation from diagonal
    scores = np.array(metrics["reliability_scores"])
    labels = np.array(metrics["reliability_labels"])
    deviations = np.abs(scores - labels)
    
    return np.max(deviations) > threshold

def _check_distribution_drift(
    calibrator: OnlineCalibrator,
    threshold: float
) -> bool:
    """Check for distribution drift using KS test."""
    if len(calibrator.history) < 2:
        return False
    
    from scipy.stats import ks_2samp
    
    # Get current and previous window scores
    current_scores = np.array(calibrator.current_window["scores"])
    prev_state = calibrator.history[-2]
    prev_scores = np.array(prev_state.get("window_scores", []))
    
    if len(prev_scores) == 0:
        return False
    
    # Perform KS test
    statistic, _ = ks_2samp(current_scores, prev_scores)
    return statistic > threshold

def _get_drift_status(calibrator: OnlineCalibrator) -> Dict:
    """Get current drift status."""
    if len(calibrator.history) < 2:
        return {"status": "stable"}
    
    metrics = calibrator.get_metrics()
    prev_metrics = calibrator.history[-2]["metrics"]
    
    ece_change = metrics["ece"] - prev_metrics["ece"]
    
    if abs(ece_change) < 0.01:
        status = "stable"
    elif ece_change > 0:
        status = "degrading"
    else:
        status = "improving"
    
    return {
        "status": status,
        "ece_change": ece_change
    }

def _calculate_update_frequency(historical_metrics: List[Dict]) -> float:
    """Calculate update frequency based on historical metrics."""
    if len(historical_metrics) < 2:
        return 0
    
    timestamps = [
        datetime.fromisoformat(m["timestamp"])
        for m in historical_metrics
    ]
    time_diffs = [
        (t2 - t1).total_seconds()
        for t1, t2 in zip(timestamps[:-1], timestamps[1:])
    ]
    return len(time_diffs) / sum(time_diffs) if time_diffs else 0

def _get_performance_stats(calibrator) -> Dict:
    """Get performance statistics from calibrator history."""
    if not hasattr(calibrator, "history") or not calibrator.history:
        return {
            "min_ece": None,
            "max_ece": None,
            "mean_ece": None,
            "std_ece": None
        }
    
    ece_values = [
        state["metrics"]["ece"]
        for state in calibrator.history
        if "metrics" in state and "ece" in state["metrics"]
    ]
    
    if not ece_values:
        return {
            "min_ece": None,
            "max_ece": None,
            "mean_ece": None,
            "std_ece": None
        }
    
    return {
        "min_ece": float(np.min(ece_values)),
        "max_ece": float(np.max(ece_values)),
        "mean_ece": float(np.mean(ece_values)),
        "std_ece": float(np.std(ece_values)) if len(ece_values) > 1 else 0.0
    }

def _calculate_distribution_drift(calibrator: OnlineCalibrator) -> float:
    """Calculate distribution drift using KS test."""
    if len(calibrator.history) < 2:
        return 0.0
    
    from scipy.stats import ks_2samp
    
    current_scores = calibrator.current_window["scores"]
    prev_state = calibrator.history[-2]
    prev_scores = np.array(prev_state["current_window"]["scores"])
    
    if len(prev_scores) == 0:
        return 0.0
    
    statistic, _ = ks_2samp(current_scores, prev_scores)
    return float(statistic)

async def _update_monitoring_metrics(
    calibrator_id: str,
    metrics: Dict,
    updated: bool
):
    """Update monitoring metrics in background."""
    MetricsCollector.update_calibration_error(
        calibrator_id,
        metrics["ece"]
    )
    
    if updated:
        MetricsCollector.record_update(calibrator_id)
    
    calibrator = streaming_calibrators[calibrator_id]
    MetricsCollector.update_version(
        calibrator_id,
        calibrator.version
    ) 