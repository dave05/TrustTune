from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from typing import List, Dict, Optional, Any
import numpy as np
from datetime import datetime, UTC
import uuid
from fastapi.encoders import jsonable_encoder, ENCODERS_BY_TYPE
from fastapi.responses import JSONResponse

from trusttune.core.factory import create_calibrator
from trusttune.core.metrics import expected_calibration_error, reliability_curve
from .streaming_routes import router as streaming_router

# Register numpy type encoders
ENCODERS_BY_TYPE[np.bool_] = bool
ENCODERS_BY_TYPE[np.integer] = int
ENCODERS_BY_TYPE[np.floating] = float
ENCODERS_BY_TYPE[np.ndarray] = lambda x: x.tolist()

app = FastAPI(title="TrustTune Calibration API")

# In-memory storage for calibrators (replace with proper database in production)
calibrators = {}
calibrator_metadata = {}

def custom_numpy_encoder(obj):
    """Custom encoder for numpy types."""
    if isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@app.middleware("http")
async def custom_json_middleware(request: Request, call_next):
    response = await call_next(request)
    
    if hasattr(response, "body"):
        try:
            import json
            body = response.body.decode()
            data = json.loads(body)
            
            def convert(obj):
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(item) for item in obj]
                else:
                    return custom_numpy_encoder(obj)
            
            converted_data = convert(data)
            response.body = json.dumps(converted_data).encode()
        except:
            pass
    return response

# Add custom encoder to FastAPI's JSON encoder
app.json_encoder = custom_numpy_encoder

class CalibrationRequest(BaseModel):
    scores: List[float]
    labels: List[int]
    calibrator_type: str

    @field_validator('scores')
    @classmethod
    def validate_scores(cls, v):
        if not all(0 <= score <= 1 for score in v):
            raise ValueError("All scores must be between 0 and 1")
        return v

    @field_validator('labels')
    @classmethod
    def validate_labels(cls, v):
        if not all(label in [0, 1] for label in v):
            raise ValueError("Labels must be binary (0 or 1)")
        return v

class CalibrationScoresRequest(BaseModel):
    scores: List[float]

    @field_validator('scores')
    @classmethod
    def validate_scores(cls, v):
        if not all(0 <= score <= 1 for score in v):
            raise ValueError("All scores must be between 0 and 1")
        return v

@app.post("/calibrators/")
async def create_calibrator_endpoint(request: CalibrationRequest):
    try:
        # Validate input lengths
        if len(request.scores) != len(request.labels):
            raise HTTPException(
                status_code=400,
                detail={"error": "Number of scores and labels must match"}
            )

        # Create and train calibrator
        try:
            calibrator = create_calibrator(request.calibrator_type)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail={"error": str(e)}
            )

        scores = np.array(request.scores)
        labels = np.array(request.labels)
        calibrator.fit(scores, labels)

        # Calculate metrics
        ece, _, _, _ = expected_calibration_error(labels, scores)
        rel_scores, rel_labels = reliability_curve(labels, scores)

        # Generate ID and store calibrator
        calibrator_id = str(uuid.uuid4())
        calibrators[calibrator_id] = calibrator
        calibrator_metadata[calibrator_id] = {
            "calibrator_type": request.calibrator_type,
            "creation_time": datetime.now(UTC).isoformat(),
            "metrics": {
                "ece": float(ece),
                "reliability_scores": rel_scores.tolist(),
                "reliability_labels": rel_labels.tolist()
            }
        }

        return {"calibrator_id": calibrator_id}

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)}
        )

@app.get("/calibrators/{calibrator_id}")
async def get_calibrator_info(calibrator_id: str):
    if calibrator_id not in calibrator_metadata:
        raise HTTPException(
            status_code=404,
            detail={"error": "Calibrator not found"}
        )
    return calibrator_metadata[calibrator_id]

@app.post("/calibrators/{calibrator_id}/calibrate")
async def calibrate_scores(calibrator_id: str, request: CalibrationScoresRequest):
    if calibrator_id not in calibrators:
        raise HTTPException(
            status_code=404,
            detail={"error": "Calibrator not found"}
        )

    try:
        scores = np.array(request.scores)
        calibrated_scores = calibrators[calibrator_id].predict_proba(scores)
        return {"calibrated_scores": calibrated_scores.tolist()}

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)}
        )

@app.get("/calibrators/{calibrator_id}/metrics")
async def get_calibrator_metrics(calibrator_id: str):
    if calibrator_id not in calibrator_metadata:
        raise HTTPException(
            status_code=404,
            detail={"error": "Calibrator not found"}
        )
    return calibrator_metadata[calibrator_id]["metrics"]

# Add after creating the FastAPI app
app.include_router(streaming_router) 