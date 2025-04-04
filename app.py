"""
Main FastAPI application for TrustTune
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, field_validator

# Import TrustTune components
from trusttune.core.factory import create_calibrator
from trusttune.core.metrics import expected_calibration_error, brier_score
from trusttune.api.app import app as api_app

# Create main application
app = FastAPI(
    title="TrustTune",
    description="Production-Ready ML Score Calibration",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Mount the API
app.mount("/api", api_app)

# Define models
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

# Routes
@app.get("/")
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/calibrate")
async def calibrate(request: CalibrationRequest):
    """Calibrate probability scores."""
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
        
        # Fit calibrator
        calibrator.fit(scores, labels)
        
        # Get calibrated scores
        calibrated_scores = calibrator.predict_proba(scores)
        
        # Calculate metrics
        ece, _, _, _ = expected_calibration_error(labels, scores)
        bs = brier_score(labels, scores)
        
        return {
            "calibrated_scores": calibrated_scores.tolist(),
            "metrics": {
                "ece": float(ece),
                "brier_score": float(bs)
            }
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
