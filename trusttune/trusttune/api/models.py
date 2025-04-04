from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class DriftMonitoringConfig(BaseModel):
    ece_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    reliability_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    distribution_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

class BatchUpdateRequest(BaseModel):
    scores: List[float]
    labels: List[int]
    drift_monitoring: Optional[DriftMonitoringConfig] = None
    
    @field_validator('scores')
    def validate_scores(cls, v):
        if not v:
            raise ValueError("Scores cannot be empty")
        if not all(0 <= x <= 1 for x in v):
            raise ValueError("Scores must be between 0 and 1")
        return v
    
    @field_validator('labels')
    def validate_labels(cls, v):
        if not v:
            raise ValueError("Labels cannot be empty")
        if not all(x in (0, 1) for x in v):
            raise ValueError("Labels must be binary (0 or 1)")
        return v 