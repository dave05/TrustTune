"""Base calibrator implementation."""
from typing import Dict, Any
import numpy as np

class BaseCalibrator:
    """Base class for all calibrators."""
    
    def __init__(self):
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator."""
        raise NotImplementedError
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate scores."""
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """Get calibrator parameters."""
        raise NotImplementedError
    
    def set_params(self, **params: Dict[str, Any]) -> None:
        """Set calibrator parameters."""
        raise NotImplementedError 