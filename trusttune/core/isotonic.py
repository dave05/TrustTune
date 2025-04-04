"""Isotonic regression calibrator implementation."""
from typing import Dict, Any
import numpy as np
from sklearn.isotonic import IsotonicRegression
from .base import BaseCalibrator

class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibrator."""
    
    def __init__(self):
        super().__init__()
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator using isotonic regression."""
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        
        self.calibrator.fit(scores, labels)
        self.fitted = True
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate scores using fitted isotonic regression."""
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        
        scores = np.asarray(scores)
        return self.calibrator.predict(scores)
    
    def get_params(self) -> Dict[str, Any]:
        """Get calibrator parameters."""
        if not self.fitted:
            return {}
        
        return {
            "X_thresholds": self.calibrator.X_thresholds_.tolist(),
            "y_thresholds": self.calibrator.y_thresholds_.tolist(),
            "increasing": bool(self.calibrator.increasing_)
        }
    
    def set_params(self, **params):
        """Set calibrator parameters."""
        if "X_thresholds" in params and "y_thresholds" in params:
            self.calibrator.X_thresholds_ = np.array(params["X_thresholds"])
            self.calibrator.y_thresholds_ = np.array(params["y_thresholds"])
            self.fitted = True
        return self