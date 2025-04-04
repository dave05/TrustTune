"""Base calibrator implementation."""
from typing import Dict, Any
import numpy as np
from sklearn.exceptions import NotFittedError

class BaseCalibrator:
    """Base class for all calibrators."""
    
    def __init__(self):
        self.fitted = False
    
    def _validate_inputs(self, scores: np.ndarray, labels: np.ndarray = None) -> None:
        """Validate input arrays."""
        scores = np.asarray(scores)
        
        if scores.size == 0:
            raise ValueError("Empty arrays are not allowed")
            
        if not (np.all(scores >= 0) and np.all(scores <= 1)):
            raise ValueError("Scores must be between 0 and 1")
        
        if labels is not None:
            labels = np.asarray(labels)
            
            if scores.shape[0] != labels.shape[0]:
                raise ValueError("Number of scores and labels must match")
            
            if not np.all(np.isin(labels, [0, 1])):
                raise ValueError("Labels must be binary (0 or 1)")
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                raise ValueError("At least two classes required")
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator."""
        self._validate_inputs(scores, labels)
        raise NotImplementedError
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate scores."""
        if not self.fitted:
            raise NotFittedError("Calibrator must be fitted before prediction")
        self._validate_inputs(scores)
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """Get calibrator parameters."""
        raise NotImplementedError
    
    def set_params(self, **params: Dict[str, Any]) -> None:
        """Set calibrator parameters."""
        raise NotImplementedError
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Alias for predict_proba."""
        return self.predict_proba(scores) 