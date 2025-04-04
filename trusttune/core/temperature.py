"""Temperature scaling calibrator."""
import numpy as np
from scipy.optimize import minimize
from .base import BaseCalibrator

class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling calibration."""
    
    def __init__(self):
        super().__init__()
        self.temperature = 1.0
    
    def _objective(self, temperature: float, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute negative log likelihood."""
        scaled_scores = 1 / (1 + np.exp(-scores / temperature))
        loss = -np.mean(
            labels * np.log(scaled_scores + 1e-7) +
            (1 - labels) * np.log(1 - scaled_scores + 1e-7)
        )
        return loss
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit temperature scaling."""
        self._validate_inputs(scores, labels)
        
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        
        result = minimize(
            self._objective,
            x0=1.0,
            args=(scores, labels),
            method='BFGS'
        )
        
        self.temperature = result.x[0]
        self.fitted = True
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
            
        scores = np.asarray(scores)
        self._validate_inputs(scores)
        
        return 1 / (1 + np.exp(-scores / self.temperature))
    
    def get_params(self) -> dict:
        """Get calibrator parameters."""
        return {"temperature": float(self.temperature)}
    
    def set_params(self, **params):
        """Set calibrator parameters."""
        if "temperature" in params:
            self.temperature = float(params["temperature"])
            self.fitted = True
        return self