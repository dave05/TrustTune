import numpy as np
from scipy.optimize import minimize
from .base import BaseCalibrator


class TemperatureCalibrator(BaseCalibrator):
    """Temperature scaling calibration.
    
    Implements a simple scaling method that divides logits by a learned temperature
    parameter. This is equivalent to scaling the confidence of the predictions.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = None
    
    def _logits_to_probs(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Convert logits to probabilities using temperature scaling."""
        scaled_logits = logits / temperature
        exp_logits = np.exp(np.clip(scaled_logits, -100, 100))
        return exp_logits / (1 + exp_logits)
    
    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        """Convert probabilities to logits."""
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        return np.log(probs / (1 - probs))
    
    def _objective(self, temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute negative log likelihood loss."""
        if temperature <= 0:
            return float('inf')
        
        probs = self._logits_to_probs(logits, temperature)
        nll = -np.mean(
            labels * np.log(probs + 1e-12) + 
            (1 - labels) * np.log(1 - probs + 1e-12)
        )
        return nll
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'TemperatureCalibrator':
        """Fit temperature scaling using score-label pairs.
        
        Args:
            scores: Raw model scores/probabilities to calibrate
            labels: True binary labels (0 or 1)
            
        Returns:
            self: The fitted calibrator
        """
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        
        # Convert probabilities to logits
        logits = self._probs_to_logits(scores)
        
        # Find optimal temperature
        result = minimize(
            self._objective,
            x0=1.0,
            args=(logits, labels),
            method='Nelder-Mead',
            bounds=[(1e-6, None)]
        )
        
        self.temperature = result.x[0]
        self.fitted = True
        return self
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate new scores using learned temperature.
        
        Args:
            scores: Raw model scores to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        
        scores = scores.reshape(-1)
        logits = self._probs_to_logits(scores)
        return self._logits_to_probs(logits, self.temperature)
    
    def get_params(self) -> dict:
        """Get calibrator parameters for serialization."""
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before getting parameters.")
        
        return {
            "temperature": float(self.temperature)
        }
    
    def set_params(self, params: dict) -> 'TemperatureCalibrator':
        """Set calibrator parameters from serialized state."""
        self.temperature = params["temperature"]
        self.fitted = True
        return self 