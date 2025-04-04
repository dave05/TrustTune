import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from typing import Dict, Union, List

from .base import BaseCalibrator


class PlattCalibrator(BaseCalibrator):
    """Platt scaling calibration.
    
    Implements logistic regression calibration as described in:
    Platt, John. "Probabilistic outputs for support vector machines and comparisons 
    to regularized likelihood methods."
    """
    
    def __init__(self, regularization: float = 1e-12):
        super().__init__()
        self.regularization = regularization
        self.a = None
        self.b = None
        self.scaler = StandardScaler()
        
    def _objective(self, ab: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute negative log likelihood loss with L2 regularization."""
        a, b = ab
        p = self._sigmoid(a * scores + b)
        log_likelihood = np.mean(
            labels * np.log(p + 1e-12) + (1 - labels) * np.log(1 - p + 1e-12)
        )
        reg_term = self.regularization * (a ** 2 + b ** 2)
        return -(log_likelihood - reg_term)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Compute sigmoid function safely handling overflow."""
        return 1 / (1 + np.exp(-np.clip(x, -100, 100)))
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'PlattCalibrator':
        """Fit Platt scaling using score-label pairs."""
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        
        # Scale scores for numerical stability
        scores = self.scaler.fit_transform(scores.reshape(-1, 1)).ravel()
        
        # Initialize parameters
        init_a = 1.0
        init_b = 0.0
        
        # Optimize parameters
        result = minimize(
            self._objective,
            x0=[init_a, init_b],
            args=(scores, labels),
            method='Nelder-Mead'
        )
        
        self.a, self.b = result.x
        self.fitted = True
        return self
    
    def predict_proba(self, scores: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Calibrate scores using fitted Platt scaling."""
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        
        # Convert to numpy array if needed
        if isinstance(scores, list):
            scores = np.array(scores)
        
        scores = self.scaler.transform(scores.reshape(-1, 1)).ravel()
        return 1 / (1 + np.exp(-(self.a * scores + self.b)))
    
    def get_params(self) -> Dict:
        """Get calibrator parameters."""
        return {
            "scaler_mean": self.scaler.mean_.tolist() if hasattr(self.scaler, "mean_") else None,
            "scaler_scale": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            "platt_a": float(self.a) if hasattr(self, "a") else None,
            "platt_b": float(self.b) if hasattr(self, "b") else None
        }
    
    def set_params(self, **params: Dict) -> None:
        """Set calibrator parameters."""
        if "scaler_mean" in params and "scaler_scale" in params:
            self.scaler.mean_ = np.array(params["scaler_mean"])
            self.scaler.scale_ = np.array(params["scaler_scale"])
        if "platt_a" in params and "platt_b" in params:
            self.a = params["platt_a"]
            self.b = params["platt_b"]
        
        self.fitted = True
        return self 