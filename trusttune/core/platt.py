"""Platt scaling calibrator implementation."""
from typing import Dict, Any
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .base import BaseCalibrator

class PlattCalibrator(BaseCalibrator):
    """Platt scaling calibrator."""
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.lr = LogisticRegression()
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator using Platt scaling."""
        scores = np.asarray(scores).reshape(-1, 1)
        labels = np.asarray(labels)
        
        # Scale the scores
        X = self.scaler.fit_transform(scores)
        
        # Fit logistic regression
        self.lr.fit(X, labels)
        self.fitted = True
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate scores using fitted Platt scaling."""
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        
        scores = np.asarray(scores).reshape(-1, 1)
        X = self.scaler.transform(scores)
        return self.lr.predict_proba(X)[:, 1]
    
    def get_params(self) -> Dict[str, Any]:
        """Get calibrator parameters."""
        if not self.fitted:
            return {}
        
        # Get the underlying parameters
        a = float(self.lr.coef_[0][0])
        b = float(self.lr.intercept_[0])
        
        return {
            "a": a,
            "b": b,
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            }
        }
    
    def set_params(self, **params: Dict[str, Any]) -> None:
        """Set calibrator parameters."""
        if not params:
            return
            
        if "a" in params and "b" in params:
            self.lr.coef_ = np.array([[params["a"]]])
            self.lr.intercept_ = np.array([params["b"]])
            
        if "scaler" in params:
            self.scaler.mean_ = np.array(params["scaler"]["mean"])
            self.scaler.scale_ = np.array(params["scaler"]["scale"])
            
        self.fitted = True 