import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import interp1d
from .base import BaseCalibrator


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration.
    
    Implements a non-parametric calibration method using isotonic regression,
    which fits a non-decreasing function to map uncalibrated scores to
    calibrated probabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.calibrator = None
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'IsotonicCalibrator':
        """Fit isotonic regression using score-label pairs.
        
        Args:
            scores: Raw model scores/probabilities to calibrate
            labels: True binary labels (0 or 1)
            
        Returns:
            self: The fitted calibrator
        """
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)
        
        # Initialize and fit isotonic regression
        self.calibrator = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds='clip'
        )
        self.calibrator.fit(scores, labels)
        self.fitted = True
        
        return self
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Calibrate new scores using fitted isotonic regression.
        
        Args:
            scores: Raw model scores to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before predicting.")
        
        scores = scores.reshape(-1)
        return self.calibrator.predict(scores)
    
    def get_params(self) -> dict:
        """Get calibrator parameters for serialization."""
        if not self.fitted:
            raise RuntimeError("Calibrator must be fitted before getting parameters.")
        
        return {
            "X_thresholds": self.calibrator.X_thresholds_.tolist(),
            "y_thresholds": self.calibrator.y_thresholds_.tolist(),
            "X_min": float(self.calibrator.X_min_),
            "X_max": float(self.calibrator.X_max_)
        }
    
    def set_params(self, params: dict) -> 'IsotonicCalibrator':
        """Set calibrator parameters from serialized state."""
        self.calibrator = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds='clip'
        )
        X_thresholds = np.array(params["X_thresholds"])
        y_thresholds = np.array(params["y_thresholds"])
        
        self.calibrator.X_thresholds_ = X_thresholds
        self.calibrator.y_thresholds_ = y_thresholds
        self.calibrator.X_min_ = params["X_min"]
        self.calibrator.X_max_ = params["X_max"]
        
        # Create the interpolation function
        self.calibrator.f_ = interp1d(
            X_thresholds,
            y_thresholds,
            kind='linear',
            bounds_error=False,
            fill_value=(0, 1)
        )
        
        self.fitted = True
        return self 