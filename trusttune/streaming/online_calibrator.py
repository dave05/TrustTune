"""Online calibrator implementation."""
import numpy as np
from typing import Dict, Any
from ..core.factory import create_calibrator
from ..core.metrics import expected_calibration_error

class OnlineCalibrator:
    """Online calibrator with drift detection."""
    
    def __init__(
        self,
        calibrator_type: str = None,
        base_calibrator: str = None,
        window_size: int = 1000,
        update_threshold: float = 0.05
    ):
        """Initialize online calibrator."""
        # Support both parameter names for backward compatibility
        self.calibrator_type = calibrator_type or base_calibrator or "platt"
        self.window_size = window_size
        self.update_threshold = update_threshold
        self.current_calibrator = create_calibrator(self.calibrator_type)
        self.base_calibrator = self.current_calibrator
        self.scores_window = []
        self.labels_window = []
        self.version = 0
        self.current_ece = None
        self.is_fitted = False
    
    def get_metrics(self):
        """Get current metrics."""
        return {
            "version": self.version,
            "window_size": len(self.scores_window),
            "ece": self.current_ece
        }

    def get_state(self):
        """Get calibrator state."""
        return {
            "calibrator_type": self.calibrator_type,
            "window_size": self.window_size,
            "update_threshold": self.update_threshold,
            "version": self.version,
            "current_ece": self.current_ece,
            "is_fitted": self.is_fitted,
            "base_calibrator": self.base_calibrator.get_params()
        }
    
    def update(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Update calibrator with new data."""
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        
        # Add new data to window
        self.scores_window.extend(scores.tolist())
        self.labels_window.extend(labels.tolist())
        
        # Trim window if needed
        if len(self.scores_window) > self.window_size:
            self.scores_window = self.scores_window[-self.window_size:]
            self.labels_window = self.labels_window[-self.window_size:]
        
        # Fit new calibrator
        window_scores = np.array(self.scores_window)
        window_labels = np.array(self.labels_window)
        
        new_calibrator = create_calibrator(self.calibrator_type)
        new_calibrator.fit(window_scores, window_labels)
        
        # Calculate new ECE
        new_probs = new_calibrator.predict_proba(window_scores)
        new_ece, _, _, _ = expected_calibration_error(window_labels, new_probs)
        
        # Check for drift
        if (self.current_ece is None or
            new_ece < self.current_ece - self.update_threshold):
            self.base_calibrator = new_calibrator
            self.current_ece = new_ece
            self.version += 1
        
        return {
            "version": self.version,
            "ece": float(self.current_ece) if self.current_ece is not None else None,
            "window_size": len(self.scores_window)
        }
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities."""
        if not self.base_calibrator.fitted:
            raise RuntimeError("Calibrator must be updated before prediction")
        return self.base_calibrator.predict_proba(scores)

    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """Fit the calibrator on initial data."""
        self.base_calibrator.fit(scores, labels)
        self.scores_window = list(scores)
        self.labels_window = list(labels)
        self.is_fitted = True
        return self 

    def set_state(self, state: dict):
        """Restore calibrator state."""
        self.calibrator_type = state["calibrator_type"]
        self.window_size = state["window_size"]
        self.update_threshold = state["update_threshold"]
        self.version = state["version"]
        self.current_ece = state["current_ece"]
        self.is_fitted = state["is_fitted"]
        self.base_calibrator.set_params(**state["base_calibrator"])
        return self 