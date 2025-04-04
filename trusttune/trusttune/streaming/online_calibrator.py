import numpy as np
from typing import Dict, Optional, Union, List, Any
from datetime import datetime, UTC
import logging
import time

from ..core.factory import create_calibrator
from ..core.metrics import expected_calibration_error, reliability_curve
from ..core.base import BaseCalibrator
from ..exceptions import CalibrationError, ValidationError, StateError
from ..utils.logging_config import setup_logger
from ..monitoring.profiler import profile, Timer
from ..monitoring.metrics import MetricsCollector
from ..monitoring.health import HealthChecker

logger = setup_logger(__name__)

class OnlineCalibrator:
    """Online calibrator that updates based on streaming data."""
    
    def __init__(
        self,
        base_calibrator: str,
        window_size: int = 1000,
        update_threshold: float = 0.1
    ):
        """Initialize online calibrator.
        
        Args:
            base_calibrator: Type of base calibrator to use
            window_size: Size of sliding window for calibration
            update_threshold: Minimum improvement in ECE required for update
            
        Raises:
            ValidationError: If parameters are invalid
        """
        try:
            if window_size <= 0:
                raise ValidationError("window_size must be positive")
            if not (0 <= update_threshold <= 1):
                raise ValidationError("update_threshold must be between 0 and 1")
            
            self.base_calibrator_type = base_calibrator
            self.window_size = window_size
            self.update_threshold = update_threshold
            self.current_calibrator = None
            self.current_window = {"scores": [], "labels": []}
            self.version = 0
            self.is_fitted = False
            self.history = []
            
            logger.info(
                "Initialized OnlineCalibrator with base_calibrator=%s, "
                "window_size=%d, update_threshold=%.3f",
                base_calibrator, window_size, update_threshold
            )
            
        except Exception as e:
            logger.error("Failed to initialize OnlineCalibrator: %s", str(e))
            raise

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the calibrator with initial data."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int32)
        
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have the same length")
        
        self.current_window = {
            "scores": scores,
            "labels": labels
        }
        
        self.current_calibrator.fit(scores, labels)
        self.is_fitted = True
        self.version = 1
        
        # Add initial state to history
        self.history.append(self.get_state())

    def _calculate_ece(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Calculate ECE for given scores and labels."""
        predictions = self.current_calibrator.predict_proba(scores)
        ece, _, _, _ = expected_calibration_error(labels, predictions)
        return ece
    
    @profile
    def update(self, scores: np.ndarray, labels: np.ndarray) -> bool:
        """Update the calibrator with new data.
        
        Args:
            scores: Uncalibrated prediction scores
            labels: True binary labels
            
        Returns:
            bool: Whether the calibrator was updated
            
        Raises:
            ValidationError: If inputs are invalid
            CalibrationError: If calibration fails
        """
        operation_id = f"update_{time.time()}"
        MetricsCollector.start_operation(operation_id)
        
        try:
            with Timer("input_validation"):
                # Input validation
                if not isinstance(scores, (np.ndarray, list)) or not isinstance(labels, (np.ndarray, list)):
                    raise ValidationError("scores and labels must be numpy arrays or lists")
                
                scores = np.asarray(scores, dtype=np.float64)
                labels = np.asarray(labels, dtype=np.int32)
                
                if len(scores) != len(labels):
                    raise ValidationError(f"scores and labels must have same length, got {len(scores)} and {len(labels)}")
                if not np.all((scores >= 0) & (scores <= 1)):
                    raise ValidationError("scores must be between 0 and 1")
                if not np.all(np.isin(labels, [0, 1])):
                    raise ValidationError("labels must be binary (0 or 1)")
            
            logger.info(
                "Processing update with %d samples (version=%d)",
                len(scores), self.version
            )
            
            with Timer("window_update"):
                # Initialize if not fitted
                if not self.is_fitted:
                    return self.fit(scores, labels)
                
                # Update window
                if isinstance(self.current_window["scores"], np.ndarray):
                    self.current_window["scores"] = np.concatenate([self.current_window["scores"], scores])
                    self.current_window["labels"] = np.concatenate([self.current_window["labels"], labels])
                else:
                    self.current_window["scores"] = scores
                    self.current_window["labels"] = labels
                
                # Keep only the most recent window_size samples
                if len(self.current_window["scores"]) > self.window_size:
                    self.current_window["scores"] = self.current_window["scores"][-self.window_size:]
                    self.current_window["labels"] = self.current_window["labels"][-self.window_size:]
            
            with Timer("calibrator_training"):
                # Train new calibrator
                new_calibrator = create_calibrator(self.base_calibrator_type)
                new_calibrator.fit(self.current_window["scores"], self.current_window["labels"])
            
            with Timer("performance_comparison"):
                # Compare performance
                old_ece = self._calculate_ece(
                    self.current_calibrator.predict_proba(self.current_window["scores"]),
                    self.current_window["labels"]
                )
                new_ece = self._calculate_ece(
                    new_calibrator.predict_proba(self.current_window["scores"]),
                    self.current_window["labels"]
                )
                
                logger.debug(
                    "Performance comparison - old_ece: %.4f, new_ece: %.4f, threshold: %.4f",
                    old_ece, new_ece, self.update_threshold
                )
                
                # Update only if new calibrator is significantly better
                if new_ece < old_ece - self.update_threshold:
                    self.current_calibrator = new_calibrator
                    self.version += 1
                    self._save_state()
                    logger.info(
                        "Updated calibrator to version %d (ECE improved by %.4f)",
                        self.version, old_ece - new_ece
                    )
                    return True
                
                logger.debug("No significant improvement, keeping current calibrator")
                return False
            
            # Record metrics
            MetricsCollector.record_update(
                calibrator_id=self.id,
                version=self.version,
                scores=scores,
                labels=labels,
                calibrated_scores=self.current_calibrator.predict_proba(scores),
                window_size=len(self.current_window["scores"]),
                operation_id=operation_id
            )
            
            # Check health
            health_status = HealthChecker.get_health_status(
                self.id,
                MetricsCollector
            )
            if health_status.status != "healthy":
                logger.warning(
                    "Health check warning: %s",
                    health_status.details
                )
            
            return True
            
        except Exception as e:
            MetricsCollector.record_update(
                calibrator_id=self.id,
                version=self.version,
                scores=scores,
                labels=labels,
                calibrated_scores=np.zeros_like(scores),
                window_size=0,
                operation_id=operation_id,
                status="failed"
            )
            logger.error(
                "Failed to update calibrator: %s",
                str(e),
                exc_info=True
            )
            raise

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities.
        
        Args:
            scores: Raw model scores to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator must be fitted before predicting")
        
        scores = np.array(scores)
        if len(scores) == 0:
            return np.array([])
            
        return self.current_calibrator.predict_proba(scores)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current calibration metrics."""
        if not self.is_fitted:
            return {
                "ece": 0.0,
                "version": self.version,
                "window_size": 0,
                "reliability_scores": [],
                "reliability_labels": []
            }
        
        # Ensure window data is numpy arrays
        scores = np.asarray(self.current_window["scores"])
        labels = np.asarray(self.current_window["labels"])
        
        if len(scores) == 0:
            return {
                "ece": 0.0,
                "version": self.version,
                "window_size": 0,
                "reliability_scores": [],
                "reliability_labels": []
            }
        
        calibrated_scores = self.current_calibrator.predict_proba(scores)
        ece = self._calculate_ece(calibrated_scores, labels)
        
        return {
            "ece": float(ece),
            "version": self.version,
            "window_size": len(scores),
            "reliability_scores": calibrated_scores.tolist(),
            "reliability_labels": labels.tolist()
        }
    
    def _save_state(self) -> None:
        """Save current state to history."""
        try:
            state = self.get_state()
            self.history.append(state)
            logger.debug("Saved state to history, version=%d", state["version"])
        except Exception as e:
            logger.error("Failed to save state: %s", str(e))
            raise StateError(f"Failed to save state: {str(e)}")
    
    def get_state(self) -> Dict:
        """Get the current state of the calibrator."""
        return {
            "base_calibrator_type": self.base_calibrator_type,
            "window_size": self.window_size,
            "update_threshold": self.update_threshold,
            "version": self.version,
            "is_fitted": self.is_fitted,
            "current_window": {
                "scores": self.current_window["scores"].tolist() if isinstance(self.current_window["scores"], np.ndarray) else [],
                "labels": self.current_window["labels"].tolist() if isinstance(self.current_window["labels"], np.ndarray) else []
            },
            "calibrator_state": self.current_calibrator.get_params() if self.current_calibrator else None,
            "metrics": self.get_metrics()
        }
    
    def set_state(self, state: Dict):
        """Set the state of the calibrator."""
        self.base_calibrator_type = state["base_calibrator_type"]
        self.window_size = state["window_size"]
        self.update_threshold = state.get("update_threshold", 0.1)
        self.version = state.get("version", 0)
        self.is_fitted = state.get("is_fitted", False)
        
        # Restore window with numpy arrays
        self.current_window = {
            "scores": np.array(state["current_window"]["scores"], dtype=np.float64),
            "labels": np.array(state["current_window"]["labels"], dtype=np.int32)
        }
        
        # Restore calibrator
        if state["calibrator_state"]:
            self.current_calibrator = create_calibrator(self.base_calibrator_type)
            self.current_calibrator.set_params(**state["calibrator_state"]) 