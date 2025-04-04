import pytest
import numpy as np
import tempfile
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from typing import Optional, Union, Tuple

@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    np.random.seed(42)
    scores = np.random.rand(100)
    # Create labels with some noise
    labels = (scores > 0.5).astype(int)
    labels[np.random.rand(100) < 0.1] = 1 - labels[np.random.rand(100) < 0.1]  # Add 10% noise
    return scores, labels

@pytest.fixture
def perfect_binary_data():
    """Generate perfect binary classification data."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    return scores, labels

@pytest.fixture
def mlflow_tracking_uri():
    """Create temporary MLflow tracking URI."""
    with tempfile.TemporaryDirectory() as tmpdir:
        uri = f"file://{tmpdir}"
        os.environ["MLFLOW_TRACKING_URI"] = uri
        yield uri

class PlattScaling(BaseEstimator, TransformerMixin):
    """
    Platt Scaling calibration method.
    
    Transforms scores to probabilities using logistic regression.
    """
    
    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver='lbfgs')
        self._is_fitted = False
    
    def _validate_inputs(self, scores: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Validate input arrays."""
        if scores.size == 0:
            raise ValueError("Empty array provided for scores")
        
        scores_2d = scores.reshape(-1, 1)
        
        if labels is not None:
            if labels.size == 0:
                raise ValueError("Empty array provided for labels")
            if scores.shape[0] != labels.shape[0]:
                raise ValueError("Mismatched shapes between scores and labels")
            if not np.all(np.isin(labels, [0, 1])):
                raise ValueError("Labels must be binary (0 or 1)")
            
        return scores_2d, labels
    
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> 'PlattScaling':
        """
        Fit the calibrator using uncalibrated scores and true labels.
        
        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Uncalibrated scores
        labels : array-like of shape (n_samples,)
            Binary true labels (0 or 1)
            
        Returns
        -------
        self : object
            Returns self.
        """
        scores_2d, labels = self._validate_inputs(scores, labels)
        self.model.fit(scores_2d, labels)
        self._is_fitted = True
        return self
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores to calibrated probabilities.
        
        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Uncalibrated scores
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before transform")
            
        scores_2d, _ = self._validate_inputs(scores)
        return self.model.predict_proba(scores_2d)[:, 1]
    
    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fit the calibrator and transform scores in one go.
        
        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Uncalibrated scores
        labels : array-like of shape (n_samples,)
            Binary true labels (0 or 1)
            
        Returns
        -------
        array-like of shape (n_samples,)
            Calibrated probabilities
        """
        return self.fit(scores, labels).transform(scores)
