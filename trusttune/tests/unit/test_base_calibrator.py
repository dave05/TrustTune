import pytest
import numpy as np
from trusttune.core.base import BaseCalibrator


class DummyCalibrator(BaseCalibrator):
    """Dummy implementation for testing BaseCalibrator."""
    
    def __init__(self):
        super().__init__()
        self.dummy_param = None
    
    def fit(self, scores, labels):
        self.dummy_param = np.mean(scores)
        self.fitted = True
        return self
    
    def predict_proba(self, scores):
        if not self.fitted:
            raise RuntimeError("Not fitted")
        return scores
    
    def get_params(self):
        return {"dummy_param": self.dummy_param}
    
    def set_params(self, params):
        self.dummy_param = params["dummy_param"]
        self.fitted = True
        return self


def test_base_calibrator_interface():
    calibrator = DummyCalibrator()
    assert not calibrator.fitted
    
    # Test fit
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 0, 1])
    calibrator.fit(scores, labels)
    assert calibrator.fitted
    
    # Test predict_proba
    new_scores = np.array([0.4, 0.5])
    predictions = calibrator.predict_proba(new_scores)
    assert np.array_equal(predictions, new_scores)
    
    # Test transform alias
    transform_output = calibrator.transform(new_scores)
    assert np.array_equal(transform_output, predictions)
    
    # Test parameter serialization
    params = calibrator.get_params()
    assert "dummy_param" in params
    assert np.isclose(params["dummy_param"], 0.2)  # mean of [0.1, 0.2, 0.3]
    
    # Test parameter loading
    new_calibrator = DummyCalibrator()
    new_calibrator.set_params(params)
    assert new_calibrator.fitted
    assert np.isclose(new_calibrator.dummy_param, params["dummy_param"])


def test_unfitted_calibrator_raises():
    calibrator = DummyCalibrator()
    with pytest.raises(RuntimeError, match="Not fitted"):
        calibrator.predict_proba(np.array([0.1, 0.2])) 

def test_validate_inputs_empty():
    calibrator = BaseCalibrator()
    with pytest.raises(ValueError, match="Empty arrays are not allowed"):
        calibrator._validate_inputs(np.array([]))

def test_validate_inputs_scores_range():
    calibrator = BaseCalibrator()
    with pytest.raises(ValueError, match="Scores must be between 0 and 1"):
        calibrator._validate_inputs(np.array([1.5]))

def test_validate_inputs_shape_mismatch():
    calibrator = BaseCalibrator()
    with pytest.raises(ValueError, match="Number of scores and labels must match"):
        calibrator._validate_inputs(np.array([0.5, 0.6]), np.array([1])) 