import pytest
import numpy as np
from trusttune.core.base import BaseCalibrator
from trusttune.core.platt import PlattCalibrator
from trusttune.core.isotonic import IsotonicCalibrator
from trusttune.core.factory import create_calibrator

def test_base_calibrator():
    """Test base calibrator interface."""
    calibrator = BaseCalibrator()
    
    with pytest.raises(NotImplementedError):
        calibrator.fit(np.array([0.1, 0.9]), np.array([0, 1]))
    
    with pytest.raises(NotImplementedError):
        calibrator.predict_proba(np.array([0.1, 0.9]))
    
    with pytest.raises(NotImplementedError):
        calibrator.get_params()
    
    with pytest.raises(NotImplementedError):
        calibrator.set_params()

def test_platt_calibrator():
    """Test Platt scaling calibrator."""
    calibrator = PlattCalibrator()
    
    # Test with simple data
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    
    # Test fitting
    calibrator.fit(scores, labels)
    assert calibrator.fitted
    
    # Test prediction
    calibrated = calibrator.predict_proba(scores)
    assert len(calibrated) == len(scores)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    
    # Test monotonicity
    assert np.all(np.diff(calibrated) >= 0)
    
    # Test params
    params = calibrator.get_params()
    assert "a" in params
    assert "b" in params
    
    # Test setting params
    new_params = {"a": 1.0, "b": 0.0}
    calibrator.set_params(**new_params)
    assert calibrator.get_params()["a"] == new_params["a"]
    assert calibrator.get_params()["b"] == new_params["b"]

def test_isotonic_calibrator():
    """Test isotonic regression calibrator."""
    calibrator = IsotonicCalibrator()
    
    # Test with simple data
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    
    # Test fitting
    calibrator.fit(scores, labels)
    assert calibrator.fitted
    
    # Test prediction
    calibrated = calibrator.predict_proba(scores)
    assert len(calibrated) == len(scores)
    assert np.all((calibrated >= 0) & (calibrated <= 1))
    
    # Test monotonicity
    assert np.all(np.diff(calibrated) >= 0)
    
    # Test params
    params = calibrator.get_params()
    assert "isotonic_regression" in params
    
    # Test setting params
    calibrator2 = IsotonicCalibrator()
    calibrator2.set_params(**params)
    np.testing.assert_array_almost_equal(
        calibrator.predict_proba(scores),
        calibrator2.predict_proba(scores)
    )

def test_factory():
    """Test calibrator factory."""
    # Test valid calibrator types
    platt = create_calibrator("platt")
    assert isinstance(platt, PlattCalibrator)
    
    isotonic = create_calibrator("isotonic")
    assert isinstance(isotonic, IsotonicCalibrator)
    
    # Test invalid calibrator type
    with pytest.raises(ValueError):
        create_calibrator("invalid_type") 