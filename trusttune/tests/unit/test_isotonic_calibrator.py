import pytest
import numpy as np
from sklearn.exceptions import NotFittedError

def test_isotonic_calibrator_fit(small_binary_data):
    from trusttune.core.isotonic import IsotonicCalibrator
    
    scores, labels = small_binary_data
    calibrator = IsotonicCalibrator()
    
    # Test fitting
    calibrator.fit(scores, labels)
    assert calibrator.fitted
    assert calibrator.calibrator is not None
    
    # Check monotonicity of predictions
    test_scores = np.sort(np.random.uniform(0, 1, 10))
    predictions = calibrator.predict_proba(test_scores)
    assert np.all(np.diff(predictions) >= -1e-10)


def test_isotonic_calibrator_predict(synthetic_binary_data):
    from trusttune.core.isotonic import IsotonicCalibrator
    
    scores, labels = synthetic_binary_data
    calibrator = IsotonicCalibrator()
    
    # Split data
    n_train = int(len(scores) * 0.7)
    train_scores, train_labels = scores[:n_train], labels[:n_train]
    test_scores = scores[n_train:]
    
    # Fit and predict
    calibrator.fit(train_scores, train_labels)
    predictions = calibrator.predict_proba(test_scores)
    
    # Check prediction properties
    assert predictions.shape == test_scores.shape
    assert np.all((0 <= predictions) & (predictions <= 1))


def test_isotonic_calibrator_serialization(small_binary_data):
    from trusttune.core.isotonic import IsotonicCalibrator
    
    scores, labels = small_binary_data
    calibrator = IsotonicCalibrator()
    calibrator.fit(scores, labels)
    
    # Test parameter serialization
    params = calibrator.get_params()
    assert "X_thresholds" in params
    assert "y_thresholds" in params
    
    # Test parameter loading
    new_calibrator = IsotonicCalibrator()
    new_calibrator.set_params(params)
    
    # Compare predictions
    test_scores = np.array([0.2, 0.5, 0.8])
    assert np.allclose(
        calibrator.predict_proba(test_scores),
        new_calibrator.predict_proba(test_scores)
    )


def test_isotonic_calibrator_edge_cases():
    from trusttune.core.isotonic import IsotonicCalibrator
    
    calibrator = IsotonicCalibrator()
    
    # Test with constant scores
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 1, 0, 1])
    calibrator.fit(scores, labels)
    
    # Should predict mean of labels for any input close to 0.5
    pred = calibrator.predict_proba(np.array([0.5]))
    assert np.abs(pred - 0.5) < 1e-10
    
    # Test with perfect separation
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    calibrator.fit(scores, labels)
    
    test_scores = np.array([0.0, 1.0])
    predictions = calibrator.predict_proba(test_scores)
    assert predictions[0] < 0.5  # Should predict low probability for 0.0
    assert predictions[1] > 0.5  # Should predict high probability for 1.0


def test_unfitted_isotonic_calibrator_raises():
    from trusttune.core.isotonic import IsotonicCalibrator
    
    calibrator = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fitted"):
        calibrator.predict_proba(np.array([0.5]))


def test_isotonic_calibrator_improves_calibration(synthetic_binary_data):
    from trusttune.core.isotonic import IsotonicCalibrator
    from trusttune.core.metrics import expected_calibration_error
    
    scores, labels = synthetic_binary_data
    calibrator = IsotonicCalibrator()
    
    # Split data
    n_train = int(len(scores) * 0.7)
    train_scores = scores[:n_train]
    train_labels = labels[:n_train]
    test_scores = scores[n_train:]
    test_labels = labels[n_train:]
    
    # Fit calibrator
    calibrator.fit(train_scores, train_labels)
    
    # Get calibrated probabilities
    calibrated_scores = calibrator.predict_proba(test_scores)
    
    # Compare ECE before and after calibration
    original_ece, _, _, _ = expected_calibration_error(test_labels, test_scores)
    calibrated_ece, _, _, _ = expected_calibration_error(test_labels, calibrated_scores)
    
    assert calibrated_ece < original_ece 