import pytest
import numpy as np

def test_temperature_calibrator_fit(small_binary_data):
    from trusttune.core.temperature import TemperatureCalibrator
    
    scores, labels = small_binary_data
    calibrator = TemperatureCalibrator()
    
    # Test fitting
    calibrator.fit(scores, labels)
    assert calibrator.fitted
    assert calibrator.temperature is not None
    assert calibrator.temperature > 0  # Temperature should be positive


def test_temperature_calibrator_predict(synthetic_binary_data):
    from trusttune.core.temperature import TemperatureCalibrator
    
    scores, labels = synthetic_binary_data
    calibrator = TemperatureCalibrator()
    
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
    
    # Check that order is preserved
    assert np.allclose(
        np.argsort(test_scores),
        np.argsort(predictions)
    )


def test_temperature_calibrator_serialization(small_binary_data):
    from trusttune.core.temperature import TemperatureCalibrator
    
    scores, labels = small_binary_data
    calibrator = TemperatureCalibrator()
    calibrator.fit(scores, labels)
    
    # Test parameter serialization
    params = calibrator.get_params()
    assert "temperature" in params
    
    # Test parameter loading
    new_calibrator = TemperatureCalibrator()
    new_calibrator.set_params(params)
    
    # Compare predictions
    test_scores = np.array([0.2, 0.5, 0.8])
    assert np.allclose(
        calibrator.predict_proba(test_scores),
        new_calibrator.predict_proba(test_scores)
    )


def test_temperature_calibrator_edge_cases():
    from trusttune.core.temperature import TemperatureCalibrator
    
    calibrator = TemperatureCalibrator()
    
    # Test with extreme probabilities
    scores = np.array([0.01, 0.02, 0.98, 0.99])
    labels = np.array([0, 0, 1, 1])
    calibrator.fit(scores, labels)
    
    test_scores = np.array([0.0, 1.0])
    predictions = calibrator.predict_proba(test_scores)
    assert predictions[0] < 0.5  # Should predict low probability for 0.0
    assert predictions[1] > 0.5  # Should predict high probability for 1.0


def test_unfitted_temperature_calibrator_raises():
    from trusttune.core.temperature import TemperatureCalibrator
    
    calibrator = TemperatureCalibrator()
    with pytest.raises(RuntimeError, match="must be fitted"):
        calibrator.predict_proba(np.array([0.5]))


def test_temperature_calibrator_improves_calibration(synthetic_binary_data):
    from trusttune.core.temperature import TemperatureCalibrator
    from trusttune.core.metrics import expected_calibration_error
    
    scores, labels = synthetic_binary_data
    calibrator = TemperatureCalibrator()
    
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