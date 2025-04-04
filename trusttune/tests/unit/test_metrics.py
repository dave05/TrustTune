import pytest
import numpy as np
from trusttune.core.metrics import (
    expected_calibration_error,
    reliability_curve,
    calibration_drift,
    brier_score
)


def test_ece_perfect_calibration():
    # Test with perfectly calibrated predictions
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    
    ece, bin_confs, bin_accs, bin_sizes = expected_calibration_error(
        y_true, y_prob, n_bins=2
    )
    
    assert np.isclose(ece, 0.0)  # Perfect calibration
    assert np.all(
        np.isclose(bin_confs[bin_sizes > 0], bin_accs[bin_sizes > 0])
    )


def test_ece_worst_calibration():
    # Test with worst possible calibration
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    
    ece, _, _, _ = expected_calibration_error(y_true, y_prob, n_bins=2)
    assert np.isclose(ece, 1.0)


def test_reliability_curve():
    """Test reliability curve computation."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    prob_pred, prob_true = reliability_curve(y_true, y_prob, n_bins=2)
    assert len(prob_pred) == len(prob_true)
    assert all(0 <= p <= 1 for p in prob_pred)
    assert all(0 <= p <= 1 for p in prob_true)


def test_calibration_drift():
    """Test calibration drift computation."""
    old_scores = np.array([0.1, 0.4, 0.6, 0.9])
    new_scores = np.array([0.2, 0.3, 0.7, 0.8])
    drift = calibration_drift(old_scores, new_scores)
    assert isinstance(drift, float)
    assert drift >= 0


def test_calibration_drift_invalid_metric():
    scores = np.array([0.5, 0.5])
    labels = np.array([0, 1])
    
    with pytest.raises(ValueError, match="Unknown metric"):
        calibration_drift(
            scores, labels,
            scores, labels,
            metric='invalid'
        )


def test_ece_input_validation():
    # Test invalid probabilities
    with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
        expected_calibration_error(
            np.array([0, 1]),
            np.array([-0.1, 1.1])
        )
    
    # Test mismatched shapes
    with pytest.raises(ValueError, match="Shape mismatch"):
        expected_calibration_error(
            np.array([0, 1]),
            np.array([0.5])
        )
    
    # Test invalid number of bins
    with pytest.raises(ValueError, match="n_bins must be positive"):
        expected_calibration_error(
            np.array([0, 1]),
            np.array([0.5, 0.5]),
            n_bins=0
        )


def test_expected_calibration_error():
    """Test ECE computation."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    ece = expected_calibration_error(y_true, y_prob, n_bins=2)
    assert isinstance(ece, float)
    assert 0 <= ece <= 1


def test_brier_score():
    """Test Brier score computation."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    score = brier_score(y_true, y_prob)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_ece_validation():
    with pytest.raises(ValueError, match="n_bins must be positive"):
        expected_calibration_error(np.array([0, 1]), np.array([0.5, 0.5]), n_bins=0)


def test_ece_return_values():
    y_true = np.array([0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.3])
    ece, conf, acc, sizes = expected_calibration_error(
        y_true, y_prob, n_bins=2
    )
    assert isinstance(ece, float)
    assert isinstance(conf, np.ndarray)
    assert isinstance(acc, np.ndarray)
    assert isinstance(sizes, np.ndarray)
    assert 0 <= ece <= 1
    assert np.all(0 <= conf <= 1)
    assert np.all(0 <= acc <= 1)
    assert np.all(sizes >= 0)
