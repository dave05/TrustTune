import pytest
import numpy as np
from trusttune.streaming.online_calibrator import OnlineCalibrator
from trusttune.core.platt import PlattCalibrator

@pytest.fixture
def sample_stream_data():
    """Generate sample streaming data for testing."""
    np.random.seed(42)
    n_samples = 1000
    scores = np.random.uniform(0, 1, n_samples)
    labels = (scores + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    return scores, labels

def test_online_calibrator_initialization():
    calibrator = OnlineCalibrator(
        base_calibrator="platt",
        window_size=100,
        update_threshold=0.1
    )
    assert calibrator.window_size == 100
    assert calibrator.update_threshold == 0.1
    assert not calibrator.is_fitted
    assert isinstance(calibrator.current_calibrator, PlattCalibrator)

def test_online_calibrator_update(sample_stream_data):
    scores, labels = sample_stream_data
    calibrator = OnlineCalibrator(
        base_calibrator="platt",
        window_size=100
    )
    
    # Initial fit
    initial_scores = scores[:100]
    initial_labels = labels[:100]
    calibrator.fit(initial_scores, initial_labels)
    assert calibrator.is_fitted
    
    # Update with new data
    new_scores = scores[100:200]
    new_labels = labels[100:200]
    metrics_before = calibrator.get_metrics()
    
    calibrator.update(new_scores, new_labels)
    metrics_after = calibrator.get_metrics()
    
    assert "ece" in metrics_before
    assert "ece" in metrics_after

def test_online_calibrator_predict():
    # Create synthetic data
    np.random.seed(42)
    scores = np.random.uniform(0, 1, 200)
    labels = (scores + np.random.normal(0, 0.1, 200) > 0.5).astype(int)
    
    calibrator = OnlineCalibrator(
        base_calibrator="platt",
        window_size=100
    )
    
    # Fit on first batch
    calibrator.fit(scores[:100], labels[:100])
    
    # Predict
    predictions = calibrator.predict_proba(np.array([0.3, 0.7]))
    assert len(predictions) == 2
    assert np.all((0 <= predictions) & (predictions <= 1))
    
    # Update and predict again
    calibrator.update(scores[100:], labels[100:])
    new_predictions = calibrator.predict_proba(np.array([0.3, 0.7]))
    assert len(new_predictions) == 2
    assert np.all((0 <= new_predictions) & (new_predictions <= 1))

def test_online_calibrator_drift_detection():
    np.random.seed(42)
    
    # Generate initial well-calibrated data
    scores1 = np.random.uniform(0, 1, 100)
    labels1 = (scores1 + np.random.normal(0, 0.1, 100) > 0.5).astype(int)
    
    # Generate drifted data (inverse relationship)
    scores2 = np.random.uniform(0, 1, 100)
    labels2 = (1 - scores2 + np.random.normal(0, 0.1, 100) > 0.5).astype(int)
    
    calibrator = OnlineCalibrator(
        base_calibrator="platt",
        window_size=100,
        update_threshold=0.1
    )
    
    # Initial fit
    calibrator.fit(scores1, labels1)
    initial_version = calibrator.version
    
    # Update with drifted data
    calibrator.update(scores2, labels2)
    
    # Version should have changed due to drift
    assert calibrator.version > initial_version

def test_online_calibrator_no_update_needed():
    np.random.seed(42)
    
    # Generate consistent data
    scores1 = np.random.uniform(0, 1, 100)
    labels1 = (scores1 + np.random.normal(0, 0.05, 100) > 0.5).astype(int)
    
    # Generate similar data
    scores2 = np.random.uniform(0, 1, 100)
    labels2 = (scores2 + np.random.normal(0, 0.05, 100) > 0.5).astype(int)
    
    calibrator = OnlineCalibrator(
        base_calibrator="platt",
        window_size=100,
        update_threshold=0.2  # High threshold
    )
    
    # Initial fit
    calibrator.fit(scores1, labels1)
    initial_version = calibrator.version
    
    # Update with similar data
    calibrator.update(scores2, labels2)
    
    # Version should not change as data is similar
    assert calibrator.version == initial_version

def test_online_calibrator_serialization():
    calibrator = OnlineCalibrator(
        base_calibrator="platt",
        window_size=100
    )
    
    # Fit with some data
    scores = np.random.uniform(0, 1, 100)
    labels = (scores > 0.5).astype(int)
    calibrator.fit(scores, labels)
    
    # Get state
    state = calibrator.get_state()
    
    # Create new calibrator and restore state
    new_calibrator = OnlineCalibrator(
        base_calibrator="platt",
        window_size=100
    )
    new_calibrator.set_state(state)
    
    # Compare predictions
    test_scores = np.array([0.3, 0.7])
    assert np.allclose(
        calibrator.predict_proba(test_scores),
        new_calibrator.predict_proba(test_scores)
    ) 