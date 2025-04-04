"""Integration tests for calibration service."""
import pytest
from fastapi.testclient import TestClient
import numpy as np
from trusttune.api.app import app
from trusttune.core.factory import create_calibrator

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def sample_training_data():
    """Generate sample training data for testing."""
    np.random.seed(42)
    scores = np.random.uniform(0, 1, 100)
    labels = (scores + np.random.normal(0, 0.1, 100) > 0.5).astype(int)
    return {
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "calibrator_type": "platt"
    }

@pytest.fixture
def trained_calibrator_id(sample_training_data):
    """Create a trained calibrator and return its ID."""
    response = client().post("/calibrators/", json=sample_training_data)
    assert response.status_code == 200
    return response.json()["calibrator_id"]

def test_create_calibrator():
    data = {
        "scores": [0.1, 0.2, 0.8, 0.9],
        "labels": [0, 0, 1, 1],
        "calibrator_type": "platt"
    }
    response = client().post("/calibrators/", json=data)
    assert response.status_code == 200
    assert "calibrator_id" in response.json()

def test_get_calibrator_info(trained_calibrator_id):
    response = client().get(f"/calibrators/{trained_calibrator_id}")
    assert response.status_code == 200
    info = response.json()
    assert info["calibrator_type"] == "platt"
    assert "creation_time" in info
    assert "metrics" in info

def test_calibrate_scores(trained_calibrator_id):
    data = {
        "scores": [0.2, 0.5, 0.8]
    }
    response = client().post(
        f"/calibrators/{trained_calibrator_id}/calibrate",
        json=data
    )
    assert response.status_code == 200
    result = response.json()
    assert "calibrated_scores" in result
    assert len(result["calibrated_scores"]) == 3
    assert all(0 <= score <= 1 for score in result["calibrated_scores"])

def test_calibrator_metrics(trained_calibrator_id):
    response = client().get(f"/calibrators/{trained_calibrator_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "ece" in metrics
    assert "reliability_scores" in metrics
    assert "reliability_labels" in metrics

def test_invalid_calibrator_type():
    data = {
        "scores": [0.1, 0.2, 0.8, 0.9],
        "labels": [0, 0, 1, 1],
        "calibrator_type": "invalid"
    }
    response = client().post("/calibrators/", json=data)
    assert response.status_code == 400
    assert "error" in response.json()["detail"]

def test_invalid_scores():
    data = {
        "scores": [-0.1, 1.2],  # Invalid probabilities
        "labels": [0, 1],
        "calibrator_type": "platt"
    }
    response = client().post("/calibrators/", json=data)
    # FastAPI's built-in validation returns 422
    assert response.status_code == 422
    assert "detail" in response.json()

def test_mismatched_data():
    data = {
        "scores": [0.1, 0.2, 0.3],
        "labels": [0, 1],  # Mismatched length
        "calibrator_type": "platt"
    }
    response = client().post("/calibrators/", json=data)
    assert response.status_code == 400
    assert "error" in response.json()["detail"]

def test_calibrate_endpoint(client):
    """Test calibration endpoint."""
    response = client.post(
        "/calibrate",
        json={
            "scores": [0.1, 0.9],
            "labels": [0, 1],
            "calibrator_type": "platt"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "calibrated_scores" in data 