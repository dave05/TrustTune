"""Integration tests for the FastAPI application."""
import pytest
from fastapi.testclient import TestClient
import numpy as np
from trusttune.api.app import app

@pytest.fixture
def client():
    """Create a test client."""
    from fastapi.testclient import TestClient
    from trusttune.api.app import app
    return TestClient(app)

def test_calibration_endpoint(client):
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
    assert "metrics" in data

def test_calibration_endpoint_invalid_scores():
    """Test calibration request with invalid scores."""
    response = client.post(
        "/calibrate",
        json={
            "scores": [1.5, -0.2],
            "labels": [0, 1],
            "calibrator_type": "platt"
        }
    )
    assert response.status_code == 400
    assert "Scores must be between 0 and 1" in response.json()["detail"]

def test_calibration_endpoint_invalid_labels():
    """Test calibration request with invalid labels."""
    response = client.post(
        "/calibrate",
        json={
            "scores": [0.1, 0.9],
            "labels": [2, 3],
            "calibrator_type": "platt"
        }
    )
    assert response.status_code == 400
    assert "Labels must be binary" in response.json()["detail"]

def test_calibration_endpoint_invalid_calibrator():
    """Test calibration request with invalid calibrator type."""
    response = client.post(
        "/calibrate",
        json={
            "scores": [0.1, 0.9],
            "labels": [0, 1],
            "calibrator_type": "invalid"
        }
    )
    assert response.status_code == 400
