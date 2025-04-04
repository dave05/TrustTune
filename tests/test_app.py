"""Test the main application."""
from fastapi.testclient import TestClient
import pytest
import numpy as np
from app import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(transport=app)

def test_home_page(client):
    """Test that the home page loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "TrustTune" in response.text

def test_calibration_endpoint(client):
    """Test the calibration endpoint."""
    # Create test data
    scores = [0.1, 0.9]
    labels = [0, 1]

    # Call the endpoint
    response = client.post(
        "/calibrate",
        json={
            "scores": scores,
            "labels": labels,
            "calibrator_type": "platt"
        }
    )

    # Check the response
    assert response.status_code == 200
    data = response.json()
    assert "calibrated_scores" in data
    assert "metrics" in data
    assert len(data["calibrated_scores"]) == len(scores)
    assert "ece" in data["metrics"]
    assert "brier_score" in data["metrics"]
