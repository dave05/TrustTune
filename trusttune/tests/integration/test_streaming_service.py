"""Integration tests for streaming service."""
import pytest
from fastapi.testclient import TestClient
from trusttune.api.app import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def stream_calibrator_id(client):
    """Create a streaming calibrator and return its ID."""
    data = {
        "calibrator_type": "platt",
        "window_size": 100,
        "update_threshold": 0.1,
        "initial_scores": [0.1, 0.2, 0.8, 0.9],
        "initial_labels": [0, 0, 1, 1]
    }
    response = client.post("/streaming/calibrators/", json=data)
    assert response.status_code == 200
    return response.json()["calibrator_id"]

def test_streaming_update(client):
    """Test streaming update endpoint."""
    # First create a calibrator
    create_response = client.post(
        "/streaming/create",
        json={
            "calibrator_type": "platt",
            "window_size": 1000
        }
    )
    assert create_response.status_code == 200
    calibrator_id = create_response.json()["calibrator_id"]
    
    # Then test update
    update_response = client.post(
        f"/streaming/{calibrator_id}/update",
        json={
            "scores": [0.1, 0.9],
            "labels": [0, 1]
        }
    )
    assert update_response.status_code == 200
    data = update_response.json()
    assert "version" in data

def test_create_streaming_calibrator():
    data = {
        "calibrator_type": "platt",
        "window_size": 100,
        "update_threshold": 0.1,
        "initial_scores": [0.1, 0.2, 0.8, 0.9],
        "initial_labels": [0, 0, 1, 1]
    }
    response = client.post("/streaming/calibrators/", json=data)
    assert response.status_code == 200
    assert "calibrator_id" in response.json()

def test_update_streaming_calibrator(stream_calibrator_id):
    data = {
        "scores": [0.3, 0.4, 0.6, 0.7],
        "labels": [0, 0, 1, 1]
    }
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/update",
        json=data
    )
    assert response.status_code == 200
    result = response.json()
    assert "updated" in result
    assert "current_metrics" in result

def test_get_streaming_calibrator_metrics(stream_calibrator_id):
    response = client.get(f"/streaming/calibrators/{stream_calibrator_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "ece" in metrics
    assert "version" in metrics
    assert "update_history" in metrics

def test_streaming_calibrate_scores(stream_calibrator_id):
    data = {
        "scores": [0.2, 0.5, 0.8]
    }
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/calibrate",
        json=data
    )
    assert response.status_code == 200
    result = response.json()
    assert "calibrated_scores" in result
    assert len(result["calibrated_scores"]) == 3

def test_streaming_calibrator_version_history(stream_calibrator_id):
    # First update
    update_data = {
        "scores": [0.1, 0.2, 0.8, 0.9],
        "labels": [1, 1, 0, 0]  # Deliberately inverse relationship
    }
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/update",
        json=update_data
    )
    assert response.status_code == 200
    
    # Get version history
    response = client.get(f"/streaming/calibrators/{stream_calibrator_id}/versions")
    assert response.status_code == 200
    versions = response.json()
    assert "versions" in versions
    assert len(versions["versions"]) > 0

def test_invalid_streaming_calibrator_params():
    data = {
        "calibrator_type": "invalid",
        "window_size": 100,
        "update_threshold": 0.1,
        "initial_scores": [0.1, 0.2],
        "initial_labels": [0, 1]
    }
    response = client.post("/streaming/calibrators/", json=data)
    assert response.status_code == 400
    assert "detail" in response.json() 