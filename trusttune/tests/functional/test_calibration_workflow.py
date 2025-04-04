"""Functional tests for calibration workflow."""
import pytest
from fastapi.testclient import TestClient
from trusttune.api.app import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(transport=app)  # Use transport instead of app

def test_end_to_end_calibration(client):
    """Test end-to-end calibration workflow."""
    # Test calibration
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

def test_end_to_end_calibration_workflow():
    # 1. Create and train a calibrator
    train_data = {
        "scores": [0.1, 0.2, 0.8, 0.9],
        "labels": [0, 0, 1, 1],
        "calibrator_type": "platt"
    }
    response = client.post("/calibrators/", json=train_data)
    assert response.status_code == 200
    calibrator_id = response.json()["calibrator_id"]

    # 2. Get calibrator info
    response = client.get(f"/calibrators/{calibrator_id}")
    assert response.status_code == 200
    info = response.json()
    assert info["calibrator_type"] == "platt"

    # 3. Get initial metrics
    response = client.get(f"/calibrators/{calibrator_id}/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["ece"] >= 0

    # 4. Calibrate new scores
    new_scores = {
        "scores": [0.3, 0.4, 0.6, 0.7]
    }
    response = client.post(
        f"/calibrators/{calibrator_id}/calibrate",
        json=new_scores
    )
    assert response.status_code == 200
    calibrated = response.json()["calibrated_scores"]
    assert len(calibrated) == 4
    assert all(0 <= score <= 1 for score in calibrated)

def test_multiple_calibrator_types():
    # Test all calibrator types
    calibrator_types = ["platt", "isotonic", "temperature"]
    train_data = {
        "scores": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        "labels": [0, 0, 0, 1, 1, 1]
    }

    for cal_type in calibrator_types:
        data = {**train_data, "calibrator_type": cal_type}
        response = client.post("/calibrators/", json=data)
        assert response.status_code == 200
        
        calibrator_id = response.json()["calibrator_id"]
        
        # Test calibration
        new_scores = {
            "scores": [0.4, 0.5, 0.6]
        }
        response = client.post(
            f"/calibrators/{calibrator_id}/calibrate",
            json=new_scores
        )
        assert response.status_code == 200

def test_error_handling():
    # Test various error conditions
    
    # 1. Invalid scores
    response = client.post("/calibrators/", json={
        "scores": [-0.1, 1.2],
        "labels": [0, 1],
        "calibrator_type": "platt"
    })
    # FastAPI's built-in validation returns 422
    assert response.status_code == 422
    assert "detail" in response.json()
    
    # 2. Invalid labels
    response = client.post("/calibrators/", json={
        "scores": [0.1, 0.9],
        "labels": [2, 3],
        "calibrator_type": "platt"
    })
    assert response.status_code == 422
    assert "detail" in response.json()
    
    # 3. Mismatched lengths
    response = client.post("/calibrators/", json={
        "scores": [0.1, 0.2, 0.3],
        "labels": [0, 1],
        "calibrator_type": "platt"
    })
    assert response.status_code == 400
    assert "error" in response.json()["detail"]
    
    # 4. Invalid calibrator type
    response = client.post("/calibrators/", json={
        "scores": [0.1, 0.9],
        "labels": [0, 1],
        "calibrator_type": "invalid"
    })
    assert response.status_code == 400
    assert "error" in response.json()["detail"]

    # 5. Invalid calibrator ID
    response = client.get("/calibrators/invalid-id")
    assert response.status_code == 404 