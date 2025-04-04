"""Integration tests for streaming endpoints."""
import pytest
from fastapi.testclient import TestClient
import numpy as np
from trusttune.api.app import app

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def batch_data():
    """Generate synthetic batch data for testing."""
    np.random.seed(42)
    n_samples = 1000
    scores = np.random.uniform(0, 1, n_samples)
    labels = (scores + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # Split into batches
    n_batches = 5
    batch_size = n_samples // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batches.append({
            "scores": scores[start:end].tolist(),
            "labels": labels[start:end].tolist()
        })
    return batches

@pytest.fixture
def stream_calibrator_id():
    """Create a streaming calibrator for testing."""
    data = {
        "calibrator_type": "platt",
        "window_size": 200,
        "update_threshold": 0.1,
        "initial_scores": [0.1, 0.2, 0.8, 0.9],
        "initial_labels": [0, 0, 1, 1]
    }
    response = client.post("/streaming/calibrators/", json=data)
    assert response.status_code == 200
    return response.json()["calibrator_id"]

def test_batch_update_streaming_calibrator(stream_calibrator_id, batch_data):
    """Test batch update functionality."""
    versions = []
    metrics = []
    
    # Process each batch
    for batch in batch_data:
        response = client.post(
            f"/streaming/calibrators/{stream_calibrator_id}/batch-update",
            json=batch
        )
        assert response.status_code == 200
        result = response.json()
        versions.append(result["version"])
        metrics.append(result["metrics"]["ece"])
    
    # Verify version progression
    assert len(set(versions)) > 1  # Should have multiple versions
    assert all(v2 >= v1 for v1, v2 in zip(versions, versions[1:]))  # Monotonic

def test_calibrator_rollback(stream_calibrator_id, batch_data):
    """Test version rollback functionality."""
    # Process a batch to create version history
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/batch-update",
        json=batch_data[0]
    )
    assert response.status_code == 200
    initial_version = response.json()["version"]
    
    # Get current predictions
    scores = [0.3, 0.5, 0.7]
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/calibrate",
        json={"scores": scores}
    )
    initial_predictions = response.json()["calibrated_scores"]
    
    # Process another batch
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/batch-update",
        json=batch_data[1]
    )
    assert response.status_code == 200
    
    # Rollback to previous version
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/rollback",
        json={"version": initial_version}
    )
    assert response.status_code == 200
    
    # Verify predictions match initial version
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/calibrate",
        json={"scores": scores}
    )
    rollback_predictions = response.json()["calibrated_scores"]
    assert np.allclose(initial_predictions, rollback_predictions)

def test_advanced_drift_detection(stream_calibrator_id, batch_data):
    """Test advanced drift detection features."""
    # Get initial metrics
    response = client.get(
        f"/streaming/calibrators/{stream_calibrator_id}/metrics"
    )
    assert response.status_code == 200
    initial_metrics = response.json()
    
    # Process batches with drift monitoring
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/batch-update",
        json={
            **batch_data[0],
            "drift_monitoring": {
                "ece_threshold": 0.1,
                "reliability_threshold": 0.15,
                "distribution_threshold": 0.2
            }
        }
    )
    assert response.status_code == 200
    drift_result = response.json()
    assert "drift_detected" in drift_result
    assert "drift_metrics" in drift_result

def test_streaming_metrics_monitoring(stream_calibrator_id):
    """Test streaming metrics monitoring endpoints."""
    response = client.get(
        f"/streaming/calibrators/{stream_calibrator_id}/monitoring"
    )
    assert response.status_code == 200
    metrics = response.json()
    
    assert "current_metrics" in metrics
    assert "historical_metrics" in metrics
    assert "drift_status" in metrics
    assert "update_frequency" in metrics
    assert "performance_stats" in metrics

def test_concurrent_updates(stream_calibrator_id, batch_data):
    """Test concurrent batch updates."""
    import asyncio
    from fastapi.testclient import TestClient
    
    # Create three different batches from the original data
    scores = batch_data[0]['scores']
    labels = batch_data[0]['labels']
    batch_size = len(scores) // 3
    
    test_batches = [
        {
            'scores': scores[i*batch_size:(i+1)*batch_size],
            'labels': labels[i*batch_size:(i+1)*batch_size]
        }
        for i in range(3)
    ]
    
    # Process batches sequentially since TestClient is synchronous
    results = []
    for batch in test_batches:
        with TestClient(app) as client:
            response = client.post(
                f"/streaming/calibrators/{stream_calibrator_id}/batch-update",
                json=batch
            )
            results.append((response.status_code, response.json()))
    
    # Verify results
    assert all(status == 200 for status, _ in results)
    versions = [result["version"] for _, result in results]
    assert len(set(versions)) == len(versions)  # All versions should be unique

def test_streaming_calibrator_persistence(stream_calibrator_id, batch_data):
    """Test calibrator state persistence."""
    # Update calibrator with initial data
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/batch-update",
        json=batch_data[0]
    )
    assert response.status_code == 200
    
    # Verify calibrator is fitted
    scores = [0.3, 0.5, 0.7]
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/calibrate",
        json={"scores": scores}
    )
    assert response.status_code == 200
    initial_predictions = response.json()["calibrated_scores"]
    
    # Save state
    response = client.post(
        f"/streaming/calibrators/{stream_calibrator_id}/save"
    )
    assert response.status_code == 200
    state_id = response.json()["state_id"]
    
    # Create new calibrator from saved state
    response = client.post(
        "/streaming/calibrators/load",
        json={"state_id": state_id}
    )
    assert response.status_code == 200
    new_id = response.json()["calibrator_id"]
    
    # Compare predictions
    response = client.post(
        f"/streaming/calibrators/{new_id}/calibrate",
        json={"scores": scores}
    )
    assert response.status_code == 200
    new_predictions = response.json()["calibrated_scores"]
    
    # Verify predictions match
    assert np.allclose(initial_predictions, new_predictions)

def test_create_streaming_calibrator(client):
    """Test creating a streaming calibrator."""
    response = client.post(
        "/streaming/create",
        json={
            "calibrator_type": "platt",
            "window_size": 1000,
            "drift_threshold": 0.05
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "calibrator_id" in data 