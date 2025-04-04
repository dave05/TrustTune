import numpy as np
import pytest
from sklearn.datasets import make_classification
from fastapi.testclient import TestClient
from httpx import AsyncClient
from trusttune.api.app import app


@pytest.fixture
def synthetic_binary_data():
    """Generate synthetic binary classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Generate synthetic probabilities
    raw_scores = 1 / (1 + np.exp(-X[:, 0]))  # Use first feature for scores
    return raw_scores, y


@pytest.fixture
def small_binary_data():
    """Small dataset for quick tests."""
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    return scores, labels


@pytest.fixture(scope="module")
def client():
    """Create a test client."""
    with TestClient(app) as client:
        yield client
