"""Test that all modules can be imported correctly."""
import pytest

def test_imports():
    """Test that all required modules can be imported."""
    # Core imports
    from trusttune.core.base import BaseCalibrator
    from trusttune.core.factory import create_calibrator
    from trusttune.core.metrics import expected_calibration_error
    
    # API imports
    from trusttune.api.app import app
    from trusttune.api.models import BatchUpdateRequest
    
    # Monitoring imports
    from trusttune.monitoring.metrics import MetricsCollector
    from trusttune.monitoring.health import HealthChecker
    from trusttune.monitoring.profiler import Profiler
    
    # Streaming imports
    from trusttune.streaming.online_calibrator import OnlineCalibrator
    
    assert True, "All imports successful" 