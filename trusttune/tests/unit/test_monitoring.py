import pytest
import numpy as np
from trusttune.monitoring.metrics import MetricsCollector
from trusttune.monitoring.health import HealthChecker
from trusttune.monitoring.profiler import Profiler, Timer

def test_metrics_collector_basic():
    """Test basic metrics collection functionality."""
    collector = MetricsCollector()
    
    # Test operation timing
    collector.start_operation("test_op")
    import time
    time.sleep(0.1)
    duration = collector.end_operation("test_op")
    assert duration >= 100  # At least 100ms
    
    # Test metrics recording
    scores = np.array([0.1, 0.9, 0.3])
    labels = np.array([0, 1, 0])
    calibrated = np.array([0.2, 0.8, 0.4])
    
    collector.record_update(
        "test_calibrator",
        1,
        scores,
        labels,
        calibrated,
        100,
        "test_op"
    )
    
    history = collector.get_metrics_history("test_calibrator")
    assert len(history) == 1
    assert history[0]["version"] == 1
    assert "calibration" in history[0]
    assert "performance" in history[0]

def test_health_checker_basic():
    """Test basic health checking functionality."""
    checker = HealthChecker()
    
    # Test basic health checks (should work without psutil)
    resource_checks = checker.check_system_resources()
    assert isinstance(resource_checks, dict)
    
    # Test overall health status
    status = checker.get_health_status()
    assert status.status in ["healthy", "degraded", "unhealthy"]
    assert isinstance(status.checks, dict)
    assert isinstance(status.details, dict)

def test_profiler():
    """Test profiling functionality."""
    # Test profiler context manager
    with Profiler() as p:
        # Do something
        sum(range(1000))
        
    stats = p.get_stats()
    assert stats is not None
    assert "function calls" in stats
    
    # Test timer context manager
    with Timer("test_timer") as t:
        # Do something
        sum(range(1000))
    
    assert t.end_time > t.start_time 