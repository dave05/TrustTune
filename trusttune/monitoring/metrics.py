"""Metrics collection and monitoring."""
from contextlib import contextmanager
import time
from typing import Optional, List, Dict, Any
from prometheus_client import Counter, Histogram, Gauge, REGISTRY
import numpy as np
from ..core.metrics import expected_calibration_error
from collections import defaultdict

class MetricsCollector:
    """Simple metrics collector without Prometheus dependencies."""
    
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.operation_start_times = {}
        self.request_count = 0
        self.error_count = 0
        self.latencies = []

    def start_operation(self, operation_id: str):
        self.operation_start_times[operation_id] = time.time()

    def end_operation(self, operation_id: str, success: bool = True) -> float:
        if operation_id in self.operation_start_times:
            duration = time.time() - self.operation_start_times[operation_id]
            self.latencies.append(duration)
            self.request_count += 1
            if not success:
                self.error_count += 1
            del self.operation_start_times[operation_id]
            return duration * 1000  # Convert to milliseconds
        return 0.0

    def get_operation_latency(self, operation_id: str) -> float:
        if operation_id in self.operation_start_times:
            return time.time() - self.operation_start_times[operation_id]
        return 0.0

    def get_throughput(self) -> float:
        if not self.latencies:
            return 0.0
        return len(self.latencies) / sum(self.latencies)

    def get_metrics(self):
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "average_latency": np.mean(self.latencies) if self.latencies else 0
        }

    def record_update(self, calibrator_id, version, scores, labels, calibrated_scores, window_size, operation_id):
        metrics = {
            "version": version,
            "timestamp": time.time(),
            "window_size": window_size,
            "ece": expected_calibration_error(labels, calibrated_scores)[0],
            "calibration": {
                "accuracy": float(np.mean(labels == (calibrated_scores > 0.5))),
                "mean_confidence": float(np.mean(calibrated_scores))
            },
            "performance": {
                "latency": self.get_operation_latency(operation_id),
                "throughput": self.get_throughput()
            }
        }
        self.metrics_history[calibrator_id].append(metrics)

    def get_metrics_history(self, calibrator_type: str) -> List[Dict[str, Any]]:
        """Get metrics history for a calibrator."""
        return self.metrics_history.get(calibrator_type, [])

    def record_request(self, calibrator_type: str, input_size: int):
        """Record a calibration request."""
        self.request_count += 1
    
    def record_error(self, error_type: str):
        """Record a calibration error."""
        self.error_count += 1
    
    @contextmanager
    def measure_latency(self):
        """Measure request latency."""
        start_time = time.time()
        try:
            yield
        finally:
            self.latencies.append(time.time() - start_time)

    def record_update(self, calibrator_id, version, scores, labels, calibrated_scores, window_size, operation_id):
        """Record metrics for a calibrator update."""
        metrics = {
            "version": version,
            "timestamp": time.time(),
            "window_size": window_size,
            "ece": expected_calibration_error(labels, calibrated_scores)[0],
            "calibration": {
                "accuracy": float(np.mean(labels == (calibrated_scores > 0.5))),
                "mean_confidence": float(np.mean(calibrated_scores))
            },
            "performance": {
                "latency": self.get_operation_latency(operation_id),
                "throughput": self.get_throughput()
            }
        }
        self.metrics_history[calibrator_id].append(metrics)

    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation."""
        self.operation_start_times[operation_name] = time.time()

    def end_operation(self, operation_name: str) -> float:
        """End timing an operation and return duration in milliseconds."""
        if operation_name not in self.operation_start_times:
            raise ValueError(f"Operation {operation_name} was not started")
        duration = time.time() - self.operation_start_times[operation_name]
        self.latencies.append(duration)
        return duration * 1000  # Convert to milliseconds 