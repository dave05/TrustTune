from prometheus_client import Counter, Gauge, Histogram
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Optional
import numpy as np
import logging
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed - system metrics will be limited")

# Calibration request metrics
CALIBRATION_REQUESTS = Counter(
    'calibration_requests_total',
    'Total number of calibration requests',
    ['calibrator_type', 'endpoint']
)

# Calibration latency
CALIBRATION_LATENCY = Histogram(
    'calibration_latency_seconds',
    'Latency of calibration operations',
    ['calibrator_type', 'operation']
)

# Calibration error metrics
CALIBRATION_ERROR = Gauge(
    'calibration_error',
    'Current calibration error (ECE)',
    ['calibrator_id']
)

# Update metrics
CALIBRATOR_UPDATES = Counter(
    'calibrator_updates_total',
    'Total number of calibrator updates',
    ['calibrator_id']
)

# Version metrics
CALIBRATOR_VERSION = Gauge(
    'calibrator_version',
    'Current version of the calibrator',
    ['calibrator_id']
)

@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    processing_time_ms: float
    memory_used_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None

@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    ece: float
    reliability_score: float
    sharpness_score: float
    brier_score: float
    auc_score: float

@dataclass
class DriftMetrics:
    """Drift detection metrics."""
    ece_drift: float
    distribution_drift: float
    reliability_drift: float
    concept_drift_score: float
    data_quality_score: float

@dataclass
class MetricsSnapshot:
    """Complete metrics snapshot."""
    calibrator_id: str
    version: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    performance: PerformanceMetrics = field(default_factory=lambda: PerformanceMetrics(0))
    calibration: CalibrationMetrics = field(default_factory=lambda: CalibrationMetrics(0, 0, 0, 0, 0))
    drift: DriftMetrics = field(default_factory=lambda: DriftMetrics(0, 0, 0, 0, 0))
    window_size: int = 0
    update_count: int = 0
    failed_updates: int = 0
    last_update_status: str = "none"

class MetricsCollector:
    """Enhanced metrics collector with performance monitoring."""
    
    _metrics: Dict[str, List[MetricsSnapshot]] = {}
    _start_times: Dict[str, float] = {}
    
    @classmethod
    def start_operation(cls, operation_id: str) -> None:
        """Start timing an operation."""
        cls._start_times[operation_id] = time.time()
    
    @classmethod
    def end_operation(cls, operation_id: str) -> float:
        """End timing an operation and return duration in ms."""
        if operation_id in cls._start_times:
            duration = (time.time() - cls._start_times.pop(operation_id)) * 1000
            return duration
        return 0.0
    
    @classmethod
    def get_system_metrics(cls) -> PerformanceMetrics:
        """Get current system performance metrics."""
        try:
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent()
                memory = psutil.Process().memory_info()
                memory_percent = psutil.Process().memory_percent()
                
                return PerformanceMetrics(
                    processing_time_ms=0,  # Will be updated later
                    memory_used_mb=memory.rss / (1024 * 1024),
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent
                )
            else:
                return PerformanceMetrics(processing_time_ms=0)
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return PerformanceMetrics(processing_time_ms=0)

    @classmethod
    def calculate_calibration_metrics(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
        calibrated_scores: np.ndarray
    ) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics."""
        try:
            from sklearn.metrics import brier_score_loss, roc_auc_score
            
            # Calculate reliability (calibration curve)
            reliability = np.mean(np.abs(calibrated_scores - labels))
            
            # Calculate sharpness (confidence distribution)
            sharpness = np.std(calibrated_scores)
            
            return CalibrationMetrics(
                ece=cls._calculate_ece(calibrated_scores, labels),
                reliability_score=1 - reliability,
                sharpness_score=sharpness,
                brier_score=brier_score_loss(labels, calibrated_scores),
                auc_score=roc_auc_score(labels, calibrated_scores)
            )
        except Exception as e:
            logger.error(f"Failed to calculate calibration metrics: {e}")
            return CalibrationMetrics(0, 0, 0, 0, 0)

    @classmethod
    def record_update(
        cls,
        calibrator_id: str,
        version: int,
        scores: np.ndarray,
        labels: np.ndarray,
        calibrated_scores: np.ndarray,
        window_size: int,
        operation_id: str,
        status: str = "success"
    ) -> None:
        """Record comprehensive metrics for a calibrator update."""
        try:
            if calibrator_id not in cls._metrics:
                cls._metrics[calibrator_id] = []
            
            # Get performance metrics
            performance = cls.get_system_metrics()
            performance.processing_time_ms = cls.end_operation(operation_id)
            
            # Calculate calibration metrics
            calibration = cls.calculate_calibration_metrics(
                scores, labels, calibrated_scores
            )
            
            # Calculate drift metrics
            drift = cls.calculate_drift_metrics(
                calibrator_id, scores, labels, calibrated_scores
            )
            
            metrics = MetricsSnapshot(
                calibrator_id=calibrator_id,
                version=version,
                performance=performance,
                calibration=calibration,
                drift=drift,
                window_size=window_size,
                update_count=len(cls._metrics[calibrator_id]) + 1,
                failed_updates=sum(
                    1 for m in cls._metrics[calibrator_id]
                    if m.last_update_status == "failed"
                ),
                last_update_status=status
            )
            
            cls._metrics[calibrator_id].append(metrics)
            
            # Log metrics summary
            logger.info(
                "Metrics recorded for calibrator %s: "
                "version=%d, ece=%.4f, processing_time=%.2fms, "
                "memory_used=%.2fMB, status=%s",
                calibrator_id, version, calibration.ece,
                performance.processing_time_ms,
                performance.memory_used_mb, status
            )
            
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
            raise

    @classmethod
    def get_metrics_history(
        cls,
        calibrator_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get metrics history for a calibrator."""
        try:
            if calibrator_id not in cls._metrics:
                return []
            
            metrics = cls._metrics[calibrator_id]
            if limit:
                metrics = metrics[-limit:]
            
            return [
                {
                    "version": m.version,
                    "timestamp": m.timestamp.isoformat(),
                    "ece": m.ece,
                    "window_size": m.window_size,
                    "update_count": m.update_count,
                    "drift_detected": m.drift_detected,
                    "drift_metrics": m.drift_metrics
                }
                for m in metrics
            ]
            
        except Exception as e:
            logger.error(
                "Failed to get metrics history for calibrator %s: %s",
                calibrator_id, str(e)
            )
            raise

    @staticmethod
    def record_request(calibrator_type: str, endpoint: str):
        CALIBRATION_REQUESTS.labels(
            calibrator_type=calibrator_type,
            endpoint=endpoint
        ).inc()
    
    @staticmethod
    def record_latency(calibrator_type: str, operation: str):
        return CALIBRATION_LATENCY.labels(
            calibrator_type=calibrator_type,
            operation=operation
        ).time()
    
    @staticmethod
    def update_calibration_error(calibrator_id: str, error: float):
        CALIBRATION_ERROR.labels(
            calibrator_id=calibrator_id
        ).set(error)
    
    @staticmethod
    def record_update(calibrator_id: str):
        CALIBRATOR_UPDATES.labels(
            calibrator_id=calibrator_id
        ).inc()
    
    @staticmethod
    def update_version(calibrator_id: str, version: int):
        CALIBRATOR_VERSION.labels(
            calibrator_id=calibrator_id
        ).set(version)
