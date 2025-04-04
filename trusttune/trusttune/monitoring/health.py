from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Optional
import logging
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed - system health checks will be limited")

@dataclass
class HealthStatus:
    """Health check status."""
    status: str  # "healthy", "degraded", or "unhealthy"
    checks: Dict[str, bool]
    details: Dict[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

class HealthChecker:
    """System health checker."""
    
    @classmethod
    def check_system_resources(cls) -> Dict[str, bool]:
        """Check system resource usage."""
        try:
            if PSUTIL_AVAILABLE:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.Process().memory_percent()
                disk_space = psutil.disk_usage("/").percent
                
                return {
                    "cpu_usage": cpu_percent < 80,
                    "memory_usage": memory_percent < 80,
                    "disk_space": disk_space < 80
                }
            else:
                # Return basic check when psutil is not available
                return {"basic_health": True}
        except Exception as e:
            logger.error(f"Failed to check system resources: {e}")
            return {"system_resources": False}
    
    @classmethod
    def check_calibrator_health(
        cls,
        calibrator_id: str,
        metrics_collector: MetricsCollector
    ) -> Dict[str, bool]:
        """Check calibrator health."""
        try:
            metrics = metrics_collector.get_metrics_history(calibrator_id, limit=10)
            if not metrics:
                return {"calibrator_status": False}
            
            recent_failures = sum(
                1 for m in metrics
                if m.last_update_status == "failed"
            )
            recent_ece = [m.calibration.ece for m in metrics]
            
            return {
                "recent_failures": recent_failures < 3,
                "ece_stability": np.std(recent_ece) < 0.1,
                "update_frequency": len(metrics) > 0
            }
        except Exception as e:
            logger.error(f"Failed to check calibrator health: {e}")
            return {"calibrator_health": False}
    
    @classmethod
    def get_health_status(
        cls,
        calibrator_id: Optional[str] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ) -> HealthStatus:
        """Get complete health status."""
        try:
            # Check system resources
            resource_checks = cls.check_system_resources()
            
            # Check calibrator health if specified
            calibrator_checks = {}
            if calibrator_id and metrics_collector:
                calibrator_checks = cls.check_calibrator_health(
                    calibrator_id, metrics_collector
                )
            
            # Combine all checks
            all_checks = {**resource_checks, **calibrator_checks}
            
            # Determine overall status
            if all(all_checks.values()):
                status = "healthy"
            elif sum(all_checks.values()) / len(all_checks) > 0.5:
                status = "degraded"
            else:
                status = "unhealthy"
            
            # Get detailed messages
            details = {
                check: cls._get_health_message(check, passed)
                for check, passed in all_checks.items()
            }
            
            return HealthStatus(
                status=status,
                checks=all_checks,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return HealthStatus(
                status="unhealthy",
                checks={"system": False},
                details={"error": str(e)}
            )
    
    @staticmethod
    def _get_health_message(check: str, passed: bool) -> str:
        """Get detailed health check message."""
        messages = {
            "cpu_usage": {
                True: "CPU usage is normal",
                False: "High CPU usage detected"
            },
            "memory_usage": {
                True: "Memory usage is normal",
                False: "High memory usage detected"
            },
            "disk_space": {
                True: "Sufficient disk space",
                False: "Low disk space warning"
            },
            "recent_failures": {
                True: "No recent calibration failures",
                False: "Multiple calibration failures detected"
            },
            "ece_stability": {
                True: "Calibration performance is stable",
                False: "Unstable calibration performance"
            },
            "update_frequency": {
                True: "Regular updates occurring",
                False: "No recent updates detected"
            }
        }
        return messages.get(check, {}).get(passed, "Unknown check status") 