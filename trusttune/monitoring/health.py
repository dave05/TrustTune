"""Health check implementation."""
import time
from typing import Dict, Any

class HealthStatus:
    def __init__(self, status: str, details: dict):
        self.status = status
        self.details = details
        self.checks = details

class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.start_time = time.time()

    def check_system_resources(self) -> dict:
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            return {
                "memory_usage": memory.percent,
                "cpu_usage": cpu_percent,
                "disk_usage": psutil.disk_usage('/').percent
            }
        except ImportError:
            return {"error": "psutil not available"}

    def get_health_status(self) -> HealthStatus:
        health_info = {
            "uptime_seconds": time.time() - self.start_time
        }
        
        resources = self.check_system_resources()
        if "error" in resources:
            return HealthStatus("degraded", {**health_info, **resources})
            
        health_info.update(resources)
        status = "healthy"
        if resources.get("memory_usage", 0) > 90 or resources.get("cpu_usage", 0) > 90:
            status = "degraded"
            
        return HealthStatus(status, health_info)