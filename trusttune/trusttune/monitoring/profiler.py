import cProfile
import pstats
import io
import time
import logging
import functools
from typing import Optional, Callable
from ..utils.logging_config import setup_logger

logger = setup_logger(__name__)

class Profiler:
    """Performance profiler for code sections."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiler = cProfile.Profile()
        self.stats = None
    
    def __enter__(self):
        if self.enabled:
            self.profiler.enable()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.profiler.disable()
            s = io.StringIO()
            stats = pstats.Stats(self.profiler, stream=s)
            stats.sort_stats('cumulative')
            stats.print_stats()
            self.stats = s.getvalue()
    
    def get_stats(self) -> Optional[str]:
        """Get profiling statistics."""
        return self.stats

def profile(func: Callable) -> Callable:
    """Decorator to profile a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with Profiler() as p:
            result = func(*args, **kwargs)
            stats = p.get_stats()
            if stats:
                logger.debug(
                    "Performance profile for %s:\n%s",
                    func.__name__, stats
                )
        return result
    return wrapper

class Timer:
    """Context manager for timing code sections."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = (self.end_time - self.start_time) * 1000
        logger.debug(
            "Timer '%s': %.2f ms",
            self.name, duration
        ) 