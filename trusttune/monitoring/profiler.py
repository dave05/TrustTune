"""Performance profiling utilities."""
import time
from contextlib import contextmanager
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, profiler: 'Profiler'):
        self.name = name
        self.profiler = profiler
        self._start_time = None
    
    def __enter__(self):
        self._start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            duration = time.time() - self._start_time
            self.profiler.timings[self.name] = self.profiler.timings.get(self.name, 0) + duration
            self.profiler.counts[self.name] = self.profiler.counts.get(self.name, 0) + 1

class Profiler:
    """Performance profiler."""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            duration = time.time() - self._start_time
            self.timings["total"] = duration
            self.counts["total"] = 1

    def profile(self, name: str):
        """Profile a code block."""
        return Timer(name, self)

    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        return {
            name: {
                "total_time": self.timings[name],
                "count": self.counts[name],
                "average_time": self.timings[name] / self.counts[name]
            }
            for name in self.timings
        }

    def reset(self):
        """Reset profiling statistics."""
        self.timings.clear()
        self.counts.clear() 