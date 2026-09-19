"""Rate limiting middleware."""
from fastapi import HTTPException, Request
import time
from collections import defaultdict
from typing import Dict, Tuple
from trusttune.config import settings

class RateLimiter:
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        
    def is_rate_limited(self, key: str) -> bool:
        now = time.time()
        minute_ago = now - 60
        
        # Clean old requests
        self.requests[key] = [req_time for req_time in self.requests[key] 
                            if req_time > minute_ago]
        
        # Check if rate limit is exceeded
        if len(self.requests[key]) >= settings.RATE_LIMIT_PER_MINUTE:
            return True
            
        self.requests[key].append(now)
        return False

rate_limiter = RateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    
    # Check rate limit
    if rate_limiter.is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    response = await call_next(request)
    return response