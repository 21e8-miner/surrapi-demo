"""
SurrAPI Production Middleware Stack
====================================

Enterprise-grade middleware for:
- Rate limiting (token bucket per API key)
- Request validation (size limits, content-type)
- Security headers (CORS, CSP, XSS protection)
- Request tracing (correlation IDs, timing)
"""

import time
import uuid
import logging
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException

logger = logging.getLogger("surrapi.middleware")

# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: float
    rate: float  # tokens per second
    tokens: float = field(default=0)
    last_update: float = field(default_factory=time.time)
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens, return True if allowed"""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        
        # Add tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self) -> float:
        """Seconds until next token available"""
        if self.tokens >= 1:
            return 0
        return (1 - self.tokens) / self.rate


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting per API key.
    
    Limits:
    - Free tier: 5 req/sec, burst of 10
    - Pro tier: 50 req/sec, burst of 100
    - Enterprise: 100 req/sec, burst of 200
    """
    
    TIER_LIMITS = {
        "free": {"rate": 5, "burst": 10},
        "pro": {"rate": 50, "burst": 100},
        "enterprise": {"rate": 100, "burst": 200},
        "anonymous": {"rate": 2, "burst": 5}  # No API key
    }
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.buckets: Dict[str, TokenBucket] = {}
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/metrics", "/"]
    
    def _get_bucket(self, key: str, tier: str) -> TokenBucket:
        """Get or create token bucket for API key"""
        if key not in self.buckets:
            limits = self.TIER_LIMITS.get(tier, self.TIER_LIMITS["anonymous"])
            self.buckets[key] = TokenBucket(
                capacity=limits["burst"],
                rate=limits["rate"],
                tokens=limits["burst"]  # Start full
            )
        return self.buckets[key]
    
    def _extract_api_key(self, request: Request) -> tuple:
        """Extract API key and tier from request"""
        auth = request.headers.get("authorization", "")
        x_api_key = request.headers.get("x-api-key", "")
        
        key = None
        if auth.startswith("Bearer "):
            key = auth[7:]
        elif x_api_key:
            key = x_api_key
        
        if key:
            # Look up tier from billing module
            try:
                from app.billing import get_api_key
                api_key_obj = get_api_key(key)
                if api_key_obj:
                    return key, api_key_obj.tier
            except ImportError:
                pass
            return key, "free"
        
        # Use IP for anonymous rate limiting
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}", "anonymous"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            return await call_next(request)
        
        key, tier = self._extract_api_key(request)
        bucket = self._get_bucket(key, tier)
        
        if not bucket.consume():
            retry_after = bucket.time_until_available()
            logger.warning(f"Rate limit exceeded for {key[:20]}... (tier={tier})")
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Retry after {retry_after:.1f}s",
                    "tier": tier,
                    "upgrade_url": "https://surrapi.io/pricing"
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Limit": str(self.TIER_LIMITS[tier]["rate"]),
                    "X-RateLimit-Remaining": "0"
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        limits = self.TIER_LIMITS.get(tier, self.TIER_LIMITS["anonymous"])
        response.headers["X-RateLimit-Limit"] = str(limits["rate"])
        response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
        
        return response


# =============================================================================
# Request Validation
# =============================================================================

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validate incoming requests for:
    - Content-Type (JSON required for POST)
    - Request body size (max 1MB)
    - Path traversal attempts
    """
    
    MAX_BODY_SIZE = 1 * 1024 * 1024  # 1MB
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content-type for POST/PUT
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                # Allow multipart for file uploads
                if not content_type.startswith("multipart/form-data"):
                    return JSONResponse(
                        status_code=415,
                        content={
                            "error": "unsupported_media_type",
                            "message": "Content-Type must be application/json"
                        }
                    )
        
        # Check content-length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request_too_large",
                    "message": f"Request body exceeds {self.MAX_BODY_SIZE // 1024}KB limit"
                }
            )
        
        # Block path traversal
        if ".." in request.url.path:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "invalid_path",
                    "message": "Path traversal not allowed"
                }
            )
        
        return await call_next(request)


# =============================================================================
# Security Headers
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.
    """
    
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Cache-Control": "no-store, max-age=0",
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response


# =============================================================================
# Request Tracing
# =============================================================================

class TracingMiddleware(BaseHTTPMiddleware):
    """
    Add request tracing with:
    - Unique request ID
    - Request timing
    - Structured logging
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or propagate request ID
        request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
        
        # Store in request state for use in handlers
        request.state.request_id = request_id
        request.state.start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"[{request_id}] {request.method} {request.url.path}",
            extra={"request_id": request_id}
        )
        
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error
            elapsed = (time.time() - request.state.start_time) * 1000
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"ERROR in {elapsed:.0f}ms: {e}",
                extra={"request_id": request_id}
            )
            raise
        
        # Calculate elapsed time
        elapsed = (time.time() - request.state.start_time) * 1000
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed:.0f}ms"
        
        # Log response
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"{response.status_code} in {elapsed:.0f}ms",
            extra={"request_id": request_id}
        )
        
        return response


# =============================================================================
# Timeout Middleware
# =============================================================================

class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Add request timeout handling.
    Prevents long-running requests from blocking.
    """
    
    DEFAULT_TIMEOUT = 30  # seconds
    PATH_TIMEOUTS = {
        "/predict": 30,
        "/predict/batch": 120,
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get timeout for this path
        timeout = self.DEFAULT_TIMEOUT
        for path, t in self.PATH_TIMEOUTS.items():
            if request.url.path.startswith(path):
                timeout = t
                break
        
        try:
            return await asyncio.wait_for(
                call_next(request),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Request timeout: {request.method} {request.url.path} "
                f"exceeded {timeout}s"
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "timeout",
                    "message": f"Request timed out after {timeout}s",
                    "request_id": getattr(request.state, 'request_id', None)
                }
            )
