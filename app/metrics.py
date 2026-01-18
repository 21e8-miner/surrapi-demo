"""
SurrAPI Prometheus Metrics
==========================

Observability metrics for monitoring and alerting.
Exposes /metrics endpoint for Prometheus scraping.
"""

import time
import logging
from functools import wraps
from typing import Callable

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, multiprocess, REGISTRY
)
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("surrapi.metrics")

# =============================================================================
# Metric Definitions
# =============================================================================

# Request metrics
REQUESTS_TOTAL = Counter(
    'surrapi_requests_total',
    'Total HTTP requests',
    ['method', 'path', 'status']
)

REQUEST_LATENCY = Histogram(
    'surrapi_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'path'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

REQUESTS_IN_PROGRESS = Gauge(
    'surrapi_requests_in_progress',
    'Number of requests currently being processed',
    ['method']
)

# Prediction metrics
PREDICTIONS_TOTAL = Counter(
    'surrapi_predictions_total',
    'Total predictions made',
    ['tier', 'status']
)

PREDICTION_LATENCY = Histogram(
    'surrapi_prediction_latency_seconds',
    'Prediction inference latency',
    ['resolution'],
    buckets=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
)

# Model metrics
MODEL_LOADED = Gauge(
    'surrapi_model_loaded',
    'Whether the FNO model is loaded (1=yes, 0=no)'
)

MODEL_PARAMETERS = Gauge(
    'surrapi_model_parameters',
    'Number of model parameters'
)

# Billing metrics
API_KEYS_TOTAL = Gauge(
    'surrapi_api_keys_total',
    'Total API keys by tier',
    ['tier']
)

MONTHLY_PREDICTIONS = Gauge(
    'surrapi_monthly_predictions',
    'Predictions this month by tier',
    ['tier']
)

# Rate limiting metrics
RATE_LIMIT_HITS = Counter(
    'surrapi_rate_limit_hits_total',
    'Number of rate limit rejections',
    ['tier']
)

# Error metrics
ERRORS_TOTAL = Counter(
    'surrapi_errors_total',
    'Total errors by type',
    ['error_type']
)

# App info
APP_INFO = Info(
    'surrapi',
    'SurrAPI application info'
)


# =============================================================================
# Metrics Middleware
# =============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect request metrics.
    """
    
    # Paths to skip for metrics (high cardinality)
    SKIP_PATHS = ['/metrics', '/health']
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip metrics endpoint itself
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)
        
        # Normalize path for metric labels (avoid high cardinality)
        path = self._normalize_path(request.url.path)
        method = request.method
        
        # Track in-progress requests
        REQUESTS_IN_PROGRESS.labels(method=method).inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception as e:
            status = "500"
            ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            # Record metrics
            elapsed = time.time() - start_time
            
            REQUESTS_TOTAL.labels(
                method=method,
                path=path,
                status=status
            ).inc()
            
            REQUEST_LATENCY.labels(
                method=method,
                path=path
            ).observe(elapsed)
            
            REQUESTS_IN_PROGRESS.labels(method=method).dec()
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path to avoid high cardinality"""
        # Group paths with IDs
        parts = path.split('/')
        normalized = []
        for part in parts:
            # Replace UUIDs and long tokens with placeholder
            if len(part) > 20 or (len(part) == 36 and '-' in part):
                normalized.append('{id}')
            else:
                normalized.append(part)
        return '/'.join(normalized) or '/'


# =============================================================================
# Metrics Endpoint
# =============================================================================

async def metrics_endpoint(request: Request) -> Response:
    """
    Expose Prometheus metrics.
    
    Usage:
        # In main.py
        from app.metrics import metrics_endpoint
        app.add_api_route("/metrics", metrics_endpoint)
    """
    # Update app info
    APP_INFO.info({
        'version': '0.1.0',
        'environment': 'production'
    })
    
    # Generate metrics output
    metrics_output = generate_latest(REGISTRY)
    
    return Response(
        content=metrics_output,
        media_type=CONTENT_TYPE_LATEST
    )


# =============================================================================
# Decorator for Tracking Predictions
# =============================================================================

def track_prediction(func: Callable) -> Callable:
    """
    Decorator to track prediction metrics.
    
    Usage:
        @track_prediction
        async def predict(request: PredictRequest):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract tier from API key if available
        tier = "free"
        api_key = kwargs.get('api_key')
        if api_key:
            tier = getattr(api_key, 'tier', 'free')
        
        # Extract resolution from request
        resolution = "128"
        request = kwargs.get('request')
        if request and hasattr(request, 'resolution'):
            resolution = str(request.resolution)
        
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            PREDICTIONS_TOTAL.labels(tier=tier, status="success").inc()
            return result
        except Exception as e:
            PREDICTIONS_TOTAL.labels(tier=tier, status="error").inc()
            ERRORS_TOTAL.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            elapsed = time.time() - start_time
            PREDICTION_LATENCY.labels(resolution=resolution).observe(elapsed)
    
    return wrapper


# =============================================================================
# Utility Functions
# =============================================================================

def update_model_metrics(model, device: str):
    """Update model-related metrics"""
    if model is not None:
        MODEL_LOADED.set(1)
        param_count = sum(p.numel() for p in model.parameters())
        MODEL_PARAMETERS.set(param_count)
    else:
        MODEL_LOADED.set(0)
        MODEL_PARAMETERS.set(0)


def update_billing_metrics():
    """Update billing-related metrics"""
    try:
        from app.billing import _api_keys, TIERS
        
        # Count keys by tier
        tier_counts = {"free": 0, "pro": 0, "enterprise": 0}
        tier_predictions = {"free": 0, "pro": 0, "enterprise": 0}
        
        for key in _api_keys.values():
            tier_counts[key.tier] = tier_counts.get(key.tier, 0) + 1
            tier_predictions[key.tier] = tier_predictions.get(key.tier, 0) + key.predictions_this_month
        
        for tier, count in tier_counts.items():
            API_KEYS_TOTAL.labels(tier=tier).set(count)
        
        for tier, count in tier_predictions.items():
            MONTHLY_PREDICTIONS.labels(tier=tier).set(count)
            
    except ImportError:
        pass
