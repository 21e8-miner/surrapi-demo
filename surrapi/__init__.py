"""
SurrAPI Python SDK
==================

Official Python client for SurrAPI - instant CFD predictions via neural operators.

Quick Start:
    >>> from surrapi import Client
    >>> client = Client(api_key="sk_...")
    >>> result = client.predict(reynolds=5000, angle=5.0, mach=0.3)
    >>> print(f"Inference: {result.inference_time_ms}ms")
    >>> print(f"Physics Score: {result.physics_score:.1%}")
"""

from .client import Client, SurrAPIError
from .types import PredictRequest, PredictResponse, FlowField

__version__ = "0.1.0"
__all__ = ["Client", "SurrAPIError", "PredictRequest", "PredictResponse", "FlowField"]
