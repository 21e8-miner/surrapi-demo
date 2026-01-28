"""
SurrAPI Python Client
=====================

Pythonic interface to the SurrAPI CFD prediction service.
"""

import os
import requests
from typing import Optional, List
from .types import PredictRequest, PredictResponse


class SurrAPIError(Exception):
    """Base exception for SurrAPI errors"""
    
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RateLimitError(SurrAPIError):
    """Raised when API rate limit is exceeded"""
    pass


class AuthenticationError(SurrAPIError):
    """Raised when API key is invalid or missing"""
    pass


class QuotaExceededError(SurrAPIError):
    """Raised when monthly prediction quota is exceeded"""
    pass


class Client:
    """
    SurrAPI Python Client.
    
    Example:
        >>> client = Client(api_key="sk_...")
        >>> result = client.predict(reynolds=5000, angle=5.0)
        >>> ux = result.ux.to_numpy()  # 128x128 numpy array
    """
    
    DEFAULT_BASE_URL = "https://api.surrapi.io"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0
    ):
        """
        Initialize SurrAPI client.
        
        Args:
            api_key: API key (or set SURRAPI_API_KEY env var)
            base_url: API base URL (default: https://api.surrapi.io)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("SURRAPI_API_KEY")
        self.base_url = (base_url or os.getenv("SURRAPI_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
    
    def _headers(self) -> dict:
        """Build request headers"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _handle_error(self, response: requests.Response) -> None:
        """Convert HTTP errors to typed exceptions"""
        if response.status_code == 401:
            raise AuthenticationError(
                "Invalid or missing API key",
                status_code=401,
                response=response.json() if response.content else None
            )
        elif response.status_code == 429:
            data = response.json() if response.content else {}
            if "quota" in str(data).lower():
                raise QuotaExceededError(
                    "Monthly prediction quota exceeded. Upgrade at https://surrapi.io/pricing",
                    status_code=429,
                    response=data
                )
            raise RateLimitError(
                "Rate limit exceeded. Please slow down.",
                status_code=429,
                response=data
            )
        elif response.status_code >= 400:
            try:
                data = response.json()
                message = data.get("detail", str(data))
            except:
                message = response.text
            raise SurrAPIError(
                f"API error: {message}",
                status_code=response.status_code,
                response=data if 'data' in dir() else None
            )
    
    def predict(
        self,
        reynolds: float = 2000.0,
        angle: float = 0.0,
        mach: float = 0.2,
        resolution: int = 128,
        inlet_velocity: Optional[float] = None,
        validate_physics: bool = True
    ) -> PredictResponse:
        """
        Predict flow field for given parameters.
        
        Args:
            reynolds: Reynolds number (500-10000)
            angle: Angle of attack in degrees (-15 to 15)
            mach: Mach number (0.05-0.6)
            resolution: Output grid resolution (64-256)
            inlet_velocity: Optional inlet velocity
            validate_physics: Warn if physics score is low
        
        Returns:
            PredictResponse with flow fields and metrics
        
        Raises:
            SurrAPIError: On API errors
            AuthenticationError: On invalid API key
            RateLimitError: On rate limit exceeded
        
        Example:
            >>> result = client.predict(reynolds=5000, angle=5.0, mach=0.3)
            >>> print(f"Max velocity: {result.velocity_magnitude.max():.2f}")
        """
        request = PredictRequest(
            reynolds=reynolds,
            angle=angle,
            mach=mach,
            resolution=resolution,
            inlet_velocity=inlet_velocity
        )
        
        response = self._session.post(
            f"{self.base_url}/predict",
            json=request.to_dict(),
            headers=self._headers(),
            timeout=self.timeout
        )
        
        self._handle_error(response)
        
        result = PredictResponse.from_api_response(response.json())
        
        # Warn on low physics score
        if validate_physics and result.physics_score is not None:
            if result.physics_score < 0.5:
                import warnings
                warnings.warn(
                    f"Low physics score ({result.physics_score:.1%}). "
                    "Prediction may not satisfy mass conservation.",
                    UserWarning
                )
        
        return result
    
    def predict_batch(
        self,
        requests: List[PredictRequest],
        validate_physics: bool = True
    ) -> List[PredictResponse]:
        """
        Batch prediction for parameter sweeps.
        
        Args:
            requests: List of prediction requests (max 100)
            validate_physics: Warn if any physics score is low
        
        Returns:
            List of PredictResponse objects
        
        Example:
            >>> requests = [PredictRequest(reynolds=r) for r in range(1000, 10001, 1000)]
            >>> results = client.predict_batch(requests)
        """
        if len(requests) > 100:
            raise ValueError("Batch size cannot exceed 100")
        
        payload = {"predictions": [r.to_dict() for r in requests]}
        
        response = self._session.post(
            f"{self.base_url}/predict/batch",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout * len(requests)  # Scale timeout
        )
        
        self._handle_error(response)
        
        data = response.json()
        results = [
            PredictResponse.from_api_response(r)
            for r in data["results"]
        ]
        
        # Warn on low physics scores
        if validate_physics:
            low_scores = [i for i, r in enumerate(results) 
                         if r.physics_score and r.physics_score < 0.5]
            if low_scores:
                import warnings
                warnings.warn(
                    f"Low physics scores on predictions: {low_scores[:5]}...",
                    UserWarning
                )
        
        return results
    
    def health(self) -> dict:
        """
        Check API health status.
        
        Returns:
            Health status dict with model_loaded, device, uptime, etc.
        """
        response = self._session.get(
            f"{self.base_url}/health",
            timeout=self.timeout
        )
        self._handle_error(response)
        return response.json()
    
    def usage(self) -> dict:
        """
        Get current API usage for this key.
        
        Returns:
            Usage dict with predictions_this_month, limit, tier, etc.
        
        Raises:
            AuthenticationError: If no API key is set
        """
        if not self.api_key:
            raise AuthenticationError("API key required for usage endpoint")
        
        response = self._session.get(
            f"{self.base_url}/api/usage",
            headers=self._headers(),
            timeout=self.timeout
        )
        self._handle_error(response)
        return response.json()
