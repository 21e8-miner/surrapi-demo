"""
Pydantic schemas for SurrAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class FlowType(str, Enum):
    """Supported flow simulation types"""
    NAVIER_STOKES = "navier_stokes"
    EULER = "euler"
    STOKES = "stokes"
    POTENTIAL = "potential"


class BoundaryCondition(str, Enum):
    """Boundary condition types"""
    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    NO_SLIP = "no_slip"


class GeometryType(str, Enum):
    """Supported obstacle geometries"""
    NONE = "none"
    CYLINDER = "cylinder"
    SQUARE = "square"
    AIRFOIL = "airfoil"
    RANDOM = "random"


# -------------------------------------------------------------------
# Request Schemas
# -------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Input parameters for flow field prediction.
    
    All parameters are normalized to physically reasonable ranges
    validated against The Well dataset parameter distributions.
    """
    
    reynolds: float = Field(
        default=2000.0,
        ge=500.0,
        le=10000.0,
        description="Reynolds number (500-10000). Controls flow regime from laminar to turbulent."
    )
    
    angle: float = Field(
        default=0.0,
        ge=-15.0,
        le=15.0,
        description="Angle of attack in degrees (-15 to 15). Orientation of flow relative to geometry."
    )
    
    mach: float = Field(
        default=0.2,
        ge=0.05,
        le=0.6,
        description="Mach number (0.05-0.6). Compressibility effects; <0.3 is effectively incompressible."
    )
    
    flow_type: FlowType = Field(
        default=FlowType.NAVIER_STOKES,
        description="Type of flow equations to use for prediction."
    )
    
    resolution: int = Field(
        default=128,
        ge=64,
        le=256,
        description="Output grid resolution (64-256). Higher = more detail but slower."
    )
    
    # Optional advanced parameters
    prandtl: Optional[float] = Field(
        default=0.71,
        ge=0.1,
        le=10.0,
        description="Prandtl number for thermal coupling (air ≈ 0.71, water ≈ 7)."
    )
    
    inlet_velocity: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=100.0,
        description="Inlet velocity magnitude (m/s)."
    )
    
    enforce_conservation: bool = Field(
        default=False,
        description="Apply adaptive correction to enforce mass conservation (∇·u = 0). Increases physics_score but adds ~50ms latency."
    )
    
    return_uncertainty: bool = Field(
        default=False,
        description="Return uncertainty estimates via Monte Carlo dropout. Adds ~200ms latency (10 samples)."
    )
    
    geometry_type: GeometryType = Field(
        default=GeometryType.CYLINDER,
        description="Type of obstacle to insert into the flow."
    )
    
    geometry_params: Dict[str, Any] = Field(
        default_factory=lambda: {"x": 0.3, "y": 0.5, "radius": 0.05},
        description="Parameters for the selected geometry (x, y, scale, etc.)."
    )
    
    # Validators for physics-based input checking
    @validator('reynolds')
    def validate_reynolds(cls, v):
        if v > 15000:
            import warnings
            warnings.warn(
                f"Re={v} exceeds safe operating range (15000). "
                "Predictions will degrade significantly. See LIMITATIONS.md",
                UserWarning
            )
        return v
    
    @validator('angle')
    def validate_angle(cls, v):
        if abs(v) > 12:
            import warnings
            warnings.warn(
                f"α={v}° may cause flow separation. "
                "Model not trained on separated flows. See LIMITATIONS.md",
                UserWarning
            )
        return v
    
    @validator('mach')
    def validate_mach(cls, v):
        if v > 0.3:
            import warnings
            warnings.warn(
                f"Mach={v} violates incompressible flow assumption (M < 0.3). "
                "Results may be unreliable for compressibility effects.",
                UserWarning
            )
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "reynolds": 5000,
                "angle": 5.0,
                "mach": 0.2,
                "flow_type": "navier_stokes",
                "resolution": 128
            }
        }


class BatchPredictRequest(BaseModel):
    """Batch prediction for parameter sweeps"""
    
    predictions: List[PredictRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of prediction requests (max 100 per batch)."
    )


class GeometryUploadRequest(BaseModel):
    """Upload custom geometry for prediction"""
    
    sdf_base64: str = Field(
        ...,
        description="Base64-encoded signed distance field (128x128 float32)."
    )
    
    parameters: PredictRequest = Field(
        default_factory=PredictRequest,
        description="Flow parameters for this geometry."
    )


# -------------------------------------------------------------------
# Response Schemas
# -------------------------------------------------------------------

class PredictResponse(BaseModel):
    """
    Flow field prediction response.
    
    Contains both structured data (JSON arrays) and
    visualization-ready formats (VTK, base64-encoded).
    """
    
    # Base64-encoded VTK ImageData file
    vtk: str = Field(
        ...,
        description="Base64-encoded .vti file for ParaView/VisIt import."
    )
    
    # Flattened field arrays (row-major order)
    ux: List[float] = Field(
        ...,
        description="X-velocity component (flattened 128x128 or specified resolution)."
    )
    
    uy: List[float] = Field(
        ...,
        description="Y-velocity component."
    )
    
    p: List[float] = Field(
        ...,
        description="Pressure field (normalized)."
    )
    
    # Derived quantities
    velocity_magnitude: Optional[List[float]] = Field(
        default=None,
        description="Velocity magnitude |u| = sqrt(ux² + uy²)."
    )
    
    vorticity: Optional[List[float]] = Field(
        default=None,
        description="Z-component of vorticity ω = ∂uy/∂x - ∂ux/∂y."
    )
    
    divergence: Optional[List[float]] = Field(
        default=None,
        description="Mass conservation error ∇·u. Should be close to 0."
    )
    
    physics_score: Optional[float] = Field(
        default=None,
        description="Confidence score based on mass conservation (0.0 to 1.0)."
    )
    
    # Uncertainty quantification (Monte Carlo dropout)
    ux_std: Optional[List[float]] = Field(
        default=None,
        description="Uncertainty (std dev) for x-velocity field."
    )
    uy_std: Optional[List[float]] = Field(
        default=None,
        description="Uncertainty (std dev) for y-velocity field."
    )
    p_std: Optional[List[float]] = Field(
        default=None,
        description="Uncertainty (std dev) for pressure field."
    )
    
    # Engineering Metrics
    enstrophy: Optional[float] = Field(
        default=None,
        description="Total enstrophy (mean square vorticity)."
    )
    drag_coefficient: Optional[float] = Field(
        default=None,
        description="Estimates drag coefficient (normalized pressure force)."
    )
    lift_coefficient: Optional[float] = Field(
        default=None,
        description="Estimated lift coefficient (transverse pressure force)."
    )
    
    # Geometry metadata
    geometry_mask: Optional[List[float]] = Field(
        default=None,
        description="Boolean mask of the obstacle geometry (1=solid, 0=fluid)."
    )
    
    # Metadata
    resolution: int = Field(
        default=128,
        description="Grid resolution of output fields."
    )
    
    inference_time_ms: float = Field(
        ...,
        description="Model inference time in milliseconds."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "vtk": "PHZ0a1...",
                "ux": [0.1, 0.2, 0.3],
                "uy": [0.0, 0.01, 0.02],
                "p": [1.0, 0.99, 0.98],
                "resolution": 128,
                "inference_time_ms": 285.4
            }
        }


class BatchPredictResponse(BaseModel):
    """Response for batch predictions"""
    
    results: List[PredictResponse] = Field(
        ...,
        description="List of prediction results in same order as requests."
    )
    
    total_time_ms: float = Field(
        ...,
        description="Total batch processing time."
    )


class IntegralQuantitiesResponse(BaseModel):
    """
    Derived integral quantities for engineering analysis.
    """
    
    drag_coefficient: float = Field(
        ...,
        description="Cd - integrated pressure and viscous drag."
    )
    
    lift_coefficient: float = Field(
        ...,
        description="Cl - integrated lift force coefficient."
    )
    
    pressure_drop: float = Field(
        ...,
        description="Inlet-outlet pressure difference."
    )
    
    max_velocity: float = Field(
        ...,
        description="Maximum velocity magnitude in domain."
    )
    
    enstrophy: float = Field(
        ...,
        description="Integrated vorticity squared (turbulence intensity proxy)."
    )


class HealthResponse(BaseModel):
    """API health check response"""
    
    status: str = Field(default="ok")
    device: str = Field(..., description="Compute device (cpu/cuda/mps).")
    model_loaded: bool = Field(..., description="Whether model weights are loaded.")
    version: str = Field(default="0.1.0")
    uptime_seconds: float = Field(..., description="Server uptime.")
    model_architecture: str = Field(default="FNO-2D", description="Details of the running neural operator.")
    
    # Extended metrics
    checks: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed dependency checks."
    )
    total_predictions: Optional[int] = Field(
        default=None,
        description="Total predictions served since startup."
    )


class ErrorResponse(BaseModel):
    """Standard error response"""
    
    error: str = Field(..., description="Error type.")
    message: str = Field(..., description="Human-readable error message.")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context."
    )


# -------------------------------------------------------------------
# Usage/Billing Schemas
# -------------------------------------------------------------------

class UsageStats(BaseModel):
    """API usage statistics for billing"""
    
    api_key: str = Field(..., description="Hashed API key identifier.")
    predictions_today: int = Field(default=0)
    predictions_month: int = Field(default=0)
    tier: str = Field(default="free", description="Subscription tier.")
    remaining_quota: int = Field(..., description="Remaining predictions this period.")


class RateLimitInfo(BaseModel):
    """Rate limit headers information"""
    
    limit: int = Field(..., description="Max requests per window.")
    remaining: int = Field(..., description="Remaining requests in window.")
    reset_at: str = Field(..., description="ISO timestamp when window resets.")
