"""
SurrAPI Type Definitions
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class FlowField:
    """
    2D flow field with convenience methods.
    
    Attributes:
        data: Flattened field values
        resolution: Grid resolution (e.g., 128 for 128x128)
    """
    data: List[float]
    resolution: int = 128
    
    def to_numpy(self) -> np.ndarray:
        """Convert to 2D numpy array"""
        return np.array(self.data).reshape(self.resolution, self.resolution)
    
    @property
    def shape(self) -> tuple:
        return (self.resolution, self.resolution)
    
    def max(self) -> float:
        return max(self.data)
    
    def min(self) -> float:
        return min(self.data)
    
    def mean(self) -> float:
        return sum(self.data) / len(self.data)


@dataclass
class PredictRequest:
    """
    CFD prediction request parameters.
    
    Args:
        reynolds: Reynolds number (500-10000)
        angle: Angle of attack in degrees (-15 to 15)
        mach: Mach number (0.05-0.6)
        resolution: Output grid resolution (64-256)
        inlet_velocity: Optional inlet velocity (m/s)
    """
    reynolds: float = 2000.0
    angle: float = 0.0
    mach: float = 0.2
    resolution: int = 128
    inlet_velocity: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to API request payload"""
        d = {
            "reynolds": self.reynolds,
            "angle": self.angle,
            "mach": self.mach,
            "resolution": self.resolution,
        }
        if self.inlet_velocity is not None:
            d["inlet_velocity"] = self.inlet_velocity
        return d


@dataclass
class PredictResponse:
    """
    CFD prediction response with flow fields and metrics.
    
    Attributes:
        ux: X-velocity field
        uy: Y-velocity field
        p: Pressure field
        vorticity: Vorticity field (optional)
        divergence: Divergence field for physics validation (optional)
        velocity_magnitude: Speed field (optional)
        physics_score: Mass conservation confidence (0-1, higher = better)
        inference_time_ms: Server-side inference time
        vtk: Base64-encoded VTK file for ParaView
    """
    ux: FlowField
    uy: FlowField
    p: FlowField
    resolution: int
    inference_time_ms: float
    vtk: str
    
    # Optional derived fields
    vorticity: Optional[FlowField] = None
    divergence: Optional[FlowField] = None
    velocity_magnitude: Optional[FlowField] = None
    physics_score: Optional[float] = None
    
    # Uncertainty quantification (future)
    ux_std: Optional[FlowField] = None
    uy_std: Optional[FlowField] = None
    p_std: Optional[FlowField] = None
    
    @classmethod
    def from_api_response(cls, data: dict) -> "PredictResponse":
        """Parse API JSON response into typed object"""
        resolution = data.get("resolution", 128)
        
        response = cls(
            ux=FlowField(data["ux"], resolution),
            uy=FlowField(data["uy"], resolution),
            p=FlowField(data["p"], resolution),
            resolution=resolution,
            inference_time_ms=data["inference_time_ms"],
            vtk=data["vtk"],
            physics_score=data.get("physics_score"),
        )
        
        # Parse optional fields
        if data.get("vorticity"):
            response.vorticity = FlowField(data["vorticity"], resolution)
        if data.get("divergence"):
            response.divergence = FlowField(data["divergence"], resolution)
        if data.get("velocity_magnitude"):
            response.velocity_magnitude = FlowField(data["velocity_magnitude"], resolution)
        
        return response
    
    def save_vtk(self, filepath: str) -> None:
        """Save VTK file for ParaView visualization"""
        import base64
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(self.vtk))
    
    def is_physically_valid(self, threshold: float = 0.8) -> bool:
        """
        Check if prediction satisfies physics constraints.
        
        Args:
            threshold: Minimum physics score (default 0.8 = 80%)
        
        Returns:
            True if physics score exceeds threshold
        """
        if self.physics_score is None:
            return True  # No score = assume valid
        return self.physics_score >= threshold
