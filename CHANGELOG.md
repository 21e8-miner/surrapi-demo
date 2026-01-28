# Changelog

## [0.2.0] - 2026-01-28 - BREAKTHROUGH: Differentiable Design Mode

### Added
- **Inverse Design Mode**: Automated geometry parameter optimization to minimize Drag.
- **GeometricOptimizer**: Differentiable SDF-to-Flow optimization loop.
- **Adjoint Sensitivity Mapping**: X-CFD visualization showing optimization zones.
- **Trust Index**: Statistical correlation between Adjoint Sensitivity and Uncertainty (UQ).
- **Custom Geometry Upload**: Base64 SDF injection support for proprietary shapes.
- **Hallucination Detection**: Pre-flight physics residual checks with adaptive PINC scaling.
- **QC v0.2 Report**: Documenting the "Middle-Out" hardening of the physics pipeline.

### Fixed
- **Dynamic Resolution Bug**: Navier-Stokes residuals now use dynamic `dx` based on grid resolution.
- **Spatial Gradient Error**: Fixed dimension mismatch in `torch.gradient` calls within physical optimizer.
- **SDF Discontinuity**: Implemented Sigmoid masking for differentiable solid-fluid boundaries.

### Changed
- **README Update**: Reflects v0.2 Design Mode capabilities and roadmap.
- **API Response**: Added `trust_index` and `sensitivity_map` fields.
