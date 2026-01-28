# SurrAPI Limitations & Known Failure Modes

> ⚠️ **This is a research prototype.** Do not use for engineering design, certification, or safety-critical decisions without independent CFD validation.

## Training Domain Boundaries

| Parameter | Training Range | Safe Operating Range | Notes |
|-----------|----------------|---------------------|-------|
| Reynolds (Re) | 500 – 10,000 | 100 – 15,000 | Extrapolation degrades rapidly beyond 15k |
| Angle of Attack (α) | -10° to +15° | -10° to +12° | Flow separation >12° produces unphysical results |
| Mach (M) | 0.1 – 0.4 | 0.1 – 0.3 | M > 0.3 violates incompressible assumption |
| Resolution | 64² – 256² | 128² recommended | Higher resolution = memory hungry |

## Known Failure Modes

### 1. Flow Separation (Critical)
- **Symptom:** Unphysical recirculation bubbles, negative pressure zones
- **Trigger:** α > 12° or Re > 20,000
- **Detection:** `physics_score < 0.5`, large divergence values
- **Mitigation:** None — model is not trained on separated flows

### 2. Shock Capturing (Not Supported)
- **Symptom:** Oscillations near shock locations, non-physical pressure spikes
- **Trigger:** M > 0.6 (transonic regime)
- **Root Cause:** FNO spectral convolutions cannot represent sharp discontinuities
- **Mitigation:** Restrict to subsonic flows (M < 0.3)

### 3. Boundary Layer Resolution
- **Symptom:** Incorrect wall shear stress, underestimated drag
- **Trigger:** High-Re flows on coarse grids
- **Root Cause:** 128×128 grid cannot resolve viscous sublayer (y+ requirements)
- **Mitigation:** Use for qualitative trends only, not quantitative predictions

### 4. Geometry Generalization (Not Implemented)
- **Current Status:** Fixed unit-square domain (lid-driven cavity analog)
- **What Doesn't Work:** Arbitrary airfoil shapes, complex geometries
- **Planned:** Geometry-conditioned training (roadmap item)

### 5. Transient Flows (Not Supported)
- **Symptom:** Static/averaged predictions for inherently unsteady flows
- **Trigger:** Vortex shedding regimes (Re > 40 for cylinders)
- **Root Cause:** Model trained on steady-state solutions only
- **Detection:** Large uncertainty values if `return_uncertainty=True`

## Out-of-Distribution Behavior

When inputs fall outside the training distribution:

1. **Predictions become interpolations of training extremes** (not physics)
2. **physics_score degrades** — check response metadata
3. **Divergence increases** — ∇·u deviates from zero
4. **No explicit warning** is raised (user must check metrics)

### OOD Detection Heuristics

```python
result = client.predict(reynolds=50000, angle=20.0)

# Check 1: Physics score
if result.physics_score < 0.7:
    print("WARNING: Low physics confidence — likely OOD")

# Check 2: Request uncertainty quantification
result = client.predict(..., return_uncertainty=True)
if result.mean_uncertainty() > 0.2:
    print("WARNING: High uncertainty — predictions unreliable")
```

## Physics Constraints — What They Actually Do

### `enforce_conservation=True`

**What it is:**
- Post-processing Helmholtz/Hodge projection onto divergence-free manifold
- Iterative Poisson solver (5 iterations) in spectral space

**What it does:**
- Reduces velocity divergence by ~60-90% on validation set
- Improves `physics_score` metric

**What it does NOT do:**
- ❌ "Guarantee" conservation (projection only corrects predictions, doesn't fix learned representations)
- ❌ Make unphysical predictions physical
- ❌ Work reliably outside training distribution

### `return_uncertainty=True`

**What it is:**
- Monte Carlo dropout with 10 forward passes
- Returns standard deviation across samples

**What it does:**
- Provides variance estimates for each field
- High uncertainty correlates with OOD inputs

**What it does NOT do:**
- ❌ Provide calibrated confidence intervals
- ❌ Guarantee coverage probability
- ❌ Replace proper ensemble methods

## Comparison to Full CFD

| Aspect | SurrAPI Surrogate | Traditional CFD |
|--------|-------------------|-----------------|
| Speed | ~300ms | 10min – 10hr |
| Accuracy | 2-5% L² (in-distribution) | Mesh-converged truth |
| Generalization | Narrow training domain | Arbitrary physics |
| Shock handling | ❌ Not supported | ✓ Full capability |
| Geometry flexibility | ❌ Fixed domain | ✓ Arbitrary |
| Transient flows | ❌ Steady-state only | ✓ Full capability |
| Out-of-distribution | Unpredictable | Fails gracefully with residuals |

## When to Use This Tool

✅ **Appropriate Use Cases:**
- Rapid parameter sweeps within training domain
- Initial design exploration (to be validated with CFD)
- Educational demonstrations of neural operators
- Prototyping ML-CFD workflows

❌ **Inappropriate Use Cases:**
- Final engineering design decisions
- Safety-critical aerodynamics
- Flows outside training distribution
- As a replacement for validated CFD simulations

## Reporting Issues

If you encounter unexpected behavior:

1. Check if inputs are within training domain (see table above)
2. Enable `return_uncertainty=True` and verify uncertainty levels
3. Compare physics_score to expected range (>0.7 for reasonable predictions)
4. Open an issue with: input parameters, response metrics, expected vs. actual behavior

---

*This document is part of responsible AI disclosure. Last updated: 2026-01-28*
