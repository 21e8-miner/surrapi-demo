# SurrAPI — Fast Neural CFD Surrogate

![Python](https://img.shields.io/badge/Python-3.11+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Production_Ready-brightgreen)

> **300ms flow field predictions.** Production-grade API with physics validation, uncertainty quantification, and honest benchmarks.

---

## Why SurrAPI?

Traditional CFD takes **hours to days** per simulation. Neural surrogates provide **sub-second** predictions for parameter exploration, optimization loops, and real-time applications — where waiting for full CFD isn't practical.

### What Makes Us Different

| Feature | SurrAPI | Generic ML Demo |
|---------|---------|-----------------|
| **Physics Validation** | Real-time divergence checking, conservation correction | Hope for the best |
| **Uncertainty Quantification** | Monte Carlo dropout, confidence bounds | Single point estimate |
| **Honest Benchmarks** | Published error metrics with ranges | "Works great!" |
| **Production Ready** | Rate limiting, API keys, Stripe billing | Jupyter notebook |
| **Failure Documentation** | [LIMITATIONS.md](./LIMITATIONS.md) tells you when NOT to use it | Silent failures |

---

## Performance

### Validated Accuracy (In-Distribution)

| Case | Reynolds | L² Error | Physics Score | Inference |
|------|----------|----------|---------------|-----------|
| Lid-Driven Cavity | 1,000 | 1.8% ± 0.4% | 0.92 | 285ms |
| Lid-Driven Cavity | 5,000 | 2.4% ± 0.6% | 0.88 | 290ms |
| Channel Flow | 2,000 | 1.2% ± 0.3% | 0.95 | 278ms |

*Error metrics computed on held-out test set. See [MODEL_CARD.md](./MODEL_CARD.md) for methodology.*

### Speed Comparison

| Method | Time | Use Case |
|--------|------|----------|
| **SurrAPI** | 300ms | Parameter sweeps, optimization |
| OpenFOAM (coarse) | 10 min | Quick validation |
| Ansys Fluent (fine) | 2-8 hrs | Final design verification |

**Our Position**: We're not replacing Fluent. We're replacing the 1,000 Fluent runs you'd do during design exploration.

---

## Quick Start

```bash
git clone https://github.com/21e8-miner/surrapi-demo.git
cd surrapi-demo
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### Python SDK

```python
from surrapi import Client

client = Client(api_key="sk_...")

# Fast prediction
result = client.predict(reynolds=5000, angle=5.0, mach=0.2)
print(f"Inference: {result.inference_time_ms:.0f}ms")
print(f"Physics score: {result.physics_score:.1%}")

# With physics correction (recommended)
result = client.predict(reynolds=5000, enforce_conservation=True)
# physics_score jumps from ~85% to ~95%

# With uncertainty (for critical decisions)
result = client.predict(reynolds=5000, return_uncertainty=True)
if result.mean_uncertainty() < 0.05:
    print("High confidence prediction")
```

---

## API Capabilities

### Core Features

| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Single flow field prediction |
| `POST /predict/batch` | Up to 100 predictions in one call |
| `GET /health` | Model status and training domain |
| `GET /api/usage` | Billing and quota information |

### Advanced Options

| Parameter | What It Does |
|-----------|--------------|
| `enforce_conservation=True` | Helmholtz projection reduces divergence 60-90% |
| `return_uncertainty=True` | MC dropout for confidence intervals |
| `resolution=256` | Higher fidelity output (64-256 supported) |

---

## Architecture

**PhysicsAwareFNO** — Our flagship model combining 4 techniques:

```
Input → LocalFeatureExtractor → FNO (32 modes) → SpecBoost → Conservation → Output
         ↓                       ↓                ↓            ↓
    Boundary layers        Global physics     High-freq    ∇·u = 0
```

- **8.4M parameters** — Runs on CPU or GPU
- **128×128 default** — Higher resolutions available
- **Steady-state 2D** — Transient and 3D on roadmap

See [MODEL_CARD.md](./MODEL_CARD.md) for full architecture specification.

---

## Scope & Limitations

We're transparent about what the model can and can't do:

### ✅ Works Well

| Parameter | Range | Expected Error |
|-----------|-------|----------------|
| Reynolds | 500 – 10,000 | 1.5 – 3% |
| Angle | -10° to +10° | 2 – 4% |
| Mach | 0.1 – 0.3 | 1.5 – 2.5% |

### ⚠️ Degrades Gracefully

| Condition | What Happens |
|-----------|--------------|
| Re > 15,000 | Error increases to 8-12%, physics_score drops |
| α > 12° | Flow separation not captured accurately |
| M > 0.3 | Compressibility effects underestimated |

### ❌ Don't Use For

- Transonic/supersonic flows (M > 0.6)
- Arbitrary geometry (fixed domain currently)
- Time-dependent vortex shedding
- Final certification (always validate with CFD)

**Full details**: [LIMITATIONS.md](./LIMITATIONS.md)

---

## Business Model

### Open Core

| Tier | Included | Price |
|------|----------|-------|
| **Demo** (this repo) | Full inference code, local deployment | Free (MIT) |
| **API** | Hosted endpoints, no GPU needed | $0.25/prediction |
| **Enterprise** | On-prem Docker, custom training, support | Contact us |

### Why Open Source?

1. **Trust** — You can verify our claims
2. **Adoption** — Easy to evaluate before buying
3. **Community** — Improvements benefit everyone

---

## Research Foundation

Built on peer-reviewed work:

- **Li et al., ICLR 2021**: Fourier Neural Operator ([arXiv:2010.08895](https://arxiv.org/abs/2010.08895))
- **Kovachki et al., JMLR 2023**: Neural Operator theory ([arXiv:2108.08481](https://arxiv.org/abs/2108.08481))
- **Training data**: Synthetic OpenFOAM simulations (methodology in MODEL_CARD.md)

---

## Breakthrough Features (Design Mode)

SurrAPI now includes **Design Mode**, a suite of differentiable tools for interactive engineering exploration.

### 🪄 Inverse Design (Auto-Optimize)
Automatically find the optimal geometry parameters (e.g., position) to minimize drag. The system differentiates through the SDF generation process and flow prediction to iteratively improve your design in real-time.

### 🔍 Adjoint Sensitivity Analysis (X-CFD)
Visualize "Explainable CFD" heatmaps showing which regions of the flow are most sensitive to design changes. This is computed using Torch auto-grad through the physics-informed architecture.

### 🎨 Custom Geometry Support
Upload your own proprietary designs using base64-encoded Signed Distance Fields (SDFs). SurrAPI's **decode_sdf** pipeline enables enterprise-grade simulation of custom CAD profiles.

### 🛡️ Hallucination Correction (PINC+)
Our **Adaptive Physics Optimizer** detects and corrects "plausible but wrong" neural artifacts by scaling iterations (3x) when physics residuals exceed safety thresholds (Hallucination Detection).

### 🤝 Trust Index
A novel diagnostic metric correlating **Adjoint Sensitivity** with **Uncertainty Quantification (UQ)**.
- **High Trust**: Low uncertainty in sensitive optimization zones.
- **Risky**: Uncertainty spikes in critical flow regions (e.g., wake or separation points).

---

## Roadmap

| Feature | Status |
|---------|--------|
| 2D steady-state | ✅ Production |
| Physics validation | ✅ Production |
| Uncertainty quantification | ✅ Production |
| **Inverse Design Mode** | ✅ Breakthrough (v0.2) |
| **Adjoint Sensitivity** | ✅ Breakthrough (v0.2) |
| **Custom Geometry Support** | ✅ Breakthrough (v0.2) |
| **Hallucination Correction** | ✅ Breakthrough (v0.2) |
| 3D support | 🔄 Planned |
| Transient flows | 📋 Planned |

---

## Get Started

```bash
# Local demo (CPU/GPU)
python -m uvicorn app.main:app --port 8000
```

**Questions?** team@surrapi.io

---

*Verified Physics. Explainable Design. Production Ready.*
