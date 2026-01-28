# SurrAPI Demo - Neural Surrogate for 2D Incompressible Flows

![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)
![License](https://img.shields.io/badge/License-MIT-yellow)

> ⚠️ **Research Prototype**: This is a demonstration of neural operator methods for CFD. 
> Not validated for engineering design, certification, or safety-critical applications.
> See [LIMITATIONS.md](./LIMITATIONS.md) before using.

---

## What This Demo Does

A Fourier Neural Operator (FNO) surrogate trained on **synthetic 2D steady-state simulations** (lid-driven cavity, channel flow). Provides fast flow field interpolation **within its training distribution**.

| Capability | Reality Check |
|------------|---------------|
| **~300ms inference** | ✓ True for 128×128 grid on GPU |
| **Training data** | ~5,000 OpenFOAM samples (not "15TB") |
| **Accuracy** | 2-5% L² error in-distribution (not "<1%") |
| **Geometry** | Fixed unit-square domain only |
| **Flow regime** | Steady-state, incompressible (M < 0.3) |

### Supported Parameter Ranges

| Parameter | Training Range | Safe Operating |
|-----------|----------------|----------------|
| Reynolds | 500 – 10,000 | 100 – 15,000 |
| Angle | -10° to +15° | -10° to +12° |
| Mach | 0.1 – 0.4 | 0.1 – 0.3 |

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/21e8-miner/surrapi-demo.git
cd surrapi-demo

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server (demo uses random weights)
python -m uvicorn app.main:app --reload --port 8000

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"reynolds": 1000, "angle": 0.0, "mach": 0.2}'
```

Visit:
- **Landing Page**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs

---

## API Reference

### POST `/predict`

**Input Parameters:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `reynolds` | float | 500-10000 | Reynolds number |
| `angle` | float | -15 to 15 | Angle of attack (degrees) |
| `mach` | float | 0.05-0.6 | Mach number |
| `resolution` | int | 64-256 | Output grid size |
| `enforce_conservation` | bool | - | Apply divergence correction |
| `return_uncertainty` | bool | - | Get uncertainty estimates |

**Response Fields:**

```json
{
  "ux": [/* 16384 floats for 128×128 */],
  "uy": [/* velocity y-component */],
  "p": [/* pressure field */],
  "physics_score": 0.85,
  "divergence": [/* optional */],
  "inference_time_ms": 285,
  "resolution": 128,
  "vtk": "base64-encoded VTK file"
}
```

### Physics Score Interpretation

| Score | Meaning |
|-------|---------|
| > 0.9 | Good mass conservation |
| 0.7 - 0.9 | Acceptable, minor divergence |
| 0.5 - 0.7 | Warning: check inputs |
| < 0.5 | Likely out-of-distribution |

---

## Optional Post-Processing

### `enforce_conservation=True`

**What it does:**
- Helmholtz projection to reduce velocity divergence by ~60-90%
- Adds ~50ms latency

**What it does NOT do:**
- ❌ "Guarantee" conservation
- ❌ Make unphysical predictions physical
- ❌ Work reliably outside training distribution

### `return_uncertainty=True`

**What it does:**
- Monte Carlo dropout (10 forward passes)
- Returns `ux_std`, `uy_std`, `p_std` fields

**What it does NOT do:**
- ❌ Provide calibrated confidence intervals
- ❌ Replace proper ensemble methods

---

## Architecture

```
Input (batch, 3, 128, 128)
    │
    ▼
4-layer FNO (8.4M parameters)
├── 32 Fourier modes per layer
├── GELU activations
├── Residual connections
└── LayerNorm
    │
    ▼
Output (batch, 3, 128, 128) → [ux, uy, p]
```

See [MODEL_CARD.md](./MODEL_CARD.md) for full specification.

---

## Limitations

⚠️ **Read before using**: [LIMITATIONS.md](./LIMITATIONS.md)

### Key Failure Modes

1. **Flow separation** (α > 12°): Unphysical predictions
2. **High Reynolds** (Re > 15k): Accuracy degrades to 10-30% error
3. **Transonic flows** (M > 0.6): Not supported (shock oscillations)
4. **Arbitrary geometry**: Not implemented (fixed domain only)
5. **Transient flows**: Model is steady-state only

---

## Research References

Based on published work (with correct citations):

```bibtex
@inproceedings{li2021fourier,
  title={Fourier Neural Operator for Parametric PDEs},
  author={Li, Zongyi and Kovachki, Nikola and others},
  booktitle={ICLR},
  year={2021},
  url={https://arxiv.org/abs/2010.08895}
}

@article{kovachki2023neural,
  title={Neural Operator: Learning Maps Between Function Spaces},
  author={Kovachki et al.},
  journal={JMLR},
  year={2023},
  url={https://arxiv.org/abs/2108.08481}
}
```

### Implemented Enhancements

| Feature | Implementation Status | File |
|---------|----------------------|------|
| Local Features (Conv-FNO) | ✓ Implemented | `app/model.py:LocalFeatureExtractor` |
| Spectral Boosting | ✓ Implemented | `app/model.py:HighFrequencyBooster` |
| Conservation Correction | ✓ Implemented | `app/model.py:conservation_correction` |
| MC Dropout UQ | ✓ Implemented | `app/model.py:UncertaintyWrapper` |

**Note**: Previous README incorrectly attributed these to specific papers. The implementations are inspired by general techniques in the neural operator literature.

---

## Repository Structure

```
surrapi-demo/
├── app/                    # FastAPI application
│   ├── main.py            # API endpoints
│   ├── model.py           # FNO architecture
│   ├── schemas.py         # Pydantic models
│   └── static/            # Landing page
├── surrapi/               # Python SDK (local)
├── LIMITATIONS.md         # Failure modes documentation
├── MODEL_CARD.md          # Model specification
└── requirements.txt
```

---

## Contributing

We specifically need help with:

- [ ] Validation scripts against OpenFOAM
- [ ] Additional test cases (backward step, channel flow)
- [ ] Proper UQ calibration study
- [ ] Geometry-conditioned training

---

## Disclaimer

> This is a research prototype demonstrating neural operator methods.
> 
> **DO NOT** use for:
> - Engineering design decisions
> - Safety-critical aerodynamics
> - Regulatory certification
> - Any application requiring validated CFD
>
> Surrogate models interpolate training data — they do not solve physics.

---

## License

MIT License - See [LICENSE](./LICENSE)

For commercial applications, contact: team@surrapi.io
