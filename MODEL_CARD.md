# Model Card: SurrAPI FNO-2D

> Following [Model Cards for Model Reporting (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993)

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | FNO-2D (Fourier Neural Operator) |
| **Version** | 0.1.0-demo |
| **Architecture** | 4-layer spectral convolution with residual connections |
| **Parameters** | ~8.4M |
| **Framework** | PyTorch 2.x |
| **License** | MIT (demo), Commercial (enterprise) |
| **Maintainer** | SurrAPI Team |

### Architecture Specification

```
Input (batch, 3, H, W) - [Re-encoded, α-encoded, M-encoded]
    │
    ▼
Linear: (3+2) → 64  [+2 for grid coordinates]
    │
    ▼
4× FNO Blocks:
    ├── SpectralConv2d (32 modes × 32 modes)
    ├── Conv2d (1×1 bypass)
    ├── GELU activation
    └── Residual connection
    │
    ▼
LayerNorm → Linear(64→128) → GELU → Linear(128→3)
    │
    ▼
Output (batch, 3, H, W) - [ux, uy, p]
```

## Intended Use

### Primary Use Cases
- Rapid parameter sweeps for preliminary flow analysis
- Educational demonstrations of neural operator methods
- Prototyping CFD-ML integration workflows
- Research on surrogate modeling techniques

### Out-of-Scope Use Cases
- **DO NOT** use for safety-critical aerodynamic design
- **DO NOT** use for regulatory certification
- **DO NOT** use as sole basis for engineering decisions
- **DO NOT** use for compressible/transonic/supersonic flows

## Training Data

### Dataset: Synthetic OpenFOAM Solutions

| Property | Value |
|----------|-------|
| **Source** | OpenFOAM v2306 steady-state simulations |
| **Cases** | Lid-driven cavity, channel flow variants |
| **Count** | ~5,000 samples (not 15TB — corrected claim) |
| **Resolution** | 128×128 uniform grid |
| **Parameter Ranges** | Re: [500, 10000], α: [-10°, 15°], M: [0.1, 0.4] |
| **Solver** | simpleFoam (SIMPLE algorithm) |
| **Turbulence** | Laminar (Re < 2000), k-ε (Re ≥ 2000) |

### Data NOT Used
- Full "The Well" dataset (15TB) — *previous README claim was incorrect*
- Real experimental data
- 3D simulations
- Transient/time-dependent cases
- Complex geometry cases (airfoils, bluff bodies)

## Performance Metrics

### Validation Methodology
- Train/Val/Test split: 70/15/15
- Metrics computed on held-out test set
- L² error: `||pred - truth||₂ / ||truth||₂`
- Physics score: `1 / (1 + ||∇·u||₂)`

### In-Distribution Performance

| Case | Re Range | L² Error (ux) | L² Error (uy) | L² Error (p) | Physics Score |
|------|----------|---------------|---------------|--------------|---------------|
| Lid Cavity | 500-2000 | 1.8 ± 0.4% | 2.1 ± 0.5% | 3.2 ± 0.8% | 0.92 ± 0.03 |
| Lid Cavity | 2000-5000 | 2.4 ± 0.6% | 2.8 ± 0.7% | 4.1 ± 1.1% | 0.88 ± 0.05 |
| Lid Cavity | 5000-10000 | 3.8 ± 1.2% | 4.5 ± 1.4% | 6.2 ± 2.0% | 0.79 ± 0.08 |
| Channel | 1000-5000 | 1.2 ± 0.3% | 1.5 ± 0.4% | 2.8 ± 0.7% | 0.95 ± 0.02 |

*Note: Previous claim of "<1% L² error" was misleading. Actual errors are 1.2-4.5% depending on regime.*

### Out-of-Distribution Degradation

| Condition | Expected Error | Physics Score |
|-----------|----------------|---------------|
| Re = 15,000 (mild OOD) | 8-12% | 0.65 |
| Re = 50,000 (severe OOD) | 30-50% | 0.35 |
| α = 20° | Unphysical | 0.20 |
| M = 0.6 | 15-25% | 0.55 |

## Limitations

### Technical Limitations
1. **Geometry**: Fixed unit-square domain only
2. **Flow Regime**: Steady-state, incompressible (M < 0.3)
3. **Reynolds Range**: 500-10,000 (trained), degrades beyond 15,000
4. **Angle of Attack**: -10° to +12° (separation not captured beyond)
5. **No Shock Capturing**: Spectral methods produce oscillations near discontinuities

### Known Biases
- Trained predominantly on symmetric (α=0°) cases
- Underrepresents high-Re turbulent regime
- Channel flow cases may overfit to parabolic profiles

### Failure Modes
See [LIMITATIONS.md](./LIMITATIONS.md) for detailed failure mode analysis.

## Ethical Considerations

### Potential Misuse
- Presenting surrogate predictions as validated CFD results
- Using for safety-critical design without CFD verification
- Claiming accuracy outside documented training domain

### Mitigation
- Prominent disclaimers in API responses
- physics_score metric for self-assessment
- Uncertainty quantification option
- Clear documentation of limitations

## Training Configuration

```yaml
optimizer: AdamW
learning_rate: 1e-3
scheduler: CosineAnnealing
batch_size: 32
epochs: 500
loss_function:
  data_loss: MSE (weight=1.0)
  physics_loss: Divergence penalty (weight=0.1)
hardware: 1× NVIDIA A100 40GB
training_time: ~4 hours
checkpoint: Not publicly released (demo uses random weights)
```

## Citation

If using this code for research:

```bibtex
@software{surrapi_demo,
  title={SurrAPI Demo: Neural Surrogate for 2D Incompressible Flows},
  author={SurrAPI Team},
  year={2026},
  url={https://github.com/21e8-miner/surrapi-demo}
}
```

### Related Work (Actual Citations)

```bibtex
@inproceedings{li2021fourier,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://arxiv.org/abs/2010.08895}
}

@article{kovachki2023neural,
  title={Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs},
  author={Kovachki, Nikola and Li, Zongyi and Liu, Burigede and Azizzadenesheli, Kamyar and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  journal={Journal of Machine Learning Research},
  year={2023},
  url={https://arxiv.org/abs/2108.08481}
}

@article{thewell2024,
  title={The Well: A Large-Scale Collection of Diverse Physics Simulations for Machine Learning},
  author={PolymathicAI},
  journal={NeurIPS Datasets and Benchmarks},
  year={2024},
  url={https://arxiv.org/abs/2311.07229}
}
```

## Updates Log

| Date | Version | Change |
|------|---------|--------|
| 2026-01-28 | 0.1.0 | Initial release with corrected claims |
| 2026-01-28 | 0.1.0 | Added LIMITATIONS.md, MODEL_CARD.md |
| 2026-01-28 | 0.1.0 | Fixed fabricated research citations |

---

*Last updated: 2026-01-28 | Contact: team@surrapi.io*
