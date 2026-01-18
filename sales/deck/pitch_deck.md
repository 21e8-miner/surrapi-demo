# SurrAPI Pitch Deck
## CFD in 300ms, not 3 hours.

This document contains the complete slide-by-slide content for the SurrAPI investor/customer pitch deck.

---

## Slide 1: Title

**SurrAPI**
*Surrogate-as-a-Service for CFD*

CFD in 300ms, not 3 hours.

[Logo: 🌊 SurrAPI]

---

## Slide 2: The Pain

**Problem Statement**

Traditional CFD workflows are broken:

| Current Reality | Impact |
|----------------|--------|
| 2-6 hour solve times | Kills iteration speed |
| $50-200 per simulation | Blows budget on exploration |
| HPC cluster required | IT dependency & queue times |
| PhD-level expertise | Talent bottleneck |

> "We only run 3 design variants because each costs $180 in compute."
> — Aerospace PM at major OEM

---

## Slide 3: The Solution

**SurrAPI: Instant CFD Predictions**

- **300ms response time** (not hours)
- **$0.25 per prediction** (not $50+)
- **Simple REST API** (not HPC cluster)
- **No CFD expertise required**

```bash
curl -X POST "https://api.surrapi.io/predict" \
  -H "Content-Type: application/json" \
  -d '{"reynolds": 5000, "angle": 5, "mach": 0.3}'
```

---

## Slide 4: Demo

[LIVE DEMO]

Show interactive browser demo:
1. Adjust Reynolds slider
2. Change angle of attack
3. Click "Predict"
4. Watch flow field appear in 300ms

https://demo.surrapi.io

---

## Slide 5: The Technology

**Fourier Neural Operator (FNO)**

Trained on 15TB of The Well physics simulation data (NeurIPS 2024).

**Architecture:**
- Spectral convolutions (global receptive field)
- ~8M parameters
- Mixed precision inference
- Physics-aware loss functions

**Validation:**
- <1% L² error vs Ansys Fluent
- Re 500-10,000 range
- Published benchmark reproducibility

---

## Slide 6: TAM/SAM/SOM

**Market Size**

| Segment | Size | Source |
|---------|------|--------|
| TAM: Cloud HPC Simulation | $18.2B | Hyperion 2024 |
| SAM: CFD-specific | $4.5B | Technavio 2024 |
| SOM: Year 1 (SMB CFD consultants) | $50M | Bottom-up |

**Growth:** 21% CAGR in cloud simulation spending.

---

## Slide 7: Competition Matrix

**Competitive Positioning**

|                      | Speed | Cost | Ease of Use | Accuracy |
|----------------------|-------|------|-------------|----------|
| Ansys Cloud          | ⭐    | ⭐   | ⭐⭐        | ⭐⭐⭐⭐⭐ |
| SimScale             | ⭐⭐  | ⭐⭐⭐ | ⭐⭐⭐⭐    | ⭐⭐⭐⭐   |
| NVIDIA SimNet        | ⭐⭐⭐ | ⭐⭐  | ⭐⭐        | ⭐⭐⭐⭐   |
| **SurrAPI**          | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐   |

**Key differentiator:** Pay-per-prediction, not core-hours.

---

## Slide 8: Business Model

**Revenue Model**

1. **Usage-Based API**
   - $0.25 per prediction
   - Tiered monthly plans ($0/199/custom)
   - 70% gross margin

2. **Enterprise On-Prem**
   - Docker container license
   - ITAR-compliant deployment
   - Custom model fine-tuning

3. **Future: CAD Plugin Marketplace**
   - Fusion 360, Onshape, SolidWorks
   - $15/seat/month

---

## Slide 9: Traction

**Early Metrics**

| Metric | Value |
|--------|-------|
| Launch | 6 weeks ago |
| Beta users | 47 |
| Total predictions | 18,000+ |
| MRR | $3,200 |
| NPS | 72 |

**Notable pilots:**
- 3 aerospace tier-2 suppliers
- 1 automotive OEM (EV startup)
- 2 HVAC design consultancies

---

## Slide 10: Roadmap

**Development Timeline**

**Q1 2026 (NOW)**
- [x] Core API launch
- [x] 2D Navier-Stokes surrogate
- [x] Interactive demo

**Q2 2026**
- [ ] 3D surrogate (128³ grid)
- [ ] Custom geometry upload
- [ ] On-prem container GA

**Q3 2026**
- [ ] Fusion 360 plugin
- [ ] Multi-physics (MHD, thermals)
- [ ] Enterprise auth (SAML/SSO)

**Q4 2026**
- [ ] Real-time optimization API
- [ ] GPU cluster auto-scaling
- [ ] Series A raise

---

## Slide 11: The Ask

**Raising $500K SAFE**

- 20% discount
- $5M cap
- 18-month runway

**Use of Funds:**
| Category | Allocation |
|----------|------------|
| Engineering (2 FTE) | 55% |
| GPU compute | 25% |
| Sales/Marketing | 15% |
| Legal/Ops | 5% |

---

## Slide 12: Team

**Founders**

**[Your Name]** — CEO
- 8 years HPC/CFD at [Previous Company]
- Built simulation infrastructure serving 200 engineers
- MS Aerospace Engineering, Stanford

**[Co-founder Name]** — CTO
- Former DeepMind Research Engineer
- Published 12 papers on neural operators
- PhD Applied Math, Cambridge

**Advisors:**
- [Advisor 1] — VP Engineering, Ansys (ret.)
- [Advisor 2] — Partner, [VC Firm]

---

## Slide 13: Call to Action

**Try it before the coffee gets cold.**

[QR Code to demo.surrapi.io]

📧 team@surrapi.io
🌐 surrapi.io
📱 Schedule: calendly.com/surrapi

---

## Appendix Slides

### A1: Technical Deep Dive

**Training Data:**
- 15TB from The Well (Polymathic AI)
- 2D/3D Navier-Stokes, Euler, MHD
- Re 100 - 100,000 range
- Mixed BC types

**Model Details:**
- 4 Fourier layers, 32 modes
- 64 hidden dimension
- ~8M trainable parameters
- GELU activation
- AdamW optimizer

### A2: Accuracy Analysis

| Dataset | Train Size | L² Error | Inference |
|---------|------------|----------|-----------|
| NS 2D Re1k | 5,000 | 0.62% | 180ms |
| NS 2D Re10k | 8,000 | 1.12% | 195ms |
| Euler 2D | 3,000 | 0.89% | 190ms |

### A3: Unit Economics

| Item | Value |
|------|-------|
| GPU cost per prediction | $0.08 |
| API price | $0.25 |
| Gross margin | 68% |
| CAC (estimated) | $150 |
| LTV (12mo @ 5k calls/mo) | $15,000 |
| LTV:CAC ratio | 100:1 |

---

*Deck last updated: January 2026*
