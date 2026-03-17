# V3 Convergence Analysis: Why Both Backbones Hit the Same Ceiling

**Date**: 2026-03-17
**Models compared**: V3 frozen backbone (50 epochs), V3 trainable parallel backbone (50 epochs)
**Previous baseline**: V2 improved (10 epochs, R@30=22%)

---

## 1. Training Convergence

Both V3 variants converge to nearly identical recall and loss:

| Metric | V3 Frozen | V3 Trainable |
|--------|-------:|-------:|
| Total params | 1,697K | 422K |
| Trainable params | 363K | 422K |
| Best R@10 | 0.131 (ep 48) | 0.141 (ep 39) |
| Best R@20 | 0.187 (ep 44) | 0.197 (ep 39) |
| Best R@30 | 0.230 (ep 45) | 0.239 (ep 39) |
| Final val loss | 0.00271 | 0.00274 |
| Plateau epoch | ~15 | ~15 |

Both plateau at R@30 ~ 23% by epoch 15. The remaining 35 epochs produce < 1% improvement.

---

## 2. Per-Event Diagnostics (V3 Trainable, 200 events)

### Recall@K
| K | Mean | Median |
|---|-----:|-------:|
| 10 | 0.130 | 0.000 |
| 20 | 0.186 | 0.000 |
| 30 | 0.230 | 0.000 |
| 50 | 0.299 | 0.333 |
| 100 | 0.421 | 0.333 |

### Per-Event R@30 Distribution
| Recall | Events | Fraction |
|--------|-------:|-------:|
| 0% (0/3) | 109 | 54.8% |
| 33% (1/3) | 48 | 24.1% |
| 67% (2/3) | 31 | 15.6% |
| 100% (3/3) | 11 | 5.5% |

### Score Separation
- GT pion mean: -1.619, std: 0.495
- Background mean: -2.388, std: 1.018
- **d-prime: 0.962**

### GT Pion Rank Distribution
- mean=247, **median=151**, p75=405, p90=621
- 13.0% of GT pions rank in top 10
- 23.0% rank in top 30
- 17.5% rank beyond 500

---

## 3. Comparison with V2

| Metric | V2 (10 ep) | V3 Trainable (50 ep) | Change |
|--------|---:|---:|---|
| d-prime | 0.769 | 0.962 | +25% |
| Median GT rank | 292 | 151 | -48% (better) |
| R@30 | 0.146 | 0.230 | +57% |
| 0/3 events | 66.3% | 54.8% | -11pp |
| 3/3 events | 1.0% | 5.5% | +4.5pp |
| dxy_sig [3,5) found | 50.0% | 75.0% | +25pp |
| pT [1,2) GeV found | 27.9% | 49.4% | +21pp |
| pT [0,0.5) GeV found | 8.5% | 9.7% | +1.2pp |

V3 is meaningfully better than V2 at finding displaced and moderate-pT pions. But very low-pT pions (< 0.5 GeV, 45% of GT) barely improve.

---

## 4. Where the Model Succeeds and Fails

### Succeeds (found rate > 30%)
- High dxy_significance (|dxy| > 2): 40-75% found rate
- Higher pT (> 1 GeV): 49-80% found rate
- These are genuinely discriminable: background has much lower dxy tails

### Fails (found rate < 10%)
- Low pT (< 0.5 GeV): 9.7% found rate — these are 45% of all GT pions
- Small |dxy_significance| (< 1): 2-4% found rate — these are 43% of all GT pions
- The model essentially cannot distinguish these from background

### Root Cause
- Low-pT, non-displaced GT pions have feature distributions nearly identical to background
- pT distributions overlap almost completely below 1 GeV (see dataset analysis)
- dxy_significance distributions overlap in [-1, 1] range
- No amount of architectural innovation can separate what the features don't distinguish

---

## 5. Key Finding

**The R@30 ~ 23% ceiling is a feature-level ceiling, not an architecture ceiling.**

Three fundamentally different architectures (V2 max-pool + refinement, V3 frozen GAPLayers, V3 trainable parallel GAPLayers) all converge to the same performance. The model finds what CAN be found (displaced, moderate-pT pions) and fails on what CANNOT be found with the current 7 features.

### Implications
1. **Architecture changes alone will not break the ceiling** — the bottleneck is the feature space
2. **Loss function changes may help marginally** — better ranking of borderline cases, but won't help with indistinguishable features
3. **New features are needed** to break past 23% — candidates:
   - Track quality variables (chi2/ndf, number of hits, pixel hit pattern)
   - Impact parameter in z (dz_significance) — adds a second displacement dimension
   - Track isolation (number of nearby tracks, local density)
   - Vertex compatibility between track pairs (shared vertex probability)
   - Track origin information (if available from upstream reconstruction)
4. **The CERN low-pT tau algorithm uses vertex fitting as post-processing** — tau vertex compatibility, flight length significance, and sum of ABCNet scores. This physics-based combination step may be essential for breaking past per-track classification limits.
