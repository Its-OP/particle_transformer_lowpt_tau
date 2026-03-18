# Pre-Filter Diagnostics Report

**Date**: 2026-03-18
**Model**: TrackPreFilter (hybrid+asl, hidden=128, 2 message rounds, latent=16)
**Checkpoint**: `models/debug_checkpoints/prefilter/checkpoints/best_model.pt` (50 epochs, full dataset)
**Eval**: 800 events from subset, 2286 GT pions

---

## Overall Performance
- R@200 mean = 59.6%
- R@100 = 41.9%, R@30 = 24.4%
- d-prime = 1.29, median GT rank = 132

## 1. Per-Event R@200 Distribution

| R@200 | Events | Fraction |
|-------|-------:|-------:|
| 0% (0/3) | 133 | 16.6% |
| 33% (1/3) | 161 | 20.2% |
| 67% (2/3) | 240 | 30.0% |
| 100% (3/3) | 265 | 33.2% |

## 2. R@200 vs Event Track Count

| Track Count | Events | R@200 | Perfect (3/3) |
|------------|-------:|------:|--------:|
| [0, 500) | 2 | 0.667 | 50.0% |
| [500, 800) | 61 | 0.497 | 24.6% |
| [800, 1100) | 187 | 0.571 | 30.5% |
| [1100, 1400) | 254 | 0.619 | 37.4% |
| [1400, 1700) | 196 | 0.576 | 30.6% |
| [1700, 3000) | 99 | 0.643 | 35.4% |

No strong multiplicity dependence — performance is relatively uniform.

## 3. Found-in-Top-200 vs pT

| pT Range (GeV) | GT Pions | Found Rate |
|----------------|-------:|----------:|
| [0.0, 0.3) | 455 | **64.4%** |
| [0.3, 0.5) | 546 | **45.2%** |
| [0.5, 1.0) | 737 | 52.8% |
| [1.0, 2.0) | 381 | 76.6% |
| [2.0, 5.0) | 153 | 85.0% |
| [5.0, 50.0) | 14 | 85.7% |

**"Uncanny valley" at pT 0.3-0.5 GeV**: Lowest found rate (45.2%). These pions are energetic enough to look like typical background but not enough to stand out kinematically.

Surprisingly, very low pT (< 0.3 GeV) does better (64.4%) — possibly because these extremely soft pions are rare enough among background that they become somewhat distinctive.

## 4. Found-in-Top-200 vs |dxy_significance|

| |dxy_sig| Range | GT Pions | Found Rate |
|---------------|-------:|----------:|
| [0.0, 0.5) | 641 | **31.2%** |
| [0.5, 1.0) | 460 | 41.1% |
| [1.0, 2.0) | 465 | 72.5% |
| [2.0, 3.0) | 228 | 92.1% |
| [3.0, 5.0) | 211 | 95.3% |
| [5.0, 20.0) | 281 | 80.4% |

dxy_significance is the dominant discriminant. The 47% of GT pions with |dxy| < 0.5 have only 31% found rate.

## 5. Score Distributions

| | Mean | Median | Std |
|---|-----:|-------:|----:|
| GT pion scores | -1.446 | -1.646 | 1.294 |
| Background scores | -3.387 | -2.960 | 1.692 |
| **d-prime** | **1.288** | | |

## 6. GT Pion Rank Distribution

| Rank Range | Count | Fraction |
|-----------|------:|-------:|
| [0, 30) | 601 | 26.3% |
| [30, 100) | 417 | 18.2% |
| [100, 200) | 345 | 15.1% |
| [200, 500) | 573 | 25.1% |
| [500, 3000) | 350 | 15.3% |

Median rank = 132, p75 = 355, p90 = 596

## 7. Failure Cases (R@200 = 0%)

133 events (16.6%) have all 3 GT pions outside top-200.

Failed GT pion characteristics (n=356):
- Median pT = 0.476 GeV (in the uncanny valley)
- Median |dxy| = 0.530 (non-displaced)
- 53.4% have pT < 0.5 GeV
- **79.2% have |dxy| < 1.0**

The failure mode is clear: **soft, non-displaced pions are indistinguishable from background** with the current 7 features.

## Key Takeaway

The pre-filter effectively finds displaced and moderate-to-high pT pions (>90% at |dxy|>2, >85% at pT>2). The remaining challenge is the ~47% of GT pions in the "uncanny valley": pT 0.3-0.5 GeV with |dxy| < 0.5, where feature distributions overlap with background.
