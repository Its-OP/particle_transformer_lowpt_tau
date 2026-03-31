# Graph Space Diagnostic: GT-Neighbor Connectivity Across kNN Feature Spaces

**Date:** 2026-03-29
**Dataset:** Subset validation (1,141 GT pions across ~400 events)
**Goal:** Determine whether building kNN graphs in physics-motivated feature
spaces improves signal-signal connectivity over the baseline (eta, phi) graph.

---

## 1. Results

### Fraction of GT pions with >= 1 GT neighbor

| Space | k=4 | k=8 | k=12 | k=16 | k=24 | k=32 |
|-------|-----|-----|------|------|------|------|
| **eta_phi (baseline)** | 0.094 | 0.148 | 0.196 | **0.251** | 0.341 | **0.409** |
| opposite_sign eta_phi | 0.096 | 0.160 | 0.211 | 0.278 | 0.367 | 0.425 |
| logpt_eta_phi | 0.094 | 0.137 | 0.166 | 0.187 | 0.245 | 0.301 |
| dz_significance | 0.025 | 0.049 | 0.077 | 0.102 | 0.138 | 0.165 |
| vertex_proxy | 0.026 | 0.046 | 0.062 | 0.085 | 0.110 | 0.127 |

### Fraction of GT pions with >= 2 GT neighbors

| Space | k=4 | k=8 | k=12 | k=16 | k=24 | k=32 |
|-------|-----|-----|------|------|------|------|
| eta_phi | 0.002 | 0.015 | 0.032 | 0.054 | 0.081 | 0.116 |
| opposite_sign | 0.004 | 0.017 | 0.031 | 0.039 | 0.056 | 0.086 |
| logpt_eta_phi | 0.011 | 0.014 | 0.017 | 0.019 | 0.028 | 0.034 |
| dz_significance | 0.000 | 0.000 | 0.002 | 0.004 | 0.016 | 0.024 |
| vertex_proxy | 0.000 | 0.002 | 0.002 | 0.004 | 0.008 | 0.011 |

### Composite graph (union of all 5 spaces)

| Metric | Value |
|--------|-------|
| Effective K per track | 68 (16+12+12+16+12) |
| GT with >= 1 GT neighbor | **0.397** |
| GT with >= 2 GT neighbors | **0.266** |
| Mean GT neighbors per GT pion | 0.947 |

**Comparison:** baseline eta_phi at k=32 achieves 0.409 — nearly identical to the
68-neighbor composite. The union of 5 physics-motivated spaces performs no better
than simply doubling k in plain (eta, phi).

---

## 2. Analysis

### 2.1 Displacement spaces (dz_sig, vertex_proxy) perform worst

**Observed:** dz_significance at k=16 gives only 10.2% GT connectivity (vs 25.1%
baseline). Vertex proxy is even worse at 8.5%.

**Explanation:** The physics argument assumed signal pions would cluster at a
shared displaced dz value, separated from the dense PV pile at dz~0. In reality:

- **47% of GT pions have |dxy_sig| < 0.5** (the "uncanny valley"). These pions
  have minimal displacement, so their dz_significance is also near zero —
  placing them squarely inside the dense primary vertex pile, not in a
  low-density tail.
- With ~1100 tracks and most of them at dz~0, a kNN in dz space picks the
  k nearest tracks from the PV pile — almost all background.
- The tau flight distance is c*tau ~ 87 um. At low pT (the regime where we
  struggle most), the boost gamma is small, so the lab-frame displacement is
  often within tracking resolution. The dz_significance values for signal
  pions overlap heavily with background.
- Vertex proxy suffers the same issue compounded by the use of dxy_significance
  (d0/sigma) instead of raw d0. The sigma normalization distorts the vertex
  position mapping differently for each track, destroying the clustering
  that raw (v_x, v_y) would provide.

**Conclusion:** Displacement-based graphs only help the ~53% of signal pions that
ARE visibly displaced. But those are already the "easy" pions that the baseline
model finds with high recall (>90% at |dxy_sig| > 2). The hard pions (low
displacement) are exactly the ones that dz/vertex spaces cannot help.

### 2.2 Opposite-sign eta_phi: marginal improvement

**Observed:** +2.7% over baseline at k=16 (27.8% vs 25.1%).

**Explanation:** Filtering to opposite-sign pairs halves the candidate pool
(~550 instead of ~1100), extending the effective radius by ~sqrt(2) in 2D for
the same k. This creates a modest improvement. However:

- Only 1 of the 3 GT pions forms an opposite-sign pair with another GT pion
  (the rho daughters pi+pi-). The bachelor pion pairs with same-sign tracks.
- The improvement is real but small — the density reduction factor (2x) only
  extends ΔR reach by ~40%, which is not enough to bridge the typical
  inter-pion separation.

### 2.3 logpt_eta_phi: worse than baseline

**Observed:** 18.7% at k=16 vs 25.1% baseline.

**Explanation:** Adding log_pT as a 3rd coordinate *hurts* because it spreads
tracks along the pT axis, increasing the effective distance between signal pions
that have different momenta. In the tau → 3pi decay, the three pions have a wide
pT spread (bachelor ~0.3-0.5 GeV, rho daughters ~1-5 GeV). This means the 3
signal pions are far apart in (log_pT, eta, phi) space — further than in pure
(eta, phi). The z-score normalization doesn't fix this because the pT spread
of signal pions is genuinely large relative to the population std.

### 2.4 Composite union: no synergy

**Observed:** 39.7% with 68 effective neighbors ≈ baseline 40.9% with 32 neighbors.

**Explanation:** The additional spaces (dz, vertex, logpt) contribute edges that
largely point to the same background tracks or to different background tracks —
not to the missing signal pions. Since the non-eta-phi spaces perform worse than
baseline individually, their edges in the union are dominated by noise.
The composite's 39.7% is entirely explained by the eta_phi and opposite_sign
components contributing their ~30% combined at their respective k values, plus
some overlap reduction.

---

## 3. Root Cause: Signal Sparsity is the Fundamental Limit

The diagnostic confirms a deeper truth: **at 0.27% signal fraction, no per-track
feature space creates signal-enriched kNN neighborhoods.** The problem is not
that we're using the wrong coordinates — it's that 3 signal tracks among 1100
are too sparse for ANY fixed kNN graph to reliably connect them.

The math: with 3 signal tracks among P=1100, a random k-neighbor has probability
3/1100 ≈ 0.27% of being signal. For k=16, the expected number of signal neighbors
is 16 * 0.0027 = 0.044. The probability of >= 1 signal neighbor is
1 - (1 - 3/1100)^16 ≈ 4.3%. The observed 25% (baseline) is much higher than
random (4.3%), meaning (eta, phi) proximity IS correlated with signal — but
the correlation is weak and cannot be much improved by other feature spaces.

**What this means for the architecture:**
- Composite graph construction is NOT the path to breaking the 0.63 wall.
- The Stage 1 pre-filter's kNN graph is already near-optimal for its k budget.
- Simply increasing k (e.g., k=32 gives 41%) provides comparable gains to
  multi-space engineering, at lower complexity.
- The cascade approach (Stage 1 → top-600 → Stage 2 with O(K^2) attention)
  remains the most viable path: it sidesteps the graph problem entirely by
  giving Stage 2 access to ALL 600×600 pairwise interactions.

---
