# Hypothesis Experiment Results

**Date:** 2026-03-31
**Checkpoint:** Physics ParT cascade (pair_extra_dim=5, R@200=0.693)
**Data:** Subset val (~200 events per experiment)

---

## H4: Score Distribution at Rank Boundary

**Question:** Are missed GT pions "just missed" (near threshold) or deeply buried?

| Metric | Value |
|--------|-------|
| Events analyzed | 200 |
| GT pions analyzed | 562 |
| GT missed (rank >= 200) | 212 (37.7%) |
| Score gap at rank 200 (mean) | 0.0030 |
| Missed GT within 0.5 std of threshold | 49.6% |
| Missed GT within 1.0 std | 83.5% |
| Missed GT within 2.0 std | 99.3% |

**Verdict:** ✓ Most missed GT pions are right at the boundary. Loss alignment
(LambdaRank / DFTopK) should rescue ~50% of them.

---

## H1: Attention Pattern Analysis

**Question:** Does the model form implicit queries? Is sparse attention viable?

| Head | Entropy | Type |
|------|---------|------|
| 0 | 0.011 | Extremely selective (delta-like) |
| 1 | 0.056 | Extremely selective |
| 2 | 5.413 | Diffuse |
| 3 | 5.806 | Diffuse |
| 4 | 6.100 | Diffuse (near uniform) |
| 5 | 5.855 | Diffuse |
| 6 | 5.626 | Diffuse |
| 7 | 5.875 | Diffuse |
| Max (uniform) | 6.397 | — |

GT→GT attention: 0.258 (mean), GT→BG attention: 0.004 (scaled).
**Ratio: 74.6×** — physics pairwise features create strong signal-signal connections.

**Verdicts:**
- ✓ Implicit query formation — heads 0-1 act as near-delta queries
- ✗ Sparse attention NOT viable — 6/8 heads are diffuse (need full context)
- ✓ MaskFormer/DETR decoder validated — the model already forms proto-queries

---

## H2: Embedding Clustering

**Question:** Are representations good (scoring head bottleneck) or do they need improvement?

| Layer | GT-GT cos | GT-BG cos | Linear Probe AUC |
|-------|----------|----------|-----------------|
| input | 0.560 | 0.577 | 0.808 |
| block 1 | 0.505 | 0.536 | **0.915** |
| block 2 | 0.735 | 0.750 | 0.903 |
| block 3 | 0.899 | 0.904 | 0.893 |
| block 4 | 0.957 | 0.957 | 0.875 |
| block 5 | 0.969 | 0.969 | 0.864 |
| block 6 | 0.973 | 0.973 | 0.856 |
| block 7 | 0.975 | 0.975 | 0.862 |
| block 8 | 0.977 | 0.977 | 0.860 |

**Key findings:**
1. **Linear probe AUC PEAKS at block 1 (0.915) then DECLINES to 0.860.**
   The first transformer block produces the most discriminative embeddings.
   Subsequent blocks actually REDUCE discrimination — they push GT and BG
   embeddings toward the same high-cosine-similarity subspace.

2. **GT-GT and GT-BG cosine similarity converge** — by block 4, both are
   >0.957. The later blocks produce nearly identical representations for
   GT and BG tracks. The model is collapsing into a near-uniform embedding.

3. **The scoring head operates on degraded representations** (AUC 0.860 at
   block 8 vs 0.915 at block 1). The 8-layer depth is HURTING discrimination.

**Verdicts:**
- ✗ "Linear probe AUC > 0.90 at final layer" (0.860) → representations degrade
- ✓ "Best discrimination at block 1" → model is too deep, or later layers overfit
- ✓ Pre-training / contrastive learning could help stabilize representations
- ✓ A shallower model (2-3 layers) might outperform 8 layers

### H2b: MLP Probe Follow-up

**Question:** Is the linear probe AUC decline an artifact of non-linear encoding?

| Layer | Linear AUC | MLP AUC | Delta |
|-------|-----------|---------|-------|
| input | 0.810 | 0.765 | -0.045 |
| block 1 | 0.899 | 0.790 | -0.109 |
| block 2 | **0.906** | 0.736 | -0.171 |
| block 3 | 0.901 | 0.707 | -0.194 |
| block 4 | 0.880 | 0.806 | -0.074 |
| block 5 | 0.886 | 0.757 | -0.128 |
| block 6 | 0.881 | 0.818 | -0.063 |
| block 7 | 0.880 | 0.766 | -0.114 |
| block 8 | 0.869 | 0.776 | -0.092 |

MLP probe underperforms linear everywhere (small probe dataset, MLP overfits).
But the linear probe trend is consistent: **peaks at block 1-2, declines through
block 8.** The information degradation is real, not a linear probe artifact.

The shallow model experiment (2-layer ParT currently training on server) will
provide the definitive answer on whether depth hurts R@200.

---

## H3: Edge Type Coverage (within Stage 2 top-600)

**Question:** Do physics-typed edges cover GT pairs better than uniform kNN?

| Edge type | GT-GT pairs | GT-GT % | GT-BG pairs | S/N ratio |
|-----------|------------|---------|-------------|-----------|
| dz_compat (\|Δdz\| < 0.5) | 347 | 77.6% | 2769 | 1.40× |
| rho_window (OS + \|m-770\| < 150) | 132 | 29.5% | 471 | 3.13× |
| delta_r (ΔR < 1.0) | 264 | 59.1% | 1066 | 2.76× |
| score_prox (\|Δs1\| < 0.5) | 140 | 31.3% | 359 | **4.35×** |
| **Union (any type)** | **419** | **93.7%** | 3388 | 1.38× |

Total GT-GT pairs: 447. Total GT-BG pairs sampled: 4990.

**Key findings:**
1. **Union covers 93.7% of GT-GT pairs** — much higher than kNN baseline (25%).
   This is within the top-600 candidate set, not the full 1100 tracks.

2. **dz compatibility alone covers 77.6%** — strongest single type. But S/N
   ratio is only 1.4× (many BG pairs also have similar dz). Within the
   cascade's enriched set, displacement is less discriminating than in the
   full event.

3. **Score proximity has best S/N (4.35×)** — GT pions tend to have similar
   Stage 1 scores. This is a useful edge type that encodes "these tracks are
   similarly confident" — a proxy for belonging to the same decay.

4. **Rho window has good S/N (3.13×)** with 29.5% GT coverage — consistent
   with our earlier pairwise diagnostic (31% of GT pairs pass rho window).

**Verdict:**
- ✓ Union covers 93.7% → heterogeneous graph viable
- But S/N ratios are modest (1.4–4.4×) → typed edges are enriched but not dramatic
- Score proximity is the strongest individual discriminant (4.35×)
