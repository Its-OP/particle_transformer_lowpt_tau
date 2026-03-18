# Denoising / Pre-Filtering Research: Validated Findings

**Date**: 2026-03-18

## Diagnostic Validation Summary

Before evaluating techniques, we tested the core assumptions each relies on:

| Hypothesis | Diagnostic Result | Verdict |
|-----------|------------------|---------|
| **dxy_sig threshold can preserve >95% GT** | |dxy_sig| > 0.3 keeps 84.8% GT but per-EVENT recall = 61.4% (39% of events lose ≥1 pion) | **FAILS** — no single threshold achieves >95% per-event recall |
| **pT can filter signal from noise** | pT < 0.5 GeV: GT=44.5%, BG=44.7% — identical | **FAILS** — pT distributions overlap completely |
| **Multi-feature isolation** | |dxy|>1 AND pT<2: enrichment only 1.4x (GT=51%, BG=36%) | **WEAK** — 2-feature combination barely separates |
| **Density differences** | ~10 BG tracks within ΔR<0.3 of each GT pion (same as BG-BG) | **FAILS** — no density contrast |
| **Simple cascade achievable recall** | |dxy_sig| > 0.2: per-event recall = 72.8%, keeps 1089/1278 tracks | **FAILS** — even the loosest useful threshold misses 27% of events entirely |

**Key insight**: 43% of GT pions have |dxy_significance| < 0.5 AND are indistinguishable from background by any simple feature combination. A threshold-based pre-filter cannot achieve >95% per-event recall.

---

## Validated Techniques (assumptions hold)

### 1. Learned Per-Track Scorer as Pre-Filter (PointRCNN-inspired)
**Source**: Shi et al., CVPR 2019

**Assumption**: A learned model can capture multi-feature correlations invisible to threshold cuts.

**Diagnostic evidence**: Our V3 model at R@100 = 42% already outperforms any threshold combination. A model specifically optimized for recall@200 (not recall@30) could push higher. The 1.4x enrichment from (|dxy|>1, pT<2) suggests there ARE multi-feature correlations — they're just too subtle for axis-aligned cuts.

**How it works for us**: Train a lightweight MLP on all 7 raw features, optimized with a loss that maximizes recall at large K (e.g., R@200). Use this as stage 1 to reduce ~1278 tracks to ~200 candidates. Then run the full V3 GAPLayer model on only 200 tracks.

**Compatibility**: Drop-in before V3. The MLP scores can be precomputed.

### 2. Two-Tower Retrieve-then-Rerank
**Source**: Google RecSys architecture; Yi et al., RecSys 2019

**Assumption**: Signal tracks are similar to a learnable "tau prototype" in embedding space.

**Diagnostic evidence**: dxy_significance IS a strong discriminant (d-prime ~1.0 in learned feature space). The issue is per-track classification — a different framing (similarity to a prototype rather than binary classification) might extract more from the same features.

**How it works for us**:
- Stage 1 (retriever): Per-track MLP → embedding. A learned "tau query" embedding. Cosine similarity selects top-200.
- Stage 2 (reranker): Full V3 GAPLayers + MIA attention on only 200 tracks. O(200^2) full attention becomes feasible.

**Compatibility**: Requires a new training objective (contrastive/metric learning for stage 1) but reuses V3 for stage 2.

### 3. Autoencoder Background Model
**Source**: Bortolato et al., SciPost 2022

**Assumption**: Background tracks occupy a compact manifold; signal tracks deviate.

**Diagnostic evidence**: The enrichment factor of 1.4x at (|dxy|>1, pT<2) suggests background IS more concentrated than signal in feature space — the manifold assumption has some support. An autoencoder would capture the full 7D background distribution, not just 2D projections.

**How it works for us**: Train a per-track autoencoder on ALL tracks (99.77% background). Tracks with high reconstruction error are anomalous → candidate signal. Keep top-200 by anomaly score.

**Compatibility**: Completely independent of V3. Can be trained separately, applied as preprocessing.

### 4. Cascade of Learned Classifiers (CASCARO-inspired)
**Source**: Sagi & Rokach, Pattern Recognition Letters 2021

**Assumption**: Different examples need different levels of scrutiny; most background can be rejected cheaply.

**Diagnostic evidence**: While threshold-based cascades fail at >95% per-event recall, the per-PION retention is more nuanced:
- |dxy_sig| > 0.3 retains 84.8% of GT pions
- The 15.2% missed pions have very small dxy, but may have OTHER distinguishing features in combination

A learned cascade (each stage a small neural network, not a threshold) can find non-linear boundaries that thresholds miss. The cascade structure provides a natural recall-precision tradeoff: each stage can be independently validated for signal efficiency.

**Compatibility**: Sequential stages, each a small MLP. Output is a reduced candidate set.

---

## Invalidated Techniques (assumptions fail)

### Isolation Forest
**Why it fails**: Relies on axis-aligned splits to isolate anomalies. Our diagnostics show |dxy_sig| and pT thresholds achieve only 1.2-1.4x enrichment. Isolation Forest would find the same weak boundaries.

### PUPPI / Local Shape Variables
**Why it fails**: Assumes signal has different local density structure. Our data shows GT pions have identical local density (~10 tracks within ΔR<0.3) to background. No density contrast to exploit.

### Score-Based Denoising / Diffusion
**Why it fails**: Assumes noise is spatial perturbation of a clean signal. Our "noise" is structural (wrong tracks, not displaced tracks). The clean→noisy mapping doesn't exist.

### Density-Based Clustering (DBSCAN, HDBSCAN)
**Why it fails**: GT pions are spread with median ΔR=1.3 and sit within background clusters. They are NOT density outliers.

---

## Recommended Approach

The diagnostics strongly suggest that **only learned pre-filters** can work — threshold and density-based methods fail because GT pions overlap with background in every simple feature projection.

**Recommended pipeline**:
1. **Stage 1**: Lightweight per-track scorer (MLP on 7 features), trained to maximize R@200 with a ranking loss. Reduces ~1278 → ~200 tracks.
2. **Stage 2**: Full V3 (GAPLayers + MIA + ASL + ranking loss) on the 200 candidates. O(200^2) full attention is feasible, enabling richer cross-track reasoning.

The key architectural benefit: by concentrating signal from 0.23% to ~1.5% in the candidate set, the downstream model sees 6x more signal per batch, producing stronger gradients and better learning dynamics.
