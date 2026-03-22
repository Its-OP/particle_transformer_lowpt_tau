# Pre-Filter Improvement Blueprint

**Status:** Active implementation plan
**Date:** 2026-03-22
**Target:** R@200 >= 0.80 (from current 0.629)

---

## Phase 0: Validation & Metrics Infrastructure

### 0.1 Extend `compute_recall_at_k_metrics()` in `part/utils/training_utils.py`
- [x] Add K values: (10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 800)
- [x] Add GT rank percentiles: p50 (=median), p75, p90, p95
- [x] Add per-event breakdown: count events with N/M GT found (at K=200)
- [x] Add total_events_with_gt to returned metrics
- [x] Return all new metrics in the existing dict

### 0.2 Add conditional recall computation (new function)
- [x] Create `compute_conditional_recall()` in `part/utils/training_utils.py`
- [x] Accepts raw features (pT, dxy_sig) alongside scores/labels/mask
- [x] Computes recall by pT bins: [0, 0.3], [0.3, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0+]
- [x] Computes recall by |dxy_sig| bins: [0, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 5.0], [5.0+]
- [x] Computes 2D heatmap: recall in (pT_bin x dxy_sig_bin) grid
- [x] Returns dict with all conditional metrics (70 metrics total)

### 0.3 JSON export in training script
- [x] Create `save_epoch_metrics()` in `part/utils/training_utils.py`
- [x] Saves to `{experiment_dir}/metrics/epoch_{N}.json`
- [x] Include: all R@K, P@K, d-prime, median rank, percentiles, per-event breakdown

### 0.4 Update `validate()` in `part/train_prefilter.py`
- [x] Pass extended K values (10,20,30,50,100,200,300,400,500,600,800)
- [x] Log R@200, R@500, R@600, P@200, d-prime, median rank, p90
- [x] Save metrics JSON per epoch
- [x] Unit tests: 20 tests in test_extended_metrics.py (TDD — written before implementation)

---

## Phase 1: Loss + Training Recipe

### 1.1 RS@200 Surrogate Loss
- [ ] Implement `_rs_at_k_loss()` method in `weaver/weaver/nn/model/TrackPreFilter.py`
  - Two nested sigmoid relaxations:
    - Inner: sigma_tau2(s_pos - s_neg) estimates soft rank of each positive
    - Outer: sigma_tau1(K - soft_rank) gives soft indicator for being in top-K
  - Formula: RS@K = (1/|P|) * sum_p sigma_tau1(K - sum_n sigma_tau2(s_n - s_p))
  - Reference: Patel et al., CVPR 2022 (arXiv:2108.11179)
- [ ] Add `rs_at_k_weight_start/end` parameters to TrackPreFilter (default 0.0 = disabled)
- [ ] Add `rs_at_k_target` parameter (default 200)
- [ ] Add `rs_at_k_tau1`, `rs_at_k_tau2` temperature parameters
- [ ] Hybrid loss: `L = ranking_loss + weight * RS@200`
- [ ] Support annealing weight from 0 -> target during training (via set_temperature_progress)
- [ ] Unit tests: verify RS@K loss is differentiable, decreases with better rankings

### 1.2 Curriculum Training with Negative Subsampling
- [ ] Add `curriculum_num_negatives` parameter to `train_one_epoch()`
  - When set, randomly subsample N background tracks per event (keep all GT)
  - Implemented as mask modification before compute_loss()
- [ ] Add curriculum schedule to main training loop:
  - Phase A (epochs 0 to E/3): subsample to ~30 background tracks per event
  - Phase B (epochs E/3 to 2E/3): cosine increase from 30 -> full
  - Phase C (epochs 2E/3 to E): full tracks, activate RS@200, DRW
- [ ] Add `--curriculum` flag to `train_prefilter.py`
- [ ] Unit tests: verify subsampling preserves all GT tracks, correct count

---

## Phase 2: Two-Stage Cascade

### 2.1 R@K Diagnostic Sweep
- [ ] Create `part/diagnostics/recall_at_k_sweep.py`
  - Load trained pre-filter checkpoint
  - Run on full validation set
  - Report R@K and P@K for K = 50, 100, 200, 300, 400, 500, 600, 800, 1000
  - Report GT rank distribution: p50, p75, p90, p95, mean
  - Output: JSON + console table
  - This confirms cascade viability (expect R@600 >= 0.90)

### 2.2 Cascade Data Pipeline
- [ ] Create data generation script: `part/generate_cascade_data.py`
  - Load trained Stage 1 model
  - Run inference on train + val sets
  - For each event: select top-K1 tracks, save repacked tensors
  - Output: new parquet/pt files with K1 tracks per event
  - Include: features, lorentz_vectors, points, mask, labels, stage1_scores
- [ ] Or: implement lazy cascade in training (Stage 1 forward -> select -> Stage 2)

### 2.3 Track A: ParT-Style Pairwise-Bias Encoder (`CascadeReranker.py`)
- [ ] Create `weaver/weaver/nn/model/CascadeReranker.py`
- [ ] Implement `PairwiseBiasAttentionLayer`:
  - Standard multi-head self-attention
  - Pairwise physics features -> per-head scalar bias via MLP
  - Attn(Q,K,V) = softmax((QK^T)/sqrt(d_k) + pairwise_bias) * V
  - Formula: bias_h = MLP_h(pair_features)  # pair_features: (B, C_pair, P, P)
- [ ] Implement `CascadeReranker`:
  - Input: features (B, C, P), lorentz_vectors (B, 4, P), mask (B, 1, P)
  - Per-track embedding MLP: features -> d_model
  - Compute pairwise features for all PxP pairs:
    - pairwise_lv_fts() -> (B, 4, P, P): ln kT, ln z, ln DR, ln m^2
    - charge product: (B, 1, P, P)
    - vertex compatibility |d0_i - d0_j|: (B, 1, P, P)
    - Total: 6 pairwise channels
  - Stack of N PairwiseBiasAttentionLayers (default N=3)
  - Scoring head: MLP on per-track embeddings -> scalar score
  - Output: per-track scores (B, P)
- [ ] Implement `compute_loss()`:
  - Ranking loss (reuse from TrackPreFilter)
  - Optional RS@200 loss
  - Support DRW, temperature scheduling
- [ ] Default config: d_model=128, num_heads=4, num_layers=3, ~500K params
- [ ] Unit tests: forward pass shapes, gradient flow, pairwise feature computation

### 2.4 Track B: DETR-Style Query Decoder (`CascadeRerankerDETR.py`)
- [ ] Create `weaver/weaver/nn/model/CascadeRerankerDETR.py`
- [ ] Adapt from existing `TauTrackFinderHead` architecture:
  - Encoder: 1-2 self-attention layers on K1 tracks
  - Decoder: K=3-10 learned queries + cross-attention to encoded tracks
  - Per-query output: assignment probability over all K1 tracks
  - Per-track auxiliary head: direct binary classification
- [ ] Implement `compute_loss()`:
  - Hungarian matching between K queries and GT tracks
  - Mask CE loss on query-track assignments
  - Confidence loss on query activity
  - Per-track focal BCE auxiliary loss
  - Convert to ranking: per-track score = max query assignment probability
- [ ] Key adaptation from TauTrackFinderHead:
  - Remove backbone dependency (input is already enriched from Stage 1)
  - Reduce num_queries (3-10 vs original 30)
  - Add pairwise features as attention bias in cross-attention
- [ ] Unit tests: Hungarian matching, forward pass, gradient flow

### 2.5 Track C: Merged ParT Encoder + DETR Decoder (`CascadeRerankerMerged.py`)
- [ ] Create `weaver/weaver/nn/model/CascadeRerankerMerged.py`
- [ ] Architecture:
  - Encoder: 2-4 PairwiseBiasAttentionLayers (from Track A)
  - Decoder: 2-4 DETR cross-attention layers (from Track B)
    - Queries cross-attend to encoder output
    - Pairwise bias in cross-attention between queries and tracks
  - Per-track auxiliary scoring head
  - Per-query assignment head
- [ ] Implement `compute_loss()`:
  - Hybrid: ranking loss + Hungarian matching loss + per-track auxiliary
- [ ] Unit tests: end-to-end forward pass, gradient flow through both stages

### 2.6 Cascade Training Script (`part/train_cascade.py`)
- [ ] Create `part/train_cascade.py`
- [ ] Arguments: Stage 1 checkpoint path, K1, Stage 2 architecture choice
- [ ] Training modes:
  - Independent: freeze Stage 1, train Stage 2 on pre-selected top-K1
  - Joint (later): end-to-end with differentiable top-K
- [ ] Data pipeline: Stage 1 forward pass -> select_top_k -> Stage 2 input
- [ ] Evaluation: report BOTH Stage 1 R@K1 and end-to-end R@200
- [ ] Support all Phase 0 extended metrics
- [ ] Resume support, TensorBoard logging, checkpoint management

### 2.7 Network Wrapper (`part/networks/lowpt_tau_CascadeReranker.py`)
- [ ] Create `part/networks/lowpt_tau_CascadeReranker.py`
- [ ] `get_model()` function supporting Track A, B, C architectures
- [ ] Default hyperparameters per track
- [ ] Log parameter counts

---

## Phase 3: Physics-Informed Post-Processing (after cascade)

### 3.1 Anchor-Based Score Boosting
- [ ] Create `part/diagnostics/physics_anchoring.py`
- [ ] Algorithm:
  1. From top-200, find high-confidence opposite-charge pairs in top-50
  2. For each anchor pair, scan all ~1100 tracks:
     - M(pi_a, pi_c) or M(pi_b, pi_c) within [625, 925] MeV (rho window)
     - Charge: total triplet charge = +/-1
     - M(pi_a, pi_b, pi_c) within [1000, 1777] MeV (tau window)
  3. Boost compatible tracks: s_adj = s_orig + w * compatibility_score
- [ ] Evaluate: change in R@200, P@200, per-event 3/3 rate
- [ ] Requires: 4-vectors + charge for all ~1100 tracks (not just top-200)

---

## Phase 4: DETR on Full Event (contingency)

### 4.1 Full-Event DETR
- [ ] Only if cascade R@200 < 0.75 after Phase 2
- [ ] Adapt existing TauTrackFinder to work on 1100 tracks
- [ ] Add windowed self-attention encoder (HEPT-style or FlexAttention)
- [ ] K=3-10 queries cross-attend to all 1100 encoded tracks
- [ ] Pairwise physics features as cross-attention bias

---

## Decision Gates

| Gate | Metric | Action if met | Action if not met |
|------|--------|---------------|-------------------|
| After Phase 1 | R@200 >= 0.67 | Phase 2 with improved Stage 1 | Phase 2 anyway |
| After Phase 2 | R@200 >= 0.80 | Phase 3 for bonus | Phase 3 to close gap |
| After Phase 2 | R@200 < 0.75 | Phase 4 (DETR full event) | -- |
| After Phase 3 | R@200 >= 0.80 | Done | Phase 4 |

---

## Key References

| Paper | Relevance |
|-------|-----------|
| Patel et al., CVPR 2022 (arXiv:2108.11179) | RS@K surrogate loss |
| Qu et al., ICML 2022 (arXiv:2202.03772) | Particle Transformer, pairwise interaction ablation |
| IAFormer, 2025 (arXiv:2505.03258) | Pairwise features alone match ParT |
| Van Stroud et al., Phys Rev X 15, 2025 | MaskFormer for HEP tracking |
| Sander et al., ICML 2023 | Fast Differentiable Top-K |
| LCRON, 2025 (arXiv:2503.09492) | Joint cascade optimization |
| Locatello et al., NeurIPS 2020 | Slot Attention |
| CASED, MICCAI 2017 | Curriculum for <1% positive rates |
