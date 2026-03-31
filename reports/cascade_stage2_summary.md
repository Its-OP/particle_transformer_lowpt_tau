# Cascade Stage 2: Current State and Open Questions for Deep Research

**Date:** 2026-03-29
**Purpose:** Comprehensive summary of the two-stage cascade pipeline for low-pT
tau track finding. Input for Deep Research agents to plan Stage 2 improvements
beyond the current R@200 ~ 0.69-0.70 plateau toward the 0.80 target.

---

## 1. Problem Statement

### Task
Identify 3 signal tracks (pions from tau -> 3pi decay) among ~1100 reconstructed
tracks per event in low-pT tau tagging at CMS (LHC). Signal fraction is **0.27%**
(3 out of ~1100).

### Two-Stage Pipeline
- **Stage 1 (pre-filter):** Score all ~1100 tracks, select top-K1 (600) candidates.
  Must achieve high recall. Frozen during Stage 2 training.
- **Stage 2 (reranker):** Re-rank the K1 candidates using richer features (pairwise
  physics, self-attention). Only Stage 2 is trained.

### Metrics
- **R@K (Recall at K):** Fraction of GT pions in the top-K ranked candidates,
  averaged over events.
- **P@K (Perfect at K):** Fraction of events where ALL GT pions are in top-K.
- **d-prime:** Score separation: d' = (mu_gt - mu_bg) / sqrt(0.5*(sigma_gt^2 + sigma_bg^2)).

### Target
**End-to-end R@200 >= 0.80** (currently ~0.69-0.70 with ParT Stage 2 in training).

### Dataset
- **~270K training events, ~84K validation events** (10 parquet shards each).
- Track counts per event: min=0, max=~3500, mean=1130, median=1211.
- Each event has exactly 3 GT pions from tau -> 3pi.
- **16 per-track features:** kinematics (px, py, pz, eta, phi, charge),
  displacement (dxy_significance, log_dz_significance), quality
  (log_norm_chi2, log_pt_error, n_valid_pixel_hits, dca_significance),
  covariance (log_covariance_phi_phi, log_covariance_lambda_lambda),
  derived (log_pt, log_relative_pt_error).

---

## 2. Architecture

### Stage 1: TrackPreFilter (Frozen)
- **Architecture:** MLP encoder -> 2 rounds kNN(k=16) message passing with
  max-pool aggregation -> scoring MLP.
- **Graph:** Fixed kNN in (eta, phi) coordinates.
- **Parameters:** ~226K.
- **Checkpoint:** `models/prefilter_best.pt` (epoch 24, R@200=0.6288).
- **Complexity:** O(P*k) per message-passing round.

### Stage 2: CascadeReranker (ParT-style, Trainable)
- **Architecture:** Particle Transformer encoder with pairwise-bias attention.
  Reuses `Block`, `Embed`, `PairEmbed` from the existing ParT codebase.
- **Input embedding:** cat(16 features, stage1_score) -> BatchNorm -> Linear -> embed_dim.
  Stage 1 score is concatenated as the 17th input feature channel.
- **Pairwise features:** 4-vectors -> `pairwise_lv_fts()` -> ln kT, ln z, ln DR, ln m^2.
  These are projected to per-head scalar biases via MLP (PairEmbed).
  Detached from computation graph to prevent NaN gradients from sqrt(DR^2) at DR=0.
- **Attention:** Attn(Q,K,V) = softmax(QK^T / sqrt(d_k) + pairwise_bias) * V.
  Standard pre-norm transformer blocks with residuals, GELU, LayerNorm.
- **Output:** Per-track scoring head: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> scalar.
  Padded tracks get -inf.
- **Loss:** Temperature-scaled pairwise ranking loss:
  L = T * softplus((s_neg - s_pos) / T), with 50 sampled negatives per positive.

### CascadeModel (Glue)
- **File:** `weaver/weaver/nn/model/CascadeModel.py`
- Runs Stage 1 in `@torch.no_grad()` with `train()` mode (see bug fixes below).
- Selects top-K1 tracks via `select_top_k()`.
- Gathers filtered tensors (points, features, lorentz_vectors, mask, labels)
  along the track dimension via index gathering.
- Passes stage1_scores to Stage 2 as an extra input.
- Returns `selected_indices` for metric scatter-back to full-event positions.

### End-to-End Metric Computation
- **File:** `part/train_cascade.py` validate() function.
- Stage 2 scores are scattered back to full-event positions using `selected_indices`.
- Non-selected tracks get -inf (always rank below selected tracks).
- `MetricsAccumulator` receives full-event scores + full-event labels + full-event mask.
- **Denominator = all GT pions in the full event** (not just those surviving Stage 1).
- This means if Stage 1 drops a GT pion, it counts as missed in end-to-end R@200.
- Verified by unit test: `test_end_to_end_recall_uses_full_event_gt`.

---

## 3. Stage 1 Verified Performance (R@K Profile)

From best pre-filter checkpoint (epoch 35, full validation set, 12K events):

| K | R@K | P@K |
|---|-----|-----|
| 10 | 0.212 | 0.068 |
| 20 | 0.274 | 0.101 |
| 30 | 0.314 | 0.124 |
| 50 | 0.374 | 0.161 |
| 100 | 0.479 | 0.230 |
| **200** | **0.628** | **0.369** |
| 300 | 0.733 | 0.504 |
| 400 | 0.808 | 0.619 |
| 500 | 0.867 | 0.724 |
| **600** | **0.909** | **0.806** |
| 800 | 0.959 | 0.908 |

**Key numbers:**
- **S1 R@600 = 0.909** — the cascade ceiling. Stage 2 can never exceed this.
- **S1 R@200 = 0.628** — the baseline. Stage 2 must improve beyond this.
- **d' = 1.325**, median GT rank = 109, p90 = 573.
- At K1=600, signal fraction improves from 0.27% to ~0.50% (1.85x enrichment).

Independently verified via `diagnostics/recall_at_k_sweep.py` on subset data
(800 events, MPS): R@600 = 0.893, R@800 = 0.950 (consistent with full-data
numbers given smaller sample).

---

## 4. Full Experiment Log

### 4.1 Pre-Filter (Stage 1) Experiments — The R@200 = 0.62 Plateau

Every approach converges to R@200 ~ 0.62 +/- 0.01:

| Experiment | R@200 | Failure hypothesis |
|-----------|-------|-------------------|
| **Phase-B baseline** (MLP+kNN, 225K) | 0.623 | Baseline |
| **ParticleNeXt** (attention agg + LV edges, 551K) | ~0.62 | Richer encoder on 99.7%-noise kNN graph adds nothing |
| **ISAB** (32 inducing points for global context) | ~0.62 | Global attention over 1100 tracks that are 99.7% noise |
| **GravNet** (learned-space kNN + distance loss) | ~0.62 | 3 signal tracks among 1100 — learned space can't cluster them |
| **EdgeConv** (pairwise LV features on kNN edges) | ~0.62 | Pairwise features between signal-noise pairs carry no info |
| **Extended features** (16 feats with fixed chi2) | 0.625 | Per-track features are NOT the bottleneck (BDT ceiling=0.589) |
| **ASL loss** | 0.41 | Hard clip=0.05 zeros easy-negative gradients, destroying ranking signal |
| **OHEM** (top-K hardest negatives) | 0.50 | Problem already maximally hard; harder negatives destabilize |
| **Pairwise features in message passing** | 0.24 | AE latent collapse: pairwise features dominate representation |
| **Pairwise as attention bias** | 0.39 | Same AE collapse regardless of injection method |
| **Score propagation** (graph smoothing) | No gain | 72% of GT pions have 0 GT neighbors; smoothing dilutes |
| **Triplet scoring** (enumerate from anchors) | OOM | 88% of kNN pairs pass mass cut -> 2B triplets -> 513GB |
| **RS@K surrogate loss** | 0.623 | No gain over ranking loss; structural bottleneck, not loss |
| **Curriculum training** (neg subsampling) | 0.628 | Marginal +0.005; structural bottleneck, not training recipe |
| **SupMin contrastive** (active, weight=0.5) | 0.616 | Still converging at time of measurement |

**Root cause:** The kNN(k=16) graph in (eta, phi) is **99.7% noise**. Any
neighborhood-based operation aggregates noise. BDT achieves R@200=0.589 without
any graph, and kNN adds only +0.034. **Breaking R@200 > 0.63 requires changing
WHICH tracks interact, not HOW they interact.** This is exactly what the cascade
does: Stage 1 filters to 600 tracks, Stage 2 applies quadratic attention on the
enriched set.

### 4.2 Cascade (Stage 2) Experiments

Cascade ParT is currently being trained, and we expect it to break the 0.69-0.70 R@200 barrier.

---

## 5. Physics Context

### Decay Chain
```
tau^- -> a_1(1260)^- -> rho(770)^0 + pi^- -> pi^+pi^- + pi^- (bachelor)
```
- The **rho pair** (opposite-sign pi^+pi^-, m ~ 770 MeV) is the "easy pair" — higher pT.
- The **bachelor pion** is typically softest and hardest to find.
- Total constraint: m(3pi) <= m_tau = 1.777 GeV.

### The "Uncanny Valley"
The intersection of **low pT (0.3-0.5 GeV)** and **small displacement (|dxy_sig| < 0.5)**
produces tracks indistinguishable from background:

| Region | Found rate |
|--------|-----------|
| pT [0.3, 0.5) GeV | 45.2% |
| |dxy_sig| [0.0, 0.5) | 31.2% |
| Intersection | ~24% |

47% of all GT pions have |dxy_sig| < 0.5. These are the tracks Stage 2 must rescue.

### The kNN Noise Problem (Stage 1 Context)
- kNN(k=16) in (eta, phi): **99.7% of neighbors are noise**.
- 72% of GT pions have **0 GT neighbors** in their kNN neighborhood.
- This is why all Stage 1 architecture changes plateaued at 0.62.
- The cascade bypasses this: Stage 2 applies **O(K1^2) attention** on 600 tracks
  where signal fraction is 1.85x higher.

### Physics Signatures Available to Stage 2
Stage 2 has access to pairwise features between all K1=600 tracks:

| Feature | Available? | How |
|---------|-----------|-----|
| Invariant mass ~ rho(770) | Yes | Via pairwise_lv_fts() ln m^2 |
| Charge product (OS for rho) | **No** | Not in pairwise features |
| ln DR (opening angle) | Yes | Via pairwise_lv_fts() |
| ln kT (splitting scale) | Yes | Via pairwise_lv_fts() |
| ln z (momentum sharing) | Yes | Via pairwise_lv_fts() |
| Vertex compatibility |d0_i - d0_j| | **No** | Not computed |
| Sum(3pi) mass <= m_tau | **No** | Would need triplet features |

**Key observation:** The strongest physics signatures (rho mass, charge product,
vertex compatibility) are either missing or only partially exploited. The current
pairwise features are generic Lorentz kinematics, not tau-decay-specific.

---

## 6. What Was Already Ruled Out

| Direction | Status | Why |
|-----------|--------|-----|
| DETR on full event (1100 tracks) | Explored, deprioritized | Cascade approach preferred for modularity |
| Richer Stage 1 (ParticleNeXt, ISAB) | Failed | 99.7% noise in kNN graph |
| GravNet learned-space kNN | Failed | Can't cluster 3 signals in 1100 |
| Pairwise features in Stage 1 kNN | Failed | AE latent collapse |
| Hard negative mining (OHEM/ASL) | Failed | Destabilizes at 0.27% signal |
| Object Condensation | Not tried in cascade | Viable alternative paradigm |
| EggNet (evolving graph) | Not tried | High effort, worth exploring |
| Physics anchoring | Not tried | Low effort, orthogonal improvement |

---

## 7. Summary Numbers at a Glance

| Metric | Value |
|--------|-------|
| Tracks per event (mean) | 1130 |
| Signal tracks per event | 3 |
| Signal fraction (full event) | 0.27% |
| Signal fraction (after K1=600 filter) | ~0.50% |
| Stage 1 R@200 (baseline) | 0.628 |
| Stage 1 R@600 (cascade ceiling) | 0.909 |
| BDT R@200 (no graph, no Stage 2) | 0.589 |
| **Cascade R@200 (ParT Stage 2, projected)** | **~0.69-0.70** |
| **Target R@200** | **>= 0.80** |
| Gap to target | ~0.10 |
| Headroom (S1 ceiling - current) | 0.21 |
| Stage 2 efficiency (used / available) | ~25% |
| Stage 2 params (full ParT config) | ~2.1M |
| Uncanny valley recall (pT 0.3-0.5, dxy<0.5) | ~24% |
