# Architecture Research Report: Pre-Filter Phase C

**Date:** 2026-03-20
**Goal:** Bridge R@200 from 0.62 to 0.8–0.9 through architecture improvements.
**Constraint:** Focus on architecture, not training/loss adjustments.

---

## 1. Phase-B Results Analysis

### 1.1 Training Summary

Phase-B: MLP mode, 225K params, hidden=192, 2 rounds kNN(k=16) max-pool.
Trained for 39/100 epochs on RTX 6000 Pro (batch=96, 750 steps/epoch).
DRW activates at epoch 31 (30% warmup of 100 epochs).

### 1.2 Metric Trajectory

| Epoch | val R@200 | val P@200 | val d' | val rank | Notes |
|-------|-----------|-----------|--------|----------|-------|
| 1     | 0.566     | 0.286     | 1.202  | 153      | |
| 10    | 0.600     | 0.335     | 1.291  | 127      | |
| 20    | 0.617     | 0.358     | 1.303  | 116      | |
| 30    | 0.616     | 0.354     | 1.311  | 114      | Last pre-DRW |
| 31    | 0.623     | 0.363     | 1.304  | 112      | DRW on |
| 36    | 0.623     | 0.368     | 1.315  | 113      | Best R@200 |
| 39    | 0.623     | 0.361     | 1.302  | 111      | Latest |

### 1.3 Overfitting Analysis

| Epoch | train R@200 | val R@200 | Gap | train rank | val rank | Gap |
|-------|-------------|-----------|-----|------------|----------|-----|
| 20    | 0.622       | 0.617     | 0.005 | 115       | 116      | +1  |
| 30    | 0.623       | 0.616     | 0.007 | 110       | 114      | +4  |
| 39    | 0.631       | 0.623     | 0.008 | 106       | 111      | +5  |

The gap is growing slowly. Train metrics keep improving while val plateaus.
The model is **at capacity** — not overfitting badly, but unable to generalize
further. This indicates the architecture is the binding constraint, not data
or training procedure.

### 1.4 Controlled Subset Evaluation

Using `validate()` from `train_prefilter.py` on extended subset (1600 events):

| Metric | Phase-B | Widened (baseline) | Delta |
|--------|---------|--------------------|-------|
| R@200  | 0.598   | 0.556              | **+0.042** |
| P@200  | 0.353   | 0.301              | **+0.052** |
| d'     | 1.290   | 1.200              | **+0.090** |
| Rank   | 141     | 175                | **-34** |

Phase-B (MLP, no AE, temp scheduling, DRW) significantly outperforms widened
baseline on a controlled comparison.

### 1.5 Bottleneck Diagnosis

**Capacity-limited, not data-limited.** Evidence:
- Train-val gap is only 0.008 — the model generalizes well but can't learn more
- 225K params is small for ~1100 tracks × 13 features
- Receptive field: 2 hops × k=16 = at most ~256 tracks (23% of event) — most
  tracks are invisible to any given track's scoring decision
- Max-pool loses information: a neighborhood of 16 tracks is summarized by a
  single max value per channel

**The core limitation is local context.** Each track scores based on its
16 nearest neighbors (2 hops). Signal tracks in the "uncanny valley" (kinematically
similar to background but spatially separated from other signal tracks) cannot
access global event context that would help identify them.

---

## 2. Literature Survey

Full details in two companion reports:
- `prefilter_architecture_literature_survey.md` — Directions 1–3
- `set_selection_and_hep_tracking_research.md` — Directions 4–5

### 2.1 Richer Message Passing / GNNs

| Approach | Mechanism | Complexity | HEP use | Relevance |
|----------|-----------|-----------|---------|-----------|
| **PNA** | cat(mean,max,min,std) aggregation | Same as max-pool | No | Already implemented |
| **Dynamic kNN (DGCNN)** | Recompute kNN in feature space each round | O(P²d) per round | ParticleNet | High |
| **GATv2** | Attention-weighted neighbor aggregation | O(Pkd) | Prototype | High |
| **Multi-scale kNN** | Parallel k=8,32,64 branches | 3× params | No | Medium |

**Key finding:** PNA is already in the codebase (`aggregation_mode='pna'`). Dynamic
kNN (recompute graph in learned feature space) is ParticleNet's key innovation and
a cheap upgrade.

### 2.2 Efficient Attention for Large Sets

| Approach | Mechanism | Complexity | HEP use | Relevance |
|----------|-----------|-----------|---------|-----------|
| **ISAB (Set Transformer)** | m inducing points compress, then broadcast | O(Pmd) | Yes (collider) | **Very high** |
| **Perceiver** | Latent bottleneck cross-attention | O(NMd) | No | High |
| **HEPT** | LSH-bucketed attention for HEP | O(P^1.5 d) | ICML 2024 Oral | High |
| **Full self-attention** | Standard transformer | O(P²d) | Van Stroud 2025 | Feasible at P=1100 |

**Key finding:** ISAB with m=32 inducing points gives every track global event
context at O(P·m·d) — a 34× reduction vs full attention. HEPT is HEP-specific
and proven at scale. At P=1100, even full self-attention (1.2M ops) is feasible.

### 2.3 Hierarchical / Cascade Architectures

| Approach | Mechanism | Complexity | HEP use | Relevance |
|----------|-----------|-----------|---------|-----------|
| **U-Net decoder** | Encode→downsample→decode with skip connections | O(P log P) | No | Medium |
| **Cascaded classifier** | MLP pre-cut 1100→400, then kNN on 400 | Varies | ATLAS GNN4ITk | Medium |
| **Point Transformer V3** | Serialization + patch attention | O(P) | No | Low (paradigm shift) |

**Key finding:** U-Net is natural if using the existing HierarchicalGraphBackbone
encoder. Cascaded filtering matches HEP's standard coarse-to-fine pattern.

### 2.4 Set Prediction / Subset Selection

| Approach | Mechanism | Complexity | HEP use | Relevance |
|----------|-----------|-----------|---------|-----------|
| **DETR-like queries** | K learned queries cross-attend to tracks | O(KNd) | Van Stroud 2025 | **Very high** |
| **Slot Attention** | Competitive slots bind to objects | O(KNd) | No | Medium |
| **Differentiable top-k** | Makes hard selection differentiable | O(N log N) | No | High (training) |
| **Neural submodular** | Learned diminishing returns | O(N²d) | No | Low |

**Key finding:** Van Stroud et al. (2025) already demonstrated a Transformer
pre-filter for 100K-hit tracking at 97% efficiency / 0.6% fake rate — the
closest published work to our problem. DETR queries are the most promising
new architecture.

### 2.5 Domain-Specific HEP

| Approach | Mechanism | Scale | Relevance |
|----------|-----------|-------|-----------|
| **Exa.TrkX pipeline** | Metric learning → GNN edge classification | 100K hits | Very high (pattern) |
| **GravNet + OC** | Learned coordinates + condensation loss | CMS HGCAL | High |
| **Interaction Networks** | Edge-level message passing | HL-LHC | Medium |
| **CMS track quality NN** | Per-track quality classifier | CMS LST | Direct analogue |

**Key lessons from HEP:**
1. Learned graph construction is the critical bottleneck for recall
2. Two-stage coarse-to-fine pipelines dominate (our pattern is correct)
3. GravNet + OC is becoming the standard for clustering
4. Low-pT tracking with ML remains genuinely underexplored
5. O(N²) is forbidden at large scale — use kNN, LSH, or inducing points

---

## 3. Recommended Architectures for Phase C

### Recommendation 1: ParticleNeXt backbone (highest priority)

**What:** Replace the custom MLP+kNN pre-filter with ParticleNeXt
(`for_segmentation=True, num_classes=1`), already implemented in the
weaver repository at `weaver/weaver/nn/model/ParticleNeXt.py`.
Wrap it with the existing ranking/denoising loss from TrackPreFilter.

**Why it should work:** ParticleNeXt is the state-of-the-art GNN for
particle physics, already available in the codebase. It addresses every
identified bottleneck of the current MLP+kNN architecture:

| Bottleneck | Current MLP+kNN | ParticleNeXt |
|-----------|-----------------|--------------|
| Aggregation | Max-pool (lossy) | Attention-weighted (learned) |
| Edge features | None (node-only) | Pairwise ΔR, invariant mass, kT, z from Lorentz vectors |
| Receptive field | 2 hops × k=16 = ~256 tracks | Multi-scale dilation [(8,1),(4,1),(2,1),(1,1)] per block |
| Channel attention | None | Squeeze-and-Excitation |
| Residual connections | None | Learnable γ-scaled shortcuts |
| Data augmentation | None | pT dropout, LV smearing/scaling/shifting |

**Architecture:**
```
Input (B, 13, P) + LV (B, 4, P)
  → ParticleNeXt(for_segmentation=True, num_classes=1):
      BN → node encode (13→32)
      BN → edge encode (pairwise LV features→8)
      kNN once (k=32, static graph)
      3× MultiScaleEdgeConv:
        multi-dilation neighbors, attention aggregation,
        SE channel attention, residual + γ scaling
      BN → ReLU → Conv1d(→1) per-track score
  → ranking loss + denoising loss (from TrackPreFilter)
```

**Complexity:** ~3× current model per forward pass, but the operations
are highly optimized Conv2d + attention (GPU-friendly).
**Parameters:** ~300K–1M depending on config (vs current 225K).
**Implementation:** New network wrapper + thin loss-computation wrapper
around ParticleNeXt. No changes to ParticleNeXt itself.
**Expected impact:** +0.10–0.20 R@200 — this replaces the entire
feature extraction with a proven, physics-informed architecture.
**Reference:** ParticleNeXt (ML4Jets 2021), ParticleNet (Qu & Gouskos 2020).

### Recommendation 2: Global Context via ISAB (if ParticleNeXt plateaus)

**What:** Add an Induced Set Attention Block after ParticleNeXt's
per-track features, before the scorer. m=32–64 inducing points give
every track access to a compressed summary of the entire event.

**Complexity:** O(P·m·d) ≈ 6.8M ops. Negligible addition.
**Parameters:** ~150K additional.
**Implementation:** ~60–80 lines.
**Expected impact:** +0.05–0.10 R@200 on top of ParticleNeXt.
**Reference:** Lee et al. ICML 2019, validated in collider physics.

### Recommendation 3: Dynamic kNN (recompute graph in feature space)

**What:** After each message-passing round, recompute kNN indices using
learned features instead of fixed (eta, phi) coordinates.

**Why:** Static kNN in (eta, phi) means neighbors are fixed by detector
geometry. Signal tracks may be spatially dispersed but become similar in
learned feature space. Dynamic kNN (from ParticleNet/DGCNN) lets the graph
adapt: round 1 uses spatial neighbors, round 2 uses semantic neighbors.

**Architecture change:** In `_forward_mlp`, replace static kNN reuse with:
```python
for round_index in range(self.num_message_rounds):
    if round_index == 0:
        neighbor_indices = cross_set_knn(points, ...)  # spatial
    else:
        neighbor_indices = cross_set_knn(current, ...)  # feature-space
```

**Complexity:** Extra O(P²d) per round for exact kNN in feature space.
At P=1100, d=192: ~232M ops. Acceptable.
**Parameters:** Zero additional parameters.
**Implementation:** ~10 lines changed.
**Expected impact:** +0.01–0.03 R@200.
**Reference:** ParticleNet (Qu & Gouskos 2020).

### Recommendation 3: DETR-like query decoder (ambitious, high potential)

**What:** Add a DETR-style decoder after the encoder (kNN rounds + optional
ISAB). K=200 learned queries cross-attend to encoded track features. Each
query outputs a selection score. Training uses Hungarian matching.

**Why:** Queries can learn specialized detection patterns (e.g., "find the
displaced high-pT track," "find the low-pT track near the tau axis"). This
is fundamentally more expressive than per-track scoring because queries can
coordinate — they attend to the same feature map but focus on different
aspects.

**Architecture:**
```
features (B, 13, P)
  → track_mlp + kNN rounds  (encode)
  → ISAB(m=32)              (global context)
  → DETR decoder: K=200 queries cross-attend to encoded features
  → per-query score → select top-200 tracks by attention weights
```

**Complexity:** O(K·P·d) = O(200·1100·192) ≈ 42M ops per decoder layer.
With 2 decoder layers: ~84M ops.
**Parameters:** ~300K additional.
**Implementation:** ~150–200 lines. Requires Hungarian matching for training.
**Expected impact:** +0.10–0.20 R@200 (based on Van Stroud et al. 2025).
**Reference:** Carion et al. ECCV 2020, Van Stroud et al. Phys. Rev. X 2025.

### Recommendation 4: Increase model capacity (simple scaling)

**What:** hidden_dim 192→384, num_message_rounds 2→3, add residual connections.

**Why:** The model is at capacity (train still improves, val plateaus). Doubling
width quadruples parameters in the scoring layers. Adding a 3rd round extends
the receptive field to k³=4096 (covers the full event).

**Parameters:** ~900K (4× current).
**Implementation:** Config change only.
**Expected impact:** +0.02–0.05 R@200.

---

## 4. Proposed Implementation Order

```
Phase C-1: ParticleNeXt as pre-filter backbone      (network wrapper + loss wrapper)
Phase C-2: Train + evaluate                          (server)
Phase C-3: If plateaus, add ISAB global context      (~80 lines)
Phase C-4: If still needed, add DETR query decoder   (~200 lines)
```

Phase C-1 is the highest-leverage change: it replaces the entire feature
extraction with a proven, physics-informed architecture that already exists
in the codebase. The remaining recommendations are fallbacks if ParticleNeXt
alone doesn't reach the 0.8+ target.

---

## 5. References (Key Papers)

1. Lee et al., "Set Transformer," ICML 2019 — ISAB inducing points
2. Qu & Gouskos, "ParticleNet," Phys. Rev. D 2020 — Dynamic kNN for HEP
3. Carion et al., "DETR," ECCV 2020 — Query-based detection
4. Jaegle et al., "Perceiver," ICML 2021 — Latent bottleneck attention
5. Miao et al., "HEPT," ICML 2024 Oral — LSH attention for HEP
6. Van Stroud et al., "Transformers for Tracking," Phys. Rev. X 2025 — Transformer pre-filter
7. Kieseler, "Object Condensation," Eur. Phys. J. C 2020 — Alternative loss paradigm
8. Sander et al., "Differentiable Top-k," ICML 2023 — End-to-end selection
9. Corso et al., "PNA," NeurIPS 2020 — Multi-aggregation
10. Brody et al., "GATv2," ICLR 2022 — Dynamic graph attention
