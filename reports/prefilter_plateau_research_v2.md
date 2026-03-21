# Research Report: Breaking the R@200=0.62 Plateau (v2)

**Date:** 2026-03-21

---

## 1. Experiment Summary

| Experiment | Branch | Change | R@200 | Verdict |
|-----------|--------|--------|-------|---------|
| Phase-B baseline | `redesign` | MLP+kNN, 225K | 0.623 | Baseline |
| ParticleNeXt | `partnext-v2` | Attention aggregation, LV edges, 551K | ~0.62 | No gain — richer encoder on same noisy kNN |
| ISAB | `isab` | Global context, 32 inducing points | ~0.62 | No gain — global attention into 99.7% noise |
| Extended features (broken) | `extended_data` | Log transforms + chi2 (16 features) | 0.622 | **Invalid** — chi2 was 99.9% clipped (center=-13.8) |
| OHEM | `ohem` | Top-K hardest negatives | 0.50 | **Worse** — problem already hard, harder negatives destabilize |
| Copy-paste | — | Signal track augmentation | — | DataLoader deadlocks, abandoned |
| SupMin (inactive) | `supmin` | Contrastive auxiliary loss | — | **Invalid** — supmin_weight=0, loss never activated |
| Extended features (fixed) | `chi2` | log1p(chi2) + 16 features, 226K | **0.625** (ep29) | Same as Phase-B — **features not the bottleneck** |
| SupMin (active) | `supmin` | Contrastive aux loss, weight=0.5, 275K | **0.616** (ep16) | Slight edge over Phase-B at same epoch, still converging |

### What We Know (Updated 2026-03-21)

**Definitively ruled out:**
- Receptive field expansion (ParticleNeXt, ISAB) → noise dominates
- Input features (16 features with chi2, log transforms) → R@200=0.625 ≈ baseline
- Hard negative mining (OHEM) → destabilizes training
- Copy-paste augmentation → implementation issues

**Pending:**
- Extended features with correct chi2 — first valid test of whether
  normChi2 + log transforms improve per-track discrimination
- SupMin contrastive — first valid test of whether clustering signal
  embeddings improves the ranking loss

**Not yet tried:** Pairwise interaction features, GravNet, cascade,
BDT baseline, Object Condensation.

---

## 2. The Core Architectural Gap: No Pairwise Features

### 2.1 What the Model Currently Does

```python
# TrackPreFilter._forward_mlp():
neighbor_features = cross_set_gather(current, neighbor_indices)  # (B, H, P, K)
max_pooled = neighbor_features.max(dim=-1)[0]                    # (B, H, P)
aggregated = cat([current, max_pooled], dim=1)                   # (B, 2H, P)
```

The model gathers neighbor node features and max-pools them. It **never
computes any pairwise features** between center track i and neighbor j:
- No relative differences `(h_j − h_i)` (EdgeConv)
- No physics pair features (invariant mass, ΔR, kT, z)
- No charge product `q_i · q_j`

### 2.2 Why This Matters: Literature Evidence

**Particle Transformer ablation (Qu et al., ICML 2022):**
Removing the pairwise interaction matrix drops accuracy from 0.861 to 0.405.
Despite magnitudes being 10⁴–10⁵× smaller than QK attention scores,
pairwise features influence the highest-attention token in 85.4% of positions.

**IAFormer (2025):** Achieves ParT-level accuracy with 10× fewer parameters
(211K vs 2.14M) using ONLY pairwise features as attention scores — no
QK attention at all. Pairwise features alone carry enough information.

**ParticleNet (Qu & Gouskos, 2020):** 1.8–2.1× background rejection
improvement from EdgeConv's `(x_j − x_i)` term vs plain aggregation.

### 2.3 Physics-Motivated Pairwise Features for τ → 3π

The dominant decay chain: τ → a₁(1260) → ρ(770) + π → π⁺π⁻π.

| Pairwise Feature | Formula | Physics Motivation |
|-----------------|---------|-------------------|
| **Invariant mass** | m_ij = √((E_i+E_j)² − ‖p_i+p_j‖²) | Signal π⁺π⁻ pairs peak at ρ(770) ≈ 770 MeV |
| **Charge product** | q_i · q_j | τ has 2 same-sign + 1 opposite-sign; ρ requires opposite-sign |
| **ln ΔR** | ln√((Δη)² + (Δφ)²) | Characteristic opening angle from τ boost |
| **ln kT** | ln(min(pT_i, pT_j) · ΔR) | Transverse momentum splitting scale |
| **ln z** | ln(min(pT) / (pT_i + pT_j)) | Momentum sharing fraction |
| **ln m²** | ln((E_i+E_j)² − ‖p_i+p_j‖²) | = invariant mass squared (log-transformed) |
| **Vertex compat.** | |d0_i − d0_j| | Tracks from same displaced vertex have correlated d0 |

The `pairwise_lv_fts()` function computing (ln kT, ln z, ln ΔR, ln m²) **already
exists** in the codebase (`ParticleTransformer.py:83`) and is used by
`HierarchicalGraphBackbone`. The pre-filter just never calls it.

### 2.4 Recommended Implementation: EdgeConv + LV Pair Features

Replace the current gather-and-max-pool with EdgeConv-style aggregation
plus physics pairwise features:

```python
# Proposed flow in _forward_mlp:
center_expanded = current.unsqueeze(-1).expand_as(neighbor_features)
relative_features = neighbor_features - center_expanded       # (B, H, P, K)
lv_pair_features = pairwise_lv_fts(center_lv, neighbor_lv)   # (B, 4, P, K)
charge_product = charge_i * charge_j                          # (B, 1, P, K)
edge_features = cat([center_expanded, relative_features,
                     lv_pair_features, charge_product], dim=1) # (B, 2H+5, P, K)
messages = edge_mlp(edge_features)                            # (B, H, P, K)
aggregated = messages.max(dim=-1)[0]                          # (B, H, P)
```

**Complexity:** Same O(P·k) as current. One additional Conv2d MLP.
**Data requirements:** All inputs already available (lorentz_vectors, charge).
**Estimated code:** ~30 lines in `_forward_mlp`, new edge MLP in `__init__`.

---

## 3. GravNet: Learned Coordinate kNN

**Paper:** Qasim et al., Eur. Phys. J. C 79 (2019).

Build kNN in a **learned S-dimensional space** (S=3-4) where signal tracks
can cluster. Distance-weighted aggregation: w_ij = exp(−‖s_i − s_j‖²).

### How It Differs From What Failed

| | Current | ParticleNeXt/ISAB | GravNet |
|--|---------|-------------------|---------|
| Graph | Fixed (η, φ) | Fixed (η, φ) | **Learned space** |
| Receptive field | k=16 local | Global | **k=16, adaptive locality** |
| What changes | Nothing | Attention weights | **Both graph AND weights** |

ISAB/ParticleNeXt expanded receptive field **size** (all 1100 tracks).
GravNet keeps the field **small** but makes the **locality adaptive**.

### Expected Impact

HIGH if signal tracks can be mapped close in learned space. Risk: with
only 3 signal tracks per event at median ΔR=1.3, the learned space may not
find a clustering projection. The Gaussian weighting partially mitigates this.

### Compatibility

O(P·k), adds one projection layer. `cross_set_knn` can be reused on
projected coordinates. ~50 lines standalone implementation.

---

## 4. Two-Phase Cascade

Phase 1: cheap per-track MLP (no kNN) scores all 1100 → top-K₁.
Phase 2: kNN recomputed among K₁ survivors, richer scorer → top-200.

### Signal Concentration Improvement

| K₁ | Signal fraction | Improvement over baseline |
|----|----------------|--------------------------|
| 1100 (no cascade) | 0.27% | 1× |
| 500 | 0.60% | 2.2× |
| 400 | 0.75% | 2.8× |
| 300 | 1.00% | 3.7× |
| 200 | 1.50% | 5.6× |

### Critical Diagnostic (MUST DO FIRST)

**Measure the current model's R@K for K = 200, 300, 400, 500, 600.**
K₁ should be the smallest K where R@K₁ ≥ 0.95 (Phase 1 loses <5% GT).
If R@400 is only 0.75, cascade cannot work — Phase 1 is the bottleneck.

This single measurement determines whether the cascade approach is viable.

### Implementation

The existing `select_top_k()` + `filter_tracks()` already implements the
Phase 1 → Phase 2 handoff. Add a second scorer after `filter_tracks`.

---

## 5. BDT Ceiling Estimation

**What:** XGBoost on 16 per-track features, no kNN context. R@200 via
`scale_pos_weight` and score-based ranking.

**Why:** Answers the fundamental question — is the bottleneck in features
or architecture?
- BDT ≈ NN (~0.62): features are the ceiling, kNN adds nothing useful
- NN >> BDT: kNN context helps, improve how it aggregates (→ pairwise features)
- BDT > NN: the NN is undertrained or the loss is suboptimal

**Effort:** 30-minute experiment. No GPU needed.

---

## 6. Object Condensation (Alternative Paradigm)

**Papers:** Kieseler, Eur. Phys. J. C 80 (2020); Lieret et al., EPJ Web Conf. (2024).

Replace ranking loss with OC: per-track β (confidence) + attractive/repulsive
loss in embedding space. Asks "do these tracks belong together?" rather than
"is this track signal?" — a relational question.

**Risk:** 0.3% signal fraction is far more extreme than OC has been tested
on (HL-LHC tracking has much higher signal fractions). The attractive loss
operates on only 3 tracks per event — very weak clustering signal.

**Priority:** Try after D1-D4 if plateau persists.

---

## 7. Ranked Recommendations

| Priority | Approach | Expected Impact | Evidence | Effort |
|----------|----------|----------------|----------|--------|
| **1** | **EdgeConv + pairwise LV features** | HIGH | ParT ablation (0.86→0.40), IAFormer, ParticleNet 2× rejection | ~30 lines |
| **2** | **BDT ceiling test** | Diagnostic | Determines feature vs architecture bottleneck | 30 min |
| **3** | **Cascade R@K diagnostic** | Diagnostic | Determines if cascade is viable | 10 min |
| **4** | **GravNet learned kNN** | HIGH | Fundamentally different graph construction | ~50 lines |
| **5** | **Two-phase cascade** | HIGH (if R@K₁ is high) | Exa.TrkX, GNN4ITk pattern | Moderate |
| **6** | **Object Condensation** | Uncertain | Proven in HEP but not at 0.3% signal | Major refactor |

## 8. Suggested Experiment Sequence

```
1. Wait for pending results (extended features + SupMin)
2. BDT ceiling test on 16 features (30 min, answers features vs architecture)
3. Cascade diagnostic: measure R@300, R@400, R@500 (10 min)
4. EdgeConv + pairwise LV + charge product in pre-filter (~30 lines)
5. GravNet learned kNN (if pairwise alone doesn't break through)
6. Two-phase cascade (if R@K₁ diagnostic is favorable)
```

Steps 2-3 are zero-risk diagnostics that determine strategy.
Step 4 is the highest-expected-impact architecture change.

---

## Sources

### Pairwise Interactions
- Qu et al., "Particle Transformer for Jet Tagging," ICML 2022 — [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)
- "Why is Attention Sparse in Particle Transformer?" — [arXiv:2512.00210](https://arxiv.org/abs/2512.00210)
- "IAFormer: Interaction-Aware Transformer" — [arXiv:2505.03258](https://arxiv.org/abs/2505.03258)
- Qu & Gouskos, "ParticleNet: Jet Tagging via Particle Clouds," Phys. Rev. D 2020 — [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)
- Battaglia et al., "Interaction Networks," NeurIPS 2016 — [arXiv:1612.00222](https://arxiv.org/abs/1612.00222)

### GravNet
- Qasim et al., Eur. Phys. J. C 79, 608 (2019) — [arXiv:1902.07987](https://arxiv.org/abs/1902.07987)

### Track Reconstruction Pipelines
- Ju et al., "Exa.TrkX Pipeline," Eur. Phys. J. C 2021 — [arXiv:2103.06995](https://arxiv.org/abs/2103.06995)
- GNN4ITk ATLAS — [EPJ Web Conf. 2024](https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_03030.pdf)

### Object Condensation
- Kieseler, Eur. Phys. J. C 80, 886 (2020) — [arXiv:2002.03605](https://arxiv.org/abs/2002.03605)
- Lieret et al., EPJ Web Conf. 295, 09004 (2024) — [arXiv:2309.16754](https://arxiv.org/abs/2309.16754)

### Tau Physics
- "Three-pion decays of the tau lepton" — [arXiv:1006.2919](https://arxiv.org/abs/1006.2919)
