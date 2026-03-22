# Deep Research Input: Pre-Filter for Low-pT Tau Track Finding

**Date:** 2026-03-22
**Purpose:** Comprehensive summary of all pre-filter experiments, findings, and
open questions. To be used as input for Deep Research to explore new directions
for breaking the R@200 = 0.62 plateau.

---

## 1. Problem Statement

### Task
Identify 3 signal tracks (pions from τ → 3π decay) among ~1100 reconstructed
tracks per event in a low-pT tau tagging context at CMS (LHC). The signal
fraction is **0.27%** (3 out of ~1100).

### Two-Stage Pipeline
- **Stage 1 (pre-filter):** Score all ~1100 tracks, select top-200 candidates.
  Must achieve high recall (R@200) while staying at O(P·k) complexity.
- **Stage 2 (refinement):** Run expensive algorithms (cross-attention, DETR
  queries, etc.) on the reduced 200-track set. O(200²) is cheap.

### Metrics
- **R@K (Recall at K):** Fraction of ground-truth signal tracks that appear in
  the top-K ranked candidates, averaged over events. R@200 = 0.62 means 62% of
  GT pions land in the top-200 on average. This is the primary metric — the
  pre-filter's job is to not miss signal.
- **P@K (Perfect at K):** Fraction of events where **all 3** GT pions are found
  in the top-K. P@200 = 0.368 means only 36.8% of events have a complete set
  of signal tracks in the top-200. This is the strictest metric — even one
  missed pion means the event cannot be fully reconstructed downstream.

### Target
- **R@200 ≥ 0.80** (currently 0.62). This means 80% of ground-truth pions
  must appear in the top-200 ranked tracks.
- The pre-filter is a retrieval/ranking problem, not a classification problem.

### Why This Is Hard
- **Extreme class imbalance:** 3 signal tracks among ~1100 (0.27% positive rate).
- **Low-pT regime:** Jets degenerate into diffuse particle clouds. Standard
  jet-based methods fail because there is no collimated jet structure.
- **Signal-background overlap:** The hardest signal pions (low pT, small
  displacement) have feature distributions that overlap heavily with noise.

### Dataset
- **270K training events, 84K validation events.**
- Track counts per event: min=0, max=~3500, mean=1130, median=1211.
- YAML config pads to length=3500 to accommodate the largest events.
- Each event has exactly 3 GT pions (from τ → 3π).

---

## 2. Current Architecture (Best Model = Extended Data Model)

### Model: TrackPreFilter
- **Backbone:** MLP encoder → 2 rounds of kNN(k=16) message passing with
  max-pool aggregation → scoring MLP
- **Graph:** Fixed kNN in (η, φ) coordinates
- **Input features:** 16 features including log-transforms and normChi2:
  - Kinematic: pT, η, φ, charge, pT_error
  - Displacement: dxy_significance, dz_significance, dca_significance
  - Quality: n_valid_pixel_hits, covariance_phi_phi, covariance_lambda_lambda
  - Derived: log(pT), log(pT_error), log1p(normChi2), log1p(covariance_phi_phi),
    log1p(covariance_lambda_lambda)
- **Parameters:** 226K
- **Complexity:** O(P·k) per message-passing round

### Training Recipe
- **Loss:** Temperature-scheduled ranking loss (primary) + contrastive denoising
  (auxiliary) + reconstruction loss (auxiliary, deferred reweighting)
- **Deferred Reweighting (DRW):** Reconstruction loss weight reduced after
  initial epochs to free the latent space for discrimination
- **Temperature scheduling:** Contrastive loss temperature cycles between high
  (group discrimination) and low (instance discrimination)
- **Optimizer:** AdamW with cosine LR schedule

### Best Result (epoch 35)
| Metric | Value |
|--------|-------|
| **val R@200** | **0.629** |
| val P@200 | 0.368 |
| val median rank | 108 |
| d-prime | ~1.29 |

---

## 3. Full Experiment Log

### 3.1 Architecture Experiments

| Experiment | Branch | Description | Params | R@200 | Verdict |
|-----------|--------|-------------|--------|-------|---------|
| Phase-B baseline | `redesign` | MLP + kNN(k=16), max-pool | 225K | 0.623 | Baseline |
| ParticleNeXt | `partnext-v2` | Attention aggregation + LV edge features | 551K | ~0.62 | No gain — richer encoder on same noisy kNN graph |
| ISAB | `isab` | 32 inducing points for global context | 529K | ~0.62 | No gain — global attention into 99.7% noise |
| Pairwise LV features in MP | — | Invariant mass, ΔR, kT, z as edge attrs | — | 0.24 (collapsed) | **Failed** — AE latent collapse from relational features |
| Pairwise as attention bias | — | Pairwise → scalar attn logit, not in value stream | — | 0.39 (collapsed) | **Failed** — same AE collapse regardless of injection method |
| Score propagation | — | Post-hoc graph smoothing (α=0.3, 3 iterations) | — | No improvement | **Failed** — 72% of GT pions have 0 GT neighbors, smoothing dilutes |
| Triplet scorer | — | Enumerate candidate triplets from top-200 anchors | — | OOM (96GB) | **Failed** — 88% of kNN pairs pass mass cut → combinatorial explosion |

### 3.2 Feature Experiments

| Experiment | Branch | Description | R@200 | Verdict |
|-----------|--------|-------------|-------|---------|
| Extended 16 features (broken chi2) | `extended_data` | Log transforms + chi2 | 0.622 | **Invalid** — chi2 was 99.9% clipped (center=-13.8) |
| Extended 16 features (fixed) | `chi2` | log1p(chi2) + all 16 features | **0.625** (ep29) | Same as baseline — **features alone are not the bottleneck** |

### 3.3 Loss / Training Experiments

| Experiment | Branch | Description | R@200 | Verdict |
|-----------|--------|-------------|-------|---------|
| ASL loss | — | Asymmetric Loss replacing ranking loss | 0.41 (degraded) | **Failed** — clip=0.05 zeros easy negative gradients, destroys ranking signal |
| OHEM | `ohem` | Top-K hardest negatives only | 0.50 | **Worse** — problem already hard, harder negatives destabilize |
| SupMin (inactive) | `supmin` | Contrastive aux loss, weight=0 | — | **Invalid** — loss never activated (weight was 0) |
| SupMin (active) | `supmin` | Contrastive aux loss, weight=0.5 | 0.616 (ep16) | Slight edge over baseline at same epoch, but still converging |
| Copy-paste augmentation | — | Signal track augmentation | — | **Abandoned** — DataLoader deadlocks |

### 3.4 Additional Experiments (Post-Plateau)

| Experiment | Description | R@200 | Verdict |
|-----------|-------------|-------|---------|
| GravNet + distance loss | Learned-space kNN + metric learning to make signal cluster | ~0.62 | **Did not break the wall** — learned space didn't produce signal-enriched neighborhoods |
| EdgeConv with pairwise LV features | Pairwise physics features on kNN edges | ~0.62 | **No gain** — pairwise features on 99.7% noise neighbors are ineffective |

### Summary
**Every approach converges to R@200 ≈ 0.62 ± 0.01.** Architecture changes
(ParticleNeXt, ISAB, GravNet, EdgeConv), feature changes (16 features + chi2),
and loss changes (OHEM, SupMin) all plateau at the same wall. The ceiling
appears to be a property of operating on the fixed (η, φ) kNN graph where
neighborhoods are 99.7% noise.

---

## 4. Diagnostic Findings

### 4.1 BDT Ceiling Test
XGBoost on 16 per-track features (no kNN context, no graph):
- **R@200 = 0.589** → kNN adds only +0.034 on top of pure per-track features
- Implication: the kNN message passing provides marginal benefit

**BDT R@K at various K (cascade viability):**

| K | R@K |
|---|-----|
| 200 | 0.589 |
| 500 | 0.85 |
| 600 | 0.90 |

This means: at K=500, 85% of GT pions are in the top-500 even with a simple
BDT. A cascade that narrows to 500 first, then applies richer processing,
could be viable.

### 4.2 Per-Event Breakdown (from diagnostics on 800 events)

| R@200 | Events | Fraction |
|-------|-------:|-------:|
| 0% (0/3 found) | 133–152 | ~19% |
| 33% (1/3 found) | ~200 | ~25% |
| 67% (2/3 found) | ~208 | ~26% |
| 100% (3/3 found) | ~240 | ~30% |

**In 3-GT events:** perfect (3/3) in 30%, find 2/3 in 26%, find 1/3 in 25%,
find 0/3 in 19%.

### 4.3 Feature-Dependent Failure Analysis

**Found rate vs pT:**

| pT Range (GeV) | Found Rate |
|----------------|-----------|
| [0.0, 0.3) | 64.4% |
| **[0.3, 0.5)** | **45.2%** ← uncanny valley |
| [0.5, 1.0) | 52.8% |
| [1.0, 2.0) | 76.6% |
| [2.0, 5.0) | 85.0% |
| [5.0, 50.0) | 85.7% |

**Found rate vs |dxy_significance|:**

| |dxy_sig| Range | Found Rate |
|----------------|-----------|
| **[0.0, 0.5)** | **31.2%** ← worst |
| [0.5, 1.0) | 41.1% |
| [1.0, 2.0) | 72.5% |
| [2.0, 3.0) | 92.1% |
| [3.0, 5.0) | 95.3% |
| [5.0, 20.0) | 80.4% |

### 4.4 The "Uncanny Valley"
The intersection of **low pT (0.3–0.5 GeV)** and **small displacement
(|dxy_sig| < 0.5)** produces tracks that are virtually indistinguishable from
background. These tracks have only ~24% recall (3 out of 4 are missed).

47% of all GT pions have |dxy_sig| < 0.5 and only 31% found rate.

**Failed GT pion characteristics** (from 0/3 events, n=356 missed pions):
- Median pT = 0.476 GeV (squarely in uncanny valley)
- Median |dxy| = 0.530 (non-displaced)
- 53.4% have pT < 0.5 GeV
- **79.2% have |dxy| < 1.0**

The failure mode is clear: **soft, non-displaced pions are indistinguishable
from background** using per-track features alone.

### 4.5 The kNN Neighborhood Problem
- kNN(k=16) in (η, φ) coordinates: **99.7% of neighbors are noise**
  (because signal is 0.27% of all tracks).
- **100% of GT pions have ≥1 GT neighbor** somewhere in the event's kNN graph.
- But "found" GT pions have **2× the rate of 2+ GT neighbors** compared to
  missed pions (30% vs 17%).
- Any operation on this graph (max-pool, attention, PNA, EdgeConv) is
  aggregating noise features. This is confirmed by ParticleNeXt and ISAB
  producing identical R@200 to the baseline.

### 4.6 Cascade Viability
**2 GT pions in top-500 for 90% of events requires K=562.**

This means a cascade approach is viable: a first stage selects top-500–600
(where R@K ≈ 0.85–0.90), then a second stage with richer processing (pairwise
features, recomputed kNN among survivors) refines to top-200.

At K₁=500, the signal fraction improves from 0.27% to 0.60% (2.2× enrichment).
At K₁=300, signal fraction reaches 1.00% (3.7× enrichment).

---

## 5. What We've Ruled Out and What Remains Open

### Definitively Ruled Out

| Approach | Why It Failed |
|----------|--------------|
| **Receptive field expansion** (ParticleNeXt, ISAB) | Noise dominates — expanding into 1100 tracks that are 99.7% noise doesn't help |
| **Hard negative mining** (OHEM, ASL) | Problem is already at maximum difficulty. OHEM destabilizes; ASL zeros essential gradients |
| **Pairwise features within fixed (η,φ) kNN** | Neighbors are 99.7% noise, physics pair features (invariant mass, ΔR) are wasted on noise pairs. AE latent also collapses. |
| **Score propagation / graph smoothing** | 72% of GT pions have 0 GT neighbors — smoothing dilutes their scores |
| **Triplet enumeration among kNN neighbors** | 88% of pairs pass mass cut → combinatorial explosion (OOM at 96GB) |
| **GravNet learned-space kNN** | Learned coordinates didn't produce signal-enriched neighborhoods; same R@200 |
| **EdgeConv on (η,φ) kNN** | Same issue — richer edge features on noise edges don't help |
| **Per-track feature engineering** (chi2, log transforms) | 16 features with log1p(chi2) achieves R@200=0.625, same as 7-feature baseline with kNN. **Features alone are not the bottleneck.** |

### NOT Ruled Out (Open Directions)

| Direction | Rationale |
|-----------|-----------|
| **Cascade approaches** | BDT R@500=0.85 — narrow to 500 first, then richer processing on enriched set |
| **Object Condensation** | Learns clustering coordinates in a SEPARATE space, doesn't suffer from the kNN noise problem |
| **Slot Attention / DETR queries** | K=3 learned queries cross-attend to ALL tracks simultaneously, bypassing kNN entirely |
| **Differentiable top-K** | Current top-K breaks gradient chain; smooth selection could improve training signal |
| **Alternative graph construction** | Methods that build graphs based on learned similarity rather than (η,φ) proximity |
| **Paradigm shift from per-track scoring** | Edge classification (Exa.TrkX), set prediction, or embedding-space clustering |

---

## 6. Physics of τ → 3π and Its Implications

### Decay Chain
```
τ⁻ → a₁(1260)⁻ → ρ(770)⁰ + π⁻ → π⁺π⁻ + π⁻ (bachelor)
```

- The **ρ pair** (opposite-sign π⁺π⁻, invariant mass ≈ 770 MeV) is the
  "easy pair" — tends to carry higher pT.
- The **bachelor pion** is typically the softest and hardest to find.
- Total τ mass constraint: m(3π) ≤ m_τ = 1.777 GeV.

### What This Means for the Pre-Filter

**In "found 2, missed 1" events:**
- The missed pion is the lowest-pT one **38% of the time**.
- But any pion position can be missed — it's not exclusively the bachelor.
- The ρ mass resonance at 770 MeV is the strongest physics signature for
  identifying signal *pairs*, but the pre-filter currently has no pairwise
  feature computation.

**The model finds 2 of 3 pions in 56% of events** (26% find 2/3, 30% find
3/3). This suggests a potential strategy: **anchor on the found pair to search
for the third pion**.

### Key Physics Signatures (Currently Unexploited)

| Signature | Level | Exploited? |
|-----------|-------|-----------|
| Displaced vertex (dxy, dz) | Per-track | Yes — dominant discriminant |
| Track quality (normChi2, hits) | Per-track | Yes |
| Invariant mass ≈ ρ(770) | **Pairwise** | No |
| Charge product (OS for ρ pair) | **Pairwise** | No |
| Sum(3π) mass ≤ m_τ | **Triplet** | No |
| Correlated impact parameters | **Pairwise** | No |
| Characteristic opening angle | **Pairwise** | No |

The strongest physics signatures are relational (pairwise/triplet), not
per-track. But exploiting them requires signal-enriched neighborhoods where
signal tracks are actually neighbors.

---

## 7. Root Cause Analysis

### Why R@200 = 0.62 Is a Hard Wall

The plateau at 0.62 can be decomposed:

1. **Per-track features alone achieve R@200 ≈ 0.59** (BDT ceiling). The
   remaining 0.03 comes from kNN context — a marginal contribution.

2. **kNN in (η,φ) is 99.7% noise.** Any neighborhood-based operation
   (aggregation, attention, edge features) operates on noise. Richer
   processing of noise neighbors does not produce signal.

3. **The uncanny valley (pT 0.3–0.5, |dxy_sig| < 0.5) is a feature-space
   overlap.** These signal pions are kinematically indistinguishable from
   background using any per-track feature set we've tested.

4. **Pairwise physics features (invariant mass, ΔR, charge product) cannot
   help on the current graph** because signal pairs almost never appear as
   kNN neighbors in (η,φ). The pairwise features are physics-motivated but
   computed between wrong pairs.

### The Fundamental Tension
- Per-track features hit a ceiling at ~0.59.
- Pairwise features are the key to going higher (ParT ablation: 0.86 → 0.40
  without them; IAFormer matches ParT with pairwise features alone).
- But pairwise features require signal-enriched neighborhoods, which don't
  exist in the current fixed (η,φ) kNN graph.

**Breaking the wall requires changing WHICH tracks interact, not HOW they
interact.**

---

## 8. Open Questions for Deep Research

### Architecture Questions
1. **How to construct signal-enriched neighborhoods?** GravNet (learned kNN)
   failed — are there other approaches? Iterative graph refinement? Attention-
   based graph construction? Cascade with recomputed kNN?

2. **Can a cascade architecture work?** BDT R@500 = 0.85. If Stage 1 narrows
   to ~500, the signal fraction improves from 0.27% to 0.60%. Is this enough
   enrichment for pairwise features to become effective in Stage 2?

3. **Is per-track scoring fundamentally the wrong paradigm?** Object
   Condensation, Slot Attention, and DETR queries all operate on track
   *relationships* rather than individual track scores. Do any of these work
   at 0.3% signal fraction?

### Physics-Motivated Questions
4. **Can we anchor on the found pair to find the third pion?** The model finds
   2/3 pions in 56% of events. The found pair constrains the third pion's
   kinematics (m(3π) ≤ 1.777 GeV, charge constraint). Is there a way to
   exploit this without combinatorial explosion?

5. **Are there HEP tracking approaches designed for extreme signal sparsity?**
   Standard track finding (Exa.TrkX, GNN4ITk) operates at much higher signal
   fractions. Are there approaches from rare-decay searches or heavy-ion
   physics that handle 0.3% signal?

### Representation Questions
6. **Could a fundamentally different paradigm break the wall?**
   - Edge classification (label pairs, not individual tracks)
   - Learned graph construction (build graph based on task, not geometry)
   - Hierarchical approaches (local → regional → global)
   - Attention without explicit graph (sparse attention patterns)

7. **What does the ML literature say about retrieval at 0.3% positive rate?**
   Information retrieval, recommendation systems, and anomaly detection all
   deal with extreme class imbalance. Are there techniques from those fields
   that transfer to this geometric/physics setting?

### Practical Questions
8. **What is the theoretical ceiling for R@200 given these features?** Is there
   a Bayes-optimal bound that tells us how close 0.62 is to the maximum
   achievable with per-track features?

9. **Should we increase K?** If K=500 is where R@K becomes acceptable (0.85),
   maybe the pre-filter target should be R@500 instead of R@200, deferring
   more work to Stage 2.

---

## Appendix A: Detailed Failed Experiment Analysis

### A.1 ASL (Asymmetric Loss)
- **What:** Replaced ranking loss with ASL + ranking combination.
- **Result:** R@200 degraded from 58% to 41% over 9 epochs.
- **Root cause:** ASL's hard clip=0.05 zeros gradients from easy negatives.
  Ranking models need gradients from the FULL score distribution.

### A.2 Score Propagation
- **What:** Post-processing graph smoothing: `s = 0.3 * mean(neighbors) + 0.7 * s`.
- **Result:** No improvement.
- **Root cause:** Only 28% of GT pions have ≥1 GT neighbor. For the 72%
  without GT neighbors, smoothing dilutes their scores.

### A.3 Pairwise LV Features in Message Passing
- **What:** Added invariant mass, ΔR, kT, z as edge attributes in kNN MP.
- **Result:** R@200 collapsed from 52% (ep1) to 24% (ep9).
- **Root cause:** Pairwise features dominate the representation, AE
  reconstruction collapses to near-zero, latent space degenerates.

### A.4 Pairwise Features as Attention Bias
- **What:** Pairwise features → MLP → scalar attention logit per neighbor.
- **Result:** Same collapse. R@200 dropped from 50.7% to 38.8% by epoch 3.
- **Root cause:** Same AE collapse. The problem is not how pairwise features
  are injected — it's that ANY pairwise features cause AE reconstruction to
  shortcut through relational info.

### A.5 Triplet Scorer
- **What:** For top-200 anchors, enumerate triplets from 512 nearest neighbors.
- **Result:** OOM on 96GB GPU at batch_size=48.
- **Root cause:** 88% of kNN pairs pass the mass cut (nearby tracks are soft,
  producing low pairwise masses). C(450,2) ≈ 100K triplets per anchor ×
  200 anchors × 96 events = 2 billion triplets.

### A.6 GravNet with Distance Loss
- **What:** Learned coordinate space for kNN + distance-based metric learning.
- **Result:** R@200 ≈ 0.62, same as baseline.
- **Root cause:** With only 3 signal tracks per event at median ΔR=1.3, the
  learned space could not find a projection where signal clusters.

### A.7 EdgeConv on (η,φ) kNN
- **What:** Pairwise physics features (LV features) on standard kNN edges.
- **Result:** R@200 ≈ 0.62.
- **Root cause:** kNN neighbors are 99.7% noise — pairwise features between
  signal and noise pairs carry no discriminative information.

---

## Appendix B: Key Numbers at a Glance

| Metric | Value |
|--------|-------|
| Tracks per event (mean) | 1130 |
| Tracks per event (median) | 1211 |
| Signal tracks per event | 3 |
| Signal fraction | 0.27% |
| **Best R@200** | **0.629** |
| Best P@200 | 0.368 |
| Best median GT rank | 108 |
| BDT R@200 (no kNN) | 0.589 |
| BDT R@500 | 0.85 |
| BDT R@600 | 0.90 |
| d-prime (score separation) | 1.29 |
| GT rank p75 | 355 |
| GT rank p90 | 596 |
| K for 90% of events having ≥2 GT | 562 |
| Model parameters | 226K |
| Dataset events | 19K |
| Uncanny valley recall (pT 0.3–0.5, \|dxy\| < 0.5) | ~24% |

---

## Appendix C: References

### Papers Cited in Experiments
- Qu et al., "Particle Transformer for Jet Tagging," ICML 2022 — [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)
- "IAFormer: Interaction-Aware Transformer," 2025 — [arXiv:2505.03258](https://arxiv.org/abs/2505.03258)
- Qu & Gouskos, "ParticleNet," Phys. Rev. D 2020 — [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)
- Qasim et al., "GravNet," Eur. Phys. J. C 79, 2019 — [arXiv:1902.07987](https://arxiv.org/abs/1902.07987)
- Kieseler, "Object Condensation," Eur. Phys. J. C 80, 2020 — [arXiv:2002.03605](https://arxiv.org/abs/2002.03605)
- Lieret et al., "High Pileup OC Tracking," 2023 — [arXiv:2312.03823](https://arxiv.org/abs/2312.03823)
- Locatello et al., "Slot Attention," NeurIPS 2020 — [arXiv:2006.15055](https://arxiv.org/abs/2006.15055)
- Carion et al., "DETR," ECCV 2020 — [arXiv:2005.12872](https://arxiv.org/abs/2005.12872)
- Ju et al., "Exa.TrkX Pipeline," Eur. Phys. J. C 2021 — [arXiv:2103.06995](https://arxiv.org/abs/2103.06995)
- ATLAS GN2, "Transforming jet flavour tagging," Nature Communications 2025 — [arXiv:2505.19689](https://arxiv.org/abs/2505.19689)
- Sander et al., "Fast Differentiable Top-K," 2023 — [arXiv:2302.01425](https://arxiv.org/abs/2302.01425)
- "Why is Attention Sparse in ParT?" — [arXiv:2512.00210](https://arxiv.org/abs/2512.00210)

### HEP Track Reconstruction
- GNN4ITk ATLAS — [EPJ Web Conf. 2024](https://www.epj-conferences.org/articles/epjconf/pdf/2024/05/epjconf_chep2024_03030.pdf)
- Caron et al., "TrackFormers," 2024 — [arXiv:2407.07179](https://arxiv.org/abs/2407.07179)

### Loss Functions & Training
- Yang et al., "SoftmaxLoss@K," KDD 2025 — [arXiv:2508.05673](https://arxiv.org/abs/2508.05673)
- Kukleva et al., "Temperature Schedules," ICLR 2023 — [arXiv:2303.13664](https://arxiv.org/abs/2303.13664)
- Cao et al., "LDAM/DRW," NeurIPS 2019 — [arXiv:1906.07413](https://arxiv.org/abs/1906.07413)
- Li et al., "Equalized Focal Loss," CVPR 2022 — [arXiv:2201.02593](https://arxiv.org/abs/2201.02593)

### Alternative Architectures
- Corso et al., "PNA," NeurIPS 2020 — [arXiv:2004.05718](https://arxiv.org/abs/2004.05718)
- Wang et al., "DGCNN," ACM TOG 2019 — [arXiv:1801.07829](https://arxiv.org/abs/1801.07829)
- Brockschmidt, "GNN-FiLM," ICML 2020 — [arXiv:1906.12192](https://arxiv.org/abs/1906.12192)

### Tau Physics
- "Three-pion decays of the tau lepton" — [arXiv:1006.2919](https://arxiv.org/abs/1006.2919)
