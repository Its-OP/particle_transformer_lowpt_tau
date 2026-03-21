# Research Report: Breaking the R@200=0.62 Plateau

**Date:** 2026-03-21

---

## 1. Plateau Diagnosis

The pre-filter plateaus at R@200 ≈ 0.62 regardless of architecture:

| Model | Branch | Params | R@200 | Rank | Notes |
|-------|--------|--------|-------|------|-------|
| Phase-B (MLP+kNN) | `redesign` | 225K | 0.623 | 111 | Baseline |
| ParticleNeXt backbone | `partnext-v2` | 551K | ~0.62 | ~110 | Attention aggregation, LV edge features |
| ISAB global context | `isab` | 529K | ~0.62 | ~110 | 32 inducing points, global event summary |

**Failed hypothesis:** "Limited receptive field." Expanding context into ~1100
tracks that are 99.7% noise just averages out the signal.

**New hypothesis:** The bottleneck is **input feature quality**. The model's
ability to discriminate signal from noise is limited by what it sees, not how
it processes what it sees.

---

## 2. Input Feature Analysis (CRITICAL FINDINGS)

### 2.1 Preprocessing destroys dz_significance

The auto-generated preprocessing for `track_dz_significance`:
- center = 0.198, scale = **0.0034**
- Standardization: `(raw - 0.198) / 0.0034`, clipped to [-5, 5]
- Usable raw range: only [0.181, 0.216]

**Actual distribution:**
```
dz_significance: min=-44002, max=24767, std=488
p5=-692, p25=-156, p50=0.18, p75=159, p95=705
```

**99.9% of all tracks are clipped to ±5.** The feature effectively carries
zero information after preprocessing.

This was documented as the feature that "recovers 93% of uncanny valley GT
pions" when it was added. Its information is being destroyed.

**Signal vs noise separation EXISTS but is invisible to the model:**
- Signal: 14.5% have |dz_sig| < 5, median = 0.40
- Noise: 8.6% have |dz_sig| < 5, median = 0.18

**Fix:** Replace standardization with log-transform:
`sign(x) * log(1 + |x|)` preserves the full dynamic range.

### 2.2 Three more features with extreme scaling

| Feature | Scale | Issue |
|---------|-------|-------|
| `track_dz_significance` | 0.0034 | 99.9% clipped (see above) |
| `track_pt_error` | 67.14 | Heavy-tailed, needs log transform |
| `track_covariance_phi_phi` | 7,530 | Heavy-tailed, needs log transform |
| `track_covariance_lambda_lambda` | 86,704 | Heavy-tailed, needs log transform |

All four features have distributions with extreme tails. The auto-preprocessing
computes scale from IQR, producing tiny scales that clip most values.

**Fix:** Use `log(1 + x)` or `log(x)` transforms in the YAML `new_variables`
section, then standardize the log-transformed values.

### 2.3 Missing features available in .root file

The source file `data/low-pt/merged_noBKstar.root` contains Track branches
NOT currently exported to parquet:

| Feature | Physical meaning | Why it matters |
|---------|-----------------|----------------|
| **`Track_normChi2`** | χ²/ndof of track fit | **#1 CMS track quality discriminator.** Measures how well hits fit a coherent helix. Fake/noise tracks have high χ². |
| `Track_nValidHits` | Total silicon hits (pixel + strip) | Current model only has pixel hits. Total hits is a stronger quality indicator. |
| `Track_dzTrg` | dz to trigger vertex | Measures compatibility with event vertex. |
| `Track_vx/vy/vz` + `PV_x/y/z` | Track + primary vertex positions | Enables 3D impact parameter relative to PV. |

### 2.4 Redundancy and missing derived features

**Redundancy:** `px, py, pz` are deterministic functions of `pt, eta, phi`.
Five features encode 3 degrees of freedom. And **pT itself is not an explicit
feature** — the network must reconstruct it from sqrt(px² + py²).

**Standard HEP features not present:**
- `log(pT)` — used by every CMS/ATLAS tagger (DeepJet, DeepTau, GN2)
- `pT_error / pT` — relative momentum uncertainty (more meaningful than absolute)
- `log(dxy_significance)`, `log(dz_significance)` — log-transforms of significances
- IP3D = sqrt(dxy² + dz²) and IP3D significance

---

## 3. Physics-Derived Features from HEP Literature

### What CMS/ATLAS track classifiers use (that we don't)

Every major CMS and ATLAS track-level classifier uses **normalizedChi2** as the
top discriminating feature. The CMS Run-3 DNN track classifier, ATLAS GN2 flavour
tagger, DeepJet b-tagger, and DeepTau tau identifier all include it.

Standard HEP practice for per-track features:
1. **Log-transform** heavy-tailed quantities (pT, covariances, errors)
2. **Relative features** (pT/jetPT, deltaR to axis, pT rank among neighbors)
3. **normalizedChi2** as primary track quality measure
4. **Total hit counts** (not just pixel — include strip/total silicon)

### From DeepTau (CMS tau ID, arXiv:2201.08458):
- Uses dxy AND dxy_error **separately**, not just significance
- Uses normalizedChi2, nPixelLayers, nStripLayers, lostInnerHits
- Uses `pfCand_fromPV` quality and PUPPI weights for pileup rejection

### From CMS low-pT tau ID (CMS-DP-2020-026):
- No jet seed — starts from reconstructed charged pions (our exact approach)
- Per-track ML pre-filter before triplet building (validates our architecture)

### Tau → 3π physics constraints:
- τ lifetime: cτ = 87 μm → mean flight ~240 μm at pT=5 GeV
- Daughter pions: pT ~ 0.5–3 GeV each, spread over ΔR ~ 1+
- Invariant mass ≤ m_τ = 1.777 GeV (passes through a₁(1260))
- Charge sum |Q| = 1

### Tier 1 features to add:

| Feature | Source | Physics motivation |
|---------|--------|-------------------|
| `Track_normChi2` | .root branch | #1 CMS track quality variable. Fakes have high χ²/ndof. |
| `sqrt((vx-PV_x)²+(vy-PV_y)²+(vz-PV_z)²)` | Derived | 3D track-to-PV distance. Separates prompt / tau / pileup. |
| `Track_nValidHits` | .root branch | Total silicon hits. Complements pixel-only count. |

### Tier 2 features to add:

| Feature | Source | Physics motivation |
|---------|--------|-------------------|
| `Track_dxy` (raw, not significance) | .root branch | DeepTau uses raw + error separately, not just ratio. |
| `Track_dz` (raw) | .root branch | Same — unsquashed values carry geometric information. |
| `Track_covQopQop` | .root branch | Momentum precision. Completes diagonal covariance set. |
| `log(pT)` | Derived | Standard in every CMS/ATLAS tagger. |
| `ptError / pT` | Derived | Relative momentum uncertainty (more meaningful than absolute). |

---

## 4. Denoising and Signal-Focused Learning

Full report: `whimsical-singing-wombat-agent-a228bd14ec0715000.md`

### Top findings

**Supervised Minority contrastive loss (SupMin)** — Mildenberger et al., CVPR 2025.
Standard SupCon collapses under binary imbalance. SupMin applies contrastive
supervision only to signal tracks, NT-Xent on noise for uniformity. Teaches
"what signal looks like" rather than "what noise looks like."
*Compatible with per-track scoring. Add as auxiliary loss.*

**Online Hard Example Mining (OHEM)** — Instead of 50 random negatives, select
the ~30 hardest (highest-scoring noise tracks). Forces model to focus on the
decision boundary rather than trivially easy noise.
*Drop-in replacement for random sampling in `_ranking_loss`.*

**Denoising pretraining (Point-RTD)** — Inject foreign tracks from other events,
train discriminator to identify intruders. Teaches "what belongs" without labels.
*Pretraining phase before fine-tuning the scorer.*

**Exa.TrkX metric learning** — Embed tracks so signal clusters and noise spreads.
The HEP community's proven approach. Simple kNN in the learned space then
naturally finds signal neighborhoods.

---

## 5. Data Augmentation

Full report: `whimsical-singing-wombat-agent-a6c720dae6b0924b8.md`

### Top findings

**Signal copy-paste** — Paste GT tracks from event A into event B. Shifts
positive rate from 0.3% to 0.5-1.0%. Trivial for unordered sets (just
concatenate). Strongly supported by literature.
*Highest-impact, lowest-effort augmentation.*

**Track dropout** — Randomly drop 20-50% of background tracks during training.
Shifts positive rate and regularizes.

**Physics smearing** — Small Gaussian noise on pT, eta, phi of copied signal
tracks to increase diversity. Best combined with copy-paste.

---

## 6. Loss Redesign

Full report: `whimsical-singing-wombat-agent-a6c720dae6b0924b8.md`

### Top findings

**Post-hoc logit adjustment** (Menon et al., ICLR 2021) — Add offset
`τ * log(N_neg/N_pos)` to trained model scores. Zero-cost diagnostic.
If it helps, confirms majority-class bias → leads to decoupled training.

**Decoupled training** (Kang et al., ICLR 2020) — Train backbone on natural
distribution, freeze, retrain classification head with balanced sampling.
Outperformed all carefully designed losses in the original paper.

**NeuralNDCG** — Differentiable relaxation of NDCG/Recall@K. Directly
optimizes the evaluation metric instead of a pairwise proxy.

---

## 7. Architectural Hypotheses for Signal Discrimination

### 7.1 GravNet: learned kNN coordinates (TOP RECOMMENDATION)

Replace fixed (eta, phi) kNN with kNN in a **learned** S-dimensional space.
A linear projection maps track features → S-dim coordinates; kNN built there.
Signal tracks can be mapped close together so neighborhoods become enriched.

**Key difference from ISAB/ParticleNeXt:** Those expanded receptive field SIZE
(attend to all 1100 tracks). GravNet keeps local receptive field (k=16) but
makes the locality ADAPTIVE — the network learns which tracks should be neighbors.
This is receptive-field-quality, not receptive-field-size.

Complexity: O(P·k) — same as current. Only adds one d→S projection.
`TauTrackFinderV3` already uses dual kNN (physical + learned) at the full model
level, confirming the pattern works in this codebase.

**Reference:** Qasim et al., Eur. Phys. J. C 79 (2019). PyG `GravNetConv`.

### 7.2 Two-phase cascade within the model

Phase 1: cheap MLP scores all 1100 → top-200. Phase 2: kNN among 200 survivors
with a second scorer. Signal concentration jumps 0.3% → 1.5% (5.5×).

The current `select_top_k` + `filter_tracks` already implements Phase 1.
Phase 2 adds kNN recomputed among survivors only.

Complexity: O(P·d + K·k·d), still linear in P.

### 7.3 Auxiliary heads (proven by ATLAS GN2)

Add training-only auxiliary objectives on the shared backbone:
- **Vertex compatibility** (pair-level): predict whether track pairs share a vertex.
  ATLAS GN2 reports significant gains from this exact auxiliary (Nature Comms 2025).
- **Tau displacement** (regression): predict flight distance from per-track features.

Zero inference cost — auxiliary heads dropped after training, only backbone improves.

### 7.4 Hard negative contrastive within kNN

For each signal track, its kNN contains the hardest negatives (geometrically
close noise). Contrastive loss that pushes these apart using the already-computed
kNN graph — zero extra computation.

**Reference:** ArcFace (CVPR 2019), CurricularFace (CVPR 2020).

---

## 8. Ranked Recommendations

| Priority | Recommendation | Type | Expected impact | Effort | Evidence |
|----------|---------------|------|----------------|--------|----------|
| **1** | **Fix dz_significance preprocessing** | Feature fix | **HIGH** — 99.9% of info currently destroyed | Config change | Verified empirically |
| **2** | **Add Track_normChi2** | New feature | **HIGH** — #1 CMS discriminator | Data pipeline | Universal in HEP literature |
| **3** | **Log-transform heavy-tailed features** | Feature fix | Medium-high | Config change | Standard HEP practice |
| **4** | **Add log(pT) as explicit feature** | New feature | Medium | Config change | Used by all CMS/ATLAS taggers |
| **5** | **Signal copy-paste augmentation** | Training | Medium | ~30 lines | CVPR 2021, NeurIPS 2022 |
| **6** | **OHEM (hard negative mining)** | Training | Medium | ~20 lines in loss | CVPR 2016 |
| **7** | **Post-hoc logit adjustment** | Diagnostic | Low-medium | 1 line | ICLR 2021 |
| **8** | **Add relative pT error (ptErr/pT)** | New feature | Low-medium | Config change | DeepJet, DeepTau |
| **9** | **Decoupled training** | Training | Medium (if #7 helps) | Moderate | ICLR 2020 |
| **10** | **SupMin contrastive auxiliary loss** | Training | Unknown | ~100 lines | CVPR 2025 |
| **11** | **GravNet learned kNN coordinates** | Architecture | High | Medium | Eur. Phys. J. C 2019 |
| **12** | **Two-phase cascade (score → refilter)** | Architecture | High | Medium | Viola-Jones pattern |
| **13** | **Auxiliary vertex compatibility head** | Training | Medium-high | Low | ATLAS GN2, Nature Comms 2025 |

---

## 9. Suggested Experiment Sequence

```
Experiment 1: Fix preprocessing (zero architecture change)
  - Log-transform dz_significance, pt_error, covariance_phi_phi, covariance_lambda_lambda
  - Add log(pT) as explicit feature
  - Add relative pT error (ptErr/pT)
  → Retrain Phase-B model with fixed features

Experiment 2: Add normChi2 (requires re-exporting parquet from .root)
  - Export Track_normChi2, Track_nValidHits from .root to parquet
  - Add to YAML data config
  → Retrain with expanded feature set

Experiment 3: Signal copy-paste augmentation
  → Retrain best model from Exp 1/2 with copy-paste

Experiment 4: OHEM negative mining
  → Replace random sampling with hardest-negative selection

Experiment 5: Post-hoc logit adjustment on best model
  → Diagnostic: if helps, proceed to decoupled training
```

Experiments 1-2 target the most likely bottleneck (feature quality).
Experiments 3-5 are complementary training improvements.
