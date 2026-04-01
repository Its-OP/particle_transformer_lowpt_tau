# Experiment Log

**Project:** Low-pT tau → 3π track finding with Particle Transformer
**Date:** 2026-03-31
**Dataset:** ~270K train / ~84K val events, ~1100 tracks/event, 3 GT pions (0.27% signal)

---

## Part 1: Successful Experiments

### Filter (Stage 1)

> Baseline: R@200 = 0.623 (Phase-B MLP + kNN, 225K params)

---

#### Phase-B MLP + kNN Baseline
**Branch:** `redesign` | **Params:** 225K | **R@200:** 0.623 | **d':** 1.30 | **Rank:** 111

Per-track MLP encoder with 2 rounds of kNN(k=16) message passing and max-pool
aggregation, followed by a scoring MLP. Uses 16 input features including
log-transformed displacement, track quality, and derived kinematic variables.
Established the working baseline for all subsequent experiments.

---

#### Extended 16 Features
**Branch:** `chi2` | **Params:** 226K | **R@200:** 0.625 | **d':** — | **Rank:** —

Added log-transformed dz_significance, normalized chi-squared, covariance terms,
and derived features (log pT, relative pT error) to the input feature set.
Confirmed the expanded feature pipeline works correctly. The marginal improvement
over baseline demonstrates that per-track features alone are not the bottleneck.

---

#### SupMin Contrastive Loss
**Branch:** `supmin` | **Params:** 275K | **R@200:** 0.616 (epoch 16) | **d':** — | **Rank:** —

Auxiliary contrastive loss that clusters signal track embeddings in latent space,
with weight=0.5. Showed a slight edge over the baseline at matched epoch count
and was still converging when the run ended. Promising direction that was
superseded by the cascade approach.

---

### Refiner (Stage 2)

> Baseline: Stage 1 R@200 = 0.628

---

#### Regular ParT Cascade
**Branch:** `phase2-part` | **Params:** 25M (embed=512, 8L, 8H) | **R@200:** 0.676 | **d':** 1.15 | **Rank:** 76

Standard Particle Transformer with 4 Lorentz-vector pairwise features (ln kT,
ln z, ln ΔR, ln m²) as attention bias, operating on the top-600 tracks from
Stage 1. First demonstration that pairwise self-attention can meaningfully improve
over per-track scoring (+0.048 over Stage 1).

---

#### Physics ParT Cascade
**Branch:** `phase2-part-more-interactions` | **Params:** 25M | **R@200:** 0.693 | **d':** 1.21 | **Rank:** 61

Extended the pairwise attention bias with 5 physics-motivated features: charge
product (q_i × q_j), dz vertex compatibility (|Δdz_sig|), ρ(770) mass Gaussian
indicator, ρ-opposite-sign conjunction, and φ-corrected dxy compatibility. These
features encode τ → 3π decay physics that generic Lorentz features miss. Improved
R@200 by +0.017 over regular ParT and +0.065 over Stage 1 alone.

---

#### Pairwise Feature Diagnostics
**Type:** Diagnostic (no training)

Measured discriminative power of candidate pairwise features on ~1100 GT-GT vs
GT-BG track pairs. Validated that pairwise physics features discriminate signal
pairs from background pairs at the attention level:

| Feature | |d'| (GT-GT vs GT-BG) |
|---------|---------------------|
| Invariant mass | 1.09 |
| dz compatibility | 0.58 |
| ρ-OS indicator | 0.57 |
| ρ mass indicator | 0.53 |
| φ-corrected dxy | 0.39 |
| Charge product | 0.33 |

---

#### BatchNorm Fix
**Type:** Bug fix

The pre-filter checkpoint had stale BatchNorm running statistics because the
training validation loop ran in `train()` mode, corrupting `running_mean` and
`running_var`. Switching Stage 1 to `train()` mode inside the cascade recovered
R@600 from 0.70 to 0.91, unblocking the cascade's potential.

---

#### Shallow ParT Cascade (2 layers)
**Params:** ~6M (embed=512, 2L, 8H) | **R@200:** 0.692 | **d':** 1.18 | **Rank:** 65

Two-layer Particle Transformer with 6 physics pairwise features, matching the
8-layer Physics ParT (R@200=0.693) with 4× fewer parameters. Confirms that
layers 3-8 contribute no ranking improvement and the model over-smooths at
depth. Now the default Stage 2 configuration.

---

#### d-prime Metric Fix
**Type:** Bug fix

The `MetricsAccumulator` included `-inf` scores from non-selected tracks when
computing d-prime, producing NaN in the standard deviation and reporting d'=0.000.
Fixed by filtering to finite scores only.

---

## Part 2: Failed Experiments

### Filter (Stage 1)

---

#### ASL Loss
**R@200:** 0.41 (collapsed from 0.58 baseline)

Asymmetric Loss replaced the pairwise ranking objective. The hard probability
threshold zeroed gradients from easy negatives, destroying the ranking signal
that the pre-filter needs for relative track ordering across the full score
distribution.

---

#### OHEM (Hard Negative Mining)
**Branch:** `ohem` | **R@200:** 0.50

Focused training on the top-K hardest negatives per positive. At 0.27% signal
fraction the ranking task is already maximally difficult, and concentrating on
even harder negatives destabilized training rather than sharpening the decision
boundary.

---

#### Pairwise LV Features in Message Passing
**Branch:** `pairwise` | **R@200:** collapsed to 0.24

Added invariant mass, ΔR, kT, z as edge attributes in kNN message-passing
aggregation. The pairwise features dominated the learned representation and
caused the autoencoder latent space to collapse, losing all per-track
discrimination. dxy_significance found rate became uniform across all bins.

---

#### Pairwise Features as Attention Bias
**R@200:** 0.39

Routed pairwise features through attention weights instead of the value stream,
attempting to avoid the collapse seen with direct concatenation. The same latent
collapse occurred — the failure was not the injection method but the fundamental
incompatibility between pairwise features and the autoencoder reconstruction
pathway in the hybrid architecture.

---

#### Triplet Scoring
**Result:** OOM on 96GB GPU

Attempted to enumerate candidate triplets from kNN neighbors and score them with
an MLP. The invariant mass cut was not selective among nearby soft tracks — 88%
of kNN pairs passed — producing a combinatorial explosion that exceeded available
memory.

---

#### Copy-Paste Augmentation
**Result:** Abandoned

Attempted to augment training data by injecting GT tracks from donor events into
recipient event backgrounds. Implementation issues prevented meaningful
evaluation.

---

### Refiner (Stage 2)

---

#### Boundary Negative Sampling
**Params:** 25M (embed=512, 8L, 8H) | **R@200:** 0.600 (best at epoch 17, vs 0.676 baseline)

Restricted negative sampling to tracks ranked 150–250 (near the rank-200
boundary) instead of random sampling from all ~597 background tracks. The model
only learned to distinguish GT from tracks near position 200, losing all sense of
the broader ranking. R@200 oscillated between 0.55 and 0.60 across epochs — the
model would push GT above rank 200 in one epoch but, lacking calibration from the
full score distribution, GT would fall back below 200 in the next. Same root cause
as OHEM: restricting the negative distribution destroys the ranking signal.

---

#### Combinatorial Triplet Post-Processing
**Result:** ~120K surviving triplets per event at K=200

Applied progressive physics and Dalitz filters (charge conservation, tau mass
bound, a₁ mass window, ρ resonance, Dalitz boundary, rho band structure, bachelor
energy, rho helicity) to enumerate valid triplets from the top-K tracks. Even
after all 8 filters, too many soft-track triplets survive because low-pT tracks
easily satisfy mass and resonance criteria combinatorially. The GT triplet itself
only survived all filters in 10–25% of events.

---

#### Combinatorial Triplet Binned by pT
**Type:** Diagnostic

Repeated the combinatorial experiment stratified by mean GT pT bins (low, mid,
high). The filters were equally non-selective at all pT ranges. Physics-based
combinatorial search is not viable as a post-processing step at any momentum
scale in this dataset.

---

## Part 3: Neutral Experiments

### Filter (Stage 1)

---

#### GravNet (Learned kNN)
**Branch:** `gravnet` | **R@200:** ≈0.62

Replaced fixed (η, φ) kNN with learned-coordinate kNN plus a distance-based
metric learning loss. The learned embedding space failed to produce
signal-enriched neighborhoods because 3 signal tracks among 1100 are too sparse
for any single coordinate projection to cluster them.

---

#### EdgeConv on (η, φ) kNN
**Branch:** `edgeconv` | **R@200:** ≈0.62

Added richer edge features (pairwise Lorentz physics) to the existing kNN graph
edges. The same 99.7%-noise edge structure meant the additional features had no
useful signal-signal pairs to operate on.

---

#### Score Propagation (Graph Smoothing)
**R@200:** No improvement

Post-hoc smoothing of pre-filter scores through the kNN graph. Since 72% of GT
pions have zero GT neighbors in the (η, φ) kNN, smoothing diluted their scores
with background neighbor scores rather than boosting them.

---

#### ParticleNeXt Backbone
**Branch:** `partnext-v2` | **Params:** 551K | **R@200:** ≈0.62

Attention-weighted aggregation with multi-scale EdgeConv and Lorentz edge
features. More than double the parameters and substantially richer neighborhood
processing, but on the same noisy kNN graph. The graph structure — not the
aggregation method — is the binding constraint.

---

#### ISAB Global Context
**Branch:** `isab` | **Params:** 529K | **R@200:** ≈0.62

32 learned inducing points providing global event-level context via set
attention (Induced Set Attention Blocks). Compressing 1100 tracks that are
99.7% noise into 32 summary points loses the rare signal tracks that need
to be identified.

---

#### RS@K Surrogate Loss
**Branch:** `phase1/rsk` | **R@200:** 0.623

Differentiable approximation of R@200 using nested sigmoid relaxations as
the direct training objective. No improvement because the bottleneck was the
noisy kNN graph structure, not the loss function — the model had already
extracted all the ranking signal the graph could provide.

---

#### Curriculum Training
**Branch:** `phase1/curriculum` | **R@200:** 0.628

Three-phase negative subsampling schedule with deferred re-weighting,
progressively increasing background track count during training. Marginal
+0.005 improvement, within noise — the same structural bottleneck limits
any training recipe applied to Stage 1.

---

#### Multi-Space Composite Graph
**Type:** Diagnostic (no training)

Tested kNN graphs in five physics-motivated feature spaces (dz_significance,
transverse vertex proxy, opposite-sign η-φ, log-pT-augmented η-φ) and their
union. The full composite graph with 68 effective neighbors achieved 39.7% GT
connectivity, compared to 40.9% for simple (η, φ) kNN at k=32. Multi-space
graph construction offers no advantage over simply increasing k.

---

#### Per-Track ρ-Candidate Count
**Type:** Diagnostic (no training) | **d':** 0.15

Counted how many opposite-sign tracks near the ρ(770) mass window each track
has. The count turned out to be almost entirely a pT proxy — soft tracks produce
many low-mass pairs regardless of whether they are signal — providing negligible
signal-vs-background discrimination.

---

### Refiner (Stage 2)

---

#### DETR Decoder
**Type:** Blueprint only (not trained)

Explored in the improvement blueprint as an alternative Stage 2 architecture
using 3 learned queries with cross-attention to all K₁ tracks. Deprioritized
in favor of the ParT pairwise-bias approach, which directly injects physics
features as attention bias with simpler training dynamics.
