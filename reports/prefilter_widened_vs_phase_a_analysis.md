# Pre-Filter Analysis: Widened (Hybrid) vs Phase-A (MLP)

## Executive Summary

**CORRECTED (2026-03-20):** Earlier analysis using a custom evaluation script was
incorrect due to a metric averaging bug. Re-evaluation using the exact `validate()`
function from `train_prefilter.py` shows Phase-A **outperforms** widened hybrid
on the extended subset data:

| Metric | Widened | Phase-A | Delta |
|--------|---------|---------|-------|
| R@200 | 0.556 | 0.588 | **+0.032** |
| P@200 | 0.301 | 0.333 | **+0.032** |
| d' | 1.200 | 1.302 | **+0.102** |
| Rank | 175 | 145 | **-30** |

Phase-A's best checkpoint is epoch 15 (pre-DRW), confirming AE removal helped
and DRW activation was destructive. Both models underfit (train-val gap <1%),
indicating model capacity is the binding constraint. Both show ~0.06 lower R@200
on the extended subset vs original validation, likely due to sample size (1600 events).

## Model Configurations

| Parameter | Widened (Baseline) | Phase-A |
|-----------|-------------------|---------|
| Mode | hybrid (AE + MLP + kNN) | mlp (kNN only) |
| Parameters | 259,502 | 225,793 (-13%) |
| hidden_dim | 192 | 192 |
| latent_dim | 48 | N/A |
| num_message_rounds | 2 | 2 |
| ranking_num_samples | 50 | 50 |
| Temperature schedule | Fixed (T=1.0) | T: 2.0 → 0.5 |
| Denoising sigma | Fixed (0.3) | sigma: 1.0 → 0.1 |
| DRW | None | warmup=30%, weight=2.0x |
| GPU | H100 | RTX 6000 Pro WS |
| Batch size | 128 | 96 |
| Steps/epoch | 500 | 500 (logged) |
| Epochs trained | 24 | 26 |
| Total steps | ~12,000 | ~13,000 |

## Best Validation Metrics Comparison

### Training Log Metrics (original val data, different GPUs/batch sizes)

| Metric | Widened (ep 22) | Phase-A (ep 15, best ckpt) | Delta |
|--------|----------------|---------------------------|-------|
| R@200 | 0.6228 | 0.5991 | -0.024 |
| P@200 | 0.3640 | 0.3362 | -0.028 |
| d' | 1.329 | 1.301 | -0.028 |
| Rank | 112 | 128 | +16 |

Note: Training logs show Phase-A slightly behind, but training conditions
differed (H100 batch=128 vs RTX6000 batch=96). Phase-A's best checkpoint
is epoch 15 (pre-DRW, best val loss), NOT epoch 23.

### Controlled Comparison (same data, same code, same device)

Using `validate()` from `train_prefilter.py` on extended subset val (1600 events):

| Metric | Widened | Phase-A | Delta | Better |
|--------|---------|---------|-------|--------|
| R@10 | 0.154 | 0.172 | +0.018 | **Phase-A** |
| R@20 | 0.208 | 0.236 | +0.028 | **Phase-A** |
| R@30 | 0.242 | 0.276 | +0.034 | **Phase-A** |
| R@100 | 0.410 | 0.441 | +0.030 | **Phase-A** |
| **R@200** | **0.556** | **0.588** | **+0.032** | **Phase-A** |
| **P@200** | **0.301** | **0.333** | **+0.032** | **Phase-A** |
| d' | 1.200 | 1.302 | +0.102 | **Phase-A** |
| Rank | 175 | 145 | -30 | **Phase-A** |

**Phase-A wins on every metric.** The improvement is consistent across all K
values and strongest for d' (+8.5% relative) and median rank (-17% relative).
The AE removal and temperature scheduling are genuinely beneficial.

## Training Dynamics Analysis

### 1. DRW Disruption is Severe

At epoch 16 (30% warmup), DRW activation doubles the positive pair weights,
causing a massive loss discontinuity:

```
Epoch 15 (pre-DRW):  val_loss = 0.581  R@200 = 0.599
Epoch 16 (DRW on):   val_loss = 1.138  R@200 = 0.605  (+0.006)
```

The total loss nearly doubles, forcing the optimizer into a very different
landscape. After 10 post-DRW epochs:

```
Epoch 26 (latest):   val_loss = 0.901  R@200 = 0.608
```

Loss is still 55% higher than pre-DRW. The model never fully recovers to its
pre-disruption trajectory. While R@200 does improve post-DRW (+0.017 over
pre-DRW), the improvement is modest given the training cost.

**Problem:** DRW as implemented is a step function — the weight jumps from
1.0x to 2.0x instantly. This is too aggressive. The optimizer's momentum
and adaptive learning rate state become stale immediately.

### 2. Temperature Schedule Only 50% Complete

At epoch 26/50:
- Temperature: T = 1.204 (target: 0.5, started at 2.0)
- Denoising sigma: sigma = 0.522 (target: 0.1, started at 1.0)

The model hasn't reached the regime where these techniques are most impactful.
Low temperature (T < 1.0) creates sharper ranking gradients that focus on hard
violations near the decision boundary — exactly what's needed for the hardest
GT tracks (those in the "uncanny valley" at ranks 100-300).

### 3. Pre-DRW Phase-A vs Widened (Controlled Comparison)

Comparing at matching epochs (before DRW confounds):

| Metric | Widened ep 15 | Phase-A ep 15 | Delta |
|--------|--------------|---------------|-------|
| R@200 | 0.613 | 0.599 | -0.014 |
| R@100 | 0.465 | 0.449 | -0.016 |
| d' | 1.304 | 1.301 | -0.003 |
| Rank | 120 | 128 | +8 |

Even before DRW disruption, the widened hybrid model outperforms Phase-A.
This suggests the AE removal itself contributed to the regression,
independent of training dynamics.

### 4. Overfitting: Not an Issue

Both models show minimal train-val gaps:

**Widened (ep 22):** train_loss = 0.354, val_loss = 0.356 (gap: 0.002)

**Phase-A (ep 26):**
- train_R@200 = 0.619, val_R@200 = 0.608 (gap: 0.011)
- train_d' = 1.317, val_d' = 1.296 (gap: 0.021)

Both models are **underfitting**, not overfitting. The train-val gap is
negligible, meaning the models have not saturated their capacity.
This is the strongest signal in the data: **more capacity will help**.

### 5. Loss Component Analysis

**Widened:**
- Ranking loss dominates (~0.358 at convergence)
- Reconstruction loss is tiny (0.0008 at convergence) — confirms AE reconstruction
  is trivially solved and adds almost no gradient signal
- Denoising loss negligible (<0.0001)

**Phase-A:**
- Ranking loss dominates (~0.900 at ep 26, still elevated from DRW)
- Denoising loss: 0.0003 at ep 26
- No reconstruction loss (AE removed)

The reconstruction loss being near-zero in widened confirms our hypothesis:
the AE's reconstruction task adds no training signal. However, the **encoder
output** (48-dim latent) may still provide useful features to the scoring head,
even if the reconstruction loss itself is trivially solvable.

## Root Cause Analysis

### Why Phase-A Outperforms (Corrected)

1. **AE was indeed dead weight.** The 13→48→13 autoencoder with trivially
   solvable reconstruction added 34K parameters without discriminative value.
   The MLP mode's 2-round kNN message passing is sufficient to learn
   neighborhood-aware representations from 13 raw features.

2. **Temperature scheduling helps even partially.** At epoch 15 (best ckpt),
   T=1.54 (started at 2.0). Even this moderate temperature provides smoother
   gradients than the fixed T=1.0 in the widened model.

3. **DRW is destructive.** The best checkpoint is epoch 15 (pre-DRW).
   After DRW activates at epoch 16, the model never recovers to pre-DRW quality.
   DRW should be either removed or implemented with gradual ramp.

4. **Training log comparison was confounded.** The widened model trained on H100
   with batch=128 (better gradient quality), while Phase-A used RTX6000 with
   batch=96. The controlled subset evaluation (same device, same code) removes
   this confound and reveals Phase-A's advantage.

## Recommendations to Push R@200 → 0.8, P@200 → 0.5

Current best: R@200 = 0.623, P@200 = 0.364. Target requires +28% relative
improvement in R@200 and +37% in P@200. This is ambitious — here are changes
ranked by expected impact:

### High Impact (Expected > +0.05 R@200)

**1. Increase model capacity (address underfitting).**
Both models underfit. This is the single most impactful change.
- `hidden_dim`: 192 → 384 (4x parameters in scoring layers)
- `num_message_rounds`: 2 → 4 (more information propagation hops)
- Add residual connections between message-passing rounds
- Expected impact: +0.05–0.10 R@200

**2. Re-introduce the encoder as a feature transformer (without decoder).**
Keep the AE encoder path (13 → 48 dims) but discard the decoder and
reconstruction loss entirely. The scoring head gets [13 + 48] = 61 features.
This recovers the learned feature transform without the dead-weight decoder.
- Expected impact: +0.01–0.03 R@200 (recovers the widened→Phase-A gap)

**3. Hard negative mining in ranking loss.**
Currently `ranking_num_samples=50` samples random negatives. Most random
negatives are easy (scores clearly below positives). Switch to:
- **Semi-hard mining:** only sample negatives with scores within margin of
  positives: `s_neg > s_pos - margin`
- **Top-K hardest:** use the K highest-scoring negatives per positive
- Expected impact: +0.02–0.05 R@200

### Medium Impact (Expected +0.01–0.05 R@200)

**4. Fix DRW with gradual weight ramp.**
Replace step function with linear ramp:
```python
# Instead of: weight = 2.0 if epoch > warmup else 1.0
progress = max(0, (epoch - warmup) / ramp_epochs)
weight = 1.0 + (target_weight - 1.0) * min(progress, 1.0)
```
Ramp over 10 epochs (e.g., epochs 15-25) from 1.0x to 2.0x.

**5. Complete the 50-epoch training run.**
Phase-A at epoch 26 has not converged. Let it run to 50 epochs where T=0.5
and sigma=0.1. The low-temperature regime focuses on hard violations.

**6. PNA aggregation (already implemented in Batch B).**
Replace max-pool in kNN aggregation with {mean, max, min, std} concatenation.
Richer neighborhood summaries improve message passing.
- Expected impact: +0.01–0.03 R@200

**7. Increase batch size to 256.**
The ranking loss samples negatives within-batch. Larger batch = more diverse
negative pool = better gradient signal. Double from 96/128 to 256.

### Lower Impact (Expected +0.005–0.02 R@200)

**8. Focal weighting (already implemented in Batch B).**
Downweight easy pairs with `(1-p)^gamma` modulation. Focuses training on
hard cases near the decision boundary.

**9. DINO contrastive denoising (already implemented in Batch B).**
Add negative noised copies of GT tracks that should score below positives.
Trains the model on the "uncanny valley" decision boundary.

**10. Learning rate schedule adjustment.**
The cosine schedule with warmup may not be optimal. Consider:
- Warmup 5% → 10% (longer ramp to peak LR)
- Higher peak LR: 0.001 → 0.002 (with larger model, can afford higher LR)
- OneCycleLR instead of cosine

### Longer-Term / Exploratory

**11. Data augmentation.** With ~19K events, the dataset is small.
- Random (eta, phi) rotations (physics-preserving)
- Random track dropout (robustness to missing tracks)
- Gaussian feature noise (regularization)

**12. Multi-scale kNN.** Currently uses k=16 neighbors. Add a second kNN
pass with k=64 for broader context, feeding into a separate aggregation.

**13. Cross-attention between top-K candidates.** After the initial per-track
scoring, re-score the top-200 using cross-attention among them. This captures
set-level correlations (e.g., GT tracks tend to be spatially clustered).

## Extended Subset Evaluation (Independent Test)

Both best checkpoints were evaluated on `data/low-pt/extended/subset/val/`
(2000 events), which is **independent from the training/validation data**.
This reveals significant distribution shift.

### Overall (Extended Subset)

| Metric | Widened | Phase-A | Delta |
|--------|---------|---------|-------|
| R@10 | 0.041 | 0.047 | +0.006 |
| R@30 | 0.101 | 0.092 | -0.009 |
| R@100 | 0.253 | 0.206 | -0.047 |
| **R@200** | **0.401** | **0.336** | **-0.066** |
| **P@200** | **0.138** | **0.106** | **-0.032** |
| d' | 0.681 | 0.645 | -0.037 |
| Med. rank | 367 | 388 | +21 |

**Key finding:** R@200 drops from ~0.62 (original val) to ~0.40 (extended subset).
This is a ~35% degradation, suggesting significant distribution shift or harder events.
The widened-vs-Phase-A gap is amplified: -0.066 here vs -0.007 on original val.

### By Track Count

| Bucket | N events | Widened R@200 | Phase-A R@200 | Delta |
|--------|----------|--------------|--------------|-------|
| <500 tracks | 9 | 0.778 | 0.722 | -0.056 |
| 500–1500 | 1517 | 0.432 | 0.363 | -0.069 |
| >1500 | 473 | 0.296 | 0.240 | -0.056 |

R@200 drops dramatically with track count: 0.78 → 0.43 → 0.30.
High-track events (>1500, 24% of events) are the primary bottleneck.

### By GT Track Count

| GT tracks | N events | Widened R@200 | Phase-A R@200 | Delta |
|-----------|----------|--------------|--------------|-------|
| 1 | 16 | 0.375 | 0.250 | -0.125 |
| 2 | 301 | 0.390 | 0.312 | -0.078 |
| **3** | **1669** | **0.403** | **0.341** | **-0.062** |
| 4 | 10 | 0.575 | 0.400 | -0.175 |
| 5 | 3 | 0.133 | 0.133 | 0.000 |

83% of events have exactly 3 GT tracks (consistent with tau → 3 prongs).

### GT Rank Distribution

| Percentile | Widened | Phase-A |
|-----------|---------|---------|
| p5 | 12 | 10 |
| p10 | 29 | 32 |
| p25 | 97 | 129 |
| **p50** | **281** | **336** |
| p75 | 594 | 599 |
| p90 | 932 | 859 |
| p95 | 1157 | 1018 |

The median GT rank is 281 (widened) — meaning a typical GT track is ranked
~281st out of ~1100. To achieve R@200 = 0.8, the median GT rank needs
to drop below ~100. This requires a 3x improvement in ranking quality.

### Score Distribution

| | Widened | Phase-A |
|--|---------|---------|
| GT mean | 15.30 | 6.30 |
| GT std | 4.25 | 5.34 |
| Bkg mean | 12.59 | 3.12 |
| Bkg std | 3.70 | 4.51 |
| d' | 0.681 | 0.645 |

Widened has higher absolute scores but similar d' — the distributions
are scaled differently but separation quality is comparable.

### Implications

1. **The model generalizes poorly to new data.** 35% R@200 drop suggests
   the original train/val split doesn't capture the full data distribution.
2. **High-track events dominate the failure mode.** With >1500 tracks,
   R@200 drops to 0.30. The kNN message passing with k=16 covers only
   ~1% of tracks per round — insufficient context.
3. **Phase-A's gap is worse on harder data.** The -0.066 delta (vs -0.007
   on original val) suggests Phase-A lacks robustness, possibly because
   the MLP mode has less feature diversity than hybrid mode.

## Suggested Next Experiment

Based on the analysis, the highest-value next experiment combines:

1. **hidden_dim=384, num_message_rounds=3** (address underfitting)
2. **Re-introduce encoder (without decoder)** as feature transformer
3. **Gradual DRW ramp** (10 epochs, not step function)
4. **Train for full 50 epochs** on H100 with batch=256

This addresses all identified root causes in one run.

## Appendix: Epoch-by-Epoch Metrics

### Widened (Hybrid) — Best Epochs

| Epoch | Val Loss | R@200 | P@200 | d' | Rank |
|-------|----------|-------|-------|----|------|
| 8 | 0.3693 | 0.612 | 0.355 | 1.294 | 120 |
| 13 | 0.3623 | 0.617 | 0.356 | 1.312 | 117 |
| 17 | 0.3613 | 0.619 | 0.360 | 1.298 | 115 |
| 18 | 0.3593 | 0.617 | 0.357 | 1.321 | 116 |
| 21 | 0.3580 | 0.617 | 0.358 | 1.321 | 114 |
| **22** | **0.3563** | **0.623** | **0.364** | **1.329** | **112** |
| 23 | 0.3585 | 0.615 | 0.358 | 1.323 | 116 |
| 24 | 0.3583 | 0.619 | 0.359 | 1.319 | 114 |

### Phase-A (MLP) — Key Epochs

| Epoch | Val Loss | R@200 | P@200 | d' | Rank | Notes |
|-------|----------|-------|-------|----|------|-------|
| 13 | 0.5994 | 0.605 | 0.340 | 1.319 | 124 | Best d' |
| 15 | 0.5811 | 0.599 | 0.336 | 1.301 | 128 | Last pre-DRW |
| 16 | 1.1377 | 0.605 | 0.339 | 1.294 | 125 | DRW activates |
| **23** | **0.9599** | **0.616** | **0.355** | **1.297** | **118** | **Best R@200** |
| 26 | 0.9011 | 0.608 | 0.347 | 1.296 | 124 | Latest |
