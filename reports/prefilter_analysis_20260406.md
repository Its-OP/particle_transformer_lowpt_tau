# Prefilter Analysis — `prefilter_dim256_cutoff_bs192` (2026-04-06)

**Run ID:** `prefilter_dim256_cutoff_bs192_PreFilter_20260405_120532`
**Started:** 2026-04-05 12:05:32 | **Aborted:** epoch 75/100 (user, overfit plateau)
**Best checkpoint:** `models/debug_checkpoints/prefilter_dim256_cutoff_bs192_PreFilter_20260405_120532/checkpoints/best_model.pt` (epoch 36, val R@200 = 0.9190)
**Val subset for fresh inference:** `data/low-pt/subset/val/val_000.parquet` (7504 events, 22512 GT pions)

## TL;DR

- **Best epoch: 36 / 100**. Val R@200 = 0.9190, p90 = 180 (full 24k-event val).
  Fresh inference on the val subset reproduces this within noise: R@200 = 0.9149,
  R@256 = 0.9428, R@300 = 0.9584, p90 = 178, d′ = 2.010.
- **Overfitting confirmed.** Train R@200 continues climbing 0.932 → 0.957 while val
  R@200 drifts 0.919 → 0.911 between epochs 36 → 75. Gap widens from 0.013 → 0.046.
  Val d′ collapses from 2.04 (ep 36) → 1.49 (ep 75).
- **The metric jump from the old 0.628 baseline to 0.919 R@200 is NOT a model
  improvement — it is caused by the pT ≥ 0.5 GeV cutoff** applied to the dataset
  to match the ABSNet low-pt baseline. The cutoff removes the softest, hardest-to-
  reconstruct tracks from the problem. Prior numbers were measured on the uncut
  dataset and are not directly comparable.
- **R@600 > 0.995 throughout** — cascade K₁ = 600 is strongly covered. Cascade
  recall headroom is not the bottleneck.

## Dataset change (pT cutoff)

As of this run, the training and validation parquets apply a **track pT ≥ 0.5 GeV
cutoff**, matching the ABSNet low-pt baseline used as the reference paper. The
cutoff eliminates the hardest-to-reconstruct soft tracks — the ones that most
often fell outside the top-200 under the pre-cutoff data. Because the difficulty
distribution shifts, R@K on the cutoff dataset is systematically higher than R@K
on the pre-cutoff dataset for the same model. The `cutoff` token in the
experiment name (`prefilter_dim256_cutoff_bs192_*`) records this. Any comparison
against the historical R@200 ≈ 0.628 baseline (uncut dataset, from earlier
experiment_log entries) must account for this — those numbers belong to a
different problem.

Concretely: **the right comparison partners for this run are future runs on the
cutoff dataset, and the published ABSNet numbers**, not the older in-house
baselines.

## Training hyperparameters

From `training.log` line 2 (full args dict):

```
epochs            = 100
batch_size        = 192
lr                = 0.001
weight_decay      = 0.01
scheduler         = cosine
warmup_fraction   = 0.05
train_fraction    = 0.8
steps_per_epoch   = 500
grad_clip         = 1.0
amp               = True
no_compile        = False
keep_best_k       = 5
save_every        = 5
num_workers       = 10
device            = cuda:0
```

From `networks/lowpt_tau_TrackPreFilter.py` (model wrapper defaults):

```
mode                           = 'mlp'
input_dim                      = 16
hidden_dim                     = 256
num_neighbors                  = 16
num_message_rounds             = 2
ranking_num_samples            = 50
ranking_temperature_start/end  = 2.0 → 0.5
denoising_sigma_start/end      = 1.0 → 0.1
drw_warmup_fraction            = 0.3         # DRW activates at epoch 31
drw_positive_weight            = 2.0
```

Training data: full `data/low-pt/train/` (10 parquet files, pT ≥ 0.5 GeV cutoff).
Validation during training: full `data/low-pt/val/` (7 parquet files, 24000 events).

## Training dynamics

Sampled per-epoch metrics from `metrics/epoch_*.json`:

| Epoch | lr      | train_loss | val_loss | val R@200 | train R@200 | gap     | val d′ | val p90 | train p90 | val median |
|------:|--------:|-----------:|---------:|----------:|------------:|--------:|-------:|--------:|----------:|-----------:|
|     1 | 2.5e-4  |    0.5879  |  0.4493  |   0.8864  |    0.8869   | -0.0005 |  1.772 |    220  |     220   |        16  |
|     5 | 1.0e-3  |    0.3608  |  0.3518  |   0.9068  |    0.9105   | +0.0037 |  1.936 |    188  |     183   |         9  |
|    10 | 9.9e-4  |    0.3195  |  0.3231  |   0.9116  |    0.9158   | +0.0042 |  1.947 |    193  |     176   |         9  |
|    15 | 9.6e-4  |    0.2963  |  0.3035  |   0.9129  |    0.9183   | +0.0055 |  1.968 |    190  |     171   |         8  |
|    20 | 9.2e-4  |    0.2770  |  0.2880  |   0.9122  |    0.9206   | +0.0084 |  1.991 |    189  |     165   |         8  |
|    25 | 8.8e-4  |    0.2552  |  0.2675  |   0.9160  |    0.9224   | +0.0064 |  2.013 |    183  |     162   |         7  |
|    30 | 8.2e-4  |    0.2364  |  0.2520  |   0.9179  |    0.9287   | +0.0108 |  2.019 |    172  |     155 * |         7  |
|    31 | 8.0e-4  |    0.4773  |  0.5105  |   (DRW ON; loss jump)                                                 |
|    33 | 7.8e-4  |    0.4576  |  0.4883  |   0.9185  |    0.9301   | +0.0116 |  2.020 |    180  |     150   |         7  |
| **36** | 7.4e-4 |    0.4306  |  0.4703  | **0.9190**|    0.9317   | +0.0127 |**2.035** | **180** |    146   |         7  |
|    40 | 6.8e-4  |    0.4039  |  0.4673  |   0.9167  |    0.9322   | +0.0155 |  1.989 |    171  |     146   |         7  |
|    45 | 6.0e-4  |    0.3695  |  0.4403  |   0.9168  |    0.9361   | +0.0193 |  1.983 |    181  |     143   |         7  |
|    50 | 5.2e-4  |    0.3385  |  0.4294  |   0.9146  |    0.9406   | +0.0260 |  1.944 |    175  |     134   |         6  |
|    55 | 4.4e-4  |    0.3054  |  0.4132  |   0.9142  |    0.9427   | +0.0285 |  1.855 |    184  |     130   |         7  |
|    60 | 3.6e-4  |    0.2752  |  0.4060  |   0.9121  |    0.9461   | +0.0340 |  1.738 |    178  |     130   |         6  |
|    65 | 2.8e-4  |    0.2478  |  0.3922  |   0.9138  |    0.9497   | +0.0359 |  1.658 |    187  |     122   |         6  |
|    70 | 2.1e-4  |    0.2195  |  0.3855  |   0.9112  |    0.9498   | +0.0386 |  1.555 |    193  |     123   |         6  |
|    75 | 1.6e-4  |    0.1977  |  0.3762  |   0.9109  |    0.9570   | +0.0461 |  1.489 |    191  |     114   |         6  |

(*train p90 at epoch 30 estimated, actual value not read.)

### Three observations

**1. Peak val metrics are flat from ~ep 12 to ep 36.** Val R@200 moves only from
0.9117 to 0.9190 across 24 epochs — a 0.007 gain. The "best epoch 36" is really
a peak on a plateau, not a sharp maximum. The grep of `"new best"` in
`training.log` lists epochs 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 19, 22,
23, 30, 33, 36 — with monotonic but tiny improvements after epoch 12.

**2. DRW activation at epoch 31 causes a loss discontinuity.** With
`drw_warmup_fraction = 0.3` and 100 total epochs, DRW (Deferred Re-Weighting
with `drw_positive_weight = 2.0`) turns on at the start of epoch 31. The loss
jumps: train_loss = 0.2364 (ep 30) → 0.4773 (ep 31), val_loss = 0.2520 (ep 30)
→ 0.5105 (ep 31). This is purely from the re-weighted loss surface, not from
any real degradation — val R@200 barely moves (0.9179 → 0.919x). The val loss
is therefore not comparable before/after epoch 31, and the val-loss trajectory
should not be used to identify the best epoch (val R@200 should be).

**3. d′ peaks at epoch 36 and collapses afterward.** Val d′ grows from 1.77
(ep 1) → 2.04 (ep 36) in lockstep with the ranking improvement, then falls to
1.49 (ep 75). Train d′ follows the same arc (1.77 → 2.06 → 1.58). This is the
classic fingerprint of a temperature-annealing model that starts to saturate
its sigmoid: as the ranking temperature drops from 2.0 → ~0.9 (late in
training, see Schedule log lines), the scores are pushed toward tight extremes
and the GT/BG distributions *both* tighten, narrowing d′ even as the ordering
of the top few stays sharp. Median GT rank on train continues to fall (7 → 6)
while val median stays at 6–7 — the top is fine; the cost is distribution
sharpness.

## Best-model val-subset sweep (K sweep at epoch 36)

Fresh inference run on `data/low-pt/subset/val/val_000.parquet`
(7504 events, 22512 GT pions, `max_steps=469` to cover exactly 7504 events
without SimpleIterDataset cycling). Device: MPS. Batch size: 16.
`model.train()` mode (per the BatchNorm-stale-stats fix, see
`experiment_log.md` "BatchNorm Fix" entry).

| K       | R@K       | P@K       |
|--------:|----------:|----------:|
|      10 |   0.5416  |   0.3535  |
|      20 |   0.6357  |   0.4647  |
|      30 |   0.6882  |   0.5232  |
|      50 |   0.7523  |   0.5966  |
|     100 |   0.8324  |   0.6998  |
| **200** | **0.9149**| **0.8230**|
| **256** | **0.9428**| **0.8753**|
| **300** | **0.9584**| **0.9070**|
|     400 |   0.9798  |   0.9528  |
|     500 |   0.9900  |   0.9760  |
|     600 |   0.9961  |   0.9908  |
|     800 |   0.9996  |   0.9988  |

- **K = 200** is the ABSNet reference cut — R@200 = 0.9149 on the subset.
- **K = 256** (user request) lands midway between 200 and 300 at R@256 = 0.9428.
- **K = 300** is the smallest K where R@K reaches ~0.96 — a practical soft
  cascade target.
- **K = 600** is the current cascade K₁ and shows R@600 = 0.9961, confirming
  that virtually no GT pions are lost at that cut.

## Sanity check: subset vs training-time full val

At epoch 36 the training loop reported full-val metrics (24000 events from
`metrics/epoch_36.json`). Comparing against the subset-val sweep (7504 events):

| Metric          | Subset (7504 ev) | Training-time full (24000 ev) | Δ (subset − full) |
|-----------------|-----------------:|-----------------------------:|------------------:|
| R@100           |        0.8324   |         0.8423                |          −0.0099  |
| **R@200**       |      **0.9149** |       **0.9190**              |          −0.0041  |
| R@300           |        0.9584   |         0.9583                |          +0.0001  |
| R@500           |        0.9900   |         0.9916                |          −0.0016  |
| R@600           |        0.9961   |         0.9965                |          −0.0004  |
| d′              |        2.010    |         2.035                 |          −0.025   |
| Median GT rank  |        7.0      |         7.0                   |           0       |
| p75             |       48.0      |        47.0                   |          +1.0     |
| **p90**         |      **178.0**  |       **180.0**               |          **−2.0** |
| p95             |      274.0      |       289.0                   |         −15.0     |

All differences are within ±0.01 on R@K and within ±15 on the high-percentile
GT ranks. The subset is statistically representative of the full val population.
The sanity check passes — the best_model checkpoint reproduces its training-time
numbers under fresh inference.

## GT-rank percentiles

From the subset sweep (7504 events, 22512 GT pions):

| Percentile  | GT rank |
|-------------|--------:|
| Median (p50)|    7    |
| p75         |   48    |
| **p90**     | **178** |
| p95         |  274    |

Interpretation:
- Half of all GT pions land in the top 7 candidates — very aggressive top-of-list
  discrimination.
- 90% land in the top 178 — this is the practical floor for a soft cascade cut
  at R ≈ 0.90.
- 95% land in the top 274 — so K = 300 captures ~95% of GT pions.
- The remaining ~5% are buried deep: their ranks stretch out to hundreds or
  thousands, matching the `found_0_of_3_at_200 = 1.69%` perfect-miss rate from
  the JSON breakdown.

Per-event breakdown at K = 200 (subset):

| Outcome            | Fraction |
|--------------------|---------:|
| Found 3 / 3        |   82.3 % |
| Found 2 / 3        |   11.6 % |
| Found 1 / 3        |    4.4 % |
| Found 0 / 3        |    1.7 % |

The "0 found" events are a small but non-trivial tail. These 1.7% of events
contribute disproportionately to downstream cascade losses, since the reranker
has nothing to work with.

## Overfitting analysis

Three signals agree:

1. **Generalization gap widens monotonically from epoch 36 onward.**
   Train R@200 − val R@200 grows +0.013 (ep 36) → +0.046 (ep 75).

2. **p90 splits between train and val.** Train p90 falls monotonically from
   220 → 114 (memorization of training examples), while val p90 bottoms out
   at 172 (ep 30) and drifts back up to 191 (ep 75). This is the textbook
   overfit fingerprint on a percentile metric.

3. **d′ collapses on both sides** — not a classical overfit shape (which would
   show train d′ rising while val d′ falls). Both train and val d′ drop from
   their ~2.04 peak. Combined with the ranking-temperature annealing schedule
   (T: 2.0 → 0.879 across 75 epochs), this indicates the model is saturating
   its output range: the scores compress toward the logit extremes, tightening
   both distributions so that the pooled std-dev in the d′ formula grows
   faster than the mean separation. The model is "confident but wrong" on the
   tail, not "uncertain and accurate" as it was at epoch 36.

Root-cause ordering (most to least likely):
- **Over-annealed ranking temperature** (T = 2.0 → 0.879 by epoch 75). Late
  training overshoots the useful temperature range.
- **DRW positive up-weighting** applied from epoch 31 onward. Pushes training
  loss toward perfect-positive but allows val negatives to drift.
- **Cosine LR never reaches `min_lr`**. At epoch 75 LR is still 1.6e-4 —
  high enough to continue specializing on train but not high enough to escape
  local minima.
- **No early-stopping callback**. The loop runs to the fixed epoch budget
  unless manually aborted, which is exactly what happened at epoch 75.

## Recommendations

**Accept best_model.pt as the final Stage 1 artifact for the cutoff dataset.**
The subset sanity check confirms the checkpoint's numbers are real and
reproducible. For the current cascade experiments, this checkpoint can be
promoted to `models/prefilter_best.pt` (the cascade launch script expects
that path).

**Do NOT interpret 0.919 as a cascade breakthrough.** The big-picture
comparison should be:
- **against ABSNet (cutoff baseline)**, which is the reference the dataset was
  aligned to; and
- **against future cutoff-dataset cascade runs**.
The historical 0.628 baseline is on a different (harder) dataset and should
not be invoked in the comparison.

**For the next prefilter training run on the cutoff dataset**, the training
script should:
- Stop ranking-temperature annealing earlier (e.g., freeze at T = 1.0 for the
  last 30% of training instead of annealing to 0.5).
- Defer DRW activation (`drw_warmup_fraction = 0.4`) or reduce
  `drw_positive_weight` from 2.0 → 1.5 to soften the loss discontinuity.
- Add an early-stopping callback with patience ~15 epochs on val R@200 —
  this would have stopped training around epoch ~51 (36 + 15) and saved ~25%
  of compute.
- Optionally cap total epochs at 60: the val plateau is visible well before
  epoch 50 and the extra epochs only add overfit.

## Verification of the analysis

- Phase 2 smoke test (100 batches, 1600 events): R@200 = 0.9088 — within
  ±0.02 of 0.9190. ✓
- Phase 2 full run (469 batches, 7504 events): R@200 = 0.9149 — within ±0.01
  of 0.9190. ✓
- Subset p90 = 178 vs full p90 = 180 — within ±10. ✓
- All 12 K values present in the JSON sidecar. ✓
- Median GT rank, d′, p75/p90/p95 all within expected tolerance. ✓

## Appendix

**Diagnostic command used** (from `part/` working directory):

```bash
python diagnostics/_tmp_analyze_prefilter.py --max-steps 469
```

**JSON sidecar:** `reports/prefilter_analysis_20260406_subset_sweep.json`
(contains all 12 K values + percentiles + per-event breakdown).

**Dataset version:** cutoff dataset with pT ≥ 0.5 GeV (ABSNet baseline),
auto-preprocessing YAML `data/low-pt/lowpt_tau_trackfinder.339cebd0f2e564902a8ee0a01c35e17d.auto.yaml`
(hash from the training log preamble).

**Checkpoint provenance:** `checkpoints/best_model.pt` from run
`prefilter_dim256_cutoff_bs192_PreFilter_20260405_120532`, saved as the
epoch-36 val-R@200 peak (last `"new best"` entry in `training.log` line 1584,
timestamp 2026-04-05 16:36:55).
