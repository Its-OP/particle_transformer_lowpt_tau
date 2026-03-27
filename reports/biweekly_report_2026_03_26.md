# Oleh Prostakov, Project Report

## TL;DR

We ran 8 experiments trying to break the R@200 = 0.62 wall on the pre-filter. None succeeded. The root cause we identified is that the kNN graph in fixed (eta, phi) coordinates connects each signal track to ~16 neighbors, of which ~15.9 are noise. No architecture or loss change can overcome this - the graph itself must change. Our plan forward is a multi-stage cascade model, with more-and-more compute-heavy models introduced at later stages.

---

## What Was Done

### Evaluation Infrastructure

Built extended evaluation metrics that go beyond the single R@200 number:
- R@K curves at K = 10, 20, 30, 50, 100, 200, 300, 400, 500, 600, 800
- Conditional recall by pT bin and |dxy_significance| bin
- Per-event breakdown (what fraction of events find 0/3, 1/3, 2/3, 3/3 pions)

Key finding from diagnostics: only **33% of events recover all 3 pions** in the top-200. The hardest regime is pT in [0.3, 0.5) GeV with |dxy_sig| < 0.5 - found rate drops to **24%**, and 47% of all ground-truth pions fall there.

### Experiments

We tried 8 directions. The chart below summarizes results:

| Experiment | R@200 | Outcome |
|---|---|---|
| MLP + kNN baseline | 0.62 | Reference |
| Extended features (16 inputs) | 0.625 | Marginal, not meaningful |
| ParticleNeXt backbone | 0.62 | No gain |
| ISAB (global attention) | 0.62 | No gain |
| Score propagation | 0.58 | Degraded |
| Pairwise LV (concat) | 0.24 | Collapsed |
| Pairwise LV (attn bias) | 0.39 | Collapsed |
| ASL loss | 0.41 | Degraded |
| Triplet scorer | -- | OOM (96 GB) |

### Why Each Experiment Failed

**ParticleNeXt & ISAB (no gain):** Both try to extract more signal from the kNN neighborhood -- attention aggregation, multi-scale neighbors, global context via inducing points. The problem is that kNN neighborhoods are **99.7% noise** (3 signal tracks among ~1100). Richer aggregation over noise neighbors still produces noise features.

**Score propagation (degraded):** Graph-based label smoothing assumes signal clusters spatially. But 72% of ground-truth pions have zero GT neighbors in their kNN. Smoothing dilutes their scores with background, making things worse.

**Pairwise LV features (collapsed):** Invariant mass and delta-R between track pairs carry strong physics signal -- but only between *signal-signal* pairs. In the kNN graph, pairwise features are computed between signal and noise, where they're meaningless. Worse, the autoencoder reconstruction task finds a shortcut through relational features, causing representation collapse.

**ASL loss (degraded):** Asymmetric loss with clip=0.05 zeros gradients from easy negatives. But ranking models need the *full score distribution* to learn ordering. Removing easy negatives destroys the ranking baseline.

**Triplet scorer (OOM):** Enumerating triplets from top-200 anchors produces ~100K candidates per anchor. At batch_size * 200 anchors, this creates billions of triplets - catastrophic memory blowup even on a 96 GB GPU.

### Root Cause: The Noise Neighborhood Trap

Our main insight from these 2 weeks: **all experiments hit the same R@200 ~ 0.62 wall because they all operate on the same fixed (eta, phi) kNN graph**. With 3 signal tracks among ~1100, any local neighborhood operation, regardless of how sophisticated, aggregates noise.

An XGBoost on per-track features alone (no graph) achieves R@200 = 0.589. The kNN graph adds only +0.03. Which confirms that the current graph topology is not helping much, and no modification to the operations *on* this graph will break through.

**To break the wall, we need to change *which* tracks interact, not *how* they interact.**

---

## What's Planned (Next 2 Weeks)

### Two-Stage Cascade Architecture

The most promising direction, supported by both our diagnostics and literature (ExaTrkX, ATLAS GNN4ITk, recommendation systems all use cascades):

```
Stage 1 (current model)      Stage 2 (new)
~1100 tracks ──> top-600 ──> pairwise reranker ──> top-200
R@600 ~ 0.90                 R@200 target: 0.80+
signal: 0.27%                signal: ~0.5% (2x enrichment)
```

At K=600 (the experimental threshold to achive recall of 0.9+), signal fraction doubles from 0.27% to ~0.5%. More importantly, **pairwise features become viable**: with 600 tracks, full pairwise attention is feasible (600^2 = 360K pairs), and signal-signal pairs are now frequent enough to provide learning signal.

We plan to incorporate a ParT-based model as a second stage re-ranker. In our prelimenary experiments, such a model manages to achieve recall of 0.8+, filtering down the set to just 200 tracks.
