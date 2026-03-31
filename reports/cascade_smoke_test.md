# Cascade Model Diagnostic Analysis

**Generated:** 2026-03-30 22:42

---

## 1. R@K Sweep

| K | Stage 1 R@K | Regular R@K | Physics R@K | Delta (P-R) |
|---|-----------|------------|------------|-------------|
| 10 | 0.0833 | 0.1417 | 0.2083 | +0.0667 |
| 20 | 0.1167 | 0.2000 | 0.2583 | +0.0583 |
| 30 | 0.1750 | 0.2250 | 0.3125 | +0.0875 |
| 50 | 0.2333 | 0.2833 | 0.3917 | +0.1083 |
| 100 | 0.3625 | 0.3875 | 0.5167 | +0.1292 |
| 200 | 0.5500 | 0.5708 | 0.6375 | +0.0667 |
| 300 | 0.6667 | 0.6583 | 0.7167 | +0.0583 |
| 400 | 0.7125 | 0.7625 | 0.8125 | +0.0500 |
| 500 | 0.8042 | 0.8292 | 0.8625 | +0.0333 |
| 600 | 0.8875 | 0.8875 | 0.8875 | +0.0000 |

### Perfect recall (P@K)

| K | Stage 1 | Regular | Physics | Delta |
|---|---------|---------|---------|-------|
| 10 | 0.0000 | 0.0250 | 0.1000 | +0.0750 |
| 20 | 0.0000 | 0.0250 | 0.1250 | +0.1000 |
| 30 | 0.0250 | 0.0500 | 0.1250 | +0.0750 |
| 50 | 0.0750 | 0.1250 | 0.2250 | +0.1000 |
| 100 | 0.1250 | 0.1750 | 0.3000 | +0.1250 |
| 200 | 0.3500 | 0.3000 | 0.4000 | +0.1000 |
| 300 | 0.4000 | 0.4500 | 0.4750 | +0.0250 |
| 400 | 0.4500 | 0.5250 | 0.5500 | +0.0250 |
| 500 | 0.5750 | 0.6000 | 0.6250 | +0.0250 |
| 600 | 0.6750 | 0.6750 | 0.6750 | +0.0000 |

---

## 2. Score Distribution Analysis

| Metric | Stage 1 | Regular | Physics |
|--------|---------|---------|---------|
| d_prime | 0.996 | 0.736 | 1.067 |
| median_gt_rank | 178.000 | 164.000 | 86.000 |
| gt_rank_p75 | 466.000 | 381.000 | 323.000 |
| gt_rank_p90 | 656.000 | 613.000 | 613.000 |
| gt_rank_p95 | 866.000 | 1028.000 | 1028.000 |

### Stage 2 vs Stage 1 score correlation

- **Regular** — Pearson r (GT pion ranks): 0.832
- **Physics** — Pearson r (GT pion ranks): 0.790

---

## 3. Per-Event Breakdown at K=200

| Found | Stage 1 | Regular | Physics |
|-------|---------|---------|---------|
| 0/3 | 15.0% | 16.2% | 10.8% |
| 1/3 | 35.0% | 24.3% | 24.3% |
| 2/3 | 12.5% | 29.7% | 24.3% |
| 3/3 | 30.0% | 29.7% | 40.5% |

### "Just missed" GT pions (ranked 201+)

| Rank range | Regular | Physics |
|-----------|---------|---------|
| 201-300 | 10 (8.5%) | 9 (7.7%) |
| 301-500 | 20 (17.1%) | 17 (14.5%) |
| 500+ | 20 (17.1%) | 16 (13.7%) |

---

## 4. Failure Analysis by Physics Properties

### Found rate by pT bin

| pT range (GeV) | Stage 1 | Regular | Physics |
|----------------|---------|---------|---------|
| [0, 0.3) | 0.714 (n=21) | 0.571 | 0.714 |
| [0.3, 0.5) | 0.452 (n=31) | 0.387 | 0.452 |
| [0.5, 1.0) | 0.382 (n=34) | 0.676 | 0.588 |
| [1.0, 2.0) | 0.682 (n=22) | 0.636 | 0.909 |
| [2.0, +) | 0.778 (n=9) | 0.667 | 0.667 |

### Found rate by |dxy_sig| bin

| |dxy_sig| range | Stage 1 | Regular | Physics |
|----------------|---------|---------|---------|
| [0, 0.5) | 0.474 (n=38) | 0.553 | 0.605 |
| [0.5, 1.0) | 0.300 (n=20) | 0.350 | 0.450 |
| [1.0, 2.0) | 0.600 (n=25) | 0.520 | 0.720 |
| [2.0, 5.0) | 0.833 (n=24) | 0.833 | 0.750 |
| [5.0, +) | 0.500 (n=10) | 0.600 | 0.700 |

### Uncanny valley (pT 0.3-0.5 AND |dxy_sig| < 0.5)

- **Stage 1:** 4/12 = 33.3%
- **Regular:** 3/12 = 25.0%
- **Physics:** 5/12 = 41.7%

---

## 5. Per-Pion-Role Analysis (3-GT events only)

| Role | Stage 1 | Regular | Physics |
|------|---------|---------|---------|
| highest_pt | 0.568 (n=37) | 0.541 | 0.676 |
| middle_pt | 0.514 (n=37) | 0.622 | 0.622 |
| lowest_pt | 0.541 (n=37) | 0.568 | 0.649 |

Mean pT by role:
- Regular highest_pt: mean pT = 1.403 GeV
- Regular middle_pt: mean pT = 0.725 GeV
- Regular lowest_pt: mean pT = 0.375 GeV
- Physics highest_pt: mean pT = 1.403 GeV
- Physics middle_pt: mean pT = 0.725 GeV
- Physics lowest_pt: mean pT = 0.375 GeV

---

## 6. Stage 1 → Stage 2 Pipeline Analysis

### Regular
- GT pions dropped by Stage 1 (rank >= 600): 13/117 (11.1%)
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 37/117 (31.6%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 11/117 (9.4%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 8/117 (6.8%)
- Net rescue: +3

### Physics
- GT pions dropped by Stage 1 (rank >= 600): 13/117 (11.1%)
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 29/117 (24.8%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 19/117 (16.2%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 8/117 (6.8%)
- Net rescue: +11

---

## 7. Learning Dynamics

### Regular
- Best R@200: 0.6761 at epoch 65
- Final R@200: 0.6707 (epoch 100)
- Total epochs: 100
- Final val loss: 0.4425
- Train-val loss gap (last epoch): +0.0071
- Final d': 1.139

### Physics
- Best R@200: 0.6932 at epoch 69
- Final R@200: 0.6881 (epoch 71)
- Total epochs: 71
- Final val loss: 0.4175
- Train-val loss gap (last epoch): +0.0040
- Final d': 1.169

---

## 8. Head-to-Head Event Comparison

- Events compared: 40
- Physics model finds MORE GT: 14 events (35.0%)
- Regular model finds MORE GT: 9 events (22.5%)
- Same performance: 17 events (42.5%)
- Net: physics wins +5 events

---

## Summary

| Metric | Stage 1 | Regular ParT | Physics ParT |
|--------|---------|-------------|-------------|
| R@200 | 0.5500 | 0.5708 | 0.6375 |
| P@200 | 0.3500 | 0.3000 | 0.4000 |
| d' | 0.996 | 0.736 | 1.067 |
| Median rank | 178 | 164 | 86 |
| GT pions | 117 | 117 | 117 |
| pair_extra_dim | — | 0 | 5 |
| embed_dim | — | 512 | 512 |
| num_layers | — | 8 | 8 |