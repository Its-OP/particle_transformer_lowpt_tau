# Cascade Model Diagnostic Analysis

**Generated:** 2026-03-30 21:46

---

## 1. R@K Sweep

| K | Stage 1 R@K | Regular R@K | Physics R@K | Delta (P-R) |
|---|-----------|------------|------------|-------------|
| 10 | 0.0833 | 0.1500 | 0.2250 | +0.0750 |
| 20 | 0.1167 | 0.2333 | 0.2583 | +0.0250 |
| 30 | 0.1750 | 0.2500 | 0.3208 | +0.0708 |
| 50 | 0.2333 | 0.3250 | 0.3875 | +0.0625 |
| 100 | 0.3625 | 0.3750 | 0.5250 | +0.1500 |
| 200 | 0.5500 | 0.5542 | 0.6417 | +0.0875 |
| 300 | 0.6667 | 0.6458 | 0.7167 | +0.0708 |
| 400 | 0.7125 | 0.7542 | 0.8125 | +0.0583 |
| 500 | 0.8042 | 0.8375 | 0.8625 | +0.0250 |
| 600 | 0.8875 | 0.8875 | 0.8875 | +0.0000 |

### Perfect recall (P@K)

| K | Stage 1 | Regular | Physics | Delta |
|---|---------|---------|---------|-------|
| 10 | 0.0000 | 0.0250 | 0.1000 | +0.0750 |
| 20 | 0.0000 | 0.0750 | 0.1250 | +0.0500 |
| 30 | 0.0250 | 0.0750 | 0.1500 | +0.0750 |
| 50 | 0.0750 | 0.1500 | 0.2250 | +0.0750 |
| 100 | 0.1250 | 0.1750 | 0.3250 | +0.1500 |
| 200 | 0.3500 | 0.2500 | 0.4250 | +0.1750 |
| 300 | 0.4000 | 0.3750 | 0.4750 | +0.1000 |
| 400 | 0.4500 | 0.5000 | 0.5250 | +0.0250 |
| 500 | 0.5750 | 0.5750 | 0.6250 | +0.0500 |
| 600 | 0.6750 | 0.6750 | 0.6750 | +0.0000 |

---

## 2. Score Distribution Analysis

| Metric | Stage 1 | Regular | Physics |
|--------|---------|---------|---------|
| d_prime | 0.996 | 0.775 | 1.100 |
| median_gt_rank | 178.000 | 150.000 | 82.000 |
| gt_rank_p75 | 466.000 | 390.000 | 322.000 |
| gt_rank_p90 | 656.000 | 613.000 | 613.000 |
| gt_rank_p95 | 866.000 | 1028.000 | 1028.000 |

### Stage 2 vs Stage 1 score correlation

- **Regular** — Pearson r (GT pion ranks): 0.841
- **Physics** — Pearson r (GT pion ranks): 0.788

---

## 3. Per-Event Breakdown at K=200

| Found | Stage 1 | Regular | Physics |
|-------|---------|---------|---------|
| 0/3 | 15.0% | 13.5% | 13.5% |
| 1/3 | 35.0% | 29.7% | 21.6% |
| 2/3 | 12.5% | 32.4% | 24.3% |
| 3/3 | 30.0% | 24.3% | 40.5% |

### "Just missed" GT pions (ranked 201+)

| Rank range | Regular | Physics |
|-----------|---------|---------|
| 201-300 | 11 (9.4%) | 9 (7.7%) |
| 301-500 | 22 (18.8%) | 17 (14.5%) |
| 500+ | 19 (16.2%) | 16 (13.7%) |

---

## 4. Failure Analysis by Physics Properties

### Found rate by pT bin

| pT range (GeV) | Regular | Physics | Delta |
|----------------|---------|---------|-------|
| [0, 0.3) | 0.524 (n=21) | 0.714 (n=21) | +0.190 |
| [0.3, 0.5) | 0.419 (n=31) | 0.419 (n=31) | +0.000 |
| [0.5, 1.0) | 0.618 (n=34) | 0.588 (n=34) | -0.029 |
| [1.0, 2.0) | 0.636 (n=22) | 0.909 (n=22) | +0.273 |
| [2.0, +) | 0.667 (n=9) | 0.778 (n=9) | +0.111 |

### Found rate by |dxy_sig| bin

| |dxy_sig| range | Regular | Physics | Delta |
|----------------|---------|---------|-------|
| [0, 0.5) | 0.526 (n=38) | 0.579 (n=38) | +0.053 |
| [0.5, 1.0) | 0.350 (n=20) | 0.500 (n=20) | +0.150 |
| [1.0, 2.0) | 0.560 (n=25) | 0.640 (n=25) | +0.080 |
| [2.0, 5.0) | 0.792 (n=24) | 0.833 (n=24) | +0.042 |
| [5.0, +) | 0.500 (n=10) | 0.700 (n=10) | +0.200 |

### Uncanny valley (pT 0.3-0.5 AND |dxy_sig| < 0.5)

- **Regular:** 3/12 = 25.0%
- **Physics:** 4/12 = 33.3%

---

## 5. Per-Pion-Role Analysis (3-GT events only)

| Role | Regular found rate | Physics found rate | Delta |
|------|-------------------|-------------------|-------|
| highest_pt | 0.541 (n=37) | 0.676 (n=37) | +0.135 |
| middle_pt | 0.595 (n=37) | 0.595 (n=37) | +0.000 |
| lowest_pt | 0.541 (n=37) | 0.649 (n=37) | +0.108 |

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
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 39/117 (33.3%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 11/117 (9.4%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 10/117 (8.5%)
- Net rescue: +1

### Physics
- GT pions dropped by Stage 1 (rank >= 600): 13/117 (11.1%)
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 29/117 (24.8%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 18/117 (15.4%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 7/117 (6.0%)
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
- Physics model finds MORE GT: 13 events (32.5%)
- Regular model finds MORE GT: 5 events (12.5%)
- Same performance: 22 events (55.0%)
- Net: physics wins +8 events

---

## Summary

| Metric | Stage 1 | Regular ParT | Physics ParT |
|--------|---------|-------------|-------------|
| R@200 | 0.5500 | 0.5542 | 0.6417 |
| P@200 | 0.3500 | 0.2500 | 0.4250 |
| d' | 0.996 | 0.775 | 1.100 |
| Median rank | 178 | 150 | 82 |
| GT pions | 117 | 117 | 117 |
| pair_extra_dim | — | 0 | 5 |
| embed_dim | — | 512 | 512 |
| num_layers | — | 8 | 8 |