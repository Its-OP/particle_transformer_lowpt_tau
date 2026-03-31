# Cascade Model Diagnostic Analysis

**Generated:** 2026-03-30 22:46

---

## 1. R@K Sweep

| K | Stage 1 R@K | Regular R@K | Physics R@K | Delta (P-R) |
|---|-----------|------------|------------|-------------|
| 10 | 0.0833 | 0.1500 | 0.2167 | +0.0667 |
| 20 | 0.1167 | 0.2250 | 0.2625 | +0.0375 |
| 30 | 0.1750 | 0.2500 | 0.2958 | +0.0458 |
| 50 | 0.2333 | 0.3000 | 0.3917 | +0.0917 |
| 100 | 0.3625 | 0.3750 | 0.5083 | +0.1333 |
| 200 | 0.5500 | 0.5250 | 0.6292 | +0.1042 |
| 300 | 0.6667 | 0.6792 | 0.7250 | +0.0458 |
| 400 | 0.7125 | 0.7625 | 0.8042 | +0.0417 |
| 500 | 0.8042 | 0.8208 | 0.8625 | +0.0417 |
| 600 | 0.8875 | 0.8875 | 0.8875 | +0.0000 |

### Perfect recall (P@K)

| K | Stage 1 | Regular | Physics | Delta |
|---|---------|---------|---------|-------|
| 10 | 0.0000 | 0.0250 | 0.1000 | +0.0750 |
| 20 | 0.0000 | 0.0250 | 0.1250 | +0.1000 |
| 30 | 0.0250 | 0.0750 | 0.1250 | +0.0500 |
| 50 | 0.0750 | 0.1500 | 0.2250 | +0.0750 |
| 100 | 0.1250 | 0.1500 | 0.3000 | +0.1500 |
| 200 | 0.3500 | 0.2500 | 0.3750 | +0.1250 |
| 300 | 0.4000 | 0.4250 | 0.4750 | +0.0500 |
| 400 | 0.4500 | 0.5000 | 0.5250 | +0.0250 |
| 500 | 0.5750 | 0.5250 | 0.6250 | +0.1000 |
| 600 | 0.6750 | 0.6750 | 0.6750 | +0.0000 |

---

## 2. Score Distribution Analysis

| Metric | Stage 1 | Regular | Physics |
|--------|---------|---------|---------|
| d_prime | 0.996 | 0.734 | 1.071 |
| median_gt_rank | 178.000 | 170.000 | 94.000 |
| gt_rank_p75 | 466.000 | 388.000 | 344.000 |
| gt_rank_p90 | 656.000 | 613.000 | 613.000 |
| gt_rank_p95 | 866.000 | 1028.000 | 1028.000 |

### Stage 2 vs Stage 1 score correlation

- **Regular** — Pearson r (GT pion ranks): 0.825
- **Physics** — Pearson r (GT pion ranks): 0.792

---

## 3. Per-Event Breakdown at K=200

| Found | Stage 1 | Regular | Physics |
|-------|---------|---------|---------|
| 0/3 | 15.0% | 18.9% | 10.8% |
| 1/3 | 35.0% | 27.0% | 24.3% |
| 2/3 | 12.5% | 27.0% | 27.0% |
| 3/3 | 30.0% | 27.0% | 37.8% |

### "Just missed" GT pions (ranked 201+)

| Rank range | Regular | Physics |
|-----------|---------|---------|
| 201-300 | 17 (14.5%) | 11 (9.4%) |
| 301-500 | 17 (14.5%) | 16 (13.7%) |
| 500+ | 21 (17.9%) | 16 (13.7%) |

---

## 4. Failure Analysis by Physics Properties

### Found rate by pT bin

| pT range (GeV) | Stage 1 | Regular | Physics |
|----------------|---------|---------|---------|
| [0, 0.3) | 0.714 (n=21) | 0.524 | 0.714 |
| [0.3, 0.5) | 0.452 (n=31) | 0.387 | 0.387 |
| [0.5, 1.0) | 0.382 (n=34) | 0.588 | 0.618 |
| [1.0, 2.0) | 0.682 (n=22) | 0.636 | 0.909 |
| [2.0, +) | 0.778 (n=9) | 0.556 | 0.667 |

### Found rate by |dxy_sig| bin

| |dxy_sig| range | Stage 1 | Regular | Physics |
|----------------|---------|---------|---------|
| [0, 0.5) | 0.474 (n=38) | 0.553 | 0.553 |
| [0.5, 1.0) | 0.300 (n=20) | 0.300 | 0.450 |
| [1.0, 2.0) | 0.600 (n=25) | 0.440 | 0.640 |
| [2.0, 5.0) | 0.833 (n=24) | 0.750 | 0.875 |
| [5.0, +) | 0.500 (n=10) | 0.600 | 0.700 |

### Uncanny valley (pT 0.3-0.5 AND |dxy_sig| < 0.5)

- **Stage 1:** 4/12 = 33.3%
- **Regular:** 4/12 = 33.3%
- **Physics:** 3/12 = 25.0%

---

## 5. Per-Pion-Role Analysis (3-GT events only)

| Role | Stage 1 | Regular | Physics |
|------|---------|---------|---------|
| highest_pt | 0.568 (n=37) | 0.541 | 0.649 |
| middle_pt | 0.514 (n=37) | 0.541 | 0.622 |
| lowest_pt | 0.541 (n=37) | 0.541 | 0.649 |

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
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 42/117 (35.9%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 9/117 (7.7%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 11/117 (9.4%)
- Net rescue: -2

### Physics
- GT pions dropped by Stage 1 (rank >= 600): 13/117 (11.1%)
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 30/117 (25.6%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 19/117 (16.2%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 9/117 (7.7%)
- Net rescue: +10

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
| R@200 | 0.5500 | 0.5250 | 0.6292 |
| P@200 | 0.3500 | 0.2500 | 0.3750 |
| d' | 0.996 | 0.734 | 1.071 |
| Median rank | 178 | 170 | 94 |
| GT pions | 117 | 117 | 117 |
| pair_extra_dim | — | 0 | 5 |
| embed_dim | — | 512 | 512 |
| num_layers | — | 8 | 8 |