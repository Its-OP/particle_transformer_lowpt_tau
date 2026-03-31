# Cascade Model Diagnostic Analysis

**Generated:** 2026-03-30 21:07

---

## 1. R@K Sweep

| K | Stage 1 R@K | Regular R@K | Physics R@K | Delta (P-R) |
|---|-----------|------------|------------|-------------|
| 10 | 0.2100 | 0.2652 | 0.2897 | +0.0245 |
| 20 | 0.2719 | 0.3328 | 0.3580 | +0.0253 |
| 30 | 0.3126 | 0.3745 | 0.3989 | +0.0244 |
| 50 | 0.3725 | 0.4332 | 0.4572 | +0.0240 |
| 100 | 0.4782 | 0.5338 | 0.5562 | +0.0224 |
| 200 | 0.6239 | 0.6676 | 0.6843 | +0.0167 |
| 300 | 0.7254 | 0.7589 | 0.7713 | +0.0124 |
| 400 | 0.8024 | 0.8253 | 0.8341 | +0.0087 |
| 500 | 0.8612 | 0.8732 | 0.8776 | +0.0044 |
| 600 | 0.9049 | 0.9049 | 0.9049 | +0.0000 |

### Perfect recall (P@K)

| K | Stage 1 | Regular | Physics | Delta |
|---|---------|---------|---------|-------|
| 10 | 0.0685 | 0.1104 | 0.1277 | +0.0173 |
| 20 | 0.1027 | 0.1584 | 0.1806 | +0.0222 |
| 30 | 0.1250 | 0.1876 | 0.2112 | +0.0236 |
| 50 | 0.1605 | 0.2303 | 0.2565 | +0.0262 |
| 100 | 0.2317 | 0.3102 | 0.3396 | +0.0295 |
| 200 | 0.3650 | 0.4400 | 0.4694 | +0.0293 |
| 300 | 0.4903 | 0.5550 | 0.5772 | +0.0222 |
| 400 | 0.6097 | 0.6540 | 0.6710 | +0.0170 |
| 500 | 0.7119 | 0.7358 | 0.7442 | +0.0083 |
| 600 | 0.7960 | 0.7960 | 0.7960 | +0.0000 |

---

## 2. Score Distribution Analysis

| Metric | Stage 1 | Regular | Physics |
|--------|---------|---------|---------|
| d_prime | 1.314 | 1.141 | 1.196 |
| median_gt_rank | 110.000 | 78.000 | 66.000 |
| gt_rank_p75 | 326.000 | 284.000 | 268.000 |
| gt_rank_p90 | 585.000 | 575.000 | 571.000 |
| gt_rank_p95 | 757.000 | 1028.000 | 1028.000 |

### Stage 2 vs Stage 1 score correlation

- **Regular** — Pearson r (GT pion ranks): 0.840
- **Physics** — Pearson r (GT pion ranks): 0.824

---

## 3. Per-Event Breakdown at K=200

| Found | Stage 1 | Regular | Physics |
|-------|---------|---------|---------|
| 0/3 | 9.7% | 10.7% | 10.3% |
| 1/3 | 18.9% | 18.7% | 17.4% |
| 2/3 | 24.9% | 26.6% | 25.1% |
| 3/3 | 30.3% | 44.0% | 47.1% |

### "Just missed" GT pions (ranked 201+)

| Rank range | Regular | Physics |
|-----------|---------|---------|
| 201-300 | 22152 (9.0%) | 21055 (8.6%) |
| 301-500 | 27662 (11.3%) | 25715 (10.5%) |
| 500+ | 30800 (12.6%) | 29732 (12.1%) |

---

## 4. Failure Analysis by Physics Properties

### Found rate by pT bin

| pT range (GeV) | Stage 1 | Regular | Physics |
|----------------|---------|---------|---------|
| [0, 0.3) | 0.675 (n=48049) | 0.710 | 0.705 |
| [0.3, 0.5) | 0.485 (n=60818) | 0.547 | 0.571 |
| [0.5, 1.0) | 0.584 (n=77044) | 0.629 | 0.656 |
| [1.0, 2.0) | 0.759 (n=42277) | 0.794 | 0.809 |
| [2.0, +) | 0.867 (n=16979) | 0.893 | 0.902 |

### Found rate by |dxy_sig| bin

| |dxy_sig| range | Stage 1 | Regular | Physics |
|----------------|---------|---------|---------|
| [0, 0.5) | 0.390 (n=68895) | 0.480 | 0.503 |
| [0.5, 1.0) | 0.472 (n=49666) | 0.553 | 0.578 |
| [1.0, 2.0) | 0.717 (n=51346) | 0.735 | 0.747 |
| [2.0, 5.0) | 0.924 (n=46122) | 0.923 | 0.919 |
| [5.0, +) | 0.823 (n=29138) | 0.815 | 0.844 |

### Uncanny valley (pT 0.3-0.5 AND |dxy_sig| < 0.5)

- **Stage 1:** 5370/20303 = 26.4%
- **Regular:** 7578/20303 = 37.3%
- **Physics:** 8277/20303 = 40.8%

---

## 5. Per-Pion-Role Analysis (3-GT events only)

| Role | Stage 1 | Regular | Physics |
|------|---------|---------|---------|
| highest_pt | 0.628 (n=72331) | 0.662 | 0.684 |
| middle_pt | 0.616 (n=72331) | 0.666 | 0.683 |
| lowest_pt | 0.660 (n=72331) | 0.710 | 0.723 |

Mean pT by role:
- Regular highest_pt: mean pT = 1.362 GeV
- Regular middle_pt: mean pT = 0.735 GeV
- Regular lowest_pt: mean pT = 0.407 GeV
- Physics highest_pt: mean pT = 1.362 GeV
- Physics middle_pt: mean pT = 0.735 GeV
- Physics lowest_pt: mean pT = 0.407 GeV

---

## 6. Stage 1 → Stage 2 Pipeline Analysis

### Regular
- GT pions dropped by Stage 1 (rank >= 600): 23175/245167 (9.5%)
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 57439/245167 (23.4%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 20205/245167 (8.2%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 9406/245167 (3.8%)
- Net rescue: +10799

### Physics
- GT pions dropped by Stage 1 (rank >= 600): 23175/245167 (9.5%)
- Passed Stage 1 but missed by Stage 2 (in top-600, not in top-200): 53327/245167 (21.8%)
- **Rescued** (S1 rank 200-600 → cascade top-200): 26244/245167 (10.7%)
- **Demoted** (S1 rank < 200 → cascade rank >= 200): 11333/245167 (4.6%)
- Net rescue: +14911

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

- Events compared: 86375
- Physics model finds MORE GT: 11900 events (13.8%)
- Regular model finds MORE GT: 8401 events (9.7%)
- Same performance: 66074 events (76.5%)
- Net: physics wins +3499 events

---

## Summary

| Metric | Stage 1 | Regular ParT | Physics ParT |
|--------|---------|-------------|-------------|
| R@200 | 0.6239 | 0.6676 | 0.6843 |
| P@200 | 0.3650 | 0.4400 | 0.4694 |
| d' | 1.314 | 1.141 | 1.196 |
| Median rank | 110 | 78 | 66 |
| GT pions | 245167 | 245167 | 245167 |
| pair_extra_dim | — | 0 | 5 |
| embed_dim | — | 512 | 512 |
| num_layers | — | 8 | 8 |