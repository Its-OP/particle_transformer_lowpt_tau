# Duplet and Triplet Recall Analysis

**Date:** 2026-03-31
**Dataset:** Full validation set (72,331 events with exactly 3 GT pions, 86,375 total events)
**Models:** Stage 1 (TrackPreFilter), Regular ParT cascade, Physics ParT cascade (+5 pairwise features)

---

## 1. Duplet and Triplet Recall at Various K

### Stage 1

| K | R@K | >=1/3 | >=2/3 | 3/3 | miss all |
|---|-----|-------|-------|-----|----------|
| 30 | 0.3256 | 0.5508 | 0.2991 | 0.1269 | 0.4492 |
| 50 | 0.3856 | 0.6322 | 0.3629 | 0.1618 | 0.3678 |
| 100 | 0.4907 | 0.7594 | 0.4821 | 0.2306 | 0.2406 |
| 150 | 0.5697 | 0.8361 | 0.5786 | 0.2945 | 0.1639 |
| 200 | 0.6347 | 0.8841 | 0.6585 | 0.3613 | 0.1159 |
| 300 | 0.7336 | 0.9383 | 0.7773 | 0.4853 | 0.0617 |
| 400 | 0.8085 | 0.9659 | 0.8547 | 0.6051 | 0.0341 |
| 500 | 0.8652 | 0.9796 | 0.9083 | 0.7076 | 0.0204 |
| 600 | 0.9076 | 0.9878 | 0.9423 | 0.7926 | 0.0122 |

### Regular ParT

| K | R@K | >=1/3 | >=2/3 | 3/3 | miss all |
|---|-----|-------|-------|-----|----------|
| 30 | 0.3904 | 0.5911 | 0.3883 | 0.1918 | 0.4089 |
| 50 | 0.4488 | 0.6632 | 0.4487 | 0.2344 | 0.3368 |
| 100 | 0.5481 | 0.7773 | 0.5542 | 0.3129 | 0.2227 |
| 150 | 0.6213 | 0.8475 | 0.6381 | 0.3781 | 0.1525 |
| 200 | 0.6794 | 0.8926 | 0.7058 | 0.4400 | 0.1074 |
| 300 | 0.7678 | 0.9441 | 0.8063 | 0.5530 | 0.0559 |
| 400 | 0.8317 | 0.9693 | 0.8743 | 0.6515 | 0.0307 |
| 500 | 0.8776 | 0.9818 | 0.9182 | 0.7329 | 0.0182 |
| 600 | 0.9076 | 0.9878 | 0.9423 | 0.7926 | 0.0122 |

### Physics ParT

| K | R@K | >=1/3 | >=2/3 | 3/3 | miss all |
|---|-----|-------|-------|-----|----------|
| 30 | 0.4153 | 0.6102 | 0.4201 | 0.2157 | 0.3898 |
| 50 | 0.4733 | 0.6806 | 0.4784 | 0.2608 | 0.3194 |
| 100 | 0.5707 | 0.7894 | 0.5802 | 0.3425 | 0.2106 |
| 150 | 0.6410 | 0.8537 | 0.6603 | 0.4089 | 0.1463 |
| 200 | 0.6965 | 0.8966 | 0.7221 | 0.4708 | 0.1034 |
| 300 | 0.7801 | 0.9462 | 0.8179 | 0.5761 | 0.0538 |
| 400 | 0.8401 | 0.9712 | 0.8808 | 0.6683 | 0.0288 |
| 500 | 0.8821 | 0.9828 | 0.9217 | 0.7416 | 0.0172 |
| 600 | 0.9076 | 0.9878 | 0.9423 | 0.7926 | 0.0122 |

---

## 2. Duplet Analysis at K=200

| Model | >=2/3 (duplet) | 3/3 (triplet) | 1/3 only | 0/3 |
|-------|---------------|---------------|----------|-----|
| Stage 1 | 47,632 (65.9%) | 26,135 (36.1%) | 16,319 (22.6%) | 8,380 (11.6%) |
| Regular ParT | 51,048 (70.6%) | 31,824 (44.0%) | 13,515 (18.7%) | 7,768 (10.7%) |
| **Physics ParT** | **52,233 (72.2%)** | **34,052 (47.1%)** | **12,619 (17.4%)** | **7,479 (10.3%)** |

Physics ParT finds a duplet in **72.2%** of events — 6.3pp better than Stage 1 alone.

---

## 3. Duplet Rate (>=2/3 found) vs K

| K | Stage 1 | Regular | Physics |
|---|---------|---------|---------|
| 30 | 0.2991 | 0.3883 | 0.4201 |
| 50 | 0.3629 | 0.4487 | 0.4784 |
| 100 | 0.4821 | 0.5542 | 0.5802 |
| 150 | 0.5786 | 0.6381 | 0.6603 |
| 200 | 0.6585 | 0.7058 | 0.7221 |
| 300 | 0.7773 | 0.8063 | 0.8179 |
| 400 | 0.8547 | 0.8743 | 0.8808 |
| 500 | 0.9083 | 0.9182 | 0.9217 |
| 600 | 0.9423 | 0.9423 | 0.9423 |

At K=300, **82% of events** have at least 2 GT pions found (Physics ParT).
All three models converge at K=600 (94.2%) — the Stage 1 ceiling.

---

## 4. The Missing Third Pion

In events where exactly 2/3 GT pions are found at K=200:

### Distribution by pion role

| Model | Events | Highest pT missed | Middle pT missed | Lowest pT missed |
|-------|--------|-------------------|-----------------|-----------------|
| Stage 1 | 21,497 | 31.4% | 34.3% | 34.3% |
| Regular ParT | 19,224 | 33.4% | 32.7% | 33.8% |
| Physics ParT | 18,181 | 32.3% | 33.8% | 33.8% |

The missing pion is **spread evenly** across all three roles — it is NOT
systematically the lowest-pT (bachelor) pion. Any of the three can be missed.

### Properties of the missing third pion

| Property | Stage 1 | Regular | Physics |
|----------|---------|---------|---------|
| Mean pT | 0.627 GeV | 0.612 GeV | 0.602 GeV |
| Median pT | 0.486 GeV | 0.478 GeV | 0.472 GeV |
| Mean \|dxy_sig\| | 0.983 | 1.080 | 1.047 |
| Median \|dxy_sig\| | 0.535 | 0.559 | 0.555 |
| Mean rank | 413 | 647 | 685 |
| Median rank | 359 | 362 | 367 |

The missing pion typically has:
- **Moderate pT** (~0.47 GeV median) — squarely in the "uncanny valley"
- **Marginal displacement** (~0.55 median |dxy_sig|)
- **Rank 360-370** — just outside top-200 but well within top-500

### Implication for combinatorial search

The median rank of 367 means the missing third pion is typically ranked
300-500. A post-hoc search among tracks ranked 200-500 using physics
constraints (rho mass window, charge conservation, tau mass bound) could
recover it. The search space would be ~300 candidates per event, constrained
by the known properties of the found duplet (their invariant mass, charges,
and vertex position).
