# Combinatorial Triplet Feasibility Experiment

**Generated:** 2026-03-31 15:46
**Events:** 150 (3-GT only)
**Model:** Physics ParT cascade (pair_extra_dim=5, concat mode)

---

## K = 200

GT triplet fully in top-200: 56/150 (37.3%)

### Filter Cascade

| Stage | Filter | Survivors (mean) | Survivors (median) | Survivors (P90) | GT survival |
|-------|--------|-----------------|-------------------|----------------|-------------|
| F1 | F1: Charge (|Q|=1) | 983137 | 986436 | 989901 | 100.0% |
| F2 | F2: Tau mass (m3pi < 1.777) | 591421 | 607496 | 740665 | 100.0% |
| F3 | F3: a1 window (0.6 < m3pi < 1.5) | 444972 | 458716 | 593411 | 96.4% |
| F4 | F4: Rho resonance (OS pair near 770) | 270728 | 278653 | 348161 | 66.1% |
| D1 | D1: Dalitz boundary | 270728 | 278653 | 348161 | 66.1% |
| D2 | D2: Rho band structure (s_high > s_low) | 235554 | 241333 | 303396 | 53.6% |
| D3 | D3: Bachelor energy (250-500 MeV in a1 frame) | 150255 | 155078 | 190690 | 37.5% |
| D4 | D4: Rho helicity (|cos theta| > 0.3) | 116355 | 119684 | 146236 | 26.8% |

### GT Triplet Loss Analysis

| Killed at | Count | Fraction |
|-----------|-------|----------|
| not_in_topk | 94 | 62.7% |
| F3 | 2 | 1.3% |
| F4 | 17 | 11.3% |
| D2 | 7 | 4.7% |
| D3 | 9 | 6.0% |
| D4 | 6 | 4.0% |
| survived | 15 | 10.0% |

### Purity (events where GT triplet survives all filters)

- Events: 15
- Final survivors per event: mean=130132, median=133748, P90=147320
- Purity (1/n_survivors): mean=0.0000, median=0.0000


## K = 300

GT triplet fully in top-300: 75/150 (50.0%)

### Filter Cascade

| Stage | Filter | Survivors (mean) | Survivors (median) | Survivors (P90) | GT survival |
|-------|--------|-----------------|-------------------|----------------|-------------|
| F1 | F1: Charge (|Q|=1) | 3339042 | 3348775 | 3352351 | 100.0% |
| F2 | F2: Tau mass (m3pi < 1.777) | 1962774 | 1977138 | 2461608 | 100.0% |
| F3 | F3: a1 window (0.6 < m3pi < 1.5) | 1446203 | 1448812 | 1939732 | 96.0% |
| F4 | F4: Rho resonance (OS pair near 770) | 914889 | 918022 | 1184325 | 60.0% |
| D1 | D1: Dalitz boundary | 914889 | 918022 | 1184325 | 60.0% |
| D2 | D2: Rho band structure (s_high > s_low) | 792069 | 797268 | 1030853 | 48.0% |
| D3 | D3: Bachelor energy (250-500 MeV in a1 frame) | 517552 | 522396 | 669620 | 33.3% |
| D4 | D4: Rho helicity (|cos theta| > 0.3) | 399618 | 403782 | 513885 | 25.3% |

### GT Triplet Loss Analysis

| Killed at | Count | Fraction |
|-----------|-------|----------|
| not_in_topk | 75 | 50.0% |
| F3 | 3 | 2.0% |
| F4 | 27 | 18.0% |
| D2 | 9 | 6.0% |
| D3 | 11 | 7.3% |
| D4 | 6 | 4.0% |
| survived | 19 | 12.7% |

### Purity (events where GT triplet survives all filters)

- Events: 19
- Final survivors per event: mean=440116, median=452733, P90=532685
- Purity (1/n_survivors): mean=0.0000, median=0.0000
