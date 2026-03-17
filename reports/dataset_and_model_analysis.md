# Dataset & Model Diagnostic Analysis

**Date**: 2026-03-17
**Dataset**: `batch1.parquet` (241,400 events), subset `lowpt_tau_trackorigin.parquet` (19,093 events)
**Model analyzed**: TauTrackFinderV2 (best checkpoint from `trackfinder_simple_improved`)

---

## 1. Full Dataset Statistics (241K events)

### Track Counts
| Range | Count | Fraction |
|-------|------:|-------:|
| [0, 100) | 2 | 0.0% |
| [100, 500) | 1,699 | 0.7% |
| [500, 800) | 18,613 | 7.7% |
| [800, 1100) | 55,752 | 23.1% |
| [1100, 1400) | 79,369 | 32.9% |
| [1400, 1700) | 58,070 | 24.1% |
| [1700, 2000) | 22,191 | 9.2% |
| [2000, 3000) | 5,698 | 2.4% |

- min=51, max=3380, **mean=1278**, median=1269, std=350

### GT Tracks Per Event
| n_gt | Count | Fraction |
|-----:|------:|-------:|
| 0 | 53 | 0.0% |
| 1 | 2,779 | 1.2% |
| 2 | 35,332 | 14.6% |
| **3** | **201,897** | **83.6%** |
| 4 | 965 | 0.4% |
| 5 | 134 | 0.1% |
| 6+ | 240 | 0.1% |

**Signal ratio**: ~3 GT pions per ~1278 tracks = **0.23% positive rate**

---

## 2. Feature Distributions

### GT Pion pT (GeV) — 685,108 total GT pions
| Range (GeV) | Count | Fraction |
|-------------|------:|-------:|
| [0.0, 0.3) | 134,120 | 19.6% |
| [0.3, 0.5) | 170,614 | 24.9% |
| [0.5, 1.0) | 214,302 | 31.3% |
| [1.0, 2.0) | 118,894 | 17.4% |
| [2.0, 3.0) | 29,742 | 4.3% |
| [3.0, 5.0) | 13,645 | 2.0% |
| [5.0, 10.0) | 3,471 | 0.5% |
| [10.0, 50.0) | 320 | 0.0% |

- mean=0.819, **median=0.557**, std=0.846
- **75% of GT pions have pT < 1 GeV**

### Background pT (GeV) — sampled 50K
- mean=0.736, **median=0.544**, std=2.588
- Nearly identical to GT pion pT at low values — **pT alone cannot discriminate signal from background**

### GT Pion dxy_significance — 685,108 total
| Range | Count | Fraction |
|-------|------:|-------:|
| [-10, -3) | 63,997 | 9.3% |
| [-3, -1) | 109,298 | 16.0% |
| [-1, 0) | 148,283 | 21.6% |
| [0, 1) | 149,624 | 21.8% |
| [1, 3) | 110,445 | 16.1% |
| [3, 5) | 35,152 | 5.1% |
| [5, 10) | 30,125 | 4.4% |
| [10, 50) | 18,275 | 2.7% |

- mean=0.004, median=0.012, std=12.1
- **12.2% of GT pions have |dxy_sig| > 5** (heavy tails)

### Background dxy_significance — sampled 50K
| Range | Count | Fraction |
|-------|------:|-------:|
| [-10, -3) | 1,443 | 2.9% |
| [-3, -1) | 6,175 | 12.3% |
| [-1, 0) | 15,732 | 31.5% |
| [0, 1) | 15,584 | 31.2% |
| [1, 3) | 6,447 | 12.9% |
| [3, 5) | 728 | 1.5% |
| [5, 10) | 714 | 1.4% |
| [10, 50) | 1,182 | 2.4% |

- Background sharply peaked at [-1, 1] (**62.7%**) vs GT pions at 43.4%
- **dxy_significance is the strongest single discriminant** — GT pions have much heavier tails

### GT Pion Charge
- charge=-1: 335,797 (49.0%)
- charge=+1: 349,311 (51.0%)
- Balanced — charge alone is not discriminating

---

## 3. Spatial Structure

### 3-Prong Tau: Max Pairwise deltaR Among GT Pions (201,897 events)
| deltaR Range | Count | Fraction |
|-------------|------:|-------:|
| [0.0, 0.2) | 1,421 | 0.7% |
| [0.2, 0.5) | 17,264 | 8.6% |
| [0.5, 1.0) | 52,723 | 26.1% |
| [1.0, 2.0) | 76,173 | 37.7% |
| [2.0, 3.0) | 40,660 | 20.1% |
| [3.0, 5.0) | 13,654 | 6.8% |

- mean=1.492, **median=1.303**, std=0.850
- **27% of events have deltaR > 2.0** — GT pions can be on opposite sides of the detector
- Only 9.3% have deltaR < 0.5 (collimated)

### Background Density Near GT Pions (sampled 500 events)
| deltaR Radius | Mean BG Tracks | Median | Max |
|--------------|------:|------:|------:|
| < 0.1 | 1.1 | 1 | 7 |
| < 0.3 | 10.2 | 10 | 26 |
| < 0.5 | 28.7 | 28 | 67 |
| < 1.0 | 114.0 | 112 | 226 |

- Within deltaR < 0.3 of any GT pion: **~10 background tracks** (10:1 local noise)
- Within deltaR < 1.0: **~114 background tracks** (114:1 local noise)

---

## 4. V2 Model Diagnostic (200 events from subset, best checkpoint)

### Recall@K
| Metric | Mean | Median | Std | Min | Max |
|--------|-----:|-------:|----:|----:|----:|
| R@10 | 0.074 | 0.000 | 0.169 | 0.000 | 0.667 |
| R@20 | 0.117 | 0.000 | 0.206 | 0.000 | 1.000 |
| R@30 | 0.146 | 0.000 | 0.225 | 0.000 | 1.000 |
| R@50 | 0.204 | 0.000 | 0.268 | 0.000 | 1.000 |
| R@100 | 0.271 | 0.333 | 0.297 | 0.000 | 1.000 |

### Per-Event Recall Distribution (R@30)
| Recall | Events | Fraction |
|--------|-------:|-------:|
| 0% (0/3) | 132 | **66.3%** |
| 33% (1/3) | 45 | 22.6% |
| 67% (2/3) | 20 | 10.1% |
| 100% (3/3) | 2 | **1.0%** |

**66% of events find NONE of the 3 GT pions in top-30.**

### GT Pion Rank Distribution (560 GT pions)
| Rank Range | Count | Fraction |
|-----------|------:|-------:|
| [0, 10) | 42 | 7.5% |
| [10, 20) | 24 | 4.3% |
| [20, 30) | 16 | 2.9% |
| [30, 50) | 34 | 6.1% |
| [50, 100) | 39 | 7.0% |
| [100, 500) | 227 | **40.5%** |
| [500, 3000) | 178 | **31.8%** |

- mean=386, **median=292**, p75=605, p90=913
- GT pions are buried deep in the ranking

### Score Distributions
| | Mean | Median | Std |
|---|-----:|-------:|----:|
| GT pion scores | -1.632 | -1.633 | 0.240 |
| Background scores | -1.815 | -1.795 | 0.236 |
| **d-prime** | **0.769** | | |

Score separation is very weak (d-prime < 1).

### R@30 vs Event Track Count
| Track Count | n Events | R@30 |
|------------|-------:|-----:|
| [0, 500) | 2 | 0.167 |
| [500, 800) | 15 | 0.167 |
| [800, 1100) | 47 | 0.121 |
| [1100, 1400) | 59 | 0.130 |
| [1400, 1700) | 51 | 0.150 |
| [1700, 3000) | 25 | 0.207 |

Recall is uniformly poor across all multiplicities.

### GT Pion Found in Top-30 vs Pion pT
| pT Range (GeV) | n Pions | Found Rate |
|----------------|-------:|----------:|
| [0.0, 0.5) | 259 | 8.5% |
| [0.5, 1.0) | 181 | 13.8% |
| [1.0, 2.0) | 79 | 27.9% |
| [2.0, 3.0) | 22 | 31.8% |
| [3.0, 5.0) | 14 | 28.6% |
| [5.0, 10.0) | 5 | 40.0% |

Low-pT pions (which dominate the dataset) are nearly invisible to the model.

### GT Pion Found in Top-30 vs dxy_significance (standardized)
| dxy_sig Range | n Pions | Found Rate |
|--------------|-------:|----------:|
| [-3, -1) | 67 | 19.4% |
| [-1, 0) | 138 | **0.7%** |
| [0, 1) | 148 | 2.7% |
| [1, 2) | 68 | 7.4% |
| [2, 3) | 31 | 32.3% |
| [3, 5) | 24 | **50.0%** |
| [5, 20) | 26 | 30.8% |

**dxy_significance is the strongest discriminant**: pions with |dxy_sig| > 3 have 30-50% found rate vs <3% for non-displaced pions.

---

## 5. Key Takeaways

1. **Signal ratio is extreme**: 3 pions in ~1278 tracks = 0.23% positive rate
2. **pT distributions overlap almost completely** between signal and background — pT alone cannot discriminate
3. **dxy_significance is the strongest feature** — GT pions have heavy tails at |dxy_sig| > 3 where background is sparse
4. **GT pions are spatially spread**: median max-deltaR = 1.3 among 3 pions, with 27% spread over deltaR > 2
5. **Local noise is severe**: ~10 background tracks within deltaR < 0.3 of each GT pion
6. **V2 model barely separates signal from background**: d-prime = 0.77, median GT pion rank = 292
7. **The frozen backbone dilutes signal features**: enrichment via 5 EdgeConv layers with k=32 overwrites per-track identity with neighbor-averaged context, washing out dxy_significance and other discriminants
