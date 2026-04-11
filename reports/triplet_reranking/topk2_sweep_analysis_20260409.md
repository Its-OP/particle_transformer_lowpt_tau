# top_k2 Sweep Analysis

**Sweep root:** `models/debug_checkpoints/topk2_sweep_20260408_205250`
**Generated:** 2026-04-09 09:53
**Runs:** 9 OK, 1 failed, 10 total

## TL;DR

- **Best K by stability band:** top_k2=90 with C@100c = 0.8262 +/- 0.0077
- **Plateau:** C@100c varies by only 0.0556 across K=50-175 — the reranker is largely insensitive to input pool size.
- **Still improving at epoch 50:** K=50, K=60, K=70, K=80, K=90, K=100 — more epochs may yield marginal gains.
- **Failed:** K=200 (OOM at batch=96).

## 1. Convergence dynamics

| K | train (first->last) | val (first->last) | train-val gap | C@100 (first->best) | best_ep | converged? |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.1982 -> 0.0863 | 0.1273 -> 0.0847 | +0.0016 | 0.7758 -> 0.8300 | 48 | no |
| 60 | 0.2016 -> 0.0774 | 0.1168 -> 0.0780 | -0.0005 | 0.7825 -> 0.8292 | 48 | no |
| 70 | 0.1858 -> 0.0736 | 0.1062 -> 0.0678 | +0.0058 | 0.7746 -> 0.8300 | 31 | no |
| 80 | 0.1765 -> 0.0648 | 0.1009 -> 0.0651 | -0.0003 | 0.7783 -> 0.8379 | 48 | no |
| 90 | 0.1542 -> 0.0588 | 0.0931 -> 0.0631 | -0.0043 | 0.7804 -> 0.8358 | 48 | no |
| 100 | 0.1542 -> 0.0582 | 0.0906 -> 0.0589 | -0.0008 | 0.7704 -> 0.8304 | 28 | no |
| 125 | 0.1653 -> 0.0540 | 0.0853 -> 0.0534 | +0.0007 | 0.7708 -> 0.8308 | 48 | yes |
| 150 | 0.1479 -> 0.0505 | 0.0801 -> nan | +nan | 0.7679 -> 0.8350 | 48 | yes |
| 175 | 0.1416 -> 0.0469 | nan -> nan | +nan | 0.7600 -> 0.8100 | 36 | yes |

## 2. Stability analysis (last 5 epochs)

| K | C@100 best | C@100 mean(5) | C@100 std(5) | RC@100 mean(5) | RC@100 std(5) |
|---:|---:|---:|---:|---:|---:|
| 50 | 0.8300 | 0.8208 | 0.0083 | 0.7853 | 0.0058 |
| 60 | 0.8292 | 0.8220 | 0.0065 | 0.7866 | 0.0049 |
| 70 | 0.8300 | 0.8207 | 0.0063 | 0.7857 | 0.0046 |
| 80 | 0.8379 | 0.8248 | 0.0092 | 0.7883 | 0.0059 |
| 90 | 0.8358 | 0.8262 | 0.0077 | 0.7908 | 0.0054 |
| 100 | 0.8304 | 0.8223 | 0.0057 | 0.7869 | 0.0042 |
| 125 | 0.8308 | 0.8222 | 0.0093 | 0.7872 | 0.0073 |
| 150 | 0.8350 | 0.8028 | 0.0297 | 0.7682 | 0.0263 |
| 175 | 0.8100 | 0.7706 | 0.0256 | 0.7375 | 0.0226 |

## 3. Plateau analysis

Marginal C@100 gain vs baseline (K=50). A difference is "significant" if it exceeds 2x the pooled stability-band std.

| K | C@100 mean(5) | delta vs K=50 | 2*sigma | significant? |
|---:|---:|---:|---:|---:|
| 50 | 0.8208 | +0.0000 | 0.0166 | no |
| 60 | 0.8220 | +0.0012 | 0.0149 | no |
| 70 | 0.8207 | -0.0002 | 0.0147 | no |
| 80 | 0.8248 | +0.0039 | 0.0175 | no |
| 90 | 0.8262 | +0.0053 | 0.0160 | no |
| 100 | 0.8223 | +0.0015 | 0.0142 | no |
| 125 | 0.8222 | +0.0013 | 0.0176 | no |
| 150 | 0.8028 | -0.0180 | 0.0436 | no |
| 175 | 0.7706 | -0.0502 | 0.0380 | YES |

## 4. Conditional reranker efficiency

C@100 conditional on eligible events = C@100 * total / eligible. This strips out the cascade recall and measures the reranker alone.

| K | eligible/total | C@100 (raw) | C@100 (conditional) | RC@100 (conditional) |
|---:|---:|---:|---:|---:|
| 50 | 2061 / 2400 | 0.8300 | 0.9665 | 0.9175 |
| 60 | 2073 / 2400 | 0.8292 | 0.9600 | 0.9112 |
| 70 | 2111 / 2400 | 0.8300 | 0.9436 | 0.9043 |
| 80 | 2129 / 2400 | 0.8379 | 0.9446 | 0.8957 |
| 90 | 2157 / 2400 | 0.8358 | 0.9300 | 0.8818 |
| 100 | 2191 / 2400 | 0.8304 | 0.9096 | 0.8727 |
| 125 | 2210 / 2400 | 0.8308 | 0.9023 | 0.8566 |
| 150 | 2238 / 2400 | 0.8350 | 0.8954 | 0.8517 |
| 175 | 2271 / 2400 | 0.8100 | 0.8560 | 0.8195 |

## 5. Mean rank scaling

If relative_rank (= mean_rank / C(K,2)) decreases with K, the reranker improves with more context. If flat or rising, bigger pools just add noise.

| K | mean_rank | C(K,2) | relative_rank |
|---:|---:|---:|---:|
| 50 | 16.2 | 1225 | 0.0132 |
| 60 | 17.1 | 1770 | 0.0097 |
| 70 | 23.2 | 2415 | 0.0096 |
| 80 | 24.8 | 3160 | 0.0079 |
| 90 | 29.0 | 4005 | 0.0072 |
| 100 | 39.9 | 4950 | 0.0081 |
| 125 | 48.2 | 7750 | 0.0062 |
| 150 | 62.3 | 11175 | 0.0056 |
| 175 | 112.9 | 15225 | 0.0074 |

## 6. K=200 OOM analysis

K=200 OOMed with batch_size=96 on a 95GB GPU. The table below estimates activation memory for the couple-reranker tensor alone (does NOT include the upstream cascade).

| K | couples/event | couples/batch(96) | reranker mem (GB) | fits budget? |
|---:|---:|---:|---:|---:|
| 50 | 1,225 | 117,600 | 2.5 | yes |
| 60 | 1,770 | 169,920 | 3.6 | yes |
| 70 | 2,415 | 231,840 | 4.9 | yes |
| 80 | 3,160 | 303,360 | 6.4 | yes |
| 90 | 4,005 | 384,480 | 8.1 | yes |
| 100 | 4,950 | 475,200 | 10.0 | yes |
| 125 | 7,750 | 744,000 | 15.6 | yes |
| 150 | 11,175 | 1,072,800 | 22.5 | yes |
| 175 | 15,225 | 1,461,600 | 30.7 | yes |
| 200 | 19,900 | 1,910,400 | 40.1 | NO |

Cascade overhead estimate: ~55 GB. Reranker budget on a 95 GB GPU: ~40 GB.

**Recommendation:** retry K=200 with `BATCH_SIZE=64` (estimated reranker memory: 26.7 GB, leaving ~13 GB headroom for fragmentation).

## 7. Recommendations

| K | C(K,2) | time (min) | C@100 mean(5) | delta vs K=50 | verdict |
|---:|---:|---:|---:|---:|---:|
| 50 | 1225 | 22 | 0.8208 | +0.0000 | equivalent |
| 60 | 1770 | 22 | 0.8220 | +0.0012 | equivalent |
| 70 | 2415 | 24 | 0.8207 | -0.0002 | equivalent |
| 80 | 3160 | 25 | 0.8248 | +0.0039 | equivalent |
| 90 | 4005 | 26 | 0.8262 | +0.0053 | BEST |
| 100 | 4950 | 28 | 0.8223 | +0.0015 | equivalent |
| 125 | 7750 | 33 | 0.8222 | +0.0013 | equivalent |
| 150 | 11175 | 39 | 0.8028 | -0.0180 | worse |
| 175 | 15225 | 46 | 0.7706 | -0.0502 | worse |

### Failed runs

- **K=200**: no experiment directory found
