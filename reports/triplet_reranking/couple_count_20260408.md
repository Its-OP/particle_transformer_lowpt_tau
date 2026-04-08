# Phase 1 — Couple-Count Diagnostic

**Generated:** 2026-04-08 17:44
**Events:** 5000
**Source:** `cascade_soap_Cascade_20260406_202001/checkpoints/best_model.pt`
**Top-50 selection:** trained Stage 2 score within Stage 1 top-256.

Filters reported (Filter A is the plan-mandated loose baseline; B/C/D
are for context, NOT recommended — they bake in physics biases the new
framing must avoid):

- A: m(ij) <= m_tau                      (loose, no bias)
- B: A + opposite-sign                   (kills SS pairs, +charge bias)
- C: A + 0.4 GeV <= m(ij) <= 1.2 GeV     (wide a1 mass window, +mass bias)
- D: A + B + C                           (all three together, ~prior physics filter)

---

## 1. Surviving couples per event (per filter)

| filter | mean | median | p25 | p75 | p90 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 862 | 858 | 761 | 962 | 1055 | 1107 | 334 | 1224 |
| B | 432 | 430 | 381 | 482 | 527 | 553 | 176 | 621 |
| C | 518 | 495 | 412 | 604 | 722 | 788 | 185 | 1005 |
| D | 261 | 249 | 209 | 303 | 361 | 396 | 96 | 520 |

Theoretical max per event: 1225 = C(50, 2).

---

## 2. GT pions in top-50 (histogram)

| n_gt_in_top50 | events | fraction |
|---:|---:|---:|
| 0 | 481 | 0.0962 |
| 1 | 329 | 0.0658 |
| 2 | 661 | 0.1322 |
| 3 | 3529 | 0.7058 |

**Duplet rate (>=2/3 GT in top-50): 0.8380**
**Triplet rate (3/3 GT in top-50): 0.7058**

---

## 3. GT couples per event (histogram)

| n_gt_couples | events | fraction |
|---:|---:|---:|
| 0 | 810 | 0.1620 |
| 1 | 661 | 0.1322 |
| 2 | 0 | 0.0000 |
| 3 | 3529 | 0.7058 |

Sanity (n_gt_couples == C(n_gt_in_top50, 2)): **5000/5000** events match.

### GT couple survival rate per filter

Fraction of GT couples that pass each filter (averaged over events with
at least one GT couple). Filter A should be ~1.0 — kinematic conservation
guarantees `m(GT pair) <= m_tau`. Filters B/C/D will drop below 1.0 by
the amount the imposed bias kills GT couples.

| filter | GT survival |
|---|---:|
| A | 0.9991 |
| B | 0.6657 |
| C | 0.8350 |
| D | 0.5736 |

---

## 4. Random-rank baseline (per filter)

Expected recall@K under a uniformly random ranking of the surviving
couples, averaged over events with at least one surviving GT couple.
**This is the floor the future couple reranker must beat.**

| filter | recall@50 | recall@100 | recall@200 |
|---|---:|---:|---:|
| A | 0.1484 | 0.2802 | 0.4984 |
| B | 0.2085 | 0.3922 | 0.6848 |
| C | 0.2138 | 0.3929 | 0.6612 |
| D | 0.3060 | 0.5548 | 0.8765 |

---

## 5. Notes for Phase 2 / Phase 3

- **Filter A is the framing-correct baseline** (no charge or mass biases), and it leaves a much larger candidate pool than the user intuition of "a couple hundred". The neural couple reranker should be trained on Filter A inputs.
- **Filters B/C/D** show the cost (in surviving-couple count and GT survival rate) of progressively tighter physics. Phase 2 (physics-only baseline) can pick whichever filter minimizes the surviving count without dropping GT survival meaningfully. Filter D is the closest to the prior (failed) physics-cascade approach in `triplet_combinatorics.md`.
- **Random-rank baseline** under Filter A is the conservative floor. The neural reranker is interesting only if it beats Filter A random recall by a meaningful margin.
