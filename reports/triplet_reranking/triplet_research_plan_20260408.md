# Post-ParT Reranking — Research Plan

**Date:** 2026-04-08
**Scope:** add a Stage 3 head (or pre-ranking step) that improves the cascade's per-track metrics over the current ParT Stage 2. Two architectural directions in scope: a **per-couple reranker** (preferred, cheap, narrow) and a **triplet/per-track reranker** (broader, fallback if couples don't work).

## 1. Goal

Two metrics, two directions:

| Direction | Output | Primary metric | Architecture posture |
|---|---|---|---|
| **A — Couple reranker** | Per-couple scores over `(i, j)` pairs from ParT top-50 | **couple-recall@K** for K ∈ {50, 100}: GT couple in top-K of reranked couple list | Per-couple, no cross-couple interactions. Cheap. Narrow. |
| **B — Triplet/per-track reranker** | Per-track scores over top-256 (or top-50) | **P@50 (3/3)**, R@30, R@50 — same as the existing cascade | 3-way architectural interactions inside a per-track scoring head |

Direction A is the preferred first attempt. If it works, it becomes the input to a future triplet-completion stage. If it doesn't, the project pivots to Direction B and uses the existing top-256 directly. Both directions share the literature review and the physics-only baseline.

The new head is **not required to output a triplet** in either direction. The eventual triplet completion is a separate, downstream step out of scope here.

## 2. Constraints (apply to both directions)

- **Inference cost is not capped.** 5-10× the existing cascade is fine; single-GPU memory is the only hard limit.
- **No hardcoded charge or mass cuts in the neural architecture.** Pairwise charge product, ρ-mass indicator, dz compatibility, vertex χ², etc., are SOFT input features. A separate physics-only ranker is allowed as a baseline / ensemble component.
- **Per-couple at the OUTPUT level for Direction A.** A context encoder over top-50 BEFORE the per-couple head is fine; cross-couple reasoning AT the head is not.
- **No premise validation.** H1 (duplet@50 ≥ 65%) and H2 (third pion ∈ top-256 | duplet ∈ top-50 ≥ 90%) are pre-confirmed by the user.

## 3. Why this is plausible

| Quantity | Value | Source |
|---|---:|---|
| ParT val P@50 (3/3) | **0.7169** | `cascade_soap_Cascade_20260406_202001` ep 55 |
| ParT val R@50 | 0.8269 | same |
| Stage 1 R@K1 (K1=256) | 0.943 | the third pion is essentially always in top-256 |
| Duplet@50 (≥2/3 in top-50) | bounded **[0.764, 0.882]** | algebraic from above |

So 76-88% of events already have a GT couple in the C(50, 2) = 1225 candidate set (Direction A's hunting ground), and 71.7% have all three GT pions in top-50 (giving 3 GT couples per event). The third pion is almost certainly in top-256 (Direction B's hunting ground).

**Crucially, this is NOT the same combinatorial-explosion regime as the failed prior attempts** (`triplet_combinatorics.md`: 116k surviving triplets per event after physics filters at K=200). The new framings — "rerank a small set of couples" or "score top-256 tracks with triplet-aware features" — are bounded by O(50²) and O(256), not O(1100³).

## 4. Literature review plan

Both directions share the same review. Target volume: ~25 papers, organized by bucket. **Lit review deliverable: `reports/triplet_lit_review_<date>.md`, 5 pages max.**

### 4.1 HEP

| Bucket | Why | 3-5 starting refs |
|---|---|---|
| **Multibody decay reconstruction (the physics baseline)** | Direct prior art for τ → 3π / a₁ → ρπ reconstruction | Hulsbergen LHCb DecayTreeFitter (`arXiv:hep-ph/0503191`); Waltenberger RAVE / Adaptive Vertex Reconstruction (`arXiv:1105.5128`); Belle II Physics Book τ chapter (`arXiv:1808.10567`) |
| **GNN tracking with edge classifiers** | Closest cousin: pair scoring for "do these belong together" | Exa.TrkX (Choma `arXiv:2003.11603`, Ju `arXiv:2103.06995`); HEP.TrkX TrackML (Farrell `arXiv:1810.06111`); Kieseler Object Condensation (`arXiv:2002.03605`) |
| **Lorentz-invariant pair / k-tuple features** | Already used in CascadeReranker (channel 6) — extend | PELICAN (`arXiv:2211.00454`); LorentzNet (`arXiv:2201.08187`) |

### 4.2 CV / IR / ML

| Bucket | Why | 3-5 starting refs |
|---|---|---|
| **Pair classifiers (Direction A core)** | Per-couple independent scoring is literally a Siamese-style binary classifier | Bromley NIPS 1993 (Siamese); Chicco survey (`arXiv:2006.02295`); Sentence-BERT bi-encoder (Reimers `arXiv:1908.10084`) |
| **Relation Networks for object detection** | The cleanest CV statement of "objects refine each other via pairwise reasoning" — read first | Hu et al. CVPR 2018 (`arXiv:1711.11575`) |
| **Two-stage detection (Direction B, anchor + query)** | Architectural pattern for the broader Stage 3 head | Faster R-CNN (`arXiv:1506.01497`); DETR (`arXiv:2005.12872`); Sparse R-CNN (`arXiv:2011.12450`) |
| **Higher-order GNNs / hypergraph nets (Direction B only)** | Formal framework for k=3 interactions in the per-track head | PPGN (Maron `arXiv:1905.11136`); k-GNN (Morris `arXiv:1810.02244`); HGNN (Feng `arXiv:1809.09401`) |
| **Set-based attention (Direction B only)** | Permutation-invariant aggregation for the per-track head | Deep Sets (Zaheer `arXiv:1703.06114`); Set Transformer (Lee `arXiv:1810.00825`) |
| **IR cascade reranking** | Cost/benefit framework + pointwise vs pairwise vs listwise loss design | Wang/Lin/Metzler "Cascade Ranking" SIGIR 2011; LambdaRank/LambdaMART (Burges 2010); monoBERT (Nogueira `arXiv:1910.14424`) |

**Skip:** FaceNet triplet loss (wrong "triplet"), Visual Relationship Detection (different domain), Cascade R-CNN (less directly applicable than DETR), k-WL theory beyond what the GNN papers cite.

### 4.3 Search strategy
- **arXiv:** `hep-ex`, `hep-ph`, `physics.ins-det`, `cs.CV`, `cs.LG`, `cs.IR`
- **DBs:** INSPIRE-HEP (HEP), Google Scholar, Semantic Scholar (cross-discipline)
- **Queries:** `tau three pion neural reconstruction`, `hypergraph neural network physics`, `relation network particle physics`, `cascade reranking object detection`, `Belle II tau hadronic`, `Siamese network pair classification`
- **Citation crawl:** forward citations from ParT (`arXiv:2202.03772`), Exa.TrkX, and Relation Networks

## 5. Methodology

### 5.1 Phase 1 — couple-count and recall measurement (1 day)

A small diagnostic. Reuses `MetricsAccumulator` and the cascade_soap checkpoint.

For each event in val:
1. Take ParT top-50.
2. Enumerate all C(50, 2) = 1225 couples.
3. Apply a LOOSE physics pre-filter: keep couples with `m(ij) ≤ m_τ = 1.777 GeV`. **No charge cut, no ρ-mass window** — those are exactly the biases the previous attempts hardcoded.
4. Report:
   - Distribution of surviving-couple count per event (mean, median, p90)
   - Number of GT couples per event (1-3)
   - GT-couple-rank distribution under a NULL ranking (random) — sanity baseline

Output: `couple_count_<date>.md`, **1 page**.

### 5.2 Phase 2 — physics-only baselines (3-4 days)

Two baselines, one per direction. **Both must be measured before any neural work**, so we know what the neural variants need to beat.

**Baseline 2A (Direction A): physics-only couple ranker.** For each surviving couple `(i, j)`:
- Vertex χ² (closed-form 2-track impact-parameter least-squares; no Kalman)
- Mass deviation `|m(ij) − 770|`
- dz compatibility `|Δdz_sig|`
- Aggregate: `score = -χ² - |m − 770| / 70 - |Δdz_sig|`, coefficients tuned on a small subset
Sort couples; measure couple-recall@K for K ∈ {50, 100, 200}.

**Baseline 2B (Direction B): physics-only per-track triplet score.** For each track `i` in top-256:
- Iterate all `(j, k)` pairs from top-50 with `j ≠ k ≠ i`
- For each (i, j, k), compute 3-track vertex χ² and mass-vs-`m_τ` deviation
- Aggregate per track via soft-min over the best few (j, k)
Rerank top-256; measure P@50, R@30, R@50.

Implementation reuses `weaver/.../CascadeReranker.py:262` (`_compute_extra_pairwise_features`) for the per-pair physics quantities.

Deliverable: `reports/triplet_physics_baseline_<date>.md`, **2-3 pages** covering both baselines.

### 5.3 Phase 3 — neural pilots (sequential)

Direction A first (cheaper, narrower), then Direction B as fallback if A doesn't beat its baseline.

#### 5.3.A Couple reranker (Direction A) — 1 week

Three architectures, all per-couple at the OUTPUT level:

| # | Architecture | Sketch | Param est. |
|---|---|---|---|
| **A1** | Per-couple MLP | (track-i feats, track-j feats, pairwise (i,j) feats, vertex χ², mass deviation) → 3-layer MLP → scalar | ~50K |
| **A2** | Encoder + per-couple MLP | Frozen ParT encoder produces per-track embeddings; (emb_i, emb_j, pairwise feats) → MLP → scalar | ~200K + frozen ParT |
| **A3** | Siamese bilinear | Two-tower encoder over top-50; couple score = `e_iᵀ W e_j + biases` | ~100K + frozen ParT |

**Loss:** binary cross-entropy with positives = GT couples, negatives = ≤200 random non-GT couples per event. Pairwise ranking loss is also viable.

**Pilot protocol:** train A1 on full train, freeze Stage 1+2, 30 epochs. Measure couple-recall@50/@100 on val. Decision tree:
- couple-recall@100 ≥ baseline 2A + 0.05 → A1 wins; consider ensemble
- couple-recall@100 ∈ [baseline + 0.01, baseline + 0.05] → marginal; try A2 for more context
- couple-recall@100 < baseline + 0.01 → kill A1, try A2; if A2 fails, kill A3 too and pivot to Direction B

Deliverable per architecture: `reports/couple_pilot_<arch>_<date>.md`, **1 page each**.

**If Direction A succeeds:** the couple reranker is locked in as a feeding stage. The current research plan is complete — future work plugs the reranked couples into a triplet-completion head as a separate effort.

**If all three Direction A architectures fail:** proceed to Phase 3.B.

#### 5.3.B Triplet / per-track reranker (Direction B) — 2 weeks

Three architectures, all per-track at the output level, with 3-way interactions internally:

| # | Architecture | Sketch | Param est. |
|---|---|---|---|
| **B1** | DETR-style decoder | Top-256 = queries, top-50 = keys/values; cross-attention + small MLP head outputs per-query (per-track) score. Pairwise physics features feed the attention bias (same as existing CascadeReranker). 1-2 decoder layers. | ~500K-1M |
| **B2** | Set Transformer with PMA | Top-256 = input set; PMA inducing points produce contextual track embeddings; per-track scoring head | ~1-2M |
| **B3** | PPGN-style 3-tensor | Order-3 tensor `T[i,j,k]` over top-256 (or top-50 anchor tuples); per-track score = symmetric pool over the tensor; physics features feed each cell as inputs | ~3-5M, ~300k MLP evals/event |

**Loss:** existing pairwise ranking loss from CascadeReranker, applied to the new per-track scores. NO triplet-extraction loss. Triplet structure influences the FORWARD pass only.

**Pilot protocol:** B1 first on a 10% subset, frozen Stage 1+2, 30 epochs, measure P@50/R@30/R@50. Decision tree:
- P@50 < 0.72 → kill B1, try B2
- P@50 ∈ [0.72, 0.73] → marginal; unfreeze ParT and retry joint
- P@50 ≥ 0.74 → scale to full data + 100 epochs

Deliverable per architecture: `reports/triplet_pilot_<arch>_<date>.md`, **1 page each**.

### 5.4 Phase 4 — full-scale training (1 week)

Whichever architecture survives Phase 3 (couple or triplet) gets a full 100-epoch run on the cutoff dataset, using `train_cascade.py` extended with a `--stage3-arch` flag and a wrapper module. EMA enabled with `--ema-decay 0.999 --ema-start-step 2000` (the warmup-aware variant).

**Success criteria:**
- Direction A winner: couple-recall@100 ≥ 0.95 on val (the threshold for "the GT couple is reliably findable")
- Direction B winner: val P@50 ≥ 0.74 (≥ +0.02 absolute over the 0.7169 baseline)

If neither winner clears its threshold, write the failure report and document the kill.

### 5.5 Phase 5 — physics + neural ensemble (2-3 days, only if Phase 3 produces a working neural variant)

Evaluate ensembles of (physics-only, neural) for the winning direction:
- Linear combination: `score = α × neural + (1-α) × physics`, sweep α
- Reciprocal-rank fusion: `1/(60 + r_neural) + 1/(60 + r_physics)`
- Learned 2-layer gating MLP

Pick the best ensemble. Deliverable: `reports/triplet_ensemble_<date>.md`, **1 page**.

### 5.6 Phase 6 — final integrated report (2 days)

Single document. **Target: 6-8 pages.** Sections:

1. Numbers (Phase 1) — 0.5 page
2. Lit review highlights — 1.5 pages
3. Physics baselines (both directions) — 1 page
4. Neural pilot results — Direction A (1 page) + Direction B if needed (1 page)
5. Ensemble result — 0.5 page
6. Recommendation: which direction integrates / which is killed — 0.5 page
7. References — 0.5 page

This is the deliverable the user explicitly asked for: a report outlining directions for triplet-based reranking.

## 6. Critical files to reuse

| File | Why |
|---|---|
| `weaver/weaver/nn/model/CascadeReranker.py:262` | `_compute_extra_pairwise_features` — 6-channel pairwise physics tensor (charge product, dz, ρ-mass, ρ-OS, dxy, Lorentz dot). Reuse for couple AND triplet features. |
| `weaver/weaver/nn/model/CascadeModel.py:117` | `compute_loss` pattern — Stage 3 wrapper mirrors this |
| `part/utils/training_utils.py:417` | `MetricsAccumulator.compute()` — extend with `couple_recall_at_k` (~20 lines for Direction A) |
| `part/train_cascade.py` | The harness — Stage 3 slots in alongside Stage 2 the same way the EMA wrapper does |
| `part/networks/lowpt_tau_CascadeReranker.py` | Stage 1+2 wrapper — Stage 1+2+3 wrapper mirrors this |
| `part/reports/triplet_reranking/triplet_combinatorics.md` | **Read first.** Prior failure mode: hard physics filters → 116k surviving triplets per event. Cautionary tale for what NOT to repeat. |
| `part/reports/triplet_reranking/duplet_triplet_analysis.md` | Existing duplet/triplet recall analysis on the OLD pre-cutoff baseline; precedent for measurement methodology. |

## 7. Stop conditions

| Phase | Kill if |
|---|---|
| 1 | Median surviving-couple count > 600 (filter too loose, revisit) OR < 50 (too tight, biases) |
| 2A | Physics-only couple-recall@100 ≥ 0.97 → physics already wins for couples; promote it, skip neural couple reranker |
| 2B | Physics-only P@50 regresses ParT by > 0.01 → physics signal alone is too weak for the per-track direction; the neural variant is the right path |
| 3A | All three couple architectures fail to beat physics 2A by ≥ 0.02 → kill couple direction; pivot to 3B |
| 3B | All three triplet architectures fail to clear P@50 = 0.72 → kill the entire neural reranking project |
| 4 | Late-epoch divergence not fixable by EMA + dropout + denoising playbook |
| 5 | Best ensemble doesn't beat the better individual ranker → use the winner alone, no ensembling |

## 8. Total effort

- Direction A only: ~2.5 weeks (Phases 1, 2A, 3A, partial 4, 5, 6)
- Direction A fails → Direction B: +2 weeks (Phases 2B, 3B, full 4, 6)
- **Worst case both directions explored:** ~5 weeks sequential

Phases 1+2 partially parallel; Phases 3A → 3B → 4 sequential.

---

**Next concrete action:** Phase 1 (couple-count diagnostic). ~150 lines, reuses existing utilities, runs in <1 day. Provides the descriptive numbers needed by both directions before any architecture work begins.
