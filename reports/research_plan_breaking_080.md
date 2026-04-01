# Plan: Breaking R@200 ≥ 0.80 — Research-Backed Experiments

## Context

Current best: Physics ParT cascade R@200=0.693 (Stage 1 R@600=0.91 ceiling).
Gap to target: +0.107. Hybrid LambdaRank experiment is in-training on server.

## What's been ruled out (from experiment log)

- All Stage 1 graph improvements (kNN noise is structural, not fixable)
- Boundary negative sampling (destroys ranking signal)
- OHEM / ASL (destabilize at 0.27% signal)
- Pairwise features in Stage 1 kNN (AE collapse)
- Combinatorial triplet post-processing (not selective at low pT)
- Per-track physics counts (rho-candidate count is just a pT proxy)

## Proposed Experiments

### Experiment 1: DFTopK — Differentiable Top-K Selection Loss

**Source:** Zhu et al., "Differentiable Fast Top-K Selection for Large-Scale
Recommendation" (arXiv:2510.11472, Oct 2025). O(k log k) differentiable top-k
operator purpose-built for cascade ranking. Also: Sander et al., ICML 2023
(convex-analysis top-K).

**Hypothesis:** The current pairwise ranking loss wastes gradient on easy
pairs far from rank 200. DFTopK directly optimizes for the top-K selection
decision, providing sharper gradients at the critical boundary. Unlike
LambdaRank (which reweights pairwise gradients), DFTopK formulates the loss
as a differentiable selection operator.

**Preliminary analysis needed:** Check DFTopK paper implementation complexity.
Compare its gradient properties to our LambdaRank at convergence. The key
question: does DFTopK avoid the gradient-starvation problem that boundary
sampling suffered from, while still focusing on the rank-200 boundary?

**Not tried before:** RS@K was tried for Stage 1 (neutral), but DFTopK is a
different formulation. LambdaRank is currently in-training — DFTopK is the
backup if LambdaRank shows no gain.

**Implementation:** New `loss_mode='dftopk'` in CascadeReranker.
**Complexity:** Medium (need to implement or adapt the DFTopK operator).

---

### Experiment 2: Contrastive Pre-training with Physics Augmentation

**Source:** "PASCL: Supervised Contrastive Learning with Perturbative
Augmentation for Particle Decay Reconstruction" (arXiv:2402.11538, 2024).
Also: BCL (Zhu et al., 2022) for class-balanced contrastive with prototypes.

**Hypothesis:** The ParT Stage 2 is trained from scratch on a 0.5% signal
task. Contrastive pre-training on the 600-track candidate set could produce
better initial representations before ranking loss takes over. PASCL uses
physics-informed augmentations (perturbing decay kinematics) to generate
hard positives/negatives — directly applicable to tau→3pi.

**Preliminary analysis needed:** Check if our current Stage 2 embeddings
(after 100 epochs) show meaningful clustering in latent space. If GT tracks
DON'T cluster, contrastive pre-training could help. If they DO cluster but
ranking is still poor, the issue is in the scoring head, not representations.

**Not tried before:** SupMin contrastive was tried for Stage 1 (promising but
superseded). Contrastive pre-training for Stage 2 has not been attempted.

**Implementation:** Two-phase training: Phase 1 (contrastive on 600-track
embeddings, ~20 epochs), Phase 2 (ranking loss with pre-trained encoder frozen
initially, then unfrozen).
**Complexity:** Medium.

---

### Experiment 3: MaskFormer / DETR Decoder with 3 Learned Queries

**Source:** Van Stroud et al., "Transformers for Charged Particle Track
Reconstruction" (Phys. Rev. X 15, Dec 2025) — 97.2% efficiency on TrackML.
Also: HEPTv2 (arXiv:2510.07594) — 3000 queries, ~28ms/event.

**Hypothesis:** Instead of per-track scoring + ranking (which treats signal
identification as a ranking problem), reframe it as set prediction: 3 learned
queries cross-attend to all 600 tracks. Each query "discovers" one pion
through competitive attention. This bypasses ranking entirely — the model
directly outputs 3 track assignments.

**Preliminary analysis needed:** Check attention patterns in the current ParT
— does the model already form implicit "proto-queries" (attention heads
specializing on different GT pions)? If yes, explicit queries would formalize
this. Compute per-head attention entropy to measure specialization.

**Not tried before:** DETR was explored in the blueprint but deprioritized.
The key difference: Van Stroud et al. (Dec 2025) now validates MaskFormer
for HEP tracking at SOTA level, removing the "will it work in HEP?" risk.

**Implementation:** New `CascadeRerankerDETR` module: existing ParT encoder
(reuse blocks) + cross-attention decoder with K=3-10 queries + Hungarian
matching loss. Can share the ParT encoder with the current scoring approach
for ablation.
**Complexity:** Medium-high (Hungarian loss exists in codebase already).

---

### Experiment 4: Heterogeneous Graph with Physics-Typed Edges

**Source:** LHCb collaboration, "Heterogeneous GNN for beauty-hadron
reconstruction" (arXiv:2504.21844, 2025). Integrated pruning at every layer;
outperformed multi-stage GNN.

**Hypothesis:** Instead of a uniform kNN graph, build typed edges:
(a) vertex-compatible pairs (small |Δdz|), (b) opposite-sign pairs in rho mass
window, (c) kinematic proximity (η,φ), (d) stage1-score proximity. Each edge
type has its own message-passing MLP. This is different from our failed composite
graph diagnostic (which just UNION'd kNN indices) — here the types are processed
SEPARATELY by type-specific functions, then combined.

**Preliminary analysis needed:** Compute what fraction of GT-GT pairs fall into
each edge type. If type (b) captures 30% of GT pairs (our diagnostic showed 31%
pass rho window) and type (a) captures 58% (dz compat |d'|=0.58), the combination
should cover >70% of GT pairs with physics-enriched edges — vs 25% in plain kNN.

**Not tried before:** Our composite graph diagnostic tested UNION of kNN spaces
and found no gain. Heterogeneous GNN is fundamentally different: separate
MLPs per edge type, not a single aggregation over a mixed edge set.

**Implementation:** New Stage 2 architecture (or modify CascadeReranker to
build typed edges before attention).
**Complexity:** Medium-high.

---

### Experiment 5: Sparse Attention with Physics-Based Block Masking

**Source:** "XAttention: Block Sparse Attention with Antidiagonal Scoring"
(ICML 2025). Also: FlexAttention + DBSCAN masks (TrackFormers Part 2,
arXiv:2509.26411) — reduces attention by ~400×.

**Hypothesis:** Current ParT does full 600×600 attention (360K pairs). Most
attention is between noise-noise pairs. Physics-based masking could: (a) group
tracks by proximity in (η,φ) or (dxy,dz), (b) only compute attention within
and between groups, (c) add a "long-range" attention channel for cross-group
signal discovery. This preserves the information in pairwise features while
reducing compute, potentially allowing more layers or larger models.

**Preliminary analysis needed:** Profile current attention patterns — what
fraction of attention weight goes to distant (ΔR > 1.0) pairs? If >90% of
attention is local, sparse attention loses nothing.

**Not tried before:** No sparse attention experiments.

**Implementation:** Use PyTorch SDPA with custom block mask.
**Complexity:** Medium.

---

### Experiment 6: Self-Supervised Pre-training (MPMv2 / OmniJet)

**Source:** MPMv2 (arXiv:2409.12589, 2024) — masked particle modeling without
tokenization. OmniJet-α (SciPost 2025) — foundation model for particle
physics, 50% improvement in tau momentum resolution from pre-training.

**Hypothesis:** Pre-train the ParT encoder on a self-supervised objective
(mask per-track features, predict them) using ALL training data (not just
labeled events). This produces a general track representation that can then
be fine-tuned for ranking. Particularly useful because our signal fraction
is only 0.5% — most training events contribute almost no ranking signal, but
they DO contribute to understanding what tracks look like.

**Preliminary analysis needed:** Check how many unlabeled events are available.
If the training set is ~270K events with 3 labeled pions each, that's 270K ×
1100 = 297M tracks for self-supervised learning but only 810K labeled pions.

**Not tried before:** Backbone pretraining exists (MaskedTrackPretrainer) for
Stage 1, but was never applied to Stage 2's ParT encoder.

**Implementation:** Adapt existing `BackbonePretraining.py` masking approach
to the 600-track Stage 2 input. Pre-train, then fine-tune with ranking loss.
**Complexity:** Medium (infrastructure exists, needs adaptation).

---

## Implementation Priority

| # | Experiment | Effort | Prerequisites | Run in parallel? |
|---|-----------|--------|---------------|-----------------|
| 1 | DFTopK loss | 3 days | Wait for LambdaRank results | After LambdaRank |
| 2 | Contrastive pre-training | 5 days | Embedding analysis first | Yes (server 2) |
| 3 | MaskFormer decoder | 7 days | Hungarian matcher exists | After 1 or 2 |
| 4 | Heterogeneous GNN edges | 5 days | Edge type analysis first | After 3 |
| 5 | Sparse attention | 3 days | Attention profiling first | Yes (parallel) |
| 6 | Self-supervised pre-train | 5 days | Adapt existing code | Yes (parallel) |

## Preliminary Hypothesis Experiments (run locally, MPS)

Use the trained Physics ParT checkpoint (`partfull_physics_concat_Cascade`,
pair_extra_dim=5, R@200=0.693) for all analyses. Load via
`cascade_model_analysis.load_cascade_from_checkpoint()`.

Script: `diagnostics/hypothesis_experiments.py`
Data: subset val (max_steps=50, ~400 events)

### H1: Attention Pattern Analysis (validates Exp 3 + 5)

Hook into all 8 transformer blocks' `nn.MultiheadAttention` layers via
`register_forward_hook`. For each batch, extract attention weight matrices
(B, num_heads, K1, K1).

**Compute:**
- Per-head entropy: H = -sum(attn * log(attn+eps)) averaged over queries.
  Low entropy → head is selective (good for sparse attention).
  High entropy → head is diffuse (needs full attention).
- Locality: fraction of attention weight on pairs with ΔR < 0.5, < 1.0, < 2.0.
  High locality → sparse attention viable.
- GT-focus: mean attention weight from GT tracks to other GT tracks vs to BG.
  If GT→GT attention >> GT→BG, the model is already finding signal pairs.
- Head specialization: do different heads attend to different track subsets?
  Compute pairwise cosine similarity between attention rows across heads.

**Hypothesis to confirm/deny:**
- "90% of attention weight is on ΔR < 1.0 pairs" → sparse attention viable
- "At least one head specializes on GT tracks" → implicit query formation
- "Attention entropy decreases in later layers" → progressive signal discovery

### H2: Embedding Clustering (validates Exp 2 + 6)

Extract intermediate embeddings after each transformer block (8 layers ×
(K1, B, embed_dim)). For GT tracks and a sample of BG tracks:

**Compute:**
- Cosine similarity: mean GT-GT similarity vs mean GT-BG similarity per layer.
  If GT-GT >> GT-BG in later layers, representations ARE separating.
- Simple linear probe: train a logistic regression on layer-L embeddings to
  predict GT/BG. Report AUC per layer. If AUC is already high (>0.9) at layer
  4, the scoring head is the bottleneck. If AUC is low even at layer 8,
  representations need improvement (pre-training helps).
- Score correlation: Pearson r between scoring head output and linear probe
  prediction. If r is low, the scoring head is under-utilizing the representation.

**Hypothesis to confirm/deny:**
- "GT tracks cluster by layer 4" → scoring head bottleneck, not representation
- "Linear probe AUC > 0.95 at layer 8" → good representations, loss is bottleneck
- "Linear probe AUC << scoring head R@200" → scoring head underperforms, architecture issue

### H3: Edge Type Coverage (validates Exp 4)

For each physics-criterion edge type, compute coverage of GT-GT pairs and
signal-to-noise ratio (GT-GT edges / total edges of that type):

Edge types:
- (a) |Δdz_sig| < 0.5: vertex-compatible (our diagnostic showed |d'|=0.58)
- (b) OS + |m_ij - 770| < 150 MeV: rho candidates (31% of GT pairs pass)
- (c) ΔR < 1.0: kinematic proximity
- (d) |Δ(stage1_score)| < threshold: score-proximity

For each type, count: total edges, GT-GT edges, GT-BG edges, BG-BG edges.
Report: GT-GT coverage (what % of GT pairs are connected), signal-to-noise
ratio (GT-GT / total). Compare to uniform kNN baseline.

**Hypothesis to confirm/deny:**
- "Union of types (a)+(b)+(c) covers >60% of GT-GT pairs" → heterogeneous viable
- "Signal/noise ratio per type is > 5× baseline kNN" → typed edges are enriched
- "Type (b) alone covers 31% of GT pairs with high S/N" → rho-typed edges valuable

### H4: Score Distribution at Rank Boundary (validates Exp 1)

Analyze the score distribution around rank 200 for the current model:

**Compute:**
- Score gap: mean(score at rank 199) - mean(score at rank 200).
  Small gap → model is uncertain at boundary → loss alignment matters most.
  Large gap → model is confident → other bottlenecks dominate.
- GT pion score distribution vs rank-200 threshold.
  What fraction of missed GT pions have scores within 5% of the rank-200 cutoff?
  High fraction → "just missed" → DFTopK/LambdaRank should rescue them.
  Low fraction → GT pions are deeply buried → need representation improvement.
- Per-layer score evolution: run forward pass, score at each layer's output
  (via a simple linear probe). Track how GT pion rank changes across layers.
  If rank improves monotonically → architecture is working, just needs more signal.
  If rank degrades in later layers → overfitting or attention collapse.

**Hypothesis to confirm/deny:**
- "Score gap at rank 200 is < 0.1 std" → boundary is tight, loss alignment critical
- ">50% of missed GT pions are within 10% of the threshold" → DFTopK should help
- "GT rank improves monotonically across layers" → deeper model could help

## Verification

- Each experiment adds to `reports/experiment_log.md`
- TDD tests for new loss functions / architectures
- Smoke test on subset before server training
- Full test suite passing before any deployment
