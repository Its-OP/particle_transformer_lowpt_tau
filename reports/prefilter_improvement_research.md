# Pre-Filter Improvement Research Report

**Date:** 2026-03-19
**Scope:** Literature survey across 3 directions, 21 validated techniques for improving the hybrid autoencoder+MLP pre-filter.
**Current baseline:** R@200=58%, d-prime=1.29, median rank=132 (7-feature model). 13-feature model training in progress.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Direction 1: Architecture Improvements](#direction-1-architecture-improvements)
3. [Direction 2: Loss Function & Training Recipe](#direction-2-loss-function--training-recipe)
4. [Direction 3: Alternative Paradigms](#direction-3-alternative-paradigms)
5. [Cross-Direction Rankings](#cross-direction-rankings)
6. [Failed Experiment Conflict Matrix](#failed-experiment-conflict-matrix)
7. [Recommended Implementation Roadmap](#recommended-implementation-roadmap)

---

## Executive Summary

The research identifies **21 techniques** across three directions. After filtering for conflict with failed experiments and applicability to our data properties (~1100 tracks/event, 3 GT pions at 0.27% positive rate, uncanny valley at pT 0.3-0.5 / |dxy|<0.5), we recommend a **staged implementation** starting with low-effort high-impact changes.

**Top 5 techniques by expected impact-to-effort ratio:**

| Rank | Technique | Direction | Effort | Expected Impact |
|------|-----------|-----------|--------|-----------------|
| 1 | AE as auxiliary loss (not primary) | Architecture | Small | High — fixes root cause of all AE collapses |
| 2 | PNA multi-aggregation | Architecture | Small | Medium-High — richer neighborhood statistics |
| 3 | Temperature-scheduled contrastive | Loss | Small | Medium — targets uncanny valley via dynamic hardness |
| 4 | Deferred reweighting (DRW) | Loss | Small | Medium — better AE representations before ranking push |
| 5 | FiLM conditioning | Architecture | Small | Medium — context-dependent message interpretation |

---

## Direction 1: Architecture Improvements

### 1.1 PNA — Principal Neighbourhood Aggregation

**Source:** Corso et al., "Principal Neighbourhood Aggregation for Graph Nets," NeurIPS 2020 (arXiv:2004.05718)

**Mechanism:** Replace single max-pool aggregation with concatenation of {mean, max, min, std} + degree scalers, then linear projection back to hidden dim.

**Core assumption:** Different aggregation statistics carry complementary info. For uncanny valley tracks, the *distribution shape* of neighborhoods (mean, variance) distinguishes signal from background even when the max-neighbor looks identical.

**Application:** Drop-in replacement for the `max` in `neighbor_features.max(dim=-1)` within the kNN message-passing rounds. Aggregate with {mean, max, min, std}, concatenate (4× hidden), project to hidden_dim.

**Expected impact:** Directly addresses uncanny valley — max-pool discards distributional info that separates ambiguous tracks.
**Effort:** Small (~10-line wrapper around existing EdgeConv)
**Conflict:** None — changes aggregation method, not edge features. Autoencoder unaffected.

---

### 1.2 Multi-Scale Parallel kNN

**Source:** Ohamouddou et al., "MS-DGCNN++," 2025 (arXiv:2507.12602)

**Mechanism:** Build 2-3 kNN graphs in parallel with different k values (k=8, k=32, k=64). Each scale has its own EdgeConv branch. Fuse via hierarchical skip connections (fine→coarse→back).

**Core assumption:** Track neighborhoods have meaningful structure at multiple spatial scales. Local (k=8) captures jet core; broader (k=64) captures event-level tau topology.

**Application:** In graph construction, build multiple kNN graphs in (eta,phi). Run separate EdgeConv per scale, fuse outputs before scoring MLP.

**Expected impact:** Moderate — uncanny valley tracks may be distinguishable at broader scales where tau topology becomes visible.
**Effort:** Medium (duplicates EdgeConv layers, roughly doubles message-passing params)
**Conflict:** None — purely graph topology, no pairwise physics features.

---

### 1.3 Dynamic Graph Recomputation (DGCNN-style)

**Source:** Wang et al., "Dynamic Graph CNN for Learning on Point Clouds," ACM TOG 2019 (arXiv:1801.07829)

**Mechanism:** After round 1 of message passing (spatial kNN in eta,phi), recompute kNN in *learned feature space* before round 2. Allows semantically similar but spatially distant tracks to become neighbors.

**Core assumption:** Semantic similarity in learned features is better than spatial proximity after initial encoding. GT pions spread at ΔR=1.3 may develop similar representations after round 1.

**Application:** Replace static kNN reuse with feature-space kNN between rounds. One additional kNN computation per forward pass.

**Expected impact:** Could help with spread GT pions. Speculative — depends on whether 1 round produces discriminative enough features.
**Effort:** Small (replace static kNN with feature-space kNN between rounds)
**Conflict:** Low risk. Not the same as injecting pairwise physics (failed #1-2), but monitor AE reconstruction loss for early plateau.

---

### 1.4 FiLM Conditioning in Message Passing

**Source:** Brockschmidt, "GNN-FiLM," ICML 2020 (arXiv:1906.12192); Manoharan et al., 2025 (arXiv:2511.05456)

**Mechanism:** Modulate each message feature-wise based on the *target* node: `message_modulated = γ(target) * message(source) + β(target)`. Makes message interpretation context-dependent — a low-pT track interprets neighbor info differently than a high-pT track.

**Core assumption:** The relevance of neighbor information depends on the receiving node's properties. Uncanny valley tracks should interpret neighborhood signals differently than high-pT tracks.

**Application:** After EdgeConv edge features, apply FiLM: compute γ, β from target node via small MLP, apply element-wise scaling/shift. Manoharan et al. showed conditioning only the first 1-2 layers suffices.

**Expected impact:** Directly targets uncanny valley — enables different scoring rules per kinematic regime.
**Effort:** Small (~10% parameter increase, small MLP for γ/β)
**Conflict:** None — modulates how messages are interpreted, not what features flow. AE never sees FiLM parameters.

---

### 1.5 Feature Group Decomposition

**Source:** Qin et al., "MAPI-GNN," AAAI 2026 (arXiv:2512.20026); Shi et al., "PolaDCA," 2026 (arXiv:2603.06303)

**Mechanism:** Split 13 input features into semantic groups, process each with separate EdgeConv branches, then fuse:
- **Kinematic:** px, py, pz, eta, phi, pt_error (6 features)
- **Displacement:** dxy_significance, dz_significance, dca_significance (3 features)
- **Quality:** n_valid_pixel_hits, covariance_phi_phi, covariance_lambda_lambda, charge (4 features)

**Core assumption:** Different feature groups define different notions of "neighborhood." Kinematic kNN connects co-moving tracks; displacement kNN connects tracks from the same vertex. Separate processing prevents pT from dominating.

**Application:** Split input tensor into 3 groups. Run separate EdgeConv branches (hidden=64 each), concatenate outputs (3×64=192) before scoring MLP.

**Expected impact:** Directly targets uncanny valley — prevents kinematic features from dominating, forces network to learn from displacement and quality features independently.
**Effort:** Medium (3 parallel branches + fusion layer, similar total params)
**Conflict:** None — each stream processes per-track features only, structurally identical to current single-stream.

---

### 1.6 Local Virtual Nodes

**Source:** Karabulut & Baytas, 2025 (arXiv:2508.20597); Qian et al., NeurIPS 2024 (arXiv:2405.17311)

**Mechanism:** Add 5-10 trainable virtual nodes connected to nearby subsets of real nodes. They act as local relay stations enabling information flow beyond the 2-hop receptive field without additional message-passing rounds.

**Core assumption:** With 2 rounds on kNN(k=16), receptive field covers ≤k²=256 of ~1100 tracks. Virtual nodes bridge gaps between distant GT pions (median ΔR=1.3) in fewer hops.

**Application:** Add virtual node embeddings, connect each to nearest ~100 real nodes. They participate in standard EdgeConv rounds.

**Expected impact:** Addresses spread GT pions problem. IPR-MPNN showed these outperform graph transformers at linear complexity.
**Effort:** Medium (virtual node construction + connection logic)
**Conflict:** Low — monitor for over-smoothing if virtual nodes aggregate too broadly.

---

### 1.7 Autoencoder as Auxiliary Loss (not Primary)

**Source:** Synthesized from Exa.TrkX (Choma et al., 2020, arXiv:2007.00149), EggNet (Chan et al., 2025, arXiv:2506.03415)

**Mechanism:** The autoencoder reconstruction loss currently dominates representation learning, forcing the latent space to reconstruct ALL tracks well (including 99.7% background). Instead: keep the encoder+decoder but make reconstruction an *auxiliary regularizer* (0.1× weight), with the ranking loss as the primary objective.

**Core assumption:** With 13 features (up from 7), the scoring loss provides enough signal to learn good representations without AE domination. The AE as regularizer prevents representation collapse without constraining the latent space to preserve background track information.

**Application:** Change loss weighting: `loss = ranking_loss + 0.1 * reconstruction_loss + 0.5 * denoising_loss` (currently reconstruction has ~1.0 weight). No new modules needed.

**Expected impact:** **HIGH.** The AE IS the bottleneck that collapsed with pairwise features (failed #1-2). Its reconstruction objective forces the latent to represent all tracks well, directly counteracting the scoring objective. Demoting it frees the latent space for discrimination. This also *unblocks future experiments* — pairwise features could potentially be re-attempted with the AE as auxiliary.

**Effort:** Small (training configuration change)
**Conflict:** **Directly addresses root cause** of failed experiments #3 and #4. The AE collapsed because pairwise features made reconstruction easier by shortcutting through relational info. With AE as auxiliary, this shortcut is no longer rewarded.

---

## Direction 2: Loss Function & Training Recipe

### 2.1 SoftmaxLoss@K (Listwise Top-K Ranking Loss)

**Source:** Yang et al., "Breaking the Top-K Barrier," KDD 2025 (arXiv:2508.05673)

**Mechanism:** Listwise loss that directly optimizes NDCG@K via a quantile technique. Estimates the K-th largest score (β_K) via sampling, then weights the softmax loss by sigmoid(score - β_K). Only items near the top-K boundary contribute strongly — exactly where ranking errors matter.

**Core assumption:** Positives exist in the candidate set and score distribution has a meaningful top-K boundary. Holds perfectly (3 GT pions among ~1100, K=200).

**Application:** Replace current pairwise ranking loss with SL@K where K=200. Estimate β_200 by sampling ~100 tracks and taking the 18th-percentile score.

**Expected impact:** High — directly optimizes R@200 instead of pairwise margins. Paper reports up to 14.9% improvement on sparse-positive datasets. Should particularly help the 16.6% of events finding 0/3 GT pions.
**Effort:** Medium (quantile estimation + softmax loss implementation)
**Conflict:** None — preserves gradients from all items (unlike ASL which zeroed them).

---

### 2.2 Temperature-Scheduled Contrastive Loss

**Source:** Kukleva et al., "Temperature Schedules for Self-Supervised Contrastive Methods on Long-Tail Data," ICLR 2023 (arXiv:2303.13664)

**Mechanism:** Cosine schedule cycling τ between high (group-wise discrimination) and low (instance-level discrimination). High-τ phases learn "GT pions are different from background"; low-τ phases force distinction of hardest uncanny-valley negatives.

**Core assumption:** Data has long-tailed distribution where some classes need group-level features, others need instance-level. At 0.27% positive rate, this is extreme long-tail.

**Application:** Replace fixed temperature in contrastive denoising loss: `τ(t) = τ_min + 0.5*(τ_max - τ_min)*(1 + cos(πt/T_cycle))`. Typical: τ_min=0.05, τ_max=0.5, T_cycle=10 epochs.

**Expected impact:** Medium — targets uncanny valley via dynamic hardness. Zero additional computational cost.
**Effort:** Small (~10 lines in training loop)
**Conflict:** None — modifies temperature scheduling, not loss structure.

---

### 2.3 Deferred Reweighting (DRW)

**Source:** Cao et al., "Label-Distribution-Aware Margin Loss," NeurIPS 2019 (arXiv:1906.07413)

**Mechanism:** Two-phase training. Phase 1 (epochs 1-30): standard loss weights, let AE learn good representations. Phase 2 (epochs 30-50): increase ranking loss weight 3-5×, increase negative samples from 50 to 100-150. Key insight: reweighting from the start distorts representations.

**Core assumption:** Model needs good general features before specializing for the rare class. The AE needs to learn the track feature space before the MLP can rank effectively.

**Application:** Conditional on epoch number to change loss weights. No new loss function needed.

**Expected impact:** Medium — resolves tension between reconstruction quality and ranking quality. Phase 1 ensures stable AE representations; Phase 2 aggressively optimizes ranking.
**Effort:** Small (epoch-conditional weight change)
**Conflict:** None — changes loss weights, not structure.

---

### 2.4 Equalized Focal Weighting

**Source:** Li et al., "Equalized Focal Loss for Dense Long-Tailed Object Detection," CVPR 2022 (arXiv:2201.02593)

**Mechanism:** Per-difficulty-bin weighting with a **guaranteed minimum gradient floor** (ε≥0.1). Unlike ASL (which zeroed easy negative gradients), EFL ensures ALL negatives contribute at least ε to the gradient while amplifying hard negatives.

**Core assumption:** Different negative subpopulations need different treatment. Uncanny valley tracks (pT 0.3-0.5, |dxy|<0.5) are much harder negatives than high-pT tracks.

**Application:** Partition negatives by pT/dxy bins. Apply per-bin weighting: `w_bin * (1-p_margin)^γ_bin + ε`. The ε floor is the critical difference from ASL.

**Expected impact:** Medium-High — up-weights hard negatives (uncanny valley) without destroying ranking signal from easy negatives.
**Effort:** Small-Medium (~30-50 lines)
**Conflict:** **Explicitly avoids ASL failure mode.** ε floor guarantees minimum gradient flow vs ASL's clip=0.05 which zeroed them.

---

### 2.5 Physics-Informed Auxiliary Objectives (GN2-style)

**Source:** ATLAS Collaboration, "Transforming jet flavour tagging at ATLAS," Nature Communications 2025 (arXiv:2505.19689)

**Mechanism:** Add lightweight MLP heads on per-track embeddings:
- **Head A:** Track origin classification (pion, kaon, electron, muon, fake, secondary)
- **Head B:** dxy_significance bin prediction (which quartile)

These force the latent space to encode physics-relevant information, improving the primary ranking task by 1.8-3.5× rejection power in GN2.

**Core assumption:** Meaningful physics categories exist for each track. Track origin labels are already in the dataset (`trackorigin` column).

**Application:** Add 2 MLP heads (2 layers each) + 2 cross-entropy losses: `L_total = L_ranking + α*L_recon + β*L_contrastive + γ*L_track_origin + δ*L_dxy_bin`

**Expected impact:** Medium — auxiliary heads regularize latent space and directly teach what tau-pions look like at representation level.
**Effort:** Medium (2 additional MLP heads + losses, track origin labels already available)
**Conflict:** None — auxiliary heads add gradient signal, don't remove it.

---

### 2.6 Curriculum Event Ordering

**Source:** Wang et al., "Dynamic Curriculum Learning for Imbalanced Data Classification," ICCV 2019 (arXiv:1901.06783)

**Mechanism:** Two-level curriculum: sampling scheduler transitions data from easy to hard, loss scheduler adjusts weights. Define event difficulty as fraction of GT pions in uncanny valley.

**Application:** Weighted sampler in DataLoader:
- Epochs 1-15: 70% easy events (0/3 GT pions in uncanny valley), 30% hard
- Epochs 15-30: 50/50
- Epochs 30-50: 30% easy, 70% hard

**Expected impact:** Medium — starts with events where basic patterns are learnable, then gradually introduces degenerate cases. Should reduce the 16.6% of events finding 0/3 GT pions.
**Effort:** Small-Medium (offline difficulty scores + weighted sampler, ~40-60 lines)
**Conflict:** None — changes data ordering, not gradient computation.

---

### 2.7 Masked Track Feature Pretraining

**Source:** Young et al., "Particle Trajectory Representation Learning with Masked Point Modeling," 2025 (arXiv:2502.02558); Golling et al., "Masked Particle Modeling on Sets," 2024 (arXiv:2401.13537)

**Mechanism:** Mask 50% of input features per track, train encoder to reconstruct ALL features. Strictly harder than standard reconstruction — forces learning feature correlations rather than identity mappings. PoLAr-MAE showed 1000× data efficiency (100 labeled events matches 100K supervised).

**Core assumption:** The unlabeled track population (~21M tracks across 19K events) contains useful structure learnable without labels.

**Application:** Phase 1: pretrain with masked reconstruction only. Phase 2: fine-tune with full loss. The encoder/decoder architecture stays the same — just add a masking layer before encoder.

**Expected impact:** Medium — current AE can learn trivial identity-like mappings. Masking forces actual feature distribution learning, creating richer representations.
**Effort:** Medium (masking layer + modified reconstruction loss, ~50-70 lines)
**Conflict:** None — strengthens existing reconstruction objective.

---

## Direction 3: Alternative Paradigms

### 3.1 Object Condensation with GravNet

**Source:** Kieseler, "Object condensation," 2020 (arXiv:2002.03605); Lieret et al., "High Pileup Particle Tracking with OC," 2023 (arXiv:2312.03823); Matousek & Vossen, "OC at CLAS12," 2025 (arXiv:2503.11277)

**Mechanism:** Map every track to a **learned latent clustering space** (2-4 dims), separate from input features. Each track also predicts a condensation charge β. Loss: attractive potential pulls same-object tracks together, repulsive potential pushes different-object tracks apart. At inference, condensation points (high-β tracks) define clusters.

**Core assumption:** Objects (tau-pion groups) form separable clusters in SOME latent space, even if not in original feature space.

**Why this differs from failed experiment #1-2:** Failed experiments injected pairwise Lorentz features INTO GNN edges, collapsing the AE. Object condensation does NOT modify message passing — it adds a separate output head that learns its own abstract geometry. The GNN edges carry whatever features work; the clustering coordinates are learned in a completely decoupled space.

**Application:** Add clustering coordinate head (2-4 dims) + β output (1 dim) to existing GNN. Implement attractive/repulsive potential loss. Open-source implementations available (CERN HGCalML).

**Expected impact:** High — captures relational structure. Even when individual features are ambiguous, the GNN can learn that certain track combinations should cluster together. Extensively validated at CERN.
**Effort:** Medium (new output heads + OC loss function)
**Conflict:** None — architecturally distinct from all 5 failed experiments.

---

### 3.2 Slot Attention for Track Selection

**Source:** Locatello et al., "Object-Centric Learning with Slot Attention," NeurIPS 2020 (arXiv:2006.15055); Fan et al., "Adaptive Slot Attention," CVPR 2024 (arXiv:2406.09196)

**Mechanism:** K=3 learned slot vectors compete for tracks via iterative cross-attention. Attention is normalized over *slots* (not tracks), creating competition — each track is "claimed" by the slot attending most strongly. After T iterations (~3-7), each slot has condensed info about one GT pion.

**Core assumption:** K=3 is known (exactly right). Objects are collectively separable when attended to jointly, even if not individually distinctive.

**Application:** Replace autoencoder+MLP entirely. Input: ~1100 track features (after optional GNN encoding). 3 learned slots iterate cross-attention. Train with Hungarian matching between slot predictions and GT pion assignments.

**Expected impact:** High — instead of scoring tracks independently, slots compete relationally. Iterative refinement lets early iterations find easy pions, later iterations use those as context for hard ones.
**Effort:** Medium (~50 lines for slot attention module + Hungarian loss)
**Conflict:** None — completely replaces the per-track scorer.

---

### 3.3 DETR-Style Set Prediction

**Source:** Carion et al., "DETR," ECCV 2020 (arXiv:2005.12872); Caron et al., "TrackFormers," 2024 (arXiv:2407.07179, published EPJC 2025)

**Mechanism:** Transformer decoder with K=3 learned query vectors cross-attending to encoded tracks. Each query predicts a probability distribution over all tracks (which track is this pion?). Hungarian matching loss between 3 predictions and 3 GT pions.

**Core assumption:** K=3 is known. Encoder representations are sufficiently informative for cross-attention discrimination.

**Application:** Reframes from "score 1100 tracks" to "predict 3 track indices." Each query specializes (e.g., one for high-dxy pions, another for dz-based uncanny valley detection). TrackFormers demonstrated sub-10ms inference at O(10⁴) hits.

**Expected impact:** High — eliminates top-K selection entirely. Each query considers ALL tracks simultaneously.
**Effort:** Medium-Large (transformer decoder + Hungarian loss + extreme imbalance handling in cross-attention)
**Conflict:** None — Hungarian matching is at output level (prediction-to-GT), not input level.

---

### 3.4 Differentiable Top-K Selection

**Source:** Sander et al., "Fast, Differentiable and Sparse Top-k," 2023 (arXiv:2302.01425)

**Mechanism:** Standard `torch.topk` breaks the gradient chain — the scorer never learns whether its ranking was correct for the downstream task. Differentiable top-K reformulates selection as a linear program over the permutahedron with p-norm regularization, solvable in O(n log n).

**Core assumption:** Per-track scores exist and benefit from end-to-end gradient flow through the selection step.

**Application:** Drop-in replacement for `torch.topk()` during training. At inference, revert to hard top-K for speed. Compatible with all scoring approaches.

**Expected impact:** Moderate — makes training signal sharper by directly optimizing recall@200 rather than pairwise margins.
**Effort:** Small (drop-in replacement, open-source implementation available)
**Conflict:** None — training technique, compatible with everything.

---

### 3.5 Learned Task-Adaptive Sampling

**Source:** "PointAS," Frontiers in Computational Neuroscience 2024; "LFPS," ICLR 2025; "LTA-PCS," CVPR 2024

**Mechanism:** Replace geometric FPS in the backbone with attention-based sampling that considers both coverage AND task relevance. LFPS uses a uniformity loss matching FPS coverage + "key point" bias toward informative regions.

**Core assumption:** Spatial/feature patterns distinguish informative from uninformative regions. The sampler can learn to over-sample where GT pions are likely.

**Application:** Replace FPS in HierarchicalGraphBackbone with learned sampler. Subsequent kNN operates on signal-enriched subset.

**Expected impact:** Moderate — geometric FPS may drop uncanny valley pions early (spatially common, not at farthest points).
**Effort:** Small-Medium (replace FPS call with attention network + uniformity regularization)
**Conflict:** None — modifies sampling, not message passing or loss.

---

### 3.6 Prototype-Guided Contrastive Learning

**Source:** "Rebalanced Supervised Contrastive Learning with Prototypes," Computer Vision and Image Understanding 2025; "A Tale of Two Classes," arXiv:2503.17024 (2025)

**Mechanism:** Learn a metric space where pion tracks cluster around a learned prototype, background is pushed away. Running-average prototype for pion class, rebalanced loss so 3 pions contribute as much gradient as ~1100 background tracks, dynamic temperature.

**Core assumption:** GT pions share common structure in learned representation space, even if raw features overlap with background.

**Application:** Replace binary loss with supervised contrastive loss using class prototypes. Prototype = exponential moving average of pion embeddings.

**Expected impact:** Moderate-High — directly addresses 0.27% positive rate. Contrastive learning with prototypes optimizes inter-class distance rather than just classification boundary.
**Effort:** Small (loss function replacement, ~30 lines)
**Conflict:** None — does NOT zero gradients (unlike ASL). Re-weights through temperature and distance.

---

### 3.7 Optimal Transport for Set Matching

**Source:** "Recent Advances in Optimal Transport for Machine Learning," 2023 (arXiv:2306.16156); "Beyond Hungarian," 2025 (arXiv:2603.08514)

**Mechanism:** Differentiable relaxation of Hungarian matching via Sinkhorn algorithm. Computes soft transport plan (doubly-stochastic matrix) between predictions and GT. Fully differentiable unlike Hungarian.

**Core assumption:** Predicted and GT sets can be meaningfully compared via a cost matrix.

**Application:** Use as loss function for set prediction approaches (3.2 or 3.3). Replaces hard Hungarian with smooth Sinkhorn-based OT. Unbalanced OT naturally handles 200 predictions vs 3 GT pions.

**Expected impact:** Moderate — smoother gradients for set prediction training.
**Effort:** Small (~20 lines of PyTorch; geomloss/POT libraries available)
**Conflict:** None — loss function technique, compatible with all architectures.

---

## Cross-Direction Rankings

### By Expected Impact (High → Low)

| Rank | ID | Technique | Direction | Impact | Effort |
|------|----|-----------|-----------|--------|--------|
| 1 | 1.7 | AE as auxiliary loss | Architecture | High | Small |
| 2 | 3.1 | Object condensation | Alt Paradigm | High | Medium |
| 3 | 3.2 | Slot attention | Alt Paradigm | High | Medium |
| 4 | 2.1 | SoftmaxLoss@K | Loss | High | Medium |
| 5 | 1.1 | PNA multi-aggregation | Architecture | Medium-High | Small |
| 6 | 2.4 | Equalized focal weighting | Loss | Medium-High | Small-Med |
| 7 | 3.6 | Prototype contrastive | Alt Paradigm | Medium-High | Small |
| 8 | 1.4 | FiLM conditioning | Architecture | Medium | Small |
| 9 | 2.2 | Temperature scheduling | Loss | Medium | Small |
| 10 | 2.3 | Deferred reweighting | Loss | Medium | Small |
| 11 | 1.5 | Feature group decomposition | Architecture | Medium | Medium |
| 12 | 2.5 | GN2 auxiliary objectives | Loss | Medium | Medium |
| 13 | 2.6 | Curriculum event ordering | Loss | Medium | Small-Med |
| 14 | 3.4 | Differentiable top-K | Alt Paradigm | Moderate | Small |
| 15 | 1.3 | Dynamic graph recomputation | Architecture | Moderate | Small |
| 16 | 2.7 | Masked pretraining | Loss | Moderate | Medium |
| 17 | 3.3 | DETR set prediction | Alt Paradigm | High | Med-Large |
| 18 | 1.2 | Multi-scale kNN | Architecture | Moderate | Medium |
| 19 | 1.6 | Local virtual nodes | Architecture | Moderate | Medium |
| 20 | 3.5 | Learned sampling | Alt Paradigm | Moderate | Small-Med |
| 21 | 3.7 | OT set matching | Alt Paradigm | Moderate | Small |

### By Implementation Effort (Small → Large)

**Small effort (can test in 1 session):**
1.7 AE as auxiliary, 2.2 Temperature scheduling, 2.3 DRW, 1.1 PNA, 1.4 FiLM, 3.6 Prototype contrastive, 3.4 Differentiable top-K, 1.3 Dynamic graph, 3.7 OT matching

**Small-Medium (1-2 sessions):**
2.4 EFL weighting, 2.6 Curriculum ordering, 3.5 Learned sampling

**Medium (2-3 sessions):**
3.1 Object condensation, 3.2 Slot attention, 1.5 Feature groups, 2.5 GN2 auxiliaries, 2.7 Masked pretraining, 1.2 Multi-scale kNN, 1.6 Virtual nodes, 2.1 SoftmaxLoss@K

**Medium-Large:**
3.3 DETR set prediction

---

## Failed Experiment Conflict Matrix

| Failed Experiment | Root Cause | Techniques That Conflict | Techniques That Fix It |
|---|---|---|---|
| #1 Pairwise in MP | AE latent collapse from relational features | None of the 21 | 1.7 (AE as auxiliary) |
| #2 Pairwise as attn bias | Same AE collapse regardless of injection | None of the 21 | 1.7 (AE as auxiliary) |
| #3 Triplet enumeration | Combinatorial explosion (88% pass mass cut) | None of the 21 | 3.1, 3.2 (avoid enumeration entirely) |
| #4 Score propagation | GT pions spread (ΔR=1.3), dilutes 72% | None of the 21 | 1.6 (virtual nodes), 1.3 (dynamic graph) |
| #5 ASL loss | Zeros easy negative gradients | None of the 21 | 2.4 (EFL with ε floor) |

**Key insight:** Technique 1.7 (AE as auxiliary) unblocks future experimentation. With the AE demoted from primary to auxiliary, pairwise physics features could potentially be re-attempted because the latent space is no longer forced to reconstruct through them. This should be validated carefully in a separate experiment after 1.7 is established.

---

## Recommended Implementation Roadmap

### Phase 1: Quick Wins (1 training run, combinable)
These modify orthogonal aspects and can be combined in a single training run:
- **1.7** AE as auxiliary loss (change reconstruction weight from ~1.0 to 0.1)
- **2.2** Temperature-scheduled contrastive (cosine schedule on τ)
- **2.3** Deferred reweighting (increase ranking weight after epoch 30)

### Phase 2: Architecture Upgrades (1-2 training runs)
After Phase 1 establishes the new training regime:
- **1.1** PNA multi-aggregation (replace max-pool)
- **1.4** FiLM conditioning (context-dependent messages)
- **2.4** Equalized focal weighting (hard negative emphasis with ε floor)

### Phase 3: Evaluate Paradigm Shift vs Architecture Extension
Based on Phase 2 diagnostics, choose ONE path:

**Path A — Stay with per-track scoring (if R@200 > 75% after Phase 2):**
- **2.1** SoftmaxLoss@K (directly optimize R@200)
- **1.5** Feature group decomposition
- **2.5** GN2 auxiliary objectives

**Path B — Paradigm shift (if uncanny valley remains problematic):**
- **3.1** Object condensation (most natural fit for HEP, extensive CERN validation)
- **3.2** Slot attention (elegant K=3 match)
- **3.4** Differentiable top-K (compatible with either approach)

### Decision Point
Wait for 13-feature model training results before executing. If dz_significance alone pushes R@200 above 70%, Phases 1-2 may be sufficient. If uncanny valley persists despite dz_significance, prioritize Phase 3 Path B.

---

## References

### Direction 1 — Architecture
- Corso et al., "PNA," NeurIPS 2020, arXiv:2004.05718
- Ohamouddou et al., "MS-DGCNN++," 2025, arXiv:2507.12602
- Wang et al., "DGCNN," ACM TOG 2019, arXiv:1801.07829
- Brockschmidt, "GNN-FiLM," ICML 2020, arXiv:1906.12192
- Manoharan et al., "Parameter-Efficient FiLM," 2025, arXiv:2511.05456
- Qin et al., "MAPI-GNN," AAAI 2026, arXiv:2512.20026
- Shi et al., "PolaDCA," 2026, arXiv:2603.06303
- Karabulut & Baytas, "Local Virtual Nodes," 2025, arXiv:2508.20597
- Qian et al., "IPR-MPNN," NeurIPS 2024, arXiv:2405.17311
- Choma et al., "Exa.TrkX," 2020, arXiv:2007.00149
- Chan et al., "EggNet," 2025, arXiv:2506.03415

### Direction 2 — Loss & Training
- Yang et al., "SL@K," KDD 2025, arXiv:2508.05673
- Kukleva et al., "Temperature Schedules," ICLR 2023, arXiv:2303.13664
- Cao et al., "LDAM/DRW," NeurIPS 2019, arXiv:1906.07413
- Li et al., "Equalized Focal Loss," CVPR 2022, arXiv:2201.02593
- ATLAS Collaboration, "GN2," Nature Communications 2025, arXiv:2505.19689
- Wang et al., "DCL," ICCV 2019, arXiv:1901.06783
- Young et al., "PoLAr-MAE," 2025, arXiv:2502.02558
- Golling et al., "MPM," 2024, arXiv:2401.13537

### Direction 3 — Alternative Paradigms
- Kieseler, "Object Condensation," 2020, arXiv:2002.03605
- Lieret et al., "High Pileup OC Tracking," 2023, arXiv:2312.03823
- Matousek & Vossen, "OC at CLAS12," 2025, arXiv:2503.11277
- Locatello et al., "Slot Attention," NeurIPS 2020, arXiv:2006.15055
- Fan et al., "AdaSlot," CVPR 2024, arXiv:2406.09196
- Carion et al., "DETR," ECCV 2020, arXiv:2005.12872
- Caron et al., "TrackFormers," 2024, arXiv:2407.07179
- Sander et al., "Differentiable Top-K," 2023, arXiv:2302.01425
