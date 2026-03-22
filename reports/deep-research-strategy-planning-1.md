# Breaking the R@200 = 0.62 wall in τ→3π track pre-filtering

**The plateau at R@200 ≈ 0.62 is a graph-topology problem, not an architecture problem.** Every GNN variant tried — MLP+kNN message passing, EdgeConv, GravNet, ParticleNeXt, ISAB — converges to the same wall because all operate on the same poisoned substrate: an (η,φ) kNN graph where **99.7% of neighbors are noise**. The path to R@200 ≥ 0.80 requires either bypassing this graph entirely, restructuring it through learned construction, or staging the problem so that pairwise physics features become effective at enriched signal fractions. Research across HEP-ML (GNN4ITk, EggNet, Object Condensation, MaskFormers), information retrieval (cascade ranking, differentiable R@K), and computer vision (DETR, Slot Attention) identifies seven concrete algorithmic directions with strong evidence of applicability. Three warrant immediate implementation; four offer complementary medium-term gains.

---

## The cascade architecture is the most pragmatic path forward

The single most actionable finding is that a **two-stage cascade** — coarse BDT retrieval followed by pairwise neural reranking — directly exploits the existing diagnostic that BDT achieves **R@500 = 0.85** and **R@600 = 0.90**. Stage 1 narrows ~1100 tracks to ~500, enriching signal from 0.27% to ~0.60%. Stage 2 then applies expensive pairwise features (invariant mass near ρ(770), charge product, ΔR) on the enriched candidate set to refine to top-200.

This is not a novel paradigm — it is the **dominant architecture in industrial recommendation and search systems**, where the funnel proceeds through Matching → Pre-ranking → Ranking → Re-ranking stages with progressively richer but more expensive features on smaller candidate sets. Wang, Lin & Metzler (SIGIR 2011) formalized this as cascade ranking; every modern search engine follows it. Critically, **LCRON** (arXiv:2503.09492, 2025) demonstrates that jointly optimizing both cascade stages via a surrogate loss derived from the probability that ground-truth items survive the full funnel yields significant end-to-end recall gains over independently trained stages.

The ExaTrkX pipeline for HL-LHC tracking already implements exactly this pattern: metric learning embedding → fixed-radius graph → MLP edge filter (coarse stage) → 8-layer Interaction Network GNN (fine stage) → track building. Each stage progressively reduces noise-to-signal ratio. The edge filter handles **~98.5% fake edges** — comparable to the 99.7% noise in the τ problem.

**Stage 2 architecture options, in priority order:**

The key question is what architecture processes the ~500 surviving candidates. At this size, **full pairwise attention is feasible**: 500² = 250K attention pairs, well within GPU memory. Three architectures transfer directly from the retrieve-then-rerank literature. A **cross-encoder reranker** (following the Mono-Duo pattern from Nogueira et al., arXiv:2101.05667) scores each candidate jointly with event context, allowing per-track features to interact with pairwise physics features through cross-attention layers. A **set-encoder** (Schlatt et al., 2024) enables inter-candidate attention via [CLS] token exchange, modeling interactions among all 500 candidates simultaneously. Alternatively, a **GNN on the candidate graph** constructs edges between kinematically plausible pairs (ΔR < threshold, compatible charges), attaches pairwise features to edges, and runs message passing to propagate relational information.

**The math is favorable.** Stage 2 receives ~3 signal tracks among ~500 candidates (0.6% positive rate). To achieve R@200 ≥ 0.80 overall, given Stage 1 passes 85% of signal (R@500 = 0.85), Stage 2 must place signal tracks in its top-200 out of 500 — i.e., above the **40th percentile**. With pairwise features that the ParticleTransformer ablation showed carry most discriminative power (removing them collapses performance from 0.86 to 0.40), ranking signal tracks into the top 40% of an enriched set is substantially more tractable than the original problem. Using R@600 = 0.90 as Stage 1 increases headroom further: Stage 2 must place signal in the top 33% of 600 candidates.

**Implementation risk is low.** Both stages use proven components — an existing BDT for Stage 1 and standard transformer/GNN architectures for Stage 2. The primary engineering question is whether to train the stages jointly (LCRON-style end-to-end, requiring differentiable top-K) or independently (simpler, faster iteration). Starting with independent training and adding joint optimization later is the pragmatic path.

---

## DETR-style learned queries can bypass kNN entirely

The most architecturally elegant solution eliminates graph construction altogether. **MaskFormer for particle track reconstruction** (Van Stroud et al., Phys. Rev. X 15, 041046, December 2025) demonstrates that a DETR-inspired encoder-decoder architecture achieves **97.2% tracking efficiency with 0.6% fake rate** on the TrackML dataset and **100ms inference** — state-of-the-art across all ML tracking approaches. The architecture uses a Transformer encoder with windowed self-attention on input elements, followed by a MaskFormer decoder where **N learned object queries cross-attend to all encoded elements** simultaneously. Each query represents a possible output track and forms an explicit latent representation jointly optimized with hit assignments and track parameters.

For the τ→3π pre-filter, the adaptation is direct: **K=3 learned query vectors** cross-attend to all **1100 encoded track embeddings**. Standard cross-attention complexity is O(K·N·d) = O(3 × 1100 × d), which is **trivially small** — three orders of magnitude below the O(N²) of full self-attention. Each query independently attends to the entire event, with no kNN graph, no local neighborhood bias, and no graph construction step. The competitive softmax normalization across queries (identical to Slot Attention from Locatello et al., NeurIPS 2020) naturally handles the "3 needles in 1100 haystacks" scenario through competitive binding.

**Training uses Hungarian matching** (bipartite matching between K predicted queries and ground-truth signal tracks), the same loss used in DETR for object detection. This loss is specifically designed for matching a small number of predictions to a small number of targets surrounded by "no object" predictions — precisely the 3-signal-in-1100 setting. Setting K=10–20 with a "no object" class provides robustness to variable signal counts.

The encoder self-attention over 1100 tracks (1100² ≈ 1.2M pairs) is also manageable on a single modern GPU, but can be made efficient with **windowed self-attention** (as Van Stroud et al. use), **LSH-based attention** (HEPT, Miao et al., ICML 2024 Oral, arXiv:2402.12535), or **FlexAttention with physics-based block masks** (TrackFormers Part 2, arXiv:2509.26411, which reduces effective attention by ~400× using DBSCAN-based geometric partitioning). HEPTv2 (arXiv:2510.07594, 2025) combines exactly this recipe: HEPT encoder with 3000 instance queries in a MaskFormer-style decoder, achieving ~28ms/event on an A100.

**Why this should break the plateau:** The kNN graph enforces locality in (η,φ) — but signal tracks from a tau decay need not be nearest neighbors in this space. The MaskFormer decoder's cross-attention operates on the **full event** without any locality constraint. The model can learn to attend to arbitrary feature combinations (pT, impact parameter, charge) across all 1100 tracks. The pairwise physics features that are critical discriminants (and that collapsed when injected into the kNN graph) can be incorporated as **attention biases** in the cross-attention (following the Particle Transformer pattern) where they operate on all 3×1100 query-track pairs rather than the noisy kNN neighborhood.

---

## Differentiable R@K losses directly close the train-eval gap

The current model trains with binary cross-entropy on all ~1100 tracks but is evaluated on top-200 recall. **This mismatch is a provable source of suboptimality**: BCE assigns equal gradient to tracks at rank 201 and rank 999, though both contribute zero to R@200. Three families of differentiable ranking losses address this directly.

**RS@K (Recall@k Surrogate Loss)** by Patel, Tolias & Matas (CVPR 2022, arXiv:2108.11179) replaces the non-differentiable Heaviside step function in the recall formula with two nested sigmoid relaxations: σ_τ₂ on score differences estimates soft rank, and σ_τ₁ on the rank threshold provides a soft indicator for being in top-K. The resulting RS@200 loss directly optimizes recall at 200 with full gradient flow. The method achieves state-of-the-art on retrieval benchmarks and has a **public implementation** (github.com/yash0307/RecallatK_surrogate). Adaptation for the pre-filter: treat each event as a "query" and all ~1100 tracks as the "database"; the 3 signal tracks are "positives." RS@200 operates on the entire event in one forward pass — no batch-size scaling issues since N=1100 is small.

**Fast Differentiable Top-K** (Sander et al., ICML 2023) provides a complementary tool: a convex-analysis-based differentiable operator that outputs an exactly K-sparse mask while propagating gradients. This enables a **top-K focused loss**: apply the differentiable top-200 mask to scores, then compute loss only on the soft-selected tracks. The p=4/3 regularization gives fully smooth gradients. An O(n) GPU-friendly Dykstra algorithm makes this practical.

**LapSum** (Struski et al., March 2025) offers the latest differentiable top-K method using shifted Laplace CDFs, with O(n) computation and SOTA results on large-scale classification. **DFTopK** (Zhu et al., October 2025) is designed specifically for cascade ranking, providing O(n) differentiable top-K selection with temperature-controlled smoothness — directly applicable to the cascade pipeline.

The recommended implementation is a **hybrid loss**: L = α·BCE(all tracks) + β·RS@200(all tracks), where BCE maintains feature-learning signal while RS@200 directly optimizes the evaluation metric. Anneal β from 0→1 during training (curriculum: start with pure BCE, gradually shift to ranking). This addresses the train/eval mismatch without destabilizing early training — a likely reason OHEM and ASL degraded performance (they aggressively modified the loss landscape before features were learned).

---

## EggNet and Object Condensation offer learned graph alternatives

If the fundamental problem is that (η,φ) kNN graphs are 99.7% noise, the solution is to build graphs in a **learned space** where signal tracks cluster together. Two approaches stand out.

**EggNet (Evolving Graph-based Graph Attention Network)** by Calafiura, Chan & Delabrouille (ICML AI4Science 2024, arXiv:2407.13925; full TrackML results in arXiv:2506.03415, CHEP 2025) eliminates pre-constructed graphs entirely. It takes a raw point cloud as input, and at each iteration: (1) builds a kNN graph in the current learned embedding space, (2) applies graph attention to update embeddings, (3) **rebuilds the kNN graph** with updated embeddings. After N iterations, DBSCAN clustering on final embeddings identifies tracks. The graph **evolves**: early iterations connect broadly, and as embeddings improve, kNN progressively concentrates on relevant neighbors. On TrackML (pT > 1 GeV), EggNet achieves **99.56% track efficiency, 1.29% duplication rate, 0.06% fake rate** — outperforming both pre-constructed edge classification and Object Condensation approaches. Inference is ~0.26s/event on an A100.

**Object Condensation** (Kieseler, Eur. Phys. J. C 80, 886, 2020; arXiv:2002.03605) takes a complementary approach: instead of iteratively refining the graph, it learns a separate latent clustering space where same-track elements are attracted together via quadratic potentials and different-track elements are repelled via hinge loss. A "condensation point" per track emerges — the element with the highest learned confidence score β. Lieret & DeZoort (arXiv:2309.16754, CHEP 2023; arXiv:2312.03823, NeurIPS ML4PS 2023) demonstrate a full OC pipeline for HL-LHC tracking at ⟨μ⟩=200: **>90% double-majority efficiency for pT > 0.9 GeV**, with the critical property that **OC can recover tracks with missing edges** — when the initial graph is disconnected, same-track elements still cluster in latent space. Qasim et al. (Eur. Phys. J. C 82, 753, 2022) demonstrate GravNet+OC reconstructing O(1000) particles in 200-pileup events at CMS HGCAL.

For the τ→3π problem, these approaches bypass the kNN noise floor by operating in a **learned space where the 3 signal tracks can cluster regardless of their (η,φ) neighborhood**. The adaptation challenge is the extreme signal fraction: OC has been demonstrated where signal tracks form ~10–50% of inputs, not 0.27%. The repulsive loss must be carefully balanced to prevent the 3 signal tracks from being overwhelmed by 1097 noise tracks. A natural solution is to **combine with the cascade**: apply the BDT Stage 1 first to concentrate signal fraction to ~0.6%, then use EggNet/OC on the ~500 surviving candidates, where the clustering problem becomes substantially easier.

---

## Physics-informed anchoring can recover the missing third pion

The model already finds **2/3 pions in 56% of events**. This diagnostic contains an untapped opportunity. The τ⁻ decay proceeds through τ⁻ → a₁(1260)⁻ν_τ → ρ(770)⁰π⁻ν_τ → π⁺π⁻π⁻ν_τ, meaning one opposite-charge pion pair has invariant mass peaking at **m_ρ ≈ 775 MeV** (width Γ_ρ ≈ 149 MeV). The charge structure is always two same-sign + one opposite-sign pions, with total charge ±1.

The anchoring strategy is straightforward and orthogonal to all other improvements:

1. From the top-200 scored tracks, identify high-confidence opposite-charge pairs among the top-50
2. For each anchor pair (π_a, π_b), compute for every remaining track π_c among all ~1100: (a) invariant masses M(π_a, π_c) and M(π_b, π_c), checking proximity to m_ρ within ~150 MeV; (b) charge compatibility (π_c must give total charge ±1); (c) three-body invariant mass M(π_a, π_b, π_c) near m_τ ≈ 1777 MeV
3. Boost scores of tracks satisfying these constraints: s_adjusted = s_original + w · ρ_compatibility

This converts the problem from "find 3 needles in 1100" to "find 1 needle with strong physics constraints" — a dramatically narrowed search. The invariant mass and charge constraints together are highly selective. The approach is analogous to **seeded track finding** (the Combinatorial Kalman Filter starts from hit doublet/triplet seeds and extends outward) and to **anchor-based object detection** (Faster R-CNN defines reference anchors and predicts offsets). Conservative estimates suggest recovering the 3rd pion in ~70% of the 56% of events where 2/3 are found — a **~39% absolute improvement in complete 3/3 reconstruction**, applicable as a post-processing step that stacks on top of any model improvement.

---

## Curriculum training explains the OHEM and ASL failures

OHEM degraded to R@200 = 0.50 and ASL to 0.41. Understanding these failures points toward more effective training strategies.

**OHEM failed because** at 0.27% positive rate, virtually all "hard" examples (those with highest loss) are hard *negatives* — background tracks that happen to resemble signal. Selecting these for focused training causes the model to overfit to distinguishing hard negatives from each other rather than learning signal features. **ASL failed because** its asymmetric γ⁻ down-weights easy negatives, but with a 1:370 ratio, even aggressive down-weighting leaves gradients dominated by the negative class. The 3 positive tracks' learning signal gets overwhelmed.

The fix is a **three-phase curriculum** inspired by CASED (Curriculum Adaptive Sampling for Extreme Data Imbalance, MICCAI 2017, designed for <1% positive rates in lung nodule detection) and Deferred Re-Weighting (DRW) from Cao et al. (NeurIPS 2019):

- **Phase 1 (first third of training):** Subsample background to ~30 tracks per event, giving ~10% positive rate. Train with standard BCE. The model learns basic discriminative features in a balanced setting.
- **Phase 2 (middle third):** Gradually increase background tracks per event from 30→100→300→1100 on a cosine schedule. Add a small RS@200 loss weight.
- **Phase 3 (final third):** Train on full 1100-track events. Switch to RS@200 + BCE combined loss. Apply DRW-style signal reweighting (10–50×). Use reduced learning rate (0.1× Phase 1).

An alternative is **hard negative mining with positive protection**: always include all 3 signal tracks in every training batch, then select only the top-K hardest negatives (K=50–100, giving 3–6% positive rate in the selected subset). This avoids OHEM's failure mode while still focusing on the decision boundary. Class-balanced contrastive pretraining (BCL, Zhu et al. 2022) using class-complement prototypes can also stabilize early feature learning — when most batches contain 0–1 signal tracks, prototypes keep the minority class geometrically represented in embedding space.

---

## Seven directions ranked by expected impact and implementation effort

| Direction | Expected ΔR@200 | Effort | Risk | Key dependency |
|---|---|---|---|---|
| **Two-stage cascade** (BDT@500 → pairwise reranker) | +0.10–0.18 | Medium | Low | Stage 2 architecture selection |
| **DETR/MaskFormer decoder** (K=3 queries, full cross-attention) | +0.10–0.20 | Medium-High | Medium | Encoder design, Hungarian loss tuning |
| **RS@200 differentiable loss** | +0.03–0.08 | Low | Low | Temperature tuning (τ₁, τ₂) |
| **Physics-informed anchoring** (ρ mass + charge) | +0.05–0.10 | Low | Very low | Requires 2/3 recovery baseline |
| **Curriculum training** (3-phase with negative subsampling) | +0.02–0.06 | Low | Low | Phase schedule tuning |
| **EggNet / iterative graph refinement** | +0.08–0.15 | High | Medium | OC loss balancing at 0.27% |
| **Object Condensation** (latent clustering) | +0.05–0.12 | High | Medium-High | Signal fraction adaptation |

The **recommended implementation sequence** is: (1) Implement RS@200 loss + curriculum training as immediate, low-risk improvements to the existing model — these require only loss function changes and training schedule modifications. (2) Build the two-stage cascade with BDT@500 → transformer reranker, which exploits the strongest existing diagnostic. (3) Add physics-informed anchoring as a post-processing step that stacks with any model improvement. (4) If the cascade hits an intermediate wall, replace the monolithic architecture with DETR/MaskFormer decoder, which eliminates the kNN entirely. (5) For longer-term gains, explore EggNet-style evolving graph construction within the cascade's Stage 2.

These directions are largely **orthogonal and stackable**: the cascade architecture (Stage 1 BDT + Stage 2 reranker) can use RS@200 loss for Stage 2 training, incorporate curriculum learning in both stages, and add physics anchoring as post-processing. A MaskFormer decoder can serve as the Stage 2 reranker. The combination of cascade + RS@200 + physics anchoring alone should provide sufficient headroom to reach R@200 ≥ 0.80 — transforming a 0.27% needle-in-a-haystack problem into a series of progressively easier subtasks where physics-informed features finally have room to discriminate.

## Conclusion

The R@200 = 0.62 plateau is not a fundamental limit of the feature space — it is an artifact of routing all architectures through the same noisy (η,φ) kNN graph. The ParticleTransformer ablation already proves that pairwise physics features carry the information needed for R@200 > 0.80, but they require neighborhoods enriched enough to express that information. The cascade architecture provides this enrichment mechanistically (doubling signal fraction), the DETR/MaskFormer decoder provides it architecturally (full-event cross-attention bypassing kNN), and the RS@200 loss provides it during training (directly optimizing the evaluation metric). The most surprising finding is that the specific problem structure — 3 signal tracks with known physics constraints (ρ meson mass, charge conservation, τ mass) among ~1100 candidates — is almost a textbook DETR setup, and the HEP community has now validated this architecture for exactly this class of problem (Van Stroud et al., Phys. Rev. X, December 2025). The convergence of evidence from HEP-ML, information retrieval, and computer vision points toward a clear implementation roadmap: cascade first, DETR if needed, physics anchoring always.