# Architecture Literature Survey for Track Pre-Filter

**Date:** 2026-03-20
**Context:** Track pre-filter scores ~1100 tracks/event, selects top-200 candidates (3 signal, rest noise). Current: MLP + 2 rounds kNN(k=16) max-pool, 225K params, R@200=0.62. Target: R@200=0.8-0.9.

---

## Table of Contents

1. [Direction 1: Richer Message Passing / GNNs](#direction-1-richer-message-passing--gnns)
2. [Direction 2: Efficient Attention for Large Sets](#direction-2-efficient-attention-for-large-sets-p1100)
3. [Direction 3: Hierarchical / Cascade Architectures](#direction-3-hierarchical--cascade-architectures)
4. [Bonus: State Space Models (Mamba)](#bonus-state-space-models-mamba)
5. [HEP-Specific Efficient Transformers](#hep-specific-efficient-transformers)
6. [Comparison Table](#comparison-table)
7. [Integration Analysis](#integration-analysis-with-existing-knn-message-passing)
8. [Recommendations](#recommendations)
9. [Sources](#sources)

---

## Direction 1: Richer Message Passing / GNNs

### 1.1 EdgeConv / DGCNN (Dynamic Graph CNN)

**Paper:** Wang et al., "Dynamic Graph CNN for Learning on Point Clouds," ACM TOG 2019 (arXiv:1801.07829). Originally submitted 2018, published 2019.

**Core mechanism:** EdgeConv constructs edge features h_{ij} = MLP(x_i || x_j - x_i) between each point and its k-nearest neighbors, then aggregates via max-pool. The key innovation is *dynamic graph updates*: after each layer, kNN is recomputed in the learned feature space (not fixed input coordinates). This means semantic neighbors (tracks with similar learned representations) replace spatial neighbors as the network deepens.

**Complexity for P=1100:**
- Per layer: kNN construction O(P^2) = O(1.21M), or O(P * k * d) with approximate methods
- Edge feature computation: O(P * k * d_in * d_out) where k=16-20, d ~ 64-128
- Total per layer with k=16: ~1100 * 16 * (MLP ops) = ~17.6K edge evaluations
- Dynamic kNN recomputation adds O(P^2 * d) per layer for exact kNN in feature space
- For P=1100, k=16: total flops per layer ~ 1100 * 16 * 2 * 64 * 64 = ~145M multiply-adds

**Known results:** State-of-the-art on ModelNet40 (93.5% acc) and S3DIS (56.1 mIoU) at time of publication. ParticleNet (Qu & Gouskos 2020) is a direct HEP adaptation.

**Used in particle physics:** YES. ParticleNet (Phys. Rev. D 101, 2020) uses EdgeConv with k=16 as its core building block for jet tagging. Three EdgeConv blocks with channels (64,64,64), (128,128,128), (256,256,256). ParticleNet is the immediate predecessor of the Particle Transformer and remains widely deployed in CMS. The current pre-filter already uses EdgeConv-style message passing.

**Integration with existing kNN:** The current model already uses static kNN + EdgeConv-like aggregation. The main upgrade would be **dynamic graph recomputation** between rounds (recompute kNN in feature space, not just eta-phi). This is a small code change: after round 1, replace `points` with `current` (learned features) in the kNN call. However, feature-space kNN on P=1100 with d=64 is O(P^2 * d) = ~77M ops, which may be acceptable but is more expensive than the current static graph.

---

### 1.2 GATv2 (Graph Attention Network v2)

**Paper:** Brody, Alon, Yahav, "How Attentive are Graph Attention Networks?," ICLR 2022 (arXiv:2105.14491).

**Core mechanism:** GATv2 fixes the "static attention" problem in GAT. In GATv1, the attention ranking of neighbors is fixed regardless of the query node. GATv2 reorders the internal operations: it first concatenates query-key features, then applies a shared linear transformation, then LeakyReLU, then the attention coefficient. This allows *dynamic attention* where the ranking of neighbors can change depending on the query. Mathematically: alpha_{ij} = a^T LeakyReLU(W [h_i || h_j]) (GATv2) vs. LeakyReLU(a^T [Wh_i || Wh_j]) (GATv1).

**Complexity for P=1100:**
- O(|V| * d * d' + |E| * d') where |V|=1100, |E|=P*k=17,600 for k=16
- Same asymptotic complexity as GATv1
- Per layer: ~1100 * 64 * 64 (node transforms) + 17,600 * 64 (attention) = ~5.6M multiply-adds
- Attention adds negligible overhead vs plain EdgeConv because it operates on pre-computed features

**Known results:** Outperforms GAT across 12 benchmarks of node-, link-, and graph-prediction. The improvement is most pronounced when the graph requires dynamic attention (i.e., different nodes genuinely need different neighbor rankings).

**Used in particle physics:** GATv2 is available in PyTorch Geometric and has been used in some HEP prototype studies, but no major published result yet. The original GAT has been explored in ATLAS tracking pipelines.

**Integration with existing kNN:** Replace the max-pool aggregation in each kNN message-passing round with attention-weighted aggregation. Each track computes attention weights over its k=16 neighbors using GATv2's dynamic attention mechanism. This is a drop-in replacement: instead of `neighbor_features.max(dim=-1)`, compute attention-weighted sum. Adds a small attention MLP (~2 * hidden_dim * 1 parameters per head per layer). With the existing kNN graph, complexity increase is minimal. NOTE: The current model already uses attention for Lorentz vector propagation but max-pool for feature aggregation. GATv2 would extend attention to feature aggregation.

---

### 1.3 GraphSAGE with Neighbor Sampling

**Paper:** Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs," NeurIPS 2017 (arXiv:1706.02216).

**Core mechanism:** Instead of using all neighbors, GraphSAGE samples a fixed-size subset of neighbors (S=25 at layer 1, S=10 at layer 2 in the original paper). The aggregation function is applied only to the sampled subset. This bounds the computational cost per node regardless of actual degree. The key insight is inductive learning: the model learns an aggregation function (not node-specific embeddings), so it generalizes to unseen nodes.

**Complexity for P=1100:**
- Per layer: O(P * S * d) where S is the sample budget (not actual degree)
- With S=25 at layer 1, S=10 at layer 2: worst-case ~1100 * 25 * 64 = 1.76M ops at layer 1
- Two-layer computational budget: O(P * S_1 * S_2 * d) = ~1100 * 25 * 10 * 64 = ~176M ops total
- Key advantage: bounded cost even if some nodes have many neighbors

**Known results:** Competitive with GCN on citation networks, significantly better on protein-protein interaction networks (inductive setting). The sampling strategy sacrifices some local accuracy for scalability.

**Used in particle physics:** Not directly in published HEP track finding. The Exa.TrkX pipeline uses interaction networks and GNNs without the sampling trick because graph sparsity (from geometric kNN) already bounds the degree.

**Integration with existing kNN:** Since the current model uses kNN with fixed k=16, the degree is already bounded. GraphSAGE's sampling mechanism would only help if k were increased significantly (e.g., k=64 or k=128). In the current setup, GraphSAGE offers little advantage because the neighborhood size is already fixed by kNN. However, the SAGE aggregation functions (mean, LSTM, pool) could be used as alternative aggregators within the existing framework.

---

### 1.4 Multi-Scale kNN (Parallel Different k Values)

**Paper:** No single canonical paper. Related work: Ohamouddou et al., "MS-DGCNN++," 2025 (arXiv:2507.12602); PointNet++ multi-scale grouping (Qi et al., NeurIPS 2017); Zhai et al., "Multiscale Dynamic GCN," 2023.

**Core mechanism:** Build 2-3 kNN graphs in parallel with different k values (e.g., k=8, k=32, k=64). Each scale has its own EdgeConv branch processing the corresponding neighborhood. Features from all scales are fused (concatenation or attention-weighted combination). The local scale (k=8) captures immediate jet core structure; the broad scale (k=64) captures event-level tau topology.

**Complexity for P=1100:**
- kNN construction for each scale: O(P^2) per scale (all use same pairwise distances)
- Message passing per scale: O(P * k_i * d_in * d_out)
- For k=8,32,64 with d=64: (8+32+64) * 1100 * 2 * 64 * 64 = ~942M multiply-adds
- Memory: 3 parallel graphs + 3 sets of edge features, roughly 3x the current model
- Can share pairwise distance computation across scales (compute once, top-k with different k)

**Known results:** MS-DGCNN++ shows improvement over single-scale DGCNN on ModelNet40. MLGCN (2024) uses multi-level kNN graphs shared among GCN blocks, achieving efficient multi-scale feature extraction. PointNet++ multi-scale grouping showed consistent gains over single-scale.

**Used in particle physics:** The HierarchicalGraphBackbone already uses different k values across stages (k=32,24,16), but these are sequential (not parallel at the same resolution). True parallel multi-scale kNN at the full P=1100 resolution has not been used in published HEP work.

**Integration with existing kNN:** Build multiple kNN graphs from the same eta-phi coordinates (reuse distance matrix). Run separate EdgeConv per scale, concatenate outputs, project back to hidden_dim. The existing `cross_set_knn` function can be called with different `num_neighbors`. Main cost: roughly 2-3x parameters in the message-passing layers.

---

### 1.5 PNA (Principal Neighbourhood Aggregation)

**Paper:** Corso et al., "Principal Neighbourhood Aggregation for Graph Nets," NeurIPS 2020 (arXiv:2004.05718).

**Core mechanism:** Replace single aggregation (e.g., max-pool) with the concatenation of MULTIPLE aggregation functions: {mean, max, min, std} combined with degree scalers {identity, amplification (sqrt(d/d_avg)), attenuation (sqrt(d_avg/d))}. The insight: different aggregation functions capture fundamentally different neighborhood properties (max captures strongest signal, mean captures average behavior, std captures diversity), and no single aggregator dominates across all tasks. The 4 aggregators x 3 scalers = 12 terms, concatenated and projected.

**Complexity for P=1100:**
- Aggregation: O(P * k * d) for each of the 4 aggregators (same as max-pool alone)
- The 4 aggregators are computed from the same gathered neighbor features (no extra kNN)
- Output: 4 * d features per node (or 12 * d with degree scalers), projected to d
- Projection: O(P * 12d * d) for full PNA with scalers
- For P=1100, k=16, d=64: aggregation ~4 * 1100 * 16 * 64 = ~4.5M; projection ~1100 * 768 * 64 = ~54M
- Total per layer: ~58M (vs ~18M for max-pool only, ~3x overhead)

**Known results:** Significantly outperforms GCN, GAT, GIN, MPNN on multi-task graph theory benchmark. The paper reports -3.13 average log10 MSE vs -2.53 for the next best (MPNN). Most impactful on tasks requiring distributional information.

**Used in particle physics:** Not directly in published HEP work. However, the current pre-filter already has a PNA mode (`aggregation_mode='pna'`) that implements cat([mean, max, min, std]) without the degree scalers. This was added in an earlier iteration.

**Integration with existing kNN:** Already partially implemented in the codebase (see `_pna_aggregate` in `TrackPreFilter.py`). Full PNA would add degree scalers, which require computing the average degree of the kNN graph (trivially k for a fixed-k graph, unless masking creates variable degrees). The degree scalers add negligible compute. The existing implementation uses 5 * hidden_dim (current + 4 aggregators) before projection.

---

### 1.6 GIN (Graph Isomorphism Network)

**Paper:** Xu et al., "How Powerful are Graph Neural Networks?," ICLR 2019 (arXiv:1810.00826).

**Core mechanism:** GIN uses sum aggregation with a learnable epsilon parameter: h_v = MLP((1 + epsilon) * h_v + SUM_{u in N(v)} h_u). The theoretical contribution is that GIN is provably as powerful as the Weisfeiler-Lehman (WL) graph isomorphism test, making it maximally expressive among standard GNNs. Sum is the critical choice: it preserves multiset information that mean and max aggregation lose.

**Complexity for P=1100:**
- Per layer: O(P * k * d) for aggregation + O(P * d * d') for MLP
- For P=1100, k=16, d=64: ~1100 * 16 * 64 (sum) + 1100 * 64 * 64 (MLP) = ~5.6M
- Slightly cheaper than EdgeConv because no edge feature construction
- Total for 2 layers: ~11M multiply-adds

**Known results:** Achieves highest or near-highest accuracy on standard graph classification benchmarks (MUTAG, PTC, PROTEINS, COLLAB). The advantage over GCN/GAT is most pronounced on tasks requiring distinguishing non-isomorphic graph structures.

**Used in particle physics:** GIN has been explored in some HEP contexts but is less popular than EdgeConv/ParticleNet for jet tagging. The reason: GIN's strength is *graph-level* classification (distinguishing entire graphs), while track pre-filtering is a *node-level* task (scoring individual tracks). For node-level tasks, the theoretical advantage of sum-over-mean is less clear.

**Integration with existing kNN:** Replace EdgeConv with GIN: use sum aggregation of neighbor features (instead of max-pool of edge features). Removes the edge feature construction (concat + difference) which simplifies the architecture. The epsilon parameter is a single learnable scalar per layer. Implementation: change `max_pooled = neighbor_features.max(dim=-1)[0]` to `sum_pooled = (neighbor_features * neighbor_validity).sum(dim=-1)` and add epsilon scaling. Very simple change, but loses the relative (x_j - x_i) edge feature that EdgeConv provides.

---

## Direction 2: Efficient Attention for Large Sets (P=1100)

### 2.1 Set Transformer with ISAB

**Paper:** Lee et al., "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks," ICML 2019 (arXiv:1810.00825).

**Core mechanism:** The Induced Set Attention Block (ISAB) reduces self-attention from O(n^2) to O(n*m) using m trainable inducing points. Two cross-attention steps: (1) inducing points attend to input set (m x n attention), producing m summarized vectors; (2) input set attends to the m summaries (n x m attention). This bottleneck forces the model to compress set-level information through m representative vectors. The inducing points are trainable parameters (not data-dependent), learned end-to-end.

**Complexity for P=1100:**
- Standard self-attention: O(P^2 * d) = O(1.21M * d) = ~77.4M with d=64
- ISAB with m=32 inducing points: O(P * m * d) = O(1100 * 32 * 64) = ~2.3M per block
- ISAB with m=64: O(1100 * 64 * 64) = ~4.5M per block
- Memory: O(P * m) instead of O(P^2) -> 35.2K vs 1.21M entries
- Speed-up factor vs full attention: P/m = 1100/32 = 34x faster

**Known results:** On point cloud classification (n=100-5000), Set Transformer with m=16 inducing points *outperformed* the full Set Transformer, likely due to regularization through the bottleneck. On amortized clustering, consistent improvements over DeepSets.

**Used in particle physics:** Yes. "Point Cloud Transformers applied to Collider Physics" (Mikuni & Canelli, 2021, arXiv:2102.05073) and "Particle Hit Clustering and Identification Using Point Set Transformers in Liquid Argon TPCs" (2025, arXiv:2504.08182) directly apply Set Transformers to HEP data. The ISAB variant is particularly suited to the variable-size, permutation-invariant nature of particle sets.

**Integration with existing kNN:** ISAB replaces (or augments) the kNN message passing entirely. Instead of local message passing via kNN, ISAB provides *global* set-level attention through inducing points. Could be used as: (a) replacement for kNN rounds entirely, (b) global attention layer after local kNN rounds, or (c) hybrid where kNN provides local context and ISAB provides global context. Option (c) is the most promising: keep the kNN EdgeConv for local geometry, add ISAB for global event-level patterns. The m=32 inducing points would learn to represent event-level track categories (signal-like, background-like, etc.).

---

### 2.2 Perceiver / Perceiver IO

**Paper:** Jaegle et al., "Perceiver: General Perception with Iterative Attention," ICML 2021 (arXiv:2103.03206). Follow-up: "Perceiver IO: A General Architecture for Structured Inputs & Outputs" (arXiv:2107.14795).

**Core mechanism:** The Perceiver uses a latent array of N << P vectors that cross-attends to the full input. Only Q (queries) come from the latent array (size N), while K,V come from the input (size P). This makes the attention matrix N x P instead of P x P. The latent array then undergoes self-attention (N x N, cheap when N << P). Multiple iterations of cross-attention + self-attention allow progressive refinement. Perceiver IO adds output queries that cross-attend to the latent array, enabling task-specific outputs.

**Complexity for P=1100:**
- Cross-attention: O(N * P * d) where N is latent size (e.g., N=64)
- Self-attention in latent: O(N^2 * d) = O(4096 * d)
- Total per iteration: O(N * P * d + N^2 * d) = O(64 * 1100 * 64 + 4096 * 64) = ~4.8M
- For T=4 iterations: ~19M multiply-adds
- Comparison: full self-attention on P=1100 would be ~77M. Perceiver is ~4x cheaper.
- Memory: O(N * P + N^2) = O(70,400 + 4,096) = ~75K entries vs O(P^2) = 1.21M

**Known results:** Competitive with specialized models across images, audio, video, and point clouds. On ModelNet40: comparable to PointNet++, slightly below DGCNN. The architecture trades some task-specific performance for generality.

**Used in particle physics:** Perceiver IO was applied in the HuggingFace Perceiver for multi-modal understanding. In HEP specifically, not yet widely adopted, but the architecture is conceptually similar to DETR-style approaches already explored for tracking (TrackFormers, ATLAS GNN4ITk).

**Integration with existing kNN:** The Perceiver can sit on top of the kNN message-passing layers. After 2 rounds of local kNN EdgeConv produce per-track features (B, 64, 1100), a small latent array (e.g., N=32-64 vectors) cross-attends to these track features. The latent array then self-attends. Per-track output scores are obtained by having each track cross-attend back to the latent array (Perceiver IO style). This provides global reasoning on top of the local kNN features without O(P^2) cost.

---

### 2.3 Performer / Linear Attention

**Paper:** Choromanski et al., "Rethinking Attention with Performers," ICLR 2021 (arXiv:2009.14794).

**Core mechanism:** Performers approximate softmax attention using random feature maps (FAVOR+). The key insight: softmax(QK^T) can be approximated by phi(Q) * phi(K)^T where phi is a random feature map using positive orthogonal random features. By changing the order of matrix multiplication from (Q * K^T) * V to Q * (K^T * V), the complexity drops from O(P^2 * d) to O(P * r * d) where r is the number of random features (typically r ~ d).

**Complexity for P=1100:**
- Standard attention: O(P^2 * d) = O(1.21M * 64) = ~77M
- Performer with r=64 features: O(P * r * d) = O(1100 * 64 * 64) = ~4.5M
- Speed-up: ~17x for P=1100
- Memory: O(P * r + r * d) instead of O(P^2) -> 70K + 4K vs 1.21M

**Known results:** FAVOR+ provides unbiased estimation of the attention matrix with uniform convergence guarantees. In practice, the approximation quality degrades for sharp attention patterns (when few keys get most of the attention weight). Performance is within 1-2% of exact attention on most NLP tasks but can degrade more on tasks requiring precise sparse attention.

**Used in particle physics:** Not directly published in HEP. Linear attention variants have been explored in the broader point cloud community.

**Integration with existing kNN:** Could replace any self-attention layer if one were added to the architecture. However, for P=1100, even exact self-attention is feasible (~77M ops is not prohibitive for modern GPUs), so the benefit is marginal. More useful if P were larger (>5000). For the current problem size, ISAB or Perceiver provide better accuracy-efficiency tradeoffs than the Performer approximation.

---

### 2.4 Neighborhood Attention (NAtten)

**Paper:** Hassani et al., "Neighborhood Attention Transformer," CVPR 2023 (arXiv:2204.07143).

**Core mechanism:** Neighborhood Attention (NA) restricts each token's attention to its k nearest neighbors in a sliding-window pattern. Unlike window self-attention (Swin Transformer), the window is centered on each token and the receptive field grows naturally across layers. NA has linear time and space complexity O(P * k * d) where k is the neighborhood size. Efficient C++ and CUDA kernels (NATTEN package) run up to 40% faster than Swin's WSA with 25% less memory.

**Complexity for P=1100:**
- O(P * k * d) where k is the attention neighborhood size
- With k=16 (same as kNN): O(1100 * 16 * 64) = ~1.1M per layer
- This is equivalent to the kNN message-passing complexity
- Memory: O(P * k) = 17,600 entries per attention layer

**Known results:** State-of-the-art on ImageNet classification and ADE20K segmentation for its parameter budget. However, NA was designed for grid-structured data (images) where "neighborhood" is defined by spatial coordinates on a regular grid.

**Used in particle physics:** Not directly. The main limitation for HEP is that NAtten assumes a regular grid structure for efficient kernel implementation. Point clouds (and particle sets) are irregular, so the optimized CUDA kernels cannot be directly used.

**Integration with existing kNN:** Conceptually, the current kNN + attention-weighted aggregation IS a form of neighborhood attention. The difference is that NAtten's CUDA kernels are optimized for grid patterns, which don't apply to irregular kNN graphs. To use NAtten on particle data, one would need to serialize the points (as in PTv3's approach), losing the irregular structure. For P=1100, the overhead of kNN-based attention is small enough that NAtten's kernel optimizations are not necessary.

---

### 2.5 Point Cloud Transformer (PCT)

**Paper:** Guo et al., "PCT: Point Cloud Transformer," Computational Visual Media 2021 (arXiv:2012.09688).

**Core mechanism:** Three innovations: (1) **Coordinate-based input embedding** via point-wise MLP, (2) **Offset-attention**: the output is computed as LBR(F_in - F_sa) + F_in, where the offset is the element-wise difference between input features and self-attention output, forming a residual attention mechanism, (3) **Neighbor embedding**: before global attention, a local kNN-based aggregation (using Euclidean distance) injects local geometric information that global attention alone might miss.

**Complexity for P=1100:**
- Neighbor embedding: O(P * k * d) for kNN aggregation ~ 1.1M with k=16
- Self-attention (4 layers): O(P^2 * d * 4) = O(1.21M * 64 * 4) = ~310M
- Total: ~311M multiply-adds
- The self-attention dominates. For P=1100, this is feasible but expensive.

**Known results:** State-of-the-art on ModelNet40 (93.2%) and ShapeNet part segmentation (86.4 mIoU) at time of publication. The combination of local (kNN) and global (attention) features is consistently better than either alone.

**Integration with existing kNN:** PCT's architecture mirrors the proposed hybrid: kNN for local context + global attention for event-level patterns. The offset-attention mechanism is a simple modification to standard self-attention that could be added after the kNN rounds. However, the O(P^2) self-attention on P=1100 may be too expensive for a pre-filter. A Perceiver-style bottleneck (Section 2.2) or ISAB (Section 2.1) would be a better-scaling alternative to PCT's full self-attention.

---

### 2.6 Flash Attention for Sparse Patterns

**Paper:** Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention," NeurIPS 2022 (arXiv:2205.14135). Follow-ups: FlashAttention-2 (2023), FlashAttention-3 (2024).

**Core mechanism:** FlashAttention is not an approximation but an IO-aware implementation of exact attention. It tiles the Q, K, V matrices to fit in GPU SRAM, avoids materializing the full P x P attention matrix in HBM, and fuses the softmax + matmul into a single kernel. This provides 2-4x speedup and significant memory savings (O(P) instead of O(P^2) in memory). Block-sparse variants support sparse attention patterns (e.g., only attending to kNN neighbors).

**Complexity for P=1100:**
- FlashAttention on full P=1100: still O(P^2 * d) compute, but 2-4x faster wall-clock due to IO optimization
- Memory: O(P) instead of O(P^2) for the attention matrix
- Block-sparse FlashAttention with kNN sparsity pattern: O(P * k * d), truly linear
- For P=1100, k=16: ~1.1M multiply-adds per layer with sparse patterns

**Known results:** FlashAttention-2 achieves up to 2.4x speedup over FlashAttention-1. FlashAttention-3 further optimizes for FP8 and asynchronous processing. Block-sparse attention supports various patterns including streaming and kNN.

**Used in particle physics:** Not directly in published HEP work, but Flash Attention is used in the general ML ecosystem and could accelerate any transformer layer applied to particle data.

**Integration with existing kNN:** For P=1100, Flash Attention makes full self-attention feasible. The P^2 = 1.21M attention matrix fits in modern GPU SRAM (a few MB), so FlashAttention would make full self-attention practical. However, this requires a CUDA GPU (Flash Attention has no MPS implementation). For local development on MPS, the kNN-sparse variant would be needed. The most practical integration: after kNN rounds, add one layer of full self-attention using FlashAttention for GPU training, fall back to ISAB for MPS local runs.

---

## Direction 3: Hierarchical / Cascade Architectures

### 3.1 PointNet++ (Set Abstraction + Feature Propagation)

**Paper:** Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space," NeurIPS 2017 (arXiv:1706.02413).

**Core mechanism:** Hierarchical set abstraction with three components: (1) **Sampling layer**: Farthest Point Sampling (FPS) selects M centroids from P points, (2) **Grouping layer**: ball query or kNN finds neighbors around each centroid, (3) **PointNet layer**: per-group MLP + max-pool produces M features. Feature Propagation reverses this via inverse-distance interpolation + skip connections. Multi-scale grouping (MSG) processes each centroid with multiple radii/k values in parallel.

**Complexity for P=1100:**
- FPS: O(P * M) where M is number of centroids. For M=256: ~281K iterations
- kNN grouping: O(P * M) for computing distances between all points and centroids
- Per stage (P=1100 -> M=256): FPS ~281K + kNN ~281K + MLP ~256 * k * d_in * d_out
- Full 3-stage pipeline (1100->256->128->64): dominated by first stage FPS
- FPS is the bottleneck: 88-94% of inference time in PointNet++ models

**Known results:** State-of-the-art at publication for 3D classification and segmentation. MSG variant handles varying point density well. The hierarchical approach is robust and well-understood.

**Used in particle physics:** YES. The current HierarchicalGraphBackbone (`weaver/weaver/nn/model/HierarchicalGraphBackbone.py`) IS a PointNet++ variant with 3 set abstraction stages (1100->256->128->64), FPS in (eta, phi), kNN grouping, and EdgeConv aggregation. It adds physics-informed edge features (pairwise Lorentz-vector features) not present in standard PointNet++.

**Integration with existing kNN:** Already fully integrated. The HierarchicalGraphBackbone is the downstream model that receives the pre-filter's top-200 tracks. The question is whether PointNet++ ideas can improve the pre-filter itself (which operates at full P=1100 without downsampling). The relevant PointNet++ technique for the pre-filter is **feature propagation**: after downsampling to a small set, propagate refined features back to ALL original points via inverse-distance interpolation + skip connections. This would give each of the 1100 tracks access to hierarchically-refined global context.

---

### 3.2 Point Transformer V2

**Paper:** Wu et al., "Point Transformer V2: Grouped Vector Attention and Partition-based Pooling," NeurIPS 2022 (arXiv:2210.05666).

**Core mechanism:** Two innovations over Point Transformer V1: (1) **Grouped Vector Attention (GVA)**: groups attention channels (like multi-head but with vector-valued attention instead of scalar), more expressive than scalar attention. (2) **Partition-based Pooling**: divides the point cloud into spatial partitions for pooling, better preserving spatial alignment than FPS-based downsampling. Also adds a position encoding multiplier to strengthen position information in attention.

**Complexity for P=1100:**
- GVA per layer: O(P * k * d) where k is the neighborhood size (similar to standard kNN attention)
- Partition-based pooling: O(P) for spatial binning + O(M * k * d) for pooled attention
- Total for hierarchical processing: dominated by the first (largest) stage
- For P=1100, k=16, d=64: ~1.1M per attention layer
- Multiple stages (1100->256->64): ~1.1M + 0.26M + 0.065M = ~1.4M per pass

**Known results:** State-of-the-art on ScanNet (75.4 mIoU) and S3DIS at publication. Superseded by PTv3 in 2024.

**Used in particle physics:** Not directly. The grouped vector attention concept is applicable to particle physics but hasn't been adopted in published HEP work.

**Integration with existing kNN:** GVA can replace the current max-pool aggregation. Instead of per-channel max over neighbors, compute attention weights grouped across channel subsets (e.g., 4 groups of 16 channels each). Each group gets its own vector-valued attention weight. This is more expressive than scalar attention (GAT/GATv2) because different channel groups can attend to different neighbors. Partition-based pooling could replace FPS in the downstream backbone.

---

### 3.3 Point Transformer V3

**Paper:** Wu et al., "Point Transformer V3: Simpler, Faster, Stronger," CVPR 2024 Oral (arXiv:2312.10035).

**Core mechanism:** PTv3 prioritizes simplicity and scale over intricate design. Key idea: **serialization-based attention**. Points are serialized using space-filling curves (Z-order, Hilbert), then grouped into patches. Attention operates within and between patches using a "Shuffle Order" mechanism. This replaces kNN search entirely with O(P) serialization. The receptive field is expanded from 16 (kNN) to 1024 points while achieving 3x speed and 10x memory improvement over PTv2.

**Complexity for P=1100:**
- Serialization: O(P log P) for sorting along space-filling curves
- Patch grouping: O(P) for chunking the serialized sequence into patches of size w
- Intra-patch attention: O(n_patches * w^2 * d) where w is patch size
- With P=1100, w=64: n_patches=17, attention per patch = 64^2 * 64 = 262K, total = ~4.5M
- Cross-patch interaction (Shuffle Order): O(P * d) per shift
- Total: ~5-10M per layer, significantly cheaper than full P^2 attention
- Key insight: kNN search accounts for 28% of forward time in PTv2; serialization eliminates this

**Known results:** State-of-the-art on multiple 3D benchmarks. 3.3x faster and 10.2x lower memory than PTv2. The paper's thesis is that scaling (larger models, larger receptive fields) matters more than architectural cleverness.

**Used in particle physics:** Not directly. The serialization approach is promising for particle physics because track ordering in (eta, phi) can be achieved via space-filling curves, but the fixed-grid assumption may not perfectly match the variable-density track distributions in particle physics events.

**Integration with existing kNN:** PTv3's serialization could replace kNN entirely. Order tracks by Hilbert curve in (eta, phi), group into patches of ~64 tracks, apply attention within patches. This provides 64-point receptive fields with O(P) cost instead of the current k=16 via kNN. The Shuffle Order mechanism provides cross-patch communication. For P=1100, this gives ~17 patches, which is manageable. However, the physics-informed edge features (Lorentz-vector pairs) would need adaptation since there are no explicit edges in the serialized attention.

---

### 3.4 Cascaded Classifiers with Progressive Narrowing

**Paper:** Viola & Jones (2001, original cascaded classifier); Cascade R-CNN (Cai & Vasconcelos, CVPR 2018); general principle.

**Core mechanism:** A cascade of progressively more expensive classifiers. Stage 1 (cheap): eliminates obvious negatives (e.g., MLP-only, cuts 1100 to 400). Stage 2 (moderate): kNN message passing on the reduced set (400 to 200). Stage 3 (expensive, optional): attention or GNN on the 200 candidates. Each stage has its own threshold, tightened progressively.

**Complexity for P=1100:**
- Stage 1 (MLP): O(P * d * d') = O(1100 * 64 * 64) = ~4.5M
- Stage 2 (kNN on 400): O(400 * 16 * 64 * 64) = ~26M (much cheaper than kNN on 1100)
- Stage 3 (attention on 200): O(200^2 * 64) = ~2.6M (full attention is now feasible!)
- Total: ~33M vs ~70M+ for a single expensive model on all 1100 tracks
- Key advantage: the expensive operations only run on the reduced set

**Known results:** Cascade R-CNN improved AP by 2-4 points over single-stage detectors on COCO. The GNN4ITk pipeline at ATLAS uses a 3-stage cascade: graph construction, GNN edge classification (loose threshold), walk-through algorithm (tighter threshold). Their per-edge efficiency is >98% for pT < 10 GeV.

**Used in particle physics:** YES. The ATLAS GNN4ITk pipeline is exactly this pattern: build graph (stage 1), classify edges with GNN (stage 2, loose score threshold 0.08-0.09), build track candidates with walk-through (stage 3, tighter requirements). The current two-stage pipeline (pre-filter -> Particle Transformer) is already a cascade.

**Integration with existing kNN:** The current pre-filter IS stage 1 of a cascade. To make the pre-filter itself a cascade: add a cheap MLP-only pre-pre-filter that cuts 1100 to ~400 (no kNN, just per-track features). Then the kNN message passing operates on 400 tracks (cheaper kNN, fewer edges). This reduces kNN cost by (400/1100)^2 = 0.13x for the distance matrix, and 400/1100 = 0.36x for message passing. The MLP pre-stage would need very high recall (>99%) to avoid dropping signal tracks.

---

### 3.5 Stratified Transformer

**Paper:** Lai et al., "Stratified Transformer for 3D Point Cloud Segmentation," CVPR 2022 (arXiv:2203.14508).

**Core mechanism:** For each query point, sample nearby points densely and distant points sparsely as keys. This "stratified key sampling" enables long-range context at low cost. Also includes: first-layer point embedding for aggregating local information (aids convergence), and contextual relative position encoding. The stratified sampling enlarges the effective receptive field without O(P^2) attention.

**Complexity for P=1100:**
- Stratified sampling per query: k_near (dense, e.g., 16) + k_far (sparse, e.g., 8) = 24 keys
- Attention per query: O(24 * d) = O(24 * 64) = 1536 ops
- Total: O(P * 24 * d) = O(1100 * 24 * 64) = ~1.7M per layer
- Similar to kNN(k=24) in cost, but with long-range connections
- Sampling: O(P) per query for distance-based stratification

**Known results:** State-of-the-art on S3DIS (72.0 mIoU) and ScanNetv2 at publication. The stratified key sampling is particularly effective for large scenes where distant context matters (e.g., room-scale understanding).

**Used in particle physics:** Not directly. The concept maps well to particle physics: nearby tracks share local jet structure (dense sampling), while distant tracks may share event-level tau topology (sparse sampling).

**Integration with existing kNN:** Can be implemented as an extension of the current kNN: for each track, take 16 nearest neighbors (existing kNN) + sample 8 random distant tracks (uniform sample from tracks outside the k=64 neighborhood). The additional 8 distant connections provide long-range context. The EdgeConv MLP would process both local and distant edges identically. Cost increase: ~50% more edges per track.

---

### 3.6 U-Net Style Architectures for Point Clouds

**Paper:** PointNet++ (Qi et al., NeurIPS 2017) is the original U-Net for point clouds. Recent: U-Net with attention skip blocks (2025, Neurocomputing).

**Core mechanism:** Encoder-decoder with skip connections. Encoder: hierarchical downsampling (FPS + kNN + MLP, as in PointNet++). Decoder: interpolation-based upsampling + skip connections from encoder. The skip connections allow fine-grained input information to flow directly to the decoder, preventing loss of detail during downsampling. Recent work adds attention mechanisms to the skip connections for selective information transfer.

**Complexity for P=1100:**
- Encoder (same as PointNet++ hierarchical backbone): ~5-10M
- Decoder (upsampling + skip connections): ~3-5M for inverse-distance interpolation
- Total: ~8-15M, roughly 2x the encoder-only cost
- Skip connections: O(P_stage * d_stage) per stage for feature concatenation

**Known results:** U-Net designs consistently outperform encoder-only models for per-point tasks (segmentation, labeling). The skip connections are critical for maintaining fine-grained spatial information.

**Used in particle physics:** The HierarchicalGraphBackbone is the encoder half. Adding a decoder with skip connections would create a full U-Net. Point cloud U-Net variants have been explored for 3D segmentation in autonomous driving but not specifically in HEP track scoring.

**Integration with existing kNN:** The pre-filter needs per-track scores for ALL 1100 tracks. A U-Net approach would: (1) encode 1100 -> 256 -> 128 -> 64 with the existing backbone stages, (2) decode 64 -> 128 -> 256 -> 1100 via interpolation + skip connections, (3) apply the scoring head to the decoded 1100-dimensional features. This gives every track access to hierarchical multi-scale context while producing per-track scores. The decoder adds ~50-80% more parameters. This approach is particularly interesting because the downstream backbone (HierarchicalGraphBackbone) already implements the encoder; the pre-filter could share it and add only the decoder.

---

## Bonus: State Space Models (Mamba)

### Mamba for Point Clouds

**Papers:** Mamba3D (2024, arXiv:2404.14966); Point Mamba (2024, arXiv:2403.06467); PointMamba (2024, arXiv:2402.10739); Voxel Mamba (2024, arXiv:2406.10700).

**Core mechanism:** Mamba (Gu & Dao, 2023) is a selective state space model (SSM) that processes sequences with O(P) complexity and linear memory. For point clouds, the key challenge is serialization: converting unordered 3D points to a 1D sequence. Methods include Z-order curves, Hilbert curves, and octree-based ordering. Mamba3D enhances local features via a Local Norm Pooling technique. The selective scan mechanism allows content-dependent information propagation along the serialized sequence.

**Complexity for P=1100:**
- Serialization: O(P log P) for sorting
- Mamba scan: O(P * d * state_dim) where state_dim is typically 16
- For P=1100, d=64, state_dim=16: ~1100 * 64 * 16 = ~1.1M per layer
- Memory: O(P * state_dim) = O(17,600) per layer -- extremely efficient
- Comparison: O(P * k * d) for kNN = 1.1M (similar compute, less memory)

**Known results:** Mamba3D achieves 92.6% on ScanObjectNN (train from scratch) and 95.1% on ModelNet40 (with pre-training). FLOPs scale linearly with sequence length, unlike quadratic for transformers.

**Used in particle physics:** Not yet. The serialization challenge is the main barrier: particle tracks don't have a natural sequential ordering, and the physics is fundamentally permutation-invariant.

**Integration with existing kNN:** Mamba could replace or augment the kNN message passing. After embedding all 1100 tracks, serialize them via Hilbert curve in (eta, phi) and run a Mamba scan. This provides O(P) global context. The challenge: Mamba's sequential scan introduces a position-dependent bias (earlier tokens influence later ones differently), which conflicts with the permutation invariance of particle physics. Mitigation: use bidirectional Mamba (two scans, forward + backward) or multiple scans with different serialization orders (as in PTv3).

---

## HEP-Specific Efficient Transformers

### HEPT (LSH-Based Efficient Point Transformer)

**Paper:** Miao et al., "Locality-Sensitive Hashing-Based Efficient Point Transformer with Applications in High-Energy Physics," ICML 2024 Oral (arXiv:2402.12535). Follow-up: "HEPT for Charged Particle Reconstruction," 2025 (arXiv:2510.07594).

**Core mechanism:** Uses Locality-Sensitive Hashing (E2LSH with OR & AND constructions) to build sparse attention patterns. Quantitative analysis of error-complexity tradeoff across sparsification methods (kNN, random, learned) shows LSH is superior for point clouds with local inductive bias. The resulting attention pattern is regular (hash-bucket-based grouping), enabling efficient GPU computation without irregular memory access.

**Complexity for P=1100:**
- LSH bucketing: O(P * L * K_hash) where L=number of hash tables, K_hash=hash functions per table
- Attention within buckets: O(P * bucket_size * d)
- Typical bucket_size ~ sqrt(P) ~ 33, so per-layer: O(P * sqrt(P) * d) = O(P^1.5 * d)
- For P=1100, d=64: ~1100 * 33 * 64 = ~2.3M per layer
- Near-linear in P, hardware-friendly regular operations

**Known results:** Significantly outperforms existing GNNs and transformers in accuracy AND computational speed on two HEP tasks: track reconstruction and jet tagging. Presented as ICML 2024 Oral, indicating high impact.

**Used in particle physics:** YES. Specifically designed for HEP. The 2025 follow-up applies it to charged particle reconstruction, directly relevant to track finding.

**Integration with existing kNN:** HEPT could replace both the kNN graph construction AND the message-passing aggregation. Instead of building kNN edges and running EdgeConv, hash tracks into buckets via LSH in (eta, phi, features) space and apply attention within buckets. This provides automatic multi-scale context (nearby points are more likely to hash together, but distant similar points also have a chance). The OR & AND construction controls the sparsity-accuracy tradeoff. Code is available at github.com/Graph-COM/HEPT.

---

## Comparison Table

| Approach | Key Paper | Complexity (P=1100, d=64) | Params vs Current | Known HEP Use | Global Context | Integration Effort | Expected Impact on R@200 |
|---|---|---|---|---|---|---|---|
| **Direction 1: Message Passing** ||||||||
| EdgeConv/DGCNN dynamic graph | Wang 2019, ACM TOG | O(P*k*d) + O(P^2*d) per recompute | +0% (arch change) | YES (ParticleNet) | 2-hop via feature kNN | Small (change kNN source) | Low-Medium |
| GATv2 | Brody 2022, ICLR | O(P*k*d') ~ 5.6M/layer | +~5K (attention MLP) | Prototypes only | No (local kNN) | Small (drop-in aggregation) | Low-Medium |
| GraphSAGE | Hamilton 2017, NeurIPS | O(P*S*d) ~ 1.8M/layer | +0% (aggregation change) | No | No (sampling bounds) | Minimal | Low |
| Multi-scale kNN | Various 2023-2025 | O(P*sum(k_i)*d) ~ 7M/layer | +2-3x (parallel branches) | No | Broader local (k=64) | Medium (parallel branches) | Medium |
| PNA | Corso 2020, NeurIPS | O(P*4k*d) + O(P*12d*d) ~ 58M/layer | +~30K (projection) | No (already in code) | No (local kNN) | **Already implemented** | Medium |
| GIN | Xu 2019, ICLR | O(P*k*d) + O(P*d*d') ~ 5.6M/layer | -10% (simpler) | Prototype only | No | Small (aggregation change) | Low |
| **Direction 2: Efficient Attention** ||||||||
| Set Transformer (ISAB m=32) | Lee 2019, ICML | O(P*m*d) ~ 2.3M/block | +~50K (inducing points) | YES (HEP collider) | **YES (global via m points)** | Medium (new module) | **High** |
| Perceiver (N=64 latent) | Jaegle 2021, ICML | O(N*P*d + N^2*d) ~ 4.8M/iter | +~100K (latent array) | Similar to DETR | **YES (latent bottleneck)** | Medium (new module) | **High** |
| Performer (linear attn) | Choromanski 2021, ICLR | O(P*r*d) ~ 4.5M/layer | +~10K (random features) | No | YES (approx global) | Medium (kernel approx) | Medium |
| NAtten | Hassani 2023, CVPR | O(P*k*d) ~ 1.1M/layer | +0% | No | No (local) | Hard (needs grid) | Low |
| PCT | Guo 2021 | O(P^2*d) ~ 77M/layer | +~200K | No | **YES (full attention)** | Medium (self-attn layers) | Medium-High |
| Flash Attention (sparse) | Dao 2022-24, NeurIPS | O(P*k*d) ~ 1.1M/layer | +0% | No | Depends on pattern | Small (kernel swap) | Low (engineering) |
| **HEPT (LSH)** | **Miao 2024, ICML Oral** | **O(P^1.5*d) ~ 2.3M/layer** | **+~50K** | **YES (HEP-specific)** | **YES (hash buckets)** | **Medium (new module)** | **High** |
| **Direction 3: Hierarchical/Cascade** ||||||||
| PointNet++ (feat propagation) | Qi 2017, NeurIPS | O(P*M) FPS + O(M*k*d) | +~50K (decoder) | YES (current backbone) | YES (hierarchical) | Medium (add decoder) | **High** |
| Point Transformer V2 (GVA) | Wu 2022, NeurIPS | O(P*k*d) ~ 1.1M/layer | +~20K (group attention) | No | No (local) | Medium (new attention) | Medium |
| Point Transformer V3 | Wu 2024, CVPR Oral | O(P*w*d) ~ 5M/layer | +~100K (serialization) | No | YES (1024 receptive field) | Large (new paradigm) | High |
| Cascaded classifiers | General principle | Stage1 O(P*d^2) + Stage2 O(P'*k*d^2) | +~50K (extra stage) | YES (ATLAS GNN4ITk) | Progressive refinement | Medium (multi-stage) | **Medium-High** |
| Stratified Transformer | Lai 2022, CVPR | O(P*24*d) ~ 1.7M/layer | +~20K | No | YES (sparse distant) | Medium (stratified sampling) | Medium |
| U-Net point cloud | Qi 2017 + recent | O(encoder + decoder) ~ 15M | +~100K (decoder) | No | YES (skip connections) | Medium (add decoder path) | **High** |
| **Bonus** ||||||||
| Mamba3D | Various 2024 | O(P*d*s) ~ 1.1M/layer | +~30K (SSM state) | No | YES (sequential scan) | Medium-Large (serialization) | Medium |

---

## Integration Analysis with Existing kNN Message Passing

The current pre-filter uses this pipeline:
```
Input(B, 13, P) -> track_mlp(13->64) -> [kNN(k=16) -> gather -> max-pool -> cat(current, pooled) -> mlp(128->64)] x 2 rounds -> scorer(64->1)
```

### Tier 1: Drop-in enhancements (change aggregation, keep kNN)

These modify the aggregation step within the existing kNN message-passing rounds:

1. **PNA** (already implemented as `aggregation_mode='pna'`): cat(mean, max, min, std) instead of max-pool. Test first since it's already in the code.
2. **GATv2**: attention-weighted aggregation instead of max-pool. Add a small attention MLP per round.
3. **GIN**: sum aggregation with learnable epsilon. Simpler but loses edge features (x_j - x_i).
4. **Dynamic graph recomputation**: recompute kNN in feature space between rounds. Changes neighbors, not aggregation.

### Tier 2: Add global context module after kNN rounds

These add a new module after the 2 kNN rounds to provide event-level context:

5. **ISAB (m=32)**: inducing points compress P=1100 tracks into 32 summaries, then broadcast back. O(P*m*d).
6. **Perceiver (N=64)**: latent array cross-attends to track features, then tracks cross-attend back. O(N*P*d).
7. **HEPT**: LSH bucketing + attention within hash buckets. O(P^1.5*d). HEP-validated.
8. **Stratified sampling**: 16 kNN neighbors + 8 random distant tracks per point. ~50% more edges.

### Tier 3: Architectural redesign

These require more substantial changes to the pre-filter architecture:

9. **U-Net with feature propagation**: encode 1100->256->64, decode 64->256->1100 with skip connections. Requires building a decoder path.
10. **Cascaded classifier**: MLP-only pre-pre-filter (1100->400), then kNN on 400 tracks. Requires careful threshold tuning.
11. **PTv3 serialization**: replace kNN entirely with Hilbert-curve serialization + patch attention. Major paradigm change.
12. **Multi-scale parallel kNN**: 2-3 parallel kNN branches with k=8,32,64. Roughly 2-3x parameters.

---

## Recommendations

### Priority 1: Test existing PNA mode (already implemented)

The `aggregation_mode='pna'` parameter in TrackPreFilter is already coded but its performance vs max-pool on the current 13-feature model should be measured. This is zero additional implementation effort.

### Priority 2: Add ISAB global context after kNN rounds

The Set Transformer's ISAB with m=32 inducing points provides global context at O(P*m*d) cost. This directly addresses a fundamental limitation of the current architecture: with 2 rounds of kNN(k=16), the receptive field is at most k^2=256 tracks out of ~1100. The ISAB allows every track to receive information from the entire event through the 32 inducing point bottleneck. This has been validated in HEP collider physics settings. Implementation: ~50-80 lines of PyTorch code for the ISAB module, added after the kNN rounds and before the scorer.

### Priority 3: HEPT (LSH-based attention)

The HEPT architecture is the most directly relevant result: it was designed specifically for HEP, presented as an ICML 2024 Oral, and demonstrated superior performance on track reconstruction tasks. The LSH-based attention provides near-linear complexity while maintaining global context. Open-source code is available. The main implementation effort is adapting the bucketing to the (eta, phi) + feature space of the current model.

### Priority 4: U-Net feature propagation

Adding a decoder path with skip connections would give every track access to hierarchical multi-scale context. The encoder already exists (HierarchicalGraphBackbone). The decoder adds inverse-distance interpolation and skip connections. This is particularly natural because the downstream task (per-track scoring) requires features at the full P=1100 resolution, which is exactly what U-Net decoders produce.

### Not recommended for now

- **Performer/Linear Attention**: For P=1100, even exact attention is feasible. The approximation quality loss is not worth it at this scale.
- **NAtten**: Requires grid structure; not applicable to irregular particle sets without serialization overhead.
- **Mamba**: Promising but introduces sequential bias that conflicts with permutation invariance. The serialization challenge is non-trivial and not yet proven in HEP.
- **Full PCT self-attention**: At O(P^2*d) = ~310M for 4 layers, this is expensive for a pre-filter. Use ISAB or HEPT instead for cheaper global context.

---

## Sources

### Direction 1: Message Passing / GNNs
- [DGCNN / EdgeConv - Wang et al. 2019](https://arxiv.org/abs/1801.07829)
- [DGCNN Official Repository](https://github.com/WangYueFt/dgcnn)
- [GATv2 - Brody et al. ICLR 2022](https://arxiv.org/abs/2105.14491)
- [GATv2 Code](https://github.com/tech-srl/how_attentive_are_gats)
- [GraphSAGE - Hamilton et al. NeurIPS 2017](https://arxiv.org/abs/1706.02216)
- [PNA - Corso et al. NeurIPS 2020](https://arxiv.org/abs/2004.05718)
- [PNA PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.PNA.html)
- [GIN - Xu et al. ICLR 2019](https://arxiv.org/abs/1810.00826)
- [ParticleNet - Qu & Gouskos 2020](https://arxiv.org/abs/1902.08570)
- [MLGCN Multi-Level kNN](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2024.1439340/full)

### Direction 2: Efficient Attention
- [Set Transformer / ISAB - Lee et al. ICML 2019](https://arxiv.org/abs/1810.00825)
- [Perceiver - Jaegle et al. ICML 2021](https://arxiv.org/abs/2103.03206)
- [Perceiver IO](https://arxiv.org/abs/2107.14795)
- [Performer / FAVOR+ - Choromanski et al. ICLR 2021](https://arxiv.org/abs/2009.14794)
- [Neighborhood Attention (NAtten) - Hassani et al. CVPR 2023](https://arxiv.org/abs/2204.07143)
- [NAtten Library](https://natten.org/)
- [PCT - Guo et al. 2021](https://arxiv.org/abs/2012.09688)
- [FlashAttention - Dao et al. NeurIPS 2022](https://arxiv.org/abs/2205.14135)
- [Block Sparse Attention](https://github.com/mit-han-lab/Block-Sparse-Attention)
- [HEPT - Miao et al. ICML 2024 Oral](https://arxiv.org/abs/2402.12535)
- [HEPT Code](https://github.com/Graph-COM/HEPT)
- [HEPT for Charged Particle Reconstruction](https://arxiv.org/abs/2510.07594)
- [Point Cloud Transformers in Collider Physics](https://arxiv.org/abs/2102.05073)
- [Particle Hit Clustering with Point Set Transformers](https://arxiv.org/abs/2504.08182)
- [Mamba-based Linear Attention for Point Cloud Segmentation](https://arxiv.org/abs/2410.21211)

### Direction 3: Hierarchical / Cascade
- [PointNet++ - Qi et al. NeurIPS 2017](https://arxiv.org/abs/1706.02413)
- [Point Transformer V2 - Wu et al. NeurIPS 2022](https://arxiv.org/abs/2210.05666)
- [Point Transformer V3 - Wu et al. CVPR 2024](https://arxiv.org/abs/2312.10035)
- [PTv3 Code](https://github.com/Pointcept/PointTransformerV3)
- [Stratified Transformer - Lai et al. CVPR 2022](https://arxiv.org/abs/2203.14508)
- [U-Net with Attention Skip for Point Cloud](https://www.sciencedirect.com/science/article/abs/pii/S0925231225002656)
- [Particle Transformer - Qu et al. ICML 2022](https://arxiv.org/abs/2202.03772)

### HEP Track Finding with GNNs
- [ATLAS GNN4ITk Track Reconstruction](https://www.epj-conferences.org/articles/epjconf/abs/2024/05/epjconf_chep2024_03030/epjconf_chep2024_03030.html)
- [Exa.TrkX Project](https://exatrkx.github.io/)
- [Hierarchical GNN for Particle Track Reconstruction](https://arxiv.org/abs/2303.01640)
- [GNN for Charged Particle Tracking on FPGAs](https://arxiv.org/abs/2112.02048)
- [GNN Track Finding in LHCb Vertex Detector](https://cds.cern.ch/record/2919388)

### Bonus: State Space Models
- [Mamba3D](https://arxiv.org/abs/2404.14966)
- [Point Mamba](https://arxiv.org/abs/2403.06467)
- [PointMamba](https://arxiv.org/abs/2402.10739)
- [Voxel Mamba](https://arxiv.org/abs/2406.10700)

### Differentiable Selection
- [Differentiable Top-K with Optimal Transport - NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)
- [Differentiable Fast Top-K Selection](https://arxiv.org/abs/2510.11472)
