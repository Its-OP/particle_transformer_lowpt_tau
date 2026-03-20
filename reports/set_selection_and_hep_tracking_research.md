# Literature Research: Set Prediction / Subset Selection (Direction 4) and Domain-Specific HEP Approaches (Direction 5)

**Context:** Pre-filter for low-pT tau track finding. ~1100 tracks/event, select top-200 candidates that must contain ~3 signal tracks among ~1100 noise. Current: MLP + kNN, R@200 = 0.62, target 0.8-0.9.

---

## Direction 4: Set Prediction / Subset Selection Models

### 4.1 Slot Attention

**Paper:** Locatello et al., "Object-Centric Learning with Slot Attention," NeurIPS 2020. [arXiv:2006.15055]

**Core mechanism:** Iterative competitive attention between a fixed set of learned "slots" and input features. Each iteration applies softmax-normalized attention *over slots* (not over inputs), forcing slots to compete for binding to input elements. This competition causes slots to specialize on distinct objects. The process runs for T iterations (typically 3-7), refining slot-object assignments.

**Scale tested:** Small synthetic scenes (CLEVR-like) with up to ~10 objects. Image inputs, not point clouds.

**Relevance to our problem:** Moderate-to-low. Slot Attention is designed for *decomposing* a scene into K objects, not for *selecting* a subset from N candidates. We would need to repurpose it: use ~3 signal slots that compete to bind to signal tracks. However:
- The mechanism assumes each slot captures one object; we need 3 slots to find 3 signal tracks among 1100.
- The softmax-over-slots normalization means each input attends to its best-matching slot -- this is object *assignment*, not *selection*.
- Not tested at scales near 1100 inputs.

**Adaptation idea:** Use K=3 signal slots + background slot. Each slot cross-attends to all 1100 tracks. Slot-track attention scores could serve as selection probabilities. Complexity: O(K * N * d) = O(3 * 1100 * d), very cheap.

**Known limitations:** Struggles with complex textures, not tested on large sets, relies on good initialization (usually random or learned).

---

### 4.2 DETR Family (Query-Based Detection)

#### 4.2.1 DETR

**Paper:** Carion et al., "End-to-End Object Detection with Transformers," ECCV 2020. [arXiv:2005.12872]

**Core mechanism:** Encoder processes input features (from CNN backbone) with self-attention. Decoder uses N learned "object queries" that cross-attend to encoder output. Each query independently predicts a detection (class + box). Training uses Hungarian matching between predictions and ground truth for bipartite assignment -- each GT object matches exactly one query.

**Scale tested:** COCO dataset, ~100 objects max per image, typically ~10-50 detections.

**Relevance to our problem:** HIGH. The query mechanism is directly applicable:
- Use K=200 learned queries, each outputting a selection score for one track candidate.
- Cross-attention: queries attend to all 1100 tracks to select the most relevant ones.
- Hungarian matching ensures each signal track is assigned to exactly one query.
- Complexity: O(N_queries * N_tracks * d) = O(200 * 1100 * d), manageable.

**Key insight:** DETR queries learn *what* to look for, not *where*. In our setting, queries would learn to identify signal-like tracks by attending to track features + local context.

**Known limitations:** Slow convergence (500 epochs on COCO), poor performance on small objects, quadratic self-attention in encoder is O(N^2) on input. The slow convergence has been addressed by successors.

#### 4.2.2 DN-DETR

**Paper:** Li et al., "DN-DETR: Accelerate DETR Training by Introducing Query DeNoising," CVPR 2022. [arXiv:2203.01305]

**Core mechanism:** Addresses DETR's slow convergence by adding an auxiliary denoising task: during training, noised ground-truth boxes are fed as additional queries alongside learned queries. The model simultaneously learns to (a) detect objects with learned queries and (b) denoise corrupted GT queries. This stabilizes the bipartite matching and provides easier learning signal early in training.

**Performance:** +1.9 AP over DETR baseline, or comparable accuracy in 50% fewer epochs.

**Relevance:** The denoising idea is directly transferable. During training, we could inject noised versions of signal-track features as auxiliary queries, helping the model learn what signal tracks look like before the learned queries can discover them on their own.

#### 4.2.3 DINO

**Paper:** Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes," arXiv 2022. [arXiv:2203.03605]

**Core mechanism:** Extends DN-DETR with (1) contrastive denoising (positive and negative noise levels), (2) mixed query selection (initializing queries from encoder features, not random), (3) look-forward-twice for iterative box refinement. Achieves 63.3 AP on COCO, state-of-the-art.

**Relevance:** The "mixed query selection" idea is particularly relevant: rather than random query initialization, use the highest-scoring tracks from a lightweight pre-scorer as initial query positions. This is essentially attention-refined rescoring.

---

### 4.3 Deep Sets

**Paper:** Zaheer et al., "Deep Sets," NeurIPS 2017. [arXiv:1703.06114]

**Core mechanism:** Proves that any permutation-invariant function on sets can be decomposed as rho(SUM(phi(x_i))). Architecture: per-element MLP phi, sum/mean pooling, set-level MLP rho. Handles variable-size sets naturally.

**Scale tested:** Population statistics, point clouds (ModelNet40 with ~2k points), set expansion. Not tested at 1000+ scale for selection tasks.

**Relevance to our problem:** LOW for selection, but foundational. Deep Sets processes sets but outputs a single set-level prediction. It cannot directly select a subset because it applies the same phi to every element independently (no inter-element interaction). However:
- Per-track scoring (phi -> score) is essentially what our current MLP does.
- Deep Sets lacks the inter-track context that kNN provides.
- The sum-decomposition theorem shows the expressivity limitation: without interactions, the model can only score tracks based on individual features.

**Known limitations:** No pairwise interactions, limited expressivity for tasks requiring relational reasoning.

---

### 4.4 Set Transformer

**Paper:** Lee et al., "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks," ICML 2019. [arXiv:1810.00825]

**Core mechanism:** Extends Deep Sets with attention. Key innovations:
- **ISAB (Induced Set Attention Block):** Uses m learned "inducing points" to reduce self-attention from O(N^2) to O(N*m). Input attends to inducing points, then inducing points attend back to input. This is the same principle as Perceiver.
- **PMA (Pooling by Multihead Attention):** Learned seed vectors cross-attend to the set to produce fixed-size output.

**Complexity:** O(N * m * d) where m is number of inducing points (typically 16-128), vs O(N^2 * d) for full self-attention.

**Relevance to our problem:** HIGH. This is architecturally very close to what we need:
- ISAB with m=32-64 inducing points could capture inter-track context for 1100 tracks at linear cost.
- PMA with K=200 seed vectors could directly output 200 selection scores.
- The inducing points act as a learned "summary" of the track set, enabling contextual scoring.

**Comparison to our current approach:** Our MLP+kNN gives local context via kNN neighbors. Set Transformer with ISAB gives *global* context via inducing points, potentially capturing event-level patterns (e.g., "this event has high track multiplicity, so signal tracks are harder to find").

---

### 4.5 Perceiver / Perceiver IO

**Paper:** Jaegle et al., "Perceiver: General Perception with Iterative Attention," ICML 2021. [arXiv:2103.03206]
**Paper:** Jaegle et al., "Perceiver IO: A General Architecture for Structured Inputs & Outputs," ICLR 2022. [arXiv:2107.14795]

**Core mechanism:** Cross-attention from a small learned latent array (M tokens, M << N) to a large input array (N tokens). The latent array is then refined with self-attention (cheap, O(M^2)). Perceiver IO adds output cross-attention: output queries attend to the latent array to produce structured outputs.

**Scale tested:** ImageNet (50k pixels), point clouds, audio, video. Demonstrated on inputs with 50,000+ elements.

**Relevance to our problem:** HIGH. The Perceiver architecture directly addresses our scale challenge:
- Input: 1100 tracks with features -> cross-attend to M=64-128 latent tokens.
- Self-attention on latent tokens: O(M^2 * d), very cheap.
- Output: 200 query tokens cross-attend to latent array -> 200 selection scores.
- Total complexity: O(N*M*d + M^2*d + K*M*d) = O(1100*128*d + 128^2*d + 200*128*d), all linear in N.

**Key advantage:** Proven to work on point clouds and large inputs. The latent bottleneck forces information compression, which may help with the needle-in-haystack problem by forcing the model to learn what's important.

---

### 4.6 Differentiable Top-k Selection

#### 4.6.1 Gumbel-Softmax

**Paper:** Jang et al., "Categorical Reparameterization with Gumbel-Softmax," ICLR 2017. [arXiv:1611.01144]

**Core mechanism:** Replaces non-differentiable categorical sampling with a differentiable approximation: add Gumbel noise to logits and apply softmax with temperature tau. As tau -> 0, the output approaches a one-hot vector. Enables backpropagation through discrete selection.

**Relevance:** Provides the gradient trick for selecting k-of-N tracks. Apply per-track scorer, add Gumbel noise, take top-k with temperature annealing. However, Gumbel-Softmax produces a *distribution* over categories, not a hard subset selection. For k>1, need Gumbel-Top-k.

#### 4.6.2 Gumbel-Top-k Trick

**Paper:** Kool et al., "Stochastic Beams and Where to Find Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement," ICML 2019. [arXiv:1903.06059]

**Core mechanism:** Extends Gumbel-Max to sample k items without replacement. Add Gumbel noise to log-probabilities, take the k largest. This produces exact samples from the k-subset distribution. Complexity linear in k and N.

**Relevance:** Directly applicable. After a neural scorer produces logits for 1100 tracks, Gumbel-Top-k selects 200 tracks differentiably during training. At inference, just take top-200 by score (no noise needed).

#### 4.6.3 Differentiable Sparse Top-k

**Paper:** Sander et al., "Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective," ICML 2023. [arXiv:2302.01425]

**Core mechanism:** Formulates top-k as a linear program over the permutahedron with p-norm regularization. Solved via isotonic optimization (pool adjacent violators). Produces sparse output where non-zero values correspond to the k largest inputs. GPU-friendly implementation.

**Relevance:** HIGH. This is the most mature differentiable top-k operator. Could replace hard top-k in our pipeline for end-to-end training. Applied successfully in vision transformers (patch selection) and mixture-of-experts (expert selection).

#### 4.6.4 Differentiable Patch Selection

**Paper:** Cordonnier et al., "Differentiable Patch Selection for Image Recognition," CVPR 2021. [arXiv:2104.03059]

**Core mechanism:** Uses a differentiable top-k operator to select the most relevant patches from high-resolution images, then processes only selected patches with a downstream network. End-to-end trainable.

**Relevance:** Direct analogue to our problem. Replace "patches" with "tracks" and "image" with "event". A lightweight scorer evaluates all 1100 tracks, differentiable top-k selects 200, and a downstream network processes only the selected 200. This is essentially our current pipeline made end-to-end differentiable.

---

### 4.7 Neural Submodular Functions for Subset Selection

**Paper:** De & Chakrabarti, "Neural Estimation of Submodular Functions with Applications to Differentiable Subset Selection," NeurIPS 2022. [arXiv:2210.11033]

**Core mechanism:** FLEXSUBNET learns submodular functions (which model diminishing returns in set selection) by recursively applying learned concave functions over modular functions. Produces an order-invariant greedy sampler that is differentiable.

**Relevance:** Moderate. Submodularity captures the intuition that "once you've selected one signal track, the marginal value of the second is slightly different." However, our problem is simpler: we want to maximize recall, which doesn't have strong diminishing-returns structure. Over-engineered for our use case.

---

### 4.8 Set Prediction Networks

**Paper:** Rezatofighi et al., "Learn to Predict Sets Using Feed-Forward Neural Networks," TPAMI 2022. [arXiv:2001.11845]

**Core mechanism:** Defines a proper likelihood for set-valued outputs by jointly modeling (1) set cardinality distribution and (2) permutation-invariant element distributions. Handles variable-size output sets.

**Relevance:** LOW. Designed for predicting what's *in* the output set, not for *selecting* from an input set. Our problem is selection (input: 1100 tracks, output: 200 track indices), not generation.

---

## Direction 5: Domain-Specific HEP Approaches

### 5.1 Exa.TrkX GNN Tracking Pipeline

**Key papers:**
- Farrell et al., "Novel deep learning methods for track reconstruction," CTD 2018. [arXiv:1810.06111]
- Choma et al., "Track Seeding and Labelling with Embedded-space Graph Neural Networks," CTD 2020. [arXiv:2007.00149]
- Ju et al., "Performance of a Geometric Deep Learning Pipeline for HL-LHC Particle Tracking," 2021. [arXiv:2103.06995]
- Lazar et al., "Accelerating the Inference of the Exa.TrkX Pipeline," ACAT 2021. [arXiv:2202.06929]
- Tripathy et al., "Scaling Graph Neural Networks for Particle Track Reconstruction," 2025. [arXiv:2504.04670]

**Pipeline (3 stages):**
1. **Graph construction via metric learning:** An MLP/embedding network maps each hit into a learned latent space where hits from the same track are close. kNN in this latent space builds the graph. This *is* a pre-filter: it reduces O(N^2) possible edges to O(N*k) by learned proximity.
2. **Edge classification via GNN:** Message-passing GNN (typically Interaction Network) classifies each edge as true (same track) or false. Multiple rounds of message passing propagate information.
3. **Track building:** Connected components on the filtered graph, or more sophisticated clustering (DBSCAN, Walkthrough).

**Scale:** TrackML dataset, ~100,000 hits per event, ~10,000 tracks. Pileup 200 (200 overlapping proton-proton collisions).

**Performance:** "Tracking efficiency and purity similar to production tracking algorithms." Computational cost scales "close to linearly" with particle count.

**Relevance to our problem:** VERY HIGH. The metric-learning embedding step is *exactly* a pre-filter:
- It maps each hit to a latent space where same-track hits cluster.
- kNN in this space selects candidate edges, dramatically reducing combinatorics.
- Our problem is analogous: map tracks to a latent space where signal tracks are distinguishable, then use kNN or attention to select them.

**Key lesson:** The HEP community found that *learned graph construction* (metric learning -> kNN) is the most critical step. A good embedding makes the downstream GNN's job easy. A bad embedding (too many false edges or missing true edges) is unrecoverable.

---

### 5.2 Interaction Networks for Tracking

**Paper:** DeZoort et al., "Charged particle tracking via edge-classifying interaction networks," Comput. Softw. Big Sci. 2021. [arXiv:2103.16701]

**Core mechanism:** Adapts Interaction Networks (Battaglia et al., NeurIPS 2016) for track finding. The IN processes graphs where nodes=hits, edges=candidate track segments. Edge network computes messages from node pairs, node network aggregates messages. After several message-passing rounds, an edge classifier predicts true/false for each edge.

**Key advantage:** The architecture is "substantially smaller than previously studied GNN tracking architectures" while achieving "excellent edge-classification accuracy."

**Scale:** HL-LHC pileup conditions, with "idealized hit filtering at various particle momenta thresholds."

**Relevance:** The Interaction Network is the standard GNN backbone for HEP tracking. It demonstrates that relatively small models work well for edge classification. For our pre-filter, the message-passing scheme could be used to propagate local context between neighboring tracks (similar to what our kNN+message passing does).

---

### 5.3 Object Condensation for Tracking

**Key papers:**
- Kieseler, "Object condensation: one-stage grid-free multi-object reconstruction," Eur. Phys. J. C, 2020. [arXiv:2002.03605]
- Lieret & DeZoort, "An Object Condensation Pipeline for Charged Particle Tracking at the HL-LHC," CHEP 2023. [arXiv:2309.16754]
- Lieret et al., "High Pileup Particle Tracking with Object Condensation," CTD 2023. [arXiv:2312.03823]
- Reuter et al., "Multi-Modal Track Reconstruction using Graph Neural Networks at Belle II," ACAT 2025. [arXiv:2602.10897]

**Core mechanism:** Instead of classifying edges, OC directly clusters hits into objects (tracks). Each hit is mapped to a latent space. The loss has two terms:
- **Attractive potential:** Pulls hits belonging to the same object toward a "condensation point" (the highest-confidence hit for that object) in latent space.
- **Repulsive potential:** Pushes condensation points of different objects apart.
- **Beta (confidence):** Each hit predicts a confidence beta in [0,1]. High-beta hits become condensation points. Low-beta hits are noise or follow their condensation point.

After training, objects are recovered by: (1) find high-beta points, (2) assign each remaining point to its nearest condensation point in latent space.

**Scale:** TrackML pixel detector, HL-LHC pileup conditions. Recent work (Belle II, 2026) reports 74.7% track efficiency for displaced particles.

**Relevance to our problem:** HIGH, but as an *alternative* paradigm, not a direct pre-filter. Instead of selecting top-200 tracks, OC would:
- Assign each of 1100 tracks a beta (confidence of being a signal-track condensation point).
- Learn a latent space where signal tracks cluster near their condensation point and noise tracks have low beta.
- The top-3 beta values would identify signal tracks directly.

**Advantage over current approach:** OC naturally handles variable number of signal tracks (could be 1, 2, or 3 per event). The beta prediction provides a natural ranking.

**Limitation:** OC was designed for *all* tracks in an event, not for finding a rare subset. With 3 signal among 1100 noise (0.3% signal fraction), the repulsive potential may be dominated by noise-noise interactions. Would need careful loss weighting.

---

### 5.4 ParticleNet

**Paper:** Qu & Gouskos, "ParticleNet: Jet Tagging via Particle Clouds," Phys. Rev. D 2020. [arXiv:1902.08570]

**Core mechanism:** Applies Dynamic Graph CNN (DGCNN) with EdgeConv to particle physics. Treats jets as unordered point clouds. Key: dynamic kNN graph is recomputed in feature space after each EdgeConv layer, allowing the graph topology to evolve.

**Scale tested:** Jet tagging with ~100 particles per jet. State-of-the-art on top tagging and quark-gluon benchmarks.

**Relevance:** MODERATE. ParticleNet's per-jet scale (~100 particles) is much smaller than our ~1100 tracks. However, the dynamic graph concept is relevant: our kNN graph in the backbone is currently static. Dynamic regraphing could help signal tracks cluster in feature space even if they're spatially dispersed.

**Known limitation:** kNN graph construction is O(N * k * d) with N=1100 particles, which is manageable. But full self-attention within EdgeConv neighborhoods is not scalable to 1100 without the hierarchical approach (which we already use via FPS downsampling).

---

### 5.5 GravNet

**Paper:** Qasim et al., "Learning representations of irregular particle-detector geometry with distance-weighted graph networks," Eur. Phys. J. C, 2019. [arXiv:1902.07987]

**Core mechanism:** Two architectures: GarNet (global, fixed aggregation) and GravNet (local, dynamic). GravNet learns to:
1. Map each hit to a coordinate space (separate from feature space).
2. Build kNN graph in the *learned* coordinate space.
3. Aggregate neighbor features with distance-dependent weights (closer neighbors contribute more).

**Scale:** Calorimeter clustering on CMS-inspired toy detector. Later used in CMS HGCAL reconstruction and combined with Object Condensation.

**Relevance:** HIGH. GravNet's "learned coordinate space + kNN" is essentially learned metric for graph construction -- the same idea as Exa.TrkX's metric learning step. For our pre-filter:
- Map each track to a learned coordinate space.
- Signal tracks should be mapped close together, noise tracks far away.
- kNN in this space naturally finds signal neighborhoods.

**Recent combination:** GravNet + Object Condensation has become a standard combination in HEP (used in CMS HGCAL, CLAS12 calorimetry, ILD reconstruction). GravNet provides the learned graph, OC provides the clustering loss.

---

### 5.6 Transformers for Tracking

**Paper:** Van Stroud et al., "Transformers for Charged Particle Track Reconstruction in High Energy Physics," Phys. Rev. X, 2025. [arXiv:2411.07149]

**Core mechanism:** Two-stage pipeline:
1. **Transformer hit filtering network:** Pre-filters hits to reduce combinatorics.
2. **MaskFormer reconstruction model:** Jointly optimizes hit assignments and track property estimation.

**Performance:** 97% efficiency at 0.6% fake rate on TrackML. Inference: ~100ms per event on GPU.

**Scale:** TrackML dataset, ~100k hits.

**Relevance:** VERY HIGH. This paper directly demonstrates:
- A **Transformer-based pre-filter** that reduces hits before reconstruction -- exactly our architecture pattern.
- The MaskFormer stage uses learned queries (like DETR) to assign hits to tracks.
- State-of-the-art results, beating GNN pipelines.

**Key insight:** The "hit filtering + query-based assignment" two-stage pattern mirrors our "pre-filter + track finder" pipeline. The Transformer pre-filter at 100k-hit scale suggests attention-based approaches are viable.

---

### 5.7 LSH-Based Point Transformer (HEPTv2)

**Paper:** Govil et al., "Locality-Sensitive Hashing-Based Efficient Point Transformer for Charged Particle Reconstruction," NeurIPS ML4PS Workshop, 2025. [arXiv:2510.07594]

**Core mechanism:** Uses locality-sensitive hashing (LSH) to achieve near-linear complexity attention on large hit sets. LSH groups nearby hits into buckets, attention computed only within buckets. Eliminates expensive graph construction step.

**Performance:** ~28ms/event on A100 GPU, competitive tracking efficiency with GNN baselines.

**Relevance:** HIGH for efficiency. LSH-attention could replace kNN graph construction in our backbone:
- Hash 1100 tracks into buckets by learned features.
- Attend within buckets (local context).
- Near-linear complexity.

---

### 5.8 Track Quality Classifiers at CMS

**Paper:** Guiang et al., "Improving tracking algorithms with machine learning: a case for line-segment tracking at the High Luminosity LHC," CTD 2023. [arXiv:2403.13166]

**Core mechanism:** Lightweight neural network replaces hand-crafted track quality cuts in CMS Line-Segment Tracking (LST) algorithm. The NN takes track features as input and predicts quality score. "Highly efficient and robust to displaced tracks while having little-to-no impact on runtime."

**Relevance:** DIRECT ANALOGUE. This is exactly a track-level binary classifier (good track vs fake), which is what our pre-filter does. The CMS experience shows:
- Simple NNs work well for track quality classification.
- Per-track features are sufficient for quality scoring (no inter-track context needed at this stage).
- The classifier must be fast enough to not bottleneck the pipeline.

---

### 5.9 Combined GNN + CKF Tracking (ACTS)

**Paper:** Heinrich et al., "Combined track finding with GNN & CKF," CTD 2023. [arXiv:2401.16016]

**Core mechanism:** GNN handles inner pixel detector (high-resolution 3D hits), identifies track seeds. Combinatorial Kalman Filter (CKF) extends these seeds through outer detector layers (lower-resolution 1D measurements). The GNN acts as a *pre-filter* that provides validated seeds to the CKF.

**Relevance:** HIGH as architectural pattern. This directly demonstrates:
- ML pre-filter -> classical algorithm pipeline.
- The pre-filter resolves the hardest part (combinatorics in dense regions).
- The downstream algorithm handles easier extrapolation.
- Our pattern: pre-filter -> track finder is the same.

---

### 5.10 Smart Pixels (In-Pixel AI)

**Paper:** Parpillon et al., "Smart Pixels: In-pixel AI for on-sensor data filtering," 2024. [arXiv:2406.14860]

**Core mechanism:** Neural network embedded directly in pixel sensor hardware for real-time data filtering at 40 MHz. Achieves 54-75% data rejection while preserving physics signal.

**Relevance:** Low for architecture, but shows that the HEP community takes the "pre-filter" concept seriously enough to embed it in hardware. Even at the detector level, aggressive data reduction is needed.

---

### 5.11 Belle II GNN + Object Condensation

**Paper:** Reuter et al., "Multi-Modal Track Reconstruction using Graph Neural Networks at Belle II," ACAT 2025. [arXiv:2602.10897]

**Performance:** Track efficiency 74.7% for displaced particles (up to 100cm displacement), purity improved by 5.5 percentage points vs previous. Uses multi-modal inputs (drift chamber + silicon vertex detector).

**Relevance:** Shows OC + GNN works for realistic tracking with beam backgrounds and noise. The 74.7% efficiency on displaced tracks is notable -- displaced tracks are analogous to our low-pT tracks in being hard to find.

---

## Lessons from HEP for Our Pre-Filter

### Lesson 1: Learned Graph Construction is the Critical Step

The entire Exa.TrkX pipeline can be viewed as: (1) learn where to look, (2) classify what you find. Step 1 (metric learning embedding -> kNN) is the bottleneck for recall. If a true edge is missed in graph construction, no amount of GNN refinement can recover it.

**Implication for us:** Our kNN-based graph construction in the backbone is the right idea, but the *feature space* in which kNN operates must be optimized for signal-noise separation. The backbone's learned features should be directly trained to make signal tracks nearest-neighbors of each other or of a learned query.

### Lesson 2: Two-Stage Pipelines Dominate

Almost every successful HEP tracking approach uses a coarse-to-fine pattern:
- Exa.TrkX: metric learning -> GNN -> track building
- Van Stroud et al.: Transformer filter -> MaskFormer
- Heinrich et al.: GNN seeds -> CKF extension
- CMS LST: segment building -> NN quality filter

**Implication:** Our pre-filter -> track finder pipeline is architecturally sound. The question is whether the pre-filter should be a simple scorer (current MLP) or a more expressive model (Set Transformer, DETR-like queries, or OC-based).

### Lesson 3: Scale Matters -- O(N^2) is Forbidden

At 100k hits, full self-attention is impossible. Every successful approach uses one of:
- kNN graphs (O(N*k))
- Metric learning embedding + local connectivity (O(N*k))
- LSH-based bucketing (O(N * bucket_size))
- Inducing points / latent bottleneck (O(N*m))

At our scale (N=1100), full self-attention O(N^2) = O(1.2M) is borderline feasible but wasteful. Using 64-128 inducing points or kNN with k=16-32 is much more efficient.

### Lesson 4: Object Condensation is a Viable Alternative to Edge Classification

OC replaces edge classification with point-level clustering + confidence scoring. This is closer to our problem (score each track) than edge classification (score each pair). OC's beta prediction is naturally a per-track signal score.

**Implication:** We could add an OC-inspired loss to the pre-filter: each track predicts a condensation confidence beta. Signal tracks should have high beta. The attractive/repulsive loss in latent space would train the backbone to cluster signal tracks together. This avoids the need for explicit edge labels.

### Lesson 5: Low-pT Tracking is an Open Problem

There is very little published work specifically on low-pT track finding with ML. CMS LST mentions "displaced tracks" as a challenge. The TrackML challenge used momentum thresholds for evaluation. Low-pT tracks have:
- More scattering -> less predictable trajectories
- More confusion with noise/fakes
- Fewer hits (shorter before curling)

**Implication:** Our problem (low-pT tau tracks among noise) is genuinely novel. We cannot directly copy a solution from the literature. But the architectural patterns (learned embeddings, attention-based scoring, coarse-to-fine filtering) are well-established.

### Lesson 6: GravNet + OC is the HEP Standard for Clustering

The combination of GravNet (learned dynamic graph construction) with Object Condensation (cluster assignment + confidence) has become the standard approach in multiple experiments (CMS HGCAL, Belle II, CLAS12, ILD). This suggests:
- Learned coordinate spaces for graph construction are robust.
- The OC loss is well-suited for variable-multiplicity problems.
- The combination scales to realistic detector data.

---

## Synthesis: Most Promising Approaches for Our Pre-Filter

### Tier 1 (Highest promise, most directly applicable)

| Approach | Why | Complexity | Key reference |
|----------|-----|-----------|---------------|
| **DETR-like queries** | K=200 learned queries cross-attend to 1100 tracks. Hungarian matching for training. Directly outputs selection scores. | O(K*N*d) | Carion et al. 2020, Van Stroud et al. 2025 |
| **Set Transformer (ISAB)** | Inducing points give global context at linear cost. PMA outputs selection scores. Most natural "upgrade" from current MLP. | O(N*m*d) | Lee et al. 2019 |
| **Perceiver-style bottleneck** | Cross-attend tracks to latent array, self-attend latent, output queries select. Proven at 50k+ scale. | O(N*M*d) | Jaegle et al. 2021 |

### Tier 2 (Strong potential, needs adaptation)

| Approach | Why | Complexity | Key reference |
|----------|-----|-----------|---------------|
| **OC-inspired beta scoring** | Per-track confidence prediction with attractive/repulsive loss. Natural ranking. Proven in HEP. | O(N*d + N*k*d) via GravNet | Kieseler 2020, Lieret et al. 2023 |
| **Differentiable top-k** | Makes current pipeline end-to-end trainable. Drop-in replacement for hard selection. | O(N log N) | Sander et al. 2023 |
| **Slot Attention (K=3)** | 3 signal slots compete to bind to signal tracks. Very cheap but untested at this scale/ratio. | O(K*N*d) | Locatello et al. 2020 |

### Tier 3 (Useful components, not standalone solutions)

| Approach | Why | Key reference |
|----------|-----|---------------|
| **Gumbel-Top-k** | Differentiable sampling for training; pairs with any scorer. | Kool et al. 2019 |
| **DN-DETR denoising** | Training stabilization for query-based models. | Li et al. 2022 |
| **LSH-attention** | Efficient local attention at 1000+ scale. | Govil et al. 2025 |

---

## Recommended Reading Priority

1. Van Stroud et al. 2025 (arXiv:2411.07149) -- Transformer pre-filter for tracking, closest to our setting
2. Carion et al. 2020 (arXiv:2005.12872) -- DETR query mechanism
3. Lee et al. 2019 (arXiv:1810.00825) -- Set Transformer / ISAB
4. Jaegle et al. 2021 (arXiv:2103.03206) -- Perceiver latent bottleneck
5. Kieseler 2020 (arXiv:2002.03605) -- Object Condensation loss
6. Ju et al. 2021 (arXiv:2103.06995) -- Exa.TrkX pipeline overview
7. Sander et al. 2023 (arXiv:2302.01425) -- Differentiable top-k
8. DeZoort et al. 2021 (arXiv:2103.16701) -- Interaction Networks for tracking
