# Improving low‑pT τ→3π Phase‑1 ML pre‑filter recall beyond R@200 = 0.8

## Executive summary

The current Phase‑1 pre‑filter is a **retrieval/ranking** problem: from **O(10³) reconstructed tracks per event**, rank so that the **three true τ→3π pion tracks** appear in the **top‑200**. The best model in the attached experiment record plateaus at **val R@200 ≈ 0.629** and **val P@200 ≈ 0.368** despite extensive architecture, feature, and loss explorations. fileciteturn0file0

The evidence in the summary points to a dominant bottleneck: **the interaction structure is wrong for this sparsity regime**. A fixed kNN graph in (η,φ) yields **neighbourhoods dominated by background**, so any message passing/attention computed on those edges mostly aggregates noise. This is consistent with: (i) only a small uplift from GNN context versus per‑track models, (ii) attempts to “make the network richer” failing, and (iii) strong relational physics signatures (ρ resonance, common secondary vertex) being inaccessible when signal tracks are rarely neighbours in the chosen graph. fileciteturn0file0

Physics further explains why the **“uncanny valley”** (very soft, near‑prompt pions) is hard: the τ lifetime corresponds to **cτ ≈ 87 μm**, so at low momentum the displacement is often within tracking resolution, making per‑track impact‑parameter significance weak as a discriminant. citeturn9view0 The **three‑charged‑prong** topology itself is common in τ decays (PDG quotes **≈14.55%** for “3‑prong” modes), but the discriminating information is largely **relational** (pair masses, charge pattern, vertex compatibility) rather than single‑track. citeturn10view0

To exceed **R@200 ≥ 0.8**, the most plausible route is a **paradigm shift in which interactions are computed on signal‑enriched subsets or via set‑prediction** rather than on a noise‑dominated geometric kNN:

- **Primary recommendation (highest expected gain):** a **cascade pre‑filter**: coarse ranking to **K₁≈600** (where even a simple per‑track BDT reaches **R@600≈0.90** in the record), then a **relational reranker** on the enriched set using **physics‑motivated pairwise features** and/or **triplet‑aware inference** to produce the final top‑200. fileciteturn0file0  
- **Secondary high‑gain track:** **set‑prediction with 3 learned queries** (DETR/Slot‑style) that directly reasons about a small set of objects amid clutter, avoiding the need for a fixed kNN graph. citeturn5search1turn5search0  
- **Supporting improvements (moderate gains):** metric‑aligned **listwise top‑K losses** and **differentiable top‑K** operators to better align training with R@200/P@200, plus **hard‑region curricula** and **feature expansions** focused on vertex consistency and track‑fit geometry. citeturn0search3turn1search0turn1search28turn3search8

If executed with disciplined ablations and decision gates, reaching **R@200 ≈ 0.80** is plausible but **not guaranteed**; a realistic forecast is that the cascade+relational reranker is the only single proposal with a credible **ΔR@200 of O(0.12–0.22)**, enough to clear the target, while other proposals are typically **ΔR@200 O(0.01–0.08)** individually.

## Current baseline extracted from the attached experiment record

### Task formulation and metrics

The pre‑filter ranks ~1100 tracks per event (with padding up to 3500) and selects the **top‑200** candidates that feed a heavier downstream stage. The primary metric is **R@200** (average fraction of the 3 truth pions found in the top‑200), with **P@200** (fraction of events with all 3 found) as a stricter companion. The target is **R@200 ≥ 0.80**. fileciteturn0file0

### Dataset composition and scaling constraints

The record states **~270k training events and ~84k validation events**, each with exactly **3 truth pions**, and typical track multiplicity around **O(10³)** (mean ~1130, median ~1211), with outliers up to ~3500 tracks (hence padding to length 3500). fileciteturn0file0  
One inconsistency is also present (“Dataset events | 19K” in an appendix table); treat that as either an older subset or a documentation artefact and verify the true counts from the training pipeline logs. fileciteturn0file0

### Baseline architecture (best model in the record)

The current best (“Extended Data Model”) is a **TrackPreFilter** with:

- **MLP encoder → two rounds of kNN message passing (k=16)** with **max‑pool aggregation → scoring MLP**,  
- graph built as **fixed kNN in (η,φ)**,  
- **~226k parameters**, intended complexity **O(P·k)** per message passing round. fileciteturn0file0

This family aligns with successful point‑cloud GNN patterns in HEP (e.g. EdgeConv/DGCNN‑style neighbourhood aggregation), but its effectiveness depends on neighbourhoods containing meaningful local structure. citeturn5search3turn5search7

### Inputs, preprocessing, augmentation

The best model uses **16 per‑track features**, including kinematics (pT, η, φ, charge), uncertainties and quality (pT_error, n_valid_pixel_hits, normChi2), displacement significance (dxy/dz/dca significance), two covariance terms, and several **log/log1p transforms**. fileciteturn0file0  
A chi² feature bug (severe clipping) was discovered and fixed, but feature additions alone did not lift the plateau materially. fileciteturn0file0  
Copy‑paste style augmentation was attempted but abandoned due to dataloader deadlocks; therefore, augmentation is effectively minimal in the current baseline. fileciteturn0file0

For context, CMS tracking and object reconstruction commonly operate near pT thresholds of a few hundred MeV in some workflows (e.g. Run‑3 HLT tracking references pT>0.3 GeV in certain configurations), which emphasises how close this problem is to tracking and selection limits at very low pT. citeturn3search9turn3search8

### Losses and training schedule

Training combines:

- a **temperature‑scheduled ranking loss** (primary objective),  
- an auxiliary **contrastive denoising** term with **temperature cycling**,  
- an auxiliary **reconstruction loss** with **deferred re‑weighting (DRW)** (down‑weighted after early epochs),  
- **AdamW** with **cosine learning‑rate schedule**. fileciteturn0file0

DRW originates from class‑imbalance learning techniques that adjust the effective gradient contribution of terms across training phases. citeturn0search3

### Best reported performance and plateau evidence

The best checkpoint (around epoch 35) achieves:

- **val R@200 ≈ 0.629**,  
- **val P@200 ≈ 0.368**,  
- median GT rank ≈ 108, d′ ≈ 1.29. fileciteturn0file0

A key diagnostic is a per‑track **XGBoost ceiling test**: **R@200 ≈ 0.589** using only per‑track features (no graph), while **R@500 ≈ 0.85** and **R@600 ≈ 0.90**, suggesting a cascade could be viable if later stages exploit relational cues. fileciteturn0file0

Across many architecture/loss variations (ParticleNeXt‑style attention aggregation, global attention via inducing points, GravNet/learned‑space kNN, EdgeConv variants, OHEM, ASL, score propagation), performance repeatedly converges near **R@200 ≈ 0.62 ± 0.01**, implying a structural bottleneck rather than capacity. fileciteturn0file0

## Failure modes and bottlenecks in low‑pT 3‑prong reconstruction

### Physics and detector constraints that shape the low‑pT regime

The τ lepton has **cτ ≈ 87 μm** (mean life ~290 fs); at low momentum, the average flight distance is often comparable to impact‑parameter resolution, so **|dxy| significance can be small even for genuine τ decay products**. citeturn9view0 This interacts directly with the observed failure cluster in the record (low pT + small |dxy_sig|). fileciteturn0file0

Three‑charged‑prong τ decays are common at the branching‑fraction level (PDG “3‑prong” category ≈14.55%) and include a prominent **π−π+π−** contribution (≈9.31%). citeturn10view0 The hadronic 3‑prong decay dynamics are dominated by resonant substructure (a₁ → ρπ being a key channel), which is precisely the kind of **pairwise invariant‑mass** signature that single‑track scoring cannot exploit. citeturn2search0turn2search4

CMS τ reconstruction algorithms historically encode such relational constraints explicitly: the HPS approach assigns decay modes and applies invariant‑mass constraints to match τh kinematics. citeturn1search15turn0search0turn1search19 (Even though your pre‑filter is not HPS itself, the design lesson is that **relational constraints matter** for 3‑prong.)

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["tau lepton three prong decay diagram","tau to three pions a1 rho pi decay diagram","CMS silicon tracker layout diagram"],"num_per_query":1}

### Empirical failure modes in the experiment record

The record identifies a sharp “uncanny valley” where **per‑track features are near‑degenerate with background**:

- pT bin **0.3–0.5 GeV** has notably low found rate,  
- **|dxy_sig| < 0.5** is the worst regime, with a large fraction of truth pions living there,  
- the intersection (soft + near‑prompt) yields ~O(25%) recall. fileciteturn0file0

These observations are consistent with general low‑pT tracking behaviour: softer tracks are more affected by multiple scattering and energy loss, and selection criteria/fit quality variables can overlap between genuine and fake/secondary tracks, increasing ambiguity. CMS tracking performance documents emphasise iterative reconstruction and pT‑dependent efficiencies/resolutions, with low‑pT regimes requiring dedicated iterations and careful tuning. citeturn3search8turn3search9turn3search27

### The neighbourhood‑selection bottleneck

The strongest diagnosis in the record is graph‑structural:

- signal is **0.27%** (3 / ~1100 tracks),  
- kNN(k=16) in (η,φ) yields **~99.7% noise neighbours**,  
- **~72% of truth pions have zero truth neighbours** in their kNN neighbourhood,  
- therefore message passing or attention over this graph aggregates mostly background, explaining why richer GNNs/attention did not help. fileciteturn0file0

This is a textbook failure mode of local aggregation under extreme sparsity: **how** you aggregate matters less than **which elements are allowed to interact**. The success of attention‑based HEP models that incorporate pairwise interactions (e.g. Particle Transformer’s pairwise bias terms; IAFormer’s sparse attention driven by boost‑invariant pairwise quantities) further supports that *accessible relational structure* is the differentiator when it exists. citeturn5search2turn4search3turn4search7

### Secondary bottlenecks worth checking explicitly

Even if the graph bottleneck is primary, several “quiet” issues can cap R@200 and should be audited because they can mimic hard physics limits:

- **Truth matching and label purity:** if some GT pions fail to produce a reconstructable track (or are mismatched), the effective ceiling for R@200 is below 1.0. This must be quantified as an “oracle” upper bound from reconstruction. (No direct evidence in the record; verify.) fileciteturn0file0  
- **Pad‑to‑3500 efficiency mismatch:** padding may distort batch statistics if masking is imperfect; ensure masked entries cannot contribute to normalisation layers or neighbourhood construction. fileciteturn0file0  
- **Objective mismatch:** optimising a generic ranking surrogate can under‑optimise the *top‑200 truncation*; listwise top‑K losses exist precisely to address this. citeturn0search3turn1search6turn1search30turn1search28

## Prioritised concrete improvement proposals

### Assumptions used for estimates

Because compute and deployment constraints were not specified, the quantitative expectations below assume:

- **Training:** O(300k) events, mixed precision, 1–4 modern GPUs (e.g. 80GB class), run time **~10–30 GPU‑hours per training** for current models;  
- **Inference budget:** per‑event latency of **≲ a few ms on GPU** is acceptable for prototyping; final latency depends on whether this is offline/HLT/L1‑adjacent.

Expected R@200 gains are **engineering estimates** (not guarantees), expressed as **ΔR@200** relative to **0.629** and with uncertainty bands capturing modelling risk.

### Summary comparison table

| Proposal | Core idea | Expected ΔR@200 | Expected R@200 | Cost (dev+compute) | Risk | Why it targets the bottleneck |
|---|---|---:|---:|---|---|---|
| A | Cascade: top‑K₁ (~600) + relational reranker to top‑200 | +0.12 to +0.22 | 0.75–0.85 | Medium–High | Medium | Creates **signal‑enriched interactions**, enabling pair/triplet physics |
| B | Set prediction with 3 learned queries (DETR/Slot‑style) | +0.08 to +0.15 | 0.71–0.78 | High | High | Bypasses fixed kNN; learns to “attend” to 3 objects in clutter |
| C | Triplet‑aware inference (triangle/vertex constrained) without combinatorial blow‑up | +0.06 to +0.14 | 0.69–0.77 | Medium | Medium–High | Directly encodes “3 tracks from same τ decay” structure |
| D | Listwise top‑K aligned loss + differentiable top‑K in training | +0.02 to +0.06 | 0.65–0.69 | Low–Medium | Low–Medium | Fixes objective mismatch at K=200; improves gradient signal |
| E | Hard‑region curriculum + confuser‑negative sampling + augmentation | +0.02 to +0.06 | 0.65–0.69 | Low–Medium | Medium | Focuses modelling capacity on uncanny‑valley failures |
| F | Feature expansion: vertexing/helix/covariance/hit pattern + event context | +0.01 to +0.05 | 0.64–0.68 | Medium | Medium | Adds information that may separate near‑prompt soft pions from pileup |
| G | Calibration + uncertainty‑aware scoring + per‑bin equalisation | +0.01 to +0.03 | 0.64–0.66 | Low | Low | Stabilises ranking, reduces pathological score compression |
| H | Domain adaptation + sim→data corrections + ensemble/distillation | +0.00 to +0.03 (sim); +robustness on data | 0.63–0.66 | Medium | Medium | Reduces sim‑data mismatch; critical for deployment stability |

### Proposal A: Cascade pre‑filter with relational reranking on an enriched candidate set

**Rationale**  
The record already demonstrates that (i) per‑track scoring achieves respectable recall at larger K (e.g. **R@600≈0.90** with a BDT), and (ii) the present kNN graph is too noisy for relational learning at full multiplicity. A cascade is the standard remedy in extreme‑imbalance retrieval: first stage maximises **coverage** (recall at a larger K₁), second stage maximises **precision within K₁** using more expensive interactions. fileciteturn0file0  
This mirrors modern large‑scale retrieval practice, and recent ML work on differentiable top‑K explicitly frames cascade ranking as a common paradigm. citeturn1search12turn1search20turn1search0

**Design sketch (pre‑filter still outputs top‑200)**  
- **Stage A1 (coarse, fast):** current TrackPreFilter (or simpler MLP/BDT) produces scores for all tracks and keeps **top‑K₁** (recommend starting K₁=600).  
- **Stage A2 (relational reranker):** operate only on K₁ tracks using:
  - **pairwise physics features** (ΔR, invariant mass assuming mπ, charge product, pT asymmetry z, kT‑like measures) as **attention bias** (Particle Transformer style) or sparse‑attention masks (IAFormer style). citeturn5search2turn4search3turn4search7  
  - optional **vertex‑compatibility approximations** (see Proposal F) to favour “common SV” structure. Track/vertex reconstruction in CMS is well‑studied and provides the conceptual basis for such compatibility variables. citeturn3search8turn3search2  
- Output is a **re‑scored ranking** over K₁; take **top‑200**.

**Expected quantitative impact**  
- **ΔR@200: +0.12 to +0.22** (expected value ~+0.17), giving **R@200 ≈ 0.75–0.85**.  
- Uncertainty drivers: (i) whether K₁ enrichment is sufficient for the relational stage to reliably “rescue” the third pion, (ii) whether pairwise masses are informative at the very lowest pT given resolution/smearing.

**Data / compute requirements**  
- No new labels required.  
- Compute increases due to A2; but K₁=600 makes dense or moderately sparse attention feasible (O(600²) ~ 3.6e5 pair interactions/layer).  
- Training A2 should be ~comparable or modestly heavier than baseline; A2 can be much larger than 226k params because sequence length is smaller.

**Implementation steps**  
1. **Freeze your existing evaluator** for R@K curves (K=50..800), P@K, and conditional recall surfaces (pT, |dxy_sig|).  
2. Implement **A1→top‑K₁ selection** in the dataloader so A2 sees exactly the candidates it will see at inference.  
3. Build pairwise features on K₁:
   - compute 4‑vectors from (pT,η,φ) with pion mass hypothesis; derive pairwise m(ij), ΔR, z, kT. (Particle Transformer uses pairwise terms inside attention and shows large performance gains in jet tagging tasks.) citeturn5search2  
4. Train A2 with a top‑K aware objective (Proposal D) and optionally an auxiliary “triplet consistency” objective (Proposal C).  
5. Integrate A2 ranking with A1 fallback: if A2 is uncertain (e.g. low confidence), use A1 scores to avoid catastrophic drops.

**Risks / failure modes**  
- If A1 drops a GT pion, A2 cannot recover it; hence K₁ must be chosen to ensure **R@K₁ ≳ 0.9** (record suggests 600 is a plausible starting point). fileciteturn0file0  
- Pairwise features may be noisy at low pT; mass peaks (ρ) may broaden. Still, CMS τ algorithms historically use invariant‑mass constraints successfully as part of decay‑mode assignment. citeturn1search15turn0search0turn1search19  
- Engineering complexity: efficient pairwise computation and memory management.

**Validation experiments**  
- Primary: R@200, P@200, and R@K for K∈{200,300,500,600}.  
- Conditional: R@200 vs pT bins; vs |dxy_sig|; and a 2D heatmap (pT × |dxy_sig|) to check whether the uncanny‑valley is improving. fileciteturn0file0  
- Ablations:
  - A2 without pairwise features (attention over node embeddings only) vs with pairwise bias.  
  - Vary K₁ (400/600/800) to identify the recall‑compute Pareto front.  
  - Compare dense attention vs sparse attention (top‑M neighbours per node) inspired by sparse‑interaction transformer designs. citeturn4search3turn4search7  
- Runtime profiling: per‑event latency and memory peak; ensure deployment feasibility.

### Proposal B: Set prediction with 3 learned queries (DETR/Slot‑style “find the 3 pions”)

**Rationale**  
Your problem has a fixed, small number of “objects” (3 charged pions) hidden in clutter. DETR reframes detection as **set prediction** using a small set of learned object queries and bipartite matching loss, removing hand‑engineered post‑processing. citeturn5search1turn5search5 Slot Attention similarly learns a small set of object‑centric slots via competitive attention. citeturn5search0turn5search4  
Unlike kNN message passing, query‑based set prediction does not require signal tracks to be neighbours in a fixed graph; it can, in principle, learn global assignments.

**Model sketch**  
- Encoder: per‑track MLP → compact embeddings.  
- Decoder: **3 queries** (or 4–6 with “null” queries) cross‑attend to all tracks, producing:
  - either (a) a probability distribution over tracks per query, or (b) a continuous pointer (via Gumbel/Sinkhorn relaxations). citeturn1search9turn1search17turn1search1  
- Loss: Hungarian matching between predicted queries and the 3 GT tracks (DETR), plus auxiliary penalties discouraging duplicate selection. citeturn5search1turn5search5  
- Convert to ranking: per‑track score = max query assignment probability; output top‑200.

**Expected quantitative impact**  
- **ΔR@200: +0.08 to +0.15** (expected ~+0.10).  
- Upside: can directly recover “the missing third pion” by jointly reasoning about a 3‑object set.  
- Downside: extreme clutter may still cause attention to diffuse without strong inductive bias.

**Data / compute**  
- Training is heavier than current GNN because cross‑attention scales with track count, though 1100 tracks is still moderate.  
- If full attention is heavy, use low‑rank cross‑attention (Perceiver‑style) or two‑stage (operate on top‑K₁ from A1) to reduce cost.

**Implementation steps**  
1. Prototype on **top‑K₁ input only** to reduce clutter and isolate the benefit (combine with Proposal A).  
2. Implement bipartite matching loss and query outputs; start with straightforward softmax assignment.  
3. Add physics priors as query embeddings (e.g. charge pattern constraints) or as additional pairwise bias (Proposal A’s pairwise features) to reduce degeneracy.  
4. Calibrate output to ranking metric with top‑K aligned loss (Proposal D).

**Risks**  
- Training instability / mode collapse (queries all attend to the same easy signal‑like track). DETR literature often needs auxiliary losses and careful training schedules. citeturn5search1turn5search5  
- Without explicit pairwise features, may revert to per‑track discrimination and fail in the uncanny valley.

**Validation**  
- Monitor not only R@200, but **inter‑query diversity** (how often the 3 selected tracks are distinct).  
- Event‑level success rate approximating P@200 (or “3/3 recovered in top‑N” for smaller N).  
- Ablate: queries=3 vs queries>3; with/without pairwise bias; with/without cascade input restriction.

### Proposal C: Triplet‑aware modelling and inference without combinatorial blow‑up

**Rationale**  
The record notes: the model often finds **2 of 3** pions; rescuing the third requires exploiting **triplet structure** (mass constraint, common vertex, charge pattern) but naïve triplet enumeration explodes. fileciteturn0file0  
Physics provides strong constraints: τ mass and 3‑prong topology; PDG provides τ mass and lifetime for constraints and scale. citeturn9view0turn10view0 CMS τ reconstruction similarly uses mass/decay‑mode constraints at higher levels. citeturn1search15turn0search0

**Algorithmic approach (beam‑search / factorised scoring)**  
Within an enriched candidate set (ideally from Proposal A, K₁≈600):

1. Train an **edge scorer** \(s_{ij}\) (or compute from pairwise‑aware transformer) predicting compatibility “same τ decay”.  
2. For each anchor i, keep top‑L neighbours by \(s_{ij}\) (L~16–32).  
3. Evaluate a limited number of triplets (i,j,k) via:
   - triplet score = unary(i)+unary(j)+unary(k) + \(s_{ij}+s_{ik}+s_{jk}\) + physics penalties (mass windows, charge pattern).  
4. Promote tracks belonging to top‑M triplets in the final ranking.

This yields complexity roughly **O(K₁·L²)** (or better with pruning), avoiding O(K₁³) or OOM.

**Expected quantitative impact**  
- **ΔR@200: +0.06 to +0.14** (expected ~+0.09), especially via converting many “2/3 found” events into “3/3 found.”

**Data / compute**  
- Needs pairwise labels (signal‑signal pair edges are positive; others negative). Labels are derivable from existing truth.  
- Training cost moderate; inference cost modest if pruning is aggressive.

**Implementation steps**  
- Start from K₁=600 candidates, build pairwise features (as in Proposal A).  
- Train an edge classifier with strong negative sampling (to handle extreme imbalance).  
- Implement beam search triplet selection and score “promotion” into track ranking.

**Risks**  
- If pairwise scores are not selective, pruning fails and cost grows (this is exactly what happened with the earlier mass‑cut attempt). fileciteturn0file0  
- Over‑fitting to simulation mass shapes; may generalise poorly without calibration (Proposal H).

**Validation**  
- Report: (i) gain in P@200, (ii) conditional “rescue rate” in events where baseline has exactly 2/3 in top‑200, (iii) calibration plots of pairwise score vs true edge label.

### Proposal D: Metric‑aligned listwise top‑K objectives and differentiable top‑K training

**Rationale**  
Top‑K metrics are discontinuous; optimising generic ranking surrogates can leave performance on the table at K=200. Recent work proposes smooth surrogates that explicitly address top‑K truncation (e.g. SoftmaxLoss@K), and the differentiable top‑K literature provides practical relaxations. citeturn0search3turn1search0turn1search28turn1search9

**Concrete options**  
- **Listwise loss within event:** treat each event as a list of items; positives are the 3 truth tracks; optimise a surrogate of recall@200 / NDCG@200 or AP‑style retrieval loss. citeturn1search6turn1search30turn1search18  
- **Differentiable top‑K mask:** replace hard top‑K selection in a cascade with a differentiable approximation to propagate gradients end‑to‑end. citeturn1search0turn1search12turn1search9turn1search28

**Expected quantitative impact**  
- **ΔR@200: +0.02 to +0.06** (expected ~+0.03). Larger impact if training is currently strongly misaligned with truncation at 200.

**Data / compute**  
- Mostly compute‑neutral; may slightly increase cost due to sorting/thresholding in differentiable operators.

**Implementation steps**  
1. Implement a listwise baseline: softmax over scores within event; maximise log‑probability mass assigned to the three positives, with down‑weighting of far‑tail negatives.  
2. Add a top‑K truncation surrogate (quantile thresholding / smooth mask). citeturn0search3turn1search12  
3. Compare against your best existing ranking loss on identical data splits.

**Risks**  
- Can improve average recall but reduce stability / increase variance if the surrogate is poorly tuned.  
- Surrogates can be sensitive to batch composition under extreme imbalance.

**Validation**  
- Ablate loss only (keep architecture fixed).  
- Track learning curves for R@200 and especially pT×|dxy_sig| heatmaps (ensure gains occur where needed, not just in easy regions). fileciteturn0file0

### Proposal E: Hard‑region curriculum, confuser‑negative sampling, and stable augmentation

**Rationale**  
The attached diagnostics show a dominant error slice: soft, near‑prompt pions. If the objective is dominated by easy displaced pions, the model may “settle” into a solution that ignores the valley. A targeted curriculum and confuser mining can reallocate gradient budget to the right region, a common tactic in extreme‑imbalance retrieval.

**Concrete actions**  
- **Curriculum over pT and |dxy_sig|:** begin training emphasising the valley positives (and their confusable negatives), then gradually anneal back to full distribution to avoid hurting overall performance.  
- **Confuser‑negative sampling:** for each positive, sample negatives matched in (pT, |dxy_sig|, hit counts, χ²) to focus on hard discriminations rather than random negatives.  
- **Fix copy‑paste augmentation engineering:** implement deterministic, deadlock‑free augmentation (e.g. offline augmentation cache, or augmentation as a preprocessing step rather than in dataloader workers). fileciteturn0file0

**Expected quantitative impact**  
- **ΔR@200: +0.02 to +0.06** (expected ~+0.04), mostly by raising valley recall without harming high‑pT regions.

**Data / compute**  
- No new data required.  
- Compute modestly increased if extra sampling logic is heavy; can be done offline.

**Risks**  
- Over‑fitting to a narrow slice; may reduce generalisation or harm displaced‑track performance.  
- If valley tracks are fundamentally indistinguishable per‑track, gains may be limited unless combined with relational modelling (Proposal A/C).

**Validation**  
- Report per‑bin recall improvements; ensure global R@200 increases, not just slice‑specific.  
- Compare against a “reweighted loss only” control to isolate whether sampling or weighting drives gains.

### Proposal F: Feature and representation expansion focused on vertex compatibility and track‑fit geometry

**Rationale**  
Current features include only a subset of track‑fit and covariance information. CMS tracking uses a five‑parameter helix representation and rich covariance; vertex reconstruction is a dedicated algorithmic area. citeturn3search8turn2search21 Adding geometry/vertex compatibility features can supply separative information precisely where dxy significance alone fails.

**Candidate feature additions (prioritised)**  
1. **Full helix parameters** (where available): d0, z0, φ0, tanλ (or cotθ), q/pT; plus additional covariance elements beyond φφ and λλ. citeturn2search21turn3search8  
2. **Track‑to‑PV association features:** dz, dxy w.r.t PV, vertex index/quality if present; CMS defines impact parameters and uses them widely in identification contexts. citeturn2search6turn3search8  
3. **Vertex‑compatibility approximations:** pairwise Δz0 compatibility, signed dxy consistency with a common SV direction, or fast approximate two‑track vertex fit quality (even a crude proxy can help). (Grounded in general vertexing practice.) citeturn3search2turn3search8  
4. **Hit pattern / quality flags:** number of strip hits, lost hits, algorithm category, highPurity flag, etc. (Common in CMS track quality, though exact variables depend on your ntuples.) citeturn3search8turn3search27

**Expected quantitative impact**  
- **ΔR@200: +0.01 to +0.05** (expected ~+0.03). Potentially higher if current feature set omits a key discriminant used in standard track quality.

**Data / compute**  
- Requires rebuilding input pipelines and validating feature availability and distributions; training compute similar.

**Risks**  
- Feature drift between simulation and data; may require domain adaptation/calibration (Proposal H).  
- Added features may correlate with pileup conditions and overfit.

**Validation**  
- XGBoost re‑run with new features to establish a new “per‑track ceiling.” fileciteturn0file0  
- Ablation study: add one feature family at a time; monitor not only R@200 but calibration and stability across pileup bins if available.

### Proposal G: Calibration, uncertainty‑aware scoring, and per‑bin equalisation

**Rationale**  
Although calibration does not always change ranking, in practice it can reduce **score compression** (small separations causing rank instability) and improve cascade behaviour (where thresholds and score comparability matter). CMS τ identification work includes explicit calibration between simulation and data, and modern τ taggers incorporate domain adaptation and workflow refinements. citeturn11search22turn11search34turn11search6

**Concrete steps**  
- Apply temperature scaling or isotonic regression **per pT bin** to stabilise score distributions (validation‑only).  
- Train with an auxiliary head predicting **aleatoric uncertainty** (e.g. score variance) and rank by expected utility (e.g. mean + α·uncertainty) to favour “potentially signal” tracks in ambiguous regions.

**Expected quantitative impact**  
- **ΔR@200: +0.01 to +0.03** (expected ~+0.02). Often more valuable for robustness than raw gains.

**Cost / risk**  
- Low cost; low risk.  
- Main risk is over‑tuning to the validation split.

**Validation**  
- Evaluate calibration error (ECE) and rank stability across resamples; verify that R@200 improves or at least does not regress where Proposal A/C depend on thresholds.

### Proposal H: Domain adaptation, sim‑to‑real corrections, and ensemble→distillation

**Rationale**  
Even if the current objective is “beat R@200 on simulation,” deployment in CMS‑like environments often requires mitigation of data–simulation mismodelling. CMS has recently documented **domain adaptation by backpropagation** in DeepTau v2.5 to reduce data‑simulation discrepancies, conceptually aligned with Domain‑Adversarial Neural Networks (gradient reversal). citeturn11search22turn11search34turn11search3turn11search7  
Broader HEP literature also discusses sim→real domain adaptation and reweighting approaches (e.g. DANN in HEP settings, OmniFold‑style reweighting) depending on what supervision is available. citeturn11search1turn11search0

**Concrete plan**  
- Add a **domain classifier head** (data vs simulation) with gradient reversal, trained on a high‑purity control region if accessible (analogy to DeepTau), or on unsupervised mixtures where labels are not needed. citeturn11search34turn11search7  
- Use **ensemble of 3–5 seeds** for development; then **distil** to a single deployable model to recover inference speed (teacher‑student).

**Expected quantitative impact**  
- On *simulation metrics alone*: **ΔR@200: 0.00 to +0.03**.  
- On **real data robustness**: potentially substantial, but requires a data‑driven evaluation plan.

**Risks**  
- Requires careful control‑region definition; otherwise the domain head can suppress genuinely discriminative features.

**Validation**  
- If real data is available, validate score shapes in control samples and monitor domain‑classifier AUC (should approach chance when adaptation is successful). citeturn11search3turn11search7

## Experimental roadmap with timelines, resources, and decision gates

### Roadmap principles

1. **Optimise interactions, not only aggregations.** The record’s strongest evidence is that the fixed (η,φ) kNN interaction graph is the bottleneck. fileciteturn0file0  
2. **Measure where gains occur.** A gain that does not improve the pT×|dxy_sig| valley likely will not reach 0.8 overall. fileciteturn0file0  
3. **Gate by R@200 and P@200 together.** For downstream reconstruction, P@200 is operationally crucial. fileciteturn0file0

### Proposed 10‑week plan starting 2026‑03‑23

| Window (calendar) | Deliverable | Key experiments | Decision gate |
|---|---|---|---|
| Weeks 1–2 (2026‑03‑23 → 2026‑04‑05) | Baseline audit + metric package | Verify dataset counts; implement standard plots; add oracle ceiling checks (reco‑availability). fileciteturn0file0 | If “oracle” R@200 < 0.9, identify label/reco ceiling before further ML work |
| Weeks 2–4 (2026‑04‑06 → 2026‑04‑19) | Loss/sampling/feature quick wins | Proposal D + E (listwise loss, confuser sampling) on fixed architecture; add a small batch of new features (Proposal F‑lite). citeturn0search3turn1search0turn1search30turn3search8 | Proceed if R@200 ≥ 0.67 and valley recall improves measurably |
| Weeks 4–7 (2026‑04‑20 → 2026‑05‑10) | Cascade prototype | Implement Proposal A with K₁ sweep; baseline A2 as pairwise‑bias transformer; measure compute/latency. citeturn5search2turn4search3turn1search12 | Continue if best K₁ configuration reaches R@200 ≥ 0.75 |
| Weeks 7–9 (2026‑05‑11 → 2026‑05‑24) | Triplet rescue / set‑prediction optional branch | Add Proposal C (beam‑search triplet rescue) on top of cascade; in parallel a minimal Proposal B prototype on K₁ input. citeturn5search1turn5search0turn2search0 | Stop when R@200 ≥ 0.80 and P@200 improves meaningfully |
| Weeks 9–10 (2026‑05‑25 → 2026‑06‑01) | Hardening and deployment prep | Calibration (Proposal G), runtime optimisation, ablation report; optionally domain‑adaptation plan (Proposal H). citeturn11search34turn11search7 | If on data: require stable score shapes and no large sim→data drift |

### Resource estimate (indicative)

- **Quick‑win phase (Weeks 2–4):** ~10–20 full training runs → **~100–400 GPU‑hours** total (depends on model and batch size).  
- **Cascade phase (Weeks 4–7):** fewer but heavier runs + K₁ sweeps → **~200–600 GPU‑hours**.  
- **Total:** **~300–1,000 GPU‑hours** over ~10 weeks for a focused effort.

### Mermaid flowchart for decision gates

```mermaid
flowchart TD
  A[Baseline audit\nmetrics + oracle ceilings] --> B{Ceiling OK?}
  B -- No --> B1[Fix labels/reco ceilings\ntruth matching + masking]
  B -- Yes --> C[Quick wins\nloss + sampling + features]
  C --> D{R@200 >= 0.67\n& valley improves?}
  D -- No --> C1[Iterate quick wins\nor reassess labels/features]
  D -- Yes --> E[Cascade prototype\nK1 sweep + reranker]
  E --> F{R@200 >= 0.75?}
  F -- No --> E1[Adjust K1 / reranker\npairwise bias / sparse attention]
  F -- Yes --> G[Triplet rescue or set prediction]
  G --> H{R@200 >= 0.80?}
  H -- Yes --> I[Hardening\ncalibration + runtime + ablations]
  H -- No --> G1[High-risk branch\n(set prediction / OC-style clustering)]
```

## Suggested visualisations and an analysis package

To make progress measurable and to avoid “global metric mirages,” the following plots should be standard in every experiment report:

- **Recall vs K curves (R@K)** for K=50..800, with annotated points at 200/300/500/600 (ties directly to cascade design). fileciteturn0file0  
- **P@K curves** (probability all 3 are captured), because downstream reconstruction depends on full triplets. fileciteturn0file0  
- **Recall vs pT** and **recall vs |dxy_sig|** (the record already shows these isolate the uncanny valley). fileciteturn0file0  
- **2D heatmap:** recall in (pT × |dxy_sig|) bins to prove valley gains.  
- **GT rank distribution:** histogram / CDF of each pion’s rank; include p75 and p90 (the record quotes p75~355 and p90~596, which are directly relevant to choosing K₁). fileciteturn0file0  
- **Confusion matrix at a fixed K:** treat “selected in top‑200” as predicted positive; compute rates for signal vs background (useful for stability and trade‑off discussions, even if retrieval is primary).  
- **Feature importance**:
  - SHAP values for XGBoost (baseline ceiling) to understand which features saturate,  
  - integrated gradients / attention rollout for relational models (especially to test whether ρ‑mass features are being used). citeturn5search2turn4search3  
- **Calibration curves** (reliability diagrams) if using cascade thresholds or domain adaptation. citeturn11search34turn11search7

A small but valuable addition is a **standard “failure atlas”**: for each model version, automatically save a fixed set of 50–100 hardest events (by number of missed GT pions, or by lowest valley recall) and produce per‑event diagnostic panels (scores, key features, neighbour stats). This prevents regressions being hidden by averages.

---

**Cited baseline source:** the attached experiment summary is referenced throughout for all current‑system details, diagnostics, and plateau evidence. fileciteturn0file0