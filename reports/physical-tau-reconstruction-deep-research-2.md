# Untapped physics for low-pT tau triplet selection

**The CMS ABCNet algorithm for τ → 3π ν reconstruction leaves substantial discriminating power unused.** The most consequential omission is invariant mass information — the m(π⁺π⁻) Dalitz structure from the a₁(1260) → ρ(770)π cascade and the m(3π) < m_τ kinematic bound — which together could reject 70–90% of combinatorial triplets that survive the current score-and-rank approach. Beyond invariant masses, cascade vertex topology, B-decay kinematic anchors, and Lorentz-invariant angular correlations each offer independent discrimination that the baseline's feature set of {pT, η, φ, mass, charge, d_PV} cannot learn. This report quantifies each physics handle and identifies the most promising ML architectures to exploit them.

## The ρ–a₁ resonance cascade dominates the Dalitz plot

The τ → 3π ν decay proceeds almost exclusively through **τ⁻ → a₁⁻(1260) ν_τ → ρ⁰π⁻ → π⁺π⁻π⁻ ν_τ**, with the ρπ intermediate state accounting for **~75% of the total rate**. The remaining contributions come from σ(500)π (~15% of amplitude squared) and small f₀(1370)π, f₂(1270)π, and D-wave ρπ terms, all producing strong interference patterns. The a₁ resonance parameters are broad but well-measured: **m(a₁) ≈ 1233 ± 18 MeV, Γ(a₁) ≈ 431 ± 20 MeV** from combined ALEPH/ARGUS/OPAL/CLEO fits. The ρ(770) is sharper: **m(ρ) = 775.3 MeV, Γ(ρ) = 147.4 MeV**.

The practical consequence for triplet selection is dramatic. In the Dalitz plot spanned by the two m²(π⁺π⁻) combinations, genuine tau decays concentrate into **two prominent ρ bands** at m(π⁺π⁻) ≈ 770 MeV with constructive/destructive interference patterns between them. Random combinatorial triplets, by contrast, populate the Dalitz plot approximately uniformly across phase space. This creates two powerful discriminants that ABCNet currently ignores entirely:

- **Pairwise invariant mass m(π⁺π⁻)**: For each triplet with charge pattern (−−+), the two opposite-sign pairings should show at least one combination near the ρ mass. Computing both m(π⁺π⁻) values and checking consistency with the ρ band provides rejection that is fundamentally Lorentz-invariant and does not degrade at low pT. At pion momenta of 0.5–1 GeV, the ρ band remains well-defined because the ρ width (147 MeV) is large relative to the mass resolution at these momenta (~5–15 MeV).

- **Triplet invariant mass m(3π)**: This must satisfy the hard kinematic bound **m(3π) < m_τ = 1.777 GeV** and peaks around 1.0–1.3 GeV following the a₁ resonance shape. For random soft-track triplets with individual pT ~ 0.5–1.5 GeV, the invariant mass distribution extends well beyond 2 GeV, especially for tracks separated by ΔR > 0.4. A simple m(3π) < 1.777 GeV cut eliminates a large fraction of combinatorial background, while tightening to the a₁ peak region (0.8–1.5 GeV) captures ~70–80% of signal while further suppressing backgrounds.

The fact that ABCNet uses only per-track features (4-momentum, charge, d_PV) means it **cannot compute invariant masses between track pairs or triplets** — these are inherently relational features. The network may learn weak proxies via angular and momentum correlations, but it cannot reproduce the sharp resonance structures. This is arguably the single largest source of untapped discriminating power.

## Angular correlations encode the full V−A decay dynamics

At visible tau pT ~ 3–5 GeV (γ ≈ 2–3), the three pions spread over **ΔR ~ 0.4–1.0** in η–φ space, far exceeding the standard CMS HPS signal cone maximum of 0.1. CMS confirmed this directly in the "opening up the tau cone" analysis (EXO-24-012), which demonstrated that for low-mass parent particles like Υ(1S) → ττ, decay products reach ΔR ~ 0.5–1.5. However, the *relative* angular structure among the three pions remains tightly constrained by the decay dynamics, even as the absolute cone opens.

The Kühn–Mirkes formalism describes the τ → 3π ν decay through **16 structure functions** that encode the complete angular and mass dependence. The three pion momenta define a decay plane, and the orientation of this plane relative to the tau flight direction carries information about the tau polarization (measured by OPAL to be consistent with pure V−A: γ_VA ≈ 1.0–1.3). The **acoplanarity angle ψ** — the rotation angle around the decay-plane normal — produces non-trivial distributions governed by parity-violating structure functions W_C through W_I.

Three specific angular discriminants emerge as most powerful for separating genuine triplets from random combinations:

- **Decay-plane flatness**: Three genuine tau daughters define a plane consistent with a single decay vertex. The normal vector to this plane, combined with the 3π momentum direction, creates a signed angle distribution that is non-uniform for signal but isotropic for random tracks.
- **Pion ordering in the decay plane**: The a₁ → ρπ helicity structure (predominantly S-wave, with D/S ratio ≈ −0.09) dictates specific angular distributions of the ρ-daughter pions relative to the bachelor pion. The ρ is predominantly longitudinally polarized, producing a cos²θ distribution in the ρ rest frame.
- **Triple product asymmetry**: The parity-odd observable **ε_μνρσ p₁^μ p₂^ν p₃^ρ q^σ** (where q is the 3π momentum) has a nonzero expectation value for genuine V−A decays but vanishes on average for random combinations. CLEO measured this asymmetry with **~10% precision** in tau decays, confirming its discriminating power.

At low boost, these angular correlations are paradoxically *more* experimentally resolvable than at high pT because the tracks are well-separated in the detector. The main degradation comes from the poorer angular resolution per track at low momentum, but for pT > 0.5 GeV the CMS tracker angular resolution (σ_φ ~ 1 mrad, σ_η ~ 0.001) remains adequate.

## The B→τ cascade creates a distinctive double-displacement signature

The vertex topology of Bc → J/ψ τν followed by τ → 3πν produces a cascade **PV → Bc vertex → τ vertex** with characteristic distances that CMS can resolve. The quantitative displacement budget is:

| Segment | cτ | βγ (typical) | Mean L_xy |
|---|---|---|---|
| Bc flight (PV → Bc vertex) | **154 μm** | 2.4–3.2 | **370–490 μm** |
| τ flight (Bc vertex → τ vertex) | **87 μm** | 2.3–3.9 | **200–340 μm** |
| Total (PV → τ vertex) | — | — | **570–830 μm** |

For B⁰ → D*τν, the numbers are even more favorable: cτ(B⁰) ≈ 455 μm gives mean L_xy ≈ 1.3–2.3 mm, producing total PV-to-τ displacements of **1.5–2.6 mm**. These distances are comfortably within CMS vertex resolution.

The challenge lies in the **impact parameter resolution at 0.5–1 GeV**. Multiple Coulomb scattering dominates the CMS Phase-1 pixel detector performance at low momentum, producing σ(d₀) that scales roughly as 30–40/pT μm·GeV in the barrel:

| pT (GeV) | σ(d₀) barrel [μm] | σ(d_z) barrel [μm] | d₀ significance (Bc chain) |
|---|---|---|---|
| 0.5 | **90–150** | 250–400 | 0.3–2.0σ |
| 0.7 | 60–100 | 170–280 | 0.5–3.3σ |
| 1.0 | **35–50** | 100–150 | 1.0–5.7σ |
| 1.5 | 22–35 | 65–100 | 1.4–9.1σ |

At pT = 0.5 GeV, individual track impact parameter significance is **marginal** (1–2σ) for the Bc chain. This is precisely why the ABCNet's simple "distance from PV" feature provides limited leverage at the softest momenta. However, three critical improvements exist beyond simple d₀:

**Pointing angle** — the angle between the PV-to-vertex vector and the 3π momentum — approaches zero for genuine tau decays (where the slight offset comes from the undetected neutrino) but is randomly distributed for fake combinations. This is one of the most powerful single vertex-related discriminants and is extensively used by LHCb.

**Cascade vertex fit** — simultaneously fitting the PV, B vertex (using J/ψ or D* tracks plus the 3π system), and τ vertex (3π tracks alone) enforces geometric and kinematic consistency across the full decay chain. This exploits correlations between the three vertex positions that independent vertex fits miss. The B flight direction provides a strong prior for the τ flight direction, and mass constraints (m(3π) < m_τ, the known B mass) reduce the effective degrees of freedom. Based on analogies with LHCb cascade-fit studies, this approach could yield **20–50% improvement** in signal/background over independent fits.

**Vertex isolation** — genuine τ → 3π decays produce a clean 3-track secondary vertex with no additional tracks. Counting extra tracks near the vertex rejects backgrounds from D_s → 3π or b/c-jet fragments, which typically have higher track multiplicity at their decay vertices.

## Charge-momentum correlations reveal the bachelor pion

The τ⁻ → π⁻π⁻π⁺ν_τ charge pattern (two same-sign plus one opposite-sign pion) immediately reduces combinatorial background by a **factor of ~4**, since only one of four possible charge-sign combinations matches. But deeper correlations exist within this topology.

In the dominant a₁⁻ → ρ⁰π⁻ channel, the "bachelor" pion (the like-sign π⁻ not from ρ decay) has a characteristic energy in the a₁ rest frame of **E_bachelor ≈ (m²_a₁ + m²_π − m²_ρ)/(2m_a₁) ≈ 380 MeV**. The two ρ-daughter pions share the ρ's energy according to its helicity structure. Because the ρ carries significant momentum from the a₁ decay, the opposite-sign pion (π⁺ from ρ⁰ → π⁺π⁻) tends to be harder than the bachelor pion on average.

A practical discriminant exploits the two-fold ambiguity: for the two possible π⁺π⁻ pairings, one should reconstruct near **m_ρ ≈ 770 MeV** while the other gives a broad, non-resonant distribution. A simple "minimum |m(π⁺π⁻) − m_ρ|" variable identifies the correct ρ pairing with high efficiency and simultaneously validates that the triplet has genuine resonance substructure. Random combinatorial triplets produce both pairings distributed broadly, failing this test.

## Combinatorial scale demands efficient pre-filtering

The raw combinatorial challenge at CMS is severe. In a typical Run 2/3 event with **⟨N_PU⟩ ≈ 35** pileup interactions, approximately **1,100–1,300 charged tracks** with pT > 0.5 GeV exist within |η| < 2.4 before any vertex association. After Charged Hadron Subtraction (CHS) removes pileup-associated tracks, **50–200 PV-associated tracks** remain, depending on the hard process. The combinatorial triplet count scales as C(N,3):

For 100 PV-associated tracks, there are **~162,000 possible triplets**; for 200 tracks, **~1.3 million**. The charge requirement (|Q| = 1) reduces this by ~4×, and a m(3π) < 1.777 GeV cut provides further substantial reduction. But the remaining combinatorial burden is still enormous, motivating efficient ML-based pre-filtering.

The kinematic distributions of background tracks differ from tau daughters in predictable ways. Minimum-bias tracks peak at pT ≈ 0.3–0.5 GeV with mean **⟨pT⟩ ≈ 0.55 GeV** and are approximately uniform in η and φ. Tau daughter pions, by contrast, cluster within ΔR ~ 0.4–1.0 of each other, have a flatter pT spectrum extending to ~3 GeV for the leading pion, and all point back toward a common displaced vertex. The spatial clustering, vertex consistency, and combined invariant mass provide complementary handles that together can reduce the combinatorial background by orders of magnitude.

## B-decay kinematics provide powerful global constraints

When the tau originates from Bc → J/ψ τν with J/ψ → μ⁺μ⁻ fully reconstructed, the precisely measured J/ψ 4-momentum anchors the entire event topology. The allowed kinematics are tightly constrained:

- **q² range**: The squared momentum transfer q² = (p_τ + p_ν)² is bounded between **m²_τ = 3.16 GeV²** and **(m_Bc − m_J/ψ)² = 10.09 GeV²**.
- **Missing mass**: m²_miss = (p_Bc − p_J/ψ − p_3π)² must be **≥ 0** for a physical solution. Wrong pion combinations frequently produce unphysical (negative) m²_miss values, providing a strong rejection handle. Computing this requires approximating the Bc momentum direction (from the PV-to-Bc-vertex vector) even though the Bc momentum magnitude is unknown due to the two neutrinos.
- **Visible mass**: m(J/ψ, 3π) must be **< m_Bc = 6.274 GeV**, and the distribution peaks well below this maximum.
- **Corrected mass**: The variable M_corr = √(m²_vis + p²_⊥) + p_⊥, where p_⊥ is the 3π momentum component perpendicular to the B flight direction, partially reconstructs the neutrino momentum. LHCb requires **M_corr ∈ [1.2, 2.5] GeV** for tau candidates in B_s → ττ searches, achieving strong background rejection.
- **Angular separation**: ΔR(J/ψ, 3π) is typically **0.1–0.5** for typical Bc boosts of pT ~ 15–30 GeV, providing a geometric constraint on where correct pion triplets should be found relative to the J/ψ.

The theory framework of Li et al. (JHEP 05, 2023, 173) derives **34 normalized observables** from the five-fold differential distribution of Bc → J/ψ(→μμ) τ(→hadrons) ν̄_τ expressed entirely in terms of visible kinematics. These observables encode the full spin-correlation structure and include angular variables in approximate Bc and τ rest frames that carry significant discriminating power, particularly for separating signal from the Bc → J/ψ D_s(→3π) background.

## Modern architectures can exploit relational features the ABCNet misses

The ABCNet architecture (Mikuni & Canelli, 2020) uses a GNN with attention-based graph pooling layers (GAPLayers), constructing k-nearest-neighbor graphs over point clouds of tracks. Its per-track scoring approach is well-suited for filtering but fundamentally limited: it processes individual tracks in context but does not explicitly compute relational features like pairwise invariant masses or triplet-level variables.

Several architectures designed for combinatorial assignment problems could substantially improve performance:

**SPANet** (Symmetry Preserving Attention Networks) was built specifically for assigning detected particles to resonance decay products. It uses symmetric tensor attention to solve the combinatorial assignment without enumerating permutations, improving reconstruction efficiency by **19–35%** on jet-parton matching benchmarks while reducing inference time by 2–5 orders of magnitude compared to brute-force approaches. Its partition-and-assign strategy maps directly onto the τ → 3π problem: assign 3 tracks from N candidates to the tau decay, respecting the (−−+) or (+−+) charge symmetry.

**PELICAN** achieves state-of-the-art jet classification using only **pairwise Lorentz dot products** as inputs — exactly the invariant mass information missing from ABCNet. With only ~45,000 parameters, it matches or exceeds models with millions of parameters, demonstrating that physics-informed input representations dramatically improve sample efficiency and generalization. Adapting PELICAN's pairwise-dot-product approach to the track-triplet problem would naturally incorporate the ρ and a₁ resonance structures.

**LorentzNet** provides Lorentz-equivariant message passing using Minkowski dot product attention, maintaining exact symmetry while learning from 4-momentum interactions. The **Particle Transformer** uses pairwise interaction features (ΔR, k_T, invariant mass between pairs) as attention biases, providing a lighter-weight approach to incorporating relational physics.

For the specific low-pT tau problem, the most promising strategy combines SPANet's combinatorial assignment framework with PELICAN-style Lorentz-invariant pairwise features. The input representation should include:

- Per-track features: pT, η, φ, charge, d₀, d_z, d₀ significance, track quality
- Pairwise features: m(π_i π_j) for all pairs, ΔR(π_i, π_j), Lorentz dot products p_i · p_j
- Triplet-level features: m(3π), Dalitz-plot coordinates, decay-plane normal, pointing angle
- Event-level features: J/ψ 4-momentum, approximate B flight direction, corrected mass, m²_miss

## The neutral pion question is largely a background consideration

The inclusive branching fractions are τ⁻ → π⁻π⁺π⁻ν_τ = **8.99%** (exclusive 3-prong) versus τ⁻ → π⁻π⁺π⁻π⁰ν_τ = **2.74%** (plus 1.95% through ω → π⁺π⁻π⁰). The 4-pion mode with a lost π⁰ is the dominant contamination of the 3-prong sample, producing a m(3π_visible) distribution shifted to higher masses (since the missing π⁰ mass is not accounted for).

At pion pT ~ 0.5–1.5 GeV, the associated π⁰ typically has pT ~ 0.3–1.0 GeV, producing photons with pT ~ 0.15–0.5 GeV each — often below the CMS ECAL strip reconstruction threshold of 0.5 GeV. Reliable π⁰ detection is therefore not feasible at these energies. However, the presence of *any* detectable ECAL activity near the 3π vertex could serve as a soft veto against the 4π background. This is a minor handle compared to the invariant mass and vertex features.

## Conclusion: a prioritized feature roadmap

The research identifies a clear hierarchy of untapped discriminating power for the low-pT tau triplet selection problem. **Invariant mass features are the highest-priority addition** — m(3π), the two m(π⁺π⁻) combinations, and ideally the full Dalitz-plot coordinates — because they exploit the dominant a₁→ρπ resonance structure that random combinatorial triplets cannot fake. These features are Lorentz-invariant and do not degrade with decreasing pT, making them especially valuable in the 2–10 GeV regime where most other discriminants weaken.

The second-priority improvements are **cascade vertex features**: the pointing angle of the 3π system relative to the B flight direction, a full cascade vertex fit (PV → B → τ) rather than independent fits, and vertex isolation. These exploit the distinctive double-displacement topology of B → τ decays that simple d₀-from-PV cannot capture.

Third, **B-decay kinematic anchors** — m²_miss, corrected mass, and angular separation from the J/ψ — provide global event-level constraints that can reject entire classes of wrong combinations. These are analysis-specific (requiring knowledge of the parent decay) but extremely powerful when available.

Finally, the **architecture choice matters**: transitioning from ABCNet's per-track scoring to an SPANet-style or Particle Transformer–style framework that natively computes relational features between track candidates would unlock all of these physics handles simultaneously, rather than requiring them to be hand-engineered as input features. The demonstrated success of PELICAN with only ~45k parameters using pairwise Lorentz dot products suggests that a lightweight, physics-informed architecture could substantially outperform the current approach while remaining computationally tractable for the O(10⁵) triplet-scale combinatorial problem.