# Low‑pT τ⁻→π⁻π⁻π⁺ν_τ reconstruction with only charged pions detected

## Executive summary

The three‑prong hadronic τ decay τ⁻→π⁻π⁻π⁺ν_τ is both common and information‑rich: its inclusive branching fraction (excluding K⁰ and ω contributions) is **0.0899 ± 0.0005**. citeturn19view0 The hadronic system is dominantly **axial‑vector** and proceeds mainly through an intermediate **a₁(1260)** resonance that decays to **ρπ**, with **interference** between the two possible ρπ combinations because the final state contains two identical π⁻. citeturn32view3turn36view0turn28view0 In the low‑p_T regime, the τ mass and three‑body kinematics cause the three charged pions to be spatially **less collimated**—in CMS simulation, the maximum pairwise ΔR among the three pions exceeds **0.4** at low visible p_T, and the efficiency of capturing all three pions inside a single R=0.4 jet falls to **≈0% below ~3 GeV** visible p_T. citeturn44view0 This undermines jet‑seeded τ reconstruction and motivates **track‑only, vertex‑centric** candidate building.

For reconstruction when **only charged pions** are detected and there is abundant **combinatorial low‑p_T pion background**, the most robust, physics‑guided discriminants are:

* **Topological consistency**: three tracks of total charge −1 forming a **common displaced secondary vertex** with good fit probability; τ flight‑length significance and “pointing” back to a candidate PV; per‑track impact parameter significance patterns consistent with a single mm‑scale decay. (τ: m = **1776.93 ± 0.09 MeV**, mean life **(290.3 ± 0.5) fs**, cτ = **87.03 μm**.) citeturn40view0  
* **Hadronic‑current (matrix‑element) structure** encoded into **invariant masses and Dalitz‑like variables**: (i) m(3π) is bounded by m_τ and typically populates the a₁ region; (ii) m(ππ) has a pronounced ρ(770) structure with interference; (iii) symmetrised Dalitz features (s_high, s_low) separate τ→a₁→ρπ from random triplets. citeturn12view0turn20view0turn46view0turn36view0  
* **Low‑p_T detector realism**: multiple scattering dominates angular and impact‑parameter resolution, scaling approximately as 1/p and √(x/X₀) (PDG gives the standard θ₀ formula). This must be treated as a first‑class nuisance in ML via augmentation and calibrated uncertainties. citeturn15view2  

The report below turns these physics properties into concrete, testable ML hypotheses: (a) candidate‑graph models that learn compatibility among tracks under charge/vertex/mass constraints; (b) hybrid architectures that combine per‑track features with symmetrised Dalitz features and vertex/flight observables; (c) losses that enforce **exact charge**, **mass endpoints**, and **vertex‑consistency**; (d) simulation‑to‑data adaptation targeted at low‑p_T tracking artefacts (fakes/splits, inefficiency, multiple scattering).

## Hadronic dynamics and matrix‑element structure in τ→3πν

### Dominant resonant picture and key physical scales

The decay is dominated by an **axial‑vector** hadronic current (odd number of pions; vector contribution suppressed in the isospin limit), produced predominantly via an intermediate **a₁(1260)** resonance. citeturn36view0 In PDG listings, a₁(1260) is broad: a “mass” estimate around **1230 ± 40 MeV** and width in the rough range **250–600 MeV** are quoted, and a pole estimate is given near **(1209\_{−10}^{+13}) − i(288\_{−12}^{+45}) MeV**. citeturn20view0

The **ρ(770)** dominates the ππ subsystem dynamics; PDG gives (for the neutral ρ mass average) **775.26 ± 0.23 MeV**, and the charged/τ‑derived determinations are consistent with this scale. citeturn12view0 These two scales (m_ρ and m_{a₁}) set the characteristic shapes of m(ππ) and m(3π) distributions used for discrimination.

### General Lorentz‑invariant hadronic current decomposition (TAUOLA / RχL baseline)

For the three‑pion final state (π⁻,π⁻,π⁺) with momenta p₁,p₂,p₃, Lorentz invariance fixes the general decomposition of the hadronic current used in RχL‑based implementations and discussed in the context of TAUOLA upgrades:

\[
J^\mu = N\left\{ T^{\mu\nu}\left[(p_2-p_3)_\nu F_1 - (p_3-p_1)_\nu F_2\right] + q^\mu F_4 - \frac{i}{4\pi^2} c_5\, \varepsilon^{\mu\nu\rho\sigma} p_{1\nu}p_{2\rho}p_{3\sigma} F_5\right\},
\]
with \(q^\mu = (p_1+p_2+p_3)^\mu\) and \(T^{\mu\nu} = g^{\mu\nu} - q^\mu q^\nu/q^2\). citeturn46view0

In the isospin limit, **F₅ vanishes by G‑parity** for the three‑pion mode, so the dominant dynamics are in F₁, F₂ (axial transverse) and a small scalar form factor F₄. citeturn46view0 This implies strong, structured correlations among the three pion momenta beyond uniform phase space—exactly the kind of structure that random triplets of low‑p_T pions generally do not reproduce.

A common invariant‑mass basis is:
* \(q^2 = m^2(3\pi)\),
* \(s_1 = (p_2+p_3)^2\),
* \(s_2 = (p_1+p_3)^2\),
with \(s_3 = (p_1+p_2)^2 = q^2 - s_1 - s_2 + 3m_\pi^2\). citeturn46view0

Within Resonance Chiral Lagrangian (RχL) parameterisations used for TAUOLA currents, the form factors are often written schematically as a sum of chiral, one‑resonance, and double‑resonance pieces:
\[
F_i = (F_i^\chi + F_i^R + F_i^{RR}) \cdot R_{3\pi}, \quad i=1,2,4,
\]
with \(R_{3\pi}=1\) for π⁻π⁻π⁺. citeturn46view0

### Kühn–Santamaría and TAUOLA implementations: ρπ dominance and interference

In the ALEPH spectral‑function analysis, the τ→3πν channels in detector simulation used **TAUOLA 1.5** with the **Kühn–Santamaría** parametrisation, describing a dominant a₁(1260) resonance (they note a₁ width parameter **~0.4 GeV** in that setup) decaying into **ρπ**, explicitly including **interference between the two ρπ combinations**. citeturn32view3 This interference arises because either π⁻ can pair with π⁺ to form a ρ candidate. In data‑driven reconstruction, this motivates **symmetrised pairwise observables** (e.g. s_high/s_low) rather than using an arbitrary π⁻ label.

### Beyond pure ρπ: isoscalars (σ/f₀) and phenomenological extensions

Empirically, purely “simple” ρπ models often fail in some regions of the Dalitz plane. In OPAL’s studies, neither Kühn–Santamaría nor Isgur‑type models gave a fully satisfactory description of the data. citeturn24view0 CLEO’s model‑dependent analysis notes that, in addition to dominant a₁→ρπ (S‑wave), **isoscalar contributions** in a₁ decay—especially **a₁→f₀(600)π**—can be large, while contributions via π(1300) are small and PCAC‑suppressed. citeturn36view0

From the RχL‑upgrade perspective, discrepancies in two‑pion invariant mass distributions led to explicit discussion of missing **σ (f₀(600))** contributions and a phenomenological inclusion via a Breit–Wigner term in the form factors (mirroring earlier CLEO‑style approaches). citeturn46view0 This matters for reconstruction because it predicts additional population in low‑m(ππ) regions (broad scalar), altering the “ρ‑peak only” expectation and thus preventing overly rigid cuts that would lower τ efficiency.

## Kinematics, masses, Dalitz‑like variables, and angular correlations

### Hard kinematic constraints and endpoints

Even without neutrino reconstruction, τ→3πν obeys strict constraints:

* **Invariant‑mass endpoint:** \(m(3\pi) < m_\tau = 1776.93 \pm 0.09\) MeV. citeturn40view0  
  In practice, detector resolution and wrong‑combinations can yield a high‑mass tail above m_τ; modelling that tail is useful for background estimation and ML calibration.
* **Charge constraint:** total track charge must be **−1** for τ⁻ candidates (or +1 for τ⁺). This is exact and should be enforced deterministically in candidate generation.
* **Resonance‑driven scales:** strong density near **m(ππ) ≈ m_ρ ≈ 775 MeV** and broad structure near **m(3π) ≈ m_{a₁} ~ 1.2 GeV**. citeturn12view0turn20view0  

These are attractive discriminants against combinatorial triplets, which tend to fill phase space more smoothly once acceptance effects are accounted for.

### Dalitz‑like variables and recommended symmetrisations

Because there are two identical π⁻, there are two π⁺π⁻ invariant masses per candidate. A robust, label‑free encoding is:

* \(s_{\mathrm{high}} = \max\{m^2(\pi^+\pi^-_1),\, m^2(\pi^+\pi^-_2)\}\)  
* \(s_{\mathrm{low}} = \min\{m^2(\pi^+\pi^-_1),\, m^2(\pi^+\pi^-_2)\}\)

This corresponds to CLEO’s convention where **s₁ is the larger** π⁺π⁻ mass‑squared and s₂ the other one. citeturn36view0 In OPAL visualisations and fits, one half of the Dalitz plane is mirrored to the other for clarity, reflecting the same exchange symmetry. citeturn45view1turn46view0

**Signal patterns:** τ→a₁→ρπ tends to populate regions where one π⁺π⁻ combination lies near the ρ band (m² ≈ 0.6 GeV²), while the other combination and s₃ accommodate remaining phase space; interference smears the band and induces characteristic asymmetries that differ from uniform 3‑body phase space. citeturn32view3turn46view0  
**Combinatorial background patterns:** random triplets typically lack a pronounced ρ band in (s_low,s_high) once same‑event correlations are removed; instead, they reflect detector/production biases (soft pions, near‑PV origins, local track density).

### Energy sharing and “softest‑track” motifs

In the τ rest frame, the neutrino removes a variable fraction of energy; in the lab with only π measured, a useful proxy is the **visible energy fraction partition** among the three pions:
\[
z_i = \frac{|\vec{p}_i|}{|\vec{p}_1|+|\vec{p}_2|+|\vec{p}_3|} \quad \text{or} \quad z_i = \frac{p_{T,i}}{\sum_j p_{T,j}} \ \text{(in the transverse plane)}.
\]
In a pure ρπ picture, two pions forming the ρ often share a correlated momentum scale, while the bachelor pion can be softer/harder depending on helicity configuration and q². The presence of broad a₁ and possible σ contributions broadens these patterns. citeturn36view0turn46view0  

For ML, these z‑variables are less brittle than attempting to fully reconstruct the τ rest frame; they also directly capture low‑p_T pathologies (missing the softest track pushes z‑distributions toward two‑body‑like configurations).

### Angular correlations and decay‑plane observables with only tracks

Even without neutrino measurement, several rotationally meaningful observables exist purely from the three pion 3‑momenta:

* **3π decay‑plane normal:** \(\hat{n} \propto \vec{p}_{\pi_a}\times\vec{p}_{\pi_b}\) in the lab (choose any two pions; symmetrise by averaging over choices or use a signed convention tied to π⁺).  
* **Acollinearity / planarity:** e.g. \(\mathrm{Det}[\hat{p}_1,\hat{p}_2,\hat{p}_3]\) or the smallest eigenvalue of the momentum tensor \(\sum_i \hat{p}_i\hat{p}_i^\top\).  
* **Helicity‑proxy angles:** With an estimate of the τ flight direction from PV→SV (see below), define angles of pion momenta in the 3π rest frame relative to that direction (approximate; sensitive to resolution).

OPAL’s analyses emphasise **structure functions** and Dalitz projections tied to hadronic dynamics and neutrino helicity. citeturn24view0turn45view0 In ML terms, decay‑plane and triad‑geometry features provide a compressed representation of these correlations that is often harder for backgrounds to mimic coherently.

## Polarisation, spin correlations, and low‑p_T manifestations

### Production polarisation (Z/W sources) and what survives without ν reconstruction

τ polarisation depends on production:

* In Z→τ⁺τ⁻ at the pole, CMS measures **P_τ(Z) = −0.144 ± 0.015**, and defines \(P_\tau = (\sigma_+ - \sigma_-)/(\sigma_+ + \sigma_-)\). citeturn38view0  
* In W→τν, ATLAS measures τ polarisation close to the left‑handed expectation, reporting **P_τ ≈ −1.06** with quoted uncertainties and a physical interval consistent with −1. citeturn38view3  
* In τ→3πν itself, OPAL measured parity‑violating asymmetry parameters consistent with left‑handed τ neutrinos (Standard Model). citeturn28view0turn45view2  

At low p_T, two effects reduce naïve polarisation sensitivity if one uses only the visible 3π:

1. **Boost dilution:** angular information in the lab becomes less aligned with the τ rest frame as the τ becomes less boosted. This is exactly the regime where track‑based reconstruction is hardest.  
2. **Direction ambiguity:** without ν, the τ momentum direction is not known; however, one can approximate it by the **flight direction** inferred from PV→SV (for displaced τ), turning polarisation extraction into a vertex‑assisted problem.

### Polarimeter vectors for τ→3πν and track‑only approximations

The optimal way to use τ→3πν for spin information is through a **polarimeter vector** constructed from the hadronic current. A recent dedicated treatment provides an algorithm for computing the polarimeter vector \(h\) for τ→3πν based on a hadronic current model fitted to CLEO data. citeturn39view0

That work starts from a CLEO‑style current ansatz:
\[
J^\mu = B_{a_1}(s)\sum_i \beta_i\, j_i^\mu,
\]
with
\[
B_{a_1}(s) = \frac{1}{s - m_{0,a_1}^2 + i\, m_{0,a_1}^2\, \Gamma^{\mathrm{tot}}_{a_1}(s)},
\]
and tabulated resonance contributions including ρ, ρ(1450), f₂, σ, f₀(1370), etc. citeturn39view0

For reconstruction (not precision spin measurement) this suggests a pragmatic hypothesis: **even without full τ rest‑frame reconstruction**, a “pseudo‑polarimeter” built from (i) the **3π rest frame**, and (ii) the **SV→PV direction as τ axis**, can supply additional discrimination because random triplets usually lack coherent spin‑driven correlations tied to a single parent direction. The same reference notes that TauSpinner uses such CLEO‑based currents for reweighting τ polarisation and spin correlations. citeturn39view0turn38view1

### Practical reweighting tools (TauSpinner; TAUOLA interface)

**TauSpinner** provides event‑by‑event weights to (re)introduce or modify τ spin correlations using only τ kinematics and decay products, useful when full spin density matrices are not stored. citeturn38view1  
The **TAUOLA universal interface** separates τ production from τ decay, with spin being the key connector, enabling consistent modelling of spin effects in event records. citeturn38view2  

For ML development, these tools support controlled experiments: train a model on unpolarised τ decays, then reweight to polarised samples and test whether candidate selection efficiency or score calibration is stable as a function of polarisation—a direct probe of spin‑related domain shift.

## Lifetime, vertex displacement, impact‑parameter scales, and secondary‑decay backgrounds

### τ decay length scales in the lab (momenta 2–50 GeV)

The mean decay length in the lab is:
\[
L \approx \beta\gamma\, c\tau \simeq \frac{p}{m_\tau}\,c\tau,
\]
with \(c\tau = 87.03\,\mu\mathrm{m}\) and \(m_\tau = 1776.93\,\mathrm{MeV}\). citeturn40view0  

Using \(L \approx (p/1.7769\,\mathrm{GeV})\times 87.03\,\mu\mathrm{m}\), typical mean decay lengths are:

| τ momentum p (GeV) | ⟨L⟩ (mm) | Comment |
|---:|---:|---|
| 2 | ~0.098 | sub‑0.1 mm: vertexing is challenging; IP‑based patterns dominate |
| 5 | ~0.245 | few‑100 μm: feasible with good silicon vertexing |
| 10 | ~0.49 | ~0.5 mm: clear SV separation in many trackers |
| 20 | ~0.98 | ~1 mm: strong topological handle |
| 50 | ~2.45 | multi‑mm: SV very distinctive |

For transverse quantities, \(L_{xy} \approx (p_T/m_\tau)\,c\tau\) can be used with p_T substituted.

These numbers motivate a key strategy: **do not rely on SV displacement alone below a few GeV**, but use **combined topological + kinematic + coherence** metrics.

### Secondary long‑lived backgrounds: K_S⁰, D⁰, B⁰

Low‑p_T pion triplets often arise from displaced hadron decays or accidental mixing of tracks from multiple vertices. Principal long‑lived sources include:

* **K_S⁰** mean life: **(0.8954 ± 0.0004)×10⁻¹⁰ s**. citeturn41view0  
  This corresponds to \(c\tau \approx 2.68\,\mathrm{cm}\). Such decays often yield π⁺π⁻ with a well‑defined vertex far from PV; pairing with an unrelated π⁻ can fake a 3‑prong vertex if vertexing is weak or tracks are mis‑associated.
* **D⁰** mean life: **(410.3 ± 1.0)×10⁻¹⁵ s**, i.e. \(c\tau \approx 0.123\,\mathrm{mm}\). citeturn17view0  
* **B⁰** mean life: **(1.517 ± 0.004)×10⁻¹² s**, i.e. \(c\tau \approx 0.455\,\mathrm{mm}\). citeturn18view2  

These scales overlap with τ in some momentum regimes (notably B hadrons), which implies that “displacement yes/no” is insufficient. Instead, discriminants should combine:

* **vertex fit probability / χ²**,  
* **pointing** (SV momentum vs PV→SV),  
* **m(π⁺π⁻)** near m(K_S) for K_S vetoes,  
* **isolation / local track density**, and  
* **consistency with τ→a₁→ρπ mass patterns**.

### Qualitative vertex‑fit / pointing behaviour for signal vs background

Modern low‑p_T τ algorithms explicitly use vertex‑quality and flight‑length significance. In a CMS low‑p_T 3‑prong algorithm (track‑based, no jet seed), the candidate choice can require τ‑vertex compatibility and a flight‑length significance >3σ, after per‑pion filtering. citeturn44view0 While your detector is unspecified, the qualitative behaviour is broadly transferable:

* **True τ triplet:** common SV with moderate χ² (3 tracks) and momentum roughly aligned with PV→SV.  
* **Random triplet:** SV fit probability typically low; “pointing” can be uncorrelated and unstable under removing a track.  
* **Triplet involving K_S:** two tracks form a good displaced V0; the third track forces a poor 3‑track vertex unless mis‑measured or if the vertex fitter wrongly merges.  
* **Heavy‑flavour decays:** good SV, but invariant‑mass patterns (3π mass, 2π masses) differ; also additional nearby tracks reduce isolation.

## Low‑p_T detector effects that shape observables

### Multiple scattering: scale formula and example magnitudes

At low momentum, multiple Coulomb scattering dominates angular deflections and thus degrades track parameter and vertex resolutions. The PDG gives (Lynch & Dahl) the standard approximation for the RMS projected scattering angle (central ~98%):

\[
\theta_0 \approx \frac{13.6\,\mathrm{MeV}}{\beta c\,p}\, z \sqrt{\frac{x}{X_0}}\left[1 + 0.038\ln\left(\frac{x z^2}{X_0 \beta^2}\right)\right],
\]
where \(x/X_0\) is the thickness in radiation lengths. citeturn15view2  

For β≈1 and z=1, rough magnitudes (ignoring the log term for intuition) are:

* p = 0.5 GeV:  
  * x/X₀ = 0.01 → θ₀ ≈ 0.0136/0.5·0.1 ≈ 2.7 mrad  
  * x/X₀ = 0.1 → θ₀ ≈ 0.0136/0.5·0.316 ≈ 8.6 mrad  
* p = 1 GeV: θ₀ ≈ 1.4 mrad (0.01 X₀) to 4.3 mrad (0.1 X₀)  
* p = 5 GeV: θ₀ ≈ 0.27 mrad (0.01 X₀) to 0.86 mrad (0.1 X₀)

These values are large enough to broaden ΔR between tracks, smear impact parameters, and inflate vertex χ²—effects that must be represented in simulation and ML training.

### Tracking inefficiency, fakes, split tracks, hit sharing

Low‑p_T tracking suffers from:
* **missed tracks** (curling in B field, insufficient hits, nuclear interactions),  
* **fake (ghost) tracks** from unrelated hits,  
* **split/clone tracks** (one particle reconstructed as multiple tracks), and  
* **hit sharing / cluster merging** in dense environments (especially important near τ 3‑prongs).

A CMS tracking performance paper highlights the difficulty of simultaneously achieving high efficiency and low fake fraction in high‑occupancy environments and reports typical prompt‑track efficiencies for p_T > 0.9 GeV (context‑dependent). citeturn43view2 A dedicated LHCb tracking note defines **ghost tracks** as those with no matching simulated particle and introduces a **ghost fraction** as a key performance metric; it also points out that electrons undergo significantly more multiple scattering and energy loss than other particles—an instructive reminder that multiple scattering is a dominant low‑momentum failure mode. citeturn47view0

### Low‑p_T opening angles and clustering failure of jet‑seeded approaches

In the low‑p_T regime, τ→3π becomes spatially wide. A CMS study explicitly shows:
* maximum pairwise ΔR among the three pions becomes **>0.4** at low visible τ p_T, and  
* the efficiency to contain all three charged pions in one R=0.4 jet drops to **≈0% below ~3 GeV** visible τ p_T. citeturn44view0  

For reconstruction, this motivates candidate generation from **track neighbourhoods around PV/SV** rather than fixed‑cone jets, and it suggests that a model should ingest **pairwise angular distances** and not assume collimation.

### Unspecified detector parameters and recommended “typical” assumptions

Your detector specifics are **unspecified**. For feasibility studies and ML prototyping in an “older‑style” silicon‑tracker experiment, a conservative but realistic assumption set (to be validated against whichever documentation you later adopt) is:

* track p_T threshold: **0.2–0.5 GeV** (note CMS low‑p_T τ algorithm uses **p_T>0.5 GeV** and ≥1 pixel hit as a working point). citeturn44view0  
* impact‑parameter resolution: a constant ⊕ multiple‑scattering term (scaling roughly like 1/p or 1/p_T), consistent with multiple‑scattering dominance at low momentum. citeturn15view2turn43view0  
* per‑layer material: **O(0.5–2)% X₀** (order‑of‑magnitude), with total tracker material at the few % X₀ level depending on design; multiple‑scattering sensitivity should be scanned across this range. citeturn15view2turn43view0  

These should be treated as nuisance parameters in a robustness study rather than fixed truths.

## Signal vs background discriminants and feature sets for ML

### Comparative table: key observables for τ→3π vs common backgrounds

The table below summarises high‑value observables for discriminating a true τ→3π candidate from typical low‑p_T backgrounds when only tracks are available (no π⁰, no ν).

| Observable | τ→3πν signal expectation | Combinatorial low‑p_T π background | Hadronic jet fragments | Secondary decays (K_S⁰, D, B) |
|---|---|---|---|---|
| Total charge | exactly −1 (or +1) | random; often not ±1 | random | depends on decay; can fake ±1 |
| Vertex fit prob / χ² | good 3‑track SV (if τ sufficiently displaced); degrades at very low p due to MS | usually poor; sensitive to removing any track | variable; often poor for random triplets | often good SV (esp. D/B), V0 for K_S |
| Flight length significance (SV–PV)/σ | scales with p: ~0.1–2.5 mm mean for p=2–50 GeV | near zero if PV‑origin | often near PV (UE/pileup), unless heavy flavour | K_S: cm‑scale; D: ~0.1 mm; B: ~0.5 mm (boost dependent) citeturn40view0turn41view0turn17view0turn18view2 |
| Pointing angle (p⃗_vis vs PV→SV) | small (aligned) when SV resolved | random / unstable | often mis‑aligned | for true long‑lived parents: aligned, but masses differ |
| m(3π) | bounded by m_τ; broad around a₁ scale ~1.2 GeV citeturn40view0turn20view0 | broad; can exceed m_τ due to wrong combos/resolution | broad; often larger | broad; may exceed m_τ; depends on parent |
| m(ππ) structure | prominent ρ(770) band with interference; two π⁺π⁻ pairings citeturn12view0turn32view3turn46view0 | no coherent ρ band | may have real ρ but with extra activity | K_S gives narrow π⁺π⁻ at m(K_S); D/B have other resonances |
| Dalitz (s_low,s_high) | structured: ρπ dominance + interference; symmetric under π⁻ exchange citeturn36view0turn45view1turn46view0 | smoother, dominated by acceptance | complex; depends on jet composition | structured but different resonances, different kinematic boundaries |
| Spatial spread (max ΔR among pions) | can exceed 0.4 at low p_T; jet containment fails below ~3 GeV visible p_T citeturn44view0 | wide, depends on track density | clustered in jets but with many tracks | depends on parent; K_S daughters can be collimated after boost |
| Isolation (nearby tracks) | relatively isolated 3‑prong (context dependent) | embedded in diffuse UE | non‑isolated | often non‑isolated in heavy flavour jets |

### Recommended ML input features (track, pair, triplet, context)

| Feature level | Examples (track‑only) | Why it helps |
|---|---|---|
| Per‑track | (p_T, η, φ), charge; impact parameters (d₀,z₀) and their significances; number of hits / χ²/ndf; track quality flags | separates prompt vs displaced; penalises fakes/splits; encodes MS‑driven uncertainty |
| Pairwise (3 choose 2) | ΔR_ij, Δφ_ij, opening angle; m(π_iπ_j) assuming pion mass; DOCA between track helices; pair‑vertex fit χ² | captures ρ and K_S patterns; stabilises candidate building; improves robustness to missing a track |
| Triplet | m(3π); s_low,s_high; triangle geometry (planarity); SV fit prob; flight length significance; pointing angle; sum p_T | directly targets τ dynamics + topology; robust against random triplets |
| Event/context | PV position and uncertainties; pileup/vertex multiplicity; local track density around candidate; isolation cones excluding candidate | controls combinatorial rate; allows domain adaptation across running conditions |

These features are deliberately agnostic to a specific detector, but depend on standard tracking and vertexing outputs. Where unavailable, surrogates can be used (e.g. approximate IP significance from track‑fit covariance).

### Pseudocode snippets for key observables and candidate generation

Invariant masses and Dalitz variables (track inputs are 4‑vectors):

```python
# pions: list of 3 objects with fields (E, px, py, pz, q)
m_pi = 0.13957039  # GeV

def fourvec_from_track(track):
    # If only (p, direction) is available, set E = sqrt(p^2 + m_pi^2)
    p2 = track.px**2 + track.py**2 + track.pz**2
    E  = (p2 + m_pi**2) ** 0.5
    return (E, track.px, track.py, track.pz)

def add(v, w):
    return (v[0]+w[0], v[1]+w[1], v[2]+w[2], v[3]+w[3])

def inv_mass(v):
    m2 = v[0]**2 - (v[1]**2 + v[2]**2 + v[3]**2)
    return max(m2, 0.0) ** 0.5

def m2_pair(v_i, v_j):
    return inv_mass(add(v_i, v_j))**2

# Candidate-level
v = [fourvec_from_track(t) for t in pions]
m3pi = inv_mass(add(add(v[0], v[1]), v[2]))

# Identify the pi+ (q=+1) and two pi- (q=-1)
pi_plus = [i for i,t in enumerate(pions) if t.q > 0][0]
pi_minus = [i for i,t in enumerate(pions) if t.q < 0]

sA = m2_pair(v[pi_plus], v[pi_minus[0]])
sB = m2_pair(v[pi_plus], v[pi_minus[1]])
s_low, s_high = (min(sA, sB), max(sA, sB))
```

Track‑list → 3‑prong candidate generation with hard constraints and pruning:

```python
# tracks: all reconstructed tracks near PV region (or within some z window)
# Preconditions: enforce pion hypothesis for mass; no PID assumed.

# Step 1: filter tracks by quality and pT threshold (tune to detector)
good = [t for t in tracks if t.pt > PT_MIN and abs(t.eta) < ETA_MAX and t.nhits >= NHITS_MIN]

# Step 2: partition by charge
pos = [t for t in good if t.q > 0]
neg = [t for t in good if t.q < 0]

# Step 3: build triplets with total charge -1: (neg, neg, pos)
# Use fast pruning: proximity in z0, small pairwise DOCA, or loose vertex prefit
cands = []
for p in pos:
    # optionally: restrict to pos tracks near candidate region
    for i in range(len(neg)):
        for j in range(i+1, len(neg)):
            t1, t2 = neg[i], neg[j]

            if not passes_fast_pruning(t1, t2, p):
                continue

            sv_fit = fit_secondary_vertex([t1, t2, p])  # returns chi2, ndof, vertex, cov
            if sv_fit.prob < VTXPROB_MIN:
                continue

            cand = build_features(t1, t2, p, sv_fit)
            cands.append(cand)

# Step 4: rank candidates (handcrafted score or NN) and optionally apply NMS-like suppression
best = select_top_candidates(cands, per_event_max=K, overlap_rule="share_tracks")
```

For vertex fit residuals/compatibility, if the full fitter is not available, a proxy is the RMS of distances of closest approach to the fitted SV, normalised by track covariance (Mahalanobis distance).

## Physics‑inspired ML hypotheses, constraints, and evaluation

### Concrete hypotheses (each with benefit, inputs, pitfalls)

| Hypothesis | Expected benefit | Required inputs | Potential pitfalls |
|---|---|---|---|
| Candidate‑graph network over tracks (nodes=tracks, edges=pair compatibility) with triplet pooling | Learns combinatorial suppression using DOCA/ΔR/m(ππ) coherence; handles variable track multiplicity naturally | per‑track kinematics + IP/quality; pairwise DOCA/ΔR + m(ππ); optional PV | Can overfit to occupancy patterns (pileup); requires careful negative sampling (hard negatives) |
| Explicit symmetrised Dalitz features (s_low,s_high,q²) fused with vertex/topology | Encodes hadronic current structure (ρπ dominance + interference) in low‑dimensional, robust features | track 4‑vectors; correct charge assignment; basic vertexing | If detector resolution smears m(ππ) strongly at low p, may down‑weight useful structure; requires resolution‑aware training |
| Multi‑task learning: (a) classify correct triplet, (b) regress m(3π) and m(ππ) consistency residuals, (c) predict SV fit probability | Forces internal representation to respect physics‑shaped observables; better calibration | same as above + vertex fit outputs | If SV fit outputs are biased/miscalibrated in data, the auxiliary task can harm domain transfer |
| Physics‑constrained loss: hard charge constraint + soft endpoint penalty for m(3π)>m_τ | Reduces unphysical candidates and stabilises score tail; improves interpretability | pion 4‑vectors; tau mass fixed | Real data has resolution tails; too strong penalty biases against high‑p uncertainty; enforce softly with uncertainty weight |
| Self‑supervised “track consistency” pretraining (contrastive): positive pairs from same SV, negatives from different vertices | Improves low‑p_T robustness and reduces need for labels; captures detector‑specific failure modes | SV association labels or pseudo‑labels from vertexing | Vertexer errors propagate to SSL; must avoid collapse in dense events |
| Domain adaptation targeted at low‑p_T scatter/efficiency: augment with θ₀‑motivated angular jitter + track drop + split tracks | Bridges sim→data difference where MS and inefficiencies dominate; improves stability vs p_T | material model range (x/X₀); track covariance; dropout rates | Risk of “over‑augmentation” and washing out real resonance structure; needs tuning vs real residuals |
| Uncertainty‑aware scoring (heteroscedastic): predict both score and per‑cand uncertainty, trained with NLL | Better ranking under varying resolution; improves calibration for selection thresholds | per‑track covariances, vertex covariances, or proxy quality metrics | If covariances are mis‑estimated, uncertainty can be mis‑calibrated; must validate with reliability curves |
| Incorporate τ spin sensitivity as a nuisance: train adversarially to be insensitive to polarisation reweighting (TauSpinner) | Reduces dependence on production model (Z vs W vs heavy flavour) | ability to reweight τ polarisation in simulation via TauSpinner; decay products only citeturn38view1turn39view0 | Adversarial training is fragile; may remove genuine discriminating power if polarisation correlates with backgrounds in data |

### Evaluation metrics and validation strategy (low‑p_T focused)

To ensure improvements translate into usable reconstruction:

* **Triplet identification efficiency vs fake‑triplet rate** as a function of visible τ p_T and per‑event track multiplicity (pileup proxy).  
* **ROC / PR curves** in the highly imbalanced regime (true τ triplets rare among all triplets). CMS low‑p_T τ work explicitly reports ROC‑type behaviour for a per‑pion prefilter. citeturn44view0  
* **Mass‑shape fidelity:** compare m(3π) and (s_low,s_high) distributions after selection to expectations; look for sculpting that could bias downstream analyses (especially near m_τ endpoint).  
* **Topological sanity checks:** distributions of SV probability, flight‑length significance, pointing angle for selected candidates; check stability vs detector conditions. citeturn44view0  
* **Calibration:** reliability diagrams / expected calibration error (ECE) for candidate scores; critical if scores will be thresholded.  
* **Robustness scans:** vary material budget (x/X₀) and track inefficiency within plausible ranges and measure performance stability, motivated by PDG multiple scattering scaling. citeturn15view2  
* **Background‑specific control regions:**  
  * K_S‑enriched (identify π⁺π⁻ vertex far from PV) to test V0 rejection,  
  * heavy‑flavour enriched (displaced jets) to test τ vs D/B confusion, using lifetime scales. citeturn41view0turn17view0turn18view2  

### Proposed ML pipeline flow (Mermaid)

```mermaid
flowchart TD
  A[Reconstructed tracks + PV list] --> B[Quality & pT filtering]
  B --> C[Charge partition: (+) and (-)]
  C --> D[Triplet generation: (-,-,+) with fast pruning]
  D --> E[3-track vertex fit + flight/pointing features]
  E --> F[Physics features: m(3π), m(ππ) pairs, s_low/s_high, ΔR geometry]
  F --> G[ML scoring model\n(Graph / Set / Hybrid)]
  G --> H[Candidate selection\n(Top-K + track-sharing suppression)]
  H --> I[Diagnostics\n(mass shapes, SV prob, Lxy, calibration)]
  I --> J[Downstream analysis\n(signal extraction / CP / etc.)]
  I --> K[Domain adaptation loop\n(augmentations + reweighting)]
  K --> G
```

### Suggested visualisations to include in your experimental study

1. **Invariant‑mass histograms**
   * m(3π) for: (i) truth‑matched τ triplets, (ii) selected candidates, (iii) background sidebands.  
   * m(π⁺π⁻) for both pairings and for s_low/s_high projections (expect ρ band). citeturn12view0turn46view0turn36view0  
2. **Dalitz‑like plots**
   * (s_low vs s_high) heatmaps for signal and background; also show symmetrised density (mirror) akin to OPAL’s approach. citeturn45view1turn36view0  
3. **Impact‑parameter and displacement**
   * d₀ significance distributions per track; SV flight‑length significance distributions for selected vs rejected. Use τ and background lifetime scales as anchors. citeturn40view0turn41view0turn17view0turn18view2  
4. **Low‑p_T spatial spread**
   * maximum pairwise ΔR among triplet vs visible p_T (or sum p_T); compare to the jet‑cone scale 0.4 and the low‑p_T containment failure. citeturn44view0  

## Short reference list (primary/official sources cited)

* entity["organization","Particle Data Group","review of particle physics 2024"]: τ properties (mass, lifetime, cτ), a₁ and ρ listings, τ branching fractions, K_S⁰, D⁰, B⁰ mean lives, and multiple scattering review. citeturn40view0turn19view0turn20view0turn12view0turn41view0turn17view0turn18view2turn15view2  
* entity["organization","OPAL Collaboration","LEP tau decays"]: hadronic current/structure functions, Dalitz projections, ν_τ helicity and model comparisons (Kühn–Santamaría vs Isgur). citeturn28view0turn24view0turn45view1  
* entity["organization","ALEPH Collaboration","LEP tau spectral functions"]: use of TAUOLA/Kühn–Santamaría, ρπ interference modelling for 3π modes. citeturn31view0turn32view3  
* entity["organization","CLEO Collaboration","CESR tau decays"]: hadronic substructure: a₁→ρπ dominance, sizeable isoscalar components (f₀(600)π), Dalitz variable definitions, background composition. citeturn36view0  
* RχL/TAUOLA improvement paper (arXiv:1310.1053): hadronic current decomposition; invariant‑mass basis; σ inclusion and Bose symmetry constraints. citeturn46view0  
* entity["organization","CMS Collaboration","LHC experiment"]: low‑p_T 3‑prong τ reconstruction note (track‑based, non‑jet‑seeded), and τ polarisation in Z decays. citeturn44view0turn38view0  
* entity["organization","ATLAS Collaboration","LHC experiment"]: τ polarisation in W→τν. citeturn38view3  
* TauSpinner and τ spin tools: reweighting using decay products only. citeturn38view1turn39view0  
* entity["organization","LHCb Collaboration","LHC experiment"]: tracking performance concepts, ghost fraction and clone tracks in long‑lived tracking (useful for ML metrics and low‑p tracking realism). citeturn47view0