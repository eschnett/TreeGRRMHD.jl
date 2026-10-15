# TreeGRRMHD.jl — Design

TreeGRRMHD.jl solves the equations of general-relativistic **resistive**
magnetohydrodynamics — the Valencia matter equations together with the full
Maxwell equations and an Ohm's law — with a high-resolution shock-capturing
finite-volume scheme on an adaptively refined mesh. It is the fourth
downstream application of [TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl)
and the first physics package that is meant to become a production code.
Its spacetime comes from [TreeGeneralizedHarmonic](../TreeGeneralizedHarmonic)
("TreeGH"); its equation of state and its primitive recovery come from
[EntropyEOS](../EntropyEOS); [TreeHydro](../TreeHydro) is the Newtonian
rehearsal whose scheme, driver and conventions this package inherits. The
end goal is neutron stars and core-collapse supernovae. The feature set is
modelled on GRaM-X (Shankar et al. 2022, arXiv:2210.17509), with the ideal
Ohm's law replaced by a resistive one.

*Status: steps 0 to 3, 5 and 6 of `PLAN.md` are done — the scaffolding;
the pointwise layer: units, the variable layouts, the point metric,
`prim2con` and the 3+1 primitive recovery; the pointwise physics a
right-hand side evaluates; the implicit stage solve; the staggered scheme
on the mesh, with the vacuum electromagnetic wave and the circularly
polarized Alfvén wave in 1D (rung 1a); and the shock tubes, the resistive
tube swept over σ, the current sheet and the atmosphere in 1D (rung 1b) —
as are steps 4a–4c in IMEXRungeKutta. Step 7, 2D and 3D with higher-order
reconstruction, is next. The numbers below about the scheme are measured
where they say so and unmeasured otherwise. (Amended in step 6.)*

Markers, as in the sibling packages: **(decided)** is a decision taken in
review; **(proposed)** is one this document makes and still wants
confirmed; **(open)** is deferred to a measurement or to a later review and
is listed under [Open questions](#open-questions). When the implementation
shows this document wrong or incomplete, amend it and say so — "(amended in
step N)", "(measured in step N)" — rather than diverging silently.

## Goals

- **Resistive GRMHD with a magnetosphere** (decided). E and B are both
  evolved; the current comes from an Ohm's law with a conductivity σ that
  is large inside stars — where the scheme must recover ideal MHD, and the
  resistive term is stiff — and goes to zero outside, where the fields obey
  the electrovacuum Maxwell equations. A neutron star's dipole magnetosphere
  is then not an atmosphere artefact but a solution of the equations.
- **Couple to a dynamical spacetime** from TreeGH, on the same forest, in
  one time integrator, with no subcycling.
- **A magnetized TOV star is the first science target** (decided), after
  the standard flat-space and fixed-background tests: a Γ = 2 polytrope in
  the Cowling approximation, then in a dynamical spacetime, then
  magnetized with a vacuum dipole outside, then with a tabulated EOS.
- **∇·B = 0 to round-off, on every level and across every regrid**
  (decided, by the vector potential — see
  [Where the fields live](#where-the-fields-live)), and **rest mass
  conserved to round-off** across coarse-fine faces.
- **Run on the caller's backend** (CPU threads, CUDA, Metal) through
  KernelAbstractions, in `Float64` for physics; `Float32` only as a device
  smoke test, since EntropyEOS is validated in `Float64` only.

## Scope and non-goals

- **No neutrinos** (decided). No leakage, no M1, no source-term hooks for
  them yet. `Yₑ` is advected passively when the EOS is tabulated.
- **No mesh machinery.** Anything about trees, ghosts, operators,
  reductions or interpolation between levels belongs upstream in TreeAMR.
  Interpolation *within a block* between centerings (the metric from
  vertices to cells, faces and edges; `T_ab` from cells to vertices) is
  application code and lives here, because it is where the physics decides
  the order.
- **No spacetime evolution.** The Einstein equations are TreeGH's; this
  package supplies `T_ab` and reads the metric. Gauge drivers, excision,
  radiative boundaries and Ψ₄ are TreeGH's and are not needed before
  collapse, binaries or supernovae.
- **No con2prim of its own.** EntropyEOS's `con2prim_safe` is used for
  every σ (see [Primitive recovery](#primitive-recovery)). The `Con2Prim`
  package in the sibling checkout is an abandoned prototype and is not a
  dependency.
- **Upstream changes are Erik's** (decided). This document states them
  precisely under [Upstream prerequisites](#upstream-prerequisites); the
  plan sequences the steps that need them to wait. Do not edit the sibling
  checkouts from here.
- **Not yet a production code.** Single node, threads or one device.
  TreeAMR has MPI since its M7 (as an extension) and checkpoints since its
  M9, which moved into the companion package TreeIOHDF5 with TreeAMR 0.2;
  this package uses neither yet. *(Amended in step 0: this item used to
  say "no MPI and no I/O until TreeAMR has them".)*

## Upstream prerequisites

What this package needs from its siblings, and why. *(Amended with
PLAN.md.)* The exact requirements — API, semantics, the tests the sibling
should carry, and the check this package adds when an item lands — are in
`PLAN.md`, "Upstream requirements", under the IDs used here. The summary
below keeps only the reasons. "(req)" blocks the step named; "(wanted)" has
a workaround listed in `PLAN.md`; "(later)" is not needed before rung 7.

**State on 2026-10-14** (amended in step 3, after the reviewer ran the
suite against the new releases), as `[compat]` guarantees it for what has
landed, as `test/prerequisite_tests.jl` reports it for what is waiting,
and, for TreeGH, as its `main` reads. A landed item whose release the
compat floor requires has no probe any more (decided by Erik 2026-10-14;
the rule is in `PLAN.md`, "Upstream requirements"): AMR-1 to AMR-6,
EOS-1, EOS-2 and SM-1. Their names are in the prerequisite test's name
lists; what this package relies on from them and does not yet exercise
through its own code is in `test/upstream_behaviour_tests.jl`.

- TreeAMR **0.2.1**, registered, and the compat floor (`TreeAMR = "0.2.1"`)
  so that what landed is guaranteed: **AMR-1, AMR-2, AMR-3 and AMR-5 have
  landed in TreeAMR 0.2.1**, AMR-4 and AMR-6 in a different form (below).
  Step 5, which needed AMR-1, is done on it (amended in step 5).
- EntropyEOS **1.0.0** on its `main`, not registered (EOS-0 waiting):
  **EOS-1 and EOS-2 have landed in EntropyEOS 1.0.0** (`EntropyEOS =
  "1"`); EOS-3 and EOS-4 have not.
- IMEXRungeKutta **1.4.0**, tagged, not registered: the plan's steps 4a–4c
  are done there, and it is a dependency from step 0.
- SpacetimeMetrics **1.7.0**, registered (`SpacetimeMetrics = "1.7"`; it
  requires Julia 1.11, the floor here): **SM-1 has landed in
  SpacetimeMetrics 1.7.0**, and its workaround is gone.
- TreeGeneralizedHarmonic **0.1.0** on its `main`, on TreeAMR 0.2, not
  registered: GH-0 has landed (the step-8 work is on `main`); GH-1 and
  GH-3 to GH-5 have not; GH-2 is open (see `PLAN.md`). TreeGH is not a
  dependency before step 10, so the suite cannot detect its items.

The unregistered dependencies are named in `Project.toml`'s `[sources]`
with their GitHub remotes; nothing resolves from a sibling checkout.

### TreeAMR

- **AMR-1 (req, step 5) A state vector spanning several field sets** —
  `statevector(sets)`, `scatter!(sets, u)`, `statearray(du, sets, i)`, as
  TreeAMR's `CODE.md` already specifies ("implemented with the first
  application that needs it"), plus segment views. The evolved state is a
  cell-centered matter set, three edge-centered sets for `A`, a
  vertex-centered set for the gauge scalar, and — when coupled — TreeGH's
  vertex-centered set; one integrator advances all of them. *(Used from
  step 5:)* the state vector is over `evolved_sets(f) = (U, A₁, A₂, A₃,
  Φ̃)`, and `test/scheme_tests.jl` makes the plan's downstream check — the
  round trip over exactly that tuple in D = 1, 2, 3, on a two-level forest,
  each segment bit for bit the set's own state vector.
- **AMR-2 (wanted, step 6) Per-set boundary hooks in `regrid!`.** It takes
  one hook for every set, and an `AllVariables` hook is length-checked
  against each set's `nvars`, so one hook cannot serve this package's sets.
- **AMR-3 (wanted, step 8) `adapt_to_initial_data!` over several sets**;
  today it adapts one.
- **AMR-4 (strongly wanted, step 10) A limited conservative prolongation**
  for cell-centered data. "Limited" in the reconstruction sense: a nonlinear
  prolongation — slope-limited (minmod/MC) and scaled toward the coarse
  average — that is still exactly conservative (the subcell averages
  reproduce the coarse cell's average), second-order where the data are
  smooth, and never creates new extrema (fine values stay within the range
  of the neighbouring coarse averages). Reason: at a neutron star's surface
  ρ drops by about ten orders of magnitude, and the linear width-3 operator
  overshoots there into cells below the atmosphere; TreeHydro measured 4096
  floored cells at width 3 against none at width 1 on a shock crossing a
  coarse-fine face, at 0.47% in L1. Being nonlinear, it is not a
  fixed-weight tensor product, so it needs a new operator family or a hook
  covering both ghost filling and the regrid transfer. Workaround: width 1
  near surfaces. *(Landed in another form, TreeAMR 0.2.1, commit 902042a,
  amended 2026-10-14:)* not a choice of limiter but a property,
  `Operators(; prolongation=3, restriction=2, family=Conservative,
  bounded=true)` — the order-3 conservative prolongation with an MC slope
  and a scaling that keeps every child within the range of its parent and
  the parent's face neighbours, exactly conservative, in the ghost fill and
  the regrid transfer alike. `test/upstream_behaviour_tests.jl` checks
  that a surface refined everywhere stays positive and within range and
  keeps its total, where the unbounded operator overshoots (until step 8's
  refinement tests take it over). See [Refinement](#refinement)
  for what upstream's design assumes of this package.
- **AMR-5 (wanted, step 14) `workgroupsize` on `map_blocks!`** (also on
  TreeGH's list).
- **AMR-6 (later) A device-capable boundary hook that reads the interior**,
  for outflow conditions. Workaround: Dirichlet from the initial or
  analytic exterior, which is what the TOV milestones need anyway.
  *(Landed in another form, TreeAMR 0.2.1, commit a983419, amended
  2026-10-14:)* not a hook but a face kind, `Forest(roots; …, outflow =
  ((lo, hi), …))`, whose ghosts the schedule fills with the nearest owned
  value (zero-gradient, first order at the face), on every backend and
  across ranks. A nonlinear condition, such as forbidding inflow, stays
  with the application.
- *(later)* The use of MPI (M7) and of checkpoints (M9, now TreeIOHDF5's)
  and point interpolation (M11), all three of which TreeAMR has; none is
  an upstream item any more. *(Amended in step 0.)*

This is a **physics package** in TreeAMR's sense (its `CODE.md`,
"Companion interface", 2026-10-08): it uses TreeAMR's exports only, and
bounds TreeAMR by a minor series, `TreeAMR = "0.2"` (since 2026-10-14
`"0.2.1"`, the release the landed items are in). The prerequisite
tests check the names it calls against the exports, and `src/device.jl`
uploads its own metadata with `KernelAbstractions.allocate` rather than
reaching for TreeAMR's internal `todevice` (proposed in step 0).

### TreeGeneralizedHarmonic

- **GH-0 (req, step 10) Step 8 on `main`.** The unmerged step-8 branches
  widen `GHProblem`, `gh_rhs_kernel!` and `evolve!`; the items below target
  `main` only.
- **GH-1 (req, step 10) Public access to the vertex metric**: the packing
  helpers under public names, the layout of `U.work` as a documented
  contract, and one exported pointwise function returning `h`, `Π` and `∂h`
  at a vertex. The Cowling runs read the metric from a vertex set in
  TreeGH's layout.
- **GH-2 (req, step 10) Registration, and compat satisfiable from the
  registry.** A dependency's `[sources]` is ignored downstream, so every
  TreeAMR feature TreeGH uses must be in a released TreeAMR. The Julia
  floor is discussed under [Julia version](#julia-version).
- **GH-3 (req, step 11) A stress-energy input.** A vertex-centered `Tsrc`
  field set (10 components, `G = 0`) read at the owned point in
  `gh_rhs_kernel!`, following the `Hsrc`/`gauge_at` pattern, switched by a
  `Val` in `GHProblem`, entering as `∂ₜΠ_ab += 16π α√γ (T_ab − ½ g_ab T)`
  (derived from `notes/formulation.md` §1–2: `R^H_ab = 8π(T_ab − ½g_abT)`
  turns `S0` into `S0 − 16πT̄_ab`, and the source enters as `−α√γ S0`). The
  ADM monitor gains the matter terms, `ℋ − 16πρ_ADM` and `ℳ_i − 8π S_i`.
- **GH-4 (req, step 11) A right-hand side callable inside an external
  integrator**: a kernel-only entry point for a caller that has already
  scattered and ghost-filled, integrator-free limiters, all accepting
  segment views of a multi-set state vector, and `evolve!` as a composition
  of exported functions. The coupled driver lives here and calls them.
- **GH-5 (wanted, step 10) A ghost width of at least `q/2 + 1`**, so that
  the 4th-order vertex→cell metric interpolation reaches every stored cell
  of a `G = 3` matter set (see [Metric at cells, faces and
  edges](#metric-at-cells-faces-and-edges)).
- **GH-6 (later)** A damped-harmonic gauge driver, radiative or
  constraint-preserving outer boundaries, excision or a generalized
  interior for collapse, Ψ₄ extraction.

The CFL factor of TreeGH under an SSP3 explicit tableau (`cfl = 1/4` was
measured with RK4, whose stability region on the imaginary axis is wider)
is *not* an upstream item: this package owns the integrator and measures it
in step 11. *(Amended with PLAN.md.)*

### EntropyEOS

- **EOS-0 (req, step 0; workaround) Registration** in General. Until then
  `Project.toml`'s `[sources]` adds it by URL, for CI, a clean checkout
  and a developer alike. *(Amended in step 0: "CI adds it by URL".)*
- **EOS-1 (req, step 5) An abstract EOS interface** (decided): an
  `AbstractEOS` supertype, the solver and policy retyped to it, and the
  four bounds read through accessors instead of fields, so that `evaluate`,
  `con2prim_safe`, the atmosphere policy and the device view work unchanged
  for any EOS that implements `evaluate`, `srange`, `srange_extended`, the
  bounds and `κ`. *Landed in EntropyEOS 1.0.0*, with one difference from
  the request: `κ` stays unexported and is documented as `EntropyEOS.κ`,
  so this package calls it qualified (amended in step 0).
- **EOS-2 (req, step 5) Analytic EOSes behind that interface**: a Γ-law
  and a hybrid (piecewise polytrope plus thermal Γ), as potentials
  `U(ρ, s)`. **A polytrope is the Γ-law at fixed entropy**,
  `U = K(s) ρ^{Γ−1}/(Γ−1)` with `K(s) = K_ref exp((Γ−1)(s − s_ref))`; a
  purely barotropic EOS (`U_s ≡ 0`) cannot be offered, because EntropyEOS's
  `(s, w)` Newton is singular there. *(Amended with PLAN.md, which also
  moved this item from rung 4 to step 5: the synthetic table cannot host
  the flat-space tests — see [Thermodynamics](#thermodynamics).)*
  *Landed in EntropyEOS 1.0.0* as `IdealGasEOS`, `HybridEOS` and
  `polytropic_entropy`, with the requested keyword constructors; both are
  `isbits`, `κ = 1`, and the TOV polytrope `K = 100, Γ = 2` is
  `IdealGasEOS` at `polytropic_entropy(eos, 100)` (checked in step 0).
- **EOS-3 (wanted, step 12) A public fixed-entropy inversion** — the "at
  fixed `s`, solve for `w`" step that `con2prim` already contains — for the
  entropy fallback, and a public entropy-from-energy solve for initial
  data.
- **EOS-4 (later, step 13)** A cold β-equilibrium slice `Yₑ,β(ρ)` and
  `s_cold(ρ)` for tabulated-EOS stars.

### SpacetimeMetrics

- **SM-1 (wanted, step 9; workaround) An allocation-free extrinsic
  curvature** *(added in step 2)*: `ExtrinsicCurvature(m, p)` allocates
  432 bytes per call (measured in step 2, Julia 1.11 and 1.13), which a
  device kernel cannot, and it differentiates the metric again where the
  `Analytic` provider already holds one `dmetric` result; the request adds
  `ExtrinsicCurvature(g, dg)` on that result. *(Landed in SpacetimeMetrics
  1.7.0, 2026-10-14:)* `src/metric.jl` calls `ExtrinsicCurvature(g, dg)`,
  and its local `extrinsic_curvature` is gone; `test/pointwise_tests.jl`
  keeps that formula as the check on upstream's.

## The equations

### Conventions

Signature `(−,+,+,+)`, `G = c = 1`, lengths and times in `M☉` (TreeGH's
unit is `M`; with `M = M☉` they agree). Electromagnetic fields in
Heaviside–Lorentz form with the `√(4π)` absorbed, so the magnetic pressure
is `B²/2`. Latin indices are spatial; `γ_ij` is the 3-metric, `α` the
lapse, `β^i` the shift, `K_ij` the extrinsic curvature, `n^μ` the unit
normal. `[ijk]` is the Levi-Civita *symbol* and `ε_ijk = √γ [ijk]` the
tensor. `E^i`, `B^i` are the fields measured by the normal observer, `v^i`
the fluid's velocity relative to that observer, `W = (1 − v²)^{−1/2}`.
Densitized fields carry fraktur: `𝔈^i = √γ E^i`, `𝔅^i = √γ B^i`.

### Matter: total energy and momentum

With `D = ρ* W` the baryon density, `S_j` the **total** (fluid plus
electromagnetic) momentum density and `𝓔` the total energy density seen by
the normal observer, and `τ = 𝓔 − D`, the evolved variables are the
densitized `√γ D`, `√γ S_j`, `√γ τ`, … under the time derivatives. (Where
this document lists "`D`, `S_i`, `τ`" as stored variables it means these
densitized ones.)

```
∂ₜ(√γ D)   + ∂ᵢ[√γ D (α vⁱ − βⁱ)]         = 0
∂ₜ(√γ Sⱼ)  + ∂ᵢ[√γ (α Sⁱⱼ − βⁱ Sⱼ)]       = √γ [½ α S^{ik} ∂ⱼγ_ik + S_i ∂ⱼβⁱ − 𝓔 ∂ⱼα]
∂ₜ(√γ τ)   + ∂ᵢ[√γ (α (Sⁱ − D vⁱ) − βⁱ τ)] = √γ [α S^{ij} K_ij − Sʲ ∂ⱼα]
∂ₜ(√γ D Yₑ) + ∂ᵢ[√γ D Yₑ (α vⁱ − βⁱ)]      = 0
∂ₜ(√γ D s)  + ∂ᵢ[√γ D s (α vⁱ − βⁱ)]       = 0          (entropy backup)
```

with the fluid and electromagnetic parts

```
𝓔    = ρ h W² − p               + ½ (E² + B²)
S_i  = ρ h W² v_i               + ε_ijk Eʲ Bᵏ
S^ij = ρ h W² vⁱ vʲ + p γ^ij    − Eⁱ Eʲ − Bⁱ Bʲ + ½ γ^ij (E² + B²)
```

(the Valencia form, e.g. Baumgarte & Shapiro; the electromagnetic parts as
in Palenzuela 2013 and Dionysopoulou et al. 2013). The momentum source is
equivalently `½ √−g T^{ab} ∂ⱼ g_ab`, which is the cheaper form when the
metric derivatives come from TreeGH (proposed; the two are checked against
each other in a test). *(Step 2:)* both are `geometric_source` in
`src/sources.jl`, and on boosted Kerr-Schild with fields up to 100 times
the fluid's energy they agree to 5.4e−16 of the magnitude of their terms;
the covariant form also agrees with `√−g T^{μν}(∂_μ g_νj − Γ^δ_νμ g_δj)`
from SpacetimeMetrics' Christoffels to 1.2e−15, and the energy source with
Banyuls et al. (1997)'s `√−g (T^{μt} ∂_μα − α T^{μν} Γᵗ_μν)` to 5.6e−16
(measured in step 2). **The covariant form is the default** (proposed in
step 2): every provider produces `∂ⱼ g_ab` natively, and the Valencia form
must first derive `∂ⱼα` and `∂ⱼβⁱ` from it. Measured in step 2 (Julia
1.11, `Float64`, one call, Apple M3 Pro): 36 ns against 50 ns for the
geometric source, 41 ns against 74 ns for a cell's whole explicit source.
The entropy equation is not a conservation law
through shocks; `D s` is advected only as a fallback for the primitive
recovery and is reset from the recovered `s` wherever the energy inversion
succeeds (proposed; whether it earns its keep is (open)).

### Maxwell

In the 3+1 form of Komissarov (2004), densitized:

```
Gauss:    ∂ᵢ 𝔈ⁱ = √γ q                      (defines q — see below)
Ampère:   ∂ₜ 𝔈ⁱ + ∂ⱼ[ −[ijk] α B_k + βⁱ 𝔈ʲ − βʲ 𝔈ⁱ ] = −√γ (α jⁱ − q βⁱ)
Faraday:  ∂ₜ A_i = −(α E_i + ε_ijk βʲ Bᵏ) − ∂ᵢ(α Φ − βʲ A_j)
          𝔅ⁱ = [ijk] ∂ⱼ A_k  (+ 𝔅̄ⁱ, see below)
Gauge:    ∂ₜ(√γ Φ) + ∂ⱼ[α √γ Aʲ − βʲ √γ Φ] = −ξ α √γ Φ
```

The flux of `𝔈ⁱ` in direction `j` is `−[ijk] α B_k + βⁱ 𝔈ʲ − βʲ 𝔈ⁱ`
(Komissarov's `H = αB − β × E`), which in flat space is `∂ₜE = ∇ × B − j`.
The sign of the shift term in Faraday's law follows from Komissarov's
`E_aux = αD + β × B`, and agrees with IllinoisGRMHD's ideal
`∂ₜA_i = ε_ijk vʲ Bᵏ − ∂ᵢ(αΦ − βʲA_j)` — whose `vⁱ = uⁱ/u⁰ = α v_Eⁱ − βⁱ`
is the coordinate velocity — once `E = −v_E × B` (checked while writing
this; a test pins both signs). *(Measured in step 2:)* the EMF agrees with
IllinoisGRMHD's form to 1.4e−15 of its terms on random states in boosted
Kerr-Schild; the flux of `𝔈` is Komissarov's `−[kdl] H_l` exactly; and in
flat space the fluxes, the EMF, the gauge potential and the gauge flux
reproduce an exact electromagnetic plane wave in the Lorenz gauge —
`∂ₜE = ∇ × B`, `∂ₜA = −E − ∇Φ`, `∂ₜΦ = −∇·A` — to 6.3e−16, with
Ampère's source equal to `−j` to 2.2e−16 (`test/pointwise_tests.jl`). The gauge is generalized Lorenz with damping `ξ`
(Farris et al. 2012), which removes the zero-speed gauge mode that
otherwise produces spurious `B` at refinement boundaries. `ξ` is (open).

**The charge density is derived, not evolved** (decided): `q := ∇·E`,
computed from the ghost-filled cell-centered `𝔈` in every right-hand-side
evaluation. Gauss's law then holds identically and charge conservation is
implied; there is no evolved `q` and no cleaning scalar for `E`. The known
cost is that any numerical divergence of `E` — at coarse-fine interfaces,
at the stellar surface — appears as charge and drives `q v` currents; the
charge is monitored as a diagnostic.

**A uniform background field** `𝔅̄ⁱ = √γ B̄ⁱ` with constant `B̄ⁱ` is added
to `curl A` (proposed). It is what lets a vector potential represent a net
flux through a periodic box, and a field along a direction the mesh does
not resolve: a 1D shock tube's constant `Bˣ` is `B̄ˣ`, since no
translation-invariant `A` produces it. It is zero for stars. *(Amended in
step 5:)* **the densitized `𝔅̄ⁱ` is the constant** (proposed in step 5), not
`B̄ⁱ`: with `√γ` varying, `∂ᵢ(√γ B̄ⁱ) = B̄ⁱ∂ᵢ√γ ≠ 0`, a divergence the curl
cannot cancel, while a constant `𝔅̄ⁱ` is divergence-free on any metric. In
flat space the two are the same. It is `MHDProblem`'s `𝔅̄` and a case's
(`src/curl.jl`); the Alfvén wave's guide field is the first user.

### Ohm's law and the conductivity

The current seen by the normal observer, for an isotropic conductor in the
fluid frame (Palenzuela et al. 2009; Palenzuela 2013):

```
jⁱ = q vⁱ + σ W [Eⁱ + ε^{ijk} v_j B_k − (v_k Eᵏ) vⁱ]
```

`σ` is an `isbits` model type (proposed):

| model | σ | use |
|---|---|---|
| `Vacuum()` | 0 | electrovacuum tests |
| `Constant(σ₀)` | σ₀ | resistive shock tubes, current sheet |
| `DensityProfile(σ₀, ρ_atm, …)` | σ₀ inside, → 0 toward `ρ_atm` | stars with a magnetosphere (Palenzuela 2013) |
| `IdealConductor()` | ∞ | the exact ideal limit: `E = −v × B`, for Balsara 1–5 and comparison with GRaM-X |
| `MagnetizationCutoff(model, μ_max)` | `model`'s, → 0 where `B²/D → μ_max` | any finite σ, where the plasma is magnetically dominated (step 3) |

The `q v` part is explicit. The `σ` part is **stiff** inside a star
(`σ Δt ≫ 1`), and is the implicit part of the time integrator.

*(Proposed in step 2:)* the models are in `src/ohm.jl`, and
`conductivity(model, prim)` takes the whole primitive state, so that a
later model may read `s` or `Yₑ`. `DensityProfile(σ₀, ρ_atm, cutoff = 2)`
is `σ₀ max(0, 1 − cutoff·ρ_atm/ρ)`: continuous, `σ₀` deep inside matter
and zero at and below `cutoff·ρ_atm`, so the atmosphere carries no current;
rung 6 settles its shape against Palenzuela (2013). For `IdealConductor`
the conduction current is not a function of the state — it is whatever
holds `E = −v × B`, which the stage solve imposes — so `conductivity` is
`Inf` and `ohm_current` throws an `ArgumentError` rather than return
`∞ · 0`. Where `σ = 0` the conduction term is not formed (a mask
branches). Measured in step 2: at `E = −v × B` the current is 2.3e−16 of
`σ W |B|`, and in flat space the bracket is the fluid-frame field
`F^{ab} u_b` with the charge the normal observer sees, to 4.4e−16.
*(Proposed in step 3:)* `conductivity_slope(model, prim)` is `∂σ/∂ρ`, for
the stage solve's Newton Jacobian; a later model that reads `s` or `Yₑ`
adds their derivatives. Extension slots, not planned: a mean-field dynamo
term, and force-free currents in the magnetosphere.

**σ → 0 where the plasma is magnetically dominated** (decided by Erik
2026-10-14, after step 3's review), and not only where `ρ → ρ_atm`. At
`b²/ρ ≳ 10³` the Alfvén speed is within `ρh/b²` of the speed of light, the
plasma's inertia is negligible and it is force-free. There one implicit
stage hands the fluid more energy than it has: the share of stages the
iteration cannot solve rises with `X`, the field energy the stage hands over
per unit of the fluid's energy, from 1.7% (and 0.1% falling back) at
`X < 1` to 46% (0.8%) at `X > 1000` (review's measurement); the stage
equations there have several admissible roots; and a scalar Ohm's law in the
fluid frame has no physical basis. Resistive neutron-star codes take σ → 0
outside the star for this reason (Palenzuela 2013); this makes the same
choice by magnetization as well as by density. A globally convergent stage
solve for that regime, a bracketed search over the fluid's rapidity that
leaves no fallback at `f ≤ 0.03` and 0.02% at `f = 0.1`, is kept on the
branch `claude/step-3-bracketed-stage-solve` and is not in the code.

*As built* (proposed in step 3): `MagnetizationCutoff(model, μ_max)` wraps
any finite-σ model and multiplies its σ by a taper `g(μ)` of the
magnetization `μ = magnetization(m, B, D) = γ_ij Bⁱ Bʲ / D` — the normal
observer's field over the rest-mass density, which bounds the fluid
frame's `μ/W ≤ b²/ρ ≤ W μ`. *(Proposed in step 6:)* `μ = +∞` where
`D ≤ 0`, which a stage predictor next to a near vacuum can carry: no rest
mass, no conductor. `B²/D` handed the taper's logarithm a negative number
there, and the magnetized atmosphere tube threw a `DomainError`.
- **Constant within a stage.** `B` is explicit in the stage and the stage
  moves no rest mass, so `μ` is computed once from the stage's `B` and `D`:
  the cutoff adds no nonlinearity to the stage solve and nothing to its
  Newton Jacobian (`conductivity_slope` is scaled by `g`). In Ohm's current
  and Ampère's stiff source, `D = ρW` of the primitive state.
- **Smooth and monotone.** `g = 1 − t²(3 − 2t)`, `t = log₁₀(10 μ/μ_max)`
  clamped to `[0, 1]`: σ is the model's for `μ ≤ μ_max/10`, zero for
  `μ ≥ μ_max`, and `C¹` and monotone between. Where `g = 0` the inner
  model is not read, and the stage is the vacuum's (`E = E★`,
  `OHM_SIGMA_ZERO`).
- **Composable and `isbits`.** `conductivity(model, prim, μ)` is the
  three-argument form every caller uses; it is `conductivity(model, prim)`
  for the plain models, and the wrapper has no two-argument form (it
  throws, saying why).
- **`IdealConductor` is not tapered** (the constructor refuses it): its σ is
  infinite, `∞ · g` is not a conductivity, and the ideal recovery is well
  posed at any magnetization — it carries the field's inertia in its own
  model, which is why it is the iteration's seed in the stiff limit.
- **`μ_max` is required and has no default** (decided by Erik
  2026-10-14). It trades the stage's robustness against how much of the
  magnetically dominated plasma stays resistive, which is a property of the
  case, not of the scheme: each case and driver states its own value, no
  code path wraps a model implicitly, and `MagnetizationCutoff(model)`
  throws an `ArgumentError` saying so. The tests pass it explicitly.
- **The measured safe value, `μ_max = 10`** (measured in step 3): the
  largest of `{5, 10, 20, 30, 50, 70, 100, 300, 1000}` at which the
  iteration, with no other help, has no fallback at any fixed `f ≤ 0.1`
  over the whole grid (`σΔt ∈ 10⁻³…10⁸`, `b²/ρ ∈ 10⁻⁴…10⁴`,
  `W ∈ {1, 2, 10}`, eight problems a point) at every Γ across sixteen seeds
  of the test's generator — 41,472 problems per Γ and `f`. At `μ_max = 10`
  none falls back up to `f = 0.3`. The taper begins at `μ = μ_max/10 = 1`;
  since `b²/ρ ≤ W μ`, σ is zero by `b²/ρ ≈ 10` for a fluid at rest and
  `≈ 100` at `W = 10`.
- **A physically motivated larger value trades a small fallback rate**
  (decided by Erik 2026-10-14). Ideal-GRMHD codes cap the magnetization at
  `b²/ρ ~ 50–100` (adding mass where this switches off the conduction
  current instead), and a case that wants the plasma resistive up to there
  may take `μ_max` of that order. Measured, summed over Γ ∈ {4/3, 5/3, 2}
  at sixteen seeds (124,416 problems per `f`):

  | `μ_max` | f = 10⁻³ | 10⁻² | 0.03 | 0.1 | 0.3 | 1 |
  |---|---|---|---|---|---|---|
  | 10 | 0 | 0 | 0 | 0 | 0 | 6 |
  | 20 | 0 | 1 | 0 | 4 | 6 | 35 |
  | 30 | 1 | 0 | 2 | 3 | 17 | 44 |
  | 100 | 0 | 0 | 2 | 1 | 24 | 208 |

  — at `f ≤ 0.1` a rate of order `10⁻⁵`, at `f = 1` up to 0.2%, every one
  caught by the lagged-velocity fallback and flagged `OHM_FALLBACK`.

Fallbacks of 41,472 problems per cell (16 seeds × 2592), at a fixed
`f = |E★ + v × B|/|B|`, the predictor's departure from the ideal field:

| | f = 10⁻³ | 10⁻² | 0.03 | 0.1 | 0.3 | 1 | uniform |
|---|---|---|---|---|---|---|---|
| Γ = 4/3, no cutoff | 2 | 1 | 26 | 76 | 236 | 684 | 361 |
| Γ = 4/3, `μ_max = 10` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Γ = 5/3, no cutoff | 2 | 118 | 161 | 434 | 844 | 1600 | 1055 |
| Γ = 5/3, `μ_max = 10` | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| Γ = 2, no cutoff | 58 | 266 | 406 | 754 | 1199 | 2034 | 1434 |
| Γ = 2, `μ_max = 10` | 0 | 0 | 0 | 0 | 0 | 3 | 1 |

The problems are built with the tapered σ, so each known solution is the
stage's; the residual at every converged point stays at most 1.8e−12 of
`τ + D`.

### Stress-energy for the spacetime

`T^{ab} = 𝓔 n^a n^b + S^a n^b + n^a S^b + S^{ab}` from the same total
`𝓔`, `S_i`, `S^{ij}` as above; lowered with `g_ab`, it is what TreeGH reads.
*(Step 2:)* `stress_energy` and `T4_lower` in `src/stress_energy.jl`.
`nᵃnᵇT_ab = 𝓔` and `−nᵇT_ib = S_i` hold to 8.0e−16 of `max |T_ab|`; the
fluid part is `ρh uᵃuᵇ + p gᵃᵇ` to 5.5e−16, the field part is trace-free to
1.2e−15, and in flat space it is `F^{ac}F^b_c − ¼ηᵃᵇF²` with
`F_{i0} = E_i`, `F_ij = [ijk]Bᵏ` to 3.3e−16, which pins the sign of the
Poynting term (measured in step 2).

## Thermodynamics

EntropyEOS everywhere (decided): its run-time variables are `(ρ*, s, Yₑ)`,
with `ρ* = κ ρ` a rescaled baryon density chosen by the table build, `s`
the entropy per baryon, and a single C² potential `U(ρ*, s, Yₑ)` from which
`p`, `h`, `c_s²` and the consistent temperature follow. Consequences:

- **The primitive thermodynamic variable is `s`**, not `ε` or `T`. `D`
  is `√γ ρ* W`, so a checkpoint is tied to the table's `κ`.
- **A units adapter** converts between geometric units (`G = c = M☉ = 1`)
  and EntropyEOS's run units (cgs-like: densities in g/cm³, `U = ε/c²`),
  including `κ`. It is the only place a unit conversion is written.
  *(Proposed in step 1:)* it is `GeometricEOS(eos; density)` in
  `src/units.jl`, an `AbstractEOS` around an EOS in its own units whose
  densities are the scheme's: `evaluate` scales `ρ★` by the density unit on
  the way in and `U_ρ`, `U_ρs` (once), `U_ρρ` (twice) and `p` (inversely)
  on the way out, the entropy windows are the inner EOS's at the scaled
  density, and the density bounds shift by its logarithm. The recovery's
  equations are homogeneous of degree one in the densities (`D`, `τ`, `S`,
  `B²`, `ρhW²` and `p` scale together), so `con2prim_safe`, `prim2con` and
  the atmosphere policy then run in geometric units unchanged, with no
  conversion at any call site. The density unit is `c⁶/(G (GM☉)²)` =
  6.1758e17 g/cm³ (measured in step 1, from the IAU 2015 nominal `GM☉` and
  CODATA 2018 `G`). `κ` enters only where a physical rest-mass density
  `ρ★/κ` is wanted (initial data in `ρ`, a baryon-mass diagnostic), through
  `EOSUnits`; the scheme's `D` is `√γ ρ★ W` throughout. An analytic EOS
  built in geometric units — the TOV polytrope, the shock tubes — needs no
  adapter at all. Measured in step 1: the adapter reproduces an ideal gas
  written directly in geometric units to 9.1e-15 in every `EOSPoint` field,
  and a table recovered through it agrees with the same table recovered in
  g/cm³ to 3.0e-15 in `(ρ, w)` and 8.4e-13 in `s`, conditioned.
- **Warm starts** (`u_guess`, the previous `log₁₀ T`) are carried per cell
  in the primitive set and passed to every EOS call, including those at
  reconstructed face states. *(Amended in step 5; proposed in step 5:)*
  every **recovery** is seeded from the state itself — `s` from the advected
  `(√γ D s)/(√γ D)`, a cold rapidity and EOS warm start (`state_guesses` in
  `src/evolution.jl`) — and not from `P`. A warm start carried in `P` makes
  each recovery depend, at the solver's tolerance, on whatever was evaluated
  before: the right-hand side would not be a function of `(u, t)` bit for
  bit (the plan's purity claim), and a restart from a checkpoint, which
  holds no `P`, would not reproduce the run. The face states, reconstructed
  from `P` within one evaluation, still take their cell's EOS warm start.
  The cost is a cold rapidity: a recovery over the stored cells is 0.86 µs a
  cell on the Alfvén wave (measured in step 5, below). A table's `log₁₀ T`
  is the warm start this loses; if step 13 finds it matters, the state can
  carry it the way it carries `D s`.
- **Analytic EOSes come from EntropyEOS** (decided; EOS-1, EOS-2). They are
  needed from the first evolution (step 5) on, not only from rung 4: the
  synthetic ideal-gas table is `Γ = 5/3` only, in cgs, with `T ≤ 50 MeV`,
  so it cannot host the flat-space tests (Balsara 1 is `Γ = 2`, and
  Balsara 3 has `ε = 1500`). *(Amended with PLAN.md.)* EOS-2 landed
  before step 1, so the synthetic table serves only as the table the units
  adapter is checked on (amended in step 1).
- **A polytrope is a Γ-law at fixed entropy**, not a barotropic EOS:
  EntropyEOS's `(s, w)` Newton needs `U_s ≠ 0`. The TOV star's
  `p = Kρ^Γ` is the Γ-law evaluated at the `s` with `K(s) = K`.
  *(Amended with PLAN.md.)*

## Primitive recovery

Because `E` is evolved, the electromagnetic energy `½(E² + B²)` and
momentum `ε_ijk Eʲ Bᵏ` do not depend on the fluid velocity. Primitive
recovery is therefore **pure hydrodynamics**: subtract them from `τ` and
`S_i`, project with `γ^{ij}` to EntropyEOS's scalar form
`(D, τ, D_Y, S_∥, S_⊥, B²)` with `S_∥ = 0`, `S_⊥ = |S|`, `B² = 0`, and
call `con2prim_safe` (decided). *(Amended in step 1: this said `S_⊥ = 0`.
With `B² = 0` the two splits give the same solve, but EntropyEOS requires
`S_⊥ ≥ 0` while `S_∥` is signed, and its own `prim2con` calls all of the
momentum perpendicular when there is no field to project against; the
repaired states it packages then carry the speed in `v_⊥`.)* The velocity
is rebuilt with its direction from `Sⁱ = γ^{ij} S_j` and its magnitude from
the returned rapidity, `vⁱ = tanh w · Sⁱ/|S|` (proposed in step 1). That
makes the primitive state self-consistent to rounding, and right on a
repaired state; the literal `vⁱ = Sⁱ/(ρhW²)` carries the momentum residual
into `atanh |v|` amplified by `W²`, which left 7e-11 in the conserved `D`
of the round trip against 4e-14 (measured in step 1). This is
`con2prim_hydro` in `src/con2prim.jl`; `con2prim_ideal` is the `B² ≠ 0`
call on the totals, splitting `S` along and across `B` in the metric, used
only as the stage solve's seed. It never fails: atmosphere, ceilings, rapidity
cap, the entropy window and non-finite input are its policy, every repair
is reported in its flags, and every repaired state is exactly invertible.

How exactly it inverts is conditioned, not uniform (amended in step 0).
EntropyEOS states its round-trip accuracy with the errors divided by
`κ_ρ = hW²` (for `ρ` and the rapidity `w`) and by
`κ_s = (1 + τ/(ρW²U))·max(1, U/(s U_s))` (for `s`): the entropy is
ill-determined where the thermal energy is a small part of `τ`, and the
density where `hW² ≫ 1`. Its suite asserts the conditioned errors below
3e−11 and 1e−10; a raw relative 1e−12 is not attainable — step 0 measured
8.9e−10 in `s` at `ρ = 1e−8`, `s = 5.2`, `W = cosh 2` on the Γ = 2 gas,
while over step 0's prerequisite test's 24 states (retired with EOS-2's
probe, 2026-10-14) the conditioned errors were at most 2.1e−14 in `ρ` and
`w` and 1.7e−13 in `s`.

Through the 3+1 layer (measured in step 1, `test/recovery_tests.jl`): on
1000 random states per `Γ ∈ {4/3, 5/3, 2}` — `ρ` across the box, `s`
across the window, `W ≤ 10`, metrics from a boosted Kerr-Schild hole with
`a ≤ 0.9` at random events between `r = 3M` and `20M`, and `E`, `B`
carrying 1e−4 to 100 times the fluid's `τ` — `con2prim_hydro ∘ prim2con`
has conditioned worst errors 4.5e−13 in `(ρ, w, v)` and 8.4e−12 in `s`
(raw `s` up to 5.7e−7 at `Γ = 2`, where the thermal energy is a tiny part
of `τ` and `κ_s` is large). In the conserved variables, with `S_i` and `τ` measured against the energy density
`𝓔 = τ + D` they are components of (proposed in step 1), the round trip at
EntropyEOS's default tolerance is 9.8e−13 relative (`τ` against itself:
9.9e−13). That is the solver's stopping criterion — `con2prim` stops when
its energy residual is below 1e−12 of `τ`, often at the cold seed with no
Newton step — and not this layer's rounding, and it leaves the 1e−12 of
the step-1 acceptance no margin. So the test runs the conserved round trip
with `Con2PrimOptions(tol = 1e−14)`, the tightest power of ten at which
every draw still converges by Newton (at 1e−15 some fall back to the
nested solve, and the round trip is no better): the worst error is then
4.3e−14 against `𝓔` and 4.6e−14 for `τ` against itself, both at `Γ = 2`,
at Julia 1.11 and 1.13 alike (amended in step 2: step 1 measured 5.1e−14
and 5.4e−14 at `Γ = 5/3` at 1.11, before the metric products stopped
contracting; see "Precision and devices"); and the conditioned primitives 3.0e−14 in `(ρ, w, v)` and 2.7e−13 in `s`
(measured in step 1; amended in step 1 after review, which found the
default-tolerance assertion fragile). The scheme itself runs at the
default tolerance. `√γ D s` is not an input of the
hydrodynamic recovery and is checked through `s`. `con2prim_ideal` on
states built with `E = −v × B` (the same draws, `B²` up to 100 times the
fluid's `τ`, `B` along and across `v` included) has conditioned worst
errors 1.8e−12 in `(ρ, w, v)` and 3.7e−12 in `s`. At `Float32` the
recovery runs in a kernel and lands within 3.6e−5, conditioned, of the
state (EntropyEOS's `Float32` tolerance is 6.1e−5) on states dense enough
that the thermal energy is a percent of `τ`; at `ρ = 1e−9` the gas's `U` is
1e−7 of its rest mass and `Float32` cannot determine `s` at all.

*(Step 3:)* `con2prim_hydro` is split at the fluid's share — `field_budget`
subtracts the fields' energy and momentum, `fluid_recovery` recovers from
what is left and also returns the EOS point — so that the stage solve,
which iterates on that share, calls the same recovery bit for bit.

`con2prim_hydro` returns EntropyEOS's policy flags as they are, and only
them: `policy_flags`, not the solver's result, is EntropyEOS's validity
signal, and a failed solve is always repaired and flagged (proposed in step
1).

The only coupling back to the fields is the implicit Ohm step, which is
[a nested solve](#the-implicit-stage-solve) around this same call.

**Entropy fallback** (proposed): where `τ_fluid = τ − D − ½(E² + B²)` is a
small difference of large numbers (electromagnetically dominated cells, the
magnetosphere) or the energy inversion fails, invert at the advected `s`
instead (EOS-3). The switch is a threshold on
`½(E² + B²) / τ` (open).

## The scheme

### Where the fields live

A block's cells are at half-integer positions between integer vertices.
TreeAMR's centerings (`cellcentered`, `facecentered(D, d)`,
`edgecentered(D, d)`, `vertexcentered`) put each quantity where its
equation puts it (decided):

- **Cells**: the ten conserved matter variables
  `(D, S₁, S₂, S₃, τ, D Yₑ, D s, 𝔈¹, 𝔈², 𝔈³)`. `𝔈` is cell-centered and
  flux-conservative, so Ampère's law is updated with the same Riemann
  fluxes as the matter.
- **Edges**: `A_d` on the edges along `d`. Faraday's law in potential form
  puts the electromotive force `αE_d + (β × B)_d` there.
- **Faces**: `𝔅ᵈ = [dij] ∂ᵢ A_j`, the exact discrete curl of the edge
  values around face `d`. Derived, not evolved: the sum over a cell's six
  faces cancels identically, so `∇·𝔅 = 0` per cell to round-off, on every
  level, after every prolongation and every regrid, whatever operator
  moved `A`.
- **Vertices**: the gauge scalar `Φ̃ = √γ Φ`. `∂ᵢ(αΦ − βʲA_j)` is needed on
  the `i`-edges and `∂ⱼ(α√γ Aʲ)` lands on the vertices — the IllinoisGRMHD
  staggering — and TreeGH's metric is native there.

In `D < 3` the components along missing directions take the centering
that remains (in 1D, `A_x` is cell-centered and `A_y`, `A_z` are
vertex-centered), and `𝔅̄` supplies what no translation-invariant `A` can.

### Field sets

| set | centering | nvars | `G` | operators / at `regrid!` |
|---|---|---|---|---|
| GH `U` (TreeGH) | vertex | 20 | 3 | PointValue width 6 / `=> schedule` |
| matter `U` | cell | 10 | 3 | Conservative width 3 (limited when upstream has it) / `=> schedule` |
| `A_d`, `d = 1…3` | edge `d` | 1 each | 3 | PointValue, width (open) / `=> schedule` |
| `Φ̃` | vertex | 1 | 3 | PointValue, width as `A` / `=> schedule` |
| `P` — primitives, EOS aux, flags, `E`, `B̄` at cells, `q`, cell metric | cell | 30 | 3 | `=> nothing` |
| `B_d` — `curl A` | face `d` | 1 each | 3 | `=> nothing`, recomputed every RHS |
| `F_d` — matter fluxes | face `d` | 10 | 0 | `=> nothing`; `restrict_interfaces!` is the flux correction |
| `EMF_d` | edge `d` | 1 each | 0 | `=> nothing`; `restrict_interfaces!` is the CT correction |
| `Tsrc` — `T_ab` for GH | vertex | 10 | 0 | `=> nothing` |
| `X` — per-cell tallies, light speed *(added in step 5)*, the reset's injection *(step 6)* | cell | 18 | 0 | `=> nothing` |

`P`'s 30 slots are laid out in `src/variables.jl` (amended in step 1: the
table said "~25"): the eleven the recovery returns — `ρ★`, `vⁱ`, `s`, `Yₑ`,
the rapidity `w` (also the next recovery's warm start), `p`, `h`, `cs²` and
the EOS's own warm start — then the policy flags (as a number; every bit is
below 2¹⁶, so exact even in `Float32`), `Eⁱ`, `Bⁱ`, `q`, and the metric
`α`, `βⁱ`, `γ_ij` and `√γ`, all undensitized. The metric derivatives and
`K_ij` the sources read are not in it yet: step 2 decides the form of the
sources, and with it whether they are stored or interpolated on the fly
(proposed in step 1). *(Step 2:)* they are not stored, and `P` stays at 30
slots (proposed in step 2). The sources are their only reader, once per
cell per right-hand side, so storing them (36 more slots: `∂ⱼg_ab` and
`K_ij`) would only move the same evaluation into another kernel and add a
write and a read; the source kernel asks the provider for them
(`metric_gradient`), as it would ask for the metric at a face.

*As built in step 5* (`src/fieldsets.jl`, `mhd_fields(T, forest; G = 3)`):

- **The missing directions.** In `D < 3` a quantity of a missing
  direction takes the centering that remains: an edge along a missing
  direction is vertex-like in every present one, a face normal to one is
  cell-like in every present one (`edge_centering`, `face_centering`; for a
  present direction they are TreeAMR's `edgecentered`, `facecentered`). So
  in 1D `A_y`, `A_z` sit at the vertices, `𝔅ʸ`, `𝔅ᶻ` at the cells, and `𝔅ˣ`
  at the faces holds only the background. There are `D` flux sets and always
  three `B`, `A` and `EMF` sets.
- **One `G` for every ghosted set** (`U`, `A`, `Φ̃`, `P`, `B`), checked by
  `MHDProblem`. With `G_A = G_B` every stored face has the two edges its
  curl reads in storage, and every stored cell the two faces its average
  reads, so `B` and `P` cover the whole stored extent (`src/curl.jl`).
  `N ≥ 2G + 2` follows (TreeAMR's invariant for a vertex-like dimension):
  `N ≥ 8` at `G = 3`.
- **`B_d` holds the densitized `𝔅ᵈ`**, and `P`'s `Bⁱ` is its cell average
  over `√γ`. `P`'s `q` is written at the owned cells only (it is read by the
  sources and nothing else); its ghosts hold zero.
- **`X`, the diagnostics set** (proposed in step 5): per owned cell, the
  implicit stages that set each of the nine `OhmStageResult` bits, the
  stages whose state the policy repaired and the `|` of their flags, the
  recoveries and the most one stage made, the cells the reset wrote, and the
  light speed for the time step. *(Added in step 6:)* the cells the reset
  wrote as the step limiter, and the change those writes made to `√γ D`
  and `√γ τ`, summed — the rest mass and energy the reset injects. The counts accumulate over a chunk and are
  reduced once at its end (`ohm_tally`), which is "reduced per chunk"
  without a reduction per stage.
- **The operators** (proposed in step 5; the `A` width stays (open) until
  step 8): Conservative width 3 for `U`, PointValue width 4 (prolongation and
  restriction) for `A` and `Φ̃` — the scheme's order plus one, rounded up to
  even. Nothing in step 5 refines, so they only have to pass TreeAMR's
  checks; they are `MHDProblem`'s `ops_U` and `ops_A`.

Operator widths follow TreeAMR's convention: the order of an operator is
its stencil width `w` per dimension and it is exact to degree `w − 1`;
PointValue widths are even, Conservative widths odd
(`TreeAMR/src/operators.jl`). TreeGH's `prolongation = 6` is a 6-point,
degree-5 interpolation. `B` takes one derivative of `A`, so `A`'s
PointValue width must be at least the scheme's order plus one (open, to be
measured as TreeWave and TreeGH measured theirs).

### Reconstruction

Reconstructed at faces (proposed): `ρ`, `s`, `Yₑ`, `uⁱ = W vⁱ` (so that
every reconstructed state is subluminal), `Eⁱ`, and the two **transverse**
components of `B` from their cell averages. The **normal** component of
`B` at a face is `B_d` itself, never reconstructed. Methods, in order of
arrival: PLM with minmod and MC (`G = 2`), then PPM, WENO-Z and MP5
(`G = 3`, which the table above already provides). A face state that is
unphysical (negative `ρ`, `s` outside its window) falls back to first
order, per face and per variable group. EOS evaluation at face states uses
the cell's warm start.

*As built in step 5* (`src/reconstruction.jl`, proposed in step 5): twelve
numbers a cell, `(ρ, s, Yₑ, uⁱ, Eⁱ, Bⁱ)` from `P`, undensitized, each
extrapolated as `q ± slope/2` with TreeHydro's `slope` (`:none`, `:minmod`,
`:mc`; `limiter` has no default, as in TreeHydro). All three components of
`B` are reconstructed and the normal one is then replaced by the face's
`𝔅ᵈ/√γ`. The two groups are `(ρ, s, Yₑ, uⁱ)` — inadmissible where `ρ` is
not positive and finite, `Yₑ` outside the EOS's window, `s` outside
`srange(eos, ρ, Yₑ)` or `u` not finite — and `(Eⁱ, Bⁱ)`, inadmissible only
where not finite; an inadmissible group takes its cell's values on that side
of that face. The face state is then `vⁱ = uⁱ/W`, one EOS evaluation
(`primitive_state`) and `prim2con` in the face metric, which the provider
evaluates at the face (`Flat`, `Analytic`; `GHMetric` will interpolate it,
step 10). A face's state costs two EOS evaluations, `primitive_state`'s and
`prim2con`'s.

### Riemann solver

`riemann_flux(::Val{:llf | :hlle}, …)` on the ten matter-plus-`𝔈`
variables at every face (proposed). Because `S_i` and `τ` are totals, their
fluxes contain the Maxwell stress, which propagates at the speed of light;
the default wave speeds are therefore the local light cone,
`λ_± = −β^d ± α √(γ^{dd})`, for all ten variables — the choice of the
resistive codes this design follows, robust and diffusive. Split speeds
(fluid speeds for `D`, `D Yₑ`, `D s`, light speed for the rest) and a
resistive HLLC are (open), to be decided by measurement on the TOV star.

*(Step 2:)* `src/flux.jl` and `src/riemann.jl`. A face state is the tuple
`(cons, prim, E, B)` — the densitized conserved state, the primitive state
it was built from, and the contravariant fields — which is what
`physical_flux(m, cons, prim, E, B, Val(d))` takes (proposed in step 2).
Both states are at the same face, so the light-cone speeds are the same on
both sides and the solvers need no EOS. Measured in step 2: `F(U, U)` is the
physical flux exactly for LLF and to 2.2e−16 for HLLE; the direction-`d`
flux is the direction-1 flux of the relabelled state to 4.7e−16 in flat
space and 6.8e−16 in Kerr-Schild; the fluxes are `α√γ Tᵈ_j` (momentum) and
`α²√γ T^{td}` (energy) to 4.8e−16; and the light cone is null in `g^{ab}`
to 1.6e−15.

### The electromotive force for `A`

Upwind constrained transport (proposed): at each `d`-edge the EMF
`αE_d + (β × B)_d` is the average over the four edge states reconstructed
from the surrounding cells (for `E`) and faces (for `B`), plus upwind terms
proportional to the jumps of the face-normal `B` reconstructed to the edge
from either side — the HLL form of Del Zanna et al. (2003) and Londrillo &
Del Zanna (2004), with light-cone speeds. It dissipates `B` consistently
with the LLF/HLL dissipation `𝔈` receives from the Riemann solver; without
it the Maxwell subsystem is upwinded in `E` alone. The metric at edges
comes from 1D 4th-order interpolation between the edge's vertices.

*(Step 2:)* `uct_hll_emf(m, E, Bᵃ, Bᵇ, Val(d))` in `src/emf.jl` (proposed
in step 2). With `(d, a, b)` cyclic, its four states are the full `E` of
each state (`LL, LR, RL, RR`, the first letter the side along `a`), and the
face-normal fields that cross the edge: `Bᵃ` reconstructed along `b`, `Bᵇ`
along `a`, one value per side. `Bᵈ` does not enter, since
`(β × B)_d = √γ(βᵃBᵇ − βᵇBᵃ)`. The result is
`Σ wᵃwᵇ e + √γ aᵃ(Bᵇ_R − Bᵇ_L) − √γ aᵇ(Bᵃ_R − Bᵃ_L)`, the HLL average of
`𝔅ᵇ`'s flux `−EMF_d` along `a` and `𝔅ᵃ`'s `+EMF_d` along `b`. The test
that pins it: in a constant metric (frozen Kerr-Schild samples) every jump
of the 1D vacuum Maxwell equations with continuous `Eˣ`, `Bˣ` is carried
by light waves at exactly `λ±`, so HLL with those speeds is the exact
Godunov solver, and the HLLE flux of `𝔈` together with the UCT EMFs equals
the physical flux of the HLL middle state — to 3.9e−16 (measured in step
2). A wrong sign in the EMF's shift or jump terms, or in the flux of `𝔈`,
breaks that.

*As built in step 5* (`emf_kernel!` in `src/evolution.jl`, proposed in step
5), at every edge of the closed range: the four states are the cells on
either side of the edge along `a` and along `b` (along a missing direction,
the one cell), each with its `E` extrapolated to the edge by the limited
slope along each present transverse direction, `E + σₐ slopeₐ/2 + σ_b
slope_b/2`; `𝔅ᵃ` is reconstructed along `b` from the `a`-faces and `𝔅ᵇ`
along `a` from the `b`-faces, with the same limiter, both **densitized**,
and divided by the edge's `√γ`. The flux of `𝔈` reconstructs the
undensitized cell average instead; the two agree in flat space and differ
at truncation order otherwise, which rung 3 will see if it matters. In 1D
the EMF along `x` is the cell's own `αE_x + (β × B)_x` (no transverse
direction to reconstruct along), and those along `y`, `z` are the 1D HLL
fluxes of `𝔅ᶻ` and `𝔅ʸ`.

### Sources

At cell centers, explicit: the geometric momentum and energy sources
above; the shift term `√γ q βⁱ` and the `−α√γ q vⁱ` part of the current in
Ampère's law. At vertices, explicit: the gauge damping `−ξ α Φ̃`. **The only
implicit term** is Ampère's `−α√γ σ W [E + v × B − (v·E) v]`.
*(Step 2:)* `src/sources.jl`: `geometric_source` (both forms), the cell's
whole `explicit_source` (none for `D`, `D Yₑ`, `D s`), and Ampère's source
as `ampere_explicit_source` + `ampere_stiff_source` = `ampere_source`, the
stiff part kept as a function for the stage solve's residual.

### Floors, the atmosphere and the magnetosphere

EntropyEOS's policy defines the atmosphere (`ρ_atm`, triggers, ceilings)
and `con2prim_safe` applies it; a limiter resets `v = 0` in atmosphere
cells and writes back only those cells, as TreeHydro's does (proposed).

**The reset is both the stage limiter and the step limiter** (proposed;
amended 2026-09-25). In IMEXRungeKutta a stage limiter's change reaches
`uⁿ⁺¹` only through the explicit right-hand side. This differs from
TreeHydro's `SSPRK33`, where each stage's reset persists with weight 1/6,
2/3 or 1. Each role does one job:
- as the stage limiter, it keeps every right-hand side from reading an
  unphysical state;
- as the step limiter, it is what makes the reset hold in the state.

With the stage limiter alone, the state would never be reset. The
injection that the reset accounts is then an equality, as under
TreeHydro's `reset = :step`: only the step limiter's changes reach the
state, and the fluxes stay conservative whatever state they were computed
from. See IMEXRungeKutta's `CODE.md`, "One step" and "Limiters in other
codes".

*As built in step 5* (`reset_atmosphere!`, `src/evolution.jl`): the reset
computes `B` from the state's own `A` (a scatter and ghost fill of the three
`A` sets and the curl, as the stage solve does), recovers every owned cell,
and **writes back every cell the policy repaired** — `FLAG_POL_ANY`, not the
atmosphere alone (proposed in step 5): every repaired state is exactly
invertible and is what the next right-hand side would recover anyway, so
writing it makes the state say what the scheme computes with. It writes the
seven matter slots and never `𝔈`, and counts each cell it writes into `X`.
A written state recovers as itself, so a second reset changes nothing (a
test). In step 5's smooth runs it writes nothing.

*(Step 6:)* the two roles are counted apart — `reset_atmosphere!` is the
stage limiter, `reset_step!` the step limiter — and the step limiter also sums
what each write changes in `√γ D` and `√γ τ` (`ohm_tally`'s `injected_D`,
`injected_τ`, volume-weighted). Since only its writes reach the state, the
totals of a box whose boundary fluxes vanish change by exactly those sums,
to rounding: the atmosphere tube asserts it (`test/atmosphere_tests.jl`),
which also shows the stage limiter's writes never leak into `uⁿ⁺¹`.

The conductivity model takes σ → 0 in the atmosphere, so the
exterior is electrovacuum: the atmosphere carries no current (`v = 0`
and `σ = 0` there), and the fields outside a star are those of the
initial vector potential evolved by the vacuum Maxwell equations. The
initial dipole is analytic in `A`, which makes it exactly divergence-free
on the mesh.

## Time integration

### The partition

One IMEX Runge–Kutta integrator over the multi-set state vector
(decided). The **explicit** part is everything except one term: all of
TreeGH, the matter and `𝔈` fluxes and sources, the EMF update of `A`, the
gauge. The **implicit** part is the σ current in Ampère's law, which is
local to a cell. The default tableau is Pareschi–Russo SSP3(4,3,3), whose
implicit part is L-stable, so that `σ Δt → ∞` relaxes to ideal MHD;
SSP2(2,2,2) is kept for debugging (proposed). Neither is stiffly accurate
(their weights `b` differ from the last row of the implicit `A`), so the
order *in* the stiff limit may drop; IMEXRungeKutta's validation
measures it, and the ARS schemes, which are stiffly accurate, are the alternative if it
matters. *(Amended with PLAN.md: an earlier version called SSP3(4,3,3)
stiffly accurate.)* `Δt` is fixed per chunk from the light-speed CFL on
the finest level; TreeGH's CFL factor under SSP3 is measured here, in step
11. The driver is chunked, with a regrid between chunks and a fresh
integrator after each, as in TreeHydro and TreeGH.

IMEXRungeKutta's validation (step 3) measured SSP3(4,3,3): its
implicit part is L-stable (R(∞) = 0 from the closed-form α = (9 −
√57)/6; A-stable by its E-polynomial, computed in step 1). In the
stiff limit its order drops from 3 to 2 in the stiff component (Kaps:
1.989 at ε = 1e−6, 1.988 at 1e−9), while the non-stiff component keeps
3.01. Each step ends Δt(1 − bᵀA⁻¹c̃)f = −0.2844 Δt f off the
equilibrium, where f is the explicit tendency of the relaxing
variable. This does not accumulate over steps, but where the
equilibrium has a jump it is an O(1) overshoot of 0.2844 C on each
side (total variation +1.14 on 2 at C = 1). ARS(4,4,3) keeps order 3
and lands on the equilibrium to O(ε).

**Measured in the scheme** (step 5, `test/wave_tests.jl`). Under
`IdealConductor` every stage solve imposes `E = −v × B` exactly, and the
offset above leaves the `𝔈` of each step's result `O(Δt)` off it. On the
Alfvén wave the stored `𝔈` converges at first order (L1 rate 0.85 over
N = 50…400; errors 5.5e−3 to 9.4e−4), while its ideal projection — one
stage solve of the final state, `−v × B` of its ideal recovery — and every
other variable converge at second (2.09 to 2.85). The offset does not
accumulate: the next stage solve replaces it and no right-hand side reads it.
Under ARS(4,4,3) the stored `𝔈` is the projected one (the same errors to
four digits, rate 2.10), for 4 explicit evaluations a step instead of 3:
8% more wall time on the 1D wave. What the offset can still do is reach the
step limiter, which recovers `uⁿ⁺¹` — `𝔈` included — and writes back what
the policy repairs: where the field's energy dwarfs the fluid's, an
`O(Δt)` error in `½E²` is a large error in the fluid's share. SSP3(4,3,3)
stays the default (proposed in step 5); step 6's stiff-limit and
atmosphere runs are where to compare the two, and ARS(4,4,3) is one
keyword away (`evolve!`'s `tableau`).

**SSP3(4,3,3) stays the default; the question is closed** (decided in step
6, on the measurements below — Erik left the choice to them). The two
tableaus give the same answer wherever the explicit part's stability does
not matter, and where it does, SSP3(4,3,3)'s SSPRK(3,3) explicit part keeps
the density positive and the reset idle while ARS(4,4,3)'s does not:
- the same accuracy on Balsara 1, 2, 3 and 5, the resistive tube at every
  σ (the distances to the ideal run agree to four digits) and the current
  sheet;
- on Balsara 4 (`W = 22` flows colliding) ARS(4,4,3) is 25% further from
  the reference and its `R4` plateau is 2.55% off, outside the 2% the
  prediction allows, against 0.64%; on Balsara 3 and 4 it fires the reset
  (3 and 2 cells, injecting `2.8·10⁻³` and `0.22` of energy) where
  SSP3(4,3,3) fires none;
- into a near vacuum its explicit part undershoots: a rarefaction into a
  medium at `10⁻⁸` reaches `2.1·10⁻¹⁰` (the reset writes 114 cells) where
  SSP3(4,3,3) never goes below the medium, and in the magnetized ideal
  atmosphere (`Bʸ = 10⁻³`, `B²/ρ = 100`) the undershoots, repaired where
  the field's energy dwarfs the fluid's, inject `2.6·10⁴` of rest mass
  into a box that holds 0.5, and the flow runs away to `W = 49`;
- with the magnetization cutoff it takes 7 fallbacks and 33 recoveries in a
  stage, SSP3(4,3,3) none and 6;
- it costs 10–17% more wall time.

SSP3(4,3,3)'s price is the one step 5 measured: each step's stored `𝔈` is
`O(Δt)` off the stiff equilibrium — at a shock an `O(1)` jump, up to 8.1%
of `max |𝔈|` from its ideal projection on Balsara 1 at 400 cells (ARS:
`10⁻³⁰`). Nothing reads it but the next stage solve, which replaces it, and
the step limiter, which in no Balsara, resistive or current-sheet run wrote
a cell; in the atmosphere runs the injection it allows is the table's.

The comparison (`bin/tableau_comparison.jl`, Julia 1.13, Apple M3 Pro, one
thread; MC, HLLE, `cfl = 2/5`; Balsara at 400 cells against both 3200-cell
references, which agree to `5·10⁻⁵` (Balsara 2) and `5·10⁻³` (Balsara 3) in
`D`; the plateau error the worst over `BALSARA_PLATEAUS`; "excursion" the
largest departure of `√γ D` from the reference's range over three cells
either side, as a fraction of its total range):

| case | tableau | L1 `D`, `τ`, `𝔅ʸ` | plateau | TV excess | excursion | stored `𝔈` off | resets stage/step | injected `D`, `τ` | wall |
|---|---|---|---|---|---|---|---|---|---|
| Balsara 1 | SSP3 | 0.00696, 0.0141, 0.00824 | 1.26% | 0.22% | 5.5% | 8.1% | 0/0 | 0, 0 | 2.60 s |
| | ARS | 0.00696, 0.0141, 0.00824 | 1.27% | 0.23% | 5.6% | 0 | 0/0 | 0, 0 | 3.03 s |
| Balsara 2 | SSP3 | 0.0305, 0.250, 0.0243 | 1.91% | −0.30% | 8.4% | 0.18% | 0/0 | 0, 0 | 3.45 s |
| | ARS | 0.0305, 0.250, 0.0243 | 1.91% | −0.27% | 8.6% | 0 | 0/0 | 0, 0 | 3.99 s |
| Balsara 3 | SSP3 | 0.380, 18.9, 0.0592 | 4.69% | −31% | 13.8% | 0.24% | 0/0 | 0, 0 | 6.15 s |
| | ARS | 0.375, 18.7, 0.0591 | 4.20% | −31% | 12.9% | 0 | 3/3 | 6·10⁻¹⁹, 2.8·10⁻³ | 6.68 s |
| Balsara 4 | SSP3 | 0.382, 11.4, 0.153 | 0.64% | 14.5% | 5.2% | 4.4% | 0/0 | 0, 0 | 4.57 s |
| | ARS | 0.477, 12.4, 0.185 | 2.55% | 18.8% | 5.3% | 0 | 2/2 | −10⁻¹⁴, 0.222 | 5.13 s |
| Balsara 5 | SSP3 | 0.00829, 0.0198, 0.00754 | 0.63% | 3.8% | 4.2% | 1.9% | 0/0 | 0, 0 | 4.43 s |
| | ARS | 0.00829, 0.0199, 0.00756 | 0.64% | 3.8% | 4.2% | 0 | 0/0 | 0, 0 | 5.06 s |

| case (200 cells) | SSP3 | ARS |
|---|---|---|
| resistive tube, L1(`𝔅ʸ` − ideal) at σ = 0, 10, 10², 10³, 10⁶ | 0.392, 0.202, 0.0611, 0.0134, 1.729·10⁻⁵ | 0.392, 0.202, 0.0611, 0.0134, 1.729·10⁻⁵ |
| resistive tube, stages fixed point / Newton at σ = 10² | 94,930 / 65,070 | 39,867 / 120,133 |
| current sheet to `t = 9`, L1 to the erf profile; wall | 9.30·10⁻⁴; 5.6 s | 9.30·10⁻⁴; 6.4 s |
| rarefaction into `10⁻⁸` (atmosphere `10⁻¹⁰`): min ρ; resets | 1.0·10⁻⁸; 0/0 | 2.1·10⁻¹⁰; 152/114 |
| atmosphere tube, `B = 0`: injected `D`, `τ` | −8.4·10⁻¹², −2.2·10⁻¹⁰ | 2.4·10⁻⁹, 1.2·10⁻⁸ |
| atmosphere tube, ideal, `Bʸ = 10⁻³`: injected `D`, `τ`; max W | 1.6·10⁻⁹, −2.5·10⁻⁷; 4.83 | 2.6·10⁴, 5.3·10¹³; 48.9 |
| atmosphere tube, `Constant(10³)` cut off at `μ_max = 10`: injected `D`, `τ`; fallbacks; most recoveries | 2.5·10⁻⁸, 6.9·10⁻⁸; 0; 6 | 4.6·10⁻⁸, 1.5·10⁻⁶; 7; 33 |

*As built in step 5* (`src/driver.jl`): `evolve!(T, case, Val(D); N,
t_end, chunk, cfl, limiter, …)` builds the forest of `case.roots` blocks of
`N` cells, the field sets and the problem, fills the exact initial state
and runs chunk by chunk, each a fresh integrator that takes over the last
one's scratch (`reuse`) and steps the state in place (`alias_u0`). Each
chunk's `Δt` is `cfl / (D · max λ/h)` from the light cone at the chunk's
start (`max_light_rate`, a `mesh_mapreduce` of the cells' light speed over
their spacing), rounded down to a whole number of steps with TreeHydro's
`chunk_count` tolerance; `cfl = 2/5` (proposed in step 5: TreeHydro's,
under SSP3(4,3,3)'s explicit part, which is SSPRK(3,3)). The stage
arithmetic is split evenly over the threads (`partition = :even`) for a CPU
state; the block-owner partition, which needs TreeAMR's unexported
`threadchunks`, is step 14's. A run in one chunk and the same run in four
chunks of whole steps of the same `Δt` are bit-identical (a test).

### The implicit stage solve

Per cell, per stage (proposed). Known: `D`, `S_i`, `τ`, `D Yₑ`, `D s`, the
cell-averaged `B` (explicit in the stage), the metric, and `𝔈★`, the
explicit part of the stage value. Let `c = γΔt · α σ`, with `γΔt` the
stage's implicit coefficient times `Δt`.

- **Given `v`**, `E` solves the linear system
  `(1 + cW) E − cW (E·v) v = E★ − cW v × B`, in closed form
  (Sherman–Morrison).
- **Given `E`**, the fluid primitives are one call of `con2prim_safe` in
  hydrodynamic mode, as in [Primitive recovery](#primitive-recovery).

So the stage is a three-dimensional root find for `v`,
`R(v) = v − V_c2p(S − E(v) × B, τ − ½(E(v)² + B²))` (amended in step 3:
the same map, iterated on the fluid's share of the energy and momentum
instead; see "As built" below). **One con2prim is used for every σ.** A
plain fixed-point iteration contracts roughly like `λ b²/(ρh)` with
`λ = cW/(1 + cW)`: fast where `σΔt` is small (the magnetosphere, where `E`
barely depends on `v`) or where the magnetization is low (a star's
interior); slow or divergent only where both are large, such as the
surface layer. There the outer iteration is Newton on `v`, still one
con2prim per iteration, seeded from the `σΔt → ∞` limit: there
`E = −v × B` and the whole inversion is exactly the ideal-MHD problem,
which the same EntropyEOS routine solves when called with `B² ≠ 0`. That
ideal call is a seed — and the exact answer for `IdealConductor` — not a
second con2prim. The last resort is the lagged `v` of Palenzuela et al.
(2009). Every outcome is flagged, counted, and reduced per chunk.

**The solve writes only `E` into `U`** (proposed; amended 2026-09-25).
No limiter has seen `u★`, so its fluid part may be inadmissible.
`con2prim_safe` repairs the solve's own view of it, and that repair stays
local to the solve. Written into `U`, it would become part of the implicit
increment `U − u★`. The integrator would then re-weight it by `b_k/a_kk`,
which is 2.758 at SSP3(4,3,3)'s last stage (IMEXRungeKutta's `CODE.md`,
"The callback contracts" and "One step"). PLUTO's resistive solve does the
same: it repairs its predictor and writes back only `E`.

**Clamps inside a Newton residual** (open). The Newton iteration on `v`
calls `con2prim_safe` in every residual, and `con2prim_safe` clamps
(atmosphere, ceilings, the rapidity cap). Where a clamp is active, the
residual is flat, and a Jacobian through it is singular.
- grim hit this, and moved its clamp out of the residual (reported by
  IMEXRungeKutta's survey, grim commit `df67cbb`).
- KHARMA instead backtracks the Newton step to keep `ρ` and `u` positive,
  and fills failed cells from their neighbours afterwards.

Where `con2prim_safe` reports a repair inside the iteration, the likely
answer here is to leave Newton for the fixed point or the ideal seed, and
to flag it. Rung 1's σ × magnetization sweep decides.

**As built** (step 3, `src/stage_solve.jl`):
`solve_ohm_stage(eos, opts, pol, σ, m, cons★, E★, B, γΔt, guesses; stage)
-> (E, prim, result, recoveries)`, allocation-free and bit-identical in a
CPU kernel at `Float64` and `Float32` (measured in step 3).

- **The iteration runs on the fluid's share** of the energy and momentum,
  `X = (S_f, τ_f)`, not on `v` (amended in step 3). Each iteration recovers
  the fluid from `X` (one `con2prim_safe`, hydrodynamic mode), solves Ohm's
  law for `E` at its velocity, and compares `X` with the share
  `X(E) = (S − ε E × B, τ − ½(E² + B²))` that `E` leaves. It is the same
  map as the root find for `v`, cut at a different place. The cut matters
  where `b²/ρ ≫ 1`: there `X(E)` is a small difference of large numbers, a
  velocity error `δu` moves it by `λ b²/(ρh) δu` of the fluid's own
  energy, and the recovery repairs whatever the fluid cannot hold. Newton
  on the velocity had a basin of `δu ~ 10⁻⁴` at `b²/ρ = 10³`, and fell
  back on 191 of 1296 points of the sweep below (Γ = 2); Newton on `X`
  hands the recovery only the linearly predicted share, which stays on the
  fluid's scale, and falls back on 40 (measured in step 3).
- **The velocity is carried as `u = W v`** (proposed in step 3): every `u`
  is subluminal, and the Jacobian through `v` has a `W² (E★ − E) vᵀ` term
  from `∂W/∂v` that through `u` becomes `(E★ − E) vᵀ/W`.
- **The Jacobian is analytic** (decided in step 3). `∂u/∂X` is the
  recovery's derivative by the implicit function theorem on its two
  equations `|S| = z tanh w`, `τ = z − p − D` in `(s, w)`, with
  EntropyEOS's own `z_w`, `z_s`, `p_w`, `p_s` from the EOS point the
  recovery returns; `∂E/∂u` is implicit differentiation of the linear
  system. Newton's `1 − PQ` (4×4, `P = ∂X/∂E · ∂E/∂u`, `Q = ∂u/∂X`) is
  inverted by Woodbury through the 3×3 `1 − QP`, and a density-dependent σ
  adds a rank-one term through `∂ρ/∂X` and `conductivity_slope(model, prim)
  = ∂σ/∂ρ` (proposed in step 3; a model reading `s` or `Yₑ` would add a
  column each). Both derivatives agree with central differences to the
  differences' own error, 1.1e−8 and 8.3e−8 (measured in step 3). No
  extra recovery, no finite difference.
- **Which branch** (proposed in step 3): σ evaluated at the predictor's
  recovery, at `E★`; if it is zero, `E = E★` and that recovery are the
  answer, bit for bit (`OHM_SIGMA_ZERO`). Otherwise the fixed point where
  the contraction factor `‖K‖`, the Frobenius norm in an orthonormal frame
  of the fixed point's Jacobian at the predictor, is at most `1/100`, and
  Newton otherwise; the fixed point also hands over to Newton when an
  iteration shrinks the residual by less than half (`OHM_SWITCHED`).
  Newton is seeded from the ideal recovery where `λ = cW/(1 + cW) > ½`, from
  the predictor otherwise, and from whichever of the two the policy did not
  have to repair if only one (`OHM_IDEAL_SEED`). At a threshold of `1/10`
  the fixed point needed up to 11 recoveries (4.4 on average) against
  Newton's 7; at `1/1000` the sweep below costs 3% fewer recoveries than at
  `1/100`, which is within its noise.
- **Clamps inside the residual** (decided in step 3, closing the (open)
  above): a share the recovery had to repair is replaced by the repaired
  state's own (`fluid_share`), so that every iterate is a fluid state and
  its derivative is the recovery's; a Newton step that grows the residual
  is halved, up to eight times (`OHM_BACKTRACKED`); and any repair inside
  the iteration is flagged (`OHM_REPAIRED`). Leaving Newton for the fixed
  point at a repair, the guess above, was not needed.
- **Converged** when `|X − X(E)| ≤ 10⁻¹²(τ + D)` (proposed in step 3;
  `max(1e-12, 1024 eps(T))`): the share closes to EntropyEOS's own
  recovery tolerance, measured against the total energy density as the
  recovery's round trip is. The returned `E` was solved at the returned
  state's velocity, so Ohm's law holds between them to rounding, and
  `prim2con(prim, E, B)` reproduces the totals to that tolerance plus the
  recovery's.
- **The fallback** after 30 recoveries is the lagged velocity: `E` at the
  velocity of the iterate with the smallest residual, the fluid recovered
  from that `E` (`OHM_FALLBACK`).
- `OhmStageResult(flags, policy_flags)`: one branch bit and any of the
  modifiers, and EntropyEOS's flags of the returned state; both reduce by
  `|` and count by bit (step 5 counts them per chunk).

**The sweep** (measured in step 3, `test/stage_solve_tests.jl`; Julia
1.13, 1.11 alike but where noted). Stage problems are built backwards
from a known solution: a fluid in a boosted Kerr-Schild metric,
`E = −v × B` plus a non-ideal part up to `|B|/(1 + cW)`, and the `E★` the
stage equation then requires. On `σΔt ∈ 10^{−3…8}` × `b²/ρ ∈ 10^{−4…4}`
(ideal comoving `b²`) × `W ∈ {1, 2, 10}`, eight problems a point, for
`Γ ∈ {4/3, 5/3, 2}`:

- no NaN; at every converged point Ohm's law holds to 3.3e−16 of its terms
  and the matter to 1.5e−12 of `τ + D`;
- the fixed point takes every point with `b²/ρ ≤ 10⁻³`, at every `σΔt`,
  and its boundary moves up to `b²/ρ ~ 10⁻¹` as `σΔt` falls to `10⁻³`; it
  needs at most 6 recoveries (3.6 on average). Newton takes the rest, in
  at most 10 recoveries, seeds included, where no iterate needed a repair
  (3.6 to 4.2 on average), and up to 32 where some did. Of the 2592
  points per Γ, 1248 and 907 (Γ = 4/3 and 2) by the fixed point, 1308 and
  1582 by Newton, 1084 and 1283 of those from the ideal seed;
- **the fallback** took 36, 78 (77 at 1.11) and 103 points (Γ = 4/3, 5/3, 2;
  at most 4.0%), all at `b²/ρ ≥ 10` and `σΔt ≤ 10²`, mostly `W ≥ 2`, and
  all but one where both seeds needed a repair: where `E★` departs from
  both the solution and the ideal field by far more than the fluid's
  energy, so that neither the predictor's share nor the ideal recovery is
  a fluid state, and the solution is an island in `σΔt` — continuation
  from either limit was tried and does not reach it. *(Decided
  2026-10-14:)* the scheme takes σ to zero there with the
  [magnetization cutoff](#ohms-law-and-the-conductivity), under which
  none of these problems falls back (the uniform draw: 1 of 124,416 over
  sixteen seeds, at Γ = 2). Without it, the lagged `E` there is off the solution by a median 0.11 of `|E| + |B|`, at most 0.46 (Γ = 2);
- **the stage equations have more than one root** there: 5, 2 and 2 points
  converge, to the same residual, to another solution than the one the
  problem was built from;
- `DensityProfile` with its cutoff within a factor of three of the
  density: residual 8.6e−13, and 1 of 148 not converged, a Newton iteration
  held at the entropy window's edge (3 before the Jacobian carried
  `∂σ/∂ρ`);
- 200 inadmissible predictors (the share at `E★` below the coldest state):
  all finite, and where converged the residual is 1.9e−13.

Cost (measured in step 3, Julia 1.13, Apple M3 Pro, the Γ = 2 gas, one
call): 1.3 µs on the fixed point (3 recoveries of 0.37 µs), 1.7–2.4 µs for
Newton (3–4 recoveries), 0.7 µs at `σ = 0` and 0.5 µs for
`IdealConductor`; a fallback costs 33 recoveries, 0.17 ms where the
recoveries themselves take EntropyEOS's repair path. Under IMEXRungeKutta's
SSP3(4,3,3), with `solve_imp!` writing only `𝔈`, a resting cell's field
decays as `exp(−ασt)` at orders 2.92 and 2.96, its matter variables come
back bit for bit, and one step at `ασΔt = z` leaves `6.7/z` of the field
(`z = 10⁶` and `10⁸`), SSP3(4,3,3)'s L-stability.

### The IMEX package

The integrator is **IMEXRungeKutta.jl** (decided; `~/src/jl/IMEXRungeKutta`),
a stand-alone package. Its `CODE.md` is authoritative for the method, the
tableaus, the stage semantics, and why OrdinaryDiffEq and ClimaTimeSteppers
do not fit (in short, neither lets the user solve the implicit stage). This
package relies on it for:

- one `solve_imp!(U, u★, γΔt, p, t)` call per implicit stage, which here
  is the Ohm solve of [The implicit stage solve](#the-implicit-stage-solve).
  The integrator recovers the tendency as `(U − u★)/(γΔt)` and never
  evaluates the σ current on its own;
- the tendency taken before the stage limiter, which here is the
  atmosphere reset. The stage limiter's change reaches `uⁿ⁺¹` only
  through the explicit right-hand side, so the reset is also the step
  limiter ([Floors, the atmosphere and the
  magnetosphere](#floors-the-atmosphere-and-the-magnetosphere); amended
  2026-09-25). Folding the reset into the tendency instead would
  re-weight it by `b_k/a_kk`, 2.758 at SSP3(4,3,3)'s last stage;
- explicit evaluations only where the tableau uses them: three right-hand
  sides (TreeGH plus MHD) per SSP3(4,3,3) step;
- fixed `Δt`, generic arrays and Julia 1.10.

Per SSP3(4,3,3) step, IMEXRungeKutta 1.4.0 makes 3 explicit evaluations,
4 stage solves, 3 stage-limiter calls (its first stage is implicit, so no
explicit stage is trivial) and 1 step-limiter call; the prerequisite tests
pin all four counts (measured in step 0). It is a dependency from step 0,
through `[sources]`, so that the contract is checked before step 5 builds
on it (proposed in step 0).

When its validation measures SSP3(4,3,3)'s L-stability and stiff-limit
order, amend [The partition](#the-partition). *(Amended 2026-09-24: the
specification moved to IMEXRungeKutta's `CODE.md`.)*

## Spacetime coupling

### Metric providers

The kernels dispatch on a metric provider (proposed):

- `Flat()` — Minkowski, all `D`, no storage.
- `Analytic(m::SpacetimeMetrics.AbstractMetric)` — evaluated at the point
  (cell, face, edge or vertex), all `D`; the Kerr-Schild tests.
- `GHMetric` — TreeGH's vertex-centered `(h_ab, Π_ab)`, either **frozen**
  (the Cowling approximation: the GH right-hand side is not called) or
  **evolved**. One code path for both, so the Cowling runs test exactly
  the interpolation the coupled runs use.

*(Step 2:)* `Flat` and `Analytic` are in `src/metric.jl`; every provider
answers `point_metric(provider, t, x) -> Metric3` and
`metric_gradient(provider, t, x) -> MetricGradient`. `MetricGradient`
holds `dg[j, a, b] = ∂_j g_ab` — the spatial derivative index **first**, as
in GHSO2, then the spacetime indices with 1 = `t` — and `K_ij`; the lapse
and shift derivatives are derived from it (`d_lapse`, `d_shift`, which
agree with ForwardDiff of the 3+1 split to 2.4e−16 of their terms). The
conversion from SpacetimeMetrics' derivative-last `dmetric` is written in
`src/metric.jl` and nowhere else (proposed in step 2). `K_ij` is computed
there from the same `dmetric` call: SpacetimeMetrics' `ExtrinsicCurvature`
differentiates the metric a second time and allocates 432 bytes per call at
Julia 1.11 and 1.13, which a device kernel cannot (measured in step 2); the
two agree to 4.3e−16. That local computation was the workaround of SM-1
(amended in step 2). *(Amended 2026-10-14:)* SM-1 landed in SpacetimeMetrics
1.7.0; `metric_gradient(::Analytic, …)` now calls its allocation-free
`ExtrinsicCurvature(g, dg)` on the same `dmetric` result, and the pointwise
layer is still bit-identical in a CPU kernel.

The matter code is generic in `D ∈ {1, 2, 3}` with three-component fields
and translation invariance along missing directions, so that shock tubes
are one-dimensional runs; coupling to TreeGH is `D = 3` only (proposed).

### Metric at cells, faces and edges

Interpolated within a block from the vertices (proposed):

- **Cells**: 4th-order tensor Lagrange, 64 points; `∂ᵢ g_ab` at cells by
  the staggered 4th-order derivative along `i` times the transverse
  interpolation; `K_ij` from `Π` and `∂h` (TreeGH's `adm_vars_from_state`).
- **Faces**: 2D, 16 points. **Edges**: 1D, 4 points.

Stored once per right-hand side in `P` for cells; on the fly at faces and
edges. The derivatives and `K_ij` are not stored but evaluated where the
sources are (proposed in step 2; see "Field sets"). One interaction with the ghost widths is (open): a 4th-order
interpolation to a cell center reaches one vertex beyond the cell on either
side, so reaching the outermost ghost cell of a `G = 3` cell set needs
`G = 4` on the vertex set, and TreeGH fixes `G = q/2 + 1 = 3`. The
primitives in that outermost ghost cell are read by the widest
reconstructions. The options are a larger GH ghost width (GH-5),
a 2nd-order interpolation in that one layer, or exchanging the
cell metric as a ghost-filled set.

### Stress-energy to the vertices

`T_ab` is computed at the cells from the primitives and the cell metric,
then averaged to each owned vertex from its `2^D` neighbouring cells into
`Tsrc` — second order, and positivity-preserving for the energy density at
a stellar surface where a higher-order interpolation would ring
(proposed; a 4th-order option is kept, and which one wins is (open)).
TreeGH reads `Tsrc` at the owned point (GH-3).

### Gauge for the TOV milestones

TreeGH's static-background path, with no gauge driver (proposed): the TOV
star is a static, `isbits` `SpacetimeMetrics.AbstractMetric` in isotropic
coordinates, generic over dual numbers so TreeGH can differentiate it
inside kernels. `H_a = −Γ_a[g_TOV]` is sampled and frozen, as TreeGH
already does for static backgrounds, and the outer boundary is Dirichlet
from the static exterior. A fixed gauge source is a legitimate gauge for a
perturbed or oscillating star; it is not for a collapse, which needs the
gauge driver (GH-6).

## The right-hand side

In the order the explicit part of each stage evaluates it (proposed):

1. `scatter!` the multi-set state; `fill_ghosts!` every evolved set.
2. Metric to cells (into `P`); `B_d = curl A + 𝔅̄` on all stored faces;
   `B` averaged to cells.
3. `con2prim` over all stored cells, in hydrodynamic mode (the ghost
   exchange is in conserved variables, as TreeHydro decided).
4. `q = ∇·E` at cells.
5. `T_ab` at cells, averaged to vertices into `Tsrc`; TreeGH's right-hand
   side (if evolved).
6. Reconstruction and Riemann fluxes into `F_d` over the closed face
   ranges; the EMF into `EMF_d` over the closed edge ranges;
   `restrict_interfaces!` on both.
7. Divergence of `F_d` plus the explicit sources into `du` for the matter
   set; `−EMF − ∂(αΦ − β·A)` into `du` for `A_d`; the gauge equation into
   `du` for `Φ̃`.

The implicit stage solve then runs on owned cells between explicit
evaluations, and the atmosphere stage limiter after it. The same reset runs
once more on each step's result, as the step limiter. The right-hand side
never mutates `u`.

*As built in step 5* (`src/evolution.jl`, `mhd_rhs!`; the kernels are
written for `D = 1, 2, 3` and step 5 validates `D = 1`):

- Steps 2 and 3 are one kernel over the stored cells (`primitive_kernel!`):
  the provider's metric at the cell, `B` from the faces, `E = 𝔈/√γ`, the
  recovery from `state_guesses` (see "Thermodynamics"), all thirty slots of
  `P`. `q` is a second kernel over the owned cells, by centered differences
  of the ghost-filled `𝔈`, `q = (1/√γ) Σ_d (𝔈ᵈ[n+e_d] − 𝔈ᵈ[n−e_d])/2h`
  (proposed in step 5).
- Step 7 for `A_j` is `−EMF_j − (ψ[n + e_j] − ψ[n])/h` with
  `ψ = αΦ − βᵏA_k` evaluated at the two vertices that bound the edge, `A_k`
  averaged to a vertex from the two `k`-edges that meet there; along a
  missing `j`, `−EMF_j` alone. For `Φ̃` it is the difference of the gauge
  flux `α√γ Aʲ − βʲΦ̃` on the two `j`-edges either side of the vertex, with
  `A_k`, `k ≠ j`, averaged to the `j`-edge (two or four points) and `Φ̃`
  from its two vertices, plus `−ξαΦ̃` (proposed in step 5: second-order
  averages; in flat space `ψ = Φ̃` and the flux is `A_j`, and nothing is
  averaged that enters).
- The implicit stage (`ohm_solve!`) computes `B` from `u★`'s `A` — a scatter
  and ghost fill of the three `A` sets and the curl — and calls
  `solve_ohm_stage` in every owned cell with that `B`, the provider's
  metric and `state_guesses(u★)`, writing `√γ E` into `𝔈` and nothing else
  and tallying the outcome into `X`.
- Cost (measured in step 5, Julia 1.13, Apple M3 Pro, one thread, the
  Alfvén wave on 400 cells in 40 blocks): 0.77 ms for a right-hand side, of
  which the recovery over the 640 stored cells is 0.55 ms; 0.39 ms for an
  implicit stage (`IdealConductor`) and 0.34 ms for a reset, which also
  recovers every cell. A step of SSP3(4,3,3) — 3 right-hand sides, 4
  stages, 4 resets — is 5.2 ms, a quarter of it the resets.
- In 2D and 3D, a run of either wave that is translation-invariant in `y`
  (and `z`) agreed with the 1D run to 2.7e−15 after seven steps, in a check
  made while writing the kernels and not in the suite — compiling the 2D and
  3D paths costs about 30 s, and their validation, bit for bit, is step
  7's.

## Conservation at coarse-fine faces

No subcycling, so conservation is a spatial fixup inside each
right-hand-side evaluation, exactly as in TreeHydro: `restrict_interfaces!`
on `F_d` makes each coarse face's flux the average of its fine faces', so
`D` (and the other nine cell variables) are conserved across levels to
round-off. `restrict_interfaces!` on `EMF_d` is the constrained-transport
EMF correction: a coarse edge on a coarse-fine face receives the average
of its fine edges' EMFs, so the coarse `A` there stays consistent with the
fine `A`, and the magnetic flux through a coarse face equals the sum
through its fine faces. `∇·B = 0` does not depend on this — it holds on
each level by construction — but flux conservation across levels does.

## Boundaries

Dirichlet from the initial or analytic state, through
`CellBoundary(AllVariables(…))` on every set, for everything up to the
magnetized TOV star (proposed): shock tubes until a wave arrives, the
Kerr-Schild tests from the analytic solution, the stars from the static
exterior (atmosphere, the initial `A` and `Φ̃ = 0`, `E` of the static
dipole). Outflow conditions were waiting for AMR-6; TreeAMR 0.2.1 has
outflow faces (amended 2026-10-14), for whichever later rung needs them.
*(Step 5:)* `MHDProblem` takes `boundary = nothing` for a periodic domain
(and refuses a non-periodic forest without hooks) or one hook per evolved
set, in the order `(U, A₁, A₂, A₃, Φ̃)` — AMR-2 having landed, each set's
`fill_ghosts!` gets its own. Step 5's waves are periodic; step 6's shock
tubes are the first to pass hooks.

*As built in step 6* (`DirichletBoundary`, `case_boundary` in
`src/driver.jl`; proposed in step 6). A case that is not periodic gets one
hook per evolved set from its own `fields` and `potential`, evaluated at the
time of each fill (`boundary_at(hook, t)`: every fill — a right-hand side's,
a stage's, a reset's — knows its time):
- **The matter**: Dirichlet, the case's conserved state at each ghost
  cell's center, `CellBoundary(AllVariables(…))`, a kernel on every backend.
- **The potentials**: Dirichlet *in the field*. A ghost of `A_j` or `Φ̃`
  holds the nearest evolved point's value plus the case's difference between
  the two points, so `𝔅` in the ghosts and on the boundary faces is the
  case's, and the level of `A`, which nothing physical reads, is the
  evolution's. A Dirichlet `A` itself is wrong: a uniform state with an
  electric field has `A = A₀ − tE`, and the stage solve is told
  `tⁿ + cΔt` while the stage's `A`, which only the explicit part moves, is
  at `tⁿ + c̃Δt` — SSP3(4,3,3)'s first stage has `c̃ = 0`, `c = 0.24`. A
  uniform flow at `vˣ = 0.999` lost a fifth of its density in the last
  cell within 0.05 of a crossing that way, and the step limiter wrote that
  cell every step (measured in step 6); with the field's Dirichlet data it
  stays uniform to 3.3e−13 under either tableau (a test). This hook reads
  the interior, TreeAMR's region form, CPU-only: AMR-7 in `PLAN.md`.

## Refinement

A Löhner indicator on `ρ`, optionally on `|B|` and `|E|`, following
TreeHydro's `hydro_flags`, plus geometric radius boxes around a star that
fix the levels the star is resolved with (proposed). `regrid!` is given
every set over the forest: the evolved ones with their schedules, the rest
`=> nothing`.

**A design point for steps 8 and 10** *(recorded 2026-10-14; nothing is
redesigned yet)*. TreeAMR's bounded prolongation (commit 902042a, AMR-4)
was shaped by an expectation its commit message states of this package:
that TreeGRRMHD "will leave admissibility to EntropyEOS, cell by cell,
prolongate the primitive variables in the ghost fill and the conserved ones
only in the regrid". The mesh guarantees only that a prolongated value
stays within its neighbours' range, per variable — a property that suits
primitives in a ghost fill (`ρ`, `s`, `vⁱ` each stay bounded, so a ghost is
an admissible state wherever its neighbours are) and conservation in a
regrid (the conserved variables' totals are kept). Steps 8 and 10 must
decide whether to follow that split — which set the ghost fill acts on, how
`P` is refilled after a fill, and what the reset limiter does to a
prolongated conserved state — and record it here.

## Julia version

`julia = "1.11"` (decided 2026-09-25). *(Amended: this section used to
say `julia = "1.10"` wherever possible.)* All the Tree* packages — TreeAMR,
TreeWave, TreeHydro, TreeGeneralizedHarmonic and this one — share the 1.11
floor, so that an unregistered dependency can be located with a
`[sources]` entry, a 1.11 key, rather than added by URL. That also removes
the reason for the weak-dependency split proposed earlier: TreeGH's pin to
1.11 (its `LinearAlgebra = "1.11"` compat entry, its `[sources]` entries,
and ApparentHorizonFinder and KorzynskiSpin, whose checkouts both declare
1.11) is now this package's floor too, so TreeGH can be an ordinary
dependency and `TreeGRRMHDTreeGHExt` is not needed for the version's sake.
A dependency's own `[sources]` is still ignored when this package resolves
it, so what it needs from an unregistered package must be named in this
package's own `[sources]`.

`Project.toml` names EntropyEOS and IMEXRungeKutta there, by their GitHub
URLs at `rev = "main"`, and CI adds nothing by hand: `julia-buildpkg`,
a clean `git archive` and a developer's `Pkg.instantiate()` all resolve
them the same way (amended in step 0; the plan had CI run
`Pkg.add(url = …)`). The test environment reaches them through the
package (`TreeGRRMHD.EntropyEOS`, `TreeGRRMHD.IRK`) rather than listing
them again, so that a second `[sources]` cannot disagree with the first.

## Precision and devices

*(Proposed in step 0.)* Every driver is generic in the element type `T`,
a leading positional argument defaulting to `Float64`, and in the
KernelAbstractions backend, a keyword defaulting to `CPU()`. Physics is
validated in `Float64`; `Float32` is the device smoke test and the leak
detector (a stray `Float64` constant widens a result), since EntropyEOS is
validated in `Float64` only. `src/precision.jl` carries TreeHydro's bridges
over `Base`'s gaps at a software float (`wrap`, `ceilint`, `floorint`,
`roundint`, `tofloat64`) for host-side control flow; they are tested at
`Float32x2` too, which is a claim about them and not about the physics.
`src/device.jl` carries TreeHydro's and TreeGH's `to_backend` (per-block
metadata a kernel reads, uploaded once) and `hostcopy`/`hostcopy!` (a field
set brought to the host for a viewer; on the CPU `hostcopy` returns the set
itself). Diagnostics never copy the state to the host: they reduce where
the data are.

A SpacetimeMetrics metric is an `isbits` kernel argument: inside a
`map_blocks!` kernel it gives the ADM split, `K_ij` and a ForwardDiff
derivative bit-identical to the host at `Float64` and `Float32` (measured
in step 0, Kerr-Schild at `a = M/2`). That is the premise of the `Analytic`
provider.

The pointwise layer of step 1 — `Metric3`, `prim2con`, `con2prim_hydro`,
`con2prim_ideal` — runs inside a KernelAbstractions kernel bit-identically
to the host at `Float64` and `Float32`, allocation-free and inferred, and
so does `GeometricEOS` around a table at `Float64` (measured in step 1).
So does step 2's layer — `Analytic(KerrSchild)` with its ForwardDiff
derivatives, the stress-energy, both Riemann solvers in every direction,
Ohm's law, both source forms, the UCT EMF and the gauge terms — at `Float64`
and `Float32` (measured in step 2), and step 3's stage solve on every
branch (measured in step 3). Step 2 needed one change: StaticArrays'
matrix product accumulates with `muladd`, which LLVM fused differently in
the kernel and on the host, so three of the 79 numbers differed in the last
bit; `lower`, `raise`, `dot_up` and `dot_down` now multiply with plain `*`
and `+` (`mul3` in `src/metric3.jl`), which is never fused (amended in step
2). On a device, which contracts by its own rules, bit-identity is not the
claim; step 14 states what is. `GeometricEOS` around a table is not
`isbits` (the table holds arrays) and is passed to a CPU kernel as it is;
it carries an `Adapt` rule for a device.

*(Step 5:)* every kernel of the scheme and of the driver's initial data
takes its geometry as `to_backend` arrays of the block origins and
spacings and its physics as `isbits` arguments (the EOS, the options, the
model, the provider, a case's callable `WaveFields`/`WavePotential`), and is
generic in `T`: the Alfvén wave runs in `Float32` throughout — the matter
and field kernels, the stage solve and the reset — and lands within 4e−5 of
the `Float64` run's error (rung 1a). The diagnostics (`l1_errors`,
`conserved_totals`, `ohm_tally`, `max_light_rate`) are `mesh_mapreduce`s,
where the data are.

## Validation ladder

These become the milestones of `PLAN.md`; each rung is a set of testsets
named as claims.

1. **Flat space, 1D**: the vacuum electromagnetic wave (`Vacuum`) at the
   scheme's order; the large-amplitude circularly polarized Alfvén wave
   (`IdealConductor`) at the scheme's order; the resistive shock tube swept
   over σ (Palenzuela et al. 2009, Dionysopoulou et al. 2013 — amended in
   step 6: Komissarov 2007 has no σ sweep, and the published tube is a
   Brio–Wu tube with `Bˣ = 0`, not Balsara 1), approaching the ideal
   solution as σ → ∞; the
   self-similar current sheet against its analytic profile; Balsara 1–5
   with `IdealConductor`, against GRaM-X's figures; a σ × magnetization
   sweep of the implicit stage solve, recording where each branch of it
   is taken. *(Step 3 did the sweep pointwise, on stage problems with a
   known solution — "The implicit stage solve"; rung 1 records the branches
   an evolution takes.)*

   *Rung 1a, measured in step 5* (`test/wave_tests.jl`; MC, HLLE,
   SSP3(4,3,3) at `cfl = 2/5`, one period on `[0, 1]` periodic at N = 50,
   100, 200, 400 cells, exact cell averages for the reference; L1 rates by
   least squares over the four; Julia 1.11 and 1.13, one and four threads,
   identical):
   - **The vacuum EM wave** (`Vacuum`; `a = 1/10`, `b = 1/20` over a resting
     Γ = 5/3 gas at `ρ = p = 1`): `𝔈ʸ`, `𝔈ᶻ` at 1.938 (L1 9.1e−4 to
     1.6e−5), `A_y`, `A_z` at 1.912, `S_x` and `τ` at 1.914, `D`, `D Yₑ`,
     `D s` at 2.00–2.02 — PLAN.md's prediction, ≥ 1.9, holds. `S_y`, `S_z`,
     `A_x` and `Φ̃` stay zero exactly. Every stage is `OHM_SIGMA_ZERO`.
   - **The circularly polarized Alfvén wave** (`IdealConductor`; Del Zanna
     et al.'s `ρ = p = B₀ = η = 1`, Γ = 4/3, `v_A = 0.3820`): `S_y`, `S_z`
     at 2.089, `A_y`, `A_z` at 2.10, the ideal projection of `𝔈` at 2.09,
     `D`, `S_x`, `τ`, `D Yₑ`, `D s` at 2.78–2.85; the stored `𝔈` at 0.85
     (see "The partition") — the prediction, ≥ 1.8, holds for all but the
     stored `𝔈`. Every stage is `OHM_IDEAL`, one recovery each.
   - **Conservation**: `√γ D`, `√γ S_i`, `√γ τ`, `√γ D Yₑ` and `√γ D s`
     drift at most 0.016 ulp of their own scale per step (asserted ≤ 8).
     No stage was repaired and no reset fired in either wave.
   - **Float32**: the Alfvén wave runs in `Float32` throughout, and after a
     tenth of a period its `A_y` error is the `Float64` run's to 4e−5.

   *Rung 1b, measured in step 6* (`test/shocktube_tests.jl`,
   `test/resistive_tests.jl`, `test/atmosphere_tests.jl`; MC, HLLE,
   SSP3(4,3,3) at `cfl = 2/5`, Dirichlet boundaries from each case's own
   data — "Boundaries"; Julia 1.11 and 1.13, one and four threads; the
   cases in `src/cases/shocktubes.jl`, every published number copied with
   its citation):
   - **Balsara 1–5** (`IdealConductor`; GRaM-X's Table 1, Giacomazzo &
     Rezzolla's Table 4; on `[−1/2, 1/2]` to `t = 0.4`, 0.55 for test 5) at
     100, 200 and 400 cells against a 3200-cell run of the same code
     (`bin/shocktube_reference.jl`, stored in `test/data/`, 99–285 s a test
     at four threads): L1 rates of `D`, `S_x`, `τ`, `𝔅ʸ` of 0.78, 0.85,
     0.77, 0.85 (test 1), 0.66, 0.67, 0.74, 0.81 (2), −0.02, 0.14, 0.17,
     0.53 (3), 0.95, 1.04, 0.94, 0.93 (4), 0.90, 0.75, 0.78, 0.97 (5).
     PLAN.md's prediction, ≥ 0.7, holds but for test 2's `D` and `S_x` —
     whose error is the contact's, and a contact converges as `h^{2/3}`
     under a second-order scheme — and test 3, which is not asymptotic below
     800 cells: its dense shell (`R5`, ρ = 10.1, between a contact moving
     at 0.95 and a slow shock) holds at most ρ = 8.35 at 400 cells and 9.67
     at 800, and its missing mass is the error of `D`, `S_x` and `τ` (`D`:
     0.37, 0.34, 0.38, 0.30, 0.15 at 100 … 1600 cells). The test asserts
     those measured rates with tolerances.
   - **The plateaus** against Giacomazzo & Rezzolla's exact solution
     (2006, Tables 7–11; `BALSARA_EXACT`, whose `p` is the total pressure),
     averaged at 400 cells over each region's window (`BALSARA_PLATEAUS`,
     where the 3200-cell run holds the region to 2%, less four cells either
     side), with `ρ` and `p` relative to their own value and each component
     of `v` and `B` to its largest magnitude in the solution
     (`plateau_error`, proposed in step 6): every window within 2% —
     the worst 1.26% (test 1, `R5`), 1.91% (2, `R5`), 0.13% (3, `R2`),
     0.64% (4, `R4`), 0.64% (5, `R2`) — but test 3's `R4`, 4.7% dense at
     400 cells (1.3% at 800), next to the unresolved shell. Too thin for a
     window at 400 cells: test 2's `R6`/`R7`, test 3's `R5`–`R7`, test 5's
     `R3` and `R6`; at 3200 cells all but test 3's `R6`/`R7` and test 5's
     `R3` are held, the worst to 2.0% (test 3's shell).
   - **Conservation with boundaries**: until a wave arrives, the totals
     change by the far states' physical fluxes, `−(F_R − F_L)t`, to at most
     4.1e−12 of their scale — the light-cone fluxes' exponentially small
     precursors reaching the faces at 100 cells; 3e−14 at 400 — and
     Balsara 1 on a periodic box conserves every total to 0.018 ulp a step.
     A uniform flow at `vˣ = ±0.999` or `±1/2` crosses the box unchanged to
     1.7e−12 of its scale under either tableau.
   - **The stages**: every one of Balsara 1–5 is `OHM_IDEAL`, one recovery;
     no repair, no reset, nothing non-finite.
   - **The resistive tube** (Palenzuela et al. 2009, Dionysopoulou et al.
     2013: a Brio–Wu tube with `Bʸ = ±1/2`, `Bˣ = 0`, `Γ = 2`, on `[0, 1]`
     to 0.4; 200 cells): the L1 distance of `𝔅ʸ` to the `IdealConductor`
     run is 0.392, 0.202, 0.0611, 0.0134 and 1.73e−5 at σ = 0, 10, 10²,
     10³, 10⁶ — monotone, PLAN.md's claim — and falls as `σ^−0.96` from 10³
     to 10⁶ (`σΔt ≈ 2000` there). The stages: all `OHM_SIGMA_ZERO` at σ = 0;
     all fixed point at σ = 10; 94,930 fixed point and 65,070 Newton of
     160,000 at 10²; 3,225 and 156,775 at 10³; 817 and 159,183 at 10⁶, every
     Newton stage there from the ideal seed; no fallback, at most 6
     recoveries in a stage, no repair, no reset.
   - **The current sheet** (Komissarov 2007, §6.1.4: `B₀ = 1`, `p = 50`,
     `ρ = 1`, `σ = 100`, `Γ = 4/3`, on `[−3/2, 3/2]` from the profile at
     `t = 1`, with the quasi-static `Eᶻ = ∂ₓBʸ/σ` where the paper starts
     from `E = 0`, proposed in step 6): PLAN.md predicted an L1 rate ≥ 1.8
     against the erf profile, but the profile solves the diffusion limit
     and not the system — the field's pressure moves the fluid at
     `2·10⁻⁴`, whose `v × B` is 1% of the resistive `E`, and the
     displacement current lags the diffusion by `O(1/σt)` — so the
     distance saturates at 1.1e−3 of `B₀` (L1 per unit length; 2.8e−3,
     8.2e−4, 1.06e−3, 1.14e−3 at 50 … 400 cells over `t = 1 → 2`). The
     order is therefore measured against the next finer run (amended in
     step 6): 2.53 and 2.58 over 50 … 400 cells. At Komissarov's 200 cells
     and `t = 9` the distance to the profile is 9.3e−4 — "indistinguishable"
     to 0.1% — and at `B₀ = 1/10`, where the fluid's response (quadratic in
     `B₀`) drops out, it is five times smaller relative to `B₀`. Every stage
     is a fixed point (1,068,800 of them to `t = 9`), at most 4 recoveries.
   - **The atmosphere** (`atmosphere_tube_case`, proposed in step 6: a gas
     at ρ = p = 1 expanding into the policy's own atmosphere at
     `ρ_atm = 10⁻⁸`, `W` up to 4.9; 200 cells to `t = 0.4`): hydrodynamic
     (`B = 0`), and magnetized as a star's surface would be — `Bʸ = 10⁻³`,
     `B²/ρ = 100` in the atmosphere, `Constant(10³)` cut off at
     `μ_max = 10` (the measured safe value), so the atmosphere is
     electrovacuum. No density below `ρ_atm`, nothing non-finite; the reset
     writes 35,158 cells as the stage limiter and 11,686 as the step
     limiter (hydrodynamic; 33,263 and 11,054 magnetized) — the policy
     flags every atmosphere cell, and the reset writes it back as it is —
     and injects `−8.4·10⁻¹²` of rest
     mass and `−2.2·10⁻¹⁰` of energy (`2.5·10⁻⁸` and `6.9·10⁻⁸`
     magnetized) into a box holding 0.5 of rest mass — and the totals
     change by exactly those amounts, to rounding, which also shows the
     stage limiter's writes stay out of the state. Magnetized, the stages
     are 48,362 `OHM_SIGMA_ZERO` (the cutoff), 111,572 fixed point and 66
     Newton (61 from the ideal seed); no fallback, at most 6 recoveries. A
     stage predictor next to the vacuum can carry `D < 0`, which EntropyEOS
     reports as non-finite and repairs inside the stage (the flag appears
     among the stages' policy flags, never in the state); its magnetization
     is `+∞` (see "Ohm's law and the conductivity").
   - Cost (Julia 1.13, Apple M3 Pro, one thread, 400 cells): a step of
     Balsara 1 is 6.5 ms, of Balsara 3 15 ms (the recoveries at a pressure
     ratio of 10⁴); the three test files take 45, 37 and 11 s.
2. **Flat space, 2D and 3D, with AMR**: the cylindrical explosion, the
   magnetic rotor, Orszag–Tang, field-loop advection across coarse-fine
   faces — `∇·B` at round-off, rest mass and magnetic flux conserved
   across levels to round-off, regrids included.
3. **Analytic curved backgrounds**: the Wald solution (`Vacuum`,
   Kerr-Schild) and the Michel monopole or magnetized Bondi flow
   (`IdealConductor`).
4. **TOV in the Cowling approximation**: a Γ = 2, K = 100 polytrope
   at fixed entropy (needs EOS-1, EOS-2); stationarity and mode frequencies.
5. **TOV in a dynamical spacetime**: the F, H1, H2 radial modes against
   the literature (Font et al. 2002); convergence of the GH and ADM
   constraints with matter (needs GH-3, GH-4).
6. **Magnetized TOV**: an interior poloidal field from an analytic `A`,
   σ(ρ) through `DensityProfile`, a vacuum dipole exterior (Palenzuela
   2013).
7. **TOV with a tabulated EOS and `Yₑ`** (needs EOS-4 for the initial
   data).

Later, not specified yet: rotating stars, collapse to a black hole,
binaries, supernovae with neutrinos.

## Possible extensions

Not planned; listed because they are the obvious next questions:
split-speed and HLLC fluxes; limited prolongation once upstream has it;
a mean-field dynamo in Ohm's law; force-free currents in the
magnetosphere; subgrid (LES) models for unresolved amplification;
neutrino leakage and M1 as a separate package on the same forest.

## Open questions

Each is decided by a measurement in the step named in `PLAN.md`:

- Riemann wave speeds: light cone for all variables (default), or split.
- `T_ab` to vertices: 2nd order (default) or 4th.
- Conservative prolongation width for the matter at stellar surfaces:
  width 1 or 3, until a limited operator exists.
- PointValue width for `A` and `Φ̃`; the gauge damping `ξ`.
- TreeGH's CFL factor under the SSP3 explicit tableau.
- The metric in the outermost ghost cell (see
  [Metric at cells, faces and edges](#metric-at-cells-faces-and-edges)).
- Whether `D s` earns its keep; the threshold that switches to it.
- Whether the atmosphere reset needs its stage-limiter role. Run rung 4's
  TOV star, a star in a vacuum, twice: with the reset as both limiters,
  and as the step limiter alone. TreeHydro named this case as the one
  where a per-stage reset could matter, and no code in IMEXRungeKutta's
  survey measured it (amended 2026-09-25).
- *(Closed in step 6.)* SSP3(4,3,3) or ARS(4,4,3) as the default tableau:
  SSP3(4,3,3), whose explicit part is SSP; ARS(4,4,3) undershoots into a
  near vacuum, fires the reset on Balsara 3 and 4, and floods a magnetized
  atmosphere (see "The partition").

## Known risks

- No MPI yet: single node, although TreeAMR now has its M7 (amended in
  step 0).
- Unlimited conservative prolongation at a stellar surface floors cells.
- Light-cone fluxes are diffusive in a star's interior. *(Measured in step
  6:)* they smear contacts and slow waves: Balsara 2's contact converges as
  `h^{2/3}`, and Balsara 3's dense shell (ρ = 10.1 between a contact and a
  slow shock) is not resolved below 800 cells, where the 400-cell run's
  peak density is 8.35.
- The implicit stage may converge poorly where magnetization and `σΔt`
  are both large; rung 1's sweep exists to find out where. *(Measured in
  step 3, pointwise:)* not where both are large — the ideal seed is exact
  there — but at `b²/ρ ≥ 10` with `σΔt` between `10⁻³` and `10²`, when `E★`
  is so far from both the solution and the ideal field that neither seed is
  a fluid state. *(Decided 2026-10-14:)* the
  [magnetization cutoff](#ohms-law-and-the-conductivity) takes σ to zero
  there, and with it (at the measured safe `μ_max = 10`) no fallback
  remains at `f ≤ 0.3`; the price is that above the taper the field
  evolves as in electrovacuum, not as force-free — no conduction current
  at all, where a magnetosphere's plasma would carry the force-free one —
  and the taper already begins at `μ = μ_max/10`, so σ is reduced a decade
  below the threshold a case states. Rung 1 sees how often an evolution
  meets the taper.
- `q := ∇·E` turns numerical divergence of `E` into charge.
