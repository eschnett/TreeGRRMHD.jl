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

*Status: steps 0 to 3 and 5 to 7 of `PLAN.md` are done — the scaffolding;
the pointwise layer: units, the variable layouts, the point metric,
`prim2con` and the 3+1 primitive recovery; the pointwise physics a
right-hand side evaluates; the implicit stage solve; the staggered scheme
on the mesh, with the vacuum electromagnetic wave and the circularly
polarized Alfvén wave in 1D (rung 1a); and the shock tubes, the resistive
tube swept over σ, the current sheet and the atmosphere in 1D (rung 1b);
and the scheme in 2D and 3D on a uniform mesh, with PPM, WENO-Z and MP5
beside PLM (rung 2a) — as are steps 4a–4c in IMEXRungeKutta; coarse-fine
faces and regridding (rung 2b, step 8); and the analytic curved backgrounds,
the Wald solution and the magnetized Bondi flow on a Kerr-Schild hole with
its interior (rung 3, step 9). Step 10, the vertex metric and the TOV star
in the Cowling approximation, is next. The numbers below about the scheme
are measured where they say so and unmeasured otherwise. (Amended in step
9.)*

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
the rule is in `PLAN.md`, "Upstream requirements"): AMR-1 to AMR-10,
EOS-1, EOS-2 and SM-1 (AMR-7 to AMR-10 since step 8, TreeAMR 0.2.2). Their names are in the prerequisite test's name
lists; what this package relies on from them and does not yet exercise
through its own code is in `test/upstream_behaviour_tests.jl` (empty, and
removed, since step 8 took the bounded prolongation into its refinement
tests; it returns when a landed item needs it).

- TreeAMR **0.2.2**, registered, and the compat floor (`TreeAMR = "0.2.2"`,
  amended in step 8; 0.2.3, which only raises its floor to Julia 1.13, is
  released upstream and admitted by it) so that what landed is guaranteed:
  **AMR-1, AMR-2, AMR-3 and AMR-5 have landed in TreeAMR 0.2.1**, AMR-4 and
  AMR-6 in a different form (below), and **AMR-7 to AMR-10 in TreeAMR
  0.2.2** (commit 624c99a). Step 5, which needed AMR-1, is done on it
  (amended in step 5).
- EntropyEOS **1.0.0** on its `main`, not registered (EOS-0 waiting):
  **EOS-1 and EOS-2 have landed in EntropyEOS 1.0.0** (`EntropyEOS =
  "1"`); EOS-3 and EOS-4 have not.
- IMEXRungeKutta **1.4.0**, tagged, not registered: the plan's steps 4a–4c
  are done there, and it is a dependency from step 0.
- SpacetimeMetrics **1.7.0**, registered (`SpacetimeMetrics = "1.7"`; it
  requires Julia 1.11, below the floor here, 1.13 — amended 2026-10-15):
  **SM-1 has landed in SpacetimeMetrics 1.7.0**, and its workaround is
  gone.
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
  the regrid transfer alike. `test/refinement_tests.jl` checks that a
  surface refined everywhere stays positive and within range and keeps its
  total, where the unbounded operator overshoots (moved there from
  `test/upstream_behaviour_tests.jl` in step 8), and the matter's ghost
  fill and regrid use it since step 8 — the ghost fill on the primitives
  `W` since 2026-10-23 (decided by Erik). See [Refinement](#refinement)
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
- **AMR-7 (later, step 14) A device-capable boundary hook relative to the
  nearest evolved point** *(added in step 6)*, for the potentials'
  Dirichlet data in the field ("Boundaries"). *(Landed in TreeAMR 0.2.2 as
  `CellBoundary(Relative(f))`, which also applies the complete condition in
  every fill; the region-form workaround and `outer_points!` are gone since
  step 8.)*
- **AMR-8 (wanted, no step blocked) Symmetric walls** *(added in step 8)*:
  an opt-in per field set under which `fill_ghosts!` fills the owned low
  wall of a vertex-like dimension by the rule of the high one — the hook,
  the reflecting derivation, the outflow clamp — so that a closed box is
  mirror-symmetric. Workaround: `low_walls!` and `outer_points!`, this
  package's kernels ("Boundaries"). *(Landed in TreeAMR 0.2.2 as
  `FieldSet(…; walls = :symmetric)`; the workaround is gone since step 8.)*
- **AMR-9 (wanted, no step blocked) A flux-conservative transfer of edge
  sets** *(added in step 8)*: conservative along an edge's own (cell-like)
  direction, point-value along the others, so that the magnetic flux
  through a coarse face is the sum through its fine faces in the state,
  not only in its rate ("Conservation at coarse-fine faces"). No
  workaround: until then the state's flux across levels holds to the
  potentials' interpolation error. *(Landed in TreeAMR 0.2.2 as
  `Operators(…; family = Conservative, vertex_prolongation = w)`; the
  potentials take it since step 8.)*
- **AMR-10 (req: step 8's ghost-fill re-measurement; steps 10, 12)
  Bounded PointValue interpolation** *(added in step 8; decided by Erik
  2026-10-15)*: `Operators(; prolongation=w, restriction=r,
  family=PointValue, bounded=true)` for even `w ≥ 2` on every centering, in
  the ghost fill and the regrid — every value within the range of the `2^D`
  points that enclose it — for the matter's ghost fill of the primitives
  ("Refinement"). No workaround: the conserved, bounded ghost fill runs
  until it lands. *(Landed in TreeAMR 0.2.2, `bounded = true` with
  `PointValue` at every even width; the matter's ghost fill took the
  primitives with it in step 8, at width 2. Amended 2026-10-23: the fill
  now takes the bounded conservative operator, AMR-4, centred on the
  parent — decided by Erik; "Refinement". The PointValue fill stays for
  measurements and as the width-4 control of the suite.)*
- **AMR-11 (wanted, no step blocked) The relative hook consistent across
  levels where an outer face meets a coarse-fine face** *(added in step
  8)*: there each level's hook writes its own wall edges, so the magnetic
  flux through a coarse face touching an outer face is its fine faces' sum
  only to 1.2e−5 of the largest flux, where everywhere else it is to
  rounding ("Conservation at coarse-fine faces"). No workaround needed.
- *(later)* The use of MPI (M7) and of checkpoints (M9, now TreeIOHDF5's)
  and point interpolation (M11), all three of which TreeAMR has; none is
  an upstream item any more. *(Amended in step 0.)*

This is a **physics package** in TreeAMR's sense (its `CODE.md`,
"Companion interface", 2026-10-08): it uses TreeAMR's exports only, and
bounds TreeAMR by a minor series, `TreeAMR = "0.2"` (since 2026-10-14
`"0.2.1"`, since step 8 `"0.2.2"`, the releases the landed items are
in). The prerequisite
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
- **EOS-8, EOS-9 (landed)**: the density floor reported (EntropyEOS
  1.1.0), and the endpoint branches made edge-consistent, the hottest state
  projected onto the entropy ceiling (2.0.0, as the two-level projection;
  tag `v2.0.0` through `[sources]`, `[compat]` `"2"`). See "Primitive
  recovery", "The two levels" *(amended 2026-10-24, EntropyEOS 2.0)*.
  *(Amended 2026-10-26, EntropyEOS 2.2:)* the tag is now `v2.2.0`,
  `[compat]` `"2.2"`.
- **EOS-11 (landed)**: `con2prim_safe`'s validity test of the `s` and `Yₑ`
  faces, with its slack, public for the reconstruction — `inside_window`
  (EntropyEOS 2.1.0; decided by Erik) *(added 2026-10-26)*. See
  "Reconstruction" and "Floors, the atmosphere and the magnetosphere",
  "EntropyEOS 2.0".
- **EOS-10 (landed)**: the nested fallback
  allocation-free under `--check-bounds=yes` (PLAN.md), where it allocated
  one `MVector{33,T}` (272 B at `Float64`, 144 B at `Float32`) *(added
  2026-10-24)*. *(Decided by Erik 2026-10-24:)* it stays an upstream
  request; the bounds-checked run's four allocation failures are known and
  wait on it. *(Re-checked 2026-10-26:)* EntropyEOS 2.2.0 allocates the
  same 272 B and 144 B there; still waiting. *(Landed 2026-10-30:)*
  EntropyEOS 2.2.1 keeps the scan's scratch on the stack under bounds
  checks, bit for bit; `[sources]` pins `v2.2.1`.

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
  carry it the way it carries `D s`. *(Amended 2026-10-24, decided by Erik
  2026-10-23:)* the right-hand side is pure in `(u, t)` **outside a step**;
  **a step is pure in `uⁿ`** (and `t`, `Δt`): inside a step of `evolve!`
  every recovery is seeded from the cell's latest recovery in that step,
  and a recovery whose input is that one's is reused — [the step
  cache](#primitive-recovery), invalidated at the end of every step, so a
  restart from `uⁿ`, which holds no cache, still reproduces the run.
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
1). *(Amended 2026-10-21, decided by Erik:)* one exception, the one decision
this package adds to the policy: where the flags say the policy raised the
density to the atmosphere's without resetting the state
(`FLAG_POL_ρ_FLOORED`, EntropyEOS 1.1, EOS-8), both recoveries return the
atmosphere at rest, flagged `FLAG_POL_ATMOSPHERE | FLAG_POL_ρ_FLOORED` — the
replacement and its cause (amended 2026-10-22, decided by Erik; until then
`FLAG_POL_ATMOSPHERE` added to the discarded state's flags) ("Floors, the
atmosphere and the magnetosphere"). EntropyEOS's conserved output is never used: every
caller rebuilds the conserved state from the primitive one with this
package's `prim2con`. *(Amended 2026-10-24, EntropyEOS 2.0:)* the exception
is now EntropyEOS's own level 2, below, and the flags of an atmosphere keep
level 1's bits.

**The two levels** *(amended 2026-10-24, EntropyEOS 2.0; tag `v2.0.0`,
7d68ddb, through `[sources]`, `[compat]` `"2"`)*. EntropyEOS 2.0 splits its
policy in two, and both recoveries call both, the second right after the
first, in `fluid_recovery` and `ideal_recovery` *(amended 2026-10-30, stage 4
of the recovery plan: outside the stage solve's iterations level 1 runs as
EntropyEOS's two phases and level 2 in a kernel of its own, "The two-phase
recovery" below)*:

- **Level 1, `con2prim_safe(eos, in, opts, proj, s, w, u)`**, projects onto
  the valid range `proj = ProjectionOptions(ρ_min, ρ_max, w_cap)`:
  `ρ ∈ [ρ_min, ρ_max]`, `w ≤ w_cap`, `Yₑ` in the table and `s` in the
  window. It has no atmosphere and no collapse. Each face fixes one
  primitive, frees one conservative and keeps the rest, and sets its own
  bit: `S_CEILED` (τ lowered), `S_FLOORED` (τ raised), `W_CEILED` (S
  lowered), `ρ_CEILED` (D lowered), `ρ_FLOORED` (S lowered to
  `w = acosh(D/ρ_min)`, D raised only where it was below `ρ_min`),
  `YE_FLOORED`/`YE_CEILED` (`D_Y`). `NONFINITE` (no closest state) and
  `NOT_CONVERGED` (the solve failed, but a valid state lies inside) return a
  fixed state that level 2 replaces. Which faces are active is decided by
  one diagnosis along the entropy edges, not by the solve's root, so that a
  converged and a failed solve of neighbouring states project alike.
- **Level 2, `apply_policy(eos, in, out, proj, pol)`**, with
  `pol = PolicyOptions(s_atm, ye_atm, atm_trigger, collapse_to_atmosphere)`,
  replaces the state by the atmosphere at rest at `ρ_min` on `ρ_FLOORED`,
  `NONFINITE` or `NOT_CONVERGED`, or where `D < atm_trigger·ρ_min` (1.01),
  and adds `FLAG_POL_ATMOSPHERE` to level 1's bits. This *is* this package's
  rule of 2026-10-21 — a state whose density had to be raised to the floor
  is the atmosphere at rest — so the wrapper's own `atmosphere_at_rest` and
  `floored` are gone, and `collapse_to_atmosphere` stays off. It holds
  `D_out ≤ max(D_in, D_atm)` by construction: of level 1's faces only
  `ρ_min` raises `D`, and that face goes to the atmosphere, whose
  `D = ρ_min = D_atm` (`test/recovery_tests.jl`, "A state the projection put
  on the ρ_min face is the atmosphere at rest"). One difference from
  2026-10-22's decision: the atmosphere's flags are level 1's bits plus
  `ATMOSPHERE`, not `ATMOSPHERE | ρ_FLOORED` alone (EOS-9's Q4, rejected
  upstream by Erik: the bits say why); every tally that reads them
  (`X_POLICY`, the reset's `FLAG_POL_ANY`) is unchanged by that.
- **The options.** `MHDProblem`, `MHDCase` and every recovery carry `proj`
  and `pol` (`isbits`, two kernel arguments, `proj` first). 1.x's
  `ρ_atm` is `ρ_min`, its `ρ_ceiling` `ρ_max`, and `w_cap` moved into
  `proj`; `default_policy`, `policy_derive_bounds`, `D_max` and `τ_max` are
  gone (`τ_max` stopped states the table represents, EOS-9), and with them
  `case_policy`. `recovery_options(eos, ρ_min)` states both for every case:
  `default_projection(eos, ρ_min)` (`ρ_max` the top of the density box,
  `W ≤ 100`) and the atmosphere at `s_atm` (below). `atmosphere_prim` gives
  initial data the state `apply_policy` writes, bit for bit (a test).
- **The flags** are renumbered to bits 8–17: `ATMOSPHERE` 8, `NONFINITE` 9,
  `NOT_CONVERGED` 10, `ρ_FLOORED` 11, `ρ_CEILED` 12, `YE_FLOORED` 13,
  `YE_CEILED` 14, `S_FLOORED` 15, `S_CEILED` 16, `W_CEILED` 17 (1.x's
  `W_CAPPED`); `CEILING`, `ρ_CLAMPED` and `YE_CLAMPED` are gone;
  `FLAG_POL_ANY = 0x3ff00`, still exact as a number in `Float32` (below
  2¹⁸ < 2²⁴; tested through `P` and the diagnostics). *(Amended
  2026-10-26, EntropyEOS 2.2:)* bits 18 and 19, `FLAG_CS2_ACAUSAL` and
  `FLAG_CS2_NONPOSITIVE`, say that the returned `cs²` lies outside `(0, 1)`.
  They are the EOS evaluation's, like bits 0–5, set in `EOSPoint.flags` by
  every evaluation (`cs²_flags`), never in `policy_flags`, and `FLAG_POL_ANY`
  is unchanged. In this package they reach only the step cache's copy of the
  `EOSPoint` (below 2²⁰, exact in `Float32`; read back in a test), which
  nothing reads but a reused recovery's caller; `P`, `X` and every tally hold
  the policy's bits only. No case's ideal gas sets them (`cs² < Γ − 1 ≤ 1`
  and positive wherever `p > 0`), and the scheme's signal speeds are the
  light cone's, so `signal_cs²` has nothing to clamp here.

What moves against 1.1, and why (EntropyEOS's `HISTORY.md`, 2026-10-23):

- **The face semantics.** 1.1 clamped primitives and regenerated every
  conservative; 2.0 keeps each conservative a face does not free. `w_cap`
  keeps τ (1.1 kept `s`: the capped cell now heats, bounded by the `s_max`
  face), `ρ_max` keeps `S` and τ (1.1 kept `w` and `s`), the `s` faces keep
  `S` (1.1 kept `w`). A failed solve with `D ≥ ρ_min` is slowed onto the
  `ρ_min` face rather than given mass (and so, here, the atmosphere), and an
  energy above the hottest state keeps `D` and `S` and lowers τ
  (`S_CEILED`, EOS-9) where 1.1 collapsed it to `ρ_ceiling`.
- **`seed_passes` 5** (was 3), the cold seed's passes, with a direct seed
  for tables: every cold recovery starts from a different (better) seed, so
  every result at the default tolerance moves by up to that tolerance, and
  the solver's iteration counts move with it; with `tol = 10⁻¹⁴` they do
  not (measured below).
- **`s_atm`.** 2.0's default atmosphere entropy is the window's minimum at
  `ρ_min`; 1.x's was the window's midpoint (20.5 for the shock gas's 1 … 40).
  *Kept at the midpoint, explicitly* (`recovery_options`, `s_atm` = the
  middle of the window at `ρ_min` and the middle of the `Yₑ` range — 1.x's
  value for an analytic EOS, whose window depends on neither). Measured
  both ways on Symmetry (2026-10-24; the explicit midpoint against 2.0's
  default, the window's minimum, 1 for the shock gas): with the midpoint
  every atmosphere-tube number of the suite is 1.1's to the digits printed
  (the cut-off tube's 12010 step writes, `W` 4.795, `−1.1754·10⁻⁸`,
  `−7.402·10⁻⁷`; the hydrodynamic tube's 11686 writes and `−8.3518·10⁻¹²`)
  and the suite passes; with the minimum the initial atmosphere is colder
  (the hydrodynamic tube `−8.3662·10⁻¹²`), the ideal magnetized
  atmosphere under ARS injects `3.5·10⁻⁶` (the assertion `≤ 10⁻⁶` fails)
  and the width-4 negative control of the coarse-fine atmosphere tube no
  longer fails (`τ` `−2.4·10⁻¹⁰`, not `≥ 10⁻⁵`, the 2026-10-17 sweep's
  variant C). On the 48 perturbed runs neither is better across the board:
  the minimum makes the uniform cut-off tube deterministic again (two runs
  `10⁻¹⁵` apart `1.7·10⁻¹³` apart, against `1.4·10⁻⁴` with the midpoint),
  every other row is the same within the spread ("Floors, the atmosphere
  and the magnetosphere"). The midpoint is kept because it changes no
  number of the suite. *(Decided by Erik 2026-10-24:)* `s_atm` stays the
  explicit 1.x midpoint (`recovery_options`).
- **Why EOS-9 needs no `collapse_to_atmosphere`.** The collapse branch was
  1.1's answer to "τ above the hottest state": `ρ_ceiling` at `s_max`, the
  blow-ups of the magnetized tube, which the option turned into the
  atmosphere. In 2.0 that state is projected onto the `s_max` face with `D`
  and `S` kept (`S_CEILED`); where keeping `D` at the momentum puts `ρ`
  below `ρ_min`, the diagnosis takes the `ρ_min` face instead, and level 2
  writes the atmosphere at rest — the decision 2026-10-22's experiment
  emulated (`claude/no-collapse-to-atmosphere`), now upstream. Nothing
  reaches a ceiling the table cannot hold, so the option has nothing to do:
  the atmosphere tube drops it (measured below and in "Floors, the
  atmosphere and the magnetosphere").

What moves in the suite (Julia 1.13, one thread, Symmetry cn112, against
the stage-1 branch at d94bcf5 in the same job; every other printed number
agrees to the solver's tolerance or better):

- **At the solver's tolerance**, everywhere a cold recovery runs: the
  round-trip and convergence diagnostics (`con2prim_hydro ∘ prim2con`
  conditioned errors within 2× either way, the default-tolerance conserved
  round trip 1.07·10⁻¹², 1.21·10⁻¹², 8.5·10⁻¹³ against 9.8·10⁻¹³,
  6.8·10⁻¹³, 6.8·10⁻¹³ — `seed_passes` 5), the Alfvén, oblique, Balsara,
  resistive and current-sheet L1 errors in their 12th digit or beyond, the
  drift tallies (`0.0046` → `0.0085` ulp a step and the like), the
  `Float32` smoke run (3.6·10⁻⁵ → 2.3·10⁻⁵).
- **The face rules**: the Jacobian test's central difference crossed the
  entropy floor on one draw of 200 (0.17), now stepped off the face
  (`test/stage_solve_tests.jl`); step 3's sweep (above).
- **The flags**: the atmosphere tubes' `|` of policy flags reads 2304
  (`ATMOSPHERE | ρ_FLOORED`, bits 8 and 11) and 100608 (and bits 15, 16)
  where 1.1 printed 256 and 66816; the tallies are the same.
- **ARS(4,4,3) near the vacuum** (see "Floors, the atmosphere and the
  magnetosphere"): the hydrodynamic rarefaction's resets 131/110 →
  133/116; the ideal magnetized atmosphere `5.6·10⁻⁸` → `−2.0·10⁻⁹` of
  rest mass, `W` 4.11 → 4.13 (SSP3: `−6.6·10⁻⁹` → `−7.8·10⁻⁹`, `W` 4.01 →
  3.99); the coarse-fine magnetized tube `−1.4012·10⁻⁸` → `−1.4027·10⁻⁸`.
- **The bounds-checked run**: EOS-10 (PLAN.md).

**EntropyEOS 2.1 and 2.2** *(amended 2026-10-26; tag `v2.2.0`, f26afca,
through `[sources]`, `[compat]` `"2.2"`)*. Three changes reach this package:

- **`inside_window`** (2.1.0, EOS-11): `con2prim_safe`'s own test of the `s`
  and `Yₑ` faces, with its slack, which the reconstruction now calls
  ("Reconstruction"). It answers point 1 of the 2026-10-24 discussion — a
  root within the slack is returned unflagged and as it is — in the form
  Erik chose over exporting the slack or clamping accepted roots.
- **The search confined to the density box** (2.2.0; HISTORY, 2026-10-24):
  the seed, the Newton and the fallback stay at `w ≥ acosh(D/ρ_max)`, with
  `ρ_max` widened by the face slack. Every state with `D ≤ ρ_max` comes back
  bit for bit; no state of this package's cases comes near `ρ_max` (`10⁴` and
  `10⁶` for the cases' gases), and no printed number of the suite moved by it
  (measured below).
- **The sound-speed flags** (2.2.0), bits 18 and 19 of the evaluation's flags
  ("The two levels", "The flags").

The C++ 1.2.0 table-building port of 2.2.0 (`crop_table`, `SeamDefect`, the
`T_HI_ROWS_DROPPED` attribute, the u-high ratio lock) touches only tables
built or read from files; every case here runs an analytic ideal gas, and the
tests' synthetic table is built in memory (`make_synthetic_table`), so none of
it applies.

*Measured 2026-10-26* (Symmetry, AMD EPYC 7532 throughout, Julia 1.13;
`claude/entropyeos-2.2` against stage 3 at 4929c94 on EntropyEOS 2.0.0, the
stage-3 one-thread suite in the same job on the same node, cn079). The suite's
output is stage 3's to every digit but for:

- **The reconstruction's admissibility** (the slack band, "Reconstruction"),
  which moves only ARS(4,4,3) near the vacuum, chaotic at rounding until now:
  the ideal magnetized atmosphere under ARS `2.81·10⁻⁹` → `−5.00·10⁻⁹` of rest
  mass (energy `−2.354·10⁻⁵` → `−2.434·10⁻⁵`, `W` 4.015 → 4.002; SSP3's row is
  unchanged), the ARS rarefaction's smallest density `2.1433251·10⁻¹⁰` →
  `2.1433185·10⁻¹⁰` (resets 139/113, unchanged), and the coarse-fine
  atmosphere tube's unbounded width-4 negative control, a run that goes
  non-finite, 72 → 71 ghosts below `ρ_atm`. Step 6's tableau comparison
  (`bin/tableau_comparison.jl`, both trees in one job on cn085) moves in the
  same rows and in Balsara 3 under ARS, whose undershoot the reset fills (L1
  `D`, `τ`, `𝔅ʸ` 0.369, 18.3, 0.0572 → 0.366, 18.2, 0.0573, plateau 3.84% →
  3.62%, injected `τ` `2.26·10⁻²` → `2.31·10⁻²`, resets 6/5 unchanged); its
  every other line is the same. That these moves are the reconstruction's and
  not the confined search's is inferred: the search returns every state with
  `D ≤ ρ_max` bit for bit, and the perturbed tube runs below match the
  emulation of the slack alone, on 2.0, to every digit.
- **The test count**, 26268 → 26296 (+18 the new reconstruction testset, +8 the
  step-cache entry read back, +2 names).

Unchanged to every digit: the uniform flow's four failures (below, "The step
cache"), step 3's sweep, the shock tubes against their stored references (no
reference regenerated), every SSP3 atmosphere row, the drift tallies, the
`Float32` smoke run, the seed tallies of every case of `bin/recovery_cost.jl
steptime`. The bounds-checked run fails EOS-10's four assertions as before
(272 B and 144 B). The perturbed tube runs are under "Floors, the atmosphere
and the magnetosphere", "EntropyEOS 2.0", and the step time under "The step
cache".

The only coupling back to the fields is the implicit Ohm step, which is
[a nested solve](#the-implicit-stage-solve) around this same call.

**Entropy fallback** (proposed): where `τ_fluid = τ − D − ½(E² + B²)` is a
small difference of large numbers (electromagnetically dominated cells, the
magnetosphere) or the energy inversion fails, invert at the advected `s`
instead (EOS-3). The switch is a threshold on
`½(E² + B²) / τ` (open). *(Investigation 2026-10-16, "Floors, the
atmosphere and the magnetosphere": measured on the magnetized atmosphere
tube at a threshold of `e_em > τ_f`, it removes the chaos as B + hot does, but
the advected `s` in those cells is itself noise.)*

**Measurement (2026-10-23): recovery counts and costs** (branch
`claude/recovery-cost`, not a PLAN.md step; `bin/recovery_cost.jl`, which
redefines the recovery's two `con2prim_safe` calls and the passes in-process
and is the code bit for bit with its counters off; Julia 1.13, Apple M3 Pro,
one thread, load 4–15 on 12 cores; ideal gases as the cases set them). Per
owned cell and SSP3(4,3,3) step the scheme makes **11 recoveries that run
EntropyEOS's cold seed** — 3 right-hand sides, 3 stage limiters, 1 step
limiter, 4 stage predictors (or `IdealConductor`'s single ideal recovery):
`state_guesses` gives `s` but no `w`, and `con2prim` runs `cold_seed`
whenever `s` or `w` is missing — plus the stage iterations, which are warm
(0 to 5.5 a step). The conserved bounded ghost fill (emulated: step 8's
first fill, the tree at 7605986) adds `3((N + 2G)/N)^D − 3` ghost recoveries
(1.8 in 1D at `N = 10`, 2.7 in 2D at 16, 13.1 in 3D at 8: the 5.36 holds);
the primitive fill (this tree's) none. A cold-seeded call costs 0.75–1.3 µs
(2.0–2.2 µs in the 3D blast, with about 4 Newton iterations and 4% nested
fallbacks), a warm iteration 0.22–0.25 µs; recoveries are 48–53% of the
step on the 1D waves (100 cells, where kernel launches dominate the rest),
67–75% on the tubes and in 2D, 73% in 3D, and 51–82% with the conserved
fill (whose step is 5–15% longer in 1D and 2D, 64–74% in 3D). Pointwise,
`con2prim_safe` from a warm start 10⁻³–10⁻² away costs 0.4–0.7 µs on the
Γ = 5/3 gas against 1.4–1.7 µs cold (median) and 2.4–3.5 µs on EntropyEOS's
synthetic table through `GeometricEOS` against 10–11 µs; the advected `s`
alone saves nothing (the same time and iterations as fully cold). Estimated
savings, as a fraction of the step, primitive fill / conserved fill (the
recorded calls of three steps replayed both ways):

| case | recoveries / cell / step | share of step | (a) step-local seeds | (b) reuse: RHS, first predictor | (c) ghost copies |
|---|---|---|---|---|---|
| EM wave, `Vacuum`, 100 | 11 / 12.8 | 48% / 51% | 36% / 37% | 13%, 4% | — / 7% |
| Alfvén wave, ideal, 100 | 11 / 12.8 | 53% / 56% | 39% / 40% | 14%, — | — / 8% |
| resistive tube σ = 10, 10², 10³, 200 | 13.7–16.0 / +1.8 | 70–75% / 73–76% | 52–56% / 52–55% | 17–19%, 6% | — / 9–10% |
| magnetized atmosphere tube, 200 | 12.5 (3.3 no solve) / 14.3 | 67% / 70% | 48% / 49% | ≤ 19%, 6% | — / 9% |
| rotor, 64² | 11 / 13.7 | 75% / 79% | 60% / 59% | 20%, — | — / 11% |
| Orszag–Tang, 64² | 11 / 13.7 | 75% / 79% | 58% / 56% | 20%, — | — / 15% |
| spherical blast, 16³, `N = 8`, `G = 3` | 11 / 24.1 | 73% / 82% | 69% / 64% | 22%, — | — / 26% |
| resistive tube σ = 10², two levels | 16.5 / 18.4 | 70% / 77% | 49% / 48% | 16%, 5% | — / 8% |
| rotor, two levels, 7168 cells | 11 / 13.7 | 74% / 80% | 60% / 61% | 20%, — | — / 10% |

(a) seeds every recovery but the step limiter's from the cell's latest one
in the step (the cache started by the step limiter's cold recovery of
`uⁿ⁺¹`): the right-hand side's then costs 0.1 µs (its input is the stage
limiter's bit for bit), the stage limiter's 0.1 µs (0–0.2 Newton iterations
from the stage solve's last recovery, whose input is within 10⁻¹⁰ of
`τ + D` of it everywhere — bit for bit where σ = 0, in 7–12% of the
resistive tubes' cells), a predictor 0.14–0.28 µs. (b) needs no
seed: the right-hand side's owned recoveries have the stage limiter's
input bit for bit wherever the limiter wrote nothing (100% off the
atmosphere tube, 80% on it), and the first predictor the step limiter's
(non-ideal σ), so reusing those recoveries changes no result; (a) includes
most of (b). (c) every same-level ghost of the conserved fill holds its
source's `U`, and recovers its `P`, bit for bit but the sign of a zero (1.8
million checks), so exchanging `P` would replace those recoveries.
Correctness: warm-seeded answers differ from today's by at most 10⁻¹¹ in
`ρ`, `s`, `w`, never in the policy's flags, except where the thermal energy
is a tiny part of `τ` (`s` up to 1.3·10⁻⁴ at the atmosphere tube's front,
within EntropyEOS's conditioned bound); run with real step-local seeds the
final states move by 4·10⁻¹⁴ (tube), 6·10⁻¹³ (rotor), 1·10⁻¹² (atmosphere
tube), stage counts and the injected mass unchanged, and under the
primitive fill 1D = 2D, the turned run, one chunk = two and one thread =
four stay bit for bit. Under the conserved fill, ghosts seeded from their
own history break 1D = 2D and the turn, and Orszag–Tang's totals drift 26
ulp instead of 1 (a face's two blocks see `P` that differ at the
tolerance): there the ghosts would need their owner's `P` (c). EntropyEOS,
for Erik: on the synthetic table near rest (`W = 1.005`) 16–26% of states
end in the nested fallback whatever the seed. *(Measured, for Erik; no
change to the code, which decision is his.)* *(Resolved 2026-10-23: overtaken
by Erik's decision of (a), (b) and (c) below; the share near rest is
EntropyEOS's.)* *(Amended 2026-10-23,
decided by Erik:)* (a), (b) and (c) are all to be done; (c) is the
parent-centred primitive fill, which this tree already ran, now the only
one ("Refinement"), with the dead `prim2con` of the conserved ghosts
removed.

**The step cache** *(stage 3 of the recovery plan, `claude/recovery-pipeline-3`,
2026-10-24; (a) and (b) above, decided by Erik 2026-10-23)*. The only true
input of a Runge–Kutta step is `uⁿ`; every recovery after the first is of a
state a little way from one the cell has just recovered. So inside a step:

- **What it holds.** One entry per owned cell (`MHDProblem`'s `cache`, a
  cell set with `G = 0` and 27 slots, `src/con2prim.jl`): EntropyEOS's input
  to the cell's latest recovery in the step — the key `(D, τ, D_Y, S_∥, S_⊥,
  B²)` that `Con2PrimIn` gets — and the output of its two levels: `(ρ, s,
  Yₑ, w, v_∥, v_⊥)`, the whole `EOSPoint` and the policy flags. An entry is
  valid while its `s` is a number. Built with the problem, invalid, so a
  `rebuild` after a regrid has a fresh one of the new mesh's size and
  nothing is transferred.
- **(a) Seeds.** Every recovery inside a step — the right-hand side's, both
  limiters', the stage solve's predictor and `IdealConductor`'s ideal
  recovery — is seeded with the entry's `(s, w, u_eos)` where the cell has
  one, and from the state (`state_guesses`) otherwise. The stage solve's
  chaining inside (`guesses_of`) is unchanged; it leaves the cell holding
  the recovery of the state it returns (its final iterate's, or the
  predictor's at `σ = 0`).
- **(b) Reuse.** A recovery whose key is the entry's bit for bit (`===` on
  the six floats) takes the entry's output and does not call EntropyEOS.
  `−0.0` is not `0.0` there, a conservative miss; a NaN matches only the
  same NaN, which EntropyEOS, a function of its input's bits, would answer
  the same way; an invalid entry never matches (its `s` is NaN, whatever its
  key). Every recovery, called or reused, hands its caller its state from
  an entry (`two_levels`), so a reused recovery is the one it replaces bit
  for bit, and outside a step (an invalid entry, nothing written) the code
  is the code before the cache. In practice the reuse is the right-hand
  side after a stage limiter that wrote nothing (the same state, recovered
  with the same arithmetic), the limiter after a stage at `σ = 0` (the stage
  writes `E★` back), the second stage's predictor where the first stage's
  increment is zero (vacuum) or the ideal recovery reads only the matter
  (`IdealConductor`), and uniform regions, where nothing changes in a step.
  For the keys to agree in curved space, every site computes `B`, `E` and
  the undensitized `D`, `S`, `τ` as `· inv(√γ)` (amended 2026-10-24: the
  right-hand side's recovery and the ghosts divided by `√γ`; in flat space
  `√γ = 1` exactly and the two are the same numbers, so the change moved
  nothing — the flat-space suite's output is the same with and without it).
  *(Exercised in step 9:)* on the Kerr-Schild cases, with the interior, one
  integration, two and a restart agree bit for bit ("Validation ladder" 3).
- **Rule 1: the cache lives within one step.** The step limiter
  (`reset_step!`) invalidates every cell's entry as it ends the step, and
  `with_step_cache` invalidates it before and after `evolve!`'s stepping. So
  every step starts with an invalid cache, its first recovery (the first
  stage's predictor, or the right-hand side of a trivial first stage) is
  cold, and **a step is a pure function of `uⁿ`, `t` and `Δt`**: one chunk
  or two, a restart from a saved `u` with a rebuilt problem, one thread or
  four reproduce the run bit for bit (tests). This rests on IMEXRungeKutta
  1.4's step semantics, read in its source: `step!` runs the stages, the
  update and the step limiter once, with no retry, rejection or adaptive
  step, and `solve!` is `step!` until `t1`; an exception from a callback
  propagates out of `step!` leaving `integ.u = uⁿ` (after the step limiter,
  undefined), and out of `with_step_cache`, whose `finally` switches the
  cache off and invalidates it. No step-begin hook is needed, so no IMEX
  request: the step limiter is the step's end, and `with_step_cache` the
  only way into a step with the cache on. (It relies on the integrator's
  step limiter being `reset_step!`, as `mhd_integrator` builds it.)
- **Rule 2: the cache is active only inside `evolve!`'s `step!`/`solve!`
  calls.** A host-side switch on the problem (`cache_on`, a `Ref{Bool}`;
  the kernels get a plain `Bool`), set by `with_step_cache` around them and
  cleared in its `finally`. Outside — a callback called directly, an
  `after_step` monitor (which runs between steps), `update_primitives!`
  before a regrid, the regrid's reset, an integrator a test builds and
  steps itself — no kernel reads or writes the cache, every recovery is
  seeded from the state, and the callbacks are pure in `(u, t)` as before
  (the step-5 purity testset is unchanged). `evolve!(…; step_cache = false)`
  is the negative control.
- **No ghost is ever recovered** (stage 1), so the cache has owned cells
  only.
- **Tallies.** `X` counts, per site — the right-hand side, the stage solve's
  first recovery, the stage limiter, the step limiter — the recoveries made
  inside a step as cold (from the state), warm (from the cache) and reused
  (`ohm_tally(p).seeds`); the stage solve's later recoveries stay in
  `X_RECOVERIES`, which still counts every recovery the stages make,
  reused or not. Per SSP3(4,3,3) step and cell that is 11 counted
  recoveries, of which exactly one is cold.
- **Dropped: reuse across steps** (the step limiter's recovery of `uⁿ⁺¹`
  feeding the next step's first predictor, column (b)'s second number
  above). A cache that outlives a step either breaks restart
  reproducibility (a restart from `uⁿ⁺¹` has no cache) or forces the step
  limiter cold to keep the entry canonical; it would save at most about
  0.2 µs per cell and step, 4–6% of a step and only where σ is neither zero
  nor ideal, against a rule set twice as complex. *A possible follow-up,
  for Erik.*

**Measured 2026-10-24** (`bin/recovery_cost.jl steptime`; Symmetry cn086,
AMD EPYC 7532, one job, one pinned thread, Julia 1.13; stage 3 at 1462923
against stage 2 at a9cd482, two rounds each, interleaved; ms per step, the
mean over steps 2…n, the best of three runs; "no cache" is this tree with
`step_cache = false`). Seeds per owned cell and step as cold / warm /
reused, from `X`; "iter." the stage solve's recoveries after its first:

| case | stage 2 | stage 3 | saving | no cache | RHS | stage solve | stage limiter | step limiter | iter. |
|---|---|---|---|---|---|---|---|---|---|
| EM wave, `Vacuum`, 100 | 3.408, 3.408 | 1.450, 1.435 | **57%, 58%** | 3.020, 3.088 | 0/0/3 | 1/2/1 | 0/0/3 | 0/1/0 | 0 |
| Alfvén wave, ideal, 100 | 3.904, 3.869 | 1.657, 1.673 | **58%, 57%** | 3.468, 3.480 | 0/0/3 | 1/2/1 | 0/3/0 | 0/1/0 | 0 |
| resistive tube σ = 10, 200 | 8.810, 8.698 | 3.461, 3.487 | **61%, 60%** | 8.582, 8.607 | 0/0/3 | 1/2.78/0.22 | 0/2.74/0.26 | 0/0.93/0.07 | 4.99 |
| resistive tube σ = 10², 200 | 8.698, 8.682 | 3.430, 3.454 | **61%, 60%** | 8.481, 8.500 | 0/0/3 | 1/2.86/0.14 | 0/2.74/0.26 | 0/0.96/0.04 | 5.04 |
| resistive tube σ = 10³, 200 | 8.464, 8.466 | 3.245, 3.252 | **62%, 62%** | 8.263, 8.300 | 0/0/3 | 1/2.88/0.12 | 0/2.62/0.38 | 0/0.96/0.04 | 2.73 |
| magnetized atmosphere tube, 200 | 8.843, 8.822 | 3.202, 3.216 | **64%, 64%** | 8.621, 8.644 | 0/0.60/2.40 | 1/2.50/0.51 | 0/1.23/1.77 | 0/0.93/0.07 | 1.45 |
| rotor, 64² | 158.9, 159.3 | 57.8, 58.1 | **64%, 64%** | 159.1, 160.1 | 0/0/3 | 1/2/1 | 0/3/0 | 0/0.76/0.24 | 0 |
| Orszag–Tang, 64² | 170.5, 171.0 | 65.0, 66.3 | **62%, 61%** | 172.0, 172.6 | 0/0/3 | 1/2/1 | 0/3/0 | 0/1/0 | 0 |
| spherical blast, 16³ | 257.2, 257.3 | 106.6, 104.2 | **59%, 60%** | 260.9, 259.7 | 0/0/3 | 1/2/1 | 0/3/0 | 0/0.88/0.12 | 0 |
| resistive tube σ = 10², two levels | 10.54, 10.56 | 4.357, 4.370 | **59%, 59%** | 10.63, 10.65 | 0/0/3 | 1/2.84/0.16 | 0/2.76/0.24 | 0/0.95/0.05 | 5.54 |
| rotor, two levels, 7168 cells | 293.7, 294.4 | 108.3, 109.1 | **63%, 63%** | 296.5, 297.5 | 0/0/3 | 1/2/1 | 0/3/0 | 0/0.85/0.15 | 0 |

The saving is 57–64%, at and a little above the top of the 2026-10-23
estimate (36–60% for (a), most of (b) inside it): of the 11 counted
recoveries per cell and step one is cold, the right-hand side's three are
reused everywhere the stage limiter wrote nothing (100% but on the
atmosphere tube, 80% there, as estimated), and reuse also covers what the
estimate counted as warm — the second stage's predictor in vacuum and under
`IdealConductor`, the limiter after a stage at `σ = 0`, uniform regions —
on a CPU (Zen 2) where a cold seed costs relatively more than on the M3 the
estimate was made on. The whole suite runs 25% faster (14m00s → 10m32s at
one thread on cn079). The tree with the cache off is 0–11% faster than
stage 2 (the 1D waves most); not investigated. Correctness:
`test/step_cache_tests.jl` (warm against cold — the final states 1.0·10⁻¹⁴
to 2.5·10⁻¹² apart, the recovered `ρ`, `s`, `w` at most 9.9·10⁻¹², the same
branches, resets and injected mass; one chunk = two; a restart = the run;
monitors change nothing; the reuse counts exactly), and the suite's 1D =
2D, turned, closed-box mirror, Orszag–Tang and one- = four-thread claims,
unchanged.

**What moves, against stage 2 on the same node** (every printed number of
the suite; the rest of them move in their last digits, as warm seeds answer
to the tolerance rather than to rounding): the uniform-flow testset's slow
flows (`vˣ = ±1/2`) **break their bound**: `1.06·10⁻¹¹`, `8.1·10⁻¹¹`,
`1.68·10⁻¹¹`, `8.4·10⁻¹¹` of a variable's scale (SSP3, ARS) against
`≤ 10⁻¹¹`, where stage 2 had `0.9–1.9·10⁻¹²` (the fast flows are
unchanged, `≤ 1.0·10⁻¹²`). Measured why (laptop, the same numbers): with
EntropyEOS's `tol = 10⁻¹⁴` instead of `10⁻¹²` the cached runs give
`0.7–2.9·10⁻¹²` again. A cold Newton solve ends with a quadratic step that
lands near rounding, and a uniform state's cells all take the same one; a
warm seed already within `tol` returns after no step or one, at the
tolerance, and which depends on the cell's history in the step — at the
Dirichlet faces, where the ghosts are the exact primitives, that history
differs, and the flow drifts by the tolerance per step. **The bound is kept
and fails; for Erik** *(resolved 2026-10-30, decided by Erik: option (a), the
drift to the recovery's tolerance, `≤ 10³·tol`, plus the 1000-step run —
"The uniform flow: to the tolerance" below)* (accept the tolerance-level drift and say so in the
bound; or ask EntropyEOS for a warm start that always polishes with one
Newton step after its residual test, which would restore rounding at a
small cost; or a tighter `tol` in the cases). Besides: the ARS(4,4,3)
near-vacuum rows, chaotic at rounding (2026-10-24, above), move as they do
under any perturbation — the rarefaction's min ρ `1.0·10⁻¹⁰` →
`2.1·10⁻¹⁰` and resets 133/116 → 139/113, the ideal magnetized atmosphere
`−7.85·10⁻⁹`, `−2.03·10⁻⁹` → `−7.64·10⁻⁹`, `+2.81·10⁻⁹` (SSP3, ARS), `W`
3.99, 4.13 → 4.00, 4.01; the coarse-fine magnetized tube `−1.4027·10⁻⁸` →
`−1.4012·10⁻⁸`; the `Float32` Alfvén smoke `8.5575·10⁻⁵` → `8.5421·10⁻⁵`;
drift tallies within their spread (static two-level 0.146 → 0.122 ulp,
tracked Balsara 0.015 → 0.011, oblique 1.49 → 1.31, adaptive rotor 0.040 →
0.054, Orszag–Tang's asymmetry `3.9·10⁻¹³` → `1.4·10⁻¹³`); Balsara 3 with
PPM in the 9th digit. The atmosphere tubes' counts and the resistive
tube's stage branches are unchanged. The `· inv(√γ)` unification alone
(3495217) gives the stage-2 output exactly (one thread, cn081, every line).

*Re-measured 2026-10-26, EntropyEOS 2.2* (`bin/recovery_cost.jl steptime`,
the same script; Symmetry cn086, AMD EPYC 7532, one job, one pinned thread;
`claude/entropyeos-2.2` against stage 3 at 4929c94, two rounds each,
interleaved; ms per step as above). The 2.2 tree is **1–3% slower** per step
with the cache (Alfvén 5–7%, the EM wave 2–3%, the 2D and 3D cases −1 to 4%)
and 1–5% without; the seed tallies are the same in every case:

| case | stage 3 | 2.2 | change | no cache, stage 3 | no cache, 2.2 |
|---|---|---|---|---|---|
| EM wave, `Vacuum`, 100 | 1.464, 1.487 | 1.506, 1.517 | +2.9%, +2.0% | 3.072, 3.059 | 3.202, 3.188 |
| Alfvén wave, ideal, 100 | 1.652, 1.624 | 1.736, 1.740 | +5.1%, +7.2% | 3.489, 3.446 | 3.640, 3.615 |
| resistive tube σ = 10, 200 | 3.442, 3.472 | 3.527, 3.572 | +2.5%, +2.9% | 8.574, 8.594 | 8.772, 8.761 |
| resistive tube σ = 10², 200 | 3.415, 3.444 | 3.503, 3.548 | +2.6%, +3.0% | 8.507, 8.501 | 8.626, 8.653 |
| resistive tube σ = 10³, 200 | 3.211, 3.246 | 3.312, 3.330 | +3.2%, +2.6% | 8.260, 8.279 | 8.393, 8.415 |
| magnetized atmosphere tube, 200 | 3.183, 3.183 | 3.234, 3.214 | +1.6%, +1.0% | 8.587, 8.589 | 8.733, 8.768 |
| rotor, 64² | 58.06, 57.54 | 58.73, 58.27 | +1.2%, +1.3% | 159.7, 158.8 | 162.2, 161.7 |
| Orszag–Tang, 64² | 65.78, 64.18 | 65.60, 65.54 | −0.3%, +2.1% | 171.8, 170.3 | 172.0, 172.8 |
| spherical blast, 16³ | 103.0, 103.1 | 104.5, 107.4 | +1.5%, +4.1% | 259.2, 260.5 | 262.8, 262.4 |
| resistive tube σ = 10², two levels | 4.350, 4.335 | 4.422, 4.425 | +1.7%, +2.1% | 10.63, 10.61 | 10.78, 10.76 |
| rotor, two levels, 7168 cells | 109.2, 106.5 | 108.6, 108.1 | −0.6%, +1.5% | 297.9, 295.0 | 300.0, 299.1 |

Not split between its two candidates: EntropyEOS's confined search, which
made warm `Float64` calls 2–4% slower on an H200 (its HISTORY, 2026-10-25:
the `acosh` of the density floor, recovered in the first stage only) and
which every recovery pays, and `inside_window`, which costs the reconstruction
what the exact test cost plus one multiplication per face. The cache's saving
is unchanged (57–64%). *(Decided by Erik 2026-10-30:)* the 1–3% is
accepted, and not attributed further.

**The uniform flow under 2.2** (re-measured 2026-10-26, every run above):
`1.0602831660·10⁻¹¹`, `8.1098066854·10⁻¹¹`, `1.6805728171·10⁻¹¹`,
`8.4017111996·10⁻¹¹` (`vˣ = −½`, `+½`; SSP3, ARS) — stage 3's numbers to every
digit, so the four failures stand, at one and four threads, in the clean
checkout and with bounds checks. Nothing in 2.1 or 2.2 addresses warm starts
returning at the tolerance (point 2 of the 2026-10-24 discussion).

**The uniform flow: to the tolerance** *(decided by Erik 2026-10-30)*. A
uniform flow through Dirichlet faces stays uniform to the recovery's
tolerance, not to rounding, and the testset says so: `≤ 10³·tol` (10⁻⁹)
after 13 steps, where it was `≤ 10⁻¹¹`. Rejected: a polishing Newton step on
warm starts in EntropyEOS (it would roughly double every warm call, give back
much of the cache's saving, and break `con2prim_safe`'s fixed points), and a
tighter `tol` in the cases (below the energy residual's floor on tables, and
impossible at `Float32`). Measured on the laptop the same day, how the change
grows (the testset's own setup, run longer; the worst change of a
variable's scale, `D·s` mostly, sometimes `D`; cold = `step_cache = false`):

| run | 13 steps | 50 | 250 | 1000 |
|---|---|---|---|---|
| SSP3 `vˣ = +½`, cold / cached | 8.8e−13 / 1.7e−11 | 7.0e−12 / 1.5e−11 | 5.5e−11 / 2.6e−11 | 2.0e−10 / 6.6e−11 |
| SSP3 `vˣ = −½`, cold / cached | 1.2e−12 / 1.1e−11 | 7.3e−12 / 1.1e−11 | 5.5e−11 / 2.9e−11 | 2.1e−10 / 6.5e−11 |
| ARS `vˣ = +½`, cold / cached | 1.9e−12 / 8.4e−11 | 8.1e−12 / 8.5e−11 | 1.8e−11 / 3.0e−10 | 1.2e−10 / 2.7e−10 |
| ARS `vˣ = −½`, cold / cached | 1.8e−12 / 8.1e−11 | 5.3e−12 / 7.9e−11 | 1.3e−11 / 3.2e−10 | 1.6e−10 / 3.6e−10 |
| `vˣ = ±0.999`, either | ≤ 1.1e−12 | | | 3.0–3.7e−11, cold = cached |

The cold seeds were not at rounding either beyond the testset's 13 steps:
they drift by about 2·10⁻¹³ a step and pass 10⁻¹¹ by 50. The cache's change
comes mostly in the first steps and then grows slowly — under SSP3 it ends
3 times closer to the exact state than the cold run; under ARS 2 times
further — presumably because a cell whose recovery matches its seed keeps it,
where a cold solve adds fresh rounding every step (not verified). A second
testset pins that: at `vˣ = ±½` over 1000 steps, `≤ 2·10³·tol`, which the
first steps' rate kept up (6.5·10⁻¹² a step, 6.5·10⁻⁹) would fail.

**The two-phase recovery** *(stage 4 of the recovery plan,
`claude/recovery-pipeline-4`, 2026-10-30; decided by Erik 2026-10-23: "the
two-phase con2prim everywhere", CPU only)*. EntropyEOS 2.x defines
`con2prim_safe` as two phases (its `CODE.md`, "Two stages"):
`con2prim_safe_start(eos, in, opts, proj, s, w, u) -> (out, stage)`, the
solve from the seed, done where the Newton converged on a state inside every
face — the common case — and then `out` is the answer; and, where it is not
done, `con2prim_safe_finish(eos, in, opts, proj, stage)`: the fallback, the
diagnosis and the face solves, and the inputs no solve applies to (non-finite,
below `ρ_min`, above `ρ_max cosh w_cap`). The split is bit for bit the fused
call, and the hand-off is an `isbits` struct of about 48 B. On a GPU the
second phase is the divergent one, and a later step can compact it.

- **Where.** Every recovery outside the stage solve's iterations: the
  right-hand side's (`primitive_kernel!`), both limiters' (`reset_kernel!`)
  and the stage solve's first one (`ohm_kernel!`: the predictor, or
  `IdealConductor`'s ideal recovery). The iterations — the iterates, the
  Newton step's ideal seed, the lagged-velocity fallback — keep the fused
  `con2prim_safe` and `apply_policy` per call: they are warm, mostly done in
  the first phase (measured below), and splitting the loop would need a pass
  per iteration.
- **Three passes, each a kernel over the owned cells.** (1) **start**
  (`start_kernel!`): the metric, `B` and `E` of the cell, EntropyEOS's input
  computed by the very functions the site's own kernel uses (`hydro_input`,
  or `predictor_input` for the stage: the same arithmetic, so the same bits),
  the step cache's entry and seed exactly as the third pass reads them, and
  `con2prim_safe_start`; a reused recovery (the entry's key is the input's)
  skips it. (2) **finish** (`finish_kernel!`): `con2prim_safe_finish` behind
  the hand-off's per-cell guard, over every owned cell — no compaction on the
  CPU, where it would buy nothing and would be the package's first worklist,
  and every cell's work stays its own, so one thread and four agree. (3)
  **continuation**: the site's kernel as before, its recovery taking level 1
  from the hand-off (`two_levels` with a hand-off), then `apply_policy`, the
  entry, the velocity, and the site's own work — `P_from_W` and the write-back
  of `P` and `W`, the reset's write-back and tallies, the rest of the stage
  solve.
- **The hand-off** (`Handoff`, `src/con2prim.jl`): per owned cell, as four
  arrays indexed `(n…, b)` — `todo` (finish is needed: not reused, and the
  first phase not done), `cin` and `stage` (written only where `todo`), and
  level 1's `out` (`Con2PrimSafeOut`, 232 B at `Float64`, beside 48 B each
  for `cin` and `stage`; written by
  the first pass where it was done and by the second where not, and not at
  all where the entry is reused, whose output the third pass takes from the
  step cache). Struct-of-arrays, so that a pass reads only what it needs.
  Owned by `MHDProblem` and built with it, like the step cache: a `rebuild`
  after a regrid has a fresh one of the new mesh's size, nothing is
  transferred, and each pass writes what the next reads before it is read.
- **Level 1 and level 2 in separate kernels.** The structure a GPU step
  needs, and EntropyEOS's `HISTORY.md` records a CUDA.jl 6.4.2 fault in a
  kernel that called level 1 and then level 2; the workaround is separate
  kernels. Level 2 is in the third pass; the first two call only level 1.
- **The fused path stays.** `MHDProblem(…; two_phase = false)` and
  `evolve!(…; two_phase = false)` build no hand-off and run each recovery in
  one kernel, `con2prim_safe` and `apply_policy` together, exactly as before
  (the same kernels with `nothing` for the hand-off). It is kept for the
  comparison: `test/two_phase_tests.jl` holds the two bit for bit — every
  state and every tally of seven runs (the EM wave, Alfvén, the resistive
  tube, the magnetized atmosphere tube with the step cache on and off, the
  rotor, the adaptive rotor), and every callback outside a step on a state
  with a cell below `ρ_min`, one below the coldest state, one beyond the
  rapidity cap and a non-finite one, under the tube's conductivity and under
  `IdealConductor`. Whether the CPU keeps the split or fuses is Erik's
  decision (measured below); no CPU/GPU switch beyond this one.
  *(Decided by Erik 2026-11-02:)* **fused is the default**
  (`two_phase = false` in `MHDProblem` and `evolve!`): on the CPU the split
  costs 1–10% a step and buys nothing, since only 0–2.4% of the recoveries
  reach the second phase and none of the stage solve's iterations would.
  The split stays in the code, tested bit for bit against fused, for a GPU
  step to turn on.

**Measured 2026-10-30** (`bin/recovery_cost.jl steptime`; Symmetry cn084,
AMD EPYC 7532, one job, one pinned thread, Julia 1.13; this tree at 0334aed
split and fused (`two_phase = false`), both with the step cache, against
`claude/entropyeos-2.2` at fea242a, two rounds each, interleaved; ms per
step, the mean over steps 2…n, the best of three runs). **The split is
slower on the CPU**: 1–4% on the 2D, 3D and resistive cases, 3–10% on the
atmosphere tube (its second round noisy: runs 3.98, 4.79, 3.46), 6–9% on the
1D waves. The fused path of this tree is as fast as fea242a or faster (the
1D waves 6–11%; not investigated):

| case | split | fused | fea242a | split / fused | split / fea242a | fused / fea242a |
|---|---|---|---|---|---|---|
| EM wave, `Vacuum`, 100 | 1.475, 1.455 | 1.348, 1.345 | 1.436, 1.434 | +9.4%, +8.2% | +2.7%, +1.5% | −6.1%, −6.2% |
| Alfvén wave, ideal, 100 | 1.526, 1.538 | 1.441, 1.437 | 1.609, 1.614 | +5.8%, +7.0% | −5.2%, −4.7% | −10.4%, −10.9% |
| resistive tube σ = 10, 200 | 3.489, 3.587 | 3.455, 3.464 | 3.489, 3.472 | +1.0%, +3.5% | +0.0%, +3.3% | −0.9%, −0.2% |
| resistive tube σ = 10², 200 | 3.558, 3.541 | 3.419, 3.431 | 3.449, 3.432 | +4.1%, +3.2% | +3.2%, +3.2% | −0.9%, −0.0% |
| resistive tube σ = 10³, 200 | 3.356, 3.339 | 3.222, 3.228 | 3.255, 3.243 | +4.2%, +3.4% | +3.1%, +3.0% | −1.0%, −0.5% |
| magnetized atmosphere tube, 200 | 3.282, 3.457 | 3.164, 3.130 | 3.192, 3.196 | +3.7%, +10.5% | +2.8%, +8.2% | −0.9%, −2.1% |
| rotor, 64² | 58.53, 59.03 | 57.46, 56.48 | 56.96, 56.85 | +1.9%, +4.5% | +2.8%, +3.8% | +0.9%, −0.7% |
| Orszag–Tang, 64² | 66.14, 65.78 | 64.73, 65.25 | 64.05, 64.12 | +2.2%, +0.8% | +3.3%, +2.6% | +1.1%, +1.8% |
| spherical blast, 16³ | 104.8, 105.3 | 103.6, 105.1 | 102.2, 103.7 | +1.2%, +0.2% | +2.6%, +1.6% | +1.4%, +1.4% |
| resistive tube σ = 10², two levels | 4.404, 4.433 | 4.292, 4.311 | 4.289, 4.303 | +2.6%, +2.8% | +2.7%, +3.0% | +0.1%, +0.2% |
| rotor, two levels, 7168 cells | 110.7, 110.1 | 107.6, 107.0 | 106.5, 106.9 | +2.9%, +2.9% | +3.9%, +3.0% | +1.0%, +0.1% |

Where the time goes was not profiled; per recovery the split adds two
kernel launches per site and the hand-off's stores and loads (the 232-B
`out` at every cell not reused), and the first pass recomputes the
metric, `B`, `E` and the input that the third computes again — which costs
most where a step does least else (the 1D waves, whose right-hand side
recoveries are all reused). *Whether the CPU keeps the split or fuses is
Erik's decision.* *(Resolved 2026-11-02, decided by Erik: fused is the
default.)*

**The share that reaches finish** (`bin/recovery_cost.jl finish`, the same
job and cases; per site, the recoveries whose first phase was not done, of
all and of those not reused): **none** in the EM and Alfvén waves, the three
resistive tubes, Orszag–Tang and the refined resistive tube; the magnetized
atmosphere tube 6 of 120000 at the right-hand side (0.025% of those
started), 1755 of 160000 at the stage's first recovery (1.25%), 882 of 40000
at the step limiter (2.4%), none at the stage limiter; the spherical blast
1272 of 131072 at the stage's first recovery (1.3%); the rotor 2 and the two-level rotor 4 there; nothing else. **The
stage solve's iterations reach finish nowhere**: 0 of their 57846 (the
atmosphere tube) to 531938 (the two-level resistive tube) recoveries, none
reused. So a later GPU step sees the second phase in at most about 2% of a
site's cells, at the sites above; the warm iterations would gain nothing
from the split.


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
| matter `U` | cell | 10 | 3 | Conservative width 3, bounded (decided in step 8), in the regrid; its ghosts are not written by the scheme (2026-10-23) / `=> schedule` |
| `A_d`, `d = 1…3` | edge `d` | 1 each | 3 | Conservative width 3 along the edge, point values of width 4 across it (AMR-9; decided in step 8) / `=> schedule` |
| `Φ̃` | vertex | 1 | 3 | PointValue, width 4 (decided in step 8) / `=> schedule` |
| `P` — primitives, EOS aux, flags, `E`, `B̄` at cells, `q`, cell metric | cell | 30 | 3 | `=> nothing` |
| `W` — `ρ★, s, Yₑ, uⁱ = Wvⁱ, Eⁱ`, the matter's ghost fill *(added in step 8)* | cell | 9 | 3 | Conservative width 3, bounded, centred on the parent (decided by Erik 2026-10-23; step 8's was PointValue width 2) / `=> nothing` |
| `B_d` — `curl A` | face `d` | 1 each | 3 | `=> nothing`, recomputed every RHS |
| `F_d` — matter fluxes | face `d` | 10 | 0 | `=> nothing`; `restrict_interfaces!` is the flux correction |
| `EMF_d` | edge `d` | 1 each | 0 | `=> nothing`; `restrict_interfaces!` is the CT correction |
| `Tsrc` — `T_ab` for GH | vertex | 10 | 0 | `=> nothing` |
| `X` — per-cell tallies, light speed *(added in step 5)*, the reset's injection *(step 6)*, `h∇·𝔅` *(step 7)*, the recoveries' seeds per site *(2026-10-24)* | cell | 31 | 0 | `=> nothing` |
| step cache — the cell's latest recovery in the step: EntropyEOS's input and output *(2026-10-24)*; `MHDProblem`'s, not `MHDFields`' | cell | 27 | 0 | not passed: a `rebuild` makes a new one, invalid |

`P`'s 30 slots are laid out in `src/variables.jl` (amended in step 1: the
table said "~25"): the eleven the recovery returns — `ρ★`, `vⁱ`, `s`, `Yₑ`,
the rapidity `w` (also the next recovery's warm start), `p`, `h`, `cs²` and
the EOS's own warm start — then the policy flags (as a number; every bit is
below 2¹⁸ — bits 8–17, `FLAG_POL_W_CEILED` the highest, since EntropyEOS
2.0; amended 2026-10-21 and 2026-10-24, EntropyEOS 2.0 — so exact even in
`Float32`, which holds every integer below 2²⁴, and tested at `Float32`
with bit 17), `Eⁱ`, `Bⁱ`, `q`, and the metric
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
  checks; they are `MHDProblem`'s `ops_U` and `ops_A`. *(Decided in step
  8:)* width 4 for `A` and `Φ̃` by the interface-order table, and the
  matter's width 3 **bounded** (TreeAMR's MC-bounded prolongation) in the
  ghost fill and the regrid — see "Refinement". *(Amended in step 8, when
  TreeAMR 0.2.2 landed AMR-9:)* `A_j` take the conservative family on a
  staggered set, `Operators(; prolongation=3, restriction=2, family =
  Conservative, vertex_prolongation=4)` — the two-edge mean and a width-3
  conservative prolongation along the edge's own (cell-like) direction,
  point values of width 4 across it — so that the magnetic flux through a
  coarse face is the sum through its fine faces as a state; `Φ̃`, a vertex
  set with no cell-like dimension, would get the very same operators from
  that family and stays PointValue width 4 (`ops_Φ`, its own keyword since
  step 8). The width 4 was re-measured under the new family (unchanged).

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

*As built in step 7* (`src/reconstruction.jl`; proposed in step 7 unless
marked otherwise). `limiter` names the reconstruction: `:none`, `:minmod`,
`:mc` (PLM, `G ≥ 2`) or `:ppm`, `:wenoz`, `:mp5` (`G ≥ 3`, which
`MHDProblem` checks). The keyword keeps its step-5 name rather than becoming
`reconstruction`, so that no caller changes.

- **Every method is an offset from the cell's value**, `q_face = q₀ + δ`,
  with `δ` a function of the differences `q_k − q₀` alone. So a stencil of
  equal values reconstructs to itself exactly, `δ` is odd under `q → −q`
  exactly, and the lower face is the upper face of the mirrored stencil.
  That is what makes a direction a solution does not vary in add nothing at
  all, and the direction-independence claims of rung 2a hold bit for bit
  (see "Validation ladder" 2a); a method with its weights multiplied out
  (`(2a − 13b + 47c + 27d − 3e)/60`) rounds a constant. PLM's offset is
  `slope/2` and computes the step-5 bits unchanged.
- **PPM** (Colella & Woodward 1984, eqs. 1.6–1.10): the fourth-order
  interface value with MC-limited slopes, then the monotonicity
  constraints, without contact steepening or flattening.
- **WENO-Z** (Borges et al. 2008) of fifth order, with Castro et al.'s
  power 2 and `ε = floatmin(T)^{1/4}` — below any resolved smoothness
  indicator, and large enough that `(τ₅/ε)²` cannot overflow into
  `Inf/Inf` (`1e−40`, the usual choice, is subnormal in `Float32`).
- **MP5** (Suresh & Huynh 1997, `α = 4`), with the test
  `(v − q₀)(v − v_MP) ≤ 0` taken without their `ε`. Its bound holds for
  CFL `≤ 1/(1 + α) = 1/5`, and at the default 2/5 it is unstable: Balsara 3's
  dense shell grew to `ρ ≈ 600` at 200 cells and the reset fired on Balsara
  3 and 4 (measured in step 7). So `evolve!`'s `cfl` defaults to
  `default_cfl(:mp5) = 1/5`, and to 2/5 for every other method.
- Face values on smooth data (the cell averages of `sin 2πx`, a test)
  converge at 2.04 (MC), 3.97 (PPM), 4.99 (WENO-Z) and 5.02 (MP5); on
  monotone stencils PPM's and MP5's face values stay between the face's two
  cells and WENO-Z's overshoots by at most 5.6% of the stencil's rise.
- **The scheme stays second order** whatever the faces get: the primitives
  are recovered pointwise from cell averages and `E = −v × B` is imposed at
  cell centres, both `O(h²)` from the averages a fifth-order reconstruction
  would need. The Alfvén wave (`IdealConductor`, one period, 50 … 400 cells;
  measured in step 7, below) converges at 2.67 (PPM), 2.00 (WENO-Z, MP5) in
  `S_y`, with 9, 17 and 17 times smaller errors than MC at 400 cells; the
  higher methods buy the constant, not the order. Converting averages to
  point values before the recovery (and back for the fluxes) is the known
  way past it, and is not planned.
- **The EMF's edge values** use the same offsets: `E` at a corner is
  `E₀ + δ_a + δ_b`, each offset the method's one-dimensional one along its
  direction, and the face-normal `𝔅` is reconstructed along the transverse
  direction by the method (see "The electromotive force for `A`").

*(Amended 2026-10-26, EntropyEOS 2.1.)* **The reconstruction tests validity
the way `con2prim_safe` does.** `admissible_fluid` asks EntropyEOS's
`inside_window(eos, ρ, s, Yₑ)` (EOS-11) whether `s` and `Yₑ` lie inside their
faces, with the slack `con2prim_safe` allows — `s` within `srange(eos, ρ, Yₑ)`
widened by 10⁻⁷ of its width at `Float64` (2⁻¹², 2.4·10⁻⁴, at `Float32`),
`Yₑ` within one ulp of its range —
and keeps this package's own conditions, `ρ` positive and finite and `u`
finite. Until then it tested the exact window. But since EntropyEOS 2.0
`con2prim_safe` returns a converged root within the slack of a face
unflagged and as it is, so a recovered cell could fail the exact test: on the
reproducer of 2026-10-24 (a cold re-solve of a state the stage limiter had
written onto the `s_min` face) `s` lies 5.5·10⁻⁹ of the window below `s_min`
with `policy_flags == 0`. Every state the recovery returns now passes, and
the EOS can be evaluated throughout the band (`inside_window`'s contract).
`ρ` is still not tested against `[ρ_min, ρ_max]` nor `w` against the cap, as
`check_prim_state` would: that would change the reconstruction near vacuum.

**The positivity fallback substitutes the cell's own value, so it cannot
repair a face whose cell is itself inadmissible**: that face gets the cell's
state, which fails the same test. A TVD limiter cannot produce such a face
from admissible cells (its face values lie between the cells'), so the case
arises only where a cell holds a state the test rejects — which, since the
test is `con2prim_safe`'s own, no recovered cell does (test:
`test/scheme_tests.jl`, "The positivity fallback admits every state
con2prim_safe returns unflagged"). What it cost while the test was exact is
under "Floors, the atmosphere and the magnetosphere", "EntropyEOS 2.0".

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

*(Step 7:)* under every reconstruction the corner value of `E` is
additive, `E₀ + δ_a + δ_b` with the method's one-dimensional offsets — for
PLM exactly step 5's `E + σₐ slopeₐ/2 + σ_b slope_b/2` — and not a tensor
product (reconstruct along `a`, then those values along `b`), which would
read 25 cells instead of 9 for a cross term that is `O(h²)`, the scheme's
order in more than one dimension anyway (proposed in step 7). Measured on
the Alfvén wave along the diagonal of a 2D box: `A_z`, whose EMF is the
one with two transverse directions, converges at 2.31 (MC) and 2.01
(WENO-Z).

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
*(Amended 2026-10-24, EntropyEOS 2.0:)* the valid range is level 1's
(`ProjectionOptions`: `ρ_min`, what was `ρ_atm`; `ρ_max`, what was
`ρ_ceiling`; `w_cap`), the atmosphere level 2's (`apply_policy` with
`PolicyOptions`: at rest at `ρ_min`, at `s_atm`, below `1.01 ρ_min`), and
there is no collapse ceiling ("Primitive recovery", "The two levels"). The
text below up to "EntropyEOS 2.0" is 1.1's, kept as the record.

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

**A density raised to the atmosphere's is the atmosphere at rest**
*(decided by Erik 2026-10-21, with EntropyEOS's review; EOS-5 withdrawn,
EOS-6 superseded, the decision kept in this package and keyed on a flag
that reports a fact — the investigation below)*. EntropyEOS 1.1.0 (EOS-8,
unregistered, through `[sources]`; `[compat]` `"1.1"`) reports
`FLAG_POL_ρ_FLOORED` (bit 16) where a repair raised `ρ` to `ρ_atm` without
an atmosphere reset, i.e. wrote `ρ cosh w > D` — its entropy-floor branch,
which kept the coldest state's rapidity at the momentum, `W ≈ |S|/ρ_atm`.
(The rapidity cap raises `ρ` too but keeps `D`, and does not set it.)
- **Where:** `src/con2prim.jl`, the one wrapper around `con2prim_safe` —
  `fluid_recovery` (under `con2prim_hydro`) and `ideal_recovery` (under
  `con2prim_ideal`) are its only two calls, so the reset, the right-hand
  side's `P`, the stage solve's predictor, iterates and ideal seed, and
  `IdealConductor`'s stages all see it; none of them uses EntropyEOS's
  conserved output, every one rebuilds the conserved state with this
  package's `prim2con` (checked 2026-10-21).
- **What:** `flags & FLAG_POL_ρ_FLOORED != 0`, tested alone — not tied to
  `S_FLOORED`, EntropyEOS's internal path — returns the atmosphere at rest,
  `policy_atmosphere(eos, pol, D Yₑ/D)`, its EOS point by `evaluate` (the
  stage solve's Jacobian reads it), flagged `FLAG_POL_ATMOSPHERE |
  FLAG_POL_ρ_FLOORED` — the replacement and its cause, not the discarded
  state's clamps (amended 2026-10-22, decided by Erik: as EOS-9 asks of
  EntropyEOS's own atmosphere replacements; on the branch
  `claude/no-collapse-to-atmosphere`). A branch, no mask multiplied;
  `isbits`, allocation-free, inferred, and the same bit for bit in a CPU
  kernel at `Float64` and `Float32` (`test/recovery_tests.jl`).
- **This package's rule** *(decided by Erik 2026-10-22, option (b):
  EntropyEOS reports facts, the application decides)*: a density the policy
  floored is the atmosphere at rest, so every state this package writes
  holds `D_out ≤ max(D_in, D_atm)` (`D_atm = ρ_atm`, the atmosphere at
  rest). The criterion is TreeGRRMHD's, enforced by this wrapper — not a
  property EntropyEOS promises; EntropyEOS's floored output itself raises
  `D`. The tallies are unchanged by the narrower flags: a floored stage is
  still counted as repaired (`X_POLICY`), its two bits reach the stages' `|`
  (`X_POLICY_OR`), and the reset writes it (`FLAG_POL_ANY`).
- **The collapse branch** *(gone in EntropyEOS 2.0, and with it the
  option's use; see "EntropyEOS 2.0" below)*: a hot near-vacuum cell whose energy lies above
  the hottest state of the window is, to the default policy, past the
  collapse ceiling, and is written at `ρ_ceiling` (10⁶ for the shock gas):
  every blow-up of the magnetized tube. The cases with a magnetized
  atmosphere set EntropyEOS's existing `collapse_to_atmosphere`
  (`case_policy` in `src/cases/shocktubes.jl`; today `atmosphere_tube_case`,
  hydrodynamic or magnetized — without a field the branch never fires and
  the hydrodynamic tube is the same bit for bit). Not the other cases: the
  shock tubes, the current sheet, the waves and the 2D/3D cases hold no
  near vacuum, and the near-vacuum rarefaction is hydrodynamic and never
  reaches the branch (its numbers below are unchanged by it); a star's
  magnetosphere (steps 10 and 12) will need it and states it in its case.
  Its cost, measured with the option set in every policy of the suite
  (2026-10-17): step 3's sweep gains fallbacks — without the cutoff 36,
  78, 103 → 35, 69, 92, and with it at `μ_max = 10` the row `Γ = 2`,
  `f = 1` goes from 1 to 4 (every row `f ≤ 0.3` stays at none); as set
  now, per case, the sweep is untouched. Writing such a state at the
  entropy ceiling with `D` kept instead (the note in PLAN.md's EOS-5) does
  not cost the sweep anything, and is Erik's call. *(Resolved 2026-10-24:
  overtaken by EntropyEOS 2.0's level 1, EOS-9, whose `S_CEILED` keeps `D`.)*
- **Measured with the real 1.1.0** (`bin/front_investigation.jl`; the same
  48 perturbed runs as before), every number the emulated "B + coll" gave,
  to every digit printed: on the uniform mesh, the refined mesh with the
  parent-centred primitive fill and with the conserved bounded fill, 0/48
  blow-ups, no fallback, largest `W` 4.8, 5.0, 5.0, injected `D`
  −1.2·10⁻⁸, −1.4·10⁻⁸, −1.4·10⁻⁸ and `τ` −7.4·10⁻⁷, −1.5·10⁻⁶,
  −1.5·10⁻⁶ in every run; two runs 10⁻¹⁵ apart 1.6·10⁻¹³, 7.6·10⁻¹³,
  2.9·10⁻¹³ apart; 12010, 27973, 28057 step-limiter writes a run; the
  hydrodynamic tube bit for bit on all three. Beyond the tube (uniform):
  ARS(4,4,3) 0/48, `W` 7.0, 23 fallbacks; `Bʸ = 10⁻²` 0/48, `W` 16; the
  ideal magnetized atmosphere 0/48 under SSP3 (`W` 4.8) and under ARS
  (`W` 4.3, where the reset-only form reached 53); 400 cells 0/48, `W`
  5.6; `μ_max = 30` 0/48, `W` 4.8. The only differences from the emulation
  are its reach and a reported bit: the stages' `|` of policy flags now
  carries bit 16 (66816 where it was 1280), and step 3's sweep, whose
  policies no longer set `collapse_to_atmosphere`, prints what it did
  before.
- **The suite** (Julia 1.13, one thread; and the clean-checkout check,
  which resolves EntropyEOS 1.1.0 from `main`): 25873 pass and one broken —
  the ARS row of the ideal magnetized atmosphere (`ars.t.injected_D ≥ 1`;
  ARS now injects 5.6·10⁻⁸ at `W` 4.1), marked `@test_broken` pending
  Erik's decision. *(Decided by Erik 2026-10-22:)* that row now asserts
  what is true — neither tableau runs away there: injected rest mass
  within 10⁻⁶ (measured −6.6·10⁻⁹ and 5.6·10⁻⁸), energy within 3·10⁻⁴
  (−1.7·10⁻⁵, −2.4·10⁻⁵), final `W ≤ 5` (4.01, 4.11) — a regression check of
  this recovery ("The partition" has the re-measured comparison); and the
  magnetized tube asserts no fallback again, on the uniform mesh and on the
  refined one of `test/refinement_cases_tests.jl` (none in any of 48
  perturbed runs on all three meshes). The suite then passes whole, 25876
  tests (Julia 1.13, one thread). What moves against the released policy: the magnetized
  tube (`W` 6.63 → 4.80, fallbacks 92 → 0, injected `5.0·10⁻⁸`,
  `1.7·10⁻⁶` → `−1.18·10⁻⁸`, `−7.4·10⁻⁷`) and its coarse-fine run
  (`−1.4·10⁻⁸`, `−1.5·10⁻⁶`); SSP3's ideal magnetized atmosphere
  (`1.56·10⁻⁹` → `−6.63·10⁻⁹`, `W` 4.83 → 4.01); and the one
  hydrodynamic case, ARS's near-vacuum rarefaction, whose undershoot the
  wrapper now puts at rest (`ρ` 2.7·10⁻¹⁰ → 1.0·10⁻¹⁰, resets 161/118 →
  131/110; its assertion, `≤ 10⁻⁹`, holds; SSP3's row is unchanged).

**EntropyEOS 2.0** *(amended 2026-10-24, EntropyEOS 2.0; tag `v2.0.0`; stage 2 of the
recovery plan, `claude/recovery-pipeline-2`; Symmetry, AMD EPYC 7532 and
7543 nodes, Julia 1.13, `bin/front_investigation.jl` — its new `beyond`
mode runs the rows beyond the tube, the 1.1 control with this file's
`beyond` copied onto the stage-1 tree, `HOT_PROJECTION=0`)*. The recovery
calls `con2prim_safe` and `apply_policy`; the atmosphere tube no longer
sets `collapse_to_atmosphere`. The rule of 2026-10-21 is unchanged: a
state whose density had to be raised to the floor is the atmosphere at
rest, `D_out ≤ max(D_in, D_atm)` — now `apply_policy`'s own rule on 2.0's
`ρ_FLOORED`, the `ρ_min` face. EOS-9 is level 1's diagnosis: the hottest
state keeps `D` and `S` on the `s_max` face (`S_CEILED`), or, where that
would put `ρ` below `ρ_min`, takes the `ρ_min` face and so the
atmosphere; no state is ever written at a density ceiling the table
cannot hold, and the option has nothing to act on (`ρ_CEILED` never
appears in any run below).

The 48 perturbed runs (as before; blow-ups with a 95% Wilson interval;
injected `D` min / median / max; fallbacks; largest `W` min / median /
max; two runs `10⁻¹⁵` apart, the largest relative difference of `D`), 2.0
with `s_atm` at the midpoint (the package's choice), 2.0 with its default
`s_atm` (the window's minimum), and 1.1 with `collapse_to_atmosphere`
(the control, reproducing the numbers above to every digit on the uniform
mesh and beyond the tube; its refined row as recorded above):

| | 2.0, `s_atm` midpoint | 2.0, `s_atm` minimum | 1.1 + collapse (control) |
|---|---|---|---|
| uniform | 0/48; −1.2e−8 (all); 0; 4.8 / 4.8 / 7.1; **1.4e−4** | 0/48; −1.2e−8; 0; 4.8 / 4.8 / 6.0; 1.7e−13 | 0/48; −1.2e−8; 0; 4.8; 1.6e−13 |
| refined, primitive fill | 0/48; −1.4e−8; 0; 5.0 / 5.4 / 11.0; **1.3e−4** | 0/48; −1.4e−8; 0; 5.0 / 5.0 / 11.8; **2.0e−4** | 0/48; −1.4e−8; 0; 5.0; 7.6e−13 |
| hydrodynamic tube (uniform; refined) | −8.3517666767e−12, `W` 4.9356884465, 11686 writes; −1.6335e−11 | −8.3661885391e−12, `W` 4.9356868767, 11686; −1.5695e−11 | −8.3517666780e−12, 4.9356884465, 11686; −1.6e−11 |
| ARS(4,4,3) | **48/48** (injected `D` 5.1e−6 … 1.1e−5, 12 runs on the laptop) | **48/48** | 0/48; −1.2e−8; 23; 7.0 |
| `Bʸ = 10⁻²` | 0/48; −2.1e−8 / −1.1e−8 / −8.7e−9; 0 / 1 / 1; 4.5 / **26** / **100** | 0/48; 0 / 1 / 1; 4.1 / **25** / **100** | 0/48; −8.5e−9; 1; 16 |
| ideal, SSP3 | 0/48; −9.0e−9 / −7.8e−9 / −7.2e−9; 0; 4.7 / 4.8 / 5.1 | 0/48; −8.1e−9 / −7.8e−9 / −1.4e−9; 0; 4.7 / 4.8 / 5.1 | 0/48; −6.6e−9; 0; 4.8 |
| ideal, ARS | **42/48** (injected 5e−7 … 9e−7 where not) ; 14 / 26 / 58 | **47/48** | 0/48; +5.6e−8; 0; 4.3 |
| 400 cells | 0/48; −1.5e−8; 0; 5.6 | 0/48; −1.5e−8; 0; 5.6 / 5.6 / 10.3 | 0/48; −1.5e−8; 0; 5.6 |
| `μ_max = 30` | 0/48; −1.2e−8; 0; 4.8 | 0/48; −1.2e−8; 0; 4.8 | 0/48; −1.2e−8; 0; 4.8 |

"Blow-up" is the harness's: injected `D > 10⁻⁶` or `|τ| > 10⁻³`; under
ARS it is the rest mass, `10⁻⁶` to `10⁻⁵` of a box holding 0.5, at a
modest `W` (5–7 in the cut-off tube), not 1.1's runaway to `10⁴`. Step 6's
tableau comparison re-measured (`bin/tableau_comparison.jl`, unperturbed):
Balsara, the resistive tube and the current sheet as before to the digits
of its table but Balsara 3 under ARS (L1 `D`, `τ`, `𝔅ʸ` 0.375, 18.7,
0.0591 → 0.360, 18.1, 0.0608, plateau 4.20% → 3.34%, resets 3/3 → 8/6, injected τ
`2.8·10⁻³` → `6.1·10⁻²`: ARS's undershoot at a strong shock, which the
reset fills, here too); the atmosphere rows SSP3 as before (hydrodynamic
`−8.4·10⁻¹²`; ideal `−7.8·10⁻⁹`, `W` 3.99; cut off `−1.2·10⁻⁸`, 0
fallbacks), ARS hydrodynamic `4.4·10⁻⁷` (1.1: `1.2·10⁻⁹`), ideal
`−2.0·10⁻⁹`, `W` 4.13, cut off `5.6·10⁻⁶` with 22 fallbacks and 33
recoveries, the rarefaction `1.5·10⁻⁶`, resets 133/116.

So **SSP3(4,3,3), the default, is unchanged on the uniform mesh** (but
the cut-off tube with the midpoint `s_atm` is no longer the same run in
every perturbation, though every one is clean), **and ARS(4,4,3) near the
vacuum regresses**: 1.1 was deterministic in every row; 2.0 puts large
negative `D` (to `−2.7·10⁻⁴` in the cut-off tube, `−7·10⁻⁵` in the
hydrodynamic one) into a few cells at the dense front at the end of a
step, which the reset fills with the atmosphere — all of the injected
rest mass. What was ruled out, by experiments on the laptop (12 perturbed
runs each, hydrodynamic and cut-off tube under ARS): the entropy faces on
the converged path (1.1's rule — `s` clamped, the root's `ρ` and `w` kept
— emulated in-process changes nothing), `seed_passes` (3 changes
nothing), `NOT_CONVERGED` (none occurs), a discontinuous recovery (none of
the stage limiter's recoveries jumps under a `10⁻¹³` perturbation, in
either version), and the repairs themselves (the states repaired above the
trigger are the same kinds and counts, 25 `S_FLOORED` and 22 floored to
the atmosphere against 1.1's 28 and 25, and 1.1's `con2prim_safe` on 2.0's
inputs returns the same conserved state to `10⁻⁸`). Bisected upstream: the
stage-1 tree on EntropyEOS 8879f4f (1.1's API, the commit before 2.0's
projection) is 1.1 to every digit; this tree on ab8eb9b ("Implement the
two-level projection") already regresses. In the magnetized ideal tube
under ARS the `S_CEILED` writes add to it: keeping `S` while lowering τ
turns heat into bulk motion, `W` 1.08 → 2.99 for one write (75 writes, a
median 1.11 times the unprojected root's `W`, at most 2.77), where 1.1
kept the rapidity. *(Decided by Erik 2026-10-24:)* accepted for now —
ARS is not the default and SSP3 is clean — and being isolated separately
(a TreeGRRMHD level 2 of its own cannot undo level 1's choice of face).
*(Amended 2026-10-26:)* isolated, and fixed by EntropyEOS 2.1's
`inside_window` (below); no longer accepted, gone.

**The ARS(4,4,3) regression: cause, mechanism and fix** *(amended
2026-10-26, EntropyEOS 2.1; the investigation of 2026-10-24 on
`claude/recovery-pipeline-2` and `-3`, Symmetry job 581987)*.

- **Cause.** Since ab8eb9b (2.0's two-level projection) `con2prim_safe`
  accepts a converged root unflagged when it lies inside every face *to the
  slack* — 10⁻⁷ of the window's width for `s` at `Float64` — and returns it
  as it is; 1.1 clamped the same root onto `s_min` and flagged it
  `S_FLOORED`. The reproducer (ideal gas Γ = 5/3, `s` window (1, 40), `ρ_min =
  10⁻⁸`, the right-hand side's cold re-solve of a cell the stage limiter had
  just written onto the `s_min` face): 2.0 returns `s = 0.999999784528953`,
  5.5·10⁻⁹ of the window below `s_min`, with `policy_flags == 0`, where 1.1
  returned `s = 1`, `S_FLOORED` (`ρ`, `w` the same). Thermal energy is about
  10⁻¹⁰ of τ there, so `s` is fixed only to the solve's conditioning. The
  reconstruction tested face states against the *exact* window.
- **Mechanism.** A cell in the slack band is a local minimum of `s`; MC gives
  it zero slope, so its face values are its own value and fail the exact
  test, and the positivity fallback, which substitutes the cell's value,
  cannot repair them: the whole fluid group of that face goes to first order.
  Under ARS(4,4,3) the switch happens at stage 4, whose explicit weight is
  `b̃₄ = −7/4`: the first-order face raises the `D` flux into the neighbouring
  cell, and the negative weight subtracts it from a cell holding about
  `ρ_min`. Every large negative end-of-step `D`, down to `−7·10⁻⁵`, lay one
  cell downstream of a stage-4 slack cell, and together they accounted for
  all of the injected mass. Under SSP3 every weight is positive: no blow-up,
  but the same switch made the runs sensitive at rounding (the uniform
  cut-off tube's perturbed runs 1.4·10⁻⁴ apart, the `Bʸ = 10⁻²` tube at
  `W` 26–100).
- **The emulation** (job 581987, cn112, the stage-2 tree with the exact test
  widened by a slack of 0 or 10⁻⁷ of the window, 48 perturbed runs a row): at
  slack 0 (the code as it was) the faces the window test sent to the fallback
  numbered 1216 (cut-off tube, ARS), 27331 (ideal, ARS), 259 (hydrodynamic,
  ARS), 85 (cut-off tube, SSP3), 5683 (ideal, SSP3) and 923 (`Bʸ = 10⁻²`) over
  the 48 runs, with 48/48 and 42/48 blow-ups in the first two and the
  hydrodynamic ARS tube injecting `1.2·10⁻⁹ … 8.9·10⁻⁷` (median `4.3·10⁻⁷`); at
  slack 10⁻⁷ none, every row deterministic, the cut-off and ideal ARS rows
  0/48, hydrodynamic ARS `1.2·10⁻⁹` in every run (1.1's), `Bʸ = 10⁻²` SSP3
  1.1's `−8.5·10⁻⁹` and `W` 16.
- **The fix**: the reconstruction tests validity the way `con2prim_safe`
  does, `inside_window` (EOS-11; "Reconstruction"). Clamping the root onto
  the window, upstream or here, was rejected upstream: the primitives would
  no longer reproduce the returned conservatives to the tolerance.

*Re-measured 2026-10-26* (`bin/front_investigation.jl`'s harness, the same
48 perturbed runs; Symmetry cn082, AMD EPYC 7532, one thread; EntropyEOS
2.2.0 with `inside_window`; the reconstruction's test instrumented in-process
to count its fallbacks). Columns as in the table above: blow-ups; injected
`D`; fallbacks; largest `W`; two runs `10⁻¹⁵` apart. "(all)" where all 48
runs agree to the digits printed:

| | 2.2, `inside_window` | 2.0, stage 2 (above) | emulation, slack 10⁻⁷ (job 581987) |
|---|---|---|---|
| uniform | 0/48; −1.2e−8 (all); 0; 4.8 (all); 3.0e−14 | 0/48; −1.2e−8; 0; 4.8 / 4.8 / 7.1; **1.4e−4** | 0/48; −1.2e−8 (all); 0; 4.8 (all) |
| refined, primitive fill | 0/48; −1.4e−8 (all); 0; 5.0 (all); 7.4e−13 | 0/48; −1.4e−8; 0; 5.0 / 5.4 / 11.0; **1.3e−4** | — |
| hydrodynamic tube (uniform; refined) | −8.3517666767e−12, `W` 4.9356884465, 11686 writes; −1.6334932204e−11 | the same | — |
| ARS(4,4,3) | 0/48; −1.2e−8 (all); 23; 7.0 (all) | **48/48** | 0/48; −1.2e−8 (all); 23; 7.0 |
| hydrodynamic, ARS(4,4,3) | 0/48; 1.2e−9 (all); 0; 5.0 | 0/48; 1.2e−9 / 4.3e−7 / 8.9e−7 (job 581987's slack 0) | 0/48; 1.2e−9 (all); 0; 5.0 |
| `Bʸ = 10⁻²` | 0/48; −8.5e−9 (all); 1; 16 (all) | 0/48; −2.1e−8 / −1.1e−8 / −8.7e−9; 0 / 1 / 1; 4.5 / **26** / **100** | 0/48; −8.5e−9 (all); 1; 16 |
| ideal, SSP3 | 0/48; −7.6e−9 (all); 0; 4.8 (all) | 0/48; −9.0e−9 / −7.8e−9 / −7.2e−9; 0; 4.7 / 4.8 / 5.1 | 0/48; −7.6e−9 (all); 0; 4.8 |
| ideal, ARS | 0/48; −5.0e−9 (all); 0; **100** (all) | **42/48** | 0/48; −5.0e−9 (all); 0; 100 |
| 400 cells | 0/48; −1.5e−8 (all); 0; 5.6 (all) | 0/48; −1.5e−8; 0; 5.6 | — |
| `μ_max = 30` | 0/48; −1.2e−8 (all); 0; 4.8 (all) | 0/48; −1.2e−8; 0; 4.8 | — |

The reconstruction's window test sent **no** face to the fallback in any row;
the faces it admitted that the exact test would have rejected — the slack band
— numbered 1254 (ARS), 23281 (ideal, ARS), 298 (hydrodynamic, ARS), 67 and 69
(the cut-off tube under SSP3, in the batch and in the uniform matrix), 93
(refined), 5248 (ideal, SSP3), 603 (`Bʸ = 10⁻²`), 50 (400 cells) and 12
(`μ_max = 30`), close to the emulation's slack-0 fallbacks. **Every row is the
emulation's to every digit it printed**, step-limiter writes and `S_FLOORED`
counts included (12021 and 17 under ARS, 11687 and 42 ideal under ARS, 11737
and 5 hydrodynamic under ARS, 12010 and 3, 12028 and 9, 12034 and 10): the
confined search changes nothing here (no state comes near `ρ_max`), and the
step cache, which the emulation's tree did not have, nothing at these digits.
So every row is deterministic again, as on 1.1, and the uniform and refined
cut-off tube, ARS(4,4,3) (23 fallbacks, `W` 7.0), the hydrodynamic tube under
ARS (`1.2·10⁻⁹`), `Bʸ = 10⁻²`, 400 cells and `μ_max = 30` are 1.1's to the
digits printed (the control column of the table above). Two rows still differ from 1.1: the ideal magnetized atmosphere under
SSP3 (`−7.6·10⁻⁹` against `−6.6·10⁻⁹`), and under ARS, which injects
`−5.0·10⁻⁹` against 1.1's `+5.6·10⁻⁸` but reaches the `w` cap (`W` = 100) in 3
cell-steps of every run, where 1.1 peaked at `W` 4.3. That is not the slack: it
is the same with the emulation, and the cell recoveries that differ between
the versions agree on identical inputs, so it is a different trajectory — a
candidate is 2.0's `S_CEILED` keeping `S` while lowering τ (above). Not
chased; **for Erik**. *(Resolved 2026-11-02: investigated 2026-10-30 and
accepted by Erik, below.)* The suite's assertion on that row (`W ≤ 5` at the end,
injected mass within 10⁻⁶) holds: its largest `W` at the end is 4.00.

**The ideal magnetized atmosphere's `W = 100`** *(investigated 2026-10-30
on EntropyEOS 2.2.1; accepted, decided by Erik 2026-11-02)*. Both
differences from 1.1 are EntropyEOS 2.0's rule for a converged root hotter
than `s_max` (its HISTORY, 2026-10-23, "Which faces are active"): 2.x
projects it onto the `s_max` face keeping `D` and `S` and lowering only τ,
where 1.1 clamped `s` at the root's `ρ` and `w`, removing energy and momentum
together. In a hot, tenuous cell the momentum is carried by enthalpy (`h` 10
to 30); on the `s_max` edge `h` falls with `ρ`, so keeping `S` raises `W`
until `ρ` reaches `ρ_min`. Shown by running both versions in one process
(8879f4f renamed beside 2.2.1, chosen per recovery), 48 perturbed runs per
variant, each variant deterministic: 1.1's answer only where 2.2.1 flags
`S_CEILED` is 1.1 to every digit under ARS and SSP3, and 2.2.1's answer only
there is 2.2.1 to every digit; 1.1's answer on all 397,341 unflagged
recoveries changes nothing (so not the seed, the tolerance or the confined
search), nor on the `W_CEILED` and `S_FLOORED` ones, nor the reconstruction
(no fallbacks), nor `s_atm`. Under ARS(4,4,3) the `S_CEILED` writes turn
removed heat into bulk motion in the precursor (`|S_f|/(τ_f + D)` at p90 0.74
against 0.55); stage 4's predictors at the front exceed the light cone
(`|S| > τ + D`) about 100 times a run in both versions, and on 2.x a few more
of them fall in denser cells, where the `s_min` face (`S_FLOORED`, the same
answer in both) writes `W ≈ |S|/D`, up to the cap — one cell, one step, `W`
below 5 the next. The injected mass under ARS (1.1's `+5.6·10⁻⁸` → `−5.0·10⁻⁹`)
is EOS-9's no-collapse instead, as measured 2026-10-22. Under SSP3 nothing
amplifies it: 12018 atmosphere writes inject `−7.6·10⁻⁹` against 1.1's 12008
and `−6.6·10⁻⁹`.

Not worse by any integral measure: the final state at `t = 0.4` against
1600- and 3200-cell references is within 1.5% of 1.1's in L1 of `D`, `S` and
τ under both tableaus (ARS against 1.1's 3200-cell run: `D` 4.110·10⁻³ against
4.117·10⁻³), its largest `W` 4.00 against 4.12 (ARS), the injected energy and
momentum within 2–10%. Nor is 1.1's `W` 4.3 a property of 1.1: it is that
of 200 cells. Largest `W` and cell-steps with `W > 20`:

| | 200 cells | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|
| ARS, 2.2.1 | 100, 3 | 100, 4 | 100, 32 | 100, 153 | — |
| ARS, 1.1 | 4.3, 0 | 100, 7 | 100, 40 | 100, 163 | — |
| SSP3, 2.2.1 | 4.8, 0 | 4.8, 0 | 4.8, 0 | 100, 2 | 100, 537 |
| SSP3, 1.1 | 4.8, 0 | 4.8, 0 | 5.9, 0 | 25, 2 | 100, 164 |

2.x brings the spikes in about one doubling of resolution earlier and up to
three times as often; the hydrodynamic and cut-off tubes have none at 1600
cells (largest `W` 8.0 and 12.7). Not pursued: a TreeGRRMHD level 2 sending
`S_CEILED` states to the atmosphere still spikes (`W` 27; 65 if limited to
`ρ < 1.1 ρ_min`). A lead for later, single laptop runs only: `atm_trigger` 2
or 10 removes the spikes in both versions, and at 10 the two are bit for bit
the same (ARS `W` 4.96, `−4.0·10⁻⁹`; SSP3 `W` 3.99) — a policy change,
untested elsewhere.

**Investigation (2026-10-22)** *(branch `claude/no-collapse-to-atmosphere`,
an experiment for Erik, not decided — *(resolved 2026-10-24: overtaken by
EOS-9 in EntropyEOS 2.0, and no case sets `collapse_to_atmosphere`)*;
`bin/hot_projection.jl`,
`bin/endpoint_measure.jl`, `bin/front_investigation.jl`; EOS-9 in PLAN.md,
rewritten the same day after Erik's review in six questions)*: can the
atmosphere tubes do without `collapse_to_atmosphere`, an excision choice
meant for collapsing stars? **Yes, with one EntropyEOS change, EOS-9:** step
4's two endpoint branches made edge-consistent and guarded, the hot one
projecting onto the entropy ceiling with `D` and the momentum kept (up to
the rapidity cap) and the density clamped into the box and reported —
never `pol_collapse`. With it and the option off, every measured row is as
clean as with the option and the hydrodynamic tube is bit for bit; without
it the ideal magnetized atmosphere and `Bʸ = 10⁻²` blow up in every run.

*What 1.1.0 does* (origin/main 43923c7). When the solve fails, step 4 solves
the momentum at the two edges of the window `sr_ref = srange(D/cosh w_ref)`
(`w_ref` the failed solve's rapidity). τ below the coldest state there
(`f₂ > 0` at `s_min`): the state is written at `ρ₁ = D/cosh w₁`, clamped
into `[ρ_atm, ρ_ceiling]` (reported since 1.1.0), at `s_min` of
`srange(ρ₁)`; τ above the hottest (`f₂ < 0` at `s_max`): `pol_collapse`,
`ρ_ceiling` at `s_max`, unless `collapse_to_atmosphere`. Step 1 sends
`D > D_max` and `τ − B² > τ_max` to `pol_collapse` before any solve.
`D_max = ρ_ceiling W_cap`, `τ_max = max ρh(s_max) W_cap²` over a 5×3 grid:
`10⁸` and `1.5·10²⁰` for the shock gas, `10⁶` and `1.6·10¹¹` for the waves'
gas, 0.1 and 21 for EntropyEOS's test `HybridEOS`, `10¹⁷` and `1.2·10¹⁹` in
g/cm³ for the synthetic table (0.16 and 19.6 through `GeometricEOS`) — no
state of this package's cases comes near them.

*Measured on the synthetic table* (`bin/endpoint_measure.jl table`; its
window depends on `ρ`, the analytic EOSes' does not; random valid states
across the box, hydrodynamic and MHD, `τ` scaled down or up; about 4000
inputs reaching step 4 per branch and mode, the same in g/cm³ and through
`GeometricEOS`):

| | cold branch (hydro / MHD) | hot branch (hydro / MHD) |
|---|---|---|
| edge at `ρ₁` against at `ρ_ref`, fraction of the window (median / max) | 1.5·10⁻⁵ / 2.1·10⁻⁵, both | 2.1·10⁻³ / 0.11; 2.1·10⁻³ / 0.27 |
| diagnosis flips at `ρ₁`; with a solve 1000× tighter and 10× longer | 0; 0 | 0; 0 |
| endpoint solve: converged / at the precision floor / capped / non-finite | all / 0 / 0 / 0 | 99.1% / 0.9% (≤ 8 tol) / 0 / 0; 99.8% / 0.2% / 0 / 0 |
| 1.1.0: output, `\|ΔS\|/\|S\|` (median / max) | at `s_min(ρ_out)`, 1.7·10⁻⁸ / 3·10⁻⁸ | `ρ_ceiling` for every input |
| first EOS-9 emulation (`sr_ref`'s ceiling, then projected): `\|ΔS\|/\|S\|`; `s_out` below the edge | — | 0.43% / 5.5%; 0.33% / 5.6%, `s_out` up to 0.27 of the window below the edge (MHD) |
| fixed point `ρ ↦ s_edge(ρ) ↦ w ↦ D/cosh w`: passes; `\|ΔS\|/\|S\|` max | 4; 6.5·10⁻¹², 1.1·10⁻¹¹ | 13–17; 3.3·10⁻¹⁰, 1.0·10⁻⁹ |
| edge solve (one root find, `s = s_edge(D/cosh w)` inside): evaluations; `\|ΔS\|/\|S\|` max; agrees with the fixed point | 20–30; 1.3·10⁻¹⁰, 3.6·10⁻¹¹; all | 20–30; 4.6·10⁻¹⁰, 6.7·10⁻¹⁰; all |
| EOS-9 as rewritten (emulated): `s_out` at the edge; `\|ΔS\|/\|S\|` median / max; flags | yes; 7·10⁻¹⁶ / 1.3·10⁻¹⁰; `S_FLOORED`, `\| ρ_FLOORED` on 3 and 5 | yes; 2·10⁻¹⁵ / 6.7·10⁻¹⁰; `S_CEILED`, `\| ρ_CEILED` on 17 and 9 |

Step 1's `τ_max` stopped 8% (hydro) and 11% (MHD) of the inputs aimed at the
hot branch, writing `ρ_ceiling` for states the table represents. In the
tubes (`bin/endpoint_measure.jl tubes`) every endpoint solve converged — 584
diagnoses in the cut-off tube, 1042 at `Bʸ = 10⁻²`, 875 and 1318 in the
ideal atmosphere under SSP3 and ARS — and no diagnosis flipped with the
longer solve. The endpoint solves that missed the tolerance on the table
all sat at the residual's floating-point floor (`|f₁|/cosh w ≤ 8 tol`, 93
of 166 out of iterations with the bracket already collapsed, the rest
collapsed earlier): the guards accept a bracket collapsed around a sign
change as a root.

*The formulation chosen*, the edge solve: one root find gives the diagnosis
and the projection at the same density, the momentum to the solver's
tolerance, `s` exactly at the edge, in 20–30 evaluations, where the fixed
point needs 4 (cold) to 17 (hot) full solves for the same answer; the
first emulation, at `sr_ref`'s ceiling, missed the momentum by up to 5.6%.
Erik's decisions are in the request: the density clamped and reported in
the hot branch as in the cold one, never the atmosphere (Q3 (b)); flags
that describe the returned state both ways, the atmosphere replacing a
state carrying only `ATMOSPHERE` and its cause (Q4); `check_con_state`
sharing the diagnosis (Q5); step 1 on `D_max` only, the `D_max` collapse
reporting `ρ_CEILED` (Q6 (a)).

*Measured in the tubes* (48 perturbed runs each; the recovery wrapper's
atmosphere at rest on in every column; blow-ups; largest `W` median/max;
fallbacks; injected rest mass; two runs 10⁻¹⁵ apart):

| | option on, 1.1.0 (470e8d9) | **option off, EOS-9 as rewritten** | option off, first EOS-9 emulation | option off, 1.1.0 (control) |
|---|---|---|---|---|
| uniform | 0/48; 4.8; 0; −1.2e−8; 1.6e−13 | same; 4.4e−14 | same; 1.6e−13 | same |
| refined, primitive fill | 0/48; 5.0; 0; −1.4e−8; 7.6e−13 | same; 2.4e−12 | same | same |
| refined, conserved fill | 0/48; 5.0; 0; −1.4e−8; 2.9e−13 | same; 1.6e−13 | same | same |
| hydrodynamic tube (3 meshes) | −8.4e−12, −1.6e−11, −1.5e−11 | bit for bit | bit for bit | bit for bit |
| ARS(4,4,3) | 0/48; 7.0; 23; −1.2e−8 | same | same | same |
| `Bʸ = 10⁻²` | 0/48; 16; 1; −8.5e−9 | same (8928 hot projections, all floored and capped) | same | **48/48** |
| ideal, SSP3 | 0/48; 4.8; 0; −6.6e−9 | 0/48; 4.8; 0; −6.7e−9 (8976, 8448 floored) | same | **48/48** |
| ideal, ARS | 0/48; 4.3; 0; +5.6e−8 | 0/48; 4.3; 0; −4.5e−9 (17,856, 17,040 floored) | same | **48/48** |
| 400 cells; `μ_max = 30` | 0/48; 5.6; 0 — 0/48; 4.8; 0 | same | same | same |
| step 3's sweep without the cutoff: fallbacks Γ = 4/3, 5/3, 2 | 36, 78, 103 | **33, 74, 90**; the kernel test 48 of 48 at `Float64` and `Float32` (47) | 33, 74, 90 | 36, 78, 103 |
| step 3's sweep with the cutoff | 0 at `f ≤ 0.3`; 1 at Γ = 5/3 and 2, `f = 1` | same | same | same |
| suite (Julia 1.13, one thread) | 25876 pass | **25940 pass** (with this branch's tightened recovery test) | 25876 pass | 3 fail: the ideal magnetized atmosphere blows up |

In the cut-off tube, at 400 cells, at `μ_max = 30` and under ARS the hot
branch never fires: there the option was already idle. Where it fires, the
rewritten projection floors the density (the cell's `D/cosh w` lies below
`ρ_atm` nearly always) and reports it; this package's wrapper then writes
the atmosphere at rest — the same states the first emulation sent to the
atmosphere inside EntropyEOS, so the tubes' numbers agree to the digits
printed, but the decision is now ours. On this branch the plain suite
(without the emulation) fails those three assertions: it is an experiment,
and EOS-9 is what it waits for.

**Investigation (2026-10-16)** *(branch `claude/front-investigation`, not a
PLAN.md step; `bin/front_investigation.jl`, and the 2026-10-17 sweep's
`bin/policy_variants.jl`, removed 2026-10-21 with the decision above —
commit ec4f044 has it; nothing in `src/` changed until the decision)*: why the magnetized
atmosphere tube (rung 1b; `Bʸ = 10⁻³`, `Constant(10³)` cut off at
`μ_max = 10`, 200 cells, SSP3(4,3,3)) is chaotic at rounding level, with
`W` spiking to 30–100 at the front and fallbacks there, while the
hydrodynamic tube is not. The harness is `bin/ghost_robustness.jl`'s (its
48 perturbed initial states, and its numbers to every digit for the
baseline).

- **The first cause: the field's truncation error lands in the fluid's
  share.** `τ` and `S_i` are totals and the recovery subtracts
  `½(E² + B²)` and `ε E × B` of the cell's own `E` and `B`; the scheme does
  not conserve the quadratic field energy of cell values, so the
  numerical dissipation of `E` and `B` becomes fluid energy and momentum —
  as kinetic energy dissipated by a total-energy hydro scheme becomes
  heat. For a field moving with the front the deposit is nearly null
  (`δτ ≈ |δS|`), and where the fluid's own energy is small against the
  field's it *is* the fluid's share. Ahead of the dense front the
  light-cone HLLE flux leaves cells a little above the atmosphere trigger
  (`D ≈ 2–50 ρ_atm`) inside the electrovacuum (`μ = B²/D ≈ 30–500`,
  `σ = 0`) where the front piles up its field (`Bʸ` to 7·10⁻³,
  `Eᶻ ≈ −vBʸ`). Over the unperturbed run's 1894 such cell-steps
  (`1.02 ρ_atm < D < 10⁻⁴`) the share's `|S_f|/(τ_f + D)` has median 0.979
  (max 0.9999) and `s` median 38.9, up to the window's ceiling 40 (456
  `S_CEILED` writes), `e_em/τ_f` median 2.0, max 8.7·10⁴; the hydrodynamic
  tube has 370 such cell-steps at median 0.818 and `s ≤ 36.4`. The
  mechanism alone: `em_wave_case` (`σ = 0`, whose exact solution leaves the
  fluid at rest) with `½b² = 5·10⁻⁷` over a gas at `ρ = 10⁻⁷`, `s = 1`:
  after one period at 50 cells `s` is at the window's ceiling (3/2)
  everywhere and `|vˣ|` reaches 0.25 (200 cells: 1.1·10⁻³, `s` ≥ 1.28; 800
  cells: 7·10⁻⁵, `s` ≥ 1.18); at `ρ = 10⁻³` `vˣ ≤ 6·10⁻⁶`, at `ρ = 1`
  10⁻⁹.
- **What turns it into spikes, mass and chaos: EntropyEOS's repairs of an
  inadmissible share.** (a) `τ` below the coldest state at the momentum
  (`S_FLOORED`, the share crossed the light cone): the policy keeps the
  momentum at `s_min` and clamps `ρ = D/cosh w` *up* to `ρ_atm`, so the
  written state has `W = |S_f|/ρ_atm` (to the cap) and `D = ρ_atm W`.
  The `W` spikes are these states and their next steps (`D/W ≈ ρ_atm`; 23
  of the unperturbed run's 34 cell-steps with `W > 20` are that step's
  write), and its 115 `S_FLOORED` writes inject `+4.8·10⁻⁸` of rest mass
  and `+1.8·10⁻⁶` of energy, more than the run's total (`4.5·10⁻⁸`,
  `1.1·10⁻⁶`; the atmosphere and entropy-ceiling writes take some back).
  The same state 1% lower in `D` is reset to the atmosphere at rest.
  (b) `τ` above the hottest state with the solve failed: the policy
  diagnoses a collapse and writes `ρ_ceiling = 10⁶` — every blow-up traced
  (`D ≈ 5·10³ = 10⁶ h`), e.g. at `D = 2.3·10⁻⁸`, `τ_f = 3.9·10⁻⁶`,
  `e_em = 8.6·10⁻⁵`, an advected `s` of −4.7.
- **The chaos is numerical, one discrete switch.** Two runs 10⁻¹⁵ apart
  agree to 10⁻¹² for 32 steps (rounding growth, nothing exponential);
  at step 33 (`t = 0.066`) the stage-4 predictor of the cell at
  `x = 0.0675` is below the trigger in one run and `S_FLOORED` in the
  other; one run writes the atmosphere, the other `W = 7.2` and seven
  times the mass, and the relative difference of `D` jumps from 1.1·10⁻¹²
  to 0.86 in that step. The front's foot then differs at O(1) for the rest
  of the run (`bin/front_investigation.jl diverge`).
- **The fallbacks are a symptom.** All 100 of the unperturbed run are
  taper cells beside that foot, `μ ∈ [6.9, 10]`, at
  `f = |E★ + v × B|/|B| ∈ [0.71, 0.96]` — beyond step 3's sweep — every one
  with repaired iterates; scanning `u = Wvˣ` (W ≤ 200), none has an
  admissible root (7 have no admissible share at any velocity, 93 have
  admissible shares and no root): the stage has no solution, and the
  lagged velocity is the right answer. Re-solved under B + hot they still
  fall back; they do not arise once (a) is consistent.
- **Not the cause:** the `q v` current (`q = ∇·E ≡ 0` here, `E = Eᶻ(x)`;
  removing it changes nothing, bit for bit); the ghost fill (the uniform
  mesh too); resolution (400 cells: 2/48 blow-ups, fallbacks 0 / 197 /
  547).

Controls, uniform mesh, 48 runs each, the released policy (blow-ups;
largest `W` min / median / max; fallbacks):

| control | blow-ups | max `W` | fallbacks | chaotic |
|---|---|---|---|---|
| the tube (baseline) | 0/48 (0–7.4%) | 29.8 / 81.8 / 100 | 0 / 26 / 238 | yes |
| `Vacuum` everywhere | 0/48 | 4.94 | 0 | no: the hydrodynamic tube to every digit (no current, no pulse) |
| `Bʸ = 0` under the cutoff | 0/48 | 4.94 | 325* | no |
| `IdealConductor` | 0/48 | 18.6 | 0 | no |
| `Bʸ = 10⁻⁴` | 0/48 | 4.9 | 366* | no |
| `Bʸ = 10⁻²` | 47/48 | 100 | 0 | — |
| `ρ_atm = 10⁻⁶` / `10⁻⁴` | 0/48 | 4.8 / 3.9 | 367* / 580* | no |
| `μ_max = 30` / `100` | 0/48 | 19.9 / 32 / 100; 15 / 21 / 37 | 6 / 34 / 244; 325* … 344 | yes |
| ARS(4,4,3) | 15/48 (20–45%) | 75 / 100 / 100 | 14 / 42 / 358 | yes |
| ideal, ARS(4,4,3) | 48/48 | — | — | — |

\* Atmosphere predictors where σ > 0: a predictor below the trigger is
repaired, its share never equals the totals', and the stage cannot
converge (all 325 at `Bʸ = 0`, where `E★ = 0` and the fallback's `E` is
`E★`: harmless, but counted). Where `B²/ρ_atm < μ_max/10` the atmosphere
conducts under the cutoff. A predictor in the atmosphere could take
`OHM_SIGMA_ZERO` (proposed, for Erik).

Candidates (48 runs each; uniform; refined `[0, 0.2)` with this tree's
parent-centred primitive fill; refined with the conserved bounded fill,
`git archive 7605986`): blow-ups, injected `D` (median; one value where all
48 agree to the digits printed), fallbacks (median / max), largest `W`
(median / max), and what the hydrodynamic tube does:

| candidate | uniform | refined, primitive fill | refined, conserved fill | hydrodynamic tube |
|---|---|---|---|---|
| baseline | 0/48; 3.9e−8; 26 / 238; 82 / 100 | 0/48; 3.6e−8; 72 / 279; 92 / 100 | 3/48; 3.4e−8; 49 / 425; 87 / 100 | — |
| B + hot (EOS-5 as first proposed; withdrawn, see below) | 0/48; **−1.2e−8**; 0; **4.8** | 0/48; −1.4e−8; 0; 5.0 | 0/48; −1.4e−8; 0; 5.0 | bit for bit |
| Breset + coll (B in our reset, `collapse_to_atmosphere`) | as B + hot | as B + hot | as B + hot (2026-10-17) | bit for bit |
| `collapse_to_atmosphere` alone | as baseline | as baseline | 0/48, else as baseline | bit for bit |
| `w_cap` at `W = 10` | 0/48; 1.7e−8 (1.3 … 2.0); 0; 10 | 0/48; 1.3e−8; 0 / 1; 10 | **13/48**; 1.2e−8; 0 / 1; 10 | bit for bit |
| `atm_trigger = 10` | 0/48; −1.1e−8; 0; 4.8 | 0/48; −1.4e−8; 0; 4.9 | 0/48; −1.3e−8; 0; 5.0 | moves: −1.2e−8, `W` 4.87 |
| magnetization ceiling `B²/D ≤ 100` (mass added at rest) | 0/48; +7.8e−8; 0; 5.9 | 0/48; +1.2e−7; 0; 19.3 / 19.5 | 0/48; +1.2e−7; 0; 18.8 | bit for bit |
| the atmosphere where `B²/D > μ_max` | 0/48; −1.1e−8; 0; 4.8 | 0/48; **−3.0e−7**; 0; 4.9 | 0/48; **−4.6e−7**; 0; 4.9 | bit for bit |
| fixed-`s` recovery where `e_em > τ_f` (EOS-3's route) | 0/48; −1.2e−8; 0; 4.8 | 0/48; −1.4e−8; 0; 5.0 | 0/48; −1.4e−8; 0; 5.0 | bit for bit |

(A ceiling at `B²/D ≤ 10 = μ_max` went non-finite in every run; not
pursued.) Every row but the baseline,
`collapse_to_atmosphere` alone and the `W` cap is the same run in all 48
perturbations: two runs 10⁻¹⁵ apart differ by 1.6·10⁻¹³ (uniform),
7.6·10⁻¹³ (refined) and 2.9·10⁻¹³ (conserved) of `D` under B + hot — and by
0.8, 4.2 and 3.7 times `D` (relative, in a cell of the foot) under the
released policy; 1.4·10⁻¹⁰ under the ceiling.

**EOS-5, as first proposed here, is withdrawn** (Erik, 2026-10-17): its
criterion, "no policy repair creates rest mass" (`D_out ≤ D_in`),
contradicts EntropyEOS's design — a well-defined range of allowed states,
inside which `con2prim`/`prim2con` always succeed, with every state
outside projected back into it, very low densities raised back to the
atmosphere — and broke it itself for `D_in < ρ_atm`. The criterion
consistent with that design is `D_out ≤ max(D_in, D_atm)`. The defect is
narrower: the `S_FLOORED` branch raises `ρ` to `ρ_atm` but keeps the
near-light-like rapidity `w₁`, writing `(ρ_atm, W ≈ |S|/ρ_atm)` at the
edge of the valid range. Its two parts stay as notes in PLAN.md's EOS-5;
the first part is variant B below (EOS-6); the second is "hot". The
2026-10-16 measurements under that emulation are the rows "B + hot"
below (identical code), and the suite's verdict on it (one failure, the
ARS row of the ideal magnetized atmosphere) stands.

**The sweep of 2026-10-17** *(Erik's variants; proposed reading below,
Erik decides)*. `bin/policy_variants.jl` emulates every variant in-process
— it redefines `EntropyEOS.default_policy` and `con2prim_safe`, so it
reaches every recovery: the right-hand side's, the stage solve's
predictor and iterates, the reset's — and `bin/front_investigation.jl
matrix MESH VARIANT` runs it (the conserved fill from `git archive
7605986` with both files copied in). The variants:
- **A`k`**: `atm_trigger = k` (EntropyEOS's existing over-correcting
  reset: the atmosphere where `D < k ρ_atm`; default 1.01);
- **B**: where the `S_FLOORED` branch would raise the density to `ρ_atm`,
  the atmosphere at rest (EOS-6) — in-process; **Breset**: the same in
  our reset only (a repair holding more rest mass than the cell is
  written as the atmosphere), which does not reach the stage solve;
- **C**: the entropy reset in the atmosphere — a state whose density,
  after the policy's projection, lies in the band `ρ < atm_trigger ρ_atm`
  takes the atmosphere's entropy, and `s_atm` is set cold (the window's
  floor). Today EntropyEOS's `default_policy` leaves `s_atm = NaN`, which
  `policy_atmosphere` takes as the *middle* of the physical window (20.5
  of the shock gas's 1 … 40, not cold); every atmosphere write already
  sets `s` (and `D s`) to it, so advected hot entropy survives only in
  cells above the trigger, and the `S_FLOORED` branch writes `s_min`;
- **coll**: `collapse_to_atmosphere = true` (an existing option);
  **hot**: "τ above the hottest state" goes to `s_max` with `D` kept.

Per mesh (48 runs; uniform; refined `[0, 0.2)` with the parent-centred
primitive fill; refined with the conserved bounded fill), each cell:
blow-ups (95% Wilson where not 0/48) · fallbacks median / max · largest `W`
median / max · injected `D` and `τ` (the reset's net writes over the run)
median / max · how far two runs 10⁻¹⁵ apart drift (largest relative `D`
difference at `t = 0.4`) · the reset's step-limiter writes per run
(median). One value where all 48 agree to the digits printed.

| variant | change needed | uniform | refined, primitive fill | refined, conserved fill | hydrodynamic tube | suite (25405 tests) |
|---|---|---|---|---|---|---|
| released (A1.01) | — | 0/48 · 26/238 · 82/100 · 3.9e−8/7.6e−8, 8.5e−7/3.5e−6 · **0.8** · 10876 | 0/48 · 72/279 · 92/100 · 3.6e−8/5.1e−8, 1.3e−7/1.3e−6 · **4.2** · 25696 | **3/48** (2.1–17%) · 49/425 · 87/100 · 3.4e−8/5.6e−8, 8.6e−8/1.7e−6 · **3.7** · 25877 | `D` −8.4e−12, `W` 4.936, 11686 writes (refined −1.6e−11, 5.104; conserved −1.5e−11, 5.116) | pass |
| A2 | option | 0/48 · 0/1 · 22/61 · 7.8e−9/1.4e−8, −5.0e−7/−1.7e−7 · 0.82 · 11984 | 0/48 · 0/33 · 24/100 · −3.2e−9/3.0e−9, −1.4e−6/−8.4e−7 · 8.9e−3 · 27976 | 3/48 (2.1–17%) · 0/20 · 26/100 · −3.0e−9/8.2e−9, −1.4e−6/−5.9e−7 · 7.3e−4 · 28080 | moves: −2.7e−9, `W` 4.856 (refined −9.5e−9) | 3 fail: the SSP3 ideal magnetized atmosphere **blows up** (4.6e3); the two coarse-fine atmosphere assertions |
| A2.5 | option | 0/48 · 0/0 · 12/100 · −5.6e−9/1.1e−8, −6.8e−7/4.6e−7 · 4.5 · 12022 | — | — | moves: −8.1e−9, `W` 4.865 | — |
| **A3** (= A5 = A10 = A20, every digit) | option | 0/48 · 0 · 4.8 · −1.1e−8, −7.3e−7 · 4.7e−14 · 12031 | 0/48 · 0 · 4.9 · −1.4e−8, −1.5e−6 · 2.8e−13 · 28036 | 0/48 · 0 · 5.0 · −1.3e−8, −1.5e−6 · 8.1e−14 · 28143 | **moves**: −1.17e−8, `τ` −4.1e−8, `W` 4.869, 12031 writes (refined −1.41e−8, 5.048; conserved −1.40e−8, 5.046) | 3 fail: the ARS ideal row (5.1e−9, not ≥ 1); the coarse-fine hydrodynamic injection (−1.4e−8, not ≤ 1e−10); the width-4 control (`τ` −7.2e−8, not ≥ 1e−5) |
| B | EntropyEOS (EOS-6) | 0/48 · 0 · 4.8 · −1.2e−8, −7.4e−7 · 1.6e−13 · 12010 | 0/48 · 0 · 5.0 · −1.4e−8, −1.5e−6 · 7.6e−13 · 27973 | 0/48 · 0 · 5.0 · −1.4e−8, −1.5e−6 · 2.9e−13 · 28057 | bit for bit | 1 fail: the SSP3 ideal magnetized atmosphere **blows up** (1.1e4) |
| Breset | our reset | as B, every digit | as B | as B | bit for bit | 1 fail: as B (9.6e3) |
| **B + coll** | EOS-6 + option | as B | as B | as B | bit for bit | 1 fail: the ARS ideal row (5.6e−8) |
| **Breset + coll** | our reset + option | as B | as B | as B | bit for bit | 1 fail: the ARS ideal row (4.9e−8) |
| B + hot (2026-10-16's EOS-5) | EntropyEOS | as B | as B | as B | bit for bit | 1 fail: the ARS ideal row (−4.5e−9) |
| C | EntropyEOS (EOS-7) + option | 0/48 · 17/188 · 100/100 · 4.2e−8/6.9e−8, 1.2e−6/3.4e−6 · 0.85 · 10882 | 1/48 (0.4–11%) · 71/479 · 100/100 · 3.7e−8/5.7e−8, 1.8e−7/1.9e−6 · 0.78 · 25848 | **5/48** (4.5–22%) · 91/300 · 100/100 · 3.5e−8/6.4e−8, 9.9e−8/6.3e−6 · 3.8 · 25968 | the colder initial atmosphere: −8.3496e−12, `W` 4.9356869 | 1 fail: the width-4 control (`τ` −2.5e−10) |
| B + C | EOS-6, EOS-7 + option | 0/48 · 0 · 4.8 · −1.2e−8, −7.4e−7 · 1.4e−13 · 12000 | 0/48 · 0 · 5.0 · −1.4e−8, −1.5e−6 · 9.0e−13 · 28000 | 0/48 · 0 · 5.0 · −1.4e−8, −1.5e−6 · 6.9e−14 · 28077 | as C | 2 fail: B's and C's |
| A3 + C | option + EOS-7 | 0/48 · 0 · 4.8 · −1.1e−8, −7.3e−7 · 1.8e−13 · 12031 | 0/48 · 0 · 4.9 · −1.3e−8, −1.5e−6 · 8.3e−13 · 28037 | 0/48 · 0 · 5.0 · −1.3e−8, −1.5e−6 · 7.1e−14 · 28143 | as A3, to 8 digits | 3 fail: as A3 |
| A2 + B | option + EOS-6 | 0/48 · 0 · 4.8 · −1.2e−8, −7.3e−7 · 1.3e−13 · 12031 | 0/48 · 0 · 4.9 · −1.4e−8, −1.5e−6 · 1.1e−12 · 28033 | 0/48 · 0 · 5.0 · −1.4e−8, −1.5e−6 · 1.5e−13 · 28138 | as A2 | — |

Beyond the tube (uniform, 48 runs each: blow-ups; largest `W` median /
max; every entry deterministic but the released policy's and A3's under
ARS):

| variant | ARS(4,4,3) | `Bʸ = 10⁻²` | ideal, SSP3 | ideal, ARS | 400 cells | `μ_max = 30` |
|---|---|---|---|---|---|---|
| released | 15/48; 100/100 | 47/48 | 0/48; 18.6 | 48/48 | 2/48; 100/100 | 0/48; 32/100 (chaotic) |
| A2 | — | — | **48/48** | — | — | — |
| A3 | **1/48** (0.4–11%); 100/100, chaotic | 0/48; 16 | 0/48; 4.0 | 0/48; 100/100 | 0/48; 5.6 | 0/48; 4.8 |
| B, B + C | 0/48; 7.0 (7.8) | **48/48** | **48/48** | **48/48** | 0/48; 5.6 | 0/48; 4.8 |
| Breset | — | — | **48/48** | **48/48** | — | — |
| B + coll | 0/48; 7.0 | 0/48; 16 | 0/48; 4.8 | 0/48; 4.3 | 0/48; 5.6 | — |
| Breset + coll | 0/48; 7.0 | 0/48; 16 | 0/48; 5.5 | 0/48; **53** | 0/48; 5.6 | — |
| B + hot | 0/48; 7.0 | 0/48; 16 | 0/48; 4.8 | 0/48; 4.3 | 0/48; 5.6 | 0/48; 4.8 |
| A3 + C | 2/48; 68/100 | 0/48; 16 | 0/48; 4.0 | 0/48; 100/100 | 0/48; 5.6 | 0/48; 4.8 |

What else moves in the suite (Julia 1.13, one thread; the released
policy through the emulation reproduces every printed number): under
every variant with a working reset the magnetized tube (`W` 6.63 → 4.76
… 4.80, fallbacks 92 → 0, injected rest mass `5.0·10⁻⁸` → `−1.1 …
−1.2·10⁻⁸`) and its coarse-fine run; the ARS rarefaction's floor
(2.7·10⁻¹⁰ → 1.0·10⁻¹⁰ under B and A5, 2.2·10⁻¹⁰ under A10, unchanged
under A3); SSP3's ideal magnetized atmosphere (1.56·10⁻⁹ → about
−6.7·10⁻⁹, `W` 4.83 → 4.0); under A`k` the hydrodynamic tube and its
coarse-fine run. Step 3's sweep moves only where the collapse branch is
changed: under hot (fallbacks without the cutoff 36, 78, 103 → 33, 74,
90; the `Float64` kernel test 47 → 48 of 48) and under coll (35, 69, 92,
and in the cutoff sweep at `μ_max = 10` the row `Γ = 2`, `f = 1` from 1
to 4 fallbacks; *corrected 2026-10-21*, this said `Γ = 4/3`, 0 → 4 — every
row `f ≤ 0.3` stays at none under both).

**Reading** *(proposed, for Erik; decided 2026-10-21 as B + coll, with B
in this package's recovery wrapper on EntropyEOS's new flag — above)*:
- **The smallest trigger that works is 3.** A2 and A2.5 keep the chaos
  (two runs drift 0.8 and 4.5 times `D` apart on the uniform mesh, `W` to
  100) and A2 blows up the ideal magnetized atmosphere; A3, A5, A10 and
  A20 are the same run to every digit printed, deterministic on all three
  meshes, `W ≤ 5`, no fallback, and hold at 400 cells, `Bʸ = 10⁻²`,
  `μ_max = 30` and under ideal SSP3. It works by resetting the HLLE
  precursor before it builds up (a cell next to a reset one never reaches
  `3 ρ_atm` in one step here), not by touching the defect — a margin
  between 2.5 and 3 that depends on the flux and the step. It needs no
  change but the cases' policy, but it moves the hydrodynamic tube
  (−1.2·10⁻⁸ … −1.4·10⁻⁸ of rest mass, 3% more writes), fails three
  assertions written for the released policy, and leaves ARS chaotic
  (1/48 blow-ups; the ideal atmosphere at `W = 100`).
- **B is the defect's own fix** and leaves the hydrodynamic tube and
  every result without a magnetized atmosphere bit for bit — but **not on
  its own**: the collapse branch then fires (the ideal magnetized
  atmosphere and `Bʸ = 10⁻²` blow up in 48 of 48 runs), so it needs
  `collapse_to_atmosphere` (an existing option) or hot. **B + coll** is
  clean in every case measured, ARS included, deterministic everywhere,
  and fails only the ARS row of the ideal atmosphere — which every working
  variant fails, since that row was this policy artefact.
  **Breset + coll** needs no upstream change and is the in-process form
  to every digit on the tube on all three meshes and in the suite, but it
  does not reach the stage solve: the ideal atmosphere under ARS reaches
  `W = 53` (4.3 in-process).
- **C does nothing** for the tube on its own (0, 1 and 5 of 48 blow-ups,
  `W` at the cap) nor on top of A3 or B: the precursor's trouble is its
  momentum, not its entropy.
- So: **Breset + coll now** (TreeGRRMHD-side, the existing option in the
  cases' policy, no upstream wait), with EOS-6 upstream if Erik agrees that
  a density raised to the atmosphere's should take the atmosphere's
  velocity; hot instead of coll if the cutoff sweep's 4 fallbacks matter.
  A3 if Erik prefers EntropyEOS's existing reset and accepts the moved
  hydrodynamic tube. Either way the ARS assertion (`ars.t.injected_D ≥ 1`)
  and, under A3, the two coarse-fine atmosphere assertions are Erik's to
  restate.

Step 3's pointwise sweep need not be extended to decide this: the front's
fallbacks have no admissible root, and a sweep built backwards from known
solutions cannot contain them; nor does the safe `μ_max` depend on them.
It could add (proposed) a class of predictors whose fluid share is
near-null and at the entropy ceiling in the taper band, asserting a
finite, flagged fallback — what the front produced.

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
  medium at `10⁻⁸` reaches the atmosphere, `10⁻¹⁰` (the reset writes 110
  cells as the step limiter), where SSP3(4,3,3) never goes below the medium,
  and the hydrodynamic atmosphere tube injects `1.2·10⁻⁹` of rest mass
  against SSP3(4,3,3)'s `−8.4·10⁻¹²`;
- *(amended 2026-10-22, decided by Erik: the decision stands)* in the
  magnetized ideal atmosphere (`Bʸ = 10⁻³`, `B²/ρ = 100`) ARS(4,4,3) was
  recorded injecting `2.6·10⁴` of rest mass into a box that holds 0.5 and
  running away to `W = 49`. That was EntropyEOS's repairs, not ARS: the
  entropy-floor branch wrote `ρ_atm` at `W ≈ |S|/ρ_atm`, the collapse
  branch `ρ_ceiling` ("Floors, the atmosphere and the magnetosphere"). With
  a floored recovery put at rest and the tube's `collapse_to_atmosphere`
  (2026-10-21) neither tableau runs away there — ARS injects `5.6·10⁻⁸` and
  ends at `W = 4.1`, SSP3(4,3,3) `−6.6·10⁻⁹` and 4.0 — and the row no
  longer argues either way. The others still do, and the default stands on
  them: Balsara 4's accuracy and plateau, the resets on Balsara 3 and 4,
  the hydrodynamic rarefaction's undershoot to the atmosphere and the
  hydrodynamic tube's injection, the cut-off tube's stages (below), and the
  cost;
- with the magnetization cutoff it takes 23 fallbacks and 33 recoveries in
  a stage, SSP3(4,3,3) none and 6 (both now deterministic; before
  2026-10-21: 7 and 33 against none and 6 in step 6's draw);
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
| rarefaction into `10⁻⁸` (atmosphere `10⁻¹⁰`): min ρ; resets stage/step | 1.0·10⁻⁸; 0/0 | 1.0·10⁻¹⁰; 131/110 (before 2026-10-21: 2.1·10⁻¹⁰; 152/114 in step 6, 2.7·10⁻¹⁰; 161/118 since step 8) |
| atmosphere tube, `B = 0`: injected `D`, `τ` | −8.4·10⁻¹², −2.2·10⁻¹⁰ | 1.2·10⁻⁹, 6.0·10⁻⁹ (before 2026-10-21: 2.4·10⁻⁹, 1.2·10⁻⁸) |
| atmosphere tube, ideal, `Bʸ = 10⁻³`: injected `D`, `τ`; max W at the end | −6.6·10⁻⁹, −1.7·10⁻⁵; 4.01 (before 2026-10-21: 1.6·10⁻⁹, −2.5·10⁻⁷; 4.83) | 5.6·10⁻⁸, −2.4·10⁻⁵; 4.11 (before 2026-10-21: 2.6·10⁴, 5.3·10¹³; 48.9 in step 6, 1.6·10⁴, 2.7·10¹³; 100 since step 8 — the policy artefact) |
| atmosphere tube, `Constant(10³)` cut off at `μ_max = 10`: injected `D`, `τ`; fallbacks; most recoveries | −1.2·10⁻⁸, −7.4·10⁻⁷; 0; 6 (before 2026-10-21: 2.5·10⁻⁸, 6.9·10⁻⁸; 0; 6 in step 6, 5.0·10⁻⁸, 1.7·10⁻⁶; 92; 33 since step 8 — chaotic) | −1.2·10⁻⁸, −7.9·10⁻⁷; 23; 33 (before 2026-10-21: 4.6·10⁻⁸, 1.5·10⁻⁶; 7; 33 in step 6, 6.8·10³, 2.4·10¹³; 725; 33 on the step-8 tree — chaotic) |

*(Re-measured 2026-10-22, `bin/tableau_comparison.jl atmosphere`, with the
atmosphere-at-rest recovery and EntropyEOS 1.1.0: the four atmosphere
rows; the "before" values are step 6's and, where they differ, the tree
just before 2026-10-21's decision, measured the same day. Re-run in full
the same day, every Balsara and resistive number is the table's to the
digits shown, and the current sheet's distance reads 9.559·10⁻⁴ and
9.560·10⁻⁴ (SSP3, ARS), as it has since step 8's walls — rung 1b — and
untouched by the recovery; the wall times are within the load's noise.)*
*(Re-measured 2026-10-24, EntropyEOS 2.0, on Symmetry: SSP3's rows as
above; ARS's atmosphere rows and Balsara 3 move — the hydrodynamic tube
injects `4.4·10⁻⁷`, the ideal one `−2.0·10⁻⁹` at `W` 4.13, the cut-off one
`5.6·10⁻⁶` with 22 fallbacks, the rarefaction `1.5·10⁻⁶` with resets
133/116 — "Floors, the atmosphere and the magnetosphere", "EntropyEOS
2.0". The decision for SSP3(4,3,3) stands on more, not less.)*

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

*(Step 7:)* `cfl` defaults to `default_cfl(limiter)`: 2/5, and 1/5 for MP5
(see "Reconstruction"). `after_step(p, u, t)` is called after every step
when given — rung 2a's `∇·𝔅` monitor (`max_div_B`). The initial data's
cell averages are the three-point Gauss rule applied one dimension at a
time, each average written as the middle sample plus a correction in the
differences, `f₀ + 5((f₋ − f₀) + (f₊ − f₀))/18` (amended in step 7): three
equal samples average to themselves bit for bit, so a direction the case
does not vary in leaves the data exactly the 1D data. The step-5 form,
`Σ w_q f_q` over the `3^D` nodes with the weights multiplied out, rounded a
constant to a neighbouring float, and was the whole of the 2.7e−15 by which
step 5's 2D check missed the 1D run (see "The right-hand side").

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

*(Re-measured 2026-10-24, EntropyEOS 2.0; Symmetry, the same numbers on
the laptop:)* the fallback takes **49, 125 and 150** points (Γ = 4/3, 5/3,
2; at most 5.8%), still only at `b²/ρ ≥ 10` and `σΔt ≤ 10²`, and **2, 8
and 4** of them where the ideal seed was a fluid state (1.1: 0, 1, 0);
another root 4, 3 and 1; the fixed point 1247, 972, 907 points, Newton
1296, 1495, 1535; residuals unchanged (3.3e−16, 1.5e−12). 168 points fall
back under 2.0 and not 1.1, 61 the other way. At the fallback points
1.1's predictor was `S_FLOORED` (with `W_CAPPED`) and its ideal seed the
collapse (`CEILING`) or `S_CEILED`; 2.0's predictor is `S_FLOORED` (with
`W_CEILED`) and its ideal seed `S_CEILED` (with `W_CEILED`): the same
faces, but 2.0's keep the momentum and move the energy where 1.1's kept
the rapidity (or collapsed to the density ceiling), so the repaired seeds
and iterates Newton starts from are other states. The magnetization
cutoff's sweep, which is what the scheme runs, is unchanged: no fallback
at `f ≤ 0.3`, one each at `f = 1` for Γ = 5/3 and 2; the kernel test
converges 46 of 48 cells at `Float32` (47 under 1.1), 48 at `Float64`.
`test/stage_solve_tests.jl`'s bound on the seeded fallbacks moved from 3
to 12 for this (a margin of 1.5), the only assertion of the sweep that
moved. *(Decided by Erik 2026-10-24:)* the bound of 12 is approved, for
the uncut sweep only; the cutoff sweep the scheme runs is unchanged.

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
- fixed `Δt`, generic arrays and Julia 1.10;
- *(2026-10-24, the step cache)* a step that is never retried or rejected,
  whose step limiter is its last call, and an exception that leaves
  `integ.u = uⁿ`: with these the step limiter is where a step ends, and the
  step cache needs no step-begin hook ([Primitive
  recovery](#primitive-recovery), "The step cache").

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

*(Measured in step 9; the choice proposed in step 9, for Erik — resolved
2026-11-03, decided by Erik: pointwise, no cache.)* **The
`Analytic` provider is evaluated pointwise, where each kernel needs it**,
and nothing is cached. Every kernel since step 5 already called
`point_metric(provider, t, x)` at its own point — the recovery and the ghost
cells' `P` at cell centres, the fluxes at faces, the EMFs at edges, the
gauge potential `αΦ − β·A` at vertices and the gauge flux at edges, the stage
solve and the reset at cells — and `metric_gradient` at the cell for the
sources (the cell's `Metric3` is `P`'s), so rung 3 needed no change in the
scheme, only the cases. Per owned cell and right-hand side that is about 20
evaluations of `point_metric` (1 cell, 3 faces, 3 edges for the EMF, 6
vertex evaluations in `∂ₜA`'s gradient — each vertex twice per direction — and
7 in the gauge equation's divergence) plus the ghost cells' and one
`metric_gradient`; per SSP3(4,3,3) step, three right-hand sides, four stage
solves and four resets, about 70 and 3. Measured (`bin/metric_cost.jl`,
Symmetry cn079, AMD EPYC 7532, one pinned thread, Julia 1.13, 2026-11-02):
- per call, `Analytic(KerrSchild(1))`: `point_metric` 102 ns,
  `metric_gradient` 234 ns (ForwardDiff through the metric, and SM-1's
  `K_ij`); `Flat()` folds to constants;
- per step, the same physics: the spherical blast of rung 2a (16³ in 8
  blocks of 8³, closed box) with `Flat()` 99.5 ms, with `Analytic` of a
  Kerr-Schild hole of mass `10⁻³⁰⁰` far outside the box — Minkowski to
  rounding, with Kerr-Schild's whole cost — 177.0 ms: **the provider is 44%
  of a step** at this block size (a block of 8³ with `G = 3` stores 5.4
  times as many cells as it owns, and `ghost_kernel!` evaluates the metric
  at every one);
- the rung-3 cases: 69–72 µs per cell and step at 16³ (8 blocks of 8³),
  49 µs at 32³ (8 blocks of 16³), Wald and Bondi alike.

So the pointwise provider is affordable for rung 3 — the study to `t = 50M`
runs in hours, not days — and a per-stage cache would save at most that
share, less what it costs to store and load the metric at four centerings
(cells, three faces, three edges, vertices: 8 points and about 100 floats
per cell). **Decided by Erik 2026-11-03: pointwise, no cache; the background
is stationary, but the analytic cases are affordable.**

*Deduplication (asked by Erik 2026-11-03; not done, measured why):* the
vertex and edge evaluations of `∂ₜA` and `∂ₜΦ̃` come to 13 per cell where 4
distinct points (a vertex, three edges) exist — but **across** kernel
invocations, not within one: `potential_rhs_kernel!` evaluates the two
distinct vertices of its own edge, and `gauge_rhs_kernel!` the six distinct
edges around its vertex and the vertex itself, each once. So there is
nothing to reuse inside an invocation. Sharing across invocations needs either
storage (the metric, or `ψ = αΦ − β·A`, at every vertex and edge, which the
request excludes) or restructured kernels (one invocation per vertex writing
its three edges' `∂ₜA`, which still evaluates `ψ` at the three far vertices:
13 → 10). Both change what the kernels compute in which order. *(Decided
by Erik 2026-11-03: leave it; it folds into step 10's caching decision.)* Its ceiling is about 9 `point_metric`
evaluations per cell and right-hand side, 9 × 102 ns × 3 ≈ 2.8 µs per cell
and step, 4–6% of the rung-3 cases' 49–72 µs.

**Caching is step 10's decision** *(deferred by Erik 2026-11-03)*. The
production provider is the interpolated vertex metric (`GHMetric`), and step 10
decides its caching with both of its uses in view: frozen (the Cowling
approximation), where the metric is static and could be computed once per
mesh, at its creation and after each regrid; and coupled, where it changes
every stage. Measured inputs: the analytic provider's 44% of a step on the 16³
blast; about 100 floats per cell for every centering (cells, three faces,
three edges, vertices); a block of 8³ with `G = 3` storing 5.4 times the cells
it owns.

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
edges. *(Amended in step 9, deferred by Erik 2026-11-03:)* whether the vertex
metric is cached per mesh (frozen) or per stage (coupled) is step 10's
decision; "Metric providers" has the measured inputs. The derivatives and `K_ij` are not stored but evaluated where the
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

In the order the explicit part of each stage evaluates it (proposed;
*amended in step 8*, when AMR-10 let the matter's ghost fill take the
primitives, decided by Erik 2026-10-15):

1. `scatter!` the multi-set state; `fill_ghosts!` the potentials `A_j`, `Φ̃`
   (each fill applying the complete outer boundary condition). The matter's
   conserved state is not exchanged.
2. Metric to cells (into `P`); `B_d = curl A + 𝔅̄` on all stored faces;
   `B` averaged to cells.
3. `con2prim` over the **owned** cells, in hydrodynamic mode, into `P` and
   the primitive set `W` (`ρ★, s, Yₑ, uⁱ = Wvⁱ, Eⁱ`); `fill_ghosts!` on `W`
   (copies at same-level faces, the bounded conservative prolongation
   centred on the parent at refinement boundaries, with its own Dirichlet
   hook, the case's primitives); at every ghost cell `P` from `W` by one
   forward EOS evaluation, no inversion, with the ghost's own `B` and
   metric. No ghost is recovered and the conserved ghosts are not written
   ("Refinement"). (Until step 8 the exchange was in conserved variables, as
   TreeHydro decided, and `con2prim` ran over every stored cell; amended
   2026-10-23, decided by Erik.)
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
never mutates `u`. It is pure in `(u, t)` outside a step; inside a step of
`evolve!` its recovery goes through [the step cache](#primitive-recovery),
and a step is pure in `uⁿ` (amended 2026-10-24, decided by Erik
2026-10-23).

*As built in step 5* (`src/evolution.jl`, `mhd_rhs!`; the kernels are
written for `D = 1, 2, 3` and step 5 validates `D = 1`):

- Steps 2 and 3 are one kernel over the stored cells (`primitive_kernel!`):
  the provider's metric at the cell, `B` from the faces, `E = 𝔈/√γ`, the
  recovery from `state_guesses` (see "Thermodynamics"), all thirty slots of
  `P`. `q` is a second kernel over the owned cells, by centered differences
  of the ghost-filled `𝔈`, `q = (1/√γ) Σ_d (𝔈ᵈ[n+e_d] − 𝔈ᵈ[n−e_d])/2h`
  (proposed in step 5). *(Amended in step 8:)* step 3 is three passes:
  `primitive_kernel!` over the owned cells (the recovery, `W`, and `P`
  through `P_from_W`), the fill of `W`, and `ghost_kernel!` over every
  stored cell that is not owned (`P` through the same `P_from_W` with the
  ghost's `B`; until 2026-10-23 also the conserved ghost by `prim2con`,
  which nothing read). Owned cells take `P`
  through `P_from_W` too, rather than straight from the recovery (proposed
  in step 8): where a ghost is a copy of its owner — every ghost on a
  uniform mesh — the two blocks then hold the same `P` bit for bit and
  compute the same flux at their common face, as they did when both
  recovered the same conserved state; the two differ by rounding only
  (`v = u/W` from `u = cosh(w) v`, the rapidity `asinh |u|`, one forward EOS
  call). `q` is taken from `P`'s `√γ E` (the same numbers in flat space).
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
  and tallying the outcome into `X`. *(Amended 2026-10-24:)* inside a step
  every site's first recovery is seeded from the step cache where the cell
  has an entry, and reused where its input is the entry's.
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
  7's. *(Step 7:)* the whole difference was the initial data's Gauss
  weights (see the driver in "The partition"); with the average in
  difference form the 2D and 3D runs are the 1D run bit for bit, under every
  reconstruction, and nothing in the right-hand side changed (measured in
  step 7, "Validation ladder" 2a).

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

*Measured in step 8* (`test/refinement_tests.jl`; MC, HLLE, SSP3(4,3,3),
static two-level meshes and adaptive ones):
- **The totals.** The Alfvén wave of rung 1a on 64 cells with the middle
  half refined, a quarter period, drifts 0.008 ulp of a total's scale a
  step with the flux fixup and 1.5·10¹⁰ without; along the diagonal on
  2 × 2 roots of 8² with one refined, 0.12 and 3.3·10¹¹. Balsara 1 on its
  box made periodic, tracked by two levels more through ten regrids, 0.017
  and 1.4·10⁷; the rotor, adaptive, 0.036 through two regrids.
- **The magnetic flux, as a rate.** With `(d, a, b)` cyclic, `∂ₜ𝔅ᵈ =
  −(∂_a EMF_b − ∂_b EMF_a)` (the gauge terms' curl vanishes), so the rate at
  which the flux through a face changes is `−∮ EMF · dl` around it. On every
  face of every coarse block that lies on a coarse-fine face, the rate from
  that block's EMFs equals the sum of the rates through the fine faces that
  cover it to 1.7e−18 of a largest rate of 5.2e−3 (2D, Orszag–Tang's
  initial data, 16 faces) and 8.7e−19 of 7.8e−3 (3D, the spherical blast in
  its closed box, 192 faces) with the correction — rounding: the fine sum
  telescopes in floating point — and 0.025 of 0.029 and 0.0037 of 0.0078
  without it (proposed in step 8: the claim of PLAN.md's step 8, "the flux
  through a coarse face equals the sum through its fine faces", made as
  the rate it is made by). *(Re-measured at the end of step 8, with the
  potentials' conservative transfer and the matter's ghost fill in
  primitives:)* 8.7e−19 of 5.4e−3 (2D) and of 5.5e−3 (3D) with the
  correction, 0.025 of 0.029 and 0.0017 of 0.0055 without; the totals
  0.013 (1D) and 0.12 (2D) ulp a step with the fixup, 1.8·10¹⁰ and
  5.4·10¹¹ without; tracked Balsara 1 0.015 (2.2·10⁷ without), the rotor
  0.037.
- **Which block's EMF.** `restrict_interfaces!` corrects the EMFs of a
  block across whose *face* the finer neighbour lies (TreeAMR's face
  directions only). An edge at the end of a coarse-fine face is also an
  edge of a coarse block that touches the fine region only along that edge,
  whose own EMF there is not corrected, and with half-open ownership that
  block may own the edge's `A` — the corner vertex of a refined quadrant in
  2D belongs to the coarse block diagonally across. That `A` then evolves
  with the coarse EMF. Nothing breaks: `A` is single-valued, so the state's
  fluxes on either side are the same numbers, and `∇·𝔅 = 0` is structural;
  it is the one place a coarse estimate of the EMF drives a value on a
  coarse-fine face (proposed in step 8: accepted, not corrected).
- **The magnetic flux, as a state** — not to rounding under the PointValue
  operators step 8 began with. In 2D the flux through a face on the
  coarse-fine face is a difference of `A_z` at two vertices the levels
  share, which the exchange copies, so the two sides agree exactly. But the
  flux through the plane, `Σ hᴰ 𝔅ᶻ = ∮ A · dl` around the periodic box,
  telescopes across a coarse-fine face only if the coarse `A_y` on it is
  the mean of the fine ones beside it, and the PointValue operators
  interpolate along an edge's own direction where a conservative transfer
  would average: on the oblique Alfvén wave it drifted 5.8e−4 of its scale
  over an eighth of a period (5.1e−17 on the uniform mesh), with the EMF
  correction or without, 1.4e−3 at width 2 — the operators', not the EMFs'.
  In 3D the flux through a face on a coarse-fine face is the circulation of
  `A_y`, `A_z` in its plane and inherited the same interpolation. That was
  **AMR-9**, which landed in TreeAMR 0.2.2.

  *(Measured in step 8, with the potentials' conservative transfer, AMR-9;
  `test/refinement_tests.jl`:)* **the flux holds as a state, to rounding,
  after every fill and every regrid.** The flux through the plane of the
  oblique wave on the two-level box drifts 2.2e−16 of its scale (5.8e−4
  with the PointValue operators), and 1.5e−16 after a regrid that coarsens
  the refined root and refines the opposite one (5.8e−4 with PointValue).
  In 3D, on every face of a coarse block on a coarse-fine face of the
  periodic spherical blast (192 faces, then 384 after a regrid), the flux
  from the coarse block's own `𝔅` equals the sum through the fine faces
  covering it to 1.4e−16 of the largest flux (1.5e−3 with PointValue). That
  is the transfer's property, not the EMF correction's: without the
  correction it holds to 1.4e−16 too, since after every fill the edges the
  non-owning level holds are derived from the owner's (the restriction's
  mean, or a prolongation whose two children average to the parent). So
  PLAN.md's negative control "and does not without [the EMF correction]"
  applies to the rate, where it is asserted (above), and the state's
  negative control is the PointValue family (proposed in step 8).

  **Not at an outer face** *(measured in step 8; proposed: accepted,
  recorded)*. In the same blast's closed box the 147 faces away from the
  walls agree to rounding, but the 45 whose perimeter lies on an outer face
  miss by 1.2e−5 of the largest flux (2.7e−5 after the regrid): there each
  level's relative hook writes its own wall edges, the nearest evolved point
  of that level plus the case's difference, so a coarse wall edge is not
  the sum of the fine ones beside it. The flux through a face at a Dirichlet
  boundary is the boundary condition's, not conserved by anything, so
  nothing in this package relies on it; a relative hook that fills a point
  on both an outer face and a coarse-fine face from the level that owns it
  (a transfer, not the hook) would make it exact — **AMR-11** in `PLAN.md`
  (added in step 8; wanted, no step blocked).

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
  *(Amended in step 8:)* since TreeAMR 0.2.2 it is TreeAMR's relative hook,
  `CellBoundary(Relative(f))` with `f` the case's potential at the fill's
  time (`boundary_at`): a kernel, so **every case, periodic or not, runs on
  every backend** — the CPU-only restriction came from the region form
  alone and is gone (not exercised on a device here: the machine has none
  this package targets; step 14's).

*(Measured in step 7:)* the potentials' hooks are not mirror-symmetric. The
box's low boundary vertex is a block's first owned point and is evolved,
while its high boundary vertex is the last block's shared plane, which no
block owns and the hook fills from the vertex before it — so `𝔅` on the
high boundary face is the case's, and on the low one the evolution's. As
long as nothing but the light-cone fluxes' exponentially small precursors
reaches the faces the difference is that small: a 2D explosion 30 cells
from its faces kept its mirror symmetries to 2.5e−17, but a spherical blast
in a 16³ box, whose precursors reach the faces within a step, came out
1.3e−4 asymmetric there after three steps, and the 3D smoke test runs
periodic. Whichever later rung needs a symmetric closed box must fill the
high boundary vertex from the evolved side, or evolve it.

*(Amended in step 8; decided by Erik 2026-10-15, "option 1":)* **both walls
are prescribed.** At every outer face of a case with Dirichlet hooks, the
owned low wall of each potential (`A_j`, `Φ̃`; a vertex-like dimension's
first owned point `G+1`) is set by the mirror of the hook's high-wall rule —
the nearest evolved point (`G+2`) plus the case's difference between the
two points, at the time of the fill. And every point beyond an outer face
takes its nearest evolved point across a block face from the block's own
(filled) ghost along the dimensions it does not leave the domain through:
TreeAMR's region form clamped along every outward direction, so a corner
row where a wall meets another block came from the block's own last column
and the point two blocks share held a different value in each (the cell
next to it had a block-dependent `𝔅`, and the two blocks computed
different fluxes at their common face next to the wall). Both were this
package's kernels until TreeAMR 0.2.2 (`low_walls!`, `outer_points!`, and on
a multilevel mesh a second, hookless fill and `outer_points!` again); both
are TreeAMR's now (amended in step 8):
- the potentials' field sets are built with **`walls = :symmetric`** (AMR-8),
  under which the fill writes the owned low wall of a vertex-like dimension
  at a low domain face by the high wall's rule;
- their hook is the **relative hook** (AMR-7), which reads `x_r` across a
  block face from the block's own ghost and runs each region after the
  stage that fills what it reads, so that **one fill applies the complete
  condition** — `fill_ghosts!`, `regrid!`'s fill before its transfer and
  `adapt_to_initial_data!`'s alike (decided by Erik 2026-10-15: "an outer
  boundary condition needs to be applied always"). `fill_set!` is that one
  fill; `regrid_state!` hands `regrid!` every set's own hook; the initial
  adaptation's criterion reads what the cycle filled.

Measured (`test/refinement_tests.jl`, the closed `[−2, 2]²` explosion with a
wall root refined, to `t = 1/4`): a second fill changes nothing bit for bit,
and one fill of working arrays prefilled with `NaN` writes the same — every
stored point, from the state alone; a hookless fill leaves it; and a regrid
through `regrid_state!` transfers bit for bit what step 8's first
arrangement (a complete fill of this package's, then `regrid!` with no hook
for the potentials) transferred. The hook evaluates `f(x_g) + (u(x_r) −
f(x_r))` where `outer_points!` evaluated `u(x_r) + (f(x_g) − f(x_r))`, so the
potentials' outer points moved by rounding: every number the suite prints
for a non-periodic case moved in its last digits at most (see "Validation
ladder").

**The low walls in the state** *(proposed in step 8, at the reviewer's
request)*. The low wall is owned, so it is in the integrator's
state vector, but TreeAMR writes it during the fill — into the set's working
array after `scatter!`, not into `u`. `map_blocks!`'s owned range is
unchanged by symmetric walls (the trimmed range `G+2 …` is the fill's
regions' only), so the right-hand side would compute an evolved value there
and `u`'s wall entries would drift from the condition. So:
- the right-hand side writes **zero** into `du` at an owned low wall of
  `A_j` and `Φ̃` (`on_low_wall`, a branch, never a multiplied mask): the wall
  is algebraic, prescribed by the fill, and the implicit part never touches
  the potentials;
- the reset, in both its roles, after its own fill of the potentials,
  copies the filled walls — those points and no other — into `u`
  (`walls_to_state!`), so that after every step `u` holds the condition's
  walls for everything that reads `u` without a fill: output, error norms,
  `gather!`-based comparisons, a regrid's `gather!` (whose new state the
  post-regrid reset then corrects where a new block's wall was
  prolongated), a checkpoint and restart.

Nothing in the scheme reads `u`'s walls without a fill first. Tested on the
closed box, uniform and with a wall root refined: after every step `u`'s
190 (214) wall entries are what a complete fill writes, bit for bit, and
`du` is exactly zero there; and one step from a state whose wall entries are
all perturbed by 1 gives the same state, bit for bit, as from the true one.
`low_walls(forest, T)` lists, per block, the low domain faces (any
non-periodic one, since TreeAMR fills the wall at outer, reflecting and
outflow faces alike).

*Measured in step 8:* the explosion in a box its precursors reach — `[−2,
2]²` at 32², minmod, `t = 1/4` — was 7.1e−6 (`x → −x`) and 6.7e−5 (`y → −y`,
fields reversed) of a variable's scale from its mirror image before, and
the spherical blast in the closed 16³ box 1.3e−2 and 1.0e−1; with both
fixes both are their own mirror images bit for bit, every evolved set on
the closed range with both walls, but for the sign of an exact zero (a
parity of −1 turns `+0` into `−0`; stated with `==`, as step 7 stated
transposition). The 3D smoke test now runs in the closed box
(`test/multid_cases_tests.jl`). The 2D explosion at 64² on `[−6, 6]²` and
the rotor, symmetric to 2.5e−17 and 2.6e−27 in step 7, are now exactly
symmetric too. Every other number the suite prints moved in its last
digits at most — the shock tubes' L1 errors and boundary budgets, the
uniform flows through the faces (1.33e−12 → 1.24e−12 of a variable's scale
at most), the current sheet's rates — since their faces see nothing but
precursors before `t_end`; the one exception is the current sheet run on to
`t = 9`, whose field has diffused to the faces by then: its L1 distance to
the erf profile went from 9.30e−4 to 9.56e−4 of `B₀`, the low face's `𝔅`
now the profile's as the high one's was.

*(Measured in step 8, with TreeAMR 0.2.2's symmetric walls and relative
hook:)* both closed boxes are still their own mirror images bit for bit,
and nothing printed by a uniform-mesh case moved — the uniform flows, the
shock tubes' budgets, the current sheet (9.56e−4) and the atmosphere tubes
to every digit. On a refined mesh a closed box is **not** mirror-symmetric:
the explosion in its closed box with the middle 2 × 2 roots refined (a mesh
symmetric under both reflections, whose coarse-fine faces cut the blast's
transition region), to `t = 1/4`, is 1.0e−2 (`x → −x`) and 8.8e−3 (`y → −y`)
of a variable's scale from its mirror image in `U`, 1.1e−2 and 1.3e−2 in
`A_z` (3.2e−4 and 6.3e−3 in `A_z` after one step). That is TreeAMR's open
item (its `CODE.md`, "Centerings": a plane on a coarse-fine face is the fine
block's owned plane when the face is the fine block's low face, and the
coarse block's, interpolated at the fine points between, when it is its high
face; 4.7e−5 upstream on smooth data, one fill), here at a discontinuity
and through 20 steps. Recorded, not hidden: `test/refinement_tests.jl`
asserts it present and bounded (proposed in step 8: nothing in this
package needs a refined box to be symmetric before the stars of steps 10
and 12, whose own symmetry tests will say whether upstream must close it).

**The potentials' boundary must pin the gauge or keep a gradient a gradient**
*(a design rule, from step 9's round 2; decided by Erik 2026-11-03: option 1
for static potentials)*. A Dirichlet hook that copies the evolved potential
across a face — the relative hook (AMR-7): `A(x_g) = A_case(x_g) + [A −
A_case](x_r)` — copies a gauge drift `∇χ` of the evolved `A` unchanged along
the normal, where it is no longer a gradient, and the filled `A`'s curl puts
`∂_n∂_⊥χ` into the ghost and wall faces' `𝔅`. That field drives `E`
(Ampère), `E` drives `A` (Faraday), `A`'s divergence drives `Φ̃` (the
generalized Lorenz gauge) and so `χ`: a loop, boundary → `𝔅` → `E` → `A` →
`χ`, which the damping `ξ` slows but does not stop. On the Kerr-Schild cases it
grew as `e^{0.11 t/M}` at `h = M/4` ("Validation ladder" 3, "Where the error
lives"); in flat space nothing measured showed it, the shock tubes' and the
explosions' faces seeing too little of a drift before `t_end`. **Our
reading, not measured**: a static gauge mode with `Φ̃ = 0` satisfies the
elliptic `∂ᵢ(α√γ γ^{ij}∂ⱼχ) = 0`, so it is fixed by `χ`'s boundary values;
the relative hook leaves them free (a Neumann-like copy), the absolute hook
pins them to zero, and with them the static mode.

- **Static potentials** (`MHDCase(…; static_potential = true)`, decided by
  Erik 2026-11-03): the hook is absolute, the case's `A_j` and `Φ̃` at every
  ghost and on every prescribed wall (`DirichletBoundary(…; absolute =
  true)`, `CellBoundary(AllVariables(…))`). Under `walls = :symmetric`
  TreeAMR fills the owned low wall with the hook's own rule, absolute or
  relative (its "the hook's low region becomes `1 … G+1`"), so the walls
  follow; nothing upstream was needed. Valid only where `A_case` does not
  depend on `t`: under IMEX the stage's `A` is at `tⁿ + c̃Δt` while
  `solve_imp!` is told `tⁿ + cΔt`, which a potential that moves with `t` would
  get wrong (step 6, above). The Kerr-Schild cases declare it; the test
  "A static potential's boundary is the case's own" checks that a gauge
  offset vanishing on the boundary leaves `𝔅` on every stored face at the
  exact state's (1.1e−15) and every boundary-filled point untouched, where
  the relative hook shows it (5.9e−2).
- **Potentials that grow in time** keep the relative hook (every flat-space
  case, `static_potential = false` by default): their outputs are
  unchanged bit for bit.
- **Open** (for step 10's outer boundary, and for dynamic cases without an
  exact `A`): option 2, a relative hook that carries the offset's gradient
  along, `δ(x_r) + (x_g − x_r)·∇δ`, which keeps a gradient a gradient to second
  order (TreeAMR's hook form); option 3, a gauge that damps a static `χ` as
  well. Neither is requested yet.

## The interior

*(Added in step 9; proposed in step 9 throughout, the form a decision for
Erik.)* A black hole's singularity is inside the domain of the rung-3 cases,
and nothing in the scheme can evolve the neighbourhood of `r = 0` — the
Kerr-Schild metric diverges there, and Michel's density grows as `r^{−3/2}`.
PLAN.md's step 9 asks for TreeGH's answer: inside `r < r_in`, within the
horizon, the state is relaxed to the analytic solution each stage,
**position-only**.

**Why it is correct to touch the inside at all** (TreeGH's argument, its
`CODE.md`, "Why touch the interior"): inside the horizon every characteristic
points inward, so in the continuum nothing outside depends on the inside. On
the mesh the stencils of the cells just outside reach in, and there they find
the analytic solution, the best data an outflow boundary can have. In
Kerr-Schild coordinates the light-cone speeds the HLL fluxes and the UCT EMF
use, `λ± = −βᵈ ± α√γ^{dd}`, both point inward inside `r = 2M` along the
radial direction, so the upwinding carries nothing outward but what the
reconstruction's stencil reaches.

**As built** (`src/interior.jl`; `HoleInterior(r_in; center, stage, step)`, an
`MHDCase`'s `interior`, `InteriorTarget` on `MHDProblem`):

- **A hard paste**, TreeGH's `ρ → ∞` limit on a step profile, which TreeGH
  kept as a measured variant of its smooth damping layer. It replaces, at every
  owned point whose position lies inside the ball — a cell by its centre, an
  edge by its midpoint, a vertex by itself — the state by the case's own
  solution: the cell's Gauss average of the conserved state, as
  `exact_state` writes the initial data, and the potentials' point values.
  The test is the position alone, never the state, so the replaced set is the
  same at every stage and step, at every thread count and after a restart.
- **In the reset, first**, before the reset's own fill and recovery, which
  then see the pasted state (so the reset repairs nothing there that the
  solution does not need repaired). The reset is IMEXRungeKutta's stage limiter
  and step limiter ("Floors, the atmosphere and the magnetosphere"), so the
  paste runs before every explicit right-hand side but a trivial stage's —
  which reads `uⁿ`, pasted by the previous step limiter — and on every step's
  result. `stage` and `step` choose the roles; both by default.
- **Not in the right-hand side.** TreeGH relaxes inside its right-hand side,
  `∂ₜu = w F(u) − ρ (u − u_exact)`, with smooth profiles, because its
  measurement found a paste's kink at the surface worse than a layer in a
  finite-difference code. A finite-volume scheme with limited reconstruction
  meets a kink at every shock; the paste is the change that needs no new
  term in the integrator's view of the right-hand side, keeps the right-hand
  side pure, and leaves the flat path untouched: a problem without an
  interior launches nothing.
- **What it costs.** The interior's cells are still recovered and stepped —
  their tendencies are thrown away by the next paste — so a run pays for them
  like any other cell; the ball is 0.4% of the default octant at `r_in = M`.
  The stage solve and the right-hand side's recovery meet the extreme states
  near `r = 0` there: in the magnetized Bondi flow at `h = M/8` a few cells
  next to the singularity (`r ≈ 0.2M`) are repaired by the policy in the stage
  solve's recovery, tallied in `X` like any other, and pasted over.

**Measured** (`bin/curved_stationarity.jl`, both cases to `t = 50M` at `h =
M/4`, 1500 steps, Symmetry EPYC 7532, jobs 588101–588103): the L1 drift
from the exact solution outside the horizon (`r ≥ 2M`) and over the whole
octant, and what the reset did, for the paste in both limiters at `r_in =
M` (the default), in one role only, at other radii, and without an
interior:

| interior | Wald `D`, `τ`, `𝔈ˣ` (`r ≥ 2M`) | Bondi `D`, `τ`, `𝔈ˣ` (`r ≥ 2M`) | `D` everywhere, Wald / Bondi | resets (stage/step), injected `D` |
|---|---|---|---|---|
| both, `r_in = M` | 1.008e−2, 1.545e−2, 1.041e−2 | 6.845e−3, 4.413e−3, 4.660e−4 | 1.49e−2 / 1.64e−2 | none |
| step limiter only | 1.008e−2, 1.546e−2, 1.041e−2 | 6.845e−3, 4.413e−3, 4.660e−4 | 1.48e−2 / 1.64e−2 | none |
| stage limiter only | 1.009e−2, 1.544e−2, 1.040e−2 | 6.818e−3, 4.408e−3, 4.647e−4 | 20.3 / 20.2 | Wald 0/34538, 95.8; Bondi 5499/51484, 129 |
| both, `r_in = M/2` | 1.006e−2, 1.546e−2, 1.040e−2 | 6.846e−3, 4.414e−3, 4.659e−4 | 2.60e−2 / 3.33e−2 | Bondi 8931/4467, −1.2e−13 |
| both, `r_in = 3M/2` | 1.025e−2, 1.539e−2, 1.036e−2 | 6.852e−3, 4.416e−3, 4.663e−4 | 1.38e−2 / 1.00e−2 | none |
| none | 9.856e−3, 1.514e−2, 1.019e−2 | 6.846e−3, 4.414e−3, 4.659e−4 | 0.250 / 0.254 | Bondi 15264/7008, 1.7e−12 |

- **Nothing that happens inside reaches outside.** Outside the horizon
  every variant drifts by the same amount to 2% (Wald `D` at `r_in = 3M/2`)
  and mostly to the fourth digit, even where the inside is wrong by `O(10)`
  (the stage limiter alone, which leaves `uⁿ⁺¹`'s inside unpasted for the
  first, trivial stage of the next step to read — the reset then floors it
  every step) or evolves into the singularity's neighbourhood (no interior,
  which under `IdealConductor` resets about a hundred cells a step there).
  The upwind light-cone fluxes inside `r = 2M` carry nothing outward at
  `h = M/4`, the horizon four cells from `r_in = M`.
- **So the interior is about the inside's health, not the outside's
  accuracy**: without it, or with the stage limiter alone, the reset works
  inside every step and its tallies (and the totals) are the singularity's;
  at `r_in = M/2` the Bondi flow's magnetization next to the paste
  (`b²/ρ ≈ 37` at `r = M/2`) still resets the cells between. At `r_in = M`
  or `3M/2`, in both limiters or the step limiter alone, nothing is reset at
  `h = M/4` and `M/8` — but at `M/16` the Bondi flow resets inside (below).
**Resets inside the hole** *(decided by Erik 2026-11-02)*:
1. **Resets deep inside the hole are not counted**: deep inside is `r < (r_in
   + 2M)/2`, `1.5M` at `r_in = M`; whatever keeps the hydrodynamic scheme
   stable there is fine. Reported tallies, and any assertion on resets,
   policy flags or injected rest mass, cover `r ≥ (r_in + 2M)/2` only.
2. **The resets are counted in radial layers**, to see where they happen:
   the stage limiter's and the step limiter's resets and the stage solve's
   policy repairs, in layers of `M/4`, split into level 1 (EntropyEOS's
   projection: which `FLAG_POL_*` face bits) and level 2 (the atmosphere of
   `apply_policy`, `FLAG_POL_ATMOSPHERE`). On the host, from `X` after a
   chunk and, for the reset's bits, a copy of the reset kernel that counts
   them, evaluated in the measuring process only (`bin/curved_localise.jl`).
3. **No special cases for resets inside the hole**: the reset there is the
   reset everywhere — no interior-specific floors, atmosphere rules or
   thresholds, and `apply_policy`'s options and the atmosphere trigger are
   not changed without asking. Erik's view: codes reset to the atmosphere
   mostly because their EOS cannot take high densities and temperatures;
   with EntropyEOS a reset can go to whatever state the EOS prefers, and the
   layers report how many resets end in level 1's projection and how many in
   level 2's atmosphere — the evidence on whether the atmosphere is needed
   there at all.
4. The paste stays as built (both limiters, `r_in = M`) for now; its form is
   not decided.

**The resets by radius** (`bin/curved_localise.jl`, `t = 50M`; jobs 588186
at `M/8` and 588213, 588214 at `M/16`). At `h = M/8`: Wald none; the
magnetized Bondi flow no reset at all, and its 9000 stage-solve policy repairs
— 3 a step — all in the corner layer `0 ≤ r < M/4` (`r ≈ 0.11–0.2M`, the
cells next to the singularity), each `FLAG_POL_S_FLOORED`: level 1, the
entropy raised onto the window's floor where the stage's predictor came
out colder than the gas allows, deep inside and not counted. At `M/16` the
radial tables (jobs 588213, 588214, the reset kernel's copy counting the
bits by layer) were at `t = 20M` when the Symmetry tunnel dropped, so where
the magnetized Bondi flow's 126000/42000 resets of job 588113 sit, and how
many end in level 1 or level 2, is in
`/mnt/beegfs/eschnetter/claude/treegrrmhd-step9-588213-part1.log`, unread.
The whole-octant `τ` of `1.29·10³` against `9.9·10⁻⁴` outside `2M` says
the resets' states lie inside `2M`.

- **Decided by Erik 2026-11-03: the hard paste, both limiters, `r_in = M`,
  for the analytic rung-3 cases** (proposed in step 9; round 3's measurement
  below). *The matter alone needs no interior*: the magnetized Bondi flow
  without one has the paste's outside error at every resolution, and every
  reset deep inside, all level 1, none to the atmosphere — EntropyEOS's
  projection absorbs the extreme states, as Erik expected. *A vacuum field
  near the singularity does need one*: without it Wald fails at `M/8` and
  `M/16`, earlier on the finer mesh — a spurious charge with no Ohm's law to
  hold `E`, whose electromagnetic energy and momentum the recovery
  subtracts from the totals, until the fluid's share is garbage and the
  reset reaches out. The step limiter alone
  gives the same run to the fourth digit; the stage paste costs one kernel
  launch over a ball of 0.4% of the box and keeps every right-hand side
  reading the solution inside. `r_in = M` keeps the horizon `M/h` cells
  from the paste (four at `h = M/4`, TreeGH's margin `m ≥ G + 1`), and
  `3M/2` gains nothing.

  *Note (round 2):* the table above was measured under the relative
  potential hook, whose boundary mode dominated the outside error; with
  option 1 the outside error is 3–10 times smaller, and round 3 measures the
  interior again.

**No interior, the ordinary reset inside** *(option 3, measured in step 9,
round 3, for Erik's open decision on the interior; `bin/curved_localise.jl …
interior=false` against the default paste from the same tree, the absolute
potential hook, both cases; jobs 588498–588503, 588592; counting radius
`1.5M`)*. Outside `2M` at `t = 50M`:

| run | `D` | `𝔈ˣ` | `√γ q` (L1) | resets stage/step, counted at `r ≥ 1.5M` | resets deep inside | injected `D`, `τ` | `∫√γ q` over the octant |
|---|---|---|---|---|---|---|---|
| Wald `M/4`, paste | 2.9810e−3 | 4.578e−4 | 8.58e−4 | 0/0 | 0/0 | 0, 0 | 4.9e−13 |
| Wald `M/4`, none | 2.9798e−3 | 4.622e−4 | 9.46e−4 | 0/0 | 0/0 (2109/1055 by 100M) | 0, 0 (2.1e−12, −1.3e6 by 100M) | 4.2e−2 |
| Bondi `M/4`, paste | 6.3065e−3 | 3.487e−4 | 9.87e−4 | 0/0 | 0/0 | 0, 0 | −8.8e−17 |
| Bondi `M/4`, none | 6.3071e−3 | 3.486e−4 | 9.87e−4 | 0/0 | 15307/7019 | 1.7e−12, −1.5e6 | 6.7e−16 |
| Wald `M/8`, paste | 5.8805e−4 | 1.154e−4 | 2.74e−4 | 0/0 | 0/0 | 0, 0 | 1.4e−12 |
| Wald `M/8`, none | 6.0775e−4 | 1.373e−4 | 6.10e−4 | **5702/2852** | 564192/291994 | **44.7**, −2.5e6 | −0.65 |
| Bondi `M/8`, paste | 1.6497e−3 | 1.065e−4 | 3.51e−4 | 0/0 | 0/0 (9000 stage repairs) | 0, 0 | 2.9e−17 |
| Bondi `M/8`, none | 1.6497e−3 | 1.065e−4 | 3.51e−4 | 0/0 | 68588/36226 | 7.0e−13, −2.9e7 | −9.5e−5 |

- **Every reset inside is level 1 but Wald's at `M/8`.** The radial layers:
  Bondi without the interior resets only at `r < 0.75M`, every one onto an
  entropy face — `S_CEILED` in `0 ≤ r < M/4`, `S_FLOORED` from `M/4` to
  `0.75M` — and never to the atmosphere (level 2: 0); Wald at `M/4` only in
  the corner layer, `S_CEILED`, from between 50M and 100M on. Wald at `M/8`
  goes further: resets from the corner out to `1.25M` with every level-1 face
  (`S_FLOORED`, `S_CEILED`, `W_CEILED`, `ρ_FLOORED`, `NONFINITE`, a few `YE`)
  and **level 2, the atmosphere**, 3909/1594 of them, from the corner to
  `1.25M`; and `S_FLOORED` resets at `1.5–2M`, counted. A rerun watching
  every step (job 588592, cn079, EPYC 7532; the same run to every digit)
  dates it: outside `2M` the charge leaves its level from `t ≈ 33M` (`L1`
  2.87e−4 at 33M, 4.6e−4 at 35M, 1.0e−3 at 36M), and **the first counted
  reset is at `t = 37.65M`, `r = 1.575M`**. **At `M/16` Wald without the
  interior blows up**: the first counted reset at `t = 10.36M`, `r = 1.501M`
  (job 588630, cn085, EPYC 7532, watched every step), the charge outside `2M`
  up eight-fold by 11M (7.3e−4 against 9.2e−5 at 10M), and in the unwatched
  run (588496) `τ`'s drift outside `2M` 2.2·10² at 15M and 1.9·10¹¹ at 20M. The
  earlier the finer: 37.65M at `M/8`, 10.36M at `M/16`, none by 100M at `M/4`
  — the singularity's neighbourhood, resolved further, holds states the
  reset no longer tames. The magnetized Bondi flow without the interior is
  stable at `M/16` too (job 588497, cn084, EPYC 7532): outside `2M` the
  paste's run to five digits (`D` 4.1891e−4 at 50M), 387585/201062 resets
  all at `r < M/2`, all level 1 (`S_FLOORED`, `S_CEILED`, two `W_CEILED`),
  none to the atmosphere, injected `τ` `−9.7·10⁷`, the charge's total
  `−7·10⁻¹⁰`; with the paste it resets 126000/42000 at `r < M/2` as well
  (jobs 588113, 588503), the cells between the paste and the singularity's
  corner.
- **Outside, the paste and none agree** but in Wald at `M/8`: to 0.04% in
  `D` and to 1–10% in `𝔈ˣ` and the charge at `M/4`, to five digits in Bondi
  at both resolutions. Wald at `M/8` without the interior is disturbed from
  between 30M and 40M (`𝔈ˣ` 2.2 times the paste's at 40M, the charge 11
  times), and partly recovers by 50M (`D` +3%, the charge 2.2 times). So
  "nothing inside reaches outside" holds for Bondi and for Wald at `M/4`, and
  **not for Wald at `M/8` and `M/16` without an interior** (at `M/16` it
  blows up).
- **The charge's total.** `∫√γ q` over a region is the discrete flux of `𝔈`
  through its surface (the centred divergence telescopes). With the paste
  the region's inner surface sits in the exact, uncharged solution and the
  total is rounding; without it the neighbourhood of the singularity, where
  the scheme is not consistent, carries a net charge (`|√γ q|` up to 15 at `M/4`
  and 2·10³ at `M/8` in `r < M/4`), and every total that includes or excludes
  it shows it: `4.2·10⁻²` over the octant at `M/4` (`−0.65` at `M/8`),
  `1.3·10⁻³` beyond `1.5M`. Round 1's `−4.0·10⁻³` beyond `r = M`, without the
  interior, is this. Bondi's charge beyond `1.5M` stays at rounding without
  the interior (`6·10⁻¹⁵`), and over the octant `−9.5·10⁻⁵` at `M/8` (rounding
  at `M/4`): `IdealConductor` sets `E = −v × B` in every cell every stage, so
  little accumulates.
- **Injected energy**: without the interior the reset removes `10⁶`–`10⁷` of
  `τ` from the singularity's neighbourhood over the run (the entropy ceiling
  lowering `τ` where the infall heats the gas without bound), and Wald at `M/8`
  injects 44.7 of rest mass. Deep inside, not counted (decided by Erik
  2026-11-02), but the totals carry it.

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

*As built in step 8* (`src/refinement.jl`, `evolve!` in `src/driver.jl`;
proposed in step 8 unless marked otherwise):

- **The indicator** (`mhd_flags`): Löhner's, with TreeHydro's local and
  global noise floors (`ε = 1/100`, `ε_g = 1/1000`), on `ρ`, `|B|` and `|E|`
  of each cell and its neighbours along every dimension, from a current
  `P` (ghosts included) — the density for the matter's waves and the two
  field magnitudes for the electromagnetic ones, which in an electrovacuum
  move where `ρ` does not. TreeHydro's four marks (`(Refine, box)` above
  `refine_tol` below `maxlevel`, `(Keep, box)` above `coarsen_tol` as a
  travelling margin, `Coarsen` where nothing fired, else `Keep`), through
  TreeAMR's `firing_boxes`, so it runs on every backend. The global
  references are the largest `ρ` and the largest component of `Bⁱ` and of
  `Eⁱ` (a `mesh_mapreduce` reduces one variable at a time; a reference off
  by `√3` moves a noise floor). `refine_tol`, `coarsen_tol` and `maxlevel`
  have no default, and `coarsen_tol < refine_tol` is required. The
  geometric boxes around a star are step 10's.
- **The regrid** (`regrid_state!`, at the start of every chunk after the
  first): `regrid!` over every field set (`regrid_pairs`) — `U`, `A_j`, `Φ̃`
  with their schedules and each with its own Dirichlet hook at the
  regrid's time (AMR-2), `P`, `B_d`, `F_d`, `EMF_d` and `X` `=> nothing`,
  resized and never transferred (`B` is never state: the regrid moves `A`,
  and `B` is its curl). Then the problem is rebuilt over the same sets
  (`rebuild`: schedules, interface schedules, per-block geometry and outer
  faces), the state gathered, the potentials' low walls prescribed (AMR-8),
  and the reset run once as the step limiter, so that a prolongated
  conserved state the recovery would repair is repaired in the state and
  its injection counted with the step limiter's (TreeHydro's post-regrid
  reset). The integrator's scratch is dropped when the mesh moved.
  Before `regrid!`, every evolved set is filled with the complete outer
  boundary condition, and `regrid!` gets no hook for a potential, so its
  own pre-transfer fill keeps that condition bit for bit (amended in step
  8, decided by Erik 2026-10-15; see "Boundaries").
- **The initial mesh** (`adapt_case!`): TreeAMR's several-set
  `adapt_to_initial_data!` (AMR-3) over every field set, the evolved ones
  filled each pass from the case — the conserved state *at* each cell's
  centre, a point value standing in for the Gauss average while the mesh is
  chosen, and the potentials at their points — with their own hooks; the
  criterion recovers `P` from the filled sets and flags. On the converged
  mesh the exact initial data (`exact_state`, Gauss averages) is written.
- **A static mesh**: `evolve!`'s `refine(forest)` refines the root mesh
  before anything is built, for the coarse-fine tests.
- **The operators** *(decided in step 8)*: the matter Conservative width 3,
  **bounded**, in its ghost fill and in its regrid (see below); `A` and `Φ̃`
  PointValue **width 4** — prolongation and restriction (step 5's proposal,
  now measured). The Alfvén wave crossing coarse-fine faces
  (`bin/interface_order.jl`; MC, the matter bounded):

  | potentials' width | 1D, one period, `S_y` L1 / L∞ rate | 2D diagonal, an eighth, `S_x` L1 / L∞ rate |
  |---|---|---|
  | uniform coarse | 2.16 / 2.10 | 2.21 / 2.03 |
  | uniform fine | 2.07 / 2.09 | 2.52 / 1.88 |
  | 2 | 1.91 / **0.91** | 1.48 / **0.33** |
  | 4 | 2.11 / 2.04 | 2.14 / 2.02 |
  | 6 | 2.11 / 2.04 | 2.14 / 2.01 |

  (1D: 32 … 256 coarse cells with the middle half refined; 2D: 2 × 2 roots
  of 8², 16², 32² with one refined.) At width 2 the fine ghosts' `𝔅`, the
  curl of a linear interpolation, is first order, and so is the error at
  the face: L∞ loses an order in 1D and all of it in 2D. Width 6 is width 4
  to three digits, as a second-order scheme makes it. The two-level errors
  lie between the uniform coarse and fine ones (1D at 256 cells: L1 3.6e−4
  against 5.8e−4 and 1.4e−4). The matter's bound changed no rate.

  *(Re-measured in step 8 with `A_j` in the conservative family, AMR-9 —
  width 3 along the edge, the table's `w` across it — and `Φ̃` PointValue
  width `w`; `bin/interface_order.jl`:)* the 1D rows are the same to every
  digit printed — in 1D `A_y`, `A_z` are vertex sets, for which the two
  families are the same operators, and `A_x` is a constant — and in 2D
  width 2 gives 1.49 / **0.33**, width 4 2.14 / 2.02 and width 6 2.14 /
  2.01 (L1 errors at 32² 1.71e−3 at width 4 and 6, against 1.85e−3 and
  3.10e−4 uniform). **The width stays 4** (decided in step 8 again): the
  conservative direction changes nothing the reconstruction at the face
  sees to three digits, and width 2 still loses the order across it. With
  the matter's ghost fill in primitives at width 2 as well (AMR-10, below)
  the potentials' widths 2, 4, 6 give 1.91 / 0.88, 2.09 / 1.97, 2.08 / 1.97
  in 1D and 1.45 / 0.08, 2.13 / 1.85, 2.13 / 1.84 in 2D: width 4 again.
- **The gauge damping** *(decided in step 8; amended in step 8, decided
  by Erik 2026-10-15: "choose the same ξ on all meshes")*: `evolve!` uses
  `ξ = 16/L` on every mesh, uniform or refined, `L` the longest side of the
  box — the longest, so that a case's extrusions and turns, whose boxes
  differ only along the added or exchanged directions, damp alike and stay
  its run bit for bit (`default_ξ`); a caller's `ξ` overrides it. (As first
  decided it was 16 over the shortest side on a refined mesh and 0 on a
  uniform one.) On a uniform mesh the gauge never reaches `𝔅` —
  `−∇(αΦ − β·A)` is a discrete gradient, whose discrete curl vanishes to
  rounding — so there `ξ` moves the gauge part of `A` and `Φ̃` and, through
  rounding, nothing else; what it moved in the suite is listed under
  "Validation ladder" 2a and below. Across a coarse-fine face it does, since the
  prolongation of a coarse gradient is not a fine one. A pure gauge pulse
  (`gauge_pulse_case`: `A = 0`, `Φ̃` a Gaussian, nothing else;
  `bin/gauge_damping.jl`) on the periodic unit box, 2 × 2 roots of 16² with
  one refined, to `t = 2` (measured with the PointValue potentials of step
  8's first form; below the table, with AMR-9's):

  | `ξ` | 0 | 1 | 4 | 16 | 64 |
  |---|---|---|---|---|---|
  | `max |𝔅|` over the run | 0.107 | 0.103 | 0.093 | 0.079 | 0.054 |
  | `max |𝔅|` at `t = 2` | 5.3e−2 | 2.1e−2 | 6.6e−3 | 1.4e−3 | 3.3e−4 |
  | `max |Φ̃|` at `t = 2` | 0.25 | 8.3e−2 | 4.0e−3 | 9.4e−5 | 5.2e−4 |

  against at most 5.3e−15 on the uniform mesh at any `ξ`. Without damping
  the pulse keeps crossing the faces of the periodic box and the spurious
  field stays (at 8² roots it grows, 0.079 at `t = 1` to 0.18 at `t = 2`);
  the damping takes it away with the pulse. The peak at the first crossing
  is set by the interpolation, not by `ξ`. The oblique Alfvén wave on the
  same mesh moved in the fifth digit of its L1 errors at most for any of
  these `ξ`. 16 is where the remainder has fallen by 40 and `Φ̃` by 2600
  within two light crossings. Beyond it the gauge mode is overdamped: `Φ̃`
  follows `−∇·A/ξ`, and the longitudinal `A` diffuses (with coefficient
  `1/ξ`) instead of leaving at the speed of light — at 64 the field's
  remainder is smaller but five times more `Φ̃` is left at `t = 2`. And `ξΔt =
  16Δt/L ≪ 1` at any resolution a box of side `L` is run at, far inside
  the explicit part's stability.

  *(Re-measured at the end of step 8 with the potentials' conservative
  transfer, AMR-9, and the matter's ghost fill in primitives, AMR-10:)*
  the spurious field is much smaller, since a coarse edge's mean is
  what the fine edges along it carry and the prolongated gradient is closer
  to a fine one:

  | `ξ` | 0 | 1 | 4 | 16 | 64 |
  |---|---|---|---|---|---|
  | `max |𝔅|` over the run | 3.6e−2 | 1.6e−2 | 4.8e−3 | 3.9e−4 | 2.0e−5 |
  | `max |𝔅|` at `t = 2` | 1.6e−2 | 6.2e−3 | 3.5e−4 | 7.2e−6 | 5.7e−6 |
  | `max |Φ̃|` at `t = 2` | 0.25 | 8.4e−2 | 4.0e−3 | 8.0e−5 | 5.3e−4 |

  (uniform: at most 5.3e−15). The oblique wave's L1 errors again move in
  the fifth digit at most. At `ξ = 16` the remainder falls by 2200 within
  two light crossings and `Φ̃` by 3100; beyond it `Φ̃` is overdamped as
  before. `ξ = 16/L` stands (Erik's decision of one `ξ` on every mesh, and
  this table agrees). In the suite's 8² version the undamped field at
  `t = 2` went from 0.18 to 0.049 and the damped one from 1.2e−3 to 1.6e−5
  (0.044 and 1.9e−5 with AMR-9 alone).

**The ghost fill of the matter** *(decided by Erik 2026-10-15: in
primitives; amended 2026-10-23, decided by Erik: the parent-centred
primitive fill, the only one)*: **the ghost fill exchanges the
primitives** `W = (ρ★, s, Yₑ, uⁱ = Wvⁱ, Eⁱ)` — copied at same-level faces,
exactly, and at refinement boundaries prolongated with TreeAMR's bounded
conservative operator centred on the parent (`default_ops_W()` =
`Operators(; prolongation=3, restriction=2, family=Conservative,
bounded=true)`, AMR-4: the parent plus an MC-limited linear slope, scaled so
that every child stays within the parent and its face neighbours —
Athena++'s prolongation of the primitives, Stone et al. 2020, ApJS 249, 4,
§2.1.5, with MC for its minmod; a coarse ghost the `2^D` mean of the fine
cells it covers). **`P` in every ghost comes from `W` by one forward EOS
evaluation (`P_from_W`), with the ghost's own `B`; no ghost is ever
recovered**: the recovery runs on the owned cells only. **The conserved
ghosts are not written by the scheme.** Nothing reads them — the fluxes,
the EMFs, the sources, `q` and the indicator (`mhd_flags`) read `P`; the
stage solve and the reset read the owned cells of the state vector; the
totals and the L1 errors the owned cells — so `U`'s ghosts hold only what
TreeAMR's own fills of `U` leave there: `regrid!`'s before its transfer,
and the initial adaptation's (each with the case's conserved hook). **The
regrid stays conservative**, Conservative width 3, bounded, on the
conserved variables. *(Amended 2026-10-23: until then `ghost_kernel!` also
rebuilt every conserved ghost by `prim2con`, which nothing read; step 8's
PointValue width-2 fill of `W` was the default until the parent-centred
operator replaced it after the experiments below, and the experimental
density forms of `W` — `ghost_vars = :density`, `:mass` — are gone from
`src/`.)*

**Three working fills, to revisit when the physics gets harder**
*(recorded 2026-10-23 at Erik's request)*. Each ran the atmosphere tube
across a coarse-fine face and the suite. The one-line result is the 48
perturbed magnetized atmosphere tubes of `bin/ghost_robustness.jl` (the
table below; `Bʸ = 10⁻³`, cut off at `μ_max = 10`) under EntropyEOS 1.0's
policy, and in brackets under the atmosphere-at-rest recovery with
`collapse_to_atmosphere` (EntropyEOS 1.1, 2026-10-21;
`bin/front_investigation.jl`, "Floors, the atmosphere and the
magnetosphere"):

1. **Conserved, bounded**: `U` filled with Conservative width 3, bounded,
   and the recovery in every stored cell, ghosts included — step 8's first
   fill (`claude/step-8-finish` at 7605986; run as `git archive 7605986`):
   3/48 blow-ups (2.1–17%), fallbacks median 49, 83 repaired ghost
   recoveries median [0/48, no fallback, largest `W` 5.0, injected `D`
   −1.4·10⁻⁸]. It costs `3((N + 2G)/N)^D − 3` more recoveries per cell and
   right-hand side: a 3D step 64–74% longer ("Primitive recovery",
   2026-10-23).
2. **Parent-centred primitives** (**the code**): `W` with the specific
   `s`, `Yₑ`, Conservative width 3, bounded (`claude/step-8-ghost-athena`,
   1f014c8; the runs on `claude/step-8-ghost-robustness`, 6b0a72e, method
   `prim-cons3-specific`): 0/48 (0–7.4%), fallbacks median 72, no repaired
   ghost, a ghost pressure outside its neighbours' range 9 times a run
   (median) [0/48, no fallback, largest `W` 5.0, injected `D` −1.4·10⁻⁸].
3. **Density-form primitives, parent copy**: `W` holding `ρ★ s`, `ρ★ Yₑ`
   (`ghost_vars = :density`, from `claude/step-8-ghost-entropy-density`,
   19f4563), Conservative width 1 — every child its parent
   (`claude/step-8-ghost-robustness`, 6b0a72e, method
   `prim-cons1-density`): 0/48 (0–7.4%), no ghost pressure or entropy out
   of range, but fallbacks median 155 and `W` at its cap of 100 in most
   runs [not re-measured].

No difference in blow-ups is significant at 48 runs (below). What the code
lost is the second and third: they run on their branches, and
`bin/ghost_entropy.jl`, `bin/ghost_robustness.jl` and
`bin/interface_order.jl` say so when asked for them.

*As built in step 8, when AMR-10 landed* *(history: superseded 2026-10-23
by the fill above; the measurements stand as measured)* (`primitives_from_sets!`,
`P_from_W`, `ghost_kernel!` in `src/evolution.jl`; "The right-hand side"):
- **The set** `W`, cell-centred, 9 variables: `ρ★`, `s`, `Yₑ`, the
  contravariant `uⁱ = Wvⁱ` and the contravariant, undensitized `Eⁱ` — Erik's
  list, with `uⁱ` rather than `vⁱ` so that any interpolated value is a
  subluminal velocity, and `E` undensitized, as a primitive (proposed in
  step 8). Scratch: resized, never transferred by a regrid.
- **The recovery runs on the owned cells only**, before the exchange, and
  writes `W` and `P`; `W`'s fill has its own hook, the case's primitives at
  each ghost centre (`DirichletBoundary{:W}`, a function of position, so the
  fill applies the complete condition); at every ghost cell `P` comes from
  `W` by one forward EOS evaluation, no inversion, with the ghost's own `B`
  (and no recovery flags), and the conserved ghost from `prim2con`. Owned
  cells take `P` through the same function, so that a copied ghost is its
  owner bit for bit (see "The right-hand side").
- **The width** *(proposed in step 8, by the measurements below)*: **2**,
  with `bounded = true` (which width 2 does not need: it is the multilinear
  interpolation, accepted with the flag so that a width can be tried
  without touching it). Width 2 interpolates every variable with the same
  convex weights, so a ghost is a mixture of the states that bracket it. At
  widths 4 and 6 the bound clamps each variable on its own, which keeps
  every primitive in range but not the *state* consistent: where the
  atmosphere tube's front starts on a coarse-fine face, a coarse ghost
  restricted from a dense cell (`ρ = 0.088`, `s = 23.7`) and a heated
  atmosphere cell (`ρ = 10⁻⁸`, `s = 31.4`) took a density of 0.028 and the
  bracket's highest entropy, a pressure of 5.1 between neighbours at 0.2
  and 10⁻¹⁰; the reset then injected 5.4e−4 of energy (width 6: 1.8e−3),
  and the magnetized tube blew up at width 4. A per-variable bound does
  not make a thermodynamically admissible state out of a mixture; only
  common weights do. That costs the smooth interface order a little in L∞
  in 2D (below).
- **Conservation does not depend on it**: at a coarse-fine face it is
  `restrict_interfaces!` on the fluxes; the regrid's bounded transfer of the
  conserved variables is exactly conservative.

Measured on the atmosphere tube (a gas at `ρ = p = 1` into a `10⁻⁸`
atmosphere, 200 cells, `[0, 0.2)` refined, which the front crosses; over the
run: the interior ghosts the policy would repair for anything but the
atmosphere — with conserved ghosts the recovery of the ghost's conserved
state, with primitive ghosts `check_prim_state` of the ghost's primitives —
and the rest mass and energy the reset injected; `bin/ghost_fill.jl`, the
conserved rows from the tree before this change, `git archive 7605986`):

| | hydrodynamic: repaired ghosts; injected `D`, `τ` | magnetized (`Bʸ = 10⁻³`, cut off at `μ_max = 10`) |
|---|---|---|
| uniform (conserved fill) | 0; −8.4e−12, −2.2e−10 | 8; 2.5e−8, 6.9e−8 |
| uniform (primitive fill) | 0; −8.4e−12, −2.2e−10 | 0; 5.0e−8, 1.7e−6 |
| conserved, unbounded | 9 (non-finite: `D < 0`); 2.0e−9, 1.5e−8 | 120 (non-finite); 4.2e−8, 8.6e−7 |
| conserved, bounded (step 8's first) | 0; −1.5e−11, −2.5e−10 | 33 (`W` capped, entropy window); 3.5e−8, −6.1e−8 |
| primitives, width 2 (**chosen**) | 0; 5.6e−9, 2.4e−9 | 0; 5.4e−8, 1.3e−6 |
| primitives, width 4 | 0; −6.5e−12, **5.4e−4** | 0; **2.5e3, 5.3e12** |
| primitives, width 6 | 0; 1.9e−12, **1.8e−3** | 0; 4.6e−8, 1.2e−6 |
| primitives, width 4 unbounded | 71 below `ρ_atm`; −2.4e−6, non-finite | 17 (+ 72 below `ρ_atm`); non-finite |

The magnetized injections are not comparable between rows below 10⁻⁶: the
magnetized tube is chaotic at that level — perturbing its initial state by
10⁻¹⁵ moves the uniform run's injected energy between 3.9e−8 and 2.1e−6
(measured in step 8, on the tree before and after the change alike) — and
the primitive fill's owned `P` differs from the recovery's by rounding,
which moved the uniform row. The hydrodynamic tube is not chaotic: its
uniform row is the same to every digit in both fills.

What the measurements say *(proposed in step 8, for Erik; resolved
2026-10-23, decided by Erik: the parent-centred primitive fill, the only one
since)*: the primitive
fill keeps every ghost admissible — no ghost the policy would repair, none
below the atmosphere, magnetized as well (the AMR-10 downstream check,
against 33 repaired recoveries with conserved ghosts) — but on the
hydrodynamic tube it injects 5.6e−9 of rest mass where the bounded
conserved fill injected 1.5e−11 and the uniform mesh 8.4e−12: mixing
primitives (a coarse ghost restricted from a dense and an atmosphere cell)
makes more internal energy than mixing the conserved state, which is the
energy-conserving mixture. Not an error — the reset's injection is a
correction the scheme accounts — but larger than step 8's first fill, and
an argument for prolongating the pressure instead of the entropy, which
needs the entropy from `(ρ, p)` (EOS-3), or for keeping width 2. The fill
runs as Erik decided, at width 2, until that is settled. *(Settled
2026-10-23 by Erik: the parent-centred fill, above.)*

*Experiment (branch `claude/step-8-ghost-entropy-density`; decided against
2026-10-23):*
`W` holding the densities `ρ★ s`, `ρ★ Yₑ` (`ghost_vars = :density`, the
default on that branch), and also `ρ★ uⁱ` (`:mass`), so that with common
weights the ghost's entropy is mass-weighted; `bin/ghost_entropy.jl` has the
measurements, reported to Erik. In short: at width 2 every ghost's pressure
and entropy stay within their neighbours' range and the magnetized tube
never blew up in twelve perturbed runs (the specific-`s` fill did, twice),
but the hydrodynamic tube injects 1.6e−8 of rest mass, more than
the specific fill — the injection is not the entropy's but any point-value
fill's: a fine ghost inside a coarse cell at a front holds mass the coarse
cell does not (2.96e−6 against 1e−8), which the light-cone HLLE flux draws
out of the coarse cell through the coarse-fine face. Widths 4 and 6 are not
robust in either form.

*Second experiment (branch `claude/step-8-ghost-athena`; the code since,
decided by Erik 2026-10-23):*
`W` filled with TreeAMR's bounded conservative operator, `Operators(;
prolongation=3, restriction=2, family=Conservative, bounded=true)` — a
linear reconstruction centred on the parent with MC slopes, scaled to keep
every child within the parent and its face neighbours: Athena++'s ghost
prolongation of the primitives (Stone et al. 2020, §2.1.5) but for MC
against minmod. At the foot of a jump a child takes its parent's value, so
the dense neighbour no longer leaks into an atmosphere cell's children. On
the specific `s`, `Yₑ`: the hydrodynamic tube injects −1.6e−11 of rest mass
(the conserved fill −1.5e−11), nothing is refilled beyond the face, no
magnetized run of twelve blew up, the interface order is the conserved
fill's (1D 2.11 / 2.04, 2D 2.14 / 1.98). `bin/ghost_entropy.jl` has the
table, reported to Erik.

*Third experiment (branch `claude/step-8-ghost-robustness`, from
`claude/step-8-ghost-athena`; the three fills above):* robustness on the
magnetized atmosphere tube across a coarse-fine face, which is chaotic at
rounding level, over **48 perturbed initial states** per method
(`bin/ghost_robustness.jl`; runs 1–12 are the twelve of the earlier
tables, runs 13–48 amplitude 10⁻¹⁵ with seeds 2–37). A blow-up: injected
`D` above 10⁻⁶, `|τ|` above 10⁻³, or non-finite (the other runs inject
3·10⁻⁸ … 8·10⁻⁸ of rest mass). Ranges are min / median / max over the runs
that did not blow up; "p, s out" count the ghosts whose pressure or entropy
leaves the range of the owned cells within 1.5 coarse spacings, summed over
the run; wall time per run with five methods running at once on 12 cores.
The conserved rows are measured on the tree before the primitive fill
(`git archive 7605986`, the same scheme otherwise), the uniform mesh on
both trees.

| method | blow-ups (95% Wilson) | injected `D` | injected `τ` | fallbacks | max `W` | ghosts repaired | p out | s out | wall |
|---|---|---|---|---|---|---|---|---|---|
| uniform (conserved tree) | 1/48 (0.4–11%) | 2.5e−8 / 4.0e−8 / 6.0e−8 | 3.9e−8 / 8.8e−7 / 2.7e−6 | 0 / 10 / 202 | 37 / 83 / 100 | 3 / 28 / 75 | 0 | 0 | 0.90 s |
| uniform (primitive tree) | 0/48 (0–7.4%) | 2.4e−8 / 3.9e−8 / 7.6e−8 | 5e−10 / 8.5e−7 / 3.5e−6 | 0 / 26 / 238 | 30 / 82 / 100 | 0 | 0 | 0 | 0.79 s |
| (a) conserved, bounded | 3/48 (2.1–17%) | 2.7e−8 / 3.4e−8 / 5.6e−8 | −6.9e−7 / 8.6e−8 / 1.7e−6 | 0 / 49 / 425 | 40 / 87 / 100 | 7 / 83 / 163 | 21 / 47 / 69 | 1429 / 1443 / 1462 | 2.17 s |
| (b) primitives, specific, Conservative w = 3 | 0/48 (0–7.4%) | 2.5e−8 / 3.6e−8 / 5.1e−8 | −5.5e−7 / 1.3e−7 / 1.3e−6 | 0 / 72 / 279 | 47 / 92 / 100 | 0 | 1 / 9 / 22 | 0 | 1.91 s |
| (c) primitives, density, parent copy | 0/48 (0–7.4%) | 3.1e−8 / 4.0e−8 / 5.2e−8 | −3.2e−7 / 5.1e−7 / 5.1e−6 | 0 / 155 / 521 | 60 / 100 / 100 | 0 / 2 / 7 | 0 | 0 | 1.93 s |

The blow-ups: (a) run 22 (seed 12) injects `D` 2.5·10³ and `τ` 5·10¹²,
run 33 (seed 23) 88 and 2·10¹⁰, run 23 (seed 13) 0.74 and 1.4·10⁷; the
uniform mesh's (seed 12, the same seed as (a)'s run 22) 5·10³ and
1.9·10¹³. Fisher's exact test: (a) against (b) or (c), 3/48 against 0/48,
`p = 0.24`; (a) against the uniform mesh, 3/48 against 1/48, `p = 0.62` —
no difference is significant at 48 runs (telling 6% from 0% would take
about 150 runs per method).

Diagnosis (`claude/step-8-ghost-robustness`, traced step by step). Every
blow-up starts at the front running into the atmosphere, where `W` spikes
(30–40 every few steps in every method, up to the cap of 100 in the runs
that blow up): the stage solve falls back in the front cell, and a step
later the reset writes a repaired state that injects the mass. (a) run 22:
at `t = 0.157`, `x = 0.164` — sixteen fine cells inside the refined band,
not on a coarse-fine face — `W` 81 then 100, fallbacks there, and at
`t = 0.158` the reset injects `D = 10⁶` (density) at `x = 0.159`; run 33
the same at `t ≈ 0.11`, `x ≈ 0.12`; the uniform mesh's at `t = 0.364`,
`x = 0.353`. (a) run 23 is the one at a coarse-fine face: the front reaches
`x = 0.2` at `t = 0.199` with `W` 52 in the fine cell at 0.196, the first
fallback is in the last fine cell (0.199) at `t = 0.201`, and the reset
injects `D = 149` into the first coarse cell (0.2025) and `τ = −5·10¹³`
into the last fine one. So the front itself is fragile everywhere (the
uniform mesh blew up once), and one of (a)'s three is where a conserved
ghost carries the coarse field's energy across the face. (c) runs at the
`W` cap in most runs and has the most fallbacks (median 155); (b) has none
of either extreme. *(Investigation 2026-10-16, "Floors, the atmosphere and
the magnetosphere": the spikes are EntropyEOS's `S_FLOORED` repairs, the
blow-ups its collapse branch, on every mesh. *Decided by Erik 2026-10-21:*
a floored state is the atmosphere at rest (EntropyEOS 1.1's
`FLAG_POL_ρ_FLOORED`, in the recovery wrapper) and the tube's policy sets
`collapse_to_atmosphere`; measured, all three meshes are deterministic,
0/48, `W ≤ 5`, no fallback — the table above stays the record of the
released policy, the ghost fill's choice is unchanged and Erik's.)*
*(Amended 2026-10-24, EntropyEOS 2.0: without the option, on 2.0's
projection, the parent-centred fill's 48 runs are 0/48, no fallback, `W`
5.0 / 5.4 / 11.0, but two runs `10⁻¹⁵` apart differ by `1.3·10⁻⁴` of `D`
— "Floors, the atmosphere and the magnetosphere".)*

Smooth data across coarse-fine faces (the Alfvén wave of "The operators"
above, potentials at width 4, `bin/interface_order.jl ghost=w`), `S_y` /
`S_x` L1 / L∞ rates:

| matter's ghost fill | 1D, one period | 2D diagonal, an eighth |
|---|---|---|
| conserved, width 3 bounded (step 8's first) | 2.11 / 2.04 | 2.14 / 2.02 |
| primitives, width 2 (**chosen**) | 2.09 / 1.97 | 2.13 / 1.85 |
| primitives, width 4 | 2.11 / 2.01 | 2.12 / 2.06 |
| primitives, width 6 | 2.11 / 2.01 | 2.12 / 2.05 |

(L∞ of `S_x` at 32² in 2D: 1.14e−2 at width 2, 8.0e−3 at width 4, against
7.6e−3 and 2.0e−3 uniform coarse and fine.) Width 2 keeps second order in
L1 and loses a little in L∞ in 2D — the linear interpolation's error at a
fine ghost is `O(h²)`, which the reconstruction there differentiates.

## Julia version

`julia = "1.13"` *(amended 2026-10-15, twice: Erik raised the floor from
1.11 to 1.12, and then to 1.13 by his rule that the floor is raised when
the older version is more than about 20% slower; measurements below)*.
Before that it was `julia = "1.11"`, decided 2026-09-25 (and before that
`julia = "1.10"` wherever possible): all the Tree* packages — TreeAMR,
TreeWave, TreeHydro, TreeGeneralizedHarmonic and this one — shared the
1.11 floor, so that an unregistered dependency can be located with a
`[sources]` entry, a 1.11 key, rather than added by URL. That also
removed the reason for the weak-dependency split proposed earlier: TreeGH's
pin to 1.11 (its `LinearAlgebra = "1.11"` compat entry, its `[sources]`
entries, and ApparentHorizonFinder and KorzynskiSpin, whose checkouts both
declare 1.11) is satisfied by this package's floor, so TreeGH can be an
ordinary dependency and `TreeGRRMHDTreeGHExt` is not needed for the
version's sake. A dependency's own `[sources]` is still ignored when this
package resolves it, so what it needs from an unregistered package must be
named in this package's own `[sources]`.

Every Tree* package now has the 1.13 floor, and its CI tests only 1.13
*(amended 2026-10-16: Erik's decision for all of them, because supporting
the older versions slows development and is not useful at the moment;
until then the siblings kept 1.11, which did not constrain this package,
since a dependency's floor below this package's is satisfied by every
version this package supports)*. `[sources]` is still what locates the
unregistered dependencies; 1.13 reads it as 1.11 did. Every `[compat]`
entry resolves to the same releases at 1.12.7 and 1.13.1.

*(Measured 2026-10-15, the whole suite at one thread, 25293 tests, on an
Apple M3 Pro shared at a load of about 4–8 on 12 cores, 1.12.7 and 1.13.1
run back to back and in alternating order, two runs each; `Test Summary`
time, and the compile time `Base.cumulative_compile_time_ns` counts inside
the suite:)*

| bounds checks       | 1.12 suite | 1.13 suite | ratio | 1.12 compile | 1.13 compile | ratio |
|---------------------|-----------:|-----------:|------:|-------------:|-------------:|------:|
| `auto`              |   5m22s    |   5m05s    | 1.06  |    168 s     |    150 s     | 1.12  |
| `--check-bounds=yes`|   6m00s    |   5m12s    | 1.16  |    198 s     |    161 s     | 1.23  |

With bounds checks on — what every CI cell runs, and what `Pkg.test()`
itself does at 1.11 and 1.12, which force `--check-bounds=yes` where 1.13
does not — 1.12 compiles 23% longer, most of it in the 2D and 3D kernels
(`multid_tests.jl` 55 s against 42 s, `multid_cases_tests.jl` 38 s against
30 s); that is over the rule's threshold,
so the floor is 1.13. Without bounds checks the gap is half that (12% and
6%), which is the case for keeping 1.12 if the rule is meant for the
default configuration. 1.11 with bounds checks (the same suite, the same
afternoon) took 6m05s and 204 s of compiling, close to 1.12's. The
numbers the suite prints are identical across the change at each version,
and 1.12 differs from 1.13, at the same flags, only in the last digits of
six recorded worst cases.

CI runs three cells, all at 1.13: macOS/arm64, Linux, and Linux at four
threads *(amended 2026-10-16: there was a fourth, macOS at the current
release, which repeated the 1.13 macOS cell until 1.14; CI now tests only
the floor)*. The macOS cell carries the coverage, which at 1.13 costs 1.24x (6m27s against
5m12s, both with bounds checks; TreeHydro had measured 3.14x at 1.13
before, against 1.04x at 1.11).

What 1.11 and 1.12 needed and 1.13 does not is gone. The suite's
allocation probe (`rec_allocations` in `test/recovery_tests.jl`) was a
`Vararg{Any,N}`-specialized method because 1.11 did not specialize a plain
`args...` there and boxed the result, 16–672 bytes a call on all 50 of the
probe's calls, where 1.12 and 1.13 allocate nothing either way. Nothing in
the package or the suite was there for 1.12 alone. The other difference
runs the other way: 1.11 refused to redefine a `const`, and 1.12 and later
do it silently, so the duplicate-constant check at the top of
`test/runtests.jl` is now the only thing that finds a top-level name two
test files share.

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

     *Per reconstruction (measured in step 7,* `bin/reconstruction_comparison.jl`*;
     the same runs, 100–400 cells, each method against its own 3200-cell run,
     MP5 at its `cfl = 1/5`):* L1 rates of `D`, `S_x`, `τ`, `𝔅ʸ`, and the worst
     plateau error at 400 cells (region in brackets).

     | test | MC | PPM | WENO-Z | MP5 |
     |---|---|---|---|---|
     | 1 | 0.78, 0.85, 0.77, 0.85; 1.26% (R5) | 0.82, 0.85, 0.83, 0.84; 1.43% (R5) | 0.78, 0.82, 0.82, 0.75; 1.55% (R5) | 0.77, 0.91, 0.87, 0.86; 0.84% (R5) |
     | 2 | 0.66, 0.67, 0.74, 0.81; 1.91% (R5) | 0.65, 0.70, 0.76, 0.80; **2.78%** (R5) | 0.73, 0.79, 0.84, 0.91; 1.47% (R5) | 0.87, 0.79, 0.82, 0.90; 0.79% (R5) |
     | 3 | −0.02, 0.14, 0.17, 0.53; **4.69%** (R4) | 0.49, 0.51, 0.53, 0.56; 0.23% (R4) | 0.16, 0.30, 0.31, 0.54; **12.6%** (R4) | 0.45, 0.16, 0.18, 0.54; 0.17% (R4) |
     | 4 | 0.95, 1.04, 0.94, 0.93; 0.64% (R4) | 0.98, 0.87, 0.97, 1.09; 0.24% (R4) | 0.61, 0.89, 0.82, 0.89; **3.73%** (R4) | 0.82, 1.04, 1.01, 1.09; **4.53%** (R4) |
     | 5 | 0.90, 0.75, 0.78, 0.97; 0.63% (R2) | 0.95, 1.00, 0.77, 1.03; 0.49% (R2) | 0.81, 0.77, 0.78, 1.01; 0.34% (R2) | 0.99, 0.81, 0.88, 1.06; 0.09% (R5) |

     Against the stored MC reference instead (which the suite reads), PPM's
     Balsara 3 rates are 0.54, 0.76, 0.77, 0.61. Balsara 3's largest density
     at 400 cells is 8.35 (MC), 8.22 (PPM), 7.39 (WENO-Z), 8.06 (MP5), of the
     exact shell's 10.1. Every stage of every run ideal and unrepaired; no
     reset fires but MP5's in Balsara 4, which writes one cell as the step
     limiter in the first steps at each resolution (0.033 of energy at 100
     cells). So: **PPM makes Balsara 3 converge** at the tested resolutions —
     every error falls, and `R4` comes within 0.23% — and the suite now
     asserts it for PPM against PPM's own 3200-cell reference
     (`test/data/balsara3_ppm.ref`, rates ≥ 0.4, every plateau within 2%;
     added in step 7); MC keeps its pinned values. WENO-Z is the worst of the
     four at the shell (its overshoot at the contact smears `R4` by 12.6%)
     and MP5 converges in `D` and `𝔅ʸ` but barely in `S_x` and `τ`; neither is
     asserted. The MC references in `test/data/` were not regenerated: rerun,
     four of them are identical byte for byte, and Balsara 1's differs in 314
     of 4800 `Float32` values by one rounding (1.9e−9 at most), the change of
     the initial data's Gauss average (step 7) in the last bit of a `Float64`.
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
     1.7e−12 of its scale under either tableau. *(Amended 2026-10-30:)*
     with the step cache, to the recovery's tolerance — at most 8.4e−11
     after 13 steps and 3.6e−10 after 1000 — asserted `≤ 10³·tol` and
     `≤ 2·10³·tol` ("The step cache", "The uniform flow").
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
     and `t = 9` the distance to the profile is 9.3e−4 (9.6e−4 since step 8
     prescribed the potentials' low walls) — "indistinguishable"
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
     *(Amended in step 8:)* **the magnetized tube is chaotic at rounding
     level**, and its fallbacks, injected energy and largest `W` are not
     properties of the scheme: perturbing its initial state by 10⁻¹⁵ (twelve
     runs, on the tree before step 8's primitive ghost fill and after it)
     gives 0 to 144 fallbacks of its 160,000 cell-stages, an injected energy
     from 3.9·10⁻⁸ to 2.1·10⁻⁶ and a largest `W` from 4.6 to 90, while every
     run stays finite, above the atmosphere, and conserves the totals but
     for exactly what the reset injects. The primitive ghost fill, whose
     owned `P` differs from the recovery's by rounding, moved the suite's
     run to 92 fallbacks, `2.5·10⁻⁸` → `5.0·10⁻⁸` of rest mass and
     `6.9·10⁻⁸` → `1.7·10⁻⁶` of energy, and `W` 4.69 → 6.63; the test now
     asserts no fallback hydrodynamically and a fallback rate of at most
     0.2% magnetized (proposed in step 8; *replaced by "no fallback"
     magnetized as well, decided by Erik 2026-10-22*). *(Investigation 2026-10-16: the
     cause — the field's truncation error in the fluid's share, amplified
     by EntropyEOS's repairs at the edge of the valid range — the sweep of
     fixes, and the decision of 2026-10-21 (a floored state is the
     atmosphere at rest; the tube's policy sets `collapse_to_atmosphere`),
     under "Floors, the atmosphere and the magnetosphere": the tube is then
     deterministic with no fallback, and the suite's run prints `W` 4.80,
     injected −1.18·10⁻⁸, −7.4·10⁻⁷. *Amended 2026-10-24, EntropyEOS 2.0:*
     without the option on 2.0's projection the suite's run prints the
     same; across 48 perturbed runs every one is clean, but no longer the
     same run.)* The same holds for step 6's
     tableau comparison below: "SSP3 no fallback, ARS 7" on this tube is one
     draw from that spread, and the decision for SSP3(4,3,3) rests on its
     other rows (the near-vacuum rarefaction, the ideal atmosphere; *amended
     2026-10-22:* the ideal atmosphere's row was the policy artefact, and the
     re-measured comparison in "The partition" — ARS 23 fallbacks, SSP3 none,
     both deterministic — is what the tube says now). The
     hydrodynamic tube is not chaotic: both fills give its numbers to every
     digit.
   - Cost (Julia 1.13, Apple M3 Pro, one thread, 400 cells): a step of
     Balsara 1 is 6.5 ms, of Balsara 3 15 ms (the recoveries at a pressure
     ratio of 10⁴); the three test files take 45, 37 and 11 s.
2. **Flat space, 2D and 3D, with AMR**: the cylindrical explosion, the
   magnetic rotor, Orszag–Tang, field-loop advection across coarse-fine
   faces — `∇·B` at round-off, rest mass and magnetic flux conserved
   across levels to round-off, regrids included.

   *Rung 2a, measured in step 7* (`test/multid_tests.jl`,
   `test/reconstruction_tests.jl`; uniform meshes, HLLE, SSP3(4,3,3), `cfl`
   2/5 — 1/5 under MP5 — over `D` in `Δt = cfl/(D max λ/h)`; the cases in
   `src/cases/multid.jl` and `src/cases/geometry.jl`, every published number
   copied with its citation; Julia 1.11 and 1.13, one and four threads —
   identical at one and four threads, and between 1.13 and 1.11 (or
   `--check-bounds=yes`) the same to the digits recorded but the
   conservation drifts' last digit):
   - **`∇·𝔅`** after every step of every 2D and 3D run of the suite
     (`max_div_B` as the driver's `after_step`): at most 2.4e−16 of
     `max |𝔅|` (asserted ≤ 1e−12, PLAN.md's bound).
   - **Translation invariance**: the Alfvén wave extruded along `y`
     (`extruded_case`) is the 1D run bit for bit (`isequal`, every evolved
     set, every point) under MC, PPM, WENO-Z and MP5, and extruded along `y`
     and `z` in 3D under MC and WENO-Z; Balsara 1, with its Dirichlet faces,
     likewise in 2D. Step 5's 2.7e−15 was the initial data's Gauss weights
     (amended in step 7: "The partition", the driver).
   - **Turning**: the same runs turned by `R(x, y, z) = (y, x, −z)`
     (`turned_case`; a rotation, so that `B = curl A` turns as a vector) are
     the turned runs, every value `==` and every nonzero value bit for bit.
     The sign of a zero is the one freedom: Balsara 1's identically zero
     `S_z` and `𝔈ᶻ` are `+0` in both runs where the turn predicts `−0`
     (`0 − 0 = +0` either way), so `isequal` would fail on them and nothing
     else (decided in step 7: the claim is stated with `==`).
   - **The Alfvén wave along the diagonal** (`oblique_case`, `(1, 1)/√2`
     on the periodic `[0, √2]²`, an eighth of a period at 16², 32², 64²; a
     quarter in parentheses, measured once): L1 rates `S_x`, `S_z`, `A_x`,
     `A_z` of 2.21, 2.23, 2.77, 2.31 (2.28, 2.36, 2.66, 2.51) under MC and
     1.97, 1.98, 2.05, 2.01 (2.03, 1.92, 2.06, 2.04) under WENO-Z — the whole
     2D path, both fluxes, the curl and the EMF with two transverse
     reconstructions, at second order. The totals drift at most 1.6 ulp of
     their scale per step. *(Amended in step 8, decided by Erik
     2026-10-15: with the gauge damping `ξ = 16/L` on every mesh.)* `A_x`'s
     gauge part is damped: its rate goes from 2.77 to 2.769 (MC) and from
     2.05 to 2.07 (WENO-Z, L1 3.67e−4, 8.62e−5, 2.14e−5 → 3.26e−4, 7.46e−5,
     1.84e−5); `S_i`, `A_z` and the rates of `S` move in their last two
     digits only, WENO-Z's drift from 1.43 to 1.42 ulp. Nothing else the
     suite prints moved: the 1D runs' gauge is never excited (`A_x` is
     constant where `E_x = 0`, and no 1D field reads it), and the 2D and 3D
     symmetry, translation and turning claims hold bit for bit as before;
     the 3D blast's `∇·𝔅` went from 1.2925e−16 to 1.2929e−16 and the tilted
     field loop's `𝔅ᶻ` ratio in its 14th digit.
   - **The cylindrical explosion** (GRaM-X §V B; 64² on `[−6, 6]²`, minmod as
     GRaM-X runs it, `t = 1`, 27 steps): symmetric under `x → −x` to 1.6e−18
     and under `y → −y` with the fields reversed to 2.5e−17 of each
     variable's scale (asserted ≤ 1e−12); no repair, no reset. Under MC at
     64² and 128² the reset floors the entropy of a few cells behind the
     front (`FLAG_POL_S_FLOORED`; 216 cells written as the stage limiter and
     96 as the step limiter to `t = 1` at 64², injecting `7·10⁻⁵` of energy);
     minmod writes none.
   - **The rotor** (GRaM-X §V C; 64², MC, `t = 1/10`): symmetric under the
     rotation by π with the fields reversed to 2.6e−27; the totals change by
     2.9e−16 of their scale (the precursors at the faces); every stage ideal,
     no repair, no reset.
   - **Orszag–Tang** (Beckwith & Stone §4.4; 64², MC, `t = 1/8`, 40 steps):
     every total but `𝔈`'s constant to 0.056 ulp of its scale per step
     (asserted ≤ 8); symmetric under the rotation by π about the centre to
     1.9e−13 — not to the last bit, but to the recovery's tolerance
     (EntropyEOS's `1e−12`): mirrored states that differ in the last bit stop
     at different iterates. The asymmetry is 9.3e−16 in the initial data,
     1.2e−13, 2.5e−13, 1.9e−13 and 1.6e−14 at `t` = 1/32 … 1/4, and 5.3e−15 at
     1/16 with the tolerance set to `1e−14` (asserted ≤ 1e−11; decided in
     step 7: "to round-off" means to the recovery's tolerance where the
     recovery iterates).
   - **The field loop** (Beckwith & Stone §4.2; 64 × 32, MC, `t = 1/2`, a
     fiftieth of a crossing): the totals constant to 0.21 ulp (0.25 with
     `vᶻ`) a step; with `vᶻ = 0`, `Eˣ`, `Eʸ`, `A_x`, `A_y` and so `𝔅ᶻ` stay
     exactly zero, as Beckwith & Stone found; with `vᶻ = 0.1/√6` the
     in-plane EMFs `vᶻB` give `max |𝔅ᶻ|` 4.6e−4 of the in-plane field at
     `t = 1/2` (3.4e−4 at `t = 1`).
   - **3D**: a spherical blast (the explosion with `r = |x|`, periodic
     `[−2, 2]³` at 16³, minmod, to `t = 1/4` in the suite and `1/2` once)
     runs with no repair and no reset,
     and is symmetric under its three reflections exactly (0.0). In a closed
     box it is not, at the faces (see "Boundaries", measured in step 7).
     *(Amended in step 8:)* with both walls prescribed (AMR-8's workaround,
     "Boundaries") it is, bit for bit, in the closed box, where the suite
     now runs it; so is the explosion on `[−2, 2]²` at 32².
   - **Higher order in 1D** (the Alfvén wave of rung 1a, one period, 50 …
     400 cells): L1 rates of `S_y`, `A_y`, the ideal projection of `𝔈ʸ` and
     the state's own `𝔈ʸ` of 2.09, 2.10, 2.10, 0.85 (MC), 2.67, 2.47, 2.68,
     0.97 (PPM), 2.00, 2.00, 2.00, 0.99 (WENO-Z), 2.00, 2.00, 2.00, 0.98
     (MP5); `S_y` errors at 400 cells 2.3e−4, 2.6e−5, 1.3e−5, 1.3e−5. `D`,
     `S_x`, `τ` converge at 2.8–2.9 under MC and PPM and 5.6–5.7 under WENO-Z,
     and are at rounding (1e−15) under MP5. Every stage ideal, no reset.
   *Rung 2b, measured in step 8* (`test/refinement_tests.jl`,
   `test/refinement_cases_tests.jl`; HLLE, SSP3(4,3,3), `cfl` 2/5 over `D`
   on the finest level; the matter's operators Conservative width 3
   bounded, the potentials' PointValue width 4, `ξ = 16/L` on every mesh
   — "Refinement"; Julia 1.13; *amended at the end of step 8*: since
   TreeAMR 0.2.2 the potentials `A_j` conservative along the edge (AMR-9),
   the matter's ghost fill in primitives, PointValue width 2 (AMR-10), and
   the numbers below re-measured with them, the first value in brackets):
   - **Conservation at coarse-fine faces**: every total but `𝔈`'s drifts
     0.008 [0.013] (1D) and 0.12 [0.12] (2D) ulp of its scale a step on
     static two-level meshes with the flux fixup, 1.5·10¹⁰ [1.8·10¹⁰] and
     3.3·10¹¹ [5.4·10¹¹] without; through regrids, 0.017 [0.015] (Balsara
     1, tracked, ten regrids) and 0.036 [0.037] (the rotor, two). See
     "Conservation at coarse-fine faces".
   - **Magnetic flux**: the rate through every coarse face on a coarse-fine
     face equals the fine faces' sum to 1.7e−18 (2D) and 8.7e−19 (3D) of the
     largest rate with the EMF correction, 0.85 and 0.48 of it without. As a
     state, the flux through the plane drifts 5.8e−4 of its scale across a
     level — the potentials' operators, AMR-9. *(Amended in step 8, AMR-9
     landed:)* with `A_j` conservative along the edge the state holds too —
     the plane's flux to 2.2e−16 through the run and 1.5e−16 through a
     regrid, and in 3D every coarse face on a coarse-fine face its fine
     faces' sum to 1.4e−16 of the largest flux after steps and after a
     regrid (1.5e−3 with PointValue), with or without the EMF correction (the
     transfer's property); at an outer face only to 1.2e−5 (each level's
     hook, AMR-11). The rate's mismatch is now 8.7e−19 of 5.4e−3 (2D) and
     5.5e−3 (3D), and 0.84 and 0.31 of it without.
   - **`∇·𝔅`**: at most 2.0e−16 of `max |𝔅|` after every step on every
     level and right after every regrid (the adaptive rotor), as in rung 2a.
   - **The field loop across a coarse-fine face** (`x ≥ 0` refined, the face
     through the loop's centre): its magnetic energy and its `A_z` error lie
     between the uniform coarse and fine controls — at `t = 1/2`, 0.801
     [0.803] of its initial energy against 0.623 and 0.844, L1(`A_z`) 1.6e−6
     [1.56e−6] against 3.0e−6 and 8.7e−7; at `t = 4`
     (`bin/field_loop_refined.jl`, the loop a third of its radius across the
     face), 0.584 [0.589] against 0.206 and 0.619, and 4.2e−6 [4.16e−6]
     against 9.3e−6 and 3.0e−6. The refined half keeps the fine
     mesh's energy; the coarse half loses the coarse mesh's. (Adaptive, one
     level where Löhner fires, the loop with its buffer and 2:1 balance
     covers every root of so small a box: it is the uniform-fine run, to
     every digit printed.)
   - **The interface order and the gauge damping**: the tables in
     "Refinement" (width 4: L∞ 2.04 [1.97] in 1D, 2.02 [1.85] in 2D, where
     width 2 gives 0.91 [0.88] and 0.33 [0.08]; `ξ = 16/L` clears 97%
     [99.95%] of a gauge pulse's spurious field within two light
     crossings).
   - **The tracked shock tube**: Balsara 1 on its box made periodic, 100
     coarse cells and two levels more where Löhner fires (`refine_tol =
     0.08`, `coarsen_tol = 0.02`, buffer 6), regridded every 1/80: three
     adaptation passes, ten regrids, 18 to 40 blocks, 416 steps, conserved
     to 0.017 [0.015] ulp a step, nothing repaired and no reset.
   - **The atmosphere across a coarse-fine face**: the bounded ghost fill
     repairs no ghost and injects what the uniform mesh does; unbounded,
     `D < 0` in fine ghosts and 250 times the rest mass (the table in
     "Refinement"). *(Amended in step 8, AMR-10 landed:)* the ghost fill
     now prolongates the primitives, width 2 (proposed): no ghost
     inadmissible or below the atmosphere, hydrodynamically and magnetized
     (where conserved ghosts had 33 repaired recoveries), but 5.6e−9 of rest
     mass injected hydrodynamically against the conserved fill's 1.5e−11;
     at width 4, 5.4e−4 of energy, and a blow-up magnetized (the tables in
     "Refinement"). *(Amended 2026-10-23:)* the fill is now the
     parent-centred one (decided by Erik): −1.6e−11 of rest mass, as the
     conserved fill's.
   - **What the end of step 8 moved elsewhere** (measured, every line the
     suite prints, before and after, Julia 1.13): the symmetric walls and
     the relative hook moved nothing a uniform-mesh case prints; the
     primitive ghost fill, whose owned `P` goes through `P_from_W`, moved
     the uniform runs in their last digits — the waves' L1 errors in the
     12th to 15th, every rate to the digits recorded unchanged, the
     drifts (Alfvén 0.0059 → 0.0046 ulp a step, PPM 0.011 → 0.0056), the
     shock tubes' L1 errors in the 11th to 13th digit (Balsara 3 under PPM
     in the 10th) and their boundary budgets within 3·10⁻¹⁴, the uniform
     flows through the faces 4.4·10⁻¹³ … 1.5·10⁻¹² of a variable's scale,
     the current sheet's distance 9.559329e−4 → 9.559339e−4, Orszag–Tang's
     asymmetry 1.9e−13 → 3.8e−13 (the recovery's tolerance; asserted ≤
     1e−11), the rotor's drift 2.9e−16 → 4.0e−16; the 1D/2D/3D bit identity,
     the turned runs and every mirror symmetry on a uniform mesh hold bit
     for bit. Chaotic at rounding level, and so moved further: the
     magnetized atmosphere tube (rung 1b, amended there), the near-vacuum
     rarefaction under ARS (2.1e−10 → 2.7e−10, resets 152/114 → 161/118)
     and the ideal magnetized atmosphere under ARS (2.6·10⁴ → 1.6·10⁴ of
     rest mass injected, `W` 49 → 100); SSP3(4,3,3)'s rows of both are the
     same to the digits recorded. The 3200-cell references in `test/data/`
     are not regenerated (proposed in step 8): the scheme changed for them by
     rounding only, against reference errors of 10⁻³.
3. **Analytic curved backgrounds**: the Wald solution (`Vacuum`,
   Kerr-Schild) and the Michel monopole or magnetized Bondi flow
   (`IdealConductor`).

   *Rung 3, as built and measured in step 9* (`src/cases/kerr_schild.jl`,
   `src/interior.jl`, `test/curved_tests.jl`, `bin/curved_stationarity.jl`;
   proposed in step 9 unless marked):
   - **The background**: a Schwarzschild hole, `M = 1`, in Kerr-Schild
     coordinates (Cook 2000, Living Rev. Relativ. 3, 5, §3.3.1),
     `Analytic(KerrSchild(M))`. With `f = 2M/r` and `l = x/r`: `γ_ij = δ_ij +
     f l_i l_j`, `βⁱ = f lⁱ/(1 + f)`, `α = 1/√(1 + f)`, `α√γ = 1`; the
     coordinates cross the horizon, and the coordinate speeds of light are
     at most 1, so the CFL step is set by `h` everywhere, the hole's
     neighbourhood included.
   - **The matter, both cases**: Michel's (1972, Ap&SS 15, 153) transonic
     accretion flow, `Γ = 4/3`, sonic point `r_c = 8M` with `ρ_c = 1`
     (Hawley, Smarr & Wilson 1984, ApJ 277, 296, §III; Rezzolla & Zanotti
     2013, §11.8): Bernoulli's `h² (1 − 2M/r + (uʳ)²) = C₂` and the mass flux
     `r² ρ uʳ = −C₁` solved per point on the transonic branch (bisection for
     the sonic density, Newton safeguarded by bisection), to `1.4·10⁻¹³` of
     `C₂`; `u·u = −1` in the 4-metric to `1.0·10⁻¹⁴`. Kerr-Schild's covariant
     `u_r` is written in the form regular across the horizon, `u_r = ((uʳ)²(1 +
     f) − f²)/(uʳ − f E)`, `E = −u_t`, where the textbook `(uʳ + fE)/(1 − f)`
     is `0/0` at `r = 2M`. On `shock_gas(4/3)`, at its window's `s = 12.2`.
   - **The Wald solution** (`wald_case`, `Vacuum`; Wald 1974, Phys. Rev. D 10,
     1680): `𝒜 = (B₀/2) ψ`, `ψ = ∂_φ`, at `a = 0`: `A = (B₀/2)(−y, x, 0)`,
     `Φ = 0` (`ψ` is orthogonal to `l`), so `𝔅 = (0, 0, B₀)` exactly — the
     curl of a linear `A` — and the stationarity of `A` gives `E = −(β/α) ×
     B`. `B₀ = 1`. The field and the flow do not touch: `j = 0` under `Vacuum`
     with `q = 0`, so the field's stress-energy is conserved apart.
   - **The magnetized Bondi flow** (`bondi_case`, `IdealConductor`; Gammie,
     McKinney & Tóth 2003, ApJ 589, 444, §4.2; Porth et al. 2017, ComAC 4,
     1, §4.3): Michel's flow in the radial monopole `𝔅 = q x/r³`, which is a
     vacuum solution on Schwarzschild (the magnetic Reissner–Nordström
     field as a test field) and lies along the flow, so `E = −v × B = 0` and
     `j = 0`: the flow is Michel's exactly. `q = 9`: `b²/ρ = 0.98` at the
     horizon, 6.1 at `r = M`, 0.030 at the box's far corner.
   - **Magnetized Bondi, not the Michel monopole** (the plan's choice, made
     here; accepted by Erik 2026-11-03). Both need a monopole field. The Michel monopole (Michel 1973) is
     a force-free solution with negligible matter: at `a = 0` it is static
     and trivial, at `a ≠ 0` Kerr has only the Blandford–Znajek perturbative
     one, and in this package's model it would be electrovacuum or an ideal
     plasma at a magnetization above the atmosphere's, neither of them its
     physics. The magnetized Bondi flow is exact for the whole system, matter
     included, is the standard GRMHD test of the literature above, and
     exercises the ideal stage solve, the curved sources and the radial
     inflow through the horizon at once.
   - **`a = 0`** (accepted by Erik 2026-11-03): a stationary fluid is needed in both cases, and Michel's is
     Schwarzschild's; Kerr has no closed-form accretion flow. A spinning Wald
     case could carry a passive fluid at the floor and be measured on `𝔈`
     and `𝔅` alone (under `Vacuum` nothing couples) — a follow-up, for Erik.
     *(Resolved 2026-11-03: Erik wants it, as step 9's round 4.)*
   - **The octant** `[0, L]³`, `L = 4M` (accepted by Erik 2026-11-03), the hole at its corner, Dirichlet
     faces from each case's own data on all six faces
     (`DirichletBoundary`). The monopole has no vector potential regular on
     a whole sphere: `A = q (n̂ × x)/(r (r + n̂·x))` is singular on the Dirac
     string `x = −s n̂`, which carries the return flux `4πq`. With
     `n̂ = (1, 1, 1)/√3` the string leaves the octant through its corner and
     reaches only the ghosts outside all three low faces at once, which
     nothing owned reads. Both solutions are smooth across the octant's faces
     but at `r = 0`, so a Dirichlet face is consistent there; and the
     octant is an eighth of the box around the hole. The same box serves
     the Wald case, for one geometry.
   - **The gauge**: `Φ = 0` for both, so the damped Lorenz gauge's
     stationary solution is the exact one and `evolve!`'s `ξ = 16/L` is
     kept. (At `a ≠ 0` Wald's `Φ ≠ 0`: the damping would move `A` and `Φ̃`
     by a gauge, and the interior's pasted `A` would no longer be the
     evolved one's gauge — `ξ = 0` for such a case.)
   - **No 1D or 2D reduction**: Kerr-Schild in Cartesian coordinates is
     not translation-invariant along any axis, so every run is 3D; the
     octant is the reduction (an eighth), and the suite runs `h = M/4` and
     `M/8` (16³ and 32³ cells) to `t = M/2`.

   **The truncation error** (`test/curved_tests.jl`): the right-hand side of
   each exact discrete state outside `r = 2M`, at `h = M/4` and `M/8`,
   converges at 1.92–1.93 in `D`, 1.66–1.87 in `S`, 1.78–1.88 in `τ`,
   1.89–1.91 in `𝔈` (Wald), 1.85–2.10 in `A`; outside `2.5M` every one at
   1.9–2.0 — the scheme is consistent in curved space at every centering:
   a wrong sign in a shift term or a metric read at the wrong point leaves
   an `O(1)` remainder.

   **Stationarity** (PLAN.md: the L1 drift after `t = 50M` converges at
   `≥ 1.8`, **(predicted)**; *(measured in step 9: fails — Wald saturates,
   the magnetized Bondi flow converges at first order; the cause is located
   below, and the prediction holds without it at `h = M/4 → M/8`)*.) The L1
   drift outside `r = 2M` after `t = 50M`, both limiters, `r_in = M`, and the
   rates `M/4 → M/8` and `M/8 → M/16` (`bin/curved_stationarity.jl`; jobs
   588101/588102 on cn079 and cn080, 588111 and 588114 on cn079 and cn084,
   EPYC 7532, 588112 and 588113 on cn108 and cn083, EPYC 7543 and 7532, amdq,
   about 2 h at 64 threads):

   | | `h = M/4` | `M/8` | `M/16` | rates |
   |---|---|---|---|---|
   | Wald `D` | 1.008e−2 | 5.247e−3 | 4.952e−3 | 0.94, 0.08 |
   | Wald `S_x`, `S_z` | 1.507e−2, 5.500e−3 | 4.913e−3, 2.853e−3 | 4.647e−3, 2.544e−3 | 1.62, 0.08; 0.95, 0.17 |
   | Wald `τ` | 1.545e−2 | 4.143e−3 | 2.678e−3 | 1.90, 0.63 |
   | Wald `D Yₑ` | 5.039e−3 | 2.624e−3 | 2.476e−3 | 0.94, 0.08 |
   | Wald `𝔈ˣ`, `𝔈ᶻ` | 1.041e−2, 2.263e−3 | 2.253e−3, 1.459e−3 | 1.586e−3, 1.365e−3 | 2.21, 0.51; 0.63, 0.10 |
   | Wald `𝔅ˣ`, `𝔅ᶻ` (all faces) | 1.459e−2, 2.789e−2 | 2.529e−3, 5.777e−3 | 7.098e−4, 1.701e−3 | 2.53, 1.83; 2.27, 1.76 |
   | Wald `√γ q` | 1.595e−2 | 6.524e−3 | 7.038e−3 | 1.29, −0.11 |
   | Bondi `D` | 6.845e−3 | 3.284e−3 | 1.412e−3 | 1.06, 1.22 |
   | Bondi `S_x` | 2.885e−3 | 1.360e−3 | 6.301e−4 | 1.09, 1.11 |
   | Bondi `τ` | 4.413e−3 | 2.509e−3 | 9.895e−4 | 0.81, 1.34 |
   | Bondi `𝔈ˣ` | 4.660e−4 | 2.137e−4 | 9.423e−5 | 1.12, 1.18 |
   | Bondi `𝔅ˣ` (all faces) | 2.206e−3 | 1.090e−3 | 4.814e−4 | 1.02, 1.18 |
   | Bondi `D s` | 0.590 | 0.174 | 5.43e−2 | 1.76, 1.68 |
   | Bondi `√γ q` | 8.73e−4 | 5.85e−4 | 4.18e−4 | 0.58, 0.48 |

   No reset and no injected mass at `M/4` and `M/8`; the magnetized Bondi
   flow at `M/16` resets 126000/42000 cells (stage/step), all deep inside
   (below), and its `τ` over the whole octant is `1.29·10³` against `9.9·10⁻⁴`
   outside `2M`.

   **Where the error lives** *(measured in step 9, round 2;
   `bin/curved_localise.jl`: the drift and the charge every `5M`, binned at
   the end by radius, by the distance to the octant's faces, and over `r ≥
   2M` without `k` cells next to every face; jobs 588185–588224, Symmetry,
   EPYC 7532 and 7543, the node per job in the logs)*:

   - **Michel's flow alone converges at second order.** `wald_case(B₀ = 0)`
     (no field) and `bondi_case(q = 0)` are the same run bit for bit; the
     drift settles by `t = 20M` and stays: at `t = 50M`, outside `2M`, `D`
     2.862e−3 → 5.423e−4 (**2.40**), `τ` 1.181e−3 → 2.148e−4 (**2.46**), `S_x`
     4.470e−4 → 1.060e−4 (**2.08**) at `h = M/4 → M/8`. Without the limiter
     (`:none`) 2.407e−3 → 4.730e−4 (2.35). So neither MC at the coordinate
     planes, nor the Dirichlet planes through the hole, nor the paste's
     surface, nor the hydrodynamic sources limit the order: the matter's
     error with the field is the field's.
   - **Wald's drift is a growing mode, not a saturation.** Its time series
     (outside `2M`, `D`, `τ`, `𝔈ˣ`):

     | `t/M` | 5 | 10 | 20 | 30 | 40 | 45 | 50 |
     |---|---|---|---|---|---|---|---|
     | `M/4` `D` | 2.55e−3 | 2.52e−3 | 2.79e−3 | 3.38e−3 | 4.80e−3 | 6.53e−3 | 1.01e−2 |
     | `M/4` `𝔈ˣ` | 5.60e−4 | 6.18e−4 | 8.25e−4 | 1.36e−3 | 3.39e−3 | 5.94e−3 | 1.04e−2 |
     | `M/8` `D` | 4.84e−4 | 5.49e−4 | 1.06e−3 | 2.04e−3 | 3.42e−3 | 4.17e−3 | 5.25e−3 |
     | `M/8` `𝔈ˣ` | 1.59e−4 | 1.88e−4 | 3.50e−4 | 6.07e−4 | 1.03e−3 | 1.49e−3 | 2.25e−3 |

     At `M/16` (job 588214, read to `20M`): `D` 1.06e−4, 1.50e−4, 2.54e−4,
     4.14e−4 at 5, 10, 15, 20M. At `t = 5M` it converges (`D` 2.4 and 2.2,
     `𝔈ˣ` 1.8 and 1.8); then `𝔈ˣ` grows
     exponentially, about `e^{0.11 t/M}` at `M/4` and `e^{0.08 t/M}` at `M/8`,
     and at `M/4` on to `t = 100M` without saturating (`𝔈ˣ` 3.2e−2 at 60M,
     0.25 at 80M, 1.04 at 100M; `τ` 7.9). The `t = 50M` "saturation" of the
     table is this mode's amplitude, reached from a smaller start at a
     slower rate on the finer mesh. It fills the volume (the radial layers
     fall off smoothly from the hole to the outer faces; leaving 1–4 cells
     next to every face out raises the L1 by at most 12%: the mode is not a
     layer at a face), and grows in the box
     around the hole too (`[−3, 3]³`, `τ` 2.3e−3 at 5M to 6.9e−2 at 50M), so it
     is neither the octant's planes through the hole nor a layer at a face.
   - **What it is not.** Linear in the field and blind to the fluid: at `B₀
     = 0.1` the relative error is `B₀ = 1`'s to three digits, and with a fluid
     a thousand times heavier (`ρ_c = 10³`) the same, so neither the charge's
     force on the fluid nor the fluid's current is needed (a fluid a thousand
     times lighter grows somewhat faster). Not the charge's transport: with
     Ampère's explicit source `−√γ q (αv − β)` replaced by `+√γ q β` alone or
     by zero (a diagnostic in `bin/curved_localise.jl`, not a scheme), it
     grows as fast or faster. Not the interior (no interior grows alike), not
     the limiter (`:none` grows faster), not the Riemann solver (LLF grows
     alike), not the time step (`cfl = 1/5` gives the same run to five
     digits). **It depends on the gauge damping**: at `ξ = 0` it grows faster
     (`𝔈ᶻ` 5.4e−2 at 50M), at `ξ = 16` and 64 slower (`𝔈ˣ` 4.5e−3, 4.6e−3)
     — against `ξ = 16/L = 4`, `𝔈ˣ` 1.04e−2.
   - **The cause: the potentials' relative Dirichlet hook** (AMR-7, chosen in
     step 6 for potentials that grow in time). It fills a ghost or wall point
     `x_g` with `A_case(x_g) + [A(x_r) − A_case(x_r)]`, copying the evolved
     potential's offset from its reference point `x_r`. The evolved `A`
     carries a gauge drift `∇χ` of truncation order — the generalized
     Lorenz gauge's, which `ξ` damps in `Φ̃` but which leaves a static `χ`
     undamped. Copied unchanged along the normal, the offset is no longer a
     gradient across the face: the curl of the filled `A` puts `∂_n∂_⊥χ`
     into the ghost and wall faces' `𝔅` (an exact gradient's curl vanishes
     only if every point carries its own). That flux enters the boundary
     cells' EMFs and Ampère fluxes, the `E` it drives feeds `∂ₜΦ̃ ∋ −∂ⱼ(α√γ
     Aʲ)` and so `χ` again: a loop, closed through the gauge, which `ξ`
     slows. **Measured**: with the potentials' hook absolute — the case's `A`
     and `Φ̃` themselves in every ghost and on every wall, a diagnostic for
     these time-independent potentials — Wald's drift is stationary from `t
     = 10M` (outside `2M`, `D` 2.981e−3, `𝔈ˣ` 4.58e−4 at 10M and at 50M), in
     the octant and in the box around the hole, and so is the magnetized
     Bondi flow's.
   - **With the absolute hook the prediction holds** (`t = 50M`, outside
     `2M`, `M/4 → M/8`): Wald `D` 2.981e−3 → 5.881e−4 (**2.34**), `τ` 1.255e−3
     → 2.597e−4 (**2.27**), `S_x` 8.50e−4 → 2.19e−4 (1.96), `𝔈ˣ` 4.578e−4 →
     1.154e−4 (**1.99**), `𝔈ᶻ` 1.95, `√γ q` 1.65; Bondi `D` 6.307e−3 →
     1.650e−3 (**1.93**), `τ` 3.436e−3 → 8.158e−4 (**2.07**), `S_x` 2.471e−3 →
     7.677e−4 (1.69), `𝔈ˣ` 3.487e−4 → 1.065e−4 (1.71), `√γ q` 1.49.
     At `M/16` (jobs 588223, 588224, amdq, 64 threads) the runs were at
     `t = 10M` when the Symmetry tunnel dropped: Wald `D` 6.035e−4 → 1.328e−4
     (**2.18**), `𝔈ˣ` 1.179e−4 → 2.937e−5 (2.01), Bondi `D` 1.650e−3 → 4.211e−4
     (1.97), `𝔈ˣ` 1.068e−4 → 2.995e−5 (1.83), against `M/8` at `10M`; their
     `t = 50M` rows are in `/mnt/beegfs/eschnetter/claude/
     treegrrmhd-step9-5882{23,24}-part1.log`, unread.
   - **Bondi at first order** was the same mode, slower: at `M/4` its drift is
     flat from `5M`, at `M/8` it grows linearly from `t = 20M` (`D` 1.84e−3 →
     3.28e−3), so the finer mesh ends further from its own converged value;
     with the absolute hook both are flat.
   - **The charge.** Wald's non-converging `√γ q` is the growing mode's
     (`q` follows `𝔈`'s growth; with the absolute hook it is stationary and
     converges at 1.65). The matter's saturation in the Wald case is the
     mode's too, through the total momentum and energy, which carry the
     field's: with the heavy fluid `D`'s drift stays Michel's own.

   **Decided by Erik 2026-11-03: option 1** — the case's own potential in the
   ghosts where it does not depend on `t`. *As built* (`static_potential` on
   `MHDCase`, `absolute` on `DirichletBoundary`, "Boundaries"): `wald_case` and
   `bondi_case` declare their potentials static, and their potentials'
   hooks are `CellBoundary(AllVariables(…))` of the case's `A_j` and `Φ̃`;
   every other case keeps the relative hook. Options 2 (a relative hook that
   carries the offset's gradient, upstream) and 3 (a gauge that damps a
   static `χ`) are open questions for step 10's outer boundary and for
   dynamic cases without an exact `A`. The process-local diagnostic of round
   2 (`abshook`) is gone; `bin/curved_localise.jl`'s `relhook=true` is the
   negative control. The M/16 runs of round 2 with the diagnostic hook
   (jobs 588223 on cn085, EPYC 7532, and 588224 on cn112, EPYC 8534P),
   outside `2M` at `t = 50M`: Wald `D` 1.2609e−4, `τ` 5.9239e−5, `𝔈ˣ`
   2.8671e−5, `𝔈ᶻ` 7.7540e−6, `S_x` 5.6127e−5, `√γ q` 8.5733e−5, flat from `15M`
   to `50M`; Bondi `D` 4.1891e−4, `τ` 1.9593e−4, `𝔈ˣ` 2.9913e−5, `S_x`
   2.1725e−4, `√γ q` 1.1346e−4. Rates `M/4 → M/8` / `M/8 → M/16`:

   | | `D` | `τ` | `S_x` | `𝔈ˣ` | `𝔈ᶻ` | `√γ q` |
   |---|---|---|---|---|---|---|
   | Wald | 2.34 / 2.22 | 2.27 / 2.13 | 1.96 / 1.96 | 1.99 / 2.01 | 1.95 / 1.94 | 1.65 / 1.68 |
   | Bondi | 1.93 / 1.98 | 2.07 / 2.06 | 1.69 / 1.82 | 1.71 / 1.83 | 1.71 / 1.83 | 1.49 / 1.63 |

   **Re-measured with option 1 as built** *(step 9, round 3; jobs 588498–588503,
   Symmetry, the node per job in the logs)*: at `M/4` and `M/8` the new default
   reproduces the round-2 diagnostic to every printed digit (Wald `D`
   2.9810e−3, 5.8805e−4; Bondi `D` 6.3065e−3, 1.6497e−3 outside `2M` at `t =
   50M`), the same arithmetic. At `M/16` (jobs 588502 on cn083, EPYC 7532, and
   588503 on cn107, EPYC 7543, amdq, 64 threads) the same: Wald `D` 1.2609e−4,
   `τ` 5.9239e−5, `𝔈ˣ` 2.8671e−5, `𝔈ᶻ` 7.7540e−6, `S_x` 5.6127e−5, `√γ q`
   8.5733e−5; Bondi `D` 4.1891e−4, `𝔈ˣ` 2.9913e−5 — the round-2 rows above, and
   so the rates of the table. No reset, no injected mass, the charge's total
   rounding (Wald `−3.4·10⁻¹¹` over the octant). Both drifts are flat in time: Wald's
   `D` within 2% from `10M` to `50M` and to `100M` at `M/4` (2.9810e−3 at 50M
   and at 100M), Bondi's to five digits. **Verdict: the prediction holds**
   for `D` and `τ` (1.93–2.34 and 2.06–2.27 between the three resolutions)
   and Wald's `𝔈`, `S_x` (1.94–2.01); **it fails** for Bondi's `S_x` and
   `𝔈` at `M/4 → M/8` (1.69, 1.71 — measured, not loosened) and for the
   charge (1.49–1.68); `M/8 → M/16` Bondi's are 1.82–1.83. Above 1.8 from the
   second pair on, so the coarsest mesh is not yet asymptotic for the
   monopole's transverse components. *(Decided by Erik 2026-11-03: the
   prediction is judged on the finest pair, `M/8 → M/16`, where it holds for
   every variable; the `M/4 → M/8` rates 1.69 and 1.71 stand as measured — at
   `M/4` only 8 cells lie between the hole's centre and the horizon.)*


   **In the suite** (`t = M/2`, `h = M/4` and `M/8`, laptop and Symmetry
   alike): Wald `D` 2.48, `S` 1.86–1.98, `τ` 2.46, `𝔈` 2.10–2.13, `𝔅`
   1.76–1.82; magnetized Bondi `D` 1.73, `S` 1.89, `τ` 2.17, `𝔈` 1.89, `𝔅`
   2.42; asserted `≥ 1.6`. No reset, no injected mass, no stage fallback
   (Wald every stage `OHM_SIGMA_ZERO`, Bondi every stage `OHM_IDEAL`).

   **The charge** `√γ q = ∂ᵢ𝔈ⁱ`, which the exact fields do not have, outside
   `2M` after `t = M/2`: L1 `5.9·10⁻⁴`, `1.8·10⁻⁴` (Wald) and `2.0·10⁻⁴`,
   `6.1·10⁻⁵` (Bondi) at `h = M/4`, `M/8` — converging at 1.7; the largest
   `|√γ q|` `9.0·10⁻³`, `4.0·10⁻³` and `1.1·10⁻²`, `7.0·10⁻³`, next to the
   interior. After `t = 50M` at `h = M/4`: L1 `1.6·10⁻²` (Wald), `8.7·10⁻⁴`
   (Bondi), largest `0.13`, `2.8·10⁻²`; its volume integral outside `r_in` is
   rounding (`10⁻¹⁴`–`10⁻¹⁶`): the discrete Gauss law moves no net charge.

   **∇·𝔅**: at most `2.7·10⁻¹⁶` of `max |𝔅|` (`h ∇·𝔅`), after every run:
   `𝔅` is the flat curl of `A` whatever the metric, densitized, and the
   background `𝔅̄` is not used.

   **Curved-space bit identity** (`test/curved_tests.jl`): with the step
   cache (fused, the default) and the interior, 16 steps of `Δt = 1/64` in
   one integration, in two, and as a restart from the saved state with a
   rebuilt problem are the same bit for bit, for both cases; the suite prints
   their checksums, which the one- and four-thread runs print alike. The
   `· inv(√γ)` keys of "The step cache", written for curved space and first
   exercised here, hold.
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
- *(Closed in step 8.)* Conservative prolongation width for the matter at
  stellar surfaces: width 3, bounded (TreeAMR's limited operator, AMR-4);
  the TOV surface's check is step 10's.
- *(Closed in step 8.)* PointValue width for `A` and `Φ̃`: 4; the gauge
  damping: `ξ = 16/L` on every mesh, `L` the box's longest side (amended,
  decided by Erik 2026-10-15; see "Refinement").
- *(Added in step 8; closed by Erik 2026-10-15.)* Primitives or the
  conserved state in the matter's ghost fill: the primitives, with a
  bounded Lagrange prolongation (AMR-10), the conserved state in the regrid
  (see "Refinement"). *(Amended 2026-10-23, decided by Erik:)* the
  primitives with the bounded conservative prolongation centred on the
  parent; the three working fills are recorded in "Refinement", to revisit
  when the physics gets harder.
- TreeGH's CFL factor under the SSP3 explicit tableau.
- The metric in the outermost ghost cell (see
  [Metric at cells, faces and edges](#metric-at-cells-faces-and-edges)).
- Whether `D s` earns its keep; the threshold that switches to it.
  *(Measured in step 9:)* on the stationary Kerr-Schild flows it drifts
  further from `D s_exact` than `D` does (see "Validation ladder" 3).
- *(Added in step 9.)* The interior's form: the hard paste in both
  limiters at `r_in = M` (proposed), against TreeGH's smooth damping layer
  inside the right-hand side ("The interior"). *(Decided by Erik 2026-11-03:
  the paste, for the analytic rung-3 cases.)*
- *(Added in step 9, round 3, from Erik's decision on the interior.)* For a
  hole that forms in a collapse, with no analytic solution inside: a
  field-only interior — paste or damp `𝔈` and `A` inside, and leave the
  matter to the ordinary reset, which round 3 found sufficient for the
  matter. Not measured; for the step that forms a hole.
- *(Added in step 9.)* The analytic metric pointwise (44% of a step on a
  16³ blast) or cached per stage, or its repeated vertex and edge
  evaluations deduplicated ("Metric providers"). *(Decided by Erik
  2026-11-03: pointwise, no cache, no deduplication; caching is step 10's.)*
- *(Added in step 9.)* A spinning Wald case, with a passive fluid at the
  floor (Kerr has no closed-form accretion flow; "Validation ladder" 3).
- *(Added in step 9, round 3.)* The potentials' outer boundary where no
  exact `A` exists (step 10's star, dynamic cases): a gradient-extrapolating
  relative hook (option 2, upstream) or a gauge that damps a static `χ`
  (option 3); see "Boundaries", the design rule. *(Decided by Erik
  2026-11-03: in step 10. The Cowling TOV star is static, so an exterior
  field from an exact static potential takes `static_potential = true`
  (option 1) as is; options 2 and 3 matter for dynamic cases only.)*
- Whether the atmosphere reset needs its stage-limiter role. Run rung 4's
  TOV star, a star in a vacuum, twice: with the reset as both limiters,
  and as the step limiter alone. TreeHydro named this case as the one
  where a per-stage reset could matter, and no code in IMEXRungeKutta's
  survey measured it (amended 2026-09-25).
- The reconstruction the stars run with (added in step 7): MC, the
  plan's and GRaM-X's choice, or PPM, which alone of the four resolved
  Balsara 3's dense shell and held every Balsara plateau but test 2's `R5`
  (2.8%) within 2% (see "Validation ladder" 1b); WENO-Z smeared the shell
  worst, and MP5 costs twice the steps. Decided on the TOV star (rung 4).
- *(Closed in step 6.)* SSP3(4,3,3) or ARS(4,4,3) as the default tableau:
  SSP3(4,3,3), whose explicit part is SSP; ARS(4,4,3) undershoots into a
  near vacuum, fires the reset on Balsara 3 and 4, and floods a magnetized
  atmosphere (see "The partition").

## Known risks

- No MPI yet: single node, although TreeAMR now has its M7 (amended in
  step 0).
- Unlimited conservative prolongation at a stellar surface floors cells.
  *(Step 8:)* the matter's operator is bounded since; the atmosphere tube
  across a coarse-fine face repairs no ghost with it, where the unbounded
  one made `D < 0` (see "Refinement").
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
- *(Measured in step 7:)* MC floors the entropy of a few cells behind
  the cylindrical explosion's front (the reset writes them; minmod, GRaM-X's
  choice there, writes none), and MP5 is unstable above `cfl = 1/5`. A
  strong blast into a light medium is where the reconstruction's
  overshoots reach the recovery.
