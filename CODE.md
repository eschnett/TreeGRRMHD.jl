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

*Status: steps 0 to 3 and 5 to 9 of `PLAN.md` are done, as are steps 4a–4c
in IMEXRungeKutta (what each delivered is recorded in [HISTORY.md, "The
steps of the plan"](HISTORY.md#the-steps-of-the-plan)). Step 10, the TOV
star in the Cowling approximation on the analytic TOV metric, with no
TreeGH ([The coupling contract](#the-coupling-contract)), is built and
measured (rung 4), its decisions proposed for review. Step 11, the coupled
run, waits on GH-2 to GH-4 ([TreeGeneralizedHarmonic](#treegeneralizedharmonic)).
The numbers below about the scheme are measured where they say so and
unmeasured otherwise.*

**Inconsistent:** the status says step 10's decisions are proposed for
review, while `CLAUDE.md`'s former "Current state" said that on 2026-11-08 Erik
decided step 10's light cone, width 3 bounded, the reset's two roles, PPM
for stars and the per-centering binding of the contract (now in
[HISTORY.md, "Open questions"](HISTORY.md#open-questions) and ["The coupling
contract"](HISTORY.md#the-coupling-contract)).

Markers: **(proposed)** is a decision this document makes and still wants
confirmed; **(open)** is deferred to a measurement or to a later review and
is listed under [Open questions](#open-questions). When the implementation
shows this document wrong or incomplete, fix it to describe the current
state, and record in [HISTORY.md](HISTORY.md) what changed, when and why,
rather than diverging silently.

**Contents**

- [Goals](#goals)
- [Scope and non-goals](#scope-and-non-goals)
- [Upstream prerequisites](#upstream-prerequisites)
  - [TreeAMR](#treeamr)
  - [TreeGeneralizedHarmonic](#treegeneralizedharmonic)
  - [EntropyEOS](#entropyeos)
  - [SpacetimeMetrics](#spacetimemetrics)
- [The equations](#the-equations)
  - [Conventions](#conventions)
  - [Matter: total energy and momentum](#matter-total-energy-and-momentum)
  - [Maxwell](#maxwell)
  - [Ohm's law and the conductivity](#ohms-law-and-the-conductivity)
  - [Stress-energy for the spacetime](#stress-energy-for-the-spacetime)
- [Thermodynamics](#thermodynamics)
- [Primitive recovery](#primitive-recovery)
  - [Where the fields live](#where-the-fields-live)
  - [Field sets](#field-sets)
  - [Reconstruction](#reconstruction)
  - [Riemann solver](#riemann-solver)
  - [The electromotive force for `A`](#the-electromotive-force-for-a)
  - [Sources](#sources)
  - [Floors, the atmosphere and the magnetosphere](#floors-the-atmosphere-and-the-magnetosphere)
- [Time integration](#time-integration)
  - [The partition](#the-partition)
  - [The implicit stage solve](#the-implicit-stage-solve)
  - [The IMEX package](#the-imex-package)
- [Spacetime coupling](#spacetime-coupling)
  - [The coupling contract](#the-coupling-contract)
  - [Metric providers](#metric-providers)
  - [Metric at cells, faces and edges](#metric-at-cells-faces-and-edges)
  - [Matter to the vertices](#matter-to-the-vertices)
  - [Gauge for the TOV milestones](#gauge-for-the-tov-milestones)
- [The right-hand side](#the-right-hand-side)
- [Conservation at coarse-fine faces](#conservation-at-coarse-fine-faces)
- [Boundaries](#boundaries)
- [The interior](#the-interior)
- [Refinement](#refinement)
- [Julia version](#julia-version)
- [Precision and devices](#precision-and-devices)
- [Sharp edges](#sharp-edges)
- [Tests](#tests)
- [Code layout](#code-layout)
- [Validation ladder](#validation-ladder)
- [Possible extensions](#possible-extensions)
- [Open questions](#open-questions)
- [Known risks](#known-risks)

## Goals

- **Resistive GRMHD with a magnetosphere.** E and B are both
  evolved; the current comes from an Ohm's law with a conductivity σ that
  is large inside stars — where the scheme must recover ideal MHD, and the
  resistive term is stiff — and goes to zero outside, where the fields obey
  the electrovacuum Maxwell equations. A neutron star's dipole magnetosphere
  is then not an atmosphere artefact but a solution of the equations.
- **Couple to a dynamical spacetime** from TreeGH, on the same forest, in
  one time integrator, with no subcycling.
- **A magnetized TOV star is the first science target**, after
  the standard flat-space and fixed-background tests: a Γ = 2 polytrope in
  the Cowling approximation, then in a dynamical spacetime, then
  magnetized with a vacuum dipole outside, then with a tabulated EOS.
- **∇·B = 0 to round-off, on every level and across every regrid**
  (by the vector potential — see
  [Where the fields live](#where-the-fields-live)), and **rest mass
  conserved to round-off** across coarse-fine faces.
- **Run on the caller's backend** (CPU threads, CUDA, Metal) through
  KernelAbstractions, in `Float64` for physics; `Float32` only as a device
  smoke test, since EntropyEOS is validated in `Float64` only.

## Scope and non-goals

- **No neutrinos.** No leakage, no M1, no source-term hooks for
  them yet. `Yₑ` is advected passively when the EOS is tabulated.
- **No mesh machinery.** Anything about trees, ghosts, operators,
  reductions or interpolation between levels belongs upstream in TreeAMR.
  Interpolation *within a block* between centerings is application code
  and lives here, in both directions: the matter's `(ρ, S_i, S_ij)` from
  cells to TreeGH's vertices, and the metric's `(α, β^i, γ_ij, K_ij)` from
  TreeGH's vertices to this package's cells, faces and edges. This package
  is the only one with mixed centerings ([The coupling
  contract](#the-coupling-contract)).
- **No spacetime evolution.** The Einstein equations are TreeGH's; this
  package supplies the matter's `(ρ, S_i, S_ij)` and reads the metric's
  `(α, β^i, γ_ij, K_ij)`, both at TreeGH's vertices. Gauge drivers, excision,
  radiative boundaries and Ψ₄ are TreeGH's and are not needed before
  collapse, binaries or supernovae.
- **No con2prim of its own.** EntropyEOS's `con2prim_safe` is used for
  every σ (see [Primitive recovery](#primitive-recovery)). The `Con2Prim`
  package in the sibling checkout is an abandoned prototype and is not a
  dependency.
- **Upstream changes are Erik's.** This document states them
  precisely under [Upstream prerequisites](#upstream-prerequisites); the
  plan sequences the steps that need them to wait. Do not edit the sibling
  checkouts from here.
- **Not yet a production code.** Single node, threads or one device.
  TreeAMR has MPI since its M7 (as an extension) and checkpoints since its
  M9, which moved into the companion package TreeIOHDF5 with TreeAMR 0.2;
  this package uses neither yet.

## Upstream prerequisites

What this package needs from its siblings, and why. The exact
requirements — API, semantics, the tests the sibling should carry, and the
check this package adds when an item lands — are in [`PLAN.md`, "Upstream
requirements"](PLAN.md#upstream-requirements), under the IDs used here, for
the items still waiting; for those landed, withdrawn or superseded they are
in [HISTORY.md, "Upstream prerequisites"](HISTORY.md#upstream-prerequisites).
The summary below keeps only the reasons. The priorities "(req)",
"(wanted)" and "(later)" are defined with `PLAN.md`'s table. How the state
below came about is in [HISTORY.md, "Upstream
prerequisites"](HISTORY.md#upstream-prerequisites).

The state, as `[compat]` guarantees it for what has landed, as
`test/prerequisite_tests.jl` reports it for what is waiting, and, for
TreeGH, as its `main` reads. A landed item whose release the compat floor
requires has no probe any more (the rule is in `PLAN.md`, "Upstream
requirements"): AMR-1 to AMR-10, EOS-1, EOS-2 and SM-1. Their names are in
the prerequisite test's name lists; what this package relies on from them
and does not yet exercise through its own code goes in
`test/upstream_behaviour_tests.jl`, which is removed while it is empty (the
bounded prolongation is in the refinement tests) and returns when a landed
item needs it.

- TreeAMR **0.2.2**, registered, and the compat floor (`TreeAMR = "0.2.2"`;
  0.2.3, which only raises its floor to Julia 1.13, is released upstream
  and admitted by it) so that what landed is guaranteed: **AMR-1, AMR-2,
  AMR-3 and AMR-5 have landed in TreeAMR 0.2.1**, AMR-4 and AMR-6 in a
  different form (below), and **AMR-7 to AMR-10 in TreeAMR 0.2.2** (commit
  624c99a). Step 5, which needed AMR-1, is done on it.
- EntropyEOS, not registered (EOS-0 waiting), through `[sources]`, which
  pins the tag `v2.2.1` (EOS-10), with `[compat]` `"2.2"`: **EOS-1 and
  EOS-2 have landed in EntropyEOS 1.0.0**, and EOS-8 to EOS-11 have landed
  since (below); EOS-3 and EOS-4 have not.

  **Inconsistent:** `[compat]` `"2.2"` here (the last value this document
  states, with the tag `v2.2.0`; [HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos))
  against `CLAUDE.md`'s former "Current state" (now [HISTORY.md,
  "EntropyEOS"](HISTORY.md#entropyeos), 2026-10-30), which said `[sources]`
  pins the tag `v2.2.1` with `[compat]` `"2.2.1"` since 2026-10-30.
- IMEXRungeKutta **1.4.0**, tagged, not registered: the plan's steps 4a–4c
  are done there, and it is a dependency from step 0.
- SpacetimeMetrics **1.7.0**, registered (`SpacetimeMetrics = "1.7"`; it
  requires Julia 1.11, below the floor here, 1.13):
  **SM-1 has landed in SpacetimeMetrics 1.7.0**, and its workaround is
  gone.
- TreeGeneralizedHarmonic **0.1.1** on its `main`, with `adm_vars` and
  `ADMMetric`, and Adapt as a new direct dependency, on TreeAMR 0.2, not
  registered: GH-0 has landed (the step-8 work is on `main`); GH-1 is
  landed, `ADMMetric`, TreeGH's PR 25 (61bcb75); GH-3 and GH-4 have not;
  GH-2 is open (see `PLAN.md`); GH-5 is withdrawn. TreeGH is not a
  dependency of this package at all, so the suite cannot detect its items;
  how step 11's coupled tests load it is (open).

The unregistered dependencies are named in `Project.toml`'s `[sources]`
with their GitHub remotes; nothing resolves from a sibling checkout.

### TreeAMR

- **AMR-1 (req, step 5) A state vector spanning several field sets** —
  `statevector(sets)`, `scatter!(sets, u)`, `statearray(du, sets, i)`, as
  TreeAMR's `CODE.md` already specifies ("implemented with the first
  application that needs it"), plus segment views. The evolved state is a
  cell-centered matter set, three edge-centered sets for `A`, a
  vertex-centered set for the gauge scalar, and — when coupled — TreeGH's
  vertex-centered set; one integrator advances all of them. *Used:* the
  state vector is over `evolved_sets(f) = (U, A₁, A₂, A₃, Φ̃)`, and
  `test/scheme_tests.jl` makes the plan's downstream check — the round trip
  over exactly that tuple in D = 1, 2, 3, on a two-level forest, each
  segment bit for bit the set's own state vector.
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
  covering both ghost filling and the regrid transfer. *Landed in another
  form, TreeAMR 0.2.1, commit 902042a:* not a choice of limiter but a
  property, `Operators(; prolongation=3, restriction=2, family=Conservative,
  bounded=true)` — the order-3 conservative prolongation with an MC slope
  and a scaling that keeps every child within the range of its parent and
  the parent's face neighbours, exactly conservative, in the ghost fill and
  the regrid transfer alike. `test/refinement_tests.jl` checks that a
  surface refined everywhere stays positive and within range and keeps its
  total, where the unbounded operator overshoots, and the matter's ghost
  fill (on the primitives `W`) and regrid use it. See
  [Refinement](#refinement) for what upstream's design assumes of this
  package.
- **AMR-5 (wanted, step 14) `workgroupsize` on `map_blocks!`** (also on
  TreeGH's list).
- **AMR-6 (later) A device-capable boundary hook that reads the interior**,
  for outflow conditions. *Landed in another form, TreeAMR 0.2.1, commit
  a983419:* not a hook but a face kind, `Forest(roots; …, outflow =
  ((lo, hi), …))`, whose ghosts the schedule fills with the nearest owned
  value (zero-gradient, first order at the face), on every backend and
  across ranks. A nonlinear condition, such as forbidding inflow, stays
  with the application.
- **AMR-7 (later, step 14) A device-capable boundary hook relative to the
  nearest evolved point**, for the potentials' Dirichlet data in the field
  ("Boundaries"). *Landed in TreeAMR 0.2.2 as `CellBoundary(Relative(f))`,*
  which also applies the complete condition in every fill; the region-form
  workaround and `outer_points!` are gone.
- **AMR-8 (wanted, no step blocked) Symmetric walls**: an opt-in per field
  set under which `fill_ghosts!` fills the owned low wall of a vertex-like
  dimension by the rule of the high one — the hook, the reflecting
  derivation, the outflow clamp — so that a closed box is mirror-symmetric.
  *Landed in TreeAMR 0.2.2 as `FieldSet(…; walls = :symmetric)`;* the
  workaround is gone.
- **AMR-9 (wanted, no step blocked) A flux-conservative transfer of edge
  sets**: conservative along an edge's own (cell-like) direction,
  point-value along the others, so that the magnetic flux through a coarse
  face is the sum through its fine faces in the state, not only in its rate
  ("Conservation at coarse-fine faces"). *Landed in TreeAMR 0.2.2 as
  `Operators(…; family = Conservative, vertex_prolongation = w)`;* the
  potentials take it.
- **AMR-10 (req: step 8's ghost-fill re-measurement; steps 10, 12)
  Bounded PointValue interpolation**: `Operators(; prolongation=w,
  restriction=r, family=PointValue, bounded=true)` for even `w ≥ 2` on every
  centering, in the ghost fill and the regrid — every value within the range
  of the `2^D` points that enclose it — for the matter's ghost fill of the
  primitives ("Refinement"). *Landed in TreeAMR 0.2.2,* `bounded = true`
  with `PointValue` at every even width. The matter's ghost fill takes the
  bounded conservative operator, AMR-4, centred on the parent
  ("Refinement"); the PointValue fill stays for measurements and as the
  width-4 control of the suite.
- **AMR-11 (wanted, no step blocked) The relative hook consistent across
  levels where an outer face meets a coarse-fine face**: there each level's
  hook writes its own wall edges, so the magnetic flux through a coarse face
  touching an outer face is its fine faces' sum only to 1.2e−5 of the
  largest flux, where everywhere else it is to rounding ("Conservation at
  coarse-fine faces"). No workaround needed.
- *(later)* The use of MPI (M7) and of checkpoints (M9, now TreeIOHDF5's)
  and point interpolation (M11), all three of which TreeAMR has; none is
  an upstream item any more.

This is a **physics package** in TreeAMR's sense (its `CODE.md`,
"Companion interface", 2026-10-08): it uses TreeAMR's exports only, and
bounds TreeAMR by a minor series, `TreeAMR = "0.2.2"`, the release the
landed items are in. The prerequisite tests check the names it calls
against the exports, and `src/device.jl` uploads its own metadata with
`KernelAbstractions.allocate` rather than reaching for TreeAMR's internal
`todevice` (proposed).

What this package relies on from the landed items, beyond the summaries above (the requests and
their record: [HISTORY.md, "TreeAMR"](HISTORY.md#treeamr)):

- **AMR-1.** Segments are concatenated in tuple order, and each segment has exactly the single-set
  layout. So every single-set function applied to `stateview(u, sets, i)` and `sets[i]` behaves
  as on a separate vector. That covers `scatter!`, `gather!`, `statearray`, `block_mapreduce`,
  `mesh_mapreduce` and `volume_weighted_norm`. `statearray` and `stateview` alias `u`. Sets of any
  centering, `nvars` and `G` may be mixed. All sets must share one forest (`===`), one backend, one
  element type, and `nblocks == nleaves(forest)`; a violation throws an `ArgumentError` naming the
  set and the reason. The downstream check, `test/scheme_tests.jl`, "AMR-1 downstream", over
  `evolved_sets(f) = (U, A₁, A₂, A₃, Φ̃)` on a two-level forest, also asserts `scatter!` the
  inverse of `gather!`, `statearray` aliasing `u`, a segment's reduction equal to a separate
  vector's, and a set over another forest refused.
- **AMR-4.** `bounded = false` is the default and bit-identical; `bounded = true` is also accepted
  at conservative order 1 (already bounded) and refused otherwise. The slope is fixed to MC
  (minmod lost the interface order). One kernel serves the ghost fill and the regrid.
- **AMR-6.** The schedule fills an outflow face's ghosts as clamped transfers. `OutflowFace` is
  not exported: this package reaches outflow through `Forest`'s `outflow` keyword only. The step
  that first uses outflow faces tests them in place.
- **AMR-7.** `CellBoundary(Relative(f))` and `CellBoundary(Relative(AllVariables(f)))` set a ghost
  to `f(x_g) + (u(x_r) − f(x_r))`, `x_r` the nearest evolved point (clamped along the components
  that leave the domain through an outer face, `G+2` on the low side under symmetric walls; the
  block's own ghost across a block face along the others), as a kernel on every backend.
  `Relative` is in step 0's name list.
- **AMR-9.** `vertex_prolongation`, of even width, is required wherever a conservative set has a
  vertex-like dimension; `bounded` with it is refused (not implemented).
- **AMR-10.** `Operators(; prolongation=w, restriction=r, family=PointValue, bounded=true)` at
  every even `w`, on every centering, in the fill (prolongation, interpolating restriction, a
  reflecting wall's derived row) and the regrid: the order-`w` value clamped to the range of its
  bracket, the `2^D` points multilinear interpolation reads — bit for bit the unbounded value where
  no clamp acts. Width 2 takes the plain kernel.

### TreeGeneralizedHarmonic

Every item is step 11's. GH-1 and GH-3 are adapters supplied by the
producer; both answer at TreeGH's owned vertices, in 3+1 variables, and
every conversion between centerings is this package's ([The coupling
contract](#the-coupling-contract)). GH-5 is withdrawn.

- **GH-0 (req, step 11) Step 8 on `main`.** The unmerged step-8 branches
  widen `GHProblem`, `gh_rhs_kernel!` and `evolve!`; what the items below
  target is in [`PLAN.md`, GH-0](PLAN.md#gh-0--land-step-8-on-main-req-step-11).
- **GH-1 (req, step 11) The metric at TreeGH's vertices**: `(α, β^i,
  γ_ij, K_ij)` at its owned vertices, from an adapter built from its
  prepared state and bound to this package's vertex set `Mv` on the same
  forest. **Landed as `ADMMetric`** (TreeGH's PR 25, 61bcb75):
  `adm = ADMMetric(p, u, t)` (scatter and fill, as `gh_rhs!` does) or
  `ADMMetric(p)`, `at = adm(Mv)`, `(; α, β, γ, K) = at(n, b)` at an owned
  index; `isbits` once adapted; a `BoundsError` outside the owned range.
  `α`, `β`, `γ` are the decoded state, `K` converges at TreeGH's order `q`
  (so `q ≥ 4` for the contract's 4th order).
- **GH-2 (req, step 11) Registration, and compat satisfiable from the
  registry**, for whatever loads TreeGH beside this package. A dependency's
  `[sources]` is ignored downstream, so every TreeAMR feature TreeGH uses
  must be in a released TreeAMR. The Julia floor is discussed under [Julia
  version](#julia-version).
- **GH-3 (req, step 11) A matter input**: a `matter` argument to
  TreeGH's right-hand side, any adapter that, bound to TreeGH's vertex set,
  returns `(; ρ, S, Sij)` at an owned vertex; `nothing` is vacuum, switched
  by a `Val`, bit-identical to today. TreeGH assembles the covariant `T_ab`
  from them with its own `α`, `β`, `γ^ij`, and it enters as
  `∂ₜΠ_ab += 16π α√γ (T_ab − ½ g_ab T)` (derived from
  `notes/formulation.md` §1–2: `R^H_ab = 8π(T_ab − ½g_abT)` turns `S0`
  into `S0 − 16πT̄_ab`, and the source enters as `−α√γ S0`). The ADM
  monitor gains the matter terms, `ℋ − 16πρ` and `ℳ_i − 8π S_i`.
- **GH-4 (req, step 11) A right-hand side callable inside an external
  integrator**: TreeGH's own verbs under the names this package shares —
  `TreeGH.prepare!`, `TreeGH.rhs!`, the integrator-free limiters, the
  speed for `Δt` — all accepting segment views of a multi-set state
  vector, and `evolve!` as a composition of them. A script calls them;
  neither package drives a coupled run.
- **GH-5 (withdrawn) A ghost width of at least `q/2 + 1`**, so
  that a 4th-order vertex→cell metric interpolation reaches every stored
  cell of a `G = 3` matter set. Not needed: the interpolation reads this
  package's own vertex set `Mv` at `G = 4`, and TreeGH's ghost width does
  not enter the contract.
- **GH-6 (later)** A damped-harmonic gauge driver, radiative or
  constraint-preserving outer boundaries, excision or a generalized
  interior for collapse, Ψ₄ extraction.

The CFL factor of TreeGH under an SSP3 explicit tableau (`cfl = 1/4` was
measured with RK4, whose stability region on the imaginary axis is wider)
is *not* an upstream item: this package owns the integrator and measures it
in step 11.

GH-1 as landed, beyond the bullet above (TreeGH's `CODE.md`, "The coupling contract with
TreeGRRMHD"; the request and its record:
[HISTORY.md, "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic)):
- `ADMMetric(p)` wraps the field set as the caller filled it; `adm(Mv)` binds to a
  vertex-centered set on TreeGH's forest (`===`).
- Binding refuses another forest or another centering with an `ArgumentError`; a call outside
  the owned range throws a `BoundsError` unless the call site is `@inbounds`.
- `isbits` once adapted (`Adapt.@adapt_structure`); a call allocates nothing.
- `K` comes from TreeGH's own first-derivative stencil and converges at order `q` (measured 2.00,
  4.00, 5.99).
- Whether GH-4's `prepare!` returns the adapter is TreeGH's open item.

### EntropyEOS

- **EOS-0 (req, step 0; workaround) Registration** in General. Until then
  `Project.toml`'s `[sources]` adds it by URL, for CI, a clean checkout
  and a developer alike.
- **EOS-1 (req, step 5) An abstract EOS interface**: an
  `AbstractEOS` supertype, the solver and policy retyped to it, and the
  four bounds read through accessors instead of fields, so that `evaluate`,
  `con2prim_safe`, the atmosphere policy and the device view work unchanged
  for any EOS that implements `evaluate`, `srange`, `srange_extended`, the
  bounds and `κ`. *Landed in EntropyEOS 1.0.0*, with one difference from
  the request: `κ` stays unexported and is documented as `EntropyEOS.κ`,
  so this package calls it qualified.
- **EOS-2 (req, step 5) Analytic EOSes behind that interface**: a Γ-law
  and a hybrid (piecewise polytrope plus thermal Γ), as potentials
  `U(ρ, s)`. **A polytrope is the Γ-law at fixed entropy**,
  `U = K(s) ρ^{Γ−1}/(Γ−1)` with `K(s) = K_ref exp((Γ−1)(s − s_ref))`; a
  purely barotropic EOS (`U_s ≡ 0`) cannot be offered, because EntropyEOS's
  `(s, w)` Newton is singular there. It is needed from step 5, since the
  synthetic table cannot host the flat-space tests — see
  [Thermodynamics](#thermodynamics). *Landed in EntropyEOS 1.0.0* as
  `IdealGasEOS`, `HybridEOS` and `polytropic_entropy`, with the requested
  keyword constructors; both are `isbits`, `κ = 1`, and the TOV polytrope
  `K = 100, Γ = 2` is `IdealGasEOS` at `polytropic_entropy(eos, 100)`.
- **EOS-3 (wanted, step 12) A public fixed-entropy inversion** — the "at
  fixed `s`, solve for `w`" step that `con2prim` already contains — for the
  entropy fallback, and a public entropy-from-energy solve for initial
  data.
- **EOS-4 (later, step 13)** A cold β-equilibrium slice `Yₑ,β(ρ)` and
  `s_cold(ρ)` for tabulated-EOS stars.
- **EOS-8, EOS-9 (landed)**: the density floor reported (EntropyEOS
  1.1.0), and the endpoint branches made edge-consistent, the hottest state
  projected onto the entropy ceiling (2.0.0, as the two-level projection).
  See "Primitive recovery", "The two levels".
- **EOS-11 (landed)**: `con2prim_safe`'s validity test of the `s` and `Yₑ`
  faces, with its slack, public for the reconstruction — `inside_window`
  (EntropyEOS 2.1.0). See "Reconstruction" and [HISTORY.md, "Floors, the
  atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere),
  "EntropyEOS 2.0".
- **EOS-10 (landed)**: the nested fallback allocation-free under
  `--check-bounds=yes` (PLAN.md), where it allocated one `MVector{33,T}`
  (272 B at `Float64`, 144 B at `Float32`). EntropyEOS 2.2.1 keeps the
  scan's scratch on the stack under bounds checks, bit for bit; `[sources]`
  pins `v2.2.1`.

What this package relies on from the landed items, beyond the summaries above (the requests and
their record: [HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos)):
- **EOS-1.** The prerequisite test's EntropyEOS name list asserts that `κ` is unexported, so an
  export later is noticed.
- **EOS-11.** `inside_window(eos, ρ, s, yₑ) -> Bool`, exported — `Yₑ` within one ulp of the
  table's range and `s` within `srange(eos, ρ, yₑ)` widened by `face_slack` of its width; `ρ` not
  tested against a range; non-finite input or `ρ ≤ 0` gives `false`. Every state `con2prim_safe`
  or `project_prim_state` returns passes it. `inside_window` is in `test/prerequisite_tests.jl`'s
  name list.

### SpacetimeMetrics

- **SM-1 (wanted, step 9; workaround) An allocation-free extrinsic
  curvature**: `ExtrinsicCurvature(m, p)` allocates 432 bytes per call
  (Julia 1.11 and 1.13), which a device kernel cannot, and it
  differentiates the metric again where the `Analytic` provider already
  holds one `dmetric` result; the request adds `ExtrinsicCurvature(g, dg)`
  on that result. *Landed in SpacetimeMetrics 1.7.0:* `src/metric.jl`
  calls `ExtrinsicCurvature(g, dg)`, and its local `extrinsic_curvature` is
  gone; `test/pointwise_tests.jl` keeps that formula as the check on
  upstream's.

`test/pointwise_tests.jl`'s metric-provider testset compares upstream's `K_ij` with the former
workaround's formula, 3.7e−16 of `max |K|` (upstream's two methods agree bit for bit, so comparing
them checks nothing independently), and step 2's kernel test asserts that `metric_gradient`, which
calls the new method, allocates nothing.

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
each other in a test). Both are `geometric_source` in `src/sources.jl`, and
on boosted Kerr-Schild with fields up to 100 times the fluid's energy they
agree to 5.4e−16 of the magnitude of their terms; the covariant form also
agrees with `√−g T^{μν}(∂_μ g_νj − Γ^δ_νμ g_δj)` from SpacetimeMetrics'
Christoffels to 1.2e−15, and the energy source with Banyuls et al.
(1997)'s `√−g (T^{μt} ∂_μα − α T^{μν} Γᵗ_μν)` to 5.6e−16. **The covariant
form is the default** (proposed): every provider produces `∂ⱼ g_ab`
natively, and the Valencia form must first derive `∂ⱼα` and `∂ⱼβⁱ` from it.
Measured (Julia 1.11, `Float64`, one call, Apple M3 Pro): 36 ns against
50 ns for the geometric source, 41 ns against 74 ns for a cell's whole
explicit source.
The entropy equation is not a conservation law
through shocks; `D s` is advected only as a fallback for the primitive
recovery. As built, the reset rewrites `D s` only where the policy repaired
the cell; resetting it from the recovered `s` wherever the energy inversion
succeeds, the resynchronisation, was tried and not adopted ([HISTORY.md,
"Matter: total energy and
momentum"](HISTORY.md#matter-total-energy-and-momentum); [HISTORY.md, "Open
questions"](HISTORY.md#open-questions)).
Whether `D s` earns its keep is (open).

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
this; a test pins both signs). Measured: the EMF agrees with
IllinoisGRMHD's form to 1.4e−15 of its terms on random states in boosted
Kerr-Schild; the flux of `𝔈` is Komissarov's `−[kdl] H_l` exactly; and in
flat space the fluxes, the EMF, the gauge potential and the gauge flux
reproduce an exact electromagnetic plane wave in the Lorenz gauge —
`∂ₜE = ∇ × B`, `∂ₜA = −E − ∇Φ`, `∂ₜΦ = −∇·A` — to 6.3e−16, with
Ampère's source equal to `−j` to 2.2e−16 (`test/pointwise_tests.jl`). The gauge is generalized Lorenz with damping `ξ`
(Farris et al. 2012), which removes the zero-speed gauge mode that
otherwise produces spurious `B` at refinement boundaries. `ξ` is (open).

**The charge density is derived, not evolved**: `q := ∇·E`,
computed from the ghost-filled cell-centered `𝔈` in every right-hand-side
evaluation. Gauss's law then holds identically and charge conservation is
implied; there is no evolved `q` and no cleaning scalar for `E`. The known
cost is that any numerical divergence of `E` — at coarse-fine interfaces,
at the stellar surface — appears as charge and drives `q v` currents; the
charge is monitored as a diagnostic.

**A uniform background field** `𝔅̄ⁱ` is added to `curl A` (proposed). It
is what lets a vector potential represent a net flux through a periodic
box, and a field along a direction the mesh does not resolve: a 1D shock
tube's constant `Bˣ` is `B̄ˣ`, since no translation-invariant `A` produces
it. It is zero for stars. **The densitized `𝔅̄ⁱ` is the constant**
(proposed), not `B̄ⁱ`: with `√γ` varying, `∂ᵢ(√γ B̄ⁱ) = B̄ⁱ∂ᵢ√γ ≠ 0`, a
divergence the curl cannot cancel, while a constant `𝔅̄ⁱ` is
divergence-free on any metric. In flat space the two are the same. It is
`MHDProblem`'s `𝔅̄` and a case's (`src/curl.jl`); the Alfvén wave's guide
field is the first user.

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
| `MagnetizationCutoff(model, μ_max)` | `model`'s, → 0 where `B²/D → μ_max` | any finite σ, where the plasma is magnetically dominated |

The `q v` part is explicit. The `σ` part is **stiff** inside a star
(`σ Δt ≫ 1`), and is the implicit part of the time integrator.

*(Proposed:)* the models are in `src/ohm.jl`, and
`conductivity(model, prim)` takes the whole primitive state, so that a
later model may read `s` or `Yₑ`. `DensityProfile(σ₀, ρ_atm, cutoff = 2)`
is `σ₀ max(0, 1 − cutoff·ρ_atm/ρ)`: continuous, `σ₀` deep inside matter
and zero at and below `cutoff·ρ_atm`, so the atmosphere carries no current;
rung 6 settles its shape against Palenzuela (2013). For `IdealConductor`
the conduction current is not a function of the state — it is whatever
holds `E = −v × B`, which the stage solve imposes — so `conductivity` is
`Inf` and `ohm_current` throws an `ArgumentError` rather than return
`∞ · 0`. Where `σ = 0` the conduction term is not formed (a mask
branches). Measured: at `E = −v × B` the current is 2.3e−16 of
`σ W |B|`, and in flat space the bracket is the fluid-frame field
`F^{ab} u_b` with the charge the normal observer sees, to 4.4e−16.
*(Proposed:)* `conductivity_slope(model, prim)` is `∂σ/∂ρ`, for
the stage solve's Newton Jacobian; a later model that reads `s` or `Yₑ`
adds their derivatives. Extension slots, not planned: a mean-field dynamo
term, and force-free currents in the magnetosphere.

**σ → 0 where the plasma is magnetically dominated**, and not only where
`ρ → ρ_atm`. At
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
solve for that regime was built and is not in the code ([HISTORY.md, "Ohm's
law and the conductivity"](HISTORY.md#ohms-law-and-the-conductivity)).

*As built* (proposed): `MagnetizationCutoff(model, μ_max)` wraps
any finite-σ model and multiplies its σ by a taper `g(μ)` of the
magnetization `μ = magnetization(m, B, D) = γ_ij Bⁱ Bʲ / D` — the normal
observer's field over the rest-mass density, which bounds the fluid
frame's `μ/W ≤ b²/ρ ≤ W μ`. *(Proposed:)* `μ = +∞` where
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
- **`μ_max` is required and has no default.** It trades the stage's
  robustness against how much of the
  magnetically dominated plasma stays resistive, which is a property of the
  case, not of the scheme: each case and driver states its own value, no
  code path wraps a model implicitly, and `MagnetizationCutoff(model)`
  throws an `ArgumentError` saying so. The tests pass it explicitly.
- **The measured safe value, `μ_max = 10`**: the
  largest of `{5, 10, 20, 30, 50, 70, 100, 300, 1000}` at which the
  iteration, with no other help, has no fallback at any fixed `f ≤ 0.1`
  over the whole grid (`σΔt ∈ 10⁻³…10⁸`, `b²/ρ ∈ 10⁻⁴…10⁴`,
  `W ∈ {1, 2, 10}`, eight problems a point) at every Γ across sixteen seeds
  of the test's generator — 41,472 problems per Γ and `f`. At `μ_max = 10`
  none falls back up to `f = 0.3`. The taper begins at `μ = μ_max/10 = 1`;
  since `b²/ρ ≤ W μ`, σ is zero by `b²/ρ ≈ 10` for a fluid at rest and
  `≈ 100` at `W = 10`.
- **A physically motivated larger value trades a small fallback rate.**
  Ideal-GRMHD codes cap the magnetization at
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
`stress_energy` and `T4_lower` are in `src/stress_energy.jl`.
`nᵃnᵇT_ab = 𝓔` and `−nᵇT_ib = S_i` hold to 8.0e−16 of `max |T_ab|`; the
fluid part is `ρh uᵃuᵇ + p gᵃᵇ` to 5.5e−16, the field part is trace-free to
1.2e−15, and in flat space it is `F^{ac}F^b_c − ¼ηᵃᵇF²` with
`F_{i0} = E_i`, `F_ij = [ijk]Bᵏ` to 3.3e−16, which pins the sign of the
Poynting term.

## Thermodynamics

EntropyEOS everywhere: its run-time variables are `(ρ*, s, Yₑ)`,
with `ρ* = κ ρ` a rescaled baryon density chosen by the table build, `s`
the entropy per baryon, and a single C² potential `U(ρ*, s, Yₑ)` from which
`p`, `h`, `c_s²` and the consistent temperature follow. Consequences:

- **The primitive thermodynamic variable is `s`**, not `ε` or `T`. `D`
  is `√γ ρ* W`, so a checkpoint is tied to the table's `κ`.
- **A units adapter** converts between geometric units (`G = c = M☉ = 1`)
  and EntropyEOS's run units (cgs-like: densities in g/cm³, `U = ε/c²`),
  including `κ`. It is the only place a unit conversion is written.
  *(Proposed:)* it is `GeometricEOS(eos; density)` in
  `src/units.jl`, an `AbstractEOS` around an EOS in its own units whose
  densities are the scheme's: `evaluate` scales `ρ★` by the density unit on
  the way in and `U_ρ`, `U_ρs` (once), `U_ρρ` (twice) and `p` (inversely)
  on the way out, the entropy windows are the inner EOS's at the scaled
  density, and the density bounds shift by its logarithm. The recovery's
  equations are homogeneous of degree one in the densities (`D`, `τ`, `S`,
  `B²`, `ρhW²` and `p` scale together), so `con2prim_safe`, `prim2con` and
  the atmosphere policy then run in geometric units unchanged, with no
  conversion at any call site. The density unit is `c⁶/(G (GM☉)²)` =
  6.1758e17 g/cm³ (from the IAU 2015 nominal `GM☉` and CODATA 2018 `G`).
  `κ` enters only where a physical rest-mass density
  `ρ★/κ` is wanted (initial data in `ρ`, a baryon-mass diagnostic), through
  `EOSUnits`; the scheme's `D` is `√γ ρ★ W` throughout. An analytic EOS
  built in geometric units — the TOV polytrope, the shock tubes — needs no
  adapter at all. Measured: the adapter reproduces an ideal gas
  written directly in geometric units to 9.1e-15 in every `EOSPoint` field,
  and a table recovered through it agrees with the same table recovered in
  g/cm³ to 3.0e-15 in `(ρ, w)` and 8.4e-13 in `s`, conditioned.
- **Warm starts** (`u_guess`, the previous `log₁₀ T`). *(Proposed:)*
  every **recovery** is seeded from the state itself — `s` from the advected
  `(√γ D s)/(√γ D)`, a cold rapidity and EOS warm start (`state_guesses` in
  `src/evolution.jl`) — and not from `P`. A warm start carried in `P` makes
  each recovery depend, at the solver's tolerance, on whatever was evaluated
  before: the right-hand side would not be a function of `(u, t)` bit for
  bit (the plan's purity claim), and a restart from a checkpoint, which
  holds no `P`, would not reproduce the run. The face states, reconstructed
  from `P` within one evaluation, still take their cell's EOS warm start.
  The cost is a cold rapidity: a recovery over the stored cells is 0.86 µs a
  cell on the Alfvén wave (measured below). A table's `log₁₀ T`
  is the warm start this loses; if step 13 finds it matters, the state can
  carry it the way it carries `D s`. The right-hand side is pure in
  `(u, t)` **outside a step**;
  **a step is pure in `uⁿ`** (and `t`, `Δt`): inside a step of `evolve!`
  every recovery is seeded from the cell's latest recovery in that step,
  and a recovery whose input is that one's is reused — [the step
  cache](#primitive-recovery), invalidated at the end of every step, so a
  restart from `uⁿ`, which holds no cache, still reproduces the run.
- **Analytic EOSes come from EntropyEOS** (EOS-1, EOS-2). They are
  needed from the first evolution (step 5) on, not only from rung 4: the
  synthetic ideal-gas table is `Γ = 5/3` only, in cgs, with `T ≤ 50 MeV`,
  so it cannot host the flat-space tests (Balsara 1 is `Γ = 2`, and
  Balsara 3 has `ε = 1500`). EOS-2 has landed, so the synthetic table
  serves only as the table the units adapter is checked on.
- **A polytrope is a Γ-law at fixed entropy**, not a barotropic EOS:
  EntropyEOS's `(s, w)` Newton needs `U_s ≠ 0`. The TOV star's
  `p = Kρ^Γ` is the Γ-law evaluated at the `s` with `K(s) = K`.

## Primitive recovery

Because `E` is evolved, the electromagnetic energy `½(E² + B²)` and
momentum `ε_ijk Eʲ Bᵏ` do not depend on the fluid velocity. Primitive
recovery is therefore **pure hydrodynamics**: subtract them from `τ` and
`S_i`, project with `γ^{ij}` to EntropyEOS's scalar form
`(D, τ, D_Y, S_∥, S_⊥, B²)` with `S_∥ = 0`, `S_⊥ = |S|`, `B² = 0`, and
call `con2prim_safe`. With `B² = 0` the two splits give the same solve, but
EntropyEOS requires `S_⊥ ≥ 0` while `S_∥` is signed, and its own `prim2con`
calls all of the momentum perpendicular when there is no field to project
against; the repaired states it packages then carry the speed in `v_⊥`. The
velocity is rebuilt with its direction from `Sⁱ = γ^{ij} S_j` and its
magnitude from the returned rapidity, `vⁱ = tanh w · Sⁱ/|S|` (proposed).
That makes the primitive state self-consistent to rounding, and right on a
repaired state; the literal `vⁱ = Sⁱ/(ρhW²)` carries the momentum residual
into `atanh |v|` amplified by `W²`, which left 7e-11 in the conserved `D`
of the round trip against 4e-14. This is
`con2prim_hydro` in `src/con2prim.jl`; `con2prim_ideal` is the `B² ≠ 0`
call on the totals, splitting `S` along and across `B` in the metric, used
only as the stage solve's seed. It never fails: atmosphere, ceilings, rapidity
cap, the entropy window and non-finite input are its policy, every repair
is reported in its flags, and every repaired state is exactly invertible.

How exactly it inverts is conditioned, not uniform.
EntropyEOS states its round-trip accuracy with the errors divided by
`κ_ρ = hW²` (for `ρ` and the rapidity `w`) and by
`κ_s = (1 + τ/(ρW²U))·max(1, U/(s U_s))` (for `s`): the entropy is
ill-determined where the thermal energy is a small part of `τ`, and the
density where `hW² ≫ 1`. Its suite asserts the conditioned errors below
3e−11 and 1e−10; a raw relative 1e−12 is not attainable — 8.9e−10 was
measured in `s` at `ρ = 1e−8`, `s = 5.2`, `W = cosh 2` on the Γ = 2 gas
([HISTORY.md, "Primitive recovery"](HISTORY.md#primitive-recovery)).

Through the 3+1 layer (`test/recovery_tests.jl`): on
1000 random states per `Γ ∈ {4/3, 5/3, 2}` — `ρ` across the box, `s`
across the window, `W ≤ 10`, metrics from a boosted Kerr-Schild hole with
`a ≤ 0.9` at random events between `r = 3M` and `20M`, and `E`, `B`
carrying 1e−4 to 100 times the fluid's `τ` — `con2prim_hydro ∘ prim2con`
has conditioned worst errors 4.5e−13 in `(ρ, w, v)` and 8.4e−12 in `s`
(raw `s` up to 5.7e−7 at `Γ = 2`, where the thermal energy is a tiny part
of `τ` and `κ_s` is large). In the conserved variables, with `S_i` and `τ`
measured against the energy density `𝓔 = τ + D` they are components of
(proposed), the round trip at EntropyEOS's default tolerance is 9.8e−13
relative (`τ` against itself: 9.9e−13). That is the solver's stopping
criterion — `con2prim` stops when its energy residual is below 1e−12 of
`τ`, often at the cold seed with no Newton step — and not this layer's
rounding, and it leaves the 1e−12 of the step-1 acceptance no margin. So
the test runs the conserved round trip with `Con2PrimOptions(tol = 1e−14)`,
the tightest power of ten at which every draw still converges by Newton (at
1e−15 some fall back to the nested solve, and the round trip is no better):
the worst error is then 4.3e−14 against `𝓔` and 4.6e−14 for `τ` against
itself, both at `Γ = 2`, at Julia 1.11 and 1.13 alike; and the conditioned
primitives 3.0e−14 in `(ρ, w, v)` and 2.7e−13 in `s`. The scheme itself
runs at the default tolerance. `√γ D s` is not an input of the
hydrodynamic recovery and is checked through `s`. `con2prim_ideal` on
states built with `E = −v × B` (the same draws, `B²` up to 100 times the
fluid's `τ`, `B` along and across `v` included) has conditioned worst
errors 1.8e−12 in `(ρ, w, v)` and 3.7e−12 in `s`. At `Float32` the
recovery runs in a kernel and lands within 3.6e−5, conditioned, of the
state (EntropyEOS's `Float32` tolerance is 6.1e−5) on states dense enough
that the thermal energy is a percent of `τ`; at `ρ = 1e−9` the gas's `U` is
1e−7 of its rest mass and `Float32` cannot determine `s` at all.

**Inconsistent:** the default-tolerance round trip above (9.8e−13; `τ`
against itself 9.9e−13) and the `Float32` 3.6e−5 are the numbers from
before EntropyEOS 2.0. [HISTORY.md, "Primitive recovery"](HISTORY.md#primitive-recovery)
(2026-10-24) records that 2.0's `seed_passes` 5 moved the default-tolerance
conserved round trip to 1.07·10⁻¹², 1.21·10⁻¹², 8.5·10⁻¹³ (against
9.8·10⁻¹³, 6.8·10⁻¹³, 6.8·10⁻¹³), the `con2prim_hydro ∘ prim2con`
conditioned errors within 2× either way, and the `Float32` smoke run from
3.6·10⁻⁵ to 2.3·10⁻⁵.

`con2prim_hydro` is split at the fluid's share — `field_budget`
subtracts the fields' energy and momentum, `fluid_recovery` recovers from
what is left and also returns the EOS point — so that the stage solve,
which iterates on that share, calls the same recovery bit for bit.

`con2prim_hydro` returns EntropyEOS's policy flags as they are, and only
them: `policy_flags`, not the solver's result, is EntropyEOS's validity
signal, and a failed solve is always repaired and flagged (proposed). The
one decision this package adds to the policy — a state whose density the
policy had to raise to the atmosphere's is the atmosphere at rest — is
EntropyEOS's own level 2, below, and the flags of an atmosphere keep
level 1's bits ("Floors, the atmosphere and the magnetosphere").
EntropyEOS's conserved output is never used: every caller rebuilds the
conserved state from the primitive one with this package's `prim2con`.

**The two levels**. EntropyEOS 2.0 splits its policy in two, and both
recoveries call both, the second right after the first, in
`fluid_recovery` and `ideal_recovery` (outside the stage solve's iterations
level 1 runs as EntropyEOS's two phases and level 2 in a kernel of its own,
"The two-phase recovery" below):

**Inconsistent:** the parenthesis above says that outside the stage solve's
iterations level 1 runs as EntropyEOS's two phases and level 2 in a kernel
of its own; "The two-phase recovery" below says that fused
(`two_phase = false`: each recovery in one kernel, `con2prim_safe` and
`apply_policy` together) is the default, and the split an option.

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
  rule — a state whose density had to be raised to the floor is the
  atmosphere at rest — so `collapse_to_atmosphere` stays off. It holds
  `D_out ≤ max(D_in, D_atm)` by construction: of level 1's faces only
  `ρ_min` raises `D`, and that face goes to the atmosphere, whose
  `D = ρ_min = D_atm` (`test/recovery_tests.jl`, "A state the projection put
  on the ρ_min face is the atmosphere at rest"). The atmosphere's flags are
  level 1's bits plus `ATMOSPHERE`: the bits say why.
- **The options.** `MHDProblem`, `MHDCase` and every recovery carry `proj`
  and `pol` (`isbits`, two kernel arguments, `proj` first).
  `recovery_options(eos, ρ_min)` states both for every case:
  `default_projection(eos, ρ_min)` (`ρ_max` the top of the density box,
  `W ≤ 100`) and the atmosphere at `s_atm` (below). `atmosphere_prim` gives
  initial data the state `apply_policy` writes, bit for bit (a test).
- **The flags** are bits 8–17: `ATMOSPHERE` 8, `NONFINITE` 9,
  `NOT_CONVERGED` 10, `ρ_FLOORED` 11, `ρ_CEILED` 12, `YE_FLOORED` 13,
  `YE_CEILED` 14, `S_FLOORED` 15, `S_CEILED` 16, `W_CEILED` 17;
  `FLAG_POL_ANY = 0x3ff00`, still exact as a number in `Float32` (below
  2¹⁸ < 2²⁴; tested through `P` and the diagnostics). Bits 18 and 19,
  `FLAG_CS2_ACAUSAL` and `FLAG_CS2_NONPOSITIVE`, say that the returned
  `cs²` lies outside `(0, 1)`.
  They are the EOS evaluation's, like bits 0–5, set in `EOSPoint.flags` by
  every evaluation (`cs²_flags`), never in `policy_flags`, and `FLAG_POL_ANY`
  is unchanged. In this package they reach only the step cache's copy of the
  `EOSPoint` (below 2²⁰, exact in `Float32`; read back in a test), which
  nothing reads but a reused recovery's caller; `P`, `X` and every tally hold
  the policy's bits only. No case's ideal gas sets them (`cs² < Γ − 1 ≤ 1`
  and positive wherever `p > 0`), and the scheme's signal speeds are the
  light cone's, so `signal_cs²` has nothing to clamp here.
- **The face semantics.** Level 1 keeps each conservative a face does not
  free. `w_cap` keeps τ (the capped cell heats, bounded by the `s_max`
  face), `ρ_max` keeps `S` and τ, the `s` faces keep `S`. A failed solve
  with `D ≥ ρ_min` is slowed onto the `ρ_min` face rather than given mass
  (and so, here, the atmosphere), and an energy above the hottest state
  keeps `D` and `S` and lowers τ (`S_CEILED`, EOS-9).
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
  variant C, [HISTORY.md, "Floors, the atmosphere and the
  magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere)).
  On the 48 perturbed runs neither is better across the board:
  the minimum makes the uniform cut-off tube deterministic again (two runs
  `10⁻¹⁵` apart `1.7·10⁻¹³` apart, against `1.4·10⁻⁴` with the midpoint),
  every other row is the same within the spread ([HISTORY.md, "Floors, the
  atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere),
  "EntropyEOS 2.0"). The midpoint is kept because it changes no
  number of the suite: `s_atm` stays the explicit 1.x midpoint
  (`recovery_options`).
- **Why EOS-9 needs no `collapse_to_atmosphere`.** The collapse branch was
  1.1's answer to "τ above the hottest state": `ρ_ceiling` at `s_max`, the
  blow-ups of the magnetized tube, which the option turned into the
  atmosphere. In 2.0 that state is projected onto the `s_max` face with `D`
  and `S` kept (`S_CEILED`); where keeping `D` at the momentum puts `ρ`
  below `ρ_min`, the diagnosis takes the `ρ_min` face instead, and level 2
  writes the atmosphere at rest. Nothing reaches a ceiling the table cannot
  hold, so the option has nothing to do: the atmosphere tube drops it
  (measured: [HISTORY.md, "Primitive recovery"](HISTORY.md#primitive-recovery),
  2026-10-24, and [HISTORY.md, "Floors, the atmosphere and the
  magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere),
  "EntropyEOS 2.0").

What moved in the suite with EntropyEOS 2.0, 2.1 and 2.2, and the changes
of 2.1 and 2.2 that reach this package (`inside_window`, the search
confined to the density box, the sound-speed flags), are recorded in
[HISTORY.md, "Primitive recovery"](HISTORY.md#primitive-recovery).

The only coupling back to the fields is the implicit Ohm step, which is
[a nested solve](#the-implicit-stage-solve) around this same call.

**Entropy fallback** (proposed): where `τ_fluid = τ − D − ½(E² + B²)` is a
small difference of large numbers (electromagnetically dominated cells, the
magnetosphere) or the energy inversion fails, invert at the advected `s`
instead (EOS-3). The switch is a threshold on
`½(E² + B²) / τ` (open). *(Investigation 2026-10-16, [HISTORY.md, "Floors, the
atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere): measured on the magnetized atmosphere
tube at a threshold of `e_em > τ_f`, it removes the chaos as B + hot does, but
the advected `s` in those cells is itself noise.)*

**The step cache.** The only true input of a Runge–Kutta step is `uⁿ`;
every recovery after the first is of a state a little way from one the cell
has just recovered. The cache is (a) and (b) of the measurement of recovery
counts and costs that preceded it ([HISTORY.md, "Primitive recovery"](HISTORY.md#primitive-recovery),
"Measurement (2026-10-23): recovery counts and costs"). So inside a step:

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
  the undensitized `D`, `S`, `τ` as `· inv(√γ)`.
  On the Kerr-Schild cases, with the interior, one
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
- **No ghost is ever recovered**, so the cache has owned cells
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
  of the 2026-10-23 measurement). A cache that outlives a step either breaks
  restart reproducibility (a restart from `uⁿ⁺¹` has no cache) or forces the
  step limiter cold to keep the entry canonical; it would save at most about
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
estimate (36–60% for (a), most of (b) inside it; [HISTORY.md, "Primitive
recovery"](HISTORY.md#primitive-recovery)): of the 11 counted
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
unchanged. Re-measured under EntropyEOS 2.2 (2026-10-26), the cache's
saving is unchanged (57–64%) ([HISTORY.md, "Primitive recovery"](HISTORY.md#primitive-recovery)).

**The uniform flow: to the tolerance**. A
uniform flow through Dirichlet faces stays uniform to the recovery's
tolerance, not to rounding, and the testset says so: `≤ 10³·tol` (10⁻⁹)
after 13 steps. Measured why (on the laptop, which gives the same numbers
as Symmetry): with EntropyEOS's `tol = 10⁻¹⁴` instead of `10⁻¹²` the cached
runs give `0.7–2.9·10⁻¹²` again. A cold Newton solve ends with a quadratic
step that lands near rounding, and a uniform state's cells all take the
same one; a warm seed already within `tol` returns after no step or one, at
the tolerance, and which depends on the cell's history in the step — at the
Dirichlet faces, where the ghosts are the exact primitives, that history
differs, and the flow drifts by the tolerance per step. Rejected: a
polishing Newton step on
warm starts in EntropyEOS (it would roughly double every warm call, give back
much of the cache's saving, and break `con2prim_safe`'s fixed points), and a
tighter `tol` in the cases (below the energy residual's floor on tables, and
impossible at `Float32`). Measured on the laptop (2026-10-30), how the change
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

**The two-phase recovery**. EntropyEOS 2.x defines
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
  `IdealConductor`. No CPU/GPU switch beyond this one.
  **Fused is the default**
  (`two_phase = false` in `MHDProblem` and `evolve!`): on the CPU the split
  costs 1–10% a step and buys nothing, since only 0–2.4% of the recoveries
  reach the second phase and none of the stage solve's iterations would
  (measured below).
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
recoveries are all reused).

**The share that reaches finish** (`bin/recovery_cost.jl finish`, the same
job and cases; per site, the recoveries whose first phase was not done, of
all and of those not reused): **none** in the EM and Alfvén waves, the three
resistive tubes, Orszag–Tang and the refined resistive tube; the magnetized
atmosphere tube 6 of 120000 at the right-hand side (0.025% of those
started), 1755 of 160000 at the stage's first recovery (1.25%), 882 of 40000
at the step limiter (2.4%), none at the stage limiter; the spherical blast
1272 of 131072 at the stage's first recovery (1.3%); the rotor 2 and the
two-level rotor 4 there; nothing else. **The
stage solve's iterations reach finish nowhere**: 0 of their 57846 (the
atmosphere tube) to 531938 (the two-level resistive tube) recoveries, none
reused. So a later GPU step sees the second phase in at most about 2% of a
site's cells, at the sites above; the warm iterations would gain nothing
from the split.

### Where the fields live

A block's cells are at half-integer positions between integer vertices.
TreeAMR's centerings (`cellcentered`, `facecentered(D, d)`,
`edgecentered(D, d)`, `vertexcentered`) put each quantity where its
equation puts it:

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
| matter `U` | cell | 10 | 3 | Conservative width 3, bounded, in the regrid; its ghosts are not written by the scheme / `=> schedule` |
| `A_d`, `d = 1…3` | edge `d` | 1 each | 3 | Conservative width 3 along the edge, point values of width 4 across it (AMR-9) / `=> schedule` |
| `Φ̃` | vertex | 1 | 3 | PointValue, width 4 / `=> schedule` |
| `P` — primitives, EOS aux, flags, `E`, `B̄` at cells, `q`, cell metric | cell | 30 | 3 | `=> nothing` |
| `W` — `ρ★, s, Yₑ, uⁱ = Wvⁱ, Eⁱ`, the matter's ghost fill | cell | 9 | 3 | Conservative width 3, bounded, centred on the parent / `=> nothing` |
| `B_d` — `curl A` | face `d` | 1 each | 3 | `=> nothing`, recomputed every RHS |
| `F_d` — matter fluxes | face `d` | 10 | 0 | `=> nothing`; `restrict_interfaces!` is the flux correction |
| `EMF_d` | edge `d` | 1 each | 0 | `=> nothing`; `restrict_interfaces!` is the CT correction |
| `Mv` — the vertex metric `(α, β^i, γ_ij, K_ij)`, owned vertices from the producer's ADM adapter, in its own `VertexMetricSet`, not `MHDFields` | vertex | 16 | 4 (so `N ≥ 10`) | PointValue width 6 in its fill / `=> nothing` (`regrid_pairs(vm)`), refilled from the rebuilt adapter |
| `Tsrc` — `(ρ, S_i, S_ij)` at cells, only if this package's matter adapter stores it (step 11, (open)) | cell | 10 | 0 | `=> nothing` |
| `X` — per-cell tallies, light speed, the reset's injection, `h∇·𝔅`, the recoveries' seeds per site | cell | 31 | 0 | `=> nothing` |
| step cache — the cell's latest recovery in the step: EntropyEOS's input and output; `MHDProblem`'s, not `MHDFields`' | cell | 27 | 0 | not passed: a `rebuild` makes a new one, invalid |

`P`'s 30 slots are laid out in `src/variables.jl`: the eleven the recovery
returns — `ρ★`, `vⁱ`, `s`, `Yₑ`, the rapidity `w` (also the next recovery's
warm start), `p`, `h`, `cs²` and the EOS's own warm start — then the policy
flags (as a number; every bit is below 2¹⁸ — bits 8–17, `FLAG_POL_W_CEILED`
the highest, since EntropyEOS 2.0 — so exact even in `Float32`, which holds
every integer below 2²⁴, and tested at `Float32` with bit 17), `Eⁱ`, `Bⁱ`,
`q`, and the metric `α`, `βⁱ`, `γ_ij` and `√γ`, all undensitized. The metric
derivatives and `K_ij` the sources read are not stored, and `P` stays at 30
slots (proposed). The sources are their only reader, once per cell per
right-hand side, so storing them (36 more slots: `∂ⱼg_ab` and `K_ij`) would
only move the same evaluation into another kernel and add a write and a
read; the source kernel asks the provider for them (`metric_gradient`), as
it would ask for the metric at a face.

*As built* (`src/fieldsets.jl`, `mhd_fields(T, forest; G = 3)`):

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
- **`X`, the diagnostics set** (proposed): per owned cell, the
  implicit stages that set each of the nine `OhmStageResult` bits, the
  stages whose state the policy repaired and the `|` of their flags, the
  recoveries and the most one stage made, the cells the reset wrote, and the
  light speed for the time step; the cells the reset
  wrote as the step limiter, and the change those writes made to `√γ D`
  and `√γ τ`, summed — the rest mass and energy the reset injects. The counts accumulate over a chunk and are
  reduced once at its end (`ohm_tally`), which is "reduced per chunk"
  without a reduction per stage.
- **The operators**: Conservative width 3 for `U` and width 4 for `A` and
  `Φ̃` — the scheme's order plus one, rounded up to even — by the
  interface-order table, and the matter's width 3 **bounded** (TreeAMR's
  MC-bounded prolongation) — see "Refinement" and the table above. They are
  `MHDProblem`'s `ops_U` and `ops_A`. `A_j` take the conservative family on a
  staggered set, `Operators(; prolongation=3, restriction=2, family =
  Conservative, vertex_prolongation=4)` — the two-edge mean and a width-3
  conservative prolongation along the edge's own (cell-like) direction,
  point values of width 4 across it — so that the magnetic flux through a
  coarse face is the sum through its fine faces as a state; `Φ̃`, a vertex
  set with no cell-like dimension, would get the very same operators from
  that family and stays PointValue width 4 (prolongation and restriction;
  `ops_Φ`, its own keyword). The width 4 was re-measured under the new
  family (unchanged).

Operator widths follow TreeAMR's convention: the order of an operator is
its stencil width `w` per dimension and it is exact to degree `w − 1`;
PointValue widths are even, Conservative widths odd
(`TreeAMR/src/operators.jl`). TreeGH's `prolongation = 6` is a 6-point,
degree-5 interpolation. `B` takes one derivative of `A`, so `A`'s
PointValue width must be at least the scheme's order plus one.

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

*As built: PLM and the face state* (`src/reconstruction.jl`, proposed): twelve
numbers a cell, `(ρ, s, Yₑ, uⁱ, Eⁱ, Bⁱ)` from `P`, undensitized, each
extrapolated as `q ± slope/2` with TreeHydro's `slope` (`:none`, `:minmod`,
`:mc`; `limiter` has no default, as in TreeHydro). All three components of
`B` are reconstructed and the normal one is then replaced by the face's
`𝔅ᵈ/√γ`. The two groups are `(ρ, s, Yₑ, uⁱ)` — inadmissible where `ρ` is
not positive and finite, `Yₑ` outside the EOS's window, `s` outside
`srange(eos, ρ, Yₑ)` (both with the slack `con2prim_safe` allows, below)
or `u` not finite — and `(Eⁱ, Bⁱ)`, inadmissible only
where not finite; an inadmissible group takes its cell's values on that side
of that face. The face state is then `vⁱ = uⁱ/W`, one EOS evaluation
(`primitive_state`) and `prim2con` in the face metric, which the provider
evaluates at the face (`Flat`, `Analytic`; TreeGH's adapter will
interpolate it, step 11). A face's state costs two EOS evaluations, `primitive_state`'s and
`prim2con`'s.

**Inconsistent:** "TreeGH's adapter will interpolate it, step 11" contradicts
"The coupling contract" and "Metric at cells, faces and edges" (and
`CLAUDE.md`): under the redesigned contract TreeGH interpolates nothing, and
this package interpolates the metric from its own vertex set `Mv`
(`VertexMetric`) to cells, faces and edges.

*As built: the higher orders* (`src/reconstruction.jl`; proposed unless
marked otherwise). `limiter` names the reconstruction: `:none`, `:minmod`,
`:mc` (PLM, `G ≥ 2`) or `:ppm`, `:wenoz`, `:mp5` (`G ≥ 3`, which
`MHDProblem` checks). The keyword keeps the name `limiter` rather than
becoming `reconstruction`, so that no caller changes.

- **Every method is an offset from the cell's value**, `q_face = q₀ + δ`,
  with `δ` a function of the differences `q_k − q₀` alone. So a stencil of
  equal values reconstructs to itself exactly, `δ` is odd under `q → −q`
  exactly, and the lower face is the upper face of the mirrored stencil.
  That is what makes a direction a solution does not vary in add nothing at
  all, and the direction-independence claims of rung 2a hold bit for bit
  (see "Validation ladder" 2a); a method with its weights multiplied out
  (`(2a − 13b + 47c + 27d − 3e)/60`) rounds a constant. PLM's offset is
  `slope/2`.
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
  3 and 4. So `evolve!`'s `cfl` defaults to
  `default_cfl(:mp5) = 1/5`, and to 2/5 for every other method.
- Face values on smooth data (the cell averages of `sin 2πx`, a test)
  converge at 2.04 (MC), 3.97 (PPM), 4.99 (WENO-Z) and 5.02 (MP5); on
  monotone stencils PPM's and MP5's face values stay between the face's two
  cells and WENO-Z's overshoots by at most 5.6% of the stencil's rise.
- **The scheme stays second order** whatever the faces get: the primitives
  are recovered pointwise from cell averages and `E = −v × B` is imposed at
  cell centres, both `O(h²)` from the averages a fifth-order reconstruction
  would need. The Alfvén wave (`IdealConductor`, one period, 50 … 400 cells;
  below) converges at 2.67 (PPM), 2.00 (WENO-Z, MP5) in
  `S_y`, with 9, 17 and 17 times smaller errors than MC at 400 cells; the
  higher methods buy the constant, not the order. Converting averages to
  point values before the recovery (and back for the fluxes) is the known
  way past it, and is not planned.
- **The EMF's edge values** use the same offsets: `E` at a corner is
  `E₀ + δ_a + δ_b`, each offset the method's one-dimensional one along its
  direction, and the face-normal `𝔅` is reconstructed along the transverse
  direction by the method (see "The electromotive force for `A`").

**The reconstruction tests validity the way `con2prim_safe` does.**
`admissible_fluid` asks EntropyEOS's `inside_window(eos, ρ, s, Yₑ)` (EOS-11)
whether `s` and `Yₑ` lie inside their faces, with the slack `con2prim_safe`
allows — `s` within `srange(eos, ρ, Yₑ)` widened by 10⁻⁷ of its width at
`Float64` (2⁻¹², 2.4·10⁻⁴, at `Float32`), `Yₑ` within one ulp of its range —
and keeps this package's own conditions, `ρ` positive and finite and `u`
finite. Since EntropyEOS 2.0 `con2prim_safe` returns a converged root within
the slack of a face unflagged and as it is, so a recovered cell could fail an
exact test of the window: on the reproducer of 2026-10-24 (a cold re-solve of
a state the stage limiter had written onto the `s_min` face) `s` lies
5.5·10⁻⁹ of the window below `s_min` with `policy_flags == 0`. Every state
the recovery returns passes, and the EOS can be evaluated throughout the band
(`inside_window`'s contract). `ρ` is not tested against `[ρ_min, ρ_max]` nor
`w` against the cap, as `check_prim_state` would: that would change the
reconstruction near vacuum.

**The positivity fallback substitutes the cell's own value, so it cannot
repair a face whose cell is itself inadmissible**: that face gets the cell's
state, which fails the same test. A TVD limiter cannot produce such a face
from admissible cells (its face values lie between the cells'), so the case
arises only where a cell holds a state the test rejects — which, since the
test is `con2prim_safe`'s own, no recovered cell does (test:
`test/scheme_tests.jl`, "The positivity fallback admits every state
con2prim_safe returns unflagged"). What it cost while the test was exact is
in [HISTORY.md, "Floors, the atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere), "EntropyEOS 2.0" and "The ARS(4,4,3) regression".

### Riemann solver

`riemann_flux(::Val{:llf | :hlle}, …)` on the ten matter-plus-`𝔈`
variables at every face (proposed). Because `S_i` and `τ` are totals, their
fluxes contain the Maxwell stress, which propagates at the speed of light;
the default wave speeds are therefore the local light cone,
`λ_± = −β^d ± α √(γ^{dd})`, for all ten variables — the choice of the
resistive codes this design follows, robust and diffusive. Split speeds
(fluid speeds for `D`, `D Yₑ`, `D s`, light speed for the rest) are measured
below and are not the default; a resistive HLLC is (open), to be decided by
measurement on the TOV star.

*As built*: `src/flux.jl` and `src/riemann.jl`. A face state is the tuple
`(cons, prim, E, B)` — the densitized conserved state, the primitive state
it was built from, and the contravariant fields — which is what
`physical_flux(m, cons, prim, E, B, Val(d))` takes (proposed).
Both states are at the same face, so the light-cone speeds are the same on
both sides and the solvers need no EOS. Measured: `F(U, U)` is the
physical flux exactly for LLF and to 2.2e−16 for HLLE; the direction-`d`
flux is the direction-1 flux of the relabelled state to 4.7e−16 in flat
space and 6.8e−16 in Kerr-Schild; the fluxes are `α√γ Tᵈ_j` (momentum) and
`α²√γ T^{td}` (energy) to 4.8e−16; and the light cone is null in `g^{ab}`
to 1.6e−15.

The split speeds are `riemann_flux(::Val{:hlle_split}, …)` in `src/riemann.jl`
(measured; the light cone stays the default): Banyuls et
al.'s (1997) acoustic speeds, the extremes over both states, for `D`,
`D Yₑ` and `D s`, and the light cone for `S_i`, `τ` and `𝔈`. On the
Cowling star (`bin/tov_study.jl`, `h = 0.75`, MC, SSP3(4,3,3), Symmetry,
job 591972) the split run holds the central density within +5% for 100 M,
where the light cone's has fallen 6.5%, and then fails: at 105 M the
central density starts to climb, ten times its initial value by 130 M,
and by 200 M the box has lost 0.42 of its rest mass of 1.506 through the
outer faces while the reset injected 2·10⁻³. The light cone's run is
stable to 200 M (−10.5%, the rest mass net of the reset's injection to
1.3·10⁻¹⁵). A likely cause, not investigated: the split gives `D` far
less dissipation than `τ` and `S_i` at the surface and in the atmosphere,
so that the rest mass and the energy no longer move together there. HLLC
is not built.

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

*As built*: `uct_hll_emf(m, E, Bᵃ, Bᵇ, Val(d))` in `src/emf.jl`
(proposed). With `(d, a, b)` cyclic, its four states are the full `E` of
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
the physical flux of the HLL middle state — to 3.9e−16. A wrong sign in the EMF's shift or jump terms, or in the flux of `𝔈`,
breaks that.

*As built on the mesh* (`emf_kernel!` in `src/evolution.jl`, proposed), at
every edge of the closed range: the four states are the cells on
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

Under every reconstruction the corner value of `E` is
additive, `E₀ + δ_a + δ_b` with the method's one-dimensional offsets — for
PLM exactly the `E + σₐ slopeₐ/2 + σ_b slope_b/2` above — and not a tensor
product (reconstruct along `a`, then those values along `b`), which would
read 25 cells instead of 9 for a cross term that is `O(h²)`, the scheme's
order in more than one dimension anyway (proposed). Measured on
the Alfvén wave along the diagonal of a 2D box: `A_z`, whose EMF is the
one with two transverse directions, converges at 2.31 (MC) and 2.01
(WENO-Z).

### Sources

At cell centers, explicit: the geometric momentum and energy sources
above; the shift term `√γ q βⁱ` and the `−α√γ q vⁱ` part of the current in
Ampère's law. At vertices, explicit: the gauge damping `−ξ α Φ̃`. **The only
implicit term** is Ampère's `−α√γ σ W [E + v × B − (v·E) v]`.
`src/sources.jl`: `geometric_source` (both forms), the cell's
whole `explicit_source` (none for `D`, `D Yₑ`, `D s`), and Ampère's source
as `ampere_explicit_source` + `ampere_stiff_source` = `ampere_source`, the
stiff part kept as a function for the stage solve's residual.

### Floors, the atmosphere and the magnetosphere

EntropyEOS's policy defines the atmosphere and the valid range: the valid
range is level 1's (`ProjectionOptions`: `ρ_min`, what was `ρ_atm`; `ρ_max`,
what was `ρ_ceiling`; `w_cap`), the atmosphere level 2's (`apply_policy` with
`PolicyOptions`: at rest at `ρ_min`, at `s_atm`, below `1.01 ρ_min`), and
there is no collapse ceiling ("Primitive recovery", "The two levels"). How
the policy came to be what it is, under EntropyEOS 1.x and 2.0, is in
[HISTORY.md, "Floors, the atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere).

**The reset is both the stage limiter and the step limiter.** In
IMEXRungeKutta a stage limiter's change reaches
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

*As built* (`reset_atmosphere!`, `src/evolution.jl`): the reset
computes `B` from the state's own `A` (a scatter and ghost fill of the three
`A` sets and the curl, as the stage solve does), recovers every owned cell,
and **writes back every cell the policy repaired** — `FLAG_POL_ANY`, not the
atmosphere alone (proposed): every repaired state is exactly
invertible and is what the next right-hand side would recover anyway, so
writing it makes the state say what the scheme computes with. It writes the
seven matter slots and never `𝔈`, and counts each cell it writes into `X`.
A written state recovers as itself, so a second reset changes nothing (a
test).

The two roles are counted apart — `reset_atmosphere!` is the
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

**A density raised to the atmosphere's is the atmosphere at rest.** The
recovery calls `con2prim_safe` and `apply_policy`; the atmosphere tube does
not set `collapse_to_atmosphere`. A state whose density had to be raised to
the floor is the atmosphere at rest, `D_out ≤ max(D_in, D_atm)` —
`apply_policy`'s own rule on 2.0's `ρ_FLOORED`, the `ρ_min` face ("Primitive
recovery", "The two levels"). EOS-9 is level 1's diagnosis: the hottest
state keeps `D` and `S` on the `s_max` face (`S_CEILED`), or, where that
would put `ρ` below `ρ_min`, takes the `ρ_min` face and so the atmosphere; no
state is ever written at a density ceiling the table cannot hold, and the
option has nothing to act on (`ρ_CEILED` never appears in any run of the
EntropyEOS 2.0 measurement, [HISTORY.md, "Floors, the atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere)). The rule's
form under EntropyEOS 1.1, how it was found and who decided it are there
too.

**The perturbed atmosphere tubes** (`bin/front_investigation.jl`'s harness,
the same 48 perturbed runs as in the investigations recorded in [HISTORY.md, "Floors, the atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere); measured
2026-10-26 on Symmetry cn082, AMD EPYC 7532, one thread; EntropyEOS 2.2.0
with `inside_window`; the reconstruction's test instrumented in-process to
count its fallbacks). Columns as in the EntropyEOS 2.0 table there:
blow-ups; injected `D`; fallbacks; largest `W`; two runs `10⁻¹⁵` apart.
"(all)" where all 48 runs agree to the digits printed. "Blow-up" is the
harness's: injected `D > 10⁻⁶` or `|τ| > 10⁻³`.

| | 2.2, `inside_window` | 2.0, stage 2 ([HISTORY.md](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere)) | emulation, slack 10⁻⁷ (job 581987) |
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
digits printed (the control column of the EntropyEOS 2.0 table,
[HISTORY.md](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere)). Two rows still differ from 1.1: the ideal magnetized atmosphere under
SSP3 (`−7.6·10⁻⁹` against `−6.6·10⁻⁹`), and under ARS, which injects
`−5.0·10⁻⁹` against 1.1's `+5.6·10⁻⁸` but reaches the `w` cap (`W` = 100) in 3
cell-steps of every run, where 1.1 peaked at `W` 4.3. That is not the slack: it
is the same with the emulation, and the cell recoveries that differ between
the versions agree on identical inputs, so it is a different trajectory — a
candidate is 2.0's `S_CEILED` keeping `S` while lowering τ — investigated
below, "The ideal magnetized atmosphere's `W = 100`". The suite's assertion on that row (`W ≤ 5` at the end,
injected mass within 10⁻⁶) holds: its largest `W` at the end is 4.00.

**The ideal magnetized atmosphere's `W = 100`** *(investigated 2026-10-30
on EntropyEOS 2.2.1; accepted)*. Both
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
is EOS-9's no-collapse instead, as measured 2026-10-22 ([HISTORY.md, "Floors, the atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere),
"Investigation (2026-10-22)"). Under SSP3 nothing
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
cells (largest `W` 8.0 and 12.7). A TreeGRRMHD level 2 sending `S_CEILED` states to
the atmosphere was tried and not pursued ([HISTORY.md, "Floors, the atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere)). A lead for later, single laptop runs only: `atm_trigger` 2
or 10 removes the spikes in both versions, and at 10 the two are bit for bit
the same (ARS `W` 4.96, `−4.0·10⁻⁹`; SSP3 `W` 3.99) — a policy change,
untested elsewhere.

A predictor in the atmosphere could take `OHM_SIGMA_ZERO` (proposed; the
footnote to the controls of "Investigation (2026-10-16)" in [HISTORY.md, "Floors, the atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere)).

Step 3's pointwise sweep could add (proposed) a class of predictors whose
fluid share is near-null and at the entropy ceiling in the taper band,
asserting a finite, flagged fallback — what the front produced (the same
investigation).

## Time integration

### The partition

One IMEX Runge–Kutta integrator over the multi-set state vector. The
**explicit** part is everything except one term: all of TreeGH, the matter
and `𝔈` fluxes and sources, the EMF update of `A`, the gauge. The
**implicit** part is the σ current in Ampère's law, which is local to a
cell. The default tableau is Pareschi–Russo SSP3(4,3,3), whose implicit
part is L-stable, so that `σ Δt → ∞` relaxes to ideal MHD; SSP2(2,2,2) is
kept for debugging (proposed). Neither is stiffly accurate (their weights
`b` differ from the last row of the implicit `A`), so the order *in* the
stiff limit may drop; IMEXRungeKutta's validation measures it, and the ARS
schemes, which are stiffly accurate, are the alternative if it matters.
`Δt` is fixed per chunk from the light-speed CFL on the finest level;
TreeGH's CFL factor under SSP3 is measured here, in step 11. The driver is
chunked, with a regrid between chunks and a fresh integrator after each, as
in TreeHydro and TreeGH.

IMEXRungeKutta's validation measured SSP3(4,3,3): its implicit part is
L-stable (R(∞) = 0 from the closed-form α = (9 − √57)/6; A-stable by its
E-polynomial). In the stiff limit its order drops from 3 to 2 in the stiff
component (Kaps: 1.989 at ε = 1e−6, 1.988 at 1e−9), while the non-stiff
component keeps 3.01. Each step ends Δt(1 − bᵀA⁻¹c̃)f = −0.2844 Δt f off
the equilibrium, where f is the explicit tendency of the relaxing
variable. This does not accumulate over steps, but where the equilibrium
has a jump it is an O(1) overshoot of 0.2844 C on each side (total
variation +1.14 on 2 at C = 1). ARS(4,4,3) keeps order 3 and lands on the
equilibrium to O(ε).

**Measured in the scheme** (`test/wave_tests.jl`). Under `IdealConductor`
every stage solve imposes `E = −v × B` exactly, and the offset above leaves
the `𝔈` of each step's result `O(Δt)` off it. On the Alfvén wave the stored
`𝔈` converges at first order (L1 rate 0.85 over N = 50…400; errors 5.5e−3
to 9.4e−4), while its ideal projection — one stage solve of the final
state, `−v × B` of its ideal recovery — and every other variable converge
at second (2.09 to 2.85). The offset does not accumulate: the next stage
solve replaces it and no right-hand side reads it. Under ARS(4,4,3) the
stored `𝔈` is the projected one (the same errors to four digits, rate
2.10), for 4 explicit evaluations a step instead of 3: 8% more wall time on
the 1D wave. What the offset can still do is reach the step limiter, which
recovers `uⁿ⁺¹` — `𝔈` included — and writes back what the policy repairs:
where the field's energy dwarfs the fluid's, an `O(Δt)` error in `½E²` is a
large error in the fluid's share. ARS(4,4,3) is one keyword away
(`evolve!`'s `tableau`).

**SSP3(4,3,3) is the default**, on the measurements below. The two tableaus
give the same answer wherever the explicit part's stability does not
matter, and where it does, SSP3(4,3,3)'s SSPRK(3,3) explicit part keeps the
density positive and the reset idle while ARS(4,4,3)'s does not:
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
- in the magnetized ideal atmosphere (`Bʸ = 10⁻³`, `B²/ρ = 100`) neither
  tableau runs away — ARS injects `5.6·10⁻⁸` and ends at `W = 4.1`,
  SSP3(4,3,3) `−6.6·10⁻⁹` and 4.0 — and the row argues neither way (the
  runaway once recorded here was EntropyEOS's repairs, not ARS: [HISTORY.md,
  "The partition"](HISTORY.md#the-partition)). The others do, and the
  default stands on them: Balsara 4's accuracy and plateau, the resets on
  Balsara 3 and 4, the hydrodynamic rarefaction's undershoot to the
  atmosphere and the hydrodynamic tube's injection, the cut-off tube's
  stages (below), and the cost;
- with the magnetization cutoff it takes 23 fallbacks and 33 recoveries in
  a stage, SSP3(4,3,3) none and 6 (both deterministic);
- it costs 10–17% more wall time.

SSP3(4,3,3)'s price is the one measured in the scheme above: each step's
stored `𝔈` is `O(Δt)` off the stiff equilibrium — at a shock an `O(1)`
jump, up to 8.1% of `max |𝔈|` from its ideal projection on Balsara 1 at 400
cells (ARS: `10⁻³⁰`). Nothing reads it but the next stage solve, which
replaces it, and the step limiter, which in no Balsara, resistive or
current-sheet run wrote a cell; in the atmosphere runs the injection it
allows is the table's.

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
| rarefaction into `10⁻⁸` (atmosphere `10⁻¹⁰`): min ρ; resets stage/step | 1.0·10⁻⁸; 0/0 | 1.0·10⁻¹⁰; 131/110 |
| atmosphere tube, `B = 0`: injected `D`, `τ` | −8.4·10⁻¹², −2.2·10⁻¹⁰ | 1.2·10⁻⁹, 6.0·10⁻⁹ |
| atmosphere tube, ideal, `Bʸ = 10⁻³`: injected `D`, `τ`; max W at the end | −6.6·10⁻⁹, −1.7·10⁻⁵; 4.01 | 5.6·10⁻⁸, −2.4·10⁻⁵; 4.11 |
| atmosphere tube, `Constant(10³)` cut off at `μ_max = 10`: injected `D`, `τ`; fallbacks; most recoveries | −1.2·10⁻⁸, −7.4·10⁻⁷; 0; 6 | −1.2·10⁻⁸, −7.9·10⁻⁷; 23; 33 |

The four atmosphere rows were re-measured 2026-10-22
(`bin/tableau_comparison.jl atmosphere`, with the atmosphere-at-rest
recovery and EntropyEOS 1.1.0); their earlier values are in [HISTORY.md,
"The partition"](HISTORY.md#the-partition). Re-run in full the same day,
every Balsara and resistive number is the table's to the digits shown, and
the current sheet's distance reads 9.559·10⁻⁴ and 9.560·10⁻⁴ (SSP3, ARS),
untouched by the recovery; the wall times are within the load's noise.
Re-measured 2026-10-24, EntropyEOS 2.0, on Symmetry: SSP3's rows as above;
ARS's atmosphere rows and Balsara 3 move — the hydrodynamic tube injects
`4.4·10⁻⁷`, the ideal one `−2.0·10⁻⁹` at `W` 4.13, the cut-off one
`5.6·10⁻⁶` with 22 fallbacks, the rarefaction `1.5·10⁻⁶` with resets
133/116 — [HISTORY.md, "Floors, the atmosphere and the
magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere), "EntropyEOS
2.0". The decision for SSP3(4,3,3) stands on more, not less.

**As built** (`src/driver.jl`): `evolve!(T, case, Val(D); N, t_end, chunk,
cfl, limiter, …)` builds the forest of `case.roots` blocks of `N` cells,
the field sets and the problem, fills the exact initial state and runs
chunk by chunk, each a fresh integrator that takes over the last one's
scratch (`reuse`) and steps the state in place (`alias_u0`). Each chunk's
`Δt` is `cfl / (D · max λ/h)` from the light cone at the chunk's start
(`max_light_rate`, a `mesh_mapreduce` of the cells' light speed over their
spacing), rounded down to a whole number of steps with TreeHydro's
`chunk_count` tolerance; `cfl = 2/5` (proposed: TreeHydro's, under
SSP3(4,3,3)'s explicit part, which is SSPRK(3,3)). The stage arithmetic is
split evenly over the threads (`partition = :even`) for a CPU state; the
block-owner partition, which needs TreeAMR's unexported `threadchunks`, is
step 14's. A run in one chunk and the same run in four chunks of whole
steps of the same `Δt` are bit-identical (a test).

`cfl` defaults to `default_cfl(limiter)`: 2/5, and 1/5 for MP5 (see
"Reconstruction"). `after_step(p, u, t)` is called after every step when
given — rung 2a's `∇·𝔅` monitor (`max_div_B`). The initial data's cell
averages are the three-point Gauss rule applied one dimension at a time,
each average written as the middle sample plus a correction in the
differences, `f₀ + 5((f₋ − f₀) + (f₊ − f₀))/18`: three equal samples
average to themselves bit for bit, so a direction the case does not vary
in leaves the data exactly the 1D data (the earlier form, which did not:
[HISTORY.md, "The partition"](HISTORY.md#the-partition)).

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
`R(v) = v − V_c2p(S − E(v) × B, τ − ½(E(v)² + B²))`; as built, the same
map is iterated on the fluid's share of the energy and momentum instead
(see "As built" below). **One con2prim is used for every σ.** A plain
fixed-point iteration contracts roughly like `λ b²/(ρh)` with
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

**The solve writes only `E` into `U`** (proposed). No limiter has seen
`u★`, so its fluid part may be inadmissible. `con2prim_safe` repairs the
solve's own view of it, and that repair stays local to the solve. Written
into `U`, it would become part of the implicit increment `U − u★`. The
integrator would then re-weight it by `b_k/a_kk`, which is 2.758 at
SSP3(4,3,3)'s last stage (IMEXRungeKutta's `CODE.md`, "The callback
contracts" and "One step"). PLUTO's resistive solve does the same: it
repairs its predictor and writes back only `E`.

**As built** (`src/stage_solve.jl`):
`solve_ohm_stage(eos, opts, pol, σ, m, cons★, E★, B, γΔt, guesses; stage)
-> (E, prim, result, recoveries)`, allocation-free and bit-identical in a
CPU kernel at `Float64` and `Float32`.

- **The iteration runs on the fluid's share** of the energy and momentum,
  `X = (S_f, τ_f)`, not on `v`. Each iteration recovers the fluid from `X`
  (one `con2prim_safe`, hydrodynamic mode), solves Ohm's law for `E` at its
  velocity, and compares `X` with the share
  `X(E) = (S − ε E × B, τ − ½(E² + B²))` that `E` leaves. It is the same
  map as the root find for `v`, cut at a different place. The cut matters
  where `b²/ρ ≫ 1`: there `X(E)` is a small difference of large numbers, a
  velocity error `δu` moves it by `λ b²/(ρh) δu` of the fluid's own
  energy, and the recovery repairs whatever the fluid cannot hold. Newton
  on the velocity had a basin of `δu ~ 10⁻⁴` at `b²/ρ = 10³`, and fell
  back on 191 of 1296 points of the sweep below (Γ = 2); Newton on `X`
  hands the recovery only the linearly predicted share, which stays on the
  fluid's scale, and falls back on 40.
- **The velocity is carried as `u = W v`** (proposed): every `u` is
  subluminal, and the Jacobian through `v` has a `W² (E★ − E) vᵀ` term
  from `∂W/∂v` that through `u` becomes `(E★ − E) vᵀ/W`.
- **The Jacobian is analytic.** `∂u/∂X` is the recovery's derivative by the
  implicit function theorem on its two equations `|S| = z tanh w`,
  `τ = z − p − D` in `(s, w)`, with EntropyEOS's own `z_w`, `z_s`, `p_w`,
  `p_s` from the EOS point the recovery returns; `∂E/∂u` is implicit
  differentiation of the linear system. Newton's `1 − PQ` (4×4,
  `P = ∂X/∂E · ∂E/∂u`, `Q = ∂u/∂X`) is inverted by Woodbury through the 3×3
  `1 − QP`, and a density-dependent σ adds a rank-one term through `∂ρ/∂X`
  and `conductivity_slope(model, prim) = ∂σ/∂ρ` (proposed; a model reading
  `s` or `Yₑ` would add a column each). Both derivatives agree with central
  differences to the differences' own error, 1.1e−8 and 8.3e−8. No extra
  recovery, no finite difference.
- **Which branch** (proposed): σ evaluated at the predictor's recovery, at
  `E★`; if it is zero, `E = E★` and that recovery are the answer, bit for
  bit (`OHM_SIGMA_ZERO`). Otherwise the fixed point where the contraction
  factor `‖K‖`, the Frobenius norm in an orthonormal frame of the fixed
  point's Jacobian at the predictor, is at most `1/100`, and Newton
  otherwise; the fixed point also hands over to Newton when an iteration
  shrinks the residual by less than half (`OHM_SWITCHED`). Newton is seeded
  from the ideal recovery where `λ = cW/(1 + cW) > ½`, from the predictor
  otherwise, and from whichever of the two the policy did not have to
  repair if only one (`OHM_IDEAL_SEED`). At a threshold of `1/10` the fixed
  point needed up to 11 recoveries (4.4 on average) against Newton's 7; at
  `1/1000` the sweep below costs 3% fewer recoveries than at `1/100`, which
  is within its noise.
- **Clamps inside the residual.** `con2prim_safe` clamps (atmosphere,
  ceilings, the rapidity cap), and where a clamp is active a residual is
  flat and a Jacobian through it singular. A share the recovery had to
  repair is replaced by the repaired state's own (`fluid_share`), so that
  every iterate is a fluid state and its derivative is the recovery's; a
  Newton step that grows the residual is halved, up to eight times
  (`OHM_BACKTRACKED`); and any repair inside the iteration is flagged
  (`OHM_REPAIRED`). Leaving Newton for the fixed point at a repair, as
  first guessed, was not needed (the question and how grim and KHARMA
  answer it: [HISTORY.md, "The implicit stage
  solve"](HISTORY.md#the-implicit-stage-solve)).
- **Converged** when `|X − X(E)| ≤ 10⁻¹²(τ + D)` (proposed;
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
  `|` and count by bit (the scheme counts them per chunk).

**The sweep** (`test/stage_solve_tests.jl`; Julia 1.13, 1.11 alike but
where noted). Stage problems are built backwards from a known solution: a
fluid in a boosted Kerr-Schild metric, `E = −v × B` plus a non-ideal part
up to `|B|/(1 + cW)`, and the `E★` the stage equation then requires. On
`σΔt ∈ 10^{−3…8}` × `b²/ρ ∈ 10^{−4…4}` (ideal comoving `b²`) ×
`W ∈ {1, 2, 10}`, eight problems a point, for `Γ ∈ {4/3, 5/3, 2}`:

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
  from either limit was tried and does not reach it. The scheme takes σ
  to zero there with the [magnetization
  cutoff](#ohms-law-and-the-conductivity), under which none of these
  problems falls back (the uniform draw: 1 of 124,416 over sixteen seeds,
  at Γ = 2). Without it, the lagged `E` there is off the solution by a
  median 0.11 of `|E| + |B|`, at most 0.46 (Γ = 2);
- **the stage equations have more than one root** there: 5, 2 and 2 points
  converge, to the same residual, to another solution than the one the
  problem was built from;
- `DensityProfile` with its cutoff within a factor of three of the
  density: residual 8.6e−13, and 1 of 148 not converged, a Newton iteration
  held at the entropy window's edge (3 before the Jacobian carried
  `∂σ/∂ρ`);
- 200 inadmissible predictors (the share at `E★` below the coldest state):
  all finite, and where converged the residual is 1.9e−13.

Re-measured 2026-10-24 under EntropyEOS 2.0 (Symmetry, the same numbers on
the laptop): the fallback takes **49, 125 and 150** points (Γ = 4/3, 5/3,
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
`test/stage_solve_tests.jl`'s bound on the seeded fallbacks is 12 (a
margin of 1.5), for the uncut sweep only; the cutoff sweep the scheme runs
is unchanged.

Cost (Julia 1.13, Apple M3 Pro, the Γ = 2 gas, one call): 1.3 µs on the
fixed point (3 recoveries of 0.37 µs), 1.7–2.4 µs for Newton (3–4
recoveries), 0.7 µs at `σ = 0` and 0.5 µs for `IdealConductor`; a fallback
costs 33 recoveries, 0.17 ms where the recoveries themselves take
EntropyEOS's repair path. Under IMEXRungeKutta's SSP3(4,3,3), with
`solve_imp!` writing only `𝔈`, a resting cell's field decays as
`exp(−ασt)` at orders 2.92 and 2.96, its matter variables come back bit for
bit, and one step at `ασΔt = z` leaves `6.7/z` of the field (`z = 10⁶` and
`10⁸`), SSP3(4,3,3)'s L-stability.

### The IMEX package

The integrator is **IMEXRungeKutta.jl** (`~/src/jl/IMEXRungeKutta`), a
stand-alone package. Its `CODE.md` is authoritative for the method, the
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
  magnetosphere](#floors-the-atmosphere-and-the-magnetosphere)). Folding
  the reset into the tendency instead would re-weight it by `b_k/a_kk`,
  2.758 at SSP3(4,3,3)'s last stage;
- explicit evaluations only where the tableau uses them: three right-hand
  sides (TreeGH plus MHD) per SSP3(4,3,3) step;
- fixed `Δt`, generic arrays and Julia 1.10;
- a step that is never retried or rejected, whose step limiter is its last
  call, and an exception that leaves `integ.u = uⁿ`: with these the step
  limiter is where a step ends, and the step cache needs no step-begin hook
  ([Primitive recovery](#primitive-recovery), "The step cache").

Per SSP3(4,3,3) step, IMEXRungeKutta 1.4.0 makes 3 explicit evaluations,
4 stage solves, 3 stage-limiter calls (its first stage is implicit, so no
explicit stage is trivial) and 1 step-limiter call; the prerequisite tests
pin all four counts. It is a dependency through `[sources]`, so that the
contract is checked before the scheme builds on it.

## Spacetime coupling

### The coupling contract

How this package and TreeGH meet, with neither in charge. Each needs the
other: TreeGH needs the matter's stress-energy, the matter needs the
metric. TreeGH's side is in its `CODE.md`, "The coupling contract with
TreeGRRMHD". (How the contract came to be: [HISTORY.md, "The coupling
contract"](HISTORY.md#the-coupling-contract).)

- **No framework.** A short Julia script owns the loop. It builds one
  IMEXRungeKutta integrator over the multi-set state vector (AMR-1) and
  calls each package's functions in the order the data flows. Neither
  TreeGH nor this package drives a coupled run; each keeps its own
  `evolve!` for its own cases and tests. What every coupled run must get
  right — a regrid of every set, then the rebuild of both problems; a
  restart that is the uninterrupted run — are functions a script calls,
  not a protocol the packages implement.
- **The verbs are each package's own, under shared names:**
  `TreeGH.prepare!` and `TreeGRRMHD.prepare!`, then `rhs!`, the limiters,
  and the speed or rate for `Δt`. There is no shared interface package.
  They are declared `public`, not exported, so that loading both packages
  does not make the names ambiguous (proposed). `prepare!` makes a
  package's state readable at `(u, t)` — scatter, ghost fill, recovery —
  and returns what the other package reads, so the order of the calls is
  the data flow, written in the script.
- **Data passes through pointwise functions**, called at an owned index of
  a field set. The provider decides whether to compute per call or from a
  precomputed array.
- **Both directions cross at TreeGH's vertices, in 3+1 variables:**
  - the metric as `(α, β^i, γ_ij, K_ij)`, 16 values a vertex, from TreeGH's
    `ADMMetric` (GH-1);
  - the matter as `(ρ, S_i, S_ij)`, 10 values a vertex, from this package's
    matter adapter (GH-3): `ρ = n^a n^b T_ab`, `S_i = −γ_i^a n^b T_ab`,
    `S_ij = γ_i^a γ_j^b T_ab` — this package's `𝓔`, `S_i` and `S^ij`
    lowered (`src/stress_energy.jl`), covariant and not densitized, in
    `G = c = 1` and the metric's length unit.
- **Every conversion between centerings is this package's.** It is the
  only one of the two with mixed centerings. It interpolates the vertex
  metric to its cells, faces and edges, differentiates its own interpolant
  for `∂α`, `∂β` and `∂γ`, and averages `(ρ, S_i, S_ij)` from its cells to
  the vertices. TreeGH interpolates nothing, fills none of this package's
  sets, and knows neither its centerings nor its ghost widths ([Metric at
  cells, faces and edges](#metric-at-cells-faces-and-edges), [Matter to
  the vertices](#matter-to-the-vertices)).
- **The adapters live in the producers**, but each answers at the vertices
  only: TreeGH's `ADMMetric` at its owned vertices, this package's matter
  adapter at TreeGH's. The precedent is TreeGH's provider for
  ApparentHorizonFinder (`GHADMProvider`, which returns
  ApparentHorizonFinder's `ADMVars`). Neither package depends on the other.
- **The contract uses only what both packages already depend on**
  (proposed). A contract owned by the consumer, as ApparentHorizonFinder's
  is, works there because ApparentHorizonFinder is below TreeGH; here each
  package is the other's consumer, and the two dependencies would form a
  cycle.
  - *Where:* the consumer's vertex-centered TreeAMR `FieldSet`, on the
    producer's forest (`===`). The adapter is bound to it, `at = adm(M)`,
    then called at an **owned** index, `at(n, b)` with `1 ≤ n[d] ≤ N` as
    `map_blocks!` hands it to a kernel. An owned index is the same in every
    vertex set on the forest, so neither side's `G` enters. Not
    coordinates: those would need a tree search per point.
  - *What:* StaticArrays in SpacetimeMetrics' conventions: `α` a scalar,
    `β` an `SVector{3,T}`, `γ` and `K` symmetric `SMatrix{3,3,T}`, read
    by name (`(; α, β, γ, K) = at(n, b)`; `ADMMetric` also returns `sqrtγ`
    and `γu`); `ρ` a scalar, `S` an `SVector{3,T}`, `Sij` a symmetric
    `SMatrix{3,3,T}` (`(; ρ, S, Sij) = at(n, b)`).
  - *How:* a functor, `isbits` once adapted to the backend, and a snapshot
    of the producer's state when it was built (no `t` argument). A regrid
    replaces the field sets: rebuild and rebind after one.
  - *How well:* `α`, `β`, `γ` are TreeGH's stored state, exact to roundoff;
    `K` converges at TreeGH's order `q`, so the contract's 4th order needs
    `q ≥ 4` (TreeGH, measured 2026-11-08). The metric is valid at the owned
    vertices only; the ghosts of this package's vertex set are its own
    fill's. The interpolation's accuracy is this package's and its tests'.
- **The kernels' calls are indexed.** Every kernel calls
  `point_metric(prov, t, x, Val(C), n, b)` with its point's staggers and
  local index; [`VertexMetric`](#metric-providers) needs the staggers and
  the index exactly as they are. `metric=` still reaches the kernels alone:
  the initial data, the Dirichlet hooks and the interior keep the case's
  own provider.
- (open, step 11) Where the coupled script and its tests live, and how they
  load both packages; whether a prepared state is reused when a limiter did
  not change it (below); whether the script overlaps TreeGH's ghost fill
  with this package's recovery on a device; and a thread partition over
  several sets aligned with their owners (TreeGH's `state_partition` is
  copied from TreeAMR's tests; an upstream item when a step needs it).

#### Two scripts

(Sketches, proposed.) A run is then a script. The two below are steps 10
and 11. Step 11's is step 10's with TreeGH's lines added. That the
difference is this small is what this design claims, and step 11 tests the
claim. A name that does not exist today is proposed; its comment names the
piece of today's code it comes from — `evolve!` and its `build()`,
`mhd_rhs!`, `ohm_solve!`, `reset_atmosphere!`, `reset_step!`,
`max_light_rate`, `exact_state`, TreeGH's `fill_exact!` and `gh_rhs!`'s
fill. IMEXRungeKutta hands every callback the parameter `p`, which carries
the problems, so that a rebuilt problem reaches them; the limiters take
IMEXRungeKutta's `(u, integ, p, t)`.

Step 10, the Cowling TOV star; the metric is the analytic one for the whole
run:

```julia
using TreeAMR, TreeGRRMHD, IMEXRungeKutta
const MHD, IRK = TreeGRRMHD, IMEXRungeKutta

function f_exp!(du, u, p, t)
    (; mhd, metric) = p
    MHD.prepare!(mhd, u, t; metric)                   # the fill and the recovery
    MHD.rhs!(du, mhd, t; metric)                      # mhd_rhs!'s kernels
    return
end
solve_imp!(U, u★, γΔt, p, t) =
    MHD.stage_solve!(U, u★, γΔt, p.mhd, t; metric = p.metric)  # ohm_solve!
stage_limit!(u, _, p, t) =
    MHD.stage_limit!(u, p.mhd, t; metric = p.metric)           # reset_atmosphere!
step_limit!(u, _, p, t) =
    MHD.step_limit!(u, p.mhd, t; metric = p.metric)            # reset_step!

function main(T = Float64; t_end = 200, chunk = 10)
    star = MHD.tov_case(T; K = 100, Γ = 2, ρ_c = 1.28e-3)     # src/cases/tov.jl
    forest = Forest{T}(star.roots; N = 16, periodic = star.periodic,
                       extents = star.extents)
    MHD.tov_boxes!(forest, star)                     # the static geometric boxes
    mhd = MHD.MHDProblem(MHD.mhd_fields(T, forest), star; limiter = :ppm)  # build()
    u = statevector(MHD.evolved_sets(mhd))           # matter, A_d, Φ̃ (AMR-1)
    MHD.initial_state!(u, mhd, star, zero(T))        # exact_state
    metric = star.provider                           # Analytic(TOVMetric)
    p = (; mhd, metric)
    cfl = MHD.default_cfl(:ppm)
    t, integ = zero(T), nothing
    for t1 in T(chunk):T(chunk):T(t_end)
        dt = cfl / (3 * MHD.max_rate(mhd, u, t; metric))       # max_light_rate, D = 3
        nsteps = MHD.chunk_count(t1 - t, dt)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt = (t1 - t) / nsteps,
                         stage_limiter = stage_limit!, step_limiter = step_limit!,
                         partition = :even, alias_u0 = true, reuse = integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), mhd)
        t = t1
        println(t, "  ", MHD.central_density(mhd, u))  # the record: any call
    end
    return u
end
main()
```

Step 11, the same star in TreeGH's spacetime. TreeGH's problem and its
segment of the state vector are added; `metric_at` replaces the constant
`metric`; each callback calls TreeGH too; `Δt` is the smaller of the two;
and an adaptive run regrids every set at once:

```julia
using TreeAMR, TreeGeneralizedHarmonic, TreeGRRMHD, IMEXRungeKutta
const GH, MHD, IRK = TreeGeneralizedHarmonic, TreeGRRMHD, IMEXRungeKutta

# TreeGH's segment of the state vector, then the matter's (AMR-1).
segments(sets) = (staterange(sets, 1), first(staterange(sets, 2)):statelength(sets))

# TreeGH's metric at a state: its fill (gh_rhs!'s) and its adapter (GH-1), then
# this package's vertex set filled from it (owned vertices copied, its own
# ghost fill), which every kernel interpolates.
metric_at(p, u, t) =
    MHD.vertex_metric!(p.vm, GH.ADMMetric(p.gh, view(u, p.s[1]), t), t)

function f_exp!(du, u, p, t)
    (; gh, mhd, s) = p
    metric = metric_at(p, u, t)
    m = MHD.prepare!(mhd, view(u, s[2]), t; metric)  # recovered in TreeGH's metric
    GH.rhs!(view(du, s[1]), gh, t; matter = MHD.ADMMatter(m))         # GH-3
    MHD.rhs!(view(du, s[2]), mhd, t; metric)
    return
end
function solve_imp!(U, u★, γΔt, p, t)
    (; mhd, s) = p
    copyto!(view(U, s[1]), view(u★, s[1]))           # TreeGH has no implicit part,
    metric = metric_at(p, u★, t)                     # and its limiter never runs here
    MHD.stage_solve!(view(U, s[2]), view(u★, s[2]), γΔt, mhd, t; metric)
    return
end
function stage_limit!(u, _, p, t)
    GH.stage_limit!(view(u, p.s[1]), p.gh, t)        # TreeGH's first, so that the
    metric = metric_at(p, u, t)                      # matter reads the limited metric
    MHD.stage_limit!(view(u, p.s[2]), p.mhd, t; metric)
    return
end
function step_limit!(u, _, p, t)
    GH.step_limit!(view(u, p.s[1]), p.gh, t)
    MHD.step_limit!(view(u, p.s[2]), p.mhd, t; metric = metric_at(p, u, t))
    return
end
refine_wins(a, b) = a == Refine || b == Refine ? Refine :
                    a == b == Coarsen ? Coarsen : Keep

function main(T = Float64; t_end = 200, chunk = 10, adaptive = false)
    star = MHD.tov_case(T; K = 100, Γ = 2, ρ_c = 1.28e-3)
    forest = Forest{T}(star.roots; N = 16, periodic = star.periodic,
                       extents = star.extents)
    MHD.tov_boxes!(forest, star)
    ghcase = GH.GHCase(T, star.provider.metric; box = star.extents,
                       periodic = star.periodic)    # the static-background gauge
    gh = GH.GHProblem(forest, ghcase; q = 4)         # q ≥ 4: K at the contract's order
    mhd = MHD.MHDProblem(MHD.mhd_fields(T, forest), star; limiter = :ppm)
    vm = MHD.VertexMetricSet(T, forest; boundary = star.provider)  # Mv: G = 4, N ≥ 10
    sets = (GH.evolved_sets(gh)..., MHD.evolved_sets(mhd)...)
    u = statevector(sets)
    p = (; gh, mhd, vm, s = segments(sets))
    GH.initial_state!(view(u, p.s[1]), gh, zero(T))            # fill_exact!
    MHD.initial_state!(view(u, p.s[2]), mhd, star, zero(T))    # exact_state
    cfl_gh, cfl_mhd = T(1 // 4), MHD.default_cfl(:ppm)  # TreeGH's under SSP3: step 11
    t, integ = zero(T), nothing
    for t1 in T(chunk):T(chunk):T(t_end)
        metric = metric_at(p, u, t)
        dt = min(cfl_gh * minimum_spacing(forest) /
                     GH.max_speed(p.gh, view(u, p.s[1]), t),
                 cfl_mhd / (3 * MHD.max_rate(p.mhd, view(u, p.s[2]), t; metric)))
        nsteps = MHD.chunk_count(t1 - t, dt)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt = (t1 - t) / nsteps,
                         stage_limiter = stage_limit!, step_limiter = step_limit!,
                         partition = :even, alias_u0 = true, reuse = integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
        t = t1
        println(t, "  ", MHD.central_density(p.mhd, view(u, p.s[2]), t), "  ",
                GH.constraint_norms(p.gh, view(u, p.s[1]), t))
        adaptive || continue                         # the star's boxes are static
        # One regrid of every set, refined where either package asks and
        # coarsened only where both allow; then both problems rebuilt.
        metric = metric_at(p, u, t)
        flags = refine_wins.(GH.flags(p.gh, t),
                             MHD.flags(p.mhd, view(u, p.s[2]), t; metric))
        scatter!(sets, u)
        if regrid!(forest, (GH.regrid_pairs(p.gh)..., MHD.regrid_pairs(p.mhd, t)...,
                            MHD.regrid_pairs(p.vm)...); flags)
            p = (; gh = GH.rebuild(p.gh, t), mhd = MHD.rebuild(p.mhd),
                 vm = MHD.rebuild(p.vm), s = segments(sets))
            u = statevector(sets)
            gather!(u, sets)
            integ = nothing                          # its scratch has the old length;
                                                     # Mv is NaN until metric_at refills it
        end
    end
    return u
end
main()
```

**As built:** the step-10 sketch is `bin/tov_cowling.jl`, and
`test/tov_tests.jl` runs its `main` against `evolve!` bit for bit. Three
departures, each so that it is `evolve!`'s run: the chunks as `evolve!`
counts them (`t1 = c · chunk`, the last `t_end`; a float range rounds its
points otherwise), the step count as `chunk_count((t1 − t)·3·rate, cfl)`,
and `IRK = TreeGRRMHD.IRK` (a test environment cannot load IMEXRungeKutta
by name). `central_density(mhd, u, t)` takes the time of its recovery's
fill. The command line and the record are an included file's
(`bin/tov_cowling_cli.jl`), so that `main` is the run alone.

What the sketches leave visible, on purpose:
- **The order is the data flow.** `MHD.prepare!` cannot run before
  `metric_at`, nor `GH.rhs!` before `MHD.prepare!`, because each needs the
  other's result. TreeGH's limiter runs before the matter's in both
  limiters, so that the matter reads the limited metric. TreeGH's limiter
  never runs inside `solve_imp!`, whose implicit tendency
  IMEXRungeKutta recovers as `(U − u★)/(γΔt)`.
- **The repeated fill** (open). Written plainly, the script fills TreeGH's
  ghosts eleven times per SSP3(4,3,3) step: three right-hand sides, four
  stage solves, three stage limiters and one step limiter. One fill per
  distinct state, four, would do. Alone under RK4, TreeGH fills four times.
  From TreeGH's H200 measurements, a fill of its 20 variables at `G = 3` is
  0.4–1.0 ns per point, so the seven extra cost about 3–7 ns per point and
  step, against TreeGH's 20 (estimated, not measured). The stage limiter's
  fill and the next right-hand side's see the same state, and so do the step
  limiter's and the next step's first stage solve. Whether the script skips
  a fill when a limiter reports that it wrote nothing, or `prepare!` does,
  is step 11's to decide. Each `metric_at` also fills this package's vertex
  metric set (`vertex_metric!`), 16 variables at `G = 4`, about the size of
  TreeGH's own fill (1.03× at `N = 16`, TreeGH's estimate), so the same
  skip saves twice as much. `K` is defined in one TreeGH ghost layer only,
  so this package cannot read TreeGH's ghosts in place of its own fill.
- **The regrid is the script's.** One `regrid!` covers every set on the
  forest, since a block refines for all of its sets at once; the flags
  combine as `refine_wins`; and each package rebuilds its own problem.
  The TOV star's boxes are static, so step 11 exercises this branch in a
  test of its own.

### Metric providers

The kernels dispatch on a metric provider (proposed):

- `Flat()` — Minkowski, all `D`, no storage.
- `Analytic(m::SpacetimeMetrics.AbstractMetric)` — evaluated at the point
  (cell, face, edge or vertex), all `D`; the Kerr-Schild tests, and the
  Cowling TOV star as `Analytic(TOVMetric)`.
- `VertexMetric` (proposed), a producer's metric in [the coupling
  contract](#the-coupling-contract): this package's own 16-variable vertex
  set `Mv`, its owned vertices copied from a producer's ADM adapter —
  TreeGH's `ADMMetric` (GH-1, step 11), or in tests and studies
  `SampledADM`, a SpacetimeMetrics metric in the same form, which tests
  this package's side of the contract — its ghosts filled by TreeAMR, and
  interpolated here to every other centering ([Metric at cells, faces and
  edges](#metric-at-cells-faces-and-edges)). The 3+1 variables are decoded
  on TreeGH's side.

`Flat` and `Analytic` are in `src/metric.jl`; every provider answers
`point_metric(provider, t, x) -> Metric3` and
`metric_gradient(provider, t, x) -> MetricGradient`. `MetricGradient` holds
`dg[j, a, b] = ∂_j g_ab` — the spatial derivative index **first**, as in
GHSO2, then the spacetime indices with 1 = `t` — and `K_ij`; the lapse and
shift derivatives are derived from it (`d_lapse`, `d_shift`, which agree
with ForwardDiff of the 3+1 split to 2.4e−16 of their terms). The
conversion from SpacetimeMetrics' derivative-last `dmetric` is written in
`src/metric.jl` and nowhere else (proposed). `K_ij` is computed there from
the same `dmetric` call: `metric_gradient(::Analytic, …)` calls
SpacetimeMetrics' allocation-free `ExtrinsicCurvature(g, dg)` (SM-1,
SpacetimeMetrics 1.7.0) on the same `dmetric` result, and the pointwise
layer is bit-identical in a CPU kernel.

**The `Analytic` provider is evaluated pointwise, where each kernel needs
it**, and nothing is cached. Every kernel calls
`point_metric(provider, t, x)` at its own point — the recovery and the ghost
cells' `P` at cell centres, the fluxes at faces, the EMFs at edges, the
gauge potential `αΦ − β·A` at vertices and the gauge flux at edges, the stage
solve and the reset at cells — and `metric_gradient` at the cell for the
sources (the cell's `Metric3` is `P`'s). Per owned cell and right-hand side
that is about 20 evaluations of `point_metric` (1 cell, 3 faces, 3 edges for
the EMF, 6 vertex evaluations in `∂ₜA`'s gradient — each vertex twice per
direction — and 7 in the gauge equation's divergence) plus the ghost cells'
and one `metric_gradient`; per SSP3(4,3,3) step, three right-hand sides,
four stage solves and four resets, about 70 and 3. Measured
(`bin/metric_cost.jl`, Symmetry cn079, AMD EPYC 7532, one pinned thread,
Julia 1.13, 2026-11-02):
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
- per call, `Analytic(TOVMetric)` of the Cowling star over `[−12, 12]³`
  (the table inside, the closed-form exterior outside): `point_metric`
  63 ns and `metric_gradient` 115 ns, against Kerr-Schild's 87 ns and
  241 ns in the same job (`bin/metric_cost.jl percall`, Symmetry cn094,
  AMD EPYC 7543, one thread, two runs within 0.5 ns, job 592037; the EPYC
  7532 above measured Kerr-Schild at 102 and 234). The TOV metric is
  cheaper than Kerr-Schild per call: a Hermite cubic in `u = r̄²` and its
  derivative, through ForwardDiff for the gradient.

So the pointwise provider is affordable for rung 3 — the study to `t = 50M`
runs in hours, not days — and a per-stage cache would save at most that
share, less what it costs to store and load the metric at four centerings
(cells, three faces, three edges, vertices: 8 points and about 100 floats
per cell). **Analytic providers are pointwise, with no cache; the
background is stationary, but the analytic cases are affordable.**

*Deduplication (not done; measured why):* the vertex and edge evaluations
of `∂ₜA` and `∂ₜΦ̃` come to 13 per cell where 4 distinct points (a vertex,
three edges) exist — but **across** kernel invocations, not within one:
`potential_rhs_kernel!` evaluates the two distinct vertices of its own
edge, and `gauge_rhs_kernel!` the six distinct edges around its vertex and
the vertex itself, each once. So there is nothing to reuse inside an
invocation. Sharing across invocations needs either storage (the metric, or
`ψ = αΦ − β·A`, at every vertex and edge) or restructured kernels (one
invocation per vertex writing its three edges' `∂ₜA`, which still evaluates
`ψ` at the three far vertices: 13 → 10). Both change what the kernels
compute in which order. It is left, and folds into the caching decision
below. Its ceiling is about 9 `point_metric` evaluations per cell and
right-hand side, 9 × 102 ns × 3 ≈ 2.8 µs per cell and step, 4–6% of the
rung-3 cases' 49–72 µs.

**Caching is this package's decision.** The production metric crosses at
TreeGH's owned vertices only, and this package stores it there, in its own
vertex set `Mv`; what it does from there is its own choice. The measured
inputs: the analytic provider's 44% of a step on the 16³ blast; about 100
floats per cell for every centering (cells, three faces, three edges,
vertices); a block of 8³ with `G = 3` storing 5.4 times the cells it owns.
`Analytic(TOVMetric)` is pointwise, like every analytic provider.

The matter code is generic in `D ∈ {1, 2, 3}` with three-component fields
and translation invariance along missing directions, so that shock tubes
are one-dimensional runs; coupling to TreeGH is `D = 3` only (proposed).

### Metric at cells, faces and edges

The conversion is this package's ([The coupling
contract](#the-coupling-contract)).

**The vertex set** (proposed). `Mv` holds `(α, β^i, γ_ij, K_ij)`, 16
variables, vertex-centered, on the matter's forest:
- **Owned vertices** are copied from the producer's ADM adapter, bound to
  `Mv` (`vertex_metric!(vm, adm, t)`, one kernel over the owned range); no
  ghost is read from the producer.
- **`G = 4`**, so that a 4th-order interpolation reaches the outermost
  ghost cell of a `G = 3` cell set: it reaches one vertex beyond the cell
  on either side. TreeAMR's vertex invariant `N ≥ 2G + 2` then needs
  **`N ≥ 10`** for every coupled run; the TOV runs use `N = 16`.
- **The ghosts are TreeAMR's fill** (`fill_ghosts!`): copies at same-level
  faces, a PointValue prolongation at refinement boundaries. The
  derivatives taken from it must stay 4th order next to a coarse-fine face;
  TreeGH's interface rule (`p = q + 2` for second derivatives) gives `p ≥
  q + 1` for first, so **width 6** (PointValue widths are even; proposed).
- **The outer boundary** (the shared plane `n = N + 1` and the ghosts
  beyond it) is this package's hook: Dirichlet from the case's own analytic
  provider where it has one, as the TOV star's static exterior (proposed);
  what a case without one does is (open).
- A snapshot: valid until the producer's state changes; a regrid replaces
  `Mv` with the other sets, and the script refills it from the rebuilt
  adapter.

**Interpolated within a block** from `Mv` (proposed):
- **Cells**: 4th-order tensor Lagrange, 64 points; `∂ᵢα`, `∂ᵢβ`, `∂ᵢγ` at
  cells by the staggered 4th-order derivative along `i` times the
  transverse interpolation, the derivative of the same interpolant; `K_ij`
  interpolated, not differentiated.
- **Faces**: 2D, 16 points. **Edges**: 1D, 4 points. **Vertices**: read.
- **The fallback** (proposed). Where the Lagrange result has `α ≤ 0`, or a
  `γ` that fails a Cholesky factorization, the point falls back to
  multilinear weights (8, 4 or 2 vertices), counted by `metric_fallbacks`
  (below). Multilinear weights make `(α, γ_ij)` a convex combination of the
  corners, so `α > 0` and `γ` positive definite survive, and an
  `(α, β, γ)` with those properties always rebuilds a Lorentzian metric;
  interpolating `g_ab` could not promise that. At a fallback cell the
  derivatives are the multilinear interpolant's too, 2nd order (proposed).

Stored once per right-hand side in `P` for cells; on the fly at faces and
edges. The derivatives and `K_ij` are not stored but evaluated where the
sources are (proposed; see "Field sets"). `Mv`'s `G = 4` settles the
outermost ghost layer; TreeGH's own ghost width does not enter (GH-5 stays
withdrawn).

**As built in step 10, round 2** (`src/vertex_metric.jl`):
- **`Mv` is not in `MHDFields`.** It lives in its own `VertexMetricSet(T,
  forest; boundary)`, with its schedule, per-block geometry and outer
  hook, so that a run without a producer's metric allocates nothing and
  `MHDProblem` is unchanged bit for bit. `boundary` has no default: the
  case's coordinate provider (`Analytic(m)`, `Flat()`), or `nothing` on a
  periodic forest. `regrid_pairs(vm)` gives `Mv => nothing` and
  `rebuild(vm)` the new mesh's schedule, `Mv` `NaN` until the next fill.
  The sketch in [Two scripts](#two-scripts) carries it as `p.vm`.
- **`vertex_metric!(vm, adm, t)`** takes the fill's time `t`, for the outer
  hook (`CellBoundary(AllVariables(…))` of the boundary provider's `α`,
  `β`, `γ` and `K` at the ghost vertex), and returns `VertexMetric`.
- **The derivative layout:** `MetricGradient` is unchanged;
  `gradient_from_adm` assembles `∂_j g_ab` from `∂α`, `∂β`, `∂γ` by the
  product rule, and `d_lapse`, `d_shift` recover them to rounding. So no
  kernel changed, and the `Flat` and `Analytic` runs are the same bit for
  bit.
- **The fallback is counted by `metric_fallbacks(prov, f)`**, a diagnostic
  pass over every stored point of `P`, `B[d]`, `A[j]` and `Φ̃`, not by a
  tally in `X`: the provider is a pure function of the point, and a face,
  edge or vertex has no cell of its own (proposed).
- **`SampledADM(metric, vm, t)`**, a SpacetimeMetrics metric in
  `ADMMetric`'s form (exact `K`), is in `src/`, not only in the tests: the
  study (`bin/tov_study.jl metric=vertex`) and step 11 (GH-1's reference)
  use it.
- The indexed calls stay ([The coupling contract](#the-coupling-contract)).

**Measured** (2026-11-08, `test/vertex_metric_tests.jl`, every stored point
of every centering, ghosts included, the first root refined; `N = 12 → 24`):
- **Kerr-Schild at `a = M/2`** on `[2, 6]³`: rates 4.52 (cells), 4.57
  (faces), 4.72 (edges), 4.59 (`∂_j g_ab`), 4.68 (`K`), 5.87 (vertices,
  where only prolongated ghosts differ); the owned vertices are `Analytic`'s
  metric bit for bit; no fallback.
- **The TOV star** (`[−16, 16]³`, `[−16, 0]³` refined), by region with the
  surface layer `|r − R̄| < 4`: `∂_j g_ab` at 3.71 inside, 3.95 outside,
  **1.67 in the layer**, where the density's kink (`ρ ∝ R̄ − r`) makes the
  metric's third derivative jump; the values at 3.68–3.83 off the layer,
  3.25–3.58 in it; no fallback.
- **The Cowling star, four steps**, through the vertex metric against
  `Analytic(TOVMetric)`: the matter's difference is 1.21e−3 of the four
  steps' change at `h = 0.8` and 4.76e−5 at `0.4`, rate 4.67.
- **The Cowling star to 200 M** (2026-11-09, `bin/tov_study.jl
  metric=analytic|vertex`, PPM, `h = 0.5`, 0.375, 0.3, `Mv` filled once at
  `t = 0`; Symmetry jobs 593624–593629 on `amdpreq`, `bin/tov_compare.jl`
  and a localisation by radius and variable in job 593802): **no fallback**
  at any point, and the rest mass net of the reset's injection holds to
  `1.4–8.0·10⁻¹⁵` through either metric.

  | difference, analytic − vertex | `h = 0.5` | 0.375 | 0.3 | rates |
  |---|---|---|---|---|
  | max over `t` of `\|Δρ_c\|/ρ_c(0)` | 4.24e−5 | 1.37e−5 | 5.64e−6 | **3.94, 3.96** |
  | L1 of `ΔD`, inside `r < R̄ − 4` | 1.20e−5 | 4.19e−6 | 1.81e−6 | 3.66, 3.75 |
  | L1 of `ΔD`, layer `\|r − R̄\| < 4` | 1.55e−5 | 5.17e−6 | 2.31e−6 | 3.82, 3.61 |
  | L1 of `Δτ`, inside / layer | 2.51e−6 / 4.94e−7 | 8.36e−7 / 1.62e−7 | 3.62e−7 / 7.17e−8 | 3.82, 3.75 / 3.88, 3.65 |
  | L1 of `ΔS_i`, inside / layer | 7.14e−7 / 2.10e−6 | 1.99e−7 / 9.04e−7 | 1.07e−7 / 3.79e−7 | 4.44, 2.80 / 2.93, 3.90 |
  | L1 of `Δ(Ds)`, inside / layer | 5.98e−5 / 8.46e−5 | 2.08e−5 / 3.21e−5 | 9.03e−6 / 2.04e−5 | 3.68, 3.73 / **3.37, 2.03** |
  | max `\|ΔD\|` (at the centre) | 2.13e−7 | 5.86e−8 | 2.62e−8 | 4.48, 3.62 |
  | max `\|Δ(Ds)\|` | 9.86e−7 (centre) | 1.20e−6 (`r = 9.24`) | 1.62e−6 (`r = 9.12`) | none |

  The prediction holds for the star itself: the central density's
  difference converges at 3.9–4.0 and every conserved variable's L1 inside
  and in the layer at 2.8–4.4, about 3.7 on the whole, below 4 where the
  layer's `∂g` (1.67 above) feeds in. What does not converge is the
  advected `√γ D s` just outside the surface, and it measures nothing the
  scheme uses: the largest `|Δ(Ds)|` sits at `r ≈ 9.1–9.2` (`D ≈ 4·10⁻⁹`)
  and grows, 1.2e−6 → 1.6e−6, and the layer's L1 of `Δ(Ds)` falls at 2.03
  on the finest pair; the whole state's L1 relative to its own
  (`bin/tov_compare.jl`'s number, dominated by `Ds`) is 1.99e−5, 7.19e−6,
  3.79e−6, rates 3.54, 2.87. Localised (2026-11-09, job 593852 and its
  rerun, each state recovered in its own metric): the ratio
  `s★ = (√γ D s)/(√γ D)` leaves the window `[3, 7]` in the thin matter at
  the surface, `−56 … 8.2` at `8 ≤ r < 8.5` and `−410 … 1220` at
  `9 ≤ r < 9.5` (`h = 0.3`, both runs), while the recovered `s` stays in
  `[3, 7]` and its range and `T`'s agree between the runs in every shell to
  the digits printed; in the worst cell `s★` is 396 against −48, the
  recovered `s` 6.653 against 6.650, `T` 1.215e−6 against 1.212e−6. `D s`
  is only the recovery's seed (`state_guesses`), and the reset rewrites it
  only where the policy repaired the cell, not "wherever the energy
  inversion succeeds" as "Matter" proposes; where `D` is `10⁻⁹` beside
  neighbours at `10⁻⁴`, the flux differences of `D` and `D s` do not cancel
  and the ratio wanders. Outside the layer the difference is `≤ 5·10⁻¹²`.
  The potentials and `Φ̃` are zero
  in both runs (the star is unmagnetized).
- **Cost:** through the vertex metric the star's step costs **2.0–2.5
  times less** than through `Analytic(TOVMetric)`: 4.39 against 11.1 µs per
  cell and step at `h = 0.5` (EPYC 7543, both), 3.70 against 9.38 at 0.375
  (EPYC 7532, both), 3.64 against 7.21 at 0.3 (EPYC 7543, both); each pair
  on two nodes of one model, which have measured 20% apart. The TOV metric's
  Hermite evaluation (63/115 ns per call, "Metric providers") is dearer than
  reading and interpolating `Mv`.

### Matter to the vertices

`(ρ, S_i, S_ij)` are computed at the cells from the primitives, the fields
and the cell's own metric — `stress_energy`'s `𝓔`, `S_i` and `S^ij`, the
last lowered with `γ_ij` — then averaged to each owned vertex from its
`2^D` neighbouring cells: second order, and the mean is a convex
combination, so `ρ ≥ 0` holds at the vertex exactly where a higher-order
interpolation would ring at a stellar surface. TreeGH assembles `T_ab` from
them with the vertex's own `α` and `β`, the metric it evolves (the earlier
alternative: [HISTORY.md, "Matter to the
vertices"](HISTORY.md#matter-to-the-vertices)). A 4th-order option is kept,
and which one wins is (open).

This is this package's matter adapter (`TreeGRRMHD.ADMMatter`, proposed),
which TreeGH binds to its vertex set and calls at its owned vertices
(GH-3). Whether it stores `(ρ, S_i, S_ij)` at the cells during `prepare!`
(10 values a cell, the `Tsrc` row of "Field sets") and averages per call,
or computes everything per call, is this package's choice, made in step 11
(open). Estimated 2026-11-05 for `T_ab`, the stored form reads about 80
values per vertex; per call, each of the `2^D` cells needs its metric
again, a 64-point interpolation from `Mv` (about 5000 reads).

### Gauge for the TOV milestones

TreeGH's static-background path, with no gauge driver (proposed): the TOV
star is a static, `isbits` `SpacetimeMetrics.AbstractMetric` in isotropic
coordinates, generic over dual numbers so TreeGH can differentiate it
inside kernels. `H_a = −Γ_a[g_TOV]` is sampled and frozen, as TreeGH
already does for static backgrounds, and the outer boundary is Dirichlet
from the static exterior. A fixed gauge source is a legitimate gauge for a
perturbed or oscillating star; it is not for a collapse, which needs the
gauge driver (GH-6). The same `TOVMetric` is this package's provider for
the Cowling star, `Analytic(TOVMetric)`, with no TreeGH, and the frozen
TreeGH metric of the same background is checked against that run in step
11.

**As built** (proposed): `src/tov.jl`. **The solve** is Lindblom's (1992)
enthalpy form: RK4 in `η = log h` from the centre to the surface, `η = 0`,
a grid point rather than a root to find, for `q = r²`, `μ = m/r³` and
`λ = log(r̄/r)`, all smooth in `η` at the centre (where
`r ∝ √(η_c − η)`); `λ` is then matched to the isotropic Schwarzschild
exterior, and the lapse inside is `α_s/h` (`h α` is constant in a static
star). Measured: `M = 1.4001597295`, areal `R = 9.5856240172`, isotropic
`R̄ = 8.1251439016` (the literature's 1.400, 9.586, 8.125), the mass
converging at 4.2–4.4 under step halving, `3.8·10⁻¹⁴` at the default 4000
steps. **The metric**, `TOVMetric{T,NK}`, tabulates `ψ` and `α` in `u = r̄²`
— in which both are smooth at the centre, and which needs no square root
there, where a symmetric mesh has a vertex and ForwardDiff's derivative of
`√u` is not finite — with their `u`-derivatives at `NK` uniform knots over
`[0, R̄²]`, cubic Hermite inside, the closed-form exterior outside: `isbits`
`SVector`s, 4176 bytes at the default `NK = 129` (kernel parameter space,
step 14's), dual-generic. Measured: the table's error 2.8e−8, 1.8e−9,
1.2e−10 at `NK = 33, 65, 129` (fourth order); the field equations
`∇²ψ = −2πψ⁵e` and `∇²(αψ) = 2παψ⁵(e + 6p)` hold to 2.1e−5 of `2πρ_c` at
`NK = 129` (second order in the knots: the interpolant's second derivative;
the scheme reads only first derivatives), and to 2.7e−16 outside;
SpacetimeMetrics' `EinsteinTensor` agrees with `8πT_ab` to 1.8e−6 of
`8πρ_c`. The density is not tabulated: `ρ(h)` at `h = α_s/α`, the
polytrope's own relation, so the initial data and the metric are one
solution. **The case**, `src/cases/tov.jl`: the gas's entropy window is
centred on the star's entropy, so that the atmosphere `recovery_options`
writes is on the star's isentrope; the exterior is that atmosphere at rest;
`IdealConductor` with `A = Φ̃ = 0` and `static_potential = true`; the box
`[−24, 24]³` on four roots, whose first box level (`tov_boxes!`) covers
`[−12, 12]³`.

## The right-hand side

In the order the explicit part of each stage evaluates it (proposed):

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
   ("Refinement").
4. `q = ∇·E` at cells.
5. Nothing of TreeGH's. In a coupled run the script calls TreeGH's
   right-hand side, which reads `(ρ, S_i, S_ij)` through this package's
   adapter ([The coupling contract](#the-coupling-contract)).
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
and a step is pure in `uⁿ`.

*As built* (`src/evolution.jl`, `mhd_rhs!`; the kernels are written for
`D = 1, 2, 3`):

- Step 3 is three passes: `primitive_kernel!` over the owned cells (the
  recovery, from the provider's metric at the cell, `B` from the faces,
  `E = 𝔈/√γ` and `state_guesses` (see "Thermodynamics"); `W`; and all
  thirty slots of `P` through `P_from_W`), the fill of `W`, and
  `ghost_kernel!` over every stored cell that is not owned (`P` through the
  same `P_from_W` with the ghost's `B`). Owned cells take `P` through
  `P_from_W` too, rather than straight from the recovery (proposed): where a
  ghost is a copy of its owner — every ghost on a uniform mesh — the two
  blocks then hold the same `P` bit for bit and compute the same flux at
  their common face, as they did when both recovered the same conserved
  state; the two differ by rounding only (`v = u/W` from `u = cosh(w) v`,
  the rapidity `asinh |u|`, one forward EOS call). `q` is a second kernel
  over the owned cells, by centered differences,
  `q = (1/√γ) Σ_d (𝔈ᵈ[n+e_d] − 𝔈ᵈ[n−e_d])/2h` (proposed); `q` is taken
  from `P`'s `√γ E` (the same numbers in flat space).
- Step 7 for `A_j` is `−EMF_j − (ψ[n + e_j] − ψ[n])/h` with
  `ψ = αΦ − βᵏA_k` evaluated at the two vertices that bound the edge, `A_k`
  averaged to a vertex from the two `k`-edges that meet there; along a
  missing `j`, `−EMF_j` alone. For `Φ̃` it is the difference of the gauge
  flux `α√γ Aʲ − βʲΦ̃` on the two `j`-edges either side of the vertex, with
  `A_k`, `k ≠ j`, averaged to the `j`-edge (two or four points) and `Φ̃`
  from its two vertices, plus `−ξαΦ̃` (proposed: second-order averages; in
  flat space `ψ = Φ̃` and the flux is `A_j`, and nothing is averaged that
  enters).
- The implicit stage (`ohm_solve!`) computes `B` from `u★`'s `A` — a scatter
  and ghost fill of the three `A` sets and the curl — and calls
  `solve_ohm_stage` in every owned cell with that `B`, the provider's
  metric and `state_guesses(u★)`, writing `√γ E` into `𝔈` and nothing else
  and tallying the outcome into `X`. Inside a step every site's first
  recovery is seeded from the step cache where the cell has an entry, and
  reused where its input is the entry's.
- In 2D and 3D, a run of either wave that is translation-invariant in `y`
  (and `z`) is the 1D run bit for bit, under every reconstruction, with the
  initial data's Gauss average in difference form (see the driver in "The
  partition"; "Validation ladder" 2a).

The cost of a right-hand side, an implicit stage and a reset as first
built, and the first 2D and 3D check, are in
[HISTORY.md, "The right-hand side"](HISTORY.md#the-right-hand-side).

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

*Measured* (`test/refinement_tests.jl`; MC, HLLE, SSP3(4,3,3), static
two-level meshes and adaptive ones; with the potentials' conservative
transfer and the matter's ghost fill in primitives; the first measurements
are in [HISTORY.md, "Conservation at coarse-fine
faces"](HISTORY.md#conservation-at-coarse-fine-faces)):
- **The totals.** The Alfvén wave of rung 1a on 64 cells with the middle
  half refined, a quarter period, drifts 0.013 ulp of a total's scale a
  step with the flux fixup and 1.8·10¹⁰ without; along the diagonal on
  2 × 2 roots of 8² with one refined, 0.12 and 5.4·10¹¹. Balsara 1 on its
  box made periodic, tracked by two levels more through ten regrids, 0.015
  and 2.2·10⁷; the rotor, adaptive, 0.037 through two regrids.
- **The magnetic flux, as a rate.** With `(d, a, b)` cyclic, `∂ₜ𝔅ᵈ =
  −(∂_a EMF_b − ∂_b EMF_a)` (the gauge terms' curl vanishes), so the rate at
  which the flux through a face changes is `−∮ EMF · dl` around it. On every
  face of every coarse block that lies on a coarse-fine face, the rate from
  that block's EMFs equals the sum of the rates through the fine faces that
  cover it to 8.7e−19 of a largest rate of 5.4e−3 (2D, Orszag–Tang's
  initial data, 16 faces) and of 5.5e−3 (3D, the spherical blast in its
  closed box, 192 faces) with the correction — rounding: the fine sum
  telescopes in floating point — and 0.025 of 0.029 and 0.0017 of 0.0055
  without it (proposed: the claim of PLAN.md's step 8, "the flux through a
  coarse face equals the sum through its fine faces", made as the rate it
  is made by).
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
  coarse-fine face (proposed: accepted, not corrected).
- **The magnetic flux, as a state** — not to rounding under the PointValue
  operators. In 2D the flux through a face on the coarse-fine face is a
  difference of `A_z` at two vertices the levels share, which the exchange
  copies, so the two sides agree exactly. But the flux through the plane,
  `Σ hᴰ 𝔅ᶻ = ∮ A · dl` around the periodic box, telescopes across a
  coarse-fine face only if the coarse `A_y` on it is the mean of the fine
  ones beside it, and the PointValue operators interpolate along an edge's
  own direction where a conservative transfer would average: on the oblique
  Alfvén wave it drifted 5.8e−4 of its scale over an eighth of a period
  (5.1e−17 on the uniform mesh), with the EMF correction or without, 1.4e−3
  at width 2 — the operators', not the EMFs'. In 3D the flux through a face
  on a coarse-fine face is the circulation of `A_y`, `A_z` in its plane and
  inherited the same interpolation. The potentials' conservative transfer
  is **AMR-9**, in TreeAMR 0.2.2.

  With the potentials' conservative transfer (AMR-9;
  `test/refinement_tests.jl`), **the flux holds as a state, to rounding,
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
  negative control is the PointValue family (proposed).

  **Not at an outer face** *(proposed: accepted, recorded)*. In the same
  blast's closed box the 147 faces away from the walls agree to rounding,
  but the 45 whose perimeter lies on an outer face miss by 1.2e−5 of the
  largest flux (2.7e−5 after the regrid): there each level's relative hook
  writes its own wall edges, the nearest evolved point of that level plus
  the case's difference, so a coarse wall edge is not the sum of the fine
  ones beside it. The flux through a face at a Dirichlet boundary is the
  boundary condition's, not conserved by anything, so nothing in this
  package relies on it; a relative hook that fills a point on both an outer
  face and a coarse-fine face from the level that owns it (a transfer, not
  the hook) would make it exact — **AMR-11** in `PLAN.md` (wanted, no step
  blocked).

## Boundaries

Dirichlet from the initial or analytic state, through
`CellBoundary(AllVariables(…))` on every set, for everything up to the
magnetized TOV star (proposed): shock tubes until a wave arrives, the
Kerr-Schild tests from the analytic solution, the stars from the static
exterior (atmosphere, the initial `A` and `Φ̃ = 0`, `E` of the static
dipole). TreeAMR 0.2.1 has outflow faces (AMR-6), for whichever later rung
needs them. `MHDProblem` takes `boundary = nothing` for a periodic domain
(and refuses a non-periodic forest without hooks) or one hook per evolved
set, in the order `(U, A₁, A₂, A₃, Φ̃)` — each set's `fill_ghosts!` gets
its own (AMR-2). The waves are periodic; the shock tubes pass hooks.

**Inconsistent:** this paragraph has `CellBoundary(AllVariables(…))` on
every set, but the potentials' hook below is TreeAMR's relative hook,
`CellBoundary(Relative(f))`, for every case but those with static
potentials.

*As built* (`DirichletBoundary`, `case_boundary` in `src/driver.jl`;
proposed). A case that is not periodic gets one hook per evolved set from
its own `fields` and `potential`, evaluated at the time of each fill
(`boundary_at(hook, t)`: every fill — a right-hand side's, a stage's, a
reset's — knows its time):
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
  cell every step; with the field's Dirichlet data it stays uniform to
  3.3e−13 under either tableau (a test). The hook is TreeAMR's relative
  hook (AMR-7), `CellBoundary(Relative(f))` with `f` the case's potential at
  the fill's time (`boundary_at`): a kernel, so **every case, periodic or
  not, runs on every backend** (not exercised on a device here: the machine
  has none this package targets; step 14's).

**Both walls are prescribed.** At every outer face of a case with Dirichlet
hooks, the owned low wall of each potential (`A_j`, `Φ̃`; a vertex-like
dimension's first owned point `G+1`) is set by the mirror of the hook's
high-wall rule — the nearest evolved point (`G+2`) plus the case's
difference between the two points, at the time of the fill. And every point
beyond an outer face takes its nearest evolved point across a block face
from the block's own (filled) ghost along the dimensions it does not leave
the domain through: TreeAMR's region form clamped along every outward
direction, so a corner row where a wall meets another block came from the
block's own last column and the point two blocks share held a different
value in each (the cell next to it had a block-dependent `𝔅`, and the two
blocks computed different fluxes at their common face next to the wall).
Both are TreeAMR's:
- the potentials' field sets are built with **`walls = :symmetric`** (AMR-8),
  under which the fill writes the owned low wall of a vertex-like dimension
  at a low domain face by the high wall's rule;
- their hook is the **relative hook** (AMR-7), which reads `x_r` across a
  block face from the block's own ghost and runs each region after the
  stage that fills what it reads, so that **one fill applies the complete
  condition** — `fill_ghosts!`, `regrid!`'s fill before its transfer and
  `adapt_to_initial_data!`'s alike (an outer boundary condition needs to be
  applied always). `fill_set!` is that one fill; `regrid_state!` hands
  `regrid!` every set's own hook; the initial adaptation's criterion reads
  what the cycle filled.

Why the low wall is prescribed too — with the hook filling the high wall
alone the potentials' boundaries were not mirror-symmetric — is in
[HISTORY.md, "Boundaries"](HISTORY.md#boundaries).

Measured (`test/refinement_tests.jl`, the closed `[−2, 2]²` explosion with a
wall root refined, to `t = 1/4`): a second fill changes nothing bit for bit,
and one fill of working arrays prefilled with `NaN` writes the same — every
stored point, from the state alone; a hookless fill leaves it.

**The low walls in the state** *(proposed)*. The low wall is owned, so it is
in the integrator's state vector, but TreeAMR writes it during the fill —
into the set's working array after `scatter!`, not into `u`.
`map_blocks!`'s owned range is unchanged by symmetric walls (the trimmed
range `G+2 …` is the fill's regions' only), so the right-hand side would
compute an evolved value there and `u`'s wall entries would drift from the
condition. So:
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

*Measured* (with TreeAMR's symmetric walls and relative hook): the explosion
in a box its precursors reach — `[−2, 2]²` at 32², minmod, `t = 1/4` — and
the spherical blast in the closed 16³ box are their own mirror images bit for
bit, every evolved set on the closed range with both walls, but for the sign
of an exact zero (a parity of −1 turns `+0` into `−0`; stated with `==`, as
step 7 stated transposition). The 3D smoke test runs in the closed box
(`test/multid_cases_tests.jl`). The 2D explosion at 64² on `[−6, 6]²` and
the rotor are exactly symmetric too. The current sheet run on to `t = 9`,
whose field has diffused to the faces by then, is 9.56e−4 of `B₀` in L1 from
the erf profile, the low face's `𝔅` the profile's as the high one's is. What
these numbers were before both walls were prescribed is in
[HISTORY.md, "Boundaries"](HISTORY.md#boundaries).

**Inconsistent:** the comment of the explosion's symmetry testset in
`test/multid_cases_tests.jl` (minmod, 64² on `[−6, 6]²`, to `t = 1`) records
a measured asymmetry of 1.6e−18 and 2.5e−17 of each variable's scale, not an
exact symmetry.

On a refined mesh a closed box is **not** mirror-symmetric: the explosion in
its closed box with the middle 2 × 2 roots refined (a mesh symmetric under
both reflections, whose coarse-fine faces cut the blast's transition
region), to `t = 1/4`, is 1.0e−2 (`x → −x`) and 8.8e−3 (`y → −y`) of a
variable's scale from its mirror image in `U`, 1.1e−2 and 1.3e−2 in `A_z`
(3.2e−4 and 6.3e−3 in `A_z` after one step). That is TreeAMR's open item
(its `CODE.md`, "Centerings": a plane on a coarse-fine face is the fine
block's owned plane when the face is the fine block's low face, and the
coarse block's, interpolated at the fine points between, when it is its high
face; 4.7e−5 upstream on smooth data, one fill), here at a discontinuity and
through 20 steps. Recorded, not hidden: `test/refinement_tests.jl` asserts
it present and bounded (proposed: nothing in this package needs a refined
box to be symmetric before the stars of steps 10 and 12, whose own symmetry
tests will say whether upstream must close it).

**The potentials' boundary must pin the gauge or keep a gradient a gradient**
*(a design rule)*. A Dirichlet hook that copies the evolved potential
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

- **Static potentials** (`MHDCase(…; static_potential = true)`, option 1):
  the hook is absolute, the case's `A_j` and `Φ̃` at every ghost and on every
  prescribed wall (`DirichletBoundary(…; absolute = true)`,
  `CellBoundary(AllVariables(…))`). Under `walls = :symmetric` TreeAMR fills
  the owned low wall with the hook's own rule, absolute or relative (its
  "the hook's low region becomes `1 … G+1`"), so the walls follow; nothing
  upstream was needed. Valid only where `A_case` does not depend on `t`:
  under IMEX the stage's `A` is at `tⁿ + c̃Δt` while `solve_imp!` is told
  `tⁿ + cΔt`, which a potential that moves with `t` would get wrong (the
  potentials' hook, above). The Kerr-Schild cases declare it; the test "A
  static potential's boundary is the case's own" checks that a gauge offset
  vanishing on the boundary leaves `𝔅` on every stored face at the exact
  state's (1.1e−15) and every boundary-filled point untouched, where the
  relative hook shows it (5.9e−2).
- **Potentials that grow in time** keep the relative hook (every flat-space
  case, `static_potential = false` by default): their outputs are
  unchanged bit for bit.
- **Open** (for step 10's outer boundary, and for dynamic cases without an
  exact `A`): option 2, a relative hook that carries the offset's gradient
  along, `δ(x_r) + (x_g − x_r)·∇δ`, which keeps a gradient a gradient to second
  order (TreeAMR's hook form); option 3, a gauge that damps a static `χ` as
  well. Neither is requested yet.

## The interior

A black hole's singularity is inside the domain of the rung-3 cases, and
nothing in the scheme can evolve the neighbourhood of `r = 0` — the
Kerr-Schild metric diverges there, and Michel's density grows as `r^{−3/2}`.
The interior is TreeGH's answer: inside `r < r_in`, within the horizon, the
state is relaxed to the analytic solution each stage, **position-only**.

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
M/4`, 1500 steps, Symmetry EPYC 7532, jobs 588101–588103; under the relative
potential hook, below): the L1 drift from the exact solution outside the
horizon (`r ≥ 2M`) and over the whole octant, and what the reset did, for the
paste in both limiters at `r_in = M` (the default), in one role only, at other
radii, and without an interior:

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

The table above was measured under the relative potential hook, whose
boundary mode dominated the outside error; with option 1 (the absolute hook,
"Boundaries") the outside error is 3–10 times smaller, and the comparison
"No interior, the ordinary reset inside" below measures the interior again.

**Resets inside the hole**:
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

- **The hard paste, both limiters, `r_in = M`, for the analytic rung-3
  cases** (the measurement below). *The matter alone needs no interior*: the
  magnetized Bondi flow without one has the paste's outside error at every
  resolution, and every reset deep inside, all level 1, none to the
  atmosphere — EntropyEOS's projection absorbs the extreme states, as Erik
  expected. *A vacuum field near the singularity does need one*: without it
  Wald fails at `M/8` and `M/16`, earlier on the finer mesh — a spurious
  charge with no Ohm's law to hold `E`, whose electromagnetic energy and
  momentum the recovery subtracts from the totals, until the fluid's share
  is garbage and the reset reaches out. The step limiter alone gives the
  same run to the fourth digit; the stage paste costs one kernel launch over
  a ball of 0.4% of the box and keeps every right-hand side reading the
  solution inside. `r_in = M` keeps the horizon `M/h` cells from the paste
  (four at `h = M/4`, TreeGH's margin `m ≥ G + 1`), and `3M/2` gains
  nothing.

**No interior, the ordinary reset inside** *(option 3;
`bin/curved_localise.jl … interior=false` against the default paste from the
same tree, the absolute potential hook, both cases; jobs 588498–588503,
588592; counting radius `1.5M`)*. Outside `2M` at `t = 50M`:

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
  `1.3·10⁻³` beyond `1.5M`. Bondi's charge beyond `1.5M` stays at rounding
  without the interior (`6·10⁻¹⁵`), and over the octant `−9.5·10⁻⁵` at `M/8`
  (rounding at `M/4`): `IdealConductor` sets `E = −v × B` in every cell every
  stage, so little accumulates.
- **Injected energy**: without the interior the reset removes `10⁶`–`10⁷` of
  `τ` from the singularity's neighbourhood over the run (the entropy ceiling
  lowering `τ` where the infall heats the gas without bound), and Wald at `M/8`
  injects 44.7 of rest mass. Deep inside, not counted, but the totals carry
  it.

**Diagnostics** of the curved cases: `outside_l1`, `field_l1` (the error of `𝔅`, free of `A`'s
gauge), `charge_norms`.

## Refinement

A Löhner indicator on `ρ`, optionally on `|B|` and `|E|`, following
TreeHydro's `hydro_flags`, plus geometric radius boxes around a star that
fix the levels the star is resolved with (proposed). `regrid!` is given
every set over the forest: the evolved ones with their schedules, the rest
`=> nothing`.

TreeAMR's bounded prolongation (commit 902042a, AMR-4) was shaped by an
expectation its commit message states of this package: that TreeGRRMHD
"will leave admissibility to EntropyEOS, cell by cell, prolongate the
primitive variables in the ghost fill and the conserved ones only in the
regrid". The mesh guarantees only that a prolongated value stays within its
neighbours' range, per variable — a property that suits primitives in a
ghost fill (`ρ`, `s`, `vⁱ` each stay bounded, so a ghost is an admissible
state wherever its neighbours are) and conservation in a regrid (the
conserved variables' totals are kept). The ghost fill of the matter and the
regrid below follow that split.

*As built* (`src/refinement.jl`, `evolve!` in `src/driver.jl`):

- **The indicator** (`mhd_flags`) *(proposed)*: Löhner's, with TreeHydro's
  local and global noise floors (`ε = 1/100`, `ε_g = 1/1000`), on `ρ`, `|B|`
  and `|E|` of each cell and its neighbours along every dimension, from a
  current `P` (ghosts included) — the density for the matter's waves and the
  two field magnitudes for the electromagnetic ones, which in an
  electrovacuum move where `ρ` does not. TreeHydro's four marks
  (`(Refine, box)` above `refine_tol` below `maxlevel`, `(Keep, box)` above
  `coarsen_tol` as a travelling margin, `Coarsen` where nothing fired, else
  `Keep`), through TreeAMR's `firing_boxes`, so it runs on every backend.
  The global references are the largest `ρ` and the largest component of
  `Bⁱ` and of `Eⁱ` (a `mesh_mapreduce` reduces one variable at a time; a
  reference off by `√3` moves a noise floor). `refine_tol`, `coarsen_tol`
  and `maxlevel` have no default, and `coarsen_tol < refine_tol` is
  required. The geometric boxes around a star are step 10's.
- **The regrid** (`regrid_state!`, at the start of every chunk after the
  first) *(proposed)*: `regrid!` over every field set (`regrid_pairs`) —
  `U`, `A_j`, `Φ̃` with their schedules and each with its own Dirichlet hook
  at the regrid's time (AMR-2), `P`, `B_d`, `F_d`, `EMF_d` and `X`
  `=> nothing`, resized and never transferred (`B` is never state: the
  regrid moves `A`, and `B` is its curl). Then the problem is rebuilt over
  the same sets (`rebuild`: schedules, interface schedules, per-block
  geometry and outer faces), the state gathered, the potentials' low walls
  prescribed (AMR-8), and the reset run once as the step limiter, so that a
  prolongated conserved state the recovery would repair is repaired in the
  state and its injection counted with the step limiter's (TreeHydro's
  post-regrid reset). The integrator's scratch is dropped when the mesh
  moved. Before `regrid!`, every evolved set is filled with the complete
  outer boundary condition, and `regrid!` gets no hook for a potential, so
  its own pre-transfer fill keeps that condition bit for bit (see
  "Boundaries").
- **The initial mesh** (`adapt_case!`) *(proposed)*: TreeAMR's several-set
  `adapt_to_initial_data!` (AMR-3) over every field set, the evolved ones
  filled each pass from the case — the conserved state *at* each cell's
  centre, a point value standing in for the Gauss average while the mesh is
  chosen, and the potentials at their points — with their own hooks; the
  criterion recovers `P` from the filled sets and flags. On the converged
  mesh the exact initial data (`exact_state`, Gauss averages) is written.
- **A static mesh** *(proposed)*: `evolve!`'s `refine(forest)` refines the
  root mesh before anything is built, for the coarse-fine tests.
- **The operators**: the matter Conservative width 3, **bounded**, in its
  regrid (and its ghost fill's `W`, below); `A_j` the conservative family
  (AMR-9: width 3 along the edge) with point values of **width 4** across
  it, and `Φ̃` PointValue **width 4** — prolongation and restriction. The Alfvén wave crossing
  coarse-fine faces (`bin/interface_order.jl`; MC, the matter bounded;
  measured with `A_j` PointValue, the re-measurement under AMR-9 below the
  table):

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

  With `A_j` in the conservative family, AMR-9 — width 3 along the edge,
  the table's `w` across it — and `Φ̃` PointValue width `w`
  (`bin/interface_order.jl`): the 1D rows are the same to every digit
  printed — in 1D `A_y`, `A_z` are vertex sets, for which the two families
  are the same operators, and `A_x` is a constant — and in 2D width 2 gives
  1.49 / **0.33**, width 4 2.14 / 2.02 and width 6 2.14 / 2.01 (L1 errors at
  32² 1.71e−3 at width 4 and 6, against 1.85e−3 and 3.10e−4 uniform).
  **The width stays 4**: the conservative direction changes nothing the
  reconstruction at the face sees to three digits, and width 2 still loses
  the order across it.
- **The gauge damping**: `evolve!` uses `ξ = 16/L` on every mesh, uniform or
  refined, `L` the longest side of the box — the longest, so that a case's
  extrusions and turns, whose boxes differ only along the added or exchanged
  directions, damp alike and stay its run bit for bit (`default_ξ`); a
  caller's `ξ` overrides it. On a uniform mesh the gauge never reaches `𝔅` —
  `−∇(αΦ − β·A)` is a discrete gradient, whose discrete curl vanishes to
  rounding — so there `ξ` moves the gauge part of `A` and `Φ̃` and, through
  rounding, nothing else; what it moved in the suite is listed under
  "Validation ladder" 2a and in [HISTORY.md,
  "Refinement"](HISTORY.md#refinement). Across a coarse-fine face it does,
  since the prolongation of a coarse gradient is not a fine one. A pure
  gauge pulse (`gauge_pulse_case`: `A = 0`, `Φ̃` a Gaussian, nothing else;
  `bin/gauge_damping.jl`) on the periodic unit box, 2 × 2 roots of 16² with
  one refined, to `t = 2`, with the potentials' conservative transfer,
  AMR-9, and the matter's ghost fill in primitives, AMR-10 (the measurement
  with the PointValue potentials is in [HISTORY.md,
  "Refinement"](HISTORY.md#refinement)):

  | `ξ` | 0 | 1 | 4 | 16 | 64 |
  |---|---|---|---|---|---|
  | `max |𝔅|` over the run | 3.6e−2 | 1.6e−2 | 4.8e−3 | 3.9e−4 | 2.0e−5 |
  | `max |𝔅|` at `t = 2` | 1.6e−2 | 6.2e−3 | 3.5e−4 | 7.2e−6 | 5.7e−6 |
  | `max |Φ̃|` at `t = 2` | 0.25 | 8.4e−2 | 4.0e−3 | 8.0e−5 | 5.3e−4 |

  against at most 5.3e−15 on the uniform mesh at any `ξ`. The spurious field
  is much smaller than with the PointValue potentials, since a coarse edge's
  mean is what the fine edges along it carry and the prolongated gradient is
  closer to a fine one. Without damping the
  pulse keeps crossing the faces of the periodic box and the spurious field
  stays; the damping takes it away with the pulse. The peak at the first
  crossing is set by the interpolation, not by `ξ`. The oblique Alfvén wave
  on the same mesh moves in the fifth digit of its L1 errors at most for any
  of these `ξ`. At `ξ = 16` the remainder falls by 2200 within two light
  crossings and `Φ̃` by 3100. Beyond it the gauge mode is overdamped: `Φ̃`
  follows `−∇·A/ξ`, and the longitudinal `A` diffuses (with coefficient
  `1/ξ`) instead of leaving at the speed of light. And `ξΔt = 16Δt/L ≪ 1` at
  any resolution a box of side `L` is run at, far inside the explicit part's
  stability. In the suite's 8² version the undamped field at `t = 2` is
  0.049 and the damped one 1.6e−5.

**The ghost fill of the matter**: **the ghost fill exchanges the
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
conserved variables. The density forms of `W` tried by experiment —
`ghost_vars = :density`, `:mass` — are gone from `src/`.

In detail (`primitives_from_sets!`, `P_from_W`, `ghost_kernel!` in
`src/evolution.jl`; "The right-hand side"):
- **The set** `W`, cell-centred, 9 variables: `ρ★`, `s`, `Yₑ`, the
  contravariant `uⁱ = Wvⁱ` and the contravariant, undensitized `Eⁱ` — Erik's
  list, with `uⁱ` rather than `vⁱ` so that any interpolated value is a
  subluminal velocity, and `E` undensitized, as a primitive (proposed).
  Scratch: resized, never transferred by a regrid.
- **The recovery runs on the owned cells only**, before the exchange, and
  writes `W` and `P`; `W`'s fill has its own hook, the case's primitives at
  each ghost centre (`DirichletBoundary{:W}`, a function of position, so the
  fill applies the complete condition); at every ghost cell `P` comes from
  `W` by one forward EOS evaluation, no inversion, with the ghost's own `B`
  (and no recovery flags). Owned cells take `P` through the same function,
  so that a copied ghost is its owner bit for bit (see "The right-hand
  side").
- **Conservation does not depend on it**: at a coarse-fine face it is
  `restrict_interfaces!` on the fluxes; the regrid's bounded transfer of the
  conserved variables is exactly conservative.

**Three working fills, to revisit when the physics gets harder.** Each ran
the atmosphere tube across a coarse-fine face and the suite. The one-line
result is the 48 perturbed magnetized atmosphere tubes of
`bin/ghost_robustness.jl` (the table below; `Bʸ = 10⁻³`, cut off at
`μ_max = 10`) under EntropyEOS 1.0's policy, and in brackets under the
atmosphere-at-rest recovery with `collapse_to_atmosphere` (EntropyEOS 1.1,
2026-10-21; `bin/front_investigation.jl`, [HISTORY.md, "Floors, the atmosphere and the
magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere)):

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

**Inconsistent:** the list above calls the second fill, the parent-centred
primitives, "the code", while this sentence says the code lost the second
and the third.

*The parent-centred fill (branch `claude/step-8-ghost-athena`):*
`W` filled with TreeAMR's bounded conservative operator, `Operators(;
prolongation=3, restriction=2, family=Conservative, bounded=true)` — a
linear reconstruction centred on the parent with MC slopes, scaled to keep
every child within the parent and its face neighbours: Athena++'s ghost
prolongation of the primitives (Stone et al. 2020, §2.1.5) but for MC
against minmod. At the foot of a jump a child takes its parent's value, so
the dense neighbour no longer leaks into an atmosphere cell's children (why
a point-value fill leaks there: [HISTORY.md, "Refinement"](HISTORY.md#refinement)).
On the specific `s`, `Yₑ`: the hydrodynamic tube injects −1.6e−11 of rest
mass (the conserved fill −1.5e−11), nothing is refilled beyond the face, no
magnetized run of twelve blew up, the interface order is the conserved
fill's (1D 2.11 / 2.04, 2D 2.14 / 1.98). `bin/ghost_entropy.jl` has the
table.

*Robustness of the three fills (branch `claude/step-8-ghost-robustness`,
from `claude/step-8-ghost-athena`):* robustness on the magnetized atmosphere
tube across a coarse-fine face, which is chaotic at rounding level, over
**48 perturbed initial states** per method (`bin/ghost_robustness.jl`; runs
1–12 are the twelve of the earlier tables, runs 13–48 amplitude 10⁻¹⁵ with
seeds 2–37). A blow-up: injected `D` above 10⁻⁶, `|τ|` above 10⁻³, or
non-finite (the other runs inject 3·10⁻⁸ … 8·10⁻⁸ of rest mass). Ranges are
min / median / max over the runs that did not blow up; "p, s out" count the
ghosts whose pressure or entropy leaves the range of the owned cells within
1.5 coarse spacings, summed over the run; wall time per run with five
methods running at once on 12 cores. The conserved rows are measured on the
tree before the primitive fill (`git archive 7605986`, the same scheme
otherwise), the uniform mesh on both trees.

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
of either extreme. (The investigation of 2026-10-16, [HISTORY.md, "Floors, the
atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere): the spikes are EntropyEOS's
`S_FLOORED` repairs, the blow-ups its collapse branch, on every mesh. With a
floored state the atmosphere at rest (EntropyEOS 1.1's
`FLAG_POL_ρ_FLOORED`, in the recovery wrapper) and the tube's policy setting
`collapse_to_atmosphere`, measured, all three meshes are deterministic,
0/48, `W ≤ 5`, no fallback — the table above stays the record of the
released policy, and the ghost fill's choice is unchanged. On EntropyEOS
2.0's projection, without the option, the parent-centred fill's 48 runs are
0/48, no fallback, `W` 5.0 / 5.4 / 11.0, but two runs `10⁻¹⁵` apart differ
by `1.3·10⁻⁴` of `D` — [HISTORY.md, "Floors, the atmosphere and the
magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere), "EntropyEOS 2.0".)

The fills tried before the parent-centred one — the conserved fill's
variants, the PointValue primitive fill at widths 2, 4 and 6, and the
density forms at width 2 — and their measurements on the atmosphere tube
and on smooth data are in [HISTORY.md, "Refinement"](HISTORY.md#refinement).

## Julia version

`julia = "1.13"`. The floor is raised when the older version is more than
about 20% slower. A dependency's own `[sources]` is ignored when this
package resolves it, so what it needs from an unregistered package must be
named in this package's own `[sources]`.

Every Tree* package has the 1.13 floor, and its CI tests only 1.13, because
supporting the older versions slows development and is not useful at the
moment. `[sources]` is what locates the unregistered dependencies; 1.13
reads it as 1.11 did. Every `[compat]` entry resolves to the same releases
at 1.12.7 and 1.13.1.

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
30 s); that is over the rule's threshold, so the floor is 1.13. Without
bounds checks the gap is half that (12% and 6%), which is the case for
keeping 1.12 if the rule is meant for the default configuration. The
numbers the suite prints are identical across the change at each version,
and 1.12 differs from 1.13, at the same flags, only in the last digits of
six recorded worst cases.

CI runs three cells, all at 1.13: macOS/arm64, Linux, and Linux at four
threads. The macOS cell carries the coverage, which at 1.13 costs 1.24x
(6m27s against 5m12s, both with bounds checks).

The cells run in two tiers, the same in every Tree* repository: a pull
request runs the two Linux cells only, uninstrumented; a push to `main`, a
weekly schedule and a manual dispatch add the macOS cell and its coverage.
Markdown and `docs/` changes run no CI, on pull requests as well as pushes,
and tags are not tested, since a release tags a commit `main` has passed.
The pull-request cells had taken 10–33 min (the threaded one the slowest),
52 of the last 80 commits on `main` touched only Markdown, and macOS cells
queued behind the other repositories', since the account runs five at once;
TreeAMR's `HISTORY.md` ("Development environment and CI") has the
measurements across the repositories.

The suite runs in **tiers** (`test/tiers.jl`, after
TreeGeneralizedHarmonic's), chosen by `TREEGRRMHD_TEST_TIER`. On GitHub's
runners, with bounds checks, the suite had taken 14 minutes at one thread
and 32 at four (the step-9 pull request, 2026-11-05). The four-thread time
was not the 3D or curved runs, which ran faster threaded (the curved file
197 s → 127 s), but the 1D studies: 4–8 times their one-thread time there
(the reconstructions' Alfvén waves 41 s → 278 s, the Alfvén wave across
coarse-fine faces 13 s → 185 s, the shock tubes about 40 s → 296 s), where
on Symmetry four threads are faster than one throughout. So `full`, the
default, runs everything (the workstation and Symmetry before a pull
request, `ci/full-suite.sh`, and CI's weekly run); `quick`, CI's one-thread
cells, leaves out the `@full_tier` testsets (the curved cases' `h = M/8`
runs, the 1000-step uniform flows); and `threaded`, CI's four-thread cell,
also the `@serial_tier` ones, the 1D studies. The kernels are one code for
D = 1, 2, 3, so the 2D and 3D runs, refinement, the curved backgrounds, the
step cache and the two-phase recovery still exercise every threaded path,
and the 1D runs held bit for bit to 2D and 3D runs stay. A testset left out
is one skipped test ("Broken") and is named at the end of the run: 3 in
`quick`, 32 in `threaded`. The runs several testsets read are made on first
use (`computed_once`) rather than by top-level constants, and the helpers
shared between files live in files of their own.

1.12 and later redefine a `const` silently, where 1.11 refused, so the
duplicate-constant check at the top of `test/runtests.jl` is the only thing
that finds a top-level name two test files share.

`Project.toml` names EntropyEOS and IMEXRungeKutta there, by their GitHub
URLs at `rev = "main"`, and CI adds nothing by hand: `julia-buildpkg`,
a clean `git archive` and a developer's `Pkg.instantiate()` all resolve
them the same way. The test environment reaches them through the
package (`TreeGRRMHD.EntropyEOS`, `TreeGRRMHD.IRK`) rather than listing
them again, so that a second `[sources]` cannot disagree with the first.

**Inconsistent:** `CLAUDE.md`'s former "Current state" (now [HISTORY.md,
"EntropyEOS"](HISTORY.md#entropyeos), 2026-10-30) said `[sources]` pins
EntropyEOS to the tag `v2.2.1` (since 2026-10-30), not `rev = "main"`.

The history of the floor (1.10, 1.11, 1.12), of the CI cells and of what
1.11 needed is in [HISTORY.md, "Julia version"](HISTORY.md#julia-version).

## Precision and devices

*(Proposed.)* Every driver is generic in the element type `T`,
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
derivative bit-identical to the host at `Float64` and `Float32` (measured,
Kerr-Schild at `a = M/2`). That is the premise of the `Analytic` provider.

The pointwise layer — `Metric3`, `prim2con`, `con2prim_hydro`,
`con2prim_ideal` — runs inside a KernelAbstractions kernel bit-identically
to the host at `Float64` and `Float32`, allocation-free and inferred, and
so does `GeometricEOS` around a table at `Float64`. So does the pointwise
physics — `Analytic(KerrSchild)` with its ForwardDiff derivatives, the
stress-energy, both Riemann solvers in every direction, Ohm's law, both
source forms, the UCT EMF and the gauge terms — at `Float64` and `Float32`,
and the stage solve on every branch. `lower`, `raise`, `dot_up` and
`dot_down` multiply with plain `*` and `+` (`mul3` in `src/metric3.jl`),
which is never fused: StaticArrays' matrix product accumulates with
`muladd`, which LLVM fused differently in the kernel and on the host, so
three of the 79 numbers differed in the last bit. On a device, which
contracts by its own rules, bit-identity is not the claim; step 14 states
what is. `GeometricEOS` around a table is not `isbits` (the table holds
arrays) and is passed to a CPU kernel as it is; it carries an `Adapt` rule
for a device.

Every kernel of the scheme and of the driver's initial data takes its
geometry as `to_backend` arrays of the block origins and spacings and its
physics as `isbits` arguments (the EOS, the options, the model, the
provider, a case's callable `WaveFields`/`WavePotential`), and is generic in
`T`: the Alfvén wave runs in `Float32` throughout — the matter and field
kernels, the stage solve and the reset — and lands within 4e−5 of the
`Float64` run's error (rung 1a). The diagnostics (`l1_errors`,
`conserved_totals`, `ohm_tally`, `max_light_rate`) are `mesh_mapreduce`s,
where the data are.

## Sharp edges

What to know before changing the code. Each is a short rule; the section
it names has the reasons.

- **Five centerings, five offsets.**
  - Cell, face, edge and vertex sets each own points `G+1 … G+N` along each
    dimension. The shared high plane `G+N+1` of a staggered dimension is
    filled like a ghost: by the exchange, or by the boundary hook at a
    physical boundary.
  - `coordinates` takes **stored** indices.
  - `map_blocks!` adds `G` under the default and `closed = true` ranges,
    and adds nothing under `stored = true`.
  - Each kernel receives one `Val(G)` per set it reads.
- **`B` is never state.** `B_d` is recomputed from `A` (plus `𝔅̄`) in every
  RHS, on the stored range. It is not in the state vector, not in
  `regrid!`'s transfer, and not in a checkpoint ("Where the fields live",
  "Refinement").
- **EntropyEOS's calling conventions** ("Primitive recovery").
  - Inputs are **undensitized** (divide by `√γ`). `D` is `ρ* W`, in
    κ-rescaled units for tables; κ = 1 for the analytic EOSes. A table is
    seen in geometric units through `GeometricEOS`, so no call site
    converts units.
  - `S_par`/`S_perp` are scalars, and `S_perp ≥ 0`. For the hydro mode
    pass `B² = 0`, `S_par = 0` and `S_perp = √(γ^{ij}S_iS_j)`.
  - The output velocity is scalar. Its direction is `Sⁱ`'s; at the
    converged state its magnitude is `|S|/(ρhW²)`, and in the ideal seed
    `vⁱ = (Sⁱ + (S·B)Bⁱ/z)/(z + B²)` with `z = ρhW²`. Take the magnitude
    from the returned rapidity, `|v| = tanh w`, not from that quotient.
  - The warm-start guesses of `con2prim`/`con2prim_safe` are
    **positional**, although the docstrings show keywords.
  - EntropyEOS's two levels, `con2prim_safe` (the projection, onto
    `ProjectionOptions`) and `apply_policy` (the atmosphere, with
    `PolicyOptions`), are called in `src/con2prim.jl` only, the second
    right after the first, in one function, `two_levels`, which
    `fluid_recovery` and `ideal_recovery` go through; their conserved
    output `cons` is never used: every caller rebuilds the conserved
    variables from the primitive state with this package's `prim2con`.
    Test flag words with `& … != 0`; `==` only where the whole word is the
    claim ("The two levels", "The flags").
  - **The step cache** ("The step cache"). A test that steps an integrator
    itself runs without the cache unless it wraps the stepping in
    `with_step_cache` — the restart test does, with `mhd_integrator`'s
    integrator, whose step limiter is `reset_step!`; one built with
    another step limiter would carry the cache across steps. Every site
    computes `B`, `E` and the undensitized variables as `· inv(√γ)`, so
    that the keys agree; a new recovery site must too.
- **TreeGH's conventions.**
  - `h_ab = g_ab − η_ab` is stored as an offset, and
    `Π_ab = √γ n^μ∂_μ g_ab` is not `∂ₜh`.
  - Components are packed `(tt,tx,ty,tz,xx,xy,xz,yy,yz,zz)`.
  - GHSO2's `dg[a,b,c] = ∂_a g_bc`, while SpacetimeMetrics has
    `dg[a,b,c] = ∂_c g_ab`. Convert once, in `src/metric.jl`, and nowhere
    else ("Metric providers").
- **The outer boundary condition is applied in full after every fill**:
  one TreeAMR fill does it in `fill_ghosts!`, `regrid!` and
  `adapt_to_initial_data!` alike, so every evolved set is filled with its
  own hook (`fill_set!`, `regrid_pairs`). A new path that fills a potential
  without its hook leaves the outer points and the low wall as they were
  ("Boundaries").
- **A regridded run's initial totals** are `evolve!`'s `totals0`, taken on
  the initial mesh: the case's initial data on the final mesh has other
  Gauss averages ("Refinement").
- **The IMEX implicit tendency is `(U − u★)/(γΔt)`.** The stage solver must
  return the `U` it solved for, before any limiter, and write only `E` into
  it: its repair of an inadmissible `u★` stays local. The atmosphere
  limiter runs after the tendency is taken ("The IMEX package", "The
  implicit stage solve").
- **The atmosphere reset is both limiters**: pass it as `stage_limiter` and
  as `step_limiter`. As a stage limiter alone, it never resets the state
  ("Floors, the atmosphere and the magnetosphere").
- **Two abscissae.** IMEXRungeKutta calls `f_exp!` and the stage limiter at
  `tⁿ + c̃Δt` and `solve_imp!` at `tⁿ + cΔt`, and they differ. Only the
  explicit part moves `A`, so a boundary or source that depends on `t`
  through `A` is wrong in the stage solve unless it is written so that `t`
  drops out — hence the potentials' Dirichlet data in the field
  ("Boundaries").
- **The RHS never mutates `u`.** The stage limiter writes only atmosphere
  cells — and the potentials' owned low walls at the low domain faces,
  which the fill writes under symmetric walls (AMR-8) and the reset, as
  stage and step limiter, copies into `u` (`walls_to_state!`), so that `u`
  holds the boundary condition's walls. The right-hand side writes zero
  into `du` there: the wall is prescribed, not evolved ("The right-hand
  side", "Boundaries").
- **Masks branch, never multiply**, because `0·NaN = NaN`.
- **An adapter is a snapshot of its producer's prepared state.** The
  `(ρ, S_i, S_ij)` that TreeGH's RHS and ADM monitor read, and the metric
  this package's recovery, RHS and stage solve read, come from a `prepare!`
  at the same `(u, t)` — at `u★` for the stage solve; so does `Mv`,
  refilled by `vertex_metric!` from each new `ADMMetric` ("The coupling
  contract").

## Tests

The suite is `test/runtests.jl`. It runs in tiers (`test/tiers.jl`,
chosen by `TREEGRRMHD_TEST_TIER`; see "Julia version"): `full`, the
default, runs everything; `quick` leaves out the `@full_tier` testsets (the
finest curved runs, the 1000-step uniform flows); `threaded` also the
`@serial_tier` ones (the 1D studies). The rules for a new testset are in
`CLAUDE.md`, "Conventions".

- `test/prerequisite_tests.jl`: the names the scheme calls, the IMEX
  contract, and a probe per upstream ID still waiting (`PLAN.md`,
  "Upstream requirements", "Retiring a probe").
- `test/recovery_tests.jl`, and after it `test/pointwise_tests.jl` and
  `test/stage_solve_tests.jl`, which use its state generators.
- `test/step_cache_tests.jl` (the step cache) and `test/two_phase_tests.jl`
  (the two-phase recovery against fused, bit for bit).
- `test/scheme_tests.jl` and `test/wave_tests.jl` (the convergence
  studies).
- `test/shocktube_tests.jl`, `test/resistive_tests.jl`,
  `test/atmosphere_tests.jl`.
- `test/reconstruction_tests.jl` (uses `wave_tests.jl`'s helpers).
- `test/multid_tests.jl` and `test/multid_cases_tests.jl` (uses the
  former's helpers).
- `test/refinement_tests.jl` and `test/refinement_cases_tests.jl` (use
  `multid_tests.jl`'s helpers).
- `test/curved_tests.jl`: the right-hand side's truncation error, the
  drift to `t = M/2` at `h = M/4`, `M/8`, the interior, the curved-space
  bit identity; its boundary testset holds a gauge offset out of `𝔅`.
- `test/tov_tests.jl`: its runs are 3D at `N = 8` and a few `M` long, so
  its testsets run in every tier, unmarked.
- `test/vertex_metric_tests.jl`: the vertex metric at every centering.

**Inconsistent:** the helper dependencies listed above
(`recovery_tests.jl`'s state generators, `wave_tests.jl`'s and
`multid_tests.jl`'s helpers) are the older layout; `CLAUDE.md`,
"Conventions", and "Julia version" say the helpers shared between files
live in files of their own (`recovery_states.jl`, `wave_study.jl`,
`multid_runs.jl`, `refinement_meshes.jl`), included by `runtests.jl` before
their first user.

## Code layout

Where each piece of the design lives. The sections named say what it is
and why.

**The pointwise layer** (each runs inside a KernelAbstractions kernel,
bit-identically to the host; "Precision and devices"):
- the module shell; `src/precision.jl` and `src/device.jl`, ported from
  TreeHydro ("Precision and devices");
- `src/units.jl`: geometric units, `GeometricEOS` ("Thermodynamics");
- `src/metric3.jl`: the 3+1 point metric;
- `src/variables.jl`: the slot layouts of `U` and `P` ("Field sets");
- `src/prim2con.jl` and `src/con2prim.jl`: `con2prim_hydro`,
  `con2prim_ideal`; the step cache's entry and `two_levels`; the
  two-phase recovery's `Handoff` ("Primitive recovery");
- `src/metric.jl`: the `Flat`/`Analytic` providers and `MetricGradient`,
  the one place SpacetimeMetrics' derivative index order is converted
  ("Metric providers");
- `src/vertex_metric.jl`: `VertexMetricSet(T, forest; boundary)`, which
  holds `Mv`; `vertex_metric!(vm, adm, t)`, which fills it from any adapter
  in `ADMMetric`'s form and returns `VertexMetric`; `metric_fallbacks`;
  `SampledADM` ("Metric at cells, faces and edges");
- `src/stress_energy.jl`, `src/ohm.jl` (with
  `MagnetizationCutoff(model, μ_max)`, "Ohm's law and the conductivity"),
  `src/flux.jl`, `src/riemann.jl` (the light-cone Riemann solvers, and
  `:hlle_split`, measured, not the default; "Riemann solver"),
  `src/emf.jl` and `src/sources.jl`;
- `src/stage_solve.jl`: `solve_ohm_stage`, Newton on the fluid's share of
  the energy and momentum, `OhmStageResult` ("The implicit stage solve").

**The scheme on the mesh** (the kernels are written for D = 1, 2, 3):
- `src/fieldsets.jl`: `mhd_fields`, every set of the table in "Field sets"
  in D = 1, 2, 3, and the diagnostics set `X`;
- `src/curl.jl`: `curl_A!`, `𝔅 = curl A + 𝔅̄` on every stored face;
  `cell_𝔅` ("Where the fields live");
- `src/reconstruction.jl`: PLM, and with `limiter = :ppm`, `:wenoz`,
  `:mp5` (which need `G = 3`; MP5 runs at `cfl = 1/5`, `default_cfl`) PPM,
  WENO-Z and MP5; the positivity fallback; the face state
  ("Reconstruction");
- `src/evolution.jl`: `MHDProblem` (which carries the step cache and the
  hand-off), the kernels, `mhd_rhs!`, `ohm_solve!`, `reset_atmosphere!`,
  `ohm_tally`, `max_light_rate`, the conserved totals ("The right-hand
  side");
- `src/driver.jl`: `MHDCase`, exact initial data and references,
  `l1_errors`, `mhd_integrator`, `evolve!`, and `DirichletBoundary`, the
  Dirichlet hooks a non-periodic case gets from its own data ("The
  partition", "Boundaries");
- `src/refinement.jl`: `mhd_flags` (Löhner on `ρ`, `|B|`, `|E|`),
  `regrid_pairs`, `regrid_state!`; `evolve!` runs a static mesh
  (`refine`) or an adaptive one (`maxlevel`, `refine_tol`, `coarsen_tol`,
  `buffer`; `adapt_case!` over every set) ("Refinement");
- `src/interior.jl`: `HoleInterior(r_in; stage, step)`, an `MHDCase`'s
  `interior`, and the diagnostics `outside_l1`, `field_l1`,
  `charge_norms` ("The interior");
- `src/tov.jl`: the host TOV solve in Lindblom's enthalpy form,
  `TOVMetric` ("Gauge for the TOV milestones").
- `reset_step!`, the reset as the step limiter, counted with the step
  limiter's injection (`ohm_tally`); `max_div_B` and `evolve!`'s
  `after_step` monitor.
- The public verbs, which a script that owns the loop calls ("The coupling
  contract", "Two scripts"): `prepare!`, `rhs!`, `stage_solve!`,
  `stage_limit!`, `step_limit!`, `max_rate`, `initial_state!`,
  `MHDProblem(fields, case; …)`; `mhd_rhs!`, `ohm_solve!`, the resets,
  `mhd_integrator` and `evolve!` are compositions of them, bit for bit.

**The cases** (`src/cases/`; "Validation ladder"):
- `em_wave_case`, `alfven_wave_case`;
- `src/cases/shocktubes.jl`: Balsara 1–5 with Giacomazzo & Rezzolla's
  exact plateaus, the resistive tube, Komissarov's current sheet and the
  atmosphere tube;
- `src/cases/geometry.jl`: `extruded_case`, `turned_case`,
  `oblique_case`;
- `src/cases/multid.jl`: the explosion, the rotor, Orszag–Tang, the field
  loop;
- `src/cases/kerr_schild.jl`: `MichelFlow` (Michel's accretion flow,
  `michel_density`, `michel_state`), `wald_case` (`Vacuum`, Wald at
  `a = 0`), `bondi_case` (`IdealConductor`, Michel's flow in a radial
  monopole) and `spinning_wald_case` on `Analytic(KerrSchild(1))`;
  `kerr_schild_r`;
- `src/cases/tov.jl`: `tov_case`, `tov_boxes!`, `central_density` (the
  largest cell average of `ρ`).

**Scripts** (`bin/`), the measurements and studies behind this
document's tables:
- `bin/shocktube_reference.jl` made the 3200-cell shock-tube references in
  `test/data/`; `bin/tableau_comparison.jl` the SSP3-vs-ARS table;
  `bin/reconstruction_comparison.jl` the per-reconstruction Balsara table
  and `test/data/balsara3_ppm.ref` ("Validation ladder");
- `bin/interface_order.jl`, `bin/gauge_damping.jl`,
  `bin/field_loop_refined.jl`, `bin/ghost_fill.jl`: the refinement tables
  ("Refinement", "Validation ladder");
- `bin/metric_cost.jl`: the pointwise provider's cost ("Metric
  providers");
- `bin/curved_stationarity.jl`: the `t = 50M` study; `bin/curved_localise.jl`
  (time series, radial layers, face distances, the resets by layer and by
  EntropyEOS level, diagnostic hooks evaluated in its own process;
  `relhook=true` the negative control of the absolute hook,
  `interior=false` the no-interior run, `kwald` the spinning Wald case with
  `a`, `rho_fluid`, `ballks`, `watch`) ("Validation ladder" 3, "The
  interior");
- `bin/tov_cowling.jl`: the Cowling star as a script that owns the loop
  (a test runs its `main` against `evolve!`, bit for bit; "Two scripts");
  `bin/tov_study.jl` (checkpointed, for `amdpreq`; `metric=analytic|vertex`),
  `bin/tov_compare.jl`, `bin/tov_drift.jl` and `bin/tov_modes.jl`: the
  rung-4 record ("Validation ladder" 4);
- `bin/recovery_cost.jl`: `steptime` times the fused and the two-phase
  recovery, `finish` counts the share that reaches finish ("Primitive
  recovery").

`ci/full-suite.sh` and `ci/full-suite-symmetry.sh` run the full suite as a
commit status (`CLAUDE.md`, "Commands").

## Validation ladder

These become the milestones of `PLAN.md`; each rung is a set of testsets
named as claims.

1. **Flat space, 1D**: the vacuum electromagnetic wave (`Vacuum`) at the
   scheme's order; the large-amplitude circularly polarized Alfvén wave
   (`IdealConductor`) at the scheme's order; the resistive shock tube swept
   over σ (Palenzuela et al. 2009, Dionysopoulou et al. 2013; the
   published tube is a Brio–Wu tube with `Bˣ = 0`), approaching the ideal
   solution as σ → ∞; the
   self-similar current sheet against its analytic profile; Balsara 1–5
   with `IdealConductor`, against GRaM-X's figures; a σ × magnetization
   sweep of the implicit stage solve, recording where each branch of it
   is taken. *(The sweep is done pointwise, on stage problems with a
   known solution — "The implicit stage solve"; rung 1 records the branches
   an evolution takes.)*

   *Rung 1a* (`test/wave_tests.jl`; MC, HLLE,
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

   *Rung 1b* (`test/shocktube_tests.jl`,
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

     *Per reconstruction (*`bin/reconstruction_comparison.jl`*;
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
     every error falls, and `R4` comes within 0.23% — and the suite
     asserts it for PPM against PPM's own 3200-cell reference
     (`test/data/balsara3_ppm.ref`, rates ≥ 0.4, every plateau within 2%);
     MC keeps its pinned values. WENO-Z is the worst of the
     four at the shell (its overshoot at the contact smears `R4` by 12.6%)
     and MP5 converges in `D` and `𝔅ʸ` but barely in `S_x` and `τ`; neither is
     asserted. The MC references in `test/data/` are not regenerated: rerun,
     four of them are identical byte for byte, and Balsara 1's differs in 314
     of 4800 `Float32` values by one rounding (1.9e−9 at most), the change of
     the initial data's Gauss average in the last bit of a `Float64`. The
     3200-cell references in `test/data/` are not regenerated for the
     symmetric walls, the relative hook and the primitive ghost fill either
     (what those moved in the uniform runs is in
     [HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder)): the
     scheme changed for them by rounding only, against reference errors of
     10⁻³ (proposed).
   - **The plateaus** against Giacomazzo & Rezzolla's exact solution
     (2006, Tables 7–11; `BALSARA_EXACT`, whose `p` is the total pressure),
     averaged at 400 cells over each region's window (`BALSARA_PLATEAUS`,
     where the 3200-cell run holds the region to 2%, less four cells either
     side), with `ρ` and `p` relative to their own value and each component
     of `v` and `B` to its largest magnitude in the solution
     (`plateau_error`, proposed): every window within 2% —
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
     A uniform flow at `vˣ = ±0.999` or `±1/2` crosses the box unchanged,
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
     from `E = 0`, proposed): the erf profile solves the diffusion limit
     and not the system — the field's pressure moves the fluid at
     `2·10⁻⁴`, whose `v × B` is 1% of the resistive `E`, and the
     displacement current lags the diffusion by `O(1/σt)` — so the
     distance to it saturates at 1.1e−3 of `B₀` (L1 per unit length; 2.8e−3,
     8.2e−4, 1.06e−3, 1.14e−3 at 50 … 400 cells over `t = 1 → 2`). The
     order is therefore measured against the next finer run: 2.53 and 2.58
     over 50 … 400 cells. At Komissarov's 200 cells
     and `t = 9`, with the potentials' low walls prescribed, the distance to
     the profile is 9.6e−4 — "indistinguishable"
     to 0.1% — and at `B₀ = 1/10`, where the fluid's response (quadratic in
     `B₀`) drops out, it is five times smaller relative to `B₀`. Every stage
     is a fixed point (1,068,800 of them to `t = 9`), at most 4 recoveries.
   - **The atmosphere** (`atmosphere_tube_case`, proposed: a gas
     at ρ = p = 1 expanding into the policy's own atmosphere at
     `ρ_atm = 10⁻⁸`, `W` up to 4.9; 200 cells to `t = 0.4`): hydrodynamic
     (`B = 0`), and magnetized as a star's surface would be — `Bʸ = 10⁻³`,
     `B²/ρ = 100` in the atmosphere, `Constant(10³)` cut off at
     `μ_max = 10` (the measured safe value), so the atmosphere is
     electrovacuum. No density below `ρ_atm`, nothing non-finite; the reset
     writes 35,158 cells as the stage limiter and 11,686 as the step
     limiter (hydrodynamic) — the policy
     flags every atmosphere cell, and the reset writes it back as it is —
     and injects `−8.4·10⁻¹²` of rest
     mass and `−2.2·10⁻¹⁰` of energy into a box holding 0.5 of rest mass —
     and the totals
     change by exactly those amounts, to rounding, which also shows the
     stage limiter's writes stay out of the state. A
     stage predictor next to the vacuum can carry `D < 0`, which EntropyEOS
     reports as non-finite and repairs inside the stage (the flag appears
     among the stages' policy flags, never in the state); its magnetization
     is `+∞` (see "Ohm's law and the conductivity"). The
     hydrodynamic tube is not chaotic: the conserved and the primitive
     ghost fill give its numbers to every digit. The magnetized tube's
     perturbed runs are clean (48 of them), but not the same run; the
     suite's run prints `W` 4.80, injected −1.18·10⁻⁸, −7.4·10⁻⁷. The
     cause of its sensitivity — the field's truncation error in the fluid's
     share, amplified by EntropyEOS's repairs at the edge of the valid
     range — and the sweep of fixes are under [HISTORY.md, "Floors, the
     atmosphere and the magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere);
     the tableau comparison on this tube is in "The
     partition". The test asserts no fallback, hydrodynamically and
     magnetized. How the magnetized tube's numbers moved is in
     [HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder).
   - Cost (Julia 1.13, Apple M3 Pro, one thread, 400 cells): a step of
     Balsara 1 is 6.5 ms, of Balsara 3 15 ms (the recoveries at a pressure
     ratio of 10⁴); the three test files take 45, 37 and 11 s.
2. **Flat space, 2D and 3D, with AMR**: the cylindrical explosion, the
   magnetic rotor, Orszag–Tang, field-loop advection across coarse-fine
   faces — `∇·B` at round-off, rest mass and magnetic flux conserved
   across levels to round-off, regrids included.

   *Rung 2a* (`test/multid_tests.jl`,
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
     likewise in 2D.
   - **Turning**: the same runs turned by `R(x, y, z) = (y, x, −z)`
     (`turned_case`; a rotation, so that `B = curl A` turns as a vector) are
     the turned runs, every value `==` and every nonzero value bit for bit.
     The sign of a zero is the one freedom: Balsara 1's identically zero
     `S_z` and `𝔈ᶻ` are `+0` in both runs where the turn predicts `−0`
     (`0 − 0 = +0` either way), so `isequal` would fail on them and nothing
     else: the claim is stated with `==`.
   - **The Alfvén wave along the diagonal** (`oblique_case`, `(1, 1)/√2`
     on the periodic `[0, √2]²`, an eighth of a period at 16², 32², 64²; a
     quarter in parentheses, measured once): L1 rates `S_x`, `S_z`, `A_x`,
     `A_z` of 2.21, 2.23, 2.77, 2.31 (2.28, 2.36, 2.66, 2.51) under MC and
     1.97, 1.98, 2.05, 2.01 (2.03, 1.92, 2.06, 2.04) under WENO-Z — the whole
     2D path, both fluxes, the curl and the EMF with two transverse
     reconstructions, at second order. The totals drift at most 1.6 ulp of
     their scale per step. With the gauge damping `ξ = 16/L` on every mesh,
     `A_x`'s gauge part is damped: its rate goes from 2.77 to 2.769 (MC) and
     from 2.05 to 2.07 (WENO-Z, L1 3.67e−4, 8.62e−5, 2.14e−5 → 3.26e−4,
     7.46e−5, 1.84e−5); `S_i`, `A_z` and the rates of `S` move in their last
     two digits only, WENO-Z's drift from 1.43 to 1.42 ulp. The 1D runs'
     gauge is never excited (`A_x` is constant where `E_x = 0`, and no 1D
     field reads it), and the 2D and 3D symmetry, translation and turning
     claims hold bit for bit with the damping as without it.
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
     4.0e−16 of their scale (the precursors at the faces); every stage ideal,
     no repair, no reset.
   - **Orszag–Tang** (Beckwith & Stone §4.4; 64², MC, `t = 1/8`, 40 steps):
     every total but `𝔈`'s constant to 0.056 ulp of its scale per step
     (asserted ≤ 8); symmetric under the rotation by π about the centre to
     3.8e−13 — not to the last bit, but to the recovery's tolerance
     (EntropyEOS's `1e−12`): mirrored states that differ in the last bit stop
     at different iterates. Measured before the primitive ghost fill (when
     the suite's asymmetry was 1.9e−13), the asymmetry is 9.3e−16 in the
     initial data, 1.2e−13, 2.5e−13, 1.9e−13 and 1.6e−14 at `t` = 1/32 …
     1/4, and 5.3e−15 at 1/16 with the tolerance set to `1e−14` (asserted
     ≤ 1e−11: "to round-off" means to the recovery's tolerance where the
     recovery iterates).
   - **The field loop** (Beckwith & Stone §4.2; 64 × 32, MC, `t = 1/2`, a
     fiftieth of a crossing): the totals constant to 0.21 ulp (0.25 with
     `vᶻ`) a step; with `vᶻ = 0`, `Eˣ`, `Eʸ`, `A_x`, `A_y` and so `𝔅ᶻ` stay
     exactly zero, as Beckwith & Stone found; with `vᶻ = 0.1/√6` the
     in-plane EMFs `vᶻB` give `max |𝔅ᶻ|` 4.6e−4 of the in-plane field at
     `t = 1/2` (3.4e−4 at `t = 1`).
   - **3D**: a spherical blast (the explosion with `r = |x|`, `[−2, 2]³` at
     16³, minmod, to `t = 1/4` in the suite and `1/2` once)
     runs with no repair and no reset,
     and is symmetric under its three reflections exactly (0.0) on the
     periodic box, and bit for bit in the closed box with both walls
     prescribed ("Boundaries"), where the suite runs it; so is the
     explosion on `[−2, 2]²` at 32².
   - **Higher order in 1D** (the Alfvén wave of rung 1a, one period, 50 …
     400 cells): L1 rates of `S_y`, `A_y`, the ideal projection of `𝔈ʸ` and
     the state's own `𝔈ʸ` of 2.09, 2.10, 2.10, 0.85 (MC), 2.67, 2.47, 2.68,
     0.97 (PPM), 2.00, 2.00, 2.00, 0.99 (WENO-Z), 2.00, 2.00, 2.00, 0.98
     (MP5); `S_y` errors at 400 cells 2.3e−4, 2.6e−5, 1.3e−5, 1.3e−5. `D`,
     `S_x`, `τ` converge at 2.8–2.9 under MC and PPM and 5.6–5.7 under WENO-Z,
     and are at rounding (1e−15) under MP5. Every stage ideal, no reset.

   *Rung 2b* (`test/refinement_tests.jl`,
   `test/refinement_cases_tests.jl`; HLLE, SSP3(4,3,3), `cfl` 2/5 over `D`
   on the finest level; the matter's operators Conservative width 3
   bounded, the potentials' PointValue width 4, `ξ = 16/L` on every mesh
   — "Refinement"; Julia 1.13; TreeAMR 0.2.2, with the potentials `A_j`
   conservative along the edge (AMR-9) and the matter's ghost fill in
   primitives, PointValue width 2 (AMR-10), with which the numbers below
   are measured; the values before those two changes are in
   [HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder)):
   - **Conservation at coarse-fine faces**: every total but `𝔈`'s drifts
     0.013 (1D) and 0.12 (2D) ulp of its scale a step on
     static two-level meshes with the flux fixup, 1.8·10¹⁰ and
     5.4·10¹¹ without; through regrids, 0.015 (Balsara
     1, tracked, ten regrids) and 0.037 (the rotor, two). See
     "Conservation at coarse-fine faces".
   - **Magnetic flux**: with `A_j` conservative along the edge the flux
     holds as a rate and as a state. The rate through every coarse face on
     a coarse-fine face equals the fine faces' sum to 8.7e−19 of 5.4e−3 (2D)
     and 5.5e−3 (3D) with the EMF correction, and 0.84 and 0.31 of it
     without. As a state, the plane's flux holds to 2.2e−16 through the run
     and 1.5e−16 through a regrid, and in 3D every coarse face on a
     coarse-fine face its fine faces' sum to 1.4e−16 of the largest flux
     after steps and after a regrid (1.5e−3 with PointValue), with or
     without the EMF correction (the transfer's property); at an outer face
     only to 1.2e−5 (each level's hook, AMR-11).
   - **`∇·𝔅`**: at most 2.0e−16 of `max |𝔅|` after every step on every
     level and right after every regrid (the adaptive rotor), as in rung 2a.
   - **The field loop across a coarse-fine face** (`x ≥ 0` refined, the face
     through the loop's centre): its magnetic energy and its `A_z` error lie
     between the uniform coarse and fine controls — at `t = 1/2`, 0.803 of
     its initial energy against 0.623 and 0.844, L1(`A_z`) 1.56e−6 against
     3.0e−6 and 8.7e−7; at `t = 4`
     (`bin/field_loop_refined.jl`, the loop a third of its radius across the
     face), 0.589 against 0.206 and 0.619, and 4.16e−6
     against 9.3e−6 and 3.0e−6. The refined half keeps the fine
     mesh's energy; the coarse half loses the coarse mesh's. (Adaptive, one
     level where Löhner fires, the loop with its buffer and 2:1 balance
     covers every root of so small a box: it is the uniform-fine run, to
     every digit printed.)
   - **The interface order and the gauge damping**: the tables in
     "Refinement" (width 4: L∞ 1.97 in 1D, 1.85 in 2D, where
     width 2 gives 0.88 and 0.08; `ξ = 16/L` clears 99.95% of a gauge
     pulse's spurious field within two light crossings).
   - **The tracked shock tube**: Balsara 1 on its box made periodic, 100
     coarse cells and two levels more where Löhner fires (`refine_tol =
     0.08`, `coarsen_tol = 0.02`, buffer 6), regridded every 1/80: three
     adaptation passes, ten regrids, 18 to 40 blocks, 416 steps, conserved
     to 0.015 ulp a step, nothing repaired and no reset.
   - **The atmosphere across a coarse-fine face**: the bounded ghost fill
     repairs no ghost and injects what the uniform mesh does; unbounded,
     `D < 0` in fine ghosts and 250 times the rest mass (the table in
     "Refinement"). With the parent-centred fill, the matter's ghost fill
     ("Refinement"): −1.6e−11 of rest mass, as the conserved fill's. The
     primitive fill of width 2 and of width 4 is in the tables in
     "Refinement".
3. **Analytic curved backgrounds**: the Wald solution (`Vacuum`,
   Kerr-Schild) and the Michel monopole or magnetized Bondi flow
   (`IdealConductor`).

   *Rung 3* (`src/cases/kerr_schild.jl`,
   `src/interior.jl`, `test/curved_tests.jl`, `bin/curved_stationarity.jl`;
   proposed, but for the choice of the magnetized Bondi flow, `a = 0` and
   the octant, which are decided):
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
   - **Magnetized Bondi, not the Michel monopole**. Both need a monopole field. The Michel monopole (Michel 1973) is
     a force-free solution with negligible matter: at `a = 0` it is static
     and trivial, at `a ≠ 0` Kerr has only the Blandford–Znajek perturbative
     one, and in this package's model it would be electrovacuum or an ideal
     plasma at a magnetization above the atmosphere's, neither of them its
     physics. The magnetized Bondi flow is exact for the whole system, matter
     included, is the standard GRMHD test of the literature above, and
     exercises the ideal stage solve, the curved sources and the radial
     inflow through the horizon at once.
   - **`a = 0`**: a stationary fluid is needed in both cases, and Michel's is
     Schwarzschild's; Kerr has no closed-form accretion flow. The spinning
     Wald case (below) carries a passive fluid at the floor and is measured
     on `𝔈` and `𝔅` alone (under `Vacuum` nothing couples).
   - **The octant** `[0, L]³`, `L = 4M`, the hole at its corner, Dirichlet
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
   `≥ 1.8`, judged on the finest pair, `M/8 → M/16`). The L1 drift outside
   `r = 2M` after `t = 50M`, both limiters, `r_in = M`, with the potentials'
   hooks absolute (option 1, below):

   **Re-measured with option 1 as built** *(jobs 588498–588503,
   Symmetry, the node per job in the logs)*: at `M/4` and `M/8`, Wald `D`
   2.9810e−3, 5.8805e−4; Bondi `D` 6.3065e−3, 1.6497e−3 outside `2M` at `t =
   50M`; as rates (`t = 50M`, outside `2M`, `M/4 → M/8`): Wald `D` 2.981e−3
   → 5.881e−4 (**2.34**), `τ` 1.255e−3 → 2.597e−4 (**2.27**), `S_x` 8.50e−4
   → 2.19e−4 (1.96), `𝔈ˣ` 4.578e−4 → 1.154e−4 (**1.99**), `𝔈ᶻ` 1.95, `√γ q`
   1.65; Bondi `D` 6.307e−3 → 1.650e−3 (**1.93**), `τ` 3.436e−3 → 8.158e−4
   (**2.07**), `S_x` 2.471e−3 → 7.677e−4 (1.69), `𝔈ˣ` 3.487e−4 → 1.065e−4
   (1.71), `√γ q` 1.49. At `M/16` (jobs 588502 on cn083, EPYC 7532, and
   588503 on cn107, EPYC 7543, amdq, 64 threads): Wald `D` 1.2609e−4,
   `τ` 5.9239e−5, `𝔈ˣ` 2.8671e−5, `𝔈ᶻ` 7.7540e−6, `S_x` 5.6127e−5, `√γ q`
   8.5733e−5, flat from `15M` to `50M`; Bondi `D` 4.1891e−4, `τ` 1.9593e−4,
   `𝔈ˣ` 2.9913e−5, `S_x` 2.1725e−4, `√γ q` 1.1346e−4 (the absolute-hook
   diagnostic of "Where the error lives", below, to every printed digit:
   the same arithmetic). Rates `M/4 → M/8` / `M/8 → M/16`:

   | | `D` | `τ` | `S_x` | `𝔈ˣ` | `𝔈ᶻ` | `√γ q` |
   |---|---|---|---|---|---|---|
   | Wald | 2.34 / 2.22 | 2.27 / 2.13 | 1.96 / 1.96 | 1.99 / 2.01 | 1.95 / 1.94 | 1.65 / 1.68 |
   | Bondi | 1.93 / 1.98 | 2.07 / 2.06 | 1.69 / 1.82 | 1.71 / 1.83 | 1.71 / 1.83 | 1.49 / 1.63 |

   No reset, no injected mass, the charge's total
   rounding (Wald `−3.4·10⁻¹¹` over the octant). Both drifts are flat in time: Wald's
   `D` within 2% from `10M` to `50M` and to `100M` at `M/4` (2.9810e−3 at 50M
   and at 100M), Bondi's to five digits. **Verdict: the prediction holds**
   for `D` and `τ` (1.93–2.34 and 2.06–2.27 between the three resolutions)
   and Wald's `𝔈`, `S_x` (1.94–2.01); **it fails** for Bondi's `S_x` and
   `𝔈` at `M/4 → M/8` (1.69, 1.71 — measured, not loosened) and for the
   charge (1.49–1.68); `M/8 → M/16` Bondi's are 1.82–1.83. Above 1.8 from the
   second pair on, so the coarsest mesh is not yet asymptotic for the
   monopole's transverse components. The
   prediction is judged on the finest pair, `M/8 → M/16`, where it holds for
   every variable; the `M/4 → M/8` rates 1.69 and 1.71 stand as measured — at
   `M/4` only 8 cells lie between the hole's centre and the horizon.

   **Inconsistent:** the paragraph above says the prediction holds on the
   finest pair "for every variable", but the table's `M/8 → M/16` rates of
   the charge `√γ q` are 1.68 (Wald) and 1.63 (Bondi), below 1.8, and the
   verdict lists the charge (1.49–1.68) among the failures.

   **Option 1** — the case's own potential in the
   ghosts where it does not depend on `t`. *As built* (`static_potential` on
   `MHDCase`, `absolute` on `DirichletBoundary`, "Boundaries"): `wald_case` and
   `bondi_case` declare their potentials static, and their potentials'
   hooks are `CellBoundary(AllVariables(…))` of the case's `A_j` and `Φ̃`;
   every other case keeps the relative hook. Options 2 (a relative hook that
   carries the offset's gradient, upstream) and 3 (a gauge that damps a
   static `χ`) are open questions for step 10's outer boundary and for
   dynamic cases without an exact `A`. `bin/curved_localise.jl`'s
   `relhook=true` is the negative control.

   **Inconsistent:** "Open questions" (the potentials' outer boundary) says
   that the Cowling TOV star of step 10 is static and takes
   `static_potential = true` (option 1) as is, so that options 2 and 3
   matter for dynamic cases only; the paragraph above still names step 10's
   outer boundary for them.

   **With the relative hook** (AMR-7; `relhook=true`) the drift does not
   converge: Wald saturates, the magnetized Bondi flow converges at first
   order. The L1
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

   **Where the error lives** *(`bin/curved_localise.jl`: the drift and the
   charge every `5M`, binned at
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
   - **The cause: the potentials' relative Dirichlet hook** (AMR-7, the
     hook for potentials that grow in time). It fills a ghost or wall point
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
     and `Φ̃` themselves in every ghost and on every wall, valid for these
     time-independent potentials (option 1) — Wald's drift is stationary
     from `t = 10M` (outside `2M`, `D` 2.981e−3, `𝔈ˣ` 4.58e−4 at 10M and at
     50M), in the octant and in the box around the hole, and so is the
     magnetized Bondi flow's.
   - **With the absolute hook the prediction holds**: the rates are in
     "Re-measured with option 1 as built", above.
   - **Bondi at first order** was the same mode, slower: at `M/4` its drift is
     flat from `5M`, at `M/8` it grows linearly from `t = 20M` (`D` 1.84e−3 →
     3.28e−3), so the finer mesh ends further from its own converged value;
     with the absolute hook both are flat.
   - **The charge.** Wald's non-converging `√γ q` is the growing mode's
     (`q` follows `𝔈`'s growth; with the absolute hook it is stationary and
     converges at 1.65). The matter's saturation in the Wald case is the
     mode's too, through the total momentum and energy, which carry the
     field's: with the heavy fluid `D`'s drift stays Michel's own.

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

   **The spinning Wald case** *(`spinning_wald_case`, `SpinningWald`)*:
   - **The field**: Wald's uncharged `𝒜 = (B₀/2)(ψ + 2aη)` (Wald 1974), the
     Killing vectors `ψ = ∂_φ = (0, −y, x, 0)` and `η = ∂_t` of Kerr-Schild,
     whose metric rotates with the `(x, y)` plane, lowered by
     `Analytic(KerrSchild(M, a))`'s metric; `𝔅` the curl of `A_i = 𝒜_i`, and `E`
     from `A`'s stationarity, `αE_i = ∂_i𝒜_t − ε_ijk βʲBᵏ`, the derivatives from
     one `dmetric` call. Checked by centred differences
     (`test/curved_tests.jl`, `a = M/2` and `0.9M`): `curl A = √γ B` to
     4.7e−10 and 9.4e−10, Gauss and Ampère (the flux of `𝔈` divergence-free)
     to 1.3e−9 and 3.7e−8, the Lorenz gauge to 3.7e−12; the flux of `𝔈`
     through a sphere `1.1·10⁻⁴` of `ψ` alone's (the quadrature), against 2 for
     `ψ − 2aη`: uncharged. SpacetimeMetrics had everything needed.
   - **`Φ̃ = aB₀` exactly, a constant** (to 2.2e−16): the damped Lorenz gauge
     `∂ₜΦ̃ = … − ξαΦ̃` is stationary at this `Φ̃` only at `ξ = 0`, so the case
     declares `gauge_damping = 0` (`MHDCase`'s field; `nothing`, every other
     case, keeps `16/L`). With the absolute boundary (option 1) pinning the
     gauge, a damped `Φ̃` would drive `A` by a gradient the boundary does not
     share, the mechanism of the mode in "Where the error lives".
   - **The box**: the octant shifted by `L/256` in `z`, so that no grid point
     lies on the equatorial disk `x² + y² < a²`, `z = 0`, where Kerr-Schild's
     `r = 0` and SpacetimeMetrics' metric is `0/0`.
   - **The fluid**: passive, at rest, `ρ = 10⁻⁶` (`10²ρ_min`), under `Vacuum`.
     It does not dominate the field — with a fluid a million times heavier
     (`ρ = 1`) `𝔈`'s and `𝔅`'s drifts agree to 1–5% — but it **dominates the
     reset tallies**: the field's energy and momentum, whose truncation error
     is `10²`–`10³` times the fluid's share of the totals, leave the recovered
     fluid outside its window in most cells every step: `ρ_FLOORED`,
     `S_FLOORED`, `S_CEILED` (level 1) and the atmosphere (level 2), from
     `t = 0.03M` and everywhere in the box (at `M/8`, `a = M/2`, 2.3·10⁸ and
     8.3·10⁷ counted stage and step resets in 50M). The heavy fluid's run
     resets nothing beyond the counting radius. Reported, not special-cased.
   - **The interior: "inside" and the paste's ball.** Kerr-Schild `r` is the
     spheroidal radius (`kerr_schild_r`): on the axis `r = |x|`, in the
     equatorial plane `r = √(|x|² − a²)`. The horizon `r₊ = M + √(M² − a²)`
     (1.87M, 1.44M) lies at `|x| = r₊` on the axis and `√(r₊² + a²)` (1.93M,
     1.70M) in the plane, so a Euclidean ball of radius `ρ` is inside it for
     `ρ < r₊`: the paste at `r_in = M` and the uncounted ball `(r_in + r₊)/2`
     (1.43M, 1.22M, Euclidean) are. "Outside the horizon", for the drift, is
     `r ≥ r₊` in Kerr-Schild `r`. **But the Euclidean paste at `a = 0.9M` does
     not contain the inner horizon**: in the plane it reaches `r =
     √(M² − a²) = 0.44M`, short of `r₋ = M − √(M² − a²) = 0.56M`, and the
     evolved region then includes a ring of `r₋ > r > 0.44M` where not every
     characteristic points inward. Measured: at `a = 0.9M` with the Euclidean
     ball `𝔈ˣ`'s drift grows (`M/8`: 1.8e−3 at 10M, 5.1e−3 at 30M, 3.1e−2 at
     50M, worse than `M/4`'s 8.1e−3), and with the ball in Kerr-Schild `r`
     (`r < M`, which contains `r₋`; a process-local measurement,
     `bin/curved_localise.jl … ballks=true`) it is flat (`M/8`: 1.84e−4 at 10M,
     1.97e−4 at 50M). At `a = M/2` the Euclidean ball reaches `r = 0.87M >
     r₋ = 0.13M` and is flat. **(open)**, left for future black-hole work
     ("Known risks"): the paste's ball in Kerr-Schild
     `r`, or one Euclidean radius `ρ` chosen to contain `r₋` in the plane
     (`√(r₋² + a²) < ρ < r₊`, 1.06M to 1.44M at `a = 0.9M`).
     At `t = 50M` outside `r₊`, `𝔈ˣ`: Euclidean 8.08e−3 (`M/4`), 3.13e−2
     (`M/8`), against Kerr-Schild `r` 7.79e−4, 1.97e−4 (the `M/16` Euclidean
     run, 588686, was stopped at 10M: 9.4e−5 at 5M, 3.4e−4 at 10M, growing).
   - **Stationarity** (`t = 50M`, outside `r₊` in Kerr-Schild `r`,
     `bin/curved_localise.jl kwald`; jobs 588687–588689 at `M/4` and `M/8` on
     cn079–cn081, EPYC 7532, and 588704, 588705 at `M/16` on cn106, cn107,
     EPYC 7543, 64 threads; `a = 0.9M` with the ball in Kerr-Schild `r`, since
     the Euclidean one grows). The L1 drifts at `t = 50M`:

     | | `h` | `𝔈ˣ` | `𝔈ᶻ` | `𝔅ˣ` | `𝔅ʸ` | `𝔅ᶻ` |
     |---|---|---|---|---|---|---|
     | `a = M/2` | `M/4` | 6.1574e−4 | 3.4463e−4 | 3.5058e−4 | 3.7528e−4 | 4.3464e−4 |
     | | `M/8` | 1.5098e−4 | 8.1192e−5 | 6.9373e−5 | 6.8003e−5 | 8.2696e−5 |
     | | `M/16` | 5.1631e−5 | 3.9523e−5 | 4.0078e−5 | 3.8561e−5 | 4.4688e−5 |
     | `a = 0.9M` | `M/4` | 7.7896e−4 | 5.4989e−4 | 5.4814e−4 | 6.1158e−4 | 7.6461e−4 |
     | | `M/8` | 1.9675e−4 | 1.4880e−4 | 1.1388e−4 | 1.2610e−4 | 1.6772e−4 |
     | | `M/16` | 5.6544e−5 | 4.2725e−5 | 4.4849e−5 | 3.3167e−5 | 7.7233e−5 |

     Rates `M/4 → M/8` / `M/8 → M/16` at three times:

     | | `t` | `𝔈ˣ` | `𝔈ᶻ` | `𝔅ˣ` | `𝔅ʸ` | `𝔅ᶻ` |
     |---|---|---|---|---|---|---|
     | `a = M/2` | 10M | 1.94 / 2.01 | 1.95 / 2.02 | 2.10 / 2.03 | 2.12 / 1.96 | 2.13 / 2.02 |
     | | 35M | 2.02 / 2.10 | 2.16 / 2.04 | 2.46 / 2.14 | 2.42 / 2.09 | 2.38 / 2.05 |
     | | 50M | 2.03 / **1.55** | 2.09 / **1.04** | 2.34 / **0.79** | 2.46 / **0.82** | 2.39 / **0.89** |
     | `a = 0.9M` | 10M | 1.93 / 1.94 | 1.81 / 1.97 | 2.14 / 1.92 | 2.16 / 2.05 | 2.08 / 1.86 |
     | | 35M | 2.00 / 1.76 | 1.92 / 1.75 | 2.19 / 1.28 | 2.33 / 1.80 | 2.20 / 1.03 |
     | | 50M | 1.99 / 1.80 | 1.89 / 1.80 | 2.27 / **1.34** | 2.28 / 1.93 | 2.19 / **1.12** |

     **Verdict: the prediction fails on the
     finest pair at `t = 50M`, for both spins.** Every rate is second order
     at `10M`, and at `a = M/2` to `35M`; the coarse pair holds throughout.
     Two different failures, located by the layer tables (`𝔈` per cell, by
     Euclidean radius, at `50M`):
     - **`a = M/2`: a late growth in the outer region.** At `M/16` the drift
       is flat to `35M` (`𝔅ˣ` 1.2–1.5e−5) and then grows (2.6e−5 at `45M`,
       4.0e−5 at `50M`); `M/8` and `M/4` grow over the same time more slowly
       (`𝔅ˣ` 5.4e−5 → 6.9e−5 and 2.8e−4 → 3.5e−4 from `30M`). By layer,
       `M/8 → M/16` is 1.9–2.0 from `1.25M` to `3M` and falls to 0–1 beyond
       `3.5M` out to the corner, where `M/16`'s `𝔈` (3–4e−5 per cell) is
       `M/8`'s; the charge `√γ q` there is *larger* at `M/16` (rates −0.1 to
       −0.5 beyond `5M`). The cells next to the faces carry no more than the
       others, so it is not a face layer. Not the floor fluid: at `M/8` the
       heavy fluid (`ρ = 1`, no counted reset) grows alike (`𝔅ˣ` 5.4e−5 →
       6.8e−5 from `30M`).
     - **`a = 0.9M`: a step in the inner region.** At `M/16`, `𝔅ᶻ` rises from
       4.2e−5 at `10M` to 6.7e−5 at `15M` and is then flat to `50M` (7.7e−5);
       `𝔅ˣ` alike. By layer `M/8 → M/16` is 2.0–2.2 beyond `2.5M` and 1.0–1.6
       from `1.25M` to `2.5M`, the ergoregion's neighbourhood (Kerr-Schild
       `r < 2M` in the plane, `|x| < 2.19M`).
     The non-spinning Wald field (`ξ = 16/L`) was flat from `15M` to `50M` at
     `M/16`, `𝔈`'s rates 1.94–2.01 outside `2M` ("Re-measured with option 1"
     above; `𝔅` was not measured then). The
     spinning case differs in `ξ = 0` (nothing damps the gauge, the charge
     or `Φ̃`), the frame dragging, the inner horizon and the shifted box.
     **Not localised further (open)**; the runs below are left for future
     black-hole work ("Known risks"). The runs that would separate the causes: Wald at `a = 0`
     with `ξ = 0` at `M/8` and `M/16` to `50M` (the gauge damping alone, about
     45 min and 2.7 h on 64 threads); `a = M/2` with the heavy fluid at `M/16`
     (the fluid at the finest mesh); and a damping of `Φ̃ − aB₀`, the
     departure from the case's own `Φ̃`, which is a design change.
   - **The resets by layer** (`M/16`, `50M`): counted stage/step resets in
     every layer from the counting radius to the corner — `a = M/2`
     2.68·10⁹/1.01·10⁹ (1.32·10⁸/4.41·10⁷ deep inside), `a = 0.9M`
     2.75·10⁹/1.00·10⁹ (7.68·10⁷/2.56·10⁷); the atmosphere (level 2) with
     `ρ_FLOORED` and `S_CEILED` in most of them inside `5.5M`, `S_FLOORED` alone
     beyond; `W_CEILED` 10⁶ and 3·10⁶ times, and `NONFINITE` (level 1, no closest
     state; level 2 puts the atmosphere there) 1.1·10⁴ and 2.2·10³
     times over the run (`M/8`: 6.9·10³ and 6.7·10³; none with the heavy
     fluid). The step limiter's injected `τ`: −7.2·10⁴ and −2.3·10⁵ over the
     octant (`M/8`: −1.6·10⁶, −5.1·10⁶). All the floor fluid's; the field
     does not see them but through the charge's convection.
   - **∇·𝔅** at rounding in every run (the curl is metric-free): 1.05–1.47e−16
     of `max |𝔅|` (job 588706, both spins, `M/4` to `50M`, `M/8` to `20M`).

   **Curved-space bit identity** (`test/curved_tests.jl`): with the step
   cache (fused, the default) and the interior, 16 steps of `Δt = 1/64` in
   one integration, in two, and as a restart from the saved state with a
   rebuilt problem are the same bit for bit, for both cases; the suite prints
   their checksums, which the one- and four-thread runs print alike. The
   `· inv(√γ)` keys of "The step cache", written for curved space and first
   exercised here, hold.
4. **TOV in the Cowling approximation**: a Γ = 2, K = 100 polytrope
   at fixed entropy (needs EOS-1, EOS-2) on `Analytic(TOVMetric)`, with no
   TreeGH; stationarity and mode frequencies.

   `tov_case` on `[−24, 24]³` (4 roots, one level
   of boxes from `tov_boxes!`, `half = 1.25 R̄`), SSP3(4,3,3), HLLE with
   light-cone speeds, the reset as both limiters; `bin/tov_study.jl`
   writes the central density (`central_density`, the largest cell
   average of `ρ`) every `M`, and `bin/tov_drift.jl` takes the drift
   `ρ_c/ρ_c(0) − 1` against the same mesh's initial cell average (`ρ_c(0)`
   is 1.24834e−3, 1.26575e−3, 1.27195e−3, 1.27641e−3 at `h = 0.75`, 0.5,
   0.375, 0.25; the star's own is 1.28e−3). Symmetry, amdpreq, 64 threads, the node per job
   in the logs.

   **Stationarity** (PLAN.md: the drift after `200M` converges at `≥ 1.5`;
   it holds for both reconstructions):

   | | `h = 0.75` | `0.5` | `0.375` | `0.25` | orders |
   |---|---|---|---|---|---|
   | MC drift at `200M` | −1.0505e−1 | −3.7809e−2 | −1.4533e−2 | −4.6686e−3 | **2.52, 3.32, 2.80** |
   | MC, mean over `[160, 200]M` | −9.580e−2 | −3.381e−2 | −1.405e−2 | −5.071e−3 | 2.57, 3.05, 2.51 |
   | MC L1 change of `D`, `τ` | 2.70e−6, 1.06e−7 | 1.09e−6, 3.83e−8 | 5.93e−7, 1.80e−8 | 2.71e−7, 6.42e−9 | |
   | PPM drift at `200M` | −4.1806e−2 | −1.4123e−2 | −6.9735e−3 | | **2.68, 2.45** |
   | PPM, mean over `[160, 200]M` | −3.975e−2 | −1.378e−2 | −6.950e−3 | | 2.61, 2.38 |
   | PPM L1 change of `D`, `τ` | 3.83e−7, 1.94e−8 | 1.21e−7, 7.19e−9 | 6.24e−8, 3.73e−9 | | |

   (Jobs 591972, 592404, 592405, 593184 for MC; 591973, 592040, 592041 for
   PPM; MC at `h = 0.25` took 57875 s on cn101, EPYC 7543.)
   The drift is a slow monotone loss of central density with the radial
   modes on top, not a growth: MC's `h = 0.75` run reaches −60.5% at
   `2000M` (job 591971, cn086). At each resolution PPM drifts 2.1–2.7 times
   less than MC, and its L1 change of `D` is 7–10 times smaller. MC's
   order scatters about 2.8 (`0.5 → 0.25`: 3.02, of the mean 2.74): the
   drift converges above second order, so it is the scheme's truncation
   error, not a boundary or atmosphere effect of fixed size.

   **Conservation** (PLAN.md: rest mass to round-off, net of the outer
   boundary flux; it holds): in every run the change of the total
   `D` equals the mass the step limiter injected (`ohm_tally`'s
   `injected_D`) to `1.3–7.7·10⁻¹⁵` of a rest mass of 1.506 at `200M` and
   at MC's `2000M` (4.2e−15), and to 3.1e−14 at `h = 0.25` (5400 steps on
   eight times `h = 0.5`'s cells, accumulated rounding), so nothing leaves
   through the outer faces. The injection is `+1.3` to `+1.9·10⁻⁷` with MC
   and `−3.5` to `−4.2·10⁻⁷` with PPM: PPM's reset removes mass, MC's adds
   it.

   **Modes** (PLAN.md: F, H1 and H2 within 3% of Font et al. 2002, Table I,
   the fixed-spacetime values of this model from a 2D code: 2.701, 4.563,
   6.352 kHz; it holds at `h = 0.375`; H2 fails at
   `h = 0.5`). Nothing perturbs the star: the modes are the initial data's
   truncation error. `bin/tov_modes.jl` on the PPM series (the trend
   removed, Hann window, the largest interior local maximum within ±20% of
   each reference, refined by a parabola), both over `[0, 886]M` (0.23 kHz
   resolution; jobs 592406 and 592407, checkpointed through 16 and 7
   pre-emptions, their first `200M` the uninterrupted runs' bit for bit):

   | | F | H1 | H2 |
   |---|---|---|---|
   | `h = 0.5` | 2.665 (−1.33%) | 4.605 (+0.92%) | 5.845 (**−7.97%**) |
   | `h = 0.375` | 2.677 (−0.89%) | 4.563 (+0.00%) | 6.492 (+2.20%) |

   The detrending degree (1–3) moves them by at most 0.05%. F and H1
   approach Font et al.'s values with resolution; H2, with two radial
   nodes inside `R̄ = 8.1` (16 cells of `h = 0.5`), is the first to fail.
   The modes are damped by the scheme's dissipation, faster on the coarser
   mesh: F's peak over `[0, 500]M`, `[500, 1000]M` is 7.3e−5, 4.6e−6 at
   `h = 0.5`; over `[0, 500]M`, `[386, 886]M` 5.2e−5, 1.1e−5 at `0.375`.
   Past that the windows hold noise (at `h = 0.5` beyond `1000M` every peak
   is off by 5–19%), so the 2000M and 1000M runs the plan named were
   stopped at 1918M and 886M; the 2000M run's whole series gives the same F
   and H1 to 1% (2.645, 4.643) and H2 at −8.4%. Over `200M` alone (1 kHz
   resolution) H1 is +3.7% and H2 +8.1% at `0.375`: the 3% needs `≳ 500M`.
   MC is no substitute: its `h = 0.75` star decays (−60.5% by `2000M`) and
   F drifts with it, 2.15, 1.76, 1.42 kHz over `[0, 500]`, `[500, 1000]`,
   `[1000, 2000]M`.

   **What the star decides.** The questions the Cowling star settles (the
   record of how they were posed and decided is in
   [HISTORY.md, "Open questions"](HISTORY.md#open-questions)):
   - **The Riemann wave speeds**: the light cone for all variables; the
     split speeds fail on the Cowling star ("Riemann solver").
   - **The conservative prolongation width for the matter at stellar
     surfaces**: width 3, bounded (TreeAMR's limited operator, AMR-4). The
     Cowling star with a second box level of half-width `0.7 R̄`, so that
     the coarse-fine faces cut through the star and its surface
     (`bin/tov_study.jl N=8 levels=2 half=0.7`, finest `h = 0.375`, MC,
     200 M; the primitives' ghost fill `ops_W` at refinement boundaries;
     Symmetry jobs 591967, 592035, 592036):

     | `ops_W` | `ρ_c` at 200 M | L1 change of `D` | rest mass the reset injected | budget |
     |---|---|---|---|---|
     | width 3, bounded | −1.77% | 8.5·10⁻⁷ | +2.3·10⁻⁷ | −1.5·10⁻¹⁵ |
     | width 3, unbounded | −4.39% | 1.4·10⁻⁶ | **−0.143** | −1.8·10⁻⁵ |
     | width 1 | −4.44% | 3.2·10⁻⁶ | +2.9·10⁻⁷ | −9.3·10⁻¹⁶ |

     With the unbounded operator the reset takes out 9.5% of the star's
     rest mass of 1.506 in 200 M (its overshoots at the faces, presumably;
     not localised), at about 7·10⁻⁴ per M after the first 25 M, where the
     bounded operator and width 1 inject 10⁻⁹ per M, as the uniform mesh
     does. Its budget (the change of the rest mass net of what the reset
     injected) is 1.8·10⁻⁵ rather than rounding: mass through the outer
     faces or through the coarse-fine faces, not separated. Width 1 is
     stable but twice as diffusive as the bounded width 3 in the central
     density and four times in L1.
   - **The atmosphere reset's stage-limiter role**: the reset keeps both
     roles. On the Cowling star the stage role does not matter
     (`bin/tov_study.jl reset=step`, `h = 0.75`, MC, 200 M, Symmetry jobs
     591972 and 591973, on an EPYC 7543 and an EPYC 8534P): the central
     density `1.1172021167·10⁻³` at 200 M, the L1 changes of `D` and `τ`
     (`2.7043·10⁻⁶`, `1.0563·10⁻⁷`) and the rest mass the reset injected
     (`1.9354·10⁻⁷`) agree to every printed digit with the reset as the step
     limiter alone and as both, although the stage limiter wrote 2.7·10⁸
     cell-stages (where, not recorded). The default stays both roles: the
     star does not decide between them, and the stage role is what keeps
     every right-hand side's input physical.
   - **The reconstruction the stars run with**: PPM. MC is the plan's and
     GRaM-X's choice; PPM alone of the four resolved Balsara 3's dense shell
     and held every Balsara plateau but test 2's `R5` (2.8%) within 2% (see
     "Validation ladder" 1b); WENO-Z smeared the shell worst, and MP5 costs
     twice the steps. On the Cowling star PPM's central density drifts
     2.1–2.7 times less than MC's at every resolution (−4.18%, −1.41%,
     −0.70% at `200M` against −10.5%, −3.78%, −1.45% at `h = 0.75`, 0.5,
     0.375), its L1 change of `D` is 7–10 times smaller, and both converge
     above second order. PPM costs about 12% more per cell-step: 8.68 µs
     against 7.72 µs for MC with the reset as the step limiter alone (the
     same run as both roles to every digit), in one job on cn112 (EPYC
     8534P, job 591973); no pair with the default reset ran on one node.
     PPM needs `G = 3`, `mhd_fields`' default. The stars' scripts run PPM
     (`bin/tov_cowling.jl`'s default); the package's default stays MC (the
     tubes and waves of rungs 1–2 were measured with it).

5. **TOV in a dynamical spacetime**: the frozen TreeGH metric, through its
   adapter, reproducing rung 4; the F, H1, H2 radial
   modes against the literature (Font et al. 2002); convergence of the GH
   and ADM constraints with matter (needs GH-2 to GH-4).
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

**Inconsistent:** "limited prolongation once upstream has it" is listed as
not available, but "Known risks" and "Validation ladder" 4 say the matter's
operator is bounded, TreeAMR's limited operator (AMR-4); and the split
speeds are listed as an extension, but they are built
(`riemann_flux(::Val{:hlle_split}, …)`, "Riemann solver") and measured.

## Open questions

Each is decided by a measurement in the step named in `PLAN.md`. The
questions already decided, and when, are in
[HISTORY.md, "Open questions"](HISTORY.md#open-questions); those the
Cowling star settled are under "Validation ladder" 4, "What the star
decides".

- **(open)** `(ρ, S_i, S_ij)` to vertices: 2nd order (default) or 4th.
- **(open)** TreeGH's CFL factor under the SSP3 explicit tableau.
- **(open)** The metric in the outermost ghost cell (see
  [Metric at cells, faces and edges](#metric-at-cells-faces-and-edges)).

  **Inconsistent:** "Metric at cells, faces and edges" says the outermost
  ghost layer is settled by `Mv`'s `G = 4`.
- **(open)** Whether `D s` earns its keep; the threshold that switches to
  it. On the stationary Kerr-Schild flows it drifts further from
  `D s_exact` than `D` does (see "Validation ladder" 3). The reset writing
  `√γ D s = (√γ D) s` from the recovered `s` in every cell it did not
  repair, in both roles, was tried and not adopted: it breaks the
  conservation of `D s` and the reset's no-op where it repairs nothing
  ([HISTORY.md, "Open questions"](HISTORY.md#open-questions) has the
  measurement). Without it, `(√γ D s)/(√γ D)` leaves the window where `D`
  is thin — −410 … 1220 outside the Cowling star's surface at 200 M ("As
  built in step 10, round 2") — which only seeds the recovery and changes
  nothing it returns. It earns its keep only once it is the fallback of a
  failed energy inversion, which is not built.
- **(open)** For a hole that forms in a collapse, with no analytic
  solution inside: a field-only interior — paste or damp `𝔈` and `A`
  inside, and leave the matter to the ordinary reset, which rung 3 found
  sufficient for the matter. Not measured; for the step that forms a hole.
- **(open)** The potentials' outer boundary where no exact `A` exists
  (dynamic cases): a gradient-extrapolating relative hook (option 2,
  upstream) or a gauge that damps a static `χ` (option 3); see
  "Boundaries", the design rule. The Cowling TOV star is static, so an
  exterior field from an exact static potential takes
  `static_potential = true` (option 1) as is; options 2 and 3 matter for
  dynamic cases only.

## Known risks

- **Spacetimes with black holes are not expected to work out of the
  box** (the same holds for TreeGeneralizedHarmonic). Rung 3 validates the
  scheme on analytic Kerr-Schild backgrounds with the hard paste: Wald at
  `a = 0` and the magnetized Bondi flow converge at second order to
  `t = 50M`. What it also shows is left for future work on black holes:
  - Without an interior, a vacuum field near the singularity fails: Wald at
    `M/8` and `M/16`, earlier on the finer mesh.
  - The spinning Wald case's drift grows late (`a = M/2` after `35M`) or
    steps near the ergoregion (`a = 0.9M`), so its `50M` rates on the
    finest pair fall below 1.8.
  - At `a = 0.9M` the Euclidean paste ball misses the inner horizon.
  - A floor fluid under `Vacuum` is reset in almost every cell every stage.
  - The potentials need a boundary where no exact `A` exists.

  See "Validation ladder" 3 and "The interior".
- No MPI yet: single node, although TreeAMR now has its M7.
- Unlimited conservative prolongation at a stellar surface floors cells.
  The matter's operator is bounded; the atmosphere tube
  across a coarse-fine face repairs no ghost with it, where the unbounded
  one made `D < 0` (see "Refinement" and "Validation ladder" 4).
- Light-cone fluxes are diffusive in a star's interior. They smear
  contacts and slow waves: Balsara 2's contact converges as
  `h^{2/3}`, and Balsara 3's dense shell (ρ = 10.1 between a contact and a
  slow shock) is not resolved below 800 cells, where the 400-cell run's
  peak density is 8.35.
- The implicit stage may converge poorly where magnetization and `σΔt`
  are both large; rung 1's sweep exists to find out where. Measured
  pointwise: not where both are large — the ideal seed is exact
  there — but at `b²/ρ ≥ 10` with `σΔt` between `10⁻³` and `10²`, when `E★`
  is so far from both the solution and the ideal field that neither seed is
  a fluid state. The
  [magnetization cutoff](#ohms-law-and-the-conductivity) takes σ to zero
  there, and with it (at the measured safe `μ_max = 10`) no fallback
  remains at `f ≤ 0.3`; the price is that above the taper the field
  evolves as in electrovacuum, not as force-free — no conduction current
  at all, where a magnetosphere's plasma would carry the force-free one —
  and the taper already begins at `μ = μ_max/10`, so σ is reduced a decade
  below the threshold a case states. Rung 1 sees how often an evolution
  meets the taper.
- `q := ∇·E` turns numerical divergence of `E` into charge.
- MC floors the entropy of a few cells behind
  the cylindrical explosion's front (the reset writes them; minmod, GRaM-X's
  choice there, writes none), and MP5 is unstable above `cfl = 1/5`. A
  strong blast into a light medium is where the reconstruction's
  overshoots reach the recovery.
