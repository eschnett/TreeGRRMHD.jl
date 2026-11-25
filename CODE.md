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

*Status: built and validated. The seven rungs of the [validation
ladder](#validation-ladder) are done, and the scheme runs on a GPU
([Precision and devices](#precision-and-devices)); how the package was
built, step by step, and who decided what, is in [HISTORY.md, "The steps of
the plan"](HISTORY.md#the-steps-of-the-plan).
- **Flat space and curved backgrounds** (rungs 1–3): smooth waves, shock
  tubes and the stiff limit in 1D, 2D and 3D, across coarse-fine faces and
  regrids, and on Kerr-Schild backgrounds, the hole's interior relaxed to
  the analytic solution.
- **The TOV star** on the Cowling metric (rung 4) and in TreeGH's evolved
  spacetime (rung 5): F within 1%, H1 within 1.5% from `h = 0.6`, H1 and H2
  within 1% of Font et al. at `h = 0.25`; the constraints converge at 1.8 or
  better to 1000 M but for `ℋ` outside at 1.76 over one 20 M window; the
  rest mass changes by 1.3e−12 of 1.506 over 1000 M, entering with the
  atmosphere at the outer faces.
- **The magnetized star** (rung 6), coupled and on the Cowling metric:
  `∇·B` at round-off; the exterior dipole's drift converging at 1.4–2.6
  once σ is zero below `ρ = 10⁻⁶`; the literature's confined-field star
  losing its central field as GRaM-X's and Dionysopoulou et al.'s do.
- **The star on a real table** (rung 7; DD2, SFHo, cold β-equilibrium, `Yₑ`
  advected): on the Cowling metric the drift converges at 1.5–2.6 and the
  rest mass and `D Yₑ` hold to round-off over 200 M. In the evolved
  spacetime TreeGH's Dirichlet outer boundary grows the star's oscillation
  on rung 5's box (`L = 24`); on a box four times as wide it is damped, and
  the constraints converge at 1.8–3.0 on the finer pair to 1000 M except
  outside the star around 100 M.
- **Devices**: a CUDA run matches the CPU to 26 ulp in `ρ_c` over 200 M;
  `Float32` runs on CUDA and Metal; one and four threads agree bit for bit.
- **MPI** ([Running distributed](#running-distributed)): a run over one,
  two or three ranks of two threads is the serial run bit for bit, its sums
  to rounding, and a checkpoint written at one rank count restarts at
  another into the uninterrupted run.

What is not built, or measured short, is under [Known risks](#known-risks)
and [Possible extensions](#possible-extensions); the upstream items still
waiting (TreeGH's registration and regrid verbs, each with a workaround,
and EntropyEOS's registration and fixed-entropy inversion) are under
[Requests still waiting](#requests-still-waiting). The numbers below about
the scheme are measured where they say so and unmeasured otherwise.*

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
  - [Requests still waiting](#requests-still-waiting)
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
  - [Threads](#threads)
  - [Devices](#devices)
  - [The per-phase table](#the-per-phase-table)
  - [The tabulated star's cost](#the-tabulated-stars-cost)
- [Running distributed](#running-distributed)
  - [The communicator](#the-communicator)
  - [What this package makes global](#what-this-package-makes-global)
  - [Checkpoints](#checkpoints)
  - [Threads and devices under MPI](#threads-and-devices-under-mpi)
  - [Distributed tests](#distributed-tests)
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
- **Not yet a production code.** It runs on threads, devices and MPI ranks
  in any combination and checkpoints through TreeIOHDF5 ([Running
  distributed](#running-distributed)), but no production-size run has been
  made distributed yet, and how many ranks a node should hold is not
  measured.

## Upstream prerequisites

What this package needs from its siblings, and why. The exact
requirements — API, semantics, the tests the sibling should carry, and the
check this package adds when an item lands — are in [Requests still
waiting](#requests-still-waiting), under the IDs used here, for the items
still waiting; for those landed, withdrawn or superseded they are in
[HISTORY.md, "Upstream prerequisites"](HISTORY.md#upstream-prerequisites),
with how the state below came about. The bullets below keep the reasons.
The priorities "(req)", "(wanted)" and "(later)" are defined under
[Requests still waiting](#requests-still-waiting).

The state, as `[compat]` guarantees it for what has landed, as
`test/prerequisite_tests.jl` reports it for what is waiting, and, for
TreeGH, as its `main` reads. A landed item whose release the compat floor
requires has no probe any more (the rule is under [Requests still
waiting](#requests-still-waiting)): AMR-1 to AMR-10, EOS-1, EOS-2, EOS-4
and EOS-8 to EOS-11, and SM-1. Their names are in
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
  pins the tag `v2.3.0`, with `[compat]` `"2.3"`: **EOS-1 and EOS-2 have
  landed in EntropyEOS 1.0.0**, EOS-8 to EOS-11 since, and **EOS-4 in
  2.3.0** (below); EOS-3 has not.
- IMEXRungeKutta, not registered, through `[sources]` at its `main`, with
  `[compat]` `"1.4"` (1.4.0 is the tag its design was finished at).
- SpacetimeMetrics **1.7.0**, registered (`SpacetimeMetrics = "1.7"`; it
  requires Julia 1.11, below the floor here, 1.13):
  **SM-1 has landed in SpacetimeMetrics 1.7.0**, and its workaround is
  gone.
- TreeGeneralizedHarmonic **0.1.1** on its `main`, with `adm_vars` and
  `ADMMetric`, and Adapt as a new direct dependency, on TreeAMR 0.2, not
  registered: GH-0 has landed (the step-8 work is on `main`); GH-1 is
  landed, `ADMMetric`, TreeGH's PR 25 (61bcb75); GH-3 (79ad132) and GH-4
  (0cd5f00) have landed on its `main`; GH-2 is open, GH-7
  is wanted, each with a workaround; GH-5 is withdrawn. TreeGH is not a
  dependency of this package at all, so the package's suite cannot detect
  its items: `coupled/`, an environment of its own, loads TreeGH through
  `[sources]` beside this checkout, and its suite reports them ("The
  coupling contract").

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
- **AMR-2 (wanted, step 6) Per-set boundary hooks in `regrid!`**: one hook
  for every set, since an `AllVariables` hook is length-checked against
  each set's `nvars`, so one hook cannot serve this package's sets.
  *Landed in TreeAMR 0.2.1*; `regrid_pairs` passes each evolved set its own.
- **AMR-3 (wanted, step 8) `adapt_to_initial_data!` over several sets**.
  *Landed in TreeAMR 0.2.1*; `adapt_case!` adapts every set at once.
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
  TreeGH's list). *Landed in TreeAMR 0.2.1, commit a0a977b:*
  `map_blocks!(kernel!, fs, args...; workgroupsize = nothing)` hands it to
  the KernelAbstractions launch on a device and ignores it on the CPU, where
  the workgroup is the block, so a caller may pass it on every backend.
  `MHDProblem` and `evolve!` take it ("Devices").
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
- **AMR-14 (wanted, no step blocked) The curve split by a weight per
  leaf**: the threads' chunks and the ranks' ranges are equal counts of
  leaves, and on a tabulated star a block's cost varies by four: the star's
  recoveries cost 5 times the atmosphere's, so the threads and ranks that
  hold the star set the pace ("The tabulated star's cost"). No workaround.
- MPI (M7) and checkpoints (M9, now TreeIOHDF5's) are used
  ([Running distributed](#running-distributed)), with TreeAMR's public
  communicator verbs (AMR-13, landed in TreeAMR 0.2.11; `HISTORY.md`,
  "Upstream prerequisites"); point interpolation (M11) is not used yet.

This is a **physics package** in TreeAMR's sense (its `CODE.md`,
"Companion interface"): it uses TreeAMR's exports and its `public`
communicator verbs, with one exception — `tov_boxes!` reads `Forest`'s
field `leaves` —, and bounds TreeAMR by a minor series,
`TreeAMR = "0.2.11"`, the release the landed items are in (the public
communicator verbs, AMR-13, the last). The prerequisite tests check the
names it calls against the exports and the public names, and `src/device.jl` uploads its own metadata with
`KernelAbstractions.allocate` rather than reaching for TreeAMR's internal
`todevice`.

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
  not exported: this package reaches outflow through `Forest`'s `outflow` keyword only. Nothing
  here uses outflow faces yet; the first case that does tests them in place.
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
  target is in [HISTORY.md,
  "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic).
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
  **Landed on TreeGH's `main`** (79ad132, still 0.1.1, unregistered):
  `rhs!(du, p, t; matter)` and `adm_constraint!(p, u, t; matter)`, the
  adapter bound once per call as `at = matter(p.U)` and called as
  `(; ρ, S, Sij) = at(n, b)` at an owned vertex `n` of local block `b`.
- **GH-4 (req, step 11) A right-hand side callable inside an external
  integrator**: TreeGH's own verbs under the names this package shares —
  `TreeGH.prepare!`, `TreeGH.rhs!`, the integrator-free limiters, the
  speed for `Δt` — all accepting segment views of a multi-set state
  vector, and `evolve!` as a composition of them. A script calls them;
  neither package drives a coupled run. **Landed on TreeGH's `main`**
  (0cd5f00): `prepare!(p, u, t) -> p`, `rhs!`, `stage_limit!`,
  `step_limit!`, `initial_state!(u, p, t)` and `evolved_sets` are `public`;
  `max_speed(p, u, t)` is exported. A segment must be a contiguous view.
- **GH-5 (withdrawn) A ghost width of at least `q/2 + 1`**, so
  that a 4th-order vertex→cell metric interpolation reaches every stored
  cell of a `G = 3` matter set. Not needed: the interpolation reads this
  package's own vertex set `Mv` at `G = 4`, and TreeGH's ghost width does
  not enter the contract.
- **GH-6 (later)** A damped-harmonic gauge driver, radiative or
  constraint-preserving outer boundaries, excision or a generalized
  interior for collapse, Ψ₄ extraction.
- **GH-7 (wanted, step 11; workaround) The regrid verbs and the
  constraint norms of a segment**: `TreeGH.flags`, `regrid_pairs`,
  `rebuild` and `constraint_norms(p, u, t; matter)`, so that the coupled
  script regrids TreeGH's set with this package's in one `regrid!` and
  monitors a segment's constraints. Until then the script composes them
  from TreeGH's exports (`coupled/gh7.jl`; the request: [GH-7](#gh-7--the-regrid-verbs-and-the-constraint-norms-of-a-segment-wanted-workaround)).
- **GH-8 (note) Constraint damping on a star.** No change requested: at
  TreeGH's near-hole `γ0 = 1/M` the star grows a mode, and this package
  uses `γ0 = 1/(10M)` ("Two scripts", "As built";
  [the note](#gh-8--constraint-damping-on-a-star-note)).

The CFL factor of TreeGH under SSP3(4,3,3)'s explicit tableau is *not* an
upstream item: this package owns the integrator, and step 11 measured the
factor ("Time integration"). TreeGH's own `cfl = 1/4` was measured with
RK4; whether to keep it is TreeGH's decision.

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
- **EOS-4 (landed, EntropyEOS 2.3.0)** The cold, neutrinoless
  β-equilibrium slice of a table, keyed by pressure, for tabulated-EOS
  stars (`cold_beta_slice`, `beta_equilibrium`;
  [Thermodynamics](#thermodynamics), "The cold barotrope"): at each
  pressure the composition of least Gibbs free energy, with any phase
  transition a density jump at constant pressure, so that `ρ★` never
  decreases with `p`. Upstream because which zero of `μ_ν` a star takes,
  and how `μ_ν` follows from a table's chemical potentials, is the EOS's
  knowledge; the fitted `μ̃` is off by MeV on the cold edge.
- **EOS-12 (wanted; workaround)** The atmosphere decided without level 1:
  `apply_policy`'s decision for an input below the trigger, public, and its
  output without a level-1 output, for the skip of level 1 there
  ("Primitive recovery", "The two levels").
- **EOS-13 (wanted)** A state below a table's cold edge recognised without
  the Newton budget and the bracket scan, which cost 32–41 µs on DD2,
  where its atmosphere and the cells truncation cools take that path.
- **EOS-14 (wanted)** A warm evaluation without sampling its temperature
  bracket's ends, two of its five or so samples, on the evaluations that are
  a little over half of a tabulated step.
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
  `--check-bounds=yes`, where it allocated one `MVector{33,T}`
  (272 B at `Float64`, 144 B at `Float32`). EntropyEOS 2.2.1 keeps the
  scan's scratch on the stack under bounds checks, bit for bit.

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

### Requests still waiting

The exact requests for the items above that have not landed. The format
is the one every item was written in:
- **what**, and the **exact API**;
- **semantics**;
- **upstream acceptance**: the tests the sibling should carry;
- **downstream check**: the testset `test/prerequisite_tests.jl` (or, for
  TreeGH, `coupled/prerequisite_tests.jl`) gains. Until the item lands, it
  is skipped with an `@info` naming the ID. It detects the item by
  `isdefined` / `hasmethod`, never by version number.

**Priorities.** **req** blocks the work named; **wanted** has a
workaround, listed with the item and marked in the code with the item's
ID (no other workaround is allowed); **later** is not needed by anything
built. An item lands in a **registered release** (or a pushed commit where
registration is pending), never only in a local checkout; additive changes
stay in the same minor series.

**Retiring a probe.** Once an item has landed *and* `[compat]` requires a
release that contains it, its testset is deleted from
`test/prerequisite_tests.jl`. The names the item added join the
prerequisite test's name lists. A property of the release this package
relies on and does not yet exercise through its own code moves to
`test/upstream_behaviour_tests.jl` until a test uses it in place. The
landed item's request and record move to [HISTORY.md, "Upstream
prerequisites"](HISTORY.md#upstream-prerequisites), and its bullet above
says what this package relies on from it.

#### AMR-11 — The relative hook consistent across levels at an outer face (wanted)

**Why.** With the potentials' conservative transfer (AMR-9) the magnetic
flux through a coarse face on a coarse-fine face is the sum through its
fine faces to rounding (1.4e−16 of the largest flux in a periodic 3D box)
— except where the face's perimeter lies on an outer face. There each
level's relative hook writes its own wall and ghost edges,
`f(x_g) + (u(x_r) − f(x_r))` with `x_r` that level's nearest evolved
point, so a coarse wall edge is not the mean of the fine ones beside it
(1.2e−5 of the largest flux in the closed 3D blast box, 45 of 192 faces;
2.7e−5 after a regrid). The same holds for any set with symmetric walls
and a conservative transfer: at the wall the levels are filled
independently.

**API.** None new; a semantic of `CellBoundary(Relative(f))` (or an
opt-in): a point that lies both beyond (or on the wall of) an outer face
and on a coarse-fine face is filled from the level that owns that plane by
the set's own transfer — restricted from the fine level where the fine
block owns it, prolongated from the coarse level's hook-filled values
where the coarse one does — rather than by the hook on each level.

**Upstream acceptance.** On an edge set under `Conservative` with
`vertex_prolongation`, symmetric walls and a relative hook, on a two-level
mesh whose coarse-fine face meets an outer face in D = 2, 3: the line
integral of every coarse edge on the wall equals the sum of its fine
edges' after a fill and after a regrid; one fill is still the condition's
fixed point.

**Downstream check.** `test/refinement_tests.jl`'s state-flux testset
asserts the closed box's mismatch to rounding instead of recording it.
AMR-11 has no probe in `test/prerequisite_tests.jl`: a semantic change has
no name to detect, and that testset is its check.

**Workaround:** none needed: the flux through a face at a Dirichlet
boundary is the boundary condition's, which nothing conserves ("Conservation
at coarse-fine faces").

#### AMR-14 — The curve split by a weight per leaf (wanted)

**Why.** `equalsplit` cuts the curve into contiguous parts of equal leaf
counts, for the threads (`threadchunks`, `launch_by_owner!`) and for the ranks
(`blockrange`, the repartition). On the DD2 star a block's recovery costs up to
3.9 times the mean block's — a star cell's 4.0 µs against an atmosphere cell's
0.7 — and the parts holding the star set the pace: at 64 threads a step's
threads are asleep 23% of their samples, the recovery phases scale 18–34
times on 64 cores where the fluxes scale 53, and over 1, 2, 4 and 8 nodes the
slowest rank's recovery takes 1.23, 1.58, 1.73 and 1.84 times the mean rank's,
waiting on it about 0.45 s of the 0.76 s a step loses on 8 nodes (Trillium and
Fir; "The tabulated star's cost"). TreeAMR's own `equalsplit` says a
cost-weighted split would change only that function.

**API** (exported or `public`).
```julia
set_leaf_weights!(forest, w::AbstractVector{<:Real})   # one per local leaf, ≥ 0; collective
leaf_weights(forest) -> Vector                         # the weights in use, 1 by default
```

**Semantics.**
- The threads' chunks and, at the next repartition (`regrid!`, or a
  `repartition!(forest, sets…)` the application calls), the ranks' ranges
  are contiguous ranges of the curve whose weight sums differ by at most one
  leaf's weight, instead of leaf counts that differ by at most one.
- Weights travel with the leaves through a regrid (a child inherits its
  parent's weight divided among its children, a parent its children's sum)
  until set again; with all weights equal, today's split bit for bit.
- Results do not depend on the split (TreeAMR's existing claim at every
  thread and rank count), only the time does.

**Upstream acceptance.** With unequal weights, the parts' sums are balanced to
one leaf's weight; equal weights give today's chunks and ranges; a weighted
run is bit for bit the unweighted one at 1, 2 and 3 ranks.

**Downstream check.** `set_leaf_weights!` exists (`hasmethod`); then the DD2
star's `bin/benchmark.jl` rows, with the weights a step's measured per-block
recovery time.

**What it does not fix.** A part cannot be lighter than its heaviest leaf: on
the 120-block mesh at 64 threads the measured split by weight leaves the
slowest part 3.3 times the mean (3.9 by count), and at 16 it is 1.3 (2.2).
There, more and smaller blocks are the remedy, which this package's `N` and
roots already choose.

#### EOS-0 — Register EntropyEOS (req; workaround)

Register EntropyEOS in General. Until then `Project.toml`'s `[sources]`
names `url = "https://github.com/eschnett/EntropyEOS.jl"` at a tag (today
`rev = "v2.3.0"`), which CI, the clean-checkout check and a developer all
read.

#### EOS-3 — Public fixed-entropy inversion and entropy from energy (wanted)

**API.**
```julia
con2prim_entropy(eos, in::Con2PrimIn, s, opts::Con2PrimOptions, w_guess=NaN, u_guess=NaN)
    -> Con2PrimOut      # in.τ ignored
entropy_from_energy(eos, ρ★, yₑ, U; u_guess=NaN, max_iter=60) -> (s, pt::EOSPoint, converged::Bool)
```

**Semantics.**
- `con2prim_entropy` wraps EntropyEOS's internal `inner_solve_w`.
  - `result = converged_fallback` when `|f1| ≤ tol·cosh w`, otherwise
    `failed_max_iter`.
  - It sets a distinct flag when `V ≥ tanh(w_max)` pinned the rapidity.
- `entropy_from_energy` wraps the internal `seed_s_solve`. It reports
  convergence instead of silently pinning to the window.

**Upstream acceptance.**
- A round trip at fixed `s` on random states.
- Pinning and non-convergence are reported, never silent.

**What waits for it**: the entropy fallback ("Primitive recovery").

#### EOS-12 — The atmosphere decided without level 1 (wanted; workaround)

**Why.** `apply_policy` sends every finite input with `D < atm_trigger·ρ_min`
to the atmosphere whatever level 1 returned, and on a table such inputs are
most of a run's atmosphere and its dearest recoveries (EOS-13). A caller
that skips level 1 there needs the decision and the output EntropyEOS's own
call would give, without its unexported internals.

**API** (exported).
```julia
policy_decides(eos, in::Con2PrimIn, proj::ProjectionOptions, pol::PolicyOptions) -> Bool
apply_policy(eos, in, nothing, proj, pol) -> Con2PrimSafeOut   # where policy_decides
```

**Semantics.**
- `policy_decides` is true exactly where `apply_policy` writes the atmosphere
  for every level-1 output, and level 1 would run a solve: a finite input,
  `B² ≥ 0`, `proj_ρ_min ≤ D < pol_atm_trigger·proj_ρ_min`.
- `apply_policy(…, nothing, …)` there returns the state `apply_policy` writes
  after `con2prim_safe`, bit for bit, with `policy_flags = ATMOSPHERE`,
  `solved = false`, and a documented `base.result`.

**Upstream acceptance.** On random inputs either side of both bounds, where
`policy_decides`, the state, `cons` and the `EOSPoint` equal
`apply_policy(con2prim_safe(…))`'s bit for bit; elsewhere it is false.

**Downstream check.** `policy_decides` and the three-argument form exist
(`hasmethod`).

**Workaround** (marked `EOS-12` in `src/con2prim.jl`, `policy_decides`
and its two callers): the bounds from the unexported `proj_ρ_min` and
`pol_atm_trigger`, and `apply_policy` on the unexported `nan_safe_out(T)`,
whose `policy_flags` are 0. Only for the skip of level 1 in "The two levels"
("Primitive recovery"); without it, nothing.

#### EOS-13 — A state below the cold edge found without the Newton budget (wanted)

**Why.** A state colder than the table's coldest row at its density —
`τ` below the energy at `s_min` for its `D` and `S` — has no root in the
window. `con2prim_start` spends its whole Newton budget against the `s_min`
face (30 iterations, `failed_no_bracket`) and `con2prim_safe_finish` then
scans for a bracket and solves on the edge to report `S_FLOORED`: on DD2
13–15 µs and 19–26 µs, against 0.5–3 µs for a state inside (M3 Pro, one
thread). A star on a cold β-equilibrium slice lies on that row, so its
atmosphere and the cells truncation pushes colder all take this path.

**API.** None new: `con2prim_safe_start` and `con2prim_safe_finish` faster on
those states.

**Semantics.**
- The same output bit for bit, or, if the edge solve must start elsewhere,
  the same face and bits with the state within the solver's tolerance.
- The test is cheap: for instance, the Newton stops once its `s` iterate is
  clamped to `s_min` and the energy residual keeps its sign (colder) for two
  iterations, and `con2prim_safe_finish` goes to the `s_min` edge without
  the bracket scan where the diagnosis already says the energy lies below
  it.

**Upstream acceptance.** States below the `s_min` edge of a real table (DD2,
SFHo) at densities from `ρ_min` to the core, at rest and moving, recover to
`S_FLOORED` as today, with iteration counts and a timing printed.

**Downstream check.** None by `hasmethod`; `bin/benchmark.jl case=dd2`'s
`stage_solve` and `reset` rows, and the counts of `S_FLOORED`.

#### EOS-14 — An evaluation without its bracket's ends (wanted)

**Why.** `evaluate(eos, ρ, s, yₑ, u_guess)` on a table solves `s(x, u, y) = s`
for the log-temperature `u` (`solve_u`), and samples both ends of the
extended bracket first, before its warm start is used: two of the five or so
B-spline samples of a warm evaluation, whose result (the entropy window)
only `evaluate_windowed`'s callers read. On the DD2 star the face states
and the ghosts' `P` are a little over half of a step and spend half of their
`solve_u` on those two samples (the profile of `bin/profile.jl`).

**API.** None new: `evaluate` faster.

**Semantics.** From a warm start, `solve_u` iterates first and samples an
end of the bracket only when an iterate leaves it or the iteration stalls;
the same converged `u` to the solver's tolerance, the same flags.
`evaluate_windowed` keeps its window.

**Upstream acceptance.** `evaluate` on a real table at random states, from
cold and from warm starts at several distances, agrees with today's to the
solve's tolerance, with the samples counted.

**Downstream check.** None by `hasmethod`; `bin/benchmark.jl case=dd2`'s
`face_fluxes` and `fill_primitives` rows.

#### GH-2 — Registration, registry-satisfiable compat (req; workaround)

TreeGH is not a dependency of this package ("The coupling contract"); the
item is for `coupled/`, the environment that loads both.

- Register TreeGeneralizedHarmonic and KorzynskiSpin in General.
- Every `[compat]` entry must be satisfiable from registered releases
  alone. `[sources]` of a dependency is ignored when TreeGRRMHD resolves
  it, so the TreeAMR features TreeGH uses must be in a released TreeAMR.
  TreeGH is on the registered TreeAMR 0.2. What is left, as of TreeGH's
  fbfd416: TreeGH itself, TreeIOHDF5 (for its checkpoints) and
  IMEXRungeKutta are unregistered, and its `ApparentHorizonFinder = "2.1"`
  is ahead of the registry, which has 2.0.0. KorzynskiSpin 1.1.0 and
  AbstractSphericalHarmonics are registered.
- The Julia floor is 1.13 for every Tree* package ([Julia
  version](#julia-version)); nothing is asked of TreeGH.
- **Workaround until registered:** `coupled/Project.toml`'s `[sources]`
  names TreeGH, ApparentHorizonFinder, IMEXRungeKutta and EntropyEOS by
  URL; `coupled/prerequisite_tests.jl` reports the item.

#### GH-6 — Later

A damped-harmonic gauge driver, radiative or constraint-preserving outer
boundaries, excision or a generalized interior for collapse, and Ψ₄. The
radiative boundary is the one a coupled run now has a measured need for:
TreeGH's Dirichlet outer boundary grows the tabulated star's oscillation
at `L = 24` ("Validation ladder" 7).

#### GH-7 — The regrid verbs and the constraint norms of a segment (wanted; workaround)

What a coupled script needs beyond GH-4 to regrid TreeGH's set together
with this package's in one `regrid!`, and to monitor the constraints of a
segment. TreeGH's `CODE.md` ("The verbs (GH-4)") lists them as not built.

**API** (module-qualified, `public`, not exported; the names are this
package's, so that the script reads the same for both packages).
```julia
TreeGH.flags(p::GHProblem, u, t; buffer = 0, travel = 0) -> flags
                     # gh_indicator!(p, u, t; buffer, travel) on a segment `u`
TreeGH.regrid_pairs(p::GHProblem, t) -> Tuple
                     # (p.U => (p.schedule, dirichlet(p.case, t)),) and every other set of
                     # `p` that lives on the forest, `=> nothing`
TreeGH.rebuild(p::GHProblem, t) -> GHProblem
                     # after `regrid!`: a new GhostSchedule with `p`'s operators, `Hsrc`
                     # resampled, `diag` reallocated, every keyword of `p` carried over
TreeGH.constraint_norms(p::GHProblem, u, t; matter = nothing, mask = …) -> NamedTuple
                     # adm_constraint!(p, u, t; mask, matter), then constraint_norms(p)
```

**Semantics.**
- `flags` and `constraint_norms` scatter `u` (a contiguous segment, as for
  GH-4) and so invalidate a preceding `prepare!`, like every monitor.
- `regrid_pairs` returns pairs for TreeAMR's `regrid!` over several sets:
  the caller concatenates them with this package's and calls `regrid!`
  once. The hook is the one `evolve!`'s `regrid_mesh` passes today.
- `rebuild` is `regrid_mesh`'s rebuild without the `regrid!`, the
  `statevector` and the limiter, which the script does over every set. It
  does not run `stage_limit!`: the script does, on the gathered segment.

**Upstream acceptance.**
- `evolve!`'s regrid, rewritten as `flags`, `regrid!` over
  `regrid_pairs`, `rebuild`, `statevector`, `gather!` and `stage_limit!`,
  is bit-identical to today's on a refined hole fixture.
- On a segment of a two-set state vector, each verb equals the standalone
  call.

**Downstream check** (in `coupled/`). The script's regrid branch, with
TreeGH's verbs in place of the workaround, is bit-identical to the
workaround's.

**Workaround** (marked `GH-7` in `coupled/gh7.jl`): the four composed in
the script from TreeGH's exports (`gh_indicator!`, `dirichlet`,
`GHProblem(U, schedule, case; q)`, `GhostSchedule`, `adm_constraint!`,
`constraint_norms(p)`), reading `GHProblem`'s fields `U`, `schedule` and
`case`.

#### GH-8 — Constraint damping on a star (note)

A measurement for TreeGH. No change is requested; what to do with it is
TreeGH's call. Written, not sent.

**What was found.** The star is rung 5's (`K = 100`, `Γ = 2`,
`ρ_c = 1.28e−3`, `M = 1.400`, `R = 9.586`), evolved by TreeGH with a
frozen sampled `H_a`, `ε_KO = 1/2` and a constant `γ0`. At `γ0 = 1/M`,
the near-hole value of `GHCase`'s docstring, a mode grows: it doubles
every 25 M (an e-fold of about 36 M) at `h = 0.6` and at `h = 0.375`. The
rate does not change with resolution, and the amplitude is 2.5 times
smaller at the finer `h`: the truncation error seeds it at second order.
In the coupled run the star collapses near 400 M. TreeGH alone shows the
same growth, on the initial star's matter held fixed (GH-3's adapter).

The largest change of TreeGH's state, TreeGH alone on the frozen matter,
at `h = 0.6` to 500 M:

| `γ0 M` | `γ2` | result |
|---|---|---|
| 0 | 0 | bounded; `ℋ` noisy, to 3e−4 |
| 1/20, 1/10 | 0 | flat, 1.6e−2 (6.7e−3 at `h = 0.375`); `ℋ` 3–5e−5 |
| 1/4 | 0 | grows, then saturates at 2.8e−2 |
| 1/2 | 0 | grows, 0.43 at 500 M |
| 1 | 0 | doubles every 25 M |
| 1 | −1/2 | bounded, 3e−2; `ℋ` 3e−4 |
| 4 | 0 | `NaN` at 105 M |

Neither `ε_KO = 0` nor a box twice as wide changes the interior. Coupled:
`γ0 M = 1/4` is stable to 250 M, `γ0 M = 1` with `γ2 = −1/2` to 390 M,
and `γ2 = 1` crashes at 80 M. This package uses `γ0 = 1/(10M)`,
`γ2 = 0`, `GaussianDamping`'s far-zone value for a hole, and `ℋ` stays
flat to 300 M. The runs: [HISTORY.md, "Two scripts"](HISTORY.md#two-scripts).

**Not measured:** which scale sets the threshold (`M`, `R`, or the
light-crossing time); the mechanism (the mode needs both the matter and
`γ0 ≳ 1/(4M)`, and `γ2 = −1/2` suppresses it); TreeGH's Plummer star
(`test/matter_tests.jl`, `star_case`), which uses `γ0 = 1`, that is
`10/M`, on short runs.

**Possibly relevant to TreeGH:** `GHCase`'s docstring and TreeGH's
`CODE.md`, "Gauge and constraint damping", give `γ0 ≈ 1/M` as the recipe
near a hole, which a matter case inherits unless told otherwise; long
matter runs at `γ0 ≳ 1/(2M)` may grow this mode.

**Reproducer** (the `coupled/` environment, about an hour on 64 cores):
```bash
julia --project=coupled -t 64 coupled/tov_coupled_study.jl N=10 t_end=400 γ0M=1
```

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
metric derivatives come from TreeGH (the two are checked against
each other in a test). Both are `geometric_source` in `src/sources.jl`, and
on boosted Kerr-Schild with fields up to 100 times the fluid's energy they
agree to 5.4e−16 of the magnitude of their terms; the covariant form also
agrees with `√−g T^{μν}(∂_μ g_νj − Γ^δ_νμ g_δj)` from SpacetimeMetrics'
Christoffels to 1.2e−15, and the energy source with Banyuls et al.
(1997)'s `√−g (T^{μt} ∂_μα − α T^{μν} Γᵗ_μν)` to 5.6e−16. **The covariant
form is the default**: every provider produces `∂ⱼ g_ab`
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
Whether `D s` earns its keep is not decided: it waits for the entropy
fallback, and that for EOS-3 ("Possible extensions").

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
otherwise produces spurious `B` at refinement boundaries. `ξ` is `16/L` by
default ("Refinement").

**The charge density is derived, not evolved**: `q := ∇·E`,
computed from the ghost-filled cell-centered `𝔈` in every right-hand-side
evaluation. Gauss's law then holds identically and charge conservation is
implied; there is no evolved `q` and no cleaning scalar for `E`. The known
cost is that any numerical divergence of `E` — at coarse-fine interfaces,
at the stellar surface — appears as charge and drives `q v` currents; the
charge is monitored as a diagnostic.

**A uniform background field** `𝔅̄ⁱ` is added to `curl A`. It
is what lets a vector potential represent a net flux through a periodic
box, and a field along a direction the mesh does not resolve: a 1D shock
tube's constant `Bˣ` is `B̄ˣ`, since no translation-invariant `A` produces
it. It is zero for stars. **The densitized `𝔅̄ⁱ` is the constant**,
not `B̄ⁱ`: with `√γ` varying, `∂ᵢ(√γ B̄ⁱ) = B̄ⁱ∂ᵢ√γ ≠ 0`, a
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

`σ` is an `isbits` model type:

| model | σ | use |
|---|---|---|
| `Vacuum()` | 0 | electrovacuum tests |
| `Constant(σ₀)` | σ₀ | resistive shock tubes, current sheet |
| `DensityProfile(σ₀, ρ_atm, …)` | σ₀ inside, → 0 toward `ρ_atm` | magnetized stars (Dionysopoulou et al. 2013) |
| `IdealConductor()` | ∞ | the exact ideal limit: `E = −v × B`, for Balsara 1–5 and comparison with GRaM-X |
| `MagnetizationCutoff(model, μ_max)` | `model`'s, → 0 where `B²/D → μ_max` | any finite σ, where the plasma is magnetically dominated |

The `q v` part is explicit. The `σ` part is **stiff** inside a star
(`σ Δt ≫ 1`), and is the implicit part of the time integrator.

the models are in `src/ohm.jl`, and
`conductivity(model, prim)` takes the whole primitive state, so that a
later model may read `s` or `Yₑ`. `DensityProfile(σ₀, ρ_atm, cutoff = 2,
power = 1)` is `σ₀ max(0, 1 − cutoff·ρ_atm/ρ)^power`: continuous, `σ₀`
deep inside matter and zero at and below `cutoff·ρ_atm`, so the atmosphere
carries no current. The literature's profiles are of this family. For a
magnetized star, Dionysopoulou et al. (2013, arXiv:1208.3487, Eq. 50) take
`σ₀ max(1 − D_atm/D, 0)²`, and so does GRaM-X's resistive module (Azizi et
al. 2025, arXiv:2510.18968, Eq. 61) — `power = 2`, `cutoff = 1`, there in
`D` rather than `ρ`. For binaries, Dionysopoulou, Alic & Rezzolla (2015,
arXiv:1502.02021, Eq. 22) take a smoothed step at `100 D_atm`, because
winds fill the domain, and call that choice arbitrary. Palenzuela (2013,
arXiv:1212.0130) switches to a force-free exterior between 50 and 400
`ρ_atm`. *The shape is kept*: the linear form stays the
default, since the squared one changes nothing measured ("Validation
ladder" 6). *The threshold is the star's, a density*: σ is
zero below `ρ = 10⁻⁶` on the magnetized star (about `8·10⁻⁴ ρ_c`), whatever
`ρ_min` is. At `ρ_min = 10⁻¹²` a threshold of `10⁴ ρ_min = 10⁻⁸` drifts as
`100 ρ_min` does at `10⁻¹⁰`, and `10⁶ ρ_min = 10⁻⁶` as `10⁴ ρ_min` does, to
three digits ("Validation ladder" 6), so the threshold belongs to the gas
outside the star, not to the atmosphere's floor. So the case takes it as a
density, `ρ_cut` (`magnetized_tov_case`'s default `10⁻⁶`, below),
through `DensityProfile(σ₀; ρ_cut, power)`, which is `ρ_atm = ρ_cut`,
`cutoff = 1`; the form `cutoff · ρ_atm` stays `DensityProfile`'s positional
one, for a model that does follow the floor. Changing `ρ_min` leaves the
star's σ unchanged, bit for bit ("Tests"). `10⁻⁶` is far
above every published threshold (`D_atm`, `100 D_atm`, 50–400 `ρ_atm`):
the atmosphere the star accretes reaches `10⁻⁸ … 10⁻⁶` outside its surface,
in the evolved spacetime as on a fixed metric, and at `10⁻¹⁰` or `10⁻⁸`,
linear or squared, that gas conducts, ties the exterior field to its motion
and leaves the drift unconverged; at `10⁻⁶` the exterior is electrovacuum
and the stage's fallbacks there all but vanish. Its cost: it turns off the
ideal limit in the star's own outer layers, the outer 0.0093 M of
`R = 9.586 M` (`ρ < 10⁻⁵` is the outer 0.092 M, `ρ < 10⁻⁴` the outer
0.83 M). A switch on the magnetization or the field's energy is the general
answer (open; "Open questions"). For `IdealConductor`
the conduction current is not a function of the state — it is whatever
holds `E = −v × B`, which the stage solve imposes — so `conductivity` is
`Inf` and `ohm_current` throws an `ArgumentError` rather than return
`∞ · 0`. Where `σ = 0` the conduction term is not formed (a mask
branches). Measured: at `E = −v × B` the current is 2.3e−16 of
`σ W |B|`, and in flat space the bracket is the fluid-frame field
`F^{ab} u_b` with the charge the normal observer sees, to 4.4e−16.
`conductivity_slope(model, prim)` is `∂σ/∂ρ`, for
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

*As built*: `MagnetizationCutoff(model, μ_max)` wraps
any finite-σ model and multiplies its σ by a taper `g(μ)` of the
magnetization `μ = magnetization(m, B, D) = γ_ij Bⁱ Bʲ / D` — the normal
observer's field over the rest-mass density, which bounds the fluid
frame's `μ/W ≤ b²/ρ ≤ W μ`. `μ = +∞` where
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
  it is `GeometricEOS(eos; density)` in
  `src/units.jl`, an `AbstractEOS` around an EOS in its own units whose
  densities are the scheme's: `evaluate` scales `ρ★` by the density unit on
  the way in and `U_ρ`, `U_ρs` (once), `U_ρρ` (twice) and `p` (inversely)
  on the way out, the entropy windows are the inner EOS's at the scaled
  density, and the density bounds shift by its logarithm. It forwards
  EntropyEOS's two optional methods too, `evaluate_windowed` and
  `entropy_from_energy`, so that a table's own cheaper ones are reached ("Warm
  starts" below has what their defaults cost), and `eos_on` uploads it to a
  device ("Devices"). The recovery's
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
- **Warm starts** (`u_guess`, the previous `log₁₀ T`). Every
  **recovery** is seeded from the state itself — `s` from the advected
  `(√γ D s)/(√γ D)` and no rapidity or EOS warm start (`state_guesses` in
  `src/evolution.jl`) — and not from `P`. At the one place the recovery calls
  EntropyEOS (`recovery_seed` in `src/con2prim.jl`) a seed from the state
  gains the **rapidity at zero pressure**, `atanh(|S|/(τ + D))`, the cold
  seed's own first pass, where EntropyEOS's input carries no field (`B² = 0`:
  the hydrodynamic recovery, and an ideal one without a field); and **an
  advected `s` outside the EOS's entropy window** at the seed's density
  `D/cosh w` is dropped, the whole seed with it, for EntropyEOS's cold seed,
  which solves for `s` from the energy. Such an `s` cannot be the recovered
  one, and on a table's cold slice it is the rule: degenerate matter's energy
  hardly depends on `s`, so a cell average's energy a hair above the slice is
  a much hotter state (the DD2 star's initial data: advected 0.30, recovered
  1.02, below the window in 33088 of its 35928 cells), and the Newton took 8.4
  iterations from it where the cold seed's start needs none. Per cell over the
  stars' initial data, at rest and at `v ≈ 10⁻²` (M3 Pro, one thread): DD2's
  star 4.0 and 5.5 µs (the advected `s` alone 10.2; with the rapidity 6.9 and
  7.0; EntropyEOS's cold seed 3.6 and 4.6), the polytrope's 0.69 and 0.75 µs
  (the advected `s` alone 1.8 and 2.4; the cold seed 1.4 and 1.6), the
  magnetized star's 0.70 and 1.08 µs. A seed with a rapidity (the step
  cache's, a stage iterate's) is a recovered state, inside the window, and is
  passed on unchanged. The cold seed is cheap on a table only because
  `GeometricEOS` forwards EntropyEOS's two optional methods,
  `entropy_from_energy` (one temperature solve) and `evaluate_windowed` (the
  window from that solve's bracket): without them it took the generic
  defaults, a bisection in `s` over the whole extended window with an
  evaluation a step, 58–75 µs a cold seed on DD2 against 3.2–3.6, and two
  more B-spline samples in every Newton residual. With a field the
  zero-pressure velocity is a cubic's root, left to the cold seed, once a
  step for `IdealConductor`'s stage solve on a magnetized state.
  A warm start carried in `P` makes
  each recovery depend, at the solver's tolerance, on whatever was evaluated
  before: the right-hand side would not be a function of `(u, t)` bit for
  bit (the plan's purity claim), and a restart from a checkpoint, which
  holds no `P`, would not reproduce the run. The face states, reconstructed
  from `P` within one evaluation, still take their cell's EOS warm start.
  A table's `log₁₀ T` is the warm start the recovery then lacks, and it is
  worth little: a forward evaluation of DD2 costs 0.6–0.9 µs warm and
  0.95–1.2 µs cold (four iterations against one), since every evaluation
  samples the ends of its temperature bracket and evaluates the result's
  derivatives whatever its start (EOS-14); a recovery seeded fully warm
  costs 0.52–0.58 µs. The right-hand side is pure in
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
- **The cold barotrope**. A table's star is in
  cold, neutrinoless β-equilibrium: EntropyEOS's slice (EOS-4,
  `cold_beta_slice`) gives at each pressure the composition of least Gibbs
  free energy on the table's coldest row, `s` on the window's minimum.
  `cold_barotrope` (`src/tov_table.jl`) samples it at 8001 pressures uniform
  in `log p`, converts with `GeometricEOS`'s density unit (`e = ρ★h − p`),
  and integrates `η = ∫ dp/(e + p)` by Simpson's rule from the surface; the
  TOV solve of `src/tov.jl` runs in this `η`, of which the polytrope's
  `log h` is the special case. Along the slice `Yₑ` and `s` change, so
  `log h` is not the integral (they differ by 1.8·10⁻⁴ between 10⁸ and
  10¹⁵ g/cm³ on DD2, 1.9·10⁻⁴ on SFHo), and only the integral makes the
  evolved fluid static (`dp = −(e + p) d log α`). Between the knots the
  state is cubic Hermite in `η` with `d log p/dη = (e + p)/p` exact
  (measured against central differences: 1.7·10⁻⁶). The slice's `Yₑ` has
  kinks at the table's density nodes, so the solve converges at fourth
  order only to about 10⁻⁹, ample at its default 4000 steps (4.5·10⁻⁹ in
  `M` against 16000). A slice with a phase transition is refused (none on
  DD2 or SFHo; [Known risks](#known-risks)). The units hold: DD2's 1.4 M☉
  star has `R = 13.225` km, SFHo's 11.896 km, with the crust to
  6.2·10³ g/cm³ (the literature's 13.2–13.3 and 11.9 km).
- **The tabulated star's surface and atmosphere**. The barotrope's surface, `η = 0`, is at the atmosphere's density
  `ρ_min`, and the atmosphere is the slice's own state there: cold
  (`PolicyOptions`' `s_atm = NaN`, the window's minimum) with the replaced
  state's `Yₑ`. The matter below `ρ_min`, which the reset would replace,
  is left out; with the surface lower (10⁻¹⁴, 6.2·10³ g/cm³) the
  composition table below cannot follow `ρ★`'s fall by four decades within
  its last knot (errors of 600 times at 6·10⁷ g/cm³), and with it at
  `ρ_min` the error is 3% pointwise at worst. The star's matter reaches a
  kernel as a `TOVComposition`, `log ρ★`, `s` and `Yₑ` at 129 knots
  uniform in `r̄²`, kept apart from the metric so that the metric's kernel
  arguments do not grow ([Sharp edges](#sharp-edges)): 2.3·10⁻⁵ in L1 in
  `ρ★` (1.8·10⁻⁴ in `Yₑ`, 7.8·10⁻⁴ in `s`), falling at second order in the
  knots; the pointwise 3% is where the slice's `Yₑ` changes steeply in the
  crust, which no smooth table follows.

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
magnitude from the returned rapidity, `vⁱ = tanh w · Sⁱ/|S|`.
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
measured against the energy density `𝓔 = τ + D`, of which they are
components, the round trip at EntropyEOS's default tolerance is about
10⁻¹² relative (worst 1.07·10⁻¹² and 1.21·10⁻¹², at EntropyEOS 2.0's five
seed passes). That is the solver's stopping criterion — `con2prim` stops
when its energy residual is below 1e−12 of `τ`, often at the cold seed with
no Newton step — and not this layer's rounding, and it reaches the 1e−12
of the step-1 acceptance. So
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
recovery runs in a kernel and lands within 2.3e−5, conditioned, of the
state (EntropyEOS's `Float32` tolerance is 6.1e−5) on states dense enough
that the thermal energy is a percent of `τ`; at `ρ = 1e−9` the gas's `U` is
1e−7 of its rest mass and `Float32` cannot determine `s` at all.

`con2prim_hydro` is split at the fluid's share — `field_budget`
subtracts the fields' energy and momentum, `fluid_recovery` recovers from
what is left and also returns the EOS point — so that the stage solve,
which iterates on that share, calls the same recovery bit for bit.

`con2prim_hydro` returns EntropyEOS's policy flags as they are, and only
them: `policy_flags`, not the solver's result, is EntropyEOS's validity
signal, and a failed solve is always repaired and flagged. The
one decision this package adds to the policy — a state whose density the
policy had to raise to the atmosphere's is the atmosphere at rest — is
EntropyEOS's own level 2, below, and the flags of an atmosphere keep
level 1's bits ("Floors, the atmosphere and the magnetosphere").
EntropyEOS's conserved output is never used: every caller rebuilds the
conserved state from the primitive one with this package's `prim2con`.

**The two levels**. EntropyEOS 2.0 splits its policy in two, and both
recoveries call both, the second right after the first, in
`fluid_recovery` and `ideal_recovery` (on a device, outside the stage
solve's iterations, level 1 runs as EntropyEOS's two phases and level 2 in a
kernel of its own; on the CPU both run in one kernel; "The two-phase
recovery" below):

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
- **Level 1 skipped where level 2 decides alone.** Where
  `ρ_min ≤ D < atm_trigger·ρ_min` and the input is finite, `apply_policy`
  writes the atmosphere whatever level 1 returned, and builds it from the
  input alone (its `Yₑ` from `D_Y/D`, its EOS point without a warm start), so
  the recovery hands it EntropyEOS's unsolved placeholder instead of
  solving (`policy_decides` in `src/con2prim.jl`, both recoveries, fused and
  two-phase): the same state bit for bit. On a table almost every atmosphere
  cell is such an input: the atmosphere lies on the table's coldest row, the
  gravitational source gives it an infall speed at fixed `τ`, which leaves it
  colder than the table, and level 1 then spends its whole Newton budget (30
  iterations) and the bracket scan to report `S_FLOORED`: 13–15 µs and
  another 19–26 µs on DD2, against about 1 µs for the atmosphere's own EOS
  evaluation (M3 Pro). On M1 at four threads (M3 Pro, `bin/benchmark.jl`, the
  check switched off against on, two rounds each, interleaved) the skip takes a
  step from 13.4 to 10.0 µs a cell (25%): `recover` from 0.89 to 0.45 µs,
  `stage_solve` from 0.76 to 0.32. What the skip gives up is level 1's bits on
  those cells — `S_FLOORED`, sometimes `ρ_FLOORED` — so their flags are
  `ATMOSPHERE` alone, which then says "below the trigger", an exception to the
  rule that the bits say why. Below `ρ_min` level 1 decides by `D` without a
  solve and keeps its `ρ_FLOORED`. Until EntropyEOS answers this itself
  (EOS-12) the check reads its unexported `proj_ρ_min`, `pol_atm_trigger` and
  `nan_safe_out`. The alternative that keeps the bits is EOS-13, a cold edge
  found without the Newton budget, which also serves the star's own floored
  cells; when it lands, the skip is weighed against it again.
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

**Entropy fallback** (not built: a [possible extension](#possible-extensions),
waiting for EOS-3). The design: where
`τ_fluid = τ − ½(E² + B²)` is a small difference of large numbers
(electromagnetically dominated cells, the magnetosphere, a magnetized star's
cold surface) or the energy inversion fails, invert at the advected `s`
instead. The switch is a threshold on `½(E² + B²) / τ_fluid`, its value not
chosen. *Not
built:* the fixed-entropy inversion is EntropyEOS's internal
`inner_solve_w`, and this package neither adds a second inversion nor calls
an internal one, so the fallback waits for EOS-3's public
`con2prim_entropy`. What it would act on is counted
(`field_diagnostics`, `src/field_diagnostics.jl`): `trigger_*`, the cells,
the atmosphere left out, where `½(E² + B²) > τ_fluid` (the threshold of the
2026-10-16 investigation below); and `repaired_*`, the cells whose energy
inversion the policy repaired on a level-1 face, the atmosphere left out —
the failures the fallback is for. On the magnetized star the trigger fires
only outside the star, in the gas it accretes, and the energy inversion
fails outside the atmosphere in at most 10 cells at a time over 50 M
("Validation ladder" 6). The precedent is Palenzuela (2013), whose
magnetized stars fall back to the isentropic `p = Kρ^Γ` near the surface;
Dionysopoulou et al. (2013) treat a failed cell as atmosphere, as this
package's policy does, and GRaM-X's resistive module has no such fallback.
*(Investigation 2026-10-16, [HISTORY.md, "Floors, the
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
`test/step_cache_tests.jl` (warm against cold — the final states 7.6·10⁻¹⁵
to 1.7·10⁻¹² apart, the recovered `ρ`, `s`, `w` at most 3.0·10⁻¹², the same
branches and injected mass, the same resets; but for cells at the atmosphere's
threshold, which the tolerance may put on either side of it: in the magnetized
atmosphere tube 3 of 36163 resets apart and its `ρ` 4.9·10⁻¹¹); one chunk = two; a restart = the run;
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
second phase is the divergent one; compacting it is not done.

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
- **Both paths stay.** `MHDProblem(…; two_phase = false)` and
  `evolve!(…; two_phase = false)` build no hand-off and run each recovery in
  one kernel, `con2prim_safe` and `apply_policy` together (the same kernels
  with `nothing` for the hand-off); `two_phase = true` runs the three passes.
  `test/two_phase_tests.jl` holds the two bit for bit — every
  state and every tally of seven runs (the EM wave, Alfvén, the resistive
  tube, the magnetized atmosphere tube with the step cache on and off, the
  rotor, the adaptive rotor), and every callback outside a step on a state
  with a cell below `ρ_min`, one below the coldest state, one beyond the
  rapidity cap and a non-finite one, under the tube's conductivity and under
  `IdealConductor`. No CPU/GPU switch beyond this one.
  **The default is the backend's** (`two_phase = nothing` in `MHDProblem`
  and `evolve!`), picked from the array type as the integrator's partition
  is. Fused on the CPU: there the split costs 1–10% a step and buys nothing,
  since only 0–2.4% of the recoveries reach the second phase and none of the
  stage solve's iterations would (measured below). Split on a device: on an
  H200 it is 14–15% faster a step ("Devices").

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
reused. So a GPU sees the second phase in at most about 2% of a
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
| `X` — per-cell tallies, light speed, the reset's injection (`D`, `τ`, `D Yₑ`), `h∇·𝔅`, the recoveries' seeds per site, the entropy floor's count and `τ` | cell | 34 | 0 | `=> nothing` |
| step cache — the cell's latest recovery in the step: EntropyEOS's input and output; `MHDProblem`'s, not `MHDFields`' | cell | 27 | 0 | not passed: a `rebuild` makes a new one, invalid |

`P`'s 30 slots are laid out in `src/variables.jl`: the eleven the recovery
returns — `ρ★`, `vⁱ`, `s`, `Yₑ`, the rapidity `w` (also the next recovery's
warm start), `p`, `h`, `cs²` and the EOS's own warm start — then the policy
flags (as a number; every bit is below 2¹⁸ — bits 8–17, `FLAG_POL_W_CEILED`
the highest, since EntropyEOS 2.0 — so exact even in `Float32`, which holds
every integer below 2²⁴, and tested at `Float32` with bit 17), `Eⁱ`, `Bⁱ`,
`q`, and the metric `α`, `βⁱ`, `γ_ij` and `√γ`, all undensitized. The metric
derivatives and `K_ij` the sources read are not stored, and `P` stays at 30
slots. The sources are their only reader, once per cell per
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
- **`X`, the diagnostics set**: per owned cell, the
  implicit stages that set each of the nine `OhmStageResult` bits, the
  stages whose state the policy repaired and the `|` of their flags, the
  recoveries and the most one stage made, the cells the reset wrote, and the
  light speed for the time step; the cells the reset
  wrote as the step limiter, and the change those writes made to `√γ D`,
  `√γ τ` and `√γ D Yₑ`, summed — the rest mass, energy and lepton number
  the reset injects; `h∇·𝔅`; three seed slots per recovery site; and the
  cells where the entropy floor fired with the `τ` it injected
  (`src/fieldsets.jl`). The counts accumulate over a chunk and are
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

Reconstructed at faces: `ρ`, `s`, `Yₑ`, `uⁱ = W vⁱ` (so that
every reconstructed state is subluminal), `Eⁱ`, and the two **transverse**
components of `B` from their cell averages. The **normal** component of
`B` at a face is `B_d` itself, never reconstructed. Methods, in order of
arrival: PLM with minmod and MC (`G = 2`), then PPM, WENO-Z and MP5
(`G = 3`, which the table above already provides). A face state that is
unphysical (negative `ρ`, `s` outside its window) falls back to first
order, per face and per variable group. EOS evaluation at face states uses
the cell's warm start.

*As built: PLM and the face state* (`src/reconstruction.jl`): twelve
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
evaluates at the face (`Flat`, `Analytic`, or `VertexMetric`
interpolating from this package's vertex set `Mv`). A face's state costs two EOS evaluations, `primitive_state`'s and
`prim2con`'s.

*As built: the higher orders* (`src/reconstruction.jl`). `limiter` names the reconstruction: `:none`, `:minmod`,
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
variables at every face. Because `S_i` and `τ` are totals, their
fluxes contain the Maxwell stress, which propagates at the speed of light;
the default wave speeds are therefore the local light cone,
`λ_± = −β^d ± α √(γ^{dd})`, for all ten variables — the choice of the
resistive codes this design follows, robust and diffusive. Split speeds
(fluid speeds for `D`, `D Yₑ`, `D s`, light speed for the rest) are measured
below and are not the default; a resistive HLLC is not built ("Possible
extensions").

*As built*: `src/flux.jl` and `src/riemann.jl`. A face state is the tuple
`(cons, prim, E, B)` — the densitized conserved state, the primitive state
it was built from, and the contravariant fields — which is what
`physical_flux(m, cons, prim, E, B, Val(d))` takes.
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

Upwind constrained transport: at each `d`-edge the EMF
`αE_d + (β × B)_d` is the average over the four edge states reconstructed
from the surrounding cells (for `E`) and faces (for `B`), plus upwind terms
proportional to the jumps of the face-normal `B` reconstructed to the edge
from either side — the HLL form of Del Zanna et al. (2003) and Londrillo &
Del Zanna (2004), with light-cone speeds. It dissipates `B` consistently
with the LLF/HLL dissipation `𝔈` receives from the Riemann solver; without
it the Maxwell subsystem is upwinded in `E` alone. The metric at edges
comes from 1D 4th-order interpolation between the edge's vertices.

*As built*: `uct_hll_emf(m, E, Bᵃ, Bᵇ, Val(d))` in `src/emf.jl`.
With `(d, a, b)` cyclic, its four states are the full `E` of
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

*As built on the mesh* (`emf_kernel!` in `src/evolution.jl`), at
every edge of the closed range: the four states are the cells on
either side of the edge along `a` and along `b` (along a missing direction,
the one cell), each with its `E` extrapolated to the edge by the limited
slope along each present transverse direction, `E + σₐ slopeₐ/2 + σ_b
slope_b/2`; `𝔅ᵃ` is reconstructed along `b` from the `a`-faces and `𝔅ᵇ`
along `a` from the `b`-faces, with the same limiter, both **densitized**,
and divided by the edge's `√γ`. The flux of `𝔈` reconstructs the
undensitized cell average instead; the two agree in flat space and differ
at truncation order otherwise; the curved-background rungs have not
separated it. In 1D
the EMF along `x` is the cell's own `αE_x + (β × B)_x` (no transverse
direction to reconstruct along), and those along `y`, `z` are the 1D HLL
fluxes of `𝔅ᶻ` and `𝔅ʸ`.

Under every reconstruction the corner value of `E` is
additive, `E₀ + δ_a + δ_b` with the method's one-dimensional offsets — for
PLM exactly the `E + σₐ slopeₐ/2 + σ_b slope_b/2` above — and not a tensor
product (reconstruct along `a`, then those values along `b`), which would
read 25 cells instead of 9 for a cross term that is `O(h²)`, the scheme's
order in more than one dimension anyway. Measured on
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
atmosphere alone: every repaired state is exactly
invertible and is what the next right-hand side would recover anyway, so
writing it makes the state say what the scheme computes with. It writes the
seven matter slots and never `𝔈`, and counts each cell it writes into `X`.
A written state recovers as itself, so a second reset changes nothing (a
test).

The two roles are counted apart — `reset_atmosphere!` is the
stage limiter, `reset_step!` the step limiter — and the step limiter also sums
what each write changes in `√γ D`, `√γ τ` and `√γ D Yₑ` (`ohm_tally`'s
`injected_D`, `injected_τ`, `injected_DYₑ`, volume-weighted), and counts
the entropy floor's firings and their `τ` (`s_floors`, `s_floor_τ`). Since only its writes reach the state, the
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

Two ideas from that investigation are not built ("Possible extensions"):
a predictor in the atmosphere taking `OHM_SIGMA_ZERO`, and near-null
predictors in the stage solve's sweep.

## Time integration

### The partition

One IMEX Runge–Kutta integrator over the multi-set state vector. The
**explicit** part is everything except one term: all of TreeGH, the matter
and `𝔈` fluxes and sources, the EMF update of `A`, the gauge. The
**implicit** part is the σ current in Ampère's law, which is local to a
cell. The default tableau is Pareschi–Russo SSP3(4,3,3), whose implicit
part is L-stable, so that `σ Δt → ∞` relaxes to ideal MHD. IMEXRungeKutta's
SSP2(2,2,2) can be passed as `tableau =`, but nothing here runs or tests
it ("Possible extensions"). Neither is stiffly accurate (their weights
`b` differ from the last row of the implicit `A`), so the order *in* the
stiff limit may drop; IMEXRungeKutta's validation measures it, and the ARS
schemes, which are stiffly accurate, are the alternative if it matters.
`Δt` is fixed per chunk from the light-speed CFL on the finest level. In a
coupled run it is the smaller of the matter's step and TreeGH's,
`cfl_gh · h_min / λ` with `λ = GH.max_speed`.

`cfl_gh = 1/2` is half the largest factor that is
stable under SSP3(4,3,3)'s explicit part on the TOV star, with the star's
matter held fixed:
- **The star:** stable up to 1.10, on a grid in steps of 0.05. At 1.15 the
  error grows 7.7-fold, and at 1.25 it is `NaN`. This was measured with
  TreeGH's damping at `γ0 = 1/M` and not repeated at today's `1/(10M)`
  ("Two scripts", "As built").
- **TreeGH's gauge wave:** stable to the end of the grid, 1.5, under both
  SSP3(4,3,3) and RK4. The wave varies in one direction only.
- **The source of the runs:** `coupled/gh_cfl.jl`;
  [HISTORY.md, "Time integration"](HISTORY.md#time-integration) has them.

On the star, the matter's light-speed bound is the smaller of the two
either way. At `cfl_gh = 1/4`, TreeGH's bound is 6.48/h steps per M
(`λ = 1.62`), and the matter's is 6.55/h (2/5 over `3·rate`, with
`rate·h = 0.873`). So the factor changes none of the studies' step counts.

The driver is
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
`chunk_count` tolerance; `cfl = 2/5` (TreeHydro's, under
SSP3(4,3,3)'s explicit part, which is SSPRK(3,3)). The stage arithmetic is
split evenly over the threads (`partition = :even`) for a CPU state and
broadcast on a device; the block-owner partition, which needs TreeAMR's
unexported `threadchunks`, is not used ("The per-phase table"). A run in one
chunk and the same run in four chunks of whole steps of the same `Δt` are
bit-identical (a test).

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

Per cell, per stage. Known: `D`, `S_i`, `τ`, `D Yₑ`, `D s`, the
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

**The solve writes only `E` into `U`**. No limiter has seen
`u★`, so its fluid part may be inadmissible. `con2prim_safe` repairs the
solve's own view of it, and that repair stays local to the solve. Written
into `U`, it would become part of the implicit increment `U − u★`. The
integrator would then re-weight it by `b_k/a_kk`, which is 2.758 at
SSP3(4,3,3)'s last stage (IMEXRungeKutta's `CODE.md`, "The callback
contracts" and "One step"). PLUTO's resistive solve does the same: it
repairs its predictor and writes back only `E`.

**As built** (`src/stage_solve.jl`):
`solve_ohm_stage(eos, opts, proj, pol, model, m, cons★, E★, B, γΔt,
guesses; stage) -> (E, prim, result, recoveries)`, with `proj` the
recovery's `ProjectionOptions` and `model` the conductivity model,, allocation-free and bit-identical in a
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
- **The velocity is carried as `u = W v`**: every `u` is
  subluminal, and the Jacobian through `v` has a `W² (E★ − E) vᵀ` term
  from `∂W/∂v` that through `u` becomes `(E★ − E) vᵀ/W`.
- **The Jacobian is analytic.** `∂u/∂X` is the recovery's derivative by the
  implicit function theorem on its two equations `|S| = z tanh w`,
  `τ = z − p − D` in `(s, w)`, with EntropyEOS's own `z_w`, `z_s`, `p_w`,
  `p_s` from the EOS point the recovery returns; `∂E/∂u` is implicit
  differentiation of the linear system. Newton's `1 − PQ` (4×4,
  `P = ∂X/∂E · ∂E/∂u`, `Q = ∂u/∂X`) is inverted by Woodbury through the 3×3
  `1 − QP`, and a density-dependent σ adds a rank-one term through `∂ρ/∂X`
  and `conductivity_slope(model, prim) = ∂σ/∂ρ` (a model reading
  `s` or `Yₑ` would add a column each). Both derivatives agree with central
  differences to the differences' own error, 1.1e−8 and 8.3e−8. No extra
  recovery, no finite difference.
- **Which branch**: σ evaluated at the predictor's recovery, at
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
- **Converged** when `|X − X(E)| ≤ 10⁻¹²(τ + D)`
  (`max(1e-12, 1024 eps(T))`): the share closes to EntropyEOS's own
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
- fixed `Δt`, generic arrays and this package's Julia floor;
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
  does not make the names ambiguous. `prepare!` makes a
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
- **The contract uses only what both packages already depend on**.
  A contract owned by the consumer, as ApparentHorizonFinder's
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
- **The coupled script and its tests live in `coupled/`**, an environment
  of its own (decided by Erik 2026-11-10): TreeGRRMHD by path,
  TreeGeneralizedHarmonic and its unregistered dependencies by URL in its
  `[sources]` (GH-2 workaround). The package and its suite never load
  TreeGH; the script `import`s both packages and qualifies every verb, since
  each exports names the other does too (`evolve!`, `hostcopy`,
  `paste_interior!`).
- A prepared state is reused when the state it was prepared from repeats.
  The script keys the fill on that state ("Two scripts", "The repeated
  fill").
- Not measured: whether the script overlaps TreeGH's ghost fill
  with this package's recovery on a device; and a thread partition over
  several sets aligned with their owners (TreeGH's `state_partition` is
  copied from TreeAMR's tests; an upstream item when a step needs it).

#### Two scripts

A run is then a script. The two below are the Cowling star's and the
coupled star's, as sketches; the coupled one is the Cowling one with
TreeGH's lines added, and that the difference is this small is what this
design claims and the coupled suite shows. Both are built
(`bin/tov_cowling.jl`, `coupled/tov_coupled.jl`), with the departures
listed after them; every `MHD.*` name below exists, and the TreeGH names
that do not are GH-7's. A comment names the piece of the code a line
comes from — `evolve!` and its `build()`,
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
    cfl_gh, cfl_mhd = T(1 // 2), MHD.default_cfl(:ppm)  # TreeGH's under SSP3
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

The step-11 sketch is `coupled/tov_coupled.jl` (step 11), its `main` run by
the coupled suite. Its departures from the sketch:
- **`import`, not `using`,** for both packages, and `IRK = TreeGRRMHD.IRK`.
- **`problems(T; N, levels, L, roots, limiter, q, refinement, reuse_fill, γ0M, γ2, δv, order, ρ_min, case)`** builds the
  mesh, the two problems, `Mv`, the stored matter set `am` and the initial state, apart from `main`, so
  that the suite and the studies build the same run. TreeGH's problem is
  `GHProblem(U, GhostSchedule(U, ops), case; q)` on a vertex set the script
  makes (TreeGH has no `GHProblem(forest, case; q)`), with
  `ops = Operators(; prolongation = q + 2, restriction = q + 2)`, TreeGH's
  own. Its case is `GHCase(T, TOVMetric; box, periodic, ε_KO = 1/2,
  γ0 = γ0M/M, γ2)` with `γ0M = 1/10` and `γ2 = 0`.
- **The damping `γ0 = 1/(10M)`** is TreeGH's far-zone value for a hole,
  used everywhere. TreeGH's near-hole value `1/M`, used everywhere, grows a
  mode of the star that e-folds in about 36 M at every resolution, seeded
  by the truncation error; the star then collapses near 400 M. The mode
  needs the matter (TreeGH alone in vacuum does not grow it) and the
  damping (at `γ0 = 0` the run is bounded). In TreeGH alone on the star's
  frozen matter, at `h = 0.6` to 500 M:

  | `γ0 M` | `γ2` | largest change of TreeGH's state |
  |---|---|---|
  | 0 | 0 | bounded; `ℋ` noisy, to 3e−4 |
  | 1/20, 1/10 | 0 | flat, 1.6e−2; at `h = 0.375`, 6.7e−3 |
  | 1/4 | 0 | grows, saturates at 2.8e−2 |
  | 1/2 | 0 | grows, 0.43 at 500 M |
  | 1 | 0 | doubles every 25 M |
  | 1 | −1/2 | bounded, 3e−2; `ℋ` 3e−4 |
  | 4 | 0 | `NaN` at 105 M |

  `γ0 M = 1/10` is the smallest flat constraint norm in the scan. Coupled,
  `γ0 M = 1/4` and `γ0 M = 1, γ2 = −1/2` are also stable (to 250 M and
  390 M), the latter with twice the `ℋ`; `γ2 = 1` crashes at 80 M. The
  runs are [HISTORY.md, "Two scripts"](HISTORY.md#two-scripts).
- **The step count** is the larger of TreeGH's, `chunk_count((t1 − t)·λ,
  cfl_gh·h_min)` with `λ = GH.max_speed`, and the matter's, step 10's
  `chunk_count((t1 − t)·3·rate, cfl)` in TreeGH's metric (`step_count`).
  `cfl_gh = 1/2` ("Time integration").
- **The fill is keyed** (`FillKey`, `reuse_fill = true`; "The repeated fill"
  below).
- **The record** is the central density and TreeGH's matter-corrected
  constraint norms, after `central_density` has prepared the matter.
- **The regrid** is `regrid_all`: the flags reduced to plain flags (each
  package's boxes dropped) and combined by `refine_wins`; one `regrid!`
  over TreeGH's set, the matter's five evolved sets and their scratch, and
  `Mv`; both problems and `Mv` rebuilt; then the state gathered and
  `step_limit!` run once, both packages' limiters, the matter's in the
  metric of the transferred state. `MHD.regrid_state!` is not used: its
  reset reads the case's own provider, not TreeGH's metric.
- **TreeGH's missing verbs** (`GH.flags`, `GH.regrid_pairs`, `GH.rebuild`,
  `GH.constraint_norms(p, u, t)`) are `coupled/gh7.jl`'s, composed from
  TreeGH's exports (GH-7 workaround).

What the sketches leave visible, on purpose:
- **The order is the data flow.** `MHD.prepare!` cannot run before
  `metric_at`, nor `GH.rhs!` before `MHD.prepare!`, because each needs the
  other's result. TreeGH's limiter runs before the matter's in both
  limiters, so that the matter reads the limited metric. TreeGH's limiter
  never runs inside `solve_imp!`, whose implicit tendency
  IMEXRungeKutta recovers as `(U − u★)/(γΔt)`.
- **The repeated fill is keyed.** Written plainly,
  the script fills TreeGH's ghosts and `Mv` eleven times per SSP3(4,3,3)
  step: four stage solves, three stage limiters, three right-hand sides and
  one step limiter. Each fill covers TreeGH's 20 variables at `G = 3` and
  `Mv`'s 16 at `G = 4`.

  How the key works:
  - `metric_at` keeps TreeGH's segment of the last state it filled, bit for
    bit, together with its `t`. When both repeat, it skips the fill
    (`FillKey`).
  - Any other TreeGH verb the script calls (`max_speed`, the flags, the
    constraint norms) prepares TreeGH itself, so the script drops the key
    after it (`forget_fill!`), before the next `metric_at`.

  Why a step fills five times:
  - At each of the three stages that read the explicit tendency, the two
    abscissae agree. So the stage limiter and the right-hand side see the
    stage solve's state: TreeGH has no implicit part, and its limiter
    changes nothing on the star (the case has no `bounds`).
  - The step limiter's fill and the next step's first stage solve's see
    the same state, but at different `t` (the first stage's abscissa is
    `α ≈ 0.24`, not 0). The key keeps both. A key that ignored `t` would
    be correct only for static boundary data.

  The run is the same bit for bit with the key and without (the coupled
  suite). Measured on the star with one box level (`coupled/coupling_cost.jl`;
  job 595135, EPYC 7543):

  | Run | One fill | A step, every fill | A step, keyed | Shorter |
  |---|---|---|---|---|
  | `N = 16`, 64 threads | 59 ms (120 ns per owned vertex) | 2.56 s | 2.08 s | 18.8% |
  | `N = 12`, one thread | 0.50 s | 16.2 s | 13.2 s | 18.6% |

  [HISTORY.md, "Two scripts"](HISTORY.md#two-scripts) has the alternatives
  the item named. `K` is defined in one TreeGH ghost layer only, so this
  package cannot read TreeGH's ghosts in place of its own fill.
- **The regrid is the script's.** One `regrid!` covers every set on the
  forest, since a block refines for all of its sets at once; the flags
  combine as `refine_wins`; and each package rebuilds its own problem.
  The TOV star's boxes are static, so step 11 exercises this branch in a
  test of its own.

### Metric providers

The kernels dispatch on a metric provider:

- `Flat()` — Minkowski, all `D`, no storage.
- `Analytic(m::SpacetimeMetrics.AbstractMetric)` — evaluated at the point
  (cell, face, edge or vertex), all `D`; the Kerr-Schild tests, and the
  Cowling TOV star as `Analytic(TOVMetric)`.
- `VertexMetric`, a producer's metric in [the coupling
  contract](#the-coupling-contract): this package's own 16-variable vertex
  set `Mv`, its owned vertices copied from a producer's ADM adapter —
  TreeGH's `ADMMetric` (GH-1, step 11), or in tests and studies
  `SampledADM`, a SpacetimeMetrics metric in the same form, which tests
  this package's side of the contract — its ghosts filled by TreeAMR, and
  interpolated here to every other centering ([Metric at cells, faces and
  edges](#metric-at-cells-faces-and-edges)). The 3+1 variables are decoded
  on TreeGH's side.

`Flat` and `Analytic` are in `src/metric.jl`; they answer
`point_metric(provider, t, x) -> Metric3` and
`metric_gradient(provider, t, x) -> MetricGradient`, and every provider,
`VertexMetric` included, answers the indexed forms
`(provider, t, x, Val(C), n, b)`; `VertexMetric` refuses the coordinate
forms with an `ArgumentError`. `MetricGradient` holds
`dg[j, a, b] = ∂_j g_ab` — the spatial derivative index **first**, as in
GHSO2, then the spacetime indices with 1 = `t` — and `K_ij`; the lapse and
shift derivatives are derived from it (`d_lapse`, `d_shift`, which agree
with ForwardDiff of the 3+1 split to 2.4e−16 of their terms). The
conversion from SpacetimeMetrics' derivative-last `dmetric` is written in
`src/metric.jl` and nowhere else. `K_ij` is computed there from
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
are one-dimensional runs; coupling to TreeGH is `D = 3` only.

### Metric at cells, faces and edges

The conversion is this package's ([The coupling
contract](#the-coupling-contract)).

**The vertex set**. `Mv` holds `(α, β^i, γ_ij, K_ij)`, 16
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
  q + 1` for first, so **width 6** (PointValue widths are even).
- **The outer boundary** (the shared plane `n = N + 1` and the ghosts
  beyond it) is this package's hook: Dirichlet from the case's own analytic
  provider where it has one, as the TOV star's static exterior;
  a case without one has no design yet ("Possible extensions").
- A snapshot: valid until the producer's state changes; a regrid replaces
  `Mv` with the other sets, and the script refills it from the rebuilt
  adapter.

**Interpolated within a block** from `Mv`:
- **Cells**: 4th-order tensor Lagrange, 64 points; `∂ᵢα`, `∂ᵢβ`, `∂ᵢγ` at
  cells by the staggered 4th-order derivative along `i` times the
  transverse interpolation, the derivative of the same interpolant; `K_ij`
  interpolated, not differentiated.
- **Faces**: 2D, 16 points. **Edges**: 1D, 4 points. **Vertices**: read.
- **The fallback**. Where the Lagrange result has `α ≤ 0`, or a
  `γ` that fails a Cholesky factorization, the point falls back to
  multilinear weights (8, 4 or 2 vertices), counted by `metric_fallbacks`
  (below). Multilinear weights make `(α, γ_ij)` a convex combination of the
  corners, so `α > 0` and `γ` positive definite survive, and an
  `(α, β, γ)` with those properties always rebuilds a Lorentzian metric;
  interpolating `g_ab` could not promise that. At a fallback cell the
  derivatives are the multilinear interpolant's too, 2nd order.

Stored once per right-hand side in `P` for cells; on the fly at faces and
edges. The derivatives and `K_ij` are not stored but evaluated where the
sources are (see "Field sets"). `Mv`'s `G = 4` settles the
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
  edge or vertex has no cell of its own.
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
last lowered with `γ_ij` — then carried to each owned vertex in one of two
ways. TreeGH assembles `T_ab` from them with the vertex's own `α` and `β`,
the metric it evolves (the earlier alternative: [HISTORY.md, "Matter to the
vertices"](HISTORY.md#matter-to-the-vertices)).

- **Second order:** the mean of the `2^D` cells around the vertex. It is
  a convex combination, so `ρ ≥ 0` holds at the vertex.
- **Fourth order:** an interpolation from the `4^D` cells around the
  vertex. The cells hold averages: `√γ 𝓔` and `√γ S_i` are `U`'s cell
  averages to round-off, and `√γ S_ij` is one to second order, since it is
  the pressure of the average. So the densitized values are interpolated
  from averages to the vertex's point value with the weights
  `(−1, 7, 7, −1)/12` along each direction, and divided by `√γ`
  interpolated from the cells' point values with `(−1, 9, 9, −1)/16`.
  Beyond a stellar surface the interpolation rings. Where it would put
  `𝓔` or a diagonal `S_ii` below the least of the eight nearest cells'
  values, the vertex takes the mean instead. On the
  star, the fallback fires only within a cell of the surface.

The fourth order matters ("Validation ladder" 5). The transfer was the
largest error in the matter TreeGH sees. On the static star at `h = 0.5`,
the mean is 1.2% of `e_c` off inside the star, and the interpolation is
3.3 · 10⁻⁴ off. In the coupled script at `t = 0`, `ℋ` inside the star
converges at 2.0 with the mean and at 3.93–3.98 with the interpolation,
which is 160–400 times smaller at `h = 0.6` to 0.3. So the coupled runs
use the fourth order: the script's `order` (2 or 4) defaults to 4. The
perturbed star's modes do not decide between the two: at `h = 0.25` both
put H1 and H2 within 1% of Font et al., the mean the closer ("Validation
ladder" 5, "The transfer order"). The unperturbed star's do: the mean's
start excites the spacetime's lines so strongly that H1 is lost among
them, where the fourth order finds it within 1.5% ("Validation ladder" 5,
"Modes"). The constraints decide too. From 250 M to 1000 M, `ℋ` outside
the star is 10–14 times smaller with the fourth order, and converges at 1.88
or better where the mean's fell to 1.74. The mean stays for comparison
and for the per-call adapter.

This is this package's matter adapter. TreeGH binds it to its vertex set
and calls it at its owned vertices (GH-3). It comes in two forms:

- **`TreeGRRMHD.ADMMatter(p)`, per call:** the mean only. It recomputes
  each of the `2^D` cells' `(𝓔, S_i, S_ij)` from `P`, which holds the
  cell's metric, `E` and `B` after `prepare!`. So a call reads 8 × 30
  values and stores nothing.
- **`ADMMatter(p, M; order)`, stored:** first copies every stored cell's
  `(𝓔, S_i, S_ij, √γ)` into `M = adm_matter_set(p)`, a set the caller owns
  and makes anew after a regrid. Then it answers from `M`. `order` is 2 or
  4, with no default. At order 2 it is the per-call mean bit for bit. At
  order 4, recomputing 64 cells per call would cost eight times the mean's
  work, so only the stored form offers it.

For the mean, storing saves less than 1% of a step. On the star
(`coupled/coupling_cost.jl`):
- **`N = 16`, 64 threads:** TreeGH's right-hand side takes 11.5 ms with
  the per-call adapter. With the stored one it takes 7.0 ms, plus 3.5 ms
  for the fill. The step takes 2.49 s.
- **`N = 12`, one thread:** 0.223 s per call, against 0.143 s plus 0.041 s
  stored. The step takes 20.6 s.

The fourth order cost about 10% more wall time than the mean in a 4 M
smoke run at `N = 10` on the laptop, at 8 threads; this is not a benchmark.

Both forms' bound adapters are `isbits` through Adapt and allocate
nothing. Against the analytic star away from the surface
(`test/adm_matter_tests.jl`), the mean converges at second order and the
interpolation at 3.5 to 3.9.
[HISTORY.md, "Matter to the vertices"](HISTORY.md#matter-to-the-vertices)
has the estimate the per-call measurement replaced, and the fourth order's
origin.

### Gauge for the TOV milestones

TreeGH's static-background path, with no gauge driver: the TOV
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

**As built**: `src/tov.jl`. **The solve** is Lindblom's (1992)
enthalpy form: RK4 in `η = log h` from the centre to the surface, `η = 0`,
a grid point rather than a root to find, for `q = r²`, `μ = m/r³` and
`λ = log(r̄/r)`, all smooth in `η` at the centre (where
`r ∝ √(η_c − η)`); `λ` is then matched to the isotropic Schwarzschild
exterior, and the lapse inside is `α_s/h` (`h α` is constant in a static
star). Measured: `M = 1.4001597295`, areal `R = 9.5856240172`, isotropic
`R̄ = 8.1251439016` (the literature's 1.400, 9.586, 8.125), the mass
converging at 4.2–4.4 under step halving, `3.8·10⁻¹⁴` at the default 4000
steps. **The metric**, `TOVMetric{T,NK,E}` (`E` the polytrope, or
`Nothing` for a tabulated star), tabulates `ψ` and `α` in `u = r̄²`
— in which both are smooth at the centre, and which needs no square root
there, where a symmetric mesh has a vertex and ForwardDiff's derivative of
`√u` is not finite — with their `u`-derivatives at `NK` uniform knots over
`[0, R̄²]`, cubic Hermite inside, the closed-form exterior outside: `isbits`
`SVector`s, 4176 bytes at the default `NK = 129` (which pass as kernel
parameters on CUDA and Metal, "Devices"), dual-generic. Measured: the table's error 2.8e−8, 1.8e−9,
1.2e−10 at `NK = 33, 65, 129` (fourth order); the field equations
`∇²ψ = −2πψ⁵e` and `∇²(αψ) = 2παψ⁵(e + 6p)` hold to 2.1e−5 of `2πρ_c` at
`NK = 129` (second order in the knots: the interpolant's second derivative;
the scheme reads only first derivatives), and to 2.7e−16 outside;
SpacetimeMetrics' `EinsteinTensor` agrees with `8πT_ab` to 1.8e−6 of
`8πρ_c`. For the polytrope the density is not tabulated: `ρ(h)` at
`h = α_s/α`, the polytrope's own relation, so the initial data and the
metric are one solution; a tabulated star's matter is a `TOVComposition`
("Thermodynamics", "The tabulated star's surface and atmosphere"). **The case**, `src/cases/tov.jl`: the gas's entropy window is
centred on the star's entropy, so that the atmosphere `recovery_options`
writes is on the star's isentrope; the exterior is that atmosphere at rest;
`IdealConductor` with `A = Φ̃ = 0` and `static_potential = true`; the box
`[−24, 24]³` on four roots, whose first box level (`tov_boxes!`) covers
`[−12, 12]³`.

## The right-hand side

In the order the explicit part of each stage evaluates it:

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

- Step 3 is three passes (four with the two-phase recovery, a device's
  default, whose `start_kernel!` and `finish_kernel!` run first):
  `primitive_kernel!` over the owned cells (the
  recovery, from the provider's metric at the cell, `B` from the faces,
  `E = 𝔈/√γ` and `state_guesses` (see "Thermodynamics"); `W`; and all
  thirty slots of `P` through `P_from_W`), the fill of `W`, and
  `ghost_kernel!` over every stored cell that is not owned (`P` through the
  same `P_from_W` with the ghost's `B`). Owned cells take `P` through
  `P_from_W` too, rather than straight from the recovery: where a
  ghost is a copy of its owner — every ghost on a uniform mesh — the two
  blocks then hold the same `P` bit for bit and compute the same flux at
  their common face, as they did when both recovered the same conserved
  state; the two differ by rounding only (`v = u/W` from `u = cosh(w) v`,
  the rapidity `asinh |u|`, one forward EOS call). `q` is a second kernel
  over the owned cells, by centered differences,
  `q = (1/√γ) Σ_d (𝔈ᵈ[n+e_d] − 𝔈ᵈ[n−e_d])/2h`; `q` is taken
  from `P`'s `√γ E` (the same numbers in flat space).
- Step 7 for `A_j` is `−EMF_j − (ψ[n + e_j] − ψ[n])/h` with
  `ψ = αΦ − βᵏA_k` evaluated at the two vertices that bound the edge, `A_k`
  averaged to a vertex from the two `k`-edges that meet there; along a
  missing `j`, `−EMF_j` alone. For `Φ̃` it is the difference of the gauge
  flux `α√γ Aʲ − βʲΦ̃` on the two `j`-edges either side of the vertex, with
  `A_k`, `k ≠ j`, averaged to the `j`-edge (two or four points) and `Φ̃`
  from its two vertices, plus `−ξαΦ̃` (second-order averages; in
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
  (and `z`) is the 1D run bit for bit (tested under MC, PPM, WENO-Z and
  MP5 in 2D, MC and WENO-Z in 3D), with the
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
  without it (rung 2b's claim, "the flux through a
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
  coarse-fine face (accepted, not corrected).
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
  rung 2b's negative control "and does not without [the EMF correction]"
  applies to the rate, where it is asserted (above), and the state's
  negative control is the PointValue family.

  **Not at an outer face**. In the same
  blast's closed box the 147 faces away from the walls agree to rounding,
  but the 45 whose perimeter lies on an outer face miss by 1.2e−5 of the
  largest flux (2.7e−5 after the regrid): there each level's relative hook
  writes its own wall edges, the nearest evolved point of that level plus
  the case's difference, so a coarse wall edge is not the sum of the fine
  ones beside it. The flux through a face at a Dirichlet boundary is the
  boundary condition's, not conserved by anything, so nothing in this
  package relies on it; a relative hook that fills a point on both an outer
  face and a coarse-fine face from the level that owns it (a transfer, not
  the hook) would make it exact — **AMR-11** ([Requests still waiting](#requests-still-waiting); wanted,
  nothing blocked).

## Boundaries

Dirichlet from the initial or analytic state for everything up to the
magnetized TOV star — through `CellBoundary(AllVariables(…))` on the
matter's sets, and on the potentials through TreeAMR's relative hook,
`CellBoundary(Relative(f))`, or, for static potentials, the absolute one
(below): shock tubes until a wave arrives, the
Kerr-Schild tests from the analytic solution, the stars from the static
exterior (atmosphere, the initial `A` and `Φ̃ = 0`, `E` of the static
dipole). TreeAMR 0.2.1 has outflow faces (AMR-6), for whichever later rung
needs them. `MHDProblem` takes `boundary = nothing` for a periodic domain
(and refuses a non-periodic forest without hooks) or one hook per evolved
set, in the order `(U, A₁, A₂, A₃, Φ̃)` — each set's `fill_ghosts!` gets
its own (AMR-2). The waves are periodic; the shock tubes pass hooks.

*As built* (`DirichletBoundary`, `case_boundary` in `src/driver.jl`).
A case that is not periodic gets one hook per evolved set from
its own `fields` and `potential`, evaluated at the time of each fill
(`boundary_at(hook, t)`: every fill — a right-hand side's, a stage's, a
reset's — knows its time):
- **The matter**: Dirichlet, `CellBoundary(AllVariables(…))`, a kernel on
  every backend. The scheme's ghost fill is the primitives' `W`, whose hook
  (`primitive_boundary`, `DirichletBoundary{:W}`) holds the case's
  primitives at each ghost cell's center; `U`'s hook, the case's conserved
  state there, serves only TreeAMR's own fills of `U`, `regrid!`'s before
  its transfer and the initial adaptation's.
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
  not, runs on every backend**: on Metal and on an H200 the rotor's Dirichlet
  hooks, matter and potentials, run as they do on the host ("Devices").

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

**The low walls in the state**. The low wall is owned, so it is
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
it present and bounded (nothing in this package needs a refined
box to be symmetric; the stars sit at the centre of nested boxes, and no
star test asks for mirror symmetry).

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
  potentials' hook, above). The Kerr-Schild cases and the three stars
  (`tov_case`, `magnetized_tov_case`, `tabulated_tov_case`) declare it;
  the test "A
  static potential's boundary is the case's own" checks that a gauge offset
  vanishing on the boundary leaves `𝔅` on every stored face at the exact
  state's (1.1e−15) and every boundary-filled point untouched, where the
  relative hook shows it (5.9e−2).
- **Potentials that grow in time** keep the relative hook (every flat-space
  case, `static_potential = false` by default): their outputs are
  unchanged bit for bit.
- **Dynamic cases without an exact `A`** have neither yet: option 2, a relative hook that carries the offset's gradient
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
  like any other cell; the ball is about 0.8% of the default octant
  (`[0, 4]³`) at `r_in = M`.
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

A Löhner indicator on `ρ`, `|B|` and `|E|`, following
TreeHydro's `hydro_flags`, plus geometric radius boxes around a star that
fix the levels the star is resolved with. `regrid!` is given
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

- **The indicator** (`mhd_flags`): Löhner's, with TreeHydro's
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
  required. The geometric boxes around a star are not the indicator's:
  `tov_boxes!` (`src/cases/tov.jl`) refines them statically.
- **The regrid** (`regrid_state!`, at the start of every chunk after the
  first): `regrid!` over every field set (`regrid_pairs`) —
  `U`, `A_j`, `Φ̃` with their schedules and each with its own hook at the
  regrid's time (AMR-2), so that `regrid!`'s own fill before its transfer
  applies the complete outer boundary condition (the potentials' relative
  or absolute hook, "Boundaries"); `P`, `W`, `B_d`, `F_d`, `EMF_d` and `X`
  `=> nothing`, resized and never transferred (`B` is never state: the
  regrid moves `A`, and `B` is its curl). Then the problem is rebuilt over
  the same sets (`rebuild`: schedules, interface schedules, per-block
  geometry and outer faces), the state gathered, and the reset run once as
  the step limiter (`reset_step!`, which also writes the potentials'
  prescribed low walls, AMR-8), so that a prolongated conserved state the
  recovery would repair is repaired in the state and its injection counted
  with the step limiter's (TreeHydro's post-regrid reset). The
  integrator's scratch is dropped when the mesh moved.
- **The initial mesh** (`adapt_case!`): TreeAMR's several-set
  `adapt_to_initial_data!` (AMR-3) over every field set, the evolved ones
  filled each pass from the case — the conserved state *at* each cell's
  centre, a point value standing in for the Gauss average while the mesh is
  chosen, and the potentials at their points — with their own hooks; the
  criterion recovers `P` from the filled sets and flags. On the converged
  mesh the exact initial data (`exact_state`, Gauss averages) is written.
- **A static mesh**: `evolve!`'s `refine(forest)` refines the
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
  directions, damp alike and stay its run bit for bit (`default_ξ`). A
  case's own `gauge_damping` replaces it (the spinning Wald case takes 0),
  and a caller's `ξ` overrides both. On a uniform mesh the gauge never reaches `𝔅` —
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
  subluminal velocity, and `E` undensitized, as a primitive.
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
lost is the first and the third: they run on their branches, and
`bin/ghost_entropy.jl`, `bin/ghost_robustness.jl` and
`bin/interface_order.jl` say so when asked for them.

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

CI runs three cells of the package's suite, all at 1.13: macOS/arm64,
Linux, and Linux at four threads; and a fourth job, the coupled suite
(`coupled/`, Linux, one thread, bounds checks), on every trigger. The macOS cell carries the coverage, which at 1.13 costs 1.24x
(6m27s against 5m12s, both with bounds checks).

The cells run in two tiers, the same in every Tree* repository: a pull
request runs the two Linux cells and the coupled job, uninstrumented; a push to `main`, a
weekly schedule and a manual dispatch add the macOS cell and its coverage.
Markdown and `docs/` changes run no CI, on pull requests as well as pushes,
and tags are not tested, since a release tags a commit `main` has passed.
Why (most commits touch only Markdown, and macOS cells queue behind the
other repositories'): TreeAMR's `HISTORY.md` ("Development environment and
CI") and [HISTORY.md, "Julia version"](HISTORY.md#julia-version).

The suite runs in **tiers** (`test/tiers.jl`, after
TreeGeneralizedHarmonic's), chosen by `TREEGRRMHD_TEST_TIER`. On GitHub's
runners at four threads the 1D studies take 4–8 times their one-thread
time (the 3D and curved runs are faster threaded), where on Symmetry four
threads are faster than one throughout; the measurements are in
[HISTORY.md, "Julia version"](HISTORY.md#julia-version). So `full`, the
default, runs everything (the workstation and Symmetry before a pull
request, `ci/full-suite.sh`, and CI's weekly run); `quick`, CI's one-thread
cells, leaves out the `@full_tier` testsets (the curved cases' `h = M/8`
runs, the 1000-step uniform flows, the vertex metric's and the magnetized
star's convergence studies); and `threaded`, CI's four-thread cell,
also the `@serial_tier` ones, the 1D studies. The kernels are one code for
D = 1, 2, 3, so the 2D and 3D runs, refinement, the curved backgrounds, the
step cache and the two-phase recovery still exercise every threaded path,
and the 1D runs held bit for bit to 2D and 3D runs stay. A testset left out
is one skipped test ("Broken") per testset (per iteration, in a loop) and
is named at the end of the run. The runs several testsets read are made on first
use (`computed_once`) rather than by top-level constants, and the helpers
shared between files live in files of their own. A fourth tier, `mpi`, runs
the MPI comparison alone, as CI's job of its own does; every other tier
runs it too where the machine has room for three ranks (16 GB;
`test/mpi_jobs.jl`), and CI's other cells skip it
(`TREEGRRMHD_TEST_MPI=0`; "Running distributed").

1.12 and later redefine a `const` silently, where 1.11 refused, so the
duplicate-constant check at the top of `test/runtests.jl` is the only thing
that finds a top-level name two test files share.

`Project.toml` names EntropyEOS and IMEXRungeKutta there, by their GitHub
URLs, EntropyEOS at a tag (`rev = "v2.3.0"`) and IMEXRungeKutta at
`rev = "main"`, and CI adds nothing by hand: `julia-buildpkg`,
a clean `git archive` and a developer's `Pkg.instantiate()` all resolve
them the same way. The test environment reaches them through the
package (`TreeGRRMHD.EntropyEOS`, `TreeGRRMHD.IRK`) rather than listing
them again, so that a second `[sources]` cannot disagree with the first.

The history of the floor (1.10, 1.11, 1.12), of the CI cells and of what
1.11 needed is in [HISTORY.md, "Julia version"](HISTORY.md#julia-version).

## Precision and devices

Every driver is generic in the element type `T`, a leading positional argument (defaulting
to `Float64` in the cases; `mhd_fields`, `VertexMetricSet` and `evolve!(T, case, …)` require
it, and `evolve!(case, …)` takes the case's), and in the KernelAbstractions backend, a keyword
defaulting to `CPU()`.
Physics is validated in `Float64`; `Float32` is the device smoke test and the leak
detector (a stray `Float64` constant widens a result), since EntropyEOS is validated in
`Float64` only. `src/precision.jl` carries TreeHydro's bridges over `Base`'s gaps at a
software float (`wrap`, `ceilint`, `floorint`, `roundint`, `tofloat64`) for host-side
control flow; they are tested at `Float32x2` too, which is a claim about them and not about
the physics. `src/device.jl` carries TreeHydro's and TreeGH's `to_backend` (per-block
metadata a kernel reads, uploaded once) and `hostcopy`/`hostcopy!` (a field set brought to
the host for a viewer; on the CPU `hostcopy` returns the set itself).

A SpacetimeMetrics metric is an `isbits` kernel argument: inside a `map_blocks!` kernel it
gives the ADM split, `K_ij` and a ForwardDiff derivative bit-identical to the host at
`Float64` and `Float32` (measured, Kerr-Schild at `a = M/2`). That is the premise of the
`Analytic` provider.

The pointwise layer — `Metric3`, `prim2con`, `con2prim_hydro`, `con2prim_ideal` — runs
inside a KernelAbstractions kernel bit-identically to the host at `Float64` and `Float32`,
allocation-free and inferred, and so does `GeometricEOS` around a table at `Float64`. So
does the pointwise physics — `Analytic(KerrSchild)` with its ForwardDiff derivatives, the
stress-energy, both Riemann solvers in every direction, Ohm's law, both source forms, the
UCT EMF and the gauge terms — at `Float64` and `Float32`, and the stage solve on every
branch. `lower`, `raise`, `dot_up` and `dot_down` multiply with plain `*` and `+` (`mul3` in
`src/metric3.jl`), which is never fused: StaticArrays' matrix product accumulates with
`muladd`, which LLVM fused differently in the kernel and on the host, so three of the 79
numbers differed in the last bit. `GeometricEOS` around a table is not `isbits` (the table
holds arrays) and is passed to a CPU kernel as it is; it carries an `Adapt` rule for a
device.

Every kernel of the scheme and of the driver's initial data takes its geometry as
`to_backend` arrays of the block origins and spacings and its physics as `isbits` arguments
(the EOS, the options, the model, the provider, a case's callable
`WaveFields`/`WavePotential`), and is generic in `T`: the Alfvén wave runs in `Float32`
throughout — the matter and field kernels, the stage solve and the reset — and lands within
4e−5 of the `Float64` run's error (rung 1a).

**The diagnostics.** What a step and a chunk read is reduced where the data are, as a
`mesh_mapreduce`: `ohm_tally`, `max_light_rate`, `conserved_totals`, `max_div_B`,
`central_density`, `l1_errors`. The measurements of rungs 3 and 6 — `profile_1d`,
`cell_profile` and the field diagnostics, `outside_l1`, `field_l1`, `charge_norms` — read
single points in host loops, and copy what they read to the host (`Array`, or `hostcopy`)
once per call: they run outside a step, once per chunk at most, and a run that calls them
on a device pays one copy of the sets they read each time.

### Threads

**A run is bit-identical at every thread count.** Every kernel runs through TreeAMR's
`map_blocks!` and writes its own points; every reduction is a `mesh_mapreduce`, combined on
the host in block order; the integrator's stage arithmetic is elementwise however
IMEXRungeKutta's `:even` partition cuts the state; and nothing in this package threads a
loop of its own. `test/threading_tests.jl` holds it: the digests of three runs
(`test/thread_workload.jl`) — the magnetized atmosphere tube (the reset at every stage, the
cutoff's stiff stage solve, the step cache), the adaptive rotor (the adaptation, a regrid,
the Dirichlet hooks) and the magnetized star on two levels (`Analytic(TOVMetric)`, the
resistive interior, the atmosphere) — per chunk the steps, `Δt`, the tally and the blocks,
and at the end the state, the totals and `∇·𝔅`, against a subprocess at another thread
count, line for line (equal at one and four threads).

The stage arithmetic is threaded by IMEXRungeKutta's `:even` partition, not by block
owner, which would need TreeAMR's `threadchunks` exported and gains nothing over it ("The
per-phase table").

### Devices

**The storage decides where the work runs** (TreeAMR's M6): `evolve!` and `MHDProblem`'s
field sets take the backend from `mhd_fields(T, forest; backend)`, every kernel takes it
from the array it is handed, and the integrator follows the state — `mhd_integrator` and a
script pick `partition = :even` for an `Array` and `nothing`, the broadcast, otherwise.
`bin/tov_cowling.jl` and `bin/benchmark.jl` take `backend=` and `type=` through
`bin/backend.jl` (TreeHydro's), which loads the device package only when asked for; neither
CUDA nor Metal is a dependency.

**The claim on a device is not bit-identity**: a device contracts `a*b + c` into an `fma`
where the host may not, and its `exp` and `sqrt` are its own. It is that a run there answers
as the host run at the same type (`test/device_tests.jl`, four runs: the Alfvén wave, the
magnetized atmosphere tube, the adaptive rotor and the magnetized star, each with both
recoveries):

- **exactly**: the steps and `Δt` of every chunk, the blocks, the regrids and the
  adaptation's passes; the stage solve's branch counts, its flags, the policy's flags, the
  recoveries and the most a cell took; the step cache's cold seeds and the sum of its warm
  and reused ones, per site (which of the two a seed was is not structural: a reuse needs
  the input bit for bit);
- **to `sqrt(eps(T))`**: the largest density, `∇·𝔅` (also to `eps(T)` absolute, where the
  host has exact zeros) and the Alfvén wave's error norms in the components it carries —
  on 40 cells, where its error is large enough: on 100 they are 10⁻⁴ of the state, and
  `Float32`'s rounding moved them by 6·10⁻⁴ of themselves;
- **to `8 eps(T) · scale · nsteps`**, the host suite's bound on a drift: the drifts of the
  seven conserved totals, and the reset's injection;
- **to less than one block's share of the chunk's recoveries**: the policy's and the
  reset's counts, each a count of comparisons of a recovered density with the atmosphere's
  trigger. A cell within a few ulp of it can go either way and then evolves differently, so
  how many differ is the case's and not `eps`'s; what the test guards against — a block a
  kernel or a reduction did not see, or saw twice — moves a count by about a block's share.
  Measured: on an H200 at `Float64` the atmosphere tube's counts differed by 1 and 2 of
  1.3–1.8·10⁴ (a share is 2200); at `Float32` the magnetized star's 1.9·10⁶ resets a chunk
  by 82 and 114 on the H200 and by 24 and 56 on Metal (a share is 1.4·10⁴). The bound
  stays at the share, though these are at most 0.008 of it: a bound fitted to them would
  measure the cases' sensitivity, not the fault;
- **the policy's flag word exactly**, but at `Float32` without its not-converged bit:
  EntropyEOS is validated at `Float64` only, and on the H200 one cell of the star at
  `Float32` failed to converge where the host's converged.

Measured on Apple's M3 Pro (Metal.jl 1.11.1, `Float32`): every assertion holds with both
recoveries, 152 of 152; the Cowling star of `bin/tov_cowling.jl` at `N = 8`, two roots, to
`20 M`, agrees with the CPU's `Float32` run to 3 ulp in `ρ_c`, and at the script's default
size (`[−24, 24]³`, four roots of 16³, one box level) runs to `200 M` and ends 2.6·10⁻⁵ of
`ρ_c` from the `Float64` run. On an H200 (Symmetry; CUDA.jl 6.4.2, KernelAbstractions
0.9.48, Julia 1.13.4): every assertion holds at `Float32` and `Float64` with both
recoveries, 304 of 304, in 15 minutes; the Cowling star at the default size to `200 M` at
`Float64` agrees with the same run on the node's own CPU in `ρ_c` at every `10 M` to 26 ulp
(5.7·10⁻¹⁵ of it), in 1305 s against 13048 s on that node's 8 Xeon Gold 6442Y cores
(`jobs/tov-cowling-device.toml`). At `Float32` the same run takes 771 s on the H200 and
stays within 3.0·10⁻⁵ of `ρ_c` of the `Float64` run at every `10 M`, as Metal's does.
`TOVMetric`'s 4176 bytes of kernel parameters pass on both.

**The recovery on a device.** Both recoveries run on both devices: the fused kernels, which
call EntropyEOS's two levels in one kernel, pass on CUDA.jl 6.4.2, the version of the fault
EntropyEOS records ("The two-phase recovery"). On an H200 the split is the faster: a step of
the star 15% at 8.8·10⁵ cells and 14% at 7.1·10⁶, its stage solve 39–40% ("The per-phase
table"), where on the CPU it costs 5%. So the split is a device's default and fused the
CPU's ("The two-phase recovery"). On a table the fused recovery is the faster, by 6.5% a
step of the DD2 star on an H200 ("The tabulated star's cost"); the default stays the
backend's, measured on one table, one mesh and one device, and a run on a table that
wants the last 6.5% passes `two_phase = false` itself.

**A table on a device.** A kernel's adaptor turns a device-resident table into an
`isbits` view but cannot upload a host array, so a tabulated case's EOS is uploaded once
where it enters the kernels: `eos_on(backend, eos)` (`src/device.jl`, `Adapt.adapt` to the
backend: the CPU keeps its arrays, CUDA copies them) in `MHDProblem`, the initial data, the
Dirichlet hooks and the interior; the hooks' wrappers (`DirichletBoundary`, `DirichletAt`,
`AbsolutePotentialAt`) carry `Adapt` rules, without which the launch refuses the table's
arrays inside them. A hook is a kernel argument, and CUDA limits a launch's arguments to
32 KiB: each hook holds only what it reads (`case_boundary`: the matter's the fields, the
EOS and the provider; a potential's the potential), since the tabulated star's fields and
potential, each holding its metric twice (4.2 KB) and its composition (6.2 KB), are
14.5 KB apiece, and a potential's hook with everything came to 32.8 KiB. Both were found by
the first device run of a table (2026-11-25; until then the device tests ran ideal gases
only): the device tests now run the DD2 star where the table is found. On an H200
(Symmetry job 611271) every assertion holds at both types and both recoveries, 386 of 386:
at `Float64` the DD2 star's central density is the host's to 3·10⁻¹⁶ and its reset counts
are equal, at `Float32` they differ by 7 and 1 of 2·10⁶ (a block's share is 1.4·10⁴); and
the same 386 on each rank of two H200s under MPI (`jobs/device-mpi.toml`, job 611272). On
Apple's M3 Pro (Metal, `Float32`) 194 of 194, the central density within 5·10⁻⁸ of the
host's and the reset counts equal.

**One construct crashes Metal's compiler**: a kernel's sum over the present directions
written `sum(ntuple(j -> f(…, Val(j)), Val(D)))`, with the `Val` built from the closure's
argument — the gauge kernel's divergence — gave a bus error in LLVM's inliner (Metal.jl
1.11.1, Julia 1.13.4) while each term compiled alone. It is written out
(`gauge_divergence`, the same additions in the same order, so the host's bits are
unchanged).

**`workgroupsize`** (AMR-5): `MHDProblem(…; workgroupsize)` and `evolve!(…;
workgroupsize)` hand it to every launch of the step's kernels and the curl; `nothing`, the
default, leaves the choice to the backend, and the CPU ignores it. Nothing passes one: on
the H200 (the star at `Float64` on 8.8·10⁵ cells, `jobs/benchmark-gpu.toml`) workgroups that
cover the block's first axis — 8×8, 16×4, 16×8, 16×16 — give a step within −2% to +5% of
the backend's choice (583–622 ns a cell against 594), the fluxes up to 23% faster in smaller
groups and the recovery up to 14% slower, and 16×16×2 does not launch (a kernel of the step
uses 255 registers a thread). An integer is a row along the first axis, which
KernelAbstractions pads with ones, so most of its threads idle on a block 16 points wide:
64, 128 and 256 make a step 2.2, 2.6 and 6.2 times slower.

### The per-phase table

A right-hand side is seven phases, each a function the step calls, so that
`bin/benchmark.jl` times what runs: `fill_potentials!` (the ghost fill of `A_j` and `Φ̃`),
`curl_A!`, `recover_cells!` (the recovery at the owned cells), `fill_primitives!` (`W`'s
fill, `P` in the ghosts, `q`), `face_fluxes!`, `edge_emfs!` and `tendencies!`, after the
`scatter!` of the state. Beside them the table has the whole right-hand side, the stage
solve, the reset, the light-cone rate and the step.

**On the CPU** (`jobs/benchmark-cpu.toml`, Symmetry job 606359, cn101, AMD EPYC 7543,
pinned, one thread count a run; the magnetized star at `Float64`, 3³ roots of 16³ and one
box level, 216 blocks, 8.8·10⁵ cells; Julia 1.13.4; ns per cell, the best of five):

| threads | recover | fill_potentials | face_fluxes | rhs | stage_solve | reset | step | step, broadcast |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6040 | 108.8 | 1412 | 9723 | 6192 | 1523 | 34392 | 34427 |
| 2 | 3024 | 55.9 | 710 | 4873 | 3139 | 788 | 16998 | 17180 |
| 4 | 1606 | 34.4 | 356 | 2558 | 1672 | 404 | 9580 | 9699 |
| 8 | 824 | 26.3 | 176 | 1326 | 876 | 218 | 5415 | 5490 |
| 16 | 435 | 23.7 | 92 | 702 | 462 | 122 | 2962 | 3099 |
| 32 | 236 | 26.7 | 46 | 380 | 258 | 69 | 1691 | 1848 |
| 64 | 150 | 28.5 | 29 | 259 | 186 | 63 | 1256 | 1496 |

- **The recovery is the right-hand side**: 62% of it at one thread, 58% at 64 (outside a
  step, where it is cold; inside one the step cache seeds and reuses it, which is why a step
  is less than its four stages' right-hand sides, solves and resets). The fluxes are 15%,
  the tendencies 13%; the rest is a few percent each.
- **A step scales 27× on 64 cores** (a right-hand side 38×); what stops scaling is
  TreeAMR's: the potentials' ghost fill reaches its floor, 24–29 ns a cell, at 8–16
  threads.
- **The two-phase recovery costs 5% a step** on 64 threads (1322 against 1256), as it did
  per site (1–10%, "The two-phase recovery"): fused is the CPU's default.
- **The partition**: IMEXRungeKutta's `:even` against the broadcast is worth 16% of a step
  at 64 threads and nothing at 8 and below. The block-owner partition gains nothing over
  `:even` (below), so TreeAMR is not asked to export `threadchunks`.

**The partition by block owner** (`jobs/benchmark-partition.toml`, Symmetry job 606773,
cn095, AMD EPYC 7543 as cn101; `bin/benchmark.jl`'s `step`, `step_broadcast` and
`step_owner` rows; the star on 3³ roots and on 6³, 7.1·10⁶ cells, about 800 MB a state
vector; ns per cell and step). `:even` cuts the state vector of five sets into equal
ranges, so a thread mostly combines entries of blocks that other threads' kernels wrote
and first touched; by owner (`owner_partition`, from TreeAMR's unexported `threadchunks`)
it combines only its own blocks' entries:

| threads | cells | `:even` | broadcast | by owner | owner / `:even` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 8.8·10⁵ | 5451 | 5539 | 5446 | 0.999 |
| 16 | 8.8·10⁵ | 2959 | 3105 | 2985 | 1.009 |
| 32 | 8.8·10⁵ | 1699 | 1857 | 1696 | 0.998 |
| 64 | 8.8·10⁵ | 1255 | 1446 | 1250 | 0.996 |
| 64 | 7.1·10⁶ | 765 | 976 | 760 | 0.993 |

Within 1% either way, on a mesh near the caches' size and on one well beyond them: the
placement of the stage arithmetic is not what a step pays for once it is threaded. The
same bits on every path (IMEXRungeKutta's contract; checked for this state at four
threads). cn095's `:even` step at 64 threads is cn101's (1255 against 1256).

**On an H200** (`jobs/benchmark-gpu.toml`, Symmetry job 606355, cn111; CUDA.jl 6.4.2,
Julia 1.13.4; the same star on the same mesh, and on 6³ roots, 1728 blocks, 7.1·10⁶ cells;
ns per cell, the best of five, the workgroup size the backend's):

| type | cells | recovery | recover | fill_potentials | face_fluxes | rhs | stage_solve | reset | step |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Float64` | 8.8·10⁵ | fused | 46.1 | 2.87 | 8.70 | 87.2 | 73.2 | 10.2 | 589 |
| `Float64` | 8.8·10⁵ | split | 42.1 | 2.78 | 8.70 | 83.0 | 44.9 | 11.2 | 501 |
| `Float32` | 8.8·10⁵ | fused | 16.3 | 2.57 | 2.73 | 35.9 | 19.4 | 24.2 | 316 |
| `Float64` | 7.1·10⁶ | fused | 40.9 | 0.42 | 8.48 | 76.6 | 64.1 | 7.08 | 429 |
| `Float64` | 7.1·10⁶ | split | 37.5 | 0.41 | 8.49 | 73.1 | 38.2 | 8.12 | 369 |
| `Float32` | 7.1·10⁶ | fused | 14.5 | 0.34 | 2.53 | 29.8 | 15.8 | 20.2 | 270 |

- **One H200 is 2.1–3.4 EPYC nodes**: a step at `Float64` is 589 ns a cell on the suite's
  mesh and 429 on the larger one, which fills the device better, against 1256 on 64 cores;
  with the split recovery 501 and 369. The Cowling star to `200 M` took a tenth of the time
  on the H200 that it took on its node's 8 CPU cores ("Devices").
- **The recovery is again the largest phase**, about half the right-hand side, and the
  stage solve the second; the split recovery takes 39–40% off the stage solve, where most
  of its recoveries are, and 8–9% off the right-hand side's.
- **The ghost fills cost a launch, not a cell**: the potentials' fill takes 2.3–3.0 ms on
  either mesh, 0.4 ns a cell on the larger.
- **`Float32` is 1.6–1.9× faster a step**, the right-hand side 2.4–2.6×, but the reset is
  2.4–2.9× slower than at `Float64`, with the same counts of resets (not examined).

### The tabulated star's cost

What a production run on a table pays, measured on the DD2 star before larger runs
(2026-11-25; `bin/benchmark.jl case=dd2`, `bin/profile.jl`; `jobs/perf-*.toml`): the 1.4 M☉
star in the Cowling approximation at `Float64`, on the Cowling study's finest mesh
(`[−24, 24]³`, 4³ roots of 16³, one box level: 120 blocks, 4.9·10⁵ cells, `h = 0.375` in
the box; "M1") and on two larger ones (two box levels, 960 blocks, 3.9·10⁶ cells, "M2";
`[−48, 48]³`, 8³ roots, three levels, 6056 blocks, 2.5·10⁷ cells, "M3"). About 93% of M1's
cells are atmosphere.

**On the CPU** (`jobs/perf-cpu.toml`, Symmetry job 611275, cn095, AMD EPYC 7543, pinned;
ns per cell, the best of three; "PR 16" is the code before this section's changes):

| threads | recover | fill_primitives | face_fluxes | rhs | stage_solve | reset | step | PR 16's step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2455 | 1845 | 8879 | 14897 | 1653 | 2428 | 58261 | |
| 2 | 1227 | 920 | 4376 | 7439 | 822 | 1209 | 29306 | |
| 4 | 652 | 478 | 2202 | 3804 | 467 | 650 | 15197 | |
| 8 | 347 | 245 | 1109 | 1947 | 263 | 349 | 7929 | 23533 |
| 16 | 203 | 137 | 597 | 1077 | 170 | 201 | 4539 | |
| 32 | 115 | 77 | 318 | 589 | 101 | 120 | 2548 | |
| 64 | 72 | 43 | 166 | 365 | 92 | 94 | 1653 | 4293 |
| 64, M2 | 72 | 39 | 157 | 314 | 74 | 75 | 1246 | |

- **A step is 3.0 times PR 16's at 8 threads and 2.6 at 64.** PR 16 spent 13 µs a
  cell (at 8 threads) in each recovery outside a step, against 0.35 now: three faults
  of the recovery on a table, each a few lines ("Thermodynamics", "Warm starts";
  "Primitive recovery", "The two levels"): `GeometricEOS` did not forward EntropyEOS's
  two optional methods, so a cold seed took a bisection in `s` (58–75 µs); the
  advected `s` seeded the Newton from below the table's window in most of the star
  (8.4 iterations); and the atmosphere's cells, colder than the table after the
  gravitational source, each spent the solver's whole budget on an answer the policy
  discards (32–41 µs; the skip of level 1).
- **The EOS evaluations are now the step**: the face states (`face_fluxes`, six
  evaluations a cell) and the primitives' ghosts (`fill_primitives`) are 72% of a
  right-hand side at one thread, and half of each evaluation samples the ends of its
  temperature bracket (EOS-14). A table's evaluation is about 0.6–1 µs; the ideal gas's
  is analytic.
- **Threads: 35 times on 64 cores** (a right-hand side 41 times), where the
  magnetized polytrope's step scaled 27 times ("The per-phase table"). The fluxes
  scale 53 times, the recoveries 18–34 (`stage_solve` the least): a star cell's
  recovery costs 4.0 µs against an atmosphere cell's 0.7, a block's up to 3.9 times the
  mean block's, and the threads take equal counts of blocks in curve order, so those
  that hold the star set the pace. At 64 threads a step's threads are asleep 23% of
  their samples (`bin/profile.jl`). A split by measured cost would help where a thread
  has several blocks (AMR-14); at 64 threads on 120 blocks one star block is the limit,
  and the larger mesh's step is 25% cheaper a cell (1246 against 1653).
- **GC and allocation**: a step at 64 threads allocated 215 MB and spent 0.06 s of its
  0.87 in GC (7%; an earlier measurement of the same step, before the EOS forwards,
  66 MB and none); at one thread 46 MB and 0.02 s of 29 s. Whether a collection lands
  in a step is chance; what allocates in proportion to the threads was not examined.

**Over MPI ranks** (`jobs/perf-mpi-node192.toml`, `jobs/perf-mpi-nodes192.toml`; Trillium
jobs 2930886 and 2930888, Fir jobs 6128536 and 6128672; nodes of 2× AMD EPYC 9655, 192
cores in 8 NUMA domains; 8 ranks of 24 threads a node, one rank per NUMA domain, the site's
Open MPI through mpi_abi_wrapper; seconds a step, the slowest rank's best of three,
Trillium / Fir):

| mesh | 1 node | 2 nodes | 4 nodes | 8 nodes |
| --- | ---: | ---: | ---: | ---: |
| M2, 3.9·10⁶ cells | 1.77 / 1.70 | 1.18 / 1.15 | 0.97 / 1.00 | 0.54 / 0.55 |
| M3, 2.5·10⁷ cells | 8.81 / 8.77 | 4.94 / 4.92 | 2.84 / 2.83 | 1.86 / 1.94 |
| M3's efficiency against 1 node | | 89% / 89% | 77% / 77% | 59% / 57% |

- **The two sites agree** to 4%, on the same CPU. On M2 a 192-core node is 3.0 of
  Symmetry's 64-core nodes (449 ns a cell against 1331), as many as its cores, and one H200
  is 1.2–1.3 of these nodes (376 and 352 ns a cell).
- **Ranks on one node cost nothing, and help a little**: 1, 2, 4 and 8 ranks of 192, 96, 48
  and 24 threads give 514, 454, 462 and 449 ns a cell on M2 (Fir 522, 461, 458, 442); one
  process on the whole node is the slowest, by 13–15% (on Symmetry's nodes by 9%: 1457
  against 1328–1359, job 611209). Production runs one rank per NUMA domain.
- **Over nodes the slowest rank sets the pace.** The ranks hold equal counts of blocks (94 or
  95 of M3's on 8 nodes), but a star block costs more than an atmosphere block, in the
  recovery most: the slowest rank's `recover` takes 1.23, 1.58, 1.73 and 1.84 times the mean
  rank's on 1, 2, 4 and 8 nodes, its `tendencies` up to 1.63 and its `face_fluxes` 1.22
  (`bin/benchmark.jl`'s `rank_mean_seconds`). On 8 nodes a right-hand side takes 0.42 s
  where perfect scaling of one node's 2.38 would give 0.30; its phases' slowest ranks add
  up to 0.41 s and their mean ranks to 0.30. Waiting on the slowest rank is about 0.45 s of
  the 0.76 s a step loses on 8 nodes against perfect scaling (in its three right-hand
  sides, four stage solves and four resets); the ghost exchanges are most of the rest
  (`fill_primitives` 0.052 s on the mean rank against 0.038 scaled). A split by measured
  cost would recover the first part (AMR-14).
- **M2 is too small for 8 nodes**: 15 blocks a rank, fewer than its 24 threads, and 41% of
  perfect scaling. A production run wants several blocks a thread.
- **MPICH_jll** (SimLaunch's `jll` recipe) is 14% slower a step on Trillium's 8 nodes (2.12 s
  on M3) and 2% on Fir's (1.98 s); the site's Open MPI stays the default.
- **Symmetry's AMD nodes cannot measure scaling over nodes.** The same 4-node step took 2.1 s
  or 8.9 s on M2 by the node set, and 8 nodes or more hang in the first collective under
  HPC-X's Open MPI while MPICH_jll runs (`HISTORY.md`, "Precision and devices"). A run there
  on 8 nodes or more takes `mpi = "jll"`.

**On an H200** (`jobs/perf-gpu.toml`, Symmetry job 611277, cn113; ns per cell, the best of
three; "PR 16" with the device fixes alone, which it needs to run a table at all):

| mesh | recovery | recover | face_fluxes | rhs | stage_solve | reset | step |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | split | 13.5 | 29.8 | 104.5 | 28.5 | 42.2 | 706 |
| M2 | split | 10.3 | 28.5 | 73.6 | 11.2 | 20.5 | 376 |
| M2 | fused | 8.6 | 28.5 | 71.8 | 9.2 | 20.1 | 352 |
| M2, PR 16 | split | 145.0 | 28.5 | 208.3 | 146.0 | 146.0 | 515 |
| M3 | split | 9.9 | 28.1 | 69.7 | 9.5 | 18.2 | 354 |

- **One H200 is 3.3–3.5 EPYC nodes** on M2 (376 and 352 against 1246 on 64 cores), and
  M1 is too small to fill it (706). The fixes take 27% off a step there: inside a step
  most recoveries were warm already.
- **On the table the fused recovery is the faster**, by 6.5% a step, where on the
  magnetized polytrope the split was 14–15% faster and became a device's default
  ("Devices"); the default stays, and a run on a table may pass `two_phase = false`.
- **Several H200s, one rank each** (`jobs/perf-gpu-mpi.toml`, job 611278, cn113, 1, 2 and 4
  devices of one node; seconds a step): M2 1.47, 0.85 and 0.57 (86% and 65% of perfect
  scaling), M3 8.72, 4.62 and 2.65 (94% and 82%). Four H200s are one 4-node step of the
  192-core sites (2.84 s on M3). The ranks are balanced on a device (the slowest rank's
  `recover` 1.01–1.11 times the mean), where on a CPU the star's ranks take 1.6–1.8 times;
  what grows is `fill_potentials`, 0.010 s on one device and 0.026 on four (M2).
- **The device is busy 92% of a step** (CUDA.jl's profile of four steps on M2,
  `bin/profile.jl backend=cuda`): the recovery's first pass is 24% of the device's time
  and its second 5%, the face fluxes 22%, the primitives' ghosts 8%; the ghost exchange's
  copies, some 30,000 launches of 2–8 µs, 3%.

## Running distributed

A run is one process on its threads or on one device ("Precision and devices"), or it is
distributed over MPI ranks, each rank on its own threads or its own device, in any
combination, through TreeAMR's M7 (its `CODE.md`, "Distributed meshes"), as TreeHydro and
TreeGeneralizedHarmonic run (TreeGH's `CODE.md`, "Running distributed").

**What TreeAMR does, and what is left here.** TreeAMR keeps the whole forest on every rank
and gives each a contiguous range of the Morton curve's leaves. It does everything that
crosses ranks: the ghost exchange (bit-identical to a serial fill), the interface
restriction, the regrid with its repartition and buffered flags, `mesh_mapreduce` (one
partial per rank, folded in rank order), the initial adaptation's cycle, and point
interpolation; TreeIOHDF5 writes the checkpoints. A field set's blocks are the rank's own
(`nblocks` is local), and every kernel, every state vector and the integrator see only
those. What is left for this package is what TreeAMR cannot know:

- a value this package combines on the host from per-block or per-cell values, which is a
  per-rank value until it is combined;
- a decision that gates a collective call — a fill, a reduction, a regrid, a checkpoint —
  which every rank must take alike: from the replicated forest, or agreed;
- output, which one rank writes.

**The claim.** A distributed run is the serial run: its state (gathered in curve order),
mesh, steps, `Δt` of every chunk, regrids, every maximum, minimum and flag, the stage
solve's and the reset's counts and every per-cell profile, bit for bit at any rank count
and any thread count under it; its sums — the conserved totals, the reset's injections, the
diagnostics' norms and ratios of sums — to rounding, and exactly at one rank. On a device
the claim is "Devices"' against the host, at every rank count. `test/mpi_tests.jl` holds it
at one, two and three ranks of two threads ([Distributed tests](#distributed-tests)).

### The communicator

- **`evolve!(…; comm = nothing)`** builds its forest over `comm` —
  `Forest{T}(rs; N, periodic, extents, comm)`, anything TreeAMR's `communicator` takes,
  normally an `MPI.Comm` after the caller's `using MPI; MPI.Init()` — and is called on
  every rank with the same arguments. `nothing` is TreeAMR's serial communicator. Its
  `nblocks`, and each chunk's, are the forest's leaves (`nleaves`), the mesh's count, not
  the rank's.
- **This package never loads MPI.** MPI.jl is a test dependency only (TreeAMR's extension
  holds the MPI methods), so a serial user's environment does not grow it.
- **The verbs need no keyword.** `mhd_fields`, `MHDProblem`, `VertexMetricSet`,
  `tov_boxes!`, `adapt_case!` and the rest take the forest, and the communicator lives on
  it (`communicator(forest)`). A script that owns the loop builds its forest over `comm`
  and changes nothing else.
- **Every forest mutation is collective.** `tov_boxes!` refines from the replicated
  `forest.leaves` by a deterministic function of the case, so every rank makes the same
  calls; that is why its one non-exported read is safe here. `mhd_flags` returns the
  rank's own blocks' flags, which is what `regrid!` takes; its scales are
  `mesh_mapreduce`s, the same on every rank.
- **A collective call is gated by the forest, never by the rank's blocks.** The reset fills
  `Φ̃` where the domain has walls (`MHDProblem`'s `haswalls`): `!all(forest.periodic)`,
  which a non-periodic dimension makes true on every rank, and not `any` over the rank's
  `low_walls`, which held only on the rank owning the low face and left the others out of
  the fill (found by the first distributed tube: `Invalid MPI_Request` in the next
  exchange). The coupled script's fill key skips a fill (collective) only where every
  rank's segment repeats (`rank_reduce(&, …)`).
- **The scripts** in `bin/` that run studies (`tov_cowling.jl`, `tov_study.jl`,
  `benchmark.jl`) and `coupled/tov_coupled.jl` and `coupled/tov_coupled_study.jl` take
  `mpi=true` (`bin/mpi.jl`), which loads MPI by `Core.eval`, as `bin/backend.jl` loads a
  device package, initializes it, builds over `MPI.COMM_WORLD`, and has rank 0 write
  (below). They run under `mpiexec` in an environment that has MPI. Every rank computes
  the record, which may be collective (`central_density`, the diagnostics); rank 0 writes
  it. `benchmark.jl` reports each phase at the slowest rank's best and gains a `ranks`
  column; `tov_study.jl`'s `state` (a serialized state vector) is serial only.

### What this package makes global

**Already global.** Every number the driver acts on is a `mesh_mapreduce`: the light-speed
rate that sets `Δt` (`max_rate`), the conserved totals, the reset's tallies and the stage
solve's counts (`ohm_tally`), `∇·B`, the refinement scales (`indicator_scales`),
`central_density` and `l1_errors`. A count summed as a float is exact while below `2⁵³`
(`2²⁴` in `Float32`), the rank-order fold included.

**`src/distributed.jl`** carries TreeHydro's helpers (copied, as `src/device.jl` is: the
packages share patterns, not code), on TreeAMR's public communicator verbs `commrank`,
`commsize`, `allgather` and `allgatherv` (AMR-13, landed in TreeAMR 0.2.11; called
qualified, since they are public, not exported):

- `rank_reduce(op, over, value; present)`: an `isbits` value combined over the ranks in
  rank order, the same bits on every rank; serially `value` itself, nothing gathered.
  `over` is a problem, a forest, a field set, a communicator, or what `communicator` takes
  (an `MPI.Comm`, `nothing`). `rank_sum` and `rank_max` are its element-wise forms on a
  tuple;
- `rank_sum!(v, over)`: a vector summed element by element over the ranks;
- `agree_any`, `agree_refusal`: a decision or an `ArgumentError` from rank-dependent inputs
  (the file system, the wall clock) made the same on every rank, so that no rank is left
  waiting in a collective another skipped;
- `isroot`: the rank that prints and writes;
- `gather_cells(over, v)`: every rank's per-cell vector concatenated in rank order, on every
  rank — the serial run's vector, since each rank holds a contiguous run of the curve.

**The host diagnostics** read the rank's own blocks and combine over the ranks, so each is
collective and the same on every rank:

- **Reductions.** `field_diagnostics` (the field energies, `Σ|E·B|/ΣB²`, `|E|/|B|`, the
  charge's L1, the trigger and repair counts, the maxima), `field_drift`, `ohm_map`'s
  per-region tallies, `metric_fallbacks`, and `src/interior.jl`'s `outside_l1`,
  `field_l1` and `charge_norms` each sum numerators and denominators per rank and form the
  ratio after the combination. `cell_profile` is the rank's own cells and carries the
  communicator (`comm`) its reductions combine over.
- **Profiles.** `profile_1d` gathers its columns to every rank in curve order before it
  sorts by `x`; `gather_profile` makes a `cell_profile` the whole mesh's, with a serial
  `comm`, so that a reduction of it is not combined a second time. Both are for test-sized
  meshes and figures; a production run records reductions.

**Output** — a study's table, a figure, a checkpoint's rotation, `@info` — is rank 0's
(`isroot`). The coupled study's constraint norms are TreeGH's, global already.

### Checkpoints

TreeIOHDF5's `save_checkpoint` and `load_checkpoint` (TreeHydro's `src/checkpoint.jl` is
the model; here `src/checkpoint.jl`) replace the studies' `serialize` of one `state.jls`,
which no other rank count could read and which carried no device array:

- **`save_run(path, p, u; run, recipe, extra, io)`** writes the forest and the five evolved
  sets (`"U"`, `"A1"`, `"A2"`, `"A3"`, `"Phi"`, each its segment of `u`), one part file
  per I/O group, atomically. `B`, `P`, `W` and the scratch sets are not in it ("Sharp
  edges": `B` is never state); the first fill of a restart rebuilds them. `extra` are
  further sets over the same forest: the coupled study stores TreeGH's (`"TreeGH.U"`).
- **`load_run(path; backend, comm)`** reads it at any rank count, or serially, the forest
  distributed over `comm`; a file of another application or format version is refused.
  A restart builds its sets on the checkpoint's forest and copies the states in
  (`restore_run!`, which refuses a problem on another forest).
- **What only this package knows**: when — at a chunk boundary, before the next chunk's
  regrid, as TreeHydro and TreeGH write, so that a restart regrids first through the loop's
  own code; the run state behind a result (`evolve!`'s `totals0`, chunk count, steps,
  regrids, passes and every chunk's tally; a study's accumulators and rows), as plain data;
  and the recipe a restart must not change (`checkpoint_recipe`: for `evolve!` the type,
  `N`, roots, chunk, `cfl`, limiter, Riemann solver, `ξ`, stage options, tableau, `G`,
  operators, `fixup`, the refinement parameters, the interior and the case's fields, box
  and periodicity), refused by name when it differs (`check_recipe`). `t_end` may change:
  that is how a run is continued.
- **`evolve!(…; checkpoint, checkpoint_every, restart, checkpoint_io)`** writes after every
  `checkpoint_every`-th chunk and the last, and continues from `restart`. The scripts take
  `checkpoint` (a file, or a directory holding `state.h5`, as SimLaunch's restarts give
  one) and resume from it when rank 0 sees it.
- **The step cache is not stored.** It holds nothing across a step (it is invalidated at
  the end of every one), so a restart at a chunk boundary is the uninterrupted run bit for
  bit: the adaptive rotor, restarted after its first chunk, serially and from a
  three-rank checkpoint at three, two and one ranks; the Cowling star's `tov_study.jl`
  resumed serially and at two ranks from a serial checkpoint, its series byte for byte.
  Only `totals0` differs, at rounding, when the checkpoint was written at another rank
  count: it is a sum.
- **TreeIOHDF5** (registered, 0.1.10) is in `[deps]`.

### Threads and devices under MPI

- **Threads.** Each rank runs its kernels on its own Julia threads, as a serial run does;
  TreeAMR calls MPI from the calling task only and needs `MPI_THREAD_SERIALIZED`,
  `MPI.Init()`'s default. The distributed tests run two threads per rank. How many ranks a
  node should hold is measured, not assumed: TreeAMR found one pinned process of 64 threads
  1.5–1.6 times as fast at the right-hand side as 8 ranks of 8 threads on one Symmetry
  node, and `bin/benchmark.jl mpi=true` gives this scheme's table (not yet measured).
- **One rank per device.** Each rank's field sets live on its own device, chosen by the
  caller from the rank's place on its node (`MPI.Comm_split_type(…, COMM_TYPE_SHARED)`;
  `bin/mpi.jl`'s `select_device!`, and `test/device_tests.jl` under MPI), and the EOS table
  is uploaded to each rank's device as it is to one. TreeAMR stages device buffers through
  page-locked host memory, or hands them to a CUDA-aware MPI with
  `communicator(comm; deviceaware = true)`; nothing here branches on which.
- **The coupled run on a device.** `coupled/tov_coupled.jl`'s `problems` takes `backend`
  and passes it to `mhd_fields`, `VertexMetricSet`, TreeGH's 20-variable set and through
  them to the adapters' sets; the fill key compares and copies by broadcast on a device,
  and the integrator's partition follows the state ("Sharp edges"). `problems` also takes
  `comm`, and a `forest` (a checkpoint's) to build on.
- **Other backends.** CUDA is the target and Metal the laptop's `Float32` smoke test.
  AMD's ROCm (Nibi's MI300A) and Intel's oneAPI are KernelAbstractions backends nothing
  here excludes; they are not claimed until a job runs them.

### Distributed tests

- **`test/mpi_tests.jl`** collects `test/mpi_workload.jl`, run by `test/mpi_jobs.jl` as one
  `mpiexec -n 3` launch of two threads per rank (TreeGH's pattern): every case over three
  ranks, then two, then one, and serially, each group writing its lines, compared line by
  line with the serial run — bit for bit, or for a line of sums (`~`) to `rtol = 10⁻¹²`,
  and exactly at one rank (`test/mpi_lines.jl`). The cases: the 1D Alfvén wave and its
  gathered `profile_1d`; the wave on two blocks, so that at three ranks one rank holds none;
  the magnetized atmosphere tube, whose reset injects; the 2D adaptive rotor, adapted and
  regridded, and restarted after one chunk from the three-rank group's checkpoint; the 3D
  magnetized star on its boxes with every host diagnostic; and a restart from a missing
  file, refused on every rank. 202 checks; the launch takes 140 s on the laptop.
- **`test/checkpoint_tests.jl`**: the serial restart bit for bit, and the refusals.
- **The coupled suite** gains `coupled/mpi_tests.jl` (`coupled/mpi_workload.jl` under
  `mpiexec -n 2`: the coupled star, the coupled regrid, and the coupled study resumed from
  a two-rank checkpoint holding TreeGH's set, at two ranks, one and serially; 52 checks) and
  `coupled/device_tests.jl` (the coupled star on a device against the host; the CPU stands
  in unless `TREEGRRMHD_TEST_BACKEND=cuda`; no coupled run has been made on a device yet,
  so its device tolerances are unmeasured).
- **MPI with devices**: `jobs/device-mpi.toml`, `test/device_tests.jl` with
  `TREEGRRMHD_TEST_DEVICE_MPI=1` at two ranks of one H200 each on Symmetry: the device
  tests' four runs at both types and both recoveries, distributed on the devices against
  the same distributed on the hosts, under the single-device tolerances; 306 of 306 on
  each rank in 19 minutes.
- **CI** runs the MPI comparison as a job of its own (`TREEGRRMHD_TEST_TIER=mpi`); the
  other cells skip it (`TREEGRRMHD_TEST_MPI=0`), and the device runs stay on Symmetry.

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
  - An EOS wrapper forwards **every** method of EntropyEOS's interface,
    the two optional ones included (`evaluate_windowed`,
    `entropy_from_energy`): a missing one falls back to a generic default,
    silently and many times slower for a table ("Thermodynamics", "Warm
    starts").
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
- **A kernel's sum over directions is written out**, not
  `sum(ntuple(j -> f(…, Val(j)), Val(D)))` with the `Val` built from the
  closure's argument: that crashed Metal's compiler in the gauge kernel
  while every term compiled alone (`gauge_divergence`; "Precision and
  devices").
- **A collective call is gated by the forest, or by an agreed value.** A
  fill, a reduction, a regrid or a checkpoint that one rank skips leaves
  the others waiting (or worse, matched against another call): decide it
  from the replicated forest (`forest.periodic`, `forest.leaves`) or combine
  it first (`rank_reduce`), never from the rank's own blocks; a host
  diagnostic combines its sums before it forms a ratio ("Running
  distributed").
- **A table on a device** ("Devices"). An EOS reaches a kernel through
  `eos_on` (uploaded once, at `MHDProblem` and the case's initial data, hooks
  and interior); a struct that carries one into a kernel needs an `Adapt`
  rule; and a kernel's arguments stay under CUDA's 32 KiB, so a hook carries
  only what it reads.
- **A script picks the integrator's partition from the state**:
  `partition = u isa Array ? :even : nothing`, as `mhd_integrator` does;
  `:even` on a device array is refused ("Precision and devices").
- **A tabulated star's matter is not in its metric.** `TOVMetric` keeps
  the polytrope (`eq`), from which `tov_density` recovers the density, or
  `nothing` for a table; a tabulated star's `ρ★`, `s` and `Yₑ` are its
  `TOVComposition`, in `TabulatedTOVStar`, beside the metric
  ([Thermodynamics](#thermodynamics), "The tabulated star's surface and
  atmosphere"). `tov_density` has no method for a table's metric.
- **A table's barotrope ends at `ρ_min`.** `tabulated_tov_case` puts the
  surface `η = 0` at the atmosphere's density, so `R̄` and `M` depend on
  `ρ_min`, slightly (DD2's 1.4 M☉ star: 13.194 km at 10⁻¹⁰ against 13.225
  km at 10⁻¹⁴); the literature's radius is the lower surface's.

## Tests

The suite is `test/runtests.jl`. It runs in tiers (`test/tiers.jl`,
chosen by `TREEGRRMHD_TEST_TIER`; see "Julia version"): `full`, the
default, runs everything; `quick` leaves out the `@full_tier` testsets (the
finest curved runs, the 1000-step uniform flows, the vertex metric's and
the magnetized star's convergence studies); `threaded` also the
`@serial_tier` ones (the 1D studies); `mpi` runs `test/mpi_tests.jl` alone. The rules for a new testset are in
`CLAUDE.md`, "Conventions".

- `test/prerequisite_tests.jl`: the names the scheme calls, the IMEX
  contract, and a probe per upstream ID still waiting ([Requests still
  waiting](#requests-still-waiting), "Retiring a probe").
- `test/precision_tests.jl` (the bridges over `Base`'s gaps at a software
  floating-point type) and `test/units_tests.jl` (the geometric units and
  the density factor to an EOS's).
- `test/recovery_tests.jl`, `test/pointwise_tests.jl` and
  `test/stage_solve_tests.jl`, which share the state generators of
  `test/recovery_states.jl`.
- `test/step_cache_tests.jl` (the step cache) and `test/two_phase_tests.jl`
  (the two-phase recovery against fused, bit for bit).
- `test/scheme_tests.jl` and `test/wave_tests.jl` (the convergence
  studies).
- `test/shocktube_tests.jl`, `test/resistive_tests.jl`,
  `test/atmosphere_tests.jl`.
- `test/reconstruction_tests.jl` (with `wave_tests.jl`, uses
  `test/wave_study.jl`).
- `test/multid_tests.jl` and `test/multid_cases_tests.jl` (use
  `test/multid_runs.jl`).
- `test/refinement_tests.jl` and `test/refinement_cases_tests.jl` (use
  `multid_runs.jl` and `test/refinement_meshes.jl`).
- `test/curved_tests.jl`: the right-hand side's truncation error, the
  drift to `t = M/2` at `h = M/4`, `M/8`, the interior, the curved-space
  bit identity; its boundary testset holds a gauge offset out of `𝔅`.
- `test/tov_tests.jl`: its runs are 3D at `N = 8` and a few `M` long, so
  its testsets run in every tier, unmarked.
- `test/vertex_metric_tests.jl`: the vertex metric at every centering.
- `test/adm_matter_tests.jl`: `ADMMatter` on the Cowling star, bound to a
  vertex set of the suite's own making (no TreeGH), both forms and both
  orders, and `flags`.
- `test/magnetized_tov_tests.jl`: the exterior dipole and its interior
  continuation pointwise; the confined field (Duez et al.'s, at GRaM-X's
  `B_c`) and the squared density profile pointwise; σ's threshold as a
  density, the same σ at `ρ_min = 10⁻¹⁰` and `10⁻¹²` on the initial data and
  the studies' `10⁴ ρ_min` bit for bit; the magnetized star at `h = 1` to `2M` (∇·B, `E·B`,
  the drift, the rest mass, the outcome map); the convergence to `h = 1/2`
  is `@full_tier`.
- `test/ye_advection_tests.jl`: a smooth `Yₑ` profile carried
  by a uniform flow at the scheme's order, and a step within its two
  values, `D Yₑ` to rounding; 1D, so `@serial_tier`.
- `test/tabulated_tov_tests.jl`: the cold barotrope against
  DD2's slice, the solve's convergence and the 1.4 M☉ radius, the
  composition table, the initial data's validity, and the star at `h = 1`
  to 4 M (the drift, rest mass and `D Yₑ` net of the reset, the entropy
  floors). Every testset reads DD2 and is gated by `test/tables.jl`
  (`@dd2_gated`): where `ENTROPYEOS_TABLE_DIR` does not hold the table, as
  on GitHub, it is one skipped test, named at the end of the run. Symmetry's
  suite sets the variable (`jobs/full-suite.toml`).
- `test/threading_tests.jl`: the digests of three runs (`test/thread_workload.jl`: the
  magnetized atmosphere tube, the adaptive rotor, the magnetized star) against a
  subprocess at another thread count, line for line ("Precision and devices").
- `test/device_tests.jl`: the same four kinds of run on a device and on the host, compared
  ("Precision and devices"); in the suite the CPU stands in for the device. It also runs by
  itself, `TREEGRRMHD_TEST_BACKEND=cuda` or `metal` in an environment with the device
  package (`jobs/device-tests.toml`), and under MPI with `TREEGRRMHD_TEST_DEVICE_MPI=1`
  (`jobs/device-mpi.toml`).
- `test/checkpoint_tests.jl`: a restart bit for bit, and the refusals; and
  `test/mpi_tests.jl`, the distributed runs against the serial ones
  (`test/mpi_workload.jl`, launched by `test/mpi_jobs.jl`, compared by
  `test/mpi_lines.jl`; "Running distributed"). The `mpi` tier runs only the latter, as
  CI's MPI job does.

**The coupled suite** is `coupled/runtests.jl`, in `coupled/`'s own
environment, which loads TreeGeneralizedHarmonic beside this checkout
("The coupling contract"); it shares `test/tiers.jl`. GitHub CI runs it in
a job of its own at one thread with bounds checks (`CLAUDE.md`,
"Commands").
- `coupled/prerequisite_tests.jl`: GH-1, GH-3 and GH-4 detected and
  asserted; GH-2 and GH-7 reported.
- `coupled/coupled_tests.jl`: TreeGH's `ADMMetric` against the analytic
  star; the frozen metric against the Cowling run; GH-3's static-matter
  check; `ℋ − 16πρ` of the initial state, with either transfer; the
  script's `main`; the fill's
  key, bit for bit; the regrid branch; the magnetized star through
  `problems(; case)`: its field's energy `½B²` in the matter TreeGH reads,
  and a run to `t = 1` (∇·B, the rest mass, the stage solve inside).
- `coupled/tabulated_tests.jl` (gated on DD2 as above): the
  tabulated star on TreeGH's frozen metric against its Cowling run, its
  `ℋ − 16πρ` with either transfer, and `main` to `t = 1` (rest mass and
  `D Yₑ` net of the reset).
- `coupled/device_tests.jl` and `coupled/mpi_tests.jl` (with
  `coupled/mpi_workload.jl`): the coupled star on a device against the host,
  and at two ranks and one against the serial run, its study resumed from a
  two-rank checkpoint ("Running distributed").

## Code layout

Where each piece of the design lives. The sections named say what it is
and why.

**The pointwise layer** (each runs inside a KernelAbstractions kernel,
bit-identically to the host; "Precision and devices"):
- the module shell, `src/TreeGRRMHD.jl`; `src/precision.jl` and `src/device.jl`, ported from
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
- `src/adm_matter.jl`: `ADMMatter`, the matter adapter of the coupling
  contract (GH-3): the mean of the `2³` cells' `(𝓔, S_i, S_ij)` at a
  producer's owned vertex, per call from `P`, or, stored in
  `adm_matter_set`'s cells, the mean or the 4th-order interpolation from
  `4³` cells ("Matter to the vertices");
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
- `src/evolution.jl`: `MHDProblem` (which carries the step cache, the
  hand-off and the kernels' `workgroupsize`), the kernels, `mhd_rhs!` and its
  phases (`fill_potentials!`, `curl_A!`, `recover_cells!`, `fill_primitives!`,
  `face_fluxes!`, `edge_emfs!`, `tendencies!`), `ohm_solve!`,
  `reset_atmosphere!`, `ohm_tally`, `max_light_rate`, the conserved totals ("The
  right-hand side", "Precision and devices");
- `src/driver.jl`: `MHDCase`, exact initial data and references,
  `l1_errors`, `mhd_integrator`, `evolve!` (a static mesh, `refine`, or an
  adaptive one, `maxlevel`, `refine_tol`, `coarsen_tol`, `buffer`;
  `adapt_case!` over every set), `central_density` (the largest cell
  average of `ρ`), and `DirichletBoundary`, the Dirichlet hooks a
  non-periodic case gets from its own data ("The partition", "Boundaries",
  "Refinement");
- `src/refinement.jl`: `mhd_flags` (Löhner on `ρ`, `|B|`, `|E|`), `flags`
  (`prepare!` then `mhd_flags`, the verb), `regrid_pairs`, `regrid_state!`
  ("Refinement");
- `src/interior.jl`: `HoleInterior(r_in; stage, step)`, an `MHDCase`'s
  `interior`, and the diagnostics `outside_l1`, `field_l1`,
  `charge_norms` ("The interior");
- `src/tov.jl`: the host TOV solve in Lindblom's enthalpy form, generic
  over the barotrope (`barotrope_state`), `TOVMetric` ("Gauge for the TOV
  milestones").
- `src/tov_table.jl`: `cold_barotrope`, a table's cold
  β-equilibrium slice as the solve's barotrope, `tov_solve(T, b; ρ_c)`,
  and `TOVComposition`, the star's matter at knots in `r̄²`
  ("Thermodynamics", "The cold barotrope").
- `src/tov_pulsation.jl`: the star's radial modes from linear perturbation
  theory, coupled and Cowling, `tov_radial_modes`, and the velocity that
  starts them, `tov_mode_velocity` ("Validation ladder" 5).
- in `src/evolution.jl` too: `reset_step!`, the reset as the step
  limiter, counted with the step limiter's injection into `D`, `τ` and
  `D Yₑ`, and its entropy floors outside the atmosphere (`ohm_tally`);
  `max_div_B` and `evolve!`'s `after_step` monitor.
- The public verbs, which a script that owns the loop calls ("The coupling
  contract", "Two scripts"): `prepare!`, `rhs!`, `stage_solve!`,
  `stage_limit!`, `step_limit!`, `max_rate`, `initial_state!`, `flags`,
  `MHDProblem(fields, case; …)`; `mhd_rhs!`, `ohm_solve!`, the resets,
  `mhd_integrator` and `evolve!` are compositions of them, bit for bit.

**The cases** (`src/cases/`; "Validation ladder"):
- `src/cases/em_wave.jl` and `src/cases/alfven_wave.jl`: `em_wave_case`,
  `alfven_wave_case`;
- `src/cases/gauge.jl`: `gauge_pulse_case`, a pure gauge pulse, the test
  of the gauge damping `ξ` ("Refinement");
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
- `src/cases/tov.jl`: `tov_case`, `tov_boxes!`;
- `src/cases/magnetized_tov.jl`: `exterior_dipole` (the static vacuum
  dipole of Schwarzschild in isotropic coordinates), `StellarDipole` and
  `stellar_dipole` (its quadratic continuation inside the star),
  `ConfinedField` and `confined_field` (Duez et al.'s interior-only field),
  `magnetized_tov_case` ("Validation ladder" 6);
- `src/cases/tabulated_tov.jl`: `tabulated_tov_case`, the star
  on a table's cold barotrope, `TabulatedTOVStar` ("Validation ladder" 7);
- `src/cases/ye_advection.jl`: `ye_advection_case`, a `Yₑ`
  profile on a uniform flow.

**The field diagnostics** (`src/field_diagnostics.jl`, host-side, outside a
step): `cell_profile`, `field_diagnostics`, `field_drift`, `ohm_map`,
`gather_profile` ("Validation ladder" 6, "Primitive recovery").

**Running distributed** ("Running distributed"):
- `src/distributed.jl`: `rank_reduce`, `rank_sum!`, `agree_any`,
  `agree_refusal`, `isroot`, `gather_cells`, on TreeAMR's public
  communicator verbs;
- `src/checkpoint.jl`: `save_run`, `load_run`, `restore_run!`,
  `checkpoint_recipe`, `check_recipe`, on TreeIOHDF5; `evolve!`'s
  `checkpoint`, `checkpoint_every` and `restart` in `src/driver.jl`;
- `bin/mpi.jl`: the scripts' `mpi=true` (`withcomm`, `withrun`,
  `select_device!`).

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
  rung-4 record ("Validation ladder" 4), `bin/tov_cowling_cli.jl` the
  script's command line; `bin/tov_study.jl table=dd2|sfho`
  the rung-7 record, its tables read by `bin/tables.jl` from
  `ENTROPYEOS_TABLE_DIR`, with `diag=` where the matter outside the star
  is ("Validation ladder" 7);
- `bin/magnetized_tov_study.jl`: the magnetized star's time series and
  outcome map per chunk ("Validation ladder" 6);
- `bin/recovery_cost.jl` (with its instrumentation,
  `bin/recovery_cost_hooks.jl`): `steptime` times the fused and the
  two-phase recovery, `finish` counts the share that reaches finish
  ("Primitive recovery");
- experiments kept for their records, not part of the package:
  `bin/endpoint_measure.jl` and `bin/hot_projection.jl` (the endpoint
  branches behind EOS-9), `bin/front_investigation.jl` (the magnetized
  atmosphere tube's chaos), `bin/ghost_entropy.jl` and
  `bin/ghost_robustness.jl` (the matter's ghost fills; "Refinement");
- `bin/benchmark.jl`: the per-phase table, ns per cell of each phase of a step on the
  magnetized star, the polytrope or a tabulated star (`case=dd2`, `sfho`, through
  `bin/tables.jl`), on any backend and type; `bin/profile.jl`, the same problem's steps
  (or one phase) under Julia's sampling profiler, with each thread's share asleep or
  spinning, the allocations and the GC, or on a device CUDA.jl's kernel profile;
  `bin/backend.jl`, TreeHydro's, resolves their and `bin/tov_cowling.jl`'s `backend=` and
  `type=` ("Precision and devices"), and `bin/mpi.jl` their `mpi=true` and every rank's
  device ("Running distributed").

**The coupled runs** (`coupled/`, its own environment; "The coupling
contract"):
- `coupled/tov_coupled.jl`: the coupled script ("Two scripts"), with its
  command line in `coupled/tov_coupled_cli.jl`; `coupled/gh7.jl`, TreeGH's
  regrid verbs and a segment's constraint norms composed from its exports
  (GH-7 workaround);
- `coupled/tov_coupled_study.jl` (checkpointed, for `amdpreq`), with
  `bin/tov_modes.jl ref=coupled` and `bin/tov_overtones.jl` (the matrix
  pencil on a perturbed run minus an unperturbed one): the rung-5 record
  ("Validation ladder" 5); `coupled/gh_cfl.jl`: TreeGH's CFL factor under SSP3 ("Time
  integration"); `coupled/coupling_cost.jl`: the fills and the matter
  adapter's cost ("Two scripts", "Matter to the vertices");
- `coupled/magnetized_tov_study.jl`: the magnetized star in the evolved
  spacetime, `bin/magnetized_tov_study.jl`'s measurements and TreeGH's
  constraints per chunk (`problems(; case)`; "Validation ladder" 6);
  `coupled/tov_coupled_study.jl table=dd2` and `coupled/tov_coupled.jl
  table=dd2`, the tabulated star ("Validation ladder" 7).

`ci/full-suite.sh` (the workstation) and `ci/full-suite-cluster.sh` (a
cluster, through SimLaunch) run the full suite as a commit status, and
`ci/suite-summary.sh` reads Test's summary from a log for both
(`CLAUDE.md`, "Commands"). `jobs/` holds the SimLaunch job specs, one per
kind of cluster job: `full-suite.toml`, the suite at the floor at one
thread, at four and with bounds checks, side by side on one node;
`device-tests.toml`, `test/device_tests.jl` on one H200 (or the laptop's GPU);
`device-mpi.toml`, the same under MPI on two H200s ("Running distributed");
`tov-cowling-device.toml`, the Cowling star on an H200 at both types and on the node's
CPU; `benchmark-gpu.toml` and `benchmark-cpu.toml`, the per-phase table on an H200 and
over thread counts on a pinned Zen 3 node, and `benchmark-partition.toml`, the stage
arithmetic's three partitions there ("Precision and devices"); `perf-cpu.toml`,
`perf-gpu.toml`, `perf-mpi-node.toml`, `perf-mpi-nodes.toml` and `perf-gpu-mpi.toml`, the
DD2 star's table and profile over threads, on an H200, over ranks on a node, over nodes and
over H200s ("The tabulated star's cost");
`tabulated-tov-cowling.toml`, `tabulated-tov-coupled.toml`, `-coupled-mid.toml` and
`-coupled-fine.toml`, the tabulated star's studies, each run a job of its own, with
`tabulated-tov-coupled-damping.toml` and `tabulated-tov-coupled-box.toml`, the same coupled
star at other constraint damping and on wider boxes, and `tabulated-tov-coupled-wide.toml`,
`-wide-mid.toml` and `-wide-fine.toml`, the coupled study at `L = 96`, and
`tabulated-tov-atmosphere.toml` and `tabulated-tov-cowling-long.toml`, where the
coarse star's matter goes and the Cowling star to 1000 M ("Validation ladder" 7).

## Validation ladder

Each rung is a set of testsets named as claims, and every rung is done
*(Done.)*: built, tested in the suite, and measured in the studies below.
What each rung's step asked for (its acceptance list) and how it went are
in [HISTORY.md, "The steps of the plan"](HISTORY.md#the-steps-of-the-plan).

1. **Flat space, 1D** *(Done.)*: the vacuum electromagnetic wave (`Vacuum`) at the
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
     `D s` at 2.00–2.02 — the rung's prediction, ≥ 1.9, holds. `S_y`, `S_z`,
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
     the rung's prediction, ≥ 0.7, holds but for test 2's `D` and `S_x` —
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
     10⁻³.
   - **The plateaus** against Giacomazzo & Rezzolla's exact solution
     (2006, Tables 7–11; `BALSARA_EXACT`, whose `p` is the total pressure),
     averaged at 400 cells over each region's window (`BALSARA_PLATEAUS`,
     where the 3200-cell run holds the region to 2%, less four cells either
     side), with `ρ` and `p` relative to their own value and each component
     of `v` and `B` to its largest magnitude in the solution
     (`plateau_error`): every window within 2% —
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
     10³, 10⁶ — monotone, the rung's claim — and falls as `σ^−0.96` from 10³
     to 10⁶ (`σΔt ≈ 2000` there). The stages: all `OHM_SIGMA_ZERO` at σ = 0;
     all fixed point at σ = 10; 94,930 fixed point and 65,070 Newton of
     160,000 at 10²; 3,225 and 156,775 at 10³; 817 and 159,183 at 10⁶, every
     Newton stage there from the ideal seed; no fallback, at most 6
     recoveries in a stage, no repair, no reset.
   - **The current sheet** (Komissarov 2007, §6.1.4: `B₀ = 1`, `p = 50`,
     `ρ = 1`, `σ = 100`, `Γ = 4/3`, on `[−3/2, 3/2]` from the profile at
     `t = 1`, with the quasi-static `Eᶻ = ∂ₓBʸ/σ` where the paper starts
     from `E = 0`): the erf profile solves the diffusion limit
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
   - **The atmosphere** (`atmosphere_tube_case`: a gas
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
2. **Flat space, 2D and 3D, with AMR** *(Done.)*: the cylindrical explosion, the
   magnetic rotor, Orszag–Tang, field-loop advection across coarse-fine
   faces — `∇·B` at round-off, rest mass and magnetic flux conserved
   across levels to round-off, regrids included.

   *Rung 2a* (`test/multid_tests.jl`, `test/multid_cases_tests.jl`,
   `test/reconstruction_tests.jl`; uniform meshes, HLLE, SSP3(4,3,3), `cfl`
   2/5 — 1/5 under MP5 — over `D` in `Δt = cfl/(D max λ/h)`; the cases in
   `src/cases/multid.jl` and `src/cases/geometry.jl`, every published number
   copied with its citation; Julia 1.11 and 1.13, one and four threads —
   identical at one and four threads, and between 1.13 and 1.11 (or
   `--check-bounds=yes`) the same to the digits recorded but the
   conservation drifts' last digit):
   - **`∇·𝔅`** after every step of every 2D and 3D run of the suite
     (`max_div_B` as the driver's `after_step`): at most 2.4e−16 of
     `max |𝔅|` (asserted ≤ 1e−12, the rung's bound).
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
     GRaM-X runs it, `t = 1`, 27 steps): symmetric under `x → −x` and under
     `y → −y` with the fields reversed, exactly since both walls are
     prescribed ("Boundaries"; 1.6e−18 and 2.5e−17 of each variable's scale
     before; asserted ≤ 1e−12); no repair, no reset. Under MC at
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
   are measured — the default fill is now the parent-centred one,
   Conservative width 3 bounded ("Refinement"), which these tables were not
   rerun with but for the number marked so below; the values before those
   two changes are in
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
3. **Analytic curved backgrounds** *(Done.)*: the Wald solution (`Vacuum`,
   Kerr-Schild) and the Michel monopole or magnetized Bondi flow
   (`IdealConductor`).

   *Rung 3* (`src/cases/kerr_schild.jl`,
   `src/interior.jl`, `test/curved_tests.jl`, `bin/curved_stationarity.jl`):
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

   **Stationarity** (the acceptance: the L1 drift after `t = 50M` converges at
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
   every variable but the charge `√γ q` (1.68 Wald, 1.63 Bondi); the
   `M/4 → M/8` rates 1.69 and 1.71 stand as measured — at
   `M/4` only 8 cells lie between the hole's centre and the horizon.

   **Option 1** — the case's own potential in the
   ghosts where it does not depend on `t`. *As built* (`static_potential` on
   `MHDCase`, `absolute` on `DirichletBoundary`, "Boundaries"): `wald_case` and
   `bondi_case` declare their potentials static (as do the stars), and
   their potentials' hooks are `CellBoundary(AllVariables(…))` of the
   case's `A_j` and `Φ̃`; the flat-space cases keep the relative hook.
   Options 2 (a relative hook that carries the offset's gradient, upstream)
   and 3 (a gauge that damps a static `χ`) are not built; they matter for
   dynamic cases without an exact `A` ("Possible extensions"). `bin/curved_localise.jl`'s
   `relhook=true` is the negative control.

   **With the relative hook** (AMR-7; `relhook=true`) the drift does not
   converge: Wald saturates, the magnetized Bondi flow converges at first
   order. The L1
   drift outside `r = 2M` after `t = 50M`, both limiters, `r_in = M`, and the
   rates `M/4 → M/8` and `M/8 → M/16` (measured with `bin/curved_stationarity.jl`
   before the cases became static, reproducible with `bin/curved_localise.jl
   relhook=true`; jobs
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
     r₋ = 0.13M` and is flat. **Not done**, left for future black-hole work
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
     **Not localised further**; the runs below are left for future
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
   cache (fused, the CPU's default) and the interior, 16 steps of `Δt = 1/64` in
   one integration, in two, and as a restart from the saved state with a
   rebuilt problem are the same bit for bit, for both cases; the suite prints
   their checksums, which the one- and four-thread runs print alike. The
   `· inv(√γ)` keys of "The step cache", written for curved space and first
   exercised here, hold.
4. **TOV in the Cowling approximation** *(Done.)*: a Γ = 2, K = 100 polytrope
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

   **Stationarity** (the acceptance: the drift after `200M` converges at `≥ 1.5`;
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

   **Conservation** (the acceptance: rest mass to round-off, net of the outer
   boundary flux; it holds): in every run the change of the total
   `D` equals the mass the step limiter injected (`ohm_tally`'s
   `injected_D`) to `1.3–7.7·10⁻¹⁵` of a rest mass of 1.506 at `200M` and
   at MC's `2000M` (4.2e−15), and to 3.1e−14 at `h = 0.25` (5400 steps on
   eight times `h = 0.5`'s cells, accumulated rounding), so nothing leaves
   through the outer faces. The injection is `+1.3` to `+1.9·10⁻⁷` with MC
   and `−3.5` to `−4.2·10⁻⁷` with PPM: PPM's reset removes mass, MC's adds
   it.

   **Modes** (the acceptance: F, H1 and H2 within 3% of Font et al. 2002, Table I,
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

5. **TOV in a dynamical spacetime** *(Done.)*: the frozen TreeGH metric, through its
   adapter, reproducing rung 4; the F, H1, H2 radial
   modes against the literature (Font et al. 2002); convergence of the GH
   and ADM constraints with matter (needs GH-2 to GH-4).

   *As built* (`coupled/`; Symmetry, EPYC 7532 and 7543). Two
   setups:
   - **The coupled suite:** step 10's star on `[−16, 16]³`, two roots, no
     boxes, MC, at `N = 10` and `N = 20`.
   - **The studies:** `[−24, 24]³`, four roots, one box level, PPM, at
     `N = 10, 12, 16` (`h = 0.6, 0.5, 0.375`), to 1000 M, with TreeGH's
     damping `γ0 = 1/(10M)`. Run on amdpreq, jobs 595407–595409.

   The results:
   - **The frozen metric:**
     - TreeGH's `ADMMetric` of the star equals `SampledADM`'s at every
       owned vertex: `α` to 1.1e−16, and `β`, `γ` and `K` exactly.
     - The Cowling run on TreeGH's frozen metric equals step 10's script
       to 3.4e−16 relative at `t = 2`. That is round-off, not the
       truncation error the rung predicted: for the static star with no
       shift, TreeGH's `K` is 0 exactly, as the analytic one is.
   - **GH-3's static-matter check:** TreeGH alone on the star, with
     `ADMMatter` held fixed, drifts from its initial state at second
     order. At `t = 4` the RMS drift is 2.33e−4 at `N = 10` and 6.39e−5 at
     `N = 20` (rate 1.87). Without the matter it is 3.49e−3 at `N = 20`,
     55 times larger.
   - **`ℋ − 16πρ` at `t = 0`:**
     - inside `r < 6`: 2.19e−3 and 6.17e−4 at `N = 10, 20` (rate 1.83);
     - outside `r ≥ 10`: 2.7e−6 and 8.8e−8;
     - without the matter, inside: 3.39e−2.
   - **Rest mass:** it changes by exactly what the reset injects, to
     1e−14 relative. At `N = 10` to `t = 1`, with `ρ_min = 1e−12`, the
     change is −1.0960122e−12 and the injection −1.0976710e−12.
   - **The constraints over time** (the studies, jobs 598575–598577, with
     the fourth-order transfer and `ρ_min = 1e−12`): the `‖·‖₂` norms with the matter, inside
     `r < 6` and outside `r ≥ 10`, each averaged over a window. The rates
     between `h = 0.6, 0.5` and between `h = 0.5, 0.375`:

     | `t` (M) | `ℋ` inside | `ℳ` inside | `ℋ` outside | `ℳ` outside |
     |---|---|---|---|---|
     | 0 | 3.93, 3.97 | — | 4.33, 4.31 | — |
     | 20–30 | 2.07, 2.30 | 2.39, 2.42 | 1.59, 2.13 | 1.70, 2.12 |
     | 95–105 | 2.28, 2.38 | 2.32, 2.43 | 2.42, 2.31 | 2.63, 2.04 |
     | 240–260 | 2.20, 2.28 | 2.21, 2.39 | 2.18, 1.95 | 2.05, 2.06 |
     | 490–510 | 2.01, 2.18 | 2.08, 2.31 | 2.27, **1.76** | 2.30, 1.88 |
     | 740–760 | 1.91, 2.10 | 2.00, 2.26 | 2.19, 2.46 | 2.29, 2.44 |
     | 900–1000 | 1.81, 2.03 | 1.88, 2.21 | 2.09, 2.50 | 2.02, 2.40 |

     **The acceptance's ≥ 1.8 holds on the finer pair, inside and outside, to
     1000 M, but for `ℋ` outside over 490–510 M (1.76): a miss.** The
     windows on either side give 2.01 (500–520 M) and 1.96 (480–500 M),
     and the whole of 250–1000 M gives 2.03 and 2.24. A 20 M window
     outside is noisy at about ±0.2: at `ρ_min = 1e−10` the same window
     gave 1.93, and a 50 M window (350–400 M) 1.76 ([HISTORY.md, "Validation
     ladder"](HISTORY.md#validation-ladder), the scan of `ρ_min`). The
     coarse pair falls below 1.8 only outside over 20–30 M. The norms do
     not grow. At `h = 0.375`, `ℋ` inside rises from 5.4e−7 at `t = 0` to
     1.3e−5 by 25 M, the matter scheme's own error once the star moves,
     and stays at 9.5–9.7e−6 to 1000 M. Outside it is 7.1e−7 over
     20–30 M and 0.9–1.3e−7 from 250 M on. At `ρ_min = 1e−10`, with the mean, the outside norm
     was 3 times larger over 20–30 M and 10–14 times from 250 M on; it
     grew after 500 M, and its rate fell to 1.74 over 900–1000 M
     ([HISTORY.md, "Matter to the
     vertices"](HISTORY.md#matter-to-the-vertices)): most of the error
     outside the star was the mean's, carried out by the spacetime.
     Inside, the fourth order's `ℋ` is 36% below the mean's at 1000 M and
     its `ℳ` 24% above (1.05e−5).
   - **The central density** oscillates in the F mode and drifts down, the
     drift converging. Relative to `t = 0`, averaged over a window:

     | `t` (M) | `h = 0.6` | `h = 0.5` | `h = 0.375` | rates |
     |---|---|---|---|---|
     | 50–70 | −3.15% | −1.96% | −0.95% | 2.60, 2.52 |
     | 240–260 | −5.03% | −2.90% | −1.23% | 3.02, 2.99 |
     | 490–510 | −7.78% | −4.50% | −1.88% | 3.00, 3.03 |
     | 900–1000 | −11.99% | −7.06% | −2.89% | 2.91, 3.11 |

     After 200 M it drifts at 9.9e−5, 5.9e−5 and 2.4e−5 per M (a
     linear fit, relative; rates 2.8 and 3.2). The mean's initial drop
     was four times larger (−3.79% over 50–70 M at `h = 0.375`) and its
     later drift about the same.
   - **Rest mass** over the 1000 M: its change minus what the reset
     injected is 5.36e−12, 3.21e−12 and 1.33e−12 of 1.506 (rates 2.8,
     3.1), a hundredth of what it is at `ρ_min = 1e−10`. The box gains it:
     the reset removes 1.57e−8 at `h = 0.375`, and the total falls by
     1.33e−12 less. That is truncation error, not round-off, unlike the
     suite's 1e−14 at `t = 1`.
   - **Both parts are the atmosphere's.** At `h = 0.5` to 300 M, with
     `ρ_min` = 1e−10, 1e−12 and 1e−14 (jobs 598455–598457), the reset
     removes 6.06e−7, 4.85e−9 and 3.58e−11, and the gain net of it is
     1.01e−10, 1.03e−12 and 1.5e−14 (the last at round-off). Both scale
     with `ρ_min`, and the gain has the same history at each value: none
     before 20 M, then increments in the same ratio. The coarse-fine
     faces conserve to round-off and the reset is counted, so the gain
     enters through the outer faces, whose Dirichlet data is the
     atmosphere at rest and where TreeGH's shift is not zero. Nothing else
     changes with `ρ_min`: `ρ_c` agrees to 3e−5 relative throughout, `ℋ`
     and `ℳ` inside to 5%, and outside within ±25% at any instant (+1–2%
     on average). The reset writes 185,000 of the 207,360 cells a step at
     each value. The energy it removes grows as `ρ_min` falls, −3.8e−5,
     −4.8e−5 and −5.0e−5 of `τ` over the 300 M, the difference arising
     after 25 M. So the coupled script's `ρ_min` is 1e−12, below
     `tov_case`'s 1e−10: the rest-mass budget falls a hundredfold, and the
     EOS's lower bound (1e−14) keeps a factor 100 of margin.
   - **Modes** (`bin/tov_modes.jl ref=coupled`, the whole series, 0.20 kHz
     resolution, of the fourth-order studies, jobs 598575–598577), against
     Font et al. (2002), Table II (1.442, 3.955, 5.916 kHz):

     | `h` | F | H1 | H2 |
     |---|---|---|---|
     | 0.6 | 1.431 (−0.74%) | 3.898 (−1.43%) | 5.292 (**−10.56%**) |
     | 0.5 | 1.432 (−0.69%) | 3.922 (−0.83%) | 5.256 (**−11.15%**) |
     | 0.375 | 1.442 (−0.02%) | 3.930 (−0.64%) | 6.497 (**+9.82%**) |

     **F and H1 hold the 3%; H2 fails it.** F's peak is 7–27 times
     H1's, and the peak in H2's window is 4–10 times below H1's and moves
     with `h`: the start excites F, H1 a little, and H2 not measurably.
     The magnetized star of rung 6 (its confined field, σ, and `ρ_min =
     10⁻¹⁰`) gives the same to four digits over 2–300 M at `h = 0.5`, and
     `ρ_min` (10⁻¹⁰, 10⁻¹², 10⁻¹⁴; jobs 598455–598457) moves F and H1 by
     at most 0.08%.
     **The transfer decides H1.** With the mean (jobs 595407–595409,
     `ρ_min = 10⁻¹⁰`) F is +0.71%, +0.78%, +0.67% and H1 misses
     (−19.81%, +6.82%, +8.38%), as does H2 (−10.71%, −10.44%, −10.56%).
     The mean's start, `ℋ` inside 100–260 times the fourth order's at
     `t = 0`, kicks F 9–14 times harder and the spacetime's lines about
     ten times harder: 2.8, 5.3 and 7.3 kHz, the strongest lines of `ℋ`
     inside the star, which appear in `ρ_c`. H1's window then holds a
     line larger than anything near H1 (3.17, 4.23 and 4.29 kHz by `h`;
     at `h = 0.375` a smaller peak at 3.82). The pair at `h = 0.25` to
     300 M that differs only in the transfer (jobs 596587, 596600) puts
     H1 at +6.10% with the mean and +0.76% with the fourth order. Why the
     mean's F sits 0.7% high at every `h` is not separated (its
     amplitude is larger, and its `ρ_c` drifts more).
   - **The perturbed star** (to excite the overtones above the
     spacetime's lines). `tov_case(T; δv = (a₁, a₂, a₃), δv_modes)` starts
     the star moving in its radial modes F, H1 and H2 at the phase of zero
     displacement, mode `n` with the amplitude `aₙ` in `ρ_c/ρ_c` (`aₙ > 0`
     expands the centre first). The velocity is the modes' eigenfunctions
     inside the star and nothing outside. The density, the entropy and the
     metric stay the static star's, and the surface and the Dirichlet
     faces are untouched. The default `(0, 0, 0)` branches to the star at
     rest, bit for bit. A sine-shaped velocity came first and did not
     excite the overtones (`HISTORY.md`, "a velocity perturbation to excite
     the overtones").
     - **The modes** (`src/tov_pulsation.jl`): adiabatic radial
       perturbations of the TOV solution, `ξ = Δr/r` and `ζ = Δn/n`, in the
       log-enthalpy on `tov_solve`'s RK4 nodes, so that the centre and the
       surface are nodes. `δv_modes = :coupled` takes Chandrasekhar's
       equation with the metric perturbed, in Gondek, Haensel & Zdunik's
       (1997) first-order form. `:cowling` holds the metric fixed. There is
       no default: a run uses the version that matches its spacetime
       (`coupled/` the first, `bin/tov_study.jl` the second). Shooting from
       the centre, the residual at the last node before the surface is the
       bracket that a regular solution makes vanish there. Its sign changes
       are bisected to round-off. The frequencies, converged to 10⁻⁶ in the
       step count:

       | | F | H1 | H2 |
       |---|---|---|---|
       | coupled | 1.4425 (+0.036%) | 3.9540 (−0.024%) | 5.9149 (−0.019%) |
       | Cowling | 2.6861 (−0.55%) | 4.5494 (−0.30%) | 6.3411 (−0.17%) |

       against Font et al.'s (2002) Table II, which is linear perturbation
       theory given to four digits, and Table I, which is a 2D code's. The
       coupled modes reproduce Table II to its last digit, so their
       equations and the star are the paper's. The Cowling ones are not
       Font et al.'s method. Rung 4's measurements at `h = 0.375` (2.677,
       4.563, 6.492) are within 0.4%, 0.3% and 2.4% of them.
     - **The velocity:** the Valencia `vⁱ = g(r̄²) xⁱ` with `g = ω ξ/(α
       √(1 − 2m/r))`, which is even and smooth in `r̄`. `Σₙ aₙ gₙ/3` is
       fitted by 12 Chebyshev terms in `2r̄²/R̄² − 1` over `r̄² ≤ 0.99 R̄²`,
       to 10⁻⁷ of `g` for H1 and H2, and extrapolated over the last 1%:
       the last nodes before the surface carry RK4's error of the stiff
       `1/η` term. `TOVStar` stores the coefficients, so it stays `isbits`.
     - **The constraints:** the data violate the momentum constraint by
       `8π S_i` (`K_ij = 0` with `S_i ≠ 0`), at order `a` and independent
       of `h`. TreeGH's damping is left to remove it. The mode's own `K_rr`
       would satisfy it at linear order, but it is TreeGH's initial data
       and is not built. The sine-shaped velocity decayed the same way
       without TreeGH, so the violation is not what removes a
       perturbation. The perturbed runs measure the modes only; the
       constraints, the drift and the rest mass stay the unperturbed runs'.
     - **The studies:** `δv = (0, 5·10⁻³, 5·10⁻³)`, so that H1 and H2 each
       move `ρ_c` by 0.5%, a sixth of the 3% the start gives F. The coupled
       runs are at `h = 0.6, 0.5, 0.375` to 1000 M (jobs 596447–596449)
       with the mean, and at `h = 0.25` to 300 M with either transfer
       (jobs 596586, 596587, 596599, 596600). A Cowling run at `h = 0.6`
       to 150 M (job 596446) is compared with the unperturbed run on the
       same mesh. The overtones are measured in the difference
       `Δρ_c/ρ_c` between the perturbed and unperturbed runs, by a
       matrix-pencil fit to damped sinusoids (`bin/tov_overtones.jl`,
       orders 6–10; the ranges below are over the orders). The unperturbed
       signal's spacetime lines cancel there.

       | run | H1 kHz | τ(H1) | H2 kHz | τ(H2) |
       |---|---|---|---|---|
       | Cowling, `h = 0.6` | 4.38–4.52 (−0.6 to −3.7%) | 50–62 M | not resolved | |
       | mean, `h = 0.6` | 3.80–3.83 (−3.2 to −3.9%) | 73–78 M | 6.12–6.39 (+3.5 to +7.9%) | 40–42 M |
       | mean, `h = 0.5` | 3.89–3.96 (−1.6 to +0.2%) | 89–101 M | 5.65–6.06 (−4.5 to +2.5%) | 47–64 M |
       | mean, `h = 0.375` | 3.92–3.95 (−0.9 to −0.1%) | 122–154 M | 5.76–6.03 (−2.6 to +1.9%) | 78–88 M |
       | mean, `h = 0.25` | 3.956–3.960 (+0.0 to +0.1%) | 435–441 M | 5.923–5.939 (+0.1 to +0.4%) | 138 M |
       | 4th order, `h = 0.25` | 3.987–3.989 (+0.8 to +0.9%) | 384–389 M | 5.943–5.952 (+0.5 to +0.6%) | 134–135 M |

       against this module's frequencies (Cowling) and Font et al.'s
       Table II (coupled). H1 and H2 are within the 3% target from
       `h = 0.375`, and within 1% at `h = 0.25`, where the fits agree
       across the model orders to 0.1%. The fits leave residuals of 7–16%
       of the signal. The overtones decay, and `τ` grows as `h` shrinks:
       H1's by 4.3–5.0 from `h = 0.5` to 0.25, about `h⁻²`. The damping
       is linear: Cowling runs at `a = 10⁻³` and `2.5 · 10⁻⁴` (jobs
       596572, 596573) decay as the `5 · 10⁻³` run does per unit
       amplitude, within 5% over the first 75 M, and the smallest if
       anything faster. So it is the scheme's dissipation of a mode a few
       cells per node across, not a surface or atmosphere effect at large
       amplitude.
   - **The transfer order:** fourth order, the script's default ("Matter to the
     vertices"). At `t = 0` the interpolation's `ℋ` inside the star is
     160–400 times smaller than the mean's and converges at 3.9–4.0; from
     250 M to 1000 M, `ℋ` outside is 10–14 times smaller and keeps its
     rate, which
     the mean's lost ("The constraints over time" above). The modes do not decide it. Its
     runs with and without `δv` to 300 M (jobs 596595–596600), against
     the mean's over the same window:

     | `h` | transfer | H1 kHz | H2 kHz |
     |---|---|---|---|
     | 0.5 | mean | 3.89–3.92 (−0.9 to −1.7%) | 5.76–6.08 |
     | 0.5 | 4th order | 3.95–4.00 (−0.2 to +1.1%) | 6.06–6.21 |
     | 0.375 | mean | 3.90–3.95 (−0.1 to −1.3%) | 5.81–6.03 (−1.8 to +2.0%) |
     | 0.375 | 4th order | 3.94–4.02 (−0.3 to +1.6%) | 5.85–6.07 (−1.1 to +2.6%) |
     | 0.25 | mean | 3.956–3.960 (+0.0 to +0.1%) | 5.923–5.939 (+0.1 to +0.4%) |
     | 0.25 | 4th order | 3.987–3.989 (+0.8 to +0.9%) | 5.943–5.952 (+0.5 to +0.6%) |

     Both transfers are within 1% at `h = 0.25`. There the 4th order
     sits 0.8% above the mean in H1 and 0.3% in H2, more than the fits'
     spread, so the difference is real. It shrinks with `h` (about 1.8%
     in H1 at `h = 0.5`, from the ranges' middles), as two consistent
     transfers' must. The mean is the closer to linear theory at
     `h = 0.25`, which one resolution with each cannot tell from a
     cancellation of the mean's error against the scheme's. Whether the
     4th order's 0.8% closes at a finer `h` is not measured ("Possible
     extensions"). H1's damping
     does not depend on the transfer (`τ` 384–389 M against 435–441 M at
     `h = 0.25`), since it comes from the matter scheme.
6. **Magnetized TOV** *(Done.)*: an interior poloidal field from an analytic `A`,
   σ(ρ) through `DensityProfile`, an electrovacuum dipole exterior: the
   magnetized star of Dionysopoulou et al. (2013), whose field is LORENE's,
   and here the exact static form (Wasserman & Shapiro 1983). Palenzuela
   (2013) has a force-free exterior around a rotating star instead.

   *As built* (`magnetized_tov_case`,
   `src/cases/magnetized_tov.jl`; `test/magnetized_tov_tests.jl`;
   `bin/magnetized_tov_study.jl`; the diagnostics in
   `src/field_diagnostics.jl`):
   - **The spacetime**: the evolved one. Rung 6 runs coupled, through
     `coupled/` (`problems(; case)`): TreeGH evolves the metric from the
     TOV star's, and the matter's stress-energy, the fields' `½(E² + B²)`
     included, crosses back through `ADMMatter`. The case itself is built
     on `Analytic(TOVMetric)`, and on that Cowling metric it, its suite
     test and `bin/magnetized_tov_study.jl` stay as the cheaper check
     the package's own suite can run without TreeGH: the field
     is a test field (below), its stress-energy `10⁻⁵` of the star's, and
     the two spacetimes give the shell's drift to three digits through
     20 M and within 7% at 50 M (measured below).
   - **The exterior**: the static, current-free dipole of the
     Schwarzschild spacetime outside the star (Wasserman & Shapiro 1983),
     `A_φ = 3μ sin²θ S(x)/r`, `x = 2M/r`, `S = Σ_{k≥3} x^{k−3}/k`, which
     is `μ sin²θ/r` far away. `A_φ` is the same covariant component in
     isotropic coordinates, so `A_i = a(r̄²) (−y, x, 0)` with
     `a = A_φ/ϖ̄²`. It is a static solution with `E = 0`, `Φ̃ = 0`: on
     `TOVMetric`, `∇ × (αB)` outside the star is the difference's own
     error (4.0 times smaller per halving of the step, 3.6e−9 of `𝔅` at
     `δ = 5·10⁻⁴`), where the flat dipole leaves 1.5–2.4e−2. The Lorenz
     gauge holds exactly in the continuum. This is stricter than the
     literature's: GRaM-X's extended field is an ad hoc
     `A_φ ∝ r₀³ r sinθ/(r³ + r₀³)` that relaxes for about 8 ms, and
     Dionysopoulou et al. (2013) take LORENE's; both report the exterior
     flux oscillating, not drifting. Neither measures the exterior's
     convergence: Dionysopoulou et al. run the extended field at one
     resolution, 0.738 km (0.50 M), and report the flux through a
     hemisphere at `r = 132.94` km against the central density's
     spectrum; GRaM-X runs it at one resolution, 0.5 M (Z4c, the
     innermost cube ±30 M, the outer ±240 M), and reports the flux
     through a hemisphere at `r = 90 M`, its spectrum only. Their
     confined-field runs, Cowling and dynamical, are at 0.30, 0.20 and
     0.15 M (0.443, 0.295, 0.222 km in Dionysopoulou et al.), with the
     same σ threshold in both spacetimes. So this rung's criterion, that
     the exterior's drift converges, is stricter than the literature's;
     it is kept. "Vacuum `E` outside" is
     therefore `E = 0` for a static star; a rotating one would need the
     exterior's own `E`.
   - **The interior**: `a(u)` continued inside `u = R̄²` by the
     quadratic in `u` that matches the exterior's value and first two
     derivatives at the surface, so `B` and the current are continuous and
     the current falls to zero at the surface, where σ does. Poloidal and
     axisymmetric, so the current and the resistive `E` are toroidal, and so
     is `−v × B` of a radial flow: `E·B = 0` in the continuum at any σ,
     which makes `E·B` inside a measure of the scheme's error alone —
     under the default tableau mostly the time integrator's, not the
     spatial truncation (measured below). The
     central field sets `μ`: `B_c` is the contravariant `Bᶻ` at the centre
     (default `10⁻⁴`; the norm there is `ψ_c² B_c = 1.43 B_c`, and
     `B²/(2p) = 6·10⁻⁵`, Heaviside–Lorentz geometric units; the
     component is the convention that reproduces GRaM-X's `B_c` from its
     `A_b`): a test field, its Lorentz force far below the Cowling star's
     own truncation error.
   - **The conductivity**:
     `MagnetizationCutoff(DensityProfile(σ₀; ρ_cut, power), μ_max)`, `σ₀`
     and `μ_max` with no default; the suite and the studies take
     `σ₀ = 10⁶` (σΔt ≈ 10⁵ at `h = 1`) and `μ_max = 10`. **σ is zero
     below `ρ_cut = 10⁻⁶`** (the case's default), a density of
     the star's and not of `ρ_min` (measured below). The default belongs
     to this star: `10⁻⁶` is about `8·10⁻⁴ ρ_c` at `ρ_c = 1.28·10⁻³`,
     and only the outer 0.0093 M of `R = 9.586 M` lies below it, so it
     sits above the gas that piles up outside the surface and below
     nearly all of the star; a star of another `ρ_c` states its own.
     That is far above every published threshold
     (`D_atm` in Dionysopoulou et al. 2013 and GRaM-X, `100 D_atm` in
     Dionysopoulou, Alic & Rezzolla 2015, a force-free switch at 50–400
     `ρ_atm` in Palenzuela 2013; "Ohm's law and the conductivity"). The
     atmosphere falls onto the star and piles up outside the surface, and
     with σ cut at `10⁻¹⁰` (the literature's `D_atm`) or `10⁻⁸` (its 100)
     that gas conducts and carries the exterior field, whose drift then does
     not converge (measured below). `DensityProfile`'s shape is kept ("Ohm's
     law and the conductivity").
   - **The diagnostics** (`src/field_diagnostics.jl`): `cell_profile`, the
     owned cells' `E`, `B`, `q`, flags and metric from one recovery;
     `field_diagnostics`, the field energy inside and outside a sphere,
     `Σ|E·B|/ΣB²` and `|E|/|B|` inside, the charge's L1, and the entropy
     fallback's trigger and the policy's repairs, the atmosphere left out
     ("Primitive recovery"); `field_drift`, `Σ|B − B₀|/Σ|B₀|` inside a
     sphere and in a shell; `ohm_map`, the diagnostics set's stage tallies
     per region.

   **Measured, coupled.** `coupled/magnetized_tov_study.jl` on the
   default box (`[−24, 24]³`, four roots, one box level, PPM, TreeGH's
   CFL 1/2, the order-4 transfer, `γ0M = 1/10`), `ρ_min = 10⁻¹⁰`, at
   `h = 0.5, 0.375, 0.25` to 50 M, at three thresholds: the literature's
   squared profile at `ρ_cut = ρ_min = 10⁻¹⁰` (`power = 2`), the linear
   one at `10⁻⁸` and at `10⁻⁶`; and at `h = 0.5` with `ρ_min = 10⁻¹²`,
   the linear one at `10⁻⁸` and `10⁻⁶`. (Run as multiples of `ρ_min`,
   before the case took a density; `σ` is the same bit for bit.) Trillium,
   AMD EPYC 9655, 96 threads, jobs 2873324–2873334; at `10⁴ ρ_min` 718 s,
   1585 s and 6259 s. The coupled suite's star (`coupled/coupled_tests.jl`):
   `[−12, 12]³`, three roots, no boxes, `h = 0.8` to `t = 1`.
   - **∇·B**: `max |∇·𝔅| h/max |𝔅|` at most 3.4e−16 in every chunk of
     every run.
   - **The exterior dipole at `ρ = 10⁻⁶`**: its drift converges, in the
     shell `R̄ + 1 ≤ r̄ < 23` and inside `R̄`, with the rates between
     `h = 0.5, 0.375` and `0.375, 0.25`:

     | `t` (M) | shell | rates | inside | rates |
     |---|---|---|---|---|
     | 2 | 1.37e−3, 7.86e−4, 3.53e−4 | 1.93, 1.97 | 4.15e−3, 2.30e−3, 9.73e−4 | 2.05, 2.12 |
     | 10 | 5.91e−3, 3.93e−3, 1.74e−3 | 1.42, 2.01 | 8.78e−3, 5.31e−3, 2.62e−3 | 1.75, 1.74 |
     | 20 | 1.18e−2, 7.93e−3, 2.75e−3 | 1.40, 2.61 | 1.35e−2, 8.46e−3, 3.98e−3 | 1.61, 1.86 |
     | 30 | 1.39e−2, 7.81e−3, 2.95e−3 | 2.01, 2.40 | 1.71e−2, 1.04e−2, 4.95e−3 | 1.72, 1.84 |
     | 50 | 1.62e−2, 8.70e−3, 3.55e−3 | 2.17, 2.21 | 2.32e−2, 1.38e−2, 6.77e−3 | 1.80, 1.76 |

     At `h = 0.5` and `0.375` the shell's drift is the Cowling star's
     (below) to three digits through 20 M and to 1% at 30 M; at 50 M it
     is 7% and 4% below it, the inside 5% above. The field energy outside `R̄` changes by
     −1.8%, −0.96% and −0.08% over the 50 M.
   - **The threshold decides.** The shell's drift at 10 M and 50 M, the
     share of stages that fall back in the band `R̄ … R̄ + 1.5` and
     beyond it (over the 50 M), and the change of the field energy
     outside `R̄`, at `h = 0.5, 0.375, 0.25`:

     | σ zero below | shell, 10 M (rates) | shell, 50 M | band | beyond | energy outside |
     |---|---|---|---|---|---|
     | `10⁻¹⁰`, squared | 3.20e−2, 2.91e−2, 2.21e−2 (0.33, 0.68) | 0.271, 0.137, 0.060 | 59%, 62%, 48% | 3.9%, 1.2%, 0.015% | +11%, −2.3%, −3.0% |
     | `10⁻⁸` | 3.35e−2, 3.04e−2, 2.25e−2 (0.34, 0.74) | 0.238, 0.139, 0.056 | 41%, 27%, 15% | 0 | +6.3%, −2.2%, −3.2% |
     | `10⁻⁶` | 5.91e−3, 3.93e−3, 1.74e−3 (1.42, 2.01) | 1.62e−2, 8.70e−3, 3.55e−3 | 0.004%, 0.0006%, 0 | 0 | −1.8%, −0.96%, −0.08% |

     At `10⁻¹⁰` and `10⁻⁸` the drift does not converge over the first
     10 M, and at `h = 0.25` it stalls near 0.06 from 20 M to 50 M,
     17 and 16 times the drift at `10⁻⁶`: the gas outside the surface
     conducts and carries the field.
   - **The threshold is a density, not a multiple of `ρ_min`.** At
     `h = 0.5`, to 50 M:

     | `ρ_min` | multiple of `ρ_min` | σ zero below | shell, 10 M | shell, 50 M | band | energy outside |
     |---|---|---|---|---|---|---|
     | `10⁻¹⁰` | 100 | `10⁻⁸` | 3.351e−2 | 0.238 | 41.2% | +6.3% |
     | `10⁻¹²` | `10⁴` | `10⁻⁸` | 3.365e−2 | 0.248 | 40.9% | +8.2% |
     | `10⁻¹⁰` | `10⁴` | `10⁻⁶` | 5.910e−3 | 1.62e−2 | 0.004% | −1.77% |
     | `10⁻¹²` | `10⁶` | `10⁻⁶` | 5.912e−3 | 1.67e−2 | 0.004% | −1.79% |

   - **Inside the star**, `Σ|E·B|/ΣB²` converges at 1.7–2.3 at every
     sample (1.97e−5, 1.10e−5, 4.52e−6 at 2 M; 8.94e−5, 5.26e−5,
     2.09e−5 at 50 M), at `10⁻⁶`; at `10⁻¹⁰` and `10⁻⁸` at 1.2–2.1.
   - **The constraints** are the same for every threshold to three
     digits: the field does not reach the spacetime at this strength.
     `ℋ` inside (`r < 6`) is 1.8e−5, 9.3e−6, 3.7e−6 at 50 M (rates 2.3,
     2.3) and outside (`r ≥ 10`) 8.5e−7, 4.1e−7, 1.9e−7 (2.6, 1.9); the
     momentum constraint 2.3e−5, 1.1e−5, 4.2e−6 inside (2.4, 2.5) and
     3.2e−7, 1.6e−7, 8.1e−8 outside (2.4, 1.7). Over every sample the
     rates of `ℋ` are 1.7–2.9 inside and 1.7–3.8 outside.
   - **Rest mass**: its change minus what the reset injects is at most
     2.2e−12, 1.2e−12 and 6.2e−13 of 1.506 at `ρ_min = 10⁻¹⁰` (by `h`,
     the same at every threshold) and 2.1e−14 at `10⁻¹²`: it scales with
     `ρ_min`, matter entering with the atmosphere at the outer faces,
     which the budget does not count, as in rung 5. On the Cowling
     metric it is 7.7e−15.
   - **The outcome map**: inside `R̄` every stage converges by the fixed
     point, at every threshold, with no fallback, repair or reset. In the
     band at `10⁻⁶`, 74%, 83% and 91% of the stages are `σ = 0`.
   - **The suite's run** (`h = 0.8`, `t = 1`): `∇·B` 2.7e−16, the rest
     mass's change the reset's injection (−2.7e−9) to round-off, 176640
     of 176640 stages inside `R̄` fixed point, the drift 0.0065 inside and
     0.0029 outside. The field's energy reaches TreeGH: the vertices'
     `ρ` at `B_c = 10⁻⁴` minus that at 0 is 9.89e−9 at its maximum
     against `½B²_max = 9.87e−9`, by the per-call, eight-cell-mean
     transfer (the order-4 one's switch to the mean near the surface
     moves its maximum there).

   **Measured on the Cowling metric**, the cheaper check. The studies:
   `bin/magnetized_tov_study.jl` on the default box (`[−24, 24]³`, four
   roots, one box level, PPM) at `h = 0.75, 0.5, 0.375` to 50 M, on
   Trillium (AMD EPYC 9655, 64 threads; jobs 2864940–2864942, 268 s,
   936 s and 2286 s). The suite's star (`test/magnetized_tov_tests.jl`):
   `[−12, 12]³`, three roots, no boxes, MC, `h = 1` and `1/2` to `t = 2M`.
   - **∇·B**: `max |∇·𝔅| h/max |𝔅|` at most 3.4e−16 in every chunk of every
     study, 1.9e−16 in the suite's runs.
   - **The exterior dipole** stays the initial field to truncation error,
     and its drift converges. `Σ|B − B₀|/Σ|B₀|` in the shell
     `R̄ + 1 ≤ r̄ < 23`, and inside `R̄`, with the rates between `h = 0.75,
     0.5` and `0.5, 0.375`:

     | `t` (M) | shell | rates | inside | rates |
     |---|---|---|---|---|
     | 2 | 2.87e−3, 1.37e−3, 7.86e−4 | 1.83, 1.93 | 9.67e−3, 4.15e−3, 2.30e−3 | 2.08, 2.05 |
     | 10 | 1.17e−2, 5.91e−3, 3.92e−3 | 1.68, 1.42 | 2.01e−2, 8.78e−3, 5.30e−3 | 2.04, 1.76 |
     | 20 | 1.92e−2, 1.18e−2, 7.93e−3 | 1.20, 1.39 | 2.57e−2, 1.34e−2, 8.41e−3 | 1.61, 1.62 |
     | 30 | 2.88e−2, 1.40e−2, 7.89e−3 | 1.78, 1.99 | 3.26e−2, 1.71e−2, 1.03e−2 | 1.59, 1.74 |
     | 50 | 4.27e−2, 1.74e−2, 9.03e−3 | 2.22, 2.28 | 4.47e−2, 2.21e−2, 1.32e−2 | 1.74, 1.79 |

     The field energy outside `R̄` changes by −5.0% at `h = 0.75` and
     −1.1% at `0.375` over the 50 M. In the suite's run the drift outside `R̄`
     at `t = 2M` is 1.21e−2 and 3.54e−3 (rate 1.77), inside 2.01.
   - **It holds only with the exterior electrovacuum** (the cutoff,
     above). With `DensityProfile`'s own `cutoff = 2` the same studies
     (jobs 2864696–2864698) drift in the shell by 3.66e−2, 3.29e−2,
     2.99e−2 at 10 M (rates 0.27, 0.32) and 0.418 at 50 M at `h = 0.75`,
     and the field energy outside grows by 20%: the gas the atmosphere
     piles up outside the star conducts and carries the field. At
     `h = 0.75` to 20 M, the shell's drift and the share of stages that
     fall back in the band `R̄ … R̄ + 1.5` and beyond it:

     | profile | threshold (`ρ_min`) | drift | band | beyond |
     |---|---|---|---|---|
     | linear | 2 | 0.139 | 49.2% | 6.0% |
     | squared (the literature's) | 1 | 0.132 | 49.2% | 9.8% |
     | linear | 100 | 0.139 | 48.1% | 1.4% |
     | squared | 100 | 0.139 | 48.1% | 1.4% |
     | linear | `10³` | 0.084 | 36.8% | 0.11% |
     | linear | `10⁴` | 0.019 | 0.69% | 0 |

     The squared profile changes nothing; the threshold does. At
     `cutoff = 2` the drift is also 0.146 at `B_c = 10⁻⁶`, 0.138 at
     `σ₀ = 10³` and 0.142 at `ρ_min = 10⁻¹²`, and 0.032 at
     `μ_max = 0.001`, where the taper takes σ to zero outside the star. The same box without its coarse-fine faces (uniform `h = 0.75`)
     drifts as with them (0.1386 against 0.1389 at 20 M), so the faces are
     not the cause.
   - **Inside the star**, `Σ|E·B|/ΣB²` converges: 5.26e−5, 1.90e−5,
     1.07e−5 at 2 M (rates 2.51, 2.00) and 1.76e−4, 8.73e−5, 5.11e−5 at
     50 M (1.73, 1.86). `|E|/|B|` inside is 9.3e−3, 5.8e−3, 4.5e−3 at
     50 M (0.93–1.16). **Both are SSP3(4,3,3)'s offset, not the star's
     spatial truncation**: the tableau is not stiffly accurate, and each
     step leaves `𝔈` `−0.2844 Δt f` off the stage's ideal equilibrium,
     `f` mostly `∇ × (αB)` ([The partition](#the-partition)), an error of
     order `Δt ∝ h`. The suite's run (`h = 1`, `2 M`) gives `Σ|E·B|/ΣB²`
     9.27e−5 and `|E|/|B|` 1.10e−2 under SSP3(4,3,3) and 2.92e−9 and
     1.92e−3 under ARS(4,4,3), which is stiffly accurate, while the field's
     drift inside is 1.684e−2 under both (`test/magnetized_tov_tests.jl`).
     So the stored `E` inside a magnetized star is about 1% of `B` off ideal
     at `h = 1` under the default tableau, and so is everything read from it
     at a step's end: `E·B`, the field energy's electric part, the charge.
   - **Rest mass**: its change equals what the reset injects to at most
     7.7e−15 of 1.506 at every chunk (−9.4e−8 to −1.2e−7 injected by
     50 M).
   - **The stage solve's outcome map** (the studies, summed over 50 M):
     inside `R̄` every stage converges by the fixed point (153 million at
     `h = 0.375`), with no fallback, no repair and no reset. In the band
     `R̄ … R̄ + 1.5` 51%, 75% and 83% of the stages are `σ = 0` (by
     resolution), nearly all the rest fixed point,
     and the lagged-velocity fallback takes 244658 at `h = 0.75` (4.1%),
     1184 at `0.5` and 624 at `0.375` (6·10⁻⁶); beyond it every stage is
     `σ = 0`. At `cutoff = 2` the band fell back in 51%, 59% and 55% of
     its stages and the exterior in 6.2%, 1.7% and 0.2%. In the suite's
     run at `h = 1` the fallbacks outside are 2032 of 646912 stages
     (0.31%; 18920 at `cutoff = 2`), all in cells whose centre is just
     outside `R̄`.
   - **The entropy fallback's trigger** fires in no cell inside `R̄`, and
     outside in 4654–21370 cells (by resolution, at 50 M; the atmosphere
     left out): the gas outside the star, whose fields' energy exceeds its
     own. The energy inversion fails outside the atmosphere in at most 10
     cells at a time, in 12 of the 50 samples at `h = 0.375`, once at
     `0.5`, never at `0.75`, and never inside `R̄`. **Whether `D s` earns
     its keep is not decided**: the fallback is blocked on EOS-3,
     and without it this star gives it almost nothing to repair.

   **The confined-field star** (`field = :confined`): the resistive star
   of the literature, against which σ's own decay of the field is
   measured. The field is Duez et al.'s (2006),
   `A_φ = ϖ² max[A_b(P − P_cut), 0]²` with `P_cut = 0.04 P_c`, zero where
   `P ≤ P_cut` (`r̄ ≳ 6.6`) and so zero at the surface and outside;
   `confined_field` sets `A_b` from `B_c` (the same contravariant `Bᶻ`),
   and GRaM-X's `A_b = 1.58` is `B_c = 4.26e−8` here, its quoted
   `4.3e−8`. This is Dionysopoulou et al.'s (2013, Sec. IV.3.1) and
   GRaM-X's (Azizi et al. 2025, Eq. 60, Sec. IV.5.1) star. The
   conductivity is theirs too, `power = 2`, `ρ_cut = ρ_min` (in `ρ`); with
   no field outside `P_cut`, the conducting gas outside the surface
   carries none. The runs are a study, not a test: the
   cheapest that tells σ₀ = 100 from 10⁶ (`h = 0.75` to 200 M) takes
   740 s on 96 threads, and its own numerical loss is 79%; the suite
   checks the initial field ("Tests"). The literature runs this star in
   both spacetimes, and so does this rung: coupled, and on the Cowling
   metric as the cheaper check, whose central field the coupled one
   follows to 0.4%.

   **Measured, coupled** (`coupled/magnetized_tov_study.jl field=confined
   power=2 rhocut=1e-10 Bc=4.26e-8`, the default box `[−24, 24]³`, four
   roots, one box level, so that the star's level has the Cowling runs'
   `h`; PPM, `μ_max = 10`, `ρ_min = 10⁻¹⁰`; to 200 M at `h = 0.5, 0.375,
   0.25` with σ₀ = 10⁶, 10³, 100, and to 2000 M at `h = 0.5` with σ₀ = 100
   and 10⁶; Trillium, AMD EPYC 9655, 96 threads, jobs 2878575–2878589,
   3430–3667 s at `h = 0.5`, 7164–7499 s at 0.375, 27.7–28.2 ks at 0.25,
   36.4 and 37.5 ks to 2000 M; and job 2880717, `ρ_min = ρ_cut = 10⁻¹²` at
   `h = 0.5`, σ₀ = 10⁶, to 200 M). `h = 0.75` is not run: a coupled block
   needs `N ≥ 10` ("Metric at cells, faces and edges"). The Cowling
   values, from the runs further below, are beside them.
   - **∇·B** at most 3.7e−16 in every chunk of every run.
   - **The central field is the Cowling star's.** `B_c(t)/B_c(0)` differs
     from it by at most 0.4% at every σ₀, `h` and sample (10, 50, 100,
     200 M), and the difference falls with `h`: −3.6e−3, −1.8e−3,
     −7.8e−4 at 200 M, σ₀ = 10⁶ (rates 2.4, 2.1). The field is far too
     weak to reach the spacetime, and the spacetime's own evolution
     barely reaches the field: `ρ_c` falls by 11.5% over 2000 M coupled
     and 4.4% on the Cowling metric, while `B_c` at σ₀ = 100 agrees to
     three digits throughout.
   - **The numerical loss** (σ₀ = 10⁶), `1 − B_c(t)/B_c(0)` at
     `h = 0.5, 0.375, 0.25` and its rates:

     | `t` (M) | coupled | rates | Cowling | rates |
     |---|---|---|---|---|
     | 10 | 0.0799, 0.0306, 0.0060 | 3.33, 4.04 | 0.0798, 0.0305, 0.0059 | 3.34, 4.04 |
     | 50 | 0.201, 0.0786, 0.0142 | 3.27, 4.22 | 0.198, 0.0767, 0.0133 | 3.30, 4.32 |
     | 100 | 0.236, 0.0983, 0.0206 | 3.05, 3.85 | 0.234, 0.0975, 0.0201 | 3.05, 3.89 |
     | 200 | 0.292, 0.125, 0.0272 | 2.94, 3.76 | 0.290, 0.124, 0.0265 | 2.95, 3.80 |

   - **σ's decay scales as 1/σ₀.** The decay rate of `B_c` over
     50–200 M (per M), and its excess over σ₀ = 10⁶'s:

     | `h` | σ₀ = 10⁶ | 10³ (excess) | 100 (excess) | Cowling: 10⁶, 10³ (excess), 100 (excess) |
     |---|---|---|---|---|
     | 0.5 | 7.67e−4 | 9.89e−4 (2.2e−4) | 4.77e−3 (4.00e−3) | 7.68e−4, 9.87e−4 (2.2e−4), 4.77e−3 (4.00e−3) |
     | 0.375 | 3.48e−4 | 6.98e−4 (3.5e−4) | 4.58e−3 (4.23e−3) | 3.47e−4, 6.98e−4 (3.5e−4), 4.59e−3 (4.24e−3) |
     | 0.25 | 8.8e−5 | 5.67e−4 (4.8e−4) | 4.34e−3 (4.26e−3) | 8.9e−5, —, 4.35e−3 (4.26e−3) |

     At σ₀ = 100 the excess converges (4.00, 4.23, 4.26e−3); σ₀ = 10³'s
     grows toward a tenth of it as the numerical loss falls below it,
     and at `h = 0.25` is 8.9 times smaller, against 10 for 1/σ₀.
   - **Two decades by 10 ms.** At σ₀ = 100, `h = 0.5`, `B_c/B_c(0)` is
     0.516 at 100 M, 0.161 at 400 M, 0.0395 at 1000 M, 0.0180 at
     1500 M, 0.0099 at 1960 M (9.65 ms; it first reaches 0.01 at 1956 M)
     and 0.0095 at 2000 M; the Cowling run's, 0.517, 0.161, 0.0396,
     0.0180, 0.0099, 0.0095. The decay rate over 50–200, 500–1000,
     1000–1500 and 1500–2000 M is 4.77e−3, 2.19e−3, 1.57e−3, 1.28e−3
     (Cowling 4.77e−3, 2.19e−3, 1.57e−3, 1.29e−3).
   - **No late rise at σ₀ = 10⁶.** At `h = 0.5`, `B_c` falls from each of
     its 1000 samples (every 2 M) to the next, to 2000 M (9.85 ms): 0.764 at 100 M, 0.464 at
     1000 M, 0.418 at 1500 M, 0.386 at 2000 M, its rate falling (7.7e−4,
     3.8e−4, 2.1e−4, 1.6e−4 per M over the four spans above). GRaM-X's
     dynamical runs grow artificially from about 7, 8 and 9 ms
     (1420–1830 M) at 0.3, 0.2 and 0.15 M, and Dionysopoulou et al.'s
     likewise; here no rise appears, at a coarser `h` whose own loss
     (61% by 2000 M) would hide only a rise slower than 1.6e−4 per M.
   - **The constraints** are the same at every σ₀ to the six digits
     printed, and the dipole star's to two: the field does not reach the
     spacetime. At 200 M, `ℋ` inside (`r < 6`) is 1.8e−5, 9.4e−6, 3.8e−6
     (rates 2.3, 2.3) and outside (`r ≥ 10`) 2.8e−7, 1.6e−7, 7.5e−8 (1.9,
     1.9); the momentum constraint 2.2e−5, 1.1e−5, 4.0e−6 inside (2.4,
     2.5) and 1.5e−7, 8.8e−8, 3.8e−8 outside (2.0, 2.0). Over every
     sample the rates are 1.6–3.8. At `h = 0.5` over 2000 M they do not
     grow: `ℋ` inside 1.7e−5 at 1000 M and 1.5e−5 at 2000 M, outside
     2.9e−7 and 4.0e−7.
   - **Rest mass**: its change minus what the reset injects is at most
     6.0e−11, 2.5e−11 and 7.3e−12 of 1.506 over 200 M (rates 3.0, 3.1),
     the same at every σ₀, and 6.8e−10 over 2000 M: round-off to 20 M,
     then growing about linearly. At `ρ_min = 10⁻¹²` it is 6.0e−13, 98
     times smaller, with `B_c` the same to four digits: atmosphere
     entering at the outer faces, as for the dipole star and in rung 5.
     On the Cowling metric's smaller box it is 3.5e−14 at `h ≤ 0.5`.
   - **Modes** (`bin/tov_modes.jl ref=coupled` on the 2000 M run's
     `ρ_c`, the whole series, 0.10 kHz resolution), against Font et
     al.'s (2002) Table II, the linear-perturbation frequencies GRaM-X's
     Fig. 6 marks: F 1.434 kHz (−0.58%), H1 3.947 kHz (−0.20%), the same
     at σ₀ = 10⁶ to four digits. **F and H1 hold the 3%.** The peak in
     H2's window, 5.23 kHz (−11.5%), is not the star's (rung 5's
     unmagnetized run has 5.26 kHz there at this `h`). Over `[0, 1000]` M, F is −0.59% and H1 −1.53%, over
     `[0, 500]` M H1 is −0.94%; the second 1000 M carries little (F's
     peak 63 times smaller). GRaM-X resolves only F at 0.3 M, F and H1
     at 0.2 and 0.15 M. The unmagnetized star with the same transfer
     gives the same (rung 5, "Modes"; the mean transfer's start is what
     hid H1 there).
   - **The outcome map**: inside `R̄` every stage at every σ₀ and `h` is
     fixed point, with no fallback, repair or reset. The band
     `R̄ … R̄ + 1.5` falls back in 32%, 36% and 28% of its stages over
     the 200 M (by `h`), beyond it 2.1%, 0.55% and 0, where the field is
     zero.

   **Measured on the Cowling metric** (`bin/magnetized_tov_study.jl field=confined power=2
   rhocut=1e-10 Bc=4.26e-8`, `[−12, 12]³`, four roots, no boxes, PPM,
   `μ_max = 10`, to 200 M at `h = 0.75, 0.5, 0.375, 0.25`; σ₀ = 10⁶,
   10³ (not at 0.25), 100; Trillium, AMD EPYC 9655, 96 threads, jobs
   2870561–2870573, 737 s at `h = 0.75` to 23.5 ks at `0.25`; and
   σ₀ = 100 at `h = 0.5` to 2000 M, job 2870574, 24.0 ks). `B_c(t)` is
   the volume-weighted mean `|B|` within `r̄ < 1`; `1 M = 4.93 μs`.
   - **∇·B** at most 2.7e−16 in every chunk; the rest mass changes by
     what the reset injects to 3.5e−14 at `h ≤ 0.5`. At `h = 0.75` the
     two differ by up to 2.8e−8, the same at every σ₀ and not monotonic
     in time: matter crossing the outer faces of this small box (3.9 M
     beyond the surface), which the budget does not count (not
     measured further).
   - **The numerical loss** (σ₀ = 10⁶, where σ's own decay time is about
     10⁷ M), `1 − B_c(t)/B_c(0)` and its rates between successive `h`:

     | `t` (M) | `h = 0.75, 0.5, 0.375, 0.25` | rates |
     |---|---|---|
     | 10 | 0.225, 0.0798, 0.0305, 0.0059 | 2.55, 3.34, 4.04 |
     | 50 | 0.516, 0.198, 0.0767, 0.0133 | 2.36, 3.30, 4.32 |
     | 100 | 0.676, 0.234, 0.0975, 0.0201 | 2.61, 3.05, 3.89 |
     | 200 | 0.790, 0.290, 0.124, 0.0265 | 2.48, 2.95, 3.80 |

     It converges at 2.4–4.3, faster than Dionysopoulou et al.'s about
     1.7 (Sec. IV.3.1, at Δx = 0.30, 0.20, 0.15 M), though at `h = 0.75`
     the star loses most of its field in 200 M. Their losses saturate at
     about 20% over 12 ms at their finest resolution; ours at
     `h = 0.25` is 2.7% at 200 M (1 ms), its rate falling.
   - **σ's decay scales as 1/σ₀.** The decay rate of `B_c` fitted over
     50–200 M (per M), and its excess over σ₀ = 10⁶'s at the same `h`:

     | `h` | σ₀ = 10⁶ | 10³ (excess) | 100 (excess) |
     |---|---|---|---|
     | 0.75 | 5.46e−3 | 5.53e−3 (6.9e−5) | 6.13e−3 (6.6e−4) |
     | 0.5 | 7.68e−4 | 9.87e−4 (2.2e−4) | 4.77e−3 (4.00e−3) |
     | 0.375 | 3.47e−4 | 6.98e−4 (3.5e−4) | 4.59e−3 (4.24e−3) |
     | 0.25 | 8.9e−5 | — | 4.35e−3 (4.26e−3) |

     At σ₀ = 100 the excess converges (4.00, 4.24, 4.26e−3); σ₀ = 10³'s
     is 12 times smaller at `h = 0.375`, against 10 for 1/σ₀, where it
     is the size of the numerical loss itself. At `h = 0.75` the scheme
     swamps both.
   - **Two decades by 10 ms.** At σ₀ = 100, `h = 0.5`, `B_c` falls to
     0.517 by 100 M, 0.161 by 400 M, 0.040 by 1000 M and 0.0100 at
     1960 M (9.65 ms; 0.0095 at 2000 M). GRaM-X's Fig. 7 has σ₀ = 100
     fall by almost two orders within 10 ms, Dionysopoulou et al.'s
     Fig. 8 within about 12 ms. The decay is not one exponential: its
     rate is 4.8e−3 over 50–200 M, 2.2e−3 over 500–1000 M, 1.3e−3 over
     1500–2000 M — the shorter scales go first, toward the slowest
     mode, for which a uniformly conducting sphere of radius `R̄` in
     vacuum gives `π²/(σ₀R̄²) = 1.5e−3` (the same size; the metric and
     the profile change it). At `h = 0.5` the numerical loss is part of
     this (7.7e−4 over 50–200 M at σ₀ = 10⁶); a 2000 M run at
     `h = 0.25`, about 65 h on 96 threads, would separate it (not run).
   - **The outcome map** at the literature's threshold: inside `R̄`
     every stage is fixed point; the band `R̄ … R̄ + 1.5` falls back in
     37% of its stages at `h = 0.375` (the last chunk), beyond it 1.4% —
     as the dipole star did at `ρ_cut = ρ_min`, and harmless here, where
     the field there is zero.
7. **TOV with a tabulated EOS and `Yₑ`** *(Done.)*: a star on a real table in cold
   β-equilibrium (EOS-4), static in the Cowling approximation and in
   TreeGH's spacetime as rungs 4 and 5 are, with `Yₑ` varying through it and
   `D Yₑ` conserved.

   *As built* ("Thermodynamics": "The cold barotrope", "The
   tabulated star's surface and atmosphere"). The stars are their tables'
   1.4 M☉ ones: DD2 at `ρ_c = 9.4·10⁻⁴` (5.8·10¹⁴ g/cm³; `M = 1.404`,
   `R̄ = 7.47`) and SFHo at `ρ_c = 1.37·10⁻³`; the atmosphere at `10⁻¹⁰`
   on the Cowling metric and `10⁻¹²` in the evolved spacetime, as for the
   polytrope.
   - **`Yₑ` advection** (`test/ye_advection_tests.jl`, the ideal gas, one
     period on 32, 64, 128 cells): a smooth profile at 1.69, 1.86 under MC
     (clipped at its extrema) and 2.05, 2.23 under PPM; a step stays
     within its two values to 4·10⁻¹³; `D Yₑ` to 6·10⁻¹⁷ of 0.35.
   - **The suite's star** (`test/tabulated_tov_tests.jl`, DD2 on
     `[−16, 16]³`, `h = 1`, MC, to 4 M): every cell of the initial data in
     the star recovers with no policy flag; `ρ_c` drifts −0.82% (the
     polytrope −3.3% on the same mesh); the rest mass and `D Yₑ` change
     by exactly what the reset injects, to 6.6·10⁻¹⁶ and 2.0·10⁻¹⁷; the
     entropy floor fires on no cell outside the atmosphere.
   - **In TreeGH's spacetime** (`coupled/tabulated_tests.jl`, `N = 10, 20`
     on the same box): the frozen metric reproduces the Cowling run to
     1.2·10⁻¹³; `ℋ − 16πρ` at `t = 0` inside `r < 6` converges at 1.83
     with the mean and 3.78 with the fourth-order transfer (3.44·10⁻²
     without the matter); to `t = 1` the rest mass and `D Yₑ` change by
     the reset's injection to 1.2·10⁻¹⁵ and 2.9·10⁻¹⁵.
   - **The Cowling studies** (`jobs/tabulated-tov-cowling.toml`,
     `bin/tov_study.jl table=…`; Symmetry, `amdq`, jobs 607056–607061):
     DD2 and SFHo on rung 4's mesh (`[−24, 24]³`, four roots, one box
     level) at `h = 0.75, 0.5, 0.375`, MC, `ρ_min = 10⁻¹⁰`, to 200 M. The
     drift `ρ_c/ρ_c(0) − 1` against the same mesh's initial cell average
     (`ρ_c(0)` 9.3029, 9.3566, 9.3756·10⁻⁴ for DD2; 1.34651, 1.35947,
     1.36406·10⁻³ for SFHo):

     | | `h = 0.75` | `0.5` | `0.375` | orders |
     |---|---|---|---|---|
     | DD2 drift at 200 M | −1.547e−2 | −8.771e−3 | −1.352e−3 | 1.40, 6.51 |
     | DD2 mean over `[160, 200]` M | −1.351e−2 | −7.257e−3 | −4.202e−3 | **1.53, 1.90** |
     | DD2 L1 change of `D`, `τ` | 2.73e−6, 1.38e−7 | 1.67e−6, 6.23e−8 | 1.04e−6, 3.34e−8 | |
     | SFHo drift at 200 M | −4.332e−2 | −7.195e−3 | −5.292e−3 | 4.43, 1.07 |
     | SFHo mean over `[160, 200]` M | −2.871e−2 | −1.062e−2 | −5.031e−3 | **2.45, 2.60** |
     | SFHo L1 change of `D`, `τ` | 3.62e−6, 1.65e−7 | 1.66e−6, 8.63e−8 | 1.11e−6, 4.85e−8 | |

     **Stationarity** (the acceptance: as in rung 4, the drift after 200 M
     converging at ≥ 1.5): **it holds for the window means, and the
     endpoints miss on one pair each** (DD2 1.40, SFHo 1.07). The
     endpoint is one sample of the radial oscillation, whose half-range
     over `[160, 200]` M (2.3e−3 for DD2 and 2.4e−3 for SFHo at
     `h = 0.375`) is larger than the drift itself, so rung 4's window mean
     is the measure here. The drift is a tenth of the polytrope's on the
     same mesh (MC at `h = 0.375`: −0.14% against −1.45%). The start rings
     harder than the polytrope's: the half-range over the first 40 M is
     2.6–3.4% at every `h`, falling to 0.2–0.4% by 160 M.
   - **Conservation** (the acceptance: `D Yₑ` to round-off): at `h ≤ 0.5` the
     rest mass and `D Yₑ` change by exactly what the reset injected, to at
     most 4.6·10⁻¹⁵ of 1.53–1.55 and 2.1·10⁻¹⁶ of 0.089–0.098. **It holds
     over 200 M** (over 1000 M, below, `D Yₑ` does not). At `h = 0.75` both
     lose matter through the outer faces, the rest mass 1.7·10⁻⁵ (DD2) and
     4.7·10⁻⁴ (SFHo) over 200 M:
   - **The coarse star sheds matter, and resolution removes it.**
     `bin/tov_study.jl`'s `diag` (`jobs/tabulated-tov-atmosphere.toml`,
     jobs 607220, 607248–607251) records each M the rest mass at
     `r > 5R̄/4` above twice the atmosphere's density, its largest radius,
     and the largest density against the outer faces:

     | | `h = 0.75` | `0.5` | `0.375` |
     |---|---|---|---|
     | DD2: largest mass outside, at | 2.98e−3, 147 M | 9.19e−5, 187 M | 1.6e−10, 137 M |
     | DD2: farthest radius | 40.3 | 23.4 | 9.4 |
     | SFHo: largest mass outside, at | 6.17e−2, 200 M | 1.54e−4, 119 M | 4.4e−10, 125 M |
     | SFHo: farthest radius | 40.3 | 9.1 | 8.3 |

     At `h = 0.75` the shed matter reaches the box's corners (`r = 41.6`)
     after about 150 M (DD2) and 140 M (SFHo); only then does the density
     against the faces rise above `ρ_min` (to 4.1·10⁻¹⁰ and 6.8·10⁻⁸) and
     the budget open. Before it, and throughout at `h ≤ 0.5`, the budget
     holds to 1.2·10⁻¹⁴. At `h = 0.375` the star sheds nothing beyond a
     single sample of 10⁻¹⁰. Where the shed matter comes from was not
     separated; the star's surface layers on the coarsest mesh are the
     candidate.
   - **The entropy floor** (the "cold edge, measure" decision of "The
     tabulated star's surface and atmosphere"): it fires on 34176, 171279
     and 383752 cells outside the atmosphere for DD2 and on 1032, 39153
     and 86713 for SFHo (by `h`), about 0.3–0.5% of the star's cell-steps
     for DD2; 81–100% of them in the first 50 M, with 91–100% of the `τ`
     it injects: 4.5, 2.9, 1.4·10⁻⁵ for DD2 (falling at about 1.2 and 2.5
     with `h`) and 3.0·10⁻⁶, 1.0·10⁻⁵, 4.6·10⁻⁶ for SFHo. It is the start's
     ringing at the cold edge, not a steady heating, so the star stays
     at the edge and no fix is proposed.
   - **The cost**: 5.1–6.9 µs per cell-step at `h ≤ 0.5`, 9.0–9.5 at
     `h = 0.75` (64 threads, EPYC 7532 and 7543).
   - **The coupled study** (`jobs/tabulated-tov-coupled*.toml`,
     `coupled/tov_coupled.jl table=dd2`; Symmetry, `amdq`, jobs 607062,
     607068 and 607064, the last two checkpointed): DD2 on rung 5's mesh
     and gauge, PPM, the fourth-order transfer, `ρ_min = 10⁻¹²`, at
     `h = 0.6, 0.5, 0.375`, to 1000 M (4.1, 7.8 and 21.2 h on 64 threads).
     The constraints, `‖·‖₂` inside `r < 6` and outside `r ≥ 10`, averaged
     over each window; the rates between `h = 0.6, 0.5` and between
     `h = 0.5, 0.375`:

     | `t` (M) | `ℋ` inside | `ℳ` inside | `ℋ` outside | `ℳ` outside |
     |---|---|---|---|---|
     | 0 | 4.16, 3.79 | — | 4.33, 4.31 | — |
     | 20–30 | 1.69, 2.10 | 1.97, 2.09 | 1.27, 1.88 | 1.30, 1.83 |
     | 95–105 | 1.36, 2.08 | 2.24, 2.84 | 1.91, 1.84 | 2.28, 2.20 |
     | 240–260 | 1.28, 2.35 | 2.32, 3.02 | 1.24, 2.10 | 1.15, 1.91 |
     | 490–510 | 1.51, 2.58 | 2.63, 2.70 | 1.27, 2.20 | 0.85, **1.54** |
     | 740–760 | 1.53, 3.26 | 2.02, 2.36 | 1.62, 2.41 | 0.76, **1.41** |
     | 900–1000 | 0.94, **1.64** | 1.95, 2.10 | 1.08, **1.66** | 1.20, **1.65** |

     **The acceptance's ≥ 1.8 (rung 5's) holds on the finer pair to 260 M,
     inside and outside, and fails after it**: `ℳ` outside from 490 M,
     `ℋ` inside and outside over 900–1000 M. The coarse pair is below 1.8
     in most windows. Unlike rung 5's, the norms grow: at `h = 0.375`, `ℋ`
     inside is 1.2–1.3·10⁻⁵ over 240–510 M and 6.6·10⁻⁵ over 900–1000 M,
     `ℳ` outside 1.1·10⁻⁶ over 240–260 M and 8.2·10⁻⁶ over 900–1000 M, and
     likewise at the other two `h`.
   - **The star's oscillation grows in the evolved spacetime on this box,
     fed by the outer boundary.** The half-range of `ρ_c/ρ_c(0)` per 100 M:

     | `t` (M) | `h = 0.6` | `0.5` | `0.375` | Cowling, `h = 0.5` |
     |---|---|---|---|---|
     | 0–100 | 6.3e−3 | 4.8e−3 | 2.7e−3 | 3.0e−3 |
     | 200–300 | 3.5e−3 | 3.2e−3 | 2.0e−3 | 4.6e−4 |
     | 500–600 | 7.0e−3 | 5.9e−3 | 4.0e−3 | 1.5e−4 |
     | 900–1000 | 2.6e−2 | 2.3e−2 | 1.4e−2 | 1.2e−4 |

     From about 250 M it grows exponentially, e-folding in 316, 333 and
     328 M by `h` — a rate independent of the resolution, so the mesh does
     not remove it; its amplitude converges at about 1 (0.7 and 1.6 over
     900–1000 M). Its line is at 2.84–2.89 kHz at every `h` over both
     halves of the run. The last column is the same star with the same
     reconstruction and atmosphere on the Cowling metric
     (`jobs/tabulated-tov-cowling-long.toml`, job 607606): damped, 25 times
     smaller by 1000 M, so the growth needs the evolved spacetime.

     TreeGH's constraint damping does not drive it
     (`jobs/tabulated-tov-coupled-damping.toml`, jobs 607869–607873:
     `h = 0.6` to 600 M; the first row is job 607062's first 600 M). The
     half-range per 100 M, and the e-folding of the half-range over
     300–600 M:

     | `γ0 M`, `γ2` | 0–100 | 200–300 | 500–600 | e-folding |
     |---|---|---|---|---|
     | 1/10, 0 | 6.3e−3 | 3.4e−3 | 7.0e−3 | 340 M |
     | 0, 0 | 9.2e−3 | 1.1e−2 | 1.9e−2 | 560 M |
     | 1/20, 0 | 7.0e−3 | 4.4e−3 | 5.1e−3 | 1700 M |
     | 1/5, 0 | 6.1e−3 | 1.5e−3 | 4.5e−3 | 345 M |
     | 1/10, −1/2 | 7.0e−3 | 7.7e−3 | 1.8e−2 | 313 M |

     It grows at every `γ0`, without damping too, at 2.8–3.0 kHz in every
     run; stronger damping deepens the decay before 300 M and not the
     growth after it, and `γ2 = −1/2`, which suppresses GH-8's mode
     ([GH-8](#gh-8--constraint-damping-on-a-star-note)),
     starts it at 100 M. `ℋ` inside stays at 2.3–5.4·10⁻⁵ in the runs
     with `γ0 > 0` and `γ2 = 0` while it grows, and the reset and the
     entropy floor inject nothing that grows with it.

     The box does (`jobs/tabulated-tov-coupled-box.toml`, jobs 607905 and
     607906, 4.3 and 5.0 h): the same star's mesh, `h = 0.6` on
     `[−12, 12]³`, inside one and two more levels out to `L = 48` and 96,
     to 600 M. The line is the strongest of `ρ_c` over 350–600 M:

     | `L` | 0–100 | 200–300 | 500–600 | e-folding | line |
     |---|---|---|---|---|---|
     | 24 | 6.3e−3 | 3.4e−3 | 7.0e−3 | 340 M | 2.82 kHz |
     | 48 | 6.4e−3 | 2.2e−3 | 2.1e−3 | 1800 M | 2.88 kHz |
     | 96 | 6.4e−3 | 2.4e−3 | 8.9e−4 | decays, 257 M | 3.04 kHz |

     The outer boundary is TreeGH's Dirichlet data from the star's static
     exterior (TreeGH's `CODE.md`, "Outer boundary: Dirichlet from the
     background, at the current time"), which reflects what the oscillating
     star sends out. At `L = 24` the reflection grows the star's coupled F
     mode and pulls its line down; at `L = 96` the mode is damped. `ℋ`
     outside (`r ≥ 10`, to each box's edge) triples from 40 to 600 M at
     `L = 24`, grows by 1.4 at `L = 48` and falls at `L = 96`. Rung 5's
     polytrope, on the `L = 24` box with the same gauge and transfer, does
     not grow; why the tabulated star does is not measured. The F mode's
     frequency on an unbounded domain is not measured either: the line
     still moves between `L = 48` and 96.
   - **The coupled study on the wide box**
     (`jobs/tabulated-tov-coupled-wide.toml`, `-wide-mid.toml`,
     `-wide-fine.toml`; Symmetry, `amdq`, jobs 608225, 608226 and 608227
     with its continuation 609114, the last two checkpointed): the study
     above with the box of the last row, `L = 96`, four roots and three
     box levels (`h = 4.8, 2.4, 1.2` outside `[−12, 12]³`), at
     `h = 0.6, 0.5, 0.375` on the star, to 1000 M (8.3, 16.2 and 46.1 h
     on 64 threads, EPYC 7543; about twice `L = 24`'s). The rates, as
     above:

     | `t` (M) | `ℋ` inside | `ℳ` inside | `ℋ` outside | `ℳ` outside |
     |---|---|---|---|---|
     | 0 | 4.16, 3.79 | — | 4.33, 4.31 | — |
     | 20–30 | 1.69, 2.10 | 1.97, 2.09 | 1.35, 2.00 | 1.34, 1.87 |
     | 95–105 | 1.43, 2.25 | 2.31, 3.00 | 0.80, **1.38** | 0.68, **1.00** |
     | 240–260 | 1.65, 2.57 | 2.37, 3.03 | 1.39, 2.18 | 1.49, 2.24 |
     | 490–510 | 1.78, 2.49 | 2.45, 2.92 | 1.25, 1.87 | 1.18, 1.96 |
     | 740–760 | 2.04, 2.46 | 2.58, 2.84 | 1.51, **1.78** | 1.54, **1.78** |
     | 900–1000 | 2.07, 2.49 | 2.51, 2.90 | 1.49, 1.90 | 1.45, 1.88 |

     **On the finer pair the acceptance's ≥ 1.8 holds inside throughout and
     outside in every window but two**: around 100 M (1.38, 1.00) and
     1.78 at 750 M. The coarse pair is below 1.8 outside, as at `L = 24`.
     The outside region (`r ≥ 10` to the box edge) now spans the three
     coarse levels; that the dip at 100 M is the start's transient
     crossing them is a guess, not measured. The norms do not grow: at
     `h = 0.375`, `ℋ` inside is 8.2·10⁻⁶ over 240–260 M and 6.8·10⁻⁶ over
     900–1000 M (at `L = 24`: 1.2·10⁻⁵ and 6.6·10⁻⁵).

     The half-range of `ρ_c/ρ_c(0)` per 100 M:

     | `t` (M) | `h = 0.6` | `0.5` | `0.375` |
     |---|---|---|---|
     | 0–100 | 6.4e−3 | 4.8e−3 | 2.7e−3 |
     | 200–300 | 2.4e−3 | 2.2e−3 | 1.4e−3 |
     | 500–600 | 8.9e−4 | 8.0e−4 | 5.3e−4 |
     | 900–1000 | 6.8e−4 | 4.6e−4 | 3.0e−4 |

     A damped sinusoid fitted to `ρ_c` over 100–500 M e-folds in 220, 270
     and 330 M by `h` — damping that weakens with resolution, so the
     scheme's — and after 500 M the oscillation levels off at an amplitude
     of 3.8–5.6·10⁻⁴, still several times the Cowling star's. **The F
     line is at 3.08–3.10 kHz over 100–500 M and 3.13–3.15 kHz over
     500–1000 M at every `h`**, so the resolution does not move it; the
     Cowling star's F is at 4.2–4.4 kHz.
     The drift, as below: −0.37, −0.21, −0.14% over 50–70 M (as at
     `L = 24`), −0.68, −0.37, −0.18% over 490–510 M (rates 3.34, 2.51) and
     −1.10, −0.60, −0.33% over 900–1000 M (3.32, 2.08). The rest mass net
     of the reset gains 1.3, 0.80 and 0.40·10⁻¹⁰ of 1.53 by 1000 M, rising
     steadily from about 100 M, and `D Yₑ` 5.8, 3.7 and 1.9·10⁻¹¹ of 0.098:
     6–10 times `L = 24`'s and of the other sign, converging at 2.3–2.7,
     not separated (the candidate is the outer faces, 16 times the area,
     on cells 4 times coarser). The entropy floor fires on no cell.
   - **The central density's drift**, relative to `t = 0` and averaged over
     a window: −0.37, −0.20, −0.14% over 50–70 M (rates 3.19, 1.44),
     −1.17, −0.83, −0.42% over 490–510 M (1.88, 2.36), −1.09, −0.89,
     −0.61% over 900–1000 M (1.09, 1.32), where the growing oscillation
     dominates it. At 490–510 M it is a quarter of the polytrope's (rung 5:
     −1.88% at `h = 0.375`).
   - **Rest mass and `D Yₑ`** over the 1000 M, net of what the reset
     injected: at `h = 0.5` and 0.375 within ±2·10⁻¹¹ of 1.53 throughout,
     −7.7·10⁻¹² and −6.5·10⁻¹² at the end (rung 5: 1.3–5.4·10⁻¹²), and
     `D Yₑ` −1.9·10⁻¹² and −3.0·10⁻¹² of 0.098 — the atmosphere at the
     outer faces, as in rung 5, not round-off. At `h = 0.6` the rest mass
     loses 1.9·10⁻⁹ over 200–250 M and 1.8·10⁻⁸ over 970–1000 M, steps
     like the coarse Cowling stars' (not measured with `diag`). The entropy
     floor fires on 0, 6 and 0 cells.
   - **`D Yₑ` over 1000 M on the Cowling metric** ("Known risks"): the run of
     the last column holds the rest mass net of the reset to 4.7·10⁻¹⁶
     (and `diag` to 6.7·10⁻¹⁵ throughout), but `D Yₑ` to only 1.6·10⁻¹³
     of 0.098, where over 200 M it held to 2·10⁻¹⁶. Something moves
     lepton number and not rest mass. Not separated; the candidate is the
     outer faces, where the Dirichlet atmosphere carries the slice's
     `Yₑ_atm` and the atmosphere inside the box the `Yₑ` of the states the
     reset replaced, so that the light-cone flux's dissipation carries
     `D Yₑ` across a face where `D` is the same on both sides.

Later, not specified yet: rotating stars, collapse to a black hole,
binaries, supernovae with neutrinos.

## Possible extensions

Not built; each is an obvious next question, with what the measurements so
far say about it.

- **The entropy fallback**, and with it whether `D s` earns its keep: the
  fallback of a failed energy inversion, at the advected `s`, waits for
  EOS-3 ("Primitive recovery"). The magnetized star gives it almost
  nothing to repair (the inversion fails outside the atmosphere in at most
  10 cells at a time; "Validation ladder" 6). The reset writing
  `√γ D s = (√γ D) s` from the recovered `s` in every cell it did not
  repair was tried and not adopted: it breaks the conservation of `D s`
  and the reset's no-op where it repairs nothing ([HISTORY.md, "Open
  questions"](HISTORY.md#open-questions)). Without it `(√γ D s)/(√γ D)`
  leaves the window where `D` is thin (−410 … 1220 outside the Cowling
  star's surface at 200 M), which only seeds the recovery. The design
  ("Primitive recovery", "Entropy fallback") switches on a threshold of
  `½(E² + B²) / τ_fluid` whose value is not chosen; the cells it would act
  on are already counted (`field_diagnostics`).
- **Two stage-solve ideas** from the investigation of the magnetized
  atmosphere tube's front ([HISTORY.md, "Floors, the atmosphere and the
  magnetosphere"](HISTORY.md#floors-the-atmosphere-and-the-magnetosphere),
  "Investigation (2026-10-16)" and the footnote to its controls): a
  predictor in the atmosphere taking `OHM_SIGMA_ZERO`; and, in the stage
  solve's pointwise sweep, a class of predictors whose fluid share is
  near-null and at the entropy ceiling in the taper band, asserting a
  finite, flagged fallback — what the front produced.
- **SSP2(2,2,2) as a debugging tableau**, run and tested here; today it
  is only available from IMEXRungeKutta ("The partition").
- **A conductivity that falls where the field's energy dwarfs the
  fluid's**, not only by density or `B²/D`: on the magnetized star the
  cold gas outside the surface fell back in half its stages until σ was
  cut at `ρ = 10⁻⁶`, a density of this star's chosen by hand ("Ohm's law
  and the conductivity", "Validation ladder" 6). The nearest published
  version is Palenzuela (2013)'s switch to a force-free exterior at
  50–400 `ρ_atm`.
- **A table's recovery warm-started from the cell's previous `log₁₀ T`**
  and rapidity, carried in the state as `D s` is ("Thermodynamics", "Warm
  starts"): it would save the step's one cold recovery a cell, 0.9–3.3 µs
  from `s` and the zero-pressure rapidity against 0.7 µs fully warm on DD2;
  the other recoveries of a step are seeded from the step cache.
- **The fourth-order matter transfer at a finer `h`**: whether its H1,
  0.8% above linear theory at `h = 0.25`, closes ("Matter to the
  vertices", "Validation ladder" 5).
- **The tabulated coupled star on boxes wider than `L = 96`** ("Validation
  ladder" 7): where its F line (3.1 kHz at `L = 96`, 2.88 kHz at `L = 48`)
  settles as `L` grows, or a radiative outer boundary in TreeGH (GH-6)
  in place of the wider box.
- **The potentials' outer boundary where no exact `A` exists** (dynamic
  cases): a gradient-extrapolating relative hook (option 2, upstream) or a
  gauge that damps a static `χ` (option 3); see "Boundaries", the design
  rule. Every case built so far takes option 1 or has periodic or
  flat-space faces. Likewise the vertex metric's outer boundary for a case
  without an analytic provider ("Metric at cells, faces and edges").
- **For a hole that forms in a collapse**, with no analytic solution
  inside: a field-only interior — paste or damp `𝔈` and `A` inside, and
  leave the matter to the ordinary reset, which rung 3 found sufficient
  for the matter ("The interior").
- **A resistive HLLC flux**; the split speeds are built and measured, and
  not the default ("Riemann solver").
- A mean-field dynamo in Ohm's law; force-free currents in the
  magnetosphere; subgrid (LES) models for unresolved amplification;
  neutrino leakage and M1 as a separate package on the same forest.

## Open questions

None is open. The questions decided, and when, are in [HISTORY.md, "Open
questions"](HISTORY.md#open-questions); those the Cowling star settled are
under "Validation ladder" 4, "What the star decides". What is measured
short is under [Known risks](#known-risks), and what is not built under
[Possible extensions](#possible-extensions). A new question is marked
**(open)** where it arises and listed here.

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
- Distributed runs are tested at test size only (three ranks of two
  threads on one machine; two H200s by `jobs/device-mpi.toml`); no study has
  run on more than one node, the coupled run has not run on a device, and
  the ranks-per-node choice is unmeasured
  ([Running distributed](#running-distributed)).
- Light-cone fluxes are diffusive in a star's interior. They smear
  contacts and slow waves: Balsara 2's contact converges as
  `h^{2/3}`, and Balsara 3's dense shell (ρ = 10.1 between a contact and a
  slow shock) is not resolved below 800 cells, where the 400-cell run's
  peak density is 8.35.
- The implicit stage may converge poorly where magnetization and `σΔt`
  are both large. Measured pointwise (the stage solve's sweep): not where both are large — the ideal seed is exact
  there — but at `b²/ρ ≥ 10` with `σΔt` between `10⁻³` and `10²`, when `E★`
  is so far from both the solution and the ideal field that neither seed is
  a fluid state. The
  [magnetization cutoff](#ohms-law-and-the-conductivity) takes σ to zero
  there, and with it (at the measured safe `μ_max = 10`) no fallback
  remains at `f ≤ 0.3`; the price is that above the taper the field
  evolves as in electrovacuum, not as force-free — no conduction current
  at all, where a magnetosphere's plasma would carry the force-free one —
  and the taper already begins at `μ = μ_max/10`, so σ is reduced a decade
  below the threshold a case states. On the magnetized star (rung 6) the stage's fallbacks
  were not the taper's to remove: with σ cut at `2ρ_min`, half the stages
  just outside the star fell back, and as many at `μ_max = 0.1` and at
  `σ₀ = 10³` — there the fluid's energy, not its rest mass, is what is
  small against the field's, which `B²/D` does not see. The case's density
  cutoff at `ρ = 10⁻⁶` removes all but 6·10⁻⁶ of them at `h = 0.375`, and
  every one at `h = 0.25` in the evolved spacetime.
- `q := ∇·E` turns numerical divergence of `E` into charge.
- A table whose cold β-equilibrium slice has a phase transition (a density
  jump at constant pressure; SRO LS220 has four, EntropyEOS's PR 15) is
  refused by `cold_barotrope`: the composition table is smooth in `r̄²`,
  and a star with a density jump needs it split at the jump. DD2 and SFHo
  have none. The radial modes (`src/tov_pulsation.jl`) are the
  polytrope's only: a table's star needs `Γ₁` from `cs²`.
- The tabulated star lies on the entropy window's minimum, the table's
  coldest row, so a cell that truncation pushes colder is floored by the
  recovery, raising `τ`. The reset counts these floors outside the
  atmosphere (`ohm_tally`'s `s_floors`, `s_floor_τ`); at `h = 1` to 4 M
  there are none ("Validation ladder" 7 has the studies).
- On the Cowling metric over 1000 M the tabulated star's `D Yₑ` moves by
  1.6·10⁻¹³ of 0.098 where its rest mass holds to 4.7·10⁻¹⁶, not
  separated; the candidate is the outer faces, where the Dirichlet
  atmosphere's `Yₑ` is the slice's and the reset's atmosphere keeps the
  replaced state's ("Validation ladder" 7).
- On the `L = 96` box the coupled tabulated star's rest mass gains up to
  1.3·10⁻¹⁰ over 1000 M, net of the reset, ten times `L = 24`'s change
  and of the other sign, and the constraints outside the star converge at
  only 1.0–1.4 on the finer pair around 100 M, perhaps the start's
  transient crossing the coarse levels; neither is separated ("Validation
  ladder" 7).
- Under the default SSP3(4,3,3), which is not stiffly accurate, the
  stored `𝔈` of a stiff conductor ends each step `O(Δt)` off its ideal
  value: inside the magnetized star about 1% of `B` at `h = 1`, against
  0.2% under ARS(4,4,3) ("Validation ladder" 6). Everything read from the
  state at a step's end inherits it — `E·B`, the field energy's electric
  part, the charge, a Poynting flux. ARS(4,4,3) is the remedy where it
  matters.
- Step 11's coupled study missed two of its acceptance numbers, and they
  stand as measured ("Validation ladder" 5): `ℋ` outside the star
  converges at 1.76 over 490–510 M on the finer pair, against 1.8, and the
  H2 mode is about 10% off at every `h`. The suite does not guard coupled
  convergence at all: `coupled/coupled_tests.jl` checks the constraints at
  `t = 0` and runs to `t = 1` at one resolution, so a regression there is
  found only by rerunning the studies.
- In a coupled run the vertex metric `Mv` is built with TreeAMR's default
  `walls = :half_open`, so its low wall vertices are TreeGH's evolved ones
  and its high wall and every ghost beyond an outer face are the case's
  analytic metric (`src/vertex_metric.jl`). The matter's interpolation
  next to an outer face then mixes the two, asymmetrically between the
  low and the high faces, by as much as TreeGH's solution has left the
  analytic star there. Not measured; it is a candidate for the `L = 96`
  rest-mass gain above (not for the Cowling `D Yₑ` drift, which has no
  `Mv`), which a budget of the fluxes through the outer faces would
  settle.
- MC floors the entropy of a few cells behind
  the cylindrical explosion's front (the reset writes them; minmod, GRaM-X's
  choice there, writes none), and MP5 is unstable above `cfl = 1/5`. A
  strong blast into a light medium is where the reconstruction's
  overshoots reach the recovery.
