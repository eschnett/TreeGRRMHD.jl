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

*Status: design only. No code exists yet; the implementation plan
(`PLAN.md`) is the next thing to be written, and every number below is
unmeasured.*

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
- **Not yet a production code.** No MPI and no I/O until TreeAMR has them
  (its M7 and M9); single node, threads or one device.

## Upstream prerequisites

What this package needs from its siblings, each with its consumer and its
reason, so that each item can be read on its own. "(req)" blocks the
milestone named; "(wanted)" has a workaround here; "(later)" is not needed
for the TOV milestones.

### TreeAMR

1. **(req, rung 1) A state vector spanning several field sets** —
   `statevector(sets)`, `scatter!(sets, u)`, `statearray(du, sets, i)`, as
   TreeAMR's `CODE.md` already specifies ("implemented with the first
   application that needs it"). Consumer: every evolution here. The
   evolved state is a cell-centered matter set, three edge-centered sets
   for `A`, a vertex-centered set for the gauge scalar, and — when coupled
   — TreeGH's vertex-centered set; one integrator advances all of them.
2. **(strongly wanted, rung 4) A limited conservative prolongation** for
   cell-centered data. "Limited" in the reconstruction sense: a nonlinear
   prolongation — slope-limited (minmod/MC) or positivity-scaled toward
   the coarse average — that is still exactly conservative (the subcell
   averages reproduce the coarse cell's average), second-order where the
   data are smooth, and never creates new extrema (fine values stay within
   the range of the neighbouring coarse averages). Reason: at a neutron
   star's surface ρ drops by about ten orders of magnitude, and the linear
   width-3 operator overshoots there into cells below the atmosphere;
   TreeHydro measured 4096 floored cells at width 3 against none at
   width 1 on a shock crossing a coarse-fine face, at 0.47% in L1. Being
   nonlinear, it is not a fixed-weight tensor product, so it needs a new
   operator family or a hook covering both ghost filling and the regrid
   transfer. Workaround: width 1 near surfaces.
3. **(wanted) A device-capable boundary hook that reads the interior**,
   for outflow conditions. Workaround: Dirichlet from the initial or
   analytic exterior, which is what the TOV milestones need anyway.
4. **(wanted) `workgroupsize` on `map_blocks!`** (also on TreeGH's list).
5. **(later)** MPI (M7), I/O and checkpointing (M9), point interpolation.

### TreeGeneralizedHarmonic

1. **(req, rung 5) A stress-energy input.** A vertex-centered `Tsrc` field
   set (10 components, `G = 0`) read at the owned point in
   `gh_rhs_kernel!`, following the `Hsrc`/`gauge_at` pattern, switched by a
   `Val` in `GHProblem`, entering as
   `∂ₜΠ_ab += 16π α√γ (T_ab − ½ g_ab T)`. (That factor is derived from
   `notes/formulation.md` §1–2 and must be checked there.) The ADM monitor
   gains the matter terms, `ℋ → ℋ − 16πρ_ADM` and `ℳ_i → ℳ_i − 8π S_i`,
   signs to be checked likewise.
2. **(req, rung 5) A right-hand side callable inside an external
   integrator**: `gh_rhs!` on its sub-state of a multi-set state vector,
   with the problem constructible without `evolve!`, and the regrid,
   indicator and diagnostics reachable as functions. The coupled driver
   lives here and calls them.
3. **(req, rung 5) The CFL factor re-validated under an SSP3 explicit
   tableau.** `cfl = 1/4` was measured with RK4, whose stability region on
   the imaginary axis is wider than SSPRK3's.
4. **(wanted) Julia 1.10.** TreeGH pins `julia = "1.11"` and
   `LinearAlgebra = "1.11"`; a grep of its `src/` found no 1.11-only
   feature (`public`, `Memory`, `ScopedValue`), so the pin looks
   incidental. If it stays, the coupling here becomes a package extension
   (see [Julia version](#julia-version)).
5. **(wanted) A ghost width of `q/2 + 2`** as an option, so that the
   4th-order vertex→cell metric interpolation reaches every stored cell of
   a `G = 3` matter set (see [Metric at cells, faces and
   edges](#metric-at-cells-faces-and-edges)).
6. **(later)** A damped-harmonic gauge driver, radiative or
   constraint-preserving outer boundaries, excision or a generalized
   interior for collapse, Ψ₄ extraction.

### EntropyEOS

1. **(req, rung 4) Analytic EOSes behind the same interface** (decided): a
   Γ-law, a polytrope and a hybrid (piecewise polytrope plus thermal Γ),
   written as potentials `U(ρ, s)` so that `evaluate`, `con2prim_safe`,
   the atmosphere policy and the device view all work unchanged. A
   polytrope is the isentrope of a Γ-law, `U = K(s) ρ^{Γ−1}/(Γ−1)`; the
   hybrid's thermal part has the same form on top of `U_cold(ρ)`.
2. **(wanted) A public fixed-`s` inversion** — the "at fixed `s`, solve
   for `w`" step that `con2prim` already contains — for the entropy
   fallback, and a public `s(ρ, ε, Yₑ)` for initial data.
3. **(later)** A cold β-equilibrium slice `Yₑ,β(ρ)` and `s_cold(ρ)` for
   tabulated-EOS stars.

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
each other in a test). The entropy equation is not a conservation law
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
this; a test pins both signs). The gauge is generalized Lorenz with damping `ξ`
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
translation-invariant `A` produces it. It is zero for stars.

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

The `q v` part is explicit. The `σ` part is **stiff** inside a star
(`σ Δt ≫ 1`), and is the implicit part of the time integrator.
Extension slots, not planned: a mean-field dynamo term, and force-free
currents in the magnetosphere.

### Stress-energy for the spacetime

`T^{ab} = 𝓔 n^a n^b + S^a n^b + n^a S^b + S^{ab}` from the same total
`𝓔`, `S_i`, `S^{ij}` as above; lowered with `g_ab`, it is what TreeGH reads.

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
- **Warm starts** (`u_guess`, the previous `log₁₀ T`) are carried per cell
  in the primitive set and passed to every EOS call, including those at
  reconstructed face states.
- **Analytic EOSes come from EntropyEOS** (decided; upstream item 1). Until
  they exist, nothing past rung 3 can run, and rungs 1–3 use its synthetic
  ideal-gas table.

## Primitive recovery

Because `E` is evolved, the electromagnetic energy `½(E² + B²)` and
momentum `ε_ijk Eʲ Bᵏ` do not depend on the fluid velocity. Primitive
recovery is therefore **pure hydrodynamics**: subtract them from `τ` and
`S_i`, project with `γ^{ij}` to EntropyEOS's scalar form
`(D, τ, D_Y, S_∥, S_⊥, B²)` with `S_⊥ = 0`, `B² = 0`, and call
`con2prim_safe` (decided). It never fails: atmosphere, ceilings, rapidity
cap, the entropy window and non-finite input are its policy, every repair
is reported in its flags, and every repaired state is exactly invertible.

The only coupling back to the fields is the implicit Ohm step, which is
[a nested solve](#the-implicit-stage-solve) around this same call.

**Entropy fallback** (proposed): where `τ_fluid = τ − D − ½(E² + B²)` is a
small difference of large numbers (electromagnetically dominated cells, the
magnetosphere) or the energy inversion fails, invert at the advected `s`
instead (upstream EntropyEOS item 2). The switch is a threshold on
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
| `P` — primitives, EOS aux, `B̄` at cells, `q`, cell metric | cell | ~25 | 3 | `=> nothing` |
| `B_d` — `curl A` | face `d` | 1 each | 3 | `=> nothing`, recomputed every RHS |
| `F_d` — matter fluxes | face `d` | 10 | 0 | `=> nothing`; `restrict_interfaces!` is the flux correction |
| `EMF_d` | edge `d` | 1 each | 0 | `=> nothing`; `restrict_interfaces!` is the CT correction |
| `Tsrc` — `T_ab` for GH | vertex | 10 | 0 | `=> nothing` |

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

### Riemann solver

`riemann_flux(::Val{:llf | :hlle}, …)` on the ten matter-plus-`𝔈`
variables at every face (proposed). Because `S_i` and `τ` are totals, their
fluxes contain the Maxwell stress, which propagates at the speed of light;
the default wave speeds are therefore the local light cone,
`λ_± = −β^d ± α √(γ^{dd})`, for all ten variables — the choice of the
resistive codes this design follows, robust and diffusive. Split speeds
(fluid speeds for `D`, `D Yₑ`, `D s`, light speed for the rest) and a
resistive HLLC are (open), to be decided by measurement on the TOV star.

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

### Sources

At cell centers, explicit: the geometric momentum and energy sources
above; the shift term `√γ q βⁱ` and the `−α√γ q vⁱ` part of the current in
Ampère's law. At vertices, explicit: the gauge damping `−ξ α Φ̃`. **The only
implicit term** is Ampère's `−α√γ σ W [E + v × B − (v·E) v]`.

### Floors, the atmosphere and the magnetosphere

EntropyEOS's policy defines the atmosphere (`ρ_atm`, triggers, ceilings)
and `con2prim_safe` applies it; a stage limiter resets `v = 0` in
atmosphere cells and writes back only those cells, as TreeHydro's does
(proposed). The conductivity model takes σ → 0 in the atmosphere, so the
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
implicit part is L-stable and stiffly accurate, so that `σ Δt → ∞`
recovers ideal MHD at the tableau's order; SSP2(2,2,2) is kept for
debugging (proposed). `Δt` is fixed per chunk from the light-speed CFL on
the finest level; TreeGH's CFL factor is re-measured under SSP3 (upstream
TreeGH item 3). The driver is chunked, with a regrid between chunks and a
fresh integrator after each, as in TreeHydro and TreeGH.

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
`R(v) = v − V_c2p(S − E(v) × B, τ − ½(E(v)² + B²))`. **One con2prim is used
for every σ.** A plain fixed-point iteration contracts roughly like
`λ b²/(ρh)` with `λ = cW/(1 + cW)`: fast where `σΔt` is small (the
magnetosphere, where `E` barely depends on `v`) or where the magnetization
is low (a star's interior); slow or divergent only where both are large,
such as the surface layer. There the outer iteration is Newton on `v`,
still one con2prim per iteration, seeded from the `σΔt → ∞` limit: there
`E = −v × B` and the whole inversion is exactly the ideal-MHD problem,
which the same EntropyEOS routine solves when called with `B² ≠ 0`. That
ideal call is a seed — and the exact answer for `IdealConductor` — not a
second con2prim. The last resort is the lagged `v` of Palenzuela et al.
(2009). Every outcome is flagged, counted, and reduced per chunk.

### The IMEX package

OrdinaryDiffEq has no IMEX-SSP scheme (the Pareschi–Russo request is open
as SciML/OrdinaryDiffEq.jl#2065); its IMEX Runge–Kutta methods (KenCarp
3/4/47/5/58, CFNLIRK3) have a non-SSP explicit part and solve their stages
by Newton on the whole vector with a Jacobian, which cannot express a
per-cell solve coupled to con2prim. So the integrator is a **new small
stand-alone package** (decided; name open, e.g. `IMEXRungeKutta`), not
ClimaTimeSteppers (whose implicit stages always go through its own Newton
loop). Its specification (proposed):

- Tableaus: SSP2(2,2,2), SSP2(3,3,2), SSP3(3,3,2), SSP3(4,3,3)
  (Pareschi & Russo 2005); ARS(2,2,2), ARS(4,4,3) (Ascher, Ruuth &
  Spiteri 1997).
- `f_exp!(du, u, p, t)`, and a user stage solver
  `solve_imp!(u, u★, γΔt, p, t)` that returns `u` with
  `u = u★ + γΔt g(u)` by any means. The implicit tendency is recovered as
  `g = (u − u★)/(γΔt)`, so no separate evaluation of `g` exists or is
  needed.
- A `stage_limiter!` hook; fixed `Δt`; generic arrays (device arrays
  through broadcasting); no SciML dependency beyond, at most, SciMLBase's
  problem types.
- Tests: convergence order on smooth non-stiff and stiff relaxation
  problems; the asymptotic-preserving limit; the SSP property on a
  Burgers equation with relaxation.

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
edges. One interaction with the ghost widths is (open): a 4th-order
interpolation to a cell center reaches one vertex beyond the cell on either
side, so reaching the outermost ghost cell of a `G = 3` cell set needs
`G = 4` on the vertex set, and TreeGH fixes `G = q/2 + 1 = 3`. The
primitives in that outermost ghost cell are read by the widest
reconstructions. The options are a larger GH ghost width (upstream TreeGH
item 5), a 2nd-order interpolation in that one layer, or exchanging the
cell metric as a ghost-filled set.

### Stress-energy to the vertices

`T_ab` is computed at the cells from the primitives and the cell metric,
then averaged to each owned vertex from its `2^D` neighbouring cells into
`Tsrc` — second order, and positivity-preserving for the energy density at
a stellar surface where a higher-order interpolation would ring
(proposed; a 4th-order option is kept, and which one wins is (open)).
TreeGH reads `Tsrc` at the owned point (upstream TreeGH item 1).

### Gauge for the TOV milestones

TreeGH's static-background path, with no gauge driver (proposed): the TOV
star is a static, `isbits` `SpacetimeMetrics.AbstractMetric` in isotropic
coordinates, generic over dual numbers so TreeGH can differentiate it
inside kernels. `H_a = −Γ_a[g_TOV]` is sampled and frozen, as TreeGH
already does for static backgrounds, and the outer boundary is Dirichlet
from the static exterior. A fixed gauge source is a legitimate gauge for a
perturbed or oscillating star; it is not for a collapse, which needs the
gauge driver (upstream, later).

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
evaluations, and the atmosphere stage limiter after it. The right-hand side
never mutates `u`.

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
dipole). Outflow conditions wait for upstream TreeAMR item 3.

## Refinement

A Löhner indicator on `ρ`, optionally on `|B|` and `|E|`, following
TreeHydro's `hydro_flags`, plus geometric radius boxes around a star that
fix the levels the star is resolved with (proposed). `regrid!` is given
every set over the forest: the evolved ones with their schedules, the rest
`=> nothing`.

## Julia version

`julia = "1.10"` wherever possible (decided). TreeAMR, EntropyEOS,
SpacetimeMetrics and TreeHydro all allow 1.10; only TreeGH pins 1.11
(upstream TreeGH item 4). If that pin stays, TreeGH becomes a weak
dependency and the coupling a package extension,
`TreeGRRMHDTreeGHExt`, so that everything on flat and analytic
backgrounds — rungs 1–4 below — stays on 1.10 (proposed).

## Validation ladder

These become the milestones of `PLAN.md`; each rung is a set of testsets
named as claims.

1. **Flat space, 1D**: the vacuum electromagnetic wave (`Vacuum`) at the
   scheme's order; the large-amplitude circularly polarized Alfvén wave
   (`IdealConductor`) at the scheme's order; the resistive shock tube swept
   over σ (Komissarov 2007), approaching Balsara 1 as σ → ∞; the
   self-similar current sheet against its analytic profile; Balsara 1–5
   with `IdealConductor`, against GRaM-X's figures; a σ × magnetization
   sweep of the implicit stage solve, recording where each branch of it
   is taken.
2. **Flat space, 2D and 3D, with AMR**: the cylindrical explosion, the
   magnetic rotor, Orszag–Tang, field-loop advection across coarse-fine
   faces — `∇·B` at round-off, rest mass and magnetic flux conserved
   across levels to round-off, regrids included.
3. **Analytic curved backgrounds**: the Wald solution (`Vacuum`,
   Kerr-Schild) and the Michel monopole or magnetized Bondi flow
   (`IdealConductor`).
4. **TOV in the Cowling approximation**: a Γ = 2, K = 100 polytrope
   (needs upstream EntropyEOS item 1); stationarity and mode frequencies.
5. **TOV in a dynamical spacetime**: the F, H1, H2 radial modes against
   the literature (Font et al. 2002); convergence of the GH and ADM
   constraints with matter (needs the TreeGH items).
6. **Magnetized TOV**: an interior poloidal field from an analytic `A`,
   σ(ρ) through `DensityProfile`, a vacuum dipole exterior (Palenzuela
   2013).
7. **TOV with a tabulated EOS and `Yₑ`** (needs upstream EntropyEOS
   item 3 for the initial data).

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
- The IMEX package's name.

## Known risks

- No MPI yet: single node until TreeAMR's M7.
- Unlimited conservative prolongation at a stellar surface floors cells.
- Light-cone fluxes are diffusive in a star's interior.
- The implicit stage may converge poorly where magnetization and `σΔt`
  are both large; rung 1's sweep exists to find out where.
- `q := ∇·E` turns numerical divergence of `E` into charge.
