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

*Status: steps 0 and 1 of `PLAN.md` are done — the scaffolding, and the
pointwise layer: units, the variable layouts, the point metric, `prim2con`
and the 3+1 primitive recovery — and step 2 is next. There is no mesh code
yet, and every number below about the scheme is unmeasured. (Amended in
step 1.)*

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

**State on 2026-10-13** (amended in step 0), as `test/prerequisite_tests.jl`
reports it and, for TreeGH, as its `main` reads:

- TreeAMR **0.2.0**, registered: none of AMR-1 to AMR-6 has landed, so
  step 5 waits for AMR-1.
- EntropyEOS **1.0.0** on its `main`, not registered (EOS-0 waiting):
  **EOS-1 and EOS-2 have landed**; EOS-3 and EOS-4 have not.
- IMEXRungeKutta **1.4.0**, tagged, not registered: the plan's steps 4a–4c
  are done there, and it is a dependency from step 0.
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
  vertex-centered set; one integrator advances all of them.
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
  near surfaces.
- **AMR-5 (wanted, step 14) `workgroupsize` on `map_blocks!`** (also on
  TreeGH's list).
- **AMR-6 (later) A device-capable boundary hook that reads the interior**,
  for outflow conditions. Workaround: Dirichlet from the initial or
  analytic exterior, which is what the TOV milestones need anyway.
- *(later)* The use of MPI (M7) and of checkpoints (M9, now TreeIOHDF5's)
  and point interpolation (M11), all three of which TreeAMR has; none is
  an upstream item any more. *(Amended in step 0.)*

This is a **physics package** in TreeAMR's sense (its `CODE.md`,
"Companion interface", 2026-10-08): it uses TreeAMR's exports only, and
bounds TreeAMR by a minor series, `TreeAMR = "0.2"`. The prerequisite
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
  reconstructed face states.
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
while over the prerequisite test's 24 states the conditioned errors are at
most 2.1e−14 in `ρ` and `w` and 1.7e−13 in `s`.

Through the 3+1 layer (measured in step 1, `test/recovery_tests.jl`): on
1000 random states per `Γ ∈ {4/3, 5/3, 2}` — `ρ` across the box, `s`
across the window, `W ≤ 10`, metrics from a boosted Kerr-Schild hole with
`a ≤ 0.9` at random events between `r = 3M` and `20M`, and `E`, `B`
carrying 1e−4 to 100 times the fluid's `τ` — `con2prim_hydro ∘ prim2con`
has conditioned worst errors 4.5e−13 in `(ρ, w, v)` and 8.4e−12 in `s`
(raw `s` up to 5.7e−7 at `Γ = 2`, where the thermal energy is a tiny part of `τ` and `κ_s` is large). In the conserved
variables the round trip is 9.8e−13 relative, with `S_i` and `τ` measured
against the energy density `𝓔 = τ + D` they are components of (`τ` against
itself: 9.9e−13). That is the solver's own tolerance — `con2prim` stops
when its energy residual is below 1e−12 of `τ`, often at the cold seed
with no Newton step — and not rounding, so 1e−12 has no margin there; the
step-1 acceptance was written at exactly that number (proposed in step 1:
the norm; see the step-1 report). `√γ D s` is not an input of the
hydrodynamic recovery and is checked through `s`. `con2prim_ideal` on
states built with `E = −v × B` (the same draws, `B²` up to 100 times the
fluid's `τ`, `B` along and across `v` included) has conditioned worst
errors 1.8e−12 in `(ρ, w, v)` and 3.7e−12 in `s`. At `Float32` the
recovery runs in a kernel and lands within 3.6e−5, conditioned, of the
state (EntropyEOS's `Float32` tolerance is 6.1e−5) on states dense enough
that the thermal energy is a percent of `τ`; at `ρ = 1e−9` the gas's `U` is
1e−7 of its rest mass and `Float32` cannot determine `s` at all.

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

`P`'s 30 slots are laid out in `src/variables.jl` (amended in step 1: the
table said "~25"): the eleven the recovery returns — `ρ★`, `vⁱ`, `s`, `Yₑ`,
the rapidity `w` (also the next recovery's warm start), `p`, `h`, `cs²` and
the EOS's own warm start — then the policy flags (as a number; every bit is
below 2¹⁶, so exact even in `Float32`), `Eⁱ`, `Bⁱ`, `q`, and the metric
`α`, `βⁱ`, `γ_ij` and `√γ`, all undensitized. The metric derivatives and
`K_ij` the sources read are not in it yet: step 2 decides the form of the
sources, and with it whether they are stored or interpolated on the fly
(proposed in step 1).

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
dipole). Outflow conditions wait for AMR-6.

## Refinement

A Löhner indicator on `ρ`, optionally on `|B|` and `|E|`, following
TreeHydro's `hydro_flags`, plus geometric radius boxes around a star that
fix the levels the star is resolved with (proposed). `regrid!` is given
every set over the forest: the evolved ones with their schedules, the rest
`=> nothing`.

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
so does `GeometricEOS` around a table at `Float64` (measured in step 1). `GeometricEOS` around a table is not
`isbits` (the table holds arrays) and is passed to a CPU kernel as it is;
it carries an `Adapt` rule for a device.

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

## Known risks

- No MPI yet: single node, although TreeAMR now has its M7 (amended in
  step 0).
- Unlimited conservative prolongation at a stellar surface floors cells.
- Light-cone fluxes are diffusive in a star's interior.
- The implicit stage may converge poorly where magnetization and `σΔt`
  are both large; rung 1's sweep exists to find out where.
- `q := ∇·E` turns numerical divergence of `E` into charge.
