# TreeGRRMHD.jl — History

How the package got where it is: who decided what and when, the
alternatives that were tried or rejected and why, measurements of code that
no longer exists or that later measurements replaced, upstream bugs since
fixed, and the version history. `CODE.md` describes the current design and
links here where a reason lives in the past; this file follows `CODE.md`'s
sections, with dates, and within a section runs in date order.

The entries use the step numbers of `PLAN.md`:

- step 0 — scaffolding;
- step 1 — units, variables and the 3+1 primitive recovery;
- step 2 — pointwise physics;
- step 3 — the implicit stage solve;
- steps 4a–4c — the IMEX package, in its own repository (IMEXRungeKutta);
- step 5 — the staggered scheme in 1D: smooth problems (rung 1a);
- step 6 — shock tubes and the stiff limit in 1D (rung 1b);
- step 7 — 2D and 3D, and higher-order reconstruction (rung 2a);
- step 8 — coarse-fine faces and regridding (rung 2b);
- step 9 — analytic curved backgrounds (rung 3);
- step 10 — the TOV star in the Cowling approximation (rung 4);
- step 11 — the TOV star in a dynamical spacetime (rung 5);
- step 12 — the magnetized TOV star (rung 6);
- step 13 — tabulated EOS and Yₑ (rung 7);
- step 14 — precision, threads, devices, benchmark;
- step 15 — review pass.

The "rungs" are those of `CODE.md`'s "Validation ladder". The upstream
items `AMR-n`, `EOS-n`, `GH-n` and `SM-n` are those of `PLAN.md`, "Upstream
requirements".

**Contents**

- [Overview](#overview)
- [The steps of the plan](#the-steps-of-the-plan)
- [Goals](#goals)
- [Scope and non-goals](#scope-and-non-goals)
- [Upstream prerequisites](#upstream-prerequisites)
  - [TreeAMR](#treeamr)
  - [TreeGeneralizedHarmonic](#treegeneralizedharmonic)
  - [EntropyEOS](#entropyeos)
  - [SpacetimeMetrics](#spacetimemetrics)
- [The equations](#the-equations)
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
  - [Two scripts](#two-scripts)
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
- [Validation ladder](#validation-ladder)
- [Open questions](#open-questions)
- [Known risks](#known-risks)
- [The test suite](#the-test-suite)

## Overview

**Undated.** `CODE.md` used the markers of the sibling packages:
**(decided)** was a decision taken in review, beside **(proposed)** and
**(open)**. Its rule for changes was: "When the implementation shows this
document wrong or incomplete, amend it and say so — '(amended in step N)',
'(measured in step N)' — rather than diverging silently." The decisions so
marked, and the amendment notes, are recorded under the topics below.

**2026-09-23.** `CODE.md` records the reviewed design (2026-09-23).

## The steps of the plan

**2026-10-13, step 0 — Scaffolding.** *(Done, 2026-10-13, on `claude/step-0-scaffolding`.)*

`CODE.md`: "Julia version", "Upstream prerequisites". Waits: none (EOS-0 by workaround).

Changes:
- **`Project.toml`:**
  - deps: KernelAbstractions, StaticArrays, Adapt, TreeAMR (`"0.2"`; amended in step 0 from
    `"0.1.2"`), SpacetimeMetrics (`"1.6"`), EntropyEOS (`"1"`, by `[sources]` until EOS-0), and
    IMEXRungeKutta (`"1.4"`, by `[sources]`; added in step 0);
  - `julia = "1.11"`.
- **`src/`:**
  - `src/TreeGRRMHD.jl`, the module shell, replacing `hello`/`domath`;
  - `src/precision.jl` and `src/device.jl`, ported from TreeHydro (`to_backend`, `hostcopy`,
    `hostcopy!`).
- **`test/`:**
  - `test/Project.toml`, and `test/runtests.jl` with TreeHydro's duplicate-`const` guard;
  - `test/prerequisite_tests.jl`: every name this package calls exists; a SpacetimeMetrics metric
    runs as a kernel argument; one skipped-until-landed testset per upstream ID.
- **CI:** `.github/workflows/CI.yml` after TreeHydro's, with cells for 1.11 on macOS, 1 on Linux,
  1 on macOS, and 1 on Linux at 4 threads. It adds the unregistered dependencies by URL.
- **Docs:** `README.md`; `CLAUDE.md`'s "Current state" and "Commands".

Accept:
- `Pkg.test()` is green on 1.11 and on the current release.
- A clean `git archive` instantiates (with the URL adds) and passes.
- The prerequisite testsets report each upstream ID as landed or waiting.

**Steps 1 to 3.** Step 1, the pointwise recovery: `src/units.jl`, `src/metric3.jl`,
`src/variables.jl`, `src/prim2con.jl` and `src/con2prim.jl`. Step 2, the
pointwise physics: `src/metric.jl`, `src/stress_energy.jl`, `src/ohm.jl`,
`src/flux.jl`, `src/riemann.jl`, `src/emf.jl` and `src/sources.jl`. Step 3,
the implicit stage solve: `src/stage_solve.jl`, with
`MagnetizationCutoff(model, μ_max)` in `src/ohm.jl` tapering σ to zero where
`B²/D` exceeds `μ_max` (decided by Erik 2026-10-14; the bracketed
alternative is on `claude/step-3-bracketed-stage-solve`, not in the code).

**2026-10-14, steps 4a–4c.** Steps 4a–4c are done in the IMEX package's own
repository (`~/src/jl/IMEXRungeKutta`):
IMEXRungeKutta 1.4.0 is tagged, unregistered, and a dependency from step 0
(`PLAN.md`'s status, amended 2026-10-14). With the upstream releases of that
date, step 5, the first evolution, waited for nothing (AMR-1, EOS-1, EOS-2,
4b).

**Step 5** (the staggered scheme in 1D, rung 1a). Step 5's mesh code:
`src/fieldsets.jl`, `src/curl.jl`, `src/reconstruction.jl`,
`src/evolution.jl`, `src/driver.jl` and `src/cases/` (`em_wave_case`,
`alfven_wave_case`). The kernels are written for D = 1, 2, 3; step 7
validated 2D and 3D.

**Step 6** (shock tubes and the stiff limit, rung 1b). Non-periodic cases
get Dirichlet hooks from their own data (`DirichletBoundary` in
`src/driver.jl`; the potentials' are Dirichlet in the field, since step 8
TreeAMR's relative hook, AMR-7), the reset is counted as stage and step
limiter with the step limiter's injection (`reset_step!`, `ohm_tally`), and
`src/cases/shocktubes.jl` has Balsara 1–5, the resistive tube, Komissarov's
current sheet and the atmosphere tube. `bin/shocktube_reference.jl` made the
3200-cell references in `test/data/`; `bin/tableau_comparison.jl` made
`CODE.md`'s SSP3-vs-ARS table. **SSP3(4,3,3) is the default tableau**
(decided in step 6). Non-periodic runs were CPU-only until AMR-7 landed
(step 8); now every backend.

**Step 7** (2D, 3D and higher-order reconstruction, rung 2a). `limiter`
also takes `:ppm`, `:wenoz`, `:mp5`; every reconstruction an offset in the
differences `q_k − q₀` so that 2D/3D runs invariant along `y`/`z` are the 1D
run bit for bit; the initial data's Gauss average is in difference form for
the same reason. `max_div_B` and `evolve!`'s `after_step` monitor;
`src/cases/geometry.jl` and `src/cases/multid.jl`.
`bin/reconstruction_comparison.jl` made the per-reconstruction Balsara table
and `test/data/balsara3_ppm.ref`.

**Step 8** (coarse-fine faces and regridding, rung 2b; finished on TreeAMR
0.2.2). Both walls of a closed box prescribed alike (decided by Erik
2026-10-15; since TreeAMR 0.2.2 the potentials' `walls = :symmetric` and
relative hook, AMR-7 and AMR-8, where step 8 first had its own `low_walls!`,
`outer_points!`), so a uniform closed box is mirror-symmetric bit for bit;
the owned low wall is prescribed: zero tendency, copied into `u` by the
reset (`walls_to_state!`). `src/refinement.jl`; `evolve!` with a static or
an adaptive mesh (`adapt_case!` over every set, AMR-3). Decided by
measurement (`CODE.md`, "Refinement"): the matter's regrid Conservative
width 3 **bounded**; `A_j` conservative along the edge with point values of
**width 4** across it (AMR-9), `Φ̃` PointValue width 4 (`ops_Φ`); `ξ = 16/L`
on every mesh (`L` the longest side; Erik's decision, 2026-10-15); the
matter's ghost fill in **primitives** (Erik's decisions, 2026-10-15 and
2026-10-23): the scratch set `W = (ρ★, s, Yₑ, uⁱ, Eⁱ)`, copied at
same-level faces and prolongated **parent-centred** at refinement
boundaries (`default_ops_W`, Conservative width 3 bounded, Athena++'s
choice), recovery on owned cells only, `P` in ghosts by forward EOS
(`P_from_W`, also for owned cells), no ghost ever recovered, the conserved
ghosts never written by the scheme (only `regrid!`'s own fill); `CODE.md`,
"Refinement", records the three working fills and their branches. Every
fill applies the complete outer boundary condition in one TreeAMR fill
(Erik, 2026-10-15). The magnetic flux across levels holds to rounding as a
rate and as a state (but at an outer face: AMR-11). `bin/interface_order.jl`,
`bin/gauge_damping.jl`, `bin/field_loop_refined.jl`, `bin/ghost_fill.jl`
made `CODE.md`'s step-8 tables.

**2026-11-02, step 9** (analytic curved backgrounds, rung 3;
`claude/step-9-curved-backgrounds`). `src/cases/kerr_schild.jl` — Michel's
flow, `wald_case` and `bondi_case` on a Schwarzschild hole in Kerr-Schild
coordinates (`Analytic(KerrSchild(1))`), on the octant `[0, 4M]³` with the
hole at its corner and Dirichlet faces (the monopole's Dirac string leaves
through the corner); `src/interior.jl` — `HoleInterior(r_in; stage, step)`,
pasted by the reset first, as stage and step limiter, position-only
(proposed; `r_in = M`). Every kernel already read the metric at its own
point, so the scheme did not change; pointwise evaluation is 44% of a step
on a 16³ blast (`bin/metric_cost.jl`; kept: decided by Erik 2026-11-03, no
cache; caching is step 10's). `bin/curved_stationarity.jl` is the `t = 50M`
study (`CODE.md`, "Validation ladder" 3). Round 2 (2026-11-02): the
`t = 50M` study fails the predicted order, and `bin/curved_localise.jl`
located it: a mode that grows through the potentials' relative Dirichlet
hook (AMR-7) copying the gauge drift of `A` across the face; with the
potentials absolute in the ghosts both cases are stationary and second
order (`CODE.md`, "Validation ladder" 3, "Where the error lives"; the fix is
Erik's). Resets deep inside the hole, `r < (r_in + 2M)/2`, are not counted
(Erik, 2026-11-02; `CODE.md`, "The interior").

**2026-11-03, step 9, rounds 3 and 4.** Round 3 (option 1, decided by
Erik): `MHDCase(…; static_potential = true)` makes the potentials'
Dirichlet hooks absolute (`DirichletBoundary(…; absolute = true)`), valid
only for potentials that do not depend on `t`; the Kerr-Schild cases
declare it, every other case keeps the relative hook (AMR-7) and its output
bit for bit; the boundary testset in `test/curved_tests.jl` holds a gauge
offset out of `𝔅`. The design rule is in `CODE.md`, "Boundaries".
`bin/curved_localise.jl`'s `relhook=true` is the negative control,
`interior=false` the no-interior run. Round 4: `spinning_wald_case` (Wald's
uncharged field on a Kerr hole, `a = M/2`, `0.9M`, a passive floor fluid,
`Φ̃ = aB₀`, so `gauge_damping = 0`, `MHDCase`'s new field), `kerr_schild_r`;
the octant shifted by `L/256` in `z` off the `r = 0` disk;
`bin/curved_localise.jl kwald`. The `50M` study (jobs 588687–588689,
588704–588706): second order at `10M`, but `M/8 → M/16` at `50M` fails —
`a = M/2` 0.79–1.55, `a = 0.9M` 1.12–1.93; the Euclidean paste at
`a = 0.9M` misses `r₋` and grows, the ball in Kerr-Schild `r` does not
(`CODE.md`, "The spinning Wald case"; both for Erik). The interior (the
paste), the coarse-pair rates and the deduplication were decided by Erik
2026-11-03.

**2026-11-05.** Step 9 was accepted as is by Erik on 2026-11-05, including
round 4's spinning Wald case, whose `50M` drift fails the prediction on the
finest pair at both spins. That spacetimes with black holes are not
expected to work out of the box is recorded under [Known
risks](#known-risks) (`CODE.md`, "Known risks").

**2026-11-06/07, step 10** (the TOV star in the Cowling approximation, rung
4; `claude/step-10-cowling-tov`, branched from
`eschnett-claude/spacetime-hydro-coupling-144389`). Step 10, the Cowling TOV
star on `Analytic(TOVMetric)` (the analytic TOV metric), waits for nothing:
it no longer reads TreeGH (decided by Erik 2026-11-06; [The coupling
contract](CODE.md#the-coupling-contract)). It built `src/tov.jl` and `src/cases/tov.jl`;
the public verbs (`prepare!`, `rhs!`, `stage_solve!`, `stage_limit!`,
`step_limit!`, `max_rate`, `initial_state!`, `MHDProblem(fields, case; …)`);
the indexed provider calls and `PointwiseMetric` (the contract binds once
per centering; replaced by `VertexMetric` in round 2, 2026-11-08);
`:hlle_split` in `src/riemann.jl`. `bin/tov_cowling.jl` is the step-10
script; `bin/tov_study.jl`, `bin/tov_drift.jl` and `bin/tov_modes.jl` made
`CODE.md`'s rung-4 record. Built and measured, for Erik's review.

**2026-11-08.** Steps 0 (scaffolding), 1 (the pointwise recovery), 2 (the
pointwise physics), 3 (the implicit stage solve), 5 to 8 and 9 are done;
step 10 is built and measured, for Erik's review (`PLAN.md`'s status
paragraph, amended 2026-10-14, in step 9's rounds 2–4, 2026-11-06 and
2026-11-08, and `CLAUDE.md`'s "Current state" of that date). What the
steps delivered, as `CODE.md`'s status paragraph listed them (amended in
step 9, on 2026-11-06 and in step 10): steps 0 to 3 and 5 to 7 of `PLAN.md`
are done — the scaffolding; the pointwise layer: units, the variable
layouts, the point metric, `prim2con` and the 3+1 primitive recovery; the
pointwise physics a right-hand side evaluates; the implicit stage solve;
the staggered scheme on the mesh, with the vacuum electromagnetic wave and
the circularly polarized Alfvén wave in 1D (rung 1a); and the shock tubes,
the resistive tube swept over σ, the current sheet and the atmosphere in 1D
(rung 1b); and the scheme in 2D and 3D on a uniform mesh, with PPM, WENO-Z
and MP5 beside PLM (rung 2a) — as are steps 4a–4c in IMEXRungeKutta;
coarse-fine faces and regridding (rung 2b, step 8); and the analytic curved
backgrounds, the Wald solution and the magnetized Bondi flow on a
Kerr-Schild hole with its interior (rung 3, step 9).

Erik redesigned the coupling contract (TreeGH's PR 25,
merged 2026-11-08 as 61bcb75, builds GH-1 as `ADMMetric`): both directions
cross at TreeGH's owned vertices in 3+1 variables, and every conversion
between centerings is this package's. `CODE.md` ("The coupling contract",
"Metric at cells, faces and edges", "Matter to the vertices") and `PLAN.md`
(GH-1, GH-3, step 10's round 2, step 11) are amended; step 10 replaces its
`PointwiseMetric` with `VertexMetric`. Step 11 waits for GH-2 to GH-4 (GH-0
has landed, and GH-1 as `ADMMetric`, TreeGH's PR 25, on 2026-11-08).

**2026-11-08, step 10, round 2** (`src/vertex_metric.jl`,
`test/vertex_metric_tests.jl`) built this package's side of the
redesigned contract: `VertexMetricSet`, `vertex_metric!`, `VertexMetric`,
`SampledADM` (GH-1's reference); `PointwiseMetric` is gone;
`Analytic(TOVMetric)` stays the Cowling star's provider. Measured to 200 M
(`bin/tov_study.jl metric=vertex` with `bin/tov_compare.jl`, jobs
593624–593629; `CODE.md`, "Metric at cells, faces and edges"; the numbers
under [Validation ladder](#validation-ladder), 2026-11-09, step 10, round
2): the advected `√γ D s` just outside the surface, which alone does not
converge, was left for Erik.

## Goals

**Undated.** Three goals were marked (decided), decisions taken in review:
resistive GRMHD with a magnetosphere; a magnetized TOV star as the first
science target; and ∇·B = 0 to round-off, on every level and across every
regrid, by the vector potential.

## Scope and non-goals

**Undated.** "No neutrinos" and "Upstream changes are Erik's" were marked
(decided).

**Step 0.** "Not yet a production code" was amended in step 0: this item
used to say "no MPI and no I/O until TreeAMR has them".

**2026-11-06.** "No mesh machinery" was amended when each producer was to
interpolate its own data.

**2026-11-08.** "No mesh machinery" was amended again: interpolation
within a block between centerings lives here in both directions, the
matter's `(ρ, S_i, S_ij)` from cells to TreeGH's vertices and the metric's
`(α, β^i, γ_ij, K_ij)` from TreeGH's vertices to this package's cells,
faces and edges; that this package is the only one with mixed centerings
was decided by Erik 2026-11-08 ([The coupling
contract](CODE.md#the-coupling-contract)). "No spacetime evolution" was
amended the same day: the matter's `(ρ, S_i, S_ij)` and the metric's
`(α, β^i, γ_ij, K_ij)` are both at TreeGH's vertices.

## Upstream prerequisites

**Undated.** The section was amended with `PLAN.md`.

**Step 0.** That `src/device.jl` uploads its own metadata with
`KernelAbstractions.allocate` rather than reaching for TreeAMR's internal
`todevice` was proposed in step 0.

**Step 0.** The rule in `PLAN.md`, "Upstream requirements", that additive changes stay in the same
minor series read 0.1.x for both TreeAMR and EntropyEOS until step 0 amended it to TreeAMR 0.2.x
and EntropyEOS 1.x. The state column of the table of upstream requirements was added in step 0.

**2026-10-14.** The state was given as the **state on 2026-10-14**
(amended in step 3, after the reviewer ran the suite against the new
releases). That a landed item whose release the compat floor requires has
no probe any more was decided by Erik 2026-10-14. On that date EntropyEOS
was **1.0.0** on its `main`, not registered (EOS-0 waiting), with
`EntropyEOS = "1"`: EOS-1 and EOS-2 had landed in it; EOS-3 and EOS-4 had
not. TreeAMR was bounded by its minor series, `TreeAMR = "0.2"`, and from
2026-10-14 by `"0.2.1"`.

**2026-10-14.** The rule for retiring a probe (`PLAN.md`, "Upstream requirements": once an item
has landed and `[compat]` requires a release that contains it, its testset is deleted from
`test/prerequisite_tests.jl`) was decided by Erik 2026-10-14. The state column was amended the
same day.

**Step 5.** The TreeAMR entry was amended in step 5: step 5, which needed
AMR-1, is done on it.

**2026-10-15.** The SpacetimeMetrics entry (it requires Julia 1.11, below
the floor here, 1.13) was amended 2026-10-15.

**Step 8.** The compat floor became `TreeAMR = "0.2.2"` (amended in
step 8), the release AMR-7 to AMR-10 are in; since step 8 AMR-7 to AMR-10
are among the landed items without a probe. `test/upstream_behaviour_tests.jl`
was empty and removed, since step 8 took the bounded prolongation into its
refinement tests.

**Step 8.** The state column was amended in step 8.

**2026-10-21.** The state column was amended.

**2026-10-24.** The state column was amended.

**2026-10-26.** The state column was amended; it is stated as of this date.

**2026-11-06.** Until this amendment, TreeGH was to become a dependency in
step 10.

**2026-11-08.** Amended: TreeGH's `main` is 0.1.1, with `adm_vars` and
`ADMMetric`, and Adapt as a new direct dependency; before, the entry read
TreeGeneralizedHarmonic **0.1.0** on its `main`. GH-1 landed 2026-11-08.

**2026-11-08 (`PLAN.md`'s status).** EOS-1 and EOS-2 have landed in
EntropyEOS 1.0.0, EOS-8 in 1.1.0, EOS-9 in 2.0.0, EOS-11 in 2.1.0 and EOS-10
in 2.2.1; the package resolves 2.0.0 since 2026-10-24, 2.2.0 since 2026-10-26
and 2.2.1 since 2026-10-30 (still unregistered: EOS-0, EOS-3 and EOS-4
waiting). TreeAMR 0.2.1 is registered with AMR-1, AMR-2, AMR-3 and AMR-5,
and AMR-4 and AMR-6 in a different form, and 0.2.2 (the compat floor since
step 8) with AMR-7 to AMR-10 (AMR-11, added in step 8, waiting);
SpacetimeMetrics 1.7.0 with SM-1, whose workaround is gone. TreeGH's `main`
is 61bcb75, 0.1.1 (GH-0 and GH-1 landed, GH-1 as its PR 25; GH-3, GH-4 and
GH-6 waiting; GH-5 withdrawn; unregistered; amended 2026-11-08).

### TreeAMR

**Undated.** The requests AMR-1 to AMR-6 as written in `PLAN.md`, "Upstream requirements":

*AMR-1 — A state vector over several field sets (req, step 5).*

**What.** Implement the multi-set state functions TreeAMR's `CODE.md` already specifies
(CODE.md:1386–1395), plus segment accessors.

**API.** These are new methods; the single-set methods in `src/state.jl` are unchanged.
```julia
statelength(sets::Tuple{Vararg{FieldSet}}) -> Int              # Σᵢ statelength(sets[i])
staterange(sets, i::Integer) -> UnitRange{Int}                 # segment of set i
statevector(sets) -> AbstractVector                             # zeroed, on the sets' backend
stateview(u::AbstractVector, sets, i) -> AbstractVector         # view(u, staterange(sets, i))
statearray(u::AbstractVector, sets, i) -> AbstractArray         # statearray(stateview(u,sets,i), sets[i])
scatter!(sets, u::AbstractVector)
gather!(u::AbstractVector, sets)
```

**Semantics.** As now in `CODE.md`, "TreeAMR" (AMR-1).

**Upstream acceptance.**
- `scatter!`/`gather!` round-trip in D = 1, 2, 3 over this tuple, with each segment bit-identical
  to that set's own `statevector`:
  - a cell set (`G=3`, 10 vars);
  - three edge sets (`G=3`, 1 var);
  - a vertex set (`G=3`, 1 var);
  - a face set (`G=0`, 10 vars).
- Aliasing holds.
- Single-set reductions on `stateview` equal those on a separate vector.
- A mismatched forest, backend or eltype throws.
- The same round trip on a device backend, where CI has one.

**Downstream check.** The same round trip over TreeGRRMHD's own tuple, in D = 1, 2, 3.

*AMR-2 — Per-set boundary hooks in `regrid!` (wanted, step 6).*

**Why.** `regrid!` passes one `boundary` to every pair (regrid.jl:556). An
`AllVariables` hook is length-checked against each set's `nvars` (ghosts.jl:326), so one hook
cannot serve sets with different `nvars`.

**API.** `regrid!(forest, pairs; flags, buffer=0, boundary=nothing, transfer=true)` also accepts
the pair `fs => (schedule, hook)`. That hook is used for this set's pre-transfer `fill_ghosts!`
and overrides `boundary`. `fs => schedule` and `fs => nothing` are unchanged.

**Acceptable instead:** document as supported what regrid.jl/ghosts.jl:461 do today. With
`boundary = nothing`, the outward ghost regions keep what the caller put there, so a caller may
fill each set with its own hook immediately before `regrid!`.

**Workaround until then:** exactly that. The code is marked `AMR-2`.

**Upstream acceptance.** Regridding a cell set and a vertex set with different `nvars` and
different Dirichlet hooks in one call transfers exactly what two single-set calls transfer.

*AMR-3 — `adapt_to_initial_data!` over several sets (wanted, step 8).*

**API.**
```julia
adapt_to_initial_data!(pairs; initial, flag | flags, buffer=0, maxpasses=10, boundary=nothing)
    -> (schedules::Tuple, passes, converged)
```
- `pairs = (fs₁ => ops₁, …)`.
- `initial = (init₁, …)` gives one `AllVariables` or per-variable callback per set.
- `flags` receives the tuple of sets.
- Every set is refilled after every pass.

**Workaround until then:** adapt the matter set alone, then create the other sets on the final
forest and fill them with `fill_by_coordinates!`. This is marked `AMR-3`.

*AMR-4 — Limited conservative prolongation (strongly wanted, step 10).*

**API.**
```julia
Operators(; prolongation=3, restriction=2, family=Conservative, limiter=:minmod | :mc | :none)
```
`:none` is the default and stays bit-identical to today. `limiter` is only allowed with
`Conservative` and `prolongation = 3`.

**Semantics**, per variable, for each coarse cell `c` with average `ū_c` and face neighbours
`ū_{c±e_d}`:
- **Slopes:** `s_d = L(ū_{c+e_d} − ū_c, ū_c − ū_{c−e_d})`, with
  `L = minmod` or `MC(a, b) = minmod(2a, 2b, (a+b)/2)`.
- **Children:** `ū_c + φ Σ_d σ_d s_d / 4`, with `σ_d = ±1` by the child's side.
- **Scale:** `φ = min(1, (M − ū_c)/Σ_d|s_d|/4, (ū_c − m)/Σ_d|s_d|/4)`, where `M` and `m` are the
  max and min over `ū_c` and its `2D` face neighbours. `φ = 1` if every `s_d = 0`.
- Hence every child lies in `[m, M]`. No new extremum means positivity is preserved.
- The `2^D` children average exactly to `ū_c`.
- It is used both by ghost-fill prolongation and by the regrid transfer.
- `G ≥ 1`, as today for width 3.
- It is nonlinear, so it is not a `Stencil1D` tensor product. It needs its own kernel, chosen in
  `run_group!` (ghosts.jl:105–129) from the group's operators. Which way to carry the family
  there is TreeAMR's design call.

**Upstream acceptance.**
- Conservation: `total_mass` is conserved to round-off through ghost fill and regrid, on random
  data.
- Linear data are reproduced exactly (the limiter is inactive) in D = 1, 2, 3.
- No new extrema on step-function and random data.
- TreeHydro's entropy wave keeps an L1 rate ≥ 1.9 across a coarse-fine face.
- TreeHydro's step-9 shock crossing floors no cell with `:minmod`.

**Downstream check.** Still standing; in `PLAN.md`, AMR-4.

*AMR-5 — `workgroupsize` on `map_blocks!` (wanted, step 14).*

`map_blocks!(kernel!, fs, args...; closed=false, stored=false, workgroupsize=nothing)` passes
`workgroupsize` through to the KernelAbstractions launch; the default is unchanged. It is also on
TreeGH's list.

*AMR-6 — A device-capable boundary hook that reads the interior (later).*

This is for outflow conditions. Today the region form is CPU-only (ghosts.jl:347–356).

**Step 0.** The *(later)* item on MPI (M7), checkpoints (M9, now
TreeIOHDF5's) and point interpolation (M11) was amended in step 0.

**Step 0.** On AMR-2: TreeAMR 0.2.0 refuses `fs => (schedule, hook)` today, but by a field access
on the tuple inside `check_regrid_set!` (regrid.jl:827) — a `FieldError` at Julia 1.12 and later,
an `ErrorException` at 1.11 — rather than an `ArgumentError` saying which pair forms it takes.
Whichever way AMR-2 is answered, that refusal should be an `ArgumentError`.

**2026-10-14.** AMR-4 landed in another form, TreeAMR 0.2.1, commit
902042a (amended 2026-10-14). Until then the workaround was width 1 near
surfaces. AMR-6 landed in another form, TreeAMR 0.2.1, commit a983419
(amended 2026-10-14). Until then the workaround was Dirichlet from the
initial or analytic exterior, which is what the TOV milestones need
anyway.

**2026-10-14.** AMR-4 landed in another form, TreeAMR 0.2.1, commit 902042a: not a choice of
limiter but a property, `Operators(; prolongation=3, restriction=2, family=Conservative,
bounded=true)`, the semantics of the request with the slope fixed to MC (the rest of its
semantics in `CODE.md`, "TreeAMR"). Its commit message records what upstream assumes of this
package (quoted in `CODE.md`, "Refinement") — a design point steps 8 and 10 take up.
`test/upstream_behaviour_tests.jl` checks, in D = 1 and 2, that a surface (1 to 1e−13) refined
everywhere stays positive and within range and keeps its total, where the unbounded operator
overshoots; it moves to step 8's refinement tests. Until 2026-10-14 this was the AMR-4 probe
in `test/prerequisite_tests.jl`, retired when `TreeAMR = "0.2.1"` came to guarantee it.

AMR-6 landed in another form, TreeAMR 0.2.1, commit a983419: not a hook but an outflow face kind
(`CODE.md`, "TreeAMR"). Its probe, which checked that both outflow faces of a one-dimensional
forest copy their nearest cell into the ghosts, was retired on 2026-10-14 (`TreeAMR = "0.2.1"`
guarantees it).

**Step 5.** AMR-1's state vector over `evolved_sets(f)` and its check in
`test/scheme_tests.jl` were used from step 5.

**Step 6.** AMR-7 was added in step 6.

**Step 5.** AMR-1's downstream check was done in step 5: `test/scheme_tests.jl`, "AMR-1
downstream" (described in `CODE.md`, "TreeAMR").

**Step 6.** AMR-7 was added to `PLAN.md`, "Upstream requirements":

*AMR-7 — A device-capable boundary hook relative to the nearest owned point (later, step 14;
workaround).*

**Why.** A non-periodic case's potentials take Dirichlet data *in the field*: each ghost of `A_j`
and `Φ̃` holds the nearest evolved point's value plus the case's difference between the two
points, so that `𝔅 = curl A` in the ghosts is the case's while the level of `A` follows the
evolution. A Dirichlet `A` itself is wrong under any tableau whose explicit and implicit abscissae
differ (`CODE.md`, "Boundaries"). That hook reads the block's interior, which only TreeAMR's region
form allows, and the region form is CPU-only (ghosts.jl:893–911). AMR-6 as landed (outflow faces)
copies the nearest owned value, which is this with a zero difference.

**API.** A per-cell form that also receives the nearest owned point:
```julia
CellBoundary(Relative((xg, xr, v, δ) -> Δ))        # ghost = value at r + Δ
CellBoundary(Relative(AllVariables((xg, xr, δ) -> Δs)))
```
with `r` the ghost's index clamped, along each outward direction, to the owned range
`G+1 … G+N` (the shared high plane of a vertex-like dimension counting as a ghost).

**Semantics.** Run as a kernel on every backend, in the same phase as today's hooks (after
the copies and restrictions, before the prolongations), so the value read at `r` is the
block's own owned value.

**Upstream acceptance.** On a CPU it writes bit for bit what the equivalent region-form hook
writes; on a device, a linear field with a constant-difference condition is reproduced in the
ghosts in D = 1, 2, 3, on every centering.

**Workaround until then:** the region form, `DirichletAt` for the potentials in
`src/driver.jl`, marked `AMR-7`. Every non-periodic case therefore runs on the CPU only.

**2026-10-15.** AMR-10 was decided by Erik 2026-10-15.

**2026-10-15, step 8.** AMR-7 gained a third requirement, decided by Erik 2026-10-15 ("an outer
boundary condition needs to be applied always"):
- **The complete condition, in every fill TreeAMR makes.** A relative hook must be able to express
  the whole condition — every point beyond an outer face, the wall's rows next to another block
  included — in every fill TreeAMR makes, `fill_ghosts!`, `regrid!`'s pre-transfer fill and
  `adapt_to_initial_data!`'s, so that nothing TreeAMR reads (a prolongation's coarse source next
  to a wall, a transfer) holds a partial value. The rows next to a coarser neighbour need its
  prolongated ghosts, so either the hook runs (again) after the prolongations for those points, or
  TreeAMR runs a second boundary phase after them; and a prolongation that reads a coarse corner
  row must read the completed one, which today takes a second exchange. **Acceptance:** with a
  relative hook, one `fill_ghosts!` is the condition's fixed point — a second fill changes nothing
  bit for bit — on a two-level mesh whose coarse-fine face meets an outer face, in D = 2, 3;
  `regrid!` with the hook transfers what a complete fill would give.

**2026-10-15, step 8.** AMR-8 was added in step 8, decided by Erik 2026-10-15, "option 1":

*AMR-8 — Symmetric walls: the owned low wall filled like the high one (wanted; workaround).*

**Why.** TreeAMR's ownership is half-open: along a vertex-like dimension a block owns
`G+1 … G+N`, so at a non-periodic face the domain's high wall plane `G+N+1` belongs to nobody and
is filled like a ghost — by the hook at an outer face, by the derivation at a reflecting one, by
the clamp at an outflow one — while the low wall `G+1` is owned and evolved (TreeAMR's `CODE.md`,
"Centerings", documents and accepts this). So `𝔅` on a box's high boundary face is the case's and
on its low one the evolution's, and a closed box is not mirror-symmetric: 1.3e−4 in a 16³ blast
after three steps (measured in step 7), 1.0e−1 of a variable's scale at `t = 1/4` (step 8). The
same asymmetry is in every vertex-like set of every application with walls.

**API.** An opt-in per field set, since it writes owned points:
```julia
FieldSet{T}(forest, nvars; G, centering, walls = :half_open)   # today's behaviour, the default
FieldSet{T}(forest, nvars; G, centering, walls = :symmetric)
```

**Semantics.** Under `walls = :symmetric`, `fill_ghosts!` also fills the owned low wall `G+1` of
every vertex-like dimension at a non-periodic low face by the same rule as the high wall at the
high face of that kind:
- an **outer** face: the hook, which is handed the wall plane as the first plane of its region
  (the low region becomes `1 … G+1`);
- a **reflecting** face: the derivation — `0` for a variable odd in that dimension, the one-sided
  fold for an even one — exactly as the high wall is derived;
- an **outflow** face: the clamp, from the nearest evolved point `G+2`.

Every ghost beyond the low wall, and every point the hook or the schedule fills relative to the
nearest evolved point, then reads `G+2` along that dimension, as the high side reads `G+N`. On every
backend, in the phases the high wall is filled in today, and across ranks. The state vector is
unchanged — the wall stays an owned point; an application that wants the filled wall in its state
gathers it after the fill. A cell-like dimension has no wall and is unaffected; `:half_open` stays
bit-identical to today.

**Upstream acceptance.** In D = 1, 2, 3 and every centering with a vertex-like dimension, on a
forest closed in every dimension by each face kind, under a reflection of the domain: data
symmetric under the reflection fill symmetrically, bit for bit, both walls included; a linear field
with a constant-difference hook is reproduced on both walls; `:half_open` writes exactly what it
writes today; a device fill matches the CPU's.

**Downstream check.** `test/prerequisite_tests.jl`, "AMR-8", detects the keyword
(`hasmethod(FieldSet{Float64}, Tuple{Forest,Int}, (:walls,))`), never a version; skipped with an
`@info` until then. When it lands, the potentials are built with `walls = :symmetric`, `low_walls!`
is deleted, and the closed-box tests of `test/multid_cases_tests.jl` stay bit for bit.

**Relation to AMR-7.** AMR-7 asks for a device-capable relative hook; with AMR-8 the evolved range
it clamps into starts at `G+2` along a vertex-like dimension at a low outer face, and the low wall
is one of the points it fills. Both are needed to delete the potentials' CPU-only region form and
this package's two kernels (`low_walls!`, `outer_points!`).

**Workaround until then** (`low_walls!` and `outer_points!` in `src/driver.jl`, marked `AMR-8`):
`map_blocks!` kernels, so on every backend, reading only in-block points. The first prescribes the
owned low walls of `A₁, A₂, A₃, Φ̃` at every outer face by the mirror of the hook's high-wall rule
(the nearest evolved point plus the case's difference) — in the state vector, inside the limiter
passed as both `stage_limiter` and `step_limiter` (beside the reset), so it persists, and in each
potential's working array before every ghost fill, so every fill sees it at its own time. The
second refills every point beyond an outer face after the fill (AMR-7, above). Only faces with this
package's Dirichlet hooks; reflecting and outflow faces are TreeAMR's machinery and untouched.
The face kinds come from exported API: the block's origin against the extent, and the forest's
`periodic`, `reflecting` and `outflow` properties (TreeAMR's documented `getproperty`); the
unexported `facekind` is not needed.

**2026-10-15, step 8.** AMR-10 was added in step 8, decided by Erik 2026-10-15:

*AMR-10 — Bounded PointValue interpolation (req: step 8's ghost-fill re-measurement; steps 10,
12).*

**Why.** The matter's ghost fill will prolongate the *primitives* (`ρ`, `s`, `Yₑ`, `uⁱ = Wvⁱ`,
`Eⁱ`) and build each conserved ghost by `prim2con` with the ghost's own `B` — the curl of the
prolongated `A` — so that a ghost is an admissible state wherever its neighbours are, and its
electromagnetic energy and momentum are its own field's (step 8 measured, with conserved ghosts
bounded, 33 repaired ghost recoveries on a magnetized atmosphere across a coarse-fine face against
8 on the uniform mesh, the coarse field's energy landing in the fluid's). Primitives are point
values, so the fill needs a PointValue prolongation, and one that creates no new extremum — `ρ`
must stay positive at a stellar surface (ten orders of magnitude), `s` and `Yₑ` in their windows.
TreeAMR's `bounded = true` is the Conservative family's at order 3 or 1 only. The regrid stays
conservative (Conservative width 3, bounded, on the conserved variables), so totals are kept.

**API.**
```julia
Operators(; prolongation=w, restriction=r, family=PointValue, bounded=true)
```
accepted for every even `w ≥ 2` (`w = 2`, linear, is bounded already, and is accepted so that an
application switches widths without touching the flag, as Conservative order 1 is today), on every
centering — cell, vertex, face, edge — in `fill_ghosts!` (the prolongation, and the restriction
where a PointValue restriction interpolates rather than injects, as along a cell-like dimension,
whose fine centres do not coincide with the coarse one) and in `regrid!`'s transfer.
`bounded = false` stays bit-identical to today.

**Semantics**, per variable. Every interpolated value lies within `[m, M]`, the minimum and maximum
of the `2^D` source points that enclose the target point (the linear stencil's points), so monotone
data stay monotone and no new extremum is created. A suggested construction (TreeAMR's call):
`u = u_lin + φ (u_w − u_lin)`, with `u_lin` the multilinear interpolant from those `2^D` points,
`u_w` the width-`w` Lagrange interpolant, and `φ ∈ [0, 1]` the largest value keeping `u` in
`[m, M]` (`φ = 1` where `u_w` is in range already). Exact on linear data; of order `w` on smooth
data wherever no limiting is active; device-capable; across reflecting and outflow faces with their
parity and clamp rules.

**Upstream acceptance.** `bounded = false` bit-identical; linear data reproduced exactly in
D = 1, 2, 3 on every centering; no new extrema on step-function and random data; order `w` on a
smooth function with no extremum in the stencil; positivity: a profile dropping from 1 to 1e−13
stays positive, in the ghost fill and in the regrid.

**Downstream check.** `test/prerequisite_tests.jl`, "AMR-10": that
`Operators(; prolongation=4, restriction=4, family=PointValue, bounded=true)` is accepted (detected
by feature, never by a version; skipped with an `@info` until then — today it throws an
`ArgumentError`). When it lands: the ghost fill of the primitives on the atmosphere tube across a
coarse-fine face repairs no ghost — magnetized, against step 8's 33 with conserved ghosts — and step
8's ghost-fill table in `CODE.md` is re-measured with it.

**Workaround:** none, and none to keep. Until it lands the conserved, bounded ghost fill of step 8
runs (it is measured, and conservation does not depend on the ghost fill); it is what runs, not a
workaround, and is replaced when AMR-10 lands. The restructuring that follows is `CODE.md`'s
("Refinement", the ghost fill of the matter).

**Step 8.** AMR-8, AMR-9, AMR-10 and AMR-11 were added in step 8. AMR-4's
check (a surface refined everywhere stays positive and within range and
keeps its total) moved to `test/refinement_tests.jl` from
`test/upstream_behaviour_tests.jl` in step 8, and the matter's ghost fill
and regrid use the bounded operator since step 8. AMR-7's region-form
workaround and `outer_points!` are gone since step 8. AMR-8's workaround,
until it landed, was `low_walls!` and `outer_points!`, this package's
kernels ("Boundaries"); gone since step 8. AMR-9 had no workaround: until
it landed the state's flux across levels held to the potentials'
interpolation error; the potentials take it since step 8. AMR-10 had no
workaround: the conserved, bounded ghost fill ran until it landed; when it
landed, the matter's ghost fill took the primitives with it in step 8, at
width 2.

**Step 8.** AMR-4's check moved in step 8 to `test/refinement_tests.jl`; the matter's ghost fill
and regrid use the bounded operator since, and `test/upstream_behaviour_tests.jl`, then empty, is
gone until a landed item needs it again.

AMR-7 was amended in step 8 with two more things a relative hook must do, found when AMR-8's
workaround made a closed box mirror-symmetric bit for bit:
- **A point beyond an outer face alongside another block** — the wall's row in a corner region
  whose direction crosses a block face along one dimension and the outer face along another — must
  take `r` from the neighbour's data along the first (the block's own ghost there), not clamp it
  to the block's last owned column. The region form clamps along every outward direction, so the
  point two blocks share (the wall's row on the shared plane) came out block-dependent, which broke
  the mirror symmetry of a closed box (7.1e−6 in a 32² explosion) and made two blocks compute
  different fluxes at their common face next to the wall. The neighbour's ghost is filled by then
  only for a same-level or finer neighbour (copies and restrictions run before the hooks, the
  prolongations after), so the per-cell form should run, for those points, after the
  prolongations.
- **Under AMR-8** the evolved range along a vertex-like dimension at a low outer face is
  `G+2 … G+N`: the low wall `G+1` is then a boundary point, clamped like the high one.

**Step 8.** AMR-7's workaround, extended in step 8: after each fill of a potential, `outer_points!`
(a `map_blocks!` kernel over the stored points) refills every point beyond an outer face from the
nearest evolved point, taken across a block face along the dimensions it does not leave the domain
through, now that every ghost is filled; on a multilevel mesh a hookless `fill_ghosts!` (which
leaves those points as written) and `outer_points!` follow once more, after which a further fill
changes nothing. `regrid_state!` fills completely and gives `regrid!` no hook for the potentials;
`adapt_case!`'s criterion fills completely before reading. Measured before this: a regrid's
transfer read values 1.8e−10 off (of 1.2e−2) next to a wall and a block face; the single-pass
fill's ghosts were off by as much. Nothing TreeAMR reads is partial now; the cost is a second fill
of each potential on a multilevel mesh. Marked `AMR-7`/`AMR-8` in `src/driver.jl`.

**Step 8.** AMR-9 was added in step 8:

*AMR-9 — A flux-conservative transfer of edge sets (wanted).*

**Why.** With `B = curl A` the magnetic flux through a face is the circulation of `A` around it.
Across a coarse-fine face the coarse circulation equals the sum of the fine ones only if the coarse
`A_j` on an edge equals the mean of the two fine edges along it — which a transfer that is
conservative along the edge's own (cell-like) direction guarantees and the PointValue family, which
interpolates there, does not. Measured in step 8: on the oblique Alfvén wave over a two-level 2D
box, `Σ hᴰ 𝔅ᶻ = ∮ A · dl` around the periodic box drifts 5.8e−4 of its scale in an eighth of a
period (5.1e−17 on the uniform mesh), whatever the EMF correction does; at width 2, twice that.
The rate — the EMF correction's business — is consistent to rounding; the state is not.
`∇·𝔅 = 0` per cell is unaffected (structural).

**API.** The `Conservative` family accepted on a set with vertex-like dimensions, meaning
conservative along its cell-like dimensions and point-value along its vertex-like ones, with the
vertex-like width its own keyword:
```julia
Operators(; prolongation=3, restriction=2, family=Conservative, vertex_prolongation=4)
```
(`vertex_prolongation` even, `prolongation` odd as today; `bounded` refused here.) Or a family of
its own; the shape is TreeAMR's call.

**Semantics.** Tensor product, per dimension by its stagger: along a cell-like dimension the
conservative 1D operators (restriction the two-cell mean, prolongation of order `p` whose two
children average to the parent); along a vertex-like one injection and the width-`w` Lagrange
prolongation of PointValue. For an edge set, the line integral of `A` along every coarse edge
then equals the sum along its fine edges after a restriction, a prolongation, a ghost fill and a
regrid, so the circulation around every coarse face equals the sum around its fine faces; for a
face set, the flux through a coarse face is its fine faces' sum.

**Upstream acceptance.** On random edge data in D = 2, 3, after a ghost fill and after a regrid,
the circulation around every coarse face on a coarse-fine face, and around every coarse face a
regrid coarsened or refined, equals the sum of its fine faces' to rounding; linear data are
reproduced; a cell-centered set's `Conservative` operators are unchanged bit for bit.

**Downstream check.** `test/prerequisite_tests.jl`, "AMR-9": an edge set's `GhostSchedule`
accepts the operators above (detected by feature, never by a version; skipped with an `@info`
until then). When it lands, the potentials take it, and `test/refinement_tests.jl` asserts the
flux through the plane to rounding across levels.

**Workaround:** none needed for correctness — the flux across levels holds to the potentials'
interpolation error, and every rate to rounding.

**Step 8.** AMR-11 was added in step 8; its request stands in `PLAN.md`.

**Step 8.** AMR-7 to AMR-10 landed in TreeAMR 0.2.2, commit 624c99a (the items amended in step 8):

AMR-7 as asked: `CellBoundary(Relative(f))` and `CellBoundary(Relative(AllVariables(f)))` (their
semantics in `CODE.md`, "TreeAMR" and "Boundaries"), so that one fill — `fill_ghosts!`,
`regrid!`'s and `adapt_to_initial_data!`'s — is the condition's fixed point. `TreeAMR = "0.2.2"`
guarantees it; the probe is retired and `Relative` joins step 0's name list. Used since step 8
(`boundary_at`); the region-form hook and `outer_points!` are deleted (the workaround dropped in
step 8), and `test/refinement_tests.jl` checks the fixed point in one fill (from `NaN`-prefilled
arrays too) and that `regrid!` with the hooks transfers what a complete fill gives, bit for bit.

AMR-8 as asked: `FieldSet(…; walls = :symmetric)` has the fill write the owned low wall plane of a
vertex-like dimension at a low domain face by the high wall's rule (hook, mirrored derivation or
clamp from `G+2`). Upstream records an open item: on a refined mesh a vertex-like set is not
mirror-symmetric under either kind of wall (4.7e−5 in its test), half-open ownership at
coarse-fine faces. `TreeAMR = "0.2.2"` guarantees it; the probe is retired. Used since step 8 for
`A_j` and `Φ̃`; `low_walls!` is deleted (the workaround dropped in step 8). Its downstream check
holds: the closed-box tests of `test/multid_cases_tests.jl` stay mirror-symmetric bit for bit; a
refined closed box is not (1.1e−2, recorded in `test/refinement_tests.jl`). The owned wall the
fill writes is copied into the state by the reset (`walls_to_state!`) and has zero tendency.

AMR-9 as `Operators(; prolongation=p, restriction=2, family=Conservative, vertex_prolongation=w)`:
conservative along the cell-like dimensions, point values of even width `w` along the vertex-like
ones; a coarse edge's line integral is the sum of its fine edges' after every fill and regrid
(the rest of its semantics in `CODE.md`, "TreeAMR"). `TreeAMR = "0.2.2"` guarantees it; the probe
is retired. The potentials take it since step 8, and `test/refinement_tests.jl` asserts the flux
through the plane to rounding across levels and through a regrid, and every coarse face's flux the
sum of its fine faces' in 3D (but at an outer face: AMR-11).

AMR-10 as asked (its semantics in `CODE.md`, "TreeAMR"). `TreeAMR = "0.2.2"` guarantees it; the
probe is retired. The matter's ghost fill takes the primitives with it since step 8, at width 2
(wider ones clamp `ρ` and `s` independently: `CODE.md`, "Refinement"); its downstream check holds
— no ghost of the magnetized tube across a coarse-fine face is inadmissible (against 33 repaired
recoveries with conserved ghosts), `test/refinement_cases_tests.jl`.

**2026-10-23.** AMR-4's bounded operator is used by the matter's ghost
fill on the primitives `W` since 2026-10-23 (decided by Erik). Amended
2026-10-23 in AMR-10: the fill takes the
bounded conservative operator, AMR-4, centred on the parent, instead of
the PointValue interpolation of AMR-10 — decided by Erik; "Refinement".

**2026-10-23.** AMR-10 amended, decided by Erik: the fill now takes the bounded conservative
operator centred on the parent (AMR-4) on `W`; the PointValue operator stays the suite's width-4
control (`CODE.md`, "Refinement").

### TreeGeneralizedHarmonic

**Undated.** GH-4's verbs were `gh_*` names, and "the coupled driver lives
here". The paragraph on TreeGH's CFL factor under an SSP3 explicit tableau
was amended with `PLAN.md`.

**Undated.** GH-0 as first written: the step-8 branches, with tip `claude/step-8f-matrix` (47
commits ahead), widen `GHProblem`, `gh_rhs_kernel!` and `evolve!`:
- `accounting`, `target`, `fits`, `t_target`;
- the stage limiter;
- an `INT = :fitted` interior.

**Undated.** GH-5 as requested (wanted, step 10), kept for TreeGH's reference:

*GH-5 — Ghost width `≥ q/2 + 1`.*

- Relax `all(g -> g == q ÷ 2 + 1, U.G)` (evolution.jl:540–546) to `≥`.
- Add a `G` keyword to `evolve!` (driver.jl:243).

The survey found every kernel offsets by `Val(G)` and every stencil reach comes from the weights,
so this is a check change plus a test. It lets a 4th-order vertex→cell interpolation reach every
stored cell of a `G = 3` matter set.

**2026-09-25.** GH-2's Julia floor was settled on 2026-09-25: every Tree* package's floor is 1.11,
so TreeGH's pin to 1.11 no longer constrains TreeGRRMHD, and TreeGH is an ordinary dependency
rather than a weak one behind `TreeGRRMHDTreeGHExt` (CODE.md "Julia version").

**2026-10-05.** TreeGH's `W` became the sixth `Val` of its kernel (GH-3's `Val(HASMATTER)` would
be the seventh).

**Step 0.** GH-0 landed by inspection: TreeGH's `main` (fbfd416, 2026-10-09) has
`accounting`, `t_target`, the `:fitted` interior and the stage limiter, and its `CLAUDE.md` records
steps 8a–8f as done. The suite cannot check it until TreeGH is a dependency.

GH-2 was amended in step 0: TreeGH 0.1.0 is on the registered TreeAMR 0.2, which settles the part
about TreeAMR features being in a released TreeAMR. Its workaround was amended in step 0: it was
"CI adds TreeGH and KorzynskiSpin".

**2026-10-15.** GH-2 amended: TreeGRRMHD's floor is now 1.13, above TreeGH's 1.11, which still
does not constrain it; nothing is asked of TreeGH.

**2026-10-16.** GH-2 amended: every Tree* package's floor is now 1.13 (Erik's decision).

**2026-11-06.** The section was amended from the decisions of 2026-11-05
in The coupling contract: every item is step 11's now, GH-1 and GH-3 are
adapters supplied by the producer, and GH-5 is withdrawn (2026-11-06).

GH-0 amended: it was step 10's, which no longer reads TreeGH.

GH-2 amended: it was step 10's. TreeGH is no longer a dependency of this package (`CODE.md`, "The
coupling contract").

GH-4 amended: the verbs are TreeGH's own under names this package shares, decided by Erik
2026-11-05, and a script calls them, not a driver here (`CODE.md`, "The coupling contract"). They
were `gh_fill_ghosts!`, `gh_rhs_prepared!`, `gh_rhs!`, `gh_max_speed`, `gh_step_limit!` and
`gh_stage_limit!`, exported, and "the coupled driver lives here". The names module-qualified were
decided; the signatures stay proposed. The bullet on `CODE.md`'s step-11 sketch ("Two scripts")
was added 2026-11-06.

GH-1 rewritten 2026-11-06 (see 2026-11-08 for the earlier forms); GH-3 amended 2026-11-06 (see
2026-11-08).

GH-5 withdrawn 2026-11-06: how the metric reaches the outermost ghost layer of a `G = 3` cell set
was then TreeGH's own choice behind GH-1.

**2026-11-08.** Amended, the contract redesigned by Erik (`CODE.md`, "The coupling contract"):
both adapters answer at TreeGH's owned vertices, in 3+1 variables, and every conversion between
centerings is this package's.

GH-1 rewritten: the metric crosses at TreeGH's owned vertices. From 2026-11-06 to 2026-11-08 the
request was a `FourMetric` adapter answering `g_ab`, alone or with `∂_c g_ab`, at every point of
any of this package's sets, ghosts included, TreeGH interpolating, 4th order in the outermost ghost layer of a `G = 3` cell set; before that, `NC`,
`unpack_sym4` and `pack_sym10` under public names, `U.work`'s layout and
`vertex_state_at(work, idx, b, inv_h, Val(q)) -> (h, Π, ∂h)`, so that this package could decode
and interpolate (the table's summary: "was `g_ab` at this package's points, before that step 10's
public vertex metric"). TreeGH's layout stays private.

The request as rewritten: **What.** `(α, β^i, γ_ij, K_ij)` at TreeGH's owned vertices, from an
adapter built from its prepared state and bound to this package's vertex set `Mv` (`CODE.md`,
"Metric at cells, faces and edges").

GH-1 landed 2026-11-08, TreeGH's PR 25, 61bcb75, as `ADMMetric` (its API and semantics in
`CODE.md`, "TreeGeneralizedHarmonic" and "The coupling contract"; TreeGH's `CODE.md`, "The coupling
contract with TreeGRRMHD"). **Upstream acceptance** (TreeGH's, in its PR 25: 17 claims in
`admvars_tests.jl`).
- On a two-level Kerr-Schild mesh at `N = 10`, the adapter is `adm_vars` bit for bit at every
  owned vertex, after its own scatter and fill.
- Inside a consumer's `map_blocks!` over a `G = 4` vertex set the metric is the host's bit for
  bit, and `K` agrees to 5.6 eps of its largest value.
- Refusals; no allocation; `Adapt` reaches both arrays. Metal at `Float32` agrees with the CPU
  to 0.76 eps (`K` 36 eps). Not yet run: an H200.

**Downstream check** (step 10 for this package's side, step 11 for TreeGH's). Step 10's sampled
ADM adapter of a SpacetimeMetrics metric, in `ADMMetric`'s form, is the reference: through `Mv`
and this package's interpolation, `g` and `∂g` converge at 4th order against `dmetric` at every
point of every centering, the outermost ghost layer of a `G = 3` cell set included. (The step-11
half stands in `PLAN.md`, GH-1.)

GH-3 amended, the contract redesigned by Erik (`CODE.md`, "The coupling contract"): the adapter
returns `(ρ, S_i, S_ij)` and TreeGH assembles `T_ab` with its own metric. From 2026-11-06 to
2026-11-08 it returned the covariant `T_ab`; before that the input was a vertex-centered `Tsrc`
field set — 10 components, `G = 0`, packed as in `NC` — that this package wrote before each
right-hand side and that TreeGH's rebuilds carried. The kernel term, the monitors and the
acceptance are otherwise unchanged.

GH-5 amended 2026-11-08: since then how the metric reaches the outermost ghost layer is this
package's: the interpolation reads its own vertex set `Mv`, which has `G = 4`, and TreeGH's ghost
width does not enter the contract; not needed since then.

**Undated, step 11.** The GH CFL factor under SSP3(4,3,3)'s explicit part is measured in step 11.
It replaces the former upstream item.

### EntropyEOS

**Undated.** EOS-1 was marked (decided). EOS-2 was amended with
`PLAN.md`, which also moved this item from rung 4 to step 5: the synthetic
table cannot host the flat-space tests.

**Undated.** The requests EOS-1 and EOS-2 as written in `PLAN.md`, "Upstream requirements":

*EOS-1 — An abstract EOS interface (req, step 5).*

**API.**
```julia
abstract type AbstractEOS{T<:AbstractFloat} end
struct EOSTableView{T,A} <: AbstractEOS{T} … end        # unchanged fields
# Required of every subtype — exactly what con2prim, con2prim_safe, prim2con and default_policy use:
evaluate(eos, ρ★::Real, s::Real, yₑ::Real, u_guess::Real) -> EOSPoint     # generic in the number
                              # type (ForwardDiff duals); allocation-free; never throws
srange(eos, ρ★, yₑ) -> SRange             # physical entropy window; finite
srange_extended(eos, ρ★, yₑ) -> SRange    # ⊇ srange; finite; U strictly increasing in s on it
logρ_bounds(eos) -> (x_lo, x_hi)           # log10 ρ★; finite
yₑ_bounds(eos) -> (y_lo, y_hi)             # finite
κ(eos) -> T                                 # exported (today unexported, on EOSTable only)
Base.eltype(eos); narrow(eos, S)           # instances isbits; Adapt is the identity for isbits
```

**Changes.**
- Retype every method now typed `eos::EOSTableView{S}` to `AbstractEOS{S}`. That is `evaluate`
  (including the promoting method, adapter_eval.jl:714), `srange`, `srange_extended`, and the
  methods in con2prim.jl, prim2con.jl and state_policy.jl.
- Replace the field reads `eos.x_lo`, `eos.x_hi`, `eos.y_lo`, `eos.y_hi` in state_policy.jl
  (lines 102, 106, 122–123, 150–153, 177, 231–236, 399, 502) with the accessors.

**Required of every `EOSPoint`.**
- `p = ρ²U_ρ`, `h = 1 + U + p/ρ`, `cs² = (2ρU_ρ + ρ²U_ρρ)/h`, `T̂ = U_s`.
- `U_ρs` is consistent with the others. con2prim's Jacobian uses `cs²`, `U_s` and `U_ρs`
  (con2prim.jl:192–214).
- An analytic EOS may return `u_solved = NaN`, `iters = 0` and `flags = 0`.

**Upstream acceptance.**
- The existing suite is bit-identical.
- A test-only toy subtype runs `con2prim_safe` end to end.

*EOS-2 — Analytic Γ-law and hybrid EOSes (req, step 5).*

**API.** Every parameter is a required keyword.
```julia
IdealGasEOS{T}(; Γ, K_ref, s_ref, s_window::NTuple{2}, ρ_bounds::NTuple{2},
               yₑ_bounds::NTuple{2}) <: AbstractEOS{T}
HybridEOS{T,N}(; ρ_breaks::NTuple{N-1}, K₀, Γs::NTuple{N}, Γ_th, K_th_ref, s_ref,
               s_window, ρ_bounds, yₑ_bounds) <: AbstractEOS{T}
polytropic_entropy(eos::IdealGasEOS, K) -> s       # the s with K(s) = K
```

**Definitions.** Units are the caller's (any consistent density unit). `κ = 1`, so `ρ★ = ρ`;
`yₑ` enters only the bounds.
- **`IdealGasEOS`:**
  - `K(s) = K_ref·exp((Γ−1)(s − s_ref))`
  - `U = K(s) ρ^{Γ−1}/(Γ−1)`, `p = K(s) ρ^Γ`
  - `T̂ = U_s = (Γ−1)U = p/ρ`, `h = 1 + ΓU`, `cs² = Γp/(ρh)`

  The polytrope `p = Kρ^Γ` is this EOS at the fixed entropy `polytropic_entropy(eos, K)`. A
  purely barotropic EOS (`U_s ≡ 0`) cannot be offered: the (s, w) Newton's Jacobian is singular
  there (con2prim.jl:213, 236).
- **`HybridEOS`:**
  - `U = U_cold(ρ) + K_th(s) ρ^{Γ_th−1}/(Γ_th−1)`.
  - `U_cold` is the piecewise polytrope with `p_cold` and `ε_cold` continuous, `ε_cold(0) = 0`,
    and the standard integration constants.
  - `K_th(s) = K_th_ref·exp((Γ_th−1)(s − s_ref))`, so `p = p_cold + (Γ_th−1)ρU_th` and
    `T̂ = (Γ_th−1)U_th > 0` on the window.
- `s_window` is finite with `s_min > 0`; `ρ_bounds` is finite (it sets the default `ρ_ceiling`).
- Everything is `isbits`.

**Upstream acceptance.**
- `EOSPoint` identities against ForwardDiff derivatives of `U`, to round-off.
- `con2prim_safe ∘ prim2con` is the identity to 1e−12 relative on random states: `ρ` across the
  bounds, `s` across the window, `W ≤ 10`, `B² = 0` and `≠ 0`. Checked in Float64, with a
  Float32 smoke run.
- `default_policy(eos, ρ_atm)` works, and its repairs are invertible.
- `isbits`, allocation-free and `@inferred`.
- The gated GPU test runs `con2prim` on both EOSes.

**Downstream check.** The step-1 3+1 round trip on `IdealGasEOS` at Γ = 4/3, 5/3 and 2.

**Step 0.** EOS-0 was amended in step 0: it used to say "CI adds it by
URL". EOS-1's difference from the request (`κ` stays unexported, called
qualified) was amended in step 0. That the TOV polytrope `K = 100, Γ = 2`
is `IdealGasEOS` at `polytropic_entropy(eos, 100)` was checked in step 0.

**Step 0.** EOS-0 was amended in step 0; until then it read 0.1.x, and a `Pkg.add(url=…)`
before `instantiate`.

EOS-1 landed in EntropyEOS 1.0.0 (recorded in step 0), except that `κ` stays unexported and is
documented as `EntropyEOS.κ(eos)`; this package calls it qualified (`CODE.md`, "EntropyEOS").

EOS-2 landed in EntropyEOS 1.0.0 (recorded in step 0). Its suite states the round trip with the
errors divided by EntropyEOS's conditioning factors rather than as a raw 1e−12 (the factors, the
bounds and step 0's measurements are in `CODE.md`, "Primitive recovery"). Step 0's prerequisite
test ran the hydrodynamic-mode call on both EOSes.

**Step 1.** EOS-2's downstream check was done in step 1: `test/recovery_tests.jl`.

**2026-10-14.** EntropyEOS 1.0.0 with `EntropyEOS = "1"` (see "Upstream
prerequisites" above).

**2026-10-14.** The EOS-1 probe was retired on 2026-10-14 (`EntropyEOS = "1"` guarantees it); its
names joined the prerequisite test's EntropyEOS name list. The EOS-2 probe was retired on
2026-10-14 (`EntropyEOS = "1"` guarantees it): `test/recovery_tests.jl` makes the same call
through the 3+1 layer on the Γ-law gas, and `HybridEOS` is next exercised here by the step that
first uses it.

**2026-10-16.** EOS-5 was added by the front investigation as "no policy repair creates rest
mass".

**2026-10-17.** EOS-5 was **withdrawn 2026-10-17** by Erik pending the sweep in [Floors,
the atmosphere and the magnetosphere](#floors-the-atmosphere-and-the-magnetosphere), "Investigation (2026-10-16)". What was kept, as the notes of
`PLAN.md`'s

*EOS-5 — Withdrawn (2026-10-17): notes on two repairs at the edge of the valid range:*

The framing was wrong: EntropyEOS's design is a well-defined range of allowed thermodynamic
states, inside which `con2prim`/`prim2con` always succeed, with every state outside projected back
into it — and very low densities must be raised back to the atmosphere, so `cons.D ≤ in.D`
contradicts the design (and the proposal itself broke it for `D_in < ρ_atm`). The defect the
investigation found is narrower. What remains are two observations, for Erik to judge on their own
merits, with the criterion consistent with the valid-range design, **`D_out ≤ max(D_in, D_atm)`**
(`D_atm` the atmosphere state's `D`, `ρ_atm` at rest):

- **(i) A state projected to the atmosphere density gets the atmosphere's velocity (rest).** The
  `S_FLOORED` branch of `con2prim_safe`'s step 4 (τ below the coldest state at the momentum) raises
  `ρ₁ = clamp(D/cosh w₁, ρ_atm, ρ_ceiling)` to `ρ_atm` but keeps the near-light-like rapidity `w₁`
  of the coldest-state solve, writing `(ρ_atm, W ≈ |S|/ρ_atm)` — up to the cap, `W = 100`, and
  `D_out = ρ_atm W` up to 100 `ρ_atm` — at the edge of the valid range rather than a natural
  projection. Writing the atmosphere state there (`policy_atmosphere`, at rest) satisfies
  `D_out ≤ max(D_in, D_atm)`; the released branch does not. Measured as variant B (both the
  in-process emulation and the reset-side form; the table in `CODE.md`).
- **(ii) A hot near-vacuum state goes to the entropy ceiling with `D` kept, not to `ρ_ceiling`.**
  The step-4 test "τ above the hottest state" calls `pol_collapse`, which writes `ρ = ρ_ceiling`
  (10⁶ for the shock gas) unless `collapse_to_atmosphere` — for a cell at `D ≈ 2 ρ_atm`, which is no
  collapse; every blow-up of the tube was this branch. The converged path's projection would have
  written `s_max` with `D` kept, which satisfies the criterion; the collapse write does not.
  Measured 2026-10-17: with `atm_trigger = 3` the branch never fires; with (i) alone (EOS-6) it
  still does — the magnetized ideal atmosphere and the cut-off tube at `Bʸ = 10⁻²` blow up in 48 of
  48 runs — and either this note or the existing option `collapse_to_atmosphere = true` closes it
  (both measured; the option needs no upstream change).

(i) was written out as EOS-6 below, now superseded (EOS-8); (ii) stays a note — the existing
`collapse_to_atmosphere` does its job in the cases that set it, and the entropy ceiling instead
remains Erik's call.

EOS-6 was added 2026-10-17, the front investigation's sweep, variant B; a candidate for Erik, not
decided:

*EOS-6 — A density raised to the atmosphere's gets the atmosphere's velocity.*

**Why.** `con2prim_safe`'s step 4, `S_FLOORED` branch, raises `ρ₁ = D/cosh w₁` to `ρ_atm` and
keeps the coldest-state rapidity `w₁`, writing `(ρ_atm, W ≈ |S|/ρ_atm)` at the edge of the valid
range; the atmosphere tube's `W` spikes, its extra rest mass and the discrete switch that makes it
chaotic are this write ([Floors, the atmosphere and the magnetosphere](#floors-the-atmosphere-and-the-magnetosphere),
"Investigation (2026-10-16)" and its 2026-10-17 sweep).

**API.** None new: a change of that branch, under the same flags.

**Semantics.** Where the branch would raise the density to `ρ_atm` (`D/cosh w₁ < ρ_atm`), return the
atmosphere state (`policy_atmosphere`: `ρ_atm` at rest, the atmosphere's `s` and `Yₑ`), flagged
`FLAG_POL_ATMOSPHERE | FLAG_POL_S_FLOORED`. Every other path unchanged bit for bit. The output is
inside the valid range and `D_out = D_atm ≤ max(D_in, D_atm)`.

**Upstream acceptance.** Random inputs with `τ` below the coldest state at their momentum, across
the density box: every output inside the valid range, `D_out ≤ max(D_in, D_atm)`, no output with
`ρ = ρ_atm` and `w > 0` from this branch, and every output a fixed point of `con2prim_safe`; the
converged path and the other branches bit for bit (the existing suite).

**Downstream check.** `test/atmosphere_tests.jl`: the magnetized atmosphere tube asserts no fallback
and `W < 6` (with `collapse_to_atmosphere = true` in its policy, see below), and runs `10⁻¹⁵` apart
agree to `10⁻¹⁰`; the ARS row of the ideal magnetized atmosphere (`ars.t.injected_D ≥ 1`) goes.

**Workaround** (measured, no upstream change): the reset writes the atmosphere at rest where the
policy's repair holds more rest mass than the cell (`S_FLOORED` or `W_CAPPED` with
`ρ cosh w > D`), and the cases' policy sets `collapse_to_atmosphere = true` — variant
`Breset+coll`: on the tube on all three meshes the same numbers as the in-process form to every
digit printed, but it does not reach the recoveries inside the stage solve (the magnetized ideal
atmosphere under ARS reaches `W = 53` where the in-process form stays at 4.3). Without the
collapse option either form leaves the collapse branch open (the note above).

EOS-7 was added 2026-10-17, variant C; for completeness, as Erik asked for every variant that needs
an EntropyEOS change:

*EOS-7 — The atmosphere's entropy in the atmosphere band.*

**Semantics.** A state whose density, after the policy's own projection, lies in the atmosphere
band `ρ < atm_trigger · ρ_atm` takes the atmosphere's entropy (`policy_atmosphere(…).s`), `ρ`, `w`,
`Yₑ` kept, flagged `S_CEILED`/`S_FLOORED` by the direction; with `s_atm` set to the window's floor
(an existing option) this is a cold atmosphere. Inside the valid range, `D_out = D_in`. **Not
recommended:** measured, it does nothing for the tube's chaos on its own (blow-ups 0, 1 and 5 of 48
on the three meshes, against 0, 0 and 3; `W` at the cap in the median run) and nothing beyond B or
the trigger when combined. Acceptance and downstream check would follow EOS-6's pattern if it were
taken up.

**2026-10-21.** EOS-6 was **superseded 2026-10-21** (design agreed by Erik with EntropyEOS's
review): EntropyEOS does not decide where the state goes; it reports what its repair did
(`FLAG_POL_ρ_FLOORED`, EOS-8, landed in 1.1.0), and this package's recovery wrapper writes the
atmosphere at rest on that flag (`CODE.md`, "Sharp edges"; [Floors, the atmosphere and
the magnetosphere](#floors-the-atmosphere-and-the-magnetosphere), 2026-10-21).

EOS-7 was not pursued, 2026-10-21.

EOS-8 was added 2026-10-21; design agreed by Erik with EntropyEOS's review; superseding EOS-6. Its
text, which is 1.1's:

*EOS-8 — Report the density floor (req; landed in EntropyEOS 1.1.0, unregistered).*

**Why.** `con2prim_safe`'s entropy-floor branch (τ below the coldest state at the momentum) raised
the density to `ρ_atm` at the coldest state's rapidity without saying so, writing `W ≈ |S|/ρ_atm`
and adding rest mass; this package needs to know when that happened to put such a point at rest
itself.

**API** (EntropyEOS 1.1.0, origin/main 43923c7): `FLAG_POL_ρ_FLOORED` (bit 16: ρ raised to `ρ_atm`
without an atmosphere reset, i.e. `ρ cosh w > D` — the rapidity cap, which raises ρ while keeping
`D`, does not set it); `FLAG_POL_ρ_CEILED` (bit 13, now also the branch's silent upper clamp);
`FLAG_POL_ρ_CLAMPED = ρ_FLOORED | ρ_CEILED` (test with `& != 0`); `FLAG_POL_ANY` including bit 16;
`policy_atmosphere` exported. Nothing else of the policy changed.

**Downstream.** `src/con2prim.jl` branches on `flags & FLAG_POL_ρ_FLOORED != 0` alone and returns
the atmosphere at rest (`policy_atmosphere`, its EOS point by `evaluate`), adding
`FLAG_POL_ATMOSPHERE`; `test/recovery_tests.jl`, "A density the policy raised to the atmosphere's is
the atmosphere at rest", at `Float64` and `Float32` (in a kernel, inferred, allocation-free, bit 16
through `P` and the diagnostics). `[compat]` `EntropyEOS = "1.1"`; the names join
`test/prerequisite_tests.jl`'s list, and no probe is kept (the rule above).

**Is it guaranteed in a clean checkout?** Yes, while EOS-0 is open: `[sources]` names
`rev = "main"`, so a tree without a `Manifest.toml` resolves EntropyEOS at `main`'s head, and
`[compat]` `"1.1"` then requires that head to be ≥ 1.1.0 and < 2 — a `main` without the flag (or a
2.0) fails to resolve rather than loading silently. An existing `Manifest.toml` pinned at 1.0.0 no
longer satisfies `[compat]`; `Pkg.update("EntropyEOS")` (or deleting the manifest) fixes it. Once
EntropyEOS is registered, `[compat]` alone guarantees it.

**2026-10-22.** EOS-9 was added 2026-10-22 on the experiment branch
`claude/no-collapse-to-atmosphere` for Erik's question whether `collapse_to_atmosphere` can go;
rewritten the same day after his review in six questions, Q1–Q6, with his decisions on Q3 (b) and
Q6 (a); proposed, not decided. Measurements: `bin/endpoint_measure.jl`,
`bin/front_investigation.jl`; emulation: `bin/hot_projection.jl`, `HOT_PROJECTION = 2`;
[Floors, the atmosphere and the magnetosphere](#floors-the-atmosphere-and-the-magnetosphere), "Investigation (2026-10-22)". The request as
written:

*EOS-9 — Step 4's endpoint branches: edge-consistent, guarded, reporting facts.*

**Why.** When `con2prim_safe`'s solve fails, step 4 diagnoses the state with two endpoint solves at
the edges of the entropy window and projects it — the cold branch (τ below the coldest state at the
momentum) onto `s_min`, the hot branch (τ above the hottest) today onto the collapse ceiling
`ρ_ceiling`. Measured on EntropyEOS's synthetic table (whose window depends on `ρ`), across the box,
hydrodynamic and MHD mode, about 4000 inputs per branch and mode:
- the hot branch writes `ρ_ceiling` for every input it diagnoses — a hot near-vacuum cell is no
  collapse; in the magnetized tubes this was every blow-up, and `collapse_to_atmosphere` (an
  excision option for collapsing stars) is today's only way round it;
- both branches solve at `srange(D/cosh w_ref)`'s edge but write the state at `ρ₁ = D/cosh w₁`: the
  edge there differs (cold: by 1.5·10⁻⁵ of the window, in 94% of inputs; hot: by 2·10⁻³ median, up
  to 0.27), so the written rapidity belongs to another `h` — the cold branch's momentum is off by
  `3·10⁻⁸` (the solver's tolerance is `10⁻¹²`), and a hot projection at `sr_ref`'s ceiling (the
  first form of this request) by 0.4% median and 5.6% at most, with `s` up to 0.27 of the window
  inside it rather than at its edge;
- the endpoint solves themselves: every cold one converged; of about 29,000 hot ones, 166 ended at
  the residual's floating-point floor just above tolerance (`|f₁|/cosh w ≤ 8 tol`; 93 out of
  iterations with the bracket already at the floor, the rest a collapsed bracket) — none capped,
  none non-finite, and no diagnosis changed with a solve 1000× tighter and 10× longer (nor at
  `ρ₁`). In the tubes every endpoint solve converged (584 to 1318 per run);
- step 1's `τ_max` (the largest `ρh` on a 5×3 grid, times `W_cap²`) stopped 8% (hydrodynamic) and
  11% (MHD) of the hot inputs before any solve and wrote `ρ_ceiling` for states the table
  represents.

**API.** None new. Changes of `con2prim_safe`'s steps 1 and 4 and of `check_con_state`; the flags
are the existing ones.

**Semantics.**
1. *(Q1) One edge-consistent root find per branch*, for the diagnosis and the projection alike: the
   rapidity `w` with `f₁(w) = 0` at `s = s_edge(D/cosh w)` inside the residual (`s_min` of
   `srange(D/cosh w, Yₑ)` for the cold branch, `s_max` for the hot), one 1D root find on
   `[0, w_max]` (measured: Illinois, 20–30 evaluations, no failure in about 15,000 inputs; it agrees
   with the fixed point `ρ ↦ s_edge(ρ) ↦ w ↦ D/cosh w`, which needs 4 passes cold and 13–17 hot,
   in every input, to `10⁻⁹` in `w`). The branch is taken on the sign of `f₂` there (cold `f₂ > 0`,
   hot `f₂ < 0`). Testable: `s_out` is at the edge of `srange(ρ_out)`; `S_out = S_in` to the
   solver's tolerance unless `W_CAPPED` (measured: at most `7·10⁻¹⁰`, median `10⁻¹⁵`); and no valid
   state exists at `ρ_out`'s window (the sign of `f₂` at the edge). The cold branch's projection
   (already shipped) is included.
2. *(Q2) Guards.* A branch is taken only if its root find found a root — `f1_converged`, or a
   bracket around a sign change collapsed to the floating-point floor (the measured endpoint
   solves that missed the tolerance are all of this kind), or the root lies beyond `w_max`
   (`f₁(w_max) < 0`: capped legitimately, `W_CAPPED`) — and the returned `w` and `f₂` are finite. A
   root find that ran out of iterations with an open bracket, or met a non-finite residual, gives no
   diagnosis; if neither branch has one, the existing "neither diagnosis fits: excise" path
   (`FLAG_POL_ATMOSPHERE`). This governs the diagnosis (the sign of `f₂`) and the projected rapidity,
   for both branches.
3. *(Q3, decided (b): EntropyEOS reports facts, the application decides.)* The hot branch's
   projection is `(ρ₁ = D/cosh w₁, s_max(ρ₁), Yₑ = ye_in, w₁ = min(w, w_cap))`, the density clamped
   into `[ρ_atm, ρ_ceiling]` exactly as the cold branch clamps it: raised to `ρ_atm` with
   `FLAG_POL_ρ_FLOORED`, lowered to `ρ_ceiling` with `FLAG_POL_ρ_CEILED` — never replaced by the
   atmosphere. Whether a floored state is put at rest is the application's rule (TreeGRRMHD's: the
   atmosphere at rest, `D_out ≤ max(D_in, D_atm)` — this package's criterion, not EntropyEOS's).
4. *(Q4) Flags describe the returned state, both ways.* Where the whole state is replaced by the
   atmosphere, return `FLAG_POL_ATMOSPHERE` and its cause (`FLAG_POL_NONFINITE`) only, not the
   clamps of the discarded state (today the converged path returns `W_CAPPED | ATMOSPHERE` when the
   capped state falls below `ρ_atm`). The hot branch's output may carry `FLAG_POL_S_CEILED` (bit 11,
   what the converged path sets for an entropy above the window), `W_CAPPED` (a legitimate cap
   only), `ρ_FLOORED`, `ρ_CEILED`, `YE_CLAMPED` (the projection is passed `ye_in`, not `ye_c`, as the
   cold branch's is); `FLAG_POL_CEILING` is left to the collapse test. Every flag set describes
   something that happened to the returned state, and every clamp that happened is flagged.
5. *(Q5) `check_con_state` changes with it:* its endpoint diagnosis and `con2prim_safe`'s step 4
   share one routine (today they bracket differently — `check_con_state` takes the union of the
   windows at `D` and at `D/cosh w_max`), its hot case returns `FLAG_POL_S_CEILED` (`CEILING` only
   for `D > D_max`), and its contract reads: it predicts which repair applies, not its clamps.
6. *(Q6, decided (a).)* Step 1 stays pure arithmetic and only `D > D_max` short-circuits to
   `pol_collapse`; `τ − B² > τ_max` with `D ≤ D_max` falls through to the solve (and to step 4 if it
   fails), and the `τ_max` test leaves `check_con_state`'s cheap mode too (whether `τ_max` stays in
   `PolicyOptions` is EntropyEOS's call). The `D_max` collapse reports `FLAG_POL_ρ_CEILED` (and
   `W_CAPPED` where it caps) alongside `FLAG_POL_CEILING`.

**Upstream acceptance.** On the synthetic table (directly and through a `ρ`-dependent window),
random inputs across the density box, hydrodynamic and MHD mode, aimed at either branch:
- `s_out` at the edge of `srange(ρ_out)`, `|S_out − S_in| ≤` a small multiple of the solver's
  tolerance unless `W_CAPPED`, and `f₂` at the edge of the right sign;
- the flags in both directions (each set flag describes the returned state; each clamp is flagged),
  and an atmosphere replacement carrying `ATMOSPHERE` and its cause only;
- `check_con_state`'s branch flag equals `con2prim_safe`'s for every input reaching step 4;
- the guards: inputs constructed so that an endpoint root find cannot converge or turns non-finite
  excise with `FLAG_POL_ATMOSPHERE`;
- every output a fixed point of `con2prim_safe`; the converged path and every other branch bit for
  bit, but the atmosphere flags of item 4; `τ_over` with `D ≤ D_max` no longer reaching
  `pol_collapse`.

**Downstream check.** `collapse_to_atmosphere` removed from the atmosphere tubes (`case_policy`), and
the suite green as it stands — measured with the emulation: 25940 pass at one thread (Julia 1.13);
without it (released 1.1.0, the option off) the three assertions of the ideal magnetized atmosphere
fail (it blows up). The tubes' numbers: those of `claude/atmosphere-at-rest` on every mesh to the
digits printed, but the ideal atmosphere under ARS (`−4.5·10⁻⁹` of rest mass where the option
added `5.6·10⁻⁸`); step 3's sweep loses fallbacks (36, 78, 103 → 33, 74, 90 without the cutoff;
the stage solve in a kernel 47 → 48 of 48 at `Float64` and `Float32`).

**Workaround:** `collapse_to_atmosphere = true` in the cases with a magnetized atmosphere (as on
`claude/atmosphere-at-rest`).

**2026-10-24.** EOS-8 and EOS-9 amended, EntropyEOS 2.0: `[compat]` is `"2"` and `[sources]`
pins the tag `v2.0.0` from 2026-10-24. In 2.0 `FLAG_POL_ρ_FLOORED` is bit 11 and means the state lies on the `ρ_min` face (the
momentum lowered to `w = acosh(D/ρ_min)`, `D` raised only where it was below `ρ_min`);
`FLAG_POL_ρ_CLAMPED` and `policy_atmosphere` are gone, and the atmosphere at rest on that flag is
EntropyEOS's `apply_policy`, which this package calls (EOS-9). (The EOS-8 text of 2026-10-21 above
is 1.1's.)

EOS-9 landed in EntropyEOS 2.0.0, tag `v2.0.0` (7d68ddb), unregistered; `[sources]` pins the tag,
`[compat]` is `"2"`, so no probe — there never was one. It landed as a redesign, EntropyEOS's
two-level projection (its `CODE.md`, "State policy"; `HISTORY.md`, 2026-10-22 and 2026-10-23),
which subsumes the request: level 1, `con2prim_safe` onto `ProjectionOptions(ρ_min, ρ_max, w_cap)`,
decides the active faces by one diagnosis along the entropy edges (items 1, 2 and 5: the
edge-consistent root find, the guards, `check_con_state` sharing the diagnosis); the hottest state
is projected onto the `s_max` face with `D` and `S` kept and τ lowered, `S_CEILED` (item 3, without
the density clamp: a density pushed below `ρ_min` is the `ρ_min` face instead, `ρ_FLOORED`);
`τ_max` and `D_max` are gone (item 6: `D > ρ_max cosh w_cap` is the `ρ_max` face, `ρ_CEILED`;
`CEILING` is retired). Item 4 was rejected upstream (decided by Erik): level 2,
`apply_policy(eos, in, out, proj, pol)`, keeps level 1's bits and adds `ATMOSPHERE`, since the bits
say why. The face rules differ from 1.1 for states 1.1 had repaired (w_cap keeps τ, ρ_max keeps S
and τ, the s faces keep S), and a failed solve is `NOT_CONVERGED`, which `apply_policy` sends to the
atmosphere. Downstream: `collapse_to_atmosphere` and `case_policy` are gone, and the recovery calls
`apply_policy` (`CODE.md`, "Primitive recovery"; the measurements are in [Floors, the atmosphere and
the magnetosphere](#floors-the-atmosphere-and-the-magnetosphere), 2026-10-22 and 2026-10-24). The request of 2026-10-22 (above) was thereby resolved, overtaken by EOS-9
landed in EntropyEOS 2.0.

EOS-5's note (ii) was resolved 2026-10-24: overtaken by EntropyEOS 2.x's level 1, EOS-9, which
projects such a state onto the entropy ceiling with `D` kept; no case sets
`collapse_to_atmosphere`.

EOS-10 was added 2026-10-24, EntropyEOS 2.0:

*EOS-10 — The nested fallback allocation-free under `--check-bounds=yes`.*

**Why.** `test/recovery_tests.jl`, "A state the projection put on the ρ_min face is the atmosphere
at rest", asserts that both recoveries allocate nothing, at `Float64` and `Float32`. Under
`--check-bounds=yes` (one of the suite's three standard runs) it fails, and only there: measured on
the laptop and on Symmetry with `@allocated` on the EntropyEOS calls alone, on the input
`Con2PrimIn(D = 5·10⁻¹⁰, τ = 10⁻⁹ − 5·10⁻¹³, D_Y = D/2, S_par = 0, S_perp = 10⁻⁸, B² = 0)` (Γ = 2
gas, `ρ ∈ [10⁻¹⁰, 10⁻²]`, `default_projection(eos, 10⁻¹⁰)`, cold guesses) — level-1 flags `0x8800`
(`ρ_FLOORED | S_FLOORED`), solver result `failed_no_bracket` — `con2prim_safe` allocates **272 B at
`Float64` and 144 B at `Float32`**, all of it in `con2prim_safe_finish` → `con2prim_finish` (the
nested fallback); `con2prim_safe_start`, `con2prim_start` and `apply_policy` allocate nothing, and
without bounds checks nothing does. 272 B and 144 B are the size of one `MVector{33,T}`
(`bracket_scan`'s stack scratch: 33·8 = 264 B plus a header; 33·4 = 132 B, aligned) — inferred from
the size, not traced: with `@inbounds` disabled the indexing's `BoundsError` path can capture the
vector, so it escapes and is not promoted to the stack. EntropyEOS 1.1 allocated twice as much,
544 B and 288 B, on the same testset's input of then. EntropyEOS's own `CODE.md` ("Allocation
gates") skips its gates under `--check-bounds=yes`, saying the property cannot hold there.

**API.** None new. `bracket_scan` (or whatever holds the scratch) non-escaping with bounds checks
on: e.g. the indices checked explicitly against `NMAX` with a throw that does not capture the
buffer, or an immutable `NTuple` scratch updated by `Base.setindex`.

**Semantics.** Unchanged results, bit for bit.

**Upstream acceptance.** EntropyEOS's allocation gates run under `--check-bounds=yes` too, on a
state that reaches the nested fallback.

**Downstream check.** The two `rec_allocations(...) == 0` assertions of that testset pass under
`--check-bounds=yes` (today 4 failures in that run, the suite's only ones). Decided by Erik
2026-10-24: it stays an upstream request, and the bounds-checked run's four failures are known,
waiting on it (the assertion is not loosened or skipped here).

**2026-10-26.** Amended, EntropyEOS 2.2: the tag is `v2.2.0` (`[sources]` pins it from
2026-10-26), `[compat]` `"2.2"`. EOS-10 re-checked 2026-10-26, EntropyEOS 2.2.0: unchanged. The same input still
allocates 272 B at `Float64` and 144 B at `Float32` under `--check-bounds=yes` (laptop, `@allocated`
on `con2prim_safe` alone; nothing without bounds checks), and the bounds-checked suite fails the
same four assertions (Symmetry). 2.1 and 2.2 changed `bracket_scan`'s arguments (the confined
search's `w_lo`), not its scratch. Still waiting.

EOS-11 was added 2026-10-26; the request was point 1 of the 2026-10-24 discussion of EntropyEOS
2.0, answered before it was filed:

*EOS-11 — The validity test of `con2prim_safe`, public (req; landed in EntropyEOS 2.1.0).*

**Why.** Since 2.0.0 `con2prim_safe` accepts a converged root unflagged when it lies within the
face slack of the entropy window (10⁻⁷ of the window's width at `Float64`; one ulp for `Yₑ`) and
returns it as it is, so a state with `policy_flags == 0` can lie a little outside the exact window
— 5.5·10⁻⁹ of the window below `s_min` on the reproducer, a cold re-solve of a state the stage
limiter had written onto that face. The reconstruction tested face states against the exact
window, rejected such a cell's faces, and its positivity fallback, which substitutes the cell's own
value, could not repair them: under ARS(4,4,3) every perturbed cut-off atmosphere tube blew up
(`CODE.md`, "Reconstruction"; [Floors, the atmosphere and the magnetosphere](#floors-the-atmosphere-and-the-magnetosphere),
"EntropyEOS 2.0" and "The ARS(4,4,3) regression").
Options put to Erik: test with `check_prim_state` (also tests `ρ` and `w`, which would change the
reconstruction near vacuum); export the slack, or a predicate for the `s` and `Yₑ` faces; clamp
accepted roots onto the window upstream or downstream.

**API** (EntropyEOS 2.1.0, fa50e3c; decided by Erik, HISTORY 2026-10-24): `inside_window`, exported
(its semantics in `CODE.md`, "EntropyEOS"). Rejected upstream: exporting `face_slack` (two
conventions, and its value would become API) and clamping accepted roots (the primitives would no
longer reproduce the returned conservatives to the tolerance, and a fed-back clamped output needs
a Newton step).

**Downstream.** `admissible_fluid` (`src/reconstruction.jl`) calls `inside_window` in place of its
exact window and `Yₑ`-range tests and keeps its own `ρ > 0`, finite, and finite-`u` conditions;
`test/scheme_tests.jl`, "The positivity fallback admits every state con2prim_safe returns
unflagged" (the reproducer's state at an MC minimum, the slack's edges, `Yₑ` one and two ulps
out). `[compat]` `EntropyEOS = "2.2"`, `[sources]` the tag `v2.2.0`; `inside_window` joins
`test/prerequisite_tests.jl`'s list, and no probe is kept (the rule above). Measured in
[CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere),
"The perturbed atmosphere tubes" (re-measured 2026-10-26). In the table its state read "landed in EntropyEOS
2.1.0, as `inside_window` (decided by Erik)".

**2026-10-30.** EOS-10 landed in EntropyEOS 2.2.1 (2026-10-30; its HISTORY, 2026-10-29), which
keeps the scan's scratch on the stack under bounds checks, bit for bit: the cause
was `MVector`'s bounds check passing the buffer to `BoundsError`; both of the scan's buffers are now
wrapped in a scratch type whose check does not capture them, with bit-identical results, and
EntropyEOS's allocation gates now run under `--check-bounds=yes` too. EntropyEOS 2.2.1 is still
unregistered: `[sources]` pins the tag `v2.2.1`, `[compat]` `"2.2.1"`, since 2026-10-30, on
`claude/entropyeos-2.2`; `[compat]` requires 2.2.1, so there is no probe. So the bounds-checked
run's four allocation failures are gone.

### SpacetimeMetrics

**Step 2.** SM-1 was added in step 2; the 432 bytes per call of
`ExtrinsicCurvature(m, p)` were measured in step 2 (Julia 1.11 and 1.13).

**Step 2.** SM-1 was added in step 2:

*SM-1 — Allocation-free extrinsic curvature (wanted, step 9; workaround).*

The `Analytic` provider needs `K_ij` at every cell of every right-hand side, inside a kernel.
SpacetimeMetrics 1.6's `ExtrinsicCurvature(m, p)` allocates 432 bytes per call (Julia 1.11 and
1.13, Kerr-Schild), which a device kernel cannot do, and it differentiates the metric a second time
where the provider already holds one `dmetric` result.

**What.** Make the existing method allocation-free, and add a method that takes the result of one
`dmetric` call.

**API.**
```julia
ExtrinsicCurvature(m::AbstractMetric, p::AbstractVector) -> SMatrix{3,3}     # unchanged, 0 bytes
ExtrinsicCurvature(g::SMatrix{4,4}, dg::SArray{Tuple{4,4,4}}) -> SMatrix{3,3} # new; dg[a,b,c] = ∂_c g_ab
```

**Semantics.**
- The same `K_ij = −(∂ₜγ_ij − D_iβ_j − D_jβ_i)/(2α)` as today, symmetric.
- `ExtrinsicCurvature(m, p) == ExtrinsicCurvature(dmetric(m, p)...)`.
- Generic in the element type (dual numbers included), `isbits` in and out, no allocation.

**Upstream acceptance.**
- `@allocated ExtrinsicCurvature(KerrSchild(1, 0.5), p) == 0` after a warm-up call, and the same
  for the new method, at `Float64` and `Float32`.
- The two methods agree bit for bit on Kerr-Schild, boosted Kerr-Schild and a gauge wave.

**Downstream check.** `test/prerequisite_tests.jl`, "SM-1": the item has landed when the new
method exists (`hasmethod`) and the old one allocates nothing (`@allocated`); then the upstream
`K_ij` must agree with this package's `extrinsic_curvature` to 1e-13 of `max |K|` on boosted
Kerr-Schild.

**Workaround** (amended in step 2): `extrinsic_curvature(α, β, sdg)` in `src/metric.jl`, marked
`SM-1`, computes `K_ij` from the provider's own `dmetric` result. It agrees with upstream's to
4.3e−16 (measured in step 2). When SM-1 lands, `metric_gradient(::Analytic, …)` calls
`ExtrinsicCurvature(g, sdg)` and the local function goes.

**2026-10-14.** SM-1 landed in SpacetimeMetrics 1.7.0.

**2026-10-14.** SM-1 landed 2026-10-14, SpacetimeMetrics 1.7.0: done; the compat is
`SpacetimeMetrics = "1.7"` (which requires Julia 1.11, the floor), and the prerequisite test
compared upstream's `K_ij` with the former workaround's formula, kept in the test, and checked that
both methods allocate nothing. The probe was retired 2026-10-14, `SpacetimeMetrics = "1.7"`
guaranteeing it: the comparison with the formula moved into `test/pointwise_tests.jl`'s
metric-provider testset (what it checks today in `CODE.md`, "SpacetimeMetrics"); the SM-1 downstream
check above was retired with it.

## The equations

**2026-10-13, step 2 — Pointwise physics.** *(Done, 2026-10-13, on `claude/step-2-pointwise`.)*

`CODE.md`: "Maxwell", "Ohm's law and the conductivity", "Stress-energy for the spacetime",
"Riemann solver", "The electromotive force for `A`", "Sources", "Metric providers" (pointwise).
Waits: none.

Changes:
- **`src/flux.jl`:** `physical_flux(m, cons, prim, E, B, ::Val{d}) -> NTuple{10}` and
  `light_cone(m, Val(d))`.
- **`src/riemann.jl`:** `riemann_flux(::Val{:llf | :hlle}, …)`.
- **`src/ohm.jl`:** `Vacuum`, `Constant`, `DensityProfile` and `IdealConductor`, plus
  `ohm_current`.
- **`src/stress_energy.jl`:** `(𝓔, S_i, S^ij)` and `T4_lower`.
- **`src/sources.jl`:** geometric sources in both the Valencia form and the `½√−g T^{ab}∂ⱼg_ab`
  form, and the Maxwell and gauge sources.
- **`src/emf.jl`:** the pointwise UCT–HLL EMF from four edge states.
- **`src/metric.jl`:** `Flat()` and `Analytic(m)`, evaluated pointwise via SpacetimeMetrics. This
  is the one place the derivative index order is converted.

*(Amended in step 2:)* as built, a face state is the tuple `(cons, prim, E, B)`, so the solvers are
`riemann_flux(Val(:llf | :hlle), m, L, R, Val(d))`; the EMF is
`uct_hll_emf(m, E, Bᵃ, Bᵇ, Val(d))`; a provider answers `point_metric(p, t, x)` and
`metric_gradient(p, t, x)`, the latter a `MetricGradient` with `dg[j,a,b] = ∂_j g_ab` and `K_ij`;
the conductivity is `conductivity(model, prim)`. `src/metric3.jl`'s products became `mul3` (plain
`*` and `+`) so that a kernel stays bit-identical to the host. The test file is
`test/pointwise_tests.jl`, and `test/Project.toml` gained ForwardDiff.

Accept:
- **Consistency:** `F(U, U)` equals the physical flux for both solvers. The direction-`d` flux
  equals the permuted direction-1 flux in flat space.
- **The signs `CODE.md` says a test pins:**
  - `∂ₜE = ∇×B − j` and `∂ₜA = −E − ∇Φ` in flat space;
  - the shift terms against IllinoisGRMHD's ideal `∂ₜA_i = ε_ijk vʲBᵏ − ∂ᵢ(αΦ − βʲA_j)`, with the
    coordinate velocity, on random states and metrics.
- **Sources:** the two momentum-source forms agree to round-off on Kerr-Schild at random points.
- **Stress-energy:** `n^a n^b T_ab = 𝓔` and `−γ_i^a n^b T_ab = S_i`.
- **Ohm:** `j = 0` for `E = −v×B` with `q = 0`.
- Everything is kernel-callable, and compiles at Float32.

*(Measured in step 2, worst over Julia 1.11 and 1.13:)* consistency 0 (LLF) and 2.2e−16 (HLLE);
the relabelled directions 4.7e−16 (flat) and 6.8e−16 (Kerr-Schild); the plane wave 6.3e−16; the
IllinoisGRMHD EMF 1.4e−15; the momentum-source forms 5.4e−16; the projections of `T_ab` 8.0e−16;
`|j|` at `E = −v×B` 2.3e−16 of `σW|B|`; the layer is bit-identical in a CPU kernel at `Float64` and
`Float32`, inferred and allocation-free. Beyond the list: HLLE with the UCT EMF is the exact upwind
solver of vacuum Maxwell in a constant curved metric (3.9e−16), and the energy source is Banyuls et
al.'s (5.6e−16). Every bound is at least ten times the measurement; `CODE.md` has the rest.

### Matter: total energy and momentum

**Step 2.** `geometric_source` in `src/sources.jl` was built in step 2,
and the agreement of its two forms (5.4e−16, 1.2e−15, 5.6e−16) was
measured in step 2. That the covariant form is the default was proposed in
step 2; its cost (Julia 1.11, `Float64`, one call, Apple M3 Pro: 36 ns
against 50 ns, 41 ns against 74 ns) was measured in step 2.

**2026-11-09.** The design said that `D s` is advected only as a fallback
for the primitive recovery and is reset from the recovered `s` wherever
the energy inversion succeeds (proposed; whether it earns its keep is
(open)). Amended 2026-11-09: as built, the reset rewrites `D s` only where
the policy repaired the cell, and the resynchronisation was tried and not
adopted (Erik, 2026-11-09; "Open questions").

### Maxwell

**Undated.** "The charge density is derived, not evolved" was marked
(decided). The uniform background field was first stated as
`𝔅̄ⁱ = √γ B̄ⁱ` with constant `B̄ⁱ`.

**Step 2.** The EMF's agreement with IllinoisGRMHD's form, the flux of
`𝔈`, and the plane wave in the Lorenz gauge were measured in step 2.

**Step 5.** Amended in step 5: the densitized `𝔅̄ⁱ` is the constant
(proposed in step 5), not `B̄ⁱ`.

### Ohm's law and the conductivity

**Step 2.** That the models are in `src/ohm.jl` and that
`conductivity(model, prim)` takes the whole primitive state, with
`DensityProfile`'s form, `IdealConductor`'s handling and the `σ = 0` mask,
was proposed in step 2; the current at `E = −v × B` (2.3e−16) and the
fluid-frame bracket (4.4e−16) were measured in step 2.

**2026-10-14.** σ → 0 where the plasma is magnetically dominated was
decided by Erik 2026-10-14, after step 3's review. That `μ_max` is required
and has no default, and that a physically motivated larger value trades a
small fallback rate, were decided by Erik 2026-10-14. A globally
convergent stage solve for that regime, a bracketed search over the
fluid's rapidity that leaves no fallback at `f ≤ 0.03` and 0.02% at
`f = 0.1`, is kept on the branch `claude/step-3-bracketed-stage-solve` and
is not in the code.

**2026-10-14, step 3, after review.** *(Amended in step 3 after review; decided by Erik 2026-10-14:)* the fallback rate was not
acceptable — with the predictor's non-ideal fraction `f = |E★ + v×B|/|B|` fixed, the corner
`b²/ρ ≥ 10`, `σΔt ≤ 10²` fell back on 0.17%, 3.5%, 8.0% and 20.9% at Γ = 2 for
`f = 10⁻³, 10⁻², 0.1, 1`. Two remedies were built: a globally convergent bracketed stage solve
(kept, unmerged, on `claude/step-3-bracketed-stage-solve` for a draft PR) and, in the code, σ → 0
where the plasma is magnetically dominated. Added: `MagnetizationCutoff(model, μ_max)`
(`src/ohm.jl`), a `C¹` taper of any finite σ in `μ = B²/D` from `μ_max/10` to `μ_max`, constant
within a stage; `conductivity(model, prim, μ)` and `magnetization(m, B, D)`. Accept: the
sweep asserts, at fixed `f ∈ {10⁻³, 10⁻², 0.03, 0.1, 0.3, 1}` with the cutoff, no fallback at
`f ≤ 0.1` over the whole grid at every Γ; the model's own testset (continuous, monotone, zero
beyond the taper, refuses `IdealConductor`, bit-identical in a CPU kernel, allocation-free,
inferred, and `MagnetizationCutoff(model)` without `μ_max` throws). `μ_max` is required, with no
default: each case and driver states its own (decided by Erik 2026-10-14), and the tests pass it
explicitly. Measured: `μ_max = 10`, the largest tested at which sixteen seeds (41,472 problems per
Γ and `f`) show no fallback at `f ≤ 0.1` (nor at `f ≤ 0.3`); at `f = 1`, 0, 3, 3 (Γ = 4/3, 5/3,
2). The table, and the fallbacks a physically motivated `μ_max = 50–100` costs, are in `CODE.md`,
"Ohm's law and the conductivity".

**Step 3.** `conductivity_slope` was proposed in step 3.
`MagnetizationCutoff` as built was proposed in step 3 (the conductivity
table gave "(step 3)" for its use). The safe value `μ_max = 10` was
measured in step 3.

**Step 6.** `μ = +∞` where `D ≤ 0` was proposed in step 6.

### Stress-energy for the spacetime

**Step 2.** `stress_energy` and `T4_lower` in `src/stress_energy.jl` were
built in step 2, and their checks (8.0e−16, 5.5e−16, 1.2e−15, 3.3e−16)
measured in step 2.

## Thermodynamics

**Undated.** "EntropyEOS everywhere" and "Analytic EOSes come from
EntropyEOS" were marked (decided). The bullets "Analytic EOSes come from
EntropyEOS" and "A polytrope is a Γ-law at fixed entropy" were amended with
`PLAN.md`.

**Step 1.** The units adapter `GeometricEOS(eos; density)` was proposed
in step 1; the density unit 6.1758e17 g/cm³ and the adapter's agreement
(9.1e-15; 3.0e-15 and 8.4e-13) were measured in step 1. EOS-2 landed
before step 1, so the synthetic table serves only as the table the units
adapter is checked on (amended in step 1).

**Step 5.** The warm-start bullet used to say: "Warm starts (`u_guess`,
the previous `log₁₀ T`) are carried per cell in the primitive set and
passed to every EOS call, including those at reconstructed face states."
Amended in step 5: every recovery is seeded from the state itself
(proposed in step 5), and not from `P`. The recovery cost of 0.86 µs a
cell on the Alfvén wave was measured in step 5.

**2026-10-23.** Decided by Erik 2026-10-23, amended 2026-10-24: the
right-hand side is pure in `(u, t)` outside a step; a step is pure in `uⁿ`
— the step cache.

## Primitive recovery

**Undated.** The pure-hydrodynamic recovery (subtract the electromagnetic
energy and momentum, project to EntropyEOS's scalar form with `B² = 0`, call
`con2prim_safe`) was marked "(decided)" in `CODE.md`.

**2026-10-13, step 1 — Units, variables and the 3+1 primitive recovery.** *(Done, 2026-10-13, on `claude/step-1-recovery`.)*

`CODE.md`: "Conventions", "Thermodynamics", "Primitive recovery". Waits: none. EOS-2 had landed,
so the tests run on `IdealGasEOS`; the synthetic table checks the units adapter (amended in
step 1).

Changes:
- **`src/units.jl`:** geometric ↔ EntropyEOS run units, with one density factor and κ. *(Step
  1:)* `EOSUnits`, and `GeometricEOS`, an `AbstractEOS` that applies the factor inside the EOS
  interface.
- **`src/variables.jl`:** index constants and `NTuple` accessors for the 10 cell variables and
  for `P`'s layout.
- **`src/metric3.jl`:** an `isbits` pointwise `Metric3{T}` (α, βⁱ, γ_ij, γ^ij, √γ), with the
  ε-tensor cross products and `ideal_E`.
- **`src/prim2con.jl`:** `prim2con(eos, m, prim, E, B) -> cons`, the 3+1 form including the EM
  parts.
- **`src/con2prim.jl`:**
  - `con2prim_hydro(eos, opts, pol, m, cons, E, B, guesses) -> (prim, flags)` subtracts the EM
    parts, projects, calls `con2prim_safe` in hydro mode, and rebuilds `vⁱ`.
  - `con2prim_ideal(…)` is the `B² ≠ 0` call used as the ideal seed.

Accept:
- `con2prim_hydro ∘ prim2con` is the identity on random states, to 1e−12 relative in the
  conserved variables and, in the primitives, to EntropyEOS's own conditioned bounds (errors
  divided by `κ_ρ` and `κ_s`, below 3e−11 and 1e−10; see EOS-2). A raw 1e−12 in the primitives
  is not attainable (proposed in step 0). *(Step 1:)* "relative" is per component for `D`,
  `D Yₑ` and `𝔈`, and against the energy density `τ + D` for `S_i` and `τ`; `√γ D s` is not an
  input of the recovery and is checked through `s` (proposed in step 1). At EntropyEOS's default
  tolerance this measures 9.8e−13, since 1e−12 is `con2prim`'s own stopping criterion; so the
  conserved round trip is asserted with `Con2PrimOptions(tol = 1e−14)`, where it measures at most
  5.4e−14 (`τ` per itself, Julia 1.11; 4.6e−14 at 1.13), over 18× inside the bound, and the
  default-tolerance number is recorded (amended in step 1 after review).
  - `W ≤ 10`;
  - random metrics, from Kerr-Schild at random points and random boosts;
  - random E and B with EM energy up to 100× the fluid's.
- `con2prim_ideal` recovers `v` when `E = −v×B`.
- The units adapter round-trips, and `M☉`'s density unit equals 6.176e17 g/cm³ to 1e−4.
- The atmosphere flags are exercised.
- Everything is callable from a CPU kernel.

**Step 0.** The statement that the recovery's inversion is conditioned, not
uniform, is an amendment of step 0. Step 0 measured the raw 8.9e−10 in `s`
that `CODE.md` cites ("Primitive recovery"), while over step
0's prerequisite test's 24 states (retired with EOS-2's probe, 2026-10-14)
the conditioned errors were at most 2.1e−14 in `ρ` and `w` and 1.7e−13 in
`s`.

**Step 1.** The scalar form first said `S_⊥ = 0`; it was amended in step 1
to `S_⊥ = |S|`, for the reason now in `CODE.md` ("Primitive recovery").
The velocity rebuilt as `vⁱ = tanh w · Sⁱ/|S|`, measuring `S_i` and `τ`
against the energy density `𝓔 = τ + D`, and `con2prim_hydro` returning
EntropyEOS's policy flags as they are were proposed in step 1. Step 1
measured the 7e-11 in the conserved `D` of the round trip (the literal
`vⁱ = Sⁱ/(ρhW²)`) against 4e-14, and the round-trip errors through the 3+1
layer (`test/recovery_tests.jl`). The conserved round trip with
`Con2PrimOptions(tol = 1e−14)` and its conditioned primitives (3.0e−14 in
`(ρ, w, v)`, 2.7e−13 in `s`) were measured in step 1, and amended in step 1
after review, which found the default-tolerance assertion fragile. Step 1
measured 5.1e−14 and 5.4e−14 (against `𝓔`, and `τ` against itself) at
`Γ = 5/3` at Julia 1.11, before the metric products stopped contracting.

**Step 2.** Amended in step 2: the conserved round trip at `tol = 1e−14`
that `CODE.md` states ("Primitive recovery": at `Γ = 2`, at Julia 1.11 and
1.13 alike) is step 2's, after the metric products stopped contracting (see
"Precision and devices").

**Step 1.** `PLAN.md`'s sharp edge on EntropyEOS's output velocity (take the
magnitude from the returned rapidity, not from `|S|/(ρhW²)`) was amended in
step 1.

**Step 3.** `con2prim_hydro` was split at the fluid's share (`field_budget`,
`fluid_recovery`) in step 3, so that the stage solve calls the same recovery
bit for bit.

**2026-10-21.** Decided by Erik: one exception, the one decision this
package adds to EntropyEOS's policy: where the flags say the policy raised
the density to the atmosphere's without resetting the state
(`FLAG_POL_ρ_FLOORED`, EntropyEOS 1.1, EOS-8), both recoveries return the
atmosphere at rest. Until 2026-10-22 it was flagged by adding
`FLAG_POL_ATMOSPHERE` to the discarded state's flags.

**2026-10-21.** The sharp edge on EntropyEOS's levels was added, with this
package's atmosphere rule of 2026-10-21.

**2026-10-22.** Decided by Erik: the atmosphere replacing a floored state is
flagged `FLAG_POL_ATMOSPHERE | FLAG_POL_ρ_FLOORED` — the replacement and its
cause ("Floors, the atmosphere and the magnetosphere"). The same day's
experiment (`claude/no-collapse-to-atmosphere`) emulated the decision
EntropyEOS 2.0 later made upstream for "τ above the hottest state": where
keeping `D` at the momentum puts `ρ` below `ρ_min`, the `ρ_min` face is
taken instead and level 2 writes the atmosphere at rest.

**2026-10-23.** **Measurement (2026-10-23): recovery counts and costs**
(branch `claude/recovery-cost`, not a PLAN.md step; `bin/recovery_cost.jl`,
which redefines the recovery's two `con2prim_safe` calls and the passes
in-process and is the code bit for bit with its counters off; Julia 1.13,
Apple M3 Pro, one thread, load 4–15 on 12 cores; ideal gases as the cases
set them). Per owned cell and SSP3(4,3,3) step the scheme makes **11
recoveries that run EntropyEOS's cold seed** — 3 right-hand sides, 3 stage
limiters, 1 step limiter, 4 stage predictors (or `IdealConductor`'s single
ideal recovery): `state_guesses` gives `s` but no `w`, and `con2prim` runs
`cold_seed` whenever `s` or `w` is missing — plus the stage iterations,
which are warm (0 to 5.5 a step). The conserved bounded ghost fill
(emulated: step 8's first fill, the tree at 7605986) adds
`3((N + 2G)/N)^D − 3` ghost recoveries (1.8 in 1D at `N = 10`, 2.7 in 2D at
16, 13.1 in 3D at 8: the 5.36 holds); the primitive fill (this tree's)
none. A cold-seeded call costs 0.75–1.3 µs (2.0–2.2 µs in the 3D blast,
with about 4 Newton iterations and 4% nested fallbacks), a warm iteration
0.22–0.25 µs; recoveries are 48–53% of the step on the 1D waves (100 cells,
where kernel launches dominate the rest), 67–75% on the tubes and in 2D,
73% in 3D, and 51–82% with the conserved fill (whose step is 5–15% longer
in 1D and 2D, 64–74% in 3D). Pointwise, `con2prim_safe` from a warm start
10⁻³–10⁻² away costs 0.4–0.7 µs on the Γ = 5/3 gas against 1.4–1.7 µs cold
(median) and 2.4–3.5 µs on EntropyEOS's synthetic table through
`GeometricEOS` against 10–11 µs; the advected `s` alone saves nothing (the
same time and iterations as fully cold). Estimated savings, as a fraction of
the step, primitive fill / conserved fill (the recorded calls of three steps
replayed both ways):

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
change to the code, which decision is his.)* *(Resolved 2026-10-23:
overtaken by Erik's decision of (a), (b) and (c) below; the share near rest
is EntropyEOS's.)*

Decided by Erik (2026-10-23): (a), (b) and (c) are all to be done; (c) is
the parent-centred primitive fill, which this tree already ran, now the only
one ("Refinement"), with the dead `prim2con` of the conserved ghosts
removed. (a) and (b) became the step cache (stage 3 of the recovery plan,
2026-10-24). The same day Erik decided "the two-phase con2prim everywhere",
CPU only (built as stage 4 of the recovery plan, 2026-10-30).

**Stage 1 of the recovery plan.** No ghost is ever recovered (stage 1), so
the step cache, built later, has owned cells only.

**2026-10-24, EntropyEOS 2.0** (stage 2 of the recovery plan). **The two
levels** were added to `CODE.md` as an amendment of 2026-10-24, EntropyEOS
2.0; tag `v2.0.0`, 7d68ddb, through `[sources]`, `[compat]` `"2"`. The
package's exception of 2026-10-21 became EntropyEOS's own level 2, and the
flags of an atmosphere keep level 1's bits. Level 2 *is* this package's rule
of 2026-10-21, so the wrapper's own `atmosphere_at_rest` and `floored` are
gone. One difference from 2026-10-22's decision: the atmosphere's flags are
level 1's bits plus `ATMOSPHERE`, not `ATMOSPHERE | ρ_FLOORED` alone
(EOS-9's Q4, rejected upstream by Erik: the bits say why); every tally that
reads them (`X_POLICY`, the reset's `FLAG_POL_ANY`) is unchanged by that.
The options: 1.x's `ρ_atm` is `ρ_min`, its `ρ_ceiling` `ρ_max`, and `w_cap`
moved into `proj`; `default_policy`, `policy_derive_bounds`, `D_max` and
`τ_max` are gone (`τ_max` stopped states the table represents, EOS-9), and
with them `case_policy`. The flags were renumbered to bits 8–17 (`W_CEILED`
17 is 1.x's `W_CAPPED`); `CEILING`, `ρ_CLAMPED` and `YE_CLAMPED` are gone.

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
- **`s_atm`**: kept at 1.x's midpoint, measured both ways (`CODE.md`,
  "Primitive recovery", "The two levels"). *(Decided by Erik 2026-10-24:)*
  `s_atm` stays the explicit 1.x midpoint (`recovery_options`).
- **Why EOS-9 needs no `collapse_to_atmosphere`**: `CODE.md`, "Primitive
  recovery", "The two levels"; the atmosphere tube drops it (measured below
  and in "Floors, the atmosphere and the magnetosphere").

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
  (`test/stage_solve_tests.jl`); step 3's sweep ("The implicit stage
  solve").
- **The flags**: the atmosphere tubes' `|` of policy flags reads 2304
  (`ATMOSPHERE | ρ_FLOORED`, bits 8 and 11) and 100608 (and bits 15, 16)
  where 1.1 printed 256 and 66816; the tallies are the same.
- **ARS(4,4,3) near the vacuum** (see "Floors, the atmosphere and the
  magnetosphere"): the hydrodynamic rarefaction's resets 131/110 →
  133/116; the ideal magnetized atmosphere `5.6·10⁻⁸` → `−2.0·10⁻⁹` of
  rest mass, `W` 4.11 → 4.13 (SSP3: `−6.6·10⁻⁹` → `−7.8·10⁻⁹`, `W` 4.01 →
  3.99); the coarse-fine magnetized tube `−1.4012·10⁻⁸` → `−1.4027·10⁻⁸`.
- **The bounds-checked run**: EOS-10 (PLAN.md).

**2026-10-24, stage 3 of the recovery plan.** **The step cache**
(`with_step_cache`; the entry and `two_levels` in `src/con2prim.jl`, the
cache on `MHDProblem`, the seed tallies in `X`; `evolve!(…; step_cache =
false)` the negative control; `test/step_cache_tests.jl`;
`bin/recovery_cost.jl steptime`) was built on
`claude/recovery-pipeline-3`: (a) and (b) of the 2026-10-23 measurement,
decided by Erik 2026-10-23. Its sharp edge was added the same day. Amended 2026-10-24: the right-hand side's
recovery and the ghosts divided by `√γ` (`· inv(√γ)`, so that the keys agree
in curved space); in flat space `√γ = 1` exactly and the two are the same
numbers, so the change moved nothing — the flat-space suite's output is the
same with and without it.

**What moves, against stage 2 on the same node** (every printed number of
the suite; the rest of them move in their last digits, as warm seeds answer
to the tolerance rather than to rounding): the uniform-flow testset's slow
flows (`vˣ = ±1/2`) **break their bound**: `1.06·10⁻¹¹`, `8.1·10⁻¹¹`,
`1.68·10⁻¹¹`, `8.4·10⁻¹¹` of a variable's scale (SSP3, ARS) against
`≤ 10⁻¹¹`, where stage 2 had `0.9–1.9·10⁻¹²` (the fast flows are
unchanged, `≤ 1.0·10⁻¹²`). Why: `CODE.md`, "Primitive recovery", "The
uniform flow: to the tolerance". **The bound is kept and fails; for Erik**
*(resolved 2026-10-30, decided by Erik: option (a), the drift to the
recovery's tolerance, `≤ 10³·tol`, plus the 1000-step run — "The uniform
flow: to the tolerance")* (accept the tolerance-level drift and say so in
the bound; or ask EntropyEOS for a warm start that always polishes with one
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

**2026-10-24, EntropyEOS 2.0.** The sharp edge was amended for EntropyEOS
2.0's two levels, `con2prim_safe` onto `ProjectionOptions` and
`apply_policy` with `PolicyOptions`. With the step cache the two calls moved
into one function, `two_levels`, which `fluid_recovery` and
`ideal_recovery` go through.

**2026-10-26, EntropyEOS 2.1 and 2.2.** **EntropyEOS 2.1 and 2.2**
*(amended 2026-10-26; tag `v2.2.0`, f26afca, through `[sources]`,
`[compat]` `"2.2"`)*. Three changes reach this package:

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
  ("The two levels", "The flags"; added to `CODE.md` as an amendment of
  2026-10-26, EntropyEOS 2.2).

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
- **The test count**, 26268 → 26296 (see "The test suite").

Unchanged to every digit: the uniform flow's four failures (above, 2026-10-24,
"What moves, against stage 2"), step 3's sweep, the shock tubes against their
stored references (no reference regenerated), every SSP3 atmosphere row, the
drift tallies, the `Float32` smoke run, the seed tallies of every case of
`bin/recovery_cost.jl steptime`. The bounds-checked run fails EOS-10's four
assertions as before (272 B and 144 B). The perturbed tube runs are under
[CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere),
"The perturbed atmosphere tubes", and the step time below.

*Re-measured 2026-10-26, EntropyEOS 2.2* (`bin/recovery_cost.jl steptime`,
the same script; Symmetry cn086, AMD EPYC 7532, one job, one pinned thread;
`claude/entropyeos-2.2` against stage 3 at 4929c94, two rounds each,
interleaved; ms per step as in `CODE.md`'s "Measured 2026-10-24"). The 2.2
tree is **1–3% slower** per step with the cache (Alfvén 5–7%, the EM wave
2–3%, the 2D and 3D cases −1 to 4%) and 1–5% without; the seed tallies are
the same in every case:

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
is unchanged (57–64%; in `CODE.md`). *(Decided by Erik 2026-10-30:)* the 1–3%
is accepted, and not attributed further.

**The uniform flow under 2.2** (re-measured 2026-10-26, every run above):
`1.0602831660·10⁻¹¹`, `8.1098066854·10⁻¹¹`, `1.6805728171·10⁻¹¹`,
`8.4017111996·10⁻¹¹` (`vˣ = −½`, `+½`; SSP3, ARS) — stage 3's numbers to every
digit, so the four failures stand, at one and four threads, in the clean
checkout and with bounds checks. Nothing in 2.1 or 2.2 addresses warm starts
returning at the tolerance (point 2 of the 2026-10-24 discussion).

**2026-10-30.** **The uniform flow: to the tolerance**, decided by Erik
2026-10-30 (option (a) of 2026-10-24's three): the testset's bound became
`≤ 10³·tol` (10⁻⁹) after 13 steps, where it was `≤ 10⁻¹¹`, plus the
1000-step run, `≤ 2·10³·tol`. The table of how the change grows was measured
on the laptop the same day.

**2026-10-30, stage 4 of the recovery plan.** **The two-phase recovery**
(`start_kernel!`, `finish_kernel!` and the site's own kernel, through the
per-owned-cell `Handoff` on `MHDProblem`; `test/two_phase_tests.jl`;
`bin/recovery_cost.jl steptime` times both, `finish` counts the share that
reaches finish) was built on `claude/recovery-pipeline-4` (decided by Erik 2026-10-23: "the
two-phase con2prim everywhere", CPU only). "The two levels" was amended the
same day: outside the stage solve's iterations level 1 runs as EntropyEOS's
two phases and level 2 in a kernel of its own. As built, whether the CPU
keeps the split or fuses was left to Erik: "Whether the CPU keeps the split
or fuses is Erik's decision (measured below)", and after the measurement
(`CODE.md`, "Measured 2026-10-30": the split is slower on the CPU) "*Whether
the CPU keeps the split or fuses is Erik's decision.*"

**2026-11-02.** Decided by Erik: **fused is the default** (`two_phase =
false` in `MHDProblem` and `evolve!`); the split stays in the code, tested
bit for bit against fused, for a GPU step to turn on. This resolved the
open decision of 2026-10-30.

**Step 9.** The step cache's curved-space keys were exercised in step 9: on
the Kerr-Schild cases, with the interior, one integration, two and a restart
agree bit for bit ("Validation ladder" 3).

### Where the fields live

**Undated.** That TreeAMR's centerings put each quantity where its equation
puts it (cells, edges, faces, vertices) was marked "(decided)".

### Field sets

**Step 1.** Step 1 laid out `P`'s 30 slots in `src/variables.jl` and amended
the table, which had said "~25". The metric derivatives and `K_ij` the
sources read were not in `P` yet: step 2 was to decide the form of the
sources, and with it whether they are stored or interpolated on the fly
(proposed in step 1).

**Step 2.** That the metric derivatives and `K_ij` are not stored and `P`
stays at 30 slots was proposed in step 2.

**Step 5.** The field sets were built in step 5 (`src/fieldsets.jl`,
`mhd_fields(T, forest; G = 3)`), with the light speed in `X`; the
diagnostics set `X` was proposed in step 5. The operators were proposed in
step 5, the `A` width staying (open) until step 8: Conservative width 3 for
`U`, PointValue width 4 (prolongation and restriction) for `A` and `Φ̃` —
the scheme's order plus one, rounded up to even. Nothing in step 5 refines,
so they only had to pass TreeAMR's checks. The design then said that `A`'s
PointValue width must be at least the scheme's order plus one "(open, to be
measured as TreeWave and TreeGH measured theirs)".

**Step 6.** Added to `X` in step 6: the cells the reset wrote as the step
limiter, and the change those writes made to `√γ D` and `√γ τ` (the reset's
injection).

**Step 7.** Added to `X` in step 7: `h∇·𝔅`.

**Step 8.** Decided in step 8: width 4 for `A` and `Φ̃` by the
interface-order table, and the matter's width 3 **bounded** (TreeAMR's
MC-bounded prolongation) in the ghost fill and the regrid — see "Refinement".
Amended in step 8, when TreeAMR 0.2.2 landed AMR-9: `A_j` take the
conservative family on a staggered set (point values of width 4 across the
edge: AMR-9, decided in step 8), and `ops_Φ` is `Φ̃`'s own keyword since step
8. The set `W` (`ρ★, s, Yₑ, uⁱ = Wvⁱ, Eⁱ`, the matter's ghost fill) was
added in step 8, prolongated with PointValue width 2.

**2026-10-21 and 2026-10-24.** The description of `P`'s policy flags was
amended on 2026-10-21 and on 2026-10-24 (EntropyEOS 2.0: bits 8–17,
`FLAG_POL_W_CEILED` the highest).

**2026-10-23.** Decided by Erik: `W` is prolongated Conservative width 3,
bounded, centred on the parent (step 8's was PointValue width 2). Since
2026-10-23 the matter `U`'s ghosts are not written by the scheme.

**2026-10-24.** Added: the recoveries' seeds per site in `X`, and the step
cache (the cell's latest recovery in the step: EntropyEOS's input and
output).

**2026-11-06.** The `Tsrc` row amended: it was the vertex set GH-3 read.

**2026-11-08.** The `Tsrc` row amended to `(ρ, S_i, S_ij)`; it was `T_ab`.
The `Mv` row added, the coupling contract redesigned; `Mv` was built in step
10, round 2.

### Reconstruction

**Step 5.** The PLM reconstruction and the face state were built in step 5
(proposed in step 5); the keyword `limiter` is step 5's name. The fluid
group's test was then exact: `Yₑ` outside the EOS's window or `s` outside
`srange(eos, ρ, Yₑ)` made it inadmissible.

**Step 7.** PPM, WENO-Z and MP5 were built in step 7 (proposed in step 7
unless marked otherwise); PLM's offset form computes the step-5 bits
unchanged. Measured in step 7: MP5's instability at the default CFL 2/5
(Balsara 3's dense shell, the reset firing on Balsara 3 and 4), and the
Alfvén wave's convergence under PPM, WENO-Z and MP5.

**2026-10-26, EntropyEOS 2.1.** The admissibility test became
`inside_window` (EOS-11). Until then it tested the exact window, which since
EntropyEOS 2.0 a recovered cell could fail. What it cost is under "Floors,
the atmosphere and the magnetosphere", 2026-10-24 ("EntropyEOS 2.0") and
2026-10-26 ("The ARS(4,4,3) regression").

### Riemann solver

**Undated.** In the design, the split speeds (fluid speeds for `D`, `D Yₑ`,
`D s`, light speed for the rest) and a resistive HLLC were (open), to be
decided by measurement on the TOV star.

**Step 2.** `src/flux.jl` and `src/riemann.jl` were built in step 2; the
face state `(cons, prim, E, B)` taken by `physical_flux` was proposed in step
2, and the consistency, direction and light-cone figures were measured in
step 2.

**Step 10; 2026-11-08.** The split speeds (`:hlle_split`) were measured in
step 10 on the Cowling star; the light cone stays, decided by Erik
2026-11-08. HLLC stays (open).

### The electromotive force for `A`

**Step 2.** `uct_hll_emf` in `src/emf.jl` was proposed in step 2, and its
agreement with the physical flux of the HLL middle state, to 3.9e−16,
measured in step 2.

**Step 5.** `emf_kernel!` was built in step 5 (proposed in step 5).

**Step 7.** The additive corner value of `E` under every reconstruction was
proposed in step 7; for PLM it is exactly step 5's
`E + σₐ slopeₐ/2 + σ_b slope_b/2`.

### Sources

**Step 2.** `src/sources.jl` (`geometric_source`, `explicit_source`,
Ampère's source split into its explicit and stiff parts) was built in step 2.

### Floors, the atmosphere and the magnetosphere

**Undated.** The design first read: "EntropyEOS's policy defines the
atmosphere (`ρ_atm`, triggers, ceilings) and `con2prim_safe` applies it; a
limiter resets `v = 0` in atmosphere cells and writes back only those cells,
as TreeHydro's does (proposed)." Step 5 built the reset to write back every
cell the policy repaired (`FLAG_POL_ANY`), not the atmosphere alone.

**2026-09-25.** "The reset is both the stage limiter and the step limiter",
then (proposed), was amended on 2026-09-25.

**Step 5.** `reset_atmosphere!` was built in step 5; writing back every cell
the policy repaired was proposed in step 5. In step 5's smooth runs it writes
nothing.

**Step 6.** From step 6 the two roles are counted apart (`reset_atmosphere!`
the stage limiter, `reset_step!` the step limiter), and the step limiter's
injection is summed.

**Step 8.** The magnetized atmosphere tube was chaotic at rounding level
(fallbacks, injected energy, `W`) until 2026-10-21's atmosphere at rest;
under EntropyEOS 2.0 its perturbed runs are clean but not all the same run.

**2026-10-16.** **Investigation (2026-10-16)** *(branch `claude/front-investigation`, not a
PLAN.md step; `bin/front_investigation.jl`, and the 2026-10-17 sweep's
`bin/policy_variants.jl`, removed 2026-10-21 with the decision below (2026-10-21) —
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
`OHM_SIGMA_ZERO` (proposed, for Erik; the proposal is in [CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere)).

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

**2026-10-17.** **EOS-5, as first proposed here, is withdrawn** (Erik, 2026-10-17): its
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
in this package's recovery wrapper on EntropyEOS's new flag — below, 2026-10-21)*:
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
The class of predictors the sweep could add is proposed in [CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere).

**2026-10-21.** **A density raised to the atmosphere's is the atmosphere at rest**
*(decided by Erik 2026-10-21, with EntropyEOS's review; EOS-5 withdrawn,
EOS-6 superseded, the decision kept in this package and keyed on a flag
that reports a fact — the investigation above, 2026-10-16 and 2026-10-17)*. EntropyEOS 1.1.0 (EOS-8,
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
  this recovery ([CODE.md, "The partition"](CODE.md#the-partition) has the re-measured comparison); and the
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

**2026-10-22.** **Investigation (2026-10-22)** *(branch `claude/no-collapse-to-atmosphere`,
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

**2026-10-24.** The section's opening was amended for EntropyEOS 2.0: the
valid range became level 1's and the atmosphere level 2's, with no collapse
ceiling, and the text of 2026-10-21 above was kept as 1.1's record. From
then on the recovery calls `con2prim_safe` and `apply_policy`, the
atmosphere tube no longer sets `collapse_to_atmosphere`, and the rule of
2026-10-21 is unchanged — now `apply_policy`'s own rule on 2.0's
`ρ_FLOORED` ([CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere), "A density raised to the atmosphere's is the atmosphere
at rest").

**EntropyEOS 2.0** *(amended 2026-10-24, EntropyEOS 2.0; tag `v2.0.0`; stage 2 of the
recovery plan, `claude/recovery-pipeline-2`; Symmetry, AMD EPYC 7532 and
7543 nodes, Julia 1.13, `bin/front_investigation.jl` — its new `beyond`
mode runs the rows beyond the tube, the 1.1 control with this file's
`beyond` copied onto the stage-1 tree, `HOT_PROJECTION=0`)*.

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

"Blow-up" is the harness's (defined in [CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere), "The perturbed atmosphere tubes"); under
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

**2026-10-26.** **The ARS(4,4,3) regression: cause, mechanism and fix** *(amended
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
  does, `inside_window` (EOS-11; [CODE.md, "Reconstruction"](CODE.md#reconstruction)). Clamping the root onto
  the window, upstream or here, was rejected upstream: the primitives would
  no longer reproduce the returned conservatives to the tolerance.

Re-measured the same day ([CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere), "The perturbed atmosphere tubes"): every
row deterministic again, but the ideal magnetized atmosphere under SSP3 and
under ARS still differing from 1.1. Not chased; **for Erik**. *(Resolved
2026-11-02: investigated 2026-10-30 and accepted by Erik, below.)*

**2026-11-02.** The ideal magnetized atmosphere's `W = 100` under
ARS(4,4,3), investigated 2026-10-30 on EntropyEOS 2.2.1 ([CODE.md, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere), "The ideal
magnetized atmosphere's `W = 100`"), accepted, decided by Erik 2026-11-02.
Not pursued: a TreeGRRMHD level 2 sending `S_CEILED` states to the
atmosphere still spikes (`W` 27; 65 if limited to `ρ < 1.1 ρ_min`).

## Time integration

### The partition

**Undated.** One IMEX Runge–Kutta integrator over the multi-set state
vector was marked "(decided)". An earlier version of the section called
SSP3(4,3,3) stiffly accurate; it was amended with PLAN.md to say that
neither SSP3(4,3,3) nor SSP2(2,2,2) is stiffly accurate.

**Undated.** IMEXRungeKutta's validation (step 3) measured SSP3(4,3,3)'s
L-stability and stiff-limit order; its A-stability by the E-polynomial was
"computed in step 1".

**Step 5.** The stored `𝔈`'s first-order convergence under
SSP3(4,3,3) and its projected second order under ARS(4,4,3) were measured
in step 5 (`test/wave_tests.jl`). The section then said: "SSP3(4,3,3) stays
the default (proposed in step 5); step 6's stiff-limit and atmosphere runs
are where to compare the two". `evolve!` as described in "As built" was
built in step 5 (`src/driver.jl`); `cfl = 2/5` was proposed in step 5
(TreeHydro's, under SSP3(4,3,3)'s explicit part, which is SSPRK(3,3)).

**Step 6.** "SSP3(4,3,3) stays the default; the question is closed"
(decided in step 6, on the comparison's measurements — Erik left the choice
to them). The cut-off atmosphere tube's stages in step 6's draw: ARS 7
fallbacks and 33 recoveries against SSP3(4,3,3)'s none and 6. In step 6
ARS(4,4,3) in the magnetized ideal atmosphere (`Bʸ = 10⁻³`, `B²/ρ = 100`)
was recorded injecting `2.6·10⁴` of rest mass into a box that holds 0.5 and
running away to `W = 49`.

**Step 7.** `cfl` began to default to `default_cfl(limiter)` (2/5, and 1/5
for MP5), and `after_step(p, u, t)` was added — rung 2a's `∇·𝔅` monitor
(`max_div_B`). The initial data's Gauss average was amended in step 7 to
its difference form. The step-5 form, `Σ w_q f_q` over the `3^D` nodes with
the weights multiplied out, rounded a constant to a neighbouring float, and
was the whole of the 2.7e−15 by which step 5's 2D check missed the 1D run
(see "The right-hand side").

**Step 8.** The current sheet's distance reads 9.559·10⁻⁴ and 9.560·10⁻⁴
(SSP3, ARS) "as it has since step 8's walls — rung 1b —".

**2026-10-21, 2026-10-22.** *(Amended 2026-10-22, decided by Erik: the
decision stands.)* The magnetized ideal atmosphere's runaway under ARS(4,4,3)
recorded in step 6 was EntropyEOS's repairs, not ARS: the entropy-floor
branch wrote `ρ_atm` at `W ≈ |S|/ρ_atm`, the collapse branch `ρ_ceiling`
("Floors, the atmosphere and the magnetosphere"). With a floored recovery
put at rest and the tube's `collapse_to_atmosphere` (2026-10-21) neither
tableau runs away there (the numbers are in `CODE.md`, "The partition"),
and the row no longer argues either way. The others still do, and the
default stands on them. The cut-off tube's stages became deterministic
(before 2026-10-21: 7 and 33 against none and 6 in step 6's draw).

The four atmosphere rows of the comparison were re-measured 2026-10-22
(`bin/tableau_comparison.jl atmosphere`, with the atmosphere-at-rest
recovery and EntropyEOS 1.1.0). The "before" values below are step 6's and,
where they differ, the tree just before 2026-10-21's decision, measured the
same day:

| case (200 cells) | SSP3, before | ARS, before |
|---|---|---|
| rarefaction into `10⁻⁸` (atmosphere `10⁻¹⁰`): min ρ; resets stage/step | — | before 2026-10-21: 2.1·10⁻¹⁰; 152/114 in step 6, 2.7·10⁻¹⁰; 161/118 since step 8 |
| atmosphere tube, `B = 0`: injected `D`, `τ` | — | before 2026-10-21: 2.4·10⁻⁹, 1.2·10⁻⁸ |
| atmosphere tube, ideal, `Bʸ = 10⁻³`: injected `D`, `τ`; max W at the end | before 2026-10-21: 1.6·10⁻⁹, −2.5·10⁻⁷; 4.83 | before 2026-10-21: 2.6·10⁴, 5.3·10¹³; 48.9 in step 6, 1.6·10⁴, 2.7·10¹³; 100 since step 8 — the policy artefact |
| atmosphere tube, `Constant(10³)` cut off at `μ_max = 10`: injected `D`, `τ`; fallbacks; most recoveries | before 2026-10-21: 2.5·10⁻⁸, 6.9·10⁻⁸; 0; 6 in step 6, 5.0·10⁻⁸, 1.7·10⁻⁶; 92; 33 since step 8 — chaotic | before 2026-10-21: 4.6·10⁻⁸, 1.5·10⁻⁶; 7; 33 in step 6, 6.8·10³, 2.4·10¹³; 725; 33 on the step-8 tree — chaotic |

("—": the original table gave no earlier value for that cell.)

### The implicit stage solve

**2026-09-25.** "The solve writes only `E` into `U`" was amended
2026-09-25 (the earlier wording is not recorded).

**2026-10-13, step 3 — The implicit stage solve.** *(Done, 2026-10-13, on `claude/step-3-stage-solve`.)* The review round that followed (2026-10-14) is under "Ohm's law and the conductivity".

`CODE.md`: "The implicit stage solve". Waits: none.

Changes: `src/stage_solve.jl`, with
```julia
solve_ohm_stage(eos, opts, pol, σ, m, cons★, E★, B, γΔt, guesses; iters) -> (E, prim, result, iters)
```
It contains:
- the Sherman–Morrison `E(v)`;
- the fixed-point iteration;
- Newton on `v`, with the Jacobian method decided and recorded;
- the ideal seed;
- the lagged-`v` fallback;
- `OhmStageResult` flags.

Accept:
- **σ = 0:** `E = E★` exactly, and the primitives are exactly the hydro recovery.
- **`IdealConductor`:** `E = −v×B` to round-off, with the ideal recovery's primitives.
- **The sweep:** on the grid `σγΔt ∈ 10^{−3…8}` × `b²/ρ ∈ 10^{−4…4}` × `W ∈ {1, 2, 10}`:
  - the full-system residual is ≤ 1e−10 at every converged point;
  - no NaN appears;
  - the branch map (fixed point, Newton or fallback) and the maximum iteration counts are
    recorded "(measured in step 3)".
- Allocation-free and kernel-callable.

*(Amended in step 3:)* as built, the signature is
`solve_ohm_stage(eos, opts, pol, σ, m, cons★, E★, B, γΔt, guesses; stage) -> (E, prim,
result, recoveries)`, with `stage::OhmStageOptions` for the tolerance and the limits, and
`result::OhmStageResult` two bit words (the solve's branch and modifiers, and EntropyEOS's
policy flags). The iteration runs on the fluid's share of the energy and momentum rather than
on `v`, with the velocity carried as `u = W v`; the Jacobian is analytic; repaired iterates are
replaced by the repaired state's share; `conductivity_slope` (`src/ohm.jl`) gives Newton
`∂σ/∂ρ`. `con2prim_hydro` and `con2prim_ideal` are split (`field_budget`, `fluid_recovery`,
`ideal_recovery`) so that the solve's recoveries are theirs bit for bit. The tests are
`test/stage_solve_tests.jl`, on stage problems built backwards from a known solution.

*(Measured in step 3, Julia 1.11 and 1.13:)* σ = 0 gives `E === E★` and the hydro recovery
`isequal`; `IdealConductor` gives `E = −v × B` to 5.6e−13 of `|B|`. Over the sweep, eight
problems a point at Γ = 4/3, 5/3 and 2: no NaN; at every converged point Ohm's law to 3.3e−16
and the matter to 1.5e−12 of `τ + D`; the fixed point takes every point with `b²/ρ ≤ 10⁻³` in
at most 6 recoveries, Newton the rest in at most 10 where no iterate needed a repair and 32
where some did; the fallback takes 36, 78 (77 at 1.11) and 103 of 2592 points, all at
`b²/ρ ≥ 10` and `σΔt ≤ 10²`, where both seeds are inadmissible but once. Beyond the list: the
analytic Jacobians agree with central differences to 1.1e−8 and 8.3e−8; the stage equations
have more than one root in that corner; under IMEXSSP3433 a resting cell's field decays at
orders 2.92 and 2.96 with its matter bit for bit; the solve is bit-identical in a CPU kernel at
`Float64` and `Float32`, inferred and allocation-free. Every bound on a residual or an error is
at least ten times the measurement, except where a bound is a theorem the measurement attains
(the stiff limit of `ohm_E`, `≤ 1 + 10⁻⁶`); the counts have room of 1.6× (recoveries) and 2×
(fallbacks); `CODE.md` has the rest.

**2026-10-14.** The magnetization cutoff, which takes σ to zero where the
uncut sweep falls back, was decided 2026-10-14.

**Step 3.** The specification was a three-dimensional root find for `v`,
`R(v) = v − V_c2p(S − E(v) × B, τ − ½(E(v)² + B²))`; step 3 amended it:
the same map, iterated on the fluid's share of the energy and momentum
instead.

Before step 3 the section had this open question:

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

Step 3 built `solve_ohm_stage` (`src/stage_solve.jl`) and closed that
question ("decided in step 3, closing the (open) above"): the repaired
state's own share (`fluid_share`), the halved Newton step
(`OHM_BACKTRACKED`) and the flag (`OHM_REPAIRED`); leaving Newton for the
fixed point at a repair, the guess above, was not needed. In step 3 also:
the iteration on the fluid's share (amended in step 3); the allocation-free,
bit-identical kernel at `Float64` and `Float32`, Newton on `X` falling back
on 40 points against Newton on the velocity's 191, the derivatives'
agreement with central differences (1.1e−8 and 8.3e−8), the sweep and the
cost (all measured in step 3); the velocity carried as `u = W v`, the
`conductivity_slope` term, the choice of branch and the convergence test
(proposed in step 3); the analytic Jacobian (decided in step 3). The
section said that "step 5 counts them per chunk" of the
`OhmStageResult` flags.

**2026-10-24.** Under EntropyEOS 2.0 `test/stage_solve_tests.jl`'s bound on
the seeded fallbacks moved from 3 to 12 (a margin of 1.5), the only
assertion of the sweep that moved. Decided by Erik 2026-10-24: the bound of
12 is approved, for the uncut sweep only; the cutoff sweep the scheme runs
is unchanged.

### The IMEX package

**Undated.** IMEXRungeKutta.jl as the integrator was marked "(decided)".

**Undated.** Steps 4a–4c, the IMEX package, are done in its own repository, `~/src/jl/IMEXRungeKutta`,
following its `CODE.md`, its own plan, and its `CLAUDE.md` for the mechanics. They are independent
of steps 1–3 and may run before them.

What TreeGRRMHD needs from it:
- **4b**, which step 5 waits for: the integrator with the contract of `CODE.md`'s "The IMEX
  package", over TreeAMR's flat state vector.
- **4c:** SSP3(4,3,3) and SSP2(2,2,2), validated, with SSP3(4,3,3)'s L-stability and
  stiff-limit order recorded. Amend `CODE.md`'s "The partition" with them.
- A tag (v0.1.0). Registration is Erik's call; until then the package is added by URL.

**2026-09-24.** The specification moved to IMEXRungeKutta's `CODE.md`. The
section then said: "When its validation measures SSP3(4,3,3)'s L-stability
and stiff-limit order, amend The partition." That was done: "The partition"
records IMEXRungeKutta's validation of SSP3(4,3,3).

**2026-09-24.** *(Amended 2026-09-24: the steps' contents moved to that repository.)*

**2026-09-25.** The item on the tendency taken before the stage limiter
(the reset as both stage and step limiter) was amended 2026-09-25.

**2026-10-13, step 0.** *(Amended in step 0:)* done. IMEXRungeKutta is at v1.4.0, with `IMEXSSP3433`, `IMEXSSP222`,
both limiters and the `solve_imp!` contract; `CODE.md`'s "The partition" carries its
measurements. It is a dependency from step 0, through `[sources]`.

**Step 0.** The four per-step counts of IMEXRungeKutta (3 explicit
evaluations, 4 stage solves, 3 stage-limiter calls, 1 step-limiter call)
were measured in step 0. IMEXRungeKutta was a dependency from step 0,
through `[sources]`, so that the contract was checked before step 5 built
on it (proposed in step 0).

**2026-10-24.** With the step cache, the requirement was added that a step
is never retried or rejected, that its step limiter is its last call, and
that an exception leaves `integ.u = uⁿ`.

## Spacetime coupling

### The coupling contract

**2026-11-05.** The step-11 discussion of 2026-11-05. Decided by Erik
2026-11-05: no framework (a script owns the loop); the verbs are each
package's own, under shared names; data passes through pointwise
functions; the adapters live in the producers. Proposed 2026-11-05: the
contract uses only what both packages already depend on.

**2026-11-06.** The section was added 2026-11-06, from that discussion.
The contract of 2026-11-06 had TreeGH's adapter answer `g_ab`, alone or
with `∂_c g_ab`, at every point of any of this package's sets, ghosts
included, 4th order in the outermost ghost layer of a `G = 3` cell set, and
this package supply a covariant `T_ab`. Step 10 built this package's side
of it: `PointwiseMetric(gm, f)` bound the producer's adapter **once per
centering** (cells to `P`, faces to `B[d]`, edges to `A[j]`, vertices to
`Φ̃`; accepted for step 10 by Erik 2026-11-08), and every kernel calls
`point_metric(prov, t, x, Val(C), n, b)` with its point's staggers and
local index. The indexed calls stay: `VertexMetric`, which replaces
`PointwiseMetric`, needs the staggers and the index exactly as they are.

**2026-11-06, step 10.** *(Amended 2026-11-06, decided by Erik: the step no longer reads TreeGH.
It was "The vertex metric and the TOV star in the Cowling approximation", waiting on GH-0, GH-1
and GH-2, with a frozen `GHMetric` provider that decoded TreeGH's vertex state and interpolated it
here. The adapters live in the producers now (`CODE.md`, "The coupling contract", decided by Erik
2026-11-05), so that interpolation is TreeGH's (GH-1), and the frozen TreeGH metric is checked
against this step's run in step 11.)*

This package's verbs were moved from step 11 to step 10 on 2026-11-06, because step 10's script
calls them.

*(Withdrawn 2026-11-06:)* from step 10's changes, `src/vertex_metric.jl`, the frozen `GHMetric`
with its caching decision, the 64-, 16- and 4-point interpolations and the outermost-ghost-cell
option: all TreeGH's now, behind GH-1.

Step 10's acceptance, the contract: *(Amended 2026-11-06: the 4th-order interpolation check moved
to GH-1's upstream acceptance.)*

Step 11: *(Amended 2026-11-06: a coupled run is a script that calls both packages' verbs and
passes data through the producers' adapters, decided by Erik 2026-11-05 (`CODE.md`, "The
coupling contract"). It was a coupled driver in this package that filled TreeGH's `Tsrc`.)*

**2026-11-06.** The sharp edge on adapters was "`T_ab` must be in `Tsrc`
before any TreeGH call that reads it"; amended 2026-11-06 to "an adapter is
a snapshot of its producer's prepared state".

**2026-11-08.** What crosses, and where, was redesigned by Erik 2026-11-08
(TreeGH's PR 25 and its `CODE.md`, "The coupling contract with
TreeGRRMHD"), superseding the contract of 2026-11-06. Decided by Erik
2026-11-08: both directions cross at TreeGH's vertices, in 3+1 variables;
every conversion between centerings is this package's (from this package's
proposal). The adapters "still" live in the producers, but each answers at
the vertices only.

**2026-11-08, step 10.** *(Amended 2026-11-08: the contract redesigned by Erik makes the
interpolation this package's again, from its own vertex set filled at TreeGH's owned vertices;
round 2 of step 10. The step still reads no TreeGH.)* The withdrawn interpolations: *(Amended
2026-11-08: the interpolations and the outermost layer return here in round 2, from `Mv` at
`G = 4`; `GHMetric`'s decoding of TreeGH's private state does not.)*

Round 2 of step 10, "The vertex metric replaces `PointwiseMetric`": *(Added 2026-11-08, the
contract redesigned by Erik; round 2, proposed, for Erik.)*

Step 10's acceptance, the contract: *(Amended 2026-11-08, round 2: the check is this package's
again, and the runs agree to the interpolation's truncation error, not round-off.)*

Step 11's matter adapter, `TreeGRRMHD.ADMMatter` (proposed): amended 2026-11-08, it was a `T_ab`
adapter, `StressEnergy`.

**2026-11-08.** The same sharp edge amended for the 3+1 variables
`(ρ, S_i, S_ij)` and `Mv`, refilled by `vertex_metric!` from each new
`ADMMetric`.

### Two scripts

**2026-11-06.** The two sketches (proposed) were added 2026-11-06.

**Step 10.** The step-10 sketch was built in step 10 as
`bin/tov_cowling.jl` (`CODE.md`, "Two scripts", "As built").

**2026-11-08.** The comment on `metric_at` in step 11's sketch ("this
package's vertex set filled from it …, which every kernel interpolates")
was amended 2026-11-08, with the contract's redesign. The repeated-fill
item was amended 2026-11-08: each `metric_at` "now also" fills this
package's vertex metric set, so the same skip saves twice as much.

### Metric providers

**Step 2.** `Flat` and `Analytic` were built in step 2 (`src/metric.jl`).
The conversion from SpacetimeMetrics' derivative-last `dmetric` in
`src/metric.jl` and nowhere else was proposed in step 2. `K_ij` was
computed there from the same `dmetric` call: SpacetimeMetrics'
`ExtrinsicCurvature` differentiated the metric a second time and allocated
432 bytes per call at Julia 1.11 and 1.13, which a device kernel cannot
(measured in step 2); the two agreed to 4.3e−16. That local computation was
the workaround of SM-1 (amended in step 2).

**2026-10-14.** SM-1 landed in SpacetimeMetrics 1.7.0;
`metric_gradient(::Analytic, …)` now calls its allocation-free
`ExtrinsicCurvature(g, dg)` on the same `dmetric` result, and the pointwise
layer is still bit-identical in a CPU kernel.

**Step 5, step 9.** Every kernel since step 5 already called
`point_metric(provider, t, x)` at its own point, so rung 3 needed no change
in the scheme, only the cases. The cost of the `Analytic` provider was
measured in step 9 (`bin/metric_cost.jl`), and the choice (pointwise, or a
per-stage cache) was proposed in step 9, for Erik.

**2026-11-03.** Resolved 2026-11-03, decided by Erik: pointwise, no cache;
the background is stationary, but the analytic cases are affordable. The
same day Erik asked for the deduplication of the vertex and edge
evaluations; it was not done, and measured why. Sharing across invocations
by storage of the metric, or of `ψ`, at every vertex and edge was excluded
by the request. Decided by Erik 2026-11-03: leave it; it folds into step
10's caching decision. Caching was step 10's, deferred by Erik 2026-11-03.

**2026-11-03, step 10's plan.** The Cowling star's provider is `Analytic(TOVMetric)`, pointwise
like every analytic provider (decided by Erik 2026-11-03).

**2026-11-06.** The producer's pointwise metric was `GHMetric`, TreeGH's
vertex-centered `(h_ab, Π_ab)` decoded and interpolated here, frozen for
the Cowling runs and evolved for the coupled ones, one code path for both.
The Cowling star runs on `Analytic(TOVMetric)` instead, decided by Erik
2026-11-06, and step 11 checks the frozen TreeGH metric against that run.
From 2026-11-06 caching was the producer's decision, and the deduplication
folded into it.

**Step 10.** `Analytic(TOVMetric)` was built in step 10, and its per-call
cost was measured in step 10. The list of providers then said: "A
producer's pointwise metric, in the coupling contract: TreeGH's (GH-1,
step 11), and a sampled SpacetimeMetrics metric in the same contract, which
tests this package's side of it (step 10)."

**2026-11-08.** The contract redesigned: the producer's metric is
`VertexMetric` now (proposed). It replaces step 10's `PointwiseMetric`, and
it is close to the `GHMetric` of before 2026-11-06, with TreeGH's private
`(h_ab, Π_ab)` replaced by the 3+1 variables, decoded on TreeGH's side.
Caching is this package's decision again (amended 2026-11-08).

### Metric at cells, faces and edges

**Step 2.** That the derivatives and `K_ij` are not stored but evaluated
where the sources are was proposed in step 2.

**Step 9.** The outermost ghost layer was open since step 9.

**2026-11-06.** From 2026-11-06 it was TreeGH's, behind a `g_ab` adapter
answering at every centering.

**Step 10.** Before 2026-11-06 the metric at cells, faces and edges was
this package's, in step 10.

**2026-11-08.** This package's again since 2026-11-08, decided by Erik in
the contract's redesign. `N ≥ 10` for every coupled run (TreeAMR's vertex
invariant `N ≥ 2G + 2` at `G = 4`) was found by TreeGH 2026-11-08. The
ghosts' PointValue width 6 was to be "measured in the step that builds
it". The fallback to multilinear weights was proposed by this package,
recorded by TreeGH 2026-11-08, with "a tally in `X`" counting it. Amended
2026-11-08: the outermost ghost layer is settled by `Mv`'s `G = 4`; GH-5
stays withdrawn. The design then had this open item:

**The derivative layout** (open). `MetricGradient` holds `∂_j g_ab` and
`K_ij`, with `d_lapse` and `d_shift` derived from them. The vertex metric
gives `∂α`, `∂β`, `∂γ` directly, so either `MetricGradient` gains them
natively (and `Analytic` converts `dmetric` to them) or `src/metric.jl`
assembles `∂_j g_ab` by the product rule, a second conversion beside
`spatial_gradient`, round-tripped by `d_lapse` and `d_shift`. Decided by
the step that builds it, bit for bit against the step-10 runs where the
kernels have not changed.

**2026-11-08, step 10, round 2.** Step 10, round 2 built `Mv`
(`src/vertex_metric.jl`) and decided the derivative layout: `MetricGradient`
unchanged, `gradient_from_adm` by the product rule. The fallback is counted
by `metric_fallbacks`, not by a tally in `X`. `PointwiseMetric` is gone; the
indexed calls stay.

**2026-11-08, step 10, round 2.** *(Built 2026-11-08, `src/vertex_metric.jl`,
`test/vertex_metric_tests.jl`; `CODE.md`, "Metric at cells, faces and edges", "As built in step
10, round 2".)* Departures from step 10's plan for round 2: `Mv` lives in its own
`VertexMetricSet(T, forest; boundary)`, not in `MHDFields`, so `MHDProblem` is unchanged;
`vertex_metric!(vm, adm, t)` takes the fill's time; the fallbacks are counted by
`metric_fallbacks`, a diagnostic pass, not in `X`; the derivative layout is the product rule
(`gradient_from_adm`), `MetricGradient` unchanged; the sampled adapter is `SampledADM` in `src/`,
since the study uses it too; `regrid_pairs` is `public`. `bin/tov_study.jl metric=vertex
state=…` and `bin/tov_compare.jl` make the 200 M comparison.

### Matter to the vertices

**2026-11-05.** The estimate of the stored form's reads (about 80 values
per vertex) against the per-call form's (about 5000 reads) was made
2026-11-05, for `T_ab`.

**2026-11-06.** The average was written into `Tsrc`, a vertex set TreeGH
read. The alternative of 2026-11-06 averaged eight cells' `T_ab`, each
assembled with that cell's interpolated lapse and shift.

**2026-11-08.** Amended 2026-11-08, the contract redesigned: the section
was "Stress-energy to the vertices", the covariant `T_ab` averaged; now
`(ρ, S_i, S_ij)` are averaged and TreeGH assembles `T_ab` with the vertex's
own `α` and `β`.

### Gauge for the TOV milestones

**2026-11-06, step 10.** `TOVMetric` as this package's provider for the
Cowling star, `Analytic(TOVMetric)`, with no TreeGH: step 10, decided by
Erik 2026-11-06. `src/tov.jl` and `src/cases/tov.jl` were built in step 10
(`CODE.md`, "Gauge for the TOV milestones", "As built", proposed).

## The right-hand side

**Step 5.** As first built (`src/evolution.jl`, `mhd_rhs!`), the kernels were
written for `D = 1, 2, 3` and step 5 validated `D = 1`. Steps 2 and 3 of the
order were one kernel over the stored cells (`primitive_kernel!`): the
provider's metric at the cell, `B` from the faces, `E = 𝔈/√γ`, the recovery
from `state_guesses` (see "Thermodynamics"), all thirty slots of `P`. `q` was
a second kernel over the owned cells, by centered differences of the
ghost-filled `𝔈` (proposed in step 5). The second-order averages of the
potentials' right-hand side were proposed in step 5.

Cost (measured in step 5, Julia 1.13, Apple M3 Pro, one thread, the
Alfvén wave on 400 cells in 40 blocks): 0.77 ms for a right-hand side, of
which the recovery over the 640 stored cells is 0.55 ms; 0.39 ms for an
implicit stage (`IdealConductor`) and 0.34 ms for a reset, which also
recovers every cell. A step of SSP3(4,3,3) — 3 right-hand sides, 4
stages, 4 resets — is 5.2 ms, a quarter of it the resets.

In 2D and 3D, a run of either wave that is translation-invariant in `y`
(and `z`) agreed with the 1D run to 2.7e−15 after seven steps, in a check
made while writing the kernels and not in the suite — compiling the 2D and
3D paths costs about 30 s, and their validation, bit for bit, was step 7's.

**2026-10-15, step 8.** The order of the right-hand side (proposed in the
design) was amended in step 8, when AMR-10 let the matter's ghost fill take
the primitives, decided by Erik 2026-10-15. Until step 8 the exchange was in
conserved variables, as TreeHydro decided, and `con2prim` ran over every
stored cell. Step 8 made step 3 three passes (`primitive_kernel!` over the
owned cells, the fill of `W`, `ghost_kernel!` over the ghosts); that owned
cells take `P` through `P_from_W` too was proposed in step 8.

**Step 7.** The whole difference between the 2D/3D runs and the 1D run was
the initial data's Gauss weights (see the driver in "The partition"); with
the average in difference form the 2D and 3D runs are the 1D run bit for
bit, under every reconstruction, and nothing in the right-hand side changed
(measured in step 7, "Validation ladder" 2a).

**Step 8.** The sharp edge "The RHS never mutates `u`" said that the stage
limiter writes only atmosphere cells; amended in step 8: also the
potentials' owned low walls at the low domain faces, which the reset copies
into `u` (`walls_to_state!`).

**2026-10-23.** Until 2026-10-23 `ghost_kernel!` also wrote the conserved
ghost by `prim2con`, which nothing read; the conserved ghosts are no longer
written (amended 2026-10-23, decided by Erik).

**2026-10-24.** That the right-hand side is pure in `(u, t)` outside a step,
and inside a step of `evolve!` recovers through the step cache so that a step
is pure in `uⁿ`, was amended 2026-10-24, decided by Erik 2026-10-23. The
implicit stage's seeding from the step cache was added the same day
(amended 2026-10-24).

**2026-11-06.** Step 5 of the order used to be "`T_ab` at cells, averaged to
vertices into `Tsrc`, and TreeGH's right-hand side, if evolved" (amended
2026-11-06: nothing of TreeGH's; in a coupled run the script calls TreeGH's
right-hand side, reading through this package's adapter).

**2026-11-08.** Until 2026-11-08 TreeGH's right-hand side read `T_ab`
through this package's adapter; since the redesigned coupling contract it
reads `(ρ, S_i, S_ij)`.

## Conservation at coarse-fine faces

**Step 8.** First measured in step 8 (`test/refinement_tests.jl`; MC, HLLE,
SSP3(4,3,3), static two-level meshes and adaptive ones), before the
potentials' conservative transfer and the matter's ghost fill in
primitives:
- The totals: the Alfvén wave of rung 1a on 64 cells with the middle half
  refined, a quarter period, drifted 0.008 ulp of a total's scale a step
  with the flux fixup and 1.5·10¹⁰ without; along the diagonal on 2 × 2
  roots of 8² with one refined, 0.12 and 3.3·10¹¹. Balsara 1 on its box made
  periodic, tracked by two levels more through ten regrids, 0.017 and
  1.4·10⁷; the rotor, adaptive, 0.036 through two regrids.
- The magnetic flux, as a rate: to 1.7e−18 of a largest rate of 5.2e−3 (2D,
  Orszag–Tang's initial data, 16 faces) and 8.7e−19 of 7.8e−3 (3D, the
  spherical blast in its closed box, 192 faces) with the correction, and
  0.025 of 0.029 and 0.0037 of 0.0078 without it. Stating PLAN.md's step-8
  claim as the rate was proposed in step 8.

The current numbers were re-measured at the end of step 8, with the
potentials' conservative transfer and the matter's ghost fill in
primitives. "Which block's EMF" (accepted, not corrected) was proposed in
step 8.

The magnetic flux as a state was not to rounding under the PointValue
operators step 8 began with; the conservative transfer was requested as
**AMR-9**, which landed in TreeAMR 0.2.2. The state's measurement with it
was made in step 8, and its negative control (the PointValue family) was
proposed in step 8. "Not at an outer face" was measured in step 8 and
proposed there as accepted and recorded; **AMR-11** was added to `PLAN.md`
in step 8.

## Boundaries

**2026-10-14.** Outflow conditions were waiting for AMR-6; TreeAMR 0.2.1 has
outflow faces (amended 2026-10-14).

**Step 5.** `MHDProblem`'s `boundary` (`nothing` or one hook per evolved
set), AMR-2 having landed. Step 5's waves were periodic; step 6's shock
tubes were the first to pass hooks.

**Step 6.** `DirichletBoundary` and `case_boundary` in `src/driver.jl`
(proposed in step 6). The uniform flow at `vˣ = 0.999` that lost a fifth of
its density in the last cell under a Dirichlet `A` was measured in step 6.
The potentials' hook, Dirichlet in the field, first read the interior in
TreeAMR's region form, CPU-only, and was requested as AMR-7 in `PLAN.md`.

**Step 6.** The sharp edge on the two abscissae, and with it the
potentials' Dirichlet data in the field, was added in step 6.

**2026-10-15, step 8.** *(Amended in step 8; decided by Erik 2026-10-15,
"option 1":)* both walls are prescribed. Both the low wall and the points
beyond an outer face were this package's kernels until TreeAMR 0.2.2
(`low_walls!`, `outer_points!`, and on a multilevel mesh a second, hookless
fill and `outer_points!` again). That one fill applies the complete
condition was decided by Erik 2026-10-15: "an outer boundary condition needs
to be applied always".

*Measured in step 8* (with this package's kernels): the explosion in a box
its precursors reach — `[−2, 2]²` at 32², minmod, `t = 1/4` — was 7.1e−6
(`x → −x`) and 6.7e−5 (`y → −y`, fields reversed) of a variable's scale from
its mirror image before, and the spherical blast in the closed 16³ box
1.3e−2 and 1.0e−1; with both fixes both are their own mirror images bit for
bit. The 3D smoke test moved to the closed box. The 2D explosion at 64² on
`[−6, 6]²` and the rotor, symmetric to 2.5e−17 and 2.6e−27 in step 7, became
exactly symmetric too. Every other number the suite prints moved in its last
digits at most — the shock tubes' L1 errors and boundary budgets, the
uniform flows through the faces (1.33e−12 → 1.24e−12 of a variable's scale
at most), the current sheet's rates — since their faces see nothing but
precursors before `t_end`; the one exception is the current sheet run on to
`t = 9`, whose field has diffused to the faces by then: its L1 distance to
the erf profile went from 9.30e−4 to 9.56e−4 of `B₀`, the low face's `𝔅`
now the profile's as the high one's was.

"The low walls in the state" was proposed in step 8, at the reviewer's
request. That a refined closed box need not be mirror-symmetric before the
stars was proposed in step 8.

**2026-10-15, step 8.** The outer boundary condition is applied in full
after every fill (decided by Erik 2026-10-15). The sharp edge was amended in
step 8 twice: first this package's `outer_points!` and a second fill, then
TreeAMR's — since TreeAMR 0.2.2 one TreeAMR fill does it, the potentials'
relative hook (AMR-7) and symmetric walls (AMR-8).

**Step 7.** *(Measured in step 7:)* the potentials' hooks were not
mirror-symmetric. The box's low boundary vertex is a block's first owned
point and was evolved, while its high boundary vertex is the last block's
shared plane, which no block owns and the hook fills from the vertex before
it — so `𝔅` on the high boundary face was the case's, and on the low one the
evolution's. As long as nothing but the light-cone fluxes' exponentially
small precursors reaches the faces the difference is that small: a 2D
explosion 30 cells from its faces kept its mirror symmetries to 2.5e−17, but
a spherical blast in a 16³ box, whose precursors reach the faces within a
step, came out 1.3e−4 asymmetric there after three steps, and the 3D smoke
test ran periodic. Whichever later rung needed a symmetric closed box would
have to fill the high boundary vertex from the evolved side, or evolve it.

**Step 8, TreeAMR 0.2.2.** *(Amended in step 8:)* since TreeAMR 0.2.2 the
potentials' hook is TreeAMR's relative hook (AMR-7), a kernel; the CPU-only
restriction came from the region form alone and is gone. The low walls and
the outer points are TreeAMR's (`walls = :symmetric`, AMR-8, and the relative
hook, AMR-7). A regrid through `regrid_state!` transfers bit for bit what
step 8's first arrangement (a complete fill of this package's, then
`regrid!` with no hook for the potentials) transferred. The hook evaluates
`f(x_g) + (u(x_r) − f(x_r))` where `outer_points!` evaluated `u(x_r) + (f(x_g)
− f(x_r))`, so the potentials' outer points moved by rounding: every number
the suite prints for a non-periodic case moved in its last digits at most
(see "Validation ladder"). *(Measured in step 8, with TreeAMR 0.2.2's
symmetric walls and relative hook:)* both closed boxes are still their own
mirror images bit for bit, and nothing printed by a uniform-mesh case moved —
the uniform flows, the shock tubes' budgets, the current sheet (9.56e−4) and
the atmosphere tubes to every digit.

**2026-11-03, step 9.** "The potentials' boundary must pin the gauge or keep
a gradient a gradient" is a design rule from step 9's round 2; decided by
Erik 2026-11-03: option 1 for static potentials (`MHDCase(…;
static_potential = true)`, decided by Erik 2026-11-03).

**2026-11-03, step 9.** *(Added in step 9, decided by Erik 2026-11-03:)* step 10's plan decides
the potentials' outer boundary.

## The interior

**2026-11-02, step 9.** The rules for resets inside the hole (not counted
deep inside, counted in radial layers, no special cases) were decided by
Erik 2026-11-02, with a fourth: "The paste stays as built (both limiters,
`r_in = M`) for now; its form is not decided." *Note (round 2):* recorded
that the first table was measured under the relative potential hook, whose
boundary mode dominated the outside error (the note, with its numbers, is in `CODE.md`,
"The interior"), and announced that round 3 measures the interior again.

**Step 9.** The interior was added in step 9, proposed in step 9 throughout,
the form a decision for Erik. PLAN.md's step 9 asked for TreeGH's answer.
The first measurement (`bin/curved_stationarity.jl`, jobs 588101–588103) was
made under the relative potential hook.

**2026-11-03, step 9, round 3.** "No interior, the ordinary reset inside"
(option 3) was measured in step 9, round 3, for Erik's open decision on the
interior. Round 1's `−4.0·10⁻³` beyond `r = M`, without the interior, is the
net charge of the singularity's neighbourhood that this measurement shows.
Decided by Erik 2026-11-03: the hard paste, both limiters, `r_in = M`, for
the analytic rung-3 cases (proposed in step 9; round 3's measurement).

## Refinement

**2026-10-14.** *A design point for steps 8 and 10* (recorded 2026-10-14;
nothing was redesigned yet): the paragraph on TreeAMR's bounded
prolongation (AMR-4) and the split it expects, now in `CODE.md`,
"Refinement", ended: "Steps 8 and 10 must decide whether to follow that split — which set the ghost fill
acts on, how `P` is refilled after a fill, and what the reset limiter does to
a prolongated conserved state — and record it here." Step 8 followed it.

**Step 5.** `A` and `Φ̃` PointValue width 4, prolongation and restriction,
was step 5's proposal; step 8 measured it.

**2026-10-15, step 8.** *(Amended in step 8, decided by Erik 2026-10-15:
"choose the same ξ on all meshes":)* `ξ = 16/L` on every mesh. The complete
outer boundary condition before `regrid!` (and no hook for a potential in
`regrid!`) was amended in step 8, decided by Erik 2026-10-15. The matter's
ghost fill in primitives was decided by Erik 2026-10-15.

*As built in step 8, when AMR-10 landed* *(superseded 2026-10-23 by the
parent-centred fill; the measurements stand as measured)*
(`primitives_from_sets!`, `P_from_W`, `ghost_kernel!` in `src/evolution.jl`):
the set `W` (proposed in step 8, with Erik's list of variables) and the
recovery on the owned cells only, as now, but at every ghost cell also the
conserved ghost from `prim2con`, and `W` filled with PointValue operators:
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
since)*: the primitive fill keeps every ghost admissible — no ghost the
policy would repair, none below the atmosphere, magnetized as well (the
AMR-10 downstream check, against 33 repaired recoveries with conserved
ghosts) — but on the hydrodynamic tube it injects 5.6e−9 of rest mass where
the bounded conserved fill injected 1.5e−11 and the uniform mesh 8.4e−12:
mixing primitives (a coarse ghost restricted from a dense and an atmosphere
cell) makes more internal energy than mixing the conserved state, which is
the energy-conserving mixture. Not an error — the reset's injection is a
correction the scheme accounts — but larger than step 8's first fill, and
an argument for prolongating the pressure instead of the entropy, which
needs the entropy from `(ρ, p)` (EOS-3), or for keeping width 2. The fill
ran as Erik decided, at width 2, until that was settled. *(Settled
2026-10-23 by Erik: the parent-centred fill.)*

Smooth data across coarse-fine faces with the width-2 primitive fill (the
Alfvén wave of "The operators", potentials at width 4,
`bin/interface_order.jl ghost=w`), `S_y` / `S_x` L1 / L∞ rates:

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

**2026-10-15, step 8 — Coarse-fine faces and regridding (rung 2b).**

`CODE.md`: "Conservation at coarse-fine faces", "Refinement", "Field sets".
Waits: AMR-2, AMR-3 (workarounds allowed).

Changes:
- **The refined paths:** `InterfaceSchedule` and `restrict_interfaces!` on both `F_d` and
  `EMF_d`.
- **Regrid over every set:**
  - `U`: Conservative, width 3;
  - `A` and `Φ̃`: PointValue, width `w`;
  - `P`, `B_d`, `F_d`, `EMF_d` and `Tsrc`: `=> nothing`.
- **The indicator:** Löhner on `ρ`, `|B|` and `|E|`.
- **The driver:** chunked, with a regrid between chunks.

Accept:
- **Static two-level mesh:** every conserved integral is conserved to round-off with the fixups,
  and leaks without them.
- **Magnetic flux:** the flux through a coarse face equals the sum through its fine faces to
  round-off with the EMF correction, and does not without it.
- **∇·B** at round-off on both levels, and after every regrid.
- **Field loop across a coarse-fine face:** energy and shape are recorded against a uniform-fine
  control.
- **The interface-order table** of the Alfvén wave crossing a coarse-fine face, for `A`/`Φ̃`
  widths `w ∈ {2, 4, 6}`. It **decides the (open) width**.
- **Spurious `B`** at refinement boundaries against the gauge damping `ξ` **decides `ξ`**.
- **A tracked shock tube** with regrids stays conservative.

*(Done, 2026-10-16: on `claude/step-8-refinement` on 2026-10-15 but for what waited on AMR-7 to
AMR-10, and the rest on `claude/step-8-finish` once TreeAMR 0.2.2 had them — see "The finish"
below.)* *(As first reported, 2026-10-15: done except **the ghost-fill re-measurement, blocked
on AMR-10**: Erik decided on 2026-10-15 that the matter's ghost fill prolongates the primitives
with a bounded Lagrange interpolation, which TreeAMR does not offer yet; the conserved, bounded
ghost fill runs until then, and step 8's atmosphere table is re-measured with the primitives when
AMR-10 lands.)* *(Amended in step 8:)* as built:
- **First, Erik's decision of 2026-10-15 ("option 1")**: both walls of a closed box are prescribed
  (AMR-8's workaround, `low_walls!`, and `outer_points!` for the corner rows the region-form hook
  got wrong; `CODE.md`, "Boundaries"), so a closed box is mirror-symmetric bit for bit and the 3D
  smoke test runs closed. New upstream item **AMR-8**; AMR-7 amended.
- **The refined paths** needed nothing new in the right-hand side: `restrict_interfaces!` on `F_d`
  and `EMF_d` was there from step 5. `src/refinement.jl`: `mhd_flags` (Löhner on `ρ`, `|B|`,
  `|E|`), `regrid_pairs`, `regrid_state!` (regrid every set, rebuild, gather, low walls, the
  post-regrid reset as step limiter); `rebuild(p; …)`; `evolve!` gains `refine` (a static mesh),
  `maxlevel`, `refine_tol`, `coarsen_tol`, `buffer`, `ε`, `ε_g`, `maxpasses`, `ops_U`, `ops_A`,
  `fixup`, `roots`, `after_regrid`, and returns `nregrids`, `passes`, `nblocks`, `totals0`;
  `adapt_case!` (AMR-3). `gauge_pulse_case` (`src/cases/gauge.jl`). `bin/interface_order.jl`,
  `bin/gauge_damping.jl`, `bin/field_loop_refined.jl`.
- **Decided by measurement** (`CODE.md`, "Refinement"): the potentials' width 4; the matter's
  width 3 *bounded*, in the ghost fill and the regrid; `ξ = 16/L` on every mesh (`default_ξ`, `L`
  the box's longest side; amended, decided by Erik 2026-10-15 — first 0 on uniform meshes); the ghost fill in conserved variables, bounded, measured — and
  *(decided by Erik 2026-10-15)* to become primitives with a bounded Lagrange prolongation and
  the conserved ghosts by `prim2con`, once AMR-10 lands; the regrid stays conservative.
  *(Amended 2026-10-23, decided by Erik:)* the primitives are copied at same-level faces and
  prolongated with the bounded conservative operator centred on the parent; the conserved ghosts
  are no longer written (nothing reads them); `ghost_vars` is gone (`CODE.md`, "Refinement",
  which records the three working fills).
- Tests: `test/refinement_tests.jl` (the AMR-4 surface moved here, conservation, the magnetic flux,
  the tracked tube, `∇·𝔅` through regrids, the flux through the plane) and
  `test/refinement_cases_tests.jl` (the field loop, the interface order, the gauge pulse, the
  atmosphere across a coarse-fine face); `test/upstream_behaviour_tests.jl` removed, empty.

*(Measured in step 8; `CODE.md`, "Conservation at coarse-fine faces", "Refinement" and "Validation
ladder" 2b, has the rest:)*
- **Static two-level mesh:** every total but `𝔈`'s to 0.008 (1D) and 0.12 (2D) ulp of its scale a
  step with the fixups; 1.5·10¹⁰ and 3.3·10¹¹ without.
- **Magnetic flux:** the rate through a coarse face equals the sum through its fine faces to
  1.7e−18 (2D) and 8.7e−19 (3D) of the largest rate with the EMF correction, 85% and 48% of it off
  without. As a state it does not hold to rounding: the flux through the plane drifts 5.8e−4 of its
  scale across a level, the potentials' interpolation along an edge — **AMR-9** (new, wanted).
- **∇·B:** at most 2.0e−16 of `max |𝔅|` on every level, after every step and every regrid.
- **Field loop across a coarse-fine face:** energy and `A_z` error between the uniform coarse and
  fine controls (0.801 against 0.623 and 0.844 of the initial energy at `t = 1/2`; 0.584 against
  0.206 and 0.619 at `t = 4`).
- **Interface order:** width 2 loses the order at the face (L∞ 0.91 in 1D, 0.33 in 2D), width 4
  keeps it (2.04, 2.02), width 6 is width 4 — **the width is 4**.
- **`ξ`:** a pure gauge pulse across a coarse-fine face leaves 5.3e−2 of spurious `|𝔅|` after two
  light crossings undamped, 1.4e−3 at `ξ = 16/L`, with no effect on the Alfvén wave's errors —
  **`ξ = 16/L`**, on every mesh (decided by Erik 2026-10-15; on a uniform mesh it moved only the
  2D oblique wave's `A_x` errors, 2.77 → 2.769 and 2.05 → 2.07 in rate, and last digits elsewhere).
- **Tracked shock tube:** Balsara 1, periodic, ten regrids, 0.017 ulp a step (1.4·10⁷ without the
  fixup), nothing repaired.

**Step 8.** *As built in step 8* (`src/refinement.jl`, `evolve!` in
`src/driver.jl`): proposed in step 8 unless marked otherwise. The operators
were decided in step 8 (the matter Conservative width 3, bounded, then in
its ghost fill and its regrid; `A` and `Φ̃` PointValue width 4). The
interface-order table was first measured with `A_j` PointValue; after AMR-9
it was re-measured, and the width stayed 4 (decided in step 8 again). With
the matter's ghost fill in primitives at width 2 as well (AMR-10) the
potentials' widths 2, 4, 6 gave 1.91 / 0.88, 2.09 / 1.97, 2.08 / 1.97 in 1D
and 1.45 / 0.08, 2.13 / 1.85, 2.13 / 1.84 in 2D: width 4 again.

The gauge damping was decided in step 8. As first decided it was 16 over
the shortest side on a refined mesh and 0 on a uniform one. The pure gauge
pulse was first measured with the PointValue potentials of step 8's first
form (2 × 2 roots of 16² with one refined, to `t = 2`):

| `ξ` | 0 | 1 | 4 | 16 | 64 |
|---|---|---|---|---|---|
| `max |𝔅|` over the run | 0.107 | 0.103 | 0.093 | 0.079 | 0.054 |
| `max |𝔅|` at `t = 2` | 5.3e−2 | 2.1e−2 | 6.6e−3 | 1.4e−3 | 3.3e−4 |
| `max |Φ̃|` at `t = 2` | 0.25 | 8.3e−2 | 4.0e−3 | 9.4e−5 | 5.2e−4 |

against at most 5.3e−15 on the uniform mesh at any `ξ`. Without damping the
pulse kept crossing the faces of the periodic box and the spurious field
stayed (at 8² roots it grew, 0.079 at `t = 1` to 0.18 at `t = 2`). The
oblique Alfvén wave on the same mesh moved in the fifth digit of its L1
errors at most for any of these `ξ`. 16 was where the remainder had fallen
by 40 and `Φ̃` by 2600 within two light crossings; at 64 the field's
remainder was smaller but five times more `Φ̃` was left at `t = 2`. In the
suite's 8² version the undamped field at `t = 2` went from 0.18 to 0.049 and
the damped one from 1.2e−3 to 1.6e−5 with AMR-9 and AMR-10 (0.044 and 1.9e−5
with AMR-9 alone).

**Step 8, the ghost-fill experiments.** *Experiment (branch
`claude/step-8-ghost-entropy-density`; decided against 2026-10-23):* `W`
holding the densities `ρ★ s`, `ρ★ Yₑ` (`ghost_vars = :density`, the default
on that branch), and also `ρ★ uⁱ` (`:mass`), so that with common weights the
ghost's entropy is mass-weighted; `bin/ghost_entropy.jl` has the
measurements, reported to Erik. In short: at width 2 every ghost's pressure
and entropy stay within their neighbours' range and the magnetized tube
never blew up in twelve perturbed runs (the specific-`s` fill did, twice),
but the hydrodynamic tube injects 1.6e−8 of rest mass, more than the
specific fill — the injection is not the entropy's but any point-value
fill's: a fine ghost inside a coarse cell at a front holds mass the coarse
cell does not (2.96e−6 against 1e−8), which the light-cone HLLE flux draws
out of the coarse cell through the coarse-fine face. Widths 4 and 6 are not
robust in either form.

The *second experiment* (branch `claude/step-8-ghost-athena`) was the
parent-centred fill, the code since (decided by Erik 2026-10-23); its table
in `bin/ghost_entropy.jl` was reported to Erik. The *third experiment*
(branch `claude/step-8-ghost-robustness`, from `claude/step-8-ghost-athena`)
measured the three fills' robustness over 48 perturbed runs; both are in
`CODE.md`, "Refinement".

**Step 8.** `PLAN.md`'s sharp edges said: **`regrid!` takes one boundary
hook for all sets** (until AMR-2); also, `adapt_to_initial_data!` adapts one
set (until AMR-3); the workarounds are in those items. Amended in step 8:
both have landed and are used without workarounds: `regrid_pairs` gives
each evolved set its own hook, `adapt_case!` adapts every set at once. The
sharp edge on a regridded run's initial totals (`evolve!`'s `totals0`) was
added in step 8.

**2026-10-16.** *(Investigation 2026-10-16, "Floors, the atmosphere and the
magnetosphere":)* the `W` spikes in the robustness runs are EntropyEOS's
`S_FLOORED` repairs, the blow-ups its collapse branch, on every mesh.

**2026-10-16, step 8, the finish.** *The finish (2026-10-16, `claude/step-8-finish`, on TreeAMR 0.2.2; `CODE.md` has the details,
marked "amended/measured/proposed in step 8"):*
- **TreeAMR 0.2.2** (`TreeAMR = "0.2.2"`): the probes of AMR-7 to AMR-10 retired, the items
  marked landed; `Relative` joins the exported-name list.
- **AMR-7, AMR-8**: the potentials have `walls = :symmetric` and the relative hook;
  `low_walls!`, `outer_points!`, the region-form hook, the second fill and the hookless
  `regrid!` are gone. One fill is the complete condition's fixed point (from `NaN` too), and
  `regrid!` with the hooks transfers bit for bit what the old arrangement did. The low wall is
  owned but prescribed: the RHS writes zero there, and the reset copies the filled walls into
  `u` (`walls_to_state!`; tested: 190/214 wall entries held bit for bit after every step, `du = 0`
  there, a perturbed wall changes nothing). Non-periodic cases run on every backend (not tried on
  a device here). Uniform closed boxes stay mirror-symmetric bit for bit; a refined one is
  asymmetric by 1.1e−2 — TreeAMR's open half-open item, recorded.
- **AMR-9**: `A_j` conservative along the edge with point values of width 4 across it, `Φ̃`
  PointValue width 4 (`ops_Φ`). The magnetic flux holds as a state: the plane's flux to 2.2e−16
  across levels and through a regrid, every coarse face's flux its fine faces' sum to 1.4e−16 in
  3D, with or without the EMF correction (the transfer's property; PointValue: 1.5e−3) — except
  at an outer face, 1.2e−5 (AMR-11, new, wanted). The interface order re-measured: width 4 stays.
  The gauge pulse's spurious field at `t = 2` fell 3-fold at `ξ = 0` and 190-fold at 16/L.
- **AMR-10**: the matter's ghost fill prolongates `W = (ρ★, s, Yₑ, uⁱ, Eⁱ)`, PointValue bounded,
  **width 2** (proposed), with its own hook; `P` in ghosts by forward EOS evaluation, conserved
  ghosts by `prim2con` with the ghost's own `B`; the regrid stays conservative. No ghost
  inadmissible on the atmosphere tubes (magnetized: against 33 repaired with conserved ghosts);
  widths 4 and 6 clamp `ρ` and `s` independently and inject energy (5.4e−4) or blow up; width 2
  injects 5.6e−9 of rest mass where the conserved fill injected 1.5e−11 — **for Erik**
  *(resolved 2026-10-23, decided by Erik: the parent-centred primitive fill)*. Smooth
  order across coarse-fine faces kept (2D L∞ 1.85 against 2.02).
- **Found**: the magnetized atmosphere tube is chaotic at rounding level (0 to 144 fallbacks,
  `W` 4.6 to 90 under 10⁻¹⁵ perturbations, before and after); its "no fallback" assertion now
  holds hydrodynamically, with a fallback rate ≤ 0.2% magnetized (proposed). *(Decided by Erik
  2026-10-22: "no fallback" magnetized too, on the uniform and the refined mesh of the suite —
  the tube is deterministic under the atmosphere-at-rest recovery of 2026-10-21; `CODE.md`,
  "Floors, the atmosphere and the magnetosphere".)*

**2026-10-21.** *Decided by Erik 2026-10-21:* a floored state is the
atmosphere at rest (EntropyEOS 1.1's `FLAG_POL_ρ_FLOORED`, in the recovery
wrapper) and the tube's policy sets `collapse_to_atmosphere`; the robustness
table stays the record of the released policy, and the ghost fill's choice
is unchanged and Erik's.

**2026-10-23.** Decided by Erik: the parent-centred primitive fill, the only
one since. *(Amended 2026-10-23: until then `ghost_kernel!` also rebuilt
every conserved ghost by `prim2con`, which nothing read; step 8's PointValue
width-2 fill of `W` was the default until the parent-centred operator
replaced it after the experiments, and the experimental density forms of `W`
— `ghost_vars = :density`, `:mass` — were removed from `src/`.)* The density
experiment was decided against. "Three working fills, to revisit when the
physics gets harder" was recorded 2026-10-23 at Erik's request.

**2026-10-24.** *(Amended 2026-10-24, EntropyEOS 2.0:)* the robustness
measurement of the parent-centred fill on 2.0's projection was added to the
record.

## Julia version

**Undated.** Before 2026-09-25 the floor was `julia = "1.10"` wherever
possible, and a weak-dependency split (`TreeGRRMHDTreeGHExt`) had been
proposed for TreeGH.

**Undated.** TreeHydro had measured the coverage's cost at 3.14x at 1.13
before, against 1.04x at 1.11.

**2026-09-25.** `julia = "1.11"` was decided: all the Tree* packages —
TreeAMR, TreeWave, TreeHydro, TreeGeneralizedHarmonic and this one — shared
the 1.11 floor, so that an unregistered dependency can be located with a
`[sources]` entry, a 1.11 key, rather than added by URL. That also removed
the reason for the weak-dependency split proposed earlier: TreeGH's pin to
1.11 (its `LinearAlgebra = "1.11"` compat entry, its `[sources]` entries,
and ApparentHorizonFinder and KorzynskiSpin, whose checkouts both declare
1.11) was satisfied by this package's floor, so TreeGH could be an ordinary
dependency and `TreeGRRMHDTreeGHExt` was not needed for the version's sake.

**Step 0.** `Project.toml` names EntropyEOS and IMEXRungeKutta by their
GitHub URLs (amended in step 0; the plan had CI run `Pkg.add(url = …)`).

**2026-10-15.** Erik raised the floor from 1.11 to 1.12, and then to 1.13 by
his rule that the floor is raised when the older version is more than about
20% slower (amended 2026-10-15, twice). 1.11 with bounds checks (the same
suite as the 1.12/1.13 comparison, the same afternoon) took 6m05s and 204 s
of compiling, close to 1.12's.

What 1.11 and 1.12 needed and 1.13 does not is gone. The suite's
allocation probe (`rec_allocations`, now in `test/recovery_states.jl`) was a
`Vararg{Any,N}`-specialized method because 1.11 did not specialize a plain
`args...` there and boxed the result, 16–672 bytes a call on all 50 of the
probe's calls, where 1.12 and 1.13 allocate nothing either way. Nothing in
the package or the suite was there for 1.12 alone.

**2026-10-16.** Every Tree* package got the 1.13 floor, and its CI tests only
1.13 (amended 2026-10-16: Erik's decision for all of them; until then the
siblings kept 1.11, which did not constrain this package, since a
dependency's floor below this package's is satisfied by every version this
package supports). CI had a fourth cell, macOS at the current release, which
repeated the 1.13 macOS cell until 1.14; CI now tests only the floor
(amended 2026-10-16).

**2026-11-05.** The CI cells run in two tiers, the same in every Tree*
repository (2026-11-05, Erik).

**2026-11-06.** *(Amended 2026-11-06:)* TreeGH is not a dependency of this
package at all now; see "The coupling contract".

## Precision and devices

**Step 0.** The design (generic `T` and backend, `Float64` physics, `Float32`
smoke test, `src/precision.jl`, `src/device.jl`) was proposed in step 0. The
SpacetimeMetrics metric as a bit-identical `isbits` kernel argument was
measured in step 0.

**Step 1.** The pointwise layer of step 1 (`Metric3`, `prim2con`,
`con2prim_hydro`, `con2prim_ideal`, `GeometricEOS` around a table) was
measured in step 1.

**Step 2.** Step 2's layer was measured in step 2. Step 2 needed one change:
StaticArrays' matrix product accumulates with `muladd`, which LLVM fused
differently in the kernel and on the host, so three of the 79 numbers
differed in the last bit; `lower`, `raise`, `dot_up` and `dot_down` were
changed to multiply with plain `*` and `+` (`mul3`; amended in step 2).

**Step 3.** The stage solve on every branch was measured in step 3.

**Step 5.** The scheme's and the driver's kernels (geometry as `to_backend`
arrays, physics as `isbits` arguments, generic in `T`, the Alfvén wave in
`Float32`) are step 5's.

## Validation ladder

**2026-10-14, step 5 — The staggered scheme in 1D: smooth problems (rung 1a).**

`CODE.md`: "Where the fields live", "Field sets", "Reconstruction", "Riemann solver", "The
electromotive force for `A`", "The right-hand side", "Time integration", "Boundaries".
Waits: **AMR-1** (EOS-1, EOS-2 and 4b have landed; amended in step 0). *(Amended 2026-10-14:)*
AMR-1 has landed in TreeAMR 0.2.1, so step 5 waits for nothing; AMR-2 and AMR-3 have landed too,
so steps 6 and 8 need no workaround for them.

Changes:
- **`src/fieldsets.jl`:** creates `U`, `A_{1,2,3}`, `Φ̃`, `P`, `B_d`, `F_d` and `EMF_d` for
  D = 1, 2, 3, with the centerings for the missing dimensions.
- **`src/curl.jl`:** `curl_A!` over the stored faces, plus `𝔅̄`, and the cell average of `B`.
- **`src/reconstruction.jl`:** PLM (minmod, MC) with the positivity fallback.
- **`src/evolution.jl`:** `MHDProblem`, and the kernels in the order of "The right-hand side":
  - con2prim (stored), `q = ∇·E`, fluxes (closed), EMF (closed edges);
  - the divergence and sources, the `A` update, and the gauge;
  - `mhd_rhs!` (the explicit part), and `ohm_solve!` (the stage solver over owned cells, calling
    step 3);
  - the atmosphere stage limiter.
- **`src/driver.jl`:** `evolve!(T, case, Val(D); N, …)` on a uniform mesh, chunked, with `Δt`
  from the light-speed CFL.
- **Cases:** `src/cases/em_wave.jl` and `src/cases/alfven_wave.jl`.

Accept:
- **Vacuum EM wave** (`Vacuum`, periodic, N = 50…400): L1 rate ≥ 1.9 with MC **(predicted)**.
- **Circularly polarized Alfvén wave** (`IdealConductor`): L1 rate ≥ 1.8 **(predicted)**.
- **Conservation:** rest mass, `S_i` and `τ` to round-off (flat and periodic, so the total energy
  and momentum have no sources).
- **Purity:** the RHS is pure, and never mutates `u`.

*(Done, 2026-10-14, on `claude/step-5-staggered-1d`.)* *(Amended in step 5:)* as built, the
files are as listed, plus `src/fieldsets.jl`'s diagnostics set `X` (per-cell tallies of the
stage outcomes and the resets, reduced per chunk by `ohm_tally`); the driver is `evolve!(T,
case::MHDCase, Val(D); N, t_end, chunk, cfl = 2/5, limiter, …)` with a case as data
(`MHDCase`: `fields(x, t)`, `potential(x, t)`, the EOS, the model, the provider, `𝔅̄`, the box),
exact initial data and references (Gauss cell averages for `U`, point values for `A`, `Φ̃`);
the cases are `em_wave_case` and `alfven_wave_case`. The kernels are written for D = 1, 2, 3 and
validated in 1D. The tests are `test/scheme_tests.jl` (centerings, the AMR-1 round trip, `∇·𝔅`
on random `A` in D = 1, 2, 3, purity, the constructor's refusals, the reconstruction's
fallback, the reset, chunking) and `test/wave_tests.jl` (the two waves). Decisions, each
(proposed in step 5) in `CODE.md`: the background `𝔅̄` is the constant *densitized* field; the
recoveries' warm starts come from the state (`s = D s/D`), not from `P`, so that the RHS is
pure bit for bit; the reset writes back any policy repair, not the atmosphere alone; `cfl =
2/5`; PointValue width 4 for `A` and `Φ̃` until step 8; `q` by centered differences at the
owned cells; SSP3(4,3,3) stays the default tableau, with a new (open) question against
ARS(4,4,3).

*(Measured in step 5, Julia 1.11 and 1.13, one and four threads, identical:)*
- **Vacuum EM wave:** L1 rates 1.938 (`𝔈ʸ`, `𝔈ᶻ`), 1.912 (`A_y`, `A_z`), 1.914 (`S_x`, `τ`),
  2.00–2.02 (`D`, `D Yₑ`, `D s`); the prediction holds.
- **Alfvén wave:** 2.089 (`S_y`, `S_z`), 2.10 (`A_y`, `A_z`), 2.09 (`𝔈`'s ideal projection),
  2.78–2.85 (`D`, `S_x`, `τ`, `D Yₑ`, `D s`); the prediction holds for all but the state's own
  `𝔈`, which converges at 0.85 because SSP3(4,3,3) is not stiffly accurate (each step ends
  `O(Δt)` off the equilibrium the stages impose; `CODE.md`, "The partition"). ARS(4,4,3) lands
  on it (2.10) for one more explicit evaluation a step, 8% more wall time.
- **Conservation:** `D`, `S_i`, `τ`, `D Yₑ`, `D s` drift at most 0.016 ulp of their scale per
  step (asserted ≤ 8). No stage was repaired, no reset fired; every stage was `OHM_SIGMA_ZERO`
  (EM wave) or `OHM_IDEAL` (Alfvén wave), one recovery each.
- **Purity:** `u` comes back bit for bit and the same `u` gives the same `du` bit for bit after
  another state was evaluated; the stage solve writes only `𝔈`.
- Beyond the list: `∇·𝔅` on random `A` (ghosts included, a two-level forest) is 0, 1.4e−16 and
  2.9e−16 of `max |𝔅|` in D = 1, 2, 3; the Alfvén wave in `Float32` lands within 4e−5 of the
  `Float64` error; one chunk and four chunks of the same `Δt` are bit-identical; a step of
  SSP3(4,3,3) on 400 cells is 5.2 ms at one thread (Apple M3 Pro, 1.13).

**2026-10-14, step 6 — Shock tubes and the stiff limit in 1D (rung 1b).**

`CODE.md`: "Validation ladder" 1, "Floors, the atmosphere and the magnetosphere".
Waits: AMR-2 (workaround allowed). *(Amended in step 5:)* AMR-2 has landed, and `MHDProblem`
takes one boundary hook per evolved set. Step 6 also decides the (open) tableau question of
step 5 (`CODE.md`, "The partition"): SSP3(4,3,3) leaves each step's `𝔈` `O(Δt)` off the stiff
equilibrium, and the step limiter reads it.

Changes:
- **`src/cases/shocktubes.jl`:**
  - Balsara 1–5 (GRaM-X Table I);
  - the resistive shock tube swept over σ (Komissarov 2007; amended in step 6: Palenzuela et
    al. 2009 and Dionysopoulou et al. 2013);
  - the self-similar current sheet (Komissarov 2007);
  - Dirichlet hooks per set.
- A host tool for reference solutions: a finer run of the same code.

Accept:
- **Current sheet:** L1 rate against the analytic erf profile ≥ 1.8 **(predicted)**.
- **σ sweep:** the L1 distance to the `IdealConductor` solution decreases monotonically over
  σ ∈ {0, 10, 10², 10³, 10⁶}.
- **Balsara 1–5:**
  - L1 self-convergence against an 8× finer reference ≥ 0.7 **(predicted)**;
  - plateau states within 2% of the published exact solutions (Giacomazzo & Rezzolla 2006;
    values copied with citation) **(predicted)**.
- **The stage solve:** outcome counts per run are recorded, and no cell ever holds an unflagged
  NaN.

*(Done, 2026-10-14, on `claude/step-6-shocktubes`.)* *(Amended in step 6:)* as built:
- `src/cases/shocktubes.jl`: `shock_tube_case` (any Riemann problem along `x`, `𝔅̄ˣ` the normal
  field, ideal or zero `E`), `balsara_case` with `BALSARA` (GRaM-X's Table 1, Giacomazzo &
  Rezzolla's Table 4), `BALSARA_EXACT` (their Tables 7–11, total pressure), `BALSARA_PLATEAUS`
  and `plateau_state`/`plateau_error`; `resistive_tube_case`; `current_sheet_case` with a
  pure-Julia `erf_series`; `atmosphere_tube_case`; `shock_gas`, a Γ-law gas with a wide
  entropy window.
- **The resistive tube is not Komissarov's** (amended in step 6): Komissarov 2007 has a current
  sheet and stationary shocks swept over η, but no shock tube swept over σ. The tube is
  Palenzuela et al. (2009, §5.1.3) and Dionysopoulou et al. (2013): `(ρ, p, Bʸ) = (1, 1, 1/2)`
  and `(1/8, 1/10, −1/2)`, `Bˣ = 0`, `Γ = 2`, σ ∈ {0, 10, 10², 10³, 10⁶} — the plan's own set.
- **Dirichlet hooks per set** in `src/driver.jl` (`DirichletBoundary`, `case_boundary`,
  `boundary_at`): the matter's Dirichlet, the potentials' Dirichlet in the field, CPU-only until
  AMR-7 (new, added in step 6). `MHDProblem` takes any hook `boundary_at` maps to one of
  TreeAMR's at the fill's time.
- The reset in its two roles (`reset_atmosphere!`, `reset_step!`), with the step limiter's
  injection in `ohm_tally`; `profile_1d`; `magnetization = +∞` at `D ≤ 0`.
- Host tools: `bin/shocktube_reference.jl` (the 3200-cell references in `test/data/`, how they
  were made in its header) and `bin/tableau_comparison.jl` (the table in `CODE.md`, "The
  partition").
- Tests: `test/shocktube_tests.jl`, `test/resistive_tests.jl`, `test/atmosphere_tests.jl`, and
  an AMR-7 probe.

*(Measured in step 6, Julia 1.11 and 1.13, one and four threads, identical; `CODE.md`,
"Validation ladder" 1b, has the rest:)*
- **Current sheet:** the prediction does not hold as stated. Against the erf profile the L1
  distance saturates at 1.1e−3 of `B₀` by 200 cells, the profile's own distance from the system's
  solution (the fluid's response and the displacement current). The acceptance is amended to the
  order against the next finer run, 2.53 and 2.58 (asserted ≥ 1.8), with the distance to the
  profile at Komissarov's resolution recorded (9.3e−4 at 200 cells, `t = 9`; asserted ≤ 2e−3).
- **σ sweep:** monotone, 0.392, 0.202, 0.0611, 0.0134, 1.73e−5 (L1 of `𝔅ʸ`, 200 cells), as
  `σ^−0.96` from 10³ to 10⁶; no fallback, at most 6 recoveries a stage.
- **Balsara 1–5:** self-convergence ≥ 0.7 for tests 1, 4, 5 and test 2's `τ`, `𝔅ʸ`; test 2's
  `D`, `S_x` at 0.66, 0.67 (a contact's `h^{2/3}`); test 3 not asymptotic below 800 cells (its
  dense shell), rates −0.02 to 0.53. The tests assert the measured rates. Plateaus within 2% in
  every window but test 3's `R4` (4.7% at 400 cells, 1.3% at 800); the thinnest regions have no
  window at 400 cells.
- **The stage solve:** every Balsara stage `OHM_IDEAL`; the resistive tube's and the current
  sheet's branch counts in `CODE.md`; no fallback in any default-tableau run; no non-finite
  value in any state (a negative `D` in a stage predictor is flagged non-finite by EntropyEOS and
  repaired inside the stage).
- **Atmosphere:** a 10⁸ ratio held, `ρ ≥ ρ_atm`; the step limiter's injection (`−8.4e−12` of
  rest mass hydrodynamically, `2.5e−8` magnetized with the cutoff, of 0.5) equals the totals'
  change to rounding.
- **The tableau:** SSP3(4,3,3) stays the default (decided in step 6 on the comparison in
  `CODE.md`, "The partition"; the open question is closed).
- **Atmosphere:** a shock tube with a 10⁸ density ratio exercises the limiter.

**Step 3.** The σ × magnetization sweep of the implicit stage solve that
rung 1 names was done in step 3, pointwise, on stage problems with a known
solution ("The implicit stage solve"); rung 1 records the branches an
evolution takes.

**Step 5.** Rung 1a was measured in step 5 (`test/wave_tests.jl`).

**Step 6.** Rung 1b was measured in step 6 (`test/shocktube_tests.jl`,
`test/resistive_tests.jl`, `test/atmosphere_tests.jl`). Proposed in step 6:
`plateau_error`; the current sheet started from the profile at `t = 1` with
the quasi-static `Eᶻ = ∂ₓBʸ/σ`; `atmosphere_tube_case`.

The resistive shock tube's citation was amended in step 6: Komissarov 2007
has no σ sweep, and the published tube is a Brio–Wu tube with `Bˣ = 0`,
not Balsara 1.

The current sheet: PLAN.md predicted an L1 rate ≥ 1.8 against the erf
profile, but the profile solves the diffusion limit and not the system, so
the distance saturates; the order was therefore measured against the next
finer run (amended in step 6). At Komissarov's 200 cells and `t = 9` the
distance to the profile was 9.3e−4 until step 8 prescribed the potentials'
low walls (9.6e−4 since).

The atmosphere tube, magnetized, as measured in step 6: the reset wrote
33,263 cells as the stage limiter and 11,054 as the step limiter and
injected `2.5·10⁻⁸` of rest mass and `6.9·10⁻⁸` of energy into the box
holding 0.5 of rest mass; the stages were 48,362 `OHM_SIGMA_ZERO` (the
cutoff), 111,572 fixed point and 66 Newton (61 from the ideal seed); no
fallback, at most 6 recoveries.

The potentials' relative Dirichlet hook (AMR-7) was chosen in step 6 for
potentials that grow in time.

**2026-10-15, step 8, decided by Erik.** The gauge damping `ξ = 16/L` on
every mesh (amended in step 8). Rung 2a's numbers before it: the Alfvén
wave along the diagonal's `A_x` rate 2.77 (MC) and 2.05 (WENO-Z, L1
3.67e−4, 8.62e−5, 2.14e−5), WENO-Z's drift 1.43 ulp. Nothing else the suite
printed moved; the 3D blast's `∇·𝔅` went from 1.2925e−16 to 1.2929e−16 and
the tilted field loop's `𝔅ᶻ` ratio in its 14th digit.

**2026-10-15, step 7 — 2D and 3D, and higher-order reconstruction (rung 2a).**

`CODE.md`: "Where the fields live", "Reconstruction". Waits: none.

Changes:
- **The scheme in D = 2, 3:** the 2D and 3D paths of steps 5–6, with the curl, the EMF in every
  direction, and the gauge at vertices.
- **Reconstruction:** PPM, WENO-Z and MP5, with `G = 3`.
- **Cases:** cylindrical explosion, magnetic rotor, Orszag–Tang, and field-loop advection on a
  uniform mesh.

Accept:
- **∇·B:** `max|∇·𝔅| h / max|𝔅| ≤ 1e−12` at every step of every 2D/3D run.
- **Direction independence:**
  - a 2D run that is translation-invariant in y equals the 1D run, bit for bit;
  - a run along y equals the run along x, transposed.
- **Higher order:** the Alfvén-wave rates of PPM, WENO-Z and MP5 are recorded.
- **Symmetry:** the rotor's and the explosion's symmetries hold to round-off.
- **Conservation:** `D`, `S_i` and `τ` to round-off.
- **3D:** a smoke run.

*(Done, 2026-10-15, on `claude/step-7-multid`.)* *(Amended in step 7:)* as built:
- **The 2D and 3D paths needed no change in the right-hand side.** Step 5's 2.7e−15 between a
  `y`-invariant 2D run and the 1D run was the initial data alone: the `3^D` Gauss weights
  multiplied out round a constant. The cell average is now the three-point rule one dimension
  at a time in difference form (`src/driver.jl`), and the runs agree bit for bit.
- **Reconstruction** (`src/reconstruction.jl`): `limiter` gains `:ppm`, `:wenoz`, `:mp5`
  (`G ≥ 3`, checked); every method an offset in the differences `q_k − q₀`, so a constant is
  reconstructed exactly and the offsets are odd and mirror-symmetric bit for bit. The EMF's
  corner `E` is `E₀ + δ_a + δ_b` with the method's offsets. MP5 runs at `cfl = 1/5`
  (`default_cfl`), its bound `1/(1 + α)`; at 2/5 it blew up Balsara 3.
- `max_div_B` and `evolve!`'s `after_step` monitor; `extruded_case`, `turned_case`,
  `oblique_case` (`src/cases/geometry.jl`); `cylindrical_explosion_case`,
  `magnetic_rotor_case` (GRaM-X), `orszag_tang_case`, `field_loop_case` (Beckwith & Stone
  2011) in `src/cases/multid.jl`; `bin/reconstruction_comparison.jl`.
- Tests: `test/reconstruction_tests.jl`, `test/multid_tests.jl` (direction independence),
  `test/multid_cases_tests.jl` (the diagonal wave and the standard tests); in
  `test/shocktube_tests.jl` the Balsara convergence testset is renamed to what it asserts, and
  Balsara 3 under PPM is asserted against PPM's own reference (`test/data/balsara3_ppm.ref`).
- TreeHydro has no PPM, WENO-Z or MP5 (the brief said it had): they are written here from
  the papers.

*(Measured in step 7; `CODE.md`, "Validation ladder" 1b and 2a, has the rest:)*
- **∇·B:** at most 2.4e−16 of `max |𝔅|` after every step of every 2D and 3D run (≤ 1e−12).
- **Direction independence:** bit for bit (`isequal`) between the 1D Alfvén wave and its
  extrusions in 2D (MC, PPM, WENO-Z, MP5) and 3D (MC, WENO-Z), and for Balsara 1 with Dirichlet
  faces; turned runs equal the turned runs with `==` — bit for bit but the sign of an exact
  zero, which `0 − 0 = +0` fixes whichever way the axes are turned.
- **Higher order:** Alfvén wave `S_y` rates 2.67 (PPM), 2.00 (WENO-Z, MP5) against MC's 2.09,
  errors 9× and 17× below MC's at 400 cells; the scheme stays second order (recovery from
  cell averages). Along the diagonal in 2D: 2.21–2.77 (MC), 1.97–2.05 (WENO-Z).
- **Symmetry:** explosion 2.5e−17 (minmod, 64²), rotor 2.6e−27 (MC, 64²); Orszag–Tang 1.9e−13,
  set by the recovery's tolerance of 1e−12 (5.3e−15 with it at 1e−14), asserted ≤ 1e−11; 3D
  periodic blast exactly symmetric. A closed box is not mirror-symmetric at its faces: the
  potentials' Dirichlet hook evolves the low boundary vertex and fills the high one (1.3e−4
  in a 16³ box whose faces the precursors reach; `CODE.md`, "Boundaries").
- **Conservation:** Orszag–Tang 0.056 ulp of a total's scale a step (0.037 at 1.11), field loop 0.25, the
  oblique wave 1.6 (asserted ≤ 8).
- **3D:** the spherical blast, 16³, 8 steps in the suite (15 once), no repair, no reset.
- **Balsara per reconstruction:** the table in `CODE.md`; PPM converges on Balsara 3
  (0.49–0.56 against itself), WENO-Z and MP5 do not usefully; WENO-Z and MP5 miss 2% in
  Balsara 4's `R4`, PPM in Balsara 2's `R5` (2.8%).

**Step 7.** Rung 2a was measured in step 7 (`test/multid_tests.jl`,
`test/reconstruction_tests.jl`), and the per-reconstruction Balsara table
of rung 1b (`bin/reconstruction_comparison.jl`) too. The suite's assertion
of Balsara 3 under PPM against PPM's own 3200-cell reference
(`test/data/balsara3_ppm.ref`, rates ≥ 0.4, every plateau within 2%) was
added in step 7. Step 7 changed the initial data's Gauss average, which
moves Balsara 1's MC reference in the last bit; the MC references were not
regenerated.

Translation invariance: step 5's 2.7e−15 was the initial data's Gauss
weights (amended in step 7: "The partition", the driver).

Decided in step 7: the turning claim is stated with `==` (the sign of a
zero is the one freedom); and for Orszag–Tang, "to round-off" means to the
recovery's tolerance where the recovery iterates.

The 3D blast in step 7 ran on the periodic `[−2, 2]³`; in a closed box it
was not symmetric, at the faces (see "Boundaries", measured in step 7).

**Step 8.** Rung 2b was measured in step 8 (`test/refinement_tests.jl`,
`test/refinement_cases_tests.jl`). With both walls prescribed (AMR-8's
workaround, "Boundaries") the 3D blast became symmetric bit for bit in the
closed box, where the suite then ran it; so did the explosion on
`[−2, 2]²` at 32² (amended in step 8).

Rung 2b's numbers before the end of step 8 (TreeAMR 0.2.2: the potentials
`A_j` conservative along the edge, AMR-9, and the matter's ghost fill in
primitives, PointValue width 2, AMR-10), each followed by the re-measured
value in brackets that `CODE.md` now gives:
- Conservation at coarse-fine faces: every total but `𝔈`'s drifted 0.008
  [0.013] (1D) and 0.12 [0.12] (2D) ulp of its scale a step on static
  two-level meshes with the flux fixup, 1.5·10¹⁰ [1.8·10¹⁰] and 3.3·10¹¹
  [5.4·10¹¹] without; through regrids, 0.017 [0.015] (Balsara 1, tracked,
  ten regrids) and 0.036 [0.037] (the rotor, two).
- Magnetic flux: the rate through every coarse face on a coarse-fine face
  equalled the fine faces' sum to 1.7e−18 (2D) and 8.7e−19 (3D) of the
  largest rate with the EMF correction, 0.85 and 0.48 of it without. As a
  state, the flux through the plane drifted 5.8e−4 of its scale across a
  level — the potentials' operators, AMR-9. *(Amended in step 8, AMR-9
  landed:)* with `A_j` conservative along the edge the state holds too
  (the numbers in `CODE.md`).
- The field loop across a coarse-fine face: at `t = 1/2`, 0.801 [0.803] of
  its initial energy, L1(`A_z`) 1.6e−6 [1.56e−6]; at `t = 4`, 0.584
  [0.589] and 4.2e−6 [4.16e−6].
- The interface order and the gauge damping: width 4 L∞ 2.04 [1.97] in 1D,
  2.02 [1.85] in 2D, width 2 0.91 [0.88] and 0.33 [0.08]; `ξ = 16/L`
  cleared 97% [99.95%] of a gauge pulse's spurious field within two light
  crossings.
- The tracked shock tube: conserved to 0.017 [0.015] ulp a step.

The atmosphere across a coarse-fine face, *(amended in step 8, AMR-10
landed:)* the ghost fill prolongated the primitives, width 2 (proposed): no
ghost inadmissible or below the atmosphere, hydrodynamically and magnetized
(where conserved ghosts had 33 repaired recoveries), but 5.6e−9 of rest
mass injected hydrodynamically against the conserved fill's 1.5e−11; at
width 4, 5.4e−4 of energy, and a blow-up magnetized (the tables in
"Refinement").

The magnetized atmosphere tube *(amended in step 8)*: **the magnetized tube
is chaotic at rounding level**, and its fallbacks, injected energy and
largest `W` are not properties of the scheme: perturbing its initial state
by 10⁻¹⁵ (twelve runs, on the tree before step 8's primitive ghost fill and
after it) gives 0 to 144 fallbacks of its 160,000 cell-stages, an injected
energy from 3.9·10⁻⁸ to 2.1·10⁻⁶ and a largest `W` from 4.6 to 90, while
every run stays finite, above the atmosphere, and conserves the totals but
for exactly what the reset injects. The primitive ghost fill, whose owned
`P` differs from the recovery's by rounding, moved the suite's run to 92
fallbacks, `2.5·10⁻⁸` → `5.0·10⁻⁸` of rest mass and `6.9·10⁻⁸` →
`1.7·10⁻⁶` of energy, and `W` 4.69 → 6.63; the test then asserted no
fallback hydrodynamically and a fallback rate of at most 0.2% magnetized
(proposed in step 8). The same holds for step 6's tableau comparison: "SSP3
no fallback, ARS 7" on this tube is one draw from that spread, and the
decision for SSP3(4,3,3) rested on its other rows (the near-vacuum
rarefaction, the ideal atmosphere). The hydrodynamic tube is not chaotic:
both fills give its numbers to every digit.

**What the end of step 8 moved elsewhere** (measured, every line the suite
prints, before and after, Julia 1.13): the symmetric walls and the relative
hook moved nothing a uniform-mesh case prints; the primitive ghost fill,
whose owned `P` goes through `P_from_W`, moved the uniform runs in their
last digits — the waves' L1 errors in the 12th to 15th, every rate to the
digits recorded unchanged, the drifts (Alfvén 0.0059 → 0.0046 ulp a step,
PPM 0.011 → 0.0056), the shock tubes' L1 errors in the 11th to 13th digit
(Balsara 3 under PPM in the 10th) and their boundary budgets within
3·10⁻¹⁴, the uniform flows through the faces 4.4·10⁻¹³ … 1.5·10⁻¹² of a
variable's scale, the current sheet's distance 9.559329e−4 → 9.559339e−4,
Orszag–Tang's asymmetry 1.9e−13 → 3.8e−13 (the recovery's tolerance;
asserted ≤ 1e−11), the rotor's drift 2.9e−16 → 4.0e−16; the 1D/2D/3D bit
identity, the turned runs and every mirror symmetry on a uniform mesh hold
bit for bit. Chaotic at rounding level, and so moved further: the
magnetized atmosphere tube (rung 1b), the near-vacuum rarefaction under ARS
(2.1e−10 → 2.7e−10, resets 152/114 → 161/118) and the ideal magnetized
atmosphere under ARS (2.6·10⁴ → 1.6·10⁴ of rest mass injected, `W` 49 →
100); SSP3(4,3,3)'s rows of both are the same to the digits recorded. The
3200-cell references in `test/data/` are not regenerated (proposed in step
8).

**2026-10-16.** The investigation of the magnetized atmosphere tube
(`bin/front_investigation.jl`): the cause — the field's truncation error in
the fluid's share, amplified by EntropyEOS's repairs at the edge of the
valid range — and the sweep of fixes, under "Floors, the atmosphere and the
magnetosphere".

**2026-10-21.** The decision of 2026-10-21 (a floored state is the
atmosphere at rest; the tube's policy sets `collapse_to_atmosphere`), under
"Floors, the atmosphere and the magnetosphere": the magnetized tube is then
deterministic with no fallback, and the suite's run prints `W` 4.80,
injected −1.18·10⁻⁸, −7.4·10⁻⁷.

**2026-10-22, decided by Erik.** The magnetized tube's assertion of a
fallback rate of at most 0.2% was replaced by "no fallback" magnetized as
well. *(Amended 2026-10-22:)* the ideal atmosphere's row of step 6's
tableau comparison was the policy artefact, and the re-measured comparison
in "The partition" — ARS 23 fallbacks, SSP3 none, both deterministic — is
what the tube said then.

**2026-10-23, decided by Erik.** The matter's ghost fill became the
parent-centred one: the atmosphere across a coarse-fine face then injects
−1.6e−11 of rest mass, as the conserved fill's.

**2026-10-24, EntropyEOS 2.0.** Without the option (`collapse_to_atmosphere`)
on 2.0's projection the magnetized tube's suite run prints the same; across
48 perturbed runs every one is clean, but no longer the same run.

**2026-10-30.** With the step cache the uniform flow crosses the box to the
recovery's tolerance (at most 8.4e−11 after 13 steps and 3.6e−10 after
1000; asserted `≤ 10³·tol` and `≤ 2·10³·tol`). Before, it crossed the box
unchanged to 1.7e−12 of its scale under either tableau.

**2026-11-02, step 9 — Analytic curved backgrounds (rung 3).**

`CODE.md`: "Metric providers", "Validation ladder" 3. Waits: none.

Changes:
- **The `Analytic` provider** at cells, faces, edges and vertices.
- **Kerr-Schild cases:** the Wald solution (`Vacuum`), and the Michel monopole or magnetized Bondi
  flow (`IdealConductor`).
- **The interior:** inside `r < r_in` (within the horizon), the state is relaxed to the analytic
  solution each stage. This is position-only, as TreeGH's interior is **(proposed)**.

Accept:
- **Stationarity:** for both cases, the L1 drift from the exact stationary solution after
  `t = 50M` converges at ≥ 1.8 **(predicted)**. *(Measured in step 9: **fails** — at `h = M/4,
  M/8, M/16` Wald's `D` 0.94 then 0.08 and its charge −0.11, the magnetized Bondi flow 1.1–1.2.
  Located in round 2 (`CODE.md`, "Validation ladder" 3, "Where the error lives"): Michel's flow
  alone converges at 2.4; the field cases carry a mode that grows through the potentials'
  relative Dirichlet hook (AMR-7), which copies the evolved potential's gauge drift across the
  face, closed through the Lorenz gauge; with the case's potential itself in the ghosts both
  cases are stationary and converge at 1.9–2.3 in `D` and `τ` (`M/4 → M/8`). The fix is a
  boundary design change, proposed for Erik. The suite asserts the short form, `t = M/2`, at
  ≥ 1.6, measured 1.73–2.48.)* *(Round 3, option 1 as built, decided by Erik 2026-11-03: the
  static potentials' boundary is the case's own; re-measured at `M/4`, `M/8`, `M/16`: `D`
  1.93–2.34, `τ` 2.06–2.27, Wald `𝔈` and `S_x` 1.94–2.01 — **holds**; Bondi's `S_x` and `𝔈`
  1.69 and 1.71 at `M/4 → M/8` (**fail**, measured, not loosened), 1.82 and 1.83 at `M/8 →
  M/16`; the charge 1.49–1.68. `CODE.md`, "Validation ladder" 3.)* *(Measured in step 9: holds
  on `M/8 → M/16`, the pair it is judged on (decided by Erik 2026-11-03): `D` 1.98–2.22, `τ`
  2.06–2.13, `S_x` 1.82–1.96, `𝔈` 1.83–2.01; the `M/4 → M/8` rates stand as measured, `M/4`
  putting only 8 cells between the hole's centre and the horizon.)*
- **Charge:** `|q|` from `∇·E` is recorded.
- **∇·B** at round-off.

*(Built, 2026-11-02, on `claude/step-9-curved-backgrounds`, and **open**: the stationarity
prediction fails — see "Accept" and `CODE.md`, "Validation ladder" 3; Erik wants the error
located before the step counts as done. `CODE.md` "Metric providers", "The interior" and
"Validation ladder" 3 have the details, marked "proposed/measured in step 9".)* *(Done,
2026-11-05: the error was located and fixed in round 3, option 1. Step 9 was accepted as is by
Erik, round 4's spinning Wald drift included. Black holes are left for future work; see
`CODE.md`, "Known risks".)* As built:
- **The provider** needed nothing new: every kernel already evaluated `point_metric` at its own
  point (cells, faces, edges, vertices) and `metric_gradient` at the cell since step 5.
  Measured: 102 ns and 234 ns a call on an EPYC 7532, **44% of a step** on the 16³ spherical
  blast (`Flat` against a Kerr-Schild hole of mass `10⁻³⁰⁰`); kept pointwise
  (*decided by Erik 2026-11-03: pointwise, no cache*; caching is step 10's, for `GHMetric`).
  The asked-for deduplication of the gauge terms' vertex and edge evaluations is not done:
  no point repeats within a kernel invocation, and sharing across invocations needs storage
  or restructured kernels (`CODE.md`, "Metric providers"; for Erik). `bin/metric_cost.jl`.
- **The cases** (`src/cases/kerr_schild.jl`): `wald_case` (Wald 1974 at `a = 0`, `Vacuum`) and
  `bondi_case` (Michel's flow in a radial monopole, `IdealConductor`) on a Schwarzschild hole
  in Kerr-Schild coordinates, each exact for matter and fields, the matter Michel's
  accretion flow (`MichelFlow`) in both. **Magnetized Bondi, not the Michel monopole**, and
  **`a = 0`** (both proposed, reasons in `CODE.md`; accepted by Erik 2026-11-03). On the
  **octant** `[0, 4M]³` with the hole at its corner and Dirichlet faces from the data (proposed;
  accepted by Erik 2026-11-03): the monopole's Dirac string
  leaves through the corner, and the octant is an eighth of the box. No 1D or 2D reduction
  exists in Cartesian Kerr-Schild.
- **The interior** (`src/interior.jl`, `HoleInterior`, **proposed**): inside `r < r_in` the
  reset pastes the case's solution — cell averages, point values — first, as stage and step
  limiter, position-only; `r_in = M`. Measured at `h = M/4` to `t = 50M`: outside the horizon
  every variant — both roles, the step limiter alone, the stage limiter alone, `r_in = M/2`,
  `3M/2`, no interior — drifts alike to 2%; the interior decides only the inside's health (the
  stage limiter alone or none leave the reset working there every step; `r_in = M/2` resets
  Bondi cells). Proposed: both limiters, `r_in = M`, for Erik.
- **Diagnostics** (now in `CODE.md`, "The interior"). `bin/curved_stationarity.jl` runs the
  study.
- **Tests** (`test/curved_tests.jl`, 72): Michel's flow against its definitions, `curl A =
  √γ B`, the right-hand side of the exact states second order outside `2M`, the drift after
  `t = M/2` at `h = M/4`, `M/8`, `∇·𝔅` and the charge, the interior after every step, and the
  curved-space bit identity of the step cache (one integration = two = a restart).

**Step 9.** Rung 3 was built and measured in step 9 (proposed in step 9
unless marked); the plan's choice of the magnetized Bondi flow over the
Michel monopole was made there. The spinning Wald case was first a
follow-up proposed for Erik: a spinning Wald case "could carry a passive
fluid at the floor and be measured on `𝔈` and `𝔅` alone (under `Vacuum`
nothing couples)".

Stationarity: PLAN.md predicted that the L1 drift after `t = 50M`
converges at `≥ 1.8` **(predicted)**; *measured in step 9: fails — Wald
saturates, the magnetized Bondi flow converges at first order; the cause is
located below, and the prediction holds without it at `h = M/4 → M/8`.* The
table of that study (the relative hook) is in `CODE.md`, "Validation
ladder" 3, "With the relative hook".

**Step 9, round 2.** "Where the error lives" was measured in step 9,
round 2. The absolute hook was then a process-local diagnostic
(`abshook`), "a diagnostic for these time-independent potentials". At
`M/16` (jobs 588223, 588224, amdq, 64 threads) the runs with it were at
`t = 10M` when the Symmetry tunnel dropped: Wald `D` 6.035e−4 → 1.328e−4
(**2.18**), `𝔈ˣ` 1.179e−4 → 2.937e−5 (2.01), Bondi `D` 1.650e−3 → 4.211e−4
(1.97), `𝔈ˣ` 1.068e−4 → 2.995e−5 (1.83), against `M/8` at `10M`; their
`t = 50M` rows were in `/mnt/beegfs/eschnetter/claude/
treegrrmhd-step9-5882{23,24}-part1.log`, unread.

**Step 9, round 3.** Re-measured with option 1 as built (jobs
588498–588503): at `M/4` and `M/8` the new default reproduces the round-2
diagnostic to every printed digit, the same arithmetic; at `M/16` (jobs
588502, 588503) the same, the round-2 rows, and so the rates of the table.

**2026-11-03, decided by Erik: option 1** — the case's own potential in the
ghosts where it does not depend on `t`. The process-local diagnostic of
round 2 (`abshook`) is gone. The `M/16` runs of round 2 with the
diagnostic hook (jobs 588223 on cn085, EPYC 7532, and 588224 on cn112, EPYC
8534P), outside `2M` at `t = 50M`: their values (Wald's flat from `15M` to
`50M`) and rates are those `CODE.md` gives under "Re-measured with
option 1 as built" ("Validation ladder" 3).

**2026-11-03, accepted by Erik.** Rung 3's magnetized Bondi flow rather
than the Michel monopole (the plan's choice, made in step 9), `a = 0`, and
the octant `[0, 4M]³`. *(Resolved 2026-11-03: Erik wants [the spinning Wald
case], as step 9's round 4.)*

**2026-11-03, decided by Erik.** The stationarity prediction is judged on
the finest pair, `M/8 → M/16`; the `M/4 → M/8` rates 1.69 and 1.71 stand as
measured.

**Step 9, round 4 (asked by Erik 2026-11-03).** The spinning Wald case
(`spinning_wald_case`, `SpinningWald`), with `gauge_damping` new on
`MHDCase`. Its verdict was measured in step 9, round 4: the prediction fails
on the finest pair at `t = 50M`, for both spins. "Not localised further;
for Erik (round 4 asked for the measurement, not a fix)." On the paste's
ball at `a = 0.9M`, "For Erik: the paste's ball in Kerr-Schild `r`, or one
Euclidean radius `ρ` chosen to contain `r₋` in the plane …; not decided
here."

**2026-11-03, step 9.** **Round 4: a spinning Wald case** *(asked by Erik 2026-11-03; acceptance written before the
code)*. Wald's uncharged field `𝒜 = (B₀/2)(ψ + 2a η)` (`ψ = ∂_φ`, `η = ∂_t`, lowered) on a Kerr
hole in Kerr-Schild coordinates, `Analytic(KerrSchild(M, a))`, `a = M/2` and `0.9M`, under
`Vacuum`; a passive fluid near the floor, which nothing couples to the field but the charge's
convection; the potential stationary, so `static_potential = true` (option 1), with `Φ̃ ≠ 0`; the
interior the default paste at `r_in = M` (decided by Erik 2026-11-03). Accept:
- **Pointwise:** the field is the curl of the potential (`√γ B`), and Maxwell's equations hold
  by finite differences of the case's own functions: Gauss (`∂ᵢ𝔈ⁱ = 0`), Ampère (the flux of `𝔈`
  divergence-free), the Lorenz gauge; the field is uncharged (the flux of `𝔈` through a sphere
  around the hole vanishes).
- **Stationarity:** the L1 drift of `𝔈` and `𝔅` outside the horizon after `t = 50M` converges at
  `≥ 1.8` **(predicted)**, judged on the finest pair. **(Measured in step 9, round 4: fails.**
  `M/8 → M/16` at `50M`: `a = M/2` `𝔈` 1.55, 1.04, `𝔅` 0.79–0.89; `a = 0.9M` (ball in
  Kerr-Schild `r`) `𝔈` 1.80, 1.80, `𝔅` 1.34, 1.93, 1.12. Second order at `10M` for both, and to
  `35M` at `a = M/2`; the coarse pair 1.81–2.46 throughout. `a = M/2` grows late in the outer
  region, `a = 0.9M` steps up between `10M` and `15M` near the ergoregion; `CODE.md`, "The
  spinning Wald case". Not localised further. *Accepted as measured by Erik 2026-11-05; its
  localisation is left for future black-hole work.*)*
- **∇·𝔅** at rounding. *(Measured: 1.05–1.47e−16 of `max |𝔅|`.)*
- **The interior:** `r_in = M` and the uncounted radius `(r_in + r₊)/2` inside the horizon,
  `r₊ = M + √(M² − a²)`, with the test for "inside" stated (Kerr-Schild `r` is not the Euclidean
  radius); the paste's ball in Euclidean against Kerr-Schild `r` measured, not decided.
  *(Measured: at `a = M/2` both are inside `r₊` and contain `r₋`; at `a = 0.9M` the Euclidean
  ball misses `r₋` in the plane and the drift grows — `𝔈ˣ` 8.1e−3 and 3.1e−2 at `M/4`, `M/8`
  after `50M`, against 7.8e−4 and 2.0e−4 in Kerr-Schild `r`. The default stays Euclidean.
  *Left for future black-hole work, Erik 2026-11-05.*)*
- **The fluid** must not dominate: resets beyond the uncounted radius reported, not special-cased.
  *(Measured: it does not dominate the field — `ρ = 1` moves `𝔈`, `𝔅` by 1–5% at `M/8` — but its
  resets fill the box, about 2.7·10⁹/1.0·10⁹ stage/step at `M/16`; none with `ρ = 1`.)*
- **Suite:** a short test within the current budget; the `50M` study in `bin/` on Symmetry.
  *(Measured at 0858776: 26442 tests, every run passing whole, 14m48s at one thread; the study
  is `bin/curved_localise.jl kwald`.)*

**2026-11-05.** The spinning Wald case was accepted as measured by Erik:
step 9 closes on it, and the runs that would separate the causes are left
for future black-hole work; the paste's ball at high spin is left for
future black-hole work too (Erik, 2026-11-05; "Known risks").

**2026-11-06, decided by Erik.** Rung 4 runs on `Analytic(TOVMetric)`, with
no TreeGH. Rung 5's "the frozen TreeGH metric, through its adapter,
reproducing rung 4" was added 2026-11-06.

**Step 10.** Rung 4 was measured in step 10. PLAN.md's predictions, each
marked **(predicted)** until then: the drift after `200M` converges at
`≥ 1.5` — measured: holds for both reconstructions; rest mass to
round-off, net of the outer boundary flux — measured: holds; F, H1 and H2
within 3% of Font et al. 2002 — measured: holds at `h = 0.375`, H2 fails
at `h = 0.5`.

**Step 10 — The TOV star in the Cowling approximation (rung 4).** Measured against the step's
acceptance list (`PLAN.md`, step 10):
- **Stationarity**, the central-density drift after 200 M predicted to converge at ≥ 1.5:
  *(Measured in step 10: holds — MC 2.52, 3.32 and 2.80 at `h = 0.75 → 0.5 → 0.375 → 0.25`,
  PPM 2.68 and 2.45 to 0.375; `CODE.md`, "Validation ladder" 4.)*
- **Conservation**, rest mass to round-off, net of the outer boundary flux: *(Measured in step
  10: holds — the change of the total `D` is the reset's injection to `≤ 7.7·10⁻¹⁵` of 1.506 in
  every run, to `2000M`; nothing leaves through the outer faces.)*
- **Modes**, the F, H1 and H2 frequencies predicted within 3% of Font et al. (2002)'s Cowling
  values: *(Measured in step 10: holds at `h = 0.375` — F −0.89%, H1 +0.00%, H2 +2.20% over
  886 M, PPM; at `h = 0.5` H2 fails, −7.97%; the modes damp within about 500–900 M, so the
  2000 M and 1000 M runs were stopped at 1918 M and 886 M; `CODE.md`, "Validation ladder" 4.)*
- **Floored cells** at a coarse-fine surface for width 1, width 3, and limited (AMR-4), which
  decide the prolongation (open): *(Measured in step 10: width 3 bounded stays, decided by Erik
  2026-11-08; the unbounded operator lets the reset remove 9.5% of the star by 200 M, width 1 is
  twice as diffusive; `CODE.md`, "Validation ladder" 4, "What the star decides".)*
- **Light-cone against split speeds** on the star, which decide the Riemann speeds (open):
  *(Measured in step 10: the light cone stays, decided by Erik 2026-11-08; the split speeds blow
  up after 105 M at `h = 0.75`; `CODE.md`, "Riemann solver".)*

**Step 10, round 2.** Measured against the round's acceptance list for the contract (`PLAN.md`,
step 10):
- Through the sampled ADM adapter and `Mv`, `g` and `∂g` predicted to converge at 4th order at
  every centering, `K` at 4th order, the vertices bit for bit: *(Measured 2026-11-08: holds on
  Kerr-Schild — 4.52–4.72, the vertices 5.87, the owned vertices bit for bit; on the star
  3.68–3.95 off the surface, but `∂_j g_ab` at **1.67 in the layer `|r − R̄| < 4`**, where the
  metric's third derivative jumps; no fallback.)*
- The Cowling star through `VertexMetric` predicted to reproduce its `Analytic(TOVMetric)` run
  to 200 M, the difference converging at 4th order, with no fallback: *(Measured over four
  steps: 1.21e−3 → 4.76e−5 of the steps' change at `h = 0.8 → 0.4`, rate 4.67, no fallback.)*

**2026-11-08.** GH-1 landed; rung 5 still needs GH-2 to GH-4.

**2026-11-09, step 10, round 2.** The Cowling star through `VertexMetric`: *(Measured to 200 M on
2026-11-09, PPM at `h = 0.5`, 0.375, 0.3 for both metrics (Symmetry jobs 593624–593629): no
fallback; the central density's difference at **3.94, 3.96**, the conserved variables' L1 inside
and in the surface layer at 2.8–4.4, about 3.7 on the whole; only the advected `√γ D s` just
outside the surface does not converge (its largest difference 1.2e−6 → 1.6e−6 at `r ≈ 9.1–9.2`,
the layer's L1 at 2.03 on the finest pair): its ratio `s★` leaves the window there (−410 …
1220), since the reset rewrites `D s` only in repaired cells, while the recovered `s` and `T`
agree. Holds; `CODE.md`, "As built in step 10, round 2". The vertex metric's step costs 2.0–2.5
times less than `Analytic(TOVMetric)`'s.)*

## Open questions

**2026-09-25.** The question whether the atmosphere reset needs its
stage-limiter role read: "Run rung 4's TOV star, a star in a vacuum, twice:
with the reset as both limiters, and as the step limiter alone. TreeHydro
named this case as the one where a per-stage reset could matter, and no
code in IMEXRungeKutta's survey measured it" (amended 2026-09-25).

**Step 6.** *(Closed in step 6.)* SSP3(4,3,3) or ARS(4,4,3) as the default
tableau: SSP3(4,3,3), whose explicit part is SSP; ARS(4,4,3) undershoots
into a near vacuum, fires the reset on Balsara 3 and 4, and floods a
magnetized atmosphere (see "The partition").

**2026-10-15, closed by Erik.** *(Added in step 8.)* Primitives or the
conserved state in the matter's ghost fill: the primitives, with a bounded
Lagrange prolongation (AMR-10), the conserved state in the regrid (see
"Refinement").

**Step 7.** Added in step 7: the reconstruction the stars run with — MC,
the plan's and GRaM-X's choice, or PPM, which alone of the four resolved
Balsara 3's dense shell and held every Balsara plateau but test 2's `R5`
(2.8%) within 2% (see "Validation ladder" 1b); WENO-Z smeared the shell
worst, and MP5 costs twice the steps. Decided on the TOV star (rung 4).

**Step 8.** *(Closed in step 8.)* Conservative prolongation width for the
matter at stellar surfaces: width 3, bounded (TreeAMR's limited operator,
AMR-4); the TOV surface's check was step 10's. *(Closed in step 8.)*
PointValue width for `A` and `Φ̃`: 4; the gauge damping: `ξ = 16/L` on
every mesh, `L` the box's longest side (amended, decided by Erik
2026-10-15; see "Refinement").

**2026-10-23, decided by Erik.** *(Amended 2026-10-23:)* the primitives
with the bounded conservative prolongation centred on the parent; the three
working fills are recorded in "Refinement", to revisit when the physics
gets harder.

**Step 9.** Added in step 9: the interior's form — the hard paste in both
limiters at `r_in = M` (proposed), against TreeGH's smooth damping layer
inside the right-hand side ("The interior"); the analytic metric pointwise
(44% of a step on a 16³ blast) or cached per stage, or its repeated vertex
and edge evaluations deduplicated ("Metric providers"); a spinning Wald
case, with a passive fluid at the floor (Kerr has no closed-form accretion
flow; "Validation ladder" 3). *(Measured in step 9:)* `D s` on the
stationary Kerr-Schild flows drifts further from `D s_exact` than `D` does.

**Step 9, round 3.** Added in step 9, round 3: from Erik's decision on the
interior, the field-only interior for a hole that forms in a collapse; and
the potentials' outer boundary where no exact `A` exists (step 10's star,
dynamic cases).

**Step 9, round 4.** The spinning Wald case was built in step 9, round 4;
its `t = 50M` drift fails the prediction on the finest pair at both spins,
not localised further ("Validation ladder" 3, "The spinning Wald case").
For Erik then: whether step 9 closes on it, and the paste's ball at
`a = 0.9M`.

**2026-11-03, decided by Erik.** The interior's form: the paste, for the
analytic rung-3 cases. The analytic metric: pointwise, no cache, no
deduplication; caching is step 10's (and TreeGH's since 2026-11-06). The
potentials' outer boundary: in step 10 — the Cowling TOV star is static, so
an exterior field from an exact static potential takes
`static_potential = true` (option 1) as is; options 2 and 3 matter for
dynamic cases only.

**2026-11-05.** *(Resolved by Erik 2026-11-05:)* step 9 is accepted as is;
the late drift, the paste's ball at high spin and the floor fluid's resets
are left for future black-hole work ("Known risks").

**2026-11-06.** The metric's caching became TreeGH's.

**2026-11-08, step 10, decided by Erik.** Closed in step 10, each measured
in step 10 (the measurements are in `CODE.md`, "Validation ladder" 4, "What
the star decides"):
- Riemann wave speeds: light cone for all variables (the default), or
  split. The light cone; the split speeds fail on the Cowling star
  ("Riemann solver").
- The conservative prolongation width for the matter at stellar surfaces:
  width 3 bounded stays.
- Whether the atmosphere reset needs its stage-limiter role: both roles
  stay.
- The reconstruction the stars run with: PPM for stars.

The question "`(ρ, S_i, S_ij)` to vertices: 2nd order (default) or 4th"
was about `T_ab` until 2026-11-08.

**2026-11-09.** *(Measured 2026-11-09, reverted by Erik the same day:)*
the reset writing `√γ D s = (√γ D) s` from the recovered `s` in every cell
it did not repair, in both roles. The suite beside cf021db's in one job
(593879, cn088, one thread): 55 of 26538 fail, every one through `D s` —
its totals no longer conserved (3·10⁻⁷ of the scale a step on the EM wave,
0.1–14% of the boundary budget on the Balsara tubes, the entropy their
shocks make), the reset no longer a no-op where it repairs nothing, the
interior's bit identity, and the Cowling star's vertex-against-analytic
difference (3.1e−2, all of it `D s`; `D`, `S_i`, `τ`, `D Yₑ` at
0.3–2.1·10⁻⁴). `D`, `S_i`, `τ`, `D Yₑ`, `A_j` and `𝔈` moved in the 9th to
12th digit only (the recovery's seed), and the `D s` errors grew eightfold
(EM wave 4.1e−6 → 3.4e−5, Alfvén 1.4e−3 → 1.1e−2).

## Known risks

**Step 0.** "No MPI yet: single node, although TreeAMR now has its M7" was
amended in step 0.

**2026-10-14.** Decided 2026-10-14: the magnetization cutoff takes σ to
zero where the implicit stage converged poorly.

**Step 3.** The implicit stage's convergence where magnetization and `σΔt`
are both large was measured in step 3, pointwise.

**Step 6.** The diffusion of the light-cone fluxes (Balsara 2's contact,
Balsara 3's dense shell) was measured in step 6.

**Step 7.** MC's entropy floors behind the cylindrical explosion's front
and MP5's instability above `cfl = 1/5` were measured in step 7.

**Step 8.** The matter's operator has been bounded since step 8.

**2026-11-05, decided by Erik.** Spacetimes with black holes are not
expected to work out of the box (the same holds for
TreeGeneralizedHarmonic). Step 9 validated the scheme on analytic
Kerr-Schild backgrounds with the hard paste; what it also showed is left
for future work on black holes.

## The test suite

**2026-10-13, step 0.** Measured at the end of step 0 (2026-10-13), each
from a clean `git archive` in a depot that already had the packages, on this
shared machine at a load of about 6 on 12 cores: **219 tests**, at every
version and thread count — at 1.13, 24 s wall per `Pkg.test` at one thread
and at four, 39 s with `--check-bounds=yes`; at 1.11, 44 s, 29 s and 30 s
(the first of them paid for compiling the test environment). The suite's
own `Test Summary` time is 15–19 s throughout. In a fresh depot (nothing
installed, registry and every package fetched) the 1.11 instantiate took
77 s and the test 92 s; it resolved EntropyEOS and IMEXRungeKutta from their
GitHub URLs, as it must. The log named every upstream ID still probed as
landed or waiting; since 2026-10-14 a landed item guaranteed by `[compat]`
has no probe.

**2026-10-13, step 2.** At the end of step 2 (2026-10-13): **20513 tests**,
a `Test Summary` time of 37–39 s at 1.11 and 28 s at 1.13 at one thread,
four and `--check-bounds=yes`; `test/pointwise_tests.jl` takes about 12 s of
it.

**2026-10-14, step 5.** At the end of step 5 (2026-10-14): **24638 tests**,
a `Test Summary` time of 1m31s at 1.13 at one thread and at four, 1m35s
with `--check-bounds=yes`, and 1m45s and 1m52s at 1.11 at one and four
threads (24292 tests and 40 s at 1.13 before the step); the two step-5
files take about 75 s of it, more than half of that compiling the scheme's
kernels (`test/scheme_tests.jl` and `test/wave_tests.jl`, the convergence
studies, ~50 s, half of it compilation).

**2026-10-14, step 6.** At the end of step 6 (2026-10-14): **24882 tests**,
a `Test Summary` time of 2m42s at 1.13 and 3m25s at 1.11 at one thread
(2m55s and 3m03s at four, 2m57s with `--check-bounds=yes`); the three
step-6 files take about 95 s of it (`test/shocktube_tests.jl`,
`test/resistive_tests.jl`, `test/atmosphere_tests.jl`: ~45, 37 and 11 s);
24638 tests and 1m31s before the step. Every printed number agrees across
versions and thread counts.

**2026-10-15, step 7.** At the end of step 7 the suite is **25293 tests**: `Test Summary` 5m23s at 1.13 and 6m43s at 1.11 at one thread,
7m17s and 8m14s at four, 5m35s with `--check-bounds=yes` (1.13) — on a machine at a load of
8–17 on 12 cores, two suites running at once; 24882 tests and 2m42s (1.13, one thread)
before the step. The new files, alone at one thread: `multid_tests.jl` 67 s (35 s of it
compiling the 2D and 3D kernels), `multid_cases_tests.jl` 60 s, `reconstruction_tests.jl`
35 s; `shocktube_tests.jl` grew to 47 s. Every printed number is identical at one and four
threads; 1.11 and `--check-bounds=yes` agree with each other exactly and differ from 1.13
only in the last one or two digits of L1 errors and conservation drifts (Orszag–Tang's
0.056 ulp a step is 0.037 there).

**Inconsistent:** `CLAUDE.md` ("Commands") recorded the end of step 7 (2026-10-15) as "a `Test
Summary` time of about 5m20s at 1.13 and 6m40s at 1.11 at one thread (7m20s and 8m15s at four,
5m35s with `--check-bounds=yes`), on this machine at a load of about 10 with two suites running at
once"; `PLAN.md`'s record, the one above, has 5m23s, 6m43s, 7m17s and 8m14s, at a load of 8–17.

**2026-10-15.** 1.11 and 1.12 forced `--check-bounds=yes`, so their plain run was the
`--check-bounds=yes` one, and the 1.11-against-1.13 times recorded here
compare the two settings as well as the two versions (measured 2026-10-15).
With the floor at 1.13 (2026-10-15, at a load of about 5–9): **25293
tests**, a `Test Summary` time of 5m01s at one thread, 5m06s at four, 5m34s
with `--check-bounds=yes` and 5m08s for the clean-checkout check, about
150 s of the one-thread run compiling (161 s with bounds checks); every
printed number identical at one and four threads, and to the runs before
the floor changed.

**2026-10-15, step 8.** At the end of step 8 (2026-10-15, Julia 1.13, at a
load of 5–10 with other jobs running): **25369 tests**, a `Test Summary`
time of 6m36s at one thread, 7m13s at four, 7m13s with `--check-bounds=yes`
and 6m34s for the clean-checkout check, against 25293 and 5m08s before the
step; identical output at one and four threads and from the clean checkout,
last digits only with bounds checks (the tracked tube's drift 0.017 → 0.015
ulp, the rotor's 0.036 → 0.040). Every number printed before the step is
unchanged but the walls' (see the first commit). The two step-8 files take
73 s and 44 s alone, compilation included. Under the 10-minute budget, so
no split into a default and a long set. After the
reviewer's round (one `ξ` on every mesh, the complete boundary condition
after every fill, AMR-10 requested): **25373 tests**, 6m45s at one thread,
7m10s at four, 7m20s with `--check-bounds=yes`; identical output at one and
four threads, last digits only with bounds checks.

**2026-10-16, step 8, the finish.** At the end of step 8's finish on
TreeAMR 0.2.2 (2026-10-16, Julia 1.13.2, at a load of 4–6): **25405
tests**, a `Test Summary` time of 6m34s at one thread, 6m59s at four, 7m14s
with `--check-bounds=yes` and 6m37s for the clean-checkout check (wall
6m43s, 7m07s, 7m41s, 6m56s), against 25374 and 6m31s on 5f82040 with
TreeAMR 0.2.2; identical output at one and four threads and from the clean checkout, last
digits only with bounds checks (60 of 256 lines; the tracked tube's drift
0.015 → 0.014 ulp). Under the 10-minute budget, so no split into a default
and a long set. The same
afternoon 1.12 took 5m22s (6m00s with bounds checks, its `Pkg.test()`
default) and 1.11 6m05s with bounds checks — see `CODE.md`, "Julia
version".

**2026-10-24.** After the parent-centred fill became the only one
(2026-10-24, on Symmetry's `amddebugq`, TreeAMR 0.2.6 as resolved there):
**25950 tests** (d456a44 had 25876: +64 from the merged recovery test, +10
the new ghost testset), a `Test Summary` time of 10m18s at one thread, 8m38s
at four and 10m17s for the clean-checkout check on cn112 (EPYC 8534P), and
14m12s with `--check-bounds=yes` on cn084 (EPYC 7532); every printed line
identical at one and four threads, from the clean checkout and to d456a44's
on the same node (but one renamed `@info` label); another CPU moves last
digits. With bounds checks four allocation assertions fail, as they do at
d456a44 (since 2aae714): EntropyEOS 1.1's `con2prim_safe` allocates 544 B on
the floored input there under `--check-bounds=yes`.

**2026-10-24, EntropyEOS 2.0.** After EntropyEOS 2.0 (2026-10-24,
`claude/recovery-pipeline-2`, Symmetry `amddebugq`): **26196 tests** (+200
the Jacobian stencil's check, +40 the rewritten ρ_min-face testset, +9 the
2.0 names, −3 the units check), a `Test Summary` time of 11m50s at one
thread and 9m11s at four on cn112 (EPYC 8534P), the stage-1 baseline
(d94bcf5, 25950 tests) 10m17s in the same job there; 11m50s for the
clean-checkout check on cn112 and 14m14s on cn080 (EPYC 7532); 15m50s with
`--check-bounds=yes` on cn079 (EPYC 7532). One and four threads identical,
and the clean checkout identical to the one-thread run on the same node
(another CPU moves last digits). With bounds checks the four allocation
assertions of the ρ_min-face testset fail, 272 B and 144 B, in EntropyEOS's
nested fallback (EOS-10 in `PLAN.md`; waiting on it, decided by Erik
2026-10-24); every other run passes whole.

**2026-10-24, the step cache.** After the step cache (2026-10-24,
`claude/recovery-pipeline-3` at 1462923, Symmetry, AMD EPYC 7532
throughout): **26268 tests** (+72, `test/step_cache_tests.jl`), a
`Test Summary` time of 10m32s at one thread on cn079 against the stage-2
baseline's 14m00s in the same job (a9cd482, 26196 tests, every test
passing), 9m42s at four threads and 10m41s for the clean-checkout check on
cn080, 12m26s with `--check-bounds=yes` on cn084. One and four threads
identical, the clean checkout identical to the one-thread run. **Four
failures in every run**: the uniform flow's slow cases leave `≤ 10⁻¹¹`
(`1.1–8.4·10⁻¹¹`, warm seeds answering to EntropyEOS's tolerance); with
bounds checks also EOS-10's four. *(Both resolved 2026-10-30: the uniform
flow is asserted to the recovery's tolerance, `≤ 10³·tol`, with a 1000-step
run besides, decided by Erik; EOS-10 landed in EntropyEOS 2.2.1.)*

**2026-10-26.** On EntropyEOS 2.2.0 with `inside_window` (2026-10-26,
`claude/entropyeos-2.2`, Symmetry, AMD EPYC 7532 throughout): **26296
tests**, 26268 → 26296 (+18 the reconstruction's slack testset, the new
reconstruction testset, +8 the step-cache entry
read back, +2 names), a `Test Summary` time of 10m46s at one thread on
cn079 against stage 3's 10m44s in the same job (4929c94, 26268 tests),
9m50s at four threads and 10m49s for the clean-checkout check on cn080,
12m27s with `--check-bounds=yes` on cn081. One and four threads identical,
the clean checkout identical to the one-thread run; against stage 3 only
ARS(4,4,3) near the vacuum moves (three lines; [Primitive
recovery](#primitive-recovery), "EntropyEOS 2.1 and 2.2"). The same **four failures** (the uniform flow, stage 3's numbers
to every digit), and with bounds checks EOS-10's four.

**2026-10-30.** On EntropyEOS 2.2.1 with the uniform flow asserted to the
recovery's tolerance (2026-10-30, `claude/entropyeos-2.2` at 5e516c6,
Symmetry, AMD EPYC 7532 throughout): **26304 tests** (+8, the 1000-step
uniform flows), **every run passing whole**: a `Test Summary` time of
10m49s at one thread on cn085 against 16c48ed's 10m42s in the same job,
9m58s at four threads and 10m48s for the clean-checkout check on cn079,
12m58s with `--check-bounds=yes` on cn080. One and four threads identical,
the clean checkout identical to the one-thread run, and every printed line
identical to 16c48ed's (EntropyEOS 2.2.0) but the four new ones.

**2026-10-30, the two-phase recovery.** After the two-phase recovery
(2026-10-30, `claude/recovery-pipeline-4` at 0334aed, Symmetry, AMD EPYC
7532 throughout): **26353 tests** (+49, `test/two_phase_tests.jl`),
**every run passing whole**: a `Test Summary` time of 11m17s at one thread
on cn082 against fea242a's 11m05s in the same job, 10m21s at four threads
there, 11m19s for the clean-checkout check and 13m16s with
`--check-bounds=yes` on cn079. One and four threads identical, the clean
checkout identical to the one-thread run, every printed line identical to
fea242a's but the nine new ones, and bounds checks moving last digits only
(68 lines).

**2026-11-02.** With fused the default (2026-11-02, 03566f4, Symmetry):
**26353 tests**, every run passing whole: 11m04s at one thread on cn079
(EPYC 7532) against fea242a's 10m54s in the same job, 10m06s at four
threads and 11m13s for the clean-checkout check on cn080 (EPYC 7532), 9m46s
with `--check-bounds=yes` on cn093 (EPYC 7543). One and four threads
identical, the clean checkout identical to the one-thread run, every printed
line of the scheme identical to fea242a's but the nine new ones; six lines
of the pointwise metric tests moved at 10⁻¹⁵ because the registry resolved
ForwardDiff 1.4.7 (and TreeAMR 0.2.7) for the new tree, against 1.4.6 (and
0.2.6) for fea242a's of 2026-10-30.

**2026-11-02, step 9.** Step 9 (2026-11-02,
`claude/step-9-curved-backgrounds` at af0c482, Symmetry, AMD EPYC 7532):
**26425 tests** (+72, `test/curved_tests.jl`), one failure (the Michel `uʳ`
bound, 1.006e−14 against 1e−14 on the EPYC, fixed in cb41add), a
`Test Summary` time of 14m35s at one thread on cn082 (the curved file about
3.5 min of it, compilation included: about 3.5 min more than step 8's, the
curved file's 72 tests and its four evolutions; close to the 15-minute
line, where a split was proposed); the same one failure at four threads (cn080) and with bounds
checks (cn081). At cb41add (jobs 588117–588119): **26425 tests, every run
passing whole** — 14m21s at one thread on cn081 (EPYC 7532) against
9e686a2's 11m08s (26353 tests) in the same job, 11m38s at four threads and
14m28s for the clean-checkout check on cn092 (EPYC 7532), 12m38s with
`--check-bounds=yes` on cn093 (EPYC 7543). One and four threads identical,
the clean checkout identical to the one-thread run, every line of the
baseline present in the one-thread run (11 new lines); bounds checks, on the
other CPU, move 70 lines. The suite stays whole at about 14 minutes, the
curved file in it (decided by Erik 2026-11-03: no split; `test/curved_tests.jl`
stays in the main suite). `test/curved_tests.jl` took ~2 min with compilation.

**2026-11-03.** With option 1, the static potentials' absolute hook
(2026-11-03, 5bea724, Symmetry, AMD EPYC 7532 throughout): **26428 tests**
(+3, the boundary testset), every run passing whole — 14m20s at one thread
on cn079 against cb41add's 14m16s in the same job, 11m51s at four threads
and 14m35s for the clean-checkout check on cn082, 16m55s with
`--check-bounds=yes` on cn080. One and four threads identical, the clean
checkout identical to the one-thread run, every line of cb41add's present
but the six curved lines the absolute hook moves (seven new ones in their
place); bounds checks move 70 lines.

**2026-11-03, step 9, round 4.** With the spinning Wald case (round 4,
2026-11-03, 0858776, Symmetry): **26442 tests** (+14), every run passing
whole — 14m48s at one thread on cn082 (EPYC 7532) against 5bea724's 14m38s
in the same job, 9m00s at four threads and 11m01s for the clean-checkout
check on cn107 (EPYC 7543), 17m03s with `--check-bounds=yes` on cn083 (EPYC
7532); one and four threads identical, the clean checkout identical to the
one-thread run, every line of 5bea724's present (three new); bounds checks
move 70 lines.

**2026-11-06, step 10.** Step 10 (2026-11-06, the tree of 14e3b87,
Symmetry, amddebugq): **26516 tests** (+74, `test/tov_tests.jl`), every run
passing whole — 13m46.7s at one thread on cn110 (EPYC 7543) against main's
11m03.6s (26442 tests) in the same job, 14m03.1s at four threads and
18m31.3s for the clean-checkout check on cn087 (EPYC 7532), 20m57.7s with
`--check-bounds=yes` on cn086 (EPYC 7532); one and four threads identical,
the clean checkout identical to the one-thread run, every line of main's
present (seven new); bounds checks move 74 lines.

**2026-11-08.** The suite runs in tiers, the ones GitHub CI runs
(`test/tiers.jl`, 2026-11-08, after TreeGeneralizedHarmonic's of
2026-11-06); the runs several testsets read
moved from top-level constants to `computed_once`, and the helpers shared
between files moved to files of their own. Measured on Symmetry (2026-11-08,
jobs 593535–593537): the full tier prints every line the suite printed
before, at one thread and at four, 26442 tests passing.

**2026-11-08.** The full suite as a commit status (`ci/full-suite.sh`,
`ci/full-suite-symmetry.sh`) was added 2026-11-08, from TreeGH's scripts.
Step 10 rebased onto main's tiers (2026-11-08):
`test/tov_tests.jl`'s testsets run in every tier, unmarked (the file alone
2m25s on the laptop, compilation included). The full tier on Symmetry
(2026-11-08, jobs 593564–593566): **26516 tests**, every run passing whole
— 13m52.5s at one thread on cn103 (EPYC 7543) against main's (a19097d)
11m12.6s (26442 tests) in the same job, 14m20.0s at four threads and
18m54.7s for the clean-checkout check on cn087 (EPYC 7532), 16m06.1s with
`--check-bounds=yes` on cn103; one and four threads identical, the clean
checkout identical to the four-thread run, every line of main's present
(seven new), every line identical to the run before the rebase; bounds
checks move 72 lines.

**2026-11-08, step 10, round 2.** The suite at round 2 (2026-11-08,
Symmetry cn103, EPYC 7543, uncommitted tree): **26538 tests, every run
passing whole** — 16m19s at one thread, 11m41s at four, 20m44s with
`--check-bounds=yes`; one and four threads identical, bounds checks moving
70 lines.
