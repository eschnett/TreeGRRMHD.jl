# Working notes for Claude in TreeGRRMHD.jl

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

## What this package is

General-relativistic **resistive** magnetohydrodynamics on
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl): the Valencia matter
equations plus the full Maxwell equations with an Ohm's law, on a
cell-centered finite-volume scheme with the vector potential on edges. It
is the fourth downstream application of TreeAMR, after TreeWave (scalar
wave), TreeHydro (Newtonian hydro, the rehearsal for this package) and
TreeGeneralizedHarmonic (the spacetime this package couples to). The
equation of state and the primitive recovery come from EntropyEOS. The end
goal is neutron stars and core-collapse supernovae; the first science
target is a magnetized TOV star.

Rules that follow from `CODE.md` and govern every change here:

- **No mesh machinery.** Trees, ghosts, inter-level operators, reductions
  and point interpolation belong upstream in TreeAMR. Interpolation
  *within a block* between centerings (metric from vertices; `T_ab` to
  vertices) is application code and lives here.
- **Upstream changes are Erik's.** When this package needs something from
  TreeAMR, TreeGeneralizedHarmonic or EntropyEOS, it goes into `CODE.md`'s
  "Upstream prerequisites" as a precise request; do not edit the sibling
  checkouts, and do not work around a missing upstream feature silently —
  say which item a step is waiting for.
- **One con2prim.** EntropyEOS's `con2prim_safe` is the only primitive
  recovery, called in hydrodynamic mode (EM energy and momentum subtracted,
  `B² = 0`) for every σ; its ideal-MHD mode is only the seed of the
  implicit stage solve. Do not add a second inversion.
- **∇·B is structural.** `B` is always the discrete curl of the edge-centered
  `A` (plus the constant background `𝔅̄`). Never evolve, prolongate or
  store `B` as independent data.
- **Spec-first.** When the implementation shows `CODE.md` wrong or
  incomplete, amend it and say so ("(amended in step N)", "(measured in
  step N)").

## Current state

**Steps 0 (scaffolding), 1 (the pointwise recovery), 2 (the pointwise
physics), 3 (the implicit stage solve), 5 (the staggered scheme in 1D,
rung 1a), 6 (shock tubes and the stiff limit, rung 1b), 7 (2D, 3D and
higher-order reconstruction, rung 2a) and 8 (coarse-fine faces and
regridding, rung 2b) are done; step 9 (analytic curved backgrounds) is next
and waits for nothing (2026-10-15).** `CODE.md` records the reviewed design
(2026-09-23); `PLAN.md` is the implementation plan: steps 0–15 on
`claude/step-N-slug` branches, run one agent per step, and the exact
upstream requirements under IDs `AMR-n`, `GH-n`, `EOS-n`, each naming the
step it blocks, with a state column. What exists: the module shell,
`src/precision.jl` and `src/device.jl` (ported from TreeHydro),
`test/prerequisite_tests.jl` (the names the scheme calls, the IMEX
contract, and a probe per upstream ID still waiting; a landed item whose
release `[compat]` requires loses its probe — `PLAN.md`, "Upstream
requirements"), step 1's pointwise layer —
`src/units.jl` (geometric units, `GeometricEOS`), `src/metric3.jl`,
`src/variables.jl` (the slot layouts of `U` and `P`), `src/prim2con.jl` and
`src/con2prim.jl` (`con2prim_hydro`, `con2prim_ideal`) — and step 2's —
`src/metric.jl` (the `Flat`/`Analytic` providers and `MetricGradient`, the
one place SpacetimeMetrics' derivative index order is converted),
`src/stress_energy.jl`, `src/ohm.jl`, `src/flux.jl`, `src/riemann.jl`,
`src/emf.jl` and `src/sources.jl` — and step 3's `src/stage_solve.jl`
(`solve_ohm_stage`, Newton on the fluid's share of the energy and momentum,
`OhmStageResult`), with `MagnetizationCutoff(model, μ_max)` in `src/ohm.jl`
tapering σ to zero where `B²/D` exceeds `μ_max` (decided by Erik
2026-10-14; the bracketed alternative is on
`claude/step-3-bracketed-stage-solve`, not in the code). **`μ_max` has no
default**: every case and driver states its own, and nothing wraps a model
implicitly; 10 is the measured safe value (`CODE.md`, "Ohm's law and the
conductivity"). `test/pointwise_tests.jl` and `test/stage_solve_tests.jl`
use `recovery_tests.jl`'s state generators and are included after it. Steps
4a–4c are done in `~/src/jl/IMEXRungeKutta` (1.4.0), whose `CODE.md` is the
IMEX design. EOS-1 and EOS-2 have landed (EntropyEOS 1.0.0, still
unregistered); TreeAMR 0.2.1 has AMR-1, -2, -3, -5, and AMR-4 and AMR-6 in
another form (`bounded = true`, outflow faces); SpacetimeMetrics 1.7.0 has
SM-1, whose workaround is gone. Step 5's mesh code: `src/fieldsets.jl`
(`mhd_fields`, every set of `CODE.md`'s table in D = 1, 2, 3, and the
diagnostics set `X`), `src/curl.jl` (`curl_A!`, `𝔅 = curl A + 𝔅̄` on every
stored face; `cell_𝔅`), `src/reconstruction.jl` (PLM, the positivity
fallback, the face state), `src/evolution.jl` (`MHDProblem`, the kernels,
`mhd_rhs!`, `ohm_solve!`, `reset_atmosphere!`, `ohm_tally`,
`max_light_rate`, the conserved totals), `src/driver.jl` (`MHDCase`, exact
initial data and references, `l1_errors`, `mhd_integrator`, `evolve!`) and
`src/cases/` (`em_wave_case`, `alfven_wave_case`). The kernels are written
for D = 1, 2, 3; step 7 validated 2D and 3D (below). The RHS is
pure bit for bit because every recovery is seeded from the state
(`state_guesses`), never from `P`. Tests: `test/scheme_tests.jl` and
`test/wave_tests.jl` (the convergence studies, ~50 s, half of it
compilation). Step 6: non-periodic cases get Dirichlet hooks from their own
data (`DirichletBoundary` in `src/driver.jl`; the potentials' are Dirichlet
in the field and CPU-only, AMR-7), the reset is counted as stage and step
limiter with the step limiter's injection (`reset_step!`, `ohm_tally`), and
`src/cases/shocktubes.jl` has Balsara 1–5 with Giacomazzo & Rezzolla's
exact plateaus, the resistive tube, Komissarov's current sheet and the
atmosphere tube. `bin/shocktube_reference.jl` made the 3200-cell
references in `test/data/` (rerun it only if the scheme changes, and say
so); `bin/tableau_comparison.jl` made `CODE.md`'s SSP3-vs-ARS table.
**SSP3(4,3,3) is the default tableau** (decided in step 6). Tests:
`test/shocktube_tests.jl`, `test/resistive_tests.jl`,
`test/atmosphere_tests.jl` (~45, 37 and 11 s). Non-periodic runs are
CPU-only until AMR-7 (waiting). Step 7: `limiter` also takes `:ppm`,
`:wenoz`, `:mp5` (need `G = 3`; MP5 runs at `cfl = 1/5`, `default_cfl`),
every reconstruction an offset in the differences `q_k − q₀` so that 2D/3D
runs invariant along `y`/`z` are the 1D run bit for bit; the initial data's
Gauss average is in difference form for the same reason. `max_div_B` and
`evolve!`'s `after_step` monitor; `src/cases/geometry.jl`
(`extruded_case`, `turned_case`, `oblique_case`) and `src/cases/multid.jl`
(explosion, rotor, Orszag–Tang, field loop). `bin/reconstruction_comparison.jl`
made the per-reconstruction Balsara table and `test/data/balsara3_ppm.ref`.
Tests: `test/reconstruction_tests.jl` (uses `wave_tests.jl`'s helpers),
`test/multid_tests.jl` and `test/multid_cases_tests.jl` (uses the former's
helpers; ~35, 67 and 60 s at one thread, 35 s of the 67 compiling the 2D and
3D kernels). Step 8: both walls of a closed box prescribed alike
(`low_walls!`, `outer_points!` in `src/driver.jl`, AMR-8's workaround,
decided by Erik 2026-10-15), so a closed box is mirror-symmetric bit for
bit; `src/refinement.jl` (`mhd_flags`, Löhner on `ρ`, `|B|`, `|E|`;
`regrid_pairs`, `regrid_state!`); `evolve!` with a static mesh
(`refine`) or an adaptive one (`maxlevel`, `refine_tol`, `coarsen_tol`,
`buffer`; `adapt_case!` over every set, AMR-3). Decided by measurement
(`CODE.md`, "Refinement"): the matter's operators Conservative width 3
**bounded**, the potentials' PointValue **width 4**, `ξ = 16/L` on every mesh
(`L` the longest side; Erik's decision, 2026-10-15), and the ghost fill in conserved variables
(primitives open until steps 10 and 12). AMR-9 (a flux-conservative
transfer of edge sets) is new and waiting: the magnetic flux across levels
holds to rounding as a rate, not as a state. `bin/interface_order.jl`,
`bin/gauge_damping.jl`, `bin/field_loop_refined.jl` made `CODE.md`'s step-8
tables. Tests: `test/refinement_tests.jl` and
`test/refinement_cases_tests.jl` (use `multid_tests.jl`'s helpers).

## Commands

TreeHydro's discipline (`TreeHydro/CLAUDE.md`, "Commands"): the whole
suite at one thread, at four threads, and with `--check-bounds=yes`, at
the floor version (1.13: `julia +1.13`) and at the current release (plain
`julia`; until 1.14 is released, the two are the same). `Pkg.test` does
not inherit `-t`, so the thread count has to be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
```

At 1.13 `Pkg.test()` runs with bounds checks at `auto`; 1.11 and 1.12
forced `--check-bounds=yes`, so their plain run was this third one, and
the 1.11-against-1.13 times recorded below compare the two settings as
well as the two versions (measured 2026-10-15).

The clean-checkout check: a tree with no `Manifest.toml` resolves TreeAMR
and SpacetimeMetrics from the General registry and EntropyEOS and
IMEXRungeKutta from their remotes through `[sources]` — never from a
sibling checkout. This is what CI does; run it under the floor too:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia +1.13 --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

A `Manifest.toml` resolved by one Julia version breaks loading under
another (1.13 refused a 1.11 manifest's `MbedTLS_jll`), so delete it, or
use a clean tree, when switching versions.

Measured at the end of step 0 (2026-10-13), each from a clean `git archive`
in a depot that already had the packages, on this shared machine at a load
of about 6 on 12 cores: **219 tests**, at every version and thread count —
at 1.13, 24 s wall per `Pkg.test` at one thread and at four, 39 s with
`--check-bounds=yes`; at 1.11, 44 s, 29 s and 30 s (the first of them
paid for compiling the test environment). The suite's own
`Test Summary` time is 15–19 s throughout. In a fresh depot (nothing
installed, registry and every package fetched) the 1.11 instantiate took
77 s and the test 92 s; it resolved EntropyEOS and IMEXRungeKutta from
their GitHub URLs, as it must. The log names every upstream ID still
probed as landed or waiting; grep it for `landed:` and `waiting:` (since
2026-10-14 a landed item guaranteed by `[compat]` has no probe, so
`landed:` should not appear). At the end of step 2
(2026-10-13): **20513 tests**, a `Test Summary` time of 37–39 s at 1.11
and 28 s at 1.13 at one thread, four and `--check-bounds=yes`;
`test/pointwise_tests.jl` takes about 12 s of it. With the floor at 1.13
(2026-10-15, at a load of about 5–9): **25293 tests**, a `Test Summary`
time of 5m01s at one thread, 5m06s at four, 5m34s with
`--check-bounds=yes` and 5m08s for the clean-checkout check, about 150 s
of the one-thread run compiling (161 s with bounds checks); every printed
number identical at one and four threads, and to the runs before the
floor changed. At the end of step 8 (2026-10-15, at a load of 5–10 with
other jobs running): **25369 tests**, a `Test Summary` time of 6m36s at
one thread, 7m13s at four, 7m13s with `--check-bounds=yes` and 6m34s for
the clean-checkout check; identical output at one and four threads and
from the clean checkout, last digits only with bounds checks. The two
step-8 files take 73 s and 44 s alone, compilation included. The same afternoon 1.12 took 5m22s (6m00s with bounds
checks, its `Pkg.test()` default) and 1.11 6m05s with bounds checks — see
`CODE.md`, "Julia version". At the end of step 7
(2026-10-15): **25293 tests**, a `Test Summary` time of about 5m20s at
1.13 and 6m40s at 1.11 at one thread (7m20s and 8m15s at four, 5m35s with
`--check-bounds=yes`), on this machine at a load of about 10 with two
suites running at once; output identical at one and four threads, and 1.11
and `--check-bounds=yes` differ from 1.13 only in last digits. At the end of step 6 (2026-10-14): **24882
tests**, a `Test Summary` time of 2m42s at 1.13 and 3m25s at 1.11 at one
thread (2m55s and 3m03s at four, 2m57s with `--check-bounds=yes`); the
three step-6 files take about 95 s of it. At the end of step 5
(2026-10-14): **24638 tests**, a `Test Summary` time of 1m31s at 1.13 at
one thread and at four, 1m35s with `--check-bounds=yes`, and 1m45s and
1m52s at 1.11 at one and four threads (24292 tests and 40 s at 1.13 before
the step); the two step-5 files take about 75 s of it, more than half of
that compiling the scheme's kernels.

## Conventions

Match TreeAMR's, TreeHydro's and TreeGeneralizedHarmonic's, since the
packages are read together:

- 4-space indent, wrap at about 80–90 columns; `return` on the last line of
  any non-trivial function.
- KernelAbstractions for every kernel; callbacks and parameter structs are
  `isbits`; `Val`s for `D`, each `G`, the direction, the limiter and the
  solver; `@inbounds` *inside* `ntuple` closures, not outside.
- No float literals in `T`-generic code: `T(1//2)`, `oftype(x, 2)`.
- `T` as a leading positional argument defaulting to `Float64`, `backend`
  a keyword defaulting to `CPU()`; keyword-heavy drivers with no default for
  anything the caller must think about.
- StaticArrays for 3- and 4-tensors, as TreeGH does; `NTuple`s for state
  vectors, as TreeHydro does.
- Unicode in mathematical contexts; `ArgumentError`s say *why*.
- Docstrings are prose-first and point at `CODE.md`.
- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.
- **Julia 1.13 is the floor** (`julia = "1.13"` in `[compat]`), raised
  from 1.11 on 2026-10-15 by Erik's rule: the floor goes up when the older
  version is more than about 20% slower (1.12 compiled the suite 23%
  longer). The sibling Tree* packages stay at 1.11, which does not
  constrain this one; `[sources]`, which locates the unregistered
  dependencies, is a 1.11 key. See `CODE.md`, "Julia version".
- Operator orders in TreeAMR's convention: the order is the stencil width
  (PointValue even, Conservative odd), exact to degree width − 1. Write
  "width w" in prose to avoid ambiguity.

## Repository facts

- No remote yet. Work on a branch, and do not commit, push or merge
  without being asked.
- `.gitignore` is TreeHydro's: `Manifest.toml` everywhere, `bin/output/`,
  `docs/build/`, editor leftovers, `TODO.md`. No `Manifest.toml` is
  tracked.
- If Erik adds a `TODO.md`, it is his personal list: **do not modify it.**
- Sibling checkouts: `~/src/jl/TreeAMR` (the mesh; its `CODE.md` for the
  API and its sharp edges), `~/src/jl/TreeHydro` (the scheme, driver and
  test patterns this package inherits), `~/src/jl/TreeGeneralizedHarmonic`
  (the spacetime; its `CODE.md` and `notes/` for the formulation),
  `~/src/jl/EntropyEOS` (EOS and con2prim), `~/src/jl/SpacetimeMetrics`
  (analytic backgrounds), `~/src/jl/IMEXRungeKutta` (the time integrator;
  edited only in steps 4a–4c). `~/src/jl/Con2Prim` is an abandoned prototype;
  do not depend on it.
