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
regridding, rung 2b; finished on TreeAMR 0.2.2) and 9 (analytic curved
backgrounds, rung 3; `claude/step-9-curved-backgrounds`) are done. Step 9 was
accepted as is by Erik on 2026-11-05, including round 4's spinning Wald case,
whose `50M` drift fails the prediction on the finest pair. **Spacetimes with
black holes are not expected to work out of the box** (as in
TreeGeneralizedHarmonic); further work on black holes is left for the
future (`CODE.md`, "Known risks"). The interior (the paste), the coarse-pair
rates and the deduplication were decided by Erik 2026-11-03. Step
10 waits for GH-1 and GH-2.** `CODE.md` records the reviewed design
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
IMEX design. EOS-1 and EOS-2 have landed (EntropyEOS 1.0.0), EOS-8
(1.1.0), EOS-9 (2.0.0), EOS-11 (2.1.0) and EOS-10 (2.2.1; still
unregistered, `[sources]` pins the tag `v2.2.1`, `[compat]` `"2.2.1"`,
since 2026-10-30, on `claude/entropyeos-2.2`; `v2.2.0` from 2026-10-26,
`v2.0.0` from 2026-10-24). The recovery calls
EntropyEOS's two levels, `con2prim_safe` onto `ProjectionOptions` (`proj`:
`ρ_min`, `ρ_max`, `w_cap`) and `apply_policy` with `PolicyOptions`
(`pol`: the atmosphere), both carried by `MHDProblem` and `MHDCase` and
built by `recovery_options(eos, ρ_min)` (`s_atm` at 1.x's window
midpoint); no case sets `collapse_to_atmosphere`; the policy flags are
bits 8–17 (`CODE.md`, "Primitive recovery", "The two levels"); 2.2's
sound-speed bits 18–19 are the EOS evaluation's and reach only the step
cache's copy of the `EOSPoint`. The reconstruction's `admissible_fluid`
tests `s` and `Yₑ` with EntropyEOS's `inside_window`, `con2prim_safe`'s own
test with its face slack (EOS-11, 2026-10-26; `CODE.md`, "Reconstruction"):
under 2.0 the exact test rejected cells `con2prim_safe` returned unflagged
within the slack, and ARS(4,4,3) near the vacuum regressed (48/48 cut-off
tubes blew up); with `inside_window` every perturbed tube row is
deterministic again and the regression is gone (`CODE.md`, "Floors, the
atmosphere and the magnetosphere", "EntropyEOS 2.0"). Step 3's uncut sweep
falls back more under 2.x (its bound of 12 approved); `s_atm` stays the
midpoint; EOS-10 landed in EntropyEOS 2.2.1 (`[sources]` pins `v2.2.1`
since 2026-10-30), so the bounds-checked run's four allocation failures are
gone. Stage 3 of the recovery plan
(`claude/recovery-pipeline-3`, 2026-10-24): **the step cache** — inside
`evolve!`'s steps (`with_step_cache`) every recovery is seeded from the
cell's latest one in the step and reused where its EntropyEOS input is that
one's bit for bit; invalidated by the step limiter and around the stepping,
so the callbacks are pure in `(u, t)` outside a step and a step is pure in
`uⁿ` (`CODE.md`, "Primitive recovery", "The step cache"; the entry and
`two_levels` in `src/con2prim.jl`, the cache on `MHDProblem`, the seed
tallies in `X`; `evolve!(…; step_cache = false)` the negative control;
`test/step_cache_tests.jl`; `bin/recovery_cost.jl steptime`). Stage 4
(`claude/recovery-pipeline-4`, 2026-10-30): **the two-phase recovery** —
every recovery outside the stage solve's iterations (the right-hand side,
both limiters, the stage's first recovery) runs as three kernels over the
owned cells: `start_kernel!` (the input and `con2prim_safe_start`),
`finish_kernel!` (`con2prim_safe_finish` behind the per-cell `todo` guard,
no compaction) and the site's own kernel (level 2 and the rest), through
the per-owned-cell `Handoff` on `MHDProblem`; level 1 and level 2 never in
one kernel; the iterations stay fused. `MHDProblem`/`evolve!(…; two_phase =
true)` turns it on; **fused is the default** (decided by Erik 2026-11-02),
bit for bit the same (`test/two_phase_tests.jl`;
`CODE.md`, "The two-phase recovery"; `bin/recovery_cost.jl steptime` times
both, `finish` counts the share that reaches finish). On the CPU the split is
1–4% slower per step than fused (the 1D waves 6–9%), and the iterations never
reach finish (measured 2026-10-30), so the CPU runs fused and the split
waits for a GPU step. The ideal magnetized atmosphere's `W = 100` spikes
under ARS are EntropyEOS 2.x's `s_max` rule (keep `D` and `S`), accepted
(Erik, 2026-11-02; `CODE.md`, "Floors, the atmosphere and the
magnetosphere"). TreeAMR 0.2.1 has AMR-1, -2, -3, -5, and AMR-4 and AMR-6 in
another form (`bounded = true`, outflow faces), and 0.2.2 (the compat floor
since step 8) AMR-7 to AMR-10 (AMR-11, new in step 8, waiting); SpacetimeMetrics 1.7.0 has
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
in the field, since step 8 TreeAMR's relative hook, AMR-7), the reset is counted as stage and step
limiter with the step limiter's injection (`reset_step!`, `ohm_tally`), and
`src/cases/shocktubes.jl` has Balsara 1–5 with Giacomazzo & Rezzolla's
exact plateaus, the resistive tube, Komissarov's current sheet and the
atmosphere tube. `bin/shocktube_reference.jl` made the 3200-cell
references in `test/data/` (rerun it only if the scheme changes, and say
so); `bin/tableau_comparison.jl` made `CODE.md`'s SSP3-vs-ARS table.
**SSP3(4,3,3) is the default tableau** (decided in step 6). Tests:
`test/shocktube_tests.jl`, `test/resistive_tests.jl`,
`test/atmosphere_tests.jl` (~45, 37 and 11 s). Non-periodic runs were
CPU-only until AMR-7 landed (step 8); now every backend. Step 7: `limiter` also takes `:ppm`,
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
3D kernels). Step 8: both walls of a closed box prescribed alike (decided
by Erik 2026-10-15; since TreeAMR 0.2.2 the potentials' `walls =
:symmetric` and relative hook, AMR-7 and AMR-8, where step 8 first had its
own `low_walls!`, `outer_points!`), so a uniform closed box is
mirror-symmetric bit for bit; the owned low wall is prescribed: zero
tendency, copied into `u` by the reset (`walls_to_state!`); `src/refinement.jl` (`mhd_flags`, Löhner on `ρ`, `|B|`, `|E|`;
`regrid_pairs`, `regrid_state!`); `evolve!` with a static mesh
(`refine`) or an adaptive one (`maxlevel`, `refine_tol`, `coarsen_tol`,
`buffer`; `adapt_case!` over every set, AMR-3). Decided by measurement
(`CODE.md`, "Refinement"): the matter's regrid Conservative width 3
**bounded**; `A_j` conservative along the edge with point values of
**width 4** across it (AMR-9), `Φ̃` PointValue width 4 (`ops_Φ`); `ξ = 16/L`
on every mesh (`L` the longest side; Erik's decision, 2026-10-15); the
matter's ghost fill in **primitives** (Erik's decisions, 2026-10-15 and
2026-10-23): the scratch set `W = (ρ★, s, Yₑ, uⁱ, Eⁱ)`, copied at
same-level faces and prolongated **parent-centred** at refinement boundaries
(`default_ops_W`, Conservative width 3 bounded, Athena++'s choice), recovery
on owned cells only, `P` in ghosts by forward EOS (`P_from_W`, also for
owned cells), no ghost ever recovered, the conserved ghosts never written by
the scheme (only `regrid!`'s own fill); `CODE.md`, "Refinement", records the
three working fills and their branches. Every fill applies the complete outer
boundary condition in one TreeAMR fill (Erik, 2026-10-15). The magnetic flux
across levels holds to rounding as a rate and as a state (but at an outer
face: AMR-11). The magnetized atmosphere tube was chaotic at rounding level
(fallbacks, injected energy, `W`) until 2026-10-21's atmosphere at rest;
under EntropyEOS 2.0 its perturbed runs are clean but not all the same run. `bin/interface_order.jl`,
`bin/gauge_damping.jl`, `bin/field_loop_refined.jl`, `bin/ghost_fill.jl`
made `CODE.md`'s step-8 tables. Tests: `test/refinement_tests.jl` and
`test/refinement_cases_tests.jl` (use `multid_tests.jl`'s helpers). Step 9:
`src/cases/kerr_schild.jl` — `MichelFlow` (Michel's accretion flow,
`michel_density`, `michel_state`), `wald_case` (`Vacuum`, Wald at `a = 0`)
and `bondi_case` (`IdealConductor`, Michel's flow in a radial monopole) on a
Schwarzschild hole in Kerr-Schild coordinates (`Analytic(KerrSchild(1))`),
on the octant `[0, 4M]³` with the hole at its corner and Dirichlet faces
(the monopole's Dirac string leaves through the corner); `src/interior.jl` —
`HoleInterior(r_in; stage, step)`, an `MHDCase`'s `interior`, pasted by the
reset first, as stage and step limiter, position-only (proposed; `r_in = M`),
and the diagnostics `outside_l1`, `field_l1`, `charge_norms`. Every kernel
already read the metric at its own point, so the scheme did not change;
pointwise evaluation is 44% of a step on a 16³ blast (`bin/metric_cost.jl`;
kept: decided by Erik 2026-11-03, no cache; caching is step 10's). `bin/curved_stationarity.jl` is the `t = 50M` study (CODE.md,
"Validation ladder" 3). Tests: `test/curved_tests.jl` (~2 min with
compilation; the right-hand side's truncation error, the drift to `t = M/2`
at `h = M/4`, `M/8`, the interior, the curved-space bit identity). Round 2
(2026-11-02): the `t = 50M` study fails the predicted order, and
`bin/curved_localise.jl` (time series, radial layers, face distances, the
resets by layer and by EntropyEOS level, diagnostic hooks evaluated in its
own process) located it: a mode that grows through the potentials' relative
Dirichlet hook (AMR-7) copying the gauge drift of `A` across the face; with
the potentials absolute in the ghosts both cases are stationary and second
order (CODE.md, "Validation ladder" 3, "Where the error lives"; the fix is
Erik's). Resets deep inside the hole, `r < (r_in + 2M)/2`, are not counted
(Erik, 2026-11-02; CODE.md, "The interior"). Round 3 (2026-11-03, option 1,
decided by Erik): `MHDCase(…; static_potential = true)` makes the potentials'
Dirichlet hooks absolute (`DirichletBoundary(…; absolute = true)`), valid only
for potentials that do not depend on `t`; the Kerr-Schild cases declare it,
every other case keeps the relative hook (AMR-7) and its output bit for bit;
the boundary testset in `test/curved_tests.jl` holds a gauge offset out of
`𝔅`. The design rule is in CODE.md, "Boundaries". `bin/curved_localise.jl`'s
`relhook=true` is the negative control, `interior=false` the no-interior run.
Round 4 (2026-11-03): `spinning_wald_case` (Wald's uncharged field on a Kerr
hole, `a = M/2`, `0.9M`, a passive floor fluid, `Φ̃ = aB₀`, so
`gauge_damping = 0`, `MHDCase`'s new field), `kerr_schild_r`; the octant
shifted by `L/256` in `z` off the `r = 0` disk; `bin/curved_localise.jl kwald`
(`a`, `rho_fluid`, `ballks`, `watch`). The `50M` study (jobs 588687–588689,
588704–588706): second order at `10M`, but `M/8 → M/16` at `50M` fails —
`a = M/2` 0.79–1.55 (a late growth in the outer region), `a = 0.9M` 1.12–1.93
(a step near the ergoregion); the Euclidean paste at `a = 0.9M` misses `r₋`
and grows, the ball in Kerr-Schild `r` does not (CODE.md, "The spinning Wald
case"; both for Erik).

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
step-8 files take 73 s and 44 s alone, compilation included. After the
reviewer's round (one `ξ` on every mesh, the complete boundary condition
after every fill, AMR-10 requested): **25373 tests**, 6m45s at one thread,
7m10s at four, 7m20s with `--check-bounds=yes`; identical output at one and
four threads, last digits only with bounds checks. At the end of step 8's
finish on TreeAMR 0.2.2 (2026-10-16, at a load of 4–6): **25405 tests**, a
`Test Summary` time of 6m34s at one thread, 6m59s at four, 7m14s with
`--check-bounds=yes` and 6m37s for the clean-checkout check (wall 6m43s,
7m07s, 7m41s, 6m56s); identical output at one and four threads and from
the clean checkout, last digits only with bounds checks (60 of 256 lines;
the tracked tube's drift 0.015 → 0.014 ulp). Under the 10-minute budget,
so no split. The same afternoon 1.12 took 5m22s (6m00s with bounds
checks, its `Pkg.test()` default) and 1.11 6m05s with bounds checks — see
`CODE.md`, "Julia version". After the parent-centred fill became the only
one (2026-10-24, on Symmetry's `amddebugq`, TreeAMR 0.2.6 as resolved
there): **25950 tests** (d456a44 had 25876: +64 from the merged recovery
test, +10 the new ghost testset), a `Test Summary` time of 10m18s at one
thread, 8m38s at four and 10m17s for the clean-checkout check on cn112 (EPYC
8534P), and 14m12s with `--check-bounds=yes` on cn084 (EPYC 7532); every
printed line identical at one and four threads, from the clean checkout and
to d456a44's on the same node (but one renamed `@info` label); another CPU
moves last digits. With bounds checks four allocation assertions fail, as
they do at d456a44 (since 2aae714): EntropyEOS 1.1's `con2prim_safe`
allocates 544 B on the floored input there under `--check-bounds=yes`.
After EntropyEOS 2.0 (2026-10-24, `claude/recovery-pipeline-2`, Symmetry
`amddebugq`): **26196 tests** (+200 the Jacobian stencil's check, +40 the
rewritten ρ_min-face testset, +9 the 2.0 names, −3 the units check), a
`Test Summary` time of 11m50s at one thread and 9m11s at four on cn112
(EPYC 8534P), the stage-1 baseline (d94bcf5, 25950 tests) 10m17s in the
same job there; 11m50s for the clean-checkout check on cn112 and 14m14s on
cn080 (EPYC 7532); 15m50s with `--check-bounds=yes` on cn079 (EPYC 7532).
One and four threads identical, and the clean checkout identical to the
one-thread run on the same node (another CPU moves last digits). With
bounds checks the four allocation assertions of the ρ_min-face testset
fail, 272 B and 144 B, in EntropyEOS's nested fallback (EOS-10 in
`PLAN.md`; waiting on it, decided by Erik 2026-10-24); every other run passes whole. After the step
cache (2026-10-24, `claude/recovery-pipeline-3` at 1462923, Symmetry, AMD
EPYC 7532 throughout): **26268 tests** (+72, `test/step_cache_tests.jl`), a
`Test Summary` time of 10m32s at one thread on cn079 against the stage-2
baseline's 14m00s in the same job (a9cd482, 26196 tests, every test
passing), 9m42s at four threads and 10m41s for the clean-checkout check on
cn080, 12m26s with `--check-bounds=yes` on cn084. One and four threads
identical, the clean checkout identical to the one-thread run. **Four
failures in every run**: the uniform flow's slow cases leave `≤ 10⁻¹¹`
(`1.1–8.4·10⁻¹¹`, warm seeds answering to EntropyEOS's tolerance); with
bounds checks also EOS-10's four. *(Both resolved 2026-10-30: the uniform
flow is asserted to the recovery's tolerance, `≤ 10³·tol`, with a 1000-step
run besides, decided by Erik; EOS-10 landed in EntropyEOS 2.2.1.)* On
EntropyEOS 2.2.0 with `inside_window` (2026-10-26, `claude/entropyeos-2.2`,
Symmetry, AMD EPYC 7532 throughout): **26296 tests** (+18 the reconstruction's
slack testset, +8 the step-cache entry read back, +2 names), a `Test Summary`
time of 10m46s at one thread on cn079 against stage 3's 10m44s in the same
job (4929c94, 26268 tests), 9m50s at four threads and 10m49s for the
clean-checkout check on cn080, 12m27s with `--check-bounds=yes` on cn081.
One and four threads identical, the clean checkout identical to the
one-thread run; against stage 3 only ARS(4,4,3) near the vacuum moves (three
lines; `CODE.md`, "EntropyEOS 2.1 and 2.2"). The same **four failures** (the
uniform flow, stage 3's numbers to every digit), and with bounds checks
EOS-10's four. On EntropyEOS 2.2.1 with the uniform flow asserted to the
recovery's tolerance (2026-10-30, `claude/entropyeos-2.2` at 5e516c6,
Symmetry, AMD EPYC 7532 throughout): **26304 tests** (+8, the 1000-step
uniform flows), **every run passing whole**: a `Test Summary` time of
10m49s at one thread on cn085 against 16c48ed's 10m42s in the same job,
9m58s at four threads and 10m48s for the clean-checkout check on cn079,
12m58s with `--check-bounds=yes` on cn080. One and four threads identical,
the clean checkout identical to the one-thread run, and every printed line
identical to 16c48ed's (EntropyEOS 2.2.0) but the four new ones. After
the two-phase recovery (2026-10-30, `claude/recovery-pipeline-4` at 0334aed,
Symmetry, AMD EPYC 7532 throughout): **26353 tests** (+49,
`test/two_phase_tests.jl`), **every run passing whole**: a `Test Summary`
time of 11m17s at one thread on cn082 against fea242a's 11m05s in the same
job, 10m21s at four threads there, 11m19s for the clean-checkout check and
13m16s with `--check-bounds=yes` on cn079. One and four threads identical,
the clean checkout identical to the one-thread run, every printed line
identical to fea242a's but the nine new ones, and bounds checks moving last
digits only (68 lines). With fused the default (2026-11-02, 03566f4,
Symmetry): **26353 tests**, every run passing whole: 11m04s at one thread on
cn079 (EPYC 7532) against fea242a's 10m54s in the same job, 10m06s at four
threads and 11m13s for the clean-checkout check on cn080 (EPYC 7532), 9m46s
with `--check-bounds=yes` on cn093 (EPYC 7543). One and four threads
identical, the clean checkout identical to the one-thread run, every printed
line of the scheme identical to fea242a's but the nine new ones; six lines
of the pointwise metric tests moved at 10⁻¹⁵ because the registry resolved
ForwardDiff 1.4.7 (and TreeAMR 0.2.7) for the new tree, against 1.4.6 (and
0.2.6) for fea242a's of 2026-10-30. Step 9 (2026-11-02,
`claude/step-9-curved-backgrounds` at af0c482, Symmetry, AMD EPYC 7532):
**26425 tests** (+72, `test/curved_tests.jl`), one failure (the Michel
`uʳ` bound on the EPYC, fixed in cb41add), a `Test Summary` time of 14m35s
at one thread on cn082 (the curved file about 3.5 min of it, compilation
included); the same one failure at four threads (cn080) and with bounds
checks (cn081). At cb41add (jobs 588117–588119): **26425 tests, every run
passing whole** — 14m21s at one thread on cn081 (EPYC 7532) against
9e686a2's 11m08s (26353 tests) in the same job, 11m38s at four threads and
14m28s for the clean-checkout check on cn092 (EPYC 7532), 12m38s with
`--check-bounds=yes` on cn093 (EPYC 7543). One and four threads identical,
the clean checkout identical to the one-thread run, every line of the
baseline present in the one-thread run (11 new lines); bounds checks, on the
other CPU, move 70 lines. The suite stays whole at about 14 minutes, the
curved file in it (decided by Erik 2026-11-03: no split). With option 1,
the static potentials' absolute hook (2026-11-03, 5bea724, Symmetry, AMD
EPYC 7532 throughout): **26428 tests** (+3, the boundary testset), every run
passing whole — 14m20s at one thread on cn079 against cb41add's 14m16s in
the same job, 11m51s at four threads and 14m35s for the clean-checkout check
on cn082, 16m55s with `--check-bounds=yes` on cn080. One and four threads
identical, the clean checkout identical to the one-thread run, every line of
cb41add's present but the six curved lines the absolute hook moves (seven new
ones in their place); bounds checks move 70 lines. With the spinning Wald
case (round 4, 2026-11-03, 0858776, Symmetry): **26442 tests** (+14), every
run passing whole — 14m48s at one thread on cn082 (EPYC 7532) against
5bea724's 14m38s in the same job, 9m00s at four threads and 11m01s for the
clean-checkout check on cn107 (EPYC 7543), 17m03s with `--check-bounds=yes`
on cn083 (EPYC 7532); one and four threads identical, the clean checkout
identical to the one-thread run, every line of 5bea724's present (three
new); bounds checks move 70 lines. At
the end of step 7
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
  longer). Since 2026-10-16 every Tree* package has the 1.13 floor, and
  CI tests only 1.13: Erik's decision, because supporting older versions
  slows development and is not useful at the moment. Do not add `VERSION`
  checks or shims for anything older. See `CODE.md`, "Julia version".
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
