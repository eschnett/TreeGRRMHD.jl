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
physics) and 3 (the implicit stage solve) are done; step 5 is next, and
is unblocked (2026-10-14).** `CODE.md` records the reviewed design
(2026-09-23); `PLAN.md` is the implementation plan: steps 0–15 on
`claude/step-N-slug` branches, run one agent per step, and the exact
upstream requirements under IDs `AMR-n`, `GH-n`, `EOS-n`, each naming the
step it blocks, with a state column. What exists: the module shell,
`src/precision.jl` and `src/device.jl` (ported from TreeHydro),
`test/prerequisite_tests.jl` (the names the scheme calls, and every
upstream ID logged as landed or waiting), step 1's pointwise layer —
`src/units.jl` (geometric units, `GeometricEOS`), `src/metric3.jl`,
`src/variables.jl` (the slot layouts of `U` and `P`), `src/prim2con.jl` and
`src/con2prim.jl` (`con2prim_hydro`, `con2prim_ideal`) — and step 2's —
`src/metric.jl` (the `Flat`/`Analytic` providers and `MetricGradient`, the
one place SpacetimeMetrics' derivative index order is converted),
`src/stress_energy.jl`, `src/ohm.jl`, `src/flux.jl`, `src/riemann.jl`,
`src/emf.jl` and `src/sources.jl` — and step 3's `src/stage_solve.jl`
(`solve_ohm_stage`, Newton on the fluid's share of the energy and momentum,
`OhmStageResult`), with `MagnetizationCutoff` in `src/ohm.jl` tapering σ to
zero where `B²/D` exceeds 10 (decided by Erik 2026-10-14; the bracketed
alternative is on `claude/step-3-bracketed-stage-solve`, not in the code). `test/pointwise_tests.jl` and `test/stage_solve_tests.jl`
use `recovery_tests.jl`'s state generators and are included after it. Steps
4a–4c are done in `~/src/jl/IMEXRungeKutta` (1.4.0), whose `CODE.md` is the
IMEX design. EOS-1 and EOS-2 have landed (EntropyEOS 1.0.0, still
unregistered); TreeAMR 0.2.1 has AMR-1, -2, -3, -5, and AMR-4 and AMR-6 in
another form (`bounded = true`, outflow faces); SpacetimeMetrics 1.7.0 has
SM-1, whose workaround is gone. Step 5 waits for nothing.

## Commands

TreeHydro's discipline (`TreeHydro/CLAUDE.md`, "Commands"): the whole
suite at one thread, at four threads, and with `--check-bounds=yes`, at
the floor version and at the current release. `Pkg.test` does not inherit
`-t`, so the thread count has to be passed explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

```bash
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
```

The clean-checkout check: a tree with no `Manifest.toml` resolves TreeAMR
and SpacetimeMetrics from the General registry and EntropyEOS and
IMEXRungeKutta from their remotes through `[sources]` — never from a
sibling checkout. This is what CI does; run it under the floor too:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia +1.11 --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

A `Manifest.toml` resolved by one Julia version breaks loading under
another (1.13 refuses a 1.11 manifest's `MbedTLS_jll`), so delete it, or
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
their GitHub URLs, as it must. The log names every upstream ID as landed
or waiting; grep it for `landed:` and `waiting:`. At the end of step 2
(2026-10-13): **20513 tests**, a `Test Summary` time of 37–39 s at 1.11
and 28 s at 1.13 at one thread, four and `--check-bounds=yes`;
`test/pointwise_tests.jl` takes about 12 s of it.

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
- **Julia 1.11 is the floor** (`julia = "1.11"` in `[compat]`), shared by
  all the Tree* packages since 2026-09-25 so that unregistered dependencies
  can be located with `[sources]`; see `CODE.md`, "Julia version".
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
