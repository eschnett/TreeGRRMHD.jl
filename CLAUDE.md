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

**Design and plan only.** `CODE.md` records the reviewed design
(2026-09-23); `PLAN.md` is the implementation plan (drafted 2026-09-23):
steps 0–15 on `claude/step-N-slug` branches, run one agent per step, and
the exact upstream requirements under IDs `AMR-n`, `GH-n`, `EOS-n`, each
naming the step it blocks. `src/TreeGRRMHD.jl` is still the Pkg template.
**Step 0 is next.** Steps 4a–4c build the IMEX package in its own
repository (`~/src/jl/IMEXRungeKutta`, name proposed).

## Commands

Nothing to run yet beyond the template. Once there is a test suite, follow
TreeHydro's discipline (`TreeHydro/CLAUDE.md`, "Commands"): the whole suite
at one thread, at four threads, and with `--check-bounds=yes`, e.g.

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

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
- **Julia 1.10 wherever possible** (`julia = "1.10"` in `[compat]`). Only
  TreeGeneralizedHarmonic pins 1.11 today; see `CODE.md`, "Julia version".
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
  (analytic backgrounds). `~/src/jl/Con2Prim` is an abandoned prototype;
  do not depend on it.
