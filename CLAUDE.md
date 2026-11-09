# Working notes for Claude in TreeGRRMHD.jl

**Contents**

- [Read first, and where things go](#read-first-and-where-things-go)
- [What this package is](#what-this-package-is)
- [Commands](#commands)
- [Conventions](#conventions)
- [Repository facts](#repository-facts)

## Read first, and where things go

Read `CODE.md` first — it is the design document and states *why* things
are the way they are. This file is only about mechanics.

The package's documents, and what goes in each:

- **`README.md`**, for users: what the package does, installation, the
  status, and a pointer to the other documents.
- **`CLAUDE.md`**, these notes, for AI agents: what to read first and this
  map; a few lines on what the package is; the commands; conventions and
  sharp edges, each a short rule pointing at the `CODE.md` section that
  explains why; repository facts. No status narrative, file inventory,
  release history or detailed measurements.
- **`CODE.md`**, the current design and the reasons for it, in the present
  tense: purpose and requirements, the method, the code and its structure
  ("Code layout", "Tests", "Sharp edges"), the interfaces and their
  contracts, current measurements that support a current decision, known
  limitations, open questions. No dates or step numbers on decisions and
  no amendment notes; only **(open)** and **(proposed)** mark what is not
  settled.
- **`PLAN.md`**, the plan: the steps, each with what it delivers and how
  one knows it is done, and the upstream requirements still waiting (a
  finished step is one line saying where its results went).
- **`HISTORY.md`**, how the package got here: who decided what and when,
  alternatives tried and rejected and why, measurements of code that no
  longer exists or that later measurements replaced, upstream bugs since
  fixed, the release history, the suite's sizes and timings over time.

Where new content goes: a design fact or its reason goes in `CODE.md` if
someone changing the code today would need it, and in `HISTORY.md`
otherwise; where a reason now lives in `HISTORY.md`, `CODE.md` links to it.
Every fact is in exactly one file; elsewhere, link to it.

**Amending the documents.** When the implementation shows `CODE.md` wrong
or incomplete, fix `CODE.md` to describe the current state, and record in
`HISTORY.md` what changed, when and why — never an "(amended …)" note in
`CODE.md`.

**Tables of contents.** `README.md`, `CLAUDE.md`, `CODE.md`, `PLAN.md` and
`HISTORY.md` each have one near the top, as links: `##` and `###` headings
for `CODE.md`, `PLAN.md` and `HISTORY.md`, `##` only for `README.md` and
`CLAUDE.md`. Keep them current whenever a heading is added, renamed, moved
or removed.

## What this package is

General-relativistic **resistive** magnetohydrodynamics on
[TreeAMR.jl](https://github.com/eschnett/TreeAMR.jl): the Valencia matter
equations plus the full Maxwell equations with an Ohm's law, on a
cell-centered finite-volume scheme with the vector potential on edges. The
equation of state and the primitive recovery come from EntropyEOS; the
spacetime it couples to is TreeGeneralizedHarmonic's (TreeGH). See
`CODE.md`, "Goals" and "Scope and non-goals"; the current status is
`CODE.md`'s status paragraph.

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

At 1.13 `Pkg.test()` runs with bounds checks at `auto`, so the third run
is not the first (`CODE.md`, "Julia version"). Each full run takes roughly
12–21 minutes on Symmetry's AMD nodes (at four threads the shortest, with
bounds checks the longest; `HISTORY.md`, "The test suite", has the
measurements). The log names every upstream ID still probed as waiting;
grep it for `waiting:` (a landed item guaranteed by `[compat]` has no
probe, so `landed:` should not appear).

The clean-checkout check: a tree with no `Manifest.toml` resolves TreeAMR
and SpacetimeMetrics from the General registry and EntropyEOS and
IMEXRungeKutta from their remotes through `[sources]` — never from a
sibling checkout. This is what CI does; run it under the floor too:

```bash
d=$(mktemp -d) && git archive HEAD | tar -x -C "$d" && \
  julia +1.13 --project="$d" -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The tiers GitHub CI runs (`test/tiers.jl`; `CODE.md`, "Julia version" and
"Tests"): `quick` at one thread leaves out the `@full_tier` testsets, and
`threaded` at four leaves out the `@serial_tier` ones too. `full` is the
default, so every command above is unchanged:

```bash
TREEGRRMHD_TEST_TIER=quick julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=yes"])'
```

```bash
TREEGRRMHD_TEST_TIER=threaded julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4", "--check-bounds=yes"])'
```

The full suite as a commit status: `ci/full-suite.sh` runs the
clean-checkout check on a commit at four threads and, with `--post`, sets
the commit's status `full-suite/<host>` to pending and then to the result.
Posting is public: ask Erik each time, as for a push; the commit must be
on GitHub. `ci/full-suite-symmetry.sh submit` does the same in an
`amddebugq` job (one fixed tree under
`/home/eschnetter/claude/full-suite/TreeGRRMHD`, log and result under
`/mnt/beegfs/eschnetter/claude/full-suite/TreeGRRMHD/<sha>/`), and
`collect` posts its result afterwards, since the cluster has no `gh`:

```bash
ci/full-suite.sh --post
```

```bash
ci/full-suite-symmetry.sh submit --post
```

```bash
ci/full-suite-symmetry.sh collect --post
```

A `Manifest.toml` resolved by one Julia version breaks loading under
another (1.13 refused a 1.11 manifest's `MbedTLS_jll`), so delete it, or
use a clean tree, when switching versions.

## Conventions

The design rules, each explained in the `CODE.md` section named:

- **No mesh machinery.** Trees, ghosts, inter-level operators, reductions
  and point interpolation belong upstream in TreeAMR. Interpolation
  *within a block* between centerings is application code and lives here,
  in both directions (`CODE.md`, "Scope and non-goals", "Metric at cells,
  faces and edges", "Matter to the vertices").
- **Neither package drives, and neither depends on the other.** A coupled
  run is a script that owns the loop and calls `TreeGH.prepare!`,
  `TreeGRRMHD.prepare!` and the rest; data passes through pointwise
  adapters that the producer supplies, called at TreeGH's owned vertices.
  Do not add a driver framework or a TreeGH dependency (`CODE.md`, "The
  coupling contract").
- **Upstream changes are Erik's.** When this package needs something from
  TreeAMR, TreeGeneralizedHarmonic or EntropyEOS, it goes in as a precise
  request — the reason in `CODE.md`'s "Upstream prerequisites", the exact
  requirement in `PLAN.md`'s "Upstream requirements"; do not edit the
  sibling checkouts, and do not work around a missing upstream feature
  silently — say which item a step is waiting for.
- **One con2prim.** EntropyEOS's `con2prim_safe` is the only primitive
  recovery, called in hydrodynamic mode (EM energy and momentum
  subtracted, `B² = 0`) for every σ; its ideal-MHD mode is only the seed of
  the implicit stage solve. Do not add a second inversion (`CODE.md`,
  "Primitive recovery", "The implicit stage solve").
- **∇·B is structural.** `B` is always the discrete curl of the
  edge-centered `A` (plus the constant background `𝔅̄`). Never evolve,
  prolongate or store `B` as independent data (`CODE.md`, "Where the
  fields live").
- **`μ_max` has no default**: every case and driver states its own, and
  nothing wraps a model implicitly (`CODE.md`, "Ohm's law and the
  conductivity").
- The shock-tube references in `test/data/` are rerun
  (`bin/shocktube_reference.jl`) only if the scheme changes, and the
  commit says so (`CODE.md`, "Code layout").

The sharp edges, each explained in `CODE.md`, "Sharp edges":

- Five centerings, five offsets: points `G+1 … G+N`, `coordinates` takes
  stored indices, one `Val(G)` per set a kernel reads.
- `B` is never state: not in the state vector, `regrid!`'s transfer or a
  checkpoint.
- EntropyEOS's inputs are undensitized; its warm starts are positional;
  take `|v|` from the returned rapidity; call the two levels only through
  `two_levels`; test flag words with `& … != 0`.
- A new recovery site computes `B`, `E` and the undensitized variables as
  `· inv(√γ)`, and a test that steps an integrator itself wraps the
  stepping in `with_step_cache` to use the step cache.
- Convert SpacetimeMetrics' derivative index order once, in
  `src/metric.jl`; TreeGH stores `h_ab` as an offset.
- Every fill applies the complete outer boundary condition; never fill a
  potential without its hook.
- A regridded run's initial totals are `evolve!`'s `totals0`.
- The stage solver returns the `U` it solved for, before any limiter,
  writing only `E`.
- The atmosphere reset is both the stage and the step limiter.
- The explicit and implicit abscissae differ; write `t`-dependent boundary
  data so that `t` drops out.
- The right-hand side never mutates `u`; an owned low wall gets zero
  tendency.
- Masks branch, never multiply.
- An adapter is a snapshot of its producer's prepared state.

Style, matching TreeAMR's, TreeHydro's and TreeGeneralizedHarmonic's,
since the packages are read together:

- 4-space indent, wrap at about 80–90 columns; `return` on the last line of
  any non-trivial function.
- KernelAbstractions for every kernel; callbacks and parameter structs are
  `isbits`; `Val`s for `D`, each `G`, the direction, the limiter and the
  solver; `@inbounds` *inside* `ntuple` closures, not outside.
- No float literals in `T`-generic code: `T(1//2)`, `oftype(x, 2)`.
- `T` as a leading positional argument defaulting to `Float64`, `backend`
  a keyword defaulting to `CPU()`; keyword-heavy drivers with no default for
  anything the caller must think about (`CODE.md`, "Precision and
  devices").
- StaticArrays for 3- and 4-tensors, as TreeGH does; `NTuple`s for state
  vectors, as TreeHydro does.
- Unicode in mathematical contexts; `ArgumentError`s say *why*.
- Docstrings are prose-first and point at `CODE.md`.
- Operator orders in TreeAMR's convention: the order is the stencil width
  (PointValue even, Conservative odd), exact to degree width − 1. Write
  "width w" in prose to avoid ambiguity.
- **Julia 1.13 is the floor** (`julia = "1.13"` in `[compat]`); the floor
  goes up when the older version is more than about 20% slower, and
  changing it is Erik's call. Do not add `VERSION` checks or shims for
  anything older (`CODE.md`, "Julia version").

Tests (`CODE.md`, "Tests"):

- **Testset names are claims**, each opening with a comment naming the
  failure mode it guards.
- **A new testset goes in a tier.** One that runs a 3D curved study, or
  anything else long, at its finest resolution is `@full_tier`; a 1D study
  (many steps on a few hundred cells, 4–8 times slower at four threads on
  GitHub's runners) is `@serial_tier`, unless it is held bit for bit to a
  2D or 3D run. A run that several testsets read is made by
  `computed_once` inside the first of them, never by a top-level `const`,
  which would run it at include whatever the tier left out. Helpers shared
  between files live in their own files (`recovery_states.jl`,
  `wave_study.jl`, `multid_runs.jl`, `refinement_meshes.jl`), included by
  `runtests.jl` before their first user.
- **Run the full suite locally before opening a pull request.** A pull
  request's CI runs only the two Linux cells, uninstrumented; macOS and
  coverage run on `main`, weekly and on dispatch, and Markdown-only changes
  run nothing (`CODE.md`, "Julia version"). GitHub runs the `quick` tier at
  one thread and the `threaded` tier at four, and the `full` tier weekly
  and on a dispatch with `tier = full`. So the local run is the macOS run
  the pull request does not get.

## Repository facts

- No remote yet. Work on a branch, and do not commit, push or merge
  without being asked.

  **Inconsistent:** "No remote yet" contradicts `README.md`'s CI and
  codecov badges for `github.com/eschnett/TreeGRRMHD.jl`, the merged pull
  requests #6 and #7 in the git log, and this file's own "Commands" ("the
  commit must be on GitHub") and "Conventions" (a pull request's CI).
- Branches: a plan step works on `claude/step-N-<slug>` (`PLAN.md`,
  "Ground rules, every step").
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
  edited only in steps 4a–4c; its `CODE.md` is the IMEX design).
  `~/src/jl/Con2Prim` is an abandoned prototype; do not depend on it.
