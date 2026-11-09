# TreeGRRMHD implementation plan

For the sessions that implement TreeGRRMHD, one step at a time. The **design** lives in
`CODE.md`. Read it first, in full: it is authoritative. This file is only the work breakdown:
- what each step changes, and what it must not change;
- what each step must measure and record;
- and, because this package stands on siblings that are still growing, exactly what it needs
  from them.

`CLAUDE.md` has the mechanics. Delete this file when the last rung is marked *(Done.)* in
`CODE.md`.

The current status is `CODE.md`'s status paragraph; how the steps went, and when each upstream
item landed, is in `HISTORY.md`, "The steps of the plan" and "Upstream prerequisites". The
"state" column of the table in [Upstream requirements](#upstream-requirements) has each item.

The steps follow `CODE.md`'s validation ladder (rungs 1–7):
- It is preceded by the pointwise physics and the IMEX package.
- It is followed by a device step and a review pass.
- Every step ends in a green suite and a `CODE.md` update.
- Each step is a brief that a single agent with a fresh context can carry.
- The order is the dependency order. [Upstream requirements](#upstream-requirements) says which
  step waits for what.

**Contents**

- [Ground rules, every step](#ground-rules-every-step)
- [Running a step as an agent](#running-a-step-as-an-agent)
- [Sharp edges to know before starting](#sharp-edges-to-know-before-starting)
- [Upstream requirements](#upstream-requirements)
  - [AMR-1 — A state vector over several field sets (req, step 5)](#amr-1--a-state-vector-over-several-field-sets-req-step-5)
  - [AMR-2 — Per-set boundary hooks in `regrid!` (wanted, step 6)](#amr-2--per-set-boundary-hooks-in-regrid-wanted-step-6)
  - [AMR-3 — `adapt_to_initial_data!` over several sets (wanted, step 8)](#amr-3--adapt_to_initial_data-over-several-sets-wanted-step-8)
  - [AMR-4 — Limited conservative prolongation (strongly wanted, step 10)](#amr-4--limited-conservative-prolongation-strongly-wanted-step-10)
  - [AMR-5 — `workgroupsize` on `map_blocks!` (wanted, step 14)](#amr-5--workgroupsize-on-map_blocks-wanted-step-14)
  - [AMR-6 — A device-capable boundary hook that reads the interior (later)](#amr-6--a-device-capable-boundary-hook-that-reads-the-interior-later)
  - [AMR-7 — A device-capable boundary hook relative to the nearest owned point (later, step 14; workaround)](#amr-7--a-device-capable-boundary-hook-relative-to-the-nearest-owned-point-later-step-14-workaround)
  - [AMR-8 — Symmetric walls: the owned low wall filled like the high one (wanted; workaround)](#amr-8--symmetric-walls-the-owned-low-wall-filled-like-the-high-one-wanted-workaround)
  - [AMR-9 — A flux-conservative transfer of edge sets (wanted)](#amr-9--a-flux-conservative-transfer-of-edge-sets-wanted)
  - [AMR-10 — Bounded PointValue interpolation (req: step 8's ghost-fill re-measurement; steps 10, 12)](#amr-10--bounded-pointvalue-interpolation-req-step-8s-ghost-fill-re-measurement-steps-10-12)
  - [AMR-11 — The relative hook consistent across levels at an outer face (wanted)](#amr-11--the-relative-hook-consistent-across-levels-at-an-outer-face-wanted)
  - [EOS-0 — Register EntropyEOS (req, step 0; workaround)](#eos-0--register-entropyeos-req-step-0-workaround)
  - [EOS-1 — An abstract EOS interface (req, step 5)](#eos-1--an-abstract-eos-interface-req-step-5)
  - [EOS-2 — Analytic Γ-law and hybrid EOSes (req, step 5)](#eos-2--analytic-γ-law-and-hybrid-eoses-req-step-5)
  - [EOS-3 — Public fixed-entropy inversion and entropy from energy (wanted, step 12)](#eos-3--public-fixed-entropy-inversion-and-entropy-from-energy-wanted-step-12)
  - [EOS-4 — Cold β-equilibrium slice (later, step 13)](#eos-4--cold-β-equilibrium-slice-later-step-13)
  - [EOS-5 — Withdrawn (2026-10-17): notes on two repairs at the edge of the valid range](#eos-5--withdrawn-2026-10-17-notes-on-two-repairs-at-the-edge-of-the-valid-range)
  - [EOS-6 — A density raised to the atmosphere's gets the atmosphere's velocity (superseded)](#eos-6--a-density-raised-to-the-atmospheres-gets-the-atmospheres-velocity-superseded)
  - [EOS-7 — The atmosphere's entropy in the atmosphere band (not pursued)](#eos-7--the-atmospheres-entropy-in-the-atmosphere-band-not-pursued)
  - [EOS-8 — Report the density floor (req; landed in EntropyEOS 1.1.0, unregistered)](#eos-8--report-the-density-floor-req-landed-in-entropyeos-110-unregistered)
  - [EOS-9 — Step 4's endpoint branches: edge-consistent, guarded, reporting facts (landed in EntropyEOS 2.0.0)](#eos-9--step-4s-endpoint-branches-edge-consistent-guarded-reporting-facts-landed-in-entropyeos-200)
  - [EOS-10 — The nested fallback allocation-free under `--check-bounds=yes` (landed)](#eos-10--the-nested-fallback-allocation-free-under---check-boundsyes-landed)
  - [EOS-11 — The validity test of `con2prim_safe`, public (req; landed in EntropyEOS 2.1.0)](#eos-11--the-validity-test-of-con2prim_safe-public-req-landed-in-entropyeos-210)
  - [GH-0 — Land step 8 on `main` (req, step 11)](#gh-0--land-step-8-on-main-req-step-11)
  - [GH-1 — The metric at TreeGH's vertices (req, step 11)](#gh-1--the-metric-at-treeghs-vertices-req-step-11)
  - [GH-2 — Registration, registry-satisfiable compat, the Julia floor (req, step 11; workaround)](#gh-2--registration-registry-satisfiable-compat-the-julia-floor-req-step-11-workaround)
  - [GH-3 — Matter input (req, step 11)](#gh-3--matter-input-req-step-11)
  - [GH-4 — RHS and limiters callable from an external integrator (req, step 11)](#gh-4--rhs-and-limiters-callable-from-an-external-integrator-req-step-11)
  - [GH-5 — Ghost width `≥ q/2 + 1` (withdrawn 2026-11-06; was wanted, step 10)](#gh-5--ghost-width--q2--1-withdrawn-2026-11-06-was-wanted-step-10)
  - [GH-6 — Later](#gh-6--later)
  - [SM-1 — Allocation-free extrinsic curvature (wanted, step 9; workaround)](#sm-1--allocation-free-extrinsic-curvature-wanted-step-9-workaround)
- [Step 0 — Scaffolding](#step-0--scaffolding)
- [Step 1 — Units, variables and the 3+1 primitive recovery](#step-1--units-variables-and-the-31-primitive-recovery)
- [Step 2 — Pointwise physics](#step-2--pointwise-physics)
- [Step 3 — The implicit stage solve](#step-3--the-implicit-stage-solve)
- [Steps 4a–4c — The IMEX package (in its own repository)](#steps-4a4c--the-imex-package-in-its-own-repository)
- [Step 5 — The staggered scheme in 1D: smooth problems (rung 1a)](#step-5--the-staggered-scheme-in-1d-smooth-problems-rung-1a)
- [Step 6 — Shock tubes and the stiff limit in 1D (rung 1b)](#step-6--shock-tubes-and-the-stiff-limit-in-1d-rung-1b)
- [Step 7 — 2D and 3D, and higher-order reconstruction (rung 2a)](#step-7--2d-and-3d-and-higher-order-reconstruction-rung-2a)
- [Step 8 — Coarse-fine faces and regridding (rung 2b)](#step-8--coarse-fine-faces-and-regridding-rung-2b)
- [Step 9 — Analytic curved backgrounds (rung 3)](#step-9--analytic-curved-backgrounds-rung-3)
- [Step 10 — The TOV star in the Cowling approximation (rung 4)](#step-10--the-tov-star-in-the-cowling-approximation-rung-4)
- [Step 11 — The TOV star in a dynamical spacetime (rung 5)](#step-11--the-tov-star-in-a-dynamical-spacetime-rung-5)
- [Step 12 — The magnetized TOV star (rung 6)](#step-12--the-magnetized-tov-star-rung-6)
- [Step 13 — Tabulated EOS and Yₑ (rung 7)](#step-13--tabulated-eos-and-yₑ-rung-7)
- [Step 14 — Precision, threads, devices, benchmark](#step-14--precision-threads-devices-benchmark)
- [Step 15 — Review pass](#step-15--review-pass)

## Ground rules, every step

- Read `CLAUDE.md` first. Then read this file's ground rules and the step, `CODE.md`'s "Sharp
  edges", then the `CODE.md` sections the step names.
- **Work on a branch** named `claude/step-N-<slug>`, off `main`. For steps 4a–4c, the branch is in
  the IMEX package's repository.
  - Commit in TreeAMR's style: implement, then measure, then record, with the measured numbers in
    the commit body.
  - **Do not merge into `main` and do not push.** Report the branch and its commits.
- **Never edit a sibling checkout** (TreeAMR, TreeGeneralizedHarmonic, EntropyEOS,
  SpacetimeMetrics, TreeHydro; IMEXRungeKutta outside steps 4a–4c).
  - If a listed upstream item has not landed, do everything that does not depend on it, and
    report the step as blocked on that item's ID.
  - If the step needs something upstream that is not listed, or a listed item turns out to be
    insufficient, write the requirement in the format of
    [Upstream requirements](#upstream-requirements) and report it.
  - A workaround listed here is allowed, and is marked in the code with the item's ID. Any other
    workaround is not.
- **Dependencies resolve from releases, never from sibling checkouts.**
  - TreeAMR and SpacetimeMetrics come from the General registry.
  - Unregistered ones are named by URL in `Project.toml`'s `[sources]`, which Julia 1.11 and
    later read; the floor is 1.13 (CODE.md "Julia version"). That covers EntropyEOS until EOS-0
    and the IMEX package, and will cover TreeGH and its unregistered dependencies until GH-2. CI,
    the clean-checkout check and a developer all resolve them the same way, so CI adds nothing by
    hand.
- **Julia 1.13 floor**. Be generic in `T` and in the backend from the first line:
  - `T` is a leading positional argument (default `Float64`); `backend` is a keyword (default
    `CPU()`).
  - No float literals where `T` is in play; every callback is `isbits`; per-block metadata goes
    through `to_backend`.
  - Physics is validated in `Float64`; `Float32` is a smoke test.
- **Tests are small.**
  - Sizes: 1D up to 400 cells, 2D up to 128², 3D with `N = 8` blocks on a few roots.
  - Time: under about 60 s per test file, and the whole suite under about 10 min at one thread.
- Before and after each step, run the full suite at the default thread count, at
  `julia_args = ["--threads=4"]`, and once with `--check-bounds=yes`, each at the floor (Julia
  1.13) and at the current release (the same until 1.14 is released), and the clean-checkout
  check under the floor (`CLAUDE.md`, "Commands").
- **Spec-first.**
  - When the step shows `CODE.md` wrong or incomplete, fix `CODE.md` to describe the current
    state, and record in `HISTORY.md` what changed, when and why.
  - Replace each **(predicted)** the step measures, and decide each **(open)** the step owns.
  - Never loosen an assertion to get green. Report the failure instead.
- **Testset names are claims.** Each opens with a comment naming the failure mode it guards.
  Rates, drifts and iteration counts are asserted as numbers with tolerances.
- One step at a time; the next starts from a green suite on `main`.

## Running a step as an agent

This is TreeGH's pattern.
- **One implementation agent per step**, in its own worktree, with the brief below. At most one
  runs at a time, because each step depends on the previous one.
- The agent does not ask questions mid-step. Where it would have asked, it decides, marks the
  decision **(proposed)** in `CODE.md`, records in `HISTORY.md` that step N proposed it, and lists
  it in the report.
- **The reviewer:**
  - checks out the branch and runs the suite at one and four threads;
  - reads the `CODE.md` diff;
  - walks the acceptance list against the report;
  - merges only then.

  A failed item goes back to the same agent with the finding, not to a new agent, and the
  reviewer does not fix it themselves.

**The brief** (replace `N`):

    You are implementing step N of PLAN.md in the Julia package at <path>. Read CLAUDE.md,
    then PLAN.md's "Ground rules", "Upstream requirements" (the items step N waits for) and
    step N, then CODE.md's "Sharp edges" and the CODE.md sections step N names. Work on the
    branch claude/step-N-<slug> in this worktree; commit in TreeAMR's style with measured
    numbers in the commit body; do not merge, push, or touch TODO.md or any sibling checkout.
    Do all of step N. If part of it is blocked — an upstream item not landed, a device you do
    not have — finish everything else and say exactly what is blocked, by which ID. Never
    loosen an assertion to get green. When you would ask a question, decide, mark it
    "(proposed)" in CODE.md, record in HISTORY.md that step N proposed it, and list it in the
    report. End with the report PLAN.md specifies.

**The report**, in this order:
1. the branch and its commits, one line each;
2. the acceptance list, each item marked *done*, *measured* (with the number), or *blocked*
   (with the ID or reason);
3. the suite's summary lines at one and at four threads, verbatim;
4. every `CODE.md` change, quoted, with the `HISTORY.md` entry that records it;
5. the decisions it took, each marked (proposed);
6. anything the next step should know, and any new upstream requirement, in the format below.

## Sharp edges to know before starting

They are in `CODE.md`, "Sharp edges".

## Upstream requirements

Each item gives:
- **what**, and the **exact API**;
- **semantics**;
- **upstream acceptance**: the tests the sibling should carry;
- **downstream check**: the testset `test/prerequisite_tests.jl` gains. Until the item lands, that
  testset is skipped with an `@info` naming the ID. It detects the item by `isdefined` /
  `hasmethod`, never by version number.

**Retiring a probe.** Once an item has landed *and* `[compat]` requires a release that contains
it, its testset is deleted from `test/prerequisite_tests.jl` and the item is marked "landed in
<package> <version>" in the table and in its section. Probes remain only for items still
waiting. The names a retired item added join the name lists of step 0's checks (which stay, with
the metric-as-kernel-argument check and the IMEX contract checks); a property of the release this
package relies on and does not yet exercise through its own code moves to
`test/upstream_behaviour_tests.jl` until the step that uses it tests it in place.

Priorities:
- **req** blocks the step named.
- **wanted** has a workaround, listed with the item.
- **later** is not needed before rung 7.

Every item lands in a **registered release** (or a pushed commit where registration is pending),
never only in a local checkout. Additive changes stay in the same minor series (TreeAMR 0.2.x,
EntropyEOS 1.x).

A landed, withdrawn, superseded or not-pursued item keeps its heading below with one line giving
its state; the request as written and its record are in
[HISTORY.md, "Upstream prerequisites"](HISTORY.md#upstream-prerequisites), and what this package
relies on from it today in [`CODE.md`, "Upstream prerequisites"](CODE.md#upstream-prerequisites).

The state column is as of 2026-10-26: for the items still waiting, what
`test/prerequisite_tests.jl` reports; for TreeGH, which the suite cannot load yet, what its
`main` reads. "Landed in" means the compat floor guarantees it and the probe is retired.

**Inconsistent:** the state column is said to be as of 2026-10-26, but its rows record EOS-10's
landing in EntropyEOS 2.2.1 (2026-10-30) and GH-1's as `ADMMetric` (2026-11-08).

| ID | package | priority | needed by step | summary | state |
|---|---|---|---|---|---|
| AMR-1 | TreeAMR | req | 5 | state vector over several field sets | landed in TreeAMR 0.2.1 |
| AMR-2 | TreeAMR | wanted | 6 | per-set boundary hooks in `regrid!` | landed in TreeAMR 0.2.1 |
| AMR-3 | TreeAMR | wanted | 8 | `adapt_to_initial_data!` over several sets | landed in TreeAMR 0.2.1 |
| AMR-4 | TreeAMR | strongly wanted | 10 | limited conservative prolongation | landed in TreeAMR 0.2.1, as `bounded = true` |
| AMR-5 | TreeAMR | wanted | 14 | `workgroupsize` on `map_blocks!` | landed in TreeAMR 0.2.1 |
| AMR-6 | TreeAMR | later | — | device boundary hook that reads the interior (outflow) | landed in TreeAMR 0.2.1, as outflow faces |
| AMR-7 | TreeAMR | later, workaround | 14 | device boundary hook relative to the nearest owned point | landed in TreeAMR 0.2.2, as `Relative` (workaround dropped) |
| AMR-8 | TreeAMR | wanted, workaround | — | symmetric walls: the owned low wall filled like the high one | landed in TreeAMR 0.2.2, as `walls = :symmetric` (workaround dropped) |
| AMR-9 | TreeAMR | wanted | — | a flux-conservative transfer of edge sets | landed in TreeAMR 0.2.2, as `vertex_prolongation` |
| AMR-10 | TreeAMR | req | 8 (re-measurement), 10, 12 | bounded PointValue interpolation | landed in TreeAMR 0.2.2 |
| AMR-11 | TreeAMR | wanted | — | the relative hook consistent across levels where an outer face meets a coarse-fine face | waiting |
| EOS-0 | EntropyEOS | req, workaround | 0 | register in General | waiting (`[sources]`) |
| EOS-1 | EntropyEOS | req | 5 | `AbstractEOS` interface | landed in EntropyEOS 1.0.0 |
| EOS-2 | EntropyEOS | req | 5 | analytic Γ-law and hybrid EOS | landed in EntropyEOS 1.0.0 |
| EOS-3 | EntropyEOS | wanted | 12 | public fixed-entropy inversion; entropy from energy | waiting |
| EOS-4 | EntropyEOS | later | 13 | cold β-equilibrium slice | waiting |
| EOS-5 | EntropyEOS | — | — | was: no policy repair creates rest mass; notes kept for Erik | withdrawn |
| EOS-6 | EntropyEOS | — | — | a density raised to the atmosphere's gets the atmosphere's velocity | superseded by EOS-8 and the wrapper |
| EOS-7 | EntropyEOS | — | — | the atmosphere's entropy in the atmosphere band | not pursued |
| EOS-8 | EntropyEOS | req | — | report the density floor: `FLAG_POL_ρ_FLOORED` | landed in EntropyEOS 1.1.0; in 2.0.0 the `ρ_min` face (bit 11) |
| EOS-9 | EntropyEOS | proposed (experiment) | 12 | step 4's endpoint branches edge-consistent and guarded, the hot one onto the entropy ceiling (`S_CEILED`, density clamped and reported), atmosphere flags without the discarded state's, `check_con_state` sharing the diagnosis, step 1 on `D_max` only | landed in EntropyEOS 2.0.0 (tag `v2.0.0`, unregistered), as the two-level projection; the atmosphere's flags keep level 1's bits |
| EOS-10 | EntropyEOS | wanted | — | `con2prim_finish`'s nested fallback allocation-free under `--check-bounds=yes` | landed in EntropyEOS 2.2.1 |
| EOS-11 | EntropyEOS | req | — | `con2prim_safe`'s validity test of the `s` and `Yₑ` faces, with its slack, public, for the reconstruction | landed in EntropyEOS 2.1.0, as `inside_window`; `[compat]` `"2.2"` |
| GH-0 | TreeGH | req | 11 | land step 8 on `main` | landed (by inspection) |
| GH-1 | TreeGH | req | 11 | `(α, β^i, γ_ij, K_ij)` at TreeGH's owned vertices, a pointwise adapter | landed as `ADMMetric` on TreeGH's `main` (PR 25, 61bcb75; still 0.1.1, unregistered) |
| GH-2 | TreeGH | req, workaround | 11 | registration; registry-satisfiable compat; Julia floor | open, see below |
| GH-3 | TreeGH | req | 11 | matter input, a pointwise `matter` adapter returning `(ρ, S_i, S_ij)` | waiting |
| GH-4 | TreeGH | req | 11 | RHS and limiters callable from a script: `TreeGH.prepare!`, `TreeGH.rhs!`, … | waiting |
| GH-5 | TreeGH | — | — | was: ghost width `≥ q/2 + 1`, wanted, step 10; not needed: the interpolation reads this package's own `G = 4` vertex set | withdrawn |
| GH-6 | TreeGH | later | — | gauge driver, radiative boundary, excision, Ψ₄ | waiting |
| SM-1 | SpacetimeMetrics | wanted, workaround | 9 | allocation-free `ExtrinsicCurvature`, also from one `dmetric` | landed in SpacetimeMetrics 1.7.0 (workaround dropped) |

**Inconsistent:** the EOS-11 row gives `[compat]` `"2.2"`; `CLAUDE.md`'s former "Current state"
(now [HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos), 2026-10-30) said `[compat]` is `"2.2.1"`
and `[sources]` pins the tag `v2.2.1` since 2026-10-30 (EOS-10).

### AMR-1 — A state vector over several field sets (req, step 5)

Landed in TreeAMR 0.2.1. The request and its record:
[HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); what this package uses, and its downstream check in
`test/scheme_tests.jl`: [`CODE.md`, "TreeAMR"](CODE.md#treeamr).

### AMR-2 — Per-set boundary hooks in `regrid!` (wanted, step 6)

Landed in TreeAMR 0.2.1. The request and its record:
[HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); its use: [`CODE.md`, "TreeAMR"](CODE.md#treeamr),
[`CODE.md`, "Boundaries"](CODE.md#boundaries).

### AMR-3 — `adapt_to_initial_data!` over several sets (wanted, step 8)

Landed in TreeAMR 0.2.1. The request and its record:
[HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); its use: [`CODE.md`, "Refinement"](CODE.md#refinement).

### AMR-4 — Limited conservative prolongation (strongly wanted, step 10)

Landed in TreeAMR 0.2.1 in another form, as `bounded = true`. The request and its record:
[HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); its use: [`CODE.md`, "TreeAMR"](CODE.md#treeamr),
[`CODE.md`, "Refinement"](CODE.md#refinement). Its downstream check stands for step 10: the TOV
surface across a coarse-fine face floors no cell at the initial regrid.

### AMR-5 — `workgroupsize` on `map_blocks!` (wanted, step 14)

Landed in TreeAMR 0.2.1. The request: [HISTORY.md, "TreeAMR"](HISTORY.md#treeamr).

### AMR-6 — A device-capable boundary hook that reads the interior (later)

Landed in TreeAMR 0.2.1 in another form, as outflow faces. The request and its record:
[HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); what this package relies on:
[`CODE.md`, "TreeAMR"](CODE.md#treeamr).

### AMR-7 — A device-capable boundary hook relative to the nearest owned point (later, step 14; workaround)

Landed in TreeAMR 0.2.2 as `CellBoundary(Relative(f))`; the workaround is gone. The request, its
amendments and its record: [HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); its use:
[`CODE.md`, "Boundaries"](CODE.md#boundaries), [`CODE.md`, "TreeAMR"](CODE.md#treeamr).

### AMR-8 — Symmetric walls: the owned low wall filled like the high one (wanted; workaround)

Landed in TreeAMR 0.2.2 as `walls = :symmetric`; the workaround is gone. The request and its
record: [HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); its use:
[`CODE.md`, "Boundaries"](CODE.md#boundaries).

### AMR-9 — A flux-conservative transfer of edge sets (wanted)

Landed in TreeAMR 0.2.2 as `vertex_prolongation`. The request and its record:
[HISTORY.md, "TreeAMR"](HISTORY.md#treeamr); its use: [`CODE.md`, "Field sets"](CODE.md#field-sets),
[`CODE.md`, "Conservation at coarse-fine faces"](CODE.md#conservation-at-coarse-fine-faces),
[`CODE.md`, "TreeAMR"](CODE.md#treeamr).

### AMR-10 — Bounded PointValue interpolation (req: step 8's ghost-fill re-measurement; steps 10, 12)

Landed in TreeAMR 0.2.2. The request and its record: [HISTORY.md, "TreeAMR"](HISTORY.md#treeamr);
its use: [`CODE.md`, "Refinement"](CODE.md#refinement), [`CODE.md`, "TreeAMR"](CODE.md#treeamr).

### AMR-11 — The relative hook consistent across levels at an outer face (wanted)

**Why.** With the potentials' conservative transfer (AMR-9) the magnetic flux through a coarse
face on a coarse-fine face is the sum through its fine faces to rounding (1.4e−16 of the largest
flux in a periodic 3D box) — except where the face's perimeter lies on an outer face: there each
level's relative hook writes its own wall and ghost edges, `f(x_g) + (u(x_r) − f(x_r))` with
`x_r` that level's nearest evolved point, so a coarse wall edge is not the mean of the fine ones
beside it (1.2e−5 of the largest flux in the closed 3D blast box, 45 of 192 faces; 2.7e−5 after a
regrid). The same holds for any set with symmetric walls and a conservative transfer: at the wall
the levels are filled independently.

**API.** None new; a semantic of `CellBoundary(Relative(f))` (or an opt-in): a point that lies
both beyond (or on the wall of) an outer face and on a coarse-fine face is filled from the level
that owns that plane by the set's own transfer — restricted from the fine level where the fine
block owns it, prolongated from the coarse level's hook-filled values where the coarse one does —
rather than by the hook on each level.

**Upstream acceptance.** On an edge set under `Conservative` with `vertex_prolongation`, symmetric
walls and a relative hook, on a two-level mesh whose coarse-fine face meets an outer face in
D = 2, 3: the line integral of every coarse edge on the wall equals the sum of its fine edges'
after a fill and after a regrid; one fill is still the condition's fixed point.

**Downstream check.** `test/refinement_tests.jl`'s state-flux testset asserts the closed box's
mismatch to rounding instead of recording it.

**Workaround:** none needed: the flux through a face at a Dirichlet boundary is the boundary
condition's, which nothing conserves; recorded in `CODE.md` ("Conservation at coarse-fine
faces").

### EOS-0 — Register EntropyEOS (req, step 0; workaround)

Register EntropyEOS in General, as 1.x (it is at 1.0.0). Until then `Project.toml`'s
`[sources]` names `url = "https://github.com/eschnett/EntropyEOS.jl", rev = "main"`, which CI
and the clean-checkout check read.

**Inconsistent:** the table records EntropyEOS releases up to 2.2.1 (EOS-9 in 2.0.0, EOS-10 in
2.2.1), and `CLAUDE.md` says `[sources]` pins the tag `v2.2.1`; this item says EntropyEOS is at
1.0.0 and `[sources]` names `rev = "main"`.

### EOS-1 — An abstract EOS interface (req, step 5)

Landed in EntropyEOS 1.0.0, except that `κ` stays unexported. The request and its record:
[HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos); what this package relies on:
[`CODE.md`, "EntropyEOS"](CODE.md#entropyeos).

### EOS-2 — Analytic Γ-law and hybrid EOSes (req, step 5)

Landed in EntropyEOS 1.0.0. The request and its record:
[HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos); its use and the round trip's conditioned
accuracy: [`CODE.md`, "EntropyEOS"](CODE.md#entropyeos),
[`CODE.md`, "Thermodynamics"](CODE.md#thermodynamics),
[`CODE.md`, "Primitive recovery"](CODE.md#primitive-recovery).

### EOS-3 — Public fixed-entropy inversion and entropy from energy (wanted, step 12)

**API.**
```julia
con2prim_entropy(eos, in::Con2PrimIn, s, opts::Con2PrimOptions, w_guess=NaN, u_guess=NaN)
    -> Con2PrimOut      # in.τ ignored
entropy_from_energy(eos, ρ★, yₑ, U; u_guess=NaN, max_iter=60) -> (s, pt::EOSPoint, converged::Bool)
```

**Semantics.**
- `con2prim_entropy` wraps `inner_solve_w` (con2prim.jl:258).
  - `result = converged_fallback` when `|f1| ≤ tol·cosh w`, otherwise `failed_max_iter`.
  - It sets a distinct flag when `V ≥ tanh(w_max)` pinned the rapidity.
- `entropy_from_energy` wraps `seed_s_solve` (con2prim.jl:469). It reports convergence instead
  of silently pinning to the window.

**Upstream acceptance.**
- A round trip at fixed s on random states.
- Pinning and non-convergence are reported, never silent.

### EOS-4 — Cold β-equilibrium slice (later, step 13)

```julia
beta_equilibrium(eos::EOSTableView, ρ★; T_MeV) -> (yₑ, s, converged::Bool)
cold_slice(eos, ρ★s::AbstractVector; T_MeV) -> (yₑ, s, p, U)
```
- `beta_equilibrium` solves `μ̃(ρ★, s(T), yₑ) = 0` on `yₑ_bounds` by bracketed secant with a
  bisection fallback (`μ̃`'s `yₑ` derivative is unavailable).
- `s(T)` comes from `sigma_extended`.

### EOS-5 — Withdrawn (2026-10-17): notes on two repairs at the edge of the valid range

Withdrawn; its notes, kept for Erik, and their resolution by EOS-9:
[HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos).

### EOS-6 — A density raised to the atmosphere's gets the atmosphere's velocity (superseded)

Superseded by EOS-8 and this package's recovery wrapper. The record:
[HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos).

### EOS-7 — The atmosphere's entropy in the atmosphere band (not pursued)

Not pursued. The record: [HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos).

### EOS-8 — Report the density floor (req; landed in EntropyEOS 1.1.0, unregistered)

Landed in EntropyEOS 1.1.0; in 2.0.0 the flag is the `ρ_min` face (bit 11). The request and its
record: [HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos); the flag's use today:
[`CODE.md`, "Primitive recovery"](CODE.md#primitive-recovery).

### EOS-9 — Step 4's endpoint branches: edge-consistent, guarded, reporting facts (landed in EntropyEOS 2.0.0)

Landed in EntropyEOS 2.0.0 as the two-level projection. The request as written, its measurements
and its record: [HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos); the recovery on it:
[`CODE.md`, "Primitive recovery"](CODE.md#primitive-recovery),
[`CODE.md`, "Floors, the atmosphere and the magnetosphere"](CODE.md#floors-the-atmosphere-and-the-magnetosphere).

### EOS-10 — The nested fallback allocation-free under `--check-bounds=yes` (landed)

Landed in EntropyEOS 2.2.1. The request and its record:
[HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos).

### EOS-11 — The validity test of `con2prim_safe`, public (req; landed in EntropyEOS 2.1.0)

Landed in EntropyEOS 2.1.0 as `inside_window`. The request and its record:
[HISTORY.md, "EntropyEOS"](HISTORY.md#entropyeos); its use:
[`CODE.md`, "Reconstruction"](CODE.md#reconstruction), [`CODE.md`, "EntropyEOS"](CODE.md#entropyeos).

### GH-0 — Land step 8 on `main` (req, step 11)

Landed by inspection (TreeGH's `main`, fbfd416). The items below are written against `main`; they
apply to whatever `main` is when they are made, and TreeGRRMHD targets `main` only. The record:
[HISTORY.md, "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic).

### GH-1 — The metric at TreeGH's vertices (req, step 11)

Landed as `ADMMetric` on TreeGH's `main` (PR 25, 61bcb75; still 0.1.1, unregistered). The request,
its earlier forms and its record:
[HISTORY.md, "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic); the contract:
[`CODE.md`, "The coupling contract"](CODE.md#the-coupling-contract),
[`CODE.md`, "TreeGeneralizedHarmonic"](CODE.md#treegeneralizedharmonic). Its downstream check
for step 11 remains: TreeGH's `ADMMetric` of the frozen TOV background agrees with the sampled
adapter to TreeGH's `K` order, and the frozen-metric TOV run reproduces step 10's Cowling run to
truncation error.

### GH-2 — Registration, registry-satisfiable compat, the Julia floor (req, step 11; workaround)

TreeGH is not a dependency of this package (`CODE.md`, "The coupling contract"); the item is for
whatever loads both for step 11's coupled runs — this package's test environment or a project of
their own, step 11's choice.

- Register TreeGeneralizedHarmonic and KorzynskiSpin in General.
- Every `[compat]` entry must be satisfiable from registered releases alone. `[sources]` of a
  dependency is ignored when TreeGRRMHD resolves it, so the TreeAMR features TreeGH uses must be
  in a released TreeAMR. TreeGH 0.1.0 is now on the registered TreeAMR 0.2, which settles that
  part. What is left, as of fbfd416: TreeGH itself, TreeIOHDF5 (new, for its checkpoints) and
  IMEXRungeKutta are unregistered, and its `ApparentHorizonFinder = "2.1"` is ahead of the
  registry, which has 2.0.0. KorzynskiSpin 1.1.0 and AbstractSphericalHarmonics are registered.
- **The Julia floor.** Every Tree* package's floor is 1.13 (CODE.md "Julia version"); nothing is
  asked of TreeGH.
- **Workaround until registered:** `[sources]` names TreeGH, TreeIOHDF5 and, if still needed,
  ApparentHorizonFinder by URL.

### GH-3 — Matter input (req, step 11)

The adapter returns `(ρ, S_i, S_ij)` and TreeGH assembles `T_ab` with its own metric (`CODE.md`,
"The coupling contract").

**API** (proposed; the verbs are GH-4's; TreeGH's `CODE.md` has the same form).
```julia
TreeGH.rhs!(du, p::GHProblem, t; matter = nothing)
at = matter(U::FieldSet)          # bound to TreeGH's vertex set
(; ρ, S, Sij) = at(n, b)          # at an owned vertex
```

**The `matter` argument.**
- `nothing`, or an adapter that, bound to TreeGH's vertex set, returns at an owned vertex `ρ`,
  `S::SVector{3,T}` and a symmetric `Sij::SMatrix{3,3,T}`: `ρ = n^a n^b T_ab`,
  `S_i = −γ_i^a n^b T_ab`, `S_ij = γ_i^a γ_j^b T_ab`, covariant, not densitized, in `G = c = 1`
  and the metric's length unit.
- TreeGH calls it at its owned vertices only and never stores it; whether the provider computes
  per call or stores is the provider's choice.
- It adds a seventh `Val`, `Val(HASMATTER)` with `HASMATTER = matter !== nothing` (`W` became
  the sixth). With `false`, the kernel is bit-identical to today.

**Kernel.** With the head's `α`, `β^i`, `γ^ij`:
```
T_tt = α² ρ − 2α β^i S_i + β^i β^j S_ij
T_ti = −α S_i + β^j S_ij
T_ij = S_ij
T    = γ^ij S_ij − ρ
msrc += 16π · α · sqrtγ · pack_sym10(T_ab − g_ab · T / 2)
```
- `∂ₜh` is unchanged. Inside an interior layer the term is weighted by `w`, like the rest of `F`.
- The coefficient follows from notes/formulation.md §1–2. `R^H_ab = 8π(T_ab − ½g_abT)` makes
  `S0 → S0 − 16πT̄_ab`, and the source enters as `−α√γ S0`.
- The adapter runs inside TreeGH's hottest kernel: it must be device-clean, and on the CPU it
  is called once per SIMD lane (TreeGH's `CODE.md`).

**Monitors.**
- `adm_constraint!` subtracts `16πρ` from `ℋ` and `8πS_i` from `ℳ_i` directly.
- The `∂ₜ∂ₜg` reconstruction inside `adm_constraint_kernel!` includes the matter term.
- `gh_constraint!` is unchanged.

**Lifecycle.** Nothing to carry: `matter` is an argument of each call, and the caller builds
it from its own prepared state, after a regrid as before it.

**Upstream acceptance.**
- `HASMATTER = false` is bit-identical to today.
- A static non-vacuum test metric: pass as `matter` the projections `(ρ, S_i, S_ij)` of
  `G_ab/8π` from SpacetimeMetrics' `EinsteinTensor` at each vertex, and freeze `H = −Γ[g]`.
  - The RHS then vanishes at order `q` under refinement.
  - The matter-corrected `ℋ` and `ℳ_i` vanish at order `q`, and do not without the matter
    terms.

**Downstream check.** The same test with TreeGRRMHD's `(ρ, S_i, S_ij)` from a TOV star
(step 11).

### GH-4 — RHS and limiters callable from an external integrator (req, step 11)

The verbs are TreeGH's own under names this package shares, and a script calls them, not a
driver here (`CODE.md`, "The coupling contract").

**API** (the names module-qualified; the signatures proposed).
```julia
TreeGH.prepare!(p::GHProblem, u::AbstractVector, t)  # scatter!(p.U, u) and the fill gh_rhs! does
                                                     # today; returns the prepared state (GH-1)
TreeGH.rhs!(du::AbstractVector, p::GHProblem, t; matter = nothing)
                                                     # kernel launch only; precondition: prepare!
                                                     # at (u, t); `matter` is GH-3's
TreeGH.max_speed(p::GHProblem, u::AbstractVector, t) # today's unexported max_speed_of
TreeGH.step_limit!(u::AbstractVector, p, t)          # integrator-free gh_step_limiter!
TreeGH.stage_limit!(u::AbstractVector, p, t)         # integrator-free gh_stage_limiter!
gh_rhs!(du, u, p, t)   # unchanged: prepare!(p, u, t); rhs!(du, p, t)
```

**Semantics.**
- Declared `public`, not exported (proposed): this package declares the same names, and a
  script that loads both qualifies them.
- `u` and `du` may be contiguous segment views (`stateview`, AMR-1).
- `evolve!` should be a composition of public functions. Name any piece that is still inlined:
  - the per-chunk `dt`/steps;
  - the rebuild after `regrid!`;
  - the buffer width;
  - the initial fill;
  - the record.

  Several are already exported: `indicator_flags`, `GHProblem`, `fill_exact!`, `dirichlet`,
  `constraint_norms` and `adm_constraint!`.
- `CODE.md`'s step-11 sketch ("Two scripts") calls these pieces as
  `TreeGH.GHProblem(forest, case; q)` (building its own vertex set on a given forest),
  `TreeGH.evolved_sets`, `TreeGH.initial_state!` (`fill_exact!` into a segment), `TreeGH.flags`
  (`indicator_flags`), `TreeGH.regrid_pairs` (`U => schedule`) and `TreeGH.rebuild`. The names
  are proposed; what the script needs is that each piece exists and takes a segment.

**Upstream acceptance.**
- `gh_rhs!` is bit-identical to today.
- `TreeGH.prepare!` and `TreeGH.rhs!` on a segment of a two-set state vector equal `gh_rhs!` on
  a standalone vector.
- The limiters on a segment equal the SciML-hook versions.

### GH-5 — Ghost width `≥ q/2 + 1` (withdrawn 2026-11-06; was wanted, step 10)

Withdrawn: not needed, since the interpolation reads this package's own vertex set `Mv` at
`G = 4`. The request, kept for TreeGH's reference:
[HISTORY.md, "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic).

### GH-6 — Later

A damped-harmonic gauge driver, radiative or constraint-preserving outer boundaries, excision or
a generalized interior for collapse, and Ψ₄.

### SM-1 — Allocation-free extrinsic curvature (wanted, step 9; workaround)

Landed in SpacetimeMetrics 1.7.0; the workaround is gone. The request and its record:
[HISTORY.md, "SpacetimeMetrics"](HISTORY.md#spacetimemetrics); its use:
[`CODE.md`, "Metric providers"](CODE.md#metric-providers),
[`CODE.md`, "SpacetimeMetrics"](CODE.md#spacetimemetrics).

## Step 0 — Scaffolding

*Done* (2026-10-13). Results in `CODE.md`, "Julia version" and "Upstream prerequisites"; the
step's changes and acceptance list are in
[HISTORY.md, "The steps of the plan"](HISTORY.md#the-steps-of-the-plan).

## Step 1 — Units, variables and the 3+1 primitive recovery

*Done* (2026-10-13). Results in `CODE.md`, "Conventions", "Thermodynamics" and "Primitive
recovery"; the step's changes, acceptance list and measurements are in
[HISTORY.md, "Primitive recovery"](HISTORY.md#primitive-recovery).

## Step 2 — Pointwise physics

*Done* (2026-10-13). Results in `CODE.md`, "Maxwell", "Ohm's law and the conductivity",
"Stress-energy for the spacetime", "Riemann solver", "The electromotive force for `A`",
"Sources" and "Metric providers"; the step's changes, acceptance list and measurements are in
[HISTORY.md, "The equations"](HISTORY.md#the-equations).

## Step 3 — The implicit stage solve

*Done* (2026-10-13; its review round 2026-10-14). Results in `CODE.md`, "The implicit stage
solve" and "Ohm's law and the conductivity"; the step's changes, acceptance list and
measurements are in [HISTORY.md, "The implicit stage solve"](HISTORY.md#the-implicit-stage-solve),
its review round in
[HISTORY.md, "Ohm's law and the conductivity"](HISTORY.md#ohms-law-and-the-conductivity).

## Steps 4a–4c — The IMEX package (in its own repository)

*Done* in `~/src/jl/IMEXRungeKutta`, where 4a (tableaus), 4b (the integrator) and 4c
(validation) name that package's milestones. Results in `CODE.md`, "The IMEX package" and "The partition"; what this package needed from it
and its state are in [HISTORY.md, "The IMEX package"](HISTORY.md#the-imex-package).

## Step 5 — The staggered scheme in 1D: smooth problems (rung 1a)

*Done* (2026-10-14). Results in `CODE.md`, "Where the fields live", "Field sets",
"Reconstruction", "Riemann solver", "The electromotive force for `A`", "The right-hand side",
"Time integration", "Boundaries" and "Validation ladder" 1; the step's changes, predictions,
acceptance list and measured outcomes are in
[HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder), the suite in
[HISTORY.md, "The test suite"](HISTORY.md#the-test-suite).

## Step 6 — Shock tubes and the stiff limit in 1D (rung 1b)

*Done* (2026-10-14). Results in `CODE.md`, "Validation ladder" 1, "Floors, the atmosphere and
the magnetosphere", "The partition" and "Boundaries"; the step's changes, predictions,
acceptance list and measured outcomes are in
[HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder), the suite in
[HISTORY.md, "The test suite"](HISTORY.md#the-test-suite).

## Step 7 — 2D and 3D, and higher-order reconstruction (rung 2a)

*Done* (2026-10-15). Results in `CODE.md`, "Where the fields live", "Reconstruction" and
"Validation ladder" 1 and 2; the step's changes, acceptance list and measured outcomes are in
[HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder), the suite in
[HISTORY.md, "The test suite"](HISTORY.md#the-test-suite).

## Step 8 — Coarse-fine faces and regridding (rung 2b)

*Done* (2026-10-16). Results in `CODE.md`, "Conservation at
coarse-fine faces", "Refinement", "Field sets", "Boundaries" and "Validation ladder" 2; the
step's changes, acceptance list, measured outcomes and its finish are in
[HISTORY.md, "Refinement"](HISTORY.md#refinement), the suite in
[HISTORY.md, "The test suite"](HISTORY.md#the-test-suite).

## Step 9 — Analytic curved backgrounds (rung 3)

*Done* (2026-11-05). Results in
`CODE.md`, "Metric providers", "Boundaries", "The interior", "Validation ladder" 3 and "Known
risks"; the step's changes, predictions, acceptance lists (round 4's included) and measured
outcomes are in [HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder), the suite in
[HISTORY.md, "The test suite"](HISTORY.md#the-test-suite).

## Step 10 — The TOV star in the Cowling approximation (rung 4)

Built and measured, round 2 included; the items marked **(proposed)** are for Erik. The step
reads no TreeGH. Its measurements against the acceptance list below are in
[HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder), the changes to its scope in
[HISTORY.md, "The coupling contract"](HISTORY.md#the-coupling-contract), and the current numbers
in `CODE.md`, "Validation ladder" 4 and "Metric at cells, faces and edges" ("As built in step
10, round 2").

**Inconsistent:** this step is not marked done, its items carry **(proposed)**, and round 2 is
"for Erik"; `CLAUDE.md`'s former "Current state" (now [HISTORY.md, "The steps of the
plan"](HISTORY.md#the-steps-of-the-plan)) said step 10 is "built and measured on
`claude/step-10-cowling-tov`, for Erik's review". But the git log of `main` has "Merge pull
request #6 from eschnett/claude/step-10-cowling-tov" (55da833) and, after "Step 10, round 2: the
vertex metric under the redesigned coupling contract" (0e81ea9), "Merge pull request #7 from
eschnett/eschnett-claude/coupling-contract-change-419d41" (772f3ec): both rounds are merged.

`CODE.md`: "The coupling contract", "Metric providers", "Gauge for the TOV milestones",
"Floors, the atmosphere and the magnetosphere", "Refinement", "Riemann solver".
Waits: nothing. AMR-4 (strongly wanted) and AMR-3 (wanted) have landed.

Changes:
- **`src/tov.jl`:** the host TOV solve in isotropic coordinates, for the polytrope K = 100,
  Γ = 2, ρ_c = 1.28e−3 (`IdealGasEOS` at `polytropic_entropy`).
- **`TOVMetric <: SpacetimeMetrics.AbstractMetric`:** `isbits` and dual-generic. The step decides
  its storage (a fixed-size spline, or an Adapt-ed device array) and records it.
  **The Cowling star's provider is `Analytic(TOVMetric)`**, pointwise like every analytic
  provider; its cost per call is measured against Kerr-Schild's 102 ns and 234 ns
  (`bin/metric_cost.jl`).
- **The potentials' outer boundary** is decided here. The Cowling TOV star is static, so if its
  exterior field comes from an exact static potential, `static_potential = true` (option 1, the
  case's own potential in the ghosts) applies as is; options 2 (a gradient-extrapolating
  relative hook) and 3 (a gauge that damps a static `χ`) matter only for dynamic cases
  (`CODE.md`, "Boundaries").
- **This package's side of the coupling contract** **(proposed)** (`CODE.md`, "The coupling
  contract"):
  - the kernels accept a producer's pointwise metric beside `Flat` and `Analytic`: each kernel
    binds it to the field set it runs over, and `point_metric` and `metric_gradient` pass the
    point's `(I, b)` beside `x` (the analytic providers ignore it); `src/metric.jl`'s one
    conversion serves both;
  - a sampled adapter of any SpacetimeMetrics metric in the contract (test code, unless a case
    needs it): bound to a field set, it evaluates `dmetric` at its points. It tests this side
    before TreeGH has GH-1, and it is GH-1's reference.
- **This package's verbs** **(proposed)**: `TreeGRRMHD.prepare!` (scatter, fill and recovery,
  given a metric), `TreeGRRMHD.rhs!`, `stage_solve!`, `stage_limit!`, `step_limit!` and
  `max_rate`, each taking the metric as a keyword, and `evolved_sets`, `initial_state!` and
  `MHDProblem(fields, case; …)` (`evolve!`'s `build()`); `public`, not exported.
  `mhd_integrator` and `evolve!` become compositions of them, bit for bit.
- **The step-10 script** **(proposed)**: `CODE.md`'s sketch ("Two scripts") as a file in `bin/`,
  which a test includes and runs for a few steps, so that the sketch stays code that runs.
- **TreeGH is not a dependency** of the package (`[deps]`), in this step or later (`CODE.md`,
  "The coupling contract"); how step 11's coupled tests load it is step 11's.
- **`src/cases/tov.jl`**, with geometric refinement boxes.
- **Round 2** **(proposed)**: **the vertex metric replaces `PointwiseMetric`** (`CODE.md`, "The
  coupling contract", "Metric at cells, faces and edges"):
  - `Mv`, a 16-variable vertex set `(α, β^i, γ_ij, K_ij)` at `G = 4`, built only for a run with
    a producer's metric (`N ≥ 10`; an `ArgumentError` says why), PointValue width 6 in its
    fill, its outer boundary Dirichlet from the case's own analytic provider;
  - `vertex_metric!(mhd, adm)`: the owned vertices from any ADM adapter in `ADMMetric`'s form
    (`at = adm(Mv)`, `(; α, β, γ, K) = at(n, b)`), then `Mv`'s ghost fill; it returns
    `VertexMetric`, the provider the verbs take as `metric`;
  - `VertexMetric`'s `point_metric` and `metric_gradient` at the indexed calls step 10 added:
    vertices read, edges 4, faces 16 and cells 64 Lagrange points, `∂α`, `∂β`, `∂γ` from the same
    interpolant at the cells, `K` interpolated; the multilinear fallback where `α ≤ 0` or `γ`
    fails Cholesky, counted in `X`; the derivative layout (`MetricGradient` native 3+1
    derivatives, or `∂_j g_ab` by the product rule) decided and recorded;
  - the sampled adapter becomes a sampled **ADM** adapter: any SpacetimeMetrics metric's
    `(α, β, γ, K)` at the owned vertices of the bound set (test code), GH-1's reference;
  - `PointwiseMetric` and its per-centering binding are removed; the step-10 script is
    unchanged, since it runs on `Analytic(TOVMetric)`.

  As built, round 2 departs from this list (`Mv` in its own `VertexMetricSet`, the fill's time
  an argument, the fallbacks counted by `metric_fallbacks`, the product rule, `SampledADM` in
  `src/`): `CODE.md`, "Metric at cells, faces and edges" ("As built in step 10, round 2"), and
  [HISTORY.md, "Metric at cells, faces and edges"](HISTORY.md#metric-at-cells-faces-and-edges).

Accept:
- **The contract:** a rung-3 case and the TOV star, run through the sampled adapter, reproduce
  their `Analytic` runs to round-off (bit for bit where both compute the point alike). In round
  2 the check is this package's, and the runs agree to the interpolation's truncation error,
  not round-off:
  - Through the sampled ADM adapter and `Mv`, on Kerr-Schild at `a = M/2` and on the TOV star,
    two levels: `g` and `∂g` converge at 4th order against `dmetric` at every centering, the
    outermost ghost layer of a `G = 3` cell set and the points next to a coarse-fine face
    included; `K` at 4th order; the vertices bit for bit.
  - The Cowling star through `VertexMetric` reproduces its `Analytic(TOVMetric)` run to
    200 M, the difference converging at 4th order, with no fallback.
  - `Flat` and `Analytic` runs are bit for bit unchanged.
- **The script:** its `main` makes the run `evolve!` makes, bit for bit.
- **Stationarity:** the central-density drift after 200 M converges at ≥ 1.5.
- **Conservation:** rest mass to round-off, net of the outer boundary flux.
- **Modes:** the F, H1 and H2 frequencies are within 3% of Font et al. (2002)'s Cowling values
  (values copied with citation).
- **Floored cells** at a coarse-fine surface for width 1, width 3, and limited (AMR-4)
  **decide the prolongation**.
- **Light-cone against split speeds** on the star **decides the Riemann speeds**.

## Step 11 — The TOV star in a dynamical spacetime (rung 5)

`CODE.md`: "The coupling contract", "Spacetime coupling", "Time integration". Waits: **GH-2, GH-3, GH-4**
(GH-1 landed 2026-11-08).

Changes:
- **This package's verbs** are step 10's; this step adds `flags`, `regrid_pairs` (today's,
  public) and `rebuild`, as the script's regrid needs them.
- **This package's matter adapter** (`TreeGRRMHD.ADMMatter`, **(proposed)**): `(ρ, S_i, S_ij)`
  at TreeGH's owned vertices, the mean of the `2^D` cells' `stress_energy` (its `S^ij` lowered),
  2nd order, with a 4th-order option. Whether it stores them at the cells during `prepare!` or
  computes them per call is measured and decided here.
- **The coupled script**, `CODE.md`'s step-11 sketch ("Two scripts"):
  - one IMEXRungeKutta integrator over both packages' sets;
  - the explicit part: `TreeGH.ADMMetric` (or `TreeGH.prepare!`), `vertex_metric!` into `Mv`,
    `TreeGRRMHD.prepare!` with that metric, and both `rhs!`;
  - the implicit part: the Ohm stage solve with TreeGH's metric at `u★` (TreeGH's limiter never
    inside it);
  - the limiters composed; `Δt = min(GH, matter)`;
  - regrid of every set, and the rebuild of both problems.

  Where the script and the coupled tests live, and how they load both packages, is decided
  here (open); so is skipping TreeGH's repeated fill (eleven per step written plainly, four
  needed; `CODE.md`, "Two scripts"), measured. The regrid branch, unused by the star's static
  boxes, gets a test of its own.
- **The GH CFL factor under SSP3(4,3,3)'s explicit part**, measured here.

Accept:
- **The frozen metric:** TreeGH's `ADMMetric` of the TOV background, with its right-hand side
  not called, reproduces step 10's Cowling run to truncation error (GH-1's downstream check).
- **GH-3's static-matter check** passes with TreeGRRMHD's `(ρ, S_i, S_ij)`.
- **Constraints:** the matter-corrected ADM constraints converge at ≥ 1.8 away from the surface
  **(predicted)**.
- **Modes:** F, H1 and H2 within 3% of Font et al. (2002)'s coupled values **(predicted)**.
- **Conservation:** rest mass to round-off.
- **Decides:** the largest stable GH `cfl` under SSP3, on the gauge wave and on the TOV star,
  which becomes the default; and **the matter's transfer order (open)**.

## Step 12 — The magnetized TOV star (rung 6)

`CODE.md`: "Ohm's law and the conductivity", "Floors, the atmosphere and the magnetosphere",
"Primitive recovery". Waits: EOS-3 (wanted).

Changes:
- `DensityProfile` for σ.
- An interior poloidal `A_φ` continued by an exterior dipole, sampled on edges.
- Vacuum `E` outside.
- The entropy fallback, wired to EOS-3.
- Diagnostics: field energy inside and outside the star, `E·B`, and `|q|`.

Accept:
- **∇·B** at round-off.
- **The exterior dipole** stays the initial field to truncation error, and the drift converges.
- **Inside the star**, `E·B → 0` at truncation error.
- **The stage-solve outcome map** in the star is recorded.
- **With and without the entropy fallback**, the failure counts **decide whether `D s` earns its
  keep (open)**.

## Step 13 — Tabulated EOS and Yₑ (rung 7)

`CODE.md`: "Thermodynamics". Waits: **EOS-4**.

Changes:
- A TOV star on a real table (SFHo or DD2), with tests gated by `ENTROPYEOS_TABLE_DIR`.
- β-equilibrium initial data.
- `Yₑ` advection.

Accept:
- Stationarity as in steps 10 and 11.
- The `D Yₑ` integral conserved to round-off.
- Gated tests skip cleanly when the variable is unset.

## Step 14 — Precision, threads, devices, benchmark

Waits: AMR-5 (wanted).

Changes:
- Device tests behind `TREEGRRMHD_TEST_BACKEND` (`cuda`, `metal`).
- A Float32 smoke run.
- Bit-identical states across thread counts.
- Benchmark scripts giving ns per cell per RHS phase, after TreeGH's step 9.

Accept:
- **CUDA Float64:** the Cowling TOV matches the CPU run to recorded differences.
- **Metal Float32:** the smoke run is recorded, pass or fail.
- **The per-phase timing table** is recorded.

## Step 15 — Review pass

- Read `CODE.md` against the code once more.
- Mark rungs 1–7 *(Done.)*.
- Update `README.md`'s status, `CODE.md`'s status paragraph and `CLAUDE.md`'s "Commands".
- Move anything still marked **(predicted)** or **(open)** to measured, or to "Possible
  extensions".
- Delete this file.
