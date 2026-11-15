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
  - [GH-7 — The regrid verbs and the constraint norms of a segment (wanted, step 11; workaround)](#gh-7--the-regrid-verbs-and-the-constraint-norms-of-a-segment-wanted-step-11-workaround)
  - [GH-8 — Constraint damping on a star (note)](#gh-8--constraint-damping-on-a-star-note)
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
| GH-3 | TreeGH | req | 11 | matter input, a pointwise `matter` adapter returning `(ρ, S_i, S_ij)` | landed on TreeGH's `main` (79ad132; 0.1.1, unregistered) |
| GH-4 | TreeGH | req | 11 | RHS and limiters callable from a script: `TreeGH.prepare!`, `TreeGH.rhs!`, … | landed on TreeGH's `main` (0cd5f00; 0.1.1, unregistered), the regrid pieces split off as GH-7 |
| GH-5 | TreeGH | — | — | was: ghost width `≥ q/2 + 1`, wanted, step 10; not needed: the interpolation reads this package's own `G = 4` vertex set | withdrawn |
| GH-6 | TreeGH | later | — | gauge driver, radiative boundary, excision, Ψ₄ | waiting |
| GH-7 | TreeGH | wanted, workaround | 11 | the regrid verbs (`flags`, `regrid_pairs`, `rebuild`) and `constraint_norms(p, u, t)` on a segment | waiting |
| GH-8 | TreeGH | note | — | constraint damping on a star: `γ0 = 1/M` grows a mode with matter; no change requested | written, not sent |
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

Landed on TreeGH's `main` (79ad132; still 0.1.1, unregistered): `rhs!(du, p, t; matter)` and
`adm_constraint!(p, u, t; matter)`, the adapter bound once per call as `at = matter(p.U)` and
called as `(; ρ, S, Sij) = at(n, b)` at an owned vertex. The request as written and its record:
[HISTORY.md, "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic); the contract:
[`CODE.md`, "The coupling contract"](CODE.md#the-coupling-contract). Its downstream check is
step 11's: TreeGH's static-matter test (its "anti-Cowling" run) passes with this package's
`(ρ, S_i, S_ij)` from the TOV star.

### GH-4 — RHS and limiters callable from an external integrator (req, step 11)

Landed on TreeGH's `main` (0cd5f00; still 0.1.1, unregistered), except the regrid branch's
pieces and `constraint_norms(p, u, t)`, which are [GH-7](#gh-7--the-regrid-verbs-and-the-constraint-norms-of-a-segment-wanted-step-11-workaround):
`prepare!(p, u, t) -> p`, `rhs!(du, p, t; matter)`, `stage_limit!`, `step_limit!`,
`initial_state!(u, p, t)` and `evolved_sets(p)` are `public`; `max_speed(p, u, t)` is exported;
a segment must be a contiguous view; `evolve!` composes them. The request as written and its
record: [HISTORY.md, "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic).

### GH-5 — Ghost width `≥ q/2 + 1` (withdrawn 2026-11-06; was wanted, step 10)

Withdrawn: not needed, since the interpolation reads this package's own vertex set `Mv` at
`G = 4`. The request, kept for TreeGH's reference:
[HISTORY.md, "TreeGeneralizedHarmonic"](HISTORY.md#treegeneralizedharmonic).

### GH-6 — Later

A damped-harmonic gauge driver, radiative or constraint-preserving outer boundaries, excision or
a generalized interior for collapse, and Ψ₄.

### GH-7 — The regrid verbs and the constraint norms of a segment (wanted, step 11; workaround)

What a coupled script needs beyond GH-4 to regrid TreeGH's set together with this package's in
one `regrid!`, and to monitor the constraints of a segment. TreeGH's `CODE.md` ("The verbs
(GH-4)") lists them as not built, waiting for step 11 to need them.

**API** (module-qualified, `public`, not exported; the names are this package's, so that the
script reads the same for both packages).
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
- `flags` and `constraint_norms` scatter `u` (a contiguous segment, as for GH-4) and so
  invalidate a preceding `prepare!`, like every monitor.
- `regrid_pairs` returns pairs for TreeAMR's `regrid!` over several sets: the caller concatenates
  them with this package's and calls `regrid!` once. The hook is the one `evolve!`'s
  `regrid_mesh` passes today.
- `rebuild` is `regrid_mesh`'s rebuild without the `regrid!`, the `statevector` and the limiter,
  which the script does over every set. It does not run `stage_limit!`: the script does, on the
  gathered segment.

**Upstream acceptance.**
- `evolve!`'s regrid, rewritten as `flags`, `regrid!` over `regrid_pairs`, `rebuild`,
  `statevector`, `gather!` and `stage_limit!`, is bit-identical to today's on a refined
  hole fixture.
- On a segment of a two-set state vector, each verb equals the standalone call.

**Downstream check** (in `coupled/`). The script's regrid branch, with TreeGH's verbs in place
of the workaround, is bit-identical to the workaround's.

**Workaround** (allowed; marked `GH-7` in `coupled/gh7.jl`): the four composed in the script from
TreeGH's exports (`gh_indicator!`, `dirichlet`, `GHProblem(U, schedule, case; q)`,
`GhostSchedule`, `adm_constraint!`, `constraint_norms(p)`), reading `GHProblem`'s fields `U`,
`schedule` and `case`.

### GH-8 — Constraint damping on a star (note)

A measurement for TreeGH, from step 11. No change is requested; what to do with it is TreeGH's
call.

**What was found.** The star is TreeGRRMHD's TOV star: `K = 100`, `Γ = 2`, `ρ_c = 1.28e−3`,
`M = 1.400`, `R = 9.586`. TreeGH evolves it with a frozen sampled `H_a`, `ε_KO = 1/2` and a
constant `γ0`. At `γ0 = 1/M`, the near-hole value of `GHCase`'s docstring, a mode grows. It
doubles every 25 M (an e-fold of about 36 M) at `h = 0.6` and at `h = 0.375`. The rate does not
change with resolution, and the amplitude is 2.5 times smaller at the finer `h`: the truncation
error seeds it at second order. In the coupled run the star collapses near 400 M. TreeGH alone
shows the same growth, on the initial star's matter held fixed (GH-3's adapter, the anti-Cowling
setting).

The largest change of TreeGH's state, TreeGH alone on the frozen matter, at `h = 0.6` to 500 M:

| `γ0 M` | `γ2` | result |
|---|---|---|
| 0 | 0 | bounded; `ℋ` noisy, to 3e−4 |
| 1/20, 1/10 | 0 | flat, 1.6e−2 (6.7e−3 at `h = 0.375`); `ℋ` 3–5e−5 |
| 1/4 | 0 | grows, then saturates at 2.8e−2 |
| 1/2 | 0 | grows, 0.43 at 500 M |
| 1 | 0 | doubles every 25 M |
| 1 | −1/2 | bounded, 3e−2; `ℋ` 3e−4 |
| 4 | 0 | `NaN` at 105 M |

Neither `ε_KO = 0` nor a box twice as wide changes the interior. Coupled: `γ0 M = 1/4` is
stable to 250 M, `γ0 M = 1` with `γ2 = −1/2` to 390 M, and `γ2 = 1` crashes at 80 M.
TreeGRRMHD now uses `γ0 = 1/(10M)`, `γ2 = 0`, which is `GaussianDamping`'s far-zone value for
a hole, and `ℋ` stays flat to 300 M. The runs:
[HISTORY.md, "Two scripts"](HISTORY.md#two-scripts).

**Not measured:**
- which scale sets the threshold (`M`, `R`, or the light-crossing time);
- the mechanism: the mode needs both the matter and `γ0 ≳ 1/(4M)`, and `γ2 = −1/2` suppresses
  it;
- TreeGH's Plummer star (`test/matter_tests.jl`, `star_case`), which uses `γ0 = 1`, that is
  `10/M`, on short runs.

**Possibly relevant to TreeGH:**
- `GHCase`'s docstring and `CODE.md`, "Gauge and constraint damping", give `γ0 ≈ 1/M` as the
  recipe near a hole. A matter case inherits it unless told otherwise.
- Long matter runs at `γ0 ≳ 1/(2M)` may grow this mode.

**Reproducer** (TreeGRRMHD's `coupled/` environment, about an hour on 64 cores):
```bash
julia --project=coupled -t 64 coupled/tov_coupled_study.jl N=10 t_end=400 γ0M=1
```

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

*Done* (both rounds; accepted by Erik 2026-11-10). Results in `CODE.md`, "The coupling
contract", "Two scripts", "Metric providers", "Metric at cells, faces and edges", "Gauge for the
TOV milestones", "Refinement" and "Validation ladder" 4; the step's changes, acceptance list and
measured outcomes are in [HISTORY.md, "Validation ladder"](HISTORY.md#validation-ladder) and
[HISTORY.md, "The steps of the plan"](HISTORY.md#the-steps-of-the-plan).

## Step 11 — The TOV star in a dynamical spacetime (rung 5)

`CODE.md`: "The coupling contract", "Spacetime coupling", "Time integration". Waits: GH-2
(workaround: `[sources]`) and GH-7 (wanted; workaround). GH-1, GH-3 and GH-4 have landed on
TreeGH's `main`.

Changes:
- **This package's verbs** are step 10's; this step adds `flags`, `regrid_pairs` (today's,
  public) and `rebuild`, as the script's regrid needs them.
- **This package's matter adapter** (`TreeGRRMHD.ADMMatter`): `(ρ, S_i, S_ij)`
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

  The script and the coupled tests live in `coupled/`, an environment of their own (decided by
  Erik 2026-11-10). TreeGH's repeated fill is skipped by a key on its input: five fills per
  step instead of eleven (accepted by Erik 2026-11-13; `CODE.md`, "Two scripts"). The regrid branch, unused
  by the star's static boxes, gets a test of its own.
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

**State (2026-11-14).** Built and measured (`CODE.md`, "Validation ladder" 5).

Done:
- the frozen metric, to round-off;
- GH-3's static-matter check (rate 1.87);
- rest mass, to 1e−14 net of the reset's injection to `t = 1`; over 1000 M with the fourth
  order and `ρ_min = 1e−12`, 5.4e−12 to 1.3e−12 of 1.506, converging at about 3 (at 1e−10,
  5.3e−10 to 1.3e−10): **a miss**.
  A scan of `ρ_min` (1e−10, 1e−12, 1e−14; jobs 598455–598457) shows it is atmosphere entering
  at the outer faces, proportional to `ρ_min`, as is the reset's part; `ρ_c` and the constraints do not change;
- the constraints at ≥ 1.8 to 1000 M on the finer pair, inside and outside, with the fourth
  order and `ρ_min = 1e−12` (jobs 598575–598577), but for `ℋ` outside over 490–510 M, 1.76:
  **a miss** in one 20 M window, between 1.96 and 2.01 on either side and 2.24 over
  250–1000 M. At 1e−10 the least was 1.88 (597892–597894). With the mean the outside rate
  fell to 1.74 after 500 M;
- the modes: F −0.02% at `h = 0.375` in the unperturbed runs (+0.67% with the mean). H1 and H2 are excited along
  their eigenfunctions (`δv`) and measured in the difference from the unperturbed run. With the
  mean they are within 3% from `h = 0.375`, and at `h = 0.25` within 0.13% and 0.39%. They decay
  by the scheme's dissipation, as `h⁻²`. Unperturbed, with the mean they missed (+8.38%,
  −10.56%); with the fourth order H1 holds (−1.43%, −0.83%, −0.64% at `h = 0.6, 0.5, 0.375`)
  and H2 misses (+9.82% at `h = 0.375`), the mean's start having hidden H1 among the
  spacetime's lines (found in step 12's review; `CODE.md`, "Validation ladder" 5, "Modes");
- the GH `cfl`: 1.10 on the star, so `cfl_gh = 1/2` (accepted by Erik, 2026-11-13);
- `ADMMatter`: the mean computed per call, and the fourth order from a stored cell set with
  its fallback to the mean at the surface (both accepted by Erik, 2026-11-13);
- TreeGH's damping `γ0 = 1/(10M)` (accepted by Erik, 2026-11-13); the first studies, at `γ0 = 1/M`,
  collapsed near 400 M (`HISTORY.md`, "Two scripts");
- the matter's transfer order: fourth order, the script's default (Erik, 2026-11-13), from the
  constraints at `t = 0`; the perturbed modes do not decide it (the unperturbed ones favour it,
  above). At `h = 0.25` to 300 M, its H1 and H2 are +0.8% and +0.5%, against the mean's +0.1%
  and +0.4%.
- the atmosphere: `ρ_min = 1e−12` for the coupled runs, the script's default (Erik,
  2026-11-14), with the three 1000 M studies rerun at it (jobs 598575–598577).

Open for Erik: whether the constraint miss in one window is accepted; a run at `h = 0.1875` to see
whether the 4th order's 0.8% in H1 closes.

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

**State (2026-11-15).** Built and measured on the laptop (`CODE.md`, "Validation ladder" 6);
the Symmetry runs wait for its CPU nodes, all down since 08:00 that day.

Done:
- `magnetized_tov_case` (`src/cases/magnetized_tov.jl`): rung 4's Cowling star threaded by the
  static vacuum dipole of Schwarzschild outside the surface (Wasserman & Shapiro 1983), its
  quadratic continuation in `r̄²` inside, `E = 0`, `Φ̃ = 0`, under
  `MagnetizationCutoff(DensityProfile(σ₀, ρ_min), μ_max)`; `DensityProfile`'s shape kept;
- the diagnostics (`src/field_diagnostics.jl`): field energy inside and outside, `E·B`, `|q|`,
  the drift of `B`, the entropy fallback's trigger, the outcome map; the study script
  `bin/magnetized_tov_study.jl`;
- ∇·B at round-off: 1.9e−16;
- the exterior dipole: the drift outside the star at `t = 2M` converges at 1.85 (`h = 1, 1/2`);
- `E·B` inside converges at 2.45;
- the outcome map, recorded: fixed point everywhere inside the star; 2.9% fallbacks outside, all
  in the cells just outside the surface, the same at any σ₀ or `μ_max` tried.
- the suite on the laptop: 26635/26635 at one thread, at four and with bounds checks, and in
  the clean-checkout check under 1.13.

Blocked:
- the entropy fallback, on **EOS-3**: without it, the counts are measured (the energy inversion
  fails nowhere outside the atmosphere at `t = 2M`); with it, nothing can be measured, so
  whether `D s` earns its keep stays open;
- the studies on the default box with boxes at three resolutions, and the suite on a cluster:
  Symmetry's CPU nodes and Nibi's (674 down, 82 drained) have been down since the morning of
  2026-11-15. The CPU runs go to Nibi (Erik, 2026-11-15; the Symmetry jobs were cancelled), queued
  as jobs 27443903 (the suite at four threads), then 27443904 and 27443905 (one thread, bounds
  checks) and 27443906–27443908 (the study at `h = 0.75, 0.5, 0.375` to 50 M), their logs in
  `/scratch/eschnett/claude/step12/`.

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
