# TreeGRRMHD implementation plan

For the sessions that implement TreeGRRMHD, one step at a time. The **design** lives in
`CODE.md`. Read it first, in full: it is authoritative. This file is only the work breakdown:
- what each step changes, and what it must not change;
- what each step must measure and record;
- and, because this package stands on siblings that are still growing, exactly what it needs
  from them.

`CLAUDE.md` has the mechanics. Delete this file when the last rung is marked *(Done.)* in
`CODE.md`.

**Steps 0 to 3 and 5 to 7 are done; step 8 is next, and waits for nothing** (AMR-2 and AMR-3
have landed). *(Amended in step 7.)* *(Amended 2026-10-14:)* Steps
4a–4c are done in the IMEX package's own repository: IMEXRungeKutta 1.4.0 is tagged,
unregistered, and a dependency from step 0. EOS-1 and EOS-2 have landed in EntropyEOS 1.0.0
(still unregistered: EOS-0, EOS-3 and EOS-4 waiting). TreeAMR 0.2.1 is registered with AMR-1,
AMR-2, AMR-3 and AMR-5, and AMR-4 and AMR-6 in a different form; SpacetimeMetrics 1.7.0 with
SM-1. So step 5, the first evolution, waits for nothing (AMR-1, EOS-1, EOS-2, 4b). TreeGH's
`main` is still fbfd416 (GH-0 landed; GH-1 to GH-6 waiting; unregistered). The "state" column
of the table below has each item.

The steps follow `CODE.md`'s validation ladder (rungs 1–7):
- It is preceded by the pointwise physics and the IMEX package.
- It is followed by a device step and a review pass.
- Every step ends in a green suite and a `CODE.md` update.
- Each step is a brief that a single agent with a fresh context can carry.
- The order is the dependency order. [Upstream requirements](#upstream-requirements) says which
  step waits for what.

## Ground rules, every step

- Read `CLAUDE.md` first. Then read this file's ground rules, sharp edges and the step, then the
  `CODE.md` sections the step names.
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
    later read; the floor is 1.13 (CODE.md "Julia version"; amended 2026-10-15: it was 1.11
    from 2026-09-25). That covers EntropyEOS until EOS-0 and the IMEX package, and will cover
    TreeGH and its unregistered dependencies until GH-2. CI, the clean-checkout check and a
    developer all resolve them the same way, so CI adds nothing by hand. *(Amended in step 0:
    this used to be `Pkg.add(url=…)` in CI.)*
- **Julia 1.13 floor** *(amended 2026-10-15: was 1.11)*. Be generic in `T` and in the backend
  from the first line:
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
  check under the floor (`CLAUDE.md`, "Commands"; amended 2026-10-15: the floor was 1.11).
- **Spec-first.**
  - Amend `CODE.md` and say so: "(amended in step N)", "(measured in step N)".
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
  decision "(proposed in step N)" in `CODE.md`, and lists it in the report.
- **The reviewer:**
  - checks out the branch and runs the suite at one and four threads;
  - reads the `CODE.md` diff;
  - walks the acceptance list against the report;
  - merges only then.

  A failed item goes back to the same agent with the finding, not to a new agent, and the
  reviewer does not fix it themselves.

**The brief** (replace `N`):

    You are implementing step N of PLAN.md in the Julia package at <path>. Read CLAUDE.md,
    then PLAN.md's "Ground rules", "Sharp edges", "Upstream requirements" (the items step N
    waits for) and step N, then the CODE.md sections step N names. Work on the branch
    claude/step-N-<slug> in this worktree; commit in TreeAMR's style with measured numbers in
    the commit body; do not merge, push, or touch TODO.md or any sibling checkout. Do all of
    step N. If part of it is blocked — an upstream item not landed, a device you do not have —
    finish everything else and say exactly what is blocked, by which ID. Never loosen an
    assertion to get green. When you would ask a question, decide, mark it "(proposed in step
    N)" in CODE.md, and list it in the report. End with the report PLAN.md specifies.

**The report**, in this order:
1. the branch and its commits, one line each;
2. the acceptance list, each item marked *done*, *measured* (with the number), or *blocked*
   (with the ID or reason);
3. the suite's summary lines at one and at four threads, verbatim;
4. every `CODE.md` change, quoted, with its marker;
5. the decisions it took, each marked (proposed);
6. anything the next step should know, and any new upstream requirement, in the format below.

## Sharp edges to know before starting

- **Five centerings, five offsets.**
  - Cell, face, edge and vertex sets each own points `G+1 … G+N` along each dimension. The shared
    high plane `G+N+1` of a staggered dimension is filled like a ghost: by the exchange, or by the
    boundary hook at a physical boundary.
  - `coordinates` takes **stored** indices.
  - `map_blocks!` adds `G` under the default and `closed = true` ranges, and adds nothing under
    `stored = true`.
  - Each kernel receives one `Val(G)` per set it reads.
- **`B` is never state.** `B_d` is recomputed from `A` (plus `𝔅̄`) in every RHS, on the stored
  range. It is not in the state vector, not in `regrid!`'s transfer, and not in a checkpoint.
- **EntropyEOS's calling conventions.**
  - Inputs are **undensitized** (divide by `√γ`). `D` is `ρ* W`, in κ-rescaled units for tables;
    κ = 1 for the analytic EOSes. A table is seen in geometric units through `GeometricEOS`
    (step 1), so no call site converts units.
  - `S_par`/`S_perp` are scalars, and `S_perp ≥ 0`. For the hydro mode pass `B² = 0`,
    `S_par = 0` and `S_perp = √(γ^{ij}S_iS_j)`.
  - The output velocity is scalar. Its direction is `Sⁱ`'s; at the converged state its magnitude
    is `|S|/(ρhW²)`, and in the ideal seed `vⁱ = (Sⁱ + (S·B)Bⁱ/z)/(z + B²)` with `z = ρhW²`. Take
    the magnitude from the returned rapidity, `|v| = tanh w`, not from that quotient: the quotient
    carries the solver's momentum residual into `atanh |v|` amplified by `W²` (7e-11 in the
    round trip's `D`), and on a repaired state it is not the repaired speed (amended in step 1).
  - The warm-start guesses of `con2prim`/`con2prim_safe` are **positional**, although the
    docstrings show keywords.
- **TreeGH's conventions.**
  - `h_ab = g_ab − η_ab` is stored as an offset, and `Π_ab = √γ n^μ∂_μ g_ab` is not `∂ₜh`.
  - Components are packed `(tt,tx,ty,tz,xx,xy,xz,yy,yz,zz)`.
  - GHSO2's `dg[a,b,c] = ∂_a g_bc`, while SpacetimeMetrics has `dg[a,b,c] = ∂_c g_ab`. Convert
    once, in `src/metric.jl`, and nowhere else.
- **`regrid!` takes one boundary hook for all sets** (until AMR-2). Also, `adapt_to_initial_data!`
  adapts one set (until AMR-3). The workarounds are in those items.
- **The IMEX implicit tendency is `(U − u★)/(γΔt)`.** The stage solver must return the `U` it
  solved for, before any limiter, and write only `E` into it: its repair of an inadmissible
  `u★` stays local. The atmosphere limiter runs after the tendency is taken.
- **The atmosphere reset is both limiters.** In IMEXRungeKutta a stage limiter's change reaches
  `uⁿ⁺¹` only through the explicit right-hand side, so pass the reset as `stage_limiter` and as
  `step_limiter`. As a stage limiter alone, it never resets the state.
- **Two abscissae.** IMEXRungeKutta calls `f_exp!` and the stage limiter at `tⁿ + c̃Δt` and
  `solve_imp!` at `tⁿ + cΔt`, and they differ (SSP3(4,3,3)'s first stage: `c̃ = 0`, `c = 0.24`).
  Only the explicit part moves `A`, so a boundary or source that depends on `t` through `A`
  is wrong in the stage solve unless it is written so that `t` drops out — hence the
  potentials' Dirichlet data in the field (added in step 6).
- **The RHS never mutates `u`.** The stage limiter writes only atmosphere cells — and *(amended in
  step 8)* the potentials' owned low walls at the outer faces of a case with Dirichlet hooks
  (AMR-8's workaround, `low_walls!`): with the reset, both limiters prescribe them, so that a
  closed box is mirror-symmetric.
- **Masks branch, never multiply**, because `0·NaN = NaN`.
- **`T_ab` must be in `Tsrc` before any TreeGH call that reads it**: the RHS, and the ADM
  monitor.

## Upstream requirements

Each item gives:
- **what**, and the **exact API**;
- **semantics**;
- **upstream acceptance**: the tests the sibling should carry;
- **downstream check**: the testset `test/prerequisite_tests.jl` gains. Until the item lands, that
  testset is skipped with an `@info` naming the ID. It detects the item by `isdefined` /
  `hasmethod`, never by version number.

**Retiring a probe** (decided by Erik 2026-10-14). Once an item has landed *and* `[compat]` requires
a release that contains it, its testset is deleted from `test/prerequisite_tests.jl` and the item
is marked "landed in <package> <version>" in the table and in its section. Probes remain only for
items still waiting. The names a retired item added join the name lists of step 0's checks (which
stay, with the metric-as-kernel-argument check and the IMEX contract checks); a property of the
release this package relies on and does not yet exercise through its own code moves to
`test/upstream_behaviour_tests.jl` until the step that uses it tests it in place.

Priorities:
- **req** blocks the step named.
- **wanted** has a workaround, listed with the item.
- **later** is not needed before rung 7.

Every item lands in a **registered release** (or a pushed commit where registration is pending),
never only in a local checkout. Additive changes stay in the same minor series (TreeAMR 0.2.x,
EntropyEOS 1.x; amended in step 0 from 0.1.x for both).

The state column is as of 2026-10-14 (added in step 0, amended 2026-10-14): for the items still
waiting, what `test/prerequisite_tests.jl` reports; for TreeGH, which the suite cannot load yet,
what its `main` reads. "Landed in" means the compat floor guarantees it and the probe is retired.

| ID | package | priority | needed by step | summary | state |
|---|---|---|---|---|---|
| AMR-1 | TreeAMR | req | 5 | state vector over several field sets | landed in TreeAMR 0.2.1 |
| AMR-2 | TreeAMR | wanted | 6 | per-set boundary hooks in `regrid!` | landed in TreeAMR 0.2.1 |
| AMR-3 | TreeAMR | wanted | 8 | `adapt_to_initial_data!` over several sets | landed in TreeAMR 0.2.1 |
| AMR-4 | TreeAMR | strongly wanted | 10 | limited conservative prolongation | landed in TreeAMR 0.2.1, as `bounded = true` |
| AMR-5 | TreeAMR | wanted | 14 | `workgroupsize` on `map_blocks!` | landed in TreeAMR 0.2.1 |
| AMR-6 | TreeAMR | later | — | device boundary hook that reads the interior (outflow) | landed in TreeAMR 0.2.1, as outflow faces |
| AMR-7 | TreeAMR | later, workaround | 14 | device boundary hook relative to the nearest owned point (added in step 6) | waiting |
| AMR-8 | TreeAMR | wanted, workaround | — | symmetric walls: the owned low wall filled like the high one (added in step 8) | waiting |
| EOS-0 | EntropyEOS | req, workaround | 0 | register in General | waiting (`[sources]`) |
| EOS-1 | EntropyEOS | req | 5 | `AbstractEOS` interface | landed in EntropyEOS 1.0.0 |
| EOS-2 | EntropyEOS | req | 5 | analytic Γ-law and hybrid EOS | landed in EntropyEOS 1.0.0 |
| EOS-3 | EntropyEOS | wanted | 12 | public fixed-entropy inversion; entropy from energy | waiting |
| EOS-4 | EntropyEOS | later | 13 | cold β-equilibrium slice | waiting |
| GH-0 | TreeGH | req | 10 | land step 8 on `main` | landed (by inspection) |
| GH-1 | TreeGH | req | 10 | public access to the vertex metric | waiting |
| GH-2 | TreeGH | req, workaround | 10 | registration; registry-satisfiable compat; Julia floor | open, see below |
| GH-3 | TreeGH | req | 11 | stress-energy input | waiting |
| GH-4 | TreeGH | req | 11 | RHS and limiters callable from an external integrator | waiting |
| GH-5 | TreeGH | wanted | 10 | ghost width `≥ q/2 + 1` | waiting |
| GH-6 | TreeGH | later | — | gauge driver, radiative boundary, excision, Ψ₄ | waiting |
| SM-1 | SpacetimeMetrics | wanted, workaround | 9 | allocation-free `ExtrinsicCurvature`, also from one `dmetric` | landed in SpacetimeMetrics 1.7.0 (workaround dropped) |

### AMR-1 — A state vector over several field sets (req, step 5)

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

**Semantics.**
- Segments are concatenated in tuple order, and each segment has exactly the single-set layout.
- So every single-set function applied to `stateview(u, sets, i)` and `sets[i]` behaves as on a
  separate vector. That covers `scatter!`, `gather!`, `statearray`, `block_mapreduce`,
  `mesh_mapreduce` and `volume_weighted_norm`.
- `statearray` and `stateview` alias `u`.
- Sets of any centering, `nvars` and `G` may be mixed.
- All sets must share:
  - one forest (`===`);
  - one backend;
  - one element type;
  - `nblocks == nleaves(forest)`.

  A violation throws an `ArgumentError` naming the set and the reason.

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
*(Done in step 5:)* `test/scheme_tests.jl`, "AMR-1 downstream", over `evolved_sets(f) = (U,
A₁, A₂, A₃, Φ̃)` on a two-level forest: each segment bit for bit the set's own state vector,
`scatter!` the inverse of `gather!`, `statearray` aliasing `u`, a segment's reduction equal to
a separate vector's, and a set over another forest refused.

### AMR-2 — Per-set boundary hooks in `regrid!` (wanted, step 6)

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

**Also** *(added in step 0)*: TreeAMR 0.2.0 refuses `fs => (schedule, hook)` today, but by a
field access on the tuple inside `check_regrid_set!` (regrid.jl:827) — a `FieldError` at Julia
1.12 and later, an `ErrorException` at 1.11 — rather than an `ArgumentError` saying which pair
forms it takes. Whichever way AMR-2 is answered, that refusal should be an `ArgumentError`.

**Upstream acceptance.** Regridding a cell set and a vertex set with different `nvars` and
different Dirichlet hooks in one call transfers exactly what two single-set calls transfer.

### AMR-3 — `adapt_to_initial_data!` over several sets (wanted, step 8)

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

### AMR-4 — Limited conservative prolongation (strongly wanted, step 10)

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

**Downstream check.** The TOV surface across a coarse-fine face floors no cell at the initial
regrid (step 10).

*(Amended 2026-10-14: landed in another form, TreeAMR 0.2.1, commit 902042a.)* Not a choice of
limiter but a property: `Operators(; prolongation=3, restriction=2, family=Conservative,
bounded=true)`, the semantics above with the slope fixed to MC (minmod lost the interface
order), `bounded = false` the default and bit-identical, `bounded = true` also accepted at
conservative order 1 (already bounded) and refused otherwise. One kernel serves the ghost fill
and the regrid. Its commit message records what upstream assumes of this package: that it
"will leave admissibility to EntropyEOS, cell by cell, prolongate the primitive variables in
the ghost fill and the conserved ones only in the regrid" — a design point steps 8 and 10 take
up (`CODE.md`, "Refinement"). `test/upstream_behaviour_tests.jl` checks, in D = 1 and 2, that a
surface (1 to 1e−13) refined everywhere stays positive and within range and keeps its total, where
the unbounded operator overshoots; it moves to step 8's refinement tests. (Until 2026-10-14 this
was the AMR-4 probe in `test/prerequisite_tests.jl`, retired when `TreeAMR = "0.2.1"` came to
guarantee it.) The step-10 downstream check stands.

### AMR-5 — `workgroupsize` on `map_blocks!` (wanted, step 14)

`map_blocks!(kernel!, fs, args...; closed=false, stored=false, workgroupsize=nothing)` passes
`workgroupsize` through to the KernelAbstractions launch; the default is unchanged. It is also on
TreeGH's list.

### AMR-6 — A device-capable boundary hook that reads the interior (later)

This is for outflow conditions. Today the region form is CPU-only (ghosts.jl:347–356).

*(Amended 2026-10-14: landed in another form, TreeAMR 0.2.1, commit a983419.)* Not a hook but an
outflow face kind, `Forest(roots; …, outflow = ((lo, hi), …))`, whose ghosts the schedule fills
as clamped transfers with the nearest owned value (zero-gradient, first order at the face), on
every backend and across ranks; a nonlinear condition (forbidding inflow) stays with the
application. Its probe, which checked that both outflow faces of a one-dimensional forest copy
their nearest cell into the ghosts, was retired on 2026-10-14 (`TreeAMR = "0.2.1"` guarantees it);
the step that first uses outflow faces tests them in place. `OutflowFace` is not exported: this
package reaches outflow through `Forest`'s `outflow` keyword only.

### AMR-7 — A device-capable boundary hook relative to the nearest owned point (later, step 14; workaround)

*(Added in step 6.)* **Why.** A non-periodic case's potentials take Dirichlet data *in the
field*: each ghost of `A_j` and `Φ̃` holds the nearest evolved point's value plus the case's
difference between the two points, so that `𝔅 = curl A` in the ghosts is the case's while the
level of `A` follows the evolution. A Dirichlet `A` itself is wrong under any tableau whose
explicit and implicit abscissae differ (`CODE.md`, "Boundaries"). That hook reads the block's
interior, which only TreeAMR's region form allows, and the region form is CPU-only
(ghosts.jl:893–911). AMR-6 as landed (outflow faces) copies the nearest owned value, which is
this with a zero difference.

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

*(Amended in step 8.)* Two more things a relative hook must do, found when AMR-8's
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

**Workaround, extended in step 8:** after each fill of a potential, `outer_points!` (a
`map_blocks!` kernel over the stored points) refills every point beyond an outer face from the
nearest evolved point, taken across a block face along the dimensions it does not leave the domain
through, now that every ghost is filled. The hook's own values stay only for the prolongations
inside the fill that read them. Marked `AMR-7`/`AMR-8` in `src/driver.jl`.

### AMR-8 — Symmetric walls: the owned low wall filled like the high one (wanted; workaround)

*(Added in step 8; decided by Erik 2026-10-15, "option 1".)* **Why.** TreeAMR's ownership is
half-open: along a vertex-like dimension a block owns `G+1 … G+N`, so at a non-periodic face the
domain's high wall plane `G+N+1` belongs to nobody and is filled like a ghost — by the hook at an
outer face, by the derivation at a reflecting one, by the clamp at an outflow one — while the low
wall `G+1` is owned and evolved (TreeAMR's `CODE.md`, "Centerings", documents and accepts this).
So `𝔅` on a box's high boundary face is the case's and on its low one the evolution's, and a closed
box is not mirror-symmetric: 1.3e−4 in a 16³ blast after three steps (measured in step 7), 1.0e−1
of a variable's scale at `t = 1/4` (step 8). The same asymmetry is in every vertex-like set of every
application with walls.

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

### EOS-0 — Register EntropyEOS (req, step 0; workaround)

Register EntropyEOS in General, as 1.x (it is at 1.0.0). Until then `Project.toml`'s
`[sources]` names `url = "https://github.com/eschnett/EntropyEOS.jl", rev = "main"`, which CI
and the clean-checkout check read. *(Amended in step 0: 0.1.x, and a `Pkg.add(url=…)` before
`instantiate`.)*

### EOS-1 — An abstract EOS interface (req, step 5)

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

**Landed** in EntropyEOS 1.0.0 *(recorded in step 0)*, except that `κ` stays unexported and is
documented as `EntropyEOS.κ(eos)`; this package calls it qualified, and the prerequisite test's
EntropyEOS name list asserts it is unexported, so an export later is noticed. The EOS-1 probe was
retired on 2026-10-14 (`EntropyEOS = "1"` guarantees it); its names joined that list.

### EOS-2 — Analytic Γ-law and hybrid EOSes (req, step 5)

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

**Downstream check.** The step-1 3+1 round trip on `IdealGasEOS` at Γ = 4/3, 5/3 and 2 (done in
step 1: `test/recovery_tests.jl`).

**Landed** in EntropyEOS 1.0.0 *(recorded in step 0)*. Its suite states the round trip with the
errors divided by EntropyEOS's conditioning factors, `κ_ρ = hW²` and
`κ_s = (1 + τ/(ρW²U))·max(1, U/(s U_s))`, below 3e−11 (ρ, w) and 1e−10 (s), rather than as a
raw 1e−12: the raw entropy error is 8.9e−10 at `ρ = 1e−8`, `W = cosh 2` on the Γ = 2 gas
(measured in step 0). Step 0's prerequisite test ran the hydrodynamic-mode call on both EOSes,
24 states, with the conditioned worst errors 2.1e−14 (ρ, w) and 1.7e−13 (s). That probe was retired
on 2026-10-14 (`EntropyEOS = "1"` guarantees it): `test/recovery_tests.jl` makes the same call
through the 3+1 layer on the Γ-law gas, and `HybridEOS` is next exercised here by the step that
first uses it.

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

### GH-0 — Land step 8 on `main` (req, step 10)

**Landed by inspection** *(step 0)*: TreeGH's `main` (fbfd416, 2026-10-09) has `accounting`,
`t_target`, the `:fitted` interior and the stage limiter, and its `CLAUDE.md` records steps
8a–8f as done. The suite cannot check it until TreeGH is a dependency.

The step-8 branches, with tip `claude/step-8f-matrix` (47 commits ahead), widen `GHProblem`,
`gh_rhs_kernel!` and `evolve!`:
- `accounting`, `target`, `fits`, `t_target`;
- the stage limiter;
- an `INT = :fitted` interior.

The items below are written against `main`. They apply to whatever `main` is when they are made.
TreeGRRMHD targets `main` only.

### GH-1 — Public access to the vertex metric (req, step 10)

**Export, with public names and docstrings:**
- `NC` (= 10).
- `unpack_sym4(v::SVector{10}) -> SMatrix{4,4}` (today `_sym4`) and
  `pack_sym10(m::SMatrix{4,4}) -> SVector{10}` (today `_pack10`), in the order
  `(tt,tx,ty,tz,xx,xy,xz,yy,yz,zz)`.
- **The layout of `U.work` as a public contract:** variables 1:10 are `h_ab = g_ab − η_ab`, and
  11:20 are `Π_ab = √γ n^μ∂_μ g_ab`, both packed.
- One exported pointwise function, preferred over exporting `axis_stencil`, `mixed_stencil`,
  `work_strides` and `point_position`:
  ```julia
  vertex_state_at(work, idx::NTuple{3}, b, inv_h, ::Val{q}) -> (h::SVector{10}, Π::SVector{10}, ∂h::NTuple{3,SVector{10}})
  ```
  It returns `h`, `Π` and the order-`q` spatial gradient at a stored vertex index, using the
  same stencils as the RHS.
- Confirm that the exported `metric_quantities`, `metric_derivatives` and `adm_vars_from_state`
  are the stable path from `(h, Π, ∂h)` to `(α, β, γ, √γ, K_ij, ∂α, ∂β, ∂γ)`.

**Downstream check.** On a Kerr-Schild background filled by `fill_exact!`, TreeGRRMHD's decoded
quantities at vertices must match SpacetimeMetrics:
- `α`, `β`, `γ_ij` and `K_ij` match `adm_decompose` / `ExtrinsicCurvature` to round-off;
- the derivatives match to finite-difference order.

### GH-2 — Registration, registry-satisfiable compat, the Julia floor (req, step 10; workaround)

- Register TreeGeneralizedHarmonic and KorzynskiSpin in General.
- Every `[compat]` entry must be satisfiable from registered releases alone. `[sources]` of a
  dependency is ignored when TreeGRRMHD resolves it, so the TreeAMR features TreeGH uses must be
  in a released TreeAMR. *(Amended in step 0:)* TreeGH 0.1.0 is now on the registered TreeAMR
  0.2, which settles that part. What is left, as of fbfd416: TreeGH itself, TreeIOHDF5 (new,
  for its checkpoints) and IMEXRungeKutta are unregistered, and its `ApparentHorizonFinder =
  "2.1"` is ahead of the registry, which has 2.0.0. KorzynskiSpin 1.1.0 and
  AbstractSphericalHarmonics are registered.
- **Julia 1.11.** Settled on 2026-09-25: every Tree* package's floor is 1.11, so TreeGH's pin
  to 1.11 no longer constrains TreeGRRMHD, and TreeGH is an ordinary dependency rather than a
  weak one behind `TreeGRRMHDTreeGHExt` (CODE.md "Julia version"). *(Amended 2026-10-15:)*
  TreeGRRMHD's floor is now 1.13, above TreeGH's 1.11, which still does not constrain it;
  nothing is asked of TreeGH.
- **Workaround until registered:** `[sources]` names TreeGH, TreeIOHDF5 and, if still needed,
  ApparentHorizonFinder by URL (amended in step 0: was "CI adds TreeGH and KorzynskiSpin").

### GH-3 — Stress-energy input (req, step 11)

**API.**
```julia
GHProblem(U, schedule, case; q, t, interior, margin_check, …, Tsrc::Union{Nothing,FieldSet}=nothing)
```

**The `Tsrc` field set.**
- `vertexcentered(3)`, `nvars == 10`, `G == 0`, on the same forest (`===`) and backend as `U`.
  Otherwise `ArgumentError`.
- Content: the covariant coordinate components `T_ab`, in `G = c = 1` and the metric's length
  unit, packed as in `NC`, at the owned vertices.
- The caller writes it before each RHS evaluation. TreeGH never writes it and never ghost-fills
  it.
- It adds a sixth `Val`, `Val(HASMATTER)` with `HASMATTER = Tsrc !== nothing`. With `false`, the
  kernel is bit-identical to today.

**Kernel** (evolution.jl:292–293 on `main`). With `Tl = unpack_sym4(Tsrc at inner)` and
`trT = Σ gu4 .* Tl`:
```
msrc += 16π · α · sqrtγ · pack_sym10(Tl − g4 · trT / 2)
```
- `∂ₜh` is unchanged. Inside an interior layer the term is weighted by `w`, like the rest of `F`.
- The coefficient follows from notes/formulation.md §1–2. `R^H_ab = 8π(T_ab − ½g_abT)` makes
  `S0 → S0 − 16πT̄_ab`, and the source enters as `−α√γ S0`.

**Monitors.**
- `adm_constraint!` reports `ℋ − 16πρ_ADM` and `ℳ_i − 8πS_i`, with `ρ_ADM = n^a n^b T_ab`,
  `S_i = −n^a T_ai` and `n^a = (1, −β^i)/α`.
- The `∂ₜ∂ₜg` reconstruction inside `adm_constraint_kernel!` (constraints.jl:388) includes the
  matter term.
- `gh_constraint!` is unchanged.

**Lifecycle.** `with_interior` and every rebuild carry `Tsrc`; `regrid!` treats it as
`=> nothing`, and the caller rebuilds it.

**Upstream acceptance.**
- `HASMATTER = false` is bit-identical to today.
- A static non-vacuum test metric: take an `isbits` test metric with a smooth compact mass
  profile, set `T_ab := G_ab[g]/8π` from SpacetimeMetrics' Einstein tensor sampled into `Tsrc`,
  and freeze `H = −Γ[g]`.
  - The RHS then vanishes at order `q` under refinement.
  - The matter-corrected `ℋ` and `ℳ_i` vanish at order `q`, and do not without the matter
    terms.

**Downstream check.** The same test with TreeGRRMHD's `T_ab` from a TOV star (step 11).

### GH-4 — RHS and limiters callable from an external integrator (req, step 11)

**API.**
```julia
gh_fill_ghosts!(p::GHProblem, t)                       # the fill gh_rhs! does today
gh_rhs_prepared!(du::AbstractVector, p::GHProblem, t)  # kernel launch only; precondition:
                                                       # p.U.work scattered and ghost-filled at t
gh_rhs!(du, u, p, t)   # = scatter!(p.U, u); gh_fill_ghosts!(p, t); gh_rhs_prepared!(du, p, t)
gh_max_speed(p::GHProblem, u::AbstractVector)          # today's unexported max_speed_of
gh_step_limit!(u::AbstractVector, p, t)                # integrator-free gh_step_limiter!
gh_stage_limit!(u::AbstractVector, p, t)               # integrator-free gh_stage_limiter!
```

**Semantics.**
- `u` and `du` may be contiguous segment views (`stateview`, AMR-1).
- `evolve!` should be a composition of exported functions. Name any piece that is still inlined:
  - the per-chunk `dt`/steps;
  - the rebuild after `regrid!`;
  - the buffer width;
  - the initial fill;
  - the record.

  Several are already exported: `indicator_flags`, `GHProblem`, `fill_exact!`, `dirichlet`,
  `constraint_norms` and `adm_constraint!`.

**Upstream acceptance.**
- `gh_rhs!` is bit-identical to today.
- `gh_rhs_prepared!` on a segment of a two-set state vector equals `gh_rhs!` on a standalone
  vector.
- The limiters on a segment equal the SciML-hook versions.

### GH-5 — Ghost width `≥ q/2 + 1` (wanted, step 10)

- Relax `all(g -> g == q ÷ 2 + 1, U.G)` (evolution.jl:540–546) to `≥`.
- Add a `G` keyword to `evolve!` (driver.jl:243).

The survey found every kernel offsets by `Val(G)` and every stencil reach comes from the weights,
so this is a check change plus a test. It lets a 4th-order vertex→cell interpolation reach every
stored cell of a `G = 3` matter set.

### GH-6 — Later

A damped-harmonic gauge driver, radiative or constraint-preserving outer boundaries, excision or
a generalized interior for collapse, and Ψ₄.

### SM-1 — Allocation-free extrinsic curvature (wanted, step 9; workaround)

*(Added in step 2.)* The `Analytic` provider needs `K_ij` at every cell of every right-hand side,
inside a kernel. SpacetimeMetrics 1.6's `ExtrinsicCurvature(m, p)` allocates 432 bytes per call
(Julia 1.11 and 1.13, Kerr-Schild), which a device kernel cannot do, and it differentiates the
metric a second time where the provider already holds one `dmetric` result.

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

**Downstream check** (retired 2026-10-14, see below). `test/prerequisite_tests.jl`, "SM-1": the
item has landed when the new method
exists (`hasmethod`) and the old one allocates nothing (`@allocated`); then the upstream `K_ij`
must agree with this package's `extrinsic_curvature` to 1e-13 of `max |K|` on boosted Kerr-Schild.

**Workaround** (amended in step 2): `extrinsic_curvature(α, β, sdg)` in `src/metric.jl`, marked
`SM-1`, computes `K_ij` from the provider's own `dmetric` result. It agrees with upstream's to
4.3e−16 (measured in step 2). When SM-1 lands, `metric_gradient(::Analytic, …)` calls
`ExtrinsicCurvature(g, sdg)` and the local function goes. *(Landed 2026-10-14, SpacetimeMetrics
1.7.0:)* done; the compat is `SpacetimeMetrics = "1.7"` (which requires Julia 1.11, the floor),
and the prerequisite test compares upstream's `K_ij` with the former workaround's formula, kept in
the test, and checks that both methods allocate nothing. *(Probe retired 2026-10-14,
`SpacetimeMetrics = "1.7"` guaranteeing it:)* the comparison with the formula moved into
`test/pointwise_tests.jl`'s metric-provider testset (upstream's two methods agree bit for bit, so
comparing them checks nothing independently; 3.7e−16 of `max |K|`), and step 2's kernel test already
asserts that `metric_gradient`, which calls the new method, allocates nothing.

## Step 0 — Scaffolding

*(Done, 2026-10-13, on `claude/step-0-scaffolding`.)*

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

## Step 1 — Units, variables and the 3+1 primitive recovery

*(Done, 2026-10-13, on `claude/step-1-recovery`.)*

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

## Step 2 — Pointwise physics

*(Done, 2026-10-13, on `claude/step-2-pointwise`.)*

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

## Step 3 — The implicit stage solve

*(Done, 2026-10-13, on `claude/step-3-stage-solve`.)*

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

*(Amended in step 3 after review; decided by Erik 2026-10-14:)* the fallback rate was not
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

## Steps 4a–4c — The IMEX package (in its own repository)

These steps are done in `~/src/jl/IMEXRungeKutta`, following its `CODE.md`, its own plan, and
its `CLAUDE.md` for the mechanics. They are independent of steps 1–3 and may run before them.
Here, 4a (tableaus), 4b (the integrator) and 4c (validation) name that package's milestones.
*(Amended 2026-09-24: the steps' contents moved to that repository.)*

What TreeGRRMHD needs from it:
- **4b**, which step 5 waits for: the integrator with the contract of `CODE.md`'s "The IMEX
  package", over TreeAMR's flat state vector.
- **4c:** SSP3(4,3,3) and SSP2(2,2,2), validated, with SSP3(4,3,3)'s L-stability and
  stiff-limit order recorded. Amend `CODE.md`'s "The partition" with them.
- A tag (v0.1.0). Registration is Erik's call; until then the package is added by URL.

*(Amended in step 0:)* done. IMEXRungeKutta is at v1.4.0, with `IMEXSSP3433`, `IMEXSSP222`,
both limiters and the `solve_imp!` contract; `CODE.md`'s "The partition" carries its
measurements. It is a dependency from step 0, through `[sources]`.

## Step 5 — The staggered scheme in 1D: smooth problems (rung 1a)

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
  SSP3(4,3,3) on 400 cells is 5.2 ms at one thread (Apple M3 Pro, 1.13). The suite is 24638
  tests, 1m31s at 1.13 and 1m45s at 1.11 (`Test Summary`, one thread; 24292 and 40 s before the
  step).

## Step 6 — Shock tubes and the stiff limit in 1D (rung 1b)

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
- The suite is 24882 tests: `Test Summary` 2m42s at 1.13 and 3m25s at 1.11 at one thread,
  2m55s and 3m03s at four, 2m57s with `--check-bounds=yes` (1.13); 24638 and 1m31s before the
  step. Every printed number agrees across versions and thread counts.
- **Atmosphere:** a shock tube with a 10⁸ density ratio exercises the limiter.

## Step 7 — 2D and 3D, and higher-order reconstruction (rung 2a)

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
- The suite is 25293 tests: `Test Summary` 5m23s at 1.13 and 6m43s at 1.11 at one thread,
  7m17s and 8m14s at four, 5m35s with `--check-bounds=yes` (1.13) — on a machine at a load of
  8–17 on 12 cores, two suites running at once; 24882 tests and 2m42s (1.13, one thread)
  before the step. The new files, alone at one thread: `multid_tests.jl` 67 s (35 s of it
  compiling the 2D and 3D kernels), `multid_cases_tests.jl` 60 s, `reconstruction_tests.jl`
  35 s; `shocktube_tests.jl` grew to 47 s. Every printed number is identical at one and four
  threads; 1.11 and `--check-bounds=yes` agree with each other exactly and differ from 1.13
  only in the last one or two digits of L1 errors and conservation drifts (Orszag–Tang's
  0.056 ulp a step is 0.037 there).

## Step 8 — Coarse-fine faces and regridding (rung 2b)

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

## Step 9 — Analytic curved backgrounds (rung 3)

`CODE.md`: "Metric providers", "Validation ladder" 3. Waits: none.

Changes:
- **The `Analytic` provider** at cells, faces, edges and vertices.
- **Kerr-Schild cases:** the Wald solution (`Vacuum`), and the Michel monopole or magnetized Bondi
  flow (`IdealConductor`).
- **The interior:** inside `r < r_in` (within the horizon), the state is relaxed to the analytic
  solution each stage. This is position-only, as TreeGH's interior is **(proposed)**.

Accept:
- **Stationarity:** for both cases, the L1 drift from the exact stationary solution after
  `t = 50M` converges at ≥ 1.8 **(predicted)**.
- **Charge:** `|q|` from `∇·E` is recorded.
- **∇·B** at round-off.

## Step 10 — The vertex metric and the TOV star in the Cowling approximation (rung 4)

`CODE.md`: "Metric at cells, faces and edges", "Gauge for the TOV milestones", "Floors, the
atmosphere and the magnetosphere", "Refinement", "Riemann solver".
Waits: **GH-0, GH-1, GH-2** (or the extension route). Also AMR-4 (strongly wanted), GH-5 and
AMR-3 (wanted).

Changes:
- **`src/tov.jl`:** the host TOV solve in isotropic coordinates, for the polytrope K = 100,
  Γ = 2, ρ_c = 1.28e−3 (`IdealGasEOS` at `polytropic_entropy`).
- **`TOVMetric <: SpacetimeMetrics.AbstractMetric`:** `isbits` and dual-generic. The step decides
  its storage (a fixed-size spline, or an Adapt-ed device array) and records it.
- **`src/vertex_metric.jl`:** the frozen `GHMetric` provider.
  - It is a vertex set in TreeGH's layout, filled via `background_state`.
  - It provides the 64-, 16- and 4-point interpolations, `∂g` at cells, and `K_ij`.
  - It **decides the outermost-ghost-cell option**: GH-5, a 2nd-order layer, or an exchange.
- **TreeGH** becomes a dependency or an extension, per GH-2.
- **`src/cases/tov.jl`**, with geometric refinement boxes.

Accept:
- **Interpolation:** the cell, face and edge metric, `∂g` and `K_ij` converge at 4th order
  against the analytic metric.
- **Stationarity:** the central-density drift after 200 M converges at ≥ 1.5 **(predicted)**.
- **Conservation:** rest mass to round-off, net of the outer boundary flux.
- **Modes:** the F, H1 and H2 frequencies are within 3% of Font et al. (2002)'s Cowling values
  **(predicted; values copied with citation)**.
- **Floored cells** at a coarse-fine surface for width 1, width 3, and limited (AMR-4)
  **decide the prolongation (open)**.
- **Light-cone against split speeds** on the star **decides the Riemann speeds (open)**.

## Step 11 — The TOV star in a dynamical spacetime (rung 5)

`CODE.md`: "Spacetime coupling", "Time integration". Waits: **GH-3, GH-4**.

Changes:
- **The coupled problem:**
  - TreeGH's `U` joins the state vector;
  - a `T_ab` kernel fills `Tsrc`, 2nd order, with a 4th-order option;
  - `gh_fill_ghosts!` and `gh_rhs_prepared!` go into the explicit part.
- **The coupled driver:**
  - `Δt = min(GH, matter)`;
  - regrid of every set;
  - rebuild of `GHProblem` with `Tsrc`.
- **The GH CFL factor under SSP3(4,3,3)'s explicit part**, measured here. It replaces the former
  upstream item.

Accept:
- **GH-3's static-matter check** passes with TreeGRRMHD's `T_ab`.
- **Constraints:** the matter-corrected ADM constraints converge at ≥ 1.8 away from the surface
  **(predicted)**.
- **Modes:** F, H1 and H2 within 3% of Font et al. (2002)'s coupled values **(predicted)**.
- **Conservation:** rest mass to round-off.
- **Decides:** the largest stable GH `cfl` under SSP3, on the gauge wave and on the TOV star,
  which becomes the default; and **the `T_ab` transfer order (open)**.

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
- Update `README.md`, and `CLAUDE.md`'s "Current state" and "Commands".
- Move anything still marked **(predicted)** or **(open)** to measured, or to "Possible
  extensions".
- Delete this file.
