# TreeGRRMHD implementation plan

For the sessions that implement TreeGRRMHD, one step at a time. The **design** lives in
`CODE.md`. Read it first, in full: it is authoritative. This file is only the work breakdown:
- what each step changes, and what it must not change;
- what each step must measure and record;
- and, because this package stands on siblings that are still growing, exactly what it needs
  from them.

`CLAUDE.md` has the mechanics. Delete this file when the last rung is marked *(Done.)* in
`CODE.md`.

**Steps 0 and 1 are done; step 2 is next.** *(Amended in step 1, 2026-10-13.)* Steps 2 and 3
need nothing new from upstream. Steps 4a–4c are done in the IMEX package's own repository: IMEXRungeKutta
1.4.0 is tagged, unregistered, and a dependency from step 0. EOS-1 and EOS-2 have landed in
EntropyEOS 1.0.0, so step 5, the first evolution, waits for AMR-1 alone. TreeAMR is at 0.2.0
(registered), and none of AMR-1 to AMR-6 has landed. The "state" column of the table below has
each item.

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
  - Unregistered ones are named by URL in `Project.toml`'s `[sources]`, which Julia 1.11 — the
    floor since 2026-09-25 — reads (CODE.md "Julia version"). That covers EntropyEOS until
    EOS-0 and the IMEX package, and will cover TreeGH and its unregistered dependencies until
    GH-2. CI, the clean-checkout check and a developer all resolve them the same way, so CI
    adds nothing by hand. *(Amended in step 0: this used to be `Pkg.add(url=…)` in CI.)*
- **Julia 1.11 floor.** Be generic in `T` and in the backend from the first line:
  - `T` is a leading positional argument (default `Float64`); `backend` is a keyword (default
    `CPU()`).
  - No float literals where `T` is in play; every callback is `isbits`; per-block metadata goes
    through `to_backend`.
  - Physics is validated in `Float64`; `Float32` is a smoke test.
- **Tests are small.**
  - Sizes: 1D up to 400 cells, 2D up to 128², 3D with `N = 8` blocks on a few roots.
  - Time: under about 60 s per test file, and the whole suite under about 10 min at one thread.
- Before and after each step, run the full suite at the default thread count, at
  `julia_args = ["--threads=4"]`, and once with `--check-bounds=yes`.
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
- **The RHS never mutates `u`.** The stage limiter writes only atmosphere cells.
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

Priorities:
- **req** blocks the step named.
- **wanted** has a workaround, listed with the item.
- **later** is not needed before rung 7.

Every item lands in a **registered release** (or a pushed commit where registration is pending),
never only in a local checkout. Additive changes stay in the same minor series (TreeAMR 0.2.x,
EntropyEOS 1.x; amended in step 0 from 0.1.x for both).

The state column is as of 2026-10-13 (added in step 0): what `test/prerequisite_tests.jl`
reports, and for TreeGH, which the suite cannot load yet, what its `main` reads.

| ID | package | priority | needed by step | summary | state |
|---|---|---|---|---|---|
| AMR-1 | TreeAMR | req | 5 | state vector over several field sets | waiting |
| AMR-2 | TreeAMR | wanted | 6 | per-set boundary hooks in `regrid!` | waiting |
| AMR-3 | TreeAMR | wanted | 8 | `adapt_to_initial_data!` over several sets | waiting |
| AMR-4 | TreeAMR | strongly wanted | 10 | limited conservative prolongation | waiting |
| AMR-5 | TreeAMR | wanted | 14 | `workgroupsize` on `map_blocks!` | waiting |
| AMR-6 | TreeAMR | later | — | device boundary hook that reads the interior (outflow) | waiting |
| EOS-0 | EntropyEOS | req, workaround | 0 | register in General | waiting (`[sources]`) |
| EOS-1 | EntropyEOS | req | 5 | `AbstractEOS` interface | landed, 1.0.0 |
| EOS-2 | EntropyEOS | req | 5 | analytic Γ-law and hybrid EOS | landed, 1.0.0 |
| EOS-3 | EntropyEOS | wanted | 12 | public fixed-entropy inversion; entropy from energy | waiting |
| EOS-4 | EntropyEOS | later | 13 | cold β-equilibrium slice | waiting |
| GH-0 | TreeGH | req | 10 | land step 8 on `main` | landed (by inspection) |
| GH-1 | TreeGH | req | 10 | public access to the vertex metric | waiting |
| GH-2 | TreeGH | req, workaround | 10 | registration; registry-satisfiable compat; Julia floor | open, see below |
| GH-3 | TreeGH | req | 11 | stress-energy input | waiting |
| GH-4 | TreeGH | req | 11 | RHS and limiters callable from an external integrator | waiting |
| GH-5 | TreeGH | wanted | 10 | ghost width `≥ q/2 + 1` | waiting |
| GH-6 | TreeGH | later | — | gauge driver, radiative boundary, excision, Ψ₄ | waiting |

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

### AMR-5 — `workgroupsize` on `map_blocks!` (wanted, step 14)

`map_blocks!(kernel!, fs, args...; closed=false, stored=false, workgroupsize=nothing)` passes
`workgroupsize` through to the KernelAbstractions launch; the default is unchanged. It is also on
TreeGH's list.

### AMR-6 — A device-capable boundary hook that reads the interior (later)

This is for outflow conditions. Today the region form is CPU-only (ghosts.jl:347–356).

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
documented as `EntropyEOS.κ(eos)`; this package calls it qualified, and the prerequisite test
asserts it is unexported, so an export later is noticed.

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
(measured in step 0). Step 0's prerequisite test runs the hydrodynamic-mode call on both EOSes,
24 states, with the conditioned worst errors 2.1e−14 (ρ, w) and 1.7e−13 (s).

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
  weak one behind `TreeGRRMHDTreeGHExt` (CODE.md "Julia version").
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
  input of the recovery and is checked through `s` (proposed in step 1). Measured 9.8e−13: 1e−12
  is `con2prim`'s own tolerance, so the bound has no margin.
  - `W ≤ 10`;
  - random metrics, from Kerr-Schild at random points and random boosts;
  - random E and B with EM energy up to 100× the fluid's.
- `con2prim_ideal` recovers `v` when `E = −v×B`.
- The units adapter round-trips, and `M☉`'s density unit equals 6.176e17 g/cm³ to 1e−4.
- The atmosphere flags are exercised.
- Everything is callable from a CPU kernel.

## Step 2 — Pointwise physics

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

## Step 3 — The implicit stage solve

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
Waits: **AMR-1** (EOS-1, EOS-2 and 4b have landed; amended in step 0).

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

## Step 6 — Shock tubes and the stiff limit in 1D (rung 1b)

`CODE.md`: "Validation ladder" 1, "Floors, the atmosphere and the magnetosphere".
Waits: AMR-2 (workaround allowed).

Changes:
- **`src/cases/shocktubes.jl`:**
  - Balsara 1–5 (GRaM-X Table I);
  - the resistive shock tube swept over σ (Komissarov 2007);
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
