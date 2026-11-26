# TreeGRRMHD implementation plan: neutrinos

For the sessions that implement the neutrinos, one step at a time. The
**design** is `CODE.md`, "Neutrinos": read it first, in full; it is
authoritative. This file is only the work breakdown: what each step
changes and must not change, what it must measure and record, and what it
waits for upstream. `CLAUDE.md` has the mechanics. When rung 8 is marked
*(Done.)* in `CODE.md`, this file's steps move to `HISTORY.md`, "The steps
of the plan", and the file is deleted, as the first plan's was in step 15.

The steps continue the first plan's numbering (steps 0–15 are in
`HISTORY.md`, "The steps of the plan") and follow rung 8 of `CODE.md`'s
validation ladder, 8a to 8g. Each ends in a green suite and a `CODE.md`
update, and each is a brief that one agent with a fresh context can carry.

**Contents**

- [Ground rules, every step](#ground-rules-every-step)
- [Running a step as an agent](#running-a-step-as-an-agent)
- [Upstream requirements](#upstream-requirements)
- [Step 16 — Rates and opacities, pointwise (rung 8a)](#step-16--rates-and-opacities-pointwise-rung-8a)
- [Step 17 — The optical depth (rung 8b)](#step-17--the-optical-depth-rung-8b)
- [Step 18 — M0 transport, without feedback (rungs 8c, 8d)](#step-18--m0-transport-without-feedback-rungs-8c-8d)
- [Step 19 — Coupling to the matter (rungs 8e, 8f)](#step-19--coupling-to-the-matter-rungs-8e-8f)
- [Step 20 — The star with neutrinos (rung 8g)](#step-20--the-star-with-neutrinos-rung-8g)
- [Step 21 — Neutrinos on MPI ranks](#step-21--neutrinos-on-mpi-ranks)
- [Step 22 — Review pass](#step-22--review-pass)

## Ground rules, every step

- Read `CLAUDE.md`, then this file's ground rules and the step, then
  `CODE.md`'s "Sharp edges" and "Neutrinos", then the other `CODE.md`
  sections the step names.
- **Work on a branch** `claude/step-N-<slug>` off `main`. Commit in
  TreeAMR's style: implement, then measure, then record, with the measured
  numbers in the commit body. **Do not merge and do not push**; report the
  branch and its commits.
- **Never edit a sibling checkout.** If an upstream item has not landed, do
  everything that does not depend on it and report the step as blocked on
  its ID. A need that is not listed goes in `CODE.md`'s "Requests still
  waiting", in its format, and in the report. No workaround that the
  request does not list.
- **A run without neutrinos is today's, bit for bit.** Every step keeps
  `neutrinos = nothing` the default and asserts that the existing suite's
  results do not change; a step that cannot keep this stops and reports.
- Generic in `T` and the backend from the first line; every callback and
  model `isbits`; no float literals where `T` is in play. Physics in
  `Float64`; `Float32` a smoke test.
- **Tests are small** (`CLAUDE.md`, "Conventions", "Tests"); each new
  testset goes in a tier, and a run several testsets read is made by
  `computed_once`. Studies at production size are scripts in `bin/`, run as
  SimLaunch specs in `jobs/`.
- Before and after each step, the full suite three ways at the floor, and
  the clean-checkout check (`CLAUDE.md`, "Commands").
- **Spec-first.** Where the step shows `CODE.md` wrong or incomplete, fix
  it to describe the current state and record in `HISTORY.md` what changed,
  when and why. Decide each **(open)** the step owns. Never loosen an
  assertion to get green; report the failure.
- One step at a time; the next starts from a green suite on `main`.

## Running a step as an agent

As in the first plan: one implementation agent per step, in its own
worktree, at most one at a time. The agent does not ask questions mid-step:
where it would have asked, it decides, marks the decision **(proposed)** in
`CODE.md`, records in `HISTORY.md` that step N proposed it, and lists it in
the report. The reviewer runs the suite at one and four threads, reads the
`CODE.md` diff, walks the acceptance list against the report, and merges
only then; a failed item goes back to the same agent.

**The brief** (replace `N`):

    You are implementing step N of PLAN.md in the Julia package at <path>.
    Read CLAUDE.md, then PLAN.md's "Ground rules", "Upstream requirements"
    and step N, then CODE.md's "Sharp edges", "Neutrinos" and the sections
    step N names. Work on the branch claude/step-N-<slug> in this worktree;
    commit in TreeAMR's style with measured numbers in the commit body; do
    not merge, push, or touch TODO.md or any sibling checkout. Do all of
    step N; if part of it is blocked, finish everything else and say
    exactly what is blocked, by which ID. Never loosen an assertion to get
    green. When you would ask a question, decide, mark it "(proposed)" in
    CODE.md, record in HISTORY.md that step N proposed it, and list it in
    the report. End with the report PLAN.md specifies.

**The report**, in this order: the branch and its commits, one line each;
the acceptance list, each item *done*, *measured* (with the number) or
*blocked* (with the ID); the suite's summary lines at one and four threads,
verbatim; every `CODE.md` change, quoted, with its `HISTORY.md` entry; the
decisions taken, each (proposed); anything the next step should know.

## Upstream requirements

One item, stated in full in `CODE.md`, "Requests still waiting":

| ID | what | needed by | state |
|---|---|---|---|
| EOS-12 | the composition on the device | step 16, step 20 | landed, EntropyEOS 2.4.0 (registered; `[compat]` `"2.4"`) |
| EOS-13 | the nucleons' own `μ_n`, `μ_p` in `Composition` | step 20 | withdrawn: the scattering blocking reads the free gas's degeneracy, from the density |

Nothing upstream waits for any step: `RJS96Rates` reads only EOS-12's
composition.

## Step 16 — Rates and opacities, pointwise (rung 8a)

*Built* (2026-11-25, on `claude/step-16-neutrino-rates`): the Fermi integrals, both
rate models, the effective rates, the table form refusing another table's
composition (EOS-12, landed in EntropyEOS 2.4.0) and their tests; `HISTORY.md`,
"The steps of the plan", has what changed from this brief. The DD2 values move
to step 20; the mass `X_l` holds is EntropyEOS's measurement (34%). The
scattering blocking was later changed to the free gas's, from the density, and
EOS-13 withdrawn (`HISTORY.md`, "EntropyEOS").

`CODE.md`: "Neutrinos" ("Leakage"), "Thermodynamics", "Precision and
devices". Waits: EOS-12 for the DD2 reference values only.

Changes:
- `src/neutrinos/fermi.jl`: `F_k(η)`, `k = 0 … 5`; decide the fits, which
  is the **(open)** this step owns, by their accuracy against quadrature
  and their cost.
- `src/neutrinos/rates.jl`: the `Composition` type (EntropyEOS's, once
  EOS-12 lands; until then a local `isbits` stand-in with the same ten
  fields and EOS-12's conventions, `μ_ν = μ_e − μ̂` and `μ̂` including `Q`,
  marked with EOS-12), `RJS96Rates` and `GrayRates`, each returning per
  species `R`, `Q`, `κ^α_a`, `κ^α_s`, `n^β`, `e^β`, `E^β` in code units; the
  references in the docstrings.
- `src/neutrinos/leakage.jl`: `R^eff`, `Q^eff` from the rates and `τ^α`.
- Two decisions on EOS-12's side of the rates: confirm against RJS96's
  appendix that the Pauli-blocking factors take `(μ̂ − Q)/T`, the one place
  `Q` is subtracted; and decide whether `Y_np` takes `2Yₑ − 1`, as for free
  nucleons, or `X_p − X_n`, where nuclei and α hold much of the matter —
  the **(open)** of that name, decided by the rates' change on DD2 between
  the two at `log ρ` 11–13 (and its limit at `η̂ → 0`, a branch).
- `bin/neutrino_reference.jl`, writing `test/data/` reference values;
  `test/neutrino_rates_tests.jl`; the EOS-12 probe.

Accept:
- The Fermi integrals within the chosen fits' stated accuracy of
  quadrature over `η ∈ [−50, 150]`, the error asserted as a number.
- The rates against the reference values to a stated tolerance; the two
  limits of `R^eff` (`τ → 0`, `τ → ∞`) and the branch at `R = 0`.
- Allocation-free in a CPU kernel at `Float64` and `Float32`, and under
  `--check-bounds=yes`.
- The mass `X_l` holds where the rates ignore it, stated as a number on
  DD2.
- With EOS-12 landed: the DD2 values through `composition`, and
  `M0Leakage` refusing a composition that does not `match` its EOS; else
  that half reported as blocked on EOS-12.

## Step 17 — The optical depth (rung 8b)

*Built* (2026-11-25, on `claude/step-17-optical-depth`): `OpticalDepth` and its tests;
`HISTORY.md`, "The steps of the plan", has what changed from this brief — after a
regrid τ is transferred and not reconverged, and `Nτ` has one ghost layer.

`CODE.md`: "Neutrinos" ("The optical depth", "Field sets and interface"),
"Field sets", "Refinement". Waits: none.

Changes:
- The `Nτ` set; `src/neutrinos/optical_depth.jl`: `κ` into `Nτ`, the sweep
  over `3^D − 1` neighbours, the outer faces' `τ = κ = 0`, the sweeps to
  convergence at the initial data and after a regrid, `τ`'s transfer at
  `regrid!`.
- `test/optical_depth_tests.jl`.

Accept:
- A uniform-`κ` sphere (`GrayRates`) against `κ (R − r)`: the overestimate
  along the diagonals measured and asserted, and the face-neighbour
  stencil's `√D` shown for contrast in the test.
- The sweeps to convergence equal to the cells across, to within a stated
  margin; one sweep per step following a sphere that moves at `v = 0.1`.
- `τ` across a coarse-fine face and through a regrid within a stated
  tolerance of the uniform mesh's.

## Step 18 — M0 transport, without feedback (rungs 8c, 8d)

*Built* (2026-11-25, on `claude/step-18-m0-transport`): M0 on the forest and its tests;
`HISTORY.md`, "The steps of the plan", has what changed from this brief — periodic forests
refused, the luminosity tally moved to step 20, 8d on the Cowling star.

`CODE.md`: "Neutrinos" ("M0", "In the time integrator", "Field sets and
interface"), "Reconstruction", "Riemann solver", "The partition",
"Conservation at coarse-fine faces", "Boundaries". Waits: none.

Changes:
- The `Nν` and `Fν_d` sets; `M0Leakage(rates; center, r_core, feedback)`
  and `MHDProblem`'s `neutrinos` keyword; `src/neutrinos/m0.jl`: `k^α`, `χ`,
  `r_core`, minmod reconstruction, the upwind flux, emission, the
  closed-form implicit absorption at the start of each implicit stage, the
  outer faces, the luminosity and number flux into `X`.
- Decide `N_min`, the **(open)** this step shares with step 19.
- `test/m0_tests.jl`; a `bin/` study for the finer resolutions.

Accept:
- The uniform sphere's steady state against `(r² n)′ + κ r² n = η r²`,
  converging at 1.8 or better away from the surface; the front at `r = t`.
- The spread across rays measured on a beam (a narrow emitting region off
  the centre), stated as a number.
- 8d: on Schwarzschild (Kerr-Schild, `a = 0`), `𝓔/N` constant along rays
  to a stated tolerance, and the mean energy at infinity `χ` times the
  emitted one.
- `N`, `𝓔 ≥ 0` everywhere; `Σ N` and `Σ 𝓔` conserved to round-off net of
  emission, absorption and the outer faces, across coarse-fine faces and a
  regrid.
- A run with `neutrinos = nothing` is the existing suite's, bit for bit.

## Step 19 — Coupling to the matter (rungs 8e, 8f)

*Built* (2026-11-25, on `claude/step-19-neutrino-coupling`): the feedback and its tests;
`HISTORY.md`, "The steps of the plan", has what changed from this brief — 8f against the
coupled problem's closed form in a uniform medium, since M0 refuses a periodic box.

`CODE.md`: "Neutrinos" ("Coupling to the matter", "In the time
integrator"), "Sources", "The implicit stage solve", "Floors, the
atmosphere and the magnetosphere". Waits: none.

Changes:
- The matter's sources: emission explicit, in the source kernel, skipped
  in atmosphere cells; the matter's share of the absorption in the implicit
  stage, from the solved `N`, `𝓔`, before Ohm's solve; `R^eff Δt/n_e` per
  chunk into `X`.
- `test/neutrino_coupling_tests.jl`.

Accept:
- 8e: `√γ D Yₑ + N_{ν_e} − N_{ν̄_e}`, and in flat space the matter's energy
  plus `Σ 𝓔`, to round-off net of the outer faces and the reset, across
  coarse-fine faces and a regrid, in 2D and 3D.
- 8f: a uniform hot periodic box (`GrayRates`), its `Yₑ`, `T`, `N` and `𝓔`
  against the ODE integrated to tolerance, through the approach to
  `R^eff = κ^eff n`, at the order the frozen coefficients allow (measured
  and stated).
- The stage solve's outcome counts unchanged on the existing cases.

## Step 20 — The star with neutrinos (rung 8g)

`CODE.md`: "Neutrinos" (all of it, "Field sets and interface" for the
composition and the checkpoint), "Thermodynamics", "Validation ladder" 7
and 8g, "The per-phase table", "Devices". Waits: nothing upstream.

Decided with Erik (2026-11-26): the composition travels with `M0Leakage`;
`T` from the warm start; the hot star a hot barotrope in hydrostatic
equilibrium; the luminosity through a spherical shell inside the domain, the
outer faces' outflow only the computational check; checkpoints within this
step, before the long studies; MPI a step of its own (21).

**20a, the composition in the kernels.**
- `M0Leakage(rates; composition, …)`, refused as `CODE.md` says;
  `TabulatedRates` into the kernels; `RJS96Rates` gains the units' density.
- `cell_rates` and the optical depth's opacities from `P`'s `ρ★`,
  `exp10(u_solved)` and `Yₑ`, zero by a branch where `ρ★ ≤ 0` or the warm
  start is not finite.
- Accept: on the synthetic table, a cell's rates in the kernel equal
  `neutrino_rates` at its state bit for bit; the refusals; the kernels
  allocation-free and inferred at `Float32` and `Float64`.

**20b, checkpoints with neutrinos.**
- `save_run` writes `Nν`'s segment and `Nτ`; `restore_run!` reads them;
  the recipe holds the `M0Leakage`. `evolve!` refuses neutrinos only with
  `comm`.
- Accept: a run with neutrinos (`GrayRates`, a static mesh, and one regrid
  of an adaptive one) restarted from its checkpoint is the run that never
  stopped, bit for bit; a restart with another `M0Leakage` is refused.

**20c, the stars.**
- `hot_barotrope(slice, eos; s, ρ_surface)` beside `cold_barotrope`;
  `tabulated_tov_case(…; s)` for the hot star (cold without `s`).
- `neutrino_luminosity(p, u, t; r_L, width)`.
- In the suite, DD2-gated, on the `h = 1` star to `t` of a few M: lepton
  number (matter plus neutrinos) conserved net of the outflow to round-off;
  the cold star's neutrino totals below a measured bound; the hot star's
  emission, its central temperature and the sign of its `dYₑ/dt` as
  measured numbers. The shell's luminosity of a uniform emitting medium
  against its closed form (`GrayRates`, no table).
- Accept: the testsets as stated, each in its tier.

**20d, the studies.**
- `bin/neutrino_star.jl` (`key=value`: `star=cold|hot`, `s`, `h`,
  `t_end`, `r_L`, `width`, `checkpoint`), its `jobs/` specs: the cold star
  to 1000 M; the hot star at three `h` for the convergence of `L` and the
  mean energies; the step's cost measured first, which decides which runs
  checkpoint.
- The same star coupled to TreeGH in `coupled/`.
- A neutrino row in `bin/benchmark.jl`; the hot star on an H200 against the
  CPU (`jobs/`).
- Accept: the cold star's luminosity below the bound found; the hot star's
  rates stated; the device within a stated number of ulp of the CPU.

## Step 21 — Neutrinos on MPI ranks

`CODE.md`: "Running distributed", "Neutrinos". Waits: step 20.

The neutrinos' fills, reductions (the optical depth's convergence, the
totals, the luminosity) and regrid across ranks, and `evolve!`'s refusal
of `comm` with neutrinos removed. Accept: the MPI comparison
(`test/mpi_tests.jl`) extended to a run with neutrinos, bit for bit against
one rank; a checkpoint written at one rank count restarted at another.
The neutrino scheme is not finished until this step is.

## Step 22 — Review pass

`CODE.md`: all of it. Waits: none.

Changes: the neutrino sections reread against the code, each (proposed)
of steps 16–21 decided with Erik, `CODE.md`'s status paragraph updated,
this plan moved into `HISTORY.md` and deleted.

Accept: rung 8 marked *(Done.)*; the full suite three ways and the
clean-checkout check green.
