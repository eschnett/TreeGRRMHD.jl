# TreeGRRMHD.jl

[![CI](https://github.com/eschnett/TreeGRRMHD.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/eschnett/TreeGRRMHD.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/eschnett/TreeGRRMHD.jl/graph/badge.svg)](https://codecov.io/gh/eschnett/TreeGRRMHD.jl)

`TreeGRRMHD` solves the equations of general-relativistic **resistive**
magnetohydrodynamics with a high-resolution shock-capturing finite-volume
scheme on the adaptively refined mesh of
[TreeAMR](https://github.com/eschnett/TreeAMR.jl).

That is the Valencia matter equations for the total (fluid plus
electromagnetic) energy and momentum, together with the full Maxwell
equations and an Ohm's law whose conductivity is large inside a star — where
the scheme must recover ideal MHD and the resistive term is stiff — and goes
to zero outside, where the fields obey the electrovacuum equations. A neutron
star's dipole magnetosphere is then a solution of the equations rather than
an atmosphere artefact. The magnetic field is the discrete curl of a vector
potential on the edges, so `∇·B = 0` holds to round-off on every level and
across every regrid; the stiff current is handled by an IMEX Runge–Kutta
integrator whose implicit stage this package solves itself.

It is the fourth application of TreeAMR, after
[TreeWave](https://github.com/eschnett/TreeWave.jl),
[TreeHydro](https://github.com/eschnett/TreeHydro.jl) — the Newtonian
rehearsal whose scheme, driver and conventions this package inherits — and
[TreeGeneralizedHarmonic](https://github.com/eschnett/TreeGeneralizedHarmonic.jl),
the spacetime it couples to. The equation of state and the one primitive
recovery come from [EntropyEOS](https://github.com/eschnett/EntropyEOS.jl),
the analytic backgrounds from
[SpacetimeMetrics](https://github.com/eschnett/SpacetimeMetrics.jl), and the
time integrator from
[IMEXRungeKutta](https://github.com/eschnett/IMEXRungeKutta.jl). The end goal
is neutron stars and core-collapse supernovae; the first science target is a
magnetized TOV star.

**Status: scaffolding (step 0 of `PLAN.md`). There is no physics yet.** What
exists is the module shell, the `Base` bridges for software floating-point
types, the host-copy helpers, and the tests that say the resolved releases
of TreeAMR, EntropyEOS, SpacetimeMetrics and IMEXRungeKutta still provide
what the scheme will be written against — including that a SpacetimeMetrics
metric runs inside a kernel, bit-identically to the host, and that the
integrator calls the stage solve and both limiters as the design assumes —
and which of the upstream items the plan waits for have landed.

`CODE.md` is the design document: what each piece is for and why it is that
way. `PLAN.md` is the implementation plan, step by step, with the exact
requirements on the sibling packages. `CLAUDE.md` has the commands.

## Installation

Julia 1.11 or later. EntropyEOS and IMEXRungeKutta are not in the General
registry yet; `Project.toml` names them under `[sources]`, so in a clone a
plain instantiate finds them:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

A dependency's `[sources]` is ignored by the environment that depends on it,
so an environment of your own that adds this package needs the same two
entries, or `Pkg.add(url = …)` of both packages first.

## Running the tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.test(; julia_args = ["--threads=4"])'
```

The log names every upstream item (`AMR-n`, `EOS-n`, `GH-n`) as landed or
waiting.
