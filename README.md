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

**Status: the pointwise layer and the evolutions on a uniform mesh (steps
0 to 3 and 5 to 7 of `PLAN.md`).** The staggered scheme runs on TreeAMR's
mesh in one, two and three dimensions: the matter and the electric field at cells, the vector potential
on edges, the gauge scalar at vertices, and the magnetic field as the
discrete curl of the potential, advanced by the IMEX integrator with the
Ohm solve as its implicit stage. A vacuum electromagnetic wave and a
large-amplitude circularly polarized Alfvén wave converge at second order,
with the conserved totals constant to round-off; Balsara's five
relativistic MHD shock tubes match Giacomazzo & Rezzolla's exact plateaus to
2% (all but one region, next to a shell 400 cells do not resolve),
a resistive shock tube approaches the ideal solution as the conductivity
grows to 10⁶, a current sheet diffuses as it should, and a blast into an
atmosphere 10⁸ times thinner is held by the atmosphere reset. In two and
three dimensions a run that does not vary along `y` is the 1D run bit for
bit, `∇·B` stays at round-off at every step, and the cylindrical explosion,
the magnetic rotor, the Orszag–Tang vortex and a field loop keep their
symmetries and conserved totals; PPM, WENO-Z and MP5 join piecewise-linear
reconstruction. Refinement across coarse-fine faces is next. Beneath it lie the module
shell and its host helpers; the tests that say the resolved releases of
TreeAMR, EntropyEOS, SpacetimeMetrics and IMEXRungeKutta still provide what
the scheme is written against, and which of the upstream items the plan
waits for have landed; and the pointwise physics the scheme stands on — geometric
units and an adapter that shows a tabulated EOS in them, the 3+1 point
metric, the layouts of the evolved and primitive variables, and the
conversion between primitive and conserved variables, electromagnetic
parts included, around EntropyEOS's primitive recovery; and the physics a
right-hand side evaluates at a point — the analytic metric providers, the
stress-energy, Ohm's law, the fluxes and the light-cone Riemann solvers,
the upwind electromotive force for the vector potential, and the sources;
and the implicit stage of the time integrator, which solves Ohm's law for
the electric field in each cell, coupled to the fluid through the one
primitive recovery. Each of these runs inside a KernelAbstractions kernel,
bit-identically to the host.

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
