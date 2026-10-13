"""
    TreeGRRMHD

General-relativistic **resistive** magnetohydrodynamics on
[TreeAMR](https://github.com/eschnett/TreeAMR.jl): the Valencia matter
equations together with the full Maxwell equations and an Ohm's law, on a
cell-centered finite-volume scheme with the vector potential on edges, so
that `∇·B = 0` holds to round-off on every level and across every regrid.

TreeAMR supplies the mesh and contains no physics. The spacetime comes from
[TreeGeneralizedHarmonic](https://github.com/eschnett/TreeGeneralizedHarmonic.jl),
the equation of state and the one primitive recovery from
[EntropyEOS](https://github.com/eschnett/EntropyEOS.jl), the analytic
backgrounds from
[SpacetimeMetrics](https://github.com/eschnett/SpacetimeMetrics.jl), and the
time integrator, whose implicit stage this package solves itself, from
[IMEXRungeKutta](https://github.com/eschnett/IMEXRungeKutta.jl).
[TreeHydro](https://github.com/eschnett/TreeHydro.jl) is the Newtonian
rehearsal whose scheme, driver and conventions this package inherits.

Every driver will take the floating-point type it computes in as a leading
positional argument and the KernelAbstractions backend it runs on as a
keyword. Physics is validated in `Float64`; `Float32` is a device smoke
test.

*Status: step 0 of `PLAN.md` — the module shell, the `Base` bridges for
software floating-point types, the host-copy helpers, and the tests that say
the resolved releases of TreeAMR, EntropyEOS, SpacetimeMetrics and
IMEXRungeKutta provide what the scheme is written against, and which
upstream items have landed. No physics yet.*

See `CODE.md` in the package root for the design document — what each piece
is for and why it is that way — and `PLAN.md` for the steps.
"""
module TreeGRRMHD

using TreeAMR

using KernelAbstractions: Backend, CPU, allocate, get_backend
using StaticArrays: SVector, SMatrix

# Imported, not `using`d: EntropyEOS exports `prim2con`, `con2prim` and
# `evaluate`, and SpacetimeMetrics `metric`, all names this package's own
# 3+1 layer will use with other signatures (steps 1 and 2). Qualified
# access says which one is meant. `IRK.` follows TreeHydro, so that
# `init`/`solve!` read as the integrator's and not CommonSolve's.
import Adapt
import EntropyEOS
import IMEXRungeKutta as IRK
import SpacetimeMetrics

# Devices
export hostcopy

include("precision.jl")
include("device.jl")

end # module TreeGRRMHD
