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

*Status: steps 1 and 2 of `PLAN.md`, the pointwise layers the scheme will
stand on.* Step 1: geometric units and the EOS adapter
([`GeometricEOS`](@ref)), the point metric ([`Metric3`](@ref)), the layouts
of the ten evolved cell variables and of the primitive set,
[`prim2con`](@ref) with the electromagnetic parts, and the 3+1 primitive
recovery around EntropyEOS's one `con2prim_safe` ([`con2prim_hydro`](@ref),
and [`con2prim_ideal`](@ref) as the stage solve's seed). Step 2: the metric
providers [`Flat`](@ref) and [`Analytic`](@ref), the stress-energy, Ohm's
law, the physical fluxes, the light-cone Riemann solvers, the UCT–HLL EMF
and the sources. No mesh code yet.

See `CODE.md` in the package root for the design document — what each piece
is for and why it is that way — and `PLAN.md` for the steps.
"""
module TreeGRRMHD

using TreeAMR

using KernelAbstractions: Backend, CPU, allocate, get_backend
using StaticArrays: SArray, SMatrix, SVector

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
# Units and the EOS adapter
export EOSUnits, GeometricEOS, MSUN_DENSITY_CGS
# The point metric
export Metric3, flat_metric, ideal_E
# The variables
export NU, NP, NPRIM
# The conversions
export primitive_state, prim2con, con2prim_hydro, con2prim_ideal, guesses, cold_guesses
# The metric providers
export MetricProvider, Flat, Analytic, MetricGradient, point_metric, metric_gradient
# The pointwise physics
export stress_energy, T4_lower
export Conductivity, Vacuum, Constant, DensityProfile, IdealConductor, conductivity,
       ohm_current
export light_cone, physical_flux, riemann_flux, gauge_flux
export faraday_emf, uct_hll_emf, gauge_potential
export geometric_source, explicit_source, ampere_explicit_source, ampere_stiff_source,
       ampere_source, gauge_damping
# The implicit stage solve
export solve_ohm_stage, OhmStageOptions, OhmStageResult, ohm_E
export OHM_SIGMA_ZERO, OHM_IDEAL, OHM_FIXED_POINT, OHM_NEWTON, OHM_FALLBACK, OHM_IDEAL_SEED,
       OHM_BACKTRACKED, OHM_SWITCHED, OHM_REPAIRED

include("precision.jl")
include("device.jl")
include("units.jl")
include("metric3.jl")
include("variables.jl")
include("prim2con.jl")
include("con2prim.jl")
include("metric.jl")
include("stress_energy.jl")
include("ohm.jl")
include("flux.jl")
include("riemann.jl")
include("emf.jl")
include("sources.jl")
include("stage_solve.jl")

end # module TreeGRRMHD
