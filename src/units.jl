# Units: geometric units for the scheme, and the EOS's own run units.
#
# The scheme computes in geometric units, `G = c = M☉ = 1` (TreeGH's unit
# is `M`; with `M = M☉` they agree). EntropyEOS's tables compute in their
# run units: densities — `ρ★`, `D`, `τ`, `S`, `p`, `B²` alike — in κ-rescaled
# g/cm³, and `U = ε/c²` dimensionless ("Thermodynamics" in `CODE.md`). The
# analytic EOSes compute in whatever consistent units their parameters were
# given in, so one built in geometric units needs no conversion at all.
#
# So the conversion is one density factor, and it is written here and
# nowhere else. `GeometricEOS` puts it *inside* the EOS interface (EOS-1):
# it is an `AbstractEOS` whose densities are geometric and whose evaluation
# scales `ρ★` on the way in and the density-like derivatives on the way out,
# so that `con2prim_safe`, `prim2con`, `default_policy` and the atmosphere
# policy all run in geometric units unchanged. That works because the
# recovery's equations are homogeneous of degree one in the densities —
# `D`, `τ`, `S`, `B²`, `z = ρhW²` and `p` all scale together — and the EOS is
# the only place a density is compared with a number (proposed in step 1).
#
# The other factor is `κ`. The EOS works in `ρ★ = κρ`, re-zeroed so that
# `ρ★(1 + U) = ρ(1 + ε)`; the scheme's `D` is `ρ★ W` (CODE.md, "Where the
# fields live"), so `κ` enters only where a *physical* rest-mass density is
# wanted — initial data given in `ρ`, a baryon-mass diagnostic. `κ = 1` for
# the analytic EOSes.

"""Speed of light in cm/s, exact (= EntropyEOS's `C_LIGHT_CM_S`)."""
const C_CGS = 2.99792458e10

"""Newton's constant in cm³/(g s²), CODATA 2018."""
const G_CGS = 6.67430e-8

"""
The nominal solar mass parameter `GM☉` in cm³/s², IAU 2015 Resolution B3.

`GM☉` is known to ten digits and `G` to five, so the units below are built
from `GM☉` and divide by `G` only where a mass in grams is asked for.
"""
const GM_SUN_CGS = 1.3271244e26

"""The geometric length unit `GM☉/c²` in cm (≈ 1.477 km)."""
const MSUN_LENGTH_CGS = GM_SUN_CGS / C_CGS^2

"""The geometric time unit `GM☉/c³` in s (≈ 4.925 μs)."""
const MSUN_TIME_CGS = GM_SUN_CGS / C_CGS^3

"""The geometric mass unit `M☉ = GM☉/G` in g."""
const MSUN_MASS_CGS = GM_SUN_CGS / G_CGS

"""
The geometric density unit `M☉/(GM☉/c²)³ = c⁶/(G (GM☉)²)` in g/cm³,
6.1758e17 (measured in step 1).

An energy density, a pressure over `c²` and `B²` in Heaviside–Lorentz units
over `c²` take the same factor, which is what makes it the one factor
between the scheme's units and an EOS table's.
"""
const MSUN_DENSITY_CGS = MSUN_MASS_CGS / MSUN_LENGTH_CGS^3

"""
    EOSUnits{T}(; density, κ)
    EOSUnits(eos; density)

The two numbers between the scheme's geometric units and an EOS's run units:
`density`, the EOS's density per geometric density unit
([`MSUN_DENSITY_CGS`](@ref) for a table in g/cm³, 1 for an EOS built in
geometric units), and `κ`, the EOS's baryon-mass rescaling `ρ★ = κρ`.

The second form takes `κ` from the EOS (`EntropyEOS.κ`, 1 for the analytic
EOSes). `isbits`. See "Thermodynamics" in `CODE.md`.
"""
struct EOSUnits{T}
    density::T
    κ::T

    function EOSUnits{T}(density, κ) where {T}
        density = T(density)
        κ = T(κ)
        (isfinite(density) && density > 0) || throw(ArgumentError(
            "EOSUnits needs a positive finite density unit, got $density: it is the " *
            "factor every density, energy density and pressure is multiplied by on " *
            "its way to the EOS, so zero, a negative or a non-finite one makes every " *
            "conversion meaningless."))
        (isfinite(κ) && κ > 0) || throw(ArgumentError(
            "EOSUnits needs a positive finite κ, got $κ: the EOS works in ρ★ = κρ, so " *
            "a physical density is recovered as ρ★/κ."))
        return new{T}(density, κ)
    end
end

EOSUnits{T}(; density, κ) where {T} = EOSUnits{T}(density, κ)
EOSUnits(eos::EntropyEOS.AbstractEOS{T}; density) where {T} =
    EOSUnits{T}(density, EntropyEOS.κ(eos))

"""
    to_eos(units, x)

A density-like quantity — `ρ★`, `D`, `τ`, `S_i`, `p`, `B²`, an energy density
— from geometric units to the EOS's run units.
"""
@inline to_eos(u::EOSUnits, x) = x * oftype(x, u.density)

"""
    from_eos(units, x)

A density-like quantity from the EOS's run units to geometric units; the
inverse of [`to_eos`](@ref).
"""
@inline from_eos(u::EOSUnits, x) = x / oftype(x, u.density)

"""
    rest_mass_density(units, ρ★)

The physical rest-mass density `ρ = ρ★/κ`, in the units `ρ★` is in.
"""
@inline rest_mass_density(u::EOSUnits, ρ★) = ρ★ / oftype(ρ★, u.κ)

"""
    rescaled_density(units, ρ)

The EOS's density `ρ★ = κρ` from a physical rest-mass density, in the units
`ρ` is in; the inverse of [`rest_mass_density`](@ref).
"""
@inline rescaled_density(u::EOSUnits, ρ) = ρ * oftype(ρ, u.κ)

"""
    GeometricEOS(eos; density)

`eos`, an EOS in its own run units, seen in geometric units: an
`AbstractEOS` whose densities are the scheme's, so that EntropyEOS's
`prim2con`, `con2prim_safe` and `default_policy` run on geometric conserved
variables unchanged. `density` is the EOS's density per geometric density
unit, [`MSUN_DENSITY_CGS`](@ref) for a table in g/cm³.

`evaluate` scales `ρ★` up by `density` and the density-like results down:
`U`, `U_s`, `T̂`, `h`, `cs²`, `T_MeV`, `μ̃` and the warm start are unchanged,
`U_ρ` and `U_ρs` gain one factor of `density`, `U_ρρ` two, and `p` loses one.
The entropy windows are the inner EOS's at the scaled density, and the
density bounds are shifted by `log₁₀ density`. `κ` is the inner EOS's.

The units adapter of "Thermodynamics" in `CODE.md` (proposed in step 1).
`isbits` when `eos` is, and adapted to a device with `eos`.
"""
struct GeometricEOS{T,E<:EntropyEOS.AbstractEOS{T}} <: EntropyEOS.AbstractEOS{T}
    eos::E
    density::T
    log10_density::T
end

function GeometricEOS(eos::EntropyEOS.AbstractEOS{T}; density) where {T}
    u = EOSUnits(eos; density=density)
    return GeometricEOS{T,typeof(eos)}(eos, u.density, log10(u.density))
end

"""
    EOSUnits(g::GeometricEOS)

The units `g` converts with.
"""
EOSUnits(g::GeometricEOS{T}) where {T} = EOSUnits{T}(g.density, EntropyEOS.κ(g.eos))

@inline function EntropyEOS.evaluate(g::GeometricEOS, ρ★::T, s::T, yₑ::T,
                                     u_guess::T) where {T<:Real}
    f = T(g.density)
    pt = EntropyEOS.evaluate(g.eos, ρ★ * f, s, yₑ, u_guess)
    return EntropyEOS.EOSPoint{T}(pt.U, pt.U_ρ * f, pt.U_s, pt.U_ρρ * f * f, pt.U_ρs * f,
                                  pt.T̂, pt.p / f, pt.h, pt.cs², pt.T_MeV, pt.μ̃,
                                  pt.u_solved, pt.iters, pt.flags)
end

@inline EntropyEOS.srange(g::GeometricEOS, ρ★::T, yₑ::T) where {T<:Real} =
    EntropyEOS.srange(g.eos, ρ★ * T(g.density), yₑ)

@inline EntropyEOS.srange_extended(g::GeometricEOS, ρ★::T, yₑ::T) where {T<:Real} =
    EntropyEOS.srange_extended(g.eos, ρ★ * T(g.density), yₑ)

function EntropyEOS.logρ_bounds(g::GeometricEOS)
    x_lo, x_hi = EntropyEOS.logρ_bounds(g.eos)
    return (x_lo - g.log10_density, x_hi - g.log10_density)
end

EntropyEOS.yₑ_bounds(g::GeometricEOS) = EntropyEOS.yₑ_bounds(g.eos)
EntropyEOS.κ(g::GeometricEOS) = EntropyEOS.κ(g.eos)

function EntropyEOS.narrow(g::GeometricEOS, ::Type{S}) where {S<:AbstractFloat}
    inner = EntropyEOS.narrow(g.eos, S)
    return GeometricEOS{S,typeof(inner)}(inner, S(g.density), S(g.log10_density))
end

function Adapt.adapt_structure(to, g::GeometricEOS{T}) where {T}
    inner = Adapt.adapt(to, g.eos)
    return GeometricEOS{T,typeof(inner)}(inner, g.density, g.log10_density)
end
