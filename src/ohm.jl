# Ohm's law and the conductivity models.
#
# The current seen by the normal observer, for an isotropic conductor in the
# fluid frame (Palenzuela et al. 2009; Palenzuela 2013):
#
#     jⁱ = q vⁱ + σ W [Eⁱ + ε^{ijk} v_j B_k − (v_k Eᵏ) vⁱ]
#
# The `q v` part is the convected charge and is explicit; the `σ` part is
# the conduction current, stiff inside a star, and the one implicit term of
# the scheme ("Ohm's law and the conductivity" and "Sources" in `CODE.md`).
# Its bracket is the electric field in the fluid frame, projected back to
# the normal observer: it vanishes exactly when `E = −v × B`, which is why
# the ideal limit is `σ → ∞` with that bracket held at zero.

"""
    Conductivity

The conductivity `σ` as an `isbits` model ("Ohm's law and the conductivity"
in `CODE.md`): [`Vacuum`](@ref), [`Constant`](@ref),
[`DensityProfile`](@ref), [`IdealConductor`](@ref). Each answers
[`conductivity`](@ref).
"""
abstract type Conductivity end

"""
    Vacuum()

`σ = 0` everywhere: electrovacuum, and the current is the convected charge
`q v` alone.
"""
struct Vacuum <: Conductivity end

"""
    Constant(σ₀)

`σ = σ₀` everywhere: the resistive shock tubes and the current sheet.
"""
struct Constant{T} <: Conductivity
    σ₀::T
end

"""
    DensityProfile(σ₀, ρ_atm, cutoff=2)

`σ = σ₀ max(0, 1 − cutoff·ρ_atm/ρ)`: `σ₀` deep inside matter, falling
continuously to zero at `ρ = cutoff·ρ_atm` and zero below it, so that the
atmosphere carries no current and the exterior of a star is electrovacuum
("Floors, the atmosphere and the magnetosphere" in `CODE.md`). The shape
follows the density-dependent conductivities of Palenzuela (2013) and
Dionysopoulou et al. (2013) (proposed in step 2; rung 6 settles it).
"""
struct DensityProfile{T} <: Conductivity
    σ₀::T
    ρ_atm::T
    cutoff::T
end
DensityProfile(σ₀, ρ_atm, cutoff=2) = DensityProfile(promote(float(σ₀), ρ_atm, cutoff)...)

"""
    IdealConductor()

`σ = ∞`: the exact ideal-MHD limit, `E = −v × B`. The conduction current is
then not a function of the state — it is whatever holds `E` at `−v × B`, and
the implicit stage solve imposes that directly (step 3) — so
[`ohm_current`](@ref) has no value for it and [`conductivity`](@ref) is
`Inf`.
"""
struct IdealConductor <: Conductivity end

"""
    conductivity(model, prim) -> σ

The conductivity of `model` at the primitive state `prim`, in `prim`'s
element type.
"""
@inline conductivity(::Vacuum, prim::NTuple{N,T}) where {N,T} = zero(T)
@inline conductivity(c::Constant, prim::NTuple{N,T}) where {N,T} = convert(T, c.σ₀)
@inline function conductivity(c::DensityProfile, prim::NTuple{N,T}) where {N,T}
    ρ = prim_ρ(prim)
    f = one(T) - convert(T, c.cutoff) * convert(T, c.ρ_atm) / ρ
    # `max` keeps a NaN density NaN, for the policy to flag upstream.
    return convert(T, c.σ₀) * max(zero(T), f)
end
@inline conductivity(::IdealConductor, prim::NTuple{N,T}) where {N,T} = T(Inf)

"""
    conductivity_slope(model, prim) -> ∂σ/∂ρ

The derivative of [`conductivity`](@ref) with respect to the rest-mass
density at the primitive state `prim`, for the implicit stage solve's Newton
Jacobian (step 3). Zero for the models that do not depend on the state. A
model that reads `s` or `Yₑ` as well would need their derivatives too, and
the stage solve a column for each (proposed in step 3).
"""
@inline conductivity_slope(::Conductivity, prim::NTuple{N,T}) where {N,T} = zero(T)
@inline function conductivity_slope(c::DensityProfile, prim::NTuple{N,T}) where {N,T}
    ρ = prim_ρ(prim)
    kρ = convert(T, c.cutoff) * convert(T, c.ρ_atm)
    return ρ > kρ ? convert(T, c.σ₀) * kρ / (ρ * ρ) : zero(T)
end

"""
    magnetization(m::Metric3, B, D) -> μ

`μ = γ_ij Bⁱ Bʲ / D`: twice the field's energy per unit rest mass, both seen
by the normal observer, from the cell's field `Bⁱ` and its undensitized `D`.
Within an implicit stage both are fixed — `B` is explicit and the stage
moves no rest mass — so a conductivity that depends on `μ` adds nothing to
the stage's nonlinearity (proposed in step 3). It bounds the fluid frame's
magnetization: `μ/W ≤ b²/ρ ≤ W μ`.
"""
@inline magnetization(m::Metric3{T}, B::SVector{3,T}, D::T) where {T} = dot_up(m, B, B) / D

"""
    MagnetizationCutoff(model, μ_max)

The conductivity of the finite-σ `model`, tapered to zero where the plasma is
magnetically dominated: `σ = g(μ) σ_model`, with [`magnetization`](@ref)
`μ = B²/D` and `g` a monotone `C¹` smoothstep in `log μ`, 1 for
`μ ≤ μ_max/10` and 0 for `μ ≥ μ_max` (proposed in step 3). There the
plasma's inertia is negligible against the field's, a stage of a scalar
fluid-frame Ohm's law hands the fluid more energy than it has, and the stage
equations stop having one admissible root; resistive neutron-star codes set
`σ = 0` outside the star for the same reason (decided by Erik 2026-10-14;
"Ohm's law and the conductivity" in `CODE.md`).

It wraps any finite-σ model — [`Constant`](@ref), [`DensityProfile`](@ref),
[`Vacuum`](@ref) — and is `isbits` when the model is. It refuses
[`IdealConductor`](@ref): the ideal recovery puts the field's inertia in its
own model and is well posed at any magnetization, and `∞ · g` is not a
conductivity.

`μ_max` is required and has no default (decided by Erik 2026-10-14): it
trades the stage's robustness against how much of the magnetosphere is
resistive, so each case or driver states its own, and nothing wraps a model
implicitly. `μ_max = 10` is the measured safe value — the largest with no
fallback at a fixed non-ideal fraction `f ≤ 0.1` — and the physically
motivated `50–100` costs a measured fallback rate of order `10⁻⁵`; see
"Ohm's law and the conductivity" in `CODE.md`. `MagnetizationCutoff(model)`
throws an `ArgumentError` saying so.
"""
struct MagnetizationCutoff{M<:Conductivity,T} <: Conductivity
    model::M
    μ_max::T
    function MagnetizationCutoff(model::M, μ_max::T) where {M<:Conductivity,T<:Real}
        model isa IdealConductor &&
            throw(ArgumentError("an ideal conductor cannot be tapered: its σ is infinite, " *
                                "and the ideal recovery is well posed at any magnetization"))
        μ_max > zero(T) ||
            throw(ArgumentError("the magnetization cutoff must be positive, got $μ_max"))
        return new{M,T}(model, μ_max)
    end
end
MagnetizationCutoff(model::Conductivity, μ_max::Integer) = MagnetizationCutoff(model, float(μ_max))
function MagnetizationCutoff(model::Conductivity)
    throw(ArgumentError("MagnetizationCutoff needs an explicit μ_max and has no default: " *
                        "the threshold trades the stage solve's robustness against how much " *
                        "of the plasma is resistive, so each case states its own " *
                        "(μ_max = 10 is the measured safe value; see \"Ohm's law and the " *
                        "conductivity\" in CODE.md)"))
end

"""
    magnetization_taper(μ_max, μ) -> g ∈ [0, 1]

The taper of [`MagnetizationCutoff`](@ref): `1 − t²(3 − 2t)` with
`t = log₁₀(10 μ/μ_max)` clamped to `[0, 1]`. A NaN `μ` stays NaN.
"""
@inline function magnetization_taper(μ_max::T, μ::T) where {T}
    t = log(10 * μ / μ_max) / log(T(10))
    t = t < zero(T) ? zero(T) : t > one(T) ? one(T) : t     # NaN passes through
    return one(T) - t * t * (3 - 2 * t)
end

"""
    conductivity(model, prim, μ) -> σ

The conductivity of `model` at the primitive state `prim` and the
[`magnetization`](@ref) `μ`: `conductivity(model, prim)` for every model but
[`MagnetizationCutoff`](@ref), which needs `μ` and has no two-argument form.
"""
@inline conductivity(model::Conductivity, prim::NTuple{N,T}, μ::T) where {N,T} =
    conductivity(model, prim)
@inline function conductivity(c::MagnetizationCutoff, prim::NTuple{N,T}, μ::T) where {N,T}
    g = magnetization_taper(convert(T, c.μ_max), μ)
    iszero(g) && return zero(T)                  # a mask: the model is not read
    return g * conductivity(c.model, prim, μ)
end
conductivity(::MagnetizationCutoff, prim::NTuple) = throw_needs_magnetization()

@noinline throw_needs_magnetization() =
    throw(ArgumentError("a MagnetizationCutoff's conductivity depends on the magnetization: " *
                        "call conductivity(model, prim, μ) with μ = magnetization(m, B, D)"))

"""
    conductivity_slope(model, prim, μ) -> ∂σ/∂ρ

[`conductivity_slope`](@ref) at the magnetization `μ`, which is constant
within a stage, so that the cutoff's taper only scales it.
"""
@inline conductivity_slope(model::Conductivity, prim::NTuple{N,T}, μ::T) where {N,T} =
    conductivity_slope(model, prim)
@inline function conductivity_slope(c::MagnetizationCutoff, prim::NTuple{N,T}, μ::T) where {N,T}
    g = magnetization_taper(convert(T, c.μ_max), μ)
    iszero(g) && return zero(T)
    return g * conductivity_slope(c.model, prim, μ)
end

"""
    conduction_drift(m::Metric3, prim, E, B) -> SVector{3}

The bracket of Ohm's law, `W [Eⁱ + ε^{ijk} v_j B_k − (v_k Eᵏ) vⁱ]`: the
conduction current per unit `σ`, contravariant. Zero exactly when
`E = −v × B`.
"""
@inline function conduction_drift(m::Metric3{T}, prim::NTuple{N,T}, E::SVector{3,T},
                                  B::SVector{3,T}) where {N,T}
    v = prim_v(prim)
    v_cross_B = raise(m, cross_down(m, v, B))     # ε^{ijk} v_j B_k
    return prim_W(prim) * (E + v_cross_B - dot_up(m, v, E) * v)
end

"""
    ohm_current(model, m::Metric3, prim, E, B, q) -> SVector{3}

The current `jⁱ = q vⁱ + σ W [Eⁱ + ε^{ijk} v_j B_k − (v_k Eᵏ) vⁱ]` seen by
the normal observer, contravariant and undensitized, for the conductivity
`model` at the primitive state `prim`, the fields `Eⁱ`, `Bⁱ` and the charge
density `q`.

Where `σ = 0` the conduction term is not formed at all (a mask branches, it
never multiplies: `0 · NaN = NaN`). For [`IdealConductor`](@ref) it throws:
the ideal conduction current is not a function of the state. See "Ohm's law
and the conductivity" in `CODE.md`.
"""
@inline function ohm_current(model::Conductivity, m::Metric3{T}, prim::NTuple{N,T},
                             E::SVector{3,T}, B::SVector{3,T}, q::T) where {N,T}
    j_conv = q * prim_v(prim)
    σ = conductivity(model, prim, magnetization(m, B, prim_ρ(prim) * prim_W(prim)))
    iszero(σ) && return j_conv
    return j_conv + σ * conduction_drift(m, prim, E, B)
end

@inline ohm_current(::Vacuum, m::Metric3{T}, prim::NTuple{N,T}, E::SVector{3,T},
                    B::SVector{3,T}, q::T) where {N,T} = q * prim_v(prim)

ohm_current(::IdealConductor, m::Metric3{T}, prim::NTuple{N,T}, E::SVector{3,T},
            B::SVector{3,T}, q::T) where {N,T} = throw_ideal_current()

@noinline throw_ideal_current() =
    throw(ArgumentError("an ideal conductor's current is not a function of the state: " *
                        "it is whatever holds E = −v × B, which the implicit stage " *
                        "solve imposes; see \"Ohm's law and the conductivity\" in CODE.md"))
