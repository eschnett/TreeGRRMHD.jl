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
    σ = conductivity(model, prim)
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
