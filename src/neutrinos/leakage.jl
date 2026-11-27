# The effective emission of the leakage scheme ("Neutrinos", "Leakage" in
# `CODE.md`; Radice et al. 2018, Eqs. 4–6; RJS96, B20–B23):
#
#     R^eff = R / (1 + t⁰_diff / t⁰_loss),   t⁰_diff = 𝒟 (τ⁰)² / (κ⁰_a + κ⁰_s),   t⁰_loss = n^β / R
#     Q^eff = Q / (1 + t¹_diff / t¹_loss),   t¹_diff = 𝒟 (τ¹)² / (κ¹_a + κ¹_s),   t¹_loss = e^β / Q
#
# in geometric units (`c = 1`, so `τ²/κ` is a time). Written as
# `R y/(y + x)` with `x = 𝒟 τ² R` and `y = κ n^β`, so that `R = 0` gives 0,
# `τ = 0` gives the free rate, and `κ n^β = 0` with `τ > 0` (an optical depth
# carried from a neighbour into a cell that does not interact) gives 0, all
# by branches rather than a division by zero.

"""The leakage scheme's diffusion constant `𝒟` (Radice et al. 2018, Eq. 5)."""
const LEAKAGE_DIFFUSION = 6

@inline function effective_rate(rate::T, τ::T, κ::T, density::T) where {T}
    iszero(rate) && return zero(T)
    x = LEAKAGE_DIFFUSION * τ * τ * rate
    iszero(x) && return rate
    y = κ * density
    iszero(y) && return zero(T)
    # `density` may be Inf (GrayRates with κa = 0, κs > 0): then y/(y + x) = 1.
    isinf(y) && return rate
    return rate * y / (y + x)
end

"""
    effective_rates(r::SpeciesRates, τ0, τ1) -> (R_eff, Q_eff)

One species' effective number and energy emission from its free rates and
its number- and energy-weighted optical depths ("Neutrinos", "Leakage" in
`CODE.md`): the free rates where `τ = 0`, and `n^β κ/(𝒟 τ²)`,
`e^β κ/(𝒟 τ²)` deep inside.
"""
@inline effective_rates(r::SpeciesRates{T}, τ0::T, τ1::T) where {T} =
    (effective_rate(r.R, τ0, r.κa0 + r.κs0, r.nβ),
     effective_rate(r.Q, τ1, r.κa1 + r.κs1, r.eβ))

"""
    absorption_opacity(r::SpeciesRates, τ0, E) -> κ_eff

M0's absorption opacity `κ^eff = e^{−τ⁰} (E/E^β)² κ⁰_a` (Radice et al. 2018,
Eq. 8), with `E` the free-streaming neutrinos' mean energy and
`E^β`, both in units of `m★c²` ("Neutrinos", "M0" in `CODE.md`). Zero
where the species does not absorb.
"""
@inline function absorption_opacity(r::SpeciesRates{T}, τ0::T, E::T) where {T}
    iszero(r.κa0) && return zero(T)
    ratio = E / r.Eβ
    return exp(-τ0) * ratio * ratio * r.κa0
end
