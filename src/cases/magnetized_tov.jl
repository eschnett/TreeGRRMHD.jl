# Rung 6: the magnetized TOV star ("Validation ladder" 6 in `CODE.md`; step 12 of
# `PLAN.md`). Rung 4's star — the polytrope `K = 100`, `Γ = 2`, `ρ_c = 1.28·10⁻³` at rest on
# its own frozen metric, `Analytic(TOVMetric)` — threaded by a poloidal field whose exterior
# is the static vacuum dipole of the Schwarzschild spacetime outside the star, under a
# resistive Ohm's law whose conductivity follows the density.
#
# **The exterior** (proposed in step 12). The static, current-free dipole on Schwarzschild
# (Wasserman & Shapiro 1983, ApJ 265, 1036), in areal coordinates `(r, θ, φ)`:
#
#     A_φ = −(3μ/(8M³)) r² sin²θ [ln(1 − 2M/r) + 2M/r + 2M²/r²] = 3μ sin²θ S(x)/r,
#
# `x = 2M/r`, `S(x) = Σ_{k≥3} x^{k−3}/k`, which is `μ sin²θ/r` as `M/r → 0`. `A_φ` is the
# covariant `φ` component, the same in isotropic coordinates (`θ`, `φ` are), so with
# `ϖ̄² = x² + y² = r̄² sin²θ` the Cartesian potential is
#
#     A_i = a(u) (−y, x, 0),   a = A_φ/ϖ̄² = 3μ S(x)/(r u),   u = r̄²,   r = r̄ (1 + M/(2r̄))².
#
# It is static with `E = 0`, `Φ = 0` (`β = 0`): Faraday's law holds, and so does Ampère's,
# since the field is current-free on the metric, `∇ × (α B_k) = 0` (a test checks it on
# `TOVMetric`, where the flat dipole fails). Its Lorenz gauge holds too: `α √γ γ^{ij} A_j` is
# `g(r̄) (−y, x, 0)`, divergence-free. `S` is its series where `x < 1/4`, where the closed
# form's cancellation would cost up to `3ε/x²` (7·10⁻¹⁴ at the box's corner), and the closed
# form elsewhere.
#
# **The interior** (proposed in step 12). `a(u)` continued inside the surface by the
# quadratic in `u` that matches the exterior's value and first two `u`-derivatives at
# `u = R̄²`, so that `B` and its first derivatives are continuous there: the current inside,
# `∇ × (α B)/α`, falls to zero at the surface, where `DensityProfile` takes σ to zero and no
# current could be held. Polynomial in `u`, so `A` is smooth at the centre; poloidal and
# axisymmetric, so the current is toroidal and `E·B = 0` in the continuum for any σ — the
# ideal `E = −v × B` of a poloidal flow is toroidal too. The derivatives at the surface come
# from the closed form, the second by a central difference of the first in 256-bit
# `BigFloat` on the host.
#
# **The strength.** The case states the field at the centre, `B_c = |B|` there for the
# normal observer (`𝔅ᶻ(0) = 2a(0)`, `B = 𝔅/ψ⁶`), and `μ` follows. The default `10⁻⁴`, in
# this package's Heaviside–Lorentz geometric units, is `B²/(2p) = 3·10⁻⁵` at the centre: a
# test field, whose Lorentz force is far below the star's own truncation error.
#
# **The conductivity.** `MagnetizationCutoff(DensityProfile(σ₀, ρ_min), μ_max)`: `σ₀` deep
# inside the star, zero at and below `2ρ_min` (the atmosphere carries no current), and zero
# where `B²/D ≥ μ_max`, the magnetically dominated layer at the surface. `σ₀` and `μ_max` have
# no default (`CODE.md`, "Ohm's law and the conductivity").

"""
    dipole_S(x) -> (S, dS/dx)

`S(x) = Σ_{k≥3} x^{k−3}/k = −(ln(1 − x) + x + x²/2)/x³` and its derivative, for
`0 ≤ x < 1`: the series below `x = 1/4`, the closed form above (`src/cases/magnetized_tov.jl`).
"""
@inline function dipole_S(x::T) where {T}
    if x < one(T) / 4
        # 40 terms: the first left out is below 4^{−40}/43 of S(0) = 1/3.
        S = zero(T)
        dS = zero(T)
        for j in 40:-1:1
            S = muladd(S, x, inv(T(j + 3)))
            dS = muladd(dS, x, T(j) / T(j + 3))
        end
        S = muladd(S, x, inv(T(3)))
        return S, dS
    else
        L = log1p(-x) + x + x * x / 2
        S = -L / x^3
        return S, inv(x * (1 - x)) - 3 * S / x
    end
end

"""
    exterior_dipole(M, u) -> (a, da/du)

The unit-moment (`μ = 1`) static vacuum dipole of a mass `M` in isotropic coordinates:
`A_i = a(u) (−y, x, 0)` at `u = r̄²` (Wasserman & Shapiro 1983; `src/cases/magnetized_tov.jl`).
Needs `r = r̄ (1 + M/(2r̄))² > 2M`, which holds for every `r̄ > 0` but `r̄ = M/2`.
"""
@inline function exterior_dipole(M::T, u::T) where {T}
    r̄ = sqrt(u)
    r = r̄ + M + M * M / (4 * r̄)
    dr = 1 - M * M / (4 * u)                         # dr/dr̄
    x = 2 * M / r
    S, dS = dipole_S(x)
    F = 3 * S / r
    dF = -3 * (S + x * dS) / (r * r)                 # dF/dr, with dx/dr = −x/r
    a = F / u
    da_dr̄ = dF * dr / u - 2 * F / (u * r̄)
    return a, da_dr̄ / (2 * r̄)
end

"""
    StellarDipole{T}

A star's poloidal field as `A_i = a(u) (−y, x, 0)`, `u = r̄²`, `isbits`: the exterior
dipole of mass `M` and moment `μ` outside `u_s = R̄²` ([`exterior_dipole`](@ref)), and
inside it `a = c₀ + c₁ (u − u_s) + c₂ (u − u_s)²`, the quadratic that matches the exterior
to second order at the surface (`src/cases/magnetized_tov.jl`). Built by
[`stellar_dipole`](@ref).
"""
struct StellarDipole{T}
    M::T
    u_s::T
    μ::T
    c₀::T
    c₁::T
    c₂::T
end

"""
    dipole_a(d::StellarDipole, u) -> (a, da/du)
"""
@inline function dipole_a(d::StellarDipole{T}, u::T) where {T}
    if u < d.u_s
        δ = u - d.u_s
        return d.c₀ + δ * (d.c₁ + δ * d.c₂), d.c₁ + 2 * d.c₂ * δ
    else
        a, da = exterior_dipole(d.M, u)
        return d.μ * a, d.μ * da
    end
end

"""
    dipole_potential(d::StellarDipole, x) -> A

The covariant potential `A_i = a(u) (−y, x, 0)` at `x`.
"""
@inline function dipole_potential(d::StellarDipole{T}, x::SVector{3,T}) where {T}
    a, _ = dipole_a(d, x[1]^2 + x[2]^2 + x[3]^2)
    return SVector(-a * x[2], a * x[1], zero(T))
end

"""
    dipole_curl(d::StellarDipole, x) -> 𝔅

The densitized field `𝔅ⁱ = [ijk] ∂ⱼ A_k` of [`dipole_potential`](@ref), in closed form:
`(−2z x a′, −2z y a′, 2a + 2(x² + y²) a′)`, `a′ = da/du`. `Bⁱ = 𝔅ⁱ/√γ`.
"""
@inline function dipole_curl(d::StellarDipole{T}, x::SVector{3,T}) where {T}
    a, da = dipole_a(d, x[1]^2 + x[2]^2 + x[3]^2)
    return SVector(-2 * x[3] * x[1] * da, -2 * x[3] * x[2] * da,
                   2 * a + 2 * (x[1]^2 + x[2]^2) * da)
end

"""
    stellar_dipole(m::TOVMetric, B_c) -> StellarDipole

The field of the magnetized star ([`StellarDipole`](@ref)) on the star `m`, scaled so that
`|B| = B_c` at its centre: the exterior dipole outside `R̄`, continued inside by the
quadratic in `u = r̄²` that matches its value and first two derivatives at the surface.
"""
function stellar_dipole(m::TOVMetric{T}, B_c) where {T}
    B_c ≥ 0 || throw(ArgumentError("the central field must not be negative, got B_c = $B_c."))
    u_s = m.R̄^2
    coeffs = setprecision(BigFloat, 256) do
        M, U = BigFloat(m.M), BigFloat(u_s)
        a, da = exterior_dipole(M, U)
        δ = U / BigFloat(10)^30
        d2a = (exterior_dipole(M, U + δ)[2] - exterior_dipole(M, U - δ)[2]) / (2 * δ)
        (a, da, d2a / 2)
    end
    c₀, c₁, c₂ = T.(coeffs)
    a₀ = c₀ - c₁ * u_s + c₂ * u_s^2                   # a(0) at μ = 1
    ψ_c, _ = tov_potentials(m, zero(T))
    μ = T(B_c) * ψ_c^6 / (2 * a₀)
    return StellarDipole{T}(m.M, u_s, μ, μ * c₀, μ * c₁, μ * c₂)
end

"""
    MagnetizedTOVStar{T,NK}

The data of the magnetized star, `isbits`: rung 4's [`TOVStar`](@ref), at rest, and its
field, a [`StellarDipole`](@ref). See `src/cases/magnetized_tov.jl`.
"""
struct MagnetizedTOVStar{T,NK}
    star::TOVStar{T,NK}
    dipole::StellarDipole{T}
end

star_metric(w::TOVStar) = w.metric
star_metric(w::MagnetizedTOVStar) = w.star.metric

@inline function wave_fields(w::MagnetizedTOVStar, x::SVector{3,T}, t::T) where {T}
    ρ, v, s, Yₑ, E, _ = wave_fields(w.star, x, t)
    m = point_metric(w.star.provider, t, x)
    return ρ, v, s, Yₑ, E, dipole_curl(w.dipole, x) / m.sqrtγ
end

@inline wave_potential(w::MagnetizedTOVStar, x::SVector{3,T}, t::T) where {T} =
    dipole_potential(w.dipole, x), zero(T)

"""
    magnetized_tov_case([T]; σ₀, μ_max, B_c = 1e-4, K = 100, Γ = 2, ρ_c = 1.28e-3, L = 24,
                        roots = 4, ρ_min = 1e-10, NK = 129, steps = 4000)

The magnetized TOV star ("Validation ladder" 6 in `CODE.md`) as a three-dimensional
[`MHDCase`](@ref): [`tov_case`](@ref)'s star at rest on `Analytic(TOVMetric)`, threaded by
the poloidal field of [`stellar_dipole`](@ref) — the static vacuum dipole of Schwarzschild
outside the surface, its quadratic continuation in `r̄²` inside, `|B| = B_c` at the centre —
with `E = 0` and `Φ̃ = 0` everywhere, the solution of the field equations outside and of the
ideal ones inside. The conductivity is
`MagnetizationCutoff(DensityProfile(σ₀, ρ_min), μ_max)`: σ follows the density to zero at
`2ρ_min`, and the magnetically dominated layer is electrovacuum; `σ₀` and `μ_max` have no
default. The potential does not depend on `t`, so its Dirichlet faces are absolute
(`static_potential = true`). The case's data is `case.fields.w`, a
[`MagnetizedTOVStar`](@ref); [`tov_boxes!`](@ref) refines around it as around rung 4's.
"""
function magnetized_tov_case(::Type{T}=Float64; σ₀, μ_max, B_c=1e-4, K=100, Γ=2,
                             ρ_c=1.28e-3, L=24, roots=4, ρ_min=1e-10, NK::Integer=129,
                             steps::Integer=4000) where {T}
    T(σ₀) > 0 || throw(ArgumentError(
        "the star's conductivity σ₀ must be positive, got σ₀ = $σ₀ (Vacuum is not a star)."))
    star = tov_case(T; K, Γ, ρ_c, L, roots, ρ_min, NK, steps)
    w0 = star.fields.w
    w = MagnetizedTOVStar(w0, stellar_dipole(w0.metric, T(B_c)))
    model = MagnetizationCutoff(DensityProfile(T(σ₀), T(ρ_min)), T(μ_max))
    return MHDCase(T, Val(3); fields=WaveFields(w), potential=WavePotential(w), eos=star.eos,
                   proj=star.proj, pol=star.pol, model, provider=star.provider,
                   periodic=false, extents=star.extents, roots=star.roots,
                   static_potential=true)
end
