# The Tolman–Oppenheimer–Volkoff star of rung 4 ("Validation ladder" 4 in
# `CODE.md`; step 10 of `PLAN.md`): a static, spherical, isentropic polytrope
# `p = K ρ^Γ` in isotropic coordinates,
#
#     ds² = −α² dt² + ψ⁴ (dx² + dy² + dz²),
#
# solved once on the host, and its metric as an `isbits` SpacetimeMetrics
# metric, `TOVMetric`, which the Cowling star reads through `Analytic` like
# every analytic background (decided by Erik 2026-11-06).
#
# **The solve** (proposed in step 10). Lindblom's (1992, ApJ 398, 569)
# enthalpy form: the log-enthalpy `η = log h`, `h = (e + p)/ρ`, is the
# independent variable, from its central value down to `η = 0` at the
# surface, so that the surface is a grid point and not a root to find; for a
# static fluid `η + log α` is constant (`dη/dr = −d log α/dr`), so the lapse
# inside is `α = α_s/h` with the surface's `α_s = √(1 − 2M/R)`. The unknowns
# are chosen smooth in `η` at the centre, where `r ∝ √(η_c − η)`:
#
#   * `q = r²` (`r` the areal radius): `dq/dη = −2 (1 − 2μq)/(μ + 4πp)`;
#   * `μ = m/r³`: `dμ/dη = (dq/dη) (4πe − 3μ)/(2q)`, at the centre
#     `(4π/5) de/dη`;
#   * `λ = log(r̄/r)` (`r̄` the isotropic radius, `dr̄/r̄ = dr/(r√(1 − 2m/r))`):
#     `dλ/dη = (dq/dη) μ/(s(1 + s))`, `s = √(1 − 2μq)`, with no cancellation.
#
# Classical RK4 in `η` from the centre (`q = 0`, `μ = 4πe_c/3`, `λ = 0`), then
# `λ` is shifted so that the surface matches the exterior's isotropic
# Schwarzschild radius `r̄_s = (R − M + √(R² − 2MR))/2`.
#
# **The metric's storage** (proposed in step 10). `ψ` and `α` as functions of
# `u = r̄² = x² + y² + z²`, in which both are smooth at the centre (the star is
# even in `r̄`), tabulated with their `u`-derivatives at `NK` points uniform in
# `u` over `[0, r̄_s²]` and evaluated by cubic Hermite interpolation; outside
# the surface the closed form `ψ = 1 + M/(2r̄)`, `α = (1 − M/(2r̄))/(1 +
# M/(2r̄))`. A table in `u` needs no square root at the centre — where a
# vertex sits on a symmetric mesh, and where ForwardDiff's derivative of `√u`
# is not finite — and the interior's knots end at the surface, so the
# interpolant never straddles it. The table is `SVector`s, so the metric is
# `isbits`: a kernel argument by value, `4 NK` floats (4.1 kB at `NK = 129`
# in `Float64`, the default), which a device's parameter space must hold
# (step 14). The density is not tabulated: `ρ = ρ(h)` with `h = α_s/α`, the
# polytrope's own relation, so the initial data and the metric are one
# solution.

"""
    Polytrope{T}(K, Γ)

The cold isentrope `p = K ρ^Γ` of an ideal gas, with `e = ρ (1 + ε)`,
`ε = p/((Γ − 1) ρ)`, as functions of the specific enthalpy `h = 1 + Γ/(Γ −
1) K ρ^{Γ − 1}` ([`polytrope_state`](@ref)); `isbits`. EntropyEOS's
`IdealGasEOS` at `polytropic_entropy(eos, K)` is the same gas.
"""
struct Polytrope{T}
    K::T
    Γ::T
end

"""
    polytrope_state(eq::Polytrope, h) -> (; ρ, p, e, dρ_dh)

The polytrope's rest-mass density, pressure and total energy density at
the specific enthalpy `h ≥ 1`, and `dρ/dh`; zero density at `h ≤ 1`.
"""
@inline function polytrope_state(eq::Polytrope, h)
    n = inv(eq.Γ - 1)
    θ = max((h - 1) / (n + 1), zero(h))              # p/ρ = K ρ^(Γ−1)
    ρ = (θ / eq.K)^n
    p = ρ * θ
    e = ρ + n * p
    dρ_dh = n * (θ / eq.K)^(n - 1) / (eq.K * (n + 1))
    return (; ρ, p, e, dρ_dh)
end

"""
    polytrope_enthalpy(eq::Polytrope, ρ) -> h
"""
@inline polytrope_enthalpy(eq::Polytrope, ρ) =
    1 + eq.Γ / (eq.Γ - 1) * eq.K * ρ^(eq.Γ - 1)

# The right-hand side `d(q, μ, λ)/dη` of the enthalpy form (top of the file).
@inline function tov_rhs(eq::Polytrope{T}, η::T, q::T, μ::T, λ::T) where {T}
    h = exp(η)
    st = polytrope_state(eq, h)
    s² = 1 - 2 * μ * q
    s = sqrt(s²)
    dq = -2 * s² / (μ + 4 * T(π) * st.p)
    dμ = q == 0 ? 4 * T(π) / 5 * (h^2 * st.dρ_dh) :
         dq * (4 * T(π) * st.e - 3 * μ) / (2 * q)
    dλ = dq * μ / (s * (1 + s))
    return SVector(dq, dμ, dλ)
end

"""
    TOVSolution{T}

The star solved on the host ([`tov_solve`](@ref)): the polytrope, the
gravitational mass `M`, the areal radius `R`, the isotropic radius `R̄`, the
surface lapse `α_s = √(1 − 2M/R)`, and the RK4 nodes in `η` (`η[1] = η_c`,
`η[end] = 0`) with `q = r²`, `μ = m/r³`, `λ = log(r̄/r)` (already matched to
the exterior) and their `η`-derivatives.
"""
struct TOVSolution{T}
    eq::Polytrope{T}
    ρ_c::T
    M::T
    R::T
    R̄::T
    α_s::T
    η::Vector{T}
    y::Vector{SVector{3,T}}
    dy::Vector{SVector{3,T}}
end

"""
    tov_solve([T]; K, Γ, ρ_c, steps = 4000) -> TOVSolution{T}

The TOV star of the polytrope `p = K ρ^Γ` with central density `ρ_c`, by
classical RK4 in the log-enthalpy with `steps` equal steps from the centre to
the surface (Lindblom's form; the top of `src/tov.jl`). Fourth order in the
step; at the default the mass and radius are converged to about 10⁻¹⁴.
"""
function tov_solve(::Type{T}=Float64; K, Γ, ρ_c, steps::Integer=4000) where {T}
    T(K) > 0 && T(Γ) > 1 && T(ρ_c) > 0 || throw(ArgumentError(
        "a polytropic star needs K > 0, Γ > 1 and ρ_c > 0, got K = $K, Γ = $Γ, ρ_c = $ρ_c."))
    steps ≥ 2 || throw(ArgumentError("the solve needs at least two steps, got $steps."))
    eq = Polytrope{T}(T(K), T(Γ))
    η_c = log(polytrope_enthalpy(eq, T(ρ_c)))
    e_c = polytrope_state(eq, exp(η_c)).e
    η = Vector{T}(undef, steps + 1)
    y = Vector{SVector{3,T}}(undef, steps + 1)
    dy = Vector{SVector{3,T}}(undef, steps + 1)
    f(η, y) = tov_rhs(eq, η, y[1], y[2], y[3])
    Δ = -η_c / steps
    η[1] = η_c
    y[1] = SVector(zero(T), 4 * T(π) * e_c / 3, zero(T))
    for i in 1:steps
        ηᵢ = η[i]
        k₁ = f(ηᵢ, y[i])
        k₂ = f(ηᵢ + Δ / 2, y[i] + Δ / 2 * k₁)
        k₃ = f(ηᵢ + Δ / 2, y[i] + Δ / 2 * k₂)
        k₄ = f(ηᵢ + Δ, y[i] + Δ * k₃)
        dy[i] = k₁
        y[i + 1] = y[i] + Δ / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
        η[i + 1] = i == steps ? zero(T) : η_c + i * Δ
    end
    dy[end] = f(η[end], y[end])
    q_s, μ_s, λ_s = y[end]
    R = sqrt(q_s)
    M = μ_s * R^3
    R̄ = (R - M + sqrt(R^2 - 2 * M * R)) / 2
    C = log(R̄ / R) - λ_s
    for i in eachindex(y)
        y[i] = SVector(y[i][1], y[i][2], y[i][3] + C)
    end
    return TOVSolution{T}(eq, T(ρ_c), M, R, R̄, sqrt(1 - 2 * M / R), η, y, dy)
end

# Cubic Hermite interpolation on `[0, 1]` in the local coordinate `t`, of the
# values `f₀`, `f₁` and the derivatives `d₀`, `d₁` already scaled by the
# interval's width.
@inline function hermite(t, f₀, d₀, f₁, d₁)
    t² = t * t
    t³ = t² * t
    return (2 * t³ - 3 * t² + 1) * f₀ + (t³ - 2 * t² + t) * d₀ +
           (-2 * t³ + 3 * t²) * f₁ + (t³ - t²) * d₁
end
@inline function hermite_deriv(t, f₀, d₀, f₁, d₁)
    t² = t * t
    return (6 * t² - 6 * t) * f₀ + (3 * t² - 4 * t + 1) * d₀ +
           (-6 * t² + 6 * t) * f₁ + (3 * t² - 2 * t) * d₁
end

# The solution's state `(q, μ, λ)` at `η` on step `i` (`η[i] ≥ η ≥ η[i+1]`),
# by cubic Hermite interpolation in `η` of the RK4 nodes and their
# right-hand sides: fourth order, like the solve.
@inline function tov_state(sol::TOVSolution{T}, i::Int, η::T) where {T}
    a, b = sol.η[i], sol.η[i + 1]
    w = b - a
    t = (η - a) / w
    return hermite(t, sol.y[i], w * sol.dy[i], sol.y[i + 1], w * sol.dy[i + 1])
end

# `u = r̄² = q e^{2λ}` and `du/dη` at a state.
@inline function tov_u(sol::TOVSolution{T}, η::T, y::SVector{3,T}) where {T}
    q, μ, λ = y
    d = tov_rhs(sol.eq, η, q, μ, λ)
    e2 = exp(2 * λ)
    return q * e2, e2 * (d[1] + 2 * q * d[3]), d
end

# The log-enthalpy at which `r̄² = u`, inside the star: the RK4 step that
# brackets it, then Newton on the step's Hermite interpolant, safeguarded by
# bisection. `u(η)` decreases from `r̄_s²` at `η = 0` to 0 at `η_c`.
function tov_eta(sol::TOVSolution{T}, u::T) where {T}
    u ≤ 0 && return sol.η[1], 1
    us = [tov_u(sol, sol.η[i], sol.y[i])[1] for i in eachindex(sol.η)]
    i = clamp(searchsortedlast(us, u), 1, length(us) - 1)
    lo, hi = sol.η[i + 1], sol.η[i]                  # u(lo) ≥ u ≥ u(hi)
    η = (lo + hi) / 2
    for _ in 1:100
        uη, du, _ = tov_u(sol, η, tov_state(sol, i, η))
        r = uη - u
        r == 0 && break
        r > 0 ? (lo = η) : (hi = η)
        η_new = η - r / du
        η_new = lo < η_new < hi ? η_new : (lo + hi) / 2
        abs(η_new - η) ≤ 4 * eps(T) * max(abs(η), one(T)) && (η = η_new; break)
        η = η_new
    end
    return η, i
end

"""
    TOVMetric{T,NK} <: SpacetimeMetrics.AbstractMetric

The static metric of a TOV star in isotropic coordinates, `ds² = −α² dt² +
ψ⁴ δ_ij dxⁱ dxʲ` centred at the origin: `ψ` and `α` tabulated in `u = r̄²`
with their derivatives at `NK` uniform knots over `[0, R̄²]` and interpolated
by cubic Hermite inside the surface, the isotropic Schwarzschild exterior
outside (the top of `src/tov.jl`). `isbits` and generic over dual numbers, so
an `Analytic` provider and SpacetimeMetrics' `dmetric` take it as they take
`KerrSchild`. Built by [`TOVMetric(sol)`](@ref TOVMetric(::TOVSolution)); see
"Gauge for the TOV milestones" in `CODE.md`.
"""
struct TOVMetric{T,NK} <: SpacetimeMetrics.AbstractMetric
    eq::Polytrope{T}
    M::T
    R̄::T
    α_s::T
    Δu::T
    ψ::SVector{NK,T}
    dψ::SVector{NK,T}
    α::SVector{NK,T}
    dα::SVector{NK,T}
end

Base.nameof(m::TOVMetric) = "TOV star (isotropic), M = $(m.M), R̄ = $(m.R̄)"

"""
    TOVMetric(sol::TOVSolution; NK = 129) -> TOVMetric

The solution's metric tabulated at `NK` knots uniform in `r̄²`.
"""
function TOVMetric(sol::TOVSolution{T}; NK::Integer=129) where {T}
    NK ≥ 2 || throw(ArgumentError("the table needs at least two knots, got NK = $NK."))
    us = sol.R̄^2
    Δu = us / (NK - 1)
    ψ = Vector{T}(undef, NK)
    dψ = similar(ψ)
    α = similar(ψ)
    dα = similar(ψ)
    for k in 1:NK
        u = k == NK ? us : (k - 1) * Δu
        η, i = k == NK ? (zero(T), length(sol.η) - 1) : tov_eta(sol, u)
        y = k == NK ? sol.y[end] : tov_state(sol, i, η)
        _, du, d = tov_u(sol, η, y)
        ψ[k] = exp(-y[3] / 2)                          # ψ² = r/r̄ = e^{−λ}
        dψ[k] = -ψ[k] * d[3] / 2 / du
        α[k] = sol.α_s * exp(-η)
        dα[k] = -α[k] / du
    end
    return TOVMetric{T,NK}(sol.eq, sol.M, sol.R̄, sol.α_s, Δu, SVector{NK}(ψ), SVector{NK}(dψ),
                           SVector{NK}(α), SVector{NK}(dα))
end

"""
    tov_potentials(m::TOVMetric, u) -> (ψ, α)

The conformal factor and the lapse at `u = r̄²`, generic over dual numbers.
"""
@inline function tov_potentials(m::TOVMetric{T,NK}, u) where {T,NK}
    if u < m.R̄^2
        ξ = u / m.Δu
        k = clamp(floor(Int, ξ), 0, NK - 2)
        t = ξ - k
        Δ = m.Δu
        ψ = @inbounds hermite(t, m.ψ[k + 1], Δ * m.dψ[k + 1], m.ψ[k + 2], Δ * m.dψ[k + 2])
        α = @inbounds hermite(t, m.α[k + 1], Δ * m.dα[k + 1], m.α[k + 2], Δ * m.dα[k + 2])
        return ψ, α
    else
        a = m.M / (2 * sqrt(u))
        return 1 + a, (1 - a) / (1 + a)
    end
end

function SpacetimeMetrics.metric(m::TOVMetric, x::AbstractVector)
    u = x[2]^2 + x[3]^2 + x[4]^2
    ψ, α = tov_potentials(m, u)
    ψ² = ψ * ψ
    g = ψ² * ψ²
    z = zero(g)
    return SMatrix{4,4}(-α * α, z, z, z, z, g, z, z, z, z, g, z, z, z, z, g)
end

"""
    tov_density(m::TOVMetric, u) -> ρ

The star's rest-mass density at `u = r̄²`: the polytrope's `ρ(h)` at the
enthalpy `h = α_s/α` of hydrostatic equilibrium (`h α` constant), zero
outside the surface.
"""
@inline function tov_density(m::TOVMetric, u)
    u < m.R̄^2 || return zero(u)
    _, α = tov_potentials(m, u)
    return polytrope_state(m.eq, m.α_s / α).ρ
end
