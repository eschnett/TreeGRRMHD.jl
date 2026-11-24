# The cold barotrope of a tabulated EOS, for the TOV star of rung 7
# ("Validation ladder" 7 in `CODE.md`; step 13 of the plan in `HISTORY.md`).
#
# EntropyEOS's cold β-equilibrium slice (EOS-4, `cold_beta_slice`) gives the
# state of least Gibbs free energy at each pressure on the table's coldest
# row: `ρ★`, `Yₑ`, `s` and the EOS there, in the table's units. The solve of
# `src/tov.jl` wants it as a function of
#
#     η = ∫_{p_s}^{p} dp/(e + p),
#
# zero at the surface pressure `p_s`, because a static fluid has `η + log α`
# constant (`dp = −(e + p) d log α`), whatever the composition does. For the
# polytrope `η = log h`; along the slice `Yₑ` and `s` change, so `log h` is
# not this integral (by 1.8·10⁻⁴ between 10⁸ and 10¹⁵ g/cm³ on DD2), and only
# the integral makes the evolved fluid static.
#
# **The tabulation.** `samples` pressures uniform in `log p` from `p_s` to the
# slice's top, each looked up on the slice (`beta_equilibrium(slice; p)`),
# converted to geometric units (`ρ★` and `p` divided by the density unit, as
# `GeometricEOS` does; `e = ρ★ h − p`). `η` is integrated in `log p` by
# Simpson's rule, `dη = p/(e + p) d log p`, so every second sample is a knot,
# and the derivatives at a knot come from the samples beside it. Between knots
# the state is cubic Hermite in `η`, with `d log p/dη = (e + p)/p` exact.
#
# **The composition.** What a kernel needs of the star's matter — `ρ★`, `s`
# and `Yₑ` — is a [`TOVComposition`](@ref), `isbits`, cubic Hermite at
# uniform knots in `η`, kept apart from the [`TOVMetric`](@ref) so that the
# metric's kernel arguments do not grow; `η = log(α_s/α)` at a point comes from
# the metric.

"""
    ColdBarotrope

A table's cold β-equilibrium slice in geometric units as a function of `η =
∫ dp/(e + p)` from the surface ([`cold_barotrope`](@ref), `src/tov_table.jl`):
knots `η` (increasing from 0), and at each the pressure, the total energy
density, the rescaled density `ρ★`, the entropy and the electron fraction,
with their `η`-derivatives. Host-side, `Float64`.
"""
struct ColdBarotrope
    η::Vector{Float64}
    logp::Vector{Float64}
    e::Vector{Float64}
    ρ::Vector{Float64}
    s::Vector{Float64}
    Yₑ::Vector{Float64}
    de::Vector{Float64}
    dρ::Vector{Float64}
    ds::Vector{Float64}
    dYₑ::Vector{Float64}
    ρ_surface::Float64
    density::Float64
end

"""
    cold_barotrope(slice::EntropyEOS.ColdBetaSlice; ρ_surface,
                   density = MSUN_DENSITY_CGS, samples = 8001) -> ColdBarotrope

The slice as the TOV solve reads it, with the surface `η = 0` at the
geometric rescaled density `ρ_surface` (no default: the star's matter below it
is left out). `density` is the table's density per geometric unit, as for
[`GeometricEOS`](@ref). A slice with a phase transition is not supported yet
(`ArgumentError`): the density would jump at one `η`.
"""
function cold_barotrope(slice::EntropyEOS.ColdBetaSlice; ρ_surface, density=MSUN_DENSITY_CGS,
                        samples::Integer=8001)
    isempty(slice.jumps) || throw(ArgumentError(
        "the slice has $(length(slice.jumps)) phase transition(s), where the density jumps " *
        "at constant pressure; the tabulated star does not support them yet."))
    samples ≥ 5 && isodd(samples) || throw(ArgumentError(
        "Simpson's rule needs an odd number of samples, at least 5; got $samples."))
    f = Float64(density)
    ρ★_s = Float64(ρ_surface) * f
    slice.ρ★[1] ≤ ρ★_s < slice.ρ★[end] || throw(ArgumentError(
        "the surface density $ρ_surface lies outside the slice's densities " *
        "[$(slice.ρ★[1] / f), $(slice.ρ★[end] / f)] (geometric)."))
    p_s = EntropyEOS.beta_equilibrium(slice; ρ★=ρ★_s).point.p
    x = range(log(p_s), log(slice.p[end]); length=samples)
    p_top = slice.p[end]
    sts = [EntropyEOS.beta_equilibrium(slice; p=clamp(exp(xᵢ), p_s, p_top)) for xᵢ in x]
    p = [st.point.p / f for st in sts]
    ρ = [st.ρ★ / f for st in sts]
    e = [(st.ρ★ * st.point.h - st.point.p) / f for st in sts]
    s = [st.s for st in sts]
    Y = [st.yₑ for st in sts]
    issorted(ρ; lt=(<=)) && issorted(e; lt=(<=)) || throw(ArgumentError(
        "the slice's density or energy density does not rise with the pressure above " *
        "ρ_surface = $ρ_surface."))
    g = p ./ (e .+ p)                                  # dη/d log p
    δ = step(x)
    K = (samples + 1) ÷ 2                              # knots at the odd samples
    kn(k) = 2k - 1
    η = zeros(K)
    for k in 2:K
        i = kn(k)
        η[k] = η[k - 1] + δ / 3 * (g[i - 2] + 4 * g[i - 1] + g[i])
    end
    # d/d log p at the knots from the samples beside them (second order),
    # then d/dη = (e + p)/p d/d log p.
    function deriv(v, i)
        i == 1 && return (-3 * v[1] + 4 * v[2] - v[3]) / (2δ)
        i == samples && return (3 * v[i] - 4 * v[i - 1] + v[i - 2]) / (2δ)
        return (v[i + 1] - v[i - 1]) / (2δ)
    end
    idx = kn.(1:K)
    dη(v) = [deriv(v, i) / g[i] for i in idx]
    return ColdBarotrope(η, log.(p[idx]), e[idx], ρ[idx], s[idx], Y[idx], dη(e), dη(ρ), dη(s),
                         dη(Y), Float64(ρ_surface), f)
end

# The knot interval holding `η` and the local coordinate in it.
@inline function barotrope_interval(b::ColdBarotrope, η::Float64)
    n = length(b.η)
    k = clamp(searchsortedlast(b.η, η), 1, n - 1)
    w = b.η[k + 1] - b.η[k]
    return k, w, (η - b.η[k]) / w
end

"""
    barotrope_state(b::ColdBarotrope, η) -> (; p, e, de_dη, ρ, s, Yₑ, dρ_dη, ds_dη, dYₑ_dη)

The barotrope at `η` (`Float64`), cubic Hermite between the knots.
"""
function barotrope_state(b::ColdBarotrope, η::Real)
    η = Float64(η)
    k, w, t = barotrope_interval(b, η)
    H(v, d) = hermite(t, v[k], w * d[k], v[k + 1], w * d[k + 1])
    D(v, d) = hermite_deriv(t, v[k], w * d[k], v[k + 1], w * d[k + 1]) / w
    dlogp(j) = (b.e[j] + exp(b.logp[j])) / exp(b.logp[j])
    logp = hermite(t, b.logp[k], w * dlogp(k), b.logp[k + 1], w * dlogp(k + 1))
    return (; p=exp(logp), e=H(b.e, b.de), de_dη=D(b.e, b.de), ρ=H(b.ρ, b.dρ),
            s=H(b.s, b.ds), Yₑ=H(b.Yₑ, b.dYₑ), dρ_dη=D(b.ρ, b.dρ), ds_dη=D(b.s, b.ds),
            dYₑ_dη=D(b.Yₑ, b.dYₑ))
end

"""
    barotrope_eta(b::ColdBarotrope, ρ) -> η

The `η` at which the barotrope's rescaled density is `ρ` (geometric), by
bisection on the interpolant.
"""
function barotrope_eta(b::ColdBarotrope, ρ::Real)
    ρ = Float64(ρ)
    b.ρ[1] ≤ ρ ≤ b.ρ[end] || throw(ArgumentError(
        "the density $ρ lies outside the barotrope's [$(b.ρ[1]), $(b.ρ[end])]."))
    lo, hi = b.η[1], b.η[end]
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        barotrope_state(b, mid).ρ < ρ ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

metric_eq(::ColdBarotrope) = nothing

"""
    tov_solve(T, b::ColdBarotrope; ρ_c, steps = 4000) -> TOVSolution{T,ColdBarotrope}

The TOV star of a table's cold barotrope ([`cold_barotrope`](@ref)) with
central rescaled density `ρ_c` (geometric), by the same RK4 in `η` as the
polytrope's. The barotrope is evaluated in `Float64`.
"""
function tov_solve(::Type{T}, b::ColdBarotrope; ρ_c, steps::Integer=4000) where {T}
    steps ≥ 2 || throw(ArgumentError("the solve needs at least two steps, got $steps."))
    η_c = barotrope_eta(b, ρ_c)
    e_c = barotrope_state(b, η_c).e
    return tov_integrate(b, T(ρ_c), T(η_c), T(e_c), steps)
end

"""
    TOVComposition{T,NC}

What a kernel needs of a tabulated star's matter, `isbits`: `log ρ★`, the
entropy `s` and the electron fraction `Yₑ` with their `u`-derivatives at `NC`
knots uniform in `u = r̄²` over `[0, R̄²]`, as the metric's `ψ` and `α` are
([`TOVMetric`](@ref)), interpolated by cubic Hermite
([`composition_state`](@ref); `src/tov_table.jl`). Uniform in `u` rather than
in `η`, the crust (where `ρ★` falls by decades in a few per cent of the
radius) gets knots in proportion to its volume; `log ρ★` is smooth there.
"""
struct TOVComposition{T,NC}
    R̄²::T
    Δu::T
    logρ::SVector{NC,T}
    dlogρ::SVector{NC,T}
    s::SVector{NC,T}
    ds::SVector{NC,T}
    Yₑ::SVector{NC,T}
    dYₑ::SVector{NC,T}
end

"""
    TOVComposition(sol::TOVSolution{T,ColdBarotrope}; NC = 129) -> TOVComposition{T,NC}

The solution's matter at `NC` knots uniform in `r̄²`: at each the `η` of the
solve ([`tov_eta`](@ref)), the barotrope there, and `d/du = (dη/du) d/dη`.
"""
function TOVComposition(sol::TOVSolution{T,ColdBarotrope}; NC::Integer=129) where {T}
    NC ≥ 2 || throw(ArgumentError("the table needs at least two knots, got NC = $NC."))
    us = sol.R̄^2
    Δu = us / (NC - 1)
    logρ, dlogρ, s, ds, Y, dY = (Vector{Float64}(undef, NC) for _ in 1:6)
    for k in 1:NC
        u = k == NC ? us : (k - 1) * Δu
        η, i = k == NC ? (zero(T), length(sol.η) - 1) : tov_eta(sol, u)
        y = k == NC ? sol.y[end] : tov_state(sol, i, η)
        _, du, _ = tov_u(sol, η, y)
        st = barotrope_state(sol.eq, η)
        dη_du = 1 / Float64(du)
        logρ[k] = log(st.ρ)
        dlogρ[k] = st.dρ_dη / st.ρ * dη_du
        s[k], ds[k] = st.s, st.ds_dη * dη_du
        Y[k], dY[k] = st.Yₑ, st.dYₑ_dη * dη_du
    end
    V(v) = SVector{NC,T}(T.(v))
    return TOVComposition{T,NC}(T(us), T(Δu), V(logρ), V(dlogρ), V(s), V(ds), V(Y), V(dY))
end

"""
    composition_state(c::TOVComposition, u) -> (ρ★, s, Yₑ)

The star's matter at `u = r̄² < R̄²`, `u ≥ 0`.
"""
@inline function composition_state(c::TOVComposition{T,NC}, u::T) where {T,NC}
    ξ = u / c.Δu
    k = clamp(floor(Int, ξ), 0, NC - 2)
    t = ξ - k
    Δ = c.Δu
    lρ = @inbounds hermite(t, c.logρ[k + 1], Δ * c.dlogρ[k + 1], c.logρ[k + 2], Δ * c.dlogρ[k + 2])
    s = @inbounds hermite(t, c.s[k + 1], Δ * c.ds[k + 1], c.s[k + 2], Δ * c.ds[k + 2])
    Y = @inbounds hermite(t, c.Yₑ[k + 1], Δ * c.dYₑ[k + 1], c.Yₑ[k + 2], Δ * c.dYₑ[k + 2])
    return exp(lρ), s, Y
end
