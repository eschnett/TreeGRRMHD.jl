# The advection of the electron fraction (step 13 of `PLAN.md`): a uniform
# fluid moving at `v` along `x` through the periodic `[0, L]`, carrying a
# profile of `Yₑ` that nothing else sees. On an EOS without `Yₑ` dependence
# (the ideal gas) `Yₑ` is a passive scalar, `∂ₜ(√γ D Yₑ) + ∂ₓ(√γ D Yₑ vˣ) = 0`
# with `D` uniform, so the exact solution is the profile shifted by `v t`, and
# the density, the pressure and the velocity stay exactly uniform.
#
# Two profiles: `:smooth`, `Yₑ = Y₀ + δY sin(2πx/L)`, for the convergence
# order; `:step`, `Y₀ + δY` on the middle half and `Y₀ − δY` outside, for the
# bounds the limiter keeps. One period is `t = L/v`.

"""
    YeWave{T}

The parameters of the `Yₑ` advection, `isbits`: the fluid's `ρ`, `s` and
velocity `v` along `x`, the profile's mean `Y₀` and amplitude `δY`, the
period `L`, and whether the profile is the step. See
`src/cases/ye_advection.jl`.
"""
struct YeWave{T}
    ρ::T
    s::T
    v::T
    Y₀::T
    δY::T
    L::T
    step::Bool
end

@inline function wave_fields(w::YeWave, x::SVector{3,T}, t::T) where {T}
    ξ = mod(x[1] - w.v * t, w.L) / w.L                # in [0, 1)
    Yₑ = if w.step
        abs(ξ - T(1 // 2)) < T(1 // 4) ? w.Y₀ + w.δY : w.Y₀ - w.δY
    else
        w.Y₀ + w.δY * sinpi(2 * ξ)
    end
    z = zero(SVector{3,T})
    return w.ρ, SVector(w.v, zero(T), zero(T)), w.s, Yₑ, z, z
end

@inline wave_potential(::YeWave, x::SVector{3,T}, t::T) where {T} = zero(SVector{3,T}), zero(T)

"""
    ye_advection_case([T]; profile = :smooth, v = 1/2, ρ = 1, p = 1, Γ = 5/3, Y₀ = 3/10,
                      δY = 1/10, L = 1, roots = 4)

The advection of a `Yₑ` profile (`:smooth` or `:step`) by a uniform fluid of
density `ρ`, pressure `p` and velocity `v` along `x`, as a one-dimensional
periodic [`MHDCase`](@ref) on `[0, L]` on the ideal gas [`case_gas`](@ref),
with no field (`IdealConductor`). `Yₑ` must stay in the gas's range `(1/20,
3/5)`. One period is `t = L/v`; the exact solution is the profile shifted.
"""
function ye_advection_case(::Type{T}=Float64; profile::Symbol=:smooth, v=1 // 2, ρ=1, p=1,
                           Γ=5 // 3, Y₀=3 // 10, δY=1 // 10, L=1, roots=4) where {T}
    profile in (:smooth, :step) || throw(ArgumentError(
        "the profile is :smooth (the convergence order) or :step (the bounds); got $profile."))
    0 < abs(T(v)) < 1 || throw(ArgumentError("the fluid's speed must lie in (0, 1), got $v."))
    eos = case_gas(T, Γ)
    lo, hi = EntropyEOS.yₑ_bounds(eos)
    lo < T(Y₀) - abs(T(δY)) && T(Y₀) + abs(T(δY)) < hi || throw(ArgumentError(
        "the profile Y₀ ± δY = $Y₀ ± $δY must lie inside the gas's Yₑ range ($lo, $hi)."))
    s = EntropyEOS.polytropic_entropy(eos, T(p) / T(ρ)^T(Γ))
    w = YeWave{T}(T(ρ), s, T(v), T(Y₀), T(δY), T(L), profile === :step)
    return MHDCase(T, Val(1); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   recovery_options(eos, T(1 // 10^9))..., model=IdealConductor(),
                   periodic=true, extents=((zero(T), T(L)),), roots=roots)
end
