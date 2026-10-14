# The vacuum electromagnetic wave: rung 1's first case ("Validation ladder"
# in `CODE.md`). A plane wave of both polarizations travelling along `x` at
# the speed of light through a fluid at rest that it does not touch:
#
#     φ = k (x − t),   E = (0, a sin φ, b sin φ),   B = (0, −b sin φ, a sin φ)
#
# from the potential `A = (0, −(a/k) cos φ, −(b/k) cos φ)`, `Φ̃ = 0` — the
# Lorenz gauge, `∂ₜΦ̃ = −∂ⱼAʲ = 0`. With `σ = 0` (`Vacuum`) and `q = 0` there
# is no current, so the field carries its own energy and momentum and the
# fluid stays at rest: the totals `S_i` and `τ` hold the fluid's plus the
# field's, and the solution is exact for the whole system.

"""
    WaveFields(w)
    WavePotential(w)

A case's `fields` and `potential` callbacks for the wave parameters `w`
(`isbits`, so kernel arguments): `WaveFields(w)(x, t)` is
`wave_fields(w, x, t)` and `WavePotential(w)(x, t)` is
`wave_potential(w, x, t)`, each case adding its methods. Callable structs
rather than closures so that a case's parameters stay readable from the
case (`case.fields.w`).
"""
struct WaveFields{W}
    w::W
end
struct WavePotential{W}
    w::W
end
@inline (f::WaveFields)(x, t) = wave_fields(f.w, x, t)
@inline (f::WavePotential)(x, t) = wave_potential(f.w, x, t)

"""
    case_gas(T, Γ) -> IdealGasEOS{T}

The ideal gas the flat-space waves run on, in geometric units: `p = K(s) ρ^Γ`
with `K(s) = exp((Γ − 1)(s − 1))`, so that `s = 1` is `p = ρ^Γ`; the entropy
window `(1/2, 3/2)`, the density box `(10⁻¹⁰, 10⁴)`, `Yₑ ∈ (0.05, 0.6)`.
"""
case_gas(::Type{T}, Γ) where {T} =
    EntropyEOS.IdealGasEOS{T}(; Γ=T(Γ), K_ref=one(T), s_ref=one(T),
                              s_window=(T(1 // 2), T(3 // 2)), ρ_bounds=(T(1 // 10^10), T(10^4)),
                              yₑ_bounds=(T(1 // 20), T(3 // 5)))

"""
    EMWave{T}(; a, b, k, ρ, s, Yₑ)

The parameters of the vacuum electromagnetic wave, `isbits`: the amplitudes
`a` of `Eʸ = Bᶻ` and `b` of `Eᶻ = −Bʸ`, the wavenumber `k`, and the resting
fluid's `ρ`, `s` and `Yₑ`. See `src/cases/em_wave.jl`.
"""
struct EMWave{T}
    a::T
    b::T
    k::T
    ρ::T
    s::T
    Yₑ::T
end
EMWave{T}(; a, b, k, ρ, s, Yₑ) where {T} = EMWave{T}(T(a), T(b), T(k), T(ρ), T(s), T(Yₑ))

@inline function wave_fields(w::EMWave, x::SVector{3,T}, t::T) where {T}
    sφ = sin(w.k * (x[1] - t))
    z = zero(T)
    return w.ρ, zero(SVector{3,T}), w.s, w.Yₑ, SVector(z, w.a * sφ, w.b * sφ),
           SVector(z, -w.b * sφ, w.a * sφ)
end

@inline function wave_potential(w::EMWave, x::SVector{3,T}, t::T) where {T}
    cφ = cos(w.k * (x[1] - t))
    return SVector(zero(T), -w.a / w.k * cφ, -w.b / w.k * cφ), zero(T)
end

"""
    em_wave_case([T]; a = 1/10, b = 1/20, ρ = 1, p = 1, Γ = 5/3, L = 1, roots = 5)

The vacuum electromagnetic wave as a one-dimensional periodic [`MHDCase`](@ref)
on `[0, L]`, one wavelength, with `Vacuum` conductivity, flat space, and a
resting ideal gas of density `ρ` and pressure `p` beneath it (the field's
energy is `(a² + b²)/2` of the fluid's `ρ`-scale, so the fluid barely notices
the numerical exchange). One period is `t = L`.
"""
function em_wave_case(::Type{T}=Float64; a=1 // 10, b=1 // 20, ρ=1, p=1, Γ=5 // 3, L=1,
                      roots=5) where {T}
    eos = case_gas(T, Γ)
    s = EntropyEOS.polytropic_entropy(eos, T(p) / T(ρ)^T(Γ))
    w = EMWave{T}(; a=a, b=b, k=2 * T(π) / T(L), ρ=ρ, s=s, Yₑ=T(1 // 2))
    return MHDCase(T, Val(1); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(1 // 10^9)), model=Vacuum(),
                   periodic=true, extents=((zero(T), T(L)),), roots=roots)
end
