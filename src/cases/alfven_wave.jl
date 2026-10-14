# The large-amplitude circularly polarized Alfvén wave: rung 1's ideal-MHD
# smooth case ("Validation ladder" in `CODE.md`), an exact nonlinear solution
# of special-relativistic ideal MHD (Del Zanna et al. 2007, the ECHO paper,
# §4.1; GRaM-X's Alfvén test). On a guide field `B₀` along `x`, with
# `φ = k (x − v_A t)`,
#
#     B = (B₀, η B₀ cos φ, η B₀ sin φ),   v = −v_A (0, Bʸ, Bᶻ)/B₀,
#     E = −v × B = v_A η B₀ (0, sin φ, −cos φ),
#
# at uniform `ρ` and `p`, and the speed
#
#     v_A² = B₀²/(ρh + B₀²(1 + η²)) · [½ (1 + √(1 − (2ηB₀²/(ρh + B₀²(1 + η²)))²))]⁻¹.
#
# The guide field is the background `𝔅̄ = (B₀, 0, 0)`, which no
# translation-invariant `A` can produce (`src/curl.jl`), and the transverse
# field is the curl of `A = −(ηB₀/k)(0, cos φ, sin φ)`, `Φ̃ = 0`. The
# conductivity is `IdealConductor`, so every implicit stage is the ideal
# recovery and `E = −v × B` exactly (step 3).

"""
    AlfvenWave{T}(; B0, η, k, vA, ρ, s, Yₑ)

The parameters of the circularly polarized Alfvén wave, `isbits`. See
`src/cases/alfven_wave.jl` and [`alfven_speed`](@ref).
"""
struct AlfvenWave{T}
    B0::T
    η::T
    k::T
    vA::T
    ρ::T
    s::T
    Yₑ::T
end
AlfvenWave{T}(; B0, η, k, vA, ρ, s, Yₑ) where {T} =
    AlfvenWave{T}(T(B0), T(η), T(k), T(vA), T(ρ), T(s), T(Yₑ))

"""
    alfven_speed(ρh, B0, η) -> v_A

The speed of the circularly polarized Alfvén wave of amplitude `η` on the
guide field `B0` in a fluid of enthalpy density `ρh` (Del Zanna et al. 2007,
eq. 54). For `η → 0` it is the relativistic Alfvén speed
`B0/√(ρh + B0²)`.
"""
function alfven_speed(ρh, B0, η)
    X = ρh + B0^2 * (1 + η^2)
    return sqrt(B0^2 / X / ((1 + sqrt(1 - (2 * η * B0^2 / X)^2)) / 2))
end

@inline function wave_fields(w::AlfvenWave, x::SVector{3,T}, t::T) where {T}
    φ = w.k * (x[1] - w.vA * t)
    sφ, cφ = sin(φ), cos(φ)
    ηB = w.η * w.B0
    v = SVector(zero(T), -w.vA * w.η * cφ, -w.vA * w.η * sφ)
    E = SVector(zero(T), w.vA * ηB * sφ, -w.vA * ηB * cφ)
    B = SVector(w.B0, ηB * cφ, ηB * sφ)
    return w.ρ, v, w.s, w.Yₑ, E, B
end

@inline function wave_potential(w::AlfvenWave, x::SVector{3,T}, t::T) where {T}
    φ = w.k * (x[1] - w.vA * t)
    a = w.η * w.B0 / w.k
    return SVector(zero(T), -a * cos(φ), -a * sin(φ)), zero(T)
end

"""
    alfven_wave_case([T]; ρ = 1, p = 1, Γ = 4/3, B0 = 1, η = 1, L = 1, roots = 5)

The circularly polarized Alfvén wave as a one-dimensional periodic
[`MHDCase`](@ref) on `[0, L]`, one wavelength, with `IdealConductor`, flat
space and an ideal gas; Del Zanna et al.'s parameters by default
(`ρ = p = B0 = η = 1`, `Γ = 4/3`, so `v_A ≈ 0.3820`). One period is
`t = L/v_A`, [`alfven_period`](@ref).
"""
function alfven_wave_case(::Type{T}=Float64; ρ=1, p=1, Γ=4 // 3, B0=1, η=1, L=1,
                          roots=5) where {T}
    eos = case_gas(T, Γ)
    s = EntropyEOS.polytropic_entropy(eos, T(p) / T(ρ)^T(Γ))
    h = EntropyEOS.evaluate(eos, T(ρ), s, T(1 // 2), T(NaN)).h
    vA = alfven_speed(T(ρ) * h, T(B0), T(η))
    w = AlfvenWave{T}(; B0=B0, η=η, k=2 * T(π) / T(L), vA=vA, ρ=ρ, s=s, Yₑ=T(1 // 2))
    return MHDCase(T, Val(1); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(1 // 10^9)), model=IdealConductor(),
                   𝔅̄=SVector(T(B0), zero(T), zero(T)), periodic=true,
                   extents=((zero(T), T(L)),), roots=roots)
end

"""
    alfven_period(case) -> L/v_A

The time the Alfvén wave of [`alfven_wave_case`](@ref) takes to cross its
box once.
"""
function alfven_period(case::MHDCase)
    L = case.extents[1][2] - case.extents[1][1]
    return L / case.fields.w.vA
end
