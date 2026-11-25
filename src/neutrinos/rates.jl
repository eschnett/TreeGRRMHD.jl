# The neutrino rates and opacities, pointwise ("Neutrinos", "Leakage" in
# `CODE.md`).
#
# A rate model answers `neutrino_rates(model, ρ★, T_MeV, yₑ, comp)` with one
# `SpeciesRates` per species — ν_e, ν̄_e, ν_x, in that order — in the scheme's
# geometric units:
#
#   * `R`, `Q`: the free number and energy emission per volume per time, in
#     the fluid frame. The number is counted in units of the baryon mass
#     `m★ = κ m_B` the EOS's `ρ★` is measured in, so that it is a density rate
#     like `D`'s and `√γ D Yₑ + N_{ν_e} − N_{ν̄_e}` is conserved without a
#     factor;
#   * `κa0`, `κa1`, `κs0`, `κs1`: the absorption and scattering opacities,
#     number-weighted (`0`) and energy-weighted (`1`), per length;
#   * `nβ`, `eβ`: the number and energy densities of neutrinos in equilibrium
#     with the matter, in the same units as `R` and `Q` times a time, and
#     `Eβ`, their mean energy, in units of `m★ c²` — carried, not taken as
#     `eβ/nβ`, which is 0/0 where the equilibrium density underflows.
#
# An energy per neutrino is then `Q/R` or `Eβ`, in units of `m★ c²`.

"""
    SpeciesRates{T}

One species' free emission `R`, `Q`, opacities `κa0`, `κa1`, `κs0`, `κs1`,
equilibrium densities `nβ`, `eβ` and mean energy `Eβ`, in geometric units, the number counted in
units of the baryon mass `m★` ("Neutrinos", "Leakage" in `CODE.md`). `isbits`.
"""
struct SpeciesRates{T}
    R::T
    Q::T
    κa0::T
    κa1::T
    κs0::T
    κs1::T
    nβ::T
    eβ::T
    Eβ::T
end

"""
    NeutrinoRates

A rate model ("Neutrinos", "Leakage" in `CODE.md`): [`GrayRates`](@ref) for
the tests, [`RJS96Rates`](@ref) on a tabulated EOS. Each answers
[`neutrino_rates`](@ref).
"""
abstract type NeutrinoRates end

# ---------------------------------------------------------------------------
# The composition (EOS-12, with EOS-13's nucleon potentials)

"""
    CompositionPoint{T}(; μ_e, μ̂, μ_n, μ_p, X_n, X_p, X_α, X_h, X_l, Ā, Z̄)

The table's composition at one state, as [`RJS96Rates`](@ref) reads it:
EntropyEOS's `Composition` (EOS-12; built from one by the method below) and
the nucleons' own chemical potentials `μ_n`, `μ_p`, without rest masses,
which RJS96's scattering blocking reads and which EOS-12 does not carry
(EOS-13; `CODE.md`, "Requests still waiting"). EOS-12's conventions:
chemical potentials in MeV, `μ_e` with the electron's rest mass, `μ̂` with
the neutron–proton mass difference `Q`, so that `μ_ν = μ_e − μ̂`; mass
fractions; `Ā`, `Z̄` of the heavy nuclei. `isbits`.
"""
struct CompositionPoint{T}
    μ_e::T
    μ̂::T
    μ_n::T
    μ_p::T
    X_n::T
    X_p::T
    X_α::T
    X_h::T
    X_l::T
    Ā::T
    Z̄::T
end
CompositionPoint{T}(; μ_e, μ̂, μ_n, μ_p, X_n, X_p, X_α, X_h, X_l, Ā, Z̄) where {T} =
    CompositionPoint{T}(μ_e, μ̂, μ_n, μ_p, X_n, X_p, X_α, X_h, X_l, Ā, Z̄)

"""
    CompositionPoint(c::EntropyEOS.Composition; μ_n, μ_p)

The point the rates read from EntropyEOS's composition (EOS-12) and the
nucleons' own chemical potentials, which it does not carry until EOS-13
(`CODE.md`, "Requests still waiting").
"""
CompositionPoint(c::EntropyEOS.Composition{T}; μ_n, μ_p) where {T} =
    CompositionPoint{T}(c.μ_e, c.μ̂, T(μ_n), T(μ_p), c.X_n, c.X_p, c.X_α, c.X_h, c.X_l, c.Ā,
                        c.Z̄)

"""The equilibrium ν_e chemical potential `μ_ν = μ_e − μ̂`, MeV (EOS-12)."""
@inline neutrino_potential(c::CompositionPoint) = c.μ_e - c.μ̂

# ---------------------------------------------------------------------------
# Gray rates, for the tests

"""
    GrayRates{T}(; κa, κs, η, E)

Gray rates for the tests, on any EOS ("Neutrinos", "Leakage" in `CODE.md`):
per species (a 3-tuple each, ν_e, ν̄_e, ν_x) an absorption and a scattering
opacity `κa ρ★`, `κs ρ★`, a number emissivity `R = η ρ★` and one emitted
energy `E` (in units of `m★ c²`), all in geometric units; the weightings
alike. The equilibrium density is Kirchhoff's, `nβ = R/κa` (`c = 1`), so
`t_loss = 1/κa`; with `κa = 0` it is infinite, and the effective rates are
the free ones. `isbits`.
"""
struct GrayRates{T} <: NeutrinoRates
    κa::NTuple{3,T}
    κs::NTuple{3,T}
    η::NTuple{3,T}
    E::NTuple{3,T}
end
function GrayRates{T}(; κa, κs, η, E) where {T}
    all(x -> all(≥(0), x), (κa, κs, η, E)) || throw(ArgumentError(
        "GrayRates needs non-negative opacities, emissivities and energies: they " *
        "are densities of interactions and of emitted neutrinos."))
    return GrayRates{T}(T.(κa), T.(κs), T.(η), T.(E))
end

@inline function gray_species(g::GrayRates{T}, i, ρ★::T) where {T}
    κa = g.κa[i] * ρ★
    κs = g.κs[i] * ρ★
    R = g.η[i] * ρ★
    nβ = iszero(R) ? zero(T) : (iszero(κa) ? T(Inf) : R / κa)
    return SpeciesRates{T}(R, R * g.E[i], κa, κa, κs, κs, nβ, nβ * g.E[i], g.E[i])
end

@inline neutrino_rates(g::GrayRates{T}, ρ★::T, T_MeV::T, yₑ::T, comp) where {T} =
    (gray_species(g, 1, ρ★), gray_species(g, 2, ρ★), gray_species(g, 3, ρ★))

# ---------------------------------------------------------------------------
# RJS96

# The physical constants, Float64, used only to build the prefactors.
const RJS_σ0 = 1.76e-44              # cm², RJS96 (A1)
const RJS_α = 1.25                   # the axial coupling g_A, RJS96 (A1)
const RJS_SIN2W = 0.23               # sin²θ_W, RJS96 (A1)
const RJS_CV = 0.5 + 2 * RJS_SIN2W   # RJS96 (A1)
const RJS_CA = 0.5                   # RJS96 (B9)
const RJS_αF = 1 / 137.036           # the fine-structure constant, RJS96 (B12)
const RJS_ME = 0.51099895            # mₑc² in MeV, CODATA 2018
const RJS_HC = 1.23984198e-10        # hc in MeV cm, CODATA 2018
const MEV_ERG = 1.602176634e-6       # erg per MeV, exact
const AMU_MEV = 931.49410242         # m_u c² in MeV, CODATA 2018
const AMU_G = 1.66053906660e-24      # m_u in g, CODATA 2018
const Q_NP_MEV = 1.29333236          # m_n c² − m_p c² in MeV, CODATA 2018

"""
    RJS96Rates(T = Float64, units::EOSUnits; baryon_mass, q_np = 1.29333236)

The leakage rates and opacities of Ruffert, Janka & Schäfer (1996, A&A 311,
532, Appendices A and B: "RJS96") with nucleon–nucleon bremsstrahlung
(Burrows, Reddy & Thompson 2006, Eq. 142) and coherent scattering on heavy
nuclei and α particles (their Eqs. 25–32), from the table's composition
([`CompositionPoint`](@ref)): "Neutrinos", "Leakage" in `CODE.md`
has the formulas and the choices among the published versions.

`units` are the EOS's ([`EOSUnits`](@ref)); `baryon_mass` (g) is the mass the
table's density counts baryons in, so that `n_b = ρ★/(κ m_B)` — no default,
since it is a property of the table family that is not in the file; `q_np`
(MeV) is the neutron–proton mass difference, EOS-12's `q_np`; the table
form below takes both from the table and its composition.
Every constant and unit conversion is folded into the prefactors here, in
`Float64`, so that a kernel multiplies numbers of order one and `Float32`
stays finite. `isbits`.
"""
struct RJS96Rates{T} <: NeutrinoRates
    c_β::T        # β-processes, B1–B2
    c_ee::T       # pairs, B8, ν_e ν̄_e, without (C1 + C2)
    C_ee_e::T     # (C1 + C2) for ν_e ν̄_e
    C_ee_x::T     # (C1 + C2) for ν_x
    c_γ_e::T      # plasmons, B11
    c_γ_x::T      # plasmons, B12
    γ0::T
    c_br::T       # bremsstrahlung
    c_κ::T        # opacities: σ₀ n_b/ρ★ (1/mₑc²)² in geometric units
    C_sn::T
    C_sp::T
    C_a::T
    C_coh::T      # coherent scattering, transport: 1/24
    W_fac::T      # 1 − 2 sin²θ_W
    c_n::T        # equilibrium densities: 4π/(hc)³ in geometric units
    inv_m★::T     # 1/(m★c²) in 1/MeV
    q_np::T
end

function RJS96Rates(::Type{T}, units::EOSUnits; baryon_mass, q_np=Q_NP_MEV) where {T}
    (isfinite(baryon_mass) && baryon_mass > 0) || throw(ArgumentError(
        "RJS96Rates needs the table's baryon mass in grams, got $baryon_mass: the " *
        "baryon density is ρ★/(κ m_B), and every rate is per baryon density."))
    (isfinite(q_np) && q_np > 0) || throw(ArgumentError(
        "RJS96Rates needs the neutron–proton mass difference in MeV, got $q_np: " *
        "the nucleons' Pauli blocking reads (μ̂ − Q)/T."))
    κ = Float64(units.κ)
    ud = Float64(units.density)
    νb = ud / (κ * baryon_mass)          # n_b in cm⁻³ per geometric ρ★
    tu = MSUN_TIME_CGS
    lu = MSUN_LENGTH_CGS
    m★ = κ * baryon_mass / AMU_G * AMU_MEV  # m★c² in MeV
    G = RJS_σ0 * C_CGS / RJS_ME^2       # cm³ s⁻¹ MeV⁻²
    P = 8π / RJS_HC^3                   # MeV⁻³ cm⁻³, the ε moments' factor
    # A number rate in cm⁻³ s⁻¹ is R·tu/νb in geometric units (counted in m★).
    c_β = (1 + 3 * RJS_α^2) / 8 * G * P * tu                 # × ρ★ Y T⁵ F₄ ⟨1−f⟩
    c_ee = G * P^2 * tu / νb                                  # × (C1+C2)/36 T⁸ F₃F₃ ⟨⟩⟨⟩
    C_ee_e = (RJS_CV - RJS_CA)^2 + (RJS_CV + RJS_CA)^2
    C_ee_x = (RJS_CV - RJS_CA)^2 + (RJS_CV + RJS_CA - 2)^2
    c_γ = π^3 / (3 * RJS_αF) * G / RJS_HC^6 * tu / νb        # × T⁸ γ⁶ e^{−γ}(1+γ) ⟨⟩⟨⟩
    γ0 = 2 * sqrt(RJS_αF / (3π))                             # 5.565e-2, RJS96 (B12)
    # Bremsstrahlung, BRT06 Eq. 142, per νν̄ pair: 1.04e30 ζ S (ρ/10¹⁴)² T^5.5
    # erg cm⁻³ s⁻¹, ζ = 0.5, with ρ = ρ★ ud/κ in g/cm³; here in MeV and as a
    # geometric number-like rate, divided by m★c² or by the mean energy later.
    c_br = 1.04e30 * 0.5 / MEV_ERG * (ud / κ / 1e14)^2 * tu / νb
    c_κ = RJS_σ0 * νb / RJS_ME^2 * lu
    C_sn = (1 + 5 * RJS_α^2) / 24
    C_sp = (4 * (RJS_CV - 1)^2 + 5 * RJS_α^2) / 24
    C_a = (1 + 3 * RJS_α^2) / 4
    c_n = 4π / RJS_HC^3 / νb
    return RJS96Rates{T}(c_β, c_ee, C_ee_e, C_ee_x, c_γ * RJS_CV^2, 4 * c_γ * (RJS_CV - 1)^2,
                         γ0, c_br, c_κ, C_sn, C_sp, C_a, 1 / 24, 1 - 2 * RJS_SIN2W, c_n,
                         1 / m★, q_np)
end
RJS96Rates(units::EOSUnits; kwargs...) = RJS96Rates(Float64, units; kwargs...)

"""
    RJS96Rates(T = Float64, units, table::EntropyEOS.EOSTable, cv::EntropyEOS.CompositionView)

The rates on a table: the baryon mass from the table's provenance
(`m_B_table_g`), the neutron–proton mass difference from the composition
(`q_np`). Refuses a composition that was not built from the same file as
the table (`matches`), and units whose `κ` is not the table's: a
composition paired with another table would convert `ρ★` silently wrong,
and `κ` does not tell tables apart (DD2's and SFHo's agree to 1e−10).
"""
function RJS96Rates(::Type{T}, units::EOSUnits, table::EntropyEOS.EOSTable,
                    cv::EntropyEOS.CompositionView) where {T}
    EntropyEOS.matches(cv, table) || throw(ArgumentError(
        "RJS96Rates: the composition was not built from the same table file as the " *
        "EOS (EntropyEOS.matches is false), so its densities and fractions belong to " *
        "another table."))
    Float64(units.κ) == Float64(EntropyEOS.κ(table)) || throw(ArgumentError(
        "RJS96Rates: the units' κ = $(units.κ) is not the table's " *
        "$(EntropyEOS.κ(table)); build them with EOSUnits(eos; density)."))
    return RJS96Rates(T, units; baryon_mass=table.m_B_table_g,
                      q_np=Float64(EntropyEOS.q_np(cv)))
end
RJS96Rates(units::EOSUnits, table::EntropyEOS.EOSTable, cv::EntropyEOS.CompositionView) =
    RJS96Rates(Float64, units, table, cv)

# Pauli blocking of a final-state fermion of mean energy `ϵ` (in units of T)
# and degeneracy `η`: 1 − f = 1/(1 + exp(η − ϵ)). Every blocking factor of
# RJS96 (A15, A16, B3, B4, B9, B13) is this one.
@inline blocking(η, ϵ) = inv(one(η) + exp(η - ϵ))

# The free-nucleon number fractions with the nucleons' Pauli blocking,
# RJS96 (A13, A14), with `X_p − X_n` for RJS96's `2Yₑ − 1` (decided in step
# 16), so that the non-degenerate limit is `X_n`, `X_p` even where nuclei
# hold the matter. `x = (μ̂ − Q)/T = η_n − η_p`. Near `x = 0`, and wherever
# the table's `x` and fractions disagree, clamped to `[0, X_n]` and
# `[0, X_p]`.
@inline function nucleon_blocking(X_n::T, X_p::T, x::T) where {T}
    small = sqrt(eps(T))
    abs(x) < small && return X_n, X_p
    # Y_pn = e^{−x} Y_np, written so that neither overflows (step 16:
    # Inf·0 at x = −213 in Float32).
    Y_np = (X_p - X_n) / expm1(-x)
    Y_pn = (X_n - X_p) / expm1(x)
    return clamp(Y_np, zero(T), X_n), clamp(Y_pn, zero(T), X_p)
end

# The scattering targets' blocking, RJS96 (A8): Y_N/(1 + (2/3) max(η_N, 0)).
@inline scattering_blocking(X::T, η::T) where {T} = X / (1 + 2 * max(η, zero(T)) / 3)

"""
    neutrino_rates(model, ρ★, T_MeV, yₑ, comp) -> (ν_e, ν̄_e, ν_x)

The free emission, the opacities and the equilibrium densities of the three
species at one state, as [`SpeciesRates`](@ref) in geometric units: `ρ★` the
EOS's density (geometric), `T_MeV` the temperature the recovery landed on,
`yₑ`, and `comp` the table's composition there ([`CompositionPoint`](@ref);
ignored by [`GrayRates`](@ref)). "Neutrinos", "Leakage" in `CODE.md`.
Allocation-free.
"""
@inline function neutrino_rates(m::RJS96Rates{T}, ρ★::T, T_MeV::T, yₑ::T,
                                comp::CompositionPoint{T}) where {T}
    t = T_MeV
    ηe = comp.μ_e / t
    ην = neutrino_potential(comp) / t       # ν_e; ν̄_e has −ην, ν_x has 0
    Y_np, Y_pn = nucleon_blocking(comp.X_n, comp.X_p, (comp.μ̂ - m.q_np) / t)

    F3e, F4e = fermi(Val(3), ηe), fermi(Val(4), ηe)
    F3p, F4p = fermi(Val(3), -ηe), fermi(Val(4), -ηe)
    t² = t * t
    t⁴ = t² * t²

    # β-processes, B1–B4 and B14: ν_e from e⁻p, ν̄_e from e⁺n.
    ϵe = fermi_ratio(Val(5), Val(4), ηe)     # mean energy of the captured e⁻, in T
    ϵp = fermi_ratio(Val(5), Val(4), -ηe)
    Rβe = m.c_β * ρ★ * Y_pn * t⁴ * t * F4e * blocking(ην, ϵe)
    Rβa = m.c_β * ρ★ * Y_np * t⁴ * t * F4p * blocking(-ην, ϵp)

    # Pair annihilation, B8–B10, B15.
    ϵee = (fermi_ratio(Val(4), Val(3), ηe) + fermi_ratio(Val(4), Val(3), -ηe)) / 2
    pairs = m.c_ee / 36 * t⁴ * t⁴ * F3e * F3p
    Ree_ea = m.C_ee_e * pairs * blocking(ην, ϵee) * blocking(-ην, ϵee)
    Ree_x = 4 * m.C_ee_x * pairs * blocking(zero(T), ϵee)^2

    # Plasmon decay, B11–B13, B16.
    γ = m.γ0 * sqrt(T(π)^2 / 3 + ηe * ηe)
    ϵγ = 1 + γ * γ / (2 * (1 + γ))           # mean neutrino energy, in T
    plasma = t⁴ * t⁴ * γ^6 * exp(-γ) * (1 + γ)
    Rγ_ea = m.c_γ_e * plasma * blocking(ην, ϵγ) * blocking(-ην, ϵγ)
    Rγ_x = m.c_γ_x * plasma * blocking(zero(T), ϵγ)^2

    # Bremsstrahlung, BRT06 Eqs. 142 and 145: per νν̄ pair; ν_e and ν̄_e each
    # half of one pair, ν_x two pairs; mean energy (2.4/1.1) T per neutrino,
    # from the spectrum of Eq. 145.
    S = comp.X_n^2 + comp.X_p^2 + 28 * comp.X_n * comp.X_p / 3
    q_br = m.c_br * ρ★ * ρ★ * S * t⁴ * t * sqrt(t)  # MeV-weighted, per pair
    ϵbr = T(12) / 5 * inv(T(11) / 10)               # 2.4/1.1, in T

    # Totals per species: number, and energy (MeV-weighted, × T).
    R_e = Rβe + Ree_ea + Rγ_ea + q_br / 2 / (ϵbr * t)
    R_a = Rβa + Ree_ea + Rγ_ea + q_br / 2 / (ϵbr * t)
    R_x = Ree_x + Rγ_x + 2 * q_br / (ϵbr * t)
    E_e = (Rβe * ϵe + Ree_ea * ϵee + Rγ_ea * ϵγ) * t + q_br / 2
    E_a = (Rβa * ϵp + Ree_ea * ϵee + Rγ_ea * ϵγ) * t + q_br / 2
    E_x = (Ree_x * ϵee + Rγ_x * ϵγ) * t + 2 * q_br

    # Opacities, A6–A16, and coherent scattering (BRT06 Eqs. 25–32, transport).
    η_n = comp.μ_n / t
    η_p = comp.μ_p / t
    Ynn = scattering_blocking(comp.X_n, η_n)
    Ypp = scattering_blocking(comp.X_p, η_p)
    W_h = 1 - 2 * comp.Z̄ / comp.Ā * m.W_fac
    W_α = 1 - m.W_fac                       # Z/A = 1/2
    coh = m.C_coh * (comp.X_h * comp.Ā * W_h^2 + 4 * comp.X_α * W_α^2)
    scat = m.C_sn * Ynn + m.C_sp * Ypp + coh
    base = m.c_κ * ρ★ * t²
    ηs = (ην, -ην, zero(T))
    abs_e = m.C_a * Y_np * blocking(ηe, fermi_ratio(Val(5), Val(4), ην))
    abs_a = m.C_a * Y_pn * blocking(-ηe, fermi_ratio(Val(5), Val(4), -ην))
    absorb = (abs_e, abs_a, zero(T))

    Rs = (R_e, R_a, R_x)
    Es = (E_e, E_a, E_x)
    gs = (one(T), one(T), T(4))
    return ntuple(Val(3)) do i
        @inbounds begin
            η = ηs[i]
            r0 = fermi_ratio(Val(4), Val(2), η)       # ⟨ε²⟩/T², number-weighted
            r1 = fermi_ratio(Val(5), Val(3), η)       # energy-weighted
            nβ = gs[i] * m.c_n * t² * t * fermi(Val(2), η)   # RJS96 (B18)
            Eβ = t * fermi_ratio(Val(3), Val(2), η) * m.inv_m★  # (B19)/(B18)
            SpeciesRates{T}(Rs[i], Es[i] * m.inv_m★,
                            base * absorb[i] * r0, base * absorb[i] * r1,
                            base * scat * r0, base * scat * r1, nβ, nβ * Eβ, Eβ)
        end
    end
end
