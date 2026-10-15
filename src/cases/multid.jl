# Rung 2a: the multidimensional cases in flat space on a uniform mesh
# ("Validation ladder" 2 in `CODE.md`; step 7 of `PLAN.md`), each an
# `MHDCase` in two (or three) dimensions under `IdealConductor`:
#
#   * the cylindrical explosion (Komissarov 1999), as GRaM-X sets it up;
#   * the relativistic magnetic rotor (Del Zanna et al. 2003), as GRaM-X sets
#     it up;
#   * the relativistic Orszag–Tang vortex (Beckwith & Stone 2011, §4.4);
#   * the advection of a weak field loop (Gardiner & Stone 2005, in Beckwith
#     & Stone's relativistic form, §4.2).
#
# **Published values, copied with their citation** (the reviewer's rule from
# step 6: nothing invented). The explosion and the rotor are GRaM-X's
# (Shankar et al. 2023, arXiv:2210.17509, §V B and §V C: eqs. 14–15 and the
# text after them); Orszag–Tang and the field loop are Beckwith & Stone's
# (2011, ApJS 193, 6; arXiv:1101.3573, §4.4 and §4.2, `γ = 5/3` "unless
# otherwise stated", their §4). Where a published setup leaves a choice, the
# docstring says what was chosen, marked (proposed in step 7).
#
# **Fields from potentials.** A uniform field is the background `𝔅̄`, and
# every other field is the discrete curl of an analytic `A`, sampled at the
# edges, so `∇·𝔅 = 0` to rounding from the first step (`src/curl.jl`). The
# electric field is the ideal `E = −v × B` the stages impose.
#
# A case's `fields` and `potential` are its initial data at every `t`: the
# non-periodic ones (the explosion and the rotor) use them as Dirichlet data
# (`DirichletBoundary`), exact until a wave reaches the box's faces, and every
# run here stops before that.

"""
    MultiDCase{K,T}

The parameters of one of the multidimensional cases, `isbits`, tagged by `K`
(`:explosion`, `:rotor`, `:orszag_tang`, `:field_loop`); the fields `p₁ … p₈`
are named in each constructor. See `src/cases/multid.jl`.
"""
struct MultiDCase{K,T}
    p₁::T
    p₂::T
    p₃::T
    p₄::T
    p₅::T
    p₆::T
    p₇::T
    p₈::T
    s₁::T          # the entropies of the case's states
    s₂::T
    Yₑ::T
    spherical::Bool
end

@inline ideal_fields(ρ, v::SVector{3,T}, s, Yₑ, B::SVector{3,T}) where {T} =
    (ρ, v, s, Yₑ, ideal_E(flat_metric(T), v, B), B)

# --- the cylindrical explosion -----------------------------------------------------

# GRaM-X's eq. 14: the profile `exp(((r_out − r) ln q_in + (r − r_in) ln q_out)/(r_out − r_in))`
# between `r_in` and `r_out`, constant inside and outside.
@inline function explosion_profile(r, r_in, r_out, q_in, q_out)
    r ≤ r_in && return q_in
    r ≥ r_out && return q_out
    return exp(((r_out - r) * log(q_in) + (r - r_in) * log(q_out)) / (r_out - r_in))
end

# p₁ … p₈ = r_in, r_out, ρ_in, ρ_out, p_in, p_out, Bˣ, Γ.
@inline function wave_fields(w::MultiDCase{:explosion}, x::SVector{3,T}, t::T) where {T}
    r = sqrt(x[1]^2 + x[2]^2 + (w.spherical ? x[3]^2 : zero(T)))
    ρ = explosion_profile(r, w.p₁, w.p₂, w.p₃, w.p₄)
    p = explosion_profile(r, w.p₁, w.p₂, w.p₅, w.p₆)
    # p = K(s) ρ^Γ with the case's gas: `s₁` is `s` at `K = 1`, and
    # `ln K = (Γ − 1)(s − s_ref)`.
    s = w.s₁ + log(p / ρ^w.p₈) / (w.p₈ - 1)
    z = zero(T)
    return ideal_fields(ρ, SVector(z, z, z), s, w.Yₑ, SVector(w.p₇, z, z))
end

@inline wave_potential(w::MultiDCase, x::SVector{3,T}, t::T) where {T} =
    multid_potential(w, x), zero(T)
@inline multid_potential(w::MultiDCase, x::SVector{3,T}) where {T} = zero(SVector{3,T})

"""
    explosion_gas(T, Γ)

The ideal gas of the explosion, `p = K(s) ρ^Γ` with `K(s) = exp((Γ − 1)(s − 30))`,
`s ∈ (1, 60)` — at Γ = 4/3, `K` from `6·10⁻⁵` to `2·10⁴`, around the case's 6.5
outside and 464 inside — `ρ ∈ (10⁻¹⁴, 10⁶)`, `Yₑ ∈ (0.05, 0.6)` (proposed in
step 7). [`shock_gas`](@ref)'s window, `K ≥ 0.002` at this Γ, floors the
entropy of a few cells behind the blast front, where the shock's numerical
heating lags its compression.
"""
explosion_gas(::Type{T}, Γ) where {T} =
    EntropyEOS.IdealGasEOS{T}(; Γ=T(Γ), K_ref=one(T), s_ref=T(30), s_window=(one(T), T(60)),
                              ρ_bounds=(T(1 // 10^14), T(10^6)),
                              yₑ_bounds=(T(1 // 20), T(3 // 5)))

"""
    cylindrical_explosion_case([T], Val(D) = Val(2); L = 6, roots = 4,
                               spherical = false, periodic = false, ρ_atm = 10⁻¹⁰)

The cylindrical explosion of Komissarov (1999) as GRaM-X sets it up
(Shankar et al. 2023, §V B, eqs. 14–15): at rest, `ρ = 10⁻²` and `p = 1`
inside `r_in = 0.8`, `ρ = 10⁻⁴` and `p = 3·10⁻⁵` outside `r_out = 1`, the
profile between them exponential in `r` (their eq. 14: `ln ρ` linear in `r`;
the printed formula has `ρ_in` and `ρ_out` exchanged, which would not
match the stated limits, and the continuous profile is used — proposed in
step 7), a uniform `Bˣ = 0.1` (Komissarov's weak-field case), `Γ = 4/3`,
on `[−L, L]^D` with `L = 6`, to `t = 4`. `r` is the distance from the `z`
axis; in 3D, `spherical = true` makes it the distance from the origin (a 3D
smoke test, proposed in step 7). Dirichlet boundaries from the initial data
(GRaM-X copies the outermost cells instead; nothing reaches the faces before
`t = 4`), or a periodic box with `periodic = true`, which is mirror-symmetric
where the Dirichlet hooks of the potentials are not: they evolve the box's
low boundary vertex and fill its high one ("Boundaries" in `CODE.md`).
"""
function cylindrical_explosion_case(::Type{T}=Float64, ::Val{D}=Val(2); L=6, roots=4,
                                    spherical::Bool=false, periodic::Bool=false,
                                    ρ_atm=1 // 10^10) where {T,D}
    2 ≤ D ≤ 3 || throw(ArgumentError("the explosion is two- or three-dimensional, not $D."))
    Γ = 4 // 3
    eos = explosion_gas(T, Γ)
    s₁ = EntropyEOS.polytropic_entropy(eos, one(T))
    w = MultiDCase{:explosion,T}(T(4 // 5), one(T), T(1 // 100), T(1 // 10^4), one(T),
                                 T(3 // 10^5), T(1 // 10), T(Γ), s₁, s₁, T(1 // 2), spherical)
    return MHDCase(T, Val(D); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(ρ_atm)), model=IdealConductor(),
                   𝔅̄=SVector(T(1 // 10), zero(T), zero(T)), periodic=periodic,
                   extents=ntuple(_ -> (-T(L), T(L)), Val(D)), roots=roots)
end

# --- the magnetic rotor -------------------------------------------------------------

# p₁ … p₈ = r_in, Ω, ρ_in, ρ_out, p, Bˣ, —, —; s₁, s₂ inside and outside.
@inline function wave_fields(w::MultiDCase{:rotor}, x::SVector{3,T}, t::T) where {T}
    r² = x[1]^2 + x[2]^2
    inside = r² < w.p₁^2
    z = zero(T)
    v = inside ? SVector(-w.p₂ * x[2], w.p₂ * x[1], z) : SVector(z, z, z)
    return ideal_fields(inside ? w.p₃ : w.p₄, v, inside ? w.s₁ : w.s₂, w.Yₑ,
                        SVector(w.p₆, z, z))
end

"""
    magnetic_rotor_case([T]; roots = 4, ρ_atm = 10⁻¹⁰)

The relativistic magnetic rotor of Del Zanna et al. (2003), as GRaM-X sets
it up (Shankar et al. 2023, §V C): a cylinder of radius `r_in = 0.1` and
density `ρ = 10` rotating rigidly at `Ω = 9.95` (`v = 0.995` at its edge,
`W ≈ 10`) in a medium at rest with `ρ = 1`, `p = 1` everywhere, a uniform
`Bˣ = 1`, `Γ = 5/3`, with no smoothing of the edge, on `[−1/2, 1/2]²`, to
`t = 0.4`. The gas is [`shock_gas`](@ref). Dirichlet boundaries from the
initial data.
"""
function magnetic_rotor_case(::Type{T}=Float64; roots=4, ρ_atm=1 // 10^10) where {T}
    Γ = 5 // 3
    eos = shock_gas(T, Γ)
    s_in = EntropyEOS.polytropic_entropy(eos, one(T) / T(10)^T(Γ))
    s_out = EntropyEOS.polytropic_entropy(eos, one(T))
    w = MultiDCase{:rotor,T}(T(1 // 10), T(995 // 100), T(10), one(T), one(T), one(T), zero(T),
                             zero(T), s_in, s_out, T(1 // 2), false)
    h = T(1 // 2)
    return MHDCase(T, Val(2); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(ρ_atm)), model=IdealConductor(),
                   𝔅̄=SVector(one(T), zero(T), zero(T)), periodic=false,
                   extents=((-h, h), (-h, h)), roots=roots)
end

# --- Orszag–Tang ------------------------------------------------------------------

# p₁ … p₈ = ρ, v₀, B₀, —, …; s₁ the entropy.
@inline function wave_fields(w::MultiDCase{:orszag_tang}, x::SVector{3,T}, t::T) where {T}
    k = 2 * T(π)
    z = zero(T)
    v = w.p₂ * SVector(-sin(k * x[2]), sin(k * x[1]), z)
    B = w.p₃ * SVector(-sin(k * x[2]), sin(2 * k * x[1]), z)
    return ideal_fields(w.p₁, v, w.s₁, w.Yₑ, B)
end

# `A_z = (B₀/4π) cos 4πx + (B₀/2π) cos 2πy`, so `Bˣ = ∂_y A_z = −B₀ sin 2πy`,
# `Bʸ = −∂ₓA_z = B₀ sin 4πx`.
@inline function multid_potential(w::MultiDCase{:orszag_tang}, x::SVector{3,T}) where {T}
    k = 2 * T(π)
    z = zero(T)
    return SVector(z, z, w.p₃ / (2 * k) * cos(2 * k * x[1]) + w.p₃ / k * cos(k * x[2]))
end

"""
    orszag_tang_case([T]; roots = 4)

The relativistic Orszag–Tang vortex of Beckwith & Stone (2011, §4.4): on the
periodic unit square, `ρ = 25/(36π)`, `p = 5/(12π)`, `Γ = 5/3`,
`v = ½(−sin 2πy, sin 2πx)` and the field of
`A_z = (B₀/4π) cos 4πx + (B₀/2π) cos 2πy`, `B₀ = 1/√(4π)`, to `t = 1`. The
paper prints `vˣ = 0.5 sin(2πy)`; the sign is the classic vortex's (Orszag &
Tang 1979; Stone & Gardiner 2009, whose result the paper reproduces at
`c = 100`), with `v` and `B` the same pattern at large scale (proposed in step
7). The gas is [`shock_gas`](@ref).
"""
function orszag_tang_case(::Type{T}=Float64; roots=4) where {T}
    Γ = 5 // 3
    eos = shock_gas(T, Γ)
    ρ = 25 / (36 * T(π))
    p = 5 / (12 * T(π))
    s = EntropyEOS.polytropic_entropy(eos, p / ρ^T(Γ))
    B₀ = inv(sqrt(4 * T(π)))
    w = MultiDCase{:orszag_tang,T}(ρ, T(1 // 2), B₀, zero(T), zero(T), zero(T), zero(T),
                                   zero(T), s, s, T(1 // 2), false)
    return MHDCase(T, Val(2); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(1 // 10^10)), model=IdealConductor(),
                   periodic=true, extents=((zero(T), one(T)), (zero(T), one(T))),
                   roots=roots)
end

# --- the field loop ---------------------------------------------------------------

# p₁ … p₈ = ρ, vˣ, vʸ, vᶻ, A₀, R, Lx, Ly; s₁ the entropy. The loop is advected
# with the flow, and the solution at `t` is the initial one at `x − vt`, folded
# into the periodic box.
@inline loop_point(w::MultiDCase{:field_loop}, x::SVector{3,T}) where {T} =
    SVector(x[1] - w.p₇ * round(x[1] / w.p₇), x[2] - w.p₈ * round(x[2] / w.p₈), zero(T))

@inline function wave_fields(w::MultiDCase{:field_loop}, x::SVector{3,T}, t::T) where {T}
    v = SVector(w.p₂, w.p₃, w.p₄)
    y = loop_point(w, x - t * v)
    r = sqrt(y[1]^2 + y[2]^2)
    z = zero(T)
    # `Bˣ = ∂_y A_z = −A₀ y/r`, `Bʸ = −∂ₓA_z = A₀ x/r` inside `R`.
    B = r < w.p₆ && r > z ? w.p₅ * SVector(-y[2] / r, y[1] / r, z) : SVector(z, z, z)
    return ideal_fields(w.p₁, v, w.s₁, w.Yₑ, B)
end

@inline function multid_potential(w::MultiDCase{:field_loop}, x::SVector{3,T}) where {T}
    y = loop_point(w, x)
    r = sqrt(y[1]^2 + y[2]^2)
    z = zero(T)
    return SVector(z, z, r < w.p₆ ? w.p₅ * (w.p₆ - r) : z)
end

# The loop moves with the flow. Its `A_z(x − vt)` is exact, since the ideal
# `E = −v × B` makes `∂ₜA_z = −E_z = −v·∇A_z`; with `vᶻ ≠ 0` the in-plane
# potential also grows, `∂ₜA_x = −vᶻBʸ`, `∂ₜA_y = vᶻBˣ`, and is not returned.
# The case is periodic, so only `t = 0` is asked of it for the state; the cell
# variables of `fields` are exact at every `t`.
@inline wave_potential(w::MultiDCase{:field_loop}, x::SVector{3,T}, t::T) where {T} =
    multid_potential(w, x - t * SVector(w.p₂, w.p₃, w.p₄)), zero(T)

"""
    field_loop_case([T]; vz = 0, roots = (4, 2))

The advection of a weak magnetic field loop (Gardiner & Stone 2005) in
Beckwith & Stone's relativistic form (2011, §4.2): `ρ = 1`, `p = 3`,
`Γ = 5/3`, `v = (0.2, 0.1, 0)/√6` or, with `vz = 1/10`, `(0.2, 0.1, 0.1)/√6`,
and the field of `A_z = A₀(R − r)` for `r ≤ R`, zero outside, `A₀ = 10⁻³`,
`R = 0.3`, on the periodic `[−1, 1] × [−1/2, 1/2]`, which the loop crosses in
`t = 2√6/0.2 ≈ 24.5`. The gas is [`shock_gas`](@ref).
"""
function field_loop_case(::Type{T}=Float64; vz=0, roots=(4, 2)) where {T}
    Γ = 5 // 3
    eos = shock_gas(T, Γ)
    s = EntropyEOS.polytropic_entropy(eos, T(3))
    r6 = sqrt(T(6))
    w = MultiDCase{:field_loop,T}(one(T), T(1 // 5) / r6, T(1 // 10) / r6, T(vz) / r6,
                                  T(1 // 1000), T(3 // 10), T(2), one(T), s, s, T(1 // 2), false)
    return MHDCase(T, Val(2); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(1 // 10^10)), model=IdealConductor(),
                   periodic=true, extents=((-one(T), one(T)), (-T(1 // 2), T(1 // 2))),
                   roots=roots)
end
