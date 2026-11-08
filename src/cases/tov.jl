# Rung 4: the TOV star in the Cowling approximation ("Validation ladder" 4 in
# `CODE.md`; step 10 of `PLAN.md`). The polytrope `K = 100`, `Γ = 2`,
# `ρ_c = 1.28·10⁻³` (`M = 1.400`, the standard model of Font et al. 2002) at
# rest on its own frozen metric, `Analytic(TOVMetric)` (`src/tov.jl`), with no
# field: `A = 0`, `Φ̃ = 0`, `E = B = 0` under `IdealConductor`, so the matter
# is pure hydrodynamics and the potentials stay zero.
#
# **The gas** (proposed in step 10). `IdealGasEOS` with `K_ref = K` and the
# entropy window centred on `s_ref`, so that the star's `polytropic_entropy` is
# `s_ref` exactly and the atmosphere's `s_atm` — `recovery_options`' midpoint
# of the window — is the star's entropy: the atmosphere is the star's
# polytrope at `ρ_min`, and a cell that the reset puts there is on the star's
# isentrope. The window `s_ref ± 2` admits `K` within a factor `e²` of the
# star's, room for the surface's shock heating.
#
# **The exterior.** Outside the surface, and wherever the polytrope's density
# is below `ρ_min` near it, the initial data is the atmosphere at rest
# (`ρ_min`, `s_ref`), the state `atmosphere_prim` gives, so that the reset
# finds the exterior already where it would put it. The box `[−L, L]³` has
# Dirichlet faces from the same static data; with `A = 0` the potentials'
# hook is absolute (`static_potential = true`, option 1 of "Boundaries";
# decided by Erik 2026-11-03 for a static star).

"""
    TOVStar{T,NK}

The data of the Cowling TOV case, `isbits`: the star's [`TOVMetric`](@ref),
the provider that serves it to the scheme, the star's entropy `s`, the
electron fraction `Yₑ` and the atmosphere's density `ρ_atm`. See
`src/cases/tov.jl`.
"""
struct TOVStar{T,NK}
    metric::TOVMetric{T,NK}
    provider::Analytic{TOVMetric{T,NK}}
    s::T
    Yₑ::T
    ρ_atm::T
end

@inline function wave_fields(w::TOVStar, x::SVector{3,T}, t::T) where {T}
    ρ = tov_density(w.metric, x[1]^2 + x[2]^2 + x[3]^2)
    z = zero(SVector{3,T})
    return max(ρ, w.ρ_atm), z, w.s, w.Yₑ, z, z
end

@inline wave_potential(::TOVStar, x::SVector{3,T}, t::T) where {T} = zero(SVector{3,T}), zero(T)

"""
    tov_gas(T; K = 100, Γ = 2, s_ref = 5) -> IdealGasEOS{T}

The star's ideal gas in geometric units: `p = K ρ^Γ` at `s = s_ref`, the
entropy window `s_ref ± 2`, densities `(10⁻¹⁴, 1)`, `Yₑ ∈ (0.05, 0.6)`
(`src/cases/tov.jl`).
"""
tov_gas(::Type{T}; K=100, Γ=2, s_ref=5) where {T} =
    EntropyEOS.IdealGasEOS{T}(; Γ=T(Γ), K_ref=T(K), s_ref=T(s_ref),
                              s_window=(T(s_ref) - 2, T(s_ref) + 2),
                              ρ_bounds=(T(1e-14), one(T)), yₑ_bounds=(T(0.05), T(0.6)))

"""
    tov_case([T]; K = 100, Γ = 2, ρ_c = 1.28e-3, L = 24, roots = 4, ρ_min = 1e-10,
             NK = 129, steps = 4000)

The TOV star in the Cowling approximation ("Validation ladder" 4 in
`CODE.md`) as a three-dimensional [`MHDCase`](@ref): the polytrope `p = K ρ^Γ`
of central density `ρ_c` ([`tov_solve`](@ref) with `steps` RK4 steps), at rest
on its own metric, `Analytic(TOVMetric)` with `NK` knots, in the closed box
`[−L, L]³` of `roots³` root blocks with Dirichlet faces from the static
exterior, the atmosphere at `ρ_min` on the star's isentrope outside
([`tov_gas`](@ref)), and no field (`IdealConductor`, `A = 0`, `Φ̃ = 0`). The
static refinement around the star is [`tov_boxes!`](@ref): on the default
box, whose root blocks are `12` wide, its first level covers `[−12, 12]³`, a
coarse-fine face outside the surface at `R̄ = 8.1` (on `[−16, 16]³` with two
roots it would refine every block). The case's data
is `case.fields.w`, a [`TOVStar`](@ref); its solution is `TOVMetric`'s.
"""
function tov_case(::Type{T}=Float64; K=100, Γ=2, ρ_c=1.28e-3, L=24, roots=4,
                  ρ_min=1e-10, NK::Integer=129, steps::Integer=4000) where {T}
    sol = tov_solve(T; K, Γ, ρ_c, steps)
    T(L) > sol.R̄ || throw(ArgumentError(
        "the box [−L, L]³ must hold the star, whose isotropic radius is $(sol.R̄); got L = $L."))
    0 < T(ρ_min) < T(ρ_c) || throw(ArgumentError(
        "the atmosphere's density must lie below the star's central density, got " *
        "ρ_min = $ρ_min, ρ_c = $ρ_c."))
    eos = tov_gas(T; K, Γ)
    m = TOVMetric(sol; NK)
    prov = Analytic(m)
    rec = recovery_options(eos, T(ρ_min))
    s = EntropyEOS.polytropic_entropy(eos, T(K))
    w = TOVStar(m, prov, s, T(1 // 2), T(ρ_min))
    return MHDCase(T, Val(3); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   rec..., model=IdealConductor(), provider=prov, periodic=false,
                   extents=ntuple(_ -> (-T(L), T(L)), Val(3)), roots=roots,
                   static_potential=true)
end

"""
    tov_boxes!(forest, case; levels = 1, half_width = 5R̄/4) -> forest

The Cowling star's static refinement: `levels` times, every leaf that meets
the cube `[−half_width, half_width]³` around the star (`R̄` its isotropic
radius) and is coarser than the level being made is refined, and the forest
rebalanced, so that the whole star and its surface lie on the finest level.
For `evolve!`'s `refine`: `refine = f -> tov_boxes!(f, case; levels = 2)`.
"""
function tov_boxes!(forest::Forest{3,T}, case::MHDCase; levels::Integer=1,
                    half_width=5 * case.fields.w.metric.R̄ / 4) where {T}
    a = T(half_width)
    meets(k) = all(e -> e[1] < a && e[2] > -a, block_extent(T, forest, k))
    for ℓ in 1:levels
        refine!(forest, [k for k in forest.leaves if level(k) < ℓ && meets(k)])
        balance!(forest)
    end
    return forest
end
