# Rung 4: the TOV star in the Cowling approximation ("Validation ladder" 4 in
# `CODE.md`; step 10 of the plan in `HISTORY.md`). The polytrope `K = 100`, `Γ = 2`,
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
#
# **The perturbation** (proposed in step 11, to excite the overtones; "Validation
# ladder" 5 in `CODE.md`). `δv = (a₁, a₂, a₃)` starts the star moving in its radial
# modes F, H1 and H2, mode `n` with the amplitude `aₙ` in `ρ_c/ρ_c`, at the phase of zero
# displacement: the velocity is the modes' eigenfunctions from linear perturbation
# theory (`src/tov_pulsation.jl`), coupled or Cowling as `δv_modes` says, inside the
# star, and nothing outside it, so the Dirichlet faces keep the static exterior. The
# density, the entropy and the metric stay the static star's, so the momentum
# constraint is violated by `8π S_i`. The default `(0, 0, 0)` branches to the star at
# rest.

"""
    TOVStar{T,NK}

The data of the Cowling TOV case, `isbits`: the star's [`TOVMetric`](@ref),
the provider that serves it to the scheme, the star's entropy `s`, the
electron fraction `Yₑ`, the atmosphere's density `ρ_atm`, and `δv`, the Chebyshev
coefficients in `2r̄²/R̄² − 1` of the initial velocity's `vⁱ/xⁱ` inside the star
([`tov_mode_velocity`](@ref)). See `src/cases/tov.jl`.
"""
struct TOVStar{T,NK}
    metric::StaticTOVMetric{T,NK,Polytrope{T}}
    provider::Analytic{StaticTOVMetric{T,NK,Polytrope{T}}}
    s::T
    Yₑ::T
    ρ_atm::T
    δv::SVector{TOV_NCHEB,T}
end

@inline function wave_fields(w::TOVStar, x::SVector{3,T}, t::T) where {T}
    r² = x[1]^2 + x[2]^2 + x[3]^2
    ρ = tov_density(w.metric, r²)
    z = zero(SVector{3,T})
    return max(ρ, w.ρ_atm), tov_velocity(w, x, r²), w.s, w.Yₑ, z, z
end

# The perturbation's velocity at `x`, `r² = |x|²`: zero for the star at rest and
# outside the surface (branches, so the star at rest is bit for bit).
@inline function tov_velocity(w::TOVStar, x::SVector{3,T}, r²::T) where {T}
    R̄² = w.metric.R̄^2
    (iszero(w.δv) || r² ≥ R̄²) && return zero(SVector{3,T})
    return chebyshev_sum(w.δv, 2 * r² / R̄² - 1) * x
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
             NK = 129, steps = 4000, δv = (0, 0, 0), δv_modes = nothing)

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

`δv = (a₁, a₂, a₃)` starts the star moving in its radial modes F, H1 and H2, mode `n`
with the amplitude `aₙ` in `ρ_c/ρ_c` (`aₙ > 0` expands the centre first), at the phase of
zero displacement (proposed in step 11, to excite the overtones; `src/cases/tov.jl`).
The velocity is the modes' eigenfunctions, `δv_modes = :coupled` with the metric
perturbed or `:cowling` with it fixed ([`tov_radial_modes`](@ref)); it has no default,
and is needed only when `δv` is not zero. The solution is then the initial data only,
not a static star.
"""
function tov_case(::Type{T}=Float64; K=100, Γ=2, ρ_c=1.28e-3, L=24, roots=4,
                  ρ_min=1e-10, NK::Integer=129, steps::Integer=4000,
                  δv=(0, 0, 0), δv_modes=nothing) where {T}
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
    w = TOVStar(m, prov, s, T(1 // 2), T(ρ_min),
                SVector{TOV_NCHEB,T}(tov_case_velocity(sol, δv, δv_modes; K, Γ, ρ_c, steps)))
    return MHDCase(T, Val(3); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   rec..., model=IdealConductor(), provider=prov, periodic=false,
                   extents=ntuple(_ -> (-T(L), T(L)), Val(3)), roots=roots,
                   static_potential=true)
end

# The Chebyshev coefficients of the perturbation's `vⁱ/xⁱ`, zero for the star at rest; the
# modes are solved in `Float64` whatever `T` is.
function tov_case_velocity(sol::TOVSolution{T}, δv, δv_modes; K, Γ, ρ_c, steps) where {T}
    length(δv) == 3 || throw(ArgumentError(
        "δv holds the amplitudes of the modes F, H1 and H2; got $(length(δv)) values."))
    all(iszero, δv) && return zero(SVector{TOV_NCHEB,Float64})
    δv_modes in (:coupled, :cowling) || throw(ArgumentError(
        "a perturbed star needs δv_modes = :coupled or :cowling, the eigenfunctions with " *
        "the metric perturbed or fixed, to match the run's spacetime; got $δv_modes."))
    sol64 = T === Float64 ? sol : tov_solve(Float64; K, Γ, ρ_c, steps)
    return tov_mode_velocity(sol64, Float64.(Tuple(δv)); cowling=δv_modes === :cowling)
end

"""
    tov_boxes!(forest, case; levels = 1, half_width = 5R̄/4) -> forest

The Cowling star's static refinement: `levels` times, every leaf that meets
the cube `[−half_width, half_width]³` around the star (`R̄` its isotropic
radius) and is coarser than the level being made is refined, and the forest
rebalanced, so that the whole star and its surface lie on the finest level. The case is
[`tov_case`](@ref)'s or [`magnetized_tov_case`](@ref)'s.
For `evolve!`'s `refine`: `refine = f -> tov_boxes!(f, case; levels = 2)`.
"""
function tov_boxes!(forest::Forest{3,T}, case::MHDCase; levels::Integer=1,
                    half_width=5 * star_metric(case.fields.w).R̄ / 4) where {T}
    a = T(half_width)
    meets(k) = all(e -> e[1] < a && e[2] > -a, block_extent(T, forest, k))
    for ℓ in 1:levels
        refine!(forest, [k for k in forest.leaves if level(k) < ℓ && meets(k)])
        balance!(forest)
    end
    return forest
end
