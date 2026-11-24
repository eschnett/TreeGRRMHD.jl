# Rung 7: the TOV star on a tabulated EOS ("Validation ladder" 7 in `CODE.md`;
# step 13 of `PLAN.md`). A table's cold β-equilibrium slice (EOS-4,
# EntropyEOS's `cold_beta_slice`) as the barotrope of the solve
# ([`ColdBarotrope`](@ref), `src/tov_table.jl`), at rest on its own frozen
# metric, `Analytic(TOVMetric)`, with no field, as rung 4's polytrope is
# (`src/cases/tov.jl`). `ρ★`, `s` and `Yₑ` vary through the star; the EOS is
# the table seen in geometric units ([`GeometricEOS`](@ref)).
#
# **The surface and the atmosphere** (proposed in step 13). The barotrope's
# surface, `η = 0`, is at the atmosphere's density `ρ_min`, so the star ends
# where the atmosphere begins, and the atmosphere is the slice's own state
# there: cold (EntropyEOS's `s_atm = NaN`, the window's minimum) with the
# replaced state's `Yₑ` (`ye_atm = NaN`). The star's matter below `ρ_min`,
# which the reset would replace anyway, is left out of the solve; the steep
# fall of `ρ★` there is what a table in `r̄²` cannot follow.
#
# **The cold edge** (decided by Erik 2026-11-20: measure). Every state of the
# slice lies on the entropy window's minimum, the table's coldest row, so a
# cell that truncation pushes colder is floored by the recovery (`S_FLOORED`,
# `τ` raised). What that costs is measured, not avoided.

"""
    TabulatedTOVStar{T,NK,NC}

The data of the tabulated TOV case, `isbits`: the star's [`TOVMetric`](@ref)
and the provider that serves it, its [`TOVComposition`](@ref), and the
atmosphere `(ρ_atm, s_atm, Yₑ_atm)`, the slice's state at `ρ_min`. See
`src/cases/tabulated_tov.jl`.
"""
struct TabulatedTOVStar{T,NK,NC}
    metric::TOVMetric{T,NK,Nothing}
    provider::Analytic{TOVMetric{T,NK,Nothing}}
    comp::TOVComposition{T,NC}
    ρ_atm::T
    s_atm::T
    Yₑ_atm::T
end

@inline function wave_fields(w::TabulatedTOVStar, x::SVector{3,T}, t::T) where {T}
    u = x[1]^2 + x[2]^2 + x[3]^2
    z = zero(SVector{3,T})
    if u < w.comp.R̄²
        ρ, s, Yₑ = composition_state(w.comp, u)
        ρ > w.ρ_atm && return ρ, z, s, Yₑ, z, z
    end
    return w.ρ_atm, z, w.s_atm, w.Yₑ_atm, z, z
end

@inline wave_potential(::TabulatedTOVStar, x::SVector{3,T}, t::T) where {T} =
    zero(SVector{3,T}), zero(T)

star_metric(w::TabulatedTOVStar) = w.metric

"""
    tabulated_tov_case([T]; table, eos, ρ_c, ρ_min, L = 24, roots = 4, NK = 129,
                       NC = 129, steps = 4000, samples = 8001)

The TOV star of a tabulated EOS in the Cowling approximation ("Validation
ladder" 7 in `CODE.md`) as a three-dimensional [`MHDCase`](@ref), as
[`tov_case`](@ref) is for the polytrope: `table` (an EntropyEOS `RawTable`)
and `eos` (the `EOSTable` built from it), the star of central rescaled density
`ρ_c` on the table's cold β-equilibrium slice ([`cold_barotrope`](@ref) with
`samples` pressures, [`tov_solve`](@ref) with `steps` RK4 steps), at rest on
its metric (`NK` knots), its composition at `NC` knots, in the closed box
`[−L, L]³` of `roots³` root blocks, the atmosphere at `ρ_min` the slice's
state there, no field. Densities are geometric (`ρ★` per
[`MSUN_DENSITY_CGS`](@ref)); `ρ_c` and `ρ_min` have no default. The case's
data is `case.fields.w`, a [`TabulatedTOVStar`](@ref); [`tov_boxes!`](@ref)
refines around it as around rung 4's.
"""
function tabulated_tov_case(::Type{T}=Float64; table, eos, ρ_c, ρ_min, L=24, roots=4,
                            NK::Integer=129, NC::Integer=129, steps::Integer=4000,
                            samples::Integer=8001) where {T}
    0 < ρ_min < ρ_c || throw(ArgumentError(
        "the atmosphere's density must lie below the star's central density, got " *
        "ρ_min = $ρ_min, ρ_c = $ρ_c."))
    slice = EntropyEOS.cold_beta_slice(table, eos)
    b = cold_barotrope(slice; ρ_surface=ρ_min, samples)
    sol = tov_solve(T, b; ρ_c, steps)
    T(L) > sol.R̄ || throw(ArgumentError(
        "the box [−L, L]³ must hold the star, whose isotropic radius is $(sol.R̄); got L = $L."))
    geos = GeometricEOS(EntropyEOS.EOSTableView(eos); density=MSUN_DENSITY_CGS)
    geos_T = T === Float64 ? geos : EntropyEOS.narrow(geos, T)
    m = TOVMetric(sol; NK)
    prov = Analytic(m)
    comp = TOVComposition(sol; NC)
    atm = barotrope_state(b, 0.0)
    proj = EntropyEOS.default_projection(geos_T, T(ρ_min))
    pol = EntropyEOS.PolicyOptions{T}()                 # cold, the state's own Yₑ
    w = TabulatedTOVStar(m, prov, comp, T(ρ_min), T(atm.s), T(atm.Yₑ))
    return MHDCase(T, Val(3); fields=WaveFields(w), potential=WavePotential(w), eos=geos_T,
                   proj, pol, model=IdealConductor(), provider=prov, periodic=false,
                   extents=ntuple(_ -> (-T(L), T(L)), Val(3)), roots=roots,
                   static_potential=true)
end
