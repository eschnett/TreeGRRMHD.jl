# The source terms, pointwise: the geometric sources of the matter
# equations, Ampère's current, and the gauge damping.
#
# At cell centers, explicit ("Sources" in `CODE.md`): the geometric momentum
# and energy sources, and the `q` part of Ampère's current, `−√γ q (α vⁱ −
# βⁱ)`. At vertices, explicit: the gauge damping `−ξ α Φ̃`. The one implicit
# term is Ampère's conduction current `−α √γ σ W [E + v × B − (v·E) v]`,
# which the stage solve of step 3 handles; it is here as a function so that
# the stage solve's residual and the tests can evaluate it.
#
# **The momentum source has two forms**, equal by algebra ("Matter" in
# `CODE.md`):
#
#     Valencia:   √γ [½ α S^{ik} ∂ⱼγ_ik + S_i ∂ⱼβⁱ − 𝓔 ∂ⱼα]
#     covariant:  ½ √−g T^{ab} ∂ⱼ g_ab,   √−g = α √γ
#
# The covariant form is the default (proposed in step 2): it reads the
# providers' `∂ⱼ g_ab` as they are, over the ten independent components,
# while the Valencia form first derives `∂ⱼα` and `∂ⱼβⁱ` from them
# (`metric.jl`). Measured in step 2 (Julia 1.11, Float64, one call on an
# Apple M3 Pro): 36 ns against 50 ns for `geometric_source`, and 41 ns
# against 74 ns for the whole `explicit_source`. Both are kept: the test that
# they agree to rounding is the check on each. The energy source is the same in
# both, `√γ [α S^{ij} K_ij − Sʲ ∂ⱼα]`.

"""
    geometric_source(::Val{:covariant}, m::Metric3, dm::MetricGradient, 𝓔, S, Sij) -> (s_S, s_τ)
    geometric_source(::Val{:valencia},  m::Metric3, dm::MetricGradient, 𝓔, S, Sij) -> (s_S, s_τ)

The densitized geometric sources of `√γ S_j` (an `SVector{3}`) and `√γ τ`,
from the total `𝓔`, `S_i` and `S^ij` of [`stress_energy`](@ref) and the
metric derivatives `dm`. The momentum source is `½ √−g T^{ab} ∂ⱼ g_ab`
(`:covariant`, the default) or `√γ [½ α S^{ik} ∂ⱼγ_ik + S_i ∂ⱼβⁱ − 𝓔 ∂ⱼα]`
(`:valencia`); the energy source is `√γ [α S^{ij} K_ij − Sʲ ∂ⱼα]` in both.
See "Matter: total energy and momentum" and "Sources" in `CODE.md`.
"""
@inline function geometric_source(::Val{:covariant}, m::Metric3{T}, dm::MetricGradient{T},
                                  𝓔::T, S::SVector{3,T}, Sij::SMatrix{3,3,T,9}) where {T}
    Tu = T4_upper(m, 𝓔, S, Sij)
    sqrtg = m.α * m.sqrtγ
    # `½ T^ab ∂ⱼg_ab` over the ten independent components: the `ti` terms
    # appear twice in the full sum and the diagonal ones once.
    s_S = SVector(ntuple(Val(3)) do j
        @inbounds begin
            dg = dm.dg
            diag = Tu[1, 1] * dg[j, 1, 1] + Tu[2, 2] * dg[j, 2, 2] + Tu[3, 3] * dg[j, 3, 3] +
                   Tu[4, 4] * dg[j, 4, 4]
            off = Tu[1, 2] * dg[j, 1, 2] + Tu[1, 3] * dg[j, 1, 3] + Tu[1, 4] * dg[j, 1, 4] +
                  Tu[2, 3] * dg[j, 2, 3] + Tu[2, 4] * dg[j, 2, 4] + Tu[3, 4] * dg[j, 3, 4]
            sqrtg * (diag / 2 + off)
        end
    end)
    return s_S, energy_source(m, dm, 𝓔, S, Sij)
end

@inline function geometric_source(::Val{:valencia}, m::Metric3{T}, dm::MetricGradient{T},
                                  𝓔::T, S::SVector{3,T}, Sij::SMatrix{3,3,T,9}) where {T}
    dα = d_lapse(m, dm)
    dβ = d_shift(m, dm)
    s_S = SVector(ntuple(Val(3)) do j
        @inbounds m.sqrtγ * (m.α * sum(Sij .* d_spatial(dm, j)) / 2 +
                             sum(S .* dβ[j, :]) - 𝓔 * dα[j])
    end)
    return s_S, energy_source(m, dm, 𝓔, S, Sij, dα)
end

# `√γ [α S^{ij} K_ij − Sʲ ∂ⱼα]`.
@inline energy_source(m::Metric3{T}, dm::MetricGradient{T}, 𝓔::T, S::SVector{3,T},
                      Sij::SMatrix{3,3,T,9}, dα=d_lapse(m, dm)) where {T} =
    m.sqrtγ * (m.α * sum(Sij .* dm.K) - sum(raise(m, S) .* dα))

"""
    ampere_explicit_source(m::Metric3, prim, q) -> SVector{3}

The explicit part of Ampère's source, `−√γ q (α vⁱ − βⁱ)`: the convected
charge and the shift's transport of it. Densitized, for `𝔈`.
"""
@inline ampere_explicit_source(m::Metric3{T}, prim::NTuple{N,T}, q::T) where {N,T} =
    -(m.sqrtγ * q) * (m.α * prim_v(prim) - m.β)

"""
    ampere_stiff_source(model, m::Metric3, prim, E, B) -> SVector{3}

The implicit part of Ampère's source, the conduction current
`−α √γ σ W [Eⁱ + ε^{ijk} v_j B_k − (v_k Eᵏ) vⁱ]`, densitized. Zero, and not
formed, where `σ = 0`; it throws for [`IdealConductor`](@ref), whose
current is not a function of the state. The stage solve of step 3 solves
with it; nothing evaluates it explicitly.
"""
@inline function ampere_stiff_source(model::Conductivity, m::Metric3{T}, prim::NTuple{N,T},
                                     E::SVector{3,T}, B::SVector{3,T}) where {N,T}
    σ = conductivity(model, prim)
    iszero(σ) && return zero(SVector{3,T})
    return -(m.α * m.sqrtγ * σ) * conduction_drift(m, prim, E, B)
end

ampere_stiff_source(::IdealConductor, m::Metric3{T}, prim::NTuple{N,T}, E::SVector{3,T},
                    B::SVector{3,T}) where {N,T} = throw_ideal_current()

"""
    ampere_source(model, m::Metric3, prim, E, B, q) -> SVector{3}

The whole of Ampère's source, `−√γ (α jⁱ − q βⁱ)` with `j` from
[`ohm_current`](@ref): the sum of [`ampere_explicit_source`](@ref) and
[`ampere_stiff_source`](@ref). Densitized, for `𝔈`. In flat space,
`∂ₜE = ∇ × B − j`. See "Maxwell" in `CODE.md`.
"""
@inline ampere_source(model::Conductivity, m::Metric3{T}, prim::NTuple{N,T}, E::SVector{3,T},
                      B::SVector{3,T}, q::T) where {N,T} =
    -m.sqrtγ * (m.α * ohm_current(model, m, prim, E, B, q) - q * m.β)

"""
    gauge_damping(ξ, m::Metric3, Φ̃) -> −ξ α Φ̃

The damping source of the generalized Lorenz gauge (Farris et al. 2012),
at a vertex. `ξ` is (open).
"""
@inline gauge_damping(ξ::T, m::Metric3{T}, Φ̃::T) where {T} = -ξ * m.α * Φ̃

"""
    explicit_source(m::Metric3, dm::MetricGradient, prim, E, B, q, form=Val(:covariant)) -> NTuple{10}

Every explicit source of the ten cell variables, in the order of
`U_D … U_E`: none for `√γ D`, `√γ D Yₑ` and `√γ D s`; the geometric sources
([`geometric_source`](@ref), in `form`) for `√γ S_j` and `√γ τ`; and
[`ampere_explicit_source`](@ref) for `𝔈`. From the primitive state, the
fields, the charge density and the metric with its derivatives.
"""
@inline function explicit_source(m::Metric3{T}, dm::MetricGradient{T}, prim::NTuple{N,T},
                                 E::SVector{3,T}, B::SVector{3,T}, q::T,
                                 form::Val=Val(:covariant)) where {N,T}
    𝓔, S, Sij = stress_energy(m, prim, E, B)
    s_S, s_τ = geometric_source(form, m, dm, 𝓔, S, Sij)
    z = zero(T)
    return pack_cons(z, s_S, s_τ, z, z, ampere_explicit_source(m, prim, q))
end
