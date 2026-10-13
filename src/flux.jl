# The physical fluxes of the ten cell variables and of the gauge scalar, and
# the light cone that bounds every wave speed.
#
# Densitized throughout, as the evolved variables are ("Matter: total energy
# and momentum" and "Maxwell" in `CODE.md`). In direction `d`, with the
# transport velocity `ṽ = α vᵈ − βᵈ`:
#
#     √γ D      →  √γ D ṽ
#     √γ S_j    →  √γ (α Sᵈ_j − βᵈ S_j),   Sᵈ_j = S^fluid_j vᵈ + p δᵈ_j + Maxwell
#     √γ τ      →  √γ (α (Sᵈ − D vᵈ) − βᵈ τ)
#     √γ D Yₑ   →  √γ D Yₑ ṽ
#     √γ D s    →  √γ D s ṽ
#     𝔈ᵏ        →  −[kdl] α B_l + βᵏ 𝔈ᵈ − βᵈ 𝔈ᵏ
#
# `S_j` and `τ` are the totals, so their fluxes carry the Maxwell stress
# `−EᵈE_j − BᵈB_j + ½ δᵈ_j (E² + B²)`, and every variable's fastest signal is
# light: the wave speeds are the light cone `−βᵈ ± α √γ^{dd}` ("Riemann
# solver" in `CODE.md`). The flux of `𝔈` is Komissarov's `H = αB − β × E`:
# in flat space `∂ₜE = ∇ × B − j`.
#
# The fluid momentum is taken as the total `S_j` minus the field's
# `ε_jkl Eᵏ Bˡ`, so that the flux transports exactly the `S_j` the state
# holds; `prim2con` builds the two consistently.

"""
    light_cone(m::Metric3, ::Val{d}) -> (λ₋, λ₊)

The coordinate speeds of light in direction `d`, `λ± = −βᵈ ± α √γ^{dd}`:
the wave speeds of every variable of the Riemann solver and of the EMF
("Riemann solver" in `CODE.md`). `λ₊ − λ₋ = 2α√γ^{dd} > 0` always.
"""
@inline function light_cone(m::Metric3{T}, ::Val{d}) where {T,d}
    c = m.α * sqrt(m.γu[d, d])
    return -m.β[d] - c, -m.β[d] + c
end

# The unit vector along `d`, and `[kdl] w_l` for every `k`: the symbol cross
# product of `e_d` with `w`.
@inline unit3(::Type{T}, ::Val{d}) where {T,d} = SVector(ntuple(i -> i == d ? one(T) : zero(T),
                                                                Val(3)))

"""
    physical_flux(m::Metric3, cons, prim, E, B, ::Val{d}) -> NTuple{10}

The flux in direction `d` of the densitized evolved state `cons`, whose
primitive state is `prim` (its `v` and `p` are read) and whose contravariant
fields are `Eⁱ`, `Bⁱ`, in the metric `m`. In the order of `U_D … U_E`; see
the top of `src/flux.jl` for the formulas and "Matter: total energy and
momentum" and "Maxwell" in `CODE.md`.

What every approximate Riemann solver reduces to when its two states are
equal. `𝔈`'s flux reads `E`, not `cons`'s `𝔈` slots: the same field, as
`prim2con` and the recovery treat it.
"""
@inline function physical_flux(m::Metric3{T}, cons::NTuple{NU,T}, prim::NTuple{N,T},
                               E::SVector{3,T}, B::SVector{3,T}, ::Val{d}) where {N,T,d}
    α, β, sg = m.α, m.β, m.sqrtγ
    v = prim_v(prim)
    vd = v[d]
    ṽ = α * vd - β[d]
    e_d = unit3(T, Val(d))

    E_low = lower(m, E)
    B_low = lower(m, B)
    u = (sum(E .* E_low) + sum(B .* B_low)) / 2

    S = cons_S(cons)                                  # √γ S_j, total
    S_fluid = S - sg * cross_down(m, E, B)            # √γ ρhW² v_j
    # √γ Sᵈ_j: the fluid's advected momentum and pressure, and Maxwell's stress.
    Sd_j = S_fluid * vd + sg * ((prim_p(prim) + u) * e_d - E[d] * E_low - B[d] * B_low)
    F_S = α * Sd_j - β[d] * S
    F_τ = α * (sum(m.γu[d, :] .* S) - cons_D(cons) * vd) - β[d] * cons_τ(cons)
    # −[kdl] α B_l = α [dkl] B_l, i.e. −α (e_d × B_low) in symbols, and the
    # shift's transport of 𝔈 = √γ E.
    F_𝔈 = -α * symbol_cross(e_d, B_low) + sg * (E[d] * β - β[d] * E)

    return pack_cons(cons_D(cons) * ṽ, F_S, F_τ, cons_DYₑ(cons) * ṽ, cons_Ds(cons) * ṽ, F_𝔈)
end

"""
    gauge_flux(m::Metric3, A, Φ̃, ::Val{j}) -> flux

The flux in direction `j` of the gauge scalar `Φ̃ = √γ Φ` in the
generalized Lorenz gauge, `α √γ Aʲ − βʲ Φ̃` with `Aʲ = γ^jk A_k` from the
covariant potential `A` ("Maxwell" in `CODE.md`): `∂ₜΦ̃ + ∂_j(flux) =
−ξ α Φ̃`. On the vertices' `j`-edges, IllinoisGRMHD's staggering.
"""
@inline function gauge_flux(m::Metric3{T}, A::SVector{3,T}, Φ̃::T, ::Val{j}) where {T,j}
    return m.α * m.sqrtγ * sum(m.γu[j, :] .* A) - m.β[j] * Φ̃
end
