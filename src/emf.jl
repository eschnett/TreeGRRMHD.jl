# Faraday's law in potential form, pointwise: the electromotive force at an
# edge, its upwind (UCT–HLL) average over the four states around the edge,
# and the gauge potential whose gradient completes `∂ₜA`.
#
#     ∂ₜ A_i = −(α E_i + ε_ijk βʲ Bᵏ) − ∂ᵢ(α Φ − βʲ A_j)
#
# ("Maxwell" in `CODE.md`). The first term is the EMF, on the `i`-edges; the
# gauge potential `αΦ − βʲA_j` lives on the vertices and its difference
# along an edge is the second. With `E = −v_E × B` the EMF is
# `−ε_ijk (α v_Eʲ − βʲ) Bᵏ`, IllinoisGRMHD's ideal form with the coordinate
# velocity `α v_E − β`; a test pins it.
#
# The UCT–HLL EMF (Del Zanna et al. 2003; Londrillo & Del Zanna 2004; "The
# electromotive force for `A`" in `CODE.md`). At a `d`-edge, with `(d, a, b)`
# cyclic, the four states are indexed by their side of the edge along `a`
# and along `b`. Each state is the full `E` (reconstructed from the cells)
# and the two face-normal fields that cross it: `Bᵃ` from the `a`-faces,
# reconstructed along `b` (so it has a `b`-side), and `Bᵇ` from the
# `b`-faces, reconstructed along `a` (an `a`-side). `Bᵈ` does not enter:
# `(β × B)_d = √γ (βᵃBᵇ − βᵇBᵃ)`. From `∂ₜ𝔅 = −[ijk] ∂ⱼ EMF_k`, `𝔅ᵇ`'s flux
# along `a` is `−EMF_d` and `𝔅ᵃ`'s along `b` is `+EMF_d`, so the HLL
# averages in each direction combine to
#
#     EMF_d = Σ_{ij} wᵃ_i wᵇ_j e_ij
#             + √γ aᵃ (Bᵇ_R − Bᵇ_L) − √γ aᵇ (Bᵃ_R − Bᵃ_L)
#
# with `wᵃ_L = a⁺/(a⁺ + a⁻)`, `wᵃ_R = a⁻/(a⁺ + a⁻)`, `aᵃ = a⁺a⁻/(a⁺ + a⁻)`,
# `a⁺ = max(0, λ₊)` and `a⁻ = max(0, −λ₋)` from the light cone along `a`
# (likewise along `b`). Its dissipation on `𝔅` is the HLL dissipation the
# Riemann solver puts on `𝔈`: together they are the exact upwind solver of
# the vacuum Maxwell equations.

"""
    faraday_emf(m::Metric3, E, B) -> SVector{3}

The electromotive force `α E_i + ε_ijk βʲ Bᵏ`, covariant, from the
contravariant fields: `∂ₜA_i = −EMF_i − ∂ᵢ(αΦ − βʲA_j)`. In flat space it
is `E`. See "Maxwell" in `CODE.md`.
"""
@inline faraday_emf(m::Metric3{T}, E::SVector{3,T}, B::SVector{3,T}) where {T} =
    m.α * lower(m, E) + cross_down(m, m.β, B)

# The HLL weights along one direction: the upwind weights of the left and
# right states, and the dissipation coefficient.
@inline function hll_weights(m::Metric3{T}, ::Val{a}) where {T,a}
    λm, λp = light_cone(m, Val(a))
    ap = max(zero(T), λp)
    am = max(zero(T), -λm)
    s = inv(ap + am)
    return ap * s, am * s, ap * am * s
end

"""
    uct_hll_emf(m::Metric3, E, Bᵃ, Bᵇ, ::Val{d}) -> EMF_d

The upwind EMF `α E_d + (β × B)_d` at a `d`-edge with metric `m`, from the
four states around it ("The electromotive force for `A`" in `CODE.md`).
With `(d, a, b)` cyclic:

- `E = (E_LL, E_LR, E_RL, E_RR)`, the contravariant `SVector{3}` electric
  field of each state, the first letter its side along `a` and the second
  along `b`;
- `Bᵃ = (Bᵃ_L, Bᵃ_R)`, the `a`-face normal field reconstructed to the edge
  from either side along `b`;
- `Bᵇ = (Bᵇ_L, Bᵇ_R)`, the `b`-face normal field reconstructed to the edge
  from either side along `a`.

All undensitized. The result is the HLL average with light-cone speeds plus
the upwind terms in the jumps of the face-normal fields; with all four states
equal and no jumps it is [`faraday_emf`](@ref)'s `d` component, to
rounding. See the top of `src/emf.jl`.
"""
@inline function uct_hll_emf(m::Metric3{T}, E::NTuple{4,SVector{3,T}}, Bᵃ::NTuple{2,T},
                             Bᵇ::NTuple{2,T}, ::Val{d}) where {T,d}
    a, b = mod1(d + 1, 3), mod1(d + 2, 3)              # (d, a, b) cyclic
    wa_L, wa_R, da = hll_weights(m, Val(a))
    wb_L, wb_R, db = hll_weights(m, Val(b))
    α, β, sg = m.α, m.β, m.sqrtγ
    γd = m.γ[d, :]
    # The EMF of state (i, j): `α E_d` lowered with the full metric, and
    # `(β × B)_d` from the normal fields that state sees.
    e(Eij, Ba, Bb) = α * sum(γd .* Eij) + sg * (β[a] * Bb - β[b] * Ba)
    @inbounds begin
        e_LL = e(E[1], Bᵃ[1], Bᵇ[1])
        e_LR = e(E[2], Bᵃ[2], Bᵇ[1])
        e_RL = e(E[3], Bᵃ[1], Bᵇ[2])
        e_RR = e(E[4], Bᵃ[2], Bᵇ[2])
        avg = wa_L * (wb_L * e_LL + wb_R * e_LR) + wa_R * (wb_L * e_RL + wb_R * e_RR)
        return avg + sg * (da * (Bᵇ[2] - Bᵇ[1]) - db * (Bᵃ[2] - Bᵃ[1]))
    end
end

"""
    gauge_potential(m::Metric3, Φ̃, A) -> αΦ − βʲA_j

The scalar whose gradient is the gauge part of Faraday's law,
`∂ₜA_i = −EMF_i − ∂ᵢ(αΦ − βʲA_j)`, from `Φ̃ = √γ Φ` and the covariant
potential `A`. Evaluated at the vertices, where `Φ̃` lives ("Where the
fields live" in `CODE.md`).
"""
@inline gauge_potential(m::Metric3{T}, Φ̃::T, A::SVector{3,T}) where {T} =
    m.α * Φ̃ / m.sqrtγ - sum(m.β .* A)
