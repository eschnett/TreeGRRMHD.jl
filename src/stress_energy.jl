# The total stress-energy: the 3+1 projections the matter equations are
# written in, and the four-tensor the spacetime reads.
#
# Fluid plus electromagnetic, from the primitives and the fields measured by
# the normal observer ("Matter: total energy and momentum" in `CODE.md`):
#
#     𝓔    = ρhW² − p                + ½(E² + B²)
#     S_i  = ρhW² v_i                + ε_ijk Eʲ Bᵏ
#     S^ij = ρhW² vⁱvʲ + p γ^ij      − EⁱEʲ − BⁱBʲ + ½ γ^ij (E² + B²)
#
# and `T^ab = 𝓔 nᵃnᵇ + Sᵃnᵇ + nᵃSᵇ + S^ab` from them ("Stress-energy for the
# spacetime"). The same three are what `prim2con` packs (densitized, with
# `τ = 𝓔 − D`), what the fluxes transport, and what the sources contract with
# the metric derivatives; `T_ab` is what TreeGH's `Tsrc` holds (GH-3).

"""
    stress_energy(m::Metric3, prim, E, B) -> (𝓔, S, Sij)

The total energy density `𝓔`, momentum density `S_i` (covariant,
`SVector{3}`) and stress `S^ij` (contravariant, `SMatrix{3,3}`) measured by
the normal observer, from the primitive state `prim` (its `ρ`, `v`, `w`, `p`
and `h`) and the contravariant fields `Eⁱ`, `Bⁱ`, all undensitized. See
"Matter: total energy and momentum" in `CODE.md`.
"""
@inline function stress_energy(m::Metric3{T}, prim::NTuple{N,T}, E::SVector{3,T},
                               B::SVector{3,T}) where {N,T}
    v = prim_v(prim)
    p = prim_p(prim)
    W = prim_W(prim)
    z = prim_ρ(prim) * prim_h(prim) * W * W
    u = (dot_up(m, E, E) + dot_up(m, B, B)) / 2
    𝓔 = z - p + u
    S = z * lower(m, v) + cross_down(m, E, B)
    Sij = z * (v * v') + (p + u) * m.γu - E * E' - B * B'
    return 𝓔, S, Sij
end

"""
    T4_upper(m::Metric3, 𝓔, S, Sij) -> SMatrix{4,4}

`T^ab = 𝓔 nᵃnᵇ + Sᵃnᵇ + nᵃSᵇ + S^ab` with `nᵃ = (1, −βⁱ)/α` and
`Sᵃ = (0, γ^ij S_j)`, index 1 being `t`:

    T^tt = 𝓔/α²,   T^ti = (Sⁱ − 𝓔βⁱ/α)/α,
    T^ij = S^ij − (Sⁱβʲ + βⁱSʲ)/α + 𝓔βⁱβʲ/α²

The momentum source contracts it with `∂_j g_ab` (`sources.jl`).
"""
@inline function T4_upper(m::Metric3{T}, 𝓔::T, S::SVector{3,T},
                          Sij::SMatrix{3,3,T,9}) where {T}
    iα = inv(m.α)
    β = m.β
    Su = raise(m, S)
    Ttt = 𝓔 * iα * iα
    Tti = (Su - (𝓔 * iα) * β) * iα
    Tij = Sij - (Su * β' + β * Su') * iα + (𝓔 * iα * iα) * (β * β')
    return SMatrix{4,4,T,16}(Ttt, Tti[1], Tti[2], Tti[3],
                             Tti[1], Tij[1, 1], Tij[2, 1], Tij[3, 1],
                             Tti[2], Tij[1, 2], Tij[2, 2], Tij[3, 2],
                             Tti[3], Tij[1, 3], Tij[2, 3], Tij[3, 3])
end

"""
    T4_lower(m::Metric3, 𝓔, S, Sij) -> SMatrix{4,4}

`T_ab`, the stress-energy TreeGH reads (GH-3), index 1 being `t`. With
`n_a = (−α, 0)`, `S_a = (βᵏS_k, S_i)` and `S_ij = γ_ik γ_jl S^kl`:

    T_tt = α²𝓔 − 2α βᵏS_k + βᵏβˡ S_kl,   T_ti = −α S_i + βᵏ S_ki,   T_ij = S_ij

See "Stress-energy for the spacetime" in `CODE.md`.
"""
@inline function T4_lower(m::Metric3{T}, 𝓔::T, S::SVector{3,T},
                          Sij::SMatrix{3,3,T,9}) where {T}
    α, β = m.α, m.β
    S_ll = mul3(mul3(m.γ, Sij), m.γ)
    Sβ = mul3(S_ll, β)
    Ttt = α * α * 𝓔 - 2 * α * sum(β .* S) + sum(β .* Sβ)
    Tti = Sβ - α * S
    return SMatrix{4,4,T,16}(Ttt, Tti[1], Tti[2], Tti[3],
                             Tti[1], S_ll[1, 1], S_ll[2, 1], S_ll[3, 1],
                             Tti[2], S_ll[1, 2], S_ll[2, 2], S_ll[3, 2],
                             Tti[3], S_ll[1, 3], S_ll[2, 3], S_ll[3, 3])
end
