# Primitive to conserved, in the 3+1 form with the electromagnetic parts.
#
# The fluid part is EntropyEOS's own `prim2con`, called in its
# hydrodynamic mode (`B² = 0`) on the scalar speed: it forms `τ` without
# cancellation, and its arithmetic is exactly the energy residual's model in
# `con2prim`, so the round trip through the recovery closes on the same
# formula in both directions ("One con2prim" in `CLAUDE.md`). What is added
# here is the geometry — the speed and the direction of `v` in the metric
# `γ_ij`, the densitization by `√γ` — and the fields, whose energy
# `½(E² + B²)` and momentum `ε_ijk Eʲ Bᵏ` do not depend on the fluid
# ("Matter: total energy and momentum" in `CODE.md`).

"""
    primitive_state(eos, m::Metric3, ρ, v, s, Yₑ, u_guess=NaN) -> NTuple{NPRIM}

The primitive state of density `ρ★`, contravariant velocity `vⁱ`, entropy
`s` and electron fraction `Yₑ`, with the slots the recovery would fill beside
them — the rapidity `w = atanh |v|`, and `p`, `h`, `cs²` and the EOS's warm
start from one evaluation of `eos`. For initial data, and for tests that
build a state the way the recovery returns it.
"""
@inline function primitive_state(eos::EntropyEOS.AbstractEOS, m::Metric3{T}, ρ::T,
                                 v::SVector{3,T}, s::T, Yₑ::T, u_guess::T=T(NaN)) where {T}
    w = rapidity(m, v)
    pt = EntropyEOS.evaluate(eos, ρ, s, Yₑ, u_guess)
    return pack_prim(ρ, v, s, Yₑ, w, pt.p, pt.h, pt.cs², pt.u_solved)
end

# The rapidity of a contravariant velocity, `atanh |v|`. A superluminal `v`
# has none: it is `Inf` rather than a `DomainError`, so that a kernel never
# throws and the recovery downstream flags the resulting non-finite state.
@inline function rapidity(m::Metric3{T}, v::SVector{3,T}) where {T}
    v² = dot_up(m, v, v)
    vmag = sqrt(v² > zero(T) ? v² : zero(T))
    return vmag < one(T) ? atanh(vmag) : T(Inf)
end

"""
    prim2con(eos, m::Metric3, prim, E, B) -> NTuple{10}

The densitized evolved state of the primitive state `prim` (its slots `ρ`,
`v`, `s`, `Yₑ` and the EOS warm start are read; `w`, `p`, `h`, `cs²` are not)
and the fields `Eⁱ`, `Bⁱ`, contravariant and undensitized, in the metric
`m`:

    D   = ρ★ W
    S_i = ρ h W² v_i + ε_ijk Eʲ Bᵏ
    τ   = ρ h W² − p − D + ½ (E² + B²)

each times `√γ`, with `√γ D Yₑ`, `√γ D s` and `𝔈ⁱ = √γ Eⁱ` beside them.
The velocity's magnitude enters through the rapidity `w = atanh |v|`, and
the fluid part is EntropyEOS's `prim2con` at `B² = 0`, whose `τ` has no
cancellation as `w → 0`. Closed form; it cannot fail, and a superluminal
`v` returns a non-finite state rather than throwing.

`B` is the field the cell holds — the average of `curl A + 𝔅̄` — and `E`
is `𝔈/√γ` in an evolution, or the stage solve's iterate inside it. See
"Matter: total energy and momentum" and "Primitive recovery" in `CODE.md`.
"""
@inline function prim2con(eos::EntropyEOS.AbstractEOS, m::Metric3{T}, prim::NTuple{N,T},
                          E::SVector{3,T}, B::SVector{3,T}) where {N,T}
    ρ, v, s, Yₑ = prim_ρ(prim), prim_v(prim), prim_s(prim), prim_Yₑ(prim)
    v_low = lower(m, v)
    v² = sum(v .* v_low)
    vmag = sqrt(v² > zero(T) ? v² : zero(T))
    w = vmag < one(T) ? atanh(vmag) : T(Inf)

    # Hydrodynamic mode, the speed all perpendicular (`cos_vB = 0`): its
    # `S_perp` is `z tanh w` with `z = ρ h W²`, and `S_par` is zero.
    c = EntropyEOS.prim2con(eos, ρ, s, Yₑ, w, zero(T), zero(T), prim_ueos(prim))
    S_fluid = vmag > zero(T) ? (c.S_perp / vmag) * v_low : zero(SVector{3,T})

    S = S_fluid + cross_down(m, E, B)
    τ = c.τ + (dot_up(m, E, E) + dot_up(m, B, B)) / 2
    sg = m.sqrtγ
    return pack_cons(sg * c.D, sg * S, sg * τ, sg * c.D_Y, sg * (c.D * s), sg * E)
end
