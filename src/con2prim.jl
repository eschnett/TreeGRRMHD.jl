# Conserved to primitive: the 3+1 wrapper around EntropyEOS's one recovery.
#
# Because `E` is evolved, the electromagnetic energy and momentum do not
# depend on the fluid velocity, and the recovery is pure hydrodynamics:
# undensitize, subtract the fields' energy and momentum, project the fluid
# momentum to its magnitude in `γ^ij`, and call `con2prim_safe` with
# `B² = 0` ("Primitive recovery" in `CODE.md`). That is `con2prim_hydro`, and
# it is the only primitive recovery the scheme performs, for every σ.
#
# `con2prim_ideal` is the same call with `B² ≠ 0` on the total energy and
# momentum. It is not a second recovery: it is the `σΔt → ∞` limit the
# implicit stage solve seeds its Newton iteration from (step 3), where
# `E = −v × B` and the whole inversion is exactly ideal MHD — and the exact
# answer for `IdealConductor`. ("The implicit stage solve" in `CODE.md`.)
#
# EntropyEOS's calling conventions ("Sharp edges" in `PLAN.md`): its inputs
# are undensitized, `S_par` and `S_perp` are scalars with `S_perp ≥ 0`, its
# warm starts are positional, and its output velocity is a pair of scalars.
# The hydrodynamic mode passes the whole momentum as `S_perp` and
# `S_par = 0`, which is what EntropyEOS's own `prim2con` does when there is
# no field to project against (amended in step 1: `CODE.md` had `S_⊥ = 0`).
#
# The velocity is rebuilt with its direction from the momentum and its
# magnitude from the returned rapidity, `|v| = tanh w`, so that the
# primitive state is self-consistent to rounding: `prim2con` of it reads
# back the `w` the recovery found. Taking the magnitude from `S/(ρhW²)`
# instead would carry the momentum residual, up to the solver's tolerance,
# into `atanh |v|` amplified by `W²`; and on a repaired state (a capped
# rapidity, the atmosphere) `S/(ρhW²)` is not the repaired speed at all
# (proposed in step 1; measured: the alternative leaves 7e-11 in the conserved
# `D` round trip, against 4e-14).

# `v` rescaled so that `|v|_γ = tanh w`, or zero where it has no direction.
@inline function velocity_at_rapidity(m::Metric3{T}, v::SVector{3,T}, w::T) where {T}
    V² = dot_up(m, v, v)
    return V² > zero(T) ? (tanh(w) / sqrt(V²)) * v : zero(SVector{3,T})
end

"""
    con2prim_hydro(eos, opts, pol, m::Metric3, cons, E, B, guesses) -> (prim, flags)

The primitive state of the densitized evolved state `cons` in the metric `m`,
with the fields `Eⁱ`, `Bⁱ` (contravariant, undensitized): undensitize, subtract
the electromagnetic energy `½(E² + B²)` from `τ` and the momentum
`ε_ijk Eʲ Bᵏ` from `S_i`, project the fluid momentum to `|S| = √(γ^ij S_i S_j)`,
call `EntropyEOS.con2prim_safe` in hydrodynamic mode (`S_par = 0`,
`S_perp = |S|`, `B² = 0`), and rebuild `vⁱ = tanh w · Sⁱ/|S|`.

`E` is passed rather than read from `cons`, because the implicit stage solve
calls this with its iterate `E(v)` (step 3); in an explicit evaluation it is
`cons_𝔈(cons)/√γ`. `cons`'s `√γ D s` and `𝔈` slots are not read.

`opts` and `pol` are EntropyEOS's `Con2PrimOptions` and `PolicyOptions`, in
the units of `eos` (wrap a table in [`GeometricEOS`](@ref)); `guesses` is
`(s, w, u_eos)`, the previous state's [`guesses`](@ref) or
[`cold_guesses`](@ref).

Returns the [`NPRIM`](@ref)-tuple `(ρ, v¹, v², v³, s, Yₑ, w, p, h, cs², u_eos)`
and EntropyEOS's policy flags. It never fails: a state the policy repaired
(atmosphere, ceilings, the rapidity cap, the entropy window, non-finite
input) has nonzero flags, and `prim2con` of the returned state is the
repaired conserved state. See "Primitive recovery" in `CODE.md`.
"""
@inline function con2prim_hydro(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, E::SVector{3,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}) where {T}
    isg = inv(m.sqrtγ)
    D = cons_D(cons) * isg
    S = cons_S(cons) * isg
    τ = cons_τ(cons) * isg
    DYₑ = cons_DYₑ(cons) * isg

    S_fluid = S - cross_down(m, E, B)
    τ_fluid = τ - (dot_up(m, E, E) + dot_up(m, B, B)) / 2
    Su = raise(m, S_fluid)
    S² = sum(S_fluid .* Su)
    # A NaN passes through to the policy, which flags it; a quadratic form
    # of a positive-definite metric is never negative but by rounding.
    S_mag = S² < zero(T) ? zero(T) : sqrt(S²)

    cin = EntropyEOS.Con2PrimIn{T}(D, τ_fluid, DYₑ, zero(T), S_mag, zero(T))
    out = EntropyEOS.con2prim_safe(eos, cin, opts, pol, guesses[1], guesses[2], guesses[3])
    b = out.base
    v = velocity_at_rapidity(m, Su, b.w)
    prim = pack_prim(b.ρ, v, b.s, b.ye, b.w, b.eos.p, b.eos.h, b.eos.cs², b.eos.u_solved)
    return prim, out.policy_flags
end

"""
    con2prim_ideal(eos, opts, pol, m::Metric3, cons, B, guesses) -> (prim, flags)

The ideal-MHD recovery: the primitive state of `cons` on the assumption that
`E = −v × B`, by `EntropyEOS.con2prim_safe` with `B² = γ_ij Bⁱ Bʲ ≠ 0` on the
*total* `τ` and `S_i`, the momentum split along and across `B` as
`S_par = S_i Bⁱ/|B|` and `S_perp = |S − S_par B̂|`. The velocity is
`v_par B̂ + v_perp Ŝ_⊥`, i.e. `(Sⁱ + (S·B)Bⁱ/z)/(z + B²)` with `z = ρhW²` at
the converged state, scaled to `|v| = tanh w` as in [`con2prim_hydro`](@ref).

`cons`'s `𝔈` slots are not read: the ideal `E` is `ideal_E(m, v, B)`.

This is the seed of the implicit stage solve in the `σΔt → ∞` limit, and its
exact answer for `IdealConductor` — not a second primitive recovery. With
`B = 0` it is `con2prim_hydro` with `E = 0`. See "The implicit stage solve"
in `CODE.md`.
"""
@inline function con2prim_ideal(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}) where {T}
    isg = inv(m.sqrtγ)
    D = cons_D(cons) * isg
    S = cons_S(cons) * isg
    τ = cons_τ(cons) * isg
    DYₑ = cons_DYₑ(cons) * isg

    Su = raise(m, S)
    B² = dot_up(m, B, B)
    if B² > zero(T)
        Bmag = sqrt(B²)
        b̂ = B / Bmag
        S_par = sum(S .* b̂)
        Su_perp = Su - S_par * b̂
    else
        # No field to project against: everything is perpendicular, as in
        # the hydrodynamic mode. A non-finite `B²` lands here too and is
        # passed on for the policy to flag.
        b̂ = zero(SVector{3,T})
        S_par = zero(T)
        Su_perp = Su
    end
    S_perp² = dot_up(m, Su_perp, Su_perp)
    S_perp = S_perp² < zero(T) ? zero(T) : sqrt(S_perp²)

    cin = EntropyEOS.Con2PrimIn{T}(D, τ, DYₑ, S_par, S_perp, B²)
    out = EntropyEOS.con2prim_safe(eos, cin, opts, pol, guesses[1], guesses[2], guesses[3])
    b = out.base
    v_perp = S_perp > zero(T) ? (b.v_perp / S_perp) * Su_perp : zero(SVector{3,T})
    v = velocity_at_rapidity(m, b.v_par * b̂ + v_perp, b.w)
    prim = pack_prim(b.ρ, v, b.s, b.ye, b.w, b.eos.p, b.eos.h, b.eos.cs², b.eos.u_solved)
    return prim, out.policy_flags
end
