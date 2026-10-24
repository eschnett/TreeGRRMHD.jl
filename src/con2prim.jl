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
#
# Both recoveries call EntropyEOS's two levels here and nowhere else —
# `con2prim_safe`, the projection onto the valid range `proj`, and right after
# it `apply_policy`, the atmosphere of `pol` (EntropyEOS 2.0) — and use their
# primitive state and flags, never their conserved output: every caller (the
# reset, the right-hand side's `P`, the stage solve's predictor, iterates and
# ideal seed) rebuilds the conserved variables with this package's `prim2con`.
# Level 2 is EntropyEOS's reference `apply_policy`, because it is this
# package's rule: a state whose density the projection had to raise
# (`FLAG_POL_ρ_FLOORED`), a non-finite input, a failed solve, or a density
# below the trigger is the atmosphere at rest at `ρ_min`, flagged
# `FLAG_POL_ATMOSPHERE` on top of level 1's bits, which say why (decided by
# Erik 2026-10-21; amended 2026-10-24, EntropyEOS 2.0: until then this file
# put a floored state at rest itself; "Floors, the atmosphere and the
# magnetosphere" in `CODE.md`).

# `v` rescaled so that `|v|_γ = tanh w`, or zero where it has no direction.
@inline function velocity_at_rapidity(m::Metric3{T}, v::SVector{3,T}, w::T) where {T}
    V² = dot_up(m, v, v)
    return V² > zero(T) ? (tanh(w) / sqrt(V²)) * v : zero(SVector{3,T})
end

"""
    con2prim_hydro(eos, opts, proj, pol, m::Metric3, cons, E, B, guesses) -> (prim, flags)

The primitive state of the densitized evolved state `cons` in the metric `m`,
with the fields `Eⁱ`, `Bⁱ` (contravariant, undensitized): undensitize, subtract
the electromagnetic energy `½(E² + B²)` from `τ` and the momentum
`ε_ijk Eʲ Bᵏ` from `S_i`, project the fluid momentum to `|S| = √(γ^ij S_i S_j)`,
call `EntropyEOS.con2prim_safe` in hydrodynamic mode (`S_par = 0`,
`S_perp = |S|`, `B² = 0`), and rebuild `vⁱ = tanh w · Sⁱ/|S|`.

`E` is passed rather than read from `cons`, because the implicit stage solve
calls this with its iterate `E(v)` (step 3); in an explicit evaluation it is
`cons_𝔈(cons)/√γ`. `cons`'s `√γ D s` and `𝔈` slots are not read.

`opts`, `proj` and `pol` are EntropyEOS's `Con2PrimOptions`,
`ProjectionOptions` (level 1, the valid range) and `PolicyOptions` (level 2,
the atmosphere), in the units of `eos` (wrap a table in
[`GeometricEOS`](@ref)); `guesses` is `(s, w, u_eos)`, the previous state's
[`guesses`](@ref) or [`cold_guesses`](@ref).

Returns the [`NPRIM`](@ref)-tuple `(ρ, v¹, v², v³, s, Yₑ, w, p, h, cs², u_eos)`
and EntropyEOS's policy flags. It never fails: a state projected onto a face
of the valid range (the density, `Yₑ` and entropy bounds, the rapidity cap)
or replaced by the atmosphere has nonzero flags, one bit per face and
`FLAG_POL_ATMOSPHERE` for the replacement, and `prim2con` of the returned
state is the repaired conserved state. The atmosphere — at rest, at `ρ_min`
— replaces a state on the `ρ_min` face, a non-finite input, a failed solve
and a density below `pol`'s trigger (EntropyEOS's `apply_policy`; amended
2026-10-24, EntropyEOS 2.0). See "Primitive recovery" in `CODE.md`.
"""
@inline function con2prim_hydro(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, E::SVector{3,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}) where {T}
    prim, flags, _ = hydro_recovery(eos, opts, proj, pol, m, cons, E, B, guesses)
    return prim, flags
end

# `con2prim_hydro`, also returning the EOS point of the recovered state, whose
# derivatives the implicit stage solve's Newton Jacobian reads (step 3). The
# stage solve iterates on the fluid's share of the energy and momentum, so the
# recovery is split at that share: `field_budget` subtracts the fields, and
# `fluid_recovery` recovers from what is left. Both are shared with the stage
# solve, so that its recovery at a field `E` is `con2prim_hydro` bit for bit.
@inline function hydro_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, E::SVector{3,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}) where {T}
    isg = inv(m.sqrtγ)
    D = cons_D(cons) * isg
    S = cons_S(cons) * isg
    τ = cons_τ(cons) * isg
    DYₑ = cons_DYₑ(cons) * isg
    S_fluid, τ_fluid = field_budget(m, S, τ, E, B)
    return fluid_recovery(eos, opts, proj, pol, m, D, S_fluid, τ_fluid, DYₑ, guesses)
end

# The fluid's share of the undensitized total momentum `S_i` and energy `τ`:
# `S_i − ε_ijk Eʲ Bᵏ` and `τ − ½(E² + B²)`.
@inline field_budget(m::Metric3{T}, S::SVector{3,T}, τ::T, E::SVector{3,T},
                     B::SVector{3,T}) where {T} =
    (S - cross_down(m, E, B), τ - (dot_up(m, E, E) + dot_up(m, B, B)) / 2)

# The hydrodynamic recovery from the fluid's own undensitized `D`, `S_i`, `τ`
# and `D Yₑ`: EntropyEOS's `con2prim_safe` at `B² = 0` and its `apply_policy`
# right after it, and the velocity rebuilt along `Sⁱ` at the returned
# rapidity.
@inline function fluid_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T}, D::T,
                                S_fluid::SVector{3,T}, τ_fluid::T, DYₑ::T,
                                guesses::NTuple{3,T}) where {T}
    Su = raise(m, S_fluid)
    S² = sum(S_fluid .* Su)
    # A NaN passes through to the policy, which flags it; a quadratic form
    # of a positive-definite metric is never negative but by rounding.
    S_mag = S² < zero(T) ? zero(T) : sqrt(S²)

    cin = EntropyEOS.Con2PrimIn{T}(D, τ_fluid, DYₑ, zero(T), S_mag, zero(T))
    out = EntropyEOS.con2prim_safe(eos, cin, opts, proj, guesses[1], guesses[2], guesses[3])
    out = EntropyEOS.apply_policy(eos, cin, out, proj, pol)
    b = out.base
    v = velocity_at_rapidity(m, Su, b.w)
    prim = pack_prim(b.ρ, v, b.s, b.ye, b.w, b.eos.p, b.eos.h, b.eos.cs², b.eos.u_solved)
    return prim, out.policy_flags, b.eos
end

"""
    atmosphere_prim(eos, proj, pol, Yₑ) -> (ρ, s, Yₑ)

The thermodynamic state of the atmosphere EntropyEOS's `apply_policy` writes,
at rest, for an input whose electron fraction is `Yₑ`: `proj.ρ_min`, `Yₑ` (or
`pol.ye_atm`) clamped into the table's range, and `pol.s_atm` clamped into
the entropy window there, or the window's minimum where `s_atm` is NaN. For
initial data that starts in the atmosphere ([`atmosphere_tube_case`](@ref));
the recovery itself calls `apply_policy`, and a test holds the two to the same
state bit for bit (amended 2026-10-24, EntropyEOS 2.0, which has no
`policy_atmosphere`). `proj.ρ_min` must be set: this package's cases always
state it.
"""
function atmosphere_prim(eos::EntropyEOS.AbstractEOS, proj::EntropyEOS.ProjectionOptions{T},
                         pol::EntropyEOS.PolicyOptions{T}, Yₑ::T) where {T}
    ρ = proj.ρ_min
    isfinite(ρ) && ρ > zero(T) || throw(ArgumentError(
        "atmosphere_prim needs `proj.ρ_min` set (finite and positive), got $ρ: unset, " *
        "EntropyEOS takes the table's lowest density, which no case of this package " *
        "means (CODE.md, \"Floors, the atmosphere and the magnetosphere\")."))
    lo, hi = EntropyEOS.yₑ_bounds(eos)
    y = clamp(isfinite(pol.ye_atm) ? pol.ye_atm : Yₑ, T(lo), T(hi))
    sr = EntropyEOS.srange(eos, ρ, y)
    s = isfinite(pol.s_atm) ? clamp(pol.s_atm, sr.s_min, sr.s_max) : sr.s_min
    return ρ, s, y
end

"""
    con2prim_ideal(eos, opts, proj, pol, m::Metric3, cons, B, guesses) -> (prim, flags)

The ideal-MHD recovery: the primitive state of `cons` on the assumption that
`E = −v × B`, by `EntropyEOS.con2prim_safe` with `B² = γ_ij Bⁱ Bʲ ≠ 0` on the
*total* `τ` and `S_i`, the momentum split along and across `B` as
`S_par = S_i Bⁱ/|B|` and `S_perp = |S − S_par B̂|`. The velocity is
`v_par B̂ + v_perp Ŝ_⊥`, i.e. `(Sⁱ + (S·B)Bⁱ/z)/(z + B²)` with `z = ρhW²` at
the converged state, scaled to `|v| = tanh w` as in [`con2prim_hydro`](@ref).

`cons`'s `𝔈` slots are not read: the ideal `E` is `ideal_E(m, v, B)`.

The two levels and the atmosphere are as in [`con2prim_hydro`](@ref).

This is the seed of the implicit stage solve in the `σΔt → ∞` limit, and its
exact answer for `IdealConductor` — not a second primitive recovery. With
`B = 0` it is `con2prim_hydro` with `E = 0`. See "The implicit stage solve"
in `CODE.md`.
"""
@inline function con2prim_ideal(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}) where {T}
    prim, flags, _ = ideal_recovery(eos, opts, proj, pol, m, cons, B, guesses)
    return prim, flags
end

# `con2prim_ideal`, also returning the EOS point, for the stage solve's seed.
@inline function ideal_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
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
    out = EntropyEOS.con2prim_safe(eos, cin, opts, proj, guesses[1], guesses[2], guesses[3])
    out = EntropyEOS.apply_policy(eos, cin, out, proj, pol)
    b = out.base
    v_perp = S_perp > zero(T) ? (b.v_perp / S_perp) * Su_perp : zero(SVector{3,T})
    v = velocity_at_rapidity(m, b.v_par * b̂ + v_perp, b.w)
    prim = pack_prim(b.ρ, v, b.s, b.ye, b.w, b.eos.p, b.eos.h, b.eos.cs², b.eos.u_solved)
    return prim, out.policy_flags, b.eos
end
