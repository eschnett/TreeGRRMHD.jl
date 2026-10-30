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
# magnetosphere" in `CODE.md`). *(Amended 2026-10-30, stage 4 of the recovery
# pipeline:)* the recoveries outside the stage solve's iterations call level 1
# as EntropyEOS's two phases, `con2prim_safe_start` and, where it is not done,
# `con2prim_safe_finish`, each in a pass of its own, and level 2 in a third
# (below, "the two-phase recovery"); bit for bit the fused call.

# --- the step cache's entry ---------------------------------------------------------
#
# A cell's entry in the step cache (stage 3 of the recovery pipeline, decided
# by Erik 2026-10-23; "Primitive recovery" in `CODE.md`, "The step cache"):
# EntropyEOS's input to the cell's latest recovery in the current step — the
# key `(D, τ, D_Y, S_∥, S_⊥, B²)` of `Con2PrimIn`, bit for bit — and the output
# of its two levels: the state `(ρ, s, Yₑ, w, v_∥, v_⊥)`, the whole `EOSPoint`
# and the policy flags. Its `(s, w, u_eos)` is the seed of the cell's next
# recovery. Every recovery goes through `two_levels`, which either finds its
# input in the entry and returns the entry, or calls EntropyEOS and returns the
# new entry; both paths hand the caller its state from the entry, so a reused
# recovery is the recovery it replaces bit for bit. Outside a step the entry
# is `invalid_entry`, and every recovery is today's.

"""Number of slots of a step-cache entry: the key (6), the state (6), the
`EOSPoint` (14) and the policy flags."""
const NC = 27
const C_KEY = 1              # D, τ, D_Y, S_∥, S_⊥, B²: EntropyEOS's `Con2PrimIn`
const C_RHO = 7
const C_S = 8                # NaN: the entry is invalid (no recovery yet in this step)
const C_YE = 9
const C_W = 10
const C_VPAR = 11
const C_VPERP = 12
const C_EOS = 13             # the `EOSPoint`'s fields in order; `iters`, `flags` as numbers
const C_POLICY = 27          # below 2¹⁸, exact in `Float32`

# The `EOSPoint`'s flags are the evaluation's — bits 0–5 and, since
# EntropyEOS 2.2, the sound-speed bits 18–19 (`FLAG_CS2_ACAUSAL`,
# `FLAG_CS2_NONPOSITIVE`) — so below 2²⁰, exact in `Float32` too. Nothing in
# this package reads them; they are kept so that a reused recovery hands back
# its `EOSPoint` bit for bit. The policy flags never hold them (amended
# 2026-10-26, EntropyEOS 2.2).

# An entry with nothing in it.
@inline invalid_entry(::Type{T}) where {T} = ntuple(_ -> T(NaN), Val(NC))

# Whether the entry holds a recovery: its `s` is a number (a recovered state
# always has one; invalidation writes NaN there).
@inline entry_valid(e::NTuple{NC,T}) where {T} = !isnan(e[C_S])

# The seed `(s, w, u_eos)` the entry's recovery hands the next one —
# `guesses(prim)` of its state.
@inline entry_seed(e::NTuple{NC,T}) where {T} = (e[C_S], e[C_W], e[C_EOS + 11])

@inline entry_key(cin::EntropyEOS.Con2PrimIn) = (cin.D, cin.τ, cin.D_Y, cin.S_par, cin.S_perp,
                                                 cin.B²)

# Whether the entry is a recovery of `cin`: a valid entry whose key is `cin`'s
# bit for bit (`===` on the floats: `−0.0` is not `0.0`, a conservative
# miss; a NaN matches only the same NaN, which EntropyEOS, a function of its
# input's bits, answers the same way).
@inline entry_hit(e::NTuple{NC,T}, cin::EntropyEOS.Con2PrimIn{T}) where {T} =
    entry_valid(e) && ntuple(k -> e[C_KEY + k - 1], Val(6)) === entry_key(cin)

@inline function make_entry(cin::EntropyEOS.Con2PrimIn{T}, out) where {T}
    b = out.base
    q = b.eos
    return (entry_key(cin)..., b.ρ, b.s, b.ye, b.w, b.v_par, b.v_perp, q.U, q.U_ρ, q.U_s,
            q.U_ρρ, q.U_ρs, q.T̂, q.p, q.h, q.cs², q.T_MeV, q.μ̃, q.u_solved, T(q.iters),
            T(q.flags), T(out.policy_flags))
end

@inline function entry_point(e::NTuple{NC,T}) where {T}
    k = C_EOS - 1
    return EntropyEOS.EOSPoint{T}(e[k + 1], e[k + 2], e[k + 3], e[k + 4], e[k + 5], e[k + 6],
                                  e[k + 7], e[k + 8], e[k + 9], e[k + 10], e[k + 11],
                                  e[k + 12], unsafe_trunc(Int32, e[k + 13]),
                                  unsafe_trunc(UInt32, e[k + 14]))
end

@inline entry_flags(e::NTuple{NC,T}) where {T} = unsafe_trunc(UInt32, e[C_POLICY])

# --- the two-phase recovery ------------------------------------------------------------
#
# Stage 4 of the recovery pipeline (decided by Erik 2026-10-23; "The two-phase
# recovery" in `CODE.md`): the recoveries outside the stage solve's iterations
# — the right-hand side's, both limiters', the stage predictor's — run as three
# passes over the owned cells, each a kernel of its own:
#
#   1. **start**: the input, the step cache's reuse test and seed, and
#      `con2prim_safe_start`, whose `(out, stage)` go into the hand-off;
#   2. **finish**: `con2prim_safe_finish` at the cells whose first stage is not
#      done, behind a per-cell guard (no compaction on the CPU);
#   3. **continuation**: the site's own kernel, with the hand-off for level 1
#      (`two_levels` below): `apply_policy`, then the rest of its work.
#
# EntropyEOS guarantees that start and, where needed, finish are
# `con2prim_safe` bit for bit (that is how it defines it), so the passes are
# the fused kernel bit for bit. Level 1 and level 2 stay in separate kernels:
# the structure a GPU step would compact, and EntropyEOS's HISTORY records a
# CUDA fault with both in one kernel.

"""
    Handoff

The per-owned-cell scratch of the two-phase recovery (stage 4 of the recovery
pipeline; "The two-phase recovery" in `CODE.md`), as four arrays indexed by the
local owned cell and the block, `(n..., b)`:

- `todo`: whether the cell needs `con2prim_safe_finish` — its recovery was not
  reused from the step cache, and `con2prim_safe_start` was not done;
- `cin`, `stage`: EntropyEOS's input and the first stage's hand-off, written
  only where `todo`;
- `out`: level 1's output, written by the first pass where it was done and
  by the second where not; not written where the step cache's entry was
  reused (the third pass takes the entry).

Built with the problem ([`MHDProblem`](@ref)'s `two_phase`), so a `rebuild`
after a regrid sizes it to the new mesh; never transferred, and every pass
writes what the next one reads before it reads it.
"""
struct Handoff{AT,AI,AS,AO}
    todo::AT
    cin::AI
    stage::AS
    out::AO
end

Adapt.@adapt_structure Handoff

# A hand-off of `dims` (the owned cells per dimension and the blocks) on
# `backend`.
function Handoff(::Type{T}, backend, dims::Tuple) where {T}
    return Handoff(allocate(backend, Bool, dims),
                   allocate(backend, EntropyEOS.Con2PrimIn{T}, dims),
                   allocate(backend, EntropyEOS.Con2PrimSafeStage{T}, dims),
                   allocate(backend, EntropyEOS.Con2PrimSafeOut{T}, dims))
end

# EntropyEOS's two levels for `cin` — `con2prim_safe` from `guesses` onto the
# projection `proj`, then `apply_policy` — as an entry, and whether it was
# `entry` reused: the one place the package calls them. With a hand-off `sc`
# (above; stage 4 of the recovery pipeline) level 1 is not called here: its
# output is the one the first two passes left in `sc` at the cell `idx`, and
# this is the third pass's level 2. With `sc = nothing` it is the fused call.
@inline function two_levels(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                            proj::EntropyEOS.ProjectionOptions{T},
                            pol::EntropyEOS.PolicyOptions{T}, cin::EntropyEOS.Con2PrimIn{T},
                            guesses::NTuple{3,T}, entry::NTuple{NC,T}, sc=nothing,
                            idx::Tuple=()) where {T}
    entry_hit(entry, cin) && return entry, true
    out = level1(eos, opts, proj, cin, guesses, sc, idx)
    out = EntropyEOS.apply_policy(eos, cin, out, proj, pol)
    return make_entry(cin, out), false
end

# Level 1: the fused `con2prim_safe`, or the hand-off's output.
@inline level1(eos, opts, proj, cin::EntropyEOS.Con2PrimIn{T}, guesses::NTuple{3,T}, ::Nothing,
               idx::Tuple) where {T} =
    EntropyEOS.con2prim_safe(eos, cin, opts, proj, guesses[1], guesses[2], guesses[3])
@inline level1(eos, opts, proj, cin::EntropyEOS.Con2PrimIn, guesses::NTuple{3}, sc::Handoff,
               idx::Tuple) = @inbounds sc.out[idx...]

# The first pass at the cell `idx`: nothing where the step cache's entry is a
# recovery of `cin` (the third pass reuses it); otherwise
# `con2prim_safe_start` from `guesses`, its output kept where it is done and
# its stage where it is not.
@inline function level1_start!(sc::Handoff, idx::Tuple, eos::EntropyEOS.AbstractEOS,
                               opts::EntropyEOS.Con2PrimOptions{T},
                               proj::EntropyEOS.ProjectionOptions{T},
                               cin::EntropyEOS.Con2PrimIn{T}, guesses::NTuple{3,T},
                               entry::NTuple{NC,T}) where {T}
    if entry_hit(entry, cin)
        @inbounds sc.todo[idx...] = false
        return nothing
    end
    out, st = EntropyEOS.con2prim_safe_start(eos, cin, opts, proj, guesses[1], guesses[2],
                                             guesses[3])
    done = EntropyEOS.is_done(st)
    @inbounds begin
        sc.todo[idx...] = !done
        if done
            sc.out[idx...] = out
        else
            sc.cin[idx...] = cin
            sc.stage[idx...] = st
        end
    end
    return nothing
end

# The second pass at the cell `idx`: `con2prim_safe_finish` where the first
# stage was not done.
@inline function level1_finish!(sc::Handoff, idx::Tuple, eos::EntropyEOS.AbstractEOS,
                                opts::EntropyEOS.Con2PrimOptions, proj::EntropyEOS.ProjectionOptions)
    @inbounds if sc.todo[idx...]
        sc.out[idx...] = EntropyEOS.con2prim_safe_finish(eos, sc.cin[idx...], opts, proj,
                                                         sc.stage[idx...])
    end
    return nothing
end

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
    prim, flags, pt, _, _ = hydro_recovery(eos, opts, proj, pol, m, cons, E, B, guesses,
                                           invalid_entry(T))
    return prim, flags, pt
end

# The same, through the cell's step-cache entry `entry` (`src/evolution.jl`;
# stage 3 of the recovery pipeline): also returns the entry of this recovery
# and whether it was reused from `entry` without calling EntropyEOS. The
# kernels take `E` and `B` as `𝔈 · inv(√γ)` and `𝔅 · inv(√γ)`, and the
# undensitizing here multiplies by the same `inv(√γ)`, so that every site
# hands EntropyEOS the same input bit for bit for the same state (amended
# 2026-10-24: the right-hand side divided by `√γ`).
#
# *(Amended 2026-10-30, stage 4:)* with a hand-off `sc`, level 1 is the one the
# first two passes left at the cell `idx` (`two_levels`).
@inline function hydro_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, E::SVector{3,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}, entry::NTuple{NC,T}, sc=nothing,
                                idx::Tuple=()) where {T}
    D, S, τ, DYₑ = undensitize(m, cons)
    S_fluid, τ_fluid = field_budget(m, S, τ, E, B)
    return fluid_recovery(eos, opts, proj, pol, m, D, S_fluid, τ_fluid, DYₑ, guesses, entry, sc,
                          idx)
end

# The undensitized `D`, `S_i`, `τ` and `D Yₑ` of `cons`, each `· inv(√γ)`.
@inline function undensitize(m::Metric3{T}, cons::NTuple{NU,T}) where {T}
    isg = inv(m.sqrtγ)
    return cons_D(cons) * isg, cons_S(cons) * isg, cons_τ(cons) * isg, cons_DYₑ(cons) * isg
end

# EntropyEOS's input of `hydro_recovery`, and the contravariant fluid
# momentum the velocity is rebuilt along: what the first pass of a two-phase
# recovery computes, by the same arithmetic.
@inline function hydro_input(m::Metric3{T}, cons::NTuple{NU,T}, E::SVector{3,T},
                             B::SVector{3,T}) where {T}
    D, S, τ, DYₑ = undensitize(m, cons)
    S_fluid, τ_fluid = field_budget(m, S, τ, E, B)
    return fluid_input(m, D, S_fluid, τ_fluid, DYₑ)
end

# The fluid's share of the undensitized total momentum `S_i` and energy `τ`:
# `S_i − ε_ijk Eʲ Bᵏ` and `τ − ½(E² + B²)`.
@inline field_budget(m::Metric3{T}, S::SVector{3,T}, τ::T, E::SVector{3,T},
                     B::SVector{3,T}) where {T} =
    (S - cross_down(m, E, B), τ - (dot_up(m, E, E) + dot_up(m, B, B)) / 2)

# EntropyEOS's input of the hydrodynamic recovery from the fluid's own
# undensitized `D`, `S_i`, `τ` and `D Yₑ`, and the contravariant `Sⁱ`.
@inline function fluid_input(m::Metric3{T}, D::T, S_fluid::SVector{3,T}, τ_fluid::T,
                             DYₑ::T) where {T}
    Su = raise(m, S_fluid)
    S² = sum(S_fluid .* Su)
    # A NaN passes through to the policy, which flags it; a quadratic form
    # of a positive-definite metric is never negative but by rounding.
    S_mag = S² < zero(T) ? zero(T) : sqrt(S²)
    return EntropyEOS.Con2PrimIn{T}(D, τ_fluid, DYₑ, zero(T), S_mag, zero(T)), Su
end

# The hydrodynamic recovery from the fluid's own undensitized `D`, `S_i`, `τ`
# and `D Yₑ`: EntropyEOS's `con2prim_safe` at `B² = 0` and its `apply_policy`
# right after it, and the velocity rebuilt along `Sⁱ` at the returned
# rapidity.
@inline function fluid_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T}, D::T,
                                S_fluid::SVector{3,T}, τ_fluid::T, DYₑ::T,
                                guesses::NTuple{3,T}) where {T}
    prim, flags, pt, _, _ = fluid_recovery(eos, opts, proj, pol, m, D, S_fluid, τ_fluid, DYₑ,
                                           guesses, invalid_entry(T))
    return prim, flags, pt
end

# The same through a step-cache entry: also returns this recovery's entry and
# whether it was reused.
@inline function fluid_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T}, D::T,
                                S_fluid::SVector{3,T}, τ_fluid::T, DYₑ::T,
                                guesses::NTuple{3,T}, entry::NTuple{NC,T}, sc=nothing,
                                idx::Tuple=()) where {T}
    cin, Su = fluid_input(m, D, S_fluid, τ_fluid, DYₑ)
    e, hit = two_levels(eos, opts, proj, pol, cin, guesses, entry, sc, idx)
    v = velocity_at_rapidity(m, Su, e[C_W])
    pt = entry_point(e)
    prim = pack_prim(e[C_RHO], v, e[C_S], e[C_YE], e[C_W], pt.p, pt.h, pt.cs², pt.u_solved)
    return prim, entry_flags(e), pt, e, hit
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
    prim, flags, pt, _, _ = ideal_recovery(eos, opts, proj, pol, m, cons, B, guesses,
                                           invalid_entry(T))
    return prim, flags, pt
end

# The same through a step-cache entry: also returns this recovery's entry and
# whether it was reused.
@inline function ideal_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                proj::EntropyEOS.ProjectionOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}, entry::NTuple{NC,T}, sc=nothing,
                                idx::Tuple=()) where {T}
    cin, b̂, S_perp, Su_perp = ideal_input(m, cons, B)
    e, hit = two_levels(eos, opts, proj, pol, cin, guesses, entry, sc, idx)
    v_perp = S_perp > zero(T) ? (e[C_VPERP] / S_perp) * Su_perp : zero(SVector{3,T})
    v = velocity_at_rapidity(m, e[C_VPAR] * b̂ + v_perp, e[C_W])
    pt = entry_point(e)
    prim = pack_prim(e[C_RHO], v, e[C_S], e[C_YE], e[C_W], pt.p, pt.h, pt.cs², pt.u_solved)
    return prim, entry_flags(e), pt, e, hit
end

# EntropyEOS's input of `ideal_recovery` — the total momentum split along and
# across `B` — with the field's direction `b̂`, `S_⊥` and the contravariant
# `S⊥ⁱ` the velocity is rebuilt from.
@inline function ideal_input(m::Metric3{T}, cons::NTuple{NU,T}, B::SVector{3,T}) where {T}
    D, S, τ, DYₑ = undensitize(m, cons)

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

    return EntropyEOS.Con2PrimIn{T}(D, τ, DYₑ, S_par, S_perp, B²), b̂, S_perp, Su_perp
end
