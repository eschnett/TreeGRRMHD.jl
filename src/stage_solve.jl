# The implicit stage solve: Ohm's conduction current, per cell, per implicit
# stage of the IMEX integrator ("The implicit stage solve" in `CODE.md`).
#
# IMEXRungeKutta's `solve_imp!(U, u★, γΔt, p, t)` asks for `U = u★ + γΔt g(U)`,
# where `g` is the one implicit term of the scheme, Ampère's conduction source
# `−α √γ σ W [E + v × B − (v·E) v]`. It acts on `𝔈` alone: `D`, `S_i`, `τ`,
# `D Yₑ` and `D s` are totals the conduction current only moves energy and
# momentum within, and `B` is explicit in the stage. Divided by `√γ`,
#
#     E = E★ − c W [E + v × B − (v·E) v],     c = γΔt α σ,
#
# where `v` and `σ` belong to the primitive state recovered from `u★` with the
# field `E`. The field and the fluid are coupled through that recovery and
# through nothing else:
#
# - **given the velocity**, `E` is linear: `(1 + cW) E − cW (E·v) v =
#   E★ − cW v × B`, a rank-one update of a multiple of the identity, solved
#   in closed form by Sherman–Morrison (`ohm_E`);
# - **given `E`**, the fluid is EntropyEOS's one recovery in hydrodynamic
#   mode, on the fluid's share of the energy and momentum,
#   `X(E) = (S_i − ε_ijk Eʲ Bᵏ, τ − ½(E² + B²))` ("One con2prim" in
#   `CLAUDE.md`).
#
# **The iteration runs on that share, `X = (S_f, τ_f)`** (decided in step 3,
# amending `CODE.md`'s "root find for `v`"): each iteration recovers the
# fluid from `X`, takes its velocity `u = W v`, solves Ohm's law for `E`, and
# compares `X` with `X(E)`. The map is the same as the root find for the
# velocity, cut at a different place, and each iteration still costs exactly
# one recovery. The cut matters where the field's energy dwarfs the fluid's
# (`b²/ρ ≫ 1`): there `X(E)` is a small difference of large numbers, and a
# velocity off by `δu` moves it by `λ b²/(ρh) δu` relative to the fluid —
# beyond what the fluid can hold, so that the recovery repairs it and Newton
# on `u` has a basin of `δu ~ 10⁻⁴` (measured in step 3, `b²/ρ = 10³`).
# Newton on `X` hands the recovery only the linearly predicted share, which
# stays on the fluid's own scale.
#
# - The **fixed point** `X ← X(E(u(X)))` contracts like `λ b²/(ρh)`,
#   `λ = cW/(1 + cW)`: fast where `σΔt` or the magnetization is small.
# - **Newton** on `R(X) = X − X(E(u(X)))` has the Jacobian `1 − P Q`, with
#   `Q = ∂u/∂X` (4 → 3), the recovery's own derivative, and
#   `P = ∂X/∂E · ∂E/∂u` (3 → 4); by Woodbury its inverse needs only the 3×3
#   `1 − Q P`, whose `Q P = K` is the fixed point's Jacobian in `u`. Both are
#   analytic (decided in step 3): `∂E/∂u` by implicit differentiation of the
#   linear system, and `∂u/∂X` by implicit differentiation of the
#   hydrodynamic inversion at the recovered state, from the EOS point the
#   recovery returns (`U_s`, `U_ρs`, `cs²`) — the derivatives EntropyEOS's own
#   Newton uses. A density-dependent σ adds a rank-one term through
#   `∂ρ/∂X`. No extra recovery and no finite difference. A step that grows
#   the residual is halved.
# - Newton is **seeded**, where `λ > ½`, from the `σΔt → ∞` limit:
#   `con2prim_ideal`, the same EntropyEOS routine with `B² ≠ 0`, exact there;
#   from the predictor otherwise; and from whichever of the two the policy
#   did not have to repair, if only one.
# - A share the recovery had to **repair** is replaced by the repaired
#   state's own, so that every iterate is a fluid state with a derivative.
# - If the iteration does not converge — where neither seed is a fluid
#   state — the **bracketed path** (`stage_bracket.jl`, added in step 3
#   after review): a scan over the fluid's rapidity for the roots of one
#   function of one variable, each root seeding the iteration again.
# - If that finds nothing either, the **lagged velocity** of Palenzuela et
#   al. (2009): `E` solved at the fixed velocity of the best iterate, and
#   the fluid recovered from that `E`.
#
# Where each branch is taken, and what it costs, is measured in
# `test/stage_solve_tests.jl` and recorded in `CODE.md`.
#
# The velocity is carried as `u = W v` (proposed in step 3): every `u` is
# subluminal, and `W = √(1 + u²)` is as well conditioned as `u`, where
# through `v` the Jacobian of `E` carries a `W²` from `∂W/∂v`.
#
# What is returned is `E` and the primitive state it was solved with: the
# stage writes only `E` into `U` ("Sharp edges" in `PLAN.md`), and the
# policy's repair of an inadmissible `u★` stays inside the solve.

# --- The result -----------------------------------------------------------------

"""
    OhmStageResult(flags, policy_flags)

What [`solve_ohm_stage`](@ref) did, as two bit words that a kernel can store
and a chunk can reduce by `|` and count by bit:

- `flags`, the solve's own: exactly one of the branch bits
  [`OHM_SIGMA_ZERO`](@ref), [`OHM_IDEAL`](@ref), [`OHM_FIXED_POINT`](@ref),
  [`OHM_NEWTON`](@ref), [`OHM_BRACKETED`](@ref), [`OHM_FALLBACK`](@ref),
  and any of the modifiers
  [`OHM_IDEAL_SEED`](@ref), [`OHM_SWITCHED`](@ref),
  [`OHM_BACKTRACKED`](@ref), [`OHM_REPAIRED`](@ref);
- `policy_flags`, EntropyEOS's policy flags of the returned primitive
  state's recovery, as [`con2prim_hydro`](@ref) returns them — the word that
  goes into `P`.

See "The implicit stage solve" in `CODE.md`.
"""
struct OhmStageResult
    flags::UInt32
    policy_flags::UInt32
end

"""`σ = 0` at the predictor's recovery: `E = E★`, and the hydrodynamic recovery at it."""
const OHM_SIGMA_ZERO = UInt32(1) << 0
"""`IdealConductor`: the ideal recovery, and `E = −v × B`."""
const OHM_IDEAL = UInt32(1) << 1
"""Converged by the fixed-point iteration."""
const OHM_FIXED_POINT = UInt32(1) << 2
"""Converged by Newton's method."""
const OHM_NEWTON = UInt32(1) << 3
"""Neither the iteration nor the bracketed search converged: the lagged velocity, the
best one the iteration saw."""
const OHM_FALLBACK = UInt32(1) << 4
"""Newton was seeded from the ideal recovery (`λ > λ_ideal`)."""
const OHM_IDEAL_SEED = UInt32(1) << 5
"""The fixed point stopped contracting, and Newton took over."""
const OHM_SWITCHED = UInt32(1) << 6
"""Newton shortened at least one step."""
const OHM_BACKTRACKED = UInt32(1) << 7
"""At least one recovery inside the iteration was repaired by the policy."""
const OHM_REPAIRED = UInt32(1) << 8
"""The iteration did not converge, and the bracketed search over the fluid's rapidity did."""
const OHM_BRACKETED = UInt32(1) << 9

"""
    converged(r::OhmStageResult)

Whether the stage equation holds to the solve's tolerance: every branch but
[`OHM_FALLBACK`](@ref). A repaired state (an inadmissible `u★`) can
converge; its `policy_flags` say so.
"""
@inline converged(r::OhmStageResult) = r.flags & OHM_FALLBACK == 0

# --- The options ----------------------------------------------------------------

"""
    OhmStageOptions{T}(; tol, maxiter=30, θ_newton=1/100, λ_ideal=1/2, max_backtrack=8,
                       nscan=32, polish=8, max_eval=800)

The knobs of [`solve_ohm_stage`](@ref), `isbits`:

- `tol`: converged when the fluid's share of the energy and momentum closes,
  `|X − X(E)| ≤ tol · (τ + D)` — relative to the total energy density,
  which is how the recovery's round trip is measured ("Primitive recovery"
  in `CODE.md`); by default `max(1e-12, 1024 eps(T))`, EntropyEOS's own
  recovery tolerance at `Float64` (proposed in step 3);
- `maxiter`: the most recoveries the iteration makes after its seeds;
- `θ_newton`: the contraction factor, estimated from the Jacobian at the
  predictor, above which the solve uses Newton rather than the fixed point;
  the fixed point also hands over to Newton when an iteration shrinks the
  residual by less than half;
- `λ_ideal`: Newton is seeded from the ideal recovery when
  `λ = cW/(1 + cW)` at the predictor exceeds it;
- `max_backtrack`: the most consecutive halvings of a Newton step that grew
  the residual;
- `nscan`: the points of the bracketed path's scan over the fluid's
  rapidity, from 0 to the policy's cap;
- `polish`: the most recoveries Newton takes from each root the bracketed
  path finds;
- `max_eval`: the most evaluations of the bracketed path's residual, which
  bounds its cost where it finds nothing.
"""
struct OhmStageOptions{T}
    tol::T
    maxiter::Int32
    θ_newton::T
    λ_ideal::T
    max_backtrack::Int32
    nscan::Int32
    polish::Int32
    max_eval::Int32
end

@inline ohm_tol(::Type{T}) where {T} = max(T(1//10^12), 1024 * eps(T))

function OhmStageOptions{T}(; tol=ohm_tol(T), maxiter=30, θ_newton=T(1//100), λ_ideal=T(1//2),
                            max_backtrack=8, nscan=32, polish=8, max_eval=800) where {T}
    return OhmStageOptions{T}(T(tol), Int32(maxiter), T(θ_newton), T(λ_ideal),
                              Int32(max_backtrack), Int32(nscan), Int32(polish), Int32(max_eval))
end

# --- Pointwise pieces -----------------------------------------------------------

# `W = √(1 + γ_ij uⁱ uʲ)`.
@inline lorentz(m::Metric3{T}, u::SVector{3,T}) where {T} = sqrt(one(T) + dot_up(m, u, u))

# The spatial four-velocity `u = W v = sinh w v̂` of a primitive state.
@inline four_velocity(prim::NTuple) = prim_W(prim) * prim_v(prim)

@inline ident3(::Type{T}) where {T} =
    SMatrix{3,3,T,9}(one(T), zero(T), zero(T), zero(T), one(T), zero(T), zero(T), zero(T),
                     one(T))

# `M⁻¹ r` for `M = (1 + cW) 1 − (c/W) u u_low`, the matrix of the linear
# system for `E` at fixed `u`: Sherman–Morrison, with `u·E` from the dot
# product of the system with `u` (`u² = W² − 1`).
@inline function sm_solve(m::Metric3{T}, c::T, W::T, u::SVector{3,T}, r::SVector{3,T}) where {T}
    cw = c / W
    return (r + (cw * dot_up(m, u, r) / (one(T) + cw)) * u) / (one(T) + c * W)
end

"""
    ohm_E(m::Metric3, c, u, E★, B) -> E

The electric field of the implicit stage at the fixed velocity `u = W v`:

    (1 + cW) E − cW (E·v) v = E★ − cW v × B,     c = γΔt α σ,

in closed form (Sherman–Morrison). Exactly `E★` where `c = 0`: the mask
branches, so nothing of `u` reaches a cell with no conductivity. As
`c → ∞` it tends to `−v × B`. See "The implicit stage solve" in `CODE.md`.
"""
@inline function ohm_E(m::Metric3{T}, c::T, u::SVector{3,T}, E★::SVector{3,T},
                       B::SVector{3,T}) where {T}
    iszero(c) && return E★
    W = lorentz(m, u)
    return sm_solve(m, c, W, u, E★ - c * raise(m, cross_down(m, u, B)))
end

# The outer product `a bᵀ`, one multiply per entry.
@inline outer3(a::SVector{3,T}, b::SVector{3,T}) where {T} = a .* transpose(b)

# `−√γ [jlk] Bᵏ`: the matrix of `E ↦ −ε_jlk Eˡ Bᵏ`, the change of the
# fluid's momentum with `E`.
@inline function cross_matrix(m::Metric3{T}, B::SVector{3,T}) where {T}
    sB = m.sqrtγ * B
    z = zero(T)
    # Column-major: the columns are l = 1, 2, 3.
    return SMatrix{3,3,T,9}(z, sB[3], -sB[2], -sB[3], z, sB[1], sB[2], -sB[1], z)
end

# `A⁻¹ b` for a general 3×3 `A`, by cofactors: no pivoting and no branch, so
# the same arithmetic on every backend. A singular `A` gives a non-finite
# result, which the caller tests for.
@inline function solve3(A::SMatrix{3,3,T,9}, b::SVector{3,T}) where {T}
    a11, a21, a31, a12, a22, a32, a13, a23, a33 = Tuple(A)
    c11 = a22 * a33 - a23 * a32
    c12 = a23 * a31 - a21 * a33
    c13 = a21 * a32 - a22 * a31
    c21 = a13 * a32 - a12 * a33
    c22 = a11 * a33 - a13 * a31
    c23 = a12 * a31 - a11 * a32
    c31 = a12 * a23 - a13 * a22
    c32 = a13 * a21 - a11 * a23
    c33 = a11 * a22 - a12 * a21
    det = a11 * c11 + a12 * c12 + a13 * c13
    x = SVector(c11 * b[1] + c21 * b[2] + c31 * b[3], c12 * b[1] + c22 * b[2] + c32 * b[3],
                c13 * b[1] + c23 * b[2] + c33 * b[3])
    return x / det
end

# The size of a mixed tensor `Kⁱⱼ` in the metric: its Frobenius norm in an
# orthonormal frame, `√tr(γ K γ⁻¹ Kᵀ)`.
@inline function mixed_norm(m::Metric3{T}, K::SMatrix{3,3,T,9}) where {T}
    n² = sum(mul3(m.γ, K) .* mul3(K, m.γu))
    return sqrt(n² > zero(T) ? n² : zero(T))
end

# The fluid's undensitized `S_i` and `τ` of a primitive state, at its own
# rapidity: EntropyEOS's `prim2con` in hydrodynamic mode, as `prim2con` here
# calls it.
@inline function fluid_share(eos::EntropyEOS.AbstractEOS, m::Metric3{T}, prim::NTuple) where {T}
    c = EntropyEOS.prim2con(eos, prim_ρ(prim), prim_s(prim), prim_Yₑ(prim), prim_w(prim), zero(T),
                            zero(T), prim_ueos(prim))
    v = prim_v(prim)
    v² = dot_up(m, v, v)
    S = v² > zero(T) ? (c.S_perp / sqrt(v²)) * lower(m, v) : zero(SVector{3,T})
    return S, c.τ
end

"""
    recovery_jacobian(m::Metric3, D, prim, pt) -> (∂u/∂S, ∂u/∂τ, ∂ρ/∂S, ∂ρ/∂τ)

The derivative of the hydrodynamic recovery's `uⁱ = W vⁱ` with respect to the
fluid's undensitized momentum `S_j` (a 3×3 matrix) and energy `τ` (a
vector), and of its density `ρ` (a vector and a scalar), at its solution
`prim` with the EOS point `pt`, for the undensitized `D`.

The recovery solves `|S| = z tanh w` and `τ = z − p − D` for the entropy `s`
and the rapidity `w`, with `z = ρhW² = D h cosh w` and `ρ = D/cosh w`, and
returns `u = sinh w Ŝ`. The derivatives of those two equations in `(s, w)`
are EntropyEOS's own (`residuals` in its `con2prim.jl`, at `B² = 0`); the
implicit function theorem gives `∂w/∂|S|` and `∂w/∂τ`, and the direction
`Ŝ` turns with `S` at the rate `sinh w/|S| = 1/(D h)`. A state the policy
repaired is not the solution of those equations, and its derivative is not
the recovery's.
"""
@inline function recovery_jacobian(m::Metric3{T}, D::T, prim::NTuple,
                                   pt::EntropyEOS.EOSPoint{T}) where {T}
    w, ρ, h, cs² = prim_w(prim), prim_ρ(prim), prim_h(prim), prim_cs²(prim)
    ch, th = cosh(w), tanh(w)
    z = D * h * ch
    z_w = z * (one(T) - cs²) * th
    z_s = D * ch * (pt.U_s + ρ * pt.U_ρs)
    p_w = -ρ * h * cs² * th
    p_s = ρ * ρ * pt.U_ρs
    a11 = z_s * th                           # ∂|S|/∂s
    a12 = z_w * th + z * (one(T) - th * th)  # ∂|S|/∂w
    a21 = z_s - p_s                          # ∂τ/∂s
    a22 = z_w - p_w                          # ∂τ/∂w
    det = a11 * a22 - a12 * a21
    w_S = -a21 / det
    w_τ = a11 / det

    v = prim_v(prim)
    v² = dot_up(m, v, v)
    Ŝ = v² > zero(T) ? v / sqrt(v²) : zero(SVector{3,T})
    ŜŜ = outer3(Ŝ, Ŝ)
    du_dS = (m.γu - ŜŜ) / (D * h) + (ch * w_S) * ŜŜ
    du_dτ = (ch * w_τ) * Ŝ
    # `ρ = D / cosh w`, for a conductivity that depends on it.
    ρ_w = -ρ * th
    return du_dS, du_dτ, (ρ_w * w_S) * Ŝ, ρ_w * w_τ
end

"""
    ohm_jacobian(m::Metric3, c, u, E, B) -> ∂E/∂u

The derivative of [`ohm_E`](@ref) with respect to `u`, at `E = ohm_E(m, c, u,
E★, B)`, by implicit differentiation of the linear system
`G = (1 + cW) E − (c/W)(E·u) u + c u × B − E★ = 0`: `∂E/∂u = −M⁻¹ ∂G/∂u`.
`σ` is held fixed.
"""
@inline function ohm_jacobian(m::Metric3{T}, c::T, u::SVector{3,T}, E::SVector{3,T},
                              B::SVector{3,T}) where {T}
    iszero(c) && return zero(SMatrix{3,3,T,9})
    W = lorentz(m, u)
    cw = c / W
    u_low = lower(m, u)
    E_low = lower(m, E)
    Eu = dot_up(m, E, u)
    X = -mul3(m.γu, cross_matrix(m, B))      # ∂(u × B)/∂u
    dG_du = outer3(cw * (E + (Eu / (W * W)) * u), u_low) -
            cw * (Eu * ident3(T) + outer3(u, E_low)) + c * X
    return -hcat(sm_solve(m, c, W, u, dG_du[:, 1]), sm_solve(m, c, W, u, dG_du[:, 2]),
                 sm_solve(m, c, W, u, dG_du[:, 3]))
end

# `∂E/∂c` of [`ohm_E`](@ref) at `E = ohm_E(m, c, u, E★, B)`: `−M⁻¹ ∂G/∂c`,
# with `∂G/∂c = W E − (E·u) u / W + u × B`.
@inline function ohm_dc(m::Metric3{T}, c::T, u::SVector{3,T}, E::SVector{3,T},
                        B::SVector{3,T}) where {T}
    W = lorentz(m, u)
    g = W * E - (dot_up(m, E, u) / W) * u + raise(m, cross_down(m, u, B))
    return -sm_solve(m, c, W, u, g)
end

"""
    newton_step(m, D, prim, pt, c, u, E, B, dc_dρ, R_S, R_τ) -> (ΔS, Δτ)

Newton's step for the fluid's share `X = (S_f, τ_f)`, `ΔX = −J⁻¹ R`, at the
iterate whose recovery is `(prim, pt)`, with the velocity `u`, the field `E`
at `c`, and the residual `R = (R_S, R_τ)`. The Jacobian is
`J = 1 − P Q − p qᵀ`: `Q = ∂u/∂X` and `P = ∂X/∂E · ∂E/∂u` through the
velocity, whose inverse is Woodbury's, `J₀⁻¹ r = r + P (1 − K)⁻¹ Q r` with
the 3×3 `K = Q P`; and `q = ∂ρ/∂X`, `p = ∂X/∂E · ∂E/∂c · dc/dρ` through a
density-dependent conductivity, a rank-one update by Sherman–Morrison. A
singular `1 − K` gives a non-finite step, which the caller tests for.
"""
@inline function newton_step(m::Metric3{T}, D::T, prim::NTuple, pt::EntropyEOS.EOSPoint{T}, c::T,
                             u::SVector{3,T}, E::SVector{3,T}, B::SVector{3,T}, dc_dρ::T,
                             R_S::SVector{3,T}, R_τ::T) where {T}
    du_dS, du_dτ, dρ_dS, dρ_dτ = recovery_jacobian(m, D, prim, pt)
    C = cross_matrix(m, B)
    E_low = lower(m, E)
    dE_du = ohm_jacobian(m, c, u, E, B)
    A = ident3(T) - mul3(mul3(du_dS, C) - outer3(du_dτ, E_low), dE_du)
    # `J₀⁻¹ r`, for `r = (r_S, r_τ)`.
    function j0(r_S, r_τ)
        δE = mul3(dE_du, solve3(A, mul3(du_dS, r_S) + r_τ * du_dτ))
        return r_S + mul3(C, δE), r_τ - sum(E_low .* δE)
    end
    a_S, a_τ = j0(R_S, R_τ)
    if !iszero(dc_dρ)
        e_c = dc_dρ * ohm_dc(m, c, u, E, B)
        b_S, b_τ = j0(mul3(C, e_c), -sum(E_low .* e_c))
        qa = sum(dρ_dS .* a_S) + dρ_dτ * a_τ
        qb = sum(dρ_dS .* b_S) + dρ_dτ * b_τ
        f = qa / (one(T) - qb)
        a_S, a_τ = a_S + f * b_S, a_τ + f * b_τ
    end
    return -a_S, -a_τ
end

"""
    stage_jacobian(m::Metric3, D, prim, pt, c, u, E, B) -> K

`K = (∂u/∂X)(∂X/∂E)(∂E/∂u)`, the Jacobian of the fixed-point map
`u ↦ u(X(E(u)))` at the iterate `u = W v` of the recovery `(prim, pt)` and
its field `E`. Its size is the fixed point's contraction factor, and
`1 − K` is the 3×3 matrix Newton on the fluid's share inverts.
"""
@inline function stage_jacobian(m::Metric3{T}, D::T, prim::NTuple, pt::EntropyEOS.EOSPoint{T},
                                c::T, u::SVector{3,T}, E::SVector{3,T},
                                B::SVector{3,T}) where {T}
    du_dS, du_dτ, _, _ = recovery_jacobian(m, D, prim, pt)
    du_dE = mul3(du_dS, cross_matrix(m, B)) - outer3(du_dτ, lower(m, E))
    return mul3(du_dE, ohm_jacobian(m, c, u, E, B))
end

# --- The solve ------------------------------------------------------------------

# The residual of an iterate: the field `E` at the velocity of its recovery
# `prim`, the fluid's share `X(E)` that field leaves of the totals `S`, `τ`,
# and how far the iterate's share `(S_f, τ_f)` is from it, relative to the
# total energy density `scale`.
@inline function stage_residual(m::Metric3{T}, c::T, prim::NTuple, S_f::SVector{3,T}, τ_f::T,
                                S::SVector{3,T}, τ::T, E★::SVector{3,T}, B::SVector{3,T},
                                scale::T) where {T}
    E = ohm_E(m, c, four_velocity(prim), E★, B)
    S_E, τ_E = field_budget(m, S, τ, E, B)
    R_S = S_f - S_E
    err = max(sqrt(dot_down(m, R_S, R_S)), abs(τ_f - τ_E)) / scale
    return err, E, S_E, τ_E
end

"""
    solve_ohm_stage(eos, opts, pol, σ::Conductivity, m::Metric3, cons★, E★, B, γΔt, guesses;
                    stage=OhmStageOptions{T}()) -> (E, prim, result::OhmStageResult, iters)

The implicit stage of Ohm's law in one cell: the field `Eⁱ` (contravariant,
undensitized) with

    E = E★ − γΔt α σ W [E + v × B − (v·E) v],

i.e. `𝔈 = 𝔈★ + γΔt ·` [`ampere_stiff_source`](@ref), where `v` and `σ`
belong to the fluid recovered from `cons★` with the field `E`. `cons★` is
the densitized stage predictor `u★`, whose `𝔈` slots are not read (`E★ =
𝔈★/√γ` is passed, as [`con2prim_hydro`](@ref) takes its `E`); `B` is the
cell's field, `γΔt` the stage's implicit coefficient times `Δt`, and
`guesses` the warm start `(s, w, u_eos)` of the cell's last recovery
([`guesses`](@ref), or [`cold_guesses`](@ref)). `opts` and `pol` are
EntropyEOS's, as for `con2prim_hydro`.

Returns `E`; the primitive state `prim` it was solved with, so that Ohm's
law holds between them to rounding and `prim2con(eos, m, prim, E, B)`
reproduces `cons★`'s `D`, `S_i` and `τ` to `stage.tol` of `τ + D` plus the
recovery's own tolerance; an
[`OhmStageResult`](@ref); and the number of recoveries made. The caller
writes `𝔈 = √γ E` into `U` and nothing else ("The implicit stage solve" in
`CODE.md`). It does not throw, and from finite input it returns a finite `E`.

- `σ = 0` at the predictor (`Vacuum`; `DensityProfile` at low density):
  `E = E★` exactly, and `prim` is `con2prim_hydro` at `E★` exactly.
- [`IdealConductor`](@ref): `prim` is [`con2prim_ideal`](@ref) and
  `E = −v × B`; `E★` is not read.
- Otherwise the fixed point or Newton on the fluid's share of the energy and
  momentum, seeded from the predictor or from the ideal recovery; if that
  does not converge in `stage.maxiter` recoveries, the bracketed path
  ([`bracketed_stage`](@ref)); and if that finds nothing either, the lagged
  velocity: `E` at the best iterate's velocity, and `prim = con2prim_hydro`
  at that `E`. See [`OhmStageOptions`](@ref).

Allocation-free and kernel-callable.
"""
@inline function solve_ohm_stage(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                 pol::EntropyEOS.PolicyOptions{T}, model::Conductivity,
                                 m::Metric3{T}, cons★::NTuple{NU,T}, E★::SVector{3,T},
                                 B::SVector{3,T}, γΔt::T, guesses::NTuple{3,T};
                                 stage::OhmStageOptions{T}=OhmStageOptions{T}()) where {T}
    isg = inv(m.sqrtγ)
    D = cons_D(cons★) * isg
    S = cons_S(cons★) * isg
    τ = cons_τ(cons★) * isg
    DYₑ = cons_DYₑ(cons★) * isg
    scale = τ + D                            # the total energy density 𝓔

    # The predictor: the fluid's share at `E★`, and its recovery — the answer
    # where there is no conductivity, the first iterate otherwise.
    S_f, τ_f = field_budget(m, S, τ, E★, B)
    prim, f, pt = fluid_recovery(eos, opts, pol, m, D, S_f, τ_f, DYₑ, guesses)
    σ = conductivity(model, prim)
    iszero(σ) && return E★, prim, OhmStageResult(OHM_SIGMA_ZERO, f), 1
    # A share the policy repaired is replaced by the repaired state's own, so
    # that every iterate is a fluid state and its recovery's derivative is
    # the recovery's.
    f == 0 || ((S_f, τ_f) = fluid_share(eos, m, prim))
    nrec = 1
    αΔt = γΔt * m.α

    # Which iteration: the fixed point where its contraction, estimated from
    # the Jacobian at the predictor, is small; Newton otherwise.
    c = αΔt * σ
    u = four_velocity(prim)
    E = ohm_E(m, c, u, E★, B)
    θ = mixed_norm(m, stage_jacobian(m, D, prim, pt, c, u, E, B))
    newton = !(θ <= stage.θ_newton)         # a NaN estimate takes Newton
    modifiers = UInt32(0)

    # The seed, where Newton is to run: the `σΔt → ∞` limit — the ideal
    # recovery, with the fluid's share it holds — where the conduction is
    # stiff (`λ = cW/(1 + cW) > λ_ideal`), the predictor otherwise; but an
    # admissible one of the two over one the policy had to repair.
    cW = c * lorentz(m, u)
    stiff = cW / (one(T) + cW) > stage.λ_ideal
    if newton && (stiff || f != 0)
        primI, fI, ptI = ideal_recovery(eos, opts, pol, m, cons★, B, guesses_of(prim))
        nrec += 1
        σI = conductivity(model, primI)
        if !iszero(σI) && (stiff ? (fI == 0 || f != 0) : fI == 0)
            S_f, τ_f = fluid_share(eos, m, primI)
            prim, f, pt, σ = primI, fI, ptI, σI
            modifiers |= OHM_IDEAL_SEED
        end
    end

    it = share_iteration(eos, opts, pol, model, m, D, S, τ, DYₑ, E★, B, αΔt, scale, S_f, τ_f,
                         prim, f, pt, σ, newton, stage.maxiter, stage)
    nrec += it.nrec
    modifiers |= it.modifiers
    if it.done
        branch = it.newton ? OHM_NEWTON : OHM_FIXED_POINT
        return it.E, it.prim, OhmStageResult(branch | modifiers, it.f), nrec
    end

    # Not converged: the bracketed search over the fluid's rapidity, which
    # does not depend on a seed ("The bracketed path" in `CODE.md`).
    br = bracketed_stage(eos, opts, pol, model, m, D, S, τ, DYₑ, E★, B, αΔt, scale, it.prim,
                         stage)
    nrec += br.nrec
    if br.done
        return br.E, br.prim, OhmStageResult(OHM_BRACKETED | modifiers | br.modifiers, br.f),
               nrec
    end

    # Neither: the lagged velocity — `E` at the best velocity the iteration
    # saw, and the fluid recovered from that `E`.
    E = ohm_E(m, αΔt * it.best_σ, it.best_u, E★, B)
    S_f, τ_f = field_budget(m, S, τ, E, B)
    prim, f, _ = fluid_recovery(eos, opts, pol, m, D, S_f, τ_f, DYₑ, guesses_of(it.prim))
    nrec += 1
    return E, prim, OhmStageResult(OHM_FALLBACK | modifiers | br.modifiers, f), nrec
end

"""
    share_iteration(eos, opts, pol, model, m, D, S, τ, DYₑ, E★, B, αΔt, scale,
                    S_f, τ_f, prim, f, pt, σ, newton, maxiter, stage) -> NamedTuple

The fast path of [`solve_ohm_stage`](@ref), and the polish of its bracketed
path: the fixed point or Newton on the fluid's share `X = (S_f, τ_f)` of the
undensitized totals `S`, `τ`, from the iterate `X` whose recovery is
`(prim, f, pt)` with the conductivity `σ`, for at most `maxiter` further
recoveries. Returns whether it converged (`done`), whether it ended on Newton,
the last field and recovery (`E`, `prim`, `f`), the recoveries it made, the
modifier bits, and the best iterate's velocity and conductivity for the
lagged-velocity fallback.
"""
function share_iteration(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                 pol::EntropyEOS.PolicyOptions{T}, model::Conductivity,
                                 m::Metric3{T}, D::T, S::SVector{3,T}, τ::T, DYₑ::T,
                                 E★::SVector{3,T}, B::SVector{3,T}, αΔt::T, scale::T,
                                 S_f::SVector{3,T}, τ_f::T, prim::NTuple{NPRIM,T}, f::UInt32,
                                 pt::EntropyEOS.EOSPoint{T}, σ::T, newton::Bool, maxiter::Int32,
                                 stage::OhmStageOptions{T}) where {T}
    modifiers = UInt32(0)
    nrec = 0
    # The best iterate, for the lagged-velocity fallback.
    best_err = T(Inf)
    best_u, best_σ = four_velocity(prim), σ
    # Newton's last accepted iterate and step, for backtracking.
    S_prev, τ_prev, ΔS, Δτ = S_f, τ_f, zero(SVector{3,T}), zero(T)
    err_prev = T(Inf)
    nback = Int32(0)
    E = E★
    done = false

    it = Int32(0)
    while true
        # The field at this iterate's velocity, and the fluid's share it
        # leaves: the residual is how far the iterate's share is from it.
        c = αΔt * σ
        u = four_velocity(prim)
        err, E, S_E, τ_E = stage_residual(m, c, prim, S_f, τ_f, S, τ, E★, B, scale)
        R_S = S_f - S_E
        R_τ = τ_f - τ_E
        isfinite(err) || break                 # non-finite input
        if err < best_err
            best_err, best_u, best_σ = err, u, σ
        end
        if err <= stage.tol
            done = true
            break
        end
        it < maxiter || break
        it += Int32(1)

        if newton && nback < stage.max_backtrack && isfinite(err_prev) && !(err < err_prev)
            # The step grew the residual: halve it.
            nback += Int32(1)
            modifiers |= OHM_BACKTRACKED
            ΔS, Δτ = ΔS / 2, Δτ / 2
            S_f, τ_f = S_prev + ΔS, τ_prev + Δτ
        elseif newton || (isfinite(err_prev) && !(err <= err_prev / 2))
            # Newton, or the fixed point handing over to it when it shrinks
            # the residual by less than half.
            newton || (modifiers |= OHM_SWITCHED)
            newton = true
            nback = Int32(0)
            ΔS, Δτ = newton_step(m, D, prim, pt, c, u, E, B,
                                 αΔt * conductivity_slope(model, prim), R_S, R_τ)
            # A singular Jacobian (a clamped recovery) takes the fixed-point
            # step instead.
            if !(all(isfinite, ΔS) && isfinite(Δτ))
                ΔS, Δτ = -R_S, -R_τ
            end
            S_prev, τ_prev, err_prev = S_f, τ_f, err
            S_f, τ_f = S_f + ΔS, τ_f + Δτ
        else
            # The fixed point: the share the field leaves.
            err_prev = err
            S_f, τ_f = S_E, τ_E
        end
        prim, f, pt = fluid_recovery(eos, opts, pol, m, D, S_f, τ_f, DYₑ, guesses_of(prim))
        nrec += 1
        if f != 0
            modifiers |= OHM_REPAIRED
            S_f, τ_f = fluid_share(eos, m, prim)
        end
        σ = conductivity(model, prim)
    end
    return (; done, newton, E, prim, f, nrec, modifiers, best_err, best_u, best_σ)
end

@inline function solve_ohm_stage(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                 pol::EntropyEOS.PolicyOptions{T}, ::IdealConductor,
                                 m::Metric3{T}, cons★::NTuple{NU,T}, E★::SVector{3,T},
                                 B::SVector{3,T}, γΔt::T, guesses::NTuple{3,T};
                                 stage::OhmStageOptions{T}=OhmStageOptions{T}()) where {T}
    prim, flags = con2prim_ideal(eos, opts, pol, m, cons★, B, guesses)
    return ideal_E(m, prim_v(prim), B), prim, OhmStageResult(OHM_IDEAL, flags), 1
end

# `guesses` the function, under a name the solve's argument does not shadow.
@inline guesses_of(prim::NTuple) = guesses(prim)
