# The variable layouts: the ten evolved cell variables, and the primitive set
# `P` the right-hand side recovers into and reads from.
#
# A state at one point is an `NTuple`, as in TreeHydro: a kernel reads the
# `nvars` slots of a cell into a tuple, works on the tuple, and writes a
# tuple back. The constants below are the only place a slot number is
# written, and the accessors the only place a slot is read by number, so
# that nothing downstream indexes a state by a literal — a `u[5]` that is `τ`
# in one file and `D Yₑ` in another is the kind of mistake that produces
# plots which look almost right. See "Where the fields live" and "Field sets"
# in `CODE.md`.
#
# Vector slots are three consecutive entries starting at the named constant,
# and are read and written as `SVector{3}`: the matter code is generic in `D`
# with three-component fields everywhere, so a vector never has fewer than
# three components.

# --- The evolved cell variables ---------------------------------------------
#
# All densitized: `√γ D`, `√γ S_i`, `√γ τ`, `√γ D Yₑ`, `√γ D s` and
# `𝔈ⁱ = √γ Eⁱ` ("Matter: total energy and momentum" in `CODE.md`). `S_i` and
# `τ` are the *totals*, fluid plus electromagnetic; `D = ρ★ W` is in the
# EOS's κ-rescaled density, so that a checkpoint is tied to the EOS's `κ`.

"""Number of evolved cell variables."""
const NU = 10
"""Slot of `√γ D`."""
const U_D = 1
"""First of the three slots of `√γ S_i` (covariant, total)."""
const U_S = 2
"""Slot of `√γ τ` (total)."""
const U_TAU = 5
"""Slot of `√γ D Yₑ`."""
const U_DYE = 6
"""Slot of `√γ D s`, the advected entropy."""
const U_DS = 7
"""First of the three slots of `𝔈ⁱ = √γ Eⁱ`."""
const U_E = 8

# --- The primitive set `P` ----------------------------------------------------
#
# The first `NPRIM` slots are the fluid primitives and the EOS quantities
# the recovery produces at the same time, in exactly the order
# `con2prim_hydro` returns them, so that its result is the leading slice of a
# `P` tuple; the rest are what the right-hand side reads beside them. All
# undensitized, all in the normal observer's frame, all in geometric units
# with `ρ = ρ★` (amended in step 1: "Field sets" in `CODE.md` said "~25").
#
#    1  ρ        rest-mass density ρ★ (the EOS's κ-rescaled density)
#  2–4  vⁱ       fluid velocity, contravariant; |v| = tanh w exactly
#    5  s        entropy per baryon
#    6  Yₑ       electron fraction
#    7  w        rapidity, W = cosh w; the recovery's warm start
#    8  p        pressure
#    9  h        specific enthalpy
#   10  cs²      sound speed squared
#   11  u_eos    the EOS's own warm start (log₁₀ T; NaN for analytic EOSes)
#   12  flags    the recovery's policy flags, as a number
# 13–15 Eⁱ       electric field
# 16–18 Bⁱ       magnetic field, the cell average of curl A plus 𝔅̄
#   19  q        charge density, ∇·E
#   20  α        lapse
# 21–23 βⁱ       shift
# 24–29 γ_ij     xx, xy, xz, yy, yz, zz
#   30  √γ

"""Number of slots of the recovered primitive state, the leading slice of `P`."""
const NPRIM = 11
"""Slot of the rest-mass density `ρ★`."""
const P_RHO = 1
"""First of the three slots of the contravariant velocity `vⁱ`."""
const P_V = 2
"""Slot of the entropy per baryon `s`."""
const P_S = 5
"""Slot of the electron fraction `Yₑ`."""
const P_YE = 6
"""Slot of the rapidity `w`, `W = cosh w`."""
const P_W = 7
"""Slot of the pressure `p`."""
const P_PRESS = 8
"""Slot of the specific enthalpy `h`."""
const P_H = 9
"""Slot of the sound speed squared `cs²`."""
const P_CS2 = 10
"""Slot of the EOS's warm start `u_solved`."""
const P_UEOS = 11
"""Slot of the recovery's policy flags."""
const P_FLAGS = 12
"""First of the three slots of `Eⁱ`."""
const P_E = 13
"""First of the three slots of `Bⁱ`."""
const P_B = 16
"""Slot of the charge density `q`."""
const P_Q = 19
"""Slot of the lapse `α`."""
const P_ALPHA = 20
"""First of the three slots of the shift `βⁱ`."""
const P_BETA = 21
"""First of the six slots of `γ_ij`, packed `xx, xy, xz, yy, yz, zz`."""
const P_GAMMA = 24
"""Slot of `√γ`."""
const P_SQRTG = 30
"""Number of slots of `P`."""
const NP = 30

@inline vec3(t::NTuple, i) = @inbounds SVector(t[i], t[i + 1], t[i + 2])

# --- Accessors: the evolved variables -----------------------------------------

"""`√γ D` of an evolved state."""
@inline cons_D(u::NTuple{NU}) = u[U_D]
"""`√γ S_i` of an evolved state, as an `SVector{3}`."""
@inline cons_S(u::NTuple{NU}) = vec3(u, U_S)
"""`√γ τ` of an evolved state."""
@inline cons_τ(u::NTuple{NU}) = u[U_TAU]
"""`√γ D Yₑ` of an evolved state."""
@inline cons_DYₑ(u::NTuple{NU}) = u[U_DYE]
"""`√γ D s` of an evolved state."""
@inline cons_Ds(u::NTuple{NU}) = u[U_DS]
"""`𝔈ⁱ = √γ Eⁱ` of an evolved state, as an `SVector{3}`."""
@inline cons_𝔈(u::NTuple{NU}) = vec3(u, U_E)

"""
    pack_cons(D, S, τ, DYₑ, Ds, 𝔈) -> NTuple{10}

An evolved state from its parts, each already densitized, in the slot order
of `U_D … U_E`.
"""
@inline pack_cons(D::T, S::SVector{3,T}, τ::T, DYₑ::T, Ds::T, 𝔈::SVector{3,T}) where {T} =
    (D, S[1], S[2], S[3], τ, DYₑ, Ds, 𝔈[1], 𝔈[2], 𝔈[3])

# --- Accessors: the primitives ---------------------------------------------------
#
# Defined on any tuple at least `NPRIM` long, so that the same accessor reads
# the recovery's result and a full `P`.

"""The rest-mass density `ρ★` of a primitive state."""
@inline prim_ρ(p::NTuple) = p[P_RHO]
"""The contravariant velocity `vⁱ` of a primitive state, as an `SVector{3}`."""
@inline prim_v(p::NTuple) = vec3(p, P_V)
"""The entropy per baryon `s` of a primitive state."""
@inline prim_s(p::NTuple) = p[P_S]
"""The electron fraction `Yₑ` of a primitive state."""
@inline prim_Yₑ(p::NTuple) = p[P_YE]
"""The rapidity `w` of a primitive state."""
@inline prim_w(p::NTuple) = p[P_W]
"""The Lorentz factor `W = cosh w` of a primitive state."""
@inline prim_W(p::NTuple) = cosh(p[P_W])
"""The pressure `p` of a primitive state."""
@inline prim_p(p::NTuple) = p[P_PRESS]
"""The specific enthalpy `h` of a primitive state."""
@inline prim_h(p::NTuple) = p[P_H]
"""The sound speed squared `cs²` of a primitive state."""
@inline prim_cs²(p::NTuple) = p[P_CS2]
"""The EOS's warm start `u_solved` of a primitive state."""
@inline prim_ueos(p::NTuple) = p[P_UEOS]

"""
    pack_prim(ρ, v, s, Yₑ, w, p, h, cs², u_eos) -> NTuple{NPRIM}

A primitive state from its parts, in the slot order of `P_RHO … P_UEOS`.
"""
@inline pack_prim(ρ::T, v::SVector{3,T}, s::T, Yₑ::T, w::T, p::T, h::T, cs²::T,
                  u_eos::T) where {T} =
    (ρ, v[1], v[2], v[3], s, Yₑ, w, p, h, cs², u_eos)

"""
    guesses(P) -> (s, w, u_eos)

The warm starts a primitive state holds for the next recovery, in the order
`con2prim_hydro` takes them ("Thermodynamics" in `CODE.md`).
"""
@inline guesses(p::NTuple) = (p[P_S], p[P_W], p[P_UEOS])

"""
    cold_guesses(T) -> (NaN, NaN, NaN)

No warm start: the recovery seeds itself.
"""
@inline cold_guesses(::Type{T}) where {T} = (T(NaN), T(NaN), T(NaN))

# --- Accessors: the rest of `P` -------------------------------------------------

"""
The recovery's policy flags held in `P`, as the `UInt32` EntropyEOS returns.
They are stored as a number of the working type, which is exact: every
policy bit is below `2^16`.
"""
@inline aux_flags(P::NTuple{NP}) = unsafe_trunc(UInt32, P[P_FLAGS])
"""The electric field `Eⁱ` held in `P`."""
@inline aux_E(P::NTuple{NP}) = vec3(P, P_E)
"""The cell-averaged magnetic field `Bⁱ` held in `P`."""
@inline aux_B(P::NTuple{NP}) = vec3(P, P_B)
"""The charge density `q` held in `P`."""
@inline aux_q(P::NTuple{NP}) = P[P_Q]

"""
    aux_metric(P) -> Metric3

The cell metric held in `P`, with `γ^ij` recomputed from `γ_ij` and the
stored `√γ` kept.
"""
@inline function aux_metric(P::NTuple{NP,T}) where {T}
    g = Metric3(P[P_ALPHA], vec3(P, P_BETA),
                SMatrix{3,3,T,9}(P[P_GAMMA], P[P_GAMMA + 1], P[P_GAMMA + 2],
                                 P[P_GAMMA + 1], P[P_GAMMA + 3], P[P_GAMMA + 4],
                                 P[P_GAMMA + 2], P[P_GAMMA + 4], P[P_GAMMA + 5]))
    return Metric3{T}(g.α, g.β, g.γ, g.γu, P[P_SQRTG])
end

"""
    pack_P(prim, flags, E, B, q, m::Metric3) -> NTuple{NP}

A full primitive-set entry: the recovered state `prim` (its first `NPRIM`
slots are used), the policy flags, the fields, the charge density and the
metric.
"""
@inline function pack_P(prim::NTuple{N,T}, flags::UInt32, E::SVector{3,T}, B::SVector{3,T},
                        q::T, m::Metric3{T}) where {N,T}
    γ = m.γ
    return (ntuple(i -> prim[i], Val(NPRIM))..., T(flags), E[1], E[2], E[3], B[1], B[2], B[3],
            q, m.α, m.β[1], m.β[2], m.β[3], γ[1, 1], γ[1, 2], γ[1, 3], γ[2, 2], γ[2, 3],
            γ[3, 3], m.sqrtγ)
end
