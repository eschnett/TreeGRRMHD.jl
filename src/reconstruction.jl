# The reconstruction of the primitive variables to a face, and the face state
# the Riemann solver takes ("Reconstruction" in `CODE.md`).
#
# What is reconstructed (proposed in `CODE.md`, as built in step 5): the
# rest-mass density `ρ`, the entropy `s`, the electron fraction `Yₑ`, the
# spatial four-velocity `uⁱ = W vⁱ` — so that every reconstructed state is
# subluminal, whatever the limiter does — the electric field `Eⁱ` and the
# magnetic field `Bⁱ` from its cell average. The face-normal `Bᵈ` is then
# replaced by the face's own `𝔅ᵈ/√γ`, which is never reconstructed: it is
# single-valued at the face by construction. Twelve numbers a cell, read from
# `P`, all undensitized.
#
# **The methods** (step 5: PLM; step 7: PPM, WENO-Z, MP5). PLM with the
# limiters `:none`, `:minmod`, `:mc` reads one cell either side (`G ≥ 2`);
# `:ppm` (Colella & Woodward 1984), `:wenoz` (Borges et al. 2008) and `:mp5`
# (Suresh & Huynh 1997) read two (`G ≥ 3`). Each is written as an *offset*
# from the cell's own value to its face, `q_face = q₀ + δ(q₋₂ … q₊₂)`, with δ a
# function of the differences `q_k − q₀` only (amended in step 7). Two
# properties follow exactly, not to rounding, and the direction-independence
# claims of rung 2a rest on them:
#
#   * a stencil of equal values reconstructs to that value — the differences
#     are zero and so is δ — so a direction a solution does not vary in adds
#     nothing, and a run that is translation-invariant along `y` is the 1D
#     run bit for bit;
#   * δ is odd under `q → −q` and the lower face is the upper face of the
#     mirrored stencil, `δ₋(q₋₂ … q₊₂) = δ₊(q₊₂ … q₋₂)`, so a turned or
#     mirrored run reconstructs the turned or mirrored values.
#
# **The positivity fallback, per face and per variable group** (proposed in
# `CODE.md`, as built in step 5). The fluid group is `(ρ, s, Yₑ, uⁱ)`: a face
# state whose `ρ` is not positive and finite, whose `Yₑ` is outside the EOS's
# window, whose `s` is outside the EOS's physical window at that `ρ` and `Yₑ`,
# or whose `u` is not finite takes the cell's own values for the whole group —
# first order at that face, on that side. The field group `(Eⁱ, Bⁱ)` has no
# admissibility condition but finiteness. Under a TVD limiter (`:minmod`,
# `:mc`) the face values lie between the neighbouring cell values, so an
# admissible stencil cannot trigger the fluid fallback unless the window
# depends on `ρ` (a table); the plain centered slope (`:none`) and the
# higher-order methods can overshoot, and the fallback is what catches it.
#
# The method is a `Val`, as in TreeHydro: a kernel specializes on it and no
# branch survives into the inner loop. Integer literals only, so that a
# `Float32` run stays in `Float32`.

"""
    slope(::Val{:none},   a, b)
    slope(::Val{:minmod}, a, b)
    slope(::Val{:mc},     a, b)

The limited slope of one variable in one cell from its one-sided differences
`a = q_i − q_{i−1}` and `b = q_{i+1} − q_i`: the full rise across the cell,
so the face values are `q_i ± σ/2`. TreeHydro's three, unchanged:

- `:none`, the centered `(a + b)/2`: no limiter, a clean `O(h²)` truncation
  error, and no monotonicity — a convergence study's choice;
- `:minmod`: zero where `a` and `b` disagree in sign, the smaller otherwise;
- `:mc`, monotonized central, `minmod(2a, 2b, (a + b)/2)`: the centered slope
  wherever that is not steeper than twice either side.

Symmetric in `(a, b)` and odd under negation.
"""
@inline slope(::Val{:none}, a, b) = (a + b) / 2

@inline function slope(::Val{:minmod}, a, b)
    z = zero(a)
    # Two `ifelse`s rather than a branch: the same choice, NaN and signed
    # zero included, without a branch on the sign of a rounding-level
    # difference (TreeHydro, measured in its step 14).
    m = ifelse(abs(a) < abs(b), a, b)
    return ifelse(a * b ≤ z, z, m)
end

@inline slope(::Val{:mc}, a, b) =
    slope(Val(:minmod), slope(Val(:minmod), 2 * a, 2 * b), (a + b) / 2)

# --- the higher-order methods, as offsets from the cell value -------------------

# Each `face_offset(lim, q₋₂, q₋₁, q₀, q₊₁, q₊₂)` is `q_{i+1/2} − q₀`, the
# value at the upper face of the middle cell minus its own, from the five cell
# values around it; the lower face is the same function of the mirrored
# stencil (see the top of this file). PLM's offset is its slope's half and
# reads only the middle three.

@inline face_offset(lim::Val{L}, q₋₂, q₋₁, q₀, q₊₁, q₊₂) where {L} =
    slope(lim, q₀ - q₋₁, q₊₁ - q₀) / 2

# PPM (Colella & Woodward 1984, eqs. 1.6–1.8 and 1.10), without contact
# steepening or flattening (proposed in step 7). The interface value
# `a_{i+1/2} = a_i + (a_{i+1} − a_i)/2 − (δa_{i+1} − δa_i)/6` with the MC-limited
# slopes `δa` (their eq. 1.8 is exactly `slope(:mc, …)`), at both faces of the
# cell; then the monotonicity constraints: a cell that is a local extremum
# (`(a_R − a)(a − a_L) ≤ 0`) is flat, and a parabola that would overshoot on
# one side has that side's value moved to `3a − 2a_other`, which in offsets is
# `−2 δ_other`.
@inline function face_offset(::Val{:ppm}, q₋₂, q₋₁, q₀, q₊₁, q₊₂)
    mc = Val(:mc)
    δ₋₁ = slope(mc, q₋₁ - q₋₂, q₀ - q₋₁)
    δ₀ = slope(mc, q₀ - q₋₁, q₊₁ - q₀)
    δ₊₁ = slope(mc, q₊₁ - q₀, q₊₂ - q₊₁)
    dR = (q₊₁ - q₀) / 2 - (δ₊₁ - δ₀) / 6
    dL = (q₋₁ - q₀) / 2 - (δ₀ - δ₋₁) / 6
    z = zero(dR)
    dR * dL ≥ z && return z                     # an extremum, or flat on one side
    s = dR - dL
    # `(a_R − a_L)(a − (a_L + a_R)/2) < −(a_R − a_L)²/6`: the parabola
    # overshoots past the upper face, so `a_R = 3a − 2a_L`.
    return -(s * ((dL + dR) / 2)) < -(s * s) / 6 ? -2 * dL : dR
end

# WENO-Z (Borges, Carmona, Costa & Don 2008) of fifth order: the three
# third-order candidates at the upper face, their Jiang–Shu smoothness
# indicators `β_k`, and the weights `α_k = d_k (1 + (τ₅/(β_k + ε))²)` with
# `τ₅ = |β₀ − β₂|`, `d = (1, 6, 3)/10`, the power 2 of Castro et al. (2011).
# All in the differences `Δ_k = q_k − q₀`. `ε` is `floatmin(T)^(1/4)` (`1e−77`
# in `Float64`, `2e−10` in `Float32`): small enough to be below any `β` a
# resolved variation has, large enough that `(τ₅/ε)²` cannot overflow, which
# would turn a weight into `Inf/Inf` (proposed in step 7).
@inline function face_offset(::Val{:wenoz}, q₋₂, q₋₁, q₀, q₊₁, q₊₂)
    T = typeof(q₀)
    Δm2, Δm1, Δp1, Δp2 = q₋₂ - q₀, q₋₁ - q₀, q₊₁ - q₀, q₊₂ - q₀
    c₀ = (2 * Δm2 - 7 * Δm1) / 6
    c₁ = (2 * Δp1 - Δm1) / 6
    c₂ = (5 * Δp1 - Δp2) / 6
    sq(x) = x * x
    β₀ = 13 * sq(Δm2 - 2 * Δm1) / 12 + sq(Δm2 - 4 * Δm1) / 4
    β₁ = 13 * sq(Δm1 + Δp1) / 12 + sq(Δm1 - Δp1) / 4
    β₂ = 13 * sq(Δp2 - 2 * Δp1) / 12 + sq(Δp2 - 4 * Δp1) / 4
    τ = abs(β₀ - β₂)
    ε = sqrt(sqrt(floatmin(T)))
    α₀ = (1 + sq(τ / (β₀ + ε))) / 10
    α₁ = 6 * (1 + sq(τ / (β₁ + ε))) / 10
    α₂ = 3 * (1 + sq(τ / (β₂ + ε))) / 10
    return (α₀ * c₀ + α₁ * c₁ + α₂ * c₂) / (α₀ + α₁ + α₂)
end

# MP5 (Suresh & Huynh 1997, eqs. 2.1, 2.12, 2.19, 2.20 and 2.24–2.27 with
# `α = 4`): the fifth-order interface value, kept where it lies between the
# cell value and the monotonicity-preserving bound `v_MP`, and otherwise
# clipped into the interval `[v_min, v_max]` built from the curvature-aware
# bounds `v_MD`, `v_LC` and the upper limit `v_UL`. In offsets from `q₀`, with
# the test `(v − q₀)(v − v_MP) ≤ 0` taken without their `ε` (proposed in step
# 7): on a stencil of equal values both factors are exactly zero.
@inline function face_offset(::Val{:mp5}, q₋₂, q₋₁, q₀, q₊₁, q₊₂)
    mm = Val(:minmod)
    Δm2, Δm1, Δp1, Δp2 = q₋₂ - q₀, q₋₁ - q₀, q₊₁ - q₀, q₊₂ - q₀
    v = (2 * Δm2 - 13 * Δm1 + 27 * Δp1 - 3 * Δp2) / 60
    vMP = slope(mm, Δp1, -4 * Δm1)
    z = zero(v)
    v * (v - vMP) ≤ z && return v
    d₋₁ = Δm2 - 2 * Δm1                         # curvatures at i−1, i, i+1
    d₀ = Δm1 + Δp1
    d₊₁ = Δp2 - 2 * Δp1
    minmod4(a, b, c, d) = slope(mm, slope(mm, a, b), slope(mm, c, d))
    dM₊ = minmod4(4 * d₀ - d₊₁, 4 * d₊₁ - d₀, d₀, d₊₁)
    dM₋ = minmod4(4 * d₀ - d₋₁, 4 * d₋₁ - d₀, d₀, d₋₁)
    vUL = -4 * Δm1
    vMD = Δp1 / 2 - dM₊ / 2
    vLC = -Δm1 / 2 + 4 * dM₋ / 3
    vmin = max(min(z, Δp1, vMD), min(z, vUL, vLC))
    vmax = min(max(z, Δp1, vMD), max(z, vUL, vLC))
    return v + slope(mm, vmin - v, vmax - v)
end

"""
    face_value(lim, q₋₂, q₋₁, q₀, q₊₁, q₊₂, σ)

The value of the middle cell's variable at its upper face (`σ = +1`) or its
lower face (`σ = −1`), `q₀ + δ`, with `δ` the method's offset of the stencil
read towards that face (see the top of `src/reconstruction.jl`).
"""
@inline face_value(lim::Val, q₋₂, q₋₁, q₀, q₊₁, q₊₂, σ::Int) =
    q₀ + (σ > 0 ? face_offset(lim, q₋₂, q₋₁, q₀, q₊₁, q₊₂) :
          face_offset(lim, q₊₂, q₊₁, q₀, q₋₁, q₋₂))

"""The admissible reconstructions, in the order `CODE.md` introduces them:
PLM with three limiters (step 5), then PPM, WENO-Z and MP5 (step 7)."""
const MHD_LIMITERS = (:none, :minmod, :mc, :ppm, :wenoz, :mp5)

"""
    stencil_radius(lim) -> 1 or 2

How many cells either side of a cell its reconstruction reads: one for PLM,
two for PPM, WENO-Z and MP5. A face reads one more on each side, so the
ghosted sets need `G ≥ stencil_radius + 1`.
"""
@inline stencil_radius(::Union{Val{:none},Val{:minmod},Val{:mc}}) = 1
@inline stencil_radius(::Val) = 2

# The number of reconstructed variables of a cell: ρ, s, Yₑ, u¹…u³, E¹…E³,
# B¹…B³.
const NREC = 12
const R_RHO, R_S, R_YE, R_U, R_E, R_B = 1, 2, 3, 4, 7, 10

"""
    recon_vars(P) -> NTuple{12}

The twelve reconstructed variables of a cell's `P` entry:
`(ρ, s, Yₑ, u¹, u², u³, E¹, E², E³, B¹, B², B³)`, with `uⁱ = W vⁱ`.
"""
@inline function recon_vars(P::NTuple{NP,T}) where {T}
    u = prim_W(P) * prim_v(P)
    E = aux_E(P)
    B = aux_B(P)
    return (prim_ρ(P), prim_s(P), prim_Yₑ(P), u[1], u[2], u[3], E[1], E[2], E[3], B[1], B[2],
            B[3])
end

# The values extrapolated from cell `c` to its face on side `σ` (`+1` its
# upper face, `−1` its lower one), from its neighbours `m` and `p` along that
# direction: `q_c ± slope/2`, per variable. PLM only.
@inline function extrapolate(lim::Val, qm::NTuple{M,T}, qc::NTuple{M,T}, qp::NTuple{M,T},
                             σ::Int) where {M,T}
    return ntuple(v -> @inbounds(qc[v] + σ * slope(lim, qc[v] - qm[v], qp[v] - qc[v]) / 2),
                  Val(M))
end

# The same from the five cells `q₋₂ … q₊₂` around `c = q₀`, for any method:
# `face_value` per variable. Under PLM it is the three-cell form bit for bit.
@inline function extrapolate(lim::Val, q₋₂::NTuple{M,T}, q₋₁::NTuple{M,T}, q₀::NTuple{M,T},
                             q₊₁::NTuple{M,T}, q₊₂::NTuple{M,T}, σ::Int) where {M,T}
    return ntuple(v -> @inbounds(face_value(lim, q₋₂[v], q₋₁[v], q₀[v], q₊₁[v], q₊₂[v], σ)),
                  Val(M))
end

# Whether a reconstructed fluid group is a state the EOS can evaluate.
@inline function admissible_fluid(eos::EntropyEOS.AbstractEOS, r::NTuple{NREC,T}) where {T}
    ρ, s, Yₑ = r[R_RHO], r[R_S], r[R_YE]
    (isfinite(ρ) && ρ > zero(T)) || return false
    (isfinite(r[R_U]) && isfinite(r[R_U + 1]) && isfinite(r[R_U + 2])) || return false
    y_lo, y_hi = EntropyEOS.yₑ_bounds(eos)
    (convert(T, y_lo) ≤ Yₑ ≤ convert(T, y_hi)) || return false
    sr = EntropyEOS.srange(eos, ρ, Yₑ)
    return sr.s_min ≤ s ≤ sr.s_max
end

@inline function admissible_fields(r::NTuple{NREC,T}) where {T}
    return all(v -> isfinite(r[v]), R_E:(R_B + 2))
end

"""
    limit_face(eos, r_face, r_cell) -> NTuple{12}

The positivity fallback: `r_face` with its fluid group replaced by the cell's
where the reconstructed fluid group is inadmissible, and likewise its field
group where it is not finite (see the top of `src/reconstruction.jl`).
"""
@inline function limit_face(eos::EntropyEOS.AbstractEOS, r::NTuple{NREC,T},
                            c::NTuple{NREC,T}) where {T}
    fluid_ok = admissible_fluid(eos, r)
    fields_ok = admissible_fields(r)
    return ntuple(Val(NREC)) do v
        @inbounds (v < R_E ? fluid_ok : fields_ok) ? r[v] : c[v]
    end
end

"""
    face_state(eos, m::Metric3, r, 𝔅n, u_guess, ::Val{d}) -> (cons, prim, E, B)

The state on one side of a face normal to `d`, in the face metric `m`, from
the reconstructed variables `r` (see [`recon_vars`](@ref)), the face's own
densitized normal field `𝔅n = 𝔅ᵈ`, and the EOS warm start of the cell it was
reconstructed from: `vⁱ = uⁱ/W` with `W = √(1 + γ_ij uⁱuʲ)`, the primitive
state from one EOS evaluation ([`primitive_state`](@ref)), `Bᵈ = 𝔅n/√γ`, and
the densitized conserved state from [`prim2con`](@ref). The tuple
[`riemann_flux`](@ref) takes.
"""
@inline function face_state(eos::EntropyEOS.AbstractEOS, m::Metric3{T}, r::NTuple{NREC,T},
                            𝔅n::T, u_guess::T, ::Val{d}) where {T,d}
    u = SVector(r[R_U], r[R_U + 1], r[R_U + 2])
    W = sqrt(one(T) + dot_up(m, u, u))
    prim = primitive_state(eos, m, r[R_RHO], u / W, r[R_S], r[R_YE], u_guess)
    E = SVector(r[R_E], r[R_E + 1], r[R_E + 2])
    Bn = 𝔅n / m.sqrtγ
    B = SVector(ntuple(k -> k == d ? Bn : r[R_B + k - 1], Val(3)))
    return prim2con(eos, m, prim, E, B), prim, E, B
end
