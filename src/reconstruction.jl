# Piecewise-linear reconstruction of the primitive variables to a face, and
# the face state the Riemann solver takes ("Reconstruction" in `CODE.md`).
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
# **The positivity fallback, per face and per variable group** (proposed in
# `CODE.md`, as built in step 5). The fluid group is `(ρ, s, Yₑ, uⁱ)`: a face
# state whose `ρ` is not positive and finite, whose `Yₑ` is outside the EOS's
# window, whose `s` is outside the EOS's physical window at that `ρ` and `Yₑ`,
# or whose `u` is not finite takes the cell's own values for the whole group —
# first order at that face, on that side. The field group `(Eⁱ, Bⁱ)` has no
# admissibility condition but finiteness. Under a TVD limiter (`:minmod`,
# `:mc`) the face values lie between the neighbouring cell values, so an
# admissible stencil cannot trigger the fluid fallback unless the window
# depends on `ρ` (a table); the plain centered slope (`:none`) can overshoot,
# and the fallback is what keeps it usable at all.
#
# The limiter is a `Val`, as in TreeHydro: a kernel specializes on it and no
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

"""The admissible limiters, in the order `CODE.md` introduces them."""
const MHD_LIMITERS = (:none, :minmod, :mc)

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
# direction: `q_c ± slope/2`, per variable.
@inline function extrapolate(lim::Val, qm::NTuple{M,T}, qc::NTuple{M,T}, qp::NTuple{M,T},
                             σ::Int) where {M,T}
    return ntuple(v -> @inbounds(qc[v] + σ * slope(lim, qc[v] - qm[v], qp[v] - qc[v]) / 2),
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
