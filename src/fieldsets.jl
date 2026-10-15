# The field sets of the scheme: where every quantity lives, in any dimension.
#
# A block's cells sit at half-integer positions between integer vertices,
# and TreeAMR's centerings put each quantity where its equation puts it
# ("Where the fields live" and "Field sets" in `CODE.md`):
#
#   * the ten conserved matter variables, with `𝔈`, at cells;
#   * `A_j` on the edges along `j` — cell-like along `j`, vertex-like across;
#   * `𝔅ᵈ = [dij] ∂ᵢ A_j` on the faces normal to `d` — vertex-like along `d`,
#     cell-like across;
#   * the gauge scalar `Φ̃ = √γ Φ` at vertices.
#
# The matter code is generic in `D ∈ {1, 2, 3}` with three-component fields,
# so a quantity that belongs to a direction the mesh does not have still
# exists: it takes the centering that remains once the missing directions are
# dropped. An edge along a missing direction is vertex-like in every present
# one (in 1D, `A_y` and `A_z` sit at the vertices), and a face normal to a
# missing direction is cell-like in every present one (in 1D, `𝔅ʸ` and `𝔅ᶻ`
# are cell-centered). That is exactly the restriction of the 3D centering to
# the present dimensions, which is what `edge_centering` and
# `face_centering` compute — TreeAMR's `edgecentered(D, d)` and
# `facecentered(D, d)` for `d ≤ D`, and the remainder for `d > D`.
#
# Ghost widths: every evolved set and every set a reconstruction reads has
# `G = 3`, the width the table in `CODE.md` reserves for the higher-order
# reconstructions of step 7 (PLM reads two). The fluxes and the EMFs are
# computed over the closed ranges and never exchanged, `G = 0`. One width for
# all the ghosted sets is what lets one stored index serve a cell of `U`, of
# `P` and the cell-like faces of `B` in 1D; the kernels still take one
# `Val(G)` per set they read ("Sharp edges" in `PLAN.md`).

"""
    edge_centering(::Val{D}, j) -> NTuple{D,Symbol}

The centering of an edge along direction `j ∈ 1:3` on a `D`-dimensional
mesh: `:cell` along `j`, `:vertex` along every other present direction.
For `j ≤ D` it is TreeAMR's `edgecentered(D, j)`; for a direction the mesh
does not have, the edge is vertex-like in every present one ("Where the
fields live" in `CODE.md`: in 1D, `A_y` and `A_z` live at the vertices).
"""
edge_centering(::Val{D}, j::Integer) where {D} = ntuple(e -> e == j ? :cell : :vertex, D)

"""
    face_centering(::Val{D}, d) -> NTuple{D,Symbol}

The centering of a face normal to direction `d ∈ 1:3` on a `D`-dimensional
mesh: `:vertex` along `d`, `:cell` along every other present direction.
For `d ≤ D` it is TreeAMR's `facecentered(D, d)`; normal to a direction
the mesh does not have, it is cell-like everywhere (in 1D, `𝔅ʸ` and `𝔅ᶻ`
are cell-centered).
"""
face_centering(::Val{D}, d::Integer) where {D} = ntuple(e -> e == d ? :vertex : :cell, D)

"""Number of slots of the diagnostics set `X`: see [`MHDFields`](@ref)."""
const NX = 19
# The diagnostics set's slots. 1–9 count the implicit stages that set each of
# the nine `OhmStageResult` flag bits, in bit order; then the stages whose
# returned state the policy had repaired, the `|` of their policy flags (a
# number below 2¹⁶, exact in any float), the recoveries the stages made and
# the most any one stage made, the cells the reset wrote back as the stage
# limiter, and the cell's light speed, for the time step. *(Added in step 6:)*
# the cells the reset wrote back as the step limiter, and what those writes
# changed in `√γ D` and `√γ τ` — the rest mass and the energy the reset
# injects into the state, since only the step limiter's writes reach it
# ("Floors, the atmosphere and the magnetosphere" in `CODE.md`). Counts and
# sums accumulate per cell over a chunk and are reduced once at its end ("The
# implicit stage solve" in `CODE.md`: "reduced per chunk").
const X_OHM = 1
const X_POLICY = 10
const X_POLICY_OR = 11
const X_RECOVERIES = 12
const X_MAX_RECOVERIES = 13
const X_RESETS = 14
const X_LIGHT = 15
const X_STEP_RESETS = 16
const X_INJECTED_D = 17
const X_INJECTED_TAU = 18
# *(Added in step 7:)* the cell's `Σ_d (𝔅ᵈ[n + e_d] − 𝔅ᵈ[n])`, `h ∇·𝔅`, written by
# `max_div_B` when it is asked.
const X_DIVB = 19
const N_OHM_BITS = 9

"""
    MHDFields

Every field set the scheme uses on one forest ("Field sets" in `CODE.md`):

| field | centering | nvars | `G` | role |
|---|---|---|---|---|
| `U` | cell | 10 | 3 | evolved: `√γ D, √γ S_i, √γ τ, √γ D Yₑ, √γ D s, 𝔈ⁱ` |
| `A[j]`, `j = 1:3` | edge along `j` | 1 | 3 | evolved: the covariant `A_j` |
| `Φ` | vertex | 1 | 3 | evolved: the gauge scalar `Φ̃ = √γ Φ` |
| `P` | cell | 30 | 3 | scratch: the primitives, `E`, `B`, `q`, the cell metric |
| `B[d]`, `d = 1:3` | face normal to `d` | 1 | 3 | scratch: `𝔅ᵈ = curl A + 𝔅̄`, every RHS |
| `F[d]`, `d = 1:D` | face normal to `d` | 10 | 0 | scratch: the matter fluxes |
| `EMF[j]`, `j = 1:3` | edge along `j` | 1 | 0 | scratch: the electromotive force |
| `X` | cell | `NX` | 0 | scratch: per-cell tallies, the reset's injection, the light speed, `h∇·𝔅` |

The state vector is over [`evolved_sets`](@ref)`(f) = (U, A[1], A[2], A[3],
Φ)`, in that order (AMR-1, TreeAMR 0.2.1). Build it with
[`mhd_fields`](@ref).
"""
struct MHDFields{D,FU,FA,FΦ,FP,FB,FF,FE,FX}
    U::FU
    A::FA          # NTuple{3,FieldSet}
    Φ::FΦ
    P::FP
    B::FB          # NTuple{3,FieldSet}
    F::FF          # NTuple{D,FieldSet}
    EMF::FE        # NTuple{3,FieldSet}
    X::FX
    valD::Val{D}
end

"""
    mhd_fields(T, forest; G=3, backend=CPU()) -> MHDFields

The field sets of the scheme over `forest`, in the element type `T` and on
`backend`, with ghost width `G` for every ghosted set (see
[`MHDFields`](@ref)). Requires `N ≥ 2G + 2` (TreeAMR's invariant for a
vertex-like dimension), which a staggered set imposes in every dimension.
"""
function mhd_fields(::Type{T}, forest::Forest{D}; G::Integer=3,
                    backend=CPU()) where {T,D}
    G ≥ 2 || throw(ArgumentError(
        "the ghosted sets need G ≥ 2: the piecewise-linear reconstruction reads " *
        "two cells beyond a face, and the curl reads one edge beyond a stored face; " *
        "got G = $G. CODE.md's table reserves G = 3 for PPM, WENO-Z and MP5 (step 7)."))
    fs(nvars, g, c) = FieldSet{T}(forest, nvars; G=g, centering=c, backend=backend)
    vD = Val(D)
    U = fs(NU, G, cellcentered(D))
    A = ntuple(j -> fs(1, G, edge_centering(vD, j)), 3)
    Φ = fs(1, G, vertexcentered(D))
    P = fs(NP, G, cellcentered(D))
    B = ntuple(d -> fs(1, G, face_centering(vD, d)), 3)
    F = ntuple(d -> fs(NU, 0, face_centering(vD, d)), D)
    EMF = ntuple(j -> fs(1, 0, edge_centering(vD, j)), 3)
    X = fs(NX, 0, cellcentered(D))
    return MHDFields{D,typeof(U),typeof(A),typeof(Φ),typeof(P),typeof(B),typeof(F),
                     typeof(EMF),typeof(X)}(U, A, Φ, P, B, F, EMF, X, vD)
end

"""
    evolved_sets(f::MHDFields) -> (U, A₁, A₂, A₃, Φ̃)

The evolved field sets, in the order of the state vector's segments: the
tuple TreeAMR's multi-set `statevector`, `scatter!`, `gather!`,
`stateview` and `statearray` take (AMR-1). `B` is not among them: it is
recomputed from `A` in every right-hand side and is never state ("Sharp
edges" in `PLAN.md`).
"""
evolved_sets(f::MHDFields) = (f.U, f.A[1], f.A[2], f.A[3], f.Φ)

# The segment of each evolved set in `evolved_sets`.
const SEG_U = 1
const SEG_A = 2            # A_j is segment SEG_A + j - 1
const SEG_Φ = 5
