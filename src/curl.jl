# The magnetic field: the discrete curl of the edge-centered vector potential
# on the faces, plus the uniform background, and its average to the cells.
#
#     𝔅ᵈ = [dij] ∂ᵢ A_j + 𝔅̄ᵈ
#
# ("Maxwell" and "Where the fields live" in `CODE.md`). With `(d, a, b)`
# cyclic, `𝔅ᵈ = ∂_a A_b − ∂_b A_a`. The face normal to `d` is cell-like along
# `a`, and `A_b` (an edge along `b`) is vertex-like along `a`, so `∂_a A_b` at
# the face is the difference of `A_b` at the two vertices that bound the face's
# cell along `a`; along `d` both are vertex-like and along `b` both cell-like,
# so there the face and the edge share an index. Likewise `∂_b A_a`. A
# derivative along a direction the mesh does not have is zero. Summed over a
# cell's faces, every edge value enters twice with opposite signs, so
# `∂_d 𝔅ᵈ = 0` per cell to rounding whatever `A` holds: `B` is structural,
# never evolved, prolongated or stored as state (`CLAUDE.md`).
#
# **The background is densitized and constant** (proposed in step 5). `CODE.md`
# wrote `𝔅̄ⁱ = √γ B̄ⁱ` with a constant `B̄ⁱ`; but then `∂ᵢ𝔅̄ⁱ = B̄ⁱ ∂ᵢ√γ`, which
# is not zero in a curved metric, and the background would carry a divergence
# the curl cannot cancel. A constant `𝔅̄ⁱ` has none on any metric, and in flat
# space it is the constant `B̄ⁱ` the plan meant: the field a 1D shock tube
# holds along its axis, or the guide field of the Alfvén wave.
#
# Index arithmetic. A kernel launched with `stored = true` over a face set
# gets the face's stored index `s`; its local index is `n = s − G_B`, owned
# points `1 … N`, and `A_j`'s stored index for the same local point is
# `n + G_A`. A cell `n` lies between vertices `n` and `n + 1`. With
# `G_A = G_B` (which `mhd_fields` builds and `MHDProblem` checks), every
# stored face — ghosts and the shared plane included — has the two edges it
# reads in storage, so the curl covers the whole stored extent and the cell
# average below covers every stored cell.

# The stored index of the point `n` (local) in a set of ghost width `G`.
@inline stored(n::NTuple{D,Int}, G::NTuple{D,Int}) where {D} = ntuple(e -> n[e] + G[e], Val(D))
# The local index `n` shifted by `k` along `e`.
@inline shift(n::NTuple{D,Int}, e::Int, k::Int) where {D} =
    ntuple(f -> f == e ? n[f] + k : n[f], Val(D))

# `∂_i A_j` at the local face point `n`, the difference along `i` between the
# vertices `n + 1` and `n`: zero along a missing direction.
@inline function edge_derivative(Aj, n::NTuple{D,Int}, b::Int, ::Val{i}, ::Val{GA},
                                 h) where {D,i,GA}
    i > D && return zero(h)
    @inbounds hi = Aj[stored(shift(n, i, 1), GA)..., 1, b]
    @inbounds lo = Aj[stored(n, GA)..., 1, b]
    return (hi - lo) / h
end

@kernel function curl_kernel!(Bd, @Const(Aa), @Const(Ab), @Const(spacings), 𝔅̄d, ::Val{D},
                              ::Val{d}, ::Val{GB}, ::Val{GAa}, ::Val{GAb}) where {D,d,GB,GAa,GAb}
    I = @index(Global, NTuple)                       # stored: `stored = true`
    b = I[D + 1]
    s = ntuple(e -> I[e], Val(D))
    n = ntuple(e -> s[e] - GB[e], Val(D))
    h = @inbounds spacings[b]
    a_ = mod1(d + 1, 3)
    b_ = mod1(d + 2, 3)
    # 𝔅ᵈ = ∂_a A_b − ∂_b A_a + 𝔅̄ᵈ.
    @inbounds Bd[s..., 1, b] = (edge_derivative(Ab, n, b, Val(a_), Val(GAb), h) -
                                edge_derivative(Aa, n, b, Val(b_), Val(GAa), h)) + 𝔅̄d
end

"""
    curl_A!(f::MHDFields, 𝔅̄, spacings)

`𝔅ᵈ = [dij] ∂ᵢ A_j + 𝔅̄ᵈ` on every **stored** face of `f.B[d]`, `d = 1:3`,
from the ghost-filled `f.A` (the shared planes and the ghosts included), with
the constant densitized background `𝔅̄::SVector{3}` (see the top of
`src/curl.jl`). `spacings` is the per-block spacing on the sets' backend.
"""
function curl_A!(f::MHDFields{D}, 𝔅̄::SVector{3}, spacings) where {D}
    ntuple(Val(3)) do d
        a, b = mod1(d + 1, 3), mod1(d + 2, 3)
        Bd = f.B[d]
        map_blocks!(curl_kernel!, Bd, Bd.work, f.A[a].work, f.A[b].work, spacings, 𝔅̄[d],
                    Val(D), Val(d), Val(Bd.G), Val(f.A[a].G), Val(f.A[b].G); stored=true)
        nothing
    end
    return f
end

"""
    cell_𝔅(B, n, b, GB, ::Val{D}) -> SVector{3}

The densitized field `𝔅ⁱ` averaged to the cell with local index `n` of
block `b`, from the three face arrays `B` (each `f.B[d].work`) of ghost
widths `GB`: along a present direction `d` the mean of the two faces that
bound the cell, `(𝔅ᵈ[n] + 𝔅ᵈ[n + e_d])/2`, and normal to a missing one the
cell-like face value itself.
"""
@inline function cell_𝔅(B::NTuple{3,Any}, n::NTuple{D,Int}, b::Int, GB::NTuple{3,Any},
                        ::Val{D}) where {D}
    return SVector(ntuple(Val(3)) do d
        @inbounds begin
            lo = B[d][stored(n, GB[d])..., 1, b]
            if d ≤ D
                hi = B[d][stored(shift(n, d, 1), GB[d])..., 1, b]
                (lo + hi) / 2
            else
                lo
            end
        end
    end)
end
