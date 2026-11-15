# This package's side of the coupling contract's matter direction (GH-3; step 11): the
# stress-energy crosses to the producer's owned vertices as its 3+1 projections
# `(ρ, S_i, S_ij)`, covariant and not densitized, and the conversion from this package's
# cells to the vertices is made here ("The coupling contract", "Matter to the vertices"
# in `CODE.md`).
#
# `ADMMatter(p)` wraps a prepared problem (after `prepare!`, whose `P` holds every cell's
# primitives, fields and metric, ghosts included). Bound to the producer's vertex set on
# the same forest, `at = ADMMatter(p)(U)`, it answers `(; ρ, S, Sij) = at(n, b)` at an
# owned vertex `n` of local block `b`: the mean of the `2³` cells around the vertex of
# `stress_energy`'s `𝓔`, `S_i` and `S^ij` lowered by each cell's own `γ_ij` — second
# order, and a convex mean, so `ρ ≥ 0` wherever the cells' are. Computed per call from
# `P`, not stored: no set of its own, nothing to regrid.
#
# The stored form, `ADMMatter(p, M; order)`, first copies every stored cell's 10 values
# and `√γ` into a cell set `M` of the caller's (`adm_matter_set`), then answers from `M`:
# `order = 2` is the same mean, bit for bit; `order = 4` reads the `4³` cells around the
# vertex. The cells hold averages of the densitized `√γ 𝓔`, `√γ S_i` (exactly, as `U`'s
# cell averages) and `√γ S_ij` (to second order: the pressure of the average), and the
# metric's `√γ` at the cell centres; so the vertex value is the average-to-point
# interpolation `(−1, 7, 7, −1)/12` along each direction of the densitized values,
# divided by the point-to-point interpolation `(−1, 9, 9, −1)/16` of `√γ`. Where it
# would undershoot the least of the eight nearest cells' `𝓔` or of a diagonal `S_ii`
# (beyond a stellar surface, where the interpolation rings), the vertex takes the
# eight-cell mean instead, so `ρ ≥ 0` still holds.

"""
    ADMMatter(p::MHDProblem)

The matter adapter of the coupling contract (GH-3): this package's stress-energy at a
producer's owned vertices, as `(ρ, S_i, S_ij)` ("Matter to the vertices" in `CODE.md`).
`p` must be prepared ([`prepare!`](@ref)) at the state and time the producer reads: the
adapter reads `p`'s cell set `P` in place, so it is a snapshot, valid until the next
`prepare!`, a limiter or a regrid.

Bound to the producer's vertex set on the same forest, as TreeGH's right-hand side binds
its `matter` argument,

```julia
at = ADMMatter(p)(U)          # U: vertex-centered, on p's forest
(; ρ, S, Sij) = at(n, b)      # at an owned vertex, 1 ≤ n[d] ≤ N, of local block b
```

`ρ = n^a n^b T_ab`, `S_i = −γ_i^a n^b T_ab` (`SVector{3}`) and `S_ij = γ_i^a γ_j^b T_ab`
(symmetric `SMatrix{3,3}`), in the state's element type, the mean over the `2³` cells
that share the vertex of [`stress_energy`](@ref)'s `𝓔`, `S_i` and `S^ij` lowered with
each cell's `γ_ij`. The bound object is `isbits` once adapted to the backend.
"""
struct ADMMatter{FS}
    P::FS
end

ADMMatter(p::MHDProblem) = ADMMatter(p.f.P)

struct ADMMatterAt{W,G}
    Pw::W
    valG::Val{G}
end

Adapt.adapt_structure(to, a::ADMMatterAt) = ADMMatterAt(Adapt.adapt(to, a.Pw), a.valG)

function (m::ADMMatter)(U::FieldSet)
    U.forest === m.P.forest || throw(ArgumentError(
        "the matter adapter binds to a vertex set on the matter's own forest (the " *
        "coupling contract crosses at the producer's owned vertices of one mesh); the " *
        "vertex set is on another forest."))
    staggers(U) == (1, 1, 1) || throw(ArgumentError(
        "the matter adapter binds to a vertex set (the coupling contract's matter " *
        "crosses at the vertices), got staggers $(staggers(U))."))
    all(≥(1), m.P.G) || throw(ArgumentError(
        "the matter adapter reads the cell below every owned vertex, so the cell set " *
        "needs G ≥ 1, got $(m.P.G)."))
    return ADMMatterAt(m.P.work, Val(m.P.G))
end

# One cell's projections, `S^ij` lowered: `(𝓔, S_i, S_ij)`.
@inline function cell_adm(Pc::NTuple{NP,T}) where {T}
    m = aux_metric(Pc)
    𝓔, S, Su = stress_energy(m, Pc, aux_E(Pc), aux_B(Pc))
    return 𝓔, S, mul3(mul3(m.γ, Su), m.γ)
end

@inline function (a::ADMMatterAt{W,G})(n::NTuple{3,Int}, b) where {W,G}
    T = eltype(a.Pw)
    # Vertex `n` is the low corner of owned cell `n`: the cells `n − 1` and `n` along
    # each direction share it, at stored indices `n − 1 + G` and `n + G`.
    s = (n[1] + G[1], n[2] + G[2], n[3] + G[3])
    ρ = zero(T)
    S = zero(SVector{3,T})
    Sij = zero(SMatrix{3,3,T,9})
    for k in -1:0, j in -1:0, i in -1:0
        Pc = read_vars(a.Pw, (s[1] + i, s[2] + j, s[3] + k), b, Val(NP))
        e, Si, Sl = cell_adm(Pc)
        ρ += e
        S += Si
        Sij += Sl
    end
    w = T(1 // 8)
    # Symmetric by construction: each cell's `γ S^ij γ` is.
    return (; ρ=w * ρ, S=w * S, Sij=w * Sij)
end

# The stored form. `M`'s 11 values per cell: `𝓔`, `S_i`, the upper triangle of `S_ij`
# (row by row), `√γ`.
const NADM = 11

"""
    adm_matter_set(p::MHDProblem) -> FieldSet

The cell set the stored [`ADMMatter`](@ref) fills: 11 values per cell (`𝓔`, `S_i`,
the six of `S_ij`, `√γ`), with `P`'s ghost width, on `p`'s forest and backend. The
caller owns it and makes a new one after a regrid; it holds nothing between calls that
a regrid would have to carry.
"""
function adm_matter_set(p::MHDProblem)
    P = p.f.P
    return FieldSet{eltype(P.work)}(P.forest, NADM; G=P.G, centering=cellcentered(3),
                                    backend=get_backend(P.work))
end

# The stored adapter before binding: `M` filled, the order a type parameter.
struct ADMMatterStored{FS,O}
    M::FS
    valO::Val{O}
end

"""
    ADMMatter(p::MHDProblem, M; order)

The stored form of the matter adapter: copies every stored cell's `(𝓔, S_i, S_ij, √γ)`
from `p`'s prepared `P` into `M` ([`adm_matter_set`](@ref)) once, then answers each
vertex from `M`. `order = 2` is the per-call form's eight-cell mean, bit for bit;
`order = 4` interpolates the densitized values from the `4³` cells around the vertex,
as cell averages, and falls back to the mean where that would undershoot the eight
nearest cells ("Matter to the vertices" in `CODE.md`). `order` has no default. A
snapshot, as the per-call form is: valid until the next `prepare!`, limiter or regrid.
"""
function ADMMatter(p::MHDProblem, M::FieldSet; order::Integer)
    order in (2, 4) || throw(ArgumentError(
        "the stored matter adapter is second order (the eight-cell mean) or fourth (the " *
        "4³-cell interpolation); got order = $order."))
    P = p.f.P
    M.forest === P.forest || throw(ArgumentError(
        "the stored matter adapter copies `P` cell by cell, so its set must be on `P`'s " *
        "forest (`adm_matter_set(p)`, made again after a regrid); it is on another."))
    (M.G == P.G && staggers(M) == staggers(P) && size(M.work, 4) == NADM) ||
        throw(ArgumentError(
            "the stored matter adapter copies `P` cell by cell, so its set must be " *
            "`adm_matter_set(p)`: $NADM cell-centered values with `P`'s ghost width " *
            "$(P.G); got $(size(M.work, 4)) values, staggers $(staggers(M)), G = $(M.G)."))
    map_blocks!(adm_store_kernel!, M, M.work, P.work; stored=true)
    return ADMMatterStored(M, Val(Int(order)))
end

@kernel function adm_store_kernel!(Mw, @Const(Pw))
    I = @index(Global, NTuple)                       # stored: `stored = true`
    b = I[4]
    c = (I[1], I[2], I[3])
    Pc = read_vars(Pw, c, b, Val(NP))
    e, S, Sl = cell_adm(Pc)
    vals = (e, S[1], S[2], S[3], Sl[1, 1], Sl[1, 2], Sl[1, 3], Sl[2, 2], Sl[2, 3],
            Sl[3, 3], Pc[P_SQRTG])
    @inbounds for v in 1:NADM
        Mw[c..., v, b] = vals[v]
    end
end

struct ADMMatterStoredAt{W,G,O}
    Mw::W
    valG::Val{G}
    valO::Val{O}
end

Adapt.adapt_structure(to, a::ADMMatterStoredAt) =
    ADMMatterStoredAt(Adapt.adapt(to, a.Mw), a.valG, a.valO)

function (m::ADMMatterStored{FS,O})(U::FieldSet) where {FS,O}
    U.forest === m.M.forest || throw(ArgumentError(
        "the matter adapter binds to a vertex set on the matter's own forest (the " *
        "coupling contract crosses at the producer's owned vertices of one mesh); the " *
        "vertex set is on another forest."))
    staggers(U) == (1, 1, 1) || throw(ArgumentError(
        "the matter adapter binds to a vertex set (the coupling contract's matter " *
        "crosses at the vertices), got staggers $(staggers(U))."))
    all(≥(O ÷ 2), m.M.G) || throw(ArgumentError(
        "the order-$O matter adapter reads $(O ÷ 2) cells below every owned vertex, so " *
        "the cell set needs G ≥ $(O ÷ 2), got $(m.M.G)."))
    return ADMMatterStoredAt(m.M.work, Val(m.M.G), Val(O))
end

@inline adm_read(Mw, c, b, ::Val{T}) where {T} =
    SVector{NADM,T}(ntuple(v -> @inbounds(Mw[c[1], c[2], c[3], v, b]), Val(NADM)))

@inline function adm_unpack(q::SVector{NADM,T}) where {T}
    @inbounds return (; ρ=q[1], S=SVector(q[2], q[3], q[4]),
                      Sij=SMatrix{3,3,T,9}(q[5], q[6], q[7], q[6], q[8], q[9], q[7], q[9],
                                           q[10]))
end

# The 1D weights at the cells `n − 2, n − 1, n, n + 1` around vertex `n` (offsets
# `i = −2 … 1`): average to point, `(−1, 7, 7, −1)/12`, and point to point,
# `(−1, 9, 9, −1)/16`.
@inline adm_wa(::Type{T}, i) where {T} = i == -2 || i == 1 ? T(-1 // 12) : T(7 // 12)
@inline adm_wp(::Type{T}, i) where {T} = i == -2 || i == 1 ? T(-1 // 16) : T(9 // 16)

@inline function (a::ADMMatterStoredAt{W,G,O})(n::NTuple{3,Int}, b) where {W,G,O}
    T = eltype(a.Mw)
    s = (n[1] + G[1], n[2] + G[2], n[3] + G[3])
    # The eight-cell mean, in the per-call form's order of summation.
    q2 = zero(SVector{NADM,T})
    for k in -1:0, j in -1:0, i in -1:0
        q2 += adm_read(a.Mw, (s[1] + i, s[2] + j, s[3] + k), b, Val(T))
    end
    m2 = adm_unpack(T(1 // 8) * q2)
    O == 2 && return m2
    # Order 4: `Σ w_a √γ q` over the 4³ cells, over `Σ w_p √γ`, and the eight nearest
    # cells' least `𝓔`, `S_11`, `S_22`, `S_33`.
    acc = zero(SVector{NADM - 1,T})
    sg = zero(T)
    lo = SVector{4,T}(typemax(T), typemax(T), typemax(T), typemax(T))
    for k in -2:1, j in -2:1, i in -2:1
        q = adm_read(a.Mw, (s[1] + i, s[2] + j, s[3] + k), b, Val(T))
        g = q[NADM]
        qd = SVector{NADM - 1,T}(ntuple(v -> @inbounds(q[v]), Val(NADM - 1)))
        acc += (adm_wa(T, i) * adm_wa(T, j) * adm_wa(T, k) * g) * qd
        sg += adm_wp(T, i) * adm_wp(T, j) * adm_wp(T, k) * g
        if -1 ≤ i ≤ 0 && -1 ≤ j ≤ 0 && -1 ≤ k ≤ 0
            lo = min.(lo, SVector(q[1], q[5], q[8], q[10]))
        end
    end
    q4 = acc / sg
    any(SVector(q4[1], q4[5], q4[8], q4[10]) .< lo) && return m2
    return adm_unpack(SVector{NADM,T}(Tuple(q4)..., sg))
end
