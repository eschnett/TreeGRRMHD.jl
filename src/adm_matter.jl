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
