# This package's side of the coupling contract (redesigned by Erik 2026-11-08;
# step 10, round 2): the producer's metric crosses at its owned vertices as
# `(α, β^i, γ_ij, K_ij)`, and every conversion between centerings is made here
# ("The coupling contract" and "Metric at cells, faces and edges" in `CODE.md`).
#
# - `VertexMetricSet` holds `Mv`, a 16-variable vertex set at `G = 4` on the
#   matter's forest, with its ghost schedule and its outer-boundary hook.
# - `vertex_metric!(vm, adm, t)` copies the producer's adapter, bound to `Mv`,
#   into the owned vertices, fills `Mv`'s ghosts by TreeAMR (PointValue width 6
#   at refinement boundaries, the case's analytic metric at the outer faces),
#   and returns the provider the kernels read, `VertexMetric`.
# - `VertexMetric` answers the indexed calls (`point_metric(prov, t, x, Val(C),
#   n, b)`, `metric_gradient` at the cells) by 4th-order Lagrange interpolation
#   from `Mv` — 4, 16 and 64 points to edges, faces and cells; the vertices
#   read — with the derivatives of the same interpolant, and falls back to
#   multilinear weights where the result has `α ≤ 0` or a `γ` that is not
#   positive definite.
#
# The producer's adapter is anything in TreeGH's `ADMMetric` form:
# `at = adm(Mv)` binds it to a vertex set on its forest, and
# `(; α, β, γ, K) = at(n, b)` answers at an owned index `n` (`1 ≤ n[d] ≤ N`) of
# block `b`. TreeGH's own (GH-1), or a sampled SpacetimeMetrics metric in the
# tests.

"""Number of variables of the vertex metric set `Mv`: `α`, `β^i`, `γ_ij`, `K_ij`."""
const NM = 16
# The slots of `Mv`: `α`, the contravariant `β^i`, then `γ_ij` and `K_ij` as
# their upper triangles in the order `xx, xy, xz, yy, yz, zz`.
const M_ALPHA = 1
const M_BETA = 2
const M_GAMMA = 5
const M_K = 11

"""
    default_ops_M()

The operators of the vertex metric set `Mv` ([`VertexMetricSet`](@ref)):
PointValue **width 6** for the prolongation and the restriction (proposed,
2026-11-08). The kernels differentiate `Mv`'s interpolant to 4th order next to
a coarse-fine face; TreeGH's interface rule, `p = q + 2` for the second
derivatives it takes, gives `p ≥ q + 1` for a first derivative, 5, and
PointValue widths are even ("Metric at cells, faces and edges" in `CODE.md`).
"""
default_ops_M() = Operators(; prolongation=6, restriction=6)

"""
    VertexMetricSet(T, forest; boundary, G = 4, ops = default_ops_M(), backend = CPU())

The vertex metric set `Mv` of the coupling contract, on the matter's `forest`
("Metric at cells, faces and edges" in `CODE.md`; step 10, round 2): 16
variables, vertex-centered, ghost width `G`, with its ghost schedule. Fill it
with [`vertex_metric!`](@ref), which returns the provider the kernels read.

- **`G = 4`** by default, so that a 4th-order interpolation reaches the
  outermost ghost cell of a `G = 3` cell set: a cell's stencil reaches one
  vertex beyond it on either side. TreeAMR's vertex invariant `N ≥ 2G + 2`
  then needs `N ≥ 10`.
- **`boundary`** is the metric at the outer faces: a provider evaluated at
  coordinates, the case's own (`Analytic(m)`, or `Flat()`), or `nothing` on
  a periodic forest. It has no default: what a producer's metric is beyond
  the domain is the case's to say.
- `D = 3` only: the coupling is three-dimensional.
"""
struct VertexMetricSet{FS,SC,OP,OR,SP,BC}
    M::FS
    schedule::SC
    ops::OP
    origins::OR                    # per block, on the set's backend
    spacings::SP
    boundary::BC                   # a coordinate provider, or `nothing`
end

function VertexMetricSet(::Type{T}, forest::Forest{D}; boundary, G::Integer=4,
                         ops::Operators=default_ops_M(), backend=CPU()) where {T,D}
    D == 3 || throw(ArgumentError(
        "the vertex metric set is three-dimensional (the coupling is D = 3), got D = $D."))
    G ≥ 4 || throw(ArgumentError(
        "the vertex metric set needs G ≥ 4: a 4th-order interpolation to the outermost " *
        "ghost cell of a G = 3 cell set reaches one vertex beyond it (CODE.md, \"Metric at " *
        "cells, faces and edges\"); got G = $G."))
    forest.N ≥ 2G + 2 || throw(ArgumentError(
        "the vertex metric set at G = $G needs N ≥ $(2G + 2) (TreeAMR's invariant for a " *
        "vertex-centered set, N ≥ 2G + 2); the forest has N = $(forest.N)."))
    boundary === nothing || boundary isa MetricProvider || throw(ArgumentError(
        "boundary is the metric at the outer faces, a provider evaluated at coordinates " *
        "(the case's own, e.g. Analytic(m)), or nothing on a periodic forest; got " *
        "$(typeof(boundary))."))
    if boundary === nothing && !all(forest.periodic)
        throw(ArgumentError(
            "the forest is not periodic in every dimension ($(forest.periodic)) but the " *
            "vertex metric set has no boundary: its ghosts beyond the outer faces would " *
            "hold whatever the allocation left."))
    end
    M = FieldSet{T}(forest, NM; G=G, centering=vertexcentered(3), backend=backend)
    M.work .= T(NaN)
    schedule = GhostSchedule(M, ops)
    origins = to_backend(backend, block_origins(forest, T))
    spacings = to_backend(backend, block_spacings(forest, T))
    return VertexMetricSet(M, schedule, ops, origins, spacings, boundary)
end

"""
    rebuild(vm::VertexMetricSet) -> VertexMetricSet

The same set, its schedule and per-block geometry rebuilt from the forest as
it is now, after a [`regrid!`](@ref) that carried `Mv` as scratch
([`regrid_pairs`](@ref)). Its values are `NaN` until the next
[`vertex_metric!`](@ref): the producer's rebuilt adapter refills it.
"""
function rebuild(vm::VertexMetricSet)
    T = eltype(vm.M.work)
    vm.M.work .= T(NaN)
    backend = get_backend(vm.M.work)
    forest = vm.M.forest
    return VertexMetricSet(vm.M, GhostSchedule(vm.M, vm.ops), vm.ops,
                           to_backend(backend, block_origins(forest, T)),
                           to_backend(backend, block_spacings(forest, T)), vm.boundary)
end

"""
    regrid_pairs(vm::VertexMetricSet) -> (Mv => nothing,)

`Mv` as `regrid!` takes it: scratch, refilled from the producer after the
regrid, never transferred.
"""
regrid_pairs(vm::VertexMetricSet) = (vm.M => nothing,)

# The outer faces of `Mv`: the boundary provider's metric at the ghost vertex,
# `(α, β, γ, K)` in `Mv`'s slots, at the time of the fill.
struct MetricBoundaryAt{P,T}
    provider::P
    t::T
end

@inline function (m::MetricBoundaryAt)(x::NTuple{3,T}, δ) where {T}
    xs = SVector(x[1], x[2], x[3])
    g = point_metric(m.provider, T(m.t), xs)
    K = metric_gradient(m.provider, T(m.t), xs).K
    return pack_adm(g.α, g.β, g.γ, K)
end

@inline pack_adm(α, β, γ, K) =
    @inbounds (α, β[1], β[2], β[3], γ[1, 1], γ[1, 2], γ[1, 3], γ[2, 2], γ[2, 3], γ[3, 3],
               K[1, 1], K[1, 2], K[1, 3], K[2, 2], K[2, 3], K[3, 3])

# The owned vertices of `Mv` from the producer's adapter, bound to `Mv`.
@kernel function vertex_copy_kernel!(Mw, at, ::Val{G}) where {G}
    I = @index(Global, NTuple)                       # local, owned
    b = I[4]
    n = (I[1], I[2], I[3])
    v = at(n, b)
    vals = pack_adm(v.α, v.β, v.γ, v.K)
    s = (n[1] + G[1], n[2] + G[2], n[3] + G[3])
    @inbounds for k in 1:NM
        Mw[s..., k, b] = vals[k]
    end
end

"""
    vertex_metric!(vm::VertexMetricSet, adm, t) -> VertexMetric

Fills `Mv` from the producer's metric adapter `adm` and returns the provider
the kernels read ("The coupling contract" in `CODE.md`):

1. `at = adm(vm.M)` binds the adapter to `Mv`, and one kernel over the owned
   vertices copies `(; α, β, γ, K) = at(n, b)` into them; no ghost is read
   from the producer;
2. `fill_ghosts!` fills `Mv`'s ghosts: copies at same-level faces, the
   PointValue prolongation at refinement boundaries, and at the outer faces
   the set's `boundary` provider at time `t`.

The returned [`VertexMetric`](@ref) is a snapshot of `Mv`: valid until the
next `vertex_metric!` or a regrid. `t` is the fill's time (the outer faces'),
and the time of the producer's state.
"""
function vertex_metric!(vm::VertexMetricSet, adm, t)
    M = vm.M
    at = adm(M)
    map_blocks!(vertex_copy_kernel!, M, M.work, at, Val(M.G))
    hook = vm.boundary === nothing ? nothing :
           CellBoundary(AllVariables(MetricBoundaryAt(vm.boundary, t)))
    fill_ghosts!(M, vm.schedule; boundary=hook)
    return VertexMetric(vm)
end

"""
    SampledADM(metric, vm::VertexMetricSet, t)

A SpacetimeMetrics metric in the producer's adapter form (TreeGH's
`ADMMetric`), at time `t`: bound to a vertex set on `vm`'s forest,
`at = adm(Mv)`, it answers `(; α, β, γ, K) = at(n, b)` at a local vertex
index from `adm_decompose` and `ExtrinsicCurvature` at the vertex's position
(the kernels' `point_x`). What step 10's tests and studies fill `Mv` with,
before TreeGH is loaded, and GH-1's reference in step 11 (step 10, round 2).
`α`, `β`, `γ` are the analytic provider's to the bit; `K` is exact, so that
what a comparison measures is this package's interpolation.
"""
struct SampledADM{M,O,H,T}
    metric::M
    origins::O
    spacings::H
    t::T
end

SampledADM(metric::SpacetimeMetrics.AbstractMetric, vm::VertexMetricSet, t) =
    SampledADM(metric, vm.origins, vm.spacings, eltype(vm.M.work)(t))

struct SampledADMAt{M,O,H,T}
    s::SampledADM{M,O,H,T}
end

function (s::SampledADM)(fs::FieldSet)
    staggers(fs) == (1, 1, 1) || throw(ArgumentError(
        "the producer's metric adapter binds to a vertex set (the coupling contract's " *
        "metric crosses at the vertices), got staggers $(staggers(fs))."))
    return SampledADMAt(s)
end

Adapt.adapt_structure(to, a::SampledADMAt) =
    SampledADMAt(SampledADM(a.s.metric, Adapt.adapt(to, a.s.origins),
                            Adapt.adapt(to, a.s.spacings), a.s.t))

@inline function (a::SampledADMAt)(n::NTuple{3,Int}, b)
    s = a.s
    x = point_x(@inbounds(s.origins[b]), @inbounds(s.spacings[b]), n, Val((1, 1, 1)))
    p = SVector(s.t, x[1], x[2], x[3])
    α, β, γ = SpacetimeMetrics.adm_decompose(s.metric, p)
    g, sdg = SpacetimeMetrics.dmetric(s.metric, p)
    return (; α, β, γ, K=SpacetimeMetrics.ExtrinsicCurvature(g, sdg))
end

"""
    VertexMetric

The metric provider of the coupling contract (step 10, round 2): this package's
vertex metric set `Mv`, filled by [`vertex_metric!`](@ref), read at a point's
staggers and local index ("Metric at cells, faces and edges" in `CODE.md`).

- **Vertices** are read; **edges**, **faces** and **cells** are 4th-order
  tensor Lagrange interpolations of the 16 variables from 4, 16 and 64
  vertices — the vertices `n − 1 … n + 2` along each cell-like dimension.
- **The derivatives** at the cells (the sources' only reader) are the same
  interpolant's: the staggered 4th-order derivative along `j`,
  `(f₋₁ − 27 f₀ + 27 f₁ − f₂)/(24h)`, times the transverse interpolation.
  `K_ij` is interpolated, not differentiated. The four-metric's `∂_j g_ab`
  is assembled from `∂α`, `∂β`, `∂γ` by the product rule
  ([`gradient_from_adm`](@ref)), so that the sources read
  [`MetricGradient`](@ref) as they do from every provider.
- **The fallback.** Where the Lagrange result has `α ≤ 0`, or a `γ_ij` that is
  not positive definite (a leading minor `≤ 0`: the test a Cholesky
  factorization makes), the point uses multilinear weights instead (2, 4 or
  8 vertices), and at a cell the derivatives of that interpolant, 2nd order.
  Multilinear weights make `(α, γ_ij)` a convex combination of the corners,
  so the result is a Lorentzian metric wherever the corners are.
  [`metric_fallbacks`](@ref) counts the points that fall back.

`isbits` once adapted to the backend. `D = 3` only. The coordinate calls
(`point_metric(prov, t, x)`) refuse it: the initial data, the boundary hooks
and the interior take the case's own provider.
"""
struct VertexMetric{W,SP} <: MetricProvider
    w::W                           # `Mv`'s working array
    spacings::SP                   # per block
    G::NTuple{3,Int}
end

VertexMetric(vm::VertexMetricSet) =
    VertexMetric(vm.M.work, vm.spacings, ntuple(e -> Int(vm.M.G[e]), 3))

Adapt.adapt_structure(to, p::VertexMetric) =
    VertexMetric(Adapt.adapt(to, p.w), Adapt.adapt(to, p.spacings), p.G)

# --- the interpolation --------------------------------------------------------------

# The weights along one dimension. A vertex-like dimension reads its own
# vertex; a cell-like one the four vertices `n − 1 … n + 2` (Lagrange, the cell
# centre midway between `n` and `n + 1`) or the two `n`, `n + 1` (linear).
@inline lagrange_w(::Type{T}) where {T} = (T(-1 // 16), T(9 // 16), T(9 // 16), T(-1 // 16))
@inline lagrange_dw(::Type{T}) where {T} = (T(1 // 24), T(-27 // 24), T(27 // 24), T(-1 // 24))
@inline linear_w(::Type{T}) where {T} = (T(1 // 2), T(1 // 2))
@inline linear_dw(::Type{T}) where {T} = (-one(T), one(T))

@inline axis_w(::Type{T}, c, ::Val{4}) where {T} = c == 1 ? (one(T),) : lagrange_w(T)
@inline axis_w(::Type{T}, c, ::Val{2}) where {T} = c == 1 ? (one(T),) : linear_w(T)
# The stored index of the first vertex a dimension reads.
@inline axis_start(n, G, c, ::Val{4}) = c == 1 ? n + G : n + G - 1
@inline axis_start(n, G, c, ::Val{2}) = c == 1 ? n + G : n + G

@inline read_m(w, i, j, k, b, ::Type{T}) where {T} =
    SVector{NM,T}(ntuple(v -> @inbounds(w[i, j, k, v, b]), Val(NM)))

# The interpolated 16 variables at a point with staggers `C`, from the stencil
# of `K` points per cell-like dimension.
@inline function interp_value(p::VertexMetric, ::Val{C}, n::NTuple{3,Int}, b, ::Type{T},
                              k::Val) where {C,T}
    wx, wy, wz = axis_w(T, C[1], k), axis_w(T, C[2], k), axis_w(T, C[3], k)
    i0 = axis_start(n[1], p.G[1], C[1], k)
    j0 = axis_start(n[2], p.G[2], C[2], k)
    k0 = axis_start(n[3], p.G[3], C[3], k)
    acc = zero(SVector{NM,T})
    for c in eachindex(wz), bj in eachindex(wy), a in eachindex(wx)
        @inbounds acc += (wx[a] * wy[bj] * wz[c]) *
                         read_m(p.w, i0 + a - 1, j0 + bj - 1, k0 + c - 1, b, T)
    end
    return acc
end

# At a cell: the value and its three derivatives, from one pass over the stencil.
@inline function interp_cell(p::VertexMetric, n::NTuple{3,Int}, b, inv_h::T,
                             k::Val{K}) where {T,K}
    w = K == 4 ? lagrange_w(T) : linear_w(T)
    dw = K == 4 ? lagrange_dw(T) : linear_dw(T)
    i0 = axis_start(n[1], p.G[1], 0, k)
    j0 = axis_start(n[2], p.G[2], 0, k)
    k0 = axis_start(n[3], p.G[3], 0, k)
    v0 = zero(SVector{NM,T})
    v1 = v0
    v2 = v0
    v3 = v0
    for c in 1:K, bj in 1:K, a in 1:K
        f = read_m(p.w, i0 + a - 1, j0 + bj - 1, k0 + c - 1, b, T)
        @inbounds begin
            v0 += (w[a] * w[bj] * w[c]) * f
            v1 += (dw[a] * w[bj] * w[c]) * f
            v2 += (w[a] * dw[bj] * w[c]) * f
            v3 += (w[a] * w[bj] * dw[c]) * f
        end
    end
    return v0, v1 * inv_h, v2 * inv_h, v3 * inv_h
end

# `α > 0` and `γ_ij` positive definite, by its leading minors (Sylvester's
# criterion, the test a Cholesky factorization makes). `false` on a NaN.
@inline function admissible_adm(v::SVector{NM,T}) where {T}
    @inbounds begin
        gxx, gxy, gxz = v[M_GAMMA], v[M_GAMMA + 1], v[M_GAMMA + 2]
        gyy, gyz, gzz = v[M_GAMMA + 3], v[M_GAMMA + 4], v[M_GAMMA + 5]
        m2 = gxx * gyy - gxy * gxy
        m3 = gxx * (gyy * gzz - gyz * gyz) - gxy * (gxy * gzz - gyz * gxz) +
             gxz * (gxy * gyz - gyy * gxz)
        return (v[M_ALPHA] > 0) & (gxx > 0) & (m2 > 0) & (m3 > 0)
    end
end

@inline sym3(v::SVector{NM,T}, o) where {T} =
    @inbounds SMatrix{3,3,T,9}(v[o], v[o + 1], v[o + 2], v[o + 1], v[o + 3], v[o + 4],
                               v[o + 2], v[o + 4], v[o + 5])
@inline vec3(v::SVector{NM}, o) = @inbounds SVector(v[o], v[o + 1], v[o + 2])

@inline metric_of(v::SVector{NM}) = Metric3(@inbounds(v[M_ALPHA]), vec3(v, M_BETA), sym3(v, M_GAMMA))

# The interpolated value at a point, with the fallback.
@inline function interp_point(p::VertexMetric, c::Val{C}, n, b, ::Type{T}) where {C,T}
    v = interp_value(p, c, n, b, T, Val(4))
    admissible_adm(v) && return v, false
    return interp_value(p, c, n, b, T, Val(2)), true
end

@inline function point_metric(p::VertexMetric, t, x::SVector{3,T}, c::Val, n::NTuple{3,Int},
                              b) where {T}
    v, _ = interp_point(p, c, n, b, T)
    return metric_of(v)
end

@inline function metric_gradient(p::VertexMetric, t, x::SVector{3,T}, ::Val{C},
                                 n::NTuple{3,Int}, b) where {T,C}
    all(==(0), C) || throw(ArgumentError(
        "metric_gradient: the sources read the metric's derivatives at the cells only; " *
        "the vertex metric differentiates its interpolant there (CODE.md, \"Metric at " *
        "cells, faces and edges\")."))
    inv_h = inv(@inbounds p.spacings[b])
    v, d1, d2, d3 = interp_cell(p, n, b, inv_h, Val(4))
    if !admissible_adm(v)
        v, d1, d2, d3 = interp_cell(p, n, b, inv_h, Val(2))
    end
    return gradient_from_adm(v, (d1, d2, d3))
end

"""
    gradient_from_adm(v, (∂₁v, ∂₂v, ∂₃v)) -> MetricGradient

The [`MetricGradient`](@ref) of the 3+1 variables `v` in `Mv`'s slots and
their spatial derivatives: `K_ij` as it is, and `∂_j g_ab` by the product rule
on `g_tt = −α² + γ_kl βᵏβˡ`, `g_tk = γ_kl βˡ`, `g_kl = γ_kl` (decided in step
10, round 2: the sources keep reading the four-metric's derivatives, and
[`d_lapse`](@ref), [`d_shift`](@ref) recover `∂α`, `∂β` from them to
rounding). The second of this package's two conversions of metric
derivatives, beside `spatial_gradient`.
"""
@inline function gradient_from_adm(v::SVector{NM,T}, dv::NTuple{3,SVector{NM,T}}) where {T}
    α = @inbounds v[M_ALPHA]
    β = vec3(v, M_BETA)
    γ = sym3(v, M_GAMMA)
    βl = γ * β
    rows = ntuple(Val(3)) do j
        d = dv[j]
        dα = @inbounds d[M_ALPHA]
        dβ = vec3(d, M_BETA)
        dγ = sym3(d, M_GAMMA)
        dβl = dγ * β + γ * dβ                        # ∂_j g_tk
        dtt = -2 * α * dα + β' * (dγ * β) + 2 * (βl' * dβ)
        (dtt, dβl, dγ)
    end
    dg = SArray{Tuple{3,4,4}}(
        begin
            (dtt, dβl, dγ) = rows[j]
            a == 1 && bb == 1 ? dtt : a == 1 ? dβl[bb - 1] : bb == 1 ? dβl[a - 1] :
            dγ[a - 1, bb - 1]
        end for j in 1:3, a in 1:4, bb in 1:4)
    return MetricGradient(dg, sym3(v, M_K))
end

point_metric(::VertexMetric, t, x) = throw(ArgumentError(
    "the vertex metric is read at a point's index, not at coordinates: call " *
    "point_metric(provider, t, x, Val(C), n, b) (CODE.md, \"The coupling contract\"). The " *
    "initial data, the boundary hooks and the interior take the case's analytic provider."))
metric_gradient(::VertexMetric, t, x) = throw(ArgumentError(
    "the vertex metric is read at a point's index, not at coordinates: call " *
    "metric_gradient(provider, t, x, Val(C), n, b) (CODE.md, \"The coupling contract\")."))

# --- the fallback count ---------------------------------------------------------------

@kernel function fallback_kernel!(flags, p, ::Val{C}, ::Val{G}) where {C,G}
    I = @index(Global, NTuple)                       # stored: `stored = true`
    b = I[4]
    n = (I[1] - G[1], I[2] - G[2], I[3] - G[3])
    T = eltype(flags)
    _, fell = interp_point(p, Val(C), n, b, T)
    @inbounds flags[I[1], I[2], I[3], b] = fell ? one(T) : zero(T)
end

function count_fallbacks(p::VertexMetric, fs)
    w = fs.work
    flags = similar(w, eltype(w), (size(w, 1), size(w, 2), size(w, 3), size(w, 5)))
    C = staggers(fs)
    map_blocks!(fallback_kernel!, fs, flags, p, Val(C),
                Val(ntuple(e -> Int(fs.G[e]), 3)); stored=true)
    return Int(sum(flags))
end

"""
    metric_fallbacks(prov::VertexMetric, f::MHDFields{3}) -> (; cells, faces, edges, vertices)

The points at which `prov` falls back to multilinear weights, at every stored
point the kernels read: the cells of `P`, the faces of `B[d]`, the edges of
`A[j]` and the vertices of `Φ̃` (`B`'s and `A`'s summed over the three
directions). A diagnostic pass over the snapshot, since the provider itself is
a pure function of the point (proposed in step 10, round 2: `X` holds per-cell
tallies only, and a vertex or edge has no cell of its own).
"""
metric_fallbacks(p::VertexMetric, f::MHDFields{3}) =
    (; cells=count_fallbacks(p, f.P), faces=sum(s -> count_fallbacks(p, s), f.B),
     edges=sum(s -> count_fallbacks(p, s), f.A), vertices=count_fallbacks(p, f.Φ))
