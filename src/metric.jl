# Metric providers, pointwise: the metric and its first derivatives at one
# point, for the kernels that compute fluxes and sources.
#
# A provider is an `isbits` value the kernels dispatch on ("Metric
# providers" in `CODE.md`): `Flat()`, Minkowski in Cartesian coordinates;
# `Analytic(m)`, a SpacetimeMetrics metric evaluated at the point; and
# (step 10) `PointwiseMetric(adapter)`, a producer's metric read at the
# point's index in the coupling contract — TreeGH's (GH-1), or a sampled
# SpacetimeMetrics metric in its tests. All answer the same two questions,
# `point_metric` and `metric_gradient`.
#
# **The derivative index order.** This package's one layout for metric
# derivatives is `MetricGradient`'s `dg[j, a, b] = ∂_j g_ab`: the spatial
# derivative index first, then the two spacetime indices with 1 = t. It is
# GHSO2's order (derivative first, `dg[a,b,c] = ∂_a g_bc`), restricted to
# spatial derivatives. SpacetimeMetrics has the derivative *last*,
# `dg[a,b,c] = ∂_c g_ab`. The conversion is written here, in
# `spatial_gradient`, and nowhere else ("Sharp edges" in `PLAN.md`); a
# producer's pointwise metric goes through it too. Every other file reads
# `MetricGradient` only.
#
# **What is stored and what is derived** (proposed in step 2). The four-metric's
# spatial derivatives `∂_j g_ab` and the extrinsic curvature `K_ij` are the
# primary data, because both providers produce `∂_j g_ab` natively — the
# analytic one by ForwardDiff, TreeGH by differencing `h_ab` — and the default
# momentum source (`½ √−g T^{ab} ∂_j g_ab`, `sources.jl`) reads them as they
# are. The lapse and shift derivatives the Valencia form and the energy
# source read are derived from them (`d_lapse`, `d_shift`). Neither is stored
# in the primitive set `P`: the sources are their only reader, once per cell
# per right-hand side, so a provider evaluates them where the source is
# computed ("Field sets" in `CODE.md`).

"""
    MetricGradient{T}

The first derivatives of the metric at one point, as the sources read them:
`dg[j, a, b] = ∂_j g_ab` for the spatial directions `j = 1:3` and the
spacetime indices `a, b = 1:4` (1 is `t`), and the extrinsic curvature
`K[i, j] = K_ij` of the slice, `K_ij = −(∂ₜγ_ij − D_iβ_j − D_jβ_i)/(2α)`.

The derivative index comes **first** (as in TreeGH's GHSO2, and the reverse
of SpacetimeMetrics' `dmetric`); the conversion is in `src/metric.jl`
only. The lapse and shift derivatives are [`d_lapse`](@ref) and
[`d_shift`](@ref). `isbits`. See "Metric providers" in `CODE.md`.
"""
struct MetricGradient{T}
    dg::SArray{Tuple{3,4,4},T,3,48}
    K::SMatrix{3,3,T,9}
end

"""
    zero_gradient(T) -> MetricGradient{T}

A metric with no derivatives and no extrinsic curvature: Minkowski in
Cartesian coordinates, or any constant metric.
"""
@inline zero_gradient(::Type{T}) where {T} =
    MetricGradient{T}(zero(SArray{Tuple{3,4,4},T,3,48}), zero(SMatrix{3,3,T,9}))

"""
    MetricProvider

Where the kernels get the metric from ("Metric providers" in `CODE.md`):
[`Flat`](@ref) and [`Analytic`](@ref) evaluate it at a point;
a producer's [`PointwiseMetric`](@ref) (step 10) reads it at the point's
index. Every provider answers
[`point_metric`](@ref) and [`metric_gradient`](@ref).
"""
abstract type MetricProvider end

"""
    Flat()

Minkowski space in Cartesian coordinates, at every point and every time:
`α = 1`, `β = 0`, `γ_ij = δ_ij`, no derivatives. No storage; all `D`.
"""
struct Flat <: MetricProvider end

"""
    Analytic(m::SpacetimeMetrics.AbstractMetric)

A SpacetimeMetrics metric, evaluated at the point — cell, face, edge or
vertex — where a kernel needs it: the Kerr-Schild tests of rung 3. `m` is
`isbits`, so the provider is a kernel argument; its element type should be
the scheme's (`KerrSchild{Float32}` for a `Float32` run), since the metric
computes in its own type.
"""
struct Analytic{M<:SpacetimeMetrics.AbstractMetric} <: MetricProvider
    metric::M
end

"""
    point_metric(provider, t, x::SVector{3}) -> Metric3

The 3+1 metric at time `t` and position `x`.
"""
@inline point_metric(::Flat, t::T, x::SVector{3,T}) where {T} = flat_metric(T)

@inline function point_metric(a::Analytic, t::T, x::SVector{3,T}) where {T}
    α, β, γ = SpacetimeMetrics.adm_decompose(a.metric, SVector(t, x[1], x[2], x[3]))
    return Metric3(α, β, γ)
end

"""
    metric_gradient(provider, t, x::SVector{3}) -> MetricGradient

The spatial derivatives `∂_j g_ab` of the four-metric and the extrinsic
curvature `K_ij` at time `t` and position `x`, in this package's index order
([`MetricGradient`](@ref)).
"""
@inline metric_gradient(::Flat, t::T, x::SVector{3,T}) where {T} = zero_gradient(T)

@inline function metric_gradient(a::Analytic, t::T, x::SVector{3,T}) where {T}
    p = SVector(t, x[1], x[2], x[3])
    g, sdg = SpacetimeMetrics.dmetric(a.metric, p)
    return gradient_from_dmetric(g, sdg)
end

"""
    spatial_gradient(sdg) -> SArray{Tuple{3,4,4}}

**The one conversion**: SpacetimeMetrics' `sdg[a, b, c] = ∂_c g_ab`, with
`c = 1` the time derivative, to this package's `dg[j, a, b] = ∂_j g_ab` over
the spatial `j` (the time derivative enters through `K_ij` alone). Every
reader of a SpacetimeMetrics-convention derivative goes through it (the
analytic provider, a producer's pointwise metric, the spinning Wald case's
potential).
"""
@inline spatial_gradient(sdg) =
    SArray{Tuple{3,4,4}}(sdg[a, b, j + 1] for j in 1:3, a in 1:4, b in 1:4)

"""
    gradient_from_dmetric(g, sdg) -> MetricGradient

The [`MetricGradient`](@ref) of a four-metric `g` and its derivatives `sdg`
in SpacetimeMetrics' convention (what `dmetric` returns): the spatial
derivatives through [`spatial_gradient`](@ref), and `K_ij` from the same
data by SpacetimeMetrics 1.7's allocation-free `ExtrinsicCurvature(g, dg)`
(SM-1, landed; it replaced a local copy).
"""
@inline gradient_from_dmetric(g, sdg) =
    MetricGradient(spatial_gradient(sdg), SpacetimeMetrics.ExtrinsicCurvature(g, sdg))

# --- the indexed calls (step 10) -------------------------------------------------------

"""
    point_metric(provider, t, x, Val(C), n, b) -> Metric3
    metric_gradient(provider, t, x, Val(C), n, b) -> MetricGradient

The calls every kernel makes (added in step 10): the point's position `x`
and, beside it, its centering — the staggers `C` (`1` vertex-like, `0`
cell-like along each dimension), its **local** index `n` in its block (owned
points `1 … N`, ghosts outside) and the block `b`. A provider that computes
from coordinates ([`Flat`](@ref), [`Analytic`](@ref)) ignores the index and
forwards to the three-argument form; a producer's pointwise metric
([`PointwiseMetric`](@ref)) ignores `x` and reads its own data at the point
("The coupling contract" in `CODE.md`).
"""
@inline point_metric(prov::MetricProvider, t, x, ::Val, n, b) = point_metric(prov, t, x)
@inline metric_gradient(prov::MetricProvider, t, x, ::Val, n, b) = metric_gradient(prov, t, x)

"""
    PointwiseMetric(adapter)

A producer's metric in the coupling contract ("The coupling contract" in
`CODE.md`; added in step 10, proposed): `adapter` is bound to this package's
mesh — its forest, `N` and the ghost widths of the sets it is read on — and
a snapshot of the producer's state, and answers, at the point of centering
`Val(C)` with local index `n` in block `b`,

    SpacetimeMetrics.metric(adapter, Val(C), n, b) -> g::SMatrix{4,4}
    SpacetimeMetrics.dmetric(adapter, Val(C), n, b) -> (g, ∂g::SArray{Tuple{4,4,4}})

in SpacetimeMetrics' conventions (`∂g[a, b, c] = ∂_c g_ab`, `c = 1` the time,
`G = c = 1`). `metric` alone where the kernel needs no derivative, so that
the 70-odd evaluations per cell and step that read only `g` do not pay for
`∂g`. Both go through this file's one conversion. The adapter is the
producer's (TreeGH's, GH-1); a sampled SpacetimeMetrics metric in the same
contract tests this side (`test/tov_tests.jl`). It has no coordinate form:
the three-argument calls refuse it.
"""
struct PointwiseMetric{A} <: MetricProvider
    adapter::A
end

Adapt.adapt_structure(to, p::PointwiseMetric) = PointwiseMetric(Adapt.adapt(to, p.adapter))

@inline function point_metric(p::PointwiseMetric, t, x, c::Val, n, b)
    α, β, γ = SpacetimeMetrics.adm_decompose(SpacetimeMetrics.metric(p.adapter, c, n, b))
    return Metric3(α, β, γ)
end

@inline function metric_gradient(p::PointwiseMetric, t, x, c::Val, n, b)
    g, sdg = SpacetimeMetrics.dmetric(p.adapter, c, n, b)
    return gradient_from_dmetric(g, sdg)
end

point_metric(::PointwiseMetric, t, x) = throw(ArgumentError(
    "a producer's pointwise metric is read at a point's index, not at coordinates: call " *
    "point_metric(provider, t, x, Val(C), n, b) (CODE.md, \"The coupling contract\"). The " *
    "initial data, the boundary hooks and the interior take the case's analytic provider."))
metric_gradient(::PointwiseMetric, t, x) = throw(ArgumentError(
    "a producer's pointwise metric is read at a point's index, not at coordinates: call " *
    "metric_gradient(provider, t, x, Val(C), n, b) (CODE.md, \"The coupling contract\")."))

# `∂_j γ_ik` for one direction `j`, as a 3×3 matrix.
@inline function d_spatial(dm::MetricGradient{T}, j) where {T}
    dg = dm.dg
    return @inbounds SMatrix{3,3,T,9}(dg[j, 2, 2], dg[j, 3, 2], dg[j, 4, 2],
                                      dg[j, 2, 3], dg[j, 3, 3], dg[j, 4, 3],
                                      dg[j, 2, 4], dg[j, 3, 4], dg[j, 4, 4])
end

# `∂_j β_k = ∂_j g_tk`, the lowered shift's derivative, for one direction.
@inline d_shift_lower(dm::MetricGradient, j) =
    @inbounds SVector(dm.dg[j, 1, 2], dm.dg[j, 1, 3], dm.dg[j, 1, 4])

"""
    d_lapse(m::Metric3, dm::MetricGradient) -> SVector{3}

`∂_j α`, from `α² = −g_tt + γ^kl g_tk g_tl`:

    ∂_j α = (−∂_j g_tt + 2 βᵏ ∂_j β_k − βᵏ βˡ ∂_j γ_kl) / (2α)
"""
@inline function d_lapse(m::Metric3{T}, dm::MetricGradient{T}) where {T}
    β = m.β
    return SVector(ntuple(Val(3)) do j
        @inbounds (-dm.dg[j, 1, 1] + 2 * sum(β .* d_shift_lower(dm, j)) -
                   sum(β .* mul3(d_spatial(dm, j), β))) / (2 * m.α)
    end)
end

"""
    d_shift(m::Metric3, dm::MetricGradient) -> SMatrix{3,3}

`∂_j βⁱ` as `dβ[j, i]` (derivative first), from `βⁱ = γ^ik β_k`:

    ∂_j βⁱ = γ^ik (∂_j β_k − ∂_j γ_kl βˡ)
"""
@inline function d_shift(m::Metric3{T}, dm::MetricGradient{T}) where {T}
    cols = ntuple(Val(3)) do j
        @inbounds mul3(m.γu, d_shift_lower(dm, j) - mul3(d_spatial(dm, j), m.β))
    end
    # Row j is ∂_j β; `hcat` of the columns gives column j, so transpose.
    return transpose(hcat(cols...))
end
