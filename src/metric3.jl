# The 3+1 metric at one point, and the handful of index operations the
# matter and Maxwell equations need on it.
#
# Every pointwise function of this package — the conversions, the fluxes,
# the sources, the implicit stage solve — takes the metric as one `isbits`
# value rather than as twelve scalars, so that a kernel reads it once from
# the primitive set (or from a provider) and hands it on. The inverse and
# `√γ` are computed once, here, where the metric is built; a kernel that
# needs `γ^{ij}` twice does not invert twice.
#
# `[ijk]` is the Levi-Civita *symbol* and `ε_ijk = √γ [ijk]` the tensor
# ("Conventions" in `CODE.md`). The cross products below are written with
# the tensor, so that `ε_ijk Eʲ Bᵏ` — the electromagnetic momentum — and
# `−ε_ijk vʲ Bᵏ` — the ideal electric field — are what they say in any
# coordinates, and reduce to the flat-space cross product only where
# `√γ = 1`.

"""
    Metric3{T}

The 3+1 split of the spacetime metric at one point: the lapse `α`, the
contravariant shift `βⁱ`, the spatial metric `γ_ij`, its inverse `γ^ij` and
`√γ = √det γ_ij`.

Build it with `Metric3(α, β, γ)`, which computes the inverse and `√γ`; the
inner constructor stores what it is given and is for code that already holds
all five (the primitive set's cached `√γ`, step 5). `isbits`, so that it is
a kernel argument and a value inside one. See "Conventions" in `CODE.md`.
"""
struct Metric3{T}
    α::T
    β::SVector{3,T}
    γ::SMatrix{3,3,T,9}
    γu::SMatrix{3,3,T,9}
    sqrtγ::T
end

"""
    Metric3(α, β, γ)

The point metric from the lapse, the contravariant shift and the spatial
metric `γ_ij`, with `γ^ij` and `√γ` computed here.

The inverse is the cofactor formula, written out from the upper triangle of
`γ_ij` so that `γ^ij` is exactly symmetric whatever rounding the lower
triangle carries; it has no branch and no pivoting, which a positive-definite
3-metric does not need, and so it is the same arithmetic on every backend.
"""
@inline function Metric3(α::T, β::SVector{3,T}, γ::SMatrix{3,3,T}) where {T<:Real}
    gxx, gxy, gxz = γ[1, 1], γ[1, 2], γ[1, 3]
    gyy, gyz, gzz = γ[2, 2], γ[2, 3], γ[3, 3]
    # Cofactors of the symmetric matrix.
    cxx = gyy * gzz - gyz * gyz
    cxy = gxz * gyz - gxy * gzz
    cxz = gxy * gyz - gxz * gyy
    cyy = gxx * gzz - gxz * gxz
    cyz = gxy * gxz - gxx * gyz
    czz = gxx * gyy - gxy * gxy
    detγ = gxx * cxx + gxy * cxy + gxz * cxz
    idet = inv(detγ)
    γsym = SMatrix{3,3,T,9}(gxx, gxy, gxz, gxy, gyy, gyz, gxz, gyz, gzz)
    γu = SMatrix{3,3,T,9}(cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz, czz) * idet
    return Metric3{T}(α, β, γsym, γu, sqrt(detγ))
end

Metric3(α::Real, β::AbstractVector, γ::AbstractMatrix) =
    Metric3(promote_metric(α, β, γ)...)

function promote_metric(α, β, γ)
    T = float(promote_type(typeof(α), eltype(β), eltype(γ)))
    return T(α), SVector{3,T}(β), SMatrix{3,3,T}(γ)
end

"""
    flat_metric(T=Float64)

Minkowski space in Cartesian coordinates: `α = 1`, `β = 0`, `γ_ij = δ_ij`.
"""
@inline function flat_metric(::Type{T}=Float64) where {T}
    I3 = SMatrix{3,3,T,9}(one(T), zero(T), zero(T), zero(T), one(T), zero(T), zero(T),
                          zero(T), one(T))
    return Metric3{T}(one(T), zero(SVector{3,T}), I3, I3, one(T))
end

"""
    lower(m::Metric3, v)

`v_i = γ_ij vʲ`.
"""
@inline lower(m::Metric3, v::SVector{3}) = m.γ * v

"""
    raise(m::Metric3, w)

`wⁱ = γ^ij w_j`.
"""
@inline raise(m::Metric3, w::SVector{3}) = m.γu * w

"""
    dot_up(m::Metric3, a, b)

`γ_ij aⁱ bʲ`, for two contravariant vectors.
"""
@inline dot_up(m::Metric3, a::SVector{3}, b::SVector{3}) = sum(a .* (m.γ * b))

"""
    dot_down(m::Metric3, a, b)

`γ^ij a_i b_j`, for two covariant vectors.
"""
@inline dot_down(m::Metric3, a::SVector{3}, b::SVector{3}) = sum(a .* (m.γu * b))

# The cross product of the symbol, `[ijk] aʲ bᵏ`, which is a component list
# and not yet a tensor.
@inline symbol_cross(a::SVector{3}, b::SVector{3}) =
    SVector(a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])

"""
    cross_down(m::Metric3, a, b)

`ε_ijk aʲ bᵏ = √γ [ijk] aʲ bᵏ`: the covariant cross product of two
contravariant vectors. The electromagnetic momentum `ε_ijk Eʲ Bᵏ` is this of
`E` and `B`.
"""
@inline cross_down(m::Metric3, a::SVector{3}, b::SVector{3}) = m.sqrtγ * symbol_cross(a, b)

"""
    cross_up(m::Metric3, a, b)

`ε^{ijk} a_j b_k = [ijk] a_j b_k / √γ`: the contravariant cross product of
two covariant vectors.
"""
@inline cross_up(m::Metric3, a::SVector{3}, b::SVector{3}) = symbol_cross(a, b) / m.sqrtγ

"""
    ideal_E(m::Metric3, v, B)

The ideal-MHD electric field `E = −v × B`, contravariant: `Eⁱ = −γ^ij ε_jkl vᵏ Bˡ`,
from the contravariant fluid velocity `vⁱ` and field `Bⁱ`, both measured by
the normal observer.

It is what the `IdealConductor` limit of Ohm's law sets `E` to, and what makes
[`con2prim_ideal`](@ref) the exact recovery of a state built with it. See
"Ohm's law and the conductivity" in `CODE.md`.
"""
@inline ideal_E(m::Metric3, v::SVector{3}, B::SVector{3}) = -raise(m, cross_down(m, v, B))
