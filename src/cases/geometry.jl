# Cases made from other cases by a change of geometry (step 7): a
# one-dimensional case extruded into two or three dimensions, and a case
# turned so that its `x` and `y` axes trade places. They exist for the
# direction-independence claims of rung 2a ("Validation ladder" in `CODE.md`):
# a run that is translation-invariant along `y` is the 1D run, bit for bit,
# and a run along `y` is the run along `x`, transposed.
#
# **The turn is a rotation, not a reflection.** Swapping `x` and `y` alone
# reverses the handedness, and `B = curl A` is a pseudovector; the map used
# here is `R(x, y, z) = (y, x, −z)`, the rotation by π about the diagonal
# `(1, 1, 0)/√2`. It is its own inverse and has determinant `+1`, so every
# vector the case holds — `v`, `E`, `B`, `A` — transforms as `R`, and every
# scalar — `ρ`, `s`, `Yₑ`, `Φ̃` — is unchanged. Its entries are `0` and `±1`,
# so the turned case evaluates the original at exactly the turned point and
# returns exactly the turned values: nothing is rounded.

# `R(x, y, z) = (y, x, −z)` on a 3-vector.
@inline turn_xy(v::SVector{3}) = @inbounds SVector(v[2], v[1], -v[3])

"""
    TurnedFields(fields)
    TurnedPotential(potential)

A case's `fields` and `potential` callbacks seen through the rotation
`R(x, y, z) = (y, x, −z)` ([`turned_case`](@ref)): the original evaluated at
`R x`, with every vector turned by `R`. `isbits` whenever the originals are.
"""
struct TurnedFields{F}
    f::F
end
struct TurnedPotential{F}
    f::F
end
@inline function (t::TurnedFields)(x::SVector{3}, τ)
    ρ, v, s, Yₑ, E, B = t.f(turn_xy(x), τ)
    return ρ, turn_xy(v), s, Yₑ, turn_xy(E), turn_xy(B)
end
@inline function (t::TurnedPotential)(x::SVector{3}, τ)
    A, Φ̃ = t.f(turn_xy(x), τ)
    return turn_xy(A), Φ̃
end

"""
    turned_case(case::MHDCase{T,D}) -> MHDCase{T,D}

`case` rotated by `R(x, y, z) = (y, x, −z)`, for `D ≥ 2`: the box's `x` and `y`
extents, periodicity and roots trade places, the background `𝔅̄` becomes
`R 𝔅̄`, and the solution is the original's at the turned point, turned (see
`src/cases/geometry.jl`). A run of the turned case is the original run with
its `x` and `y` axes exchanged and its `z` components negated — bit for bit,
when the scheme is (rung 2a, step 7).
"""
function turned_case(c::MHDCase{T,D}) where {T,D}
    D ≥ 2 || throw(ArgumentError(
        "turning x into y needs a y axis: the case is $(D)-dimensional."))
    sw(t) = ntuple(d -> d == 1 ? t[2] : d == 2 ? t[1] : t[d], Val(D))
    return MHDCase(T, Val(D); fields=TurnedFields(c.fields),
                   potential=TurnedPotential(c.potential), eos=c.eos, opts=c.opts, pol=c.pol,
                   model=c.model, provider=c.provider, 𝔅̄=turn_xy(c.𝔅̄),
                   periodic=sw(c.periodic), extents=sw(c.extents), roots=sw(c.roots))
end

"""
    extruded_case(case::MHDCase{T,1}, Val(D); N, roots = 1) -> MHDCase{T,D}

The one-dimensional `case` as a `D`-dimensional one that is translation-
invariant along `y` (and `z`): the same solution, ignoring the coordinates the
1D case never reads, on a box that is periodic along the new directions, with
`roots` root blocks of `N` cells each there, of the spacing the 1D case has
at `N` cells per block — so that the cells are square and a run is the 1D run
repeated along every new direction (rung 2a, step 7).
"""
function extruded_case(c::MHDCase{T,1}, ::Val{D}; N::Integer, roots::Integer=1) where {T,D}
    2 ≤ D ≤ 3 || throw(ArgumentError("extrude into two or three dimensions, not $D."))
    h = (c.extents[1][2] - c.extents[1][1]) / (c.roots[1] * N)
    ext = ntuple(d -> d == 1 ? c.extents[1] : (zero(T), T(roots * N) * h), Val(D))
    return MHDCase(T, Val(D); fields=c.fields, potential=c.potential, eos=c.eos, opts=c.opts,
                   pol=c.pol, model=c.model, provider=c.provider, 𝔅̄=c.𝔅̄,
                   periodic=ntuple(d -> d == 1 ? c.periodic[1] : true, Val(D)), extents=ext,
                   roots=ntuple(d -> d == 1 ? c.roots[1] : roots, Val(D)))
end

"""
    ObliqueFields(fields, n)
    ObliquePotential(potential, n)

A one-dimensional case's `fields` and `potential` turned so that its `x`
axis points along the unit vector `n = (n₁, n₂, 0)`: evaluated at the
coordinate `n·x` along it, with every vector `w` of the 1D solution turned to
`w₁ n + w₂ n⊥ + w₃ ẑ`, `n⊥ = (−n₂, n₁, 0)` ([`oblique_case`](@ref)).
"""
struct ObliqueFields{F,T}
    f::F
    n::SVector{3,T}
end
struct ObliquePotential{F,T}
    f::F
    n::SVector{3,T}
end
@inline oblique_turn(n::SVector{3,T}, w::SVector{3,T}) where {T} =
    SVector(w[1] * n[1] - w[2] * n[2], w[1] * n[2] + w[2] * n[1], w[3])
@inline function (o::ObliqueFields)(x::SVector{3,T}, t) where {T}
    ρ, v, s, Yₑ, E, B = o.f(SVector(o.n[1] * x[1] + o.n[2] * x[2], zero(T), zero(T)), t)
    return ρ, oblique_turn(o.n, v), s, Yₑ, oblique_turn(o.n, E), oblique_turn(o.n, B)
end
@inline function (o::ObliquePotential)(x::SVector{3,T}, t) where {T}
    A, Φ̃ = o.f(SVector(o.n[1] * x[1] + o.n[2] * x[2], zero(T), zero(T)), t)
    return oblique_turn(o.n, A), Φ̃
end

"""
    oblique_case(case::MHDCase{T,1}; direction = (1, 1), N, roots = 1) -> MHDCase{T,2}

The periodic one-dimensional `case`, of length `L`, as a two-dimensional
periodic case whose `x` axis points along `n = (a, b)/√(a² + b²)` for the
positive integers `direction = (a, b)`: the box `[0, L/n₁] × [0, L/n₂]` holds
exactly one period along each axis, with `roots·b` and `roots·a` root blocks
of `N` cells (square cells). Unlike [`extruded_case`](@ref) and
[`turned_case`](@ref) the solution is not exactly the 1D one sampled — `n` is
irrational in general — but it is the same exact solution, so an oblique run
measures the scheme's order across both directions at once: the curl, the
EMF along `z` with both transverse reconstructions, and the fluxes in both
directions (rung 2a, step 7). The background `𝔅̄` and the guide field turn
with the rest.
"""
function oblique_case(c::MHDCase{T,1}; direction=(1, 1), N::Integer, roots::Integer=1) where {T}
    all(c.periodic) || throw(ArgumentError("an oblique case needs a periodic 1D case."))
    a, b = direction
    (a > 0 && b > 0) || throw(ArgumentError("direction must be two positive integers, got $direction."))
    L = c.extents[1][2] - c.extents[1][1]
    nn = sqrt(T(a)^2 + T(b)^2)
    n = SVector(T(a) / nn, T(b) / nn, zero(T))
    return MHDCase(T, Val(2); fields=ObliqueFields(c.fields, n),
                   potential=ObliquePotential(c.potential, n), eos=c.eos, opts=c.opts,
                   pol=c.pol, model=c.model, provider=c.provider,
                   𝔅̄=oblique_turn(n, c.𝔅̄), periodic=(true, true),
                   extents=((zero(T), L * nn / a), (zero(T), L * nn / b)),
                   roots=(roots * b, roots * a))
end
