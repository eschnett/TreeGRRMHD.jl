# The interior of a black hole (step 9; "The interior" in `CODE.md`): inside
# a sphere well within the horizon the state is replaced by the analytic
# solution, at the points of each set — the cell averages of the matter, the
# point values of the potentials — by the limiters, so that the scheme never
# evolves the region around the singularity, and the cells just outside it read
# the analytic solution there.
#
# **Position-only** (proposed in step 9, as TreeGH's interior is): whether a
# point is replaced depends on its position alone — the distance of the cell
# centre, the edge's midpoint or the vertex from the hole's centre — never on
# the state, so the replaced set is the same at every stage, in every step and
# at every thread count, and a run with the interior is a pure function of its
# start as a run without one is. Inside the horizon every characteristic points
# inward, so in the continuum nothing outside depends on the inside; on the
# mesh the stencils of the cells just outside reach in, and find the analytic
# solution — the best data an outflow boundary can have.
#
# **A hard paste, in the limiters** (proposed in step 9). TreeGH relaxes its
# interior inside the right-hand side, `∂ₜu = w F(u) − ρ (u − u_exact)`, with
# smooth profiles; it is a finite-difference code, and its own measurement
# found the paste's kink at the surface worse than a smooth layer there. Here
# the paste is TreeGH's `ρ → ∞` limit on a step profile, applied where this
# package already applies its pointwise repairs: in the reset, which
# IMEXRungeKutta calls as the stage limiter before every explicit evaluation and
# as the step limiter on every step's result ("Floors, the atmosphere and the
# magnetosphere"). It runs first in the reset, before the reset's own fill and
# recovery, which then see the pasted state. `HoleInterior`'s `stage` and
# `step` say in which of the two roles; "The interior" in `CODE.md` has what
# each costs.

"""
    HoleInterior(r_in; center = (0, 0, 0), stage = true, step = true)

The interior of a black hole as a case states it (step 9): the ball of
Euclidean radius `r_in` about `center`, inside the horizon, where the state is
replaced by the case's own solution — at a cell centre, the cell average of the
conserved state ([`exact_state`](@ref)'s Gauss average); at an edge or a
vertex, the potential — by the reset as the stage limiter (`stage`) and as the
step limiter (`step`). Position-only and `isbits`. An [`MHDCase`](@ref)'s
`interior`; see `src/interior.jl` and "The interior" in `CODE.md`.
"""
struct HoleInterior{T}
    r_in::T
    center::SVector{3,T}
    stage::Bool
    step::Bool
end

function HoleInterior(r_in::T; center=nothing, stage::Bool=true, step::Bool=true) where {T<:Real}
    r_in > 0 || throw(ArgumentError("the interior's radius must be positive, got $r_in."))
    (stage || step) || throw(ArgumentError(
        "an interior applied in neither limiter does nothing; pass `interior = nothing` " *
        "to the case or to `evolve!` instead."))
    c = center === nothing ? zero(SVector{3,T}) : SVector{3,T}(center)
    return HoleInterior{T}(r_in, c, stage, step)
end

"""
    InteriorTarget(hole::HoleInterior, case::MHDCase)

What the interior's kernels need, `isbits`: the ball and the case's solution —
its `fields`, `potential`, EOS and metric provider. [`MHDProblem`](@ref)'s
`interior`; [`evolve!`](@ref) builds it from the case.
"""
struct InteriorTarget{T,FL,PT,EOS,MET}
    hole::HoleInterior{T}
    fields::FL
    potential::PT
    eos::EOS
    provider::MET
end

InteriorTarget(hole::HoleInterior, case) =
    InteriorTarget(hole, case.fields, case.potential, case.eos, case.provider)

interior_target(::Nothing, case) = nothing
interior_target(hole::HoleInterior, case) = InteriorTarget(hole, case)

# Whether the point `x` is inside the ball.
@inline function in_hole(h::HoleInterior{T}, x::SVector{3,T}) where {T}
    d = x - h.center
    return d[1]^2 + d[2]^2 + d[3]^2 < h.r_in^2
end

@kernel function interior_cells_kernel!(uU, fields, eos, provider, hole, t, @Const(origins),
                                        @Const(spacings), ::Val{D}) where {D}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    xc = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    if in_hole(hole, xc)
        acc = cell_average(fields, eos, provider, xc, oftype(h, t), h, Val(D))
        @inbounds for v in 1:NU
            uU[n..., v, b] = acc[v]
        end
    end
end

@kernel function interior_points_kernel!(uA, potential, hole, t, @Const(origins),
                                         @Const(spacings), ::Val{D}, ::Val{C},
                                         ::Val{j}) where {D,C,j}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(C))
    if in_hole(hole, x)
        A, Φ̃ = potential(x, oftype(h, t))
        @inbounds uA[n..., 1, b] = j ≤ 3 ? A[j] : Φ̃
    end
end

"""
    paste_interior!(u, p::MHDProblem, t)

Replace the owned points of the state vector `u` inside the problem's interior
by the case's solution at time `t` — the matter's cell averages, the
potentials' point values — and touch nothing else ([`HoleInterior`](@ref)).
A `map_blocks!` kernel per set, on every backend.
"""
function paste_interior!(u, p::MHDProblem{T,D}, t) where {T,D}
    it = p.interior
    sets = p.sets
    map_blocks!(interior_cells_kernel!, sets[SEG_U], statearray(u, sets, SEG_U), it.fields, it.eos,
                it.provider, it.hole, T(t), p.origins, p.spacings, Val(D))
    for i in SEG_A:SEG_Φ
        s = sets[i]
        map_blocks!(interior_points_kernel!, s, statearray(u, sets, i), it.potential, it.hole,
                    T(t), p.origins, p.spacings, Val(D), Val(staggers(s)), Val(i - SEG_A + 1))
    end
    return u
end

# The reset's first act, in the role `step` (`Val(true)`: the step limiter,
# `Val(false)`: the stage limiter): the paste, where the problem has an
# interior that applies in that role. Nothing at all without an interior, so
# that every problem without one runs the code it ran before step 9.
@inline paste_in_role!(u, p, ::Nothing, t, ::Val) = nothing
function paste_in_role!(u, p, it::InteriorTarget, t, ::Val{STEP}) where {STEP}
    (STEP ? it.hole.step : it.hole.stage) && paste_interior!(u, p, t)
    return nothing
end

# --- diagnostics outside the hole ---------------------------------------------------

"""
    outside_l1(p::MHDProblem, d, r_min) -> (U = NTuple{10}, A = NTuple{3}, Φ)

The volume-weighted mean of `|d|` per variable of every evolved set of the
state vector `d` (a difference of two states), over the owned points at a
Euclidean distance `≥ r_min` from the origin: [`l1_errors`](@ref) with the
inside of a hole left out, where the interior holds the solution and the
singularity's neighbourhood would dominate a norm of anything else (a
right-hand side, say). On the host; a diagnostic of step 9.
"""
function outside_l1(p::MHDProblem{T,D}, d, r_min) where {T,D}
    sets = p.sets
    o, hs = Array(p.origins), Array(p.spacings)
    one_set(i) = begin
        s = sets[i]
        a = Array(statearray(d, sets, i))
        N, C = s.forest.N, staggers(s)
        nv = size(a, D + 1)
        acc, vol = zeros(T, nv), zero(T)
        for b in axes(a, D + 2), I in CartesianIndices(ntuple(_ -> N, Val(D)))
            n = Tuple(I)
            x = point_x(o[b], hs[b], n, Val(C))
            sum(abs2, x) ≥ T(r_min)^2 || continue
            w = hs[b]^D
            vol += w
            for v in 1:nv
                acc[v] += w * abs(a[n..., v, b])
            end
        end
        acc ./ vol
    end
    U = one_set(SEG_U)
    return (; U=ntuple(v -> U[v], Val(NU)), A=ntuple(j -> one_set(SEG_A + j - 1)[1], 3),
            Φ=one_set(SEG_Φ)[1])
end

"""
    field_l1(p::MHDProblem, u, uref, t, r_min) -> NTuple{3}

The mean `|𝔅ᵈ − 𝔅ᵈ_ref|` over the owned faces normal to each `d` at a
distance `≥ r_min` from the origin, `𝔅` the curl of each state's potential
plus `𝔅̄` ([`update_fields!`](@ref)): the field's error, which, unlike `A`'s,
no gauge drift enters. Overwrites the scratch sets `A` and `B`. On the host.
"""
function field_l1(p::MHDProblem{T,D}, u, uref, t, r_min) where {T,D}
    f = p.f
    update_fields!(p, uref, t)
    ref = map(B -> Array(B.work), f.B)
    update_fields!(p, u, t)
    o, hs = Array(p.origins), Array(p.spacings)
    return ntuple(Val(3)) do d
        s = f.B[d]
        a = Array(s.work)
        N, C, G = s.forest.N, staggers(s), s.G
        acc, vol = zero(T), zero(T)
        for b in axes(a, D + 2), I in CartesianIndices(ntuple(_ -> N, Val(D)))
            n = Tuple(I)
            x = point_x(o[b], hs[b], n, Val(C))
            sum(abs2, x) ≥ T(r_min)^2 || continue
            k = stored(n, G)
            acc += hs[b]^D * abs(a[k..., 1, b] - ref[d][k..., 1, b])
            vol += hs[b]^D
        end
        acc / vol
    end
end

"""
    charge_norms(p::MHDProblem, u, t, r_min) -> (; l1, max, total)

The charge `√γ q = ∂ᵢ𝔈ⁱ` the right-hand side derives from `u` at time `t`
([`update_primitives!`](@ref); "Maxwell" in `CODE.md`) over the owned cells at
a distance `≥ r_min` from the origin: its volume mean `|√γ q|`, its largest
`|√γ q|`, and its volume integral. Zero for an exact vacuum or ideal field; on
the mesh, the divergence of the discrete `𝔈`. On the host.
"""
function charge_norms(p::MHDProblem{T,D}, u, t, r_min) where {T,D}
    P = update_primitives!(p, u, t)
    a = Array(P.work)
    o, hs = Array(p.origins), Array(p.spacings)
    N, G = P.forest.N, P.G
    l1, mx, tot, vol = zero(T), zero(T), zero(T), zero(T)
    for b in axes(a, D + 2), I in CartesianIndices(ntuple(_ -> N, Val(D)))
        n = Tuple(I)
        x = point_x(o[b], hs[b], n, Val(cell_staggers(Val(D))))
        sum(abs2, x) ≥ T(r_min)^2 || continue
        k = stored(n, G)
        q = a[k..., P_Q, b] * a[k..., P_SQRTG, b]
        w = hs[b]^D
        l1 += w * abs(q)
        mx = max(mx, abs(q))
        tot += w * q
        vol += w
    end
    return (; l1=l1 / vol, max=mx, total=tot)
end
