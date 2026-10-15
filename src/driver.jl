# The driver: a case as data, its initial data on the mesh, and one chunked
# evolution on a uniform mesh.
#
# TreeHydro's discipline, reduced to what step 5 needs: **a case is data** —
# the pointwise solution, the EOS and its recovery options, the
# conductivity, the metric, the background field and the box — and nothing in
# this file knows what an Alfvén wave is. The run is cut into chunks, each a
# fresh integrator (reusing the previous one's scratch) with its own `Δt` from
# the light-speed CFL measured at the chunk's start and its own tally of the
# implicit stages; between chunks is where a regrid will go (step 8).
#
# Initial data and references are **exact**: the matter set holds cell
# averages, so it is filled by a tensor-product three-point Gauss quadrature of
# the conserved state over each cell (exact to `O(h⁶)`), and the potentials
# hold point values, so they are sampled. With `A` exact at the edges, `𝔅` from
# the curl is the exact face average of the field — in 1D the exact cell
# average — so the scheme starts from the solution and nothing else.

"""
    MHDCase(T, Val(D); fields, potential, eos, opts = Con2PrimOptions{T}(), pol,
            model, provider = Flat(), 𝔅̄ = 0, periodic, extents, roots)

A problem, as data:

- `fields(x, t) -> (ρ, v, s, Yₑ, E, B)` — the solution at the point
  `x::SVector{3,T}` (zero along the missing directions) and time `t`: the
  rest-mass density, the contravariant velocity, the entropy, the electron
  fraction, and the contravariant fields `Eⁱ`, `Bⁱ` the normal observer sees;
- `potential(x, t) -> (A, Φ̃)` — the covariant vector potential and the gauge
  scalar there, with `[ijk] ∂ⱼA_k + 𝔅̄ⁱ = √γ Bⁱ`;
- `eos`, `opts`, `pol`: EntropyEOS's EOS, `Con2PrimOptions` and
  `PolicyOptions`; `model`: the conductivity; `provider`: the metric;
- `𝔅̄`: the constant densitized background field (`src/curl.jl`);
- `periodic`, `extents`, `roots`: the box, per dimension.

Both callbacks become kernel arguments, so they must capture `isbits` values
only. Where `fields` is an exact solution at every `t`, it is also the
reference ([`exact_state`](@ref)). A case that is not periodic in every
dimension is given Dirichlet boundaries from the same two callbacks
([`case_boundary`](@ref), [`DirichletBoundary`](@ref); step 6), so there
they must also be the solution outside the box for as long as the run lasts.
"""
struct MHDCase{T,D,FL,PT,EOS,OPT,POL,MOD,MET}
    fields::FL
    potential::PT
    eos::EOS
    opts::OPT
    pol::POL
    model::MOD
    provider::MET
    𝔅̄::SVector{3,T}
    periodic::NTuple{D,Bool}
    extents::NTuple{D,Tuple{T,T}}
    roots::NTuple{D,Int}
end

function MHDCase(::Type{T}, ::Val{D}; fields, potential, eos, opts=nothing, pol,
                 model::Conductivity, provider::MetricProvider=Flat(), 𝔅̄=nothing,
                 periodic, extents, roots) where {T,D}
    per = periodic isa Tuple ? ntuple(d -> Bool(periodic[d]), D) : ntuple(_ -> Bool(periodic), D)
    length(extents) == D || throw(ArgumentError(
        "a case needs one extent per dimension, got $(length(extents)) for D = $D."))
    ext = ntuple(d -> (T(extents[d][1]), T(extents[d][2])), D)
    all(d -> ext[d][2] > ext[d][1], 1:D) || throw(ArgumentError(
        "each extent must be nonempty and increasing, got $ext."))
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), D) : ntuple(_ -> Int(roots), D)
    all(>(0), rs) || throw(ArgumentError("root counts must be positive, got $rs."))
    o = opts === nothing ? EntropyEOS.Con2PrimOptions{T}() : opts
    B̄ = 𝔅̄ === nothing ? zero(SVector{3,T}) : SVector{3,T}(𝔅̄)
    return MHDCase{T,D,typeof(fields),typeof(potential),typeof(eos),typeof(o),typeof(pol),
                   typeof(model),typeof(provider)}(fields, potential, eos, o, pol, model,
                                                   provider, B̄, per, ext, rs)
end

# --- initial data and references ----------------------------------------------------

# The three-point Gauss–Legendre rule on a cell of width `h` centered at 0:
# nodes `−h√(3/5)/2`, `0`, `h√(3/5)/2`, weights `5/18`, `8/18`, `5/18`.
@inline gauss3_node(h, q::Int) = q == 2 ? zero(h) : (q == 1 ? -one(h) : one(h)) * sqrt(oftype(h, 3) / 5) * h / 2

# The rule's average of three samples, written as the middle sample plus a
# correction in the differences (amended in step 7):
#
#     (5f₋ + 8f₀ + 5f₊)/18 = f₀ + 5((f₋ − f₀) + (f₊ − f₀))/18,
#
# so that three equal samples average to themselves bit for bit. A
# tensor-product rule with the weights multiplied out (`Σ w_q f_q`) rounds a
# constant to a neighbouring float, and a run that is translation-invariant
# along `y` did not start from the 1D run's data (2.7e−15 apart after seven
# steps, measured in step 5). Applied one dimension at a time, the average
# along a direction the solution does not vary in is the identity, and is
# symmetric under reflection (`f₋ ↔ f₊`).
@inline gauss3_average(f₋, f₀, f₊) =
    ntuple(v -> @inbounds(f₀[v] + 5 * ((f₋[v] - f₀[v]) + (f₊[v] - f₀[v])) / 18), Val(length(f₀)))

# The conserved state of the case at the point `x` and time `t`.
@inline function case_cons(fields, eos, provider, x::SVector{3,T}, t::T) where {T}
    m = point_metric(provider, t, x)
    ρ, v, s, Yₑ, E, B = fields(x, t)
    prim = primitive_state(eos, m, ρ, v, s, Yₑ)
    return prim2con(eos, m, prim, E, B)
end

# The Gauss average over the directions `e ≤ E` of the cell centred at `x`,
# one direction at a time, innermost `e = 1`: three averages over the
# directions below `E` at the nodes along `E`, then their average.
@inline cell_average(fields, eos, provider, x::SVector{3}, t, h, ::Val{0}) =
    case_cons(fields, eos, provider, x, t)
@inline function cell_average(fields, eos, provider, x::SVector{3,T}, t, h,
                              ::Val{E}) where {T,E}
    node(q) = x + SVector(ntuple(e -> e == E ? gauss3_node(h, q) : zero(T), Val(3)))
    f₋ = cell_average(fields, eos, provider, node(1), t, h, Val(E - 1))
    f₀ = cell_average(fields, eos, provider, x, t, h, Val(E - 1))
    f₊ = cell_average(fields, eos, provider, node(3), t, h, Val(E - 1))
    return gauss3_average(f₋, f₀, f₊)
end

@kernel function cell_average_kernel!(uU, fields, eos, provider, t, @Const(origins),
                                      @Const(spacings), ::Val{D}) where {D}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    xc = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    acc = cell_average(fields, eos, provider, xc, oftype(h, t), h, Val(D))
    @inbounds for v in 1:NU
        uU[n..., v, b] = acc[v]
    end
end

@kernel function potential_kernel!(uA, potential, t, @Const(origins), @Const(spacings),
                                   ::Val{D}, ::Val{C}, ::Val{j}) where {D,C,j}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(C))
    A, Φ̃ = potential(x, oftype(h, t))
    @inbounds uA[n..., 1, b] = j ≤ 3 ? A[j] : Φ̃
end

"""
    case_state!(u, p::MHDProblem, case::MHDCase, t) -> u

The case's solution at time `t` written into the state vector `u` over
`p.sets`: the exact cell averages of the conserved matter variables (Gauss,
three points per dimension), and the exact point values of `A_j` at the edges
and of `Φ̃` at the vertices.
"""
function case_state!(u, p::MHDProblem{T,D}, case::MHDCase, t) where {T,D}
    sets = p.sets
    map_blocks!(cell_average_kernel!, sets[SEG_U], statearray(u, sets, SEG_U), case.fields,
                case.eos, case.provider, T(t), p.origins, p.spacings, Val(D))
    for j in 1:3
        i = SEG_A + j - 1
        map_blocks!(potential_kernel!, sets[i], statearray(u, sets, i), case.potential, T(t),
                    p.origins, p.spacings, Val(D), Val(staggers(sets[i])), Val(j))
    end
    map_blocks!(potential_kernel!, sets[SEG_Φ], statearray(u, sets, SEG_Φ), case.potential,
                T(t), p.origins, p.spacings, Val(D), Val(staggers(sets[SEG_Φ])), Val(4))
    return u
end

"""
    exact_state(p::MHDProblem, case::MHDCase, t) -> u

A fresh state vector holding the case's solution at time `t`
([`case_state!`](@ref)): the initial data at `t = 0`, and the reference of an
exact solution at any `t`.
"""
exact_state(p::MHDProblem, case::MHDCase, t) = case_state!(statevector(p.sets), p, case, t)

# --- Dirichlet boundaries ----------------------------------------------------------------

"""
    DirichletBoundary{R}(fields, potential, eos, provider)

The case's own solution as the boundary of one evolved set (step 6;
"Boundaries" in `CODE.md`), at the time of each fill. `R` names the set:
`:U`, or `1`, `2`, `3` (`A_R`) or `4` (`Φ̃`). [`boundary_at`](@ref) turns it
into the hook TreeAMR's `fill_ghosts!` takes.

- **The matter**, `R = :U`: every ghost cell facing outside the domain holds
  the conserved state of `fields` at its center, through
  `CellBoundary(AllVariables(…))`, which runs as a kernel on every backend.
  Ghosts of `U` hold cell averages; the point value is the average where the
  solution is uniform (a shock tube's far states) and differs by `O(h²)` of
  the second derivative where it is smooth (the current sheet's tails).
- **The potentials**, `R = 1 … 4`: Dirichlet in the *field*, not in the
  potential (proposed in step 6). Each ghost point holds the nearest evolved
  point's value plus the case's difference between the two points,
  `A(x_g) = A(x_r) + [A_case(x_g, t) − A_case(x_r, t)]`, so that the curl —
  `𝔅` on the boundary faces and in the ghosts — is the case's, while the
  potential's level, which no physical quantity reads, is the evolved one's.
  A Dirichlet `A` itself is wrong at the first implicit stage of a tableau
  whose two abscissae differ: a uniform state's `A` grows as `−tE`, and the
  stage's own `A` is the explicit combination, at `tⁿ + c̃Δt`, while
  `solve_imp!` is told `tⁿ + cΔt` (SSP3(4,3,3)'s first stage has `c̃ = 0`,
  `c = 0.24`). Measured in step 6: a uniform flow at `vˣ = 0.999` through
  the box lost a fifth of its density in the last cell within 0.05 of a
  crossing that way. The difference is the field's, so it does not depend on
  `t` where the field does not. This is TreeAMR's region form of a hook,
  which reads the block's interior, and runs on the CPU only (AMR-7).

As in TreeHydro, a Dirichlet boundary is exact only until a wave arrives; a
case's `t_end` is chosen before that.
"""
struct DirichletBoundary{R,FL,PT,EOS,MET}
    fields::FL
    potential::PT
    eos::EOS
    provider::MET
end
DirichletBoundary{R}(fields, potential, eos, provider) where {R} =
    DirichletBoundary{R,typeof(fields),typeof(potential),typeof(eos),typeof(provider)}(
        fields, potential, eos, provider)

# The time-`t` callback of a `DirichletBoundary`, `isbits` whenever the hook
# is.
struct DirichletAt{H,T}
    hook::H
    t::T
end

@inline point3(x::NTuple{D,T}) where {D,T} = SVector(ntuple(e -> e ≤ D ? x[e] : zero(T), Val(3)))

# The matter: `g(x, δ) -> vals`, `x` the ghost cell's center.
@inline function (d::DirichletAt{<:DirichletBoundary{:U}})(x::NTuple{D,T}, δ) where {D,T}
    h = d.hook
    return case_cons(h.fields, h.eos, h.provider, point3(x), T(d.t))
end

# The case's potential `R` at the point `x`.
@inline function case_potential(h::DirichletBoundary{R}, x::NTuple{D,T}, t) where {R,D,T}
    A, Φ̃ = h.potential(point3(x), T(t))
    return R ≤ 3 ? A[R] : Φ̃
end

# A potential: TreeAMR's region form, `(fs, b, key, δ, region)`, which runs on
# the CPU only — the workaround for AMR-7 (`PLAN.md`), a per-cell form that
# also receives the nearest owned point. The nearest evolved point of a
# boundary point is its index clamped, along each dimension, to the evolved
# range: `G+1 … G+N`, except that on the high side of a vertex-like
# dimension the shared plane `G+N+1`, which nobody owns, is the hook's, and
# *(amended in step 8, AMR-8)* on the low side of a vertex-like dimension at
# an outer face the owned wall `G+1` is prescribed by the mirror of the same
# rule ([`low_walls!`](@ref)), so the evolved range starts at `G+2` there.
# A ghost beyond the low wall, and a ghost alongside it (in the wall's plane,
# beyond another face), is then made from the same evolved point as its
# mirror image on the high side, and a closed box is mirror-symmetric bit for
# bit.
function (d::DirichletAt{<:DirichletBoundary{R}})(fs::FieldSet{T,D}, b, key, δ,
                                                region) where {R,T,D}
    h = d.hook
    forest = fs.forest
    N = forest.N
    G = fs.G
    c = staggers(fs)
    lw = low_outer_faces(forest, T, block_origin(T, forest, key))
    lo = ntuple(e -> c[e] == 1 && lw[e] ? G[e] + 2 : G[e] + 1, Val(D))
    w = fs.work
    for I in region
        g = Tuple(I)
        r = ntuple(e -> clamp(g[e], lo[e], G[e] + N), Val(D))
        xg = coordinates(T, fs, b, g)
        xr = coordinates(T, fs, b, r)
        @inbounds w[g..., 1, b] = w[r..., 1, b] + (case_potential(h, xg, d.t) -
                                                   case_potential(h, xr, d.t))
    end
    return nothing
end

# --- the low walls (AMR-8's workaround) --------------------------------------------

"""
    low_outer_faces(forest, T, origin) -> NTuple{D,Bool}

Along each dimension, whether the block with lower corner `origin` lies on
the domain's low face there and that face is an outer one — neither
periodic, reflecting nor outflow — whose ghosts the application's hook
fills. From exported API only: the block's origin against the domain's
extent (`block_origin` reproduces the extent's low end exactly for a block
at the low face, `lo + 0 · w`), and the forest's `periodic`, `reflecting`
and `outflow` properties.
"""
function low_outer_faces(forest::Forest{D}, ::Type{T}, origin) where {D,T}
    per, refl, out = forest.periodic, forest.reflecting, forest.outflow
    return ntuple(e -> !per[e] && !refl[e][1] && !out[e][1] &&
                       origin[e] == T(forest.extents[e][1]), Val(D))
end

"""
    outer_faces(forest, T) -> Vector{NTuple{2,NTuple{D,Bool}}}

Per block this rank stores, `(low, high)`: along each dimension, whether
the block's low (high) face is an outer face of the domain
([`low_outer_faces`](@ref)). The high face from the block's position in
whole cells, `(origin − lo)/h + N` against the domain's `(hi − lo)/h`, both
integers, rather than from a sum of floats.
"""
function outer_faces(forest::Forest{D}, ::Type{T}) where {D,T}
    per, refl, out = forest.periodic, forest.reflecting, forest.outflow
    N = forest.N
    os = block_origins(forest, Float64)
    hs = block_spacings(forest, Float64)
    return map(eachindex(os)) do b
        o, h = os[b], hs[b]
        low = low_outer_faces(forest, T, block_origins(forest, T)[b])
        high = ntuple(Val(D)) do e
            lo, hi = forest.extents[e]
            !per[e] && !refl[e][2] && !out[e][2] &&
                round(Int, (o[e] - lo) / h) + N == round(Int, (hi - lo) / h)
        end
        (low, high)
    end
end

# The case's potential `R` at a point given as an `SVector{3}` (zero along
# the missing directions), as `case_potential` evaluates it at a point given
# as TreeAMR's coordinate tuple: the same callback on the same numbers.
@inline function case_potential(h::DirichletBoundary{R}, x::SVector{3,T}, t) where {R,T}
    A, Φ̃ = h.potential(x, T(t))
    return R ≤ 3 ? A[R] : Φ̃
end

# The owned low walls of one potential: at every owned point `n` of block `b`
# that lies on the wall of a vertex-like dimension `e` at a low outer face
# (`n[e] = 1`), the value at the nearest evolved point `r` — `n` with every
# such coordinate moved to `2` — plus the case's difference between the two
# points. That is the hook's rule for the high wall, `w[G+N+1] = w[G+N] +
# [A(x_{G+N+1}) − A(x_{G+N})]`, mirrored, with the positions formed by
# TreeAMR's `coordinates` arithmetic (`point_x`) and the same callback, so a
# box symmetric under a reflection gets the two walls bit for bit alike.
# The kernel reads only evolved points and writes only wall points, so there
# is no race. `GO` is the offset of the array's owned points: the set's `G`
# for its working array, zero for a state vector's segment.
@kernel function low_wall_kernel!(w, hook, t, @Const(origins), @Const(spacings),
                                  @Const(outer), ::Val{D}, ::Val{C},
                                  ::Val{GO}) where {D,C,GO}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    lw = @inbounds outer[b][1]
    wall = ntuple(e -> C[e] == 1 && lw[e] && n[e] == 1, Val(D))
    if any(wall)
        r = ntuple(e -> wall[e] ? 2 : n[e], Val(D))
        h = @inbounds spacings[b]
        o = @inbounds origins[b]
        xg = point_x(o, h, n, Val(C))
        xr = point_x(o, h, r, Val(C))
        @inbounds w[stored(n, GO)..., 1, b] = w[stored(r, GO)..., 1, b] +
                                             (case_potential(hook, xg, t) -
                                              case_potential(hook, xr, t))
    end
end

# Every stored point of one potential beyond an outer face of block `b` —
# the ghosts, the high wall plane of a vertex-like dimension, and the low
# wall `G+1` of one (AMR-8) — from the nearest evolved point plus the case's
# difference, after the ghost fill. The evolved range along a dimension is
# `G+1 … G+N`, starting at `G+2` along a vertex-like one at a low outer face;
# `r` is the point clamped into it along every dimension in which the point
# lies beyond an outer face, and left where it is along the others — where
# it may be a ghost across another block's face, which the fill has written
# by now. The hook itself, which runs before the prolongations and may read
# only the block's own owned points, clamps along every outward direction,
# and so takes a point where a wall meets another block (the wall's row in a
# corner region) from the block's own last column rather than from the
# neighbour's — a block-dependent value at a point two blocks share, which
# broke a closed box's mirror symmetry (found in step 8). Writes only points
# beyond an outer face, reads only points within the evolved range along
# every outer dimension: no race.
@kernel function outer_point_kernel!(w, hook, t, @Const(origins), @Const(spacings),
                                     @Const(outer), N, ::Val{D}, ::Val{C},
                                     ::Val{G}) where {D,C,G}
    I = @index(Global, NTuple)                       # stored: `stored = true`
    b = I[D + 1]
    g = ntuple(e -> I[e], Val(D))
    lw, hw = @inbounds outer[b]
    lo = ntuple(e -> G[e] + 1 + (C[e] == 1 && lw[e] ? 1 : 0), Val(D))
    hi = ntuple(e -> G[e] + N, Val(D))
    beyond = ntuple(e -> (lw[e] && g[e] < lo[e]) || (hw[e] && g[e] > hi[e]), Val(D))
    if any(beyond)
        r = ntuple(e -> beyond[e] ? clamp(g[e], lo[e], hi[e]) : g[e], Val(D))
        h = @inbounds spacings[b]
        o = @inbounds origins[b]
        n = ntuple(e -> g[e] - G[e], Val(D))
        m = ntuple(e -> r[e] - G[e], Val(D))
        xg = point_x(o, h, n, Val(C))
        xr = point_x(o, h, m, Val(C))
        @inbounds w[g..., 1, b] = w[r..., 1, b] + (case_potential(hook, xg, t) -
                                                   case_potential(hook, xr, t))
    end
end

"""
    low_walls!(p::MHDProblem, u, t)
    low_walls!(p::MHDProblem, i, w, Val(G), t)

*AMR-8's workaround (added in step 8; decided by Erik 2026-10-15).* Prescribe
the owned low walls of the potentials `A_j` and `Φ̃` at every outer face of a
case with Dirichlet hooks ([`DirichletBoundary`](@ref)), at time `t`, by the
mirror of the rule the hook fills the high walls with: the nearest evolved
point's value plus the case's difference between the two points. TreeAMR
fills the high wall plane `G+N+1` of a vertex-like dimension at an outer face
as a ghost and leaves the low one, `G+1`, owned and evolved — its
half-open ownership ("Centerings" in its `CODE.md`) — so without this `𝔅` on
a box's high boundary face is the case's and on its low one the evolution's,
and a closed box is not mirror-symmetric (1.3e−4 in a 16³ blast, measured in
step 7; "Boundaries" in `CODE.md`).

The first form writes the segments of the state vector `u`; the
[`reset_atmosphere!`](@ref) and [`reset_step!`](@ref) limiters call it, so
the prescribed walls persist in the state. The second writes the working
array `w` of evolved set `i` (owned points at offset `G`) before each of
its ghost fills, so every fill sees the walls at its own time. Periodic,
reflecting and outflow faces are not touched: they have no wall the hook
fills (reflecting and outflow walls are TreeAMR's own machinery, and
AMR-8 asks TreeAMR to fill all of them alike). A `map_blocks!` kernel,
so it runs on every backend; once AMR-8 lands it is deleted.
"""
function low_walls!(p::MHDProblem, u, t)
    p.hasouter || return u
    for i in SEG_A:SEG_Φ
        low_walls!(p, i, statearray(u, p.sets, i), Val(ntuple(_ -> 0, length(p.sets[i].G))), t)
    end
    return u
end

function low_walls!(p::MHDProblem{T,D}, i::Integer, w, ::Val{GO}, t) where {T,D,GO}
    p.hasouter || return nothing
    hook = p.boundary[i]
    hook isa DirichletBoundary || return nothing
    s = p.sets[i]
    map_blocks!(low_wall_kernel!, s, w, hook, T(t), p.origins, p.spacings, p.outer, Val(D),
                Val(staggers(s)), Val(GO))
    return nothing
end

"""
    outer_points!(p::MHDProblem, i, t)

After a ghost fill of potential `i` (`A_j` or `Φ̃`) with the case's
Dirichlet hook ([`DirichletBoundary`](@ref)), refill every point beyond an
outer face — the ghosts, the high wall, and the low wall (AMR-8) — from the
nearest evolved point plus the case's difference at time `t`, with the
nearest point taken across another block's face where the point lies in a
wall's plane alongside it (see `outer_point_kernel!`; added in step 8). The
hook's own values, which a prolongation inside the fill reads, stay what
they were only for that read. A `map_blocks!` kernel over the stored points.
"""
function outer_points!(p::MHDProblem{T,D}, i::Integer, t) where {T,D}
    p.hasouter || return nothing
    hook = p.boundary[i]
    hook isa DirichletBoundary || return nothing
    s = p.sets[i]
    map_blocks!(outer_point_kernel!, s, s.work, hook, T(t), p.origins, p.spacings, p.outer,
                s.forest.N, Val(D), Val(staggers(s)), Val(s.G); stored=true)
    return nothing
end

boundary_at(hook::DirichletBoundary{:U}, t) = CellBoundary(AllVariables(DirichletAt(hook, t)))
boundary_at(hook::DirichletBoundary, t) = DirichletAt(hook, t)

"""
    case_boundary(case::MHDCase) -> NTuple{5,DirichletBoundary} or nothing

`nothing` for a case periodic in every dimension, and otherwise one
[`DirichletBoundary`](@ref) per evolved set, in the order `(U, A₁, A₂, A₃,
Φ̃)` that [`MHDProblem`](@ref)'s `boundary` takes: the case's solution as the
boundary of every set, through each set's own hook (AMR-2).
"""
function case_boundary(case::MHDCase)
    all(case.periodic) && return nothing
    mk(R) = DirichletBoundary{R}(case.fields, case.potential, case.eos, case.provider)
    return (mk(:U), mk(1), mk(2), mk(3), mk(4))
end

"""
    l1_errors(p::MHDProblem, u, uref) -> (U = NTuple{10}, A = NTuple{3}, Φ = scalar)

The volume-weighted L1 norm of `u − uref`, per variable of every evolved
set, each normalized by the domain volume: the mean absolute error of each
of the ten cell variables, of each `A_j` and of `Φ̃`.
"""
function l1_errors(p::MHDProblem{T,D}, u, uref) where {T,D}
    diff = u .- uref
    forest = p.sets[SEG_U].forest
    vol = prod(e -> forest.extents[e][2] - forest.extents[e][1], 1:D)
    l1(i, v) = mesh_mapreduce(abs, +, zero(T), p.sets[i], stateview(diff, p.sets, i);
                              vars=v, weight=key -> spacing(T, forest, key)^D) / T(vol)
    return (; U=ntuple(v -> l1(SEG_U, v), Val(NU)), A=ntuple(j -> l1(SEG_A + j - 1, 1), 3),
            Φ=l1(SEG_Φ, 1))
end

"""
    profile_1d(p::MHDProblem{T,1}, u, t) -> NamedTuple

The state `u` of a one-dimensional run as host arrays ordered by `x`: the
cell centers `x`; the evolved cell variables `U` (`ncells × 10`); and,
from one recovery of `u` at `t` ([`update_primitives!`](@ref)), the
rest-mass density `ρ`, the gas pressure `p`, the velocity `v`, the fields
`E` and `B` (each `ncells × 3`, contravariant, `B` the cell average of the
curl of `A` plus `𝔅̄`), the comoving `b² = B²/W² + (B·v)²` and the policy
flags. What the shock-tube claims read (step 6).
"""
function profile_1d(p::MHDProblem{T,1}, u, t) where {T}
    P = hostcopy(update_primitives!(p, u, T(t)))
    U = p.sets[SEG_U]
    uU = Array(statearray(u, p.sets, SEG_U))
    N = U.forest.N
    G = P.G[1]
    nb = nblocks(U)
    x = T[]
    rows = Tuple{Int,Int}[]
    for b in 1:nb, n in 1:N
        push!(x, coordinates(T, U, b, (n + U.G[1],))[1])
        push!(rows, (n, b))
    end
    order = sortperm(x)
    x = x[order]
    rows = rows[order]
    nc = length(rows)
    Pv(n, b, v) = P.work[n + G, v, b]
    vec3(slot) = [Pv(n, b, slot + k - 1) for (n, b) in rows, k in 1:3]
    ρ = [Pv(n, b, P_RHO) for (n, b) in rows]
    pr = [Pv(n, b, P_PRESS) for (n, b) in rows]
    v, E, B = vec3(P_V), vec3(P_E), vec3(P_B)
    b2 = map(1:nc) do i
        # Flat or not, the comoving field from the cell's own metric.
        n, bk = rows[i]
        m = aux_metric(ntuple(k -> Pv(n, bk, k), Val(NP)))
        vv, BB = SVector{3,T}(v[i, :]), SVector{3,T}(B[i, :])
        W² = inv(one(T) - dot_up(m, vv, vv))
        dot_up(m, BB, BB) / W² + dot_up(m, BB, vv)^2
    end
    flags = [unsafe_trunc(UInt32, Pv(n, b, P_FLAGS)) for (n, b) in rows]
    Um = [uU[n, v, b] for (n, b) in rows, v in 1:NU]
    return (; x, U=Um, ρ, p=pr, v, E, B, b2, flags)
end

# --- the integrator and the run --------------------------------------------------------

"""
    mhd_integrator(p::MHDProblem, u, t0, t1, nsteps; tableau = IMEXSSP3433(),
                   alias_u0 = false, reuse = nothing)

An IMEXRungeKutta integrator for `nsteps` fixed steps from `t0` to `t1`:
[`mhd_rhs!`](@ref) as the explicit part, [`ohm_solve!`](@ref) as the stage
solve, and the reset as both the stage limiter ([`reset_atmosphere!`](@ref)) and
the step limiter ([`reset_step!`](@ref), which also accounts what it injects)
("The IMEX package" in `CODE.md`). The stage arithmetic is threaded evenly
over a CPU state (`partition = :even`; the block-owner partition is step
14's), and broadcast on a device. The step count is asserted, as in
TreeHydro.
"""
function mhd_integrator(p::MHDProblem{T}, u, t0, t1, nsteps::Integer;
                        tableau=IRK.IMEXSSP3433(), alias_u0::Bool=false,
                        reuse=nothing) where {T}
    nsteps ≥ 1 || throw(ArgumentError("nsteps must be at least 1, got $nsteps."))
    t0, t1 = T(t0), T(t1)
    prob = IRK.IMEXProblem(mhd_rhs!, ohm_solve!, u, (t0, t1), p)
    integ = IRK.init(prob, tableau; dt=(t1 - t0) / nsteps,
                     stage_limiter=reset_atmosphere!, step_limiter=reset_step!,
                     partition=u isa Array ? :even : nothing, alias_u0=alias_u0,
                     reuse=reuse)
    integ.nsteps == nsteps || throw(ErrorException(
        "IMEXRungeKutta derived $(integ.nsteps) steps from dt = (t1 − t0)/$nsteps; the " *
        "driver sized its step for $nsteps."))
    return integ
end

"""
    chunk_count(t_end, chunk)

`⌈t_end / chunk⌉`, except that a quotient within a few ulp of an integer `m`
gives `m`, so that a run meant to be a whole number of chunks (or of steps)
is not given one more by rounding. TreeHydro's, which is IMEXRungeKutta's
`step_count` rule.
"""
function chunk_count(t_end::T, chunk::T) where {T}
    r = t_end / chunk
    m = round(r)
    tol = 4 * (eps(r) + eps(t_end) / chunk)
    return m ≥ 1 && abs(r - m) ≤ tol ? roundint(m) : ceilint(r)
end

"""
    default_cfl(limiter) -> 2/5, or 1/5 for :mp5

The light-speed CFL factor [`evolve!`](@ref) uses unless told otherwise:
TreeHydro's 2/5 under SSP3(4,3,3)'s explicit SSPRK(3,3) part (proposed in step
5), and 1/5 for MP5, whose monotonicity-preserving bound holds for a forward
Euler step of CFL `≤ 1/(1 + α) = 1/5` at its `α = 4` (Suresh & Huynh 1997,
§2.4; proposed in step 7): at 2/5 it fires the reset on Balsara 3 and 4 and
grows Balsara 3's dense shell to `ρ ≈ 600` (measured in step 7).
"""
default_cfl(limiter) = limiter === :mp5 ? 1 // 5 : 2 // 5

"""
    evolve!([T], case::MHDCase, Val(D); N, t_end, chunk = t_end, cfl = default_cfl(limiter), limiter,
            riemann = :hlle, ξ = default_ξ(case), stage = nothing, tableau = IMEXSSP3433(), G = 3,
            backend = CPU(), after_step = nothing, roots = case.roots, ops_U, ops_A,
            fixup = true, refine = nothing, maxlevel = 0, refine_tol, coarsen_tol,
            buffer = 0, ε = 1//100, ε_g = 1//1000, maxpasses = 10,
            after_regrid = nothing) -> NamedTuple

Run `case` from `t = 0` to `t_end` on a mesh of `roots` root blocks (the
case's own by default) of `N` cells per dimension, in chunks of length
`chunk`. Each chunk's `Δt` is `cfl / (D · max λ/h)` from the light cone at
the chunk's start on the finest level ([`max_light_rate`](@ref)), rounded down
to a whole number of steps, and each chunk tallies its implicit stages and
resets ([`ohm_tally`](@ref)).

Returns `(; p, u, t, nsteps, chunks, nregrids, passes, nblocks, totals0)`:
the problem, the final state vector over `p.sets`, the final time, the total
step count, per chunk `(t, nsteps, dt, tally, nblocks)`, the number of
regrids that moved the mesh, the initial adaptation's passes (0 without
one), the final block count, and the [`conserved_totals`](@ref) of the
initial state on the initial mesh (which a regridded run's own initial
data on its final mesh does not reproduce to rounding).

`after_step(p, u, t)`, if given, is called after every step with the
problem, the state and its time — a monitor, such as the `∇·𝔅` check of
rung 2a ([`max_div_B`](@ref); added in step 7); it must not change `u`.
`after_regrid(p, u, t)` likewise after every regrid that moved the mesh and
once after the initial adaptation (added in step 8).

`limiter` has no default (see [`MHDProblem`](@ref)); `cfl = 2/5` is
TreeHydro's, under the explicit part of SSP3(4,3,3) — the SSPRK(3,3)
tableau, whose SSP coefficient is 1 (proposed in step 5) — and 1/5 under
MP5 ([`default_cfl`](@ref), step 7).

**The mesh** *(added in step 8; "Refinement" in `CODE.md`)*:

- `refine(forest)`, if given, refines the root mesh before anything is
  built — a static mesh, fixed for the run (`maxlevel = 0`), such as the two
  levels of the coarse-fine tests. It must leave the forest balanced.
- `maxlevel > 0` adapts the mesh: [`adapt_to_initial_data!`](@ref) over every
  field set on the case's own data (AMR-3), then a regrid at the start of
  every chunk after the first ([`regrid_state!`](@ref)), both with
  [`mhd_flags`](@ref) — Löhner on `ρ`, `|B|`, `|E|` — whose `refine_tol` and
  `coarsen_tol` have no default; `buffer` is the margin in cells `regrid!`
  dilates the firing cells by, the case's (feature speed × `chunk`), and
  `ε`, `ε_g` the indicator's floors.
- `ops_U`, `ops_A` are the operators of the matter and of the potentials
  ([`MHDProblem`](@ref)), `fixup = false` skips `restrict_interfaces!` (the
  negative control).
"""
function evolve!(::Type{T}, case::MHDCase{T,D}, ::Val{D}; N::Integer, t_end, chunk=t_end,
                 cfl=nothing, limiter=nothing, riemann::Symbol=:hlle, ξ=nothing,
                 stage=nothing, tableau=IRK.IMEXSSP3433(), G::Integer=3,
                 backend=CPU(), after_step=nothing, roots=case.roots,
                 ops_U::Operators=default_ops_U(), ops_A::Operators=default_ops_A(),
                 fixup::Bool=true, refine=nothing, maxlevel::Integer=0, refine_tol=nothing,
                 coarsen_tol=nothing, buffer::Integer=0, ε=T(1 // 100), ε_g=T(1 // 1000),
                 maxpasses::Integer=10, after_regrid=nothing) where {T,D}
    t_end, chunk = T(t_end), T(chunk)
    cfl = T(cfl === nothing ? default_cfl(limiter) : cfl)
    t_end > 0 || throw(ArgumentError("t_end must be positive, got $t_end."))
    chunk > 0 || throw(ArgumentError("chunk must be positive, got $chunk."))
    cfl > 0 || throw(ArgumentError("cfl must be positive, got $cfl."))
    maxlevel ≥ 0 || throw(ArgumentError("maxlevel must be non-negative, got $maxlevel."))
    if maxlevel > 0
        (refine_tol === nothing || coarsen_tol === nothing) && throw(ArgumentError(
            "an adaptive run (maxlevel = $maxlevel) needs refine_tol and coarsen_tol: " *
            "they have no default, since what is under-resolved is the case's to say " *
            "(CODE.md, \"Refinement\")."))
        refine === nothing || throw(ArgumentError(
            "pass either a static mesh (`refine`) or an adaptive one (`maxlevel > 0`), not " *
            "both: the first regrid would undo the static refinement."))
    end
    ξ = resolve_ξ(T, ξ, case)
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), D) : ntuple(_ -> Int(roots), D)
    forest = Forest{T}(rs; N=N, periodic=case.periodic, extents=case.extents)
    refine === nothing || refine(forest)
    f = mhd_fields(T, forest; G=G, backend=backend)
    build() = MHDProblem(f; eos=case.eos, opts=case.opts, pol=case.pol, model=case.model,
                         provider=case.provider, 𝔅̄=case.𝔅̄, ξ=ξ, stage=stage,
                         limiter=limiter, riemann=riemann, ops_U=ops_U, ops_A=ops_A,
                         fixup=fixup, boundary=case_boundary(case))
    flagkw = (; refine_tol, coarsen_tol, maxlevel, ε, ε_g)
    passes = 0
    if maxlevel > 0
        passes = adapt_case!(f, case, build; ops_U=ops_U, ops_A=ops_A, buffer=buffer,
                             maxpasses=maxpasses, flagkw...)
    end
    p = build()
    u = exact_state(p, case, zero(T))
    low_walls!(p, u, zero(T))                         # AMR-8's workaround
    totals0 = conserved_totals(p, u)
    maxlevel > 0 && after_regrid !== nothing && after_regrid(p, u, zero(T))
    nchunks = chunk_count(t_end, chunk)
    t = zero(T)
    total = 0
    nregrids = 0
    integ = nothing
    chunks = []
    for c in 1:nchunks
        t1 = c == nchunks ? t_end : c * chunk
        clear_tally!(p)
        if maxlevel > 0 && c > 1
            update_primitives!(p, u, t)
            r = regrid_state!(p, u, t; flags=mhd_flags(p; flagkw...), buffer=buffer)
            if r.moved
                p, u = r.p, r.u
                integ = nothing                       # the scratch has the old length
                nregrids += 1
                after_regrid === nothing || after_regrid(p, u, t)
            end
        end
        rate = max_light_rate(p, t)
        nsteps = chunk_count((t1 - t) * D * rate, cfl)
        integ = mhd_integrator(p, u, t, t1, nsteps; tableau=tableau, alias_u0=true,
                               reuse=integ)
        if after_step === nothing
            IRK.solve!(integ)
        else
            while integ.nstep < integ.nsteps
                IRK.step!(integ)
                after_step(p, integ.u, integ.t)
            end
        end
        push!(chunks, (; t=t1, nsteps, dt=(t1 - t) / nsteps, tally=ohm_tally(p),
                       nblocks=nblocks(p.f.U)))
        total += nsteps
        t = t1
    end
    return (; p, u, t, nsteps=total, chunks, nregrids, passes, nblocks=nblocks(p.f.U), totals0)
end

"""
    default_ξ(case)

The gauge damping [`evolve!`](@ref) uses unless told otherwise: `16/L`, `L`
the longest side of the case's box — the damping of a gauge wave over a
sixteenth of a light crossing of the box — on every mesh, uniform or refined
(decided in step 8; amended in step 8, decided by Erik 2026-10-15: "choose
the same ξ on all meshes", where it was zero on a uniform mesh). The longest
side, so that a case and its extrusions and turns (`extruded_case`,
`turned_case`), whose boxes differ only in the directions added or
exchanged, damp alike and stay the same run bit for bit. Across a
coarse-fine face the gauge reaches `𝔅`, since the prolongation of a coarse
gradient is not a fine one: a pure gauge pulse on a two-level unit box left
`max |𝔅|` at 5.3e−2 of the pulse after two light crossings with `ξ = 0`,
6.6e−3 at `ξ = 4`, 1.4e−3 at 16 and 3.3e−4 at 64, while the oblique Alfvén
wave's errors moved in the fifth digit at most (measured in step 8,
`bin/gauge_damping.jl`). On a uniform mesh `−∇(αΦ − β·A)` is a discrete
gradient whose curl vanishes to rounding, so there `ξ` moves `A`'s gauge
part and `Φ̃`, and `𝔅` and the matter only at rounding ("Refinement" in
`CODE.md`).
"""
default_ξ(case::MHDCase{T,D}) where {T,D} =
    16 / maximum(e -> case.extents[e][2] - case.extents[e][1], 1:D)

# `evolve!`'s `ξ`: the default, or the caller's in the case's type.
resolve_ξ(::Type{T}, ξ, case) where {T} = ξ === nothing ? T(default_ξ(case)) : T(ξ)

"""
    default_ops_U(), default_ops_A()

The operators [`evolve!`](@ref) gives the matter set and the potentials
unless told otherwise ("Field sets" and "Refinement" in `CODE.md`).
"""
default_ops_U() = Operators(; prolongation=3, restriction=2, family=Conservative, bounded=true)
default_ops_A() = Operators(; prolongation=4, restriction=4)

"""
    adapt_case!(f::MHDFields, case, build; ops_U, ops_A, refine_tol, coarsen_tol,
                maxlevel, ε, ε_g, buffer, maxpasses) -> passes

The initial adaptation of every field set of `f` to `case`'s data at
`t = 0` (AMR-3, TreeAMR's several-set `adapt_to_initial_data!`): each pass
fills the evolved sets from the case — the conserved state *at* each cell's
centre, a point value standing in for the cell average while the mesh is
being chosen, and the potentials at their points — fills their ghosts with
their own hooks, recovers `P` and flags by [`mhd_flags`](@ref), and regrids
with `transfer = false`; the scratch sets are resized. On the converged mesh
the driver then writes the exact initial data ([`exact_state`](@ref)).
`build()` makes the problem over `f` as the forest stands. Throws if the
cycle has not converged within `maxpasses`.
"""
function adapt_case!(f::MHDFields{D}, case::MHDCase{T,D}, build; ops_U, ops_A, refine_tol,
                     coarsen_tol, maxlevel, ε, ε_g, buffer, maxpasses) where {T,D}
    hooks = case_boundary(case)
    hook(i) = hooks === nothing ? nothing : boundary_at(hooks[i], zero(T))
    fl, pt, eos, prov = case.fields, case.potential, case.eos, case.provider
    initU = AllVariables(x -> case_cons(fl, eos, prov, point3(x), zero(T)))
    initA(R) = AllVariables(x -> (let (A, Φ̃) = pt(point3(x), zero(T))
                                      R ≤ 3 ? A[R] : Φ̃
                                  end,))
    ev = evolved_sets(f)
    ops = (ops_U, ops_A, ops_A, ops_A, ops_A)
    evolved = ntuple(i -> ev[i] => (ops[i], hook(i)), 5)
    scratch = (f.P => nothing, map(s -> s => nothing, f.B)..., map(s -> s => nothing, f.F)...,
               map(s -> s => nothing, f.EMF)..., f.X => nothing)
    nscratch = length(scratch)
    initial = (initU, initA(1), initA(2), initA(3), initA(4), ntuple(_ -> nothing, nscratch)...)
    criterion(sets) = begin
        p = build()
        primitives_from_sets!(p, zero(T))
        mhd_flags(p; refine_tol, coarsen_tol, maxlevel, ε, ε_g)
    end
    _, passes, converged = adapt_to_initial_data!((evolved..., scratch...); initial=initial,
                                                  flags=criterion, buffer=buffer,
                                                  maxpasses=maxpasses)
    converged || throw(ErrorException(
        "the initial adaptation had not converged after $passes passes: the criterion " *
        "kept asking for a different mesh. Each pass re-evaluates the case on the new " *
        "mesh, so it stops when nothing new is asked — a refine_tol below what the data " *
        "can reach, or too high a maxlevel for a discontinuity, keeps it going."))
    return passes
end

evolve!(case::MHDCase{T}, valD::Val; kwargs...) where {T} = evolve!(T, case, valD; kwargs...)

"""
    convergence_rate(hs, errs)

Least-squares slope of `log(err)` against `log(h)`: the measured order of a
convergence study, from more than two resolutions at once so that one noisy
point does not become the answer. TreeHydro's (after TreeWave's and
TreeAMR's), copied rather than depended on.
"""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end
