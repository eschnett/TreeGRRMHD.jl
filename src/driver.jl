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

# The three-point Gauss–Legendre rule on [−1, 1]: nodes −√(3/5), 0, √(3/5),
# weights 5/9, 8/9, 5/9.
@inline gauss3_node(h, q::Int) = q == 2 ? zero(h) : (q == 1 ? -one(h) : one(h)) * sqrt(oftype(h, 3) / 5)
@inline gauss3_weight(h, q::Int) = q == 2 ? oftype(h, 8) / 9 : oftype(h, 5) / 9

# The conserved state of the case at the point `x` and time `t`.
@inline function case_cons(fields, eos, provider, x::SVector{3,T}, t::T) where {T}
    m = point_metric(provider, t, x)
    ρ, v, s, Yₑ, E, B = fields(x, t)
    prim = primitive_state(eos, m, ρ, v, s, Yₑ)
    return prim2con(eos, m, prim, E, B)
end

@kernel function cell_average_kernel!(uU, fields, eos, provider, t, @Const(origins),
                                      @Const(spacings), ::Val{D}) where {D}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    xc = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    tt = oftype(h, t)
    acc = ntuple(_ -> zero(h), Val(NU))
    for q in CartesianIndices(ntuple(_ -> 3, Val(D)))
        w = one(h)
        δ = ntuple(Val(3)) do e
            e ≤ D ? gauss3_node(h, q[e]) * h / 2 : zero(h)
        end
        for e in 1:D
            w *= gauss3_weight(h, q[e]) / 2
        end
        c = case_cons(fields, eos, provider, xc + SVector(δ), tt)
        acc = ntuple(v -> acc[v] + w * c[v], Val(NU))
    end
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
# also receives the nearest owned point. The nearest
# evolved point of a ghost is its index clamped, along each outward
# direction, to the owned range `G+1 … G+N` — on the high side of a
# vertex-like dimension that skips the shared plane `G+N+1`, which nobody
# owns and the hook fills.
function (d::DirichletAt{<:DirichletBoundary{R}})(fs::FieldSet{T,D}, b, key, δ,
                                                region) where {R,T,D}
    h = d.hook
    N = fs.forest.N
    G = fs.G
    w = fs.work
    for I in region
        g = Tuple(I)
        r = ntuple(e -> δ[e] < 0 ? G[e] + 1 : δ[e] > 0 ? G[e] + N : g[e], Val(D))
        xg = coordinates(T, fs, b, g)
        xr = coordinates(T, fs, b, r)
        @inbounds w[g..., 1, b] = w[r..., 1, b] + (case_potential(h, xg, d.t) -
                                                   case_potential(h, xr, d.t))
    end
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
    evolve!([T], case::MHDCase, Val(D); N, t_end, chunk = t_end, cfl = 2/5, limiter,
            riemann = :hlle, ξ = 0, stage = nothing, tableau = IMEXSSP3433(), G = 3,
            backend = CPU()) -> NamedTuple

Run `case` from `t = 0` to `t_end` on the uniform mesh of `case.roots` root
blocks of `N` cells per dimension, in chunks of length `chunk`. Each chunk's
`Δt` is `cfl / (D · max λ/h)` from the light cone at the chunk's start
([`max_light_rate`](@ref)), rounded down to a whole number of steps, and each
chunk tallies its implicit stages and resets ([`ohm_tally`](@ref)).

Returns `(; p, u, t, nsteps, chunks)`: the problem, the final state vector
over `p.sets`, the final time, the total step count, and per chunk
`(t, nsteps, dt, tally)`.

`limiter` has no default (see [`MHDProblem`](@ref)); `cfl = 2/5` is
TreeHydro's, under the explicit part of SSP3(4,3,3) — the SSPRK(3,3)
tableau, whose SSP coefficient is 1 (proposed in step 5).
"""
function evolve!(::Type{T}, case::MHDCase{T,D}, ::Val{D}; N::Integer, t_end, chunk=t_end,
                 cfl=T(2 // 5), limiter=nothing, riemann::Symbol=:hlle, ξ=zero(T),
                 stage=nothing, tableau=IRK.IMEXSSP3433(), G::Integer=3,
                 backend=CPU()) where {T,D}
    t_end, chunk, cfl = T(t_end), T(chunk), T(cfl)
    t_end > 0 || throw(ArgumentError("t_end must be positive, got $t_end."))
    chunk > 0 || throw(ArgumentError("chunk must be positive, got $chunk."))
    cfl > 0 || throw(ArgumentError("cfl must be positive, got $cfl."))
    forest = Forest{T}(case.roots; N=N, periodic=case.periodic, extents=case.extents)
    f = mhd_fields(T, forest; G=G, backend=backend)
    p = MHDProblem(f; eos=case.eos, opts=case.opts, pol=case.pol, model=case.model,
                   provider=case.provider, 𝔅̄=case.𝔅̄, ξ=ξ, stage=stage, limiter=limiter,
                   riemann=riemann, boundary=case_boundary(case))
    u = exact_state(p, case, zero(T))
    nchunks = chunk_count(t_end, chunk)
    t = zero(T)
    total = 0
    integ = nothing
    chunks = []
    for c in 1:nchunks
        t1 = c == nchunks ? t_end : c * chunk
        clear_tally!(p)
        rate = max_light_rate(p, t)
        nsteps = chunk_count((t1 - t) * D * rate, cfl)
        integ = mhd_integrator(p, u, t, t1, nsteps; tableau=tableau, alias_u0=true,
                               reuse=integ)
        IRK.solve!(integ)
        push!(chunks, (; t=t1, nsteps, dt=(t1 - t) / nsteps, tally=ohm_tally(p)))
        total += nsteps
        t = t1
    end
    return (; p, u, t, nsteps=total, chunks)
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
