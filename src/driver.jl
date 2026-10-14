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
reference ([`exact_state`](@ref)).
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

# --- the integrator and the run --------------------------------------------------------

"""
    mhd_integrator(p::MHDProblem, u, t0, t1, nsteps; tableau = IMEXSSP3433(),
                   alias_u0 = false, reuse = nothing)

An IMEXRungeKutta integrator for `nsteps` fixed steps from `t0` to `t1`:
[`mhd_rhs!`](@ref) as the explicit part, [`ohm_solve!`](@ref) as the stage
solve, and [`reset_atmosphere!`](@ref) as both the stage and the step limiter
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
                     stage_limiter=reset_atmosphere!, step_limiter=reset_atmosphere!,
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
                   riemann=riemann)
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
