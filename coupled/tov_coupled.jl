# Step 11 (rung 5): the TOV star in a dynamical spacetime, as a script that owns the time
# loop ("The coupling contract", "Two scripts" in `CODE.md`). TreeGeneralizedHarmonic
# evolves the spacetime and TreeGRRMHD the matter, one IMEXRungeKutta integrator over both
# packages' sets; the metric crosses at TreeGH's owned vertices as `(α, β, γ, K)` (GH-1)
# into this package's vertex set, the matter as `(ρ, S_i, S_ij)` (GH-3) back. This is
# `bin/tov_cowling.jl` with TreeGH's lines added: `problems` builds the mesh and the two
# problems, `main` holds the run.
#
#     julia --project=coupled -t T coupled/tov_coupled.jl [key=value...]
#
# Keys (`coupled/tov_coupled_cli.jl`): `N` (cells per block, 16), `levels` (box levels
# around the star, 1), `L` (the box `[−L, L]³`, 24), `roots` (4), `t_end` (200), `chunk`
# (10), `limiter` (`ppm`), `cfl_gh` (TreeGH's CFL factor), `out` (a file for the record,
# `t ρ_c ‖ℋ‖₂ ‖ℳ‖₂` per chunk; default the terminal).

using TreeAMR
import TreeGRRMHD
import TreeGeneralizedHarmonic
const MHD = TreeGRRMHD
const GH = TreeGeneralizedHarmonic
const IRK = MHD.IRK                                  # IMEXRungeKutta, as both packages load it

include(joinpath(@__DIR__, "gh7.jl"))                # GH-7 workaround

# TreeGH's segment of the state vector, then the matter's (AMR-1).
segments(sets) = (staterange(sets, 1), first(staterange(sets, 2)):statelength(sets))

# TreeGH's metric at a state: its fill and its adapter (GH-1), then this package's vertex
# set filled from it (the owned vertices copied, its own ghost fill), which every kernel
# interpolates.
#
# The fill is skipped when its input repeats (proposed in step 11, "Two scripts" in
# `CODE.md`): an SSP3(4,3,3) step calls `metric_at` eleven times, and at a stage whose two
# abscissae agree the stage limiter's call and `f_exp!`'s see the state the stage solve's
# call filled, since TreeGH has no implicit part and its limiter changes nothing on the
# star. The key is TreeGH's segment bit for bit and `t`, so a skipped fill is the fill it
# replaces; TreeGH's prepared state and `Mv` are still that fill's, since nothing else
# writes them between calls, and any other TreeGH verb the script calls on `p.gh` drops the
# key first (`forget_fill!`). `key = nothing` fills every time.
struct FillKey{V,T}
    seg::V                                           # TreeGH's segment, as last filled
    t::Base.RefValue{T}
    valid::Base.RefValue{Bool}
    count::Base.RefValue{Int}                        # fills done, for the measurements
end
FillKey(u, s) = FillKey(similar(view(u, s)), Ref(zero(eltype(u))), Ref(false), Ref(0))
forget_fill!(p) = (p.key === nothing || (p.key.valid[] = false); p)

# Bitwise equality and a copy, threaded: the segment is 20 variables at every owned vertex.
function chunked(f, n)
    nt = Threads.nthreads()
    r = fill(true, nt)
    Threads.@threads for c in 1:nt
        r[c] = f((c - 1) * n ÷ nt + 1, c * n ÷ nt)
    end
    return all(r)
end
same_bits(a, b) = chunked(length(a)) do lo, hi
    for i in lo:hi
        @inbounds a[i] === b[i] || return false
    end
    return true
end
threaded_copy!(a, b) = chunked(length(a)) do lo, hi
    @inbounds copyto!(a, lo, b, lo, hi - lo + 1)
    return true
end

fill_metric(p, u, t) = MHD.vertex_metric!(p.vm, GH.ADMMetric(p.gh, view(u, p.s[1]), t), t)
metric_at(p, u, t) = metric_at(p.key, p, u, t)
metric_at(::Nothing, p, u, t) = fill_metric(p, u, t)
function metric_at(k::FillKey, p, u, t)
    ug = view(u, p.s[1])
    k.valid[] && k.t[] == t && same_bits(ug, k.seg) && return MHD.VertexMetric(p.vm)
    k.valid[] = false                                # until the fill has finished
    m = fill_metric(p, u, t)
    threaded_copy!(k.seg, ug)
    k.t[], k.valid[] = t, true
    k.count[] += 1
    return m
end

# IMEXRungeKutta's four callbacks. The order of the calls is the data flow: the matter is
# recovered in TreeGH's metric, and TreeGH reads the matter of that recovery.
function f_exp!(du, u, p, t)
    (; gh, mhd, s) = p
    metric = metric_at(p, u, t)
    m = MHD.prepare!(mhd, view(u, s[2]), t; metric)
    GH.rhs!(view(du, s[1]), gh, t; matter=MHD.ADMMatter(m))           # GH-3
    MHD.rhs!(view(du, s[2]), mhd, t; metric)
    return nothing
end
function solve_imp!(U, u★, γΔt, p, t)
    (; mhd, s) = p
    copyto!(view(U, s[1]), view(u★, s[1]))           # TreeGH has no implicit part,
    metric = metric_at(p, u★, t)                     # and its limiter never runs here
    MHD.stage_solve!(view(U, s[2]), view(u★, s[2]), γΔt, mhd, t; metric)
    return nothing
end
function stage_limit!(u, _, p, t)
    GH.stage_limit!(view(u, p.s[1]), p.gh, t)        # TreeGH's first, so that the
    metric = metric_at(p, u, t)                      # matter reads the limited metric
    MHD.stage_limit!(view(u, p.s[2]), p.mhd, t; metric)
    return nothing
end
function step_limit!(u, _, p, t)
    GH.step_limit!(view(u, p.s[1]), p.gh, t)
    MHD.step_limit!(view(u, p.s[2]), p.mhd, t; metric=metric_at(p, u, t))
    return nothing
end

# The star of step 10 on one forest, TreeGH's problem on it, the matter's, and the vertex
# set the metric crosses into; the state vector holds TreeGH's set first. TreeGH's gauge is
# its static-background path ("Gauge for the TOV milestones" in `CODE.md`): `H_a` of the
# star's own metric, sampled and frozen. Its damping (proposed in step 11): `γ0 = 1/M`,
# `γ2 = 0`, `ε_KO = 1/2`.
function problems(T=Float64; N=16, levels=1, L=24, roots=4, limiter=:ppm, q=4,
                  refinement=nothing, reuse_fill=true)
    star = MHD.tov_case(T; K=100, Γ=2, ρ_c=1.28e-3, L, roots)
    forest = Forest{T}(star.roots; N, periodic=star.periodic, extents=star.extents)
    MHD.tov_boxes!(forest, star; levels)             # the static boxes around the star
    tov = star.provider.metric
    ghcase = GH.GHCase(T, tov; box=star.extents, periodic=star.periodic, ε_KO=T(1 // 2),
                       γ0=inv(tov.M), γ2=zero(T), refinement)
    U = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3))
    ops = Operators(; prolongation=q + 2, restriction=q + 2)
    gh = GH.GHProblem(U, GhostSchedule(U, ops), ghcase; q)   # q ≥ 4: K at the contract's order
    mhd = MHD.MHDProblem(MHD.mhd_fields(T, forest), star; limiter)
    vm = MHD.VertexMetricSet(T, forest; boundary=star.provider)  # Mv: G = 4, N ≥ 10
    sets = (GH.evolved_sets(gh)..., MHD.evolved_sets(mhd)...)
    u = statevector(sets)
    s = segments(sets)
    p = (; gh, mhd, vm, s, key=reuse_fill ? FillKey(u, s[1]) : nothing)
    GH.initial_state!(view(u, p.s[1]), gh, zero(T))             # fill_exact!
    MHD.initial_state!(view(u, p.s[2]), mhd, star, zero(T))     # exact_state
    return (; p, u, star, forest, sets)
end

# The chunk's step count: the smaller of TreeGH's step (`cfl_gh` of its speed's crossing
# time of the finest spacing) and the matter's (step 10's rule, in TreeGH's metric).
function step_count(p, u, t, t1, forest, cfl_gh, cfl_mhd)
    T = eltype(u)
    λ = GH.max_speed(p.gh, view(u, p.s[1]), t)
    forget_fill!(p)
    rate = MHD.max_rate(p.mhd, view(u, p.s[2]), t; metric=metric_at(p, u, t))
    return max(MHD.chunk_count((t1 - t) * λ, cfl_gh * minimum_spacing(T, forest)),
               MHD.chunk_count((t1 - t) * 3 * rate, cfl_mhd))
end

# One regrid of every set, refined where either package asks and coarsened only where both
# allow; then both problems rebuilt, the state gathered on the new mesh, and both limiters
# run once on it, the matter's in TreeGH's metric of the transferred state. `Mv` is `NaN`
# after the regrid until `metric_at` refills it, which the limiter does.
plain(f) = f isa Tuple ? first(f) : f
refine_wins(a, b) = a == Refine || b == Refine ? Refine : a == b == Coarsen ? Coarsen : Keep
function regrid_all(p, u, sets, forest, t; mhd_flags)
    metric = metric_at(p, u, t)
    fl = refine_wins.(plain.(gh_flags(p.gh, view(u, p.s[1]), t).flags),     # GH-7
                      plain.(MHD.flags(p.mhd, view(u, p.s[2]), t; metric, mhd_flags...)))
    forget_fill!(p)
    scatter!(sets, u)
    pairs = (gh_regrid_pairs(p.gh, t)..., MHD.regrid_pairs(p.mhd, t)...,     # GH-7
             MHD.regrid_pairs(p.vm)...)
    regrid!(forest, pairs; flags=fl) || return (; moved=false, p, u)
    s = segments(sets)
    u = statevector(sets)
    p = (; gh=gh_rebuild(p.gh, t), mhd=MHD.rebuild(p.mhd), vm=MHD.rebuild(p.vm),  # GH-7
         s, key=p.key === nothing ? nothing : FillKey(u, s[1]))
    gather!(u, sets)
    MHD.clear_tally!(p.mhd)
    step_limit!(u, nothing, p, t)
    return (; moved=true, p, u)
end

function main(T=Float64; N=16, levels=1, L=24, roots=4, t_end=200, chunk=10, limiter=:ppm,
              cfl_gh=T(1 // 2), refinement=nothing, mhd_flags=nothing, reuse_fill=true,
              record=(t, ρ_c, c) -> println(t, "  ", ρ_c, "  ", c.ham_l2, "  ", c.mom_l2))
    (; p, u, forest, sets) = problems(T; N, levels, L, roots, limiter, refinement,
                                      reuse_fill)
    cfl_mhd = T(MHD.default_cfl(limiter))
    t_end, chunk = T(t_end), T(chunk)
    nchunks = MHD.chunk_count(t_end, chunk)
    t, integ = zero(T), nothing
    for c in 1:nchunks
        t1 = c == nchunks ? t_end : c * chunk
        nsteps = step_count(p, u, t, t1, forest, cfl_gh, cfl_mhd)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=(t1 - t) / nsteps,
                         stage_limiter=stage_limit!, step_limiter=step_limit!,
                         partition=:even, alias_u0=true, reuse=integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
        t = t1
        metric = metric_at(p, u, t)
        ρ_c = MHD.central_density(p.mhd, view(u, p.s[2]), t; metric)  # prepares the matter
        record(t, ρ_c, gh_constraint_norms(p.gh, view(u, p.s[1]), t;   # GH-7
                                           matter=MHD.ADMMatter(p.mhd)))
        forget_fill!(p)
        mhd_flags === nothing && continue            # the star's boxes are static
        r = regrid_all(p, u, sets, forest, t; mhd_flags)
        r.moved || continue
        p, u, integ = r.p, r.u, nothing              # the integrator's scratch has the old length
    end
    return (; u, t, p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "tov_coupled_cli.jl"))
    run_cli(main, ARGS)
end
