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
# (10), `limiter` (`ppm`), `cfl_gh` (TreeGH's CFL factor), `γ0M` and `γ2` (TreeGH's
# damping, `γ0 = γ0M/M`), `δv` (the star's initial velocity along its coupled radial
# modes, the amplitudes of F, H1 and H2 in `ρ_c/ρ_c`, `δv=a₁,a₂,a₃`; `tov_case`), `order`
# (the matter's transfer to the vertices, 2 or 4; 4), `ρ_min` (the atmosphere's density;
# 1e−12, below `tov_case`'s 1e−10: "Validation ladder" 5 in `CODE.md`), `table` (`dd2`,
# `sfho`: the star on that table in cold β-equilibrium instead of the polytrope,
# `tabulated_tov_case`, read from `ENTROPYEOS_TABLE_DIR`; added in step 13) with `ρ_c` (its
# central density, geometric; no default), `out` (a file for the record, `t ρ_c ‖ℋ‖₂ ‖ℳ‖₂`
# per chunk; default the terminal), `backend` (`cpu`, `cuda`, `metal`; `bin/backend.jl`) and
# `mpi` (`false`; `true` over `MPI.COMM_WORLD` under `mpiexec`, one rank per device on a
# device, rank 0 writing the record; `bin/mpi.jl`, "Running distributed" in `CODE.md`).

using TreeAMR
using KernelAbstractions: CPU
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
# The fill is skipped when its input repeats ("Two scripts" in `CODE.md`): an SSP3(4,3,3)
# step calls `metric_at` eleven times, and at a stage whose two abscissae agree the stage
# limiter's call and `f_exp!`'s see the state the stage solve's call filled, since TreeGH has no implicit part and its limiter changes nothing on the
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

# Bitwise equality and a copy, threaded on the host and broadcast on a device: the segment
# is 20 variables at every owned vertex. `isequal` is `===` on floats but for NaN payloads,
# which no fill tells apart.
function chunked(f, n)
    nt = Threads.nthreads()
    r = fill(true, nt)
    Threads.@threads for c in 1:nt
        r[c] = f((c - 1) * n ÷ nt + 1, c * n ÷ nt)
    end
    return all(r)
end
host_array(a) = parent(a) isa Array
same_bits(a, b) = !host_array(a) ? mapreduce(isequal, &, a, b; init=true) :
                  chunked(length(a)) do lo, hi
    for i in lo:hi
        @inbounds a[i] === b[i] || return false
    end
    return true
end
threaded_copy!(a, b) = !host_array(a) ? (copyto!(a, b); true) :
                       chunked(length(a)) do lo, hi
    @inbounds copyto!(a, lo, b, lo, hi - lo + 1)
    return true
end

fill_metric(p, u, t) = MHD.vertex_metric!(p.vm, GH.ADMMetric(p.gh, view(u, p.s[1]), t), t)
metric_at(p, u, t) = metric_at(p.key, p, u, t)
metric_at(::Nothing, p, u, t) = fill_metric(p, u, t)
function metric_at(k::FillKey, p, u, t)
    ug = view(u, p.s[1])
    # Over MPI ranks the fill is collective, so the skip is agreed: it repeats on every rank
    # or no rank skips ("Running distributed" in `CODE.md`).
    repeats = k.valid[] && k.t[] == t && same_bits(ug, k.seg)
    MHD.rank_reduce(&, p.vm.M, repeats) && return MHD.VertexMetric(p.vm)
    k.valid[] = false                                # until the fill has finished
    m = fill_metric(p, u, t)
    threaded_copy!(k.seg, ug)
    k.t[], k.valid[] = t, true
    k.count[] += 1
    return m
end

# The matter adapter at the run's transfer order ("Matter to the vertices" in `CODE.md`):
# the per-call eight-cell mean (`order = 2`, `p.am === nothing`), or the stored form at
# fourth order, which first copies the prepared cells into `p.am`.
matter(p) = p.am === nothing ? MHD.ADMMatter(p.mhd) : MHD.ADMMatter(p.mhd, p.am; order=4)

# IMEXRungeKutta's four callbacks. The order of the calls is the data flow: the matter is
# recovered in TreeGH's metric, and TreeGH reads the matter of that recovery.
function f_exp!(du, u, p, t)
    (; gh, mhd, s) = p
    metric = metric_at(p, u, t)
    MHD.prepare!(mhd, view(u, s[2]), t; metric)
    GH.rhs!(view(du, s[1]), gh, t; matter=matter(p))                  # GH-3
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
# star's own metric, sampled and frozen. Its damping: `γ0 = γ0M/M`
# with `γ0M = 1/10`, `γ2 = 0`, `ε_KO = 1/2`. `γ0M = 1`, TreeGH's value at a hole, grows a
# mode of the star e-folding in about 36 M ("Two scripts", "As built" in `CODE.md`).
# `order` is the matter's transfer to the vertices, 2 or 4 (4, decided in step 11);
# `ρ_min` the atmosphere's density (1e−12, decided in step 11). `case` replaces the star by
# another case on the same `Analytic(TOVMetric)`, the magnetized star of rung 6
# (`magnetized_tov_study.jl`), which then holds its own `L`, `roots`, `ρ_min` and velocity.
#
# `backend` is where every set lives, `comm` the communicator the forest is distributed over
# (`nothing`, serial), and `forest` a forest to build on instead of the star's boxes: a
# checkpoint's (`MHD.load_run`), already distributed. `neutrinos` is the matter's
# `M0Leakage` (step 20; `nothing`, none): their stress-energy does not reach TreeGH.
function problems(T=Float64; N=16, levels=1, L=24, roots=4, limiter=:ppm, q=4,
                  refinement=nothing, reuse_fill=true, γ0M=1 // 10, γ2=0, δv=(0, 0, 0),
                  order=4, ρ_min=1e-12, case=nothing, backend=CPU(), comm=nothing,
                  forest=nothing, neutrinos=nothing)
    order in (2, 4) || throw(ArgumentError("the transfer order is 2 or 4, got $order"))
    case === nothing || all(iszero, δv) || throw(ArgumentError(
        "δv perturbs `tov_case`'s star; a `case` holds its own velocity."))
    star = case !== nothing ? case :
           MHD.tov_case(T; K=100, Γ=2, ρ_c=1.28e-3, L, roots, ρ_min, δv, δv_modes=:coupled)
    if forest === nothing
        forest = Forest{T}(star.roots; N, periodic=star.periodic, extents=star.extents, comm)
        MHD.tov_boxes!(forest, star; levels)         # the static boxes around the star
    end
    tov = star.provider.metric
    ghcase = GH.GHCase(T, tov; box=star.extents, periodic=star.periodic, ε_KO=T(1 // 2),
                       γ0=T(γ0M) / tov.M, γ2=T(γ2), refinement)
    U = FieldSet{T}(forest, 20; G=q ÷ 2 + 1, centering=vertexcentered(3), backend)
    ops = Operators(; prolongation=q + 2, restriction=q + 2)
    gh = GH.GHProblem(U, GhostSchedule(U, ops), ghcase; q)   # q ≥ 4: K at the contract's order
    mhd = MHD.MHDProblem(MHD.mhd_fields(T, forest; backend), star; limiter, neutrinos)
    vm = MHD.VertexMetricSet(T, forest; boundary=star.provider, backend)  # Mv: G = 4, N ≥ 10
    sets = (GH.evolved_sets(gh)..., MHD.evolved_sets(mhd)...)
    u = statevector(sets)
    s = segments(sets)
    am = order == 4 ? MHD.adm_matter_set(mhd) : nothing
    p = (; gh, mhd, vm, am, s, key=reuse_fill ? FillKey(u, s[1]) : nothing)
    GH.initial_state!(view(u, p.s[1]), gh, zero(T))             # fill_exact!
    MHD.initial_state!(view(u, p.s[2]), mhd, star, zero(T))     # exact_state
    # The neutrinos' `P` and optical depth, in TreeGH's metric (step 20).
    neutrinos === nothing ||
        MHD.initial_depth!(mhd, view(u, p.s[2]), zero(T); metric=metric_at(p, u, zero(T)))
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
    mhd = MHD.rebuild(p.mhd)
    am = p.am === nothing ? nothing : MHD.adm_matter_set(mhd)  # scratch, made anew
    p = (; gh=gh_rebuild(p.gh, t), mhd, vm=MHD.rebuild(p.vm), am,                # GH-7
         s, key=p.key === nothing ? nothing : FillKey(u, s[1]))
    gather!(u, sets)
    MHD.clear_tally!(p.mhd)
    step_limit!(u, nothing, p, t)
    return (; moved=true, p, u)
end

function main(T=Float64; N=16, levels=1, L=24, roots=4, t_end=200, chunk=10, limiter=:ppm,
              cfl_gh=T(1 // 2), refinement=nothing, mhd_flags=nothing, reuse_fill=true,
              γ0M=1 // 10, γ2=0, δv=(0, 0, 0), order=4, ρ_min=1e-12, case=nothing,
              backend=CPU(), comm=nothing,
              record=(t, ρ_c, c) -> println(t, "  ", ρ_c, "  ", c.ham_l2, "  ", c.mom_l2))
    (; p, u, forest, sets) = problems(T; N, levels, L, roots, limiter, refinement,
                                      reuse_fill, γ0M, γ2, δv, order, ρ_min, case, backend,
                                      comm)
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
                         partition=u isa Array ? :even : nothing, alias_u0=true,
                         reuse=integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
        t = t1
        metric = metric_at(p, u, t)
        ρ_c = MHD.central_density(p.mhd, view(u, p.s[2]), t; metric)  # prepares the matter
        record(t, ρ_c, gh_constraint_norms(p.gh, view(u, p.s[1]), t;   # GH-7
                                           matter=matter(p)))
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
