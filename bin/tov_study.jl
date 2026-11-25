# Step 10 (rung 4): the Cowling TOV star's studies, for `CODE.md`'s tables ("Validation
# ladder" 4, "What the star decides"): one run per invocation, its central density sampled every
# chunk, and at the end the drift from the initial data, the rest mass against what the
# reset injected, the resets, and the wall time. The loop is `bin/tov_cowling.jl`'s, through
# the same verbs, so that the reset's roles can be chosen.
#
#     julia --project=. -t T bin/tov_study.jl [key=value...]
#
# Keys: `N` (cells per block, 16), `levels` (box levels, 1), `half` (the boxes' half-width in
# units of the star's isotropic radius, 1.25), `L` (24), `roots` (4), `t_end` (200), `chunk`
# (the sampling interval, 1), `limiter` (`mc`), `riemann` (`hlle`; `hlle_split`), `opsW`
# (the primitives' ghost fill at refinement boundaries: `w3b`, the default, Conservative
# width 3 bounded; `w3` unbounded; `w1`), `reset` (`both`, the default, or `step`: the
# reset as the step limiter alone), `out` (the time series, `t ρ_c resets injected_D` per
# chunk; default none), `checkpoint` (a file, or a directory holding it as `state.h5`, as
# SimLaunch's restarts give one, that the run saves its state to every `every` chunks,
# default 20, and resumes from if it exists: for pre-emptible queues; default none;
# TreeIOHDF5's checkpoint, `save_run`, readable at any rank count),
# `metric` (`analytic`, the default, `Analytic(TOVMetric)` at every point; or `vertex`, the
# same metric sampled at the owned vertices into the vertex metric set and interpolated,
# step 10's round 2, `N ≥ 10`), `state` (a file the final state is serialized to, for
# `bin/tov_compare.jl`; default none; serial runs only), `δv` (the star's initial velocity along its
# Cowling radial modes, the amplitudes of F, H1 and H2 in `ρ_c/ρ_c`, `δv=a₁,a₂,a₃`;
# `tov_case`; none), and *(added in step 13)* `table` (`dd2`, `sfho`: the star on that
# table in cold β-equilibrium, `tabulated_tov_case`, read from `ENTROPYEOS_TABLE_DIR`;
# default the polytrope) with `ρ_c` (its central density, geometric; no default) and
# `ρ_min` (its atmosphere, 1e−10). The time series then also holds, per chunk, what the
# reset injected into `D Yₑ` and its entropy floors outside the atmosphere with their `τ`:
# `t ρ_c resets injected_D injected_DYₑ s_floors s_floor_τ`. `diag` (a file; default none;
# added in step 13) records per chunk where the matter outside the star is: `t`, the rest
# mass's change net of the reset so far, the rest mass at `r > 5R̄/4` above twice the
# atmosphere's density, the largest radius of such matter, and the largest density in the
# outermost layer of cells. The summary goes to the terminal; a resumed run's wall time is
# its own part only. `mpi` (`false`; `true` runs over `MPI.COMM_WORLD` under `mpiexec`, rank
# 0 writing every file and the summary; `bin/mpi.jl`, "Running distributed" in `CODE.md`).

using TreeAMR
using TreeGRRMHD
using Printf
using Serialization
const MHD = TreeGRRMHD
include(joinpath(@__DIR__, "tables.jl"))
include(joinpath(@__DIR__, "mpi.jl"))
const IRK = MHD.IRK

function f_exp!(du, u, p, t)
    MHD.prepare!(p.mhd, u, t; p.metric)
    MHD.rhs!(du, p.mhd, t; p.metric)
    return nothing
end
solve_imp!(U, u★, γΔt, p, t) = MHD.stage_solve!(U, u★, γΔt, p.mhd, t; p.metric)
stage_limit!(u, _, p, t) = MHD.stage_limit!(u, p.mhd, t; p.metric)
step_limit!(u, _, p, t) = MHD.step_limit!(u, p.mhd, t; p.metric)

const OPS_W = Dict(
    "w3b" => Operators(; prolongation=3, restriction=2, family=Conservative, bounded=true),
    "w3" => Operators(; prolongation=3, restriction=2, family=Conservative, bounded=false),
    "w1" => Operators(; prolongation=1, restriction=2, family=Conservative))

# Where the matter outside the star is (the `diag` key): the rest mass `√γ D` of the cells
# at `r > 5R̄/4` above twice the atmosphere's density, the largest radius of such a cell,
# and the largest density in the outermost layer of cells, against the box's faces.
function outer_matter(mhd, u, t, metric, R̄, L, ρ_min)
    T = typeof(L)
    P = MHD.hostcopy(MHD.update_primitives!(mhd, u, t; metric))
    U = mhd.sets[MHD.SEG_U]
    uU = Array(statearray(u, mhd.sets, MHD.SEG_U))
    N, G, GU = U.forest.N, P.G[1], U.G[1]
    M, r_front, ρ_edge = zero(T), zero(T), zero(T)
    for b in 1:nblocks(U)
        h = spacing(T, U.forest, blockkey(U, b))
        for k in 1:N, j in 1:N, i in 1:N
            x = coordinates(T, U, b, (i + GU, j + GU, k + GU))
            ρ = P.work[i + G, j + G, k + G, MHD.P_RHO, b]
            r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
            if ρ > 2ρ_min && r > 5R̄ / 4
                M += h^3 * uU[i, j, k, MHD.U_D, b]
                r_front = max(r_front, r)
            end
            maximum(abs, x) > L - h && (ρ_edge = max(ρ_edge, ρ))
        end
    end
    # This rank's blocks, combined over the ranks (collective).
    return MHD.rank_reduce(+, mhd, M), MHD.rank_reduce(max, mhd, r_front),
           MHD.rank_reduce(max, mhd, ρ_edge)
end

function study(T=Float64;N=16, levels=1, half=1.25, L=24, roots=4, t_end=200, chunk=1,
               limiter=:mc, riemann=:hlle, opsW="w3b", reset="both", record=nothing,
               checkpoint=nothing, every=20, metric="analytic", δv=(0, 0, 0), table=nothing,
               ρ_c=nothing, ρ_min=1e-10, diag=nothing, comm=nothing)
    star = if table === nothing
        MHD.tov_case(T; L, roots, δv, δv_modes=:cowling)
    else
        all(iszero, δv) || error("tov_study.jl: δv perturbs the polytrope only")
        d = load_table(table)
        MHD.tabulated_tov_case(T; d.table, d.eos, ρ_c, ρ_min, L, roots)
    end
    # A directory (SimLaunch's `ck/`, `mutable_restarts`) holds the checkpoint as a file.
    checkpoint !== nothing && isdir(checkpoint) && (checkpoint = joinpath(checkpoint, "state.h5"))
    # Whether to resume is rank 0's view of the file system, which is the rank that reads
    # the checkpoint's index: every rank resumes or none does.
    resume = checkpoint !== nothing && MHD.rank_reduce((a, b) -> a, comm, isfile(checkpoint))
    recipe = MHD.checkpoint_recipe(; T, N, levels, half, L, roots, chunk=T(chunk), limiter,
                                   riemann, opsW, reset, metric, δv, table, ρ_c, ρ_min)
    ck = resume ? MHD.load_run(checkpoint; comm) : nothing
    ck === nothing || MHD.check_recipe(ck.recipe, recipe)
    if ck === nothing
        forest = Forest{T}(star.roots; N, periodic=star.periodic, extents=star.extents, comm)
        MHD.tov_boxes!(forest, star; levels, half_width=half * star.fields.w.metric.R̄)
    else
        forest = ck.forest                           # its leaves, over this run's ranks
    end
    mhd = MHD.MHDProblem(MHD.mhd_fields(T, forest), star; limiter, riemann,
                         ops_W=OPS_W[opsW])
    u = statevector(MHD.evolved_sets(mhd))
    MHD.initial_state!(u, mhd, star, zero(T))        # the initial data, for `l1_errors`
    u0 = copy(u)
    fallbacks = nothing
    if metric == "analytic"
        p = (; mhd, metric=star.provider)
    elseif metric == "vertex"
        # The Cowling metric is frozen: `Mv` is filled once, at `t = 0`.
        vm = MHD.VertexMetricSet(T, forest; boundary=star.provider)
        prov = MHD.vertex_metric!(vm, MHD.SampledADM(star.provider.metric, vm, zero(T)),
                                  zero(T))
        fallbacks = MHD.metric_fallbacks(prov, mhd.f)
        p = (; mhd, metric=prov)
    else
        error("tov_study.jl: metric is `analytic` or `vertex`, got `$metric`")
    end
    tot0 = conserved_totals(mhd, u)
    M0, Y0 = tot0[MHD.U_D], tot0[MHD.U_DYE]
    ρ0 = MHD.central_density(mhd, u, zero(T); p.metric)
    cfl = T(MHD.default_cfl(limiter))
    t_end, chunk = T(t_end), T(chunk)
    nchunks = MHD.chunk_count(t_end, chunk)
    t, integ, steps = zero(T), nothing, 0
    injected, resets, stage_resets = zero(T), 0, 0
    injY, floors, floorτ = zero(T), 0, zero(T)
    series = Tuple{T,T}[(zero(T), ρ0)]
    rows = Tuple{T,T,Int,T,T,Int,T}[]
    c0 = 0
    if ck !== nothing
        MHD.restore_run!(u, mhd, ck)
        s = ck.run
        c0, t, steps = s.c, T(s.t), s.steps
        injected, resets, stage_resets = T(s.injected), s.resets, s.stage_resets
        injY, floors, floorτ = T(s.injY), s.floors, T(s.floorτ)
        series = collect(zip(T.(s.series.t), T.(s.series.ρ)))
        rows = collect(zip(T.(s.rows.t), T.(s.rows.ρ), Int.(s.rows.resets),
                           T.(s.rows.injected_D), T.(s.rows.injected_DYₑ),
                           Int.(s.rows.s_floors), T.(s.rows.s_floor_τ)))
        record === nothing || foreach(r -> record(r...), rows)
        MHD.isroot(comm) && @info "resumed from $checkpoint at chunk $c0, t = $t"
    end
    steps0 = steps
    for c in (c0 + 1):nchunks
        t1 = c == nchunks ? t_end : c * chunk
        clear_tally!(mhd)
        nsteps = MHD.chunk_count((t1 - t) * 3 * MHD.max_rate(mhd, u, t; p.metric), cfl)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=(t1 - t) / nsteps,
                         stage_limiter=reset == "both" ? stage_limit! : nothing,
                         step_limiter=step_limit!, partition=:even, alias_u0=true,
                         reuse=integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), mhd)
        steps += nsteps
        t = t1
        tally = ohm_tally(mhd)
        injected += tally.injected_D
        resets += tally.step_resets
        stage_resets += tally.resets
        injY += tally.injected_DYₑ
        floors += tally.s_floors
        floorτ += tally.s_floor_τ
        ρ = MHD.central_density(mhd, u, t; p.metric)
        if diag !== nothing
            o = outer_matter(mhd, u, t, p.metric, star.fields.w.metric.R̄, T(L), T(ρ_min))
            diag(t, conserved_totals(mhd, u)[MHD.U_D] - M0 - injected, o...)
        end
        push!(series, (t, ρ))
        push!(rows, (t, ρ, tally.step_resets, tally.injected_D, tally.injected_DYₑ,
                     tally.s_floors, tally.s_floor_τ))
        record === nothing || record(last(rows)...)
        if checkpoint !== nothing && (c % every == 0 || c == nchunks)
            col(k) = [r[k] for r in rows]
            MHD.save_run(checkpoint, mhd, u; recipe,
                         run=(; c, t, steps, injected, resets, stage_resets, injY, floors,
                              floorτ, series=(; t=first.(series), ρ=last.(series)),
                              rows=(; t=col(1), ρ=col(2), resets=col(3),
                                    injected_D=col(4), injected_DYₑ=col(5),
                                    s_floors=col(6), s_floor_τ=col(7))))
        end
    end
    tot1 = conserved_totals(mhd, u)
    l1 = l1_errors(mhd, u, u0)
    return (; series, ρ0, ρ1=last(series)[2], M0, ΔM=tot1[MHD.U_D] - M0, Y0,
            ΔY=tot1[MHD.U_DYE] - Y0, injY, floors, floorτ, injected, resets,
            stage_resets, steps, steps_run=steps - steps0, nblocks=nleaves(forest),
            h=minimum_spacing(forest), l1D=l1.U[1], l1τ=l1.U[5], fallbacks, u=Array(u))
end

function main(args)
    kw = Dict{Symbol,Any}()
    out = nothing
    state = nothing
    diagfile = nothing
    mpi = false
    for a in args
        k, v = split(a, '='; limit=2)
        if k in ("N", "levels", "roots", "every")
            kw[Symbol(k)] = parse(Int, v)
        elseif k in ("L", "t_end", "chunk", "half", "ρ_c", "ρ_min")
            kw[Symbol(k)] = parse(Float64, v)
        elseif k in ("limiter", "riemann")
            kw[Symbol(k)] = Symbol(v)
        elseif k in ("opsW", "reset", "checkpoint", "metric", "table")
            kw[Symbol(k)] = String(v)
        elseif k == "δv"
            kw[:δv] = Tuple(parse.(Float64, split(v, ',')))
        elseif k == "out"
            out = v
        elseif k == "state"
            state = v
        elseif k == "diag"
            diagfile = v
        elseif k == "mpi"
            mpi = parse(Bool, v)
        else
            error("tov_study.jl: unknown key `$k`")
        end
    end
    mpi && state !== nothing && error("tov_study.jl: `state` serializes one process's " *
                                      "state vector, so it is for a serial run only")
    return withcomm(mpi) do comm
        study_main(args, kw, out, state, diagfile, comm)
    end
end

function study_main(args, kw, out, state, diagfile, comm)
    # Every rank computes, and rank 0 writes: the `diag` callback is collective, so every
    # rank has one, writing to `devnull` but on rank 0.
    root = MHD.isroot(comm)
    sink(name) = name === nothing ? nothing : root ? open(name, "w") : devnull
    io = sink(out)
    rec = io === nothing ? nothing :
          (t, ρ, n, inj, injY, nf, fτ) -> (@printf(io, "%.6f %.16e %d %.6e %.6e %d %.6e\n", t,
                                                   ρ, n, inj, injY, nf, fτ); flush(io))
    dio = sink(diagfile)
    dg = dio === nothing ? nothing :
         (t, net, M, rf, ρe) -> (@printf(dio, "%.6f %.6e %.6e %.4f %.6e\n", t, net, M, rf, ρe);
                                 flush(dio))
    wall = @elapsed r = study(; kw..., record=rec, diag=dg, comm)
    io === nothing || io === devnull || close(io)
    dio === nothing || dio === devnull || close(dio)
    root || return r
    state === nothing || serialize(state, (; r.series, r.u, r.h, args))
    ρs = last.(r.series)
    ranks = TreeAMR.commsize(TreeAMR.communicator(comm))
    @printf("TOV study %s\n", join(args, " "))
    @printf("  h = %.4g, %d blocks, %d steps, %d rank(s) of %d threads, wall %.1f s (%.3g µs per cell-step)\n",
            r.h, r.nblocks, r.steps, ranks, Threads.nthreads(), wall,
            1e6 * wall / (r.steps_run * r.nblocks * get(kw, :N, 16)^3))
    @printf("  ρ_c: %.10e → %.10e, relative change %.4e; min %.6e, max %.6e\n", r.ρ0, r.ρ1,
            (r.ρ1 - r.ρ0) / r.ρ0, minimum(ρs), maximum(ρs))
    @printf("  L1 change of D %.4e, of τ %.4e\n", r.l1D, r.l1τ)
    @printf("  rest mass %.12e, change %.4e, injected by the reset %.4e, difference %.3e\n",
            r.M0, r.ΔM, r.injected, r.ΔM - r.injected)
    @printf("  D Yₑ %.12e, change %.4e, injected by the reset %.4e, difference %.3e\n",
            r.Y0, r.ΔY, r.injY, r.ΔY - r.injY)
    @printf("  entropy floors outside the atmosphere %d, their τ %.4e\n", r.floors, r.floorτ)
    @printf("  step-limiter writes %d, stage-limiter writes %d\n", r.resets, r.stage_resets)
    r.fallbacks === nothing || println("  vertex metric fallbacks ", r.fallbacks)
    return r
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
