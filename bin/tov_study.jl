# Step 10 (rung 4): the Cowling TOV star's studies, for `CODE.md`'s tables ("Validation
# ladder" 4, "Open questions"): one run per invocation, its central density sampled every
# chunk, and at the end the drift from the initial data, the rest mass against what the
# reset injected, the resets, and the wall time. The loop is `bin/tov_cowling.jl`'s, through
# the same verbs, so that the reset's roles can be chosen.
#
#     julia --project=. -t T bin/tov_study.jl [key=value...]
#
# Keys: `N` (cells per block, 16), `levels` (box levels, 1), `half` (the boxes' half-width in
# units of the star's isotropic radius, 1.25), `L` (16), `roots` (2), `t_end` (200), `chunk`
# (the sampling interval, 1), `limiter` (`mc`), `riemann` (`hlle`; `hlle_split`), `opsW`
# (the primitives' ghost fill at refinement boundaries: `w3b`, the default, Conservative
# width 3 bounded; `w3` unbounded; `w1`), `reset` (`both`, the default, or `step`: the
# reset as the step limiter alone), `out` (the time series, `t ρ_c resets injected_D` per
# chunk; default none). The summary goes to the terminal.

using TreeAMR
using TreeGRRMHD
using Printf
const MHD = TreeGRRMHD
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

function study(T=Float64; N=16, levels=1, half=1.25, L=16, roots=2, t_end=200, chunk=1,
               limiter=:mc, riemann=:hlle, opsW="w3b", reset="both", record=nothing)
    star = MHD.tov_case(T; L, roots)
    forest = Forest{T}(star.roots; N, periodic=star.periodic, extents=star.extents)
    MHD.tov_boxes!(forest, star; levels, half_width=half * star.fields.w.metric.R̄)
    mhd = MHD.MHDProblem(MHD.mhd_fields(T, forest), star; limiter, riemann,
                         ops_W=OPS_W[opsW])
    u = statevector(MHD.evolved_sets(mhd))
    MHD.initial_state!(u, mhd, star, zero(T))
    u0 = copy(u)
    p = (; mhd, metric=star.provider)
    M0 = conserved_totals(mhd, u)[1]
    ρ0 = MHD.central_density(mhd, u, zero(T))
    cfl = T(MHD.default_cfl(limiter))
    t_end, chunk = T(t_end), T(chunk)
    nchunks = MHD.chunk_count(t_end, chunk)
    t, integ, steps = zero(T), nothing, 0
    injected, resets, stage_resets = zero(T), 0, 0
    series = Tuple{T,T}[(zero(T), ρ0)]
    for c in 1:nchunks
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
        ρ = MHD.central_density(mhd, u, t)
        push!(series, (t, ρ))
        record === nothing || record(t, ρ, tally.step_resets, tally.injected_D)
    end
    M1 = conserved_totals(mhd, u)[1]
    l1 = l1_errors(mhd, u, u0)
    return (; series, ρ0, ρ1=last(series)[2], M0, ΔM=M1 - M0, injected, resets,
            stage_resets, steps, nblocks=nblocks(mhd.f.U), h=minimum_spacing(forest),
            l1D=l1.U[1], l1τ=l1.U[5])
end

function main(args)
    kw = Dict{Symbol,Any}()
    out = nothing
    for a in args
        k, v = split(a, '='; limit=2)
        if k in ("N", "levels", "roots")
            kw[Symbol(k)] = parse(Int, v)
        elseif k in ("L", "t_end", "chunk", "half")
            kw[Symbol(k)] = parse(Float64, v)
        elseif k in ("limiter", "riemann")
            kw[Symbol(k)] = Symbol(v)
        elseif k in ("opsW", "reset")
            kw[Symbol(k)] = String(v)
        elseif k == "out"
            out = v
        else
            error("tov_study.jl: unknown key `$k`")
        end
    end
    io = out === nothing ? nothing : open(out, "w")
    rec = io === nothing ? nothing :
          (t, ρ, n, inj) -> (@printf(io, "%.6f %.16e %d %.6e\n", t, ρ, n, inj); flush(io))
    wall = @elapsed r = study(; kw..., record=rec)
    io === nothing || close(io)
    ρs = last.(r.series)
    @printf("TOV study %s\n", join(args, " "))
    @printf("  h = %.4g, %d blocks, %d steps, %d threads, wall %.1f s (%.3g µs per cell-step)\n",
            r.h, r.nblocks, r.steps, Threads.nthreads(), wall,
            1e6 * wall / (r.steps * r.nblocks * get(kw, :N, 16)^3))
    @printf("  ρ_c: %.10e → %.10e, relative change %.4e; min %.6e, max %.6e\n", r.ρ0, r.ρ1,
            (r.ρ1 - r.ρ0) / r.ρ0, minimum(ρs), maximum(ρs))
    @printf("  L1 change of D %.4e, of τ %.4e\n", r.l1D, r.l1τ)
    @printf("  rest mass %.12e, change %.4e, injected by the reset %.4e, difference %.3e\n",
            r.M0, r.ΔM, r.injected, r.ΔM - r.injected)
    @printf("  step-limiter writes %d, stage-limiter writes %d\n", r.resets, r.stage_resets)
    return r
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
