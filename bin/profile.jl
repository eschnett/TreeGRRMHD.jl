# Where a step's time goes: `bin/benchmark.jl`'s problem, stepped under Julia's sampling
# profiler, and the allocations and garbage collection of a step. For finding the cost the
# per-phase table shows; the table itself is `bin/benchmark.jl`.
#
#     julia --project=. -t N bin/profile.jl [key=value...]
#
# Keys: `bin/benchmark.jl`'s `case`, `N`, `roots`, `levels`, `L`, `type`, `backend`,
# `two_phase`, `steps` (the profiled steps, 4), and `phase` (`step`, the default, or one of
# the table's phases, `recover`, `rhs`, `stage_solve`, `reset`: that phase alone, called
# `steps` times), `mincount` (the flat listing's cut, 1% of the samples by default), and
# `C` (`false`; `true` lists C frames too: the scheduler's, the GC's). Prints, to stdout:
#
# - per step (or call): the seconds, the bytes allocated, the GC's seconds, from `@timed`
#   after one warm-up;
# - the flat profile of every thread, sorted by self samples (`Profile.print(format=:flat,
#   sortedby=:count)`), and the same as a tree to `profile_tree.txt` in the working
#   directory;
# - the share of samples taken in the scheduler or a lock, per thread: the time a thread
#   waits for the others, which is what stops a phase scaling.
#
# With `backend=cuda`, instead of the last three: CUDA.jl's `CUDA.@profile` of the same
# calls, the host's API calls and every kernel with its launches and total time.

using Profile
using KernelAbstractions: synchronize

include(joinpath(@__DIR__, "benchmark.jl"))

const WAIT_FRAMES = ("poptask", "task_get_next", "wait", "jl_task_wait", "sleep_check",
                     "jl_safepoint_wait", "lock", "trylock", "ijl_task_get_next",
                     "jl_gc_wait_for_the_world", "uv_run", "jl_process_events")

function profiled_problem(::Type{T}; case, N, roots, levels, L, backend, two_phase) where {T}
    c = benchmark_case(T, case, L)
    forest = Forest{T}(ntuple(_ -> roots, 3); N, periodic=c.periodic, extents=c.extents)
    levels > 0 && tov_boxes!(forest, c; levels)
    p = MHD.MHDProblem(mhd_fields(T, forest; backend), c; limiter=:mc, two_phase)
    u = statevector(evolved_sets(p))
    MHD.initial_state!(u, p, c, zero(T))
    return p, u, nleaves(forest) * N^3
end

function phase_call(p, u, phase, steps)
    T = eltype(u)
    t = zero(T)
    dt = T(MHD.default_cfl(:mc)) / (3 * MHD.max_rate(p, u, t))
    MHD.prepare!(p, u, t)
    du, U, v = similar(u), similar(u), similar(u)
    phase == "recover" && return () -> MHD.recover_cells!(p, t)
    phase == "rhs" && return () -> mhd_rhs!(du, u, p, t)
    phase == "stage_solve" && return () -> (copyto!(U, u); ohm_solve!(U, u, dt / 4, p, t))
    phase == "reset" && return () -> (copyto!(v, u); MHD.stage_limit!(v, p, t))
    phase == "step" || error("profile.jl: unknown phase `$phase`")
    prev = Ref{Any}(nothing)
    return () -> begin
        integ = MHD.mhd_integrator(p, u, t, t + steps * dt, steps; alias_u0=true,
                                   reuse=prev[], partition=u isa Array ? :even : nothing)
        MHD.with_step_cache(() -> IRK.solve!(integ), p)
        prev[] = integ
        nothing
    end
end

# Per thread: the samples, those taken asleep (the profiler's own flag), and those awake
# whose innermost frames are the scheduler's or a lock's (spinning before it sleeps).
function waitshare()
    data = Profile.fetch(; include_meta=true)
    lidict = Profile.getdict(data)
    counts = Dict{Int,NTuple{3,Int}}()
    start = 1
    for i in eachindex(data)
        Profile.is_block_end(data, i) || continue
        tid = Int(data[i - Profile.META_OFFSET_THREADID])
        asleep = data[i - Profile.META_OFFSET_SLEEPSTATE] - 1 == 1
        spinning = false
        if !asleep
            for ip in Iterators.take(@view(data[start:(i - Profile.nmeta - 2)]), 16)
                for fr in get(lidict, ip, Base.StackTraces.StackFrame[])
                    name = String(fr.func)
                    spinning |= name in WAIT_FRAMES
                end
                spinning && break
            end
        end
        s, z, w = get(counts, tid, (0, 0, 0))
        counts[tid] = (s + 1, z + asleep, w + spinning)
        start = i + 1
    end
    return counts
end

function profile_main(args)
    kw = Dict{Symbol,Any}(:case => :dd2, :N => 16, :roots => 3, :levels => 1, :L => 24,
                          :two_phase => nothing)
    backendname, T, steps, phase, mincount, C = "cpu", Float64, 4, "step", nothing, false
    for a in args
        k, val = split(a, '='; limit=2)
        if k in ("N", "roots", "levels")
            kw[Symbol(k)] = parse(Int, val)
        elseif k == "L"
            kw[:L] = parse(Float64, val)
        elseif k == "case"
            kw[:case] = Symbol(val)
        elseif k == "two_phase"
            kw[:two_phase] = val == "auto" ? nothing : parse(Bool, val)
        elseif k == "type"
            T = FLOATTYPES[val]
        elseif k == "backend"
            backendname = val
        elseif k == "steps"
            steps = parse(Int, val)
        elseif k == "phase"
            phase = val
        elseif k == "mincount"
            mincount = parse(Int, val)
        elseif k == "C"
            C = parse(Bool, val)
        else
            error("profile.jl: unknown key `$k`")
        end
    end
    return withrun(backendname, T, false) do backend, _
        p, u, cells = profiled_problem(T; kw..., backend)
        f = phase_call(p, u, phase, steps)
        calls = phase == "step" ? 1 : steps
        run!() = (for _ in 1:calls; f(); end; synchronize(backend))
        run!()                                       # compile
        st = @timed run!()
        per = phase == "step" ? steps : calls
        println("# case=$(kw[:case]) N=$(kw[:N]) roots=$(kw[:roots]) levels=$(kw[:levels]) ",
                "cells=$cells threads=$(Threads.nthreads()) backend=$backendname phase=$phase")
        println("# per $(phase == "step" ? "step" : "call"): ", st.time / per, " s, ",
                1e9 * st.time / per / cells, " ns a cell, ", st.bytes / per, " bytes, GC ",
                st.gctime / per, " s")
        if backendname == "cuda"
            # CUDA.jl's integrated profiler: the host's API calls and the device's
            # kernels, each with its count and total time.
            prof = Base.invokelatest(Core.eval(Main, :(g -> CUDA.@profile g())), run!)
            show(IOContext(stdout, :displaysize => (1000, 300)), prof)
            println()
            return nothing
        end
        Profile.clear()
        # One sample a thread every `delay`, so the buffer holds a long step at 64 threads.
        Profile.init(; n=10^8, delay=0.001 * max(1, Threads.nthreads() ÷ 8))
        @profile run!()
        shares = sort!(collect(waitshare()))
        total = sum(x -> x[2][1], shares; init=0)
        println("# samples: ", total, "; per thread: samples, % asleep, % spinning")
        for (tid, (s, z, w)) in shares
            println("# thread ", tid, ": ", s, " ", round(100z / max(s, 1); digits=1), " ",
                    round(100w / max(s, 1); digits=1))
        end
        mc = mincount === nothing ? max(1, total ÷ 100) : mincount
        Profile.print(stdout; format=:flat, sortedby=:count, mincount=mc, C, noisefloor=0)
        open("profile_tree.txt", "w") do io
            Profile.print(IOContext(io, :displaysize => (1000, 300)); format=:tree,
                          mincount=mc, C, maxdepth=60)
        end
        return nothing
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && profile_main(ARGS)
