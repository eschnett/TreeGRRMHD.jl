# The per-phase timing table (step 14): ns per cell per phase of a right-hand side, the stage
# solve, the reset and the step, at one thread count, on one backend, at one type.
#
#     julia --project=. -t N bin/benchmark.jl [key=value...]
#
# The package environment, not `bin/`'s: this needs nothing but the package (and a device
# package for `backend=cuda` or `metal`, `bin/backend.jl`). Keys: `case` (`star`, the
# magnetized TOV star of rung 6, the default; `tov`, the unmagnetized star of rung 4), `N`
# (cells per block, 16), `roots` (root blocks per dimension, 3), `levels` (static box
# levels around the star, 1), `L` (the box `[−L, L]³`, 12), `type` (`f64`, `f32`),
# `backend` (`cpu`, `cuda`, `metal`), `two_phase` (`auto`, the backend's choice, fused on
# the CPU and split on a device; `false`, `true`: the recovery's two paths, "The two-phase
# recovery" in `CODE.md`), `workgroupsize` (`auto`; an `Int`, which KernelAbstractions pads
# with ones, so a 1D workgroup along the block's first axis; or a shape, `16x4`, padded the
# same way; AMR-5, ignored on the CPU), `reps` (the best of, 5), `steps` (steps in the timed
# step phase, 2).
#
# Prints a header row, then one tab-separated row per phase:
#
#     threads backend type case N roots levels blocks cells two_phase workgroupsize phase
#     seconds ns_per_cell
#
# Each phase is a function the step itself calls (`src/evolution.jl`), timed as the best of
# `reps` calls with the backend synchronized: `scatter`, `fill_potentials`, `curl`,
# `recover`, `fill_primitives`, `face_fluxes`, `edge_emfs`, `tendencies` (the eight that make
# a right-hand side), `rhs` (all of them, `mhd_rhs!`), `stage_solve` (`ohm_solve!` after a
# copy of the state, which it needs), `reset` (`stage_limit!`), `light` (`max_rate`) and
# `step` (a whole IMEX step, per step, under the step cache, as `evolve!` takes it), and on
# the CPU `step_broadcast` (the same with the stage arithmetic broadcast, as on a device,
# rather than threaded by IMEXRungeKutta's `:even` partition) and `step_owner` (threaded by
# block owner, `owner_partition`).
# `cell_updates_per_second` is a row of its own, from `step`. Rows go to stdout, which a
# SimLaunch run keeps as its `output.log`; the thread scan is a spec in `jobs/`.

using KernelAbstractions: CPU, synchronize
using TreeAMR
using TreeGRRMHD
const MHD = TreeGRRMHD
const IRK = MHD.IRK

include(joinpath(@__DIR__, "backend.jl"))

const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)
const TYPENAMES = Dict(Float32 => "f32", Float64 => "f64")

# The best of `reps` calls of `f`, in seconds, after one warm-up call.
function bestof(f, reps, backend)
    f()
    synchronize(backend)
    best = Inf
    for _ in 1:reps
        t0 = time_ns()
        f()
        synchronize(backend)
        best = min(best, (time_ns() - t0) / 1e9)
    end
    return best
end

# The block-owner partition of the state vector of `sets`, as IMEXRungeKutta's `partition`
# takes it: thread `c` owns, in every set's segment, the entries of the blocks in chunk `c`
# of TreeAMR's `threadchunks`, the blocks `launch_by_owner!` runs on thread `c` and whose
# entries it first touched. `threadchunks` is unexported, and this is a measurement, not
# something the package uses: it gains nothing over `:even` ("The per-phase table" in
# `CODE.md`).
function owner_partition(sets)
    nb = nblocks(first(sets))
    segments = [(first(TreeAMR.staterange(sets, i)) - 1, statelength(sets[i]) ÷ nb)
                for i in eachindex(sets)]
    blocks = TreeAMR.threadchunks(nb)
    while length(blocks) < Threads.nthreads()
        push!(blocks, 1:0)
    end
    return IRK.block_partition(blocks, segments)
end

function benchmark_case(::Type{T}, case::Symbol, L) where {T}
    case === :star && return magnetized_tov_case(T; σ₀=1e6, μ_max=10, L, roots=1)
    case === :tov && return tov_case(T; L, roots=1)
    error("unknown case=$case; expected star or tov")
end

function benchmark_phases(::Type{T}; case, N, roots, levels, L, backend, two_phase,
                          workgroupsize, reps, steps) where {T}
    c = benchmark_case(T, case, L)
    forest = Forest{T}(ntuple(_ -> roots, 3); N, periodic=c.periodic, extents=c.extents)
    levels > 0 && tov_boxes!(forest, c; levels)
    p = MHD.MHDProblem(mhd_fields(T, forest; backend), c; limiter=:mc, two_phase,
                       workgroupsize)
    u = statevector(evolved_sets(p))
    MHD.initial_state!(u, p, c, zero(T))
    du, v, U = similar(u), similar(u), similar(u)
    t = zero(T)
    dt = T(MHD.default_cfl(:mc)) / (3 * MHD.max_rate(p, u, t))
    γΔt = dt / 4                                      # a stage's, near SSP3(4,3,3)'s γ
    MHD.prepare!(p, u, t)
    best(f) = bestof(f, reps, backend)
    timings = Pair{String,Float64}[]
    push!(timings, "scatter" => best(() -> scatter!(p.sets, u)))
    push!(timings, "fill_potentials" => best(() -> MHD.fill_potentials!(p, t)))
    push!(timings, "curl" => best(() -> curl_A!(p.f, p.𝔅̄, p.spacings; p.workgroupsize)))
    push!(timings, "recover" => best(() -> MHD.recover_cells!(p, t)))
    push!(timings, "fill_primitives" => best(() -> MHD.fill_primitives!(p, t)))
    push!(timings, "face_fluxes" => best(() -> MHD.face_fluxes!(p, t)))
    push!(timings, "edge_emfs" => best(() -> MHD.edge_emfs!(p, t)))
    push!(timings, "tendencies" => best(() -> MHD.tendencies!(du, p, t)))
    push!(timings, "rhs" => best(() -> mhd_rhs!(du, u, p, t)))
    push!(timings, "stage_solve" => best(() -> (copyto!(U, u); ohm_solve!(U, u, γΔt, p, t))))
    copyto!(v, u)
    push!(timings, "reset" => best(() -> MHD.stage_limit!(v, p, t)))
    push!(timings, "light" => best(() -> MHD.max_rate(p, u, t)))
    # The step: `steps` steps of one integrator, rebuilt for every repetition as `evolve!`
    # builds one a chunk, on the state stepped in place.
    function stepper(partition)
        prev = Ref{Any}(nothing)
        return () -> begin
            integ = MHD.mhd_integrator(p, u, t, t + steps * dt, steps; alias_u0=true,
                                       reuse=prev[], partition)
            MHD.with_step_cache(() -> IRK.solve!(integ), p)
            prev[] = integ
            nothing
        end
    end
    push!(timings, "step" => best(stepper(u isa Array ? :even : nothing)) / steps)
    # On the CPU the stage arithmetic broadcast, as on a device, and by block owner, against
    # `:even`.
    if u isa Array
        push!(timings, "step_broadcast" => best(stepper(nothing)) / steps)
        push!(timings, "step_owner" => best(stepper(owner_partition(p.sets))) / steps)
    end
    cells = nblocks(p.f.U) * N^3
    return (; timings, blocks=nblocks(p.f.U), cells, two_phase=p.handoff !== nothing)
end

function main(args)
    kw = Dict{Symbol,Any}(:case => :star, :N => 16, :roots => 3, :levels => 1, :L => 12,
                          :two_phase => nothing, :workgroupsize => nothing, :reps => 5,
                          :steps => 2)
    backendname, T = "cpu", Float64
    for a in args
        k, val = split(a, '='; limit=2)
        if k in ("N", "roots", "levels", "reps", "steps")
            kw[Symbol(k)] = parse(Int, val)
        elseif k == "L"
            kw[:L] = parse(Float64, val)
        elseif k == "case"
            kw[:case] = Symbol(val)
        elseif k == "two_phase"
            kw[:two_phase] = val == "auto" ? nothing : parse(Bool, val)
        elseif k == "workgroupsize"
            kw[:workgroupsize] = val == "auto" ? nothing :
                                 occursin('x', val) ? Tuple(parse.(Int, split(val, 'x'))) :
                                 parse(Int, val)
        elseif k == "type"
            haskey(FLOATTYPES, val) ||
                error("benchmark.jl: unknown type=$val; expected f32 or f64")
            T = FLOATTYPES[val]
        elseif k == "backend"
            backendname = val
        else
            error("benchmark.jl: unknown key `$k` (case, N, roots, levels, L, type, " *
                  "backend, two_phase, workgroupsize, reps, steps)")
        end
    end
    println(join(("threads", "backend", "type", "case", "N", "roots", "levels", "blocks",
                  "cells", "two_phase", "workgroupsize", "phase", "seconds", "ns_per_cell"),
                 '\t'))
    return withbackend(backendname, T) do backend
        r = benchmark_phases(T; kw..., backend)
        wgs = kw[:workgroupsize] === nothing ? "auto" : join(kw[:workgroupsize], 'x')
        prefix = join((Threads.nthreads(), backendname, TYPENAMES[T], kw[:case], kw[:N],
                       kw[:roots], kw[:levels], r.blocks, r.cells, r.two_phase, wgs),
                      '\t')
        for (phase, s) in r.timings
            println(prefix, '\t', phase, '\t', s, '\t', 1e9 * s / r.cells)
        end
        step = last(only(filter(x -> first(x) == "step", r.timings)))
        println(prefix, '\t', "cell_updates_per_second", '\t', r.cells / step, '\t', "")
        flush(stdout)
        return nothing
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
