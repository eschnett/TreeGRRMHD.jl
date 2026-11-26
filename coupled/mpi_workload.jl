# The workload behind the coupled suite's test that a distributed coupled run is the serial
# one ("Running distributed" in `CODE.md`; `coupled/mpi_tests.jl`), after the package's
# `test/mpi_workload.jl`.
#
#     mpiexec -n 2 julia --threads=2 --project=coupled coupled/mpi_workload.jl OUTDIR
#
# In one launch: the cases over both ranks (`n2.txt`), then rank 0 over a communicator of
# itself alone (`n1.txt`) while rank 1 runs them with no communicator at all (`n0.txt`, the
# serial run). The lines are the package workload's kinds: `~` for sums (the constraint
# norms, the totals), compared to rounding; every other line bit for bit.
#
# The cases, through `tov_coupled.jl`'s and `tov_coupled_study.jl`'s own functions:
#
# - **The coupled star**, two chunks: TreeGH's spacetime and this package's matter, the
#   vertex metric, the matter adapter and the fill's key, whose skip every rank agrees on.
# - **The coupled star regridded**: `regrid_all` over TreeGH's set, the matter's and `Mv`.
# - **The study checkpointed**: TreeGH's set beside the matter's in one checkpoint, written
#   by the two-rank group after one chunk and resumed by every group: the resumed run is
#   the uninterrupted one.

using MPI
include(joinpath(@__DIR__, "tov_coupled_study.jl"))   # `tov_coupled.jl` and `study`

const CW = (; N=10, levels=0, L=16, roots=2, limiter=:mc)

cw_hash(v) = string(foldl((h, x) -> hash(x, h), v; init=UInt(0x5eed)); base=16, pad=16)

# TreeGH's segment and the matter's five sets, each gathered in curve order.
function cw_digest(p, u)
    g(v) = cw_hash(MHD.gather_cells(p.mhd, Array(v)))
    um = view(u, p.s[2])
    return join((g(view(u, p.s[1])),
                 (g(stateview(um, p.mhd.sets, i)) for i in 1:5)...), ",")
end

function cw_cases(comm, dir, from)
    lines = String[]
    rec(tag) = (t, ρ_c, c) -> begin
        push!(lines, string(tag, " t=", repr(t), " ρ_c=", repr(ρ_c)))
        push!(lines, string("~", tag, " t=", repr(t), " ham=", repr(c.ham_l2), " mom=",
                            repr(c.mom_l2)))
    end
    r = main(; CW..., t_end=1, chunk=1 // 2, comm, record=rec("star"))
    # The fills done, the same on every rank since the skip is agreed.
    push!(lines, string("star u=", cw_digest(r.p, r.u), " fills=", r.p.key.count[]))
    push!(lines, string("~star totals=",
                        repr(MHD.conserved_totals(r.p.mhd, view(r.u, r.p.s[2])))))

    refinement = GH.Refinement(Float64; refine_tol=0.99, coarsen_tol=0.98, maxlevel_cap=1,
                               floor_margin=0.0)
    mhd_flags = (; refine_tol=0.05, coarsen_tol=0.01, maxlevel=1)
    r = main(; CW..., t_end=1 // 2, chunk=1 // 4, refinement, mhd_flags, comm,
             record=rec("regridded"))
    forest = first(r.p.mhd.sets).forest
    push!(lines, string("regridded u=", cw_digest(r.p, r.u), " leaves=", nleaves(forest),
                        " ", cw_hash(forest.leaves)))

    put(tag, s) = begin
        for row in s.rows
            push!(lines, string(tag, " t=", repr(row.t), " ρ_c=", repr(row.ρ_c), " steps=",
                                row.steps, " resets=", row.resets))
            push!(lines, string("~", tag, " t=", repr(row.t), " ham_in=", repr(row.ham_in),
                                " ham_out=", repr(row.ham_out), " ΔM=", repr(row.ΔM),
                                " injected=", repr(row.injected)))
        end
    end
    sw = (; N=10, levels=0, L=16, roots=2, limiter=:mc, chunk=1 // 2)
    put("study", study(; sw..., t_end=1, comm))
    study(; sw..., t_end=1 // 2, comm, checkpoint=joinpath(dir, "study.h5"), every=1)
    # A copy of the two-rank group's checkpoint for this group to resume from and write to,
    # at the same path on every rank of the group (a collective load takes one path): made
    # by the group's first rank, which every rank waits for in a gather.
    resumed = joinpath(dir, "resumed")
    MHD.isroot(comm) && (mkpath(resumed);
                         cp(joinpath(from, "study.h5"), joinpath(resumed, "study.h5")))
    MHD.rank_reduce(&, comm, true)
    put("study", study(; sw..., t_end=1, comm, checkpoint=joinpath(resumed, "study.h5")))
    return lines
end

function cw_write(path, lines)
    open(path, "w") do io
        foreach(l -> println(io, l), lines)
    end
    return nothing
end

function cw_main(out)
    MPI.Init()
    world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(world)
    MPI.Comm_size(world) == 2 || error("the coupled workload runs at two ranks")
    ckpt(n) = mkpath(joinpath(out, "ckpt$n"))
    rank == 0 && foreach(ckpt, 0:2)
    MPI.Barrier(world)
    lines = cw_cases(world, ckpt(2), ckpt(2))
    rank == 0 && cw_write(joinpath(out, "n2.txt"), lines)
    # The two-rank group's checkpoint is complete, its collective calls having returned; a
    # checkpoint of one index file (one I/O group on one node) copies as a file.
    sub = MPI.Comm_split(world, rank, rank)
    if rank == 0
        cw_write(joinpath(out, "n1.txt"), cw_cases(sub, ckpt(1), ckpt(2)))
    else
        cw_write(joinpath(out, "n0.txt"), cw_cases(nothing, ckpt(0), ckpt(2)))
    end
    MPI.Barrier(world)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        cw_main(only(ARGS))
    catch err
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
end
