# The workload behind the test that a distributed run is the serial run ("Running
# distributed" in `CODE.md`; `mpi_tests.jl`).
#
# Run under MPI —
#
#     mpiexec -n 3 julia --threads=2 --project=<env> test/mpi_workload.jl OUTDIR
#
# — it runs every case below three times in one launch, so that one compilation serves
# three rank counts (TreeGeneralizedHarmonic's `test/mpi_workload.jl` is the pattern): over
# all three ranks; then over a communicator of ranks 0 and 1, while rank 2 runs the same
# cases over a communicator of itself alone and then once more with no communicator at all.
# Each group's first rank writes the group's lines to `OUTDIR/n<k>.txt` (`n0.txt` being the
# serial run), and `mpi_tests.jl` compares the four files. Every rank runs on the threads
# it was started with, so the groups are ranks of threads.
#
# Two kinds of line. A line that does not start with `~` is a claim of **bit-identity**:
# the state gathered in curve order, the leaves, every step, chunk and regrid count, every
# `Δt`, every count of the stage solve and the reset, every maximum and every per-cell
# profile gathered to every rank — none of which is a floating-point sum. A line that starts
# with `~` holds sums (the conserved totals, the reset's injections, the diagnostics' norms
# and ratios of sums), which a partition reassociates, and is compared to rounding. A line
# that starts with `#` depends on the rank count and is not compared.
#
# It may use nothing outside `Base`, TreeAMR, MPI and the package. Its digests are a fold of
# `hash` over **every** element, not `hash` of the vector, which samples a long one.

using MPI
using TreeAMR
using TreeGRRMHD

const MHD = TreeGRRMHD

mpi_hash(v) = string(foldl((h, x) -> hash(x, h), v; init=UInt(0x5eed)); base=16, pad=16)

# Every set's segment of the state, gathered in curve order: over a distributed forest each
# rank holds a contiguous run of the curve, so the ranks' segments concatenated in rank order
# are the serial segment.
mpi_digest(p, u) =
    join((mpi_hash(gather_cells(p, Array(stateview(u, p.sets, i)))) for i in 1:5), ",")

mpi_leaves(forest) = mpi_hash(forest.leaves)

# Everything `evolve!` returns that is a claim about the run, split by kind.
function mpi_result!(lines, tag, r)
    p = r.p
    push!(lines, string(tag, " nsteps=", r.nsteps, " nregrids=", r.nregrids, " passes=",
                        r.passes, " blocks=", r.nblocks, " t=", repr(r.t)))
    push!(lines, string(tag, " u=", mpi_digest(p, r.u), " leaves=",
                        mpi_leaves(first(p.sets).forest)))
    for (c, k) in enumerate(r.chunks)
        ta = k.tally
        push!(lines, string(tag, " chunk ", c, " t=", repr(k.t), " nsteps=", k.nsteps,
                            " dt=", repr(k.dt), " blocks=", k.nblocks, " bits=", ta.bits,
                            " flags=", ta.flags, " policy=", ta.policy, " policy_flags=",
                            ta.policy_flags, " recoveries=", ta.recoveries,
                            " max_recoveries=", ta.max_recoveries, " resets=", ta.resets,
                            " step_resets=", ta.step_resets, " s_floors=", ta.s_floors,
                            " seeds=", ta.seeds))
        push!(lines, string("~", tag, " chunk ", c, " injected_D=", repr(ta.injected_D),
                            " injected_τ=", repr(ta.injected_τ), " injected_DYₑ=",
                            repr(ta.injected_DYₑ), " s_floor_τ=", repr(ta.s_floor_τ)))
    end
    push!(lines, string("~", tag, " totals0=", repr(r.totals0), " totals=",
                        repr(conserved_totals(p, r.u))))
    push!(lines, string(tag, " divB=", repr(max_div_B(p, r.u, r.t)), " ρ_max=",
                        repr(MHD.central_density(p, r.u, r.t))))
    return r
end

mpi_wave() = alfven_wave_case(Float64; roots=4)
mpi_tube() = atmosphere_tube_case(Float64; By=1e-3,
                                  model=MagnetizationCutoff(Constant(1e3), 10), roots=20)
mpi_rotor() = magnetic_rotor_case(Float64; roots=4)
const MPI_ROTOR = (; N=8, chunk=1 // 80, limiter=:mc, maxlevel=1, refine_tol=0.08,
                   coarsen_tol=0.02, buffer=4)

"""
The cases, over `comm` (`nothing` for the serial run), as lines; `dir` is where this group
writes its checkpoints and `from` the directory of the three-rank group's, which every group
restarts from.

- **The Alfvén wave in `D = 1`**, two chunks, and its `profile_1d` gathered to every rank.
- **The wave on two blocks**: at three ranks one rank holds none for the whole run — the
  reductions, the integrator, the stage solve and the reset on an empty rank.
- **The magnetized atmosphere tube**: the reset fires and injects, a sum per rank.
- **The adaptive rotor in `D = 2`**: the initial adaptation's cycle and a regrid every
  chunk that moves the partition; uninterrupted, and restarted from the three-rank group's
  checkpoint after its first chunk — a checkpoint written at one rank count continued at
  another is the run that never stopped.
- **The magnetized TOV star in `D = 3`** on its static boxes, and the host diagnostics:
  `cell_profile` gathered, `field_diagnostics`, `field_drift`, `ohm_map`, `charge_norms`,
  `outside_l1`, `field_l1`.
- **A refusal**: a restart from a file that does not exist is refused on every rank.
"""
function mpi_cases(comm, dir, from)
    lines = String[]
    wave = mpi_wave()
    r = mpi_result!(lines, "wave", evolve!(wave, Val(1); N=10, t_end=1 // 8, chunk=1 // 16,
                                          limiter=:mc, comm))
    pr = profile_1d(r.p, r.u, r.t)
    push!(lines, string("wave profile x=", mpi_hash(pr.x), " U=", mpi_hash(pr.U), " B=",
                        mpi_hash(pr.B), " b2=", mpi_hash(pr.b2), " flags=",
                        mpi_hash(pr.flags)))
    mpi_result!(lines, "few", evolve!(alfven_wave_case(Float64; roots=2), Val(1); N=10,
                                      t_end=1 // 16, limiter=:mc, comm))
    mpi_result!(lines, "tube", evolve!(mpi_tube(), Val(1); N=10, t_end=1 // 5,
                                       chunk=1 // 10, limiter=:mc, comm))

    rotor = mpi_rotor()
    mpi_result!(lines, "rotor", evolve!(rotor, Val(2); MPI_ROTOR..., t_end=3 // 80, comm))
    path = joinpath(dir, "rotor.h5")
    first_ = evolve!(rotor, Val(2); MPI_ROTOR..., t_end=1 // 80, comm, checkpoint=path)
    push!(lines, string("rotor first u=", mpi_digest(first_.p, first_.u)))
    mpi_result!(lines, "rotor restarted",
                evolve!(rotor, Val(2); MPI_ROTOR..., t_end=3 // 80, comm,
                        restart=joinpath(from, "rotor.h5")))

    star = magnetized_tov_case(Float64; σ₀=1e6, μ_max=10, L=12, roots=3)
    r = mpi_result!(lines, "star",
                    evolve!(star, Val(3); N=8, t_end=1, chunk=1 // 2, limiter=:mc, comm,
                            refine=forest -> tov_boxes!(forest, star; levels=1)))
    p, u, t = r.p, r.u, r.t
    R̄ = star.fields.w.star.metric.R̄
    region(x) = sum(abs2, x) < R̄^2 ? 1 : 2
    mp = ohm_map(p, region, 2)
    push!(lines, string("star ohm_map=", mp))
    u0 = exact_state(p, star, 0.0)
    prof0 = cell_profile(p, u0, 0.0)
    prof = cell_profile(p, u, t)
    fd = field_diagnostics(prof; radius=R̄)
    push!(lines, string("star counts trigger=", (fd.trigger_in, fd.trigger_out),
                        " repaired=", (fd.repaired_in, fd.repaired_out), " cells=",
                        (fd.cells_in, fd.cells_out), " q_max=", repr(fd.q_max), " B_max=",
                        repr(fd.B_max), " EB_max_in=", repr(fd.EB_max_in)))
    push!(lines, string("~star field energy=", repr((fd.energy_in, fd.energy_out)),
                        " EB=", repr((fd.EB_in, fd.EB_out)), " E_in=", repr(fd.E_in),
                        " q=", repr((fd.q_in, fd.q_out)), " B_center=", repr(fd.B_center)))
    dr = field_drift(prof, prof0; radius=R̄, outer=2R̄)
    push!(lines, string("~star drift=", repr((dr.inside, dr.outside))))
    g = gather_profile(prof)
    push!(lines, string("star profile x=", mpi_hash(g.x), " ρ=", mpi_hash(g.ρ), " B=",
                        mpi_hash(g.B), " E=", mpi_hash(g.E), " m=", mpi_hash(g.m),
                        " flags=", mpi_hash(g.flags)))
    # The gathered profile is the serial one: its diagnostics are the distributed ones.
    gd = field_diagnostics(g; radius=R̄)
    push!(lines, string("~star gathered energy=", repr((gd.energy_in, gd.energy_out))))
    cn = charge_norms(p, u, t, R̄)
    push!(lines, string("star charge max=", repr(cn.max)))
    push!(lines, string("~star charge l1=", repr(cn.l1), " total=", repr(cn.total)))
    ol = outside_l1(p, u .- u0, R̄)
    push!(lines, string("~star outside_l1=", repr(ol)))
    push!(lines, string("~star field_l1=", repr(field_l1(p, u, u0, t, R̄))))

    refused = try
        evolve!(mpi_wave(), Val(1); N=10, t_end=1 // 8, limiter=:mc, comm,
                restart=joinpath(dir, "no-such-checkpoint.h5"))
        false
    catch e
        e isa ArgumentError
    end
    push!(lines, string("refusal agreed=", rank_reduce(&, comm, refused)))
    return lines
end

# The lines of a group, written by its first rank.
function write_lines(path, lines)
    open(path, "w") do io
        foreach(l -> println(io, l), lines)
    end
    return nothing
end

function mpi_main(out)
    MPI.Init()
    world = MPI.COMM_WORLD
    rank = MPI.Comm_rank(world)
    MPI.Comm_size(world) == 3 || error("the workload runs at three ranks")
    ckpt(n) = mkpath(joinpath(out, "ckpt$n"))
    rank == 0 && foreach(ckpt, 0:3)
    MPI.Barrier(world)

    lines = mpi_cases(world, ckpt(3), ckpt(3))
    rank == 0 && write_lines(joinpath(out, "n3.txt"), lines)

    # Ranks 0 and 1 together, and rank 2 alone; the three-rank group's checkpoints are
    # complete, its collective calls having returned.
    sub = MPI.Comm_split(world, rank < 2 ? 0 : 1, rank)
    n = MPI.Comm_size(sub)
    lines = mpi_cases(sub, ckpt(n), ckpt(3))
    MPI.Comm_rank(sub) == 0 && write_lines(joinpath(out, "n$n.txt"), lines)
    if rank == 2
        write_lines(joinpath(out, "n0.txt"), mpi_cases(nothing, ckpt(0), ckpt(3)))
    end
    MPI.Barrier(world)
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    try
        mpi_main(only(ARGS))
    catch err
        # One rank's exception would leave the others waiting in their next collective:
        # take the job down instead.
        showerror(stderr, err, catch_backtrace())
        println(stderr)
        MPI.Abort(MPI.COMM_WORLD, 1)
    end
end
