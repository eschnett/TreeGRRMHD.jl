# Rank-count independence ("Running distributed" in `CODE.md`).
#
# The claim extends the thread one: **a run distributed over MPI ranks is the serial run** —
# its state, its mesh, its steps, chunks, regrids and `Δt`, every count of the stage solve
# and the reset, every maximum, every per-cell profile, bit for bit at any rank count and on
# any number of threads per rank — **and its sums agree to rounding**, exactly at one rank.
# TreeAMR makes the exchange, the regrid and its own reductions so; what can break it here is
# this package's, and each failure mode is one a serial suite cannot see:
#
#   * a number combined on the host from per-block or per-cell values — a diagnostic's sum,
#     a fallback count, a profile — that is a rank's own over a distributed forest;
#   * a block count, or anything sized by it, that is the rank's where the claim is the
#     mesh's;
#   * a rank with no blocks, on which a reduction, a launch or a check reads a block that is
#     not there;
#   * a checkpoint that does not restart at another rank count, or restarts into another
#     run;
#   * a refusal on some ranks that leaves the others waiting in a collective.
#
# Every line of the workload that is not a sum is compared with `==`; a line of sums is
# compared number by number to rounding, and its text exactly.

isdefined(@__MODULE__, :take_mpi_job!) || include("mpi_jobs.jl")
isdefined(@__MODULE__, :TEST_TIER) || include("tiers.jl")

isdefined(@__MODULE__, :mpi_same) || include("mpi_lines.jl")

function mpi_compare(files)
    serial = files[0]
    # Every case ran, the serial run included, and is what it claims to be: the rotor
    # adapted and regridded, the tube's reset fired, and the refusal was every rank's.
    @test length(serial) ≥ 40
    rotor = only(filter(startswith("rotor nsteps="), serial))
    @test !occursin("passes=0", rotor) && !occursin("nregrids=0", rotor)
    @test any(l -> startswith(l, "tube chunk") && !occursin("step_resets=0", l), serial)
    # The restarted rotor is the uninterrupted one, in every group: its state, mesh, steps
    # and counts bit for bit, and its sums to rounding — `totals0` comes from the
    # three-rank group's checkpoint, summed at three ranks.
    for n in 0:3
        lines = files[n]
        rest = [replace(l, "rotor restarted" => "rotor"; count=1) for l in lines
                if startswith(l, "rotor restarted") || startswith(l, "~rotor restarted")]
        whole = [l for l in lines if (startswith(l, "rotor ") || startswith(l, "~rotor ")) &&
                                     !occursin("restarted", l) && !occursin("first", l)]
        @test length(rest) == length(whole) && all(map(mpi_same, whole, rest))
    end
    for n in 1:3
        lines = files[n]
        @test length(lines) == length(serial)
        @test only(filter(startswith("refusal "), lines)) == "refusal agreed=true"
        for (a, b) in zip(serial, lines)
            startswith(a, "#") && (@test startswith(b, "#"); continue)
            # At one rank nothing is reassociated, and the sums are the serial bits too.
            ok = n == 1 ? a == b : mpi_same(a, b)
            @test ok
            ok || @info "a line differs at $n rank(s)" serial = a distributed = b
        end
    end
    return nothing
end

@testset "A distributed run is the serial run, at every rank count" begin
    job = take_mpi_job!()
    job === nothing && @info "the MPI workload is skipped: TREEGRRMHD_TEST_MPI=0, or this " *
                             "machine has less than 16 GB for its three ranks " *
                             "(TREEGRRMHD_TEST_MPI=1 runs it)"
    job === nothing && @test_skip "a distributed run is the serial run"
    job === nothing || mpi_compare(finish_mpi_workload(job))
end
