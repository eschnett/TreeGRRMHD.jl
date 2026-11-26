# The coupled run distributed ("Running distributed" in `CODE.md`): `mpi_workload.jl` under
# `mpiexec -n 2`, each rank on two threads, against the serial run of the same launch. The
# package suite's `test/mpi_tests.jl` makes the claim for this package alone; what only the
# coupled script can break is its own: the fill's key, whose skip of a collective fill
# every rank must agree on; the regrid of three packages' sets together; and a checkpoint
# holding TreeGH's set beside the matter's.

using MPI: MPI
include(joinpath(@__DIR__, "..", "test", "mpi_lines.jl"))

const CW_WORKLOAD = joinpath(@__DIR__, "mpi_workload.jl")

function cw_run(; timeout=3600)
    out = mktempdir()
    mpi = MPI.mpiexec()
    cmd = `$mpi -n 2 $(Base.julia_cmd()) --threads=2 --project=$(Base.active_project())
           $CW_WORKLOAD $out`
    err = IOBuffer()
    proc = run(pipeline(setenv(cmd, mpi.env); stdout=devnull, stderr=err); wait=false)
    t0 = time()
    if timedwait(() -> process_exited(proc), timeout) !== :ok
        kill(proc)
        error("the coupled MPI workload did not finish in $timeout s; its stderr:\n" *
              String(take!(err)))
    end
    success(proc) || error("the coupled MPI workload failed; its stderr:\n" *
                           String(take!(err)))
    @info "the coupled MPI workload ran for $(round(Int, time() - t0)) s"
    return Dict(n => readlines(joinpath(out, "n$n.txt")) for n in 0:2)
end

@testset "The coupled star distributed is the serial one, and restarts at another rank count" begin
    files = cw_run()
    serial = files[0]
    # The cases ran: the record, the regrid onto more blocks, the fill's key skipping, and
    # the study twice, uninterrupted and resumed from the two-rank group's checkpoint.
    @test count(startswith("star t="), serial) == 2
    @test only(filter(startswith("star u="), serial)) |> l -> !occursin("fills=0", l)
    study = filter(l -> startswith(l, "study ") || startswith(l, "~study "), serial)
    @test iseven(length(study)) && length(study) ≥ 6
    half = length(study) ÷ 2
    # The resumed study is the uninterrupted one: its rows bit for bit, its sums to
    # rounding (its initial totals were summed at two ranks).
    @test all(map(mpi_same, study[1:half], study[(half + 1):end]))
    for n in 1:2
        lines = files[n]
        @test length(lines) == length(serial)
        for (a, b) in zip(serial, lines)
            ok = n == 1 ? a == b : mpi_same(a, b)
            @test ok
            ok || @info "a line differs at $n rank(s)" serial = a distributed = b
        end
    end
end
