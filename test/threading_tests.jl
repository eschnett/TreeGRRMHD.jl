# Thread-count independence (step 14), TreeHydro's test.
#
# There is no switch to test. TreeAMR's kernels and reductions thread themselves and place
# every block on the thread that owns it; IMEXRungeKutta's stage arithmetic is elementwise;
# and this package adds no threaded loop of its own. What has to be guarded is the
# invariant that makes that safe to rely on — the answer does not move when the thread
# count does, to the last bit — because nothing else in the suite would notice if it did
# (`CODE.md`, "Precision and devices"). Every assertion here is exact equality, never `≈`:
# roundoff-level agreement is what a reassociated sum gives, and a reassociated sum is
# exactly the bug.

include("thread_workload.jl")

@testset "A run is bit-identical across thread counts" begin
    # Guards a reduction partitioned by thread rather than by block, a kernel that writes a
    # point another block's thread owns, or a stage combination whose association depends on
    # the partition: each would pass every other test in this suite and fail here. The
    # thread count is a command-line argument to Julia and cannot change inside a session,
    # so the comparison is against a subprocess started at another count.
    reference = thread_digests()
    # Two chunks and a summary for each of the three runs.
    @test length(reference) == 3 * 3
    # A run where the reset fired, a run that regridded, and a run in three dimensions
    # whose stage solve took more than one branch: without them the digests would be a
    # claim about a quiet uniform mesh.
    @test occursin(r"[ (]resets = [1-9]", reference[1])
    @test occursin(r"nregrids=[1-9]", reference[6])
    @test !occursin(r"bits = \(0, \d+, 0, 0, 0, 0, 0, 0, 0\)", reference[7])

    other = Threads.nthreads() == 1 ? max(2, min(4, Sys.CPU_THREADS)) : 1
    script = joinpath(@__DIR__, "thread_workload.jl")
    # The active project, not `test/`: under `Pkg.test` the tests run in a sandbox, and
    # `test/Project.toml` has no manifest of its own.
    project = Base.active_project()
    out = read(`$(Base.julia_cmd()) --threads=$other --project=$project $script`, String)
    lines = split(chomp(out), '\n')
    @test lines == reference
    # Said line by line where they differ, so that a failure names the run and the chunk.
    for (a, b) in zip(lines, reference)
        a == b || @info "digest differs at $other thread(s)" subprocess = a here = b
    end
end
