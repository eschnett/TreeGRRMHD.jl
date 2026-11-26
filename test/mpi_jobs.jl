# The `mpiexec` job of the MPI test ("Running distributed" in `CODE.md`), as a helper rather
# than a test: `runtests.jl` includes it before the first test, and `mpi_tests.jl` includes
# it if it was not. TreeGeneralizedHarmonic's `test/mpi_jobs.jl` is the pattern, and
# TreeHydro's and TreeAMR's before it.
#
# The job runs `mpi_workload.jl` under `mpiexec -n 3`, each rank on `MPI_THREADS` threads
# (2 unless `TREEGRRMHD_TEST_MPI_THREADS` says otherwise), and is compilation-bound: three
# ranks compiling the driver in one, two and three dimensions. Where the machine has room for
# them beside the suite it starts at the start of the suite and compiles on otherwise idle
# cores while the rest runs; elsewhere it runs to its end before the suite, and
# `mpi_tests.jl` compares what it wrote. CI runs it as a job of its own
# (`TREEGRRMHD_TEST_TIER=mpi`).

using MPI: MPI

const MPI_WORKLOAD = joinpath(@__DIR__, "mpi_workload.jl")
const MPI_THREADS = parse(Int, get(ENV, "TREEGRRMHD_TEST_MPI_THREADS", "2"))

# Whether the job may run at all: three ranks of a few GB each, the package compiled in
# every one, beside the suite's own process. A machine with less than 16 GB skips it.
# `TREEGRRMHD_TEST_MPI=1` or `0` decides it by hand; CI's MPI job forces it on.
function mpi_runnable()
    choice = get(ENV, "TREEGRRMHD_TEST_MPI", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.total_memory() >= 16 * 2^30
end

# Whether the job may run beside the suite: three ranks, each compiling the workload.
# `TREEGRRMHD_TEST_MPI_CONCURRENT=0` or `1` decides it by hand.
function mpi_concurrent()
    choice = get(ENV, "TREEGRRMHD_TEST_MPI_CONCURRENT", "")
    choice == "1" && return true
    choice == "0" && return false
    return Sys.CPU_THREADS >= 3 * MPI_THREADS + 4 && Sys.total_memory() >= 24 * 2^30
end

# The workload under `mpiexec -n 3`, started and not waited for. The command is built from
# `MPI.mpiexec()` with its environment set on it: interpolating `mpiexec()` into a larger
# command drops the library paths it carries (TreeAMR's finding). The *active* project, not
# `test/`: under `Pkg.test` the tests run in a sandbox.
function launch_mpi_workload(; timeout=3600)
    out = mktempdir()
    mpi = MPI.mpiexec()
    project = Base.active_project()
    cmd = `$mpi -n 3 $(Base.julia_cmd()) --threads=$MPI_THREADS --project=$project
           $MPI_WORKLOAD $out`
    err = IOBuffer()
    proc = run(pipeline(setenv(cmd, mpi.env); stdout=devnull, stderr=err); wait=false)
    return (; proc, out, err, start=time(), timeout)
end

# The job's four files of lines, with the deadline from its start: a rank waiting for a
# message never sent would otherwise hang the suite.
function finish_mpi_workload(job)
    (; proc, out, err, start, timeout) = job
    left = max(1.0, timeout - (time() - start))
    if timedwait(() -> process_exited(proc), left) !== :ok
        kill(proc)
        error("the MPI workload did not finish in $timeout s; its stderr:\n" *
              String(take!(err)))
    end
    success(proc) || error("the MPI workload failed; its stderr:\n" * String(take!(err)))
    @info "the MPI workload ran for $(round(Int, time() - start)) s"
    return Dict(n => readlines(joinpath(out, "n$n.txt")) for n in 0:3)
end

const MPI_JOB = Ref{Any}(nothing)

# The job, started beside the suite, or run to its end before it where the machine has no
# room for both. Waiting throws nothing: a job that failed or overran is reported by
# `finish_mpi_workload` in `mpi_tests.jl`, inside the suite's testset.
function start_mpi_job!()
    mpi_runnable() || return nothing
    job = launch_mpi_workload()
    MPI_JOB[] = job
    atexit(() -> process_running(job.proc) && kill(job.proc))
    if !mpi_concurrent()
        @info "the MPI workload runs before the suite: this machine has no room for its " *
              "three ranks beside it (TREEGRRMHD_TEST_MPI_CONCURRENT=1 runs them together)"
        timedwait(() -> process_exited(job.proc), job.timeout)
    end
    return nothing
end

# The job, started now if it was not, or `nothing` where it may not run.
function take_mpi_job!()
    job = MPI_JOB[]
    MPI_JOB[] = nothing
    job === nothing || return job
    return mpi_runnable() ? launch_mpi_workload() : nothing
end
