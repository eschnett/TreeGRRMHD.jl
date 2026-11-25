# `mpi=true` for the scripts in this directory and in `coupled/` ("Running distributed" in
# `CODE.md`): run over `MPI.COMM_WORLD`, each rank on its own threads or its own device.
# MPI is loaded only when asked for: it is not a dependency of TreeGRRMHD, which never loads
# it (TreeAMR's extension holds the MPI methods), so the load is a `Core.eval` into `Main`,
# as `backend.jl` loads a device package, and a missing package is reported as the
# environment that fixes it.
#
#     mpiexec -n R julia --project=<env with MPI> -t T bin/tov_cowling.jl mpi=true ...
#
# Every rank runs the script with the same arguments; rank 0 prints and writes the files a
# study keeps (`isroot`).

include(joinpath(@__DIR__, "backend.jl"))

"""
    withcomm(f, mpi::Bool) -> f(comm)

`f(nothing)` for a serial run; for `mpi = true`, MPI loaded and initialized (with
`MPI_THREAD_SERIALIZED`, MPI.jl's default and what TreeAMR needs) and `f(MPI.COMM_WORLD)`
called in the latest world, since the load added methods after the caller started.
"""
function withcomm(f, mpi::Bool)
    mpi || return f(nothing)
    Base.identify_package("MPI") === nothing && error(
        "mpi=true needs MPI in this environment, which is deliberately not a dependency \
         of TreeGRRMHD. Run the script in an environment that has it (and TreeAMR's MPI \
         extension with it): a SimLaunch spec's `env`, or\n\n    julia \
         --project=/tmp/grmpi -e 'using Pkg; Pkg.develop(path=\".\"); Pkg.add(\"MPI\")'\n")
    Core.eval(Main, :(using MPI))
    return Base.invokelatest() do
        MPI = Core.eval(Main, :MPI)
        MPI.Initialized() || MPI.Init()
        f(MPI.COMM_WORLD)
    end
end

"""
    node_rank(comm) -> (rank, size)

This rank's place among the ranks on its own node (`MPI.Comm_split_type(…,
COMM_TYPE_SHARED)`, as TreeGeneralizedHarmonic's `test/octant_runs.jl` does), and how many
share the node: what a rank picks its device by. `(0, 1)` serially.
"""
function node_rank(comm)
    comm === nothing && return (0, 1)
    MPI = Core.eval(Main, :MPI)
    node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
    return (MPI.Comm_rank(node), MPI.Comm_size(node))
end

"""
    select_device!(name, comm)

One rank per device: on `backend=cuda` each rank takes device `node rank mod devices`, so
that the ranks of a node spread over its devices before they share one. Nothing on the CPU
or on Metal (one device). Called after the device package is loaded and before anything is
allocated on it.
"""
function select_device!(name::AbstractString, comm)
    name == "cuda" || return nothing
    CUDA = Core.eval(Main, :CUDA)
    rank, _ = node_rank(comm)
    n = length(CUDA.devices())
    CUDA.device!(rank % n)
    return nothing
end

"""
    withrun(f, backendname, T, mpi) -> f(backend, comm)

[`withcomm`](@ref), then [`withbackend`](@ref) with each rank's device selected, then
`f(backend, comm)` in the latest world.
"""
function withrun(f, backendname::AbstractString, ::Type{T}, mpi::Bool) where {T}
    return withcomm(mpi) do comm
        backend = resolvebackend(backendname)
        Base.invokelatest() do
            select_device!(backendname, comm)
            checkprecision(T, backend)
            f(backend, comm)
        end
    end
end
