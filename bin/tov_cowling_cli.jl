# The command line of `bin/tov_cowling.jl`: `key=value` arguments and the record, kept out of
# the script so that its `main` reads as the run alone.

include(joinpath(@__DIR__, "mpi.jl"))

const FLOATTYPES = Dict("f32" => Float32, "f64" => Float64)

function run_cli(main, args)
    kw = Dict{Symbol,Any}()
    out = nothing
    backendname, T, mpi = "cpu", Float64, false
    for a in args
        k, v = split(a, '='; limit=2)
        if k in ("N", "levels", "roots")
            kw[Symbol(k)] = parse(Int, v)
        elseif k in ("L", "t_end", "chunk")
            kw[Symbol(k)] = parse(Float64, v)
        elseif k == "limiter"
            kw[:limiter] = Symbol(v)
        elseif k == "out"
            out = v
        elseif k == "backend"
            backendname = v
        elseif k == "mpi"
            mpi = parse(Bool, v)
        elseif k == "type"
            haskey(FLOATTYPES, v) ||
                error("tov_cowling.jl: unknown type=$v; expected f32 or f64")
            T = FLOATTYPES[v]
        else
            error("tov_cowling.jl: unknown key `$k` (N, levels, L, roots, t_end, chunk, " *
                  "limiter, out, backend, type, mpi)")
        end
    end
    t0 = time()
    withrun(backendname, T, mpi) do backend, comm
        root = TreeGRRMHD.isroot(comm)              # rank 0 writes, every rank computes
        io = !root ? devnull : out === nothing ? stdout : open(out, "w")
        try
            record(t, ρ_c) = (println(io, t, "  ", ρ_c); flush(io))
            main(T; kw..., backend, comm, record)
        finally
            root && out !== nothing && close(io)
        end
        root && println("wall time ", round(time() - t0; digits=1), " s")
    end
    return nothing
end
