# The command line of `bin/tov_cowling.jl`: `key=value` arguments and the record, kept out of
# the script so that its `main` reads as the run alone.

function run_cli(main, args)
    kw = Dict{Symbol,Any}()
    out = nothing
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
        else
            error("tov_cowling.jl: unknown key `$k` (N, levels, L, roots, t_end, chunk, " *
                  "limiter, out)")
        end
    end
    io = out === nothing ? stdout : open(out, "w")
    t0 = time()
    try
        main(; kw..., record=(t, ρ_c) -> (println(io, t, "  ", ρ_c); flush(io)))
    finally
        out === nothing || close(io)
    end
    println("wall time ", round(time() - t0; digits=1), " s")
    return nothing
end
