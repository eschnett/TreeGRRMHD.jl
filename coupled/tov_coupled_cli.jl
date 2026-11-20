# The command line of `coupled/tov_coupled.jl`: `key=value` arguments and the record, kept
# out of the script so that its `main` reads as the run alone.

include(joinpath(@__DIR__, "..", "bin", "tables.jl"))

function run_cli(main, args)
    kw = Dict{Symbol,Any}()
    out = nothing
    table, ρ_c = nothing, nothing
    for a in args
        k, v = split(a, '='; limit=2)
        if k in ("N", "levels", "roots", "order")
            kw[Symbol(k)] = parse(Int, v)
        elseif k in ("L", "t_end", "chunk", "cfl_gh", "γ0M", "γ2", "ρ_min")
            kw[Symbol(k)] = parse(Float64, v)
        elseif k == "table"
            table = String(v)
        elseif k == "ρ_c"
            ρ_c = parse(Float64, v)
        elseif k == "δv"
            kw[:δv] = Tuple(parse.(Float64, split(v, ',')))
        elseif k == "limiter"
            kw[:limiter] = Symbol(v)
        elseif k == "out"
            out = v
        else
            error("tov_coupled.jl: unknown key `$k` (N, levels, L, roots, t_end, chunk, " *
                  "cfl_gh, γ0M, γ2, δv, order, ρ_min, table, ρ_c, limiter, out)")
        end
    end
    (table === nothing) == (ρ_c === nothing) ||
        error("tov_coupled.jl: `table` and `ρ_c` go together (the tabulated star)")
    if table !== nothing
        d = load_table(table)
        kw[:case] = TreeGRRMHD.tabulated_tov_case(Float64; d.table, d.eos, ρ_c,
                                                  ρ_min=get(kw, :ρ_min, 1e-12),
                                                  L=get(kw, :L, 24), roots=get(kw, :roots, 4))
    end
    io = out === nothing ? stdout : open(out, "w")
    t0 = time()
    try
        main(; kw...,
             record=(t, ρ_c, c) -> (println(io, t, "  ", ρ_c, "  ", c.ham_l2, "  ",
                                            sqrt(sum(abs2, c.mom_l2))); flush(io)))
    finally
        out === nothing || close(io)
    end
    println("wall time ", round(time() - t0; digits=1), " s")
    return nothing
end
