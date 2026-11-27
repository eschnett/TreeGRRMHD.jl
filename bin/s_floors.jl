# Where the tabulated star's recoveries reach the table's cold edge (EOS-16's downstream
# check, "Requests still waiting" in `CODE.md`): `bin/benchmark.jl`'s DD2 star stepped as
# `evolve!` steps it, under the step cache, printing per step the step limiter's entropy
# floors on states the policy did not send to the atmosphere (`ohm_tally`'s `s_floors`),
# the cells whose state recovers `S_FLOORED` outside the atmosphere (a cold recovery of
# `uⁿ⁺¹` after the step: what the next step's first recovery meets), the atmosphere's
# cells, the `|` of the stage solve's policy flags, and the step's seconds.
#
#     julia --project=. -t N bin/s_floors.jl [key=value...]
#
# Keys: `case` (`dd2`, `sfho`), `N` (16), `roots` (4), `levels` (1), `L` (24), `steps`
# (100), `every` (1, print every so many steps). Reads `ENTROPYEOS_TABLE_DIR`.

include(joinpath(@__DIR__, "benchmark.jl"))
import EntropyEOS

function s_floor_main(args)
    kw = Dict{String,Any}("case" => "dd2", "N" => 16, "roots" => 4, "levels" => 1,
                          "L" => 24.0, "steps" => 100, "every" => 1)
    for a in args
        k, val = split(a, '='; limit=2)
        haskey(kw, k) || error("s_floors.jl: unknown key `$k` ($(join(keys(kw), ", ")))")
        kw[k] = k == "case" ? val : k == "L" ? parse(Float64, val) : parse(Int, val)
    end
    T = Float64
    N = kw["N"]
    c = benchmark_case(T, Symbol(kw["case"]), kw["L"])
    forest = Forest{T}(ntuple(_ -> kw["roots"], 3); N, periodic=c.periodic, extents=c.extents)
    kw["levels"] > 0 && tov_boxes!(forest, c; levels=kw["levels"])
    p = MHD.MHDProblem(mhd_fields(T, forest; backend=CPU()), c; limiter=:mc)
    u = statevector(evolved_sets(p))
    MHD.initial_state!(u, p, c, zero(T))
    t = zero(T)
    dt = T(MHD.default_cfl(:mc)) / (3 * MHD.max_rate(p, u, t))
    SF, ATM = EntropyEOS.FLAG_POL_S_FLOORED, EntropyEOS.FLAG_POL_ATMOSPHERE
    cold(f) = (g = unsafe_trunc(UInt32, f); g & SF != 0 && g & ATM == 0 ? one(T) : zero(T))
    atm(f) = unsafe_trunc(UInt32, f) & ATM != 0 ? one(T) : zero(T)
    println("# case=$(kw["case"]) cells=$(nleaves(forest) * N^3) dt=$dt ",
            "threads=$(Threads.nthreads())")
    println("step\tt\tseconds\ts_floors\tcold_s_floored\tatmosphere\tstage_policy_flags")
    prev = nothing
    for k in 1:kw["steps"]
        MHD.clear_tally!(p)
        integ = MHD.mhd_integrator(p, u, t, t + dt, 1; alias_u0=true, reuse=prev)
        el = @elapsed MHD.with_step_cache(() -> IRK.solve!(integ), p)
        prev = integ
        t += dt
        tal = MHD.ohm_tally(p)
        if k % kw["every"] == 0 || k == 1
            P = MHD.update_primitives!(p, u, t)
            nsf = mesh_mapreduce(cold, +, zero(T), P; vars=MHD.P_FLAGS)
            na = mesh_mapreduce(atm, +, zero(T), P; vars=MHD.P_FLAGS)
            println(k, '\t', t, '\t', el, '\t', tal.s_floors, '\t', round(Int, nsf), '\t',
                    round(Int, na), "\t0x", string(tal.policy_flags; base=16))
            flush(stdout)
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && s_floor_main(ARGS)
