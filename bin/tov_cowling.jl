# Step 10 (rung 4): the TOV star in the Cowling approximation, as a script that owns the
# time loop ("The coupling contract", "Two scripts" in `CODE.md`). The metric is the star's
# own, `Analytic(TOVMetric)`, for the whole run; step 11's script is this one with TreeGH's
# lines added. `main` holds only the run: the mesh, the problem, the initial data, then per
# chunk the step count, the integrator and the record. It makes the run `evolve!` makes, bit
# for bit (`test/tov_tests.jl`), so that this file stays code that runs.
#
#     julia --project=. -t T bin/tov_cowling.jl [key=value...]
#
# Keys (`bin/tov_cowling_cli.jl`): `N` (cells per block, 16), `levels` (box levels around
# the star, 1), `L` (the box `[−L, L]³`, 16), `roots` (2), `t_end` (200), `chunk` (10),
# `limiter` (`ppm`), `out` (a file for the record, `t ρ_c` per chunk; default the terminal).

using TreeAMR
using TreeGRRMHD
const MHD = TreeGRRMHD
const IRK = MHD.IRK                                  # IMEXRungeKutta, as this package loads it

# IMEXRungeKutta's four callbacks. `p` carries the problem and the metric, so that a rebuilt
# problem reaches them; the limiters take IMEXRungeKutta's `(u, integ, p, t)`.
function f_exp!(du, u, p, t)
    MHD.prepare!(p.mhd, u, t; p.metric)              # the fill and the recovery
    MHD.rhs!(du, p.mhd, t; p.metric)                 # the explicit right-hand side
    return nothing
end
solve_imp!(U, u★, γΔt, p, t) = MHD.stage_solve!(U, u★, γΔt, p.mhd, t; p.metric)
stage_limit!(u, _, p, t) = MHD.stage_limit!(u, p.mhd, t; p.metric)
step_limit!(u, _, p, t) = MHD.step_limit!(u, p.mhd, t; p.metric)

function main(T=Float64; N=16, levels=1, L=16, roots=2, t_end=200, chunk=10, limiter=:ppm,
              record=(t, ρ_c) -> println(t, "  ", ρ_c))
    star = MHD.tov_case(T; K=100, Γ=2, ρ_c=1.28e-3, L, roots)
    forest = Forest{T}(star.roots; N, periodic=star.periodic, extents=star.extents)
    MHD.tov_boxes!(forest, star; levels)             # the static boxes around the star
    mhd = MHD.MHDProblem(MHD.mhd_fields(T, forest), star; limiter)
    u = statevector(MHD.evolved_sets(mhd))           # matter, A_d, Φ̃ (AMR-1)
    MHD.initial_state!(u, mhd, star, zero(T))
    p = (; mhd, metric=star.provider)                # Analytic(TOVMetric)
    cfl = T(MHD.default_cfl(limiter))
    t_end, chunk = T(t_end), T(chunk)
    nchunks = MHD.chunk_count(t_end, chunk)
    t, integ = zero(T), nothing
    for c in 1:nchunks
        t1 = c == nchunks ? t_end : c * chunk
        rate = MHD.max_rate(p.mhd, u, t; p.metric)   # the light cone's max λ/h
        nsteps = MHD.chunk_count((t1 - t) * 3 * rate, cfl)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=(t1 - t) / nsteps,
                         stage_limiter=stage_limit!, step_limiter=step_limit!,
                         partition=:even, alias_u0=true, reuse=integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
        t = t1
        record(t, MHD.central_density(p.mhd, u, t))
    end
    return (; u, t, p)
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "tov_cowling_cli.jl"))
    run_cli(main, ARGS)
end
