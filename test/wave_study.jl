# The 1D convergence study of `wave_tests.jl`, which `reconstruction_tests.jl`
# runs under the higher-order methods: one run per resolution, the L1 errors
# against the exact solution, the rates and the conservation of the totals. A
# helper, not a test file (moved out of `wave_tests.jl` on 2026-11-08):
# `runtests.jl` includes it before the files that use it.

using StaticArrays: SVector

const WV = TreeGRRMHD
# Total cells per run: `roots = M/10` blocks of `N = 10`.
const WV_MS = (50, 100, 200, 400)

wv_rate(ms, errs) = WV.convergence_rate(1 ./ collect(ms), collect(errs))

# One run per resolution, in four chunks, with everything the claims read.
function wv_study(make_case, t_end_of; limiter=:mc, T=Float64, Ms=WV_MS)
    map(Ms) do M
        case = make_case(T, M ÷ 10)
        t_end = t_end_of(case)
        r = evolve!(T, case, Val(1); N=10, t_end=t_end, chunk=t_end / 4, limiter=limiter)
        u0 = exact_state(r.p, case, zero(T))
        uex = exact_state(r.p, case, r.t)
        # The state's ideal projection: one implicit stage of the case's own
        # conductivity on the final state — for `IdealConductor`, `E = −v × B`
        # of its ideal recovery; for `Vacuum`, the state itself.
        proj = copy(r.u)
        ohm_solve!(proj, r.u, one(T), r.p, r.t)
        (; M, r.nsteps, ncells=M, e=l1_errors(r.p, r.u, uex), ep=l1_errors(r.p, proj, uex),
         drift=conserved_totals(r.p, r.u) .- conserved_totals(r.p, u0),
         scales=conserved_scales(r.p, u0), tallies=[c.tally for c in r.chunks])
    end
end

# The evolved variables the claims name, as (label, accessor) pairs.
const WV_MATTER = (("D", e -> e.U[WV.U_D]), ("S_x", e -> e.U[WV.U_S]),
                   ("S_y", e -> e.U[WV.U_S + 1]), ("S_z", e -> e.U[WV.U_S + 2]),
                   ("τ", e -> e.U[WV.U_TAU]), ("D Yₑ", e -> e.U[WV.U_DYE]),
                   ("D s", e -> e.U[WV.U_DS]))
const WV_FIELD = (("𝔈ʸ", e -> e.U[WV.U_E + 1]), ("𝔈ᶻ", e -> e.U[WV.U_E + 2]),
                  ("A_y", e -> e.A[2]), ("A_z", e -> e.A[3]))

function wv_rates(rs, vars; field=:e)
    return map(vars) do (name, get)
        errs = [get(getfield(r, field)) for r in rs]
        name => (rate=wv_rate([r.M for r in rs], errs), errs=errs,
                 decreasing=issorted(errs; rev=true))
    end
end

# Every conserved total constant to rounding: a few ulp of its own scale per
# step. 8 is TreeHydro's round number above what `nsteps` rounding errors of
# one ulp can reach.
function wv_conservation(rs)
    worst = 0.0
    for r in rs, v in (WV.U_D, WV.U_S, WV.U_S + 1, WV.U_S + 2, WV.U_TAU, WV.U_DYE, WV.U_DS)
        bound = 8 * eps() * r.scales[v] * r.nsteps
        @test abs(r.drift[v]) ≤ bound
        r.scales[v] > 0 && (worst = max(worst, abs(r.drift[v]) / (eps() * r.scales[v] * r.nsteps)))
    end
    return worst
end
