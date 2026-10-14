# Rung 1a: the staggered scheme on smooth problems in one dimension
# ("Validation ladder" in `CODE.md`; step 5 of `PLAN.md`). Two exact
# solutions, each run at four resolutions on the uniform periodic mesh:
#
#   * the vacuum electromagnetic wave (`Vacuum`): the Maxwell subsystem —
#     `𝔈` through the Riemann fluxes, `A` through the UCT EMF — over a resting
#     fluid it does not touch;
#   * the large-amplitude circularly polarized Alfvén wave (`IdealConductor`):
#     the whole coupled system, every implicit stage the ideal recovery.
#
# Every run is wall-clock, so each study runs once and every claim is made on
# its results — the order, the conservation of the totals and the implicit
# stages' outcomes. The initial data and the references are exact (cell
# averages by Gauss quadrature, point values of `A` and `Φ̃`), so the measured
# slope is the scheme's own. The limiter is MC, the plan's (and GRaM-X's)
# choice for these tests: a TVD limiter clips the smooth extrema, which costs
# little in L1.

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

@testset "The vacuum EM wave converges at second order, and conserves the totals" begin
    # Guards the Maxwell subsystem of the scheme ("Maxwell", "Riemann
    # solver", "The electromotive force for `A`" in `CODE.md`): `𝔈` updated
    # by the light-cone HLLE fluxes of the reconstructed `E` and `B`, `A` by
    # the UCT–HLL EMF on the edges, `B` the curl of `A`, and a resting fluid
    # carrying the field's energy and momentum in its totals. A reconstruction
    # that is first order, an EMF with the wrong upwinding, or a curl half a
    # cell off shows as a rate; a flux that is not the same on both sides of a
    # face, as a drift of the totals. PLAN.md predicted an L1 rate ≥ 1.9 with
    # MC, N = 50…400 (measured in step 5, below and in `CODE.md`).
    rs = wv_study((T, roots) -> em_wave_case(T; roots=roots), case -> one(eltype(case.𝔅̄)))
    rates = wv_rates(rs, (WV_FIELD..., WV_MATTER...))
    for (name, r) in rates
        @info "EM wave, MC: $name L1 errors $(r.errs), rate $(round(r.rate, digits=3))"
    end
    for (name, r) in rates
        name in ("S_y", "S_z") && continue           # zero, exactly: no y or z momentum
        @test r.rate ≥ 1.9
        @test r.decreasing
    end
    # The fluid does not move in y or z, and `A_x`, `Φ̃` stay zero: the wave is
    # transverse, so no EMF along x and no gauge flux arise.
    @test all(r -> r.e.U[WV.U_S + 1] == 0 && r.e.U[WV.U_S + 2] == 0, rs)
    @test all(r -> r.e.A[1] == 0 && r.e.Φ == 0, rs)
    worst = wv_conservation(rs)
    @info "EM wave: worst drift of a conserved total $(round(worst, digits=4)) ulp of its scale per step"
    # σ = 0: every stage is the vacuum's, `E = E★`, one recovery, no repair.
    for r in rs
        stages = 4 * r.ncells * r.nsteps            # SSP3(4,3,3): four stage solves a step
        bits = sum(t -> collect(t.bits), r.tallies)
        @test bits[1] == stages                      # OHM_SIGMA_ZERO
        @test sum(bits) == stages
        @test sum(t -> t.recoveries, r.tallies) == stages
        @test all(t -> t.policy == 0 && t.resets == 0, r.tallies)
    end
end

@testset "The circularly polarized Alfvén wave converges at second order under IdealConductor" begin
    # Guards the coupled system ("The implicit stage solve", "Time
    # integration"): the matter's fluxes with the field's stress in the
    # totals, the ideal recovery at every implicit stage writing `E = −v × B`
    # into `𝔈`, the guide field from the background `𝔅̄`, and the transverse
    # field from the curl of `A` — an exact nonlinear solution, at Del Zanna
    # et al.'s amplitude `η = 1`. PLAN.md predicted an L1 rate ≥ 1.8.
    #
    # The state's own `𝔈` is the exception, and the test says so. SSP3(4,3,3)
    # is not stiffly accurate: each step ends `−0.2844 Δt f` off the
    # equilibrium the stages imposed (`CODE.md`, "The partition"), so the
    # `𝔈` in `uⁿ⁺¹` is off `−v × B` by `O(Δt)`. It does not accumulate — the
    # next stage solve replaces it, and no right-hand side ever reads it — so
    # it converges at first order, and the field the state means, its ideal
    # projection, converges at second.
    rs = wv_study((T, roots) -> alfven_wave_case(T; roots=roots), alfven_period)
    second = wv_rates(rs, (WV_MATTER..., WV_FIELD[3], WV_FIELD[4]))
    projected = wv_rates(rs, WV_FIELD[1:2]; field=:ep)
    raw = wv_rates(rs, WV_FIELD[1:2])
    for (name, r) in second
        @info "Alfvén wave, MC: $name L1 errors $(r.errs), rate $(round(r.rate, digits=3))"
    end
    for ((name, r), (_, q)) in zip(projected, raw)
        @info "Alfvén wave, MC: $name projected L1 errors $(r.errs), rate " *
              "$(round(r.rate, digits=3)); the state's own: rate $(round(q.rate, digits=3))"
    end
    for (name, r) in (second..., projected...)
        @test r.rate ≥ 1.8
        @test r.decreasing
    end
    for (name, q) in raw
        # First order, and not worse: an offset that accumulated over the
        # steps would not converge at all.
        @test 0.7 ≤ q.rate ≤ 1.2
        @test q.decreasing
    end
    worst = wv_conservation(rs)
    @info "Alfvén wave: worst drift of a conserved total $(round(worst, digits=4)) ulp of its scale per step"
    # IdealConductor: every stage is the ideal recovery, one each, no repair.
    for r in rs
        stages = 4 * r.ncells * r.nsteps
        bits = sum(t -> collect(t.bits), r.tallies)
        @test bits[2] == stages                      # OHM_IDEAL
        @test sum(bits) == stages
        @test sum(t -> t.recoveries, r.tallies) == stages
        @test all(t -> t.policy == 0 && t.resets == 0, r.tallies)
    end
end

@testset "The scheme runs in Float32" begin
    # A smoke test, as `CODE.md` ("Precision and devices") makes `Float32`:
    # every kernel of the right-hand side, the stage solve and the reset in
    # the run's own type — a stray `Float64` constant would widen a result or
    # fail to compile — on the Alfvén wave, the case that exercises all of
    # them, for a tenth of a period.
    case = alfven_wave_case(Float32; roots=5)
    t_end = alfven_period(case) / 10
    r = evolve!(Float32, case, Val(1); N=10, t_end=t_end, limiter=:mc)
    @test eltype(r.u) === Float32
    @test all(isfinite, r.u)
    e = l1_errors(r.p, r.u, exact_state(r.p, case, r.t))
    @test e.A[2] isa Float32
    c64 = alfven_wave_case(Float64; roots=5)
    r64 = evolve!(Float64, c64, Val(1); N=10, t_end=Float64(t_end), limiter=:mc)
    e64 = l1_errors(r64.p, r64.u, exact_state(r64.p, c64, r64.t))
    @info "Alfvén wave, Float32 against Float64 after a tenth of a period: A_y L1 " *
          "$(e.A[2]) against $(e64.A[2])"
    # The truncation error dominates at N = 50; Float32 adds its rounding.
    @test abs(e.A[2] - e64.A[2]) ≤ 1e-2 * e64.A[2]
end
