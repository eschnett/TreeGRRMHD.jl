# Rung 1b, the atmosphere: a blast into a near vacuum, a density ratio of
# 10⁸, which the reset has to hold ("Floors, the atmosphere and the
# magnetosphere" in `CODE.md`; step 6 of `PLAN.md`), and the comparison of
# the two tableaus where they differ most — a strong rarefaction into a
# vacuum, where only an SSP explicit part keeps the density positive.

using StaticArrays: SVector

const AT = TreeGRRMHD

# Every number the claims read, from one run.
function at_run(case; tableau=nothing, M=200)
    kw = tableau === nothing ? (;) : (; tableau)
    u0 = nothing
    r = evolve!(case, Val(1); N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc, kw...)
    u0 = exact_state(r.p, case, 0.0)
    pr = profile_1d(r.p, r.u, r.t)
    t = (; stages=sum(c -> collect(c.tally.bits), r.chunks),
         policy=sum(c -> c.tally.policy, r.chunks),
         policy_flags=reduce(|, (c.tally.policy_flags for c in r.chunks)),
         resets=sum(c -> c.tally.resets, r.chunks),
         step_resets=sum(c -> c.tally.step_resets, r.chunks),
         injected_D=sum(c -> c.tally.injected_D, r.chunks),
         injected_τ=sum(c -> c.tally.injected_τ, r.chunks),
         maxrec=maximum(c -> c.tally.max_recoveries, r.chunks))
    W = maximum(1 ./ sqrt.(1 .- sum(pr.v .^ 2; dims=2)))
    return (; r, pr, t, W, drift=conserved_totals(r.p, r.u) .- conserved_totals(r.p, u0),
            scales=conserved_scales(r.p, u0), nsteps=r.nsteps)
end

@testset "A stage with no rest mass has no conductor" begin
    # Guards `magnetization` at `D ≤ 0` (proposed in step 6): a stage's
    # predictor has seen no limiter, and next to a near vacuum its `D` can be
    # negative; `B²/D` then handed the cutoff's taper the logarithm of a
    # negative number, and step 6's magnetized atmosphere tube stopped with a
    # `DomainError`. No rest mass is no conductor: `μ = ∞`, σ = 0, the stage
    # the vacuum's.
    m = flat_metric(Float64)
    B = SVector(0.0, 1e-3, 0.0)
    @test magnetization(m, B, -1e-9) == Inf
    @test magnetization(m, B, 0.0) == Inf
    @test isnan(magnetization(m, B, NaN))
    model = MagnetizationCutoff(Constant(1e3), 10)
    eos = shock_gas(Float64, 5 // 3)
    pol = AT.EntropyEOS.default_policy(eos, 1e-8)
    opts = AT.EntropyEOS.Con2PrimOptions{Float64}()
    cons★ = (-1e-9, 0.0, 0.0, 0.0, 6e-7, -5e-10, -2e-8, 0.0, 0.0, 0.0)
    E★ = SVector(0.0, 0.0, 1e-5)
    E, prim, res, nrec = solve_ohm_stage(eos, opts, pol, model, m, cons★, E★, B, 1e-3,
                                         (NaN, NaN, NaN))
    @test res.flags == OHM_SIGMA_ZERO && E === E★
    # The predictor's recovery, repaired to the atmosphere and flagged as
    # such; the EOS's own warm start (the last slot) is NaN for an analytic
    # EOS.
    @test all(isfinite, prim[1:(AT.NPRIM - 1)])
    @test res.policy_flags & AT.EntropyEOS.FLAG_POL_ATMOSPHERE != 0
end

@testset "The atmosphere tube holds a 10⁸ density ratio, and the reset injects exactly what it accounts" begin
    # Guards the reset in both roles ("Floors, the atmosphere and the
    # magnetosphere"): a gas at ρ = p = 1 expands into the policy's own
    # atmosphere at `ρ_atm = 10⁻⁸` (`atmosphere_tube_case`), at up to W ≈ 5,
    # hydrodynamically (`B = 0`, `IdealConductor`) and in the configuration
    # of a star with a magnetosphere — a field `Bʸ = 10⁻³` that dominates the
    # atmosphere (`B²/ρ = 100`), and a conductivity of 10³ that the
    # magnetization cutoff takes to zero there (`μ_max = 10`, the measured
    # safe value). The reset must keep every cell at or above the atmosphere,
    # and leave no non-finite value; and since only the step limiter's writes
    # reach the state, the totals of `√γ D` and `√γ τ` — whose boundary
    # fluxes vanish (both far states at rest, `E = 0`) — must change by
    # exactly what those writes injected (`ohm_tally`), to rounding. A stage
    # limiter whose writes leaked into the state, or a write the tally
    # missed, breaks the equality. A predictor `u★` next to the vacuum can
    # carry a negative `D`; EntropyEOS reports that recovery as non-finite and
    # repairs it, inside the stage, where it stays — so the flag may appear
    # among the stages' policy flags, but never a non-finite number in the
    # state.
    for (name, By, model) in (("hydrodynamic", 0.0, IdealConductor()),
                              ("magnetized, cutoff μ_max = 10", 1e-3,
                               MagnetizationCutoff(Constant(1e3), 10)))
        a = at_run(atmosphere_tube_case(Float64; By=By, model=model, roots=20))
        t = a.t
        @info "Atmosphere tube, $name: min ρ $(minimum(a.pr.ρ)), max W $(a.W); stage bits " *
              "$(t.stages), $(t.policy) repaired stages (flags $(t.policy_flags)), " *
              "$(t.resets) stage and $(t.step_resets) step resets; injected rest mass " *
              "$(t.injected_D), energy $(t.injected_τ); totals moved by " *
              "$(a.drift[AT.U_D]), $(a.drift[AT.U_TAU]); at most $(t.maxrec) recoveries"
        @test all(isfinite, a.r.u)
        @test minimum(a.pr.ρ) ≥ 1e-8 * (1 - 1e-12)
        @test t.step_resets > 0                          # the reset holds the atmosphere
        @test t.stages[5] == 0                           # no OHM_FALLBACK
        for (v, inj) in ((AT.U_D, t.injected_D), (AT.U_TAU, t.injected_τ))
            @test abs(a.drift[v] - inj) ≤ 8 * eps() * a.scales[v] * a.nsteps
        end
    end
end

@testset "SSP3(4,3,3) keeps a rarefaction into a near vacuum positive, where ARS(4,4,3) does not" begin
    # Guards the default tableau, decided in step 6 ("The partition"): the
    # explicit part of SSP3(4,3,3) is SSPRK(3,3), a convex combination of
    # forward-Euler steps, and the light-cone HLLE step with MC is
    # positivity-preserving at `cfl = 2/5`; ARS(4,4,3)'s explicit part is not
    # SSP. Here a gas at ρ = p = 1 expands into a medium at 10⁻⁸ above an
    # atmosphere at 10⁻¹⁰ (so no reset need fire): under SSP3(4,3,3) no
    # density undershoots the medium and the reset never writes; under
    # ARS(4,4,3) the front undershoots fiftyfold and the reset writes a hundred
    # cells. In the magnetized atmosphere (`Bʸ = 10⁻³`, `IdealConductor`) the
    # same undershoots, repaired where the field's energy dwarfs the fluid's,
    # inject orders of magnitude more mass than the box holds.
    case = shock_tube_case(Float64; Γ=5 // 3, left=(ρ=1, p=1, v=(0, 0, 0), B=(0, 0, 0)),
                           right=(ρ=1 // 10^8, p=1 // 10^12, v=(0, 0, 0), B=(0, 0, 0)),
                           ρ_atm=1 // 10^10, roots=20)
    ssp = at_run(case; tableau=IRK.IMEXSSP3433())
    ars = at_run(case; tableau=IRK.ARS443())
    @info "Rarefaction into 10⁻⁸: min ρ $(minimum(ssp.pr.ρ)) (SSP3), $(minimum(ars.pr.ρ)) (ARS); " *
          "resets $(ssp.t.resets)/$(ssp.t.step_resets) and $(ars.t.resets)/$(ars.t.step_resets)"
    @test minimum(ssp.pr.ρ) ≥ 1e-8 * (1 - 1e-6)
    @test ssp.t.resets == 0 && ssp.t.step_resets == 0
    # Measured 2.1e-10 and 152/114.
    @test minimum(ars.pr.ρ) ≤ 1e-9
    @test ars.t.step_resets > 0

    mag = atmosphere_tube_case(Float64; By=1e-3, model=IdealConductor(), roots=20)
    ssp = at_run(mag; tableau=IRK.IMEXSSP3433())
    ars = at_run(mag; tableau=IRK.ARS443())
    @info "Magnetized ideal atmosphere: injected rest mass $(ssp.t.injected_D) (SSP3), " *
          "$(ars.t.injected_D) (ARS); max W $(ssp.W), $(ars.W)"
    # Measured 1.6e-9 against 2.6e4 (the box holds 0.5).
    @test abs(ssp.t.injected_D) ≤ 1e-6
    @test ars.t.injected_D ≥ 1
end
