# Rung 1b, resistive: the shock tube swept over the conductivity and the
# self-similar current sheet ("Validation ladder" 1 in `CODE.md`; step 6 of
# `PLAN.md`) — the implicit stage solve inside an evolution, from σ = 0 to the
# stiff limit, on discontinuous and on smooth data.

const RS = TreeGRRMHD

rs_run(model; M=200) = evolve!(resistive_tube_case(Float64, model; roots=M ÷ 10), Val(1); N=10,
                               t_end=2 // 5, limiter=:mc)

@testset "The resistive tube approaches the ideal solution monotonically as σ grows" begin
    # Guards the stiff limit of the scheme ("The partition", "The implicit
    # stage solve"): Palenzuela et al.'s and Dionysopoulou et al.'s Brio–Wu
    # tube (`resistive_tube_case`) at σ ∈ {0, 10, 10², 10³, 10⁶} on 200 cells.
    # At σ = 0 the field's jump leaves as light waves; as σ grows the solution
    # must move steadily toward the `IdealConductor` run of the same mesh —
    # PLAN.md's claim, the L1 distance decreasing monotonically — and at
    # σ = 10⁶ (`σΔt ≈ 2000`, the stage stiff) be within its rounding-free
    # distance of it, `O(1/σ)`. A stage solve that lost the stiff limit, or a
    # tableau that did not damp it (SSP3(4,3,3) is L-stable), would leave
    # σ = 10⁶ far from the ideal run or oscillating.
    ideal = rs_run(IdealConductor())
    pi_ = profile_1d(ideal.p, ideal.u, ideal.t)
    σs = (0.0, 10.0, 1e2, 1e3, 1e6)
    dist = Float64[]
    for σ in σs
        r = rs_run(σ == 0 ? Vacuum() : Constant(σ))
        pr = profile_1d(r.p, r.u, r.t)
        d = sum(abs, pr.B[:, 2] .- pi_.B[:, 2]) / 200
        dD = sum(abs, pr.U[:, RS.U_D] .- pi_.U[:, RS.U_D]) / 200
        push!(dist, d)
        bits = sum(c -> collect(c.tally.bits), r.chunks)
        maxrec = maximum(c -> c.tally.max_recoveries, r.chunks)
        @info "Resistive tube σ = $σ: L1(𝔅ʸ − ideal) $d, L1(D − ideal) $dD; stage bits $bits, " *
              "at most $maxrec recoveries"
        stages = 4 * 200 * r.nsteps
        @test sum(bits[1:4]) == stages                  # one branch bit per stage
        @test bits[5] == 0                              # no OHM_FALLBACK
        σ == 0 && @test bits[1] == stages               # OHM_SIGMA_ZERO
        σ > 0 && @test bits[1] == 0
        # Measured at most 6 (σ ≥ 10²) — step 3's Newton needed up to 10.
        @test maxrec ≤ 12
        @test all(c -> c.tally.policy == 0 && c.tally.step_resets == 0, r.chunks)
        @test all(isfinite, r.u)
    end
    @test issorted(dist; rev=true) && allunique(dist)
    # The approach is O(1/σ) once the stage is stiff: from 10³ to 10⁶ the
    # distance falls by 780 (measured: 1.34e-2 to 1.73e-5), a slope of 0.96.
    slope = log(dist[4] / dist[5]) / log(1e6 / 1e3)
    @info "Resistive tube: the distance falls as σ^-$(round(slope, digits=3)) from 10³ to 10⁶"
    @test 0.85 ≤ slope ≤ 1.05
    @test dist[5] ≤ 1e-4
end

# The current sheet at `M` cells from `t₀ = 1` for `t_end`.
function rs_sheet(M, t_end; B0=1)
    case = current_sheet_case(Float64; roots=M ÷ 10, B0=B0)
    r = evolve!(case, Val(1); N=10, t_end=t_end, limiter=:mc)
    pr = profile_1d(r.p, r.u, r.t)
    ex = profile_1d(r.p, exact_state(r.p, case, r.t), r.t)
    return (; M, r, pr, ex)
end

@testset "The current sheet converges at second order, to the erf profile's own accuracy" begin
    # Guards the resistive scheme on smooth data: Komissarov's (2007)
    # self-similar current sheet (`current_sheet_case`: `B₀ = 1`, `p = 50`,
    # `σ = 100`), where Ampère's stiff current and Faraday's law together
    # diffuse `Bʸ` with `η = 1/σ`.
    #
    # PLAN.md predicted an L1 rate ≥ 1.8 *against the erf profile*. That
    # profile solves the diffusion limit, not the full system: the field's
    # pressure moves the fluid (`|v| ≈ 2·10⁻⁴`, whose `v × B` is 1% of the
    # resistive `E`) and the displacement current lags the diffusion by
    # `O(1/σt)`, and together they leave it 1.1·10⁻³ of `B₀` (L1, per unit
    # length) from the system's solution — reached by 200 cells. So the
    # order is measured against the next finer run (amended in step 6), over
    # `t = 1 → 2` at 50, 100, 200 and 400 cells; the distance to the profile
    # is asserted at Komissarov's resolution and time (200 cells, `t = 9`);
    # and that this distance is the profile's and not the scheme's is shown by
    # weakening the field, which takes the fluid's response (quadratic in
    # `B₀`) out of it.
    runs = [rs_sheet(M, 1.0) for M in (50, 100, 200, 400)]
    diffs = map(1:3) do i
        a, b = runs[i], runs[i + 1]
        fine = [(b.pr.B[2j - 1, 2] + b.pr.B[2j, 2]) / 2 for j in 1:a.M]
        sum(abs, a.pr.B[:, 2] .- fine) * 3 / a.M / 3
    end
    rates = [log2(diffs[i] / diffs[i + 1]) for i in 1:2]
    @info "Current sheet, t = 1 → 2: L1 differences of Bʸ between successive meshes $diffs, " *
          "rates $(round.(rates, digits=3))"
    # Measured 2.53 and 2.58.
    @test all(≥(1.8), rates)
    erf_dist(run) = sum(abs, run.pr.B[:, 2] .- run.ex.B[:, 2]) * 3 / run.M / 3
    @info "Current sheet, t = 1 → 2: L1 distance to the erf profile " *
          "$(map(erf_dist, runs)) at 50…400 cells"

    komissarov = rs_sheet(200, 8.0)
    d = erf_dist(komissarov)
    @info "Current sheet, 200 cells, t = 1 → 9: L1 distance to the erf profile $d of B₀"
    # Measured 9.30e-4: Komissarov's "indistinguishable" holds to 0.1%.
    @test d ≤ 2e-3
    bits = sum(c -> collect(c.tally.bits), komissarov.r.chunks)
    stages = 4 * 200 * komissarov.r.nsteps
    @info "Current sheet, 200 cells to t = 9: stage bits $bits"
    @test bits[3] + bits[4] == stages && bits[5] == 0   # fixed point or Newton, no fallback
    @test all(c -> c.tally.policy == 0 && c.tally.step_resets == 0, komissarov.r.chunks)

    weak = rs_sheet(200, 1.0; B0=1 // 10)
    ratio = erf_dist(runs[3]) / (erf_dist(weak) / (1 // 10))
    @info "Current sheet, 200 cells, t = 1 → 2: the relative distance to the erf profile " *
          "falls $(round(ratio, digits=2))-fold at B₀ = 1/10"
    # Measured 5.0: what remains is the displacement current's `O(1/σt)`,
    # linear in `B₀`.
    @test ratio ≥ 3
end
