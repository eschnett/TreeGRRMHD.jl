# Step 10 (rung 4): the TOV star in the Cowling approximation, on `Analytic(TOVMetric)`
# ("Validation ladder" 4, "Gauge for the TOV milestones" in `CODE.md`; `src/tov.jl`,
# `src/cases/tov.jl`). First the star itself, pointwise: the host solve, the tabulated
# metric and the polytrope. The scheme's tests follow.

const TV = TreeGRRMHD
const TV_STAR = (; K=100.0, Γ=2.0, ρ_c=1.28e-3)

tv3(x) = round(x; sigdigits=3)

@testset "The TOV solve converges at fourth order to the standard K = 100, Γ = 2 star" begin
    # Guards `tov_solve` (`src/tov.jl`): a wrong right-hand side in the enthalpy form, a
    # wrong centre limit of `dμ/dη`, or a wrong matching to the exterior would give another
    # star, and a non-smooth unknown at the centre would cost the order. Against the
    # literature's model (M = 1.400, areal R = 9.586, isotropic R̄ = 8.125; Font et al.
    # 2002, PRD 65, 084024, §IV) and against itself under step halving.
    sols = [tov_solve(; TV_STAR..., steps=n) for n in (250, 500, 1000, 2000)]
    ref = tov_solve(; TV_STAR..., steps=16000)
    errs = [abs(s.M - ref.M) for s in sols]
    rates = [log2(errs[i] / errs[i + 1]) for i in 1:(length(errs) - 1)]
    @info "TOV solve: M = $(ref.M), R = $(ref.R), R̄ = $(ref.R̄); mass errors " *
          "$(tv3.(errs)), rates $(round.(rates; digits=2))"
    @test all(r -> 3.7 < r < 4.5, rates)
    @test isapprox(ref.M, 1.400; rtol=1e-3)
    @test isapprox(ref.R, 9.586; rtol=1e-3)
    @test isapprox(ref.R̄, 8.125; rtol=1e-3)
    # The isotropic radius is the exterior's at the surface: `R = R̄ (1 + M/(2R̄))²`.
    @test isapprox(ref.R, ref.R̄ * (1 + ref.M / (2 * ref.R̄))^2; rtol=1e-14)
    @test_throws ArgumentError tov_solve(; K=100, Γ=1, ρ_c=1e-3)
end

@testset "The polytrope is EntropyEOS's ideal gas at the polytropic entropy" begin
    # Guards the star's thermodynamics against the gas the scheme evolves: `src/tov.jl`
    # writes `p = K ρ^Γ`, `e = ρ + p/(Γ − 1)` in closed form, and the case runs on
    # `IdealGasEOS` at `polytropic_entropy(eos, K)`; another `K`, `Γ` or `ε` would make the
    # initial data out of equilibrium by the difference.
    c = tov_case(Float64)
    eq = c.fields.w.metric.eq
    s = c.fields.w.s
    for ρ in (1e-9, 1e-6, 1e-4, 1.28e-3)
        pt = TV.EntropyEOS.evaluate(c.eos, ρ, s, 0.5, NaN)
        p = eq.K * ρ^eq.Γ
        @test isapprox(p, pt.p; rtol=1e-13)
        @test isapprox(ρ + p / (eq.Γ - 1), ρ * (1 + pt.U); rtol=1e-13)
        @test isapprox(1 + pt.U + pt.p / ρ, TV.polytrope_enthalpy(eq, ρ); rtol=1e-14)
        # `polytrope_state` inverts the enthalpy: to rounding in `h − 1`, which carries
        # `ρ` (relative `ε/(h − 1)`, the cancellation of `h = 1 + 2Kρ` at low density).
        h = TV.polytrope_enthalpy(eq, ρ)
        st = TV.polytrope_state(eq, h)
        @test isapprox(st.ρ, ρ; rtol=8 * eps() / (h - 1))
        @test isapprox(st.p, p; rtol=16 * eps() / (h - 1))
    end
end

@testset "The tabulated metric is fourth order in its knots and continuous at the surface" begin
    # Guards `TOVMetric`: a Hermite interpolant with wrong derivative data (the chain rule
    # from `η` to `u = r̄²`) is second order or worse; an exterior that does not continue
    # the table puts a kink or a jump into `ψ` or `α` at the surface, which the scheme sees
    # as a source of momentum there.
    sol = tov_solve(; TV_STAR...)
    fine = TOVMetric(sol; NK=2049)
    us = range(0, 1.1 * sol.R̄^2, 4001)
    err(NK) = (m = TOVMetric(sol; NK);
               maximum(maximum(abs.(TV.tov_potentials(m, u) .- TV.tov_potentials(fine, u)))
                       for u in us))
    errs = [err(NK) for NK in (33, 65, 129)]
    rates = [log2(errs[i] / errs[i + 1]) for i in 1:2]
    @info "TOV metric table: errors at NK = 33, 65, 129: $(tv3.(errs)), rates " *
          "$(round.(rates; digits=2)); $(sizeof(TOVMetric(sol))) bytes at NK = 129"
    @test all(r -> 3.7 < r < 4.3, rates)
    @test errs[3] < 3e-10
    m = TOVMetric(sol)
    @test isbits(m) && isbits(Analytic(m))
    # Across the surface: the values to rounding, the first derivatives (from the TOV
    # equations inside, the closed form outside) to the table's accuracy.
    u_s = sol.R̄^2
    inner, outer = TV.tov_potentials(m, prevfloat(u_s)), TV.tov_potentials(m, u_s)
    @test all(isapprox.(inner, outer; rtol=1e-14))
    δ = 1e-4 * u_s
    d_in = (TV.tov_potentials(m, u_s - δ) .- TV.tov_potentials(m, u_s - 2δ)) ./ δ
    d_out = (TV.tov_potentials(m, u_s + 2δ) .- TV.tov_potentials(m, u_s + δ)) ./ δ
    @test all(isapprox.(d_in, d_out; rtol=1e-3))
end

@testset "The TOV metric solves Einstein's equations with the star's matter" begin
    # Guards the metric and the density together. For `ds² = −α² dt² + ψ⁴ δ`, static, with
    # the perfect fluid at rest, Einstein's equations are the Hamiltonian constraint
    # `∇²ψ = −2π ψ⁵ e` and the lapse's `∇²(αψ) = 2π αψ⁵ (e + 6p)` (the isotropy of the
    # stress is the rest), `∇² = 6 ∂_u + 4u ∂²_u` on functions of `u = r̄²`; `e` and `p`
    # from `tov_density` (the polytrope at `h = α_s/α`). The `u`-derivatives by ForwardDiff
    # of the scalar interpolant, so the residual is its second-derivative error, `O(Δu²)`;
    # outside, the exterior is vacuum to rounding. (SpacetimeMetrics' `EinsteinTensor`
    # agrees to 1.8e−6 of `8πρ_c` inside and 2.3e−17 outside, measured in step 10, but
    # through the 4-metric its nested dual numbers cost 44 s to compile.)
    # Also: `dmetric` is finite at the centre, a vertex of a symmetric mesh, where it
    # vanishes by symmetry.
    sol = tov_solve(; TV_STAR...)
    d(f) = u -> ForwardDiff.derivative(f, u)
    lap(f, u) = 6 * d(f)(u) + 4u * d(d(f))(u)
    scale = 2π * 1.28e-3
    function residuals(m)
        ψ(u) = TV.tov_potentials(m, u)[1]
        αψ(u) = prod(TV.tov_potentials(m, u))
        worst, worst_out = 0.0, 0.0
        for u in vcat(range(0.05, 0.98 * m.R̄^2, 23), range(1.02 * m.R̄^2, 400, 9))
            ψ₀, α₀ = TV.tov_potentials(m, u)
            st = TV.polytrope_state(m.eq, TV.polytrope_enthalpy(m.eq, TV.tov_density(m, u)))
            rH = abs(lap(ψ, u) + 2π * ψ₀^5 * st.e) / scale
            rL = abs(lap(αψ, u) - 2π * α₀ * ψ₀^5 * (st.e + 6 * st.p)) / scale
            u < m.R̄^2 ? (worst = max(worst, rH, rL)) : (worst_out = max(worst_out, rH, rL))
        end
        return worst, worst_out
    end
    res = [residuals(TOVMetric(sol; NK)) for NK in (65, 129, 257)]
    inside = first.(res)
    rates = [log2(inside[i] / inside[i + 1]) for i in 1:2]
    @info "TOV metric: field-equation residual / 2πρ_c at NK = 65, 129, 257: inside " *
          "$(tv3.(inside)) (rates $(round.(rates; digits=2))), outside $(tv3(maximum(last.(res))))"
    # Second order: a cubic Hermite interpolant's second derivative. The scheme reads the
    # first derivatives only, which are third order.
    @test all(r -> 1.8 < r < 2.6, rates)
    @test inside[2] < 3e-5
    @test maximum(last.(res)) < 1e-12
    m = TOVMetric(sol)
    g, dg = TV.SpacetimeMetrics.dmetric(m, SVector(0.0, 0.0, 0.0, 0.0))
    @test all(isfinite, dg) && maximum(abs, dg) == 0
    # `dmetric` (ForwardDiff through the table) against central differences.
    p = SVector(0.0, 2.0, -1.0, 3.0)
    _, dg = TV.SpacetimeMetrics.dmetric(m, p)
    for j in 2:4
        e = SVector(ntuple(k -> k == j ? 1e-5 : 0.0, 4))
        fd = (TV.SpacetimeMetrics.metric(m, p + e) - TV.SpacetimeMetrics.metric(m, p - e)) / 2e-5
        @test maximum(abs, dg[:, :, j] - fd) < 1e-9
    end
end
