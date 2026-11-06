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

@testset "The split-speed HLLE is consistent, and its fluid speeds lie in the light cone" begin
    # Guards `riemann_flux(::Val{:hlle_split}, …)` and `fluid_speeds` (step 10's measurement
    # of the Riemann speeds, "Riemann solver" in `CODE.md`): a wrong sign of the shift or of
    # `vᵈ`, a wrong `γ^{dd}`, or a mismatched speed pair per variable would make the flux
    # inconsistent (`F(U, U) ≠ F(U)`) or let a fluid signal outrun light. On the star's
    # metric and a boosted Kerr-Schild one, with velocities up to `0.9`.
    eos = TV_CASE.eos
    s = TV_CASE.fields.w.s
    m_star = point_metric(TV_CASE.provider, 0.0, SVector(3.0, -2.0, 1.0))
    m_ks = point_metric(Analytic(TV.SpacetimeMetrics.KerrSchild{Float64}(1.0)), 0.0,
                        SVector(2.0, 1.5, -1.0))
    worst = 0.0
    for m in (m_star, m_ks), v in (SVector(0.0, 0.0, 0.0), SVector(0.3, -0.2, 0.1),
                                   SVector(0.0, 0.0, 0.9) / sqrt(m.γ[3, 3]))
        prim = primitive_state(eos, m, 1e-3, v, s, 0.5)
        E, B = SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)
        st = (prim2con(eos, m, prim, E, B), prim, E, B)
        for d in 1:3
            F = physical_flux(m, st..., Val(d))
            H = riemann_flux(Val(:hlle_split), m, st, st, Val(d))
            worst = max(worst, maximum(abs.(F .- H) ./ max.(abs.(F), 1e-300)))
            c₋, c₊ = light_cone(m, Val(d))
            f₋, f₊ = TV.fluid_speeds(m, prim, Val(d))
            @test c₋ ≤ f₋ ≤ f₊ ≤ c₊
            if v == zero(v)
                c = sqrt(TV.prim_cs²(prim))
                @test f₊ ≈ -m.β[d] + m.α * c * sqrt(m.γu[d, d]) rtol = 1e-14
            end
        end
    end
    @test worst < 1e-14
end

# --- the scheme on the star ----------------------------------------------------------------

# This package's side of the coupling contract, tested before TreeGH has GH-1: a
# SpacetimeMetrics metric sampled at the points of the mesh, bound to the problem's block
# origins and spacings and to a time (a snapshot, as a producer's adapter is), read at the
# point's centering and local index. It computes the point's position as the kernels do
# (`point_x`), so it is the analytic provider bit for bit — GH-1's reference
# ("The coupling contract" in `CODE.md`).
struct TVSampled{M,O,H,T}
    metric::M
    origins::O
    spacings::H
    t::T
end
@inline tv_point(s::TVSampled, c::Val, n, b) =
    SVector(s.t, TV.point_x(@inbounds(s.origins[b]), @inbounds(s.spacings[b]), n, c)...)
TV.SpacetimeMetrics.metric(s::TVSampled, c::Val, n, b) =
    TV.SpacetimeMetrics.metric(s.metric, tv_point(s, c, n, b))
TV.SpacetimeMetrics.dmetric(s::TVSampled, c::Val, n, b) =
    TV.SpacetimeMetrics.dmetric(s.metric, tv_point(s, c, n, b))

tv_sampled(p::MHDProblem) =
    PointwiseMetric(TVSampled(p.provider.metric, p.origins, p.spacings, 0.0))

const TV_CASE = tov_case(Float64)
tv_boxes(f) = tov_boxes!(f, TV_CASE; levels=1)

@testset "A producer's pointwise metric is the analytic one, to rounding: $name" for
        (name, case, refine, exact) in (("Wald", wald_case(Float64), nothing, false),
                                        ("TOV star", TV_CASE, tv_boxes, true))
    # Guards this package's side of the coupling contract (`PointwiseMetric`,
    # `point_metric(prov, t, x, Val(C), n, b)`): a kernel that passes another centering's
    # staggers or index (the vertex metric of `∂ₜA`'s gradient, read from an edge kernel;
    # the edge metric of the gauge flux, read from a vertex kernel; a ghost cell's local
    # index), or a second derivative-index conversion, gives another metric somewhere, and
    # a run that differs from the analytic one at truncation level. Four steps of the same
    # integrator over the same state, with the step cache. The sampled adapter returns the
    # analytic `g` bit for bit, but the 3+1 split of it (`adm_decompose`, StaticArrays'
    # `muladd`) rounds as the context it is inlined into lets LLVM fuse it: Wald's shift,
    # near the singularity, moves `α` by 1.4e−14 at `r = 0.22` and the run by rounding
    # (measured in step 10). The star's metric is diagonal, and its run is the same bit for
    # bit.
    forest = Forest{Float64}(case.roots; N=8, periodic=case.periodic, extents=case.extents)
    refine === nothing || refine(forest)
    p = MHDProblem(mhd_fields(Float64, forest), case; limiter=:mc)
    q = TV.rebuild(p; provider=tv_sampled(p))
    u0 = exact_state(p, case, 0.0)
    run(r) = (integ = mhd_integrator(r, copy(u0), 0.0, 1 / 8, 4);
              with_step_cache(() -> TV.IRK.solve!(integ), r); integ.u)
    a, b = run(p), run(q)
    worst = maximum(i -> (r = staterange(p.sets, i);
                          maximum(abs, a[r] - b[r]) / max(maximum(abs, a[r] - u0[r]), 1e-300)),
                    eachindex(p.sets))
    @info "$name through the sampled pointwise metric: checksum $(sum(abs, b)); identical to " *
          "the analytic provider's: $(isequal(a, b)); largest difference per set, relative " *
          "to its change in the four steps: $(tv3(worst))"
    @test exact ? isequal(a, b) : worst < 1e-9
    @test maximum(abs, a - u0) > 0
    @test_throws ArgumentError point_metric(q.provider, 0.0, SVector(1.0, 1.0, 1.0))
end

module TVScript
include(joinpath(@__DIR__, "..", "bin", "tov_cowling.jl"))
end

@testset "The step-10 script is the run evolve! makes, bit for bit" begin
    # Guards the verbs (`prepare!`, `rhs!`, `stage_solve!`, `stage_limit!`, `step_limit!`,
    # `max_rate`, `initial_state!`, `MHDProblem(fields, case)`) against `evolve!`, which is
    # a composition of them: a verb that drops the metric, a step count or a chunk boundary
    # computed otherwise, or a record that changes the state would make the script another
    # run. `bin/tov_cowling.jl`'s `main`, two chunks, against `evolve!` on the same case and
    # boxes; the script's record (the central density per chunk) is taken between chunks.
    ρs = Float64[]
    s = TVScript.main(; N=8, levels=1, t_end=1, chunk=1 // 2, limiter=:mc,
                      record=(t, ρ) -> push!(ρs, ρ))
    r = evolve!(TV_CASE, Val(3); N=8, t_end=1, chunk=1 // 2, limiter=:mc, refine=tv_boxes)
    @info "The step-10 script: checksum $(sum(abs, s.u)), $(r.nsteps) steps; identical to " *
          "evolve!'s: $(isequal(s.u, r.u)); ρ_c per chunk $(ρs)"
    @test isequal(s.u, r.u) && s.t == r.t == 1
    @test length(ρs) == 2 && ρs[end] == TV.central_density(r.p, r.u, r.t)
end

@testset "The Cowling star holds still and conserves its rest mass" begin
    # Guards the case as a whole (`tov_case`, `tov_boxes!`): initial data out of
    # equilibrium with the metric, an atmosphere off the star's isentrope or not at rest,
    # or Dirichlet faces that are not the static exterior would move the central density
    # at first order in `t`; a leak at a coarse-fine face, a boundary or the reset would
    # change the rest mass by more than what the reset injects. On `h = 1` (the star
    # resolved by 8 cells in `R̄`), MC, to `t = 4M`: measured −3.3% in `ρ_c`, the
    # truncation error of so coarse a star (the study is `bin/tov_study.jl`'s); the rest
    # mass changes by exactly the reset's injected `D`, to rounding.
    r = evolve!(TV_CASE, Val(3); N=8, t_end=4, chunk=1, limiter=:mc, refine=tv_boxes)
    ρ0 = TV.central_density(r.p, exact_state(r.p, TV_CASE, 0.0), 0.0)
    ρ1 = TV.central_density(r.p, r.u, r.t)
    M0 = r.totals0[1]
    ΔM = conserved_totals(r.p, r.u)[1] - M0
    injected = sum(c -> c.tally.injected_D, r.chunks)
    @info "Cowling star, h = 1, t = 4M: ρ_c $(ρ0) → $(ρ1) ($(tv3((ρ1 - ρ0) / ρ0))); rest " *
          "mass $(M0), change $(ΔM), injected by the reset $(injected), difference " *
          "$(ΔM - injected); $(r.nsteps) steps, $(r.nblocks) blocks; checksum $(sum(abs, r.u))"
    @test abs(ρ1 - ρ0) / ρ0 < 0.05
    @test abs(ΔM - injected) < 1e-14 * M0
    @test abs(injected) < 1e-8 * M0
end
