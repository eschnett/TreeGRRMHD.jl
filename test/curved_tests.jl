# Step 9 (rung 3): analytic curved backgrounds. The Wald solution (`Vacuum`) and the
# magnetized Bondi flow (`IdealConductor`) on a Schwarzschild hole in Kerr-Schild
# coordinates, on the octant `[0, 4]³` with the hole at its corner, Dirichlet faces from
# the solution and the interior `r < 1` replaced by it ("Validation ladder" 3, "The
# interior" in `CODE.md`; `src/cases/kerr_schild.jl`, `src/interior.jl`). The stationarity
# study to `t = 50M` is `bin/curved_stationarity.jl`'s; the suite asserts the
# truncation error of the right-hand side and the drift after `t = 1/2 M`, at `h = M/4`
# and `M/8`.

const CV = TreeGRRMHD
const CV_CASES = (("Wald", wald_case(Float64)), ("magnetized Bondi", bondi_case(Float64)))

# A fresh problem over the case's own mesh of `N` cells per block.
function cv_problem(case, N)
    forest = Forest{Float64}(case.roots; N=N, periodic=case.periodic, extents=case.extents)
    f = mhd_fields(Float64, forest; G=3)
    return MHDProblem(f; eos=case.eos, opts=case.opts, proj=case.proj, pol=case.pol,
                      model=case.model, provider=case.provider, limiter=:mc,
                      boundary=case_boundary(case), ξ=CV.default_ξ(case),
                      interior=CV.interior_target(case.interior, case))
end

cv3(x) = round(x; sigdigits=3)

@testset "Michel's flow is the transonic solution, regular across the horizon" begin
    # Guards the matter of both cases (`MichelFlow`, `michel_state`): a root on the wrong
    # branch of Bernoulli's equation (the subsonic one inside `r_c`, or the wind), a sign in
    # the Kerr-Schild velocity, or the `0/0` of the naive `u_r` at the horizon would give a
    # flow that is not stationary, or not finite at `r = 2M`. Pointwise, against the
    # definitions: Bernoulli's residual, the mass flux, `u·u = −1` in the 4-metric, the
    # sonic point, and the flow supersonic inside `r_c` and subsonic outside.
    c = CV_CASES[1][2]
    f = c.fields.w.flow
    ks = c.provider.metric
    worst_F, worst_uu, worst_ur = 0.0, 0.0, 0.0
    for r in (0.25, 0.5, 1.0, 1.5, 2.0, 2.0 + 1e-12, 3.0, 5.0, 7.5, 8.5, 12.0)
        ρ = michel_density(f, r)
        F, _ = CV.michel_residual(f, r, log(ρ))
        worst_F = max(worst_F, abs(F) / f.C₂)
        θ = f.K * ρ^(1 / f.n)
        h = 1 + (f.n + 1) * θ
        u = f.C₁ / (r^2 * ρ)
        c² = (h - 1) / (f.n * h)
        V² = u^2 / (1 - 2 / r + u^2)
        @test r < 2 || (r < f.r_c ? V² > c² : V² < c²)
        x = r * SVector(2.0, 3.0, 6.0) / 7
        m = point_metric(c.provider, 0.0, x)
        ρ2, v = michel_state(f, m, x)
        @test ρ2 ≈ ρ rtol = 1e-14
        W = 1 / sqrt(1 - CV.dot_up(m, v, v))
        U = SVector(W / m.α, (W * (v - m.β / m.α))...)
        g = SpacetimeMetrics.metric(ks, SVector(0.0, x...))
        worst_uu = max(worst_uu, abs(U' * g * U + 1))
        worst_ur = max(worst_ur, abs(U[2:4]' * x / r + u) / u)
    end
    @info "Michel's flow: Bernoulli residual $(cv3(worst_F)) of C₂, |u·u + 1| $(cv3(worst_uu)), " *
          "uʳ to $(cv3(worst_ur))"
    # Measured: 1.4e-13, 1.0e-14 and 4.6e-15 on the laptop (M3), the last 1.0e-14 on an EPYC
    # 7532 (ForwardDiff's metric and `exp` round differently).
    @test worst_F ≤ 5e-13
    @test worst_uu ≤ 5e-14
    @test worst_ur ≤ 5e-14
    @test michel_density(f, f.r_c) ≈ f.ρ_c rtol = 1e-7
end

@testset "The fields are the curl of the potentials the cases give" begin
    # Guards each case's pair `fields`/`potential` ("Where the fields live" in `CODE.md`):
    # the scheme's `𝔅` is the curl of the potential and the initial `U` holds the case's
    # `B`, so a mismatch — a monopole whose potential is not its field, a missing `√γ` —
    # would start the run off the solution. A centred difference of `A` at `δ = 10⁻⁴`
    # against `√γ B`, at points on both sides of the horizon.
    for (name, c) in CV_CASES
        worst = 0.0
        for x in (SVector(0.6, 0.7, 0.8), SVector(1.5, 0.3, 1.0), SVector(3.0, 2.5, 0.5))
            δ = 1e-4
            dA(i) = (c.potential(x + δ * SVector(ntuple(k -> k == i ? 1.0 : 0.0, 3)), 0.0)[1] -
                     c.potential(x - δ * SVector(ntuple(k -> k == i ? 1.0 : 0.0, 3)), 0.0)[1]) / 2δ
            J = hcat(dA(1), dA(2), dA(3))                    # J[k, i] = ∂ᵢ A_k
            curl = SVector(J[3, 2] - J[2, 3], J[1, 3] - J[3, 1], J[2, 1] - J[1, 2])
            m = point_metric(c.provider, 0.0, x)
            B = c.fields(x, 0.0)[6]
            worst = max(worst, maximum(abs, curl - m.sqrtγ * B) / maximum(abs, curl))
            @test c.potential(x, 0.0)[2] == 0
        end
        @info "$name: curl A against √γ B to $(cv3(worst))"
        @test worst ≤ 1e-8
    end
end

@testset "The discrete right-hand side of each exact solution is second order outside the hole" begin
    # Guards the scheme in curved space ("Metric providers", "The right-hand side" in
    # `CODE.md`): every kernel reads the metric at its own point — cells, faces, edges,
    # vertices — from `Analytic(KerrSchild)`, and the stationary solution is a solution of the
    # continuum equations, so the right-hand side of its exact discrete state is the
    # truncation error, `O(h²)`. A wrong sign in a shift or lapse term, a source missing a
    # metric derivative, or a metric read at the wrong centering leaves an `O(1)` remainder.
    # Outside `r = 2M` (the interior's neighbourhood is not evolved), at `h = M/4` and `M/8`,
    # which near the horizon is not yet the asymptotic regime. Measured (2026-11-02): Wald
    # `D` 1.93, `S` 1.66–1.86, `τ` 1.88, `𝔈` 1.89–1.91, `A` 1.92–2.10; Bondi `D` 1.92, `S`
    # 1.87, `τ` 1.78, `𝔈` 1.90, `A` 1.85. Outside `2.5M` every one is 1.9–2.0.
    for (name, c) in CV_CASES
        res = map((8, 16)) do N
            p = cv_problem(c, N)
            u = exact_state(p, c, 0.0)
            du = similar(u)
            mhd_rhs!(du, u, p, 0.0)
            outside_l1(p, du, 2.0)
        end
        rate(f) = log2(f(res[1]) / f(res[2]))
        rU = ntuple(v -> rate(r -> r.U[v]), 10)
        rA = ntuple(j -> rate(r -> r.A[j]), 3)
        @info "$name: |du| outside 2M at h = M/4: D $(cv3(res[1].U[1])), τ $(cv3(res[1].U[5])); " *
              "rates D $(cv3(rU[1])), S $(map(cv3, rU[2:4])), τ $(cv3(rU[5])), 𝔈 $(map(cv3, rU[8:10])), " *
              "A $(map(cv3, rA))"
        @test rU[1] ≥ 1.8
        @test all(≥(1.55), rU[2:5])
        name == "Wald" && @test all(≥(1.8), rU[8:10])
        @test all(≥(1.75), rA)
    end
end

const CV_RUNS = Dict{Tuple{String,Int},Any}()
for (name, c) in CV_CASES, N in (8, 16)
    CV_RUNS[(name, N)] = evolve!(c, Val(3); N=N, t_end=1 // 2, limiter=:mc)
end

@testset "Both cases stay at the exact solution, converging at second order" begin
    # Guards the stationarity claim of rung 3 ("Validation ladder" 3), the suite's short
    # form: the L1 drift from the exact stationary solution after `t = M/2`, at `h = M/4`
    # and `M/8`. `bin/curved_stationarity.jl` runs the plan's `t = 50M` and three
    # resolutions on Symmetry. A scheme that is not consistent in curved space drifts by
    # `O(1)` or converges at first order; an interior that leaks, or a boundary that is not
    # the solution's, puts an `O(h)` error at the surface. And nothing is repaired: no reset,
    # no injected mass. Measured (2026-11-02): Wald `D` 2.48, `S` 1.86–1.98, `τ` 2.46, `𝔈`
    # 2.10–2.13, `𝔅` 1.76–1.82; Bondi `D` 1.73, `S` 1.89, `τ` 2.17, `𝔈` 1.89, `𝔅` 2.42 (the
    # `t = 50M` study: `CODE.md`, "Validation ladder" 3).
    for (name, c) in CV_CASES
        res = map((8, 16)) do N
            r = CV_RUNS[(name, N)]
            uref = exact_state(r.p, c, r.t)
            (; l=outside_l1(r.p, r.u .- uref, 0.0), fl=field_l1(r.p, r.u, uref, r.t, 0.0), r)
        end
        rate(f) = log2(f(res[1]) / f(res[2]))
        rU = ntuple(v -> rate(x -> x.l.U[v]), 10)
        rB = ntuple(d -> rate(x -> x.fl[d]), 3)
        @info "$name, t = M/2: drift at h = M/8: D $(cv3(res[2].l.U[1])), τ $(cv3(res[2].l.U[5])), " *
              "𝔈 $(map(cv3, res[2].l.U[8:10])), 𝔅 $(map(cv3, res[2].fl)); rates D $(cv3(rU[1])), " *
              "S $(map(cv3, rU[2:4])), τ $(cv3(rU[5])), 𝔈 $(map(cv3, rU[8:10])), 𝔅 $(map(cv3, rB))"
        @test rU[1] ≥ 1.6
        @test all(≥(1.6), rU[2:5])
        @test all(≥(1.6), rU[8:10])
        @test all(≥(1.6), rB)
        for x in res
            ch = x.r.chunks
            @test sum(k -> k.tally.resets + k.tally.step_resets, ch) == 0
            @test sum(k -> k.tally.injected_D, ch) == 0
        end
    end
end

@testset "∇·𝔅 stays at round-off and the charge converges to zero" begin
    # Guards ∇·𝔅 = 0 in curved space ("Where the fields live"): `𝔅` is the flat curl of `A`
    # whatever the metric, so its divergence is rounding after every step, with the
    # densitized `𝔅` and not `B`; and records the charge `√γ q = ∂ᵢ𝔈ⁱ` the right-hand side
    # derives (rung 3: "`|q|` from ∇·E is recorded"), which the exact fields do not have:
    # outside the interior it is the discrete divergence of a converging `𝔈`.
    for (name, c) in CV_CASES
        qs = map((8, 16)) do N
            r = CV_RUNS[(name, N)]
            @test max_div_B(r.p, r.u, r.t) ≤ 1e-15
            charge_norms(r.p, r.u, r.t, 2.0)
        end
        @info "$name, t = M/2, outside 2M: √γ q L1 $(cv3(qs[1].l1)), $(cv3(qs[2].l1)); max " *
              "$(cv3(qs[1].max)), $(cv3(qs[2].max)) at h = M/4, M/8"
        @test qs[2].l1 < qs[1].l1
    end
end

@testset "The interior holds the solution: position-only, after every step" begin
    # Guards the interior ("The interior" in `CODE.md`): the reset pastes the case's solution
    # into the owned points inside `r_in` — cells by their centre, edges and vertices by
    # their point — and nothing else, so after every step the inside is the exact state bit
    # for bit and the outside is the evolution's. A paste that read the state, or wrote
    # outside the ball, or ran after the reset's own fill, would break one or the other.
    c = CV_CASES[2][2]
    r = CV_RUNS[("magnetized Bondi", 8)]
    p = r.p
    uref = exact_state(p, c, r.t)
    nan = fill(NaN, length(r.u))
    paste_interior!(nan, p, r.t)
    inside = .!isnan.(nan)
    @test count(inside) > 0
    @test nan[inside] == uref[inside]
    # The potentials' owned low walls are the boundary condition's, `f(x_w) + (u − f)(x_r)`
    # (AMR-8, "Boundaries"), which the reset writes after the paste; where `x_r` is outside
    # the ball, a wall point inside holds the evolved offset. Every other point inside is
    # the solution (`==`: the wall rule also turns the solution's `−0.0` into `+0.0`).
    walls = falses(length(r.u))
    lw = Array(p.lowwall)
    for i in CV.SEG_A:CV.SEG_Φ
        s = p.sets[i]
        w = CV.statearray(walls, p.sets, i)
        for b in axes(w, 5), I in CartesianIndices(ntuple(_ -> s.forest.N, 3))
            CV.on_low_wall(Tuple(I), lw[b], Val(staggers(s))) && (w[Tuple(I)..., 1, b] = true)
        end
    end
    keep = inside .& .!walls
    @test count(inside .& walls) < count(inside) ÷ 4
    @test r.u[keep] == uref[keep]
    @test r.u[.!inside] != uref[.!inside]
    # Without an interior the evolution reaches the singularity's neighbourhood within a
    # few steps; with the interior in the step limiter alone the stages run there too.
    # Both measured in `bin/curved_stationarity.jl` ("The interior" in `CODE.md`).
end

@testset "Curved space: a step is a function of its start, with the interior and the step cache" begin
    # Guards the step cache's contract ("The step cache" in `CODE.md`) where it was never
    # exercised before step 9: with `√γ ≠ 1` every recovery's key is built as `· inv(√γ)`,
    # and the interior's paste is a pure function of position, so one integration in one
    # piece, the same in two pieces, and a restart from the saved state with a rebuilt
    # problem reproduce each other bit for bit. Dyadic `Δt = 1/64` at `h = M/4`. The
    # one-thread and four-thread suites print the same checksums.
    for (name, c) in CV_CASES
        p = cv_problem(c, 8)
        u0 = exact_state(p, c, 0.0)
        one_ = mhd_integrator(p, copy(u0), 0, 1 // 4, 16)
        with_step_cache(() -> CV.IRK.solve!(one_), p)
        a = mhd_integrator(p, copy(u0), 0, 1 // 8, 8)
        with_step_cache(() -> CV.IRK.solve!(a), p)
        q = CV.rebuild(p)
        b = mhd_integrator(q, copy(a.u), 1 // 8, 1 // 4, 8)
        with_step_cache(() -> CV.IRK.solve!(b), q)
        @test isequal(b.u, one_.u)
        @test one_.u != u0
        @info "$name: 16 steps of Δt = 1/64, checksum $(sum(abs, one_.u)) (one thread = four)"
    end
end
