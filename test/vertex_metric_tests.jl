# Step 10, round 2: this package's side of the coupling contract as Erik redesigned it on
# 2026-11-08 — the producer's metric crosses at its owned vertices as `(α, β, γ, K)`, and
# this package fills its vertex set `Mv` from it and interpolates to every centering
# (`src/vertex_metric.jl`; "The coupling contract", "Metric at cells, faces and edges" in
# `CODE.md`). Tested before TreeGH is loaded, through a sampled SpacetimeMetrics metric in
# TreeGH's `ADMMetric` form (`SampledADM`, GH-1's reference), against the same metric's
# `Analytic` provider. Uses `tov_tests.jl`'s `TV_CASE`, `tv_boxes` and `tv3`.

const VM = TreeGRRMHD

# The vertex metric of `provider`'s metric on `forest`, through the sampled adapter.
function vm_filled(forest, provider)
    vm = VertexMetricSet(Float64, forest; boundary=provider)
    adm = SampledADM(provider.metric, vm, 0.0)
    return vm, VM.vertex_metric!(vm, adm, 0.0)
end

# Two roots per dimension, the first refined: coarse-fine faces through the box.
function vm_forest(roots, extents, N)
    forest = Forest{Float64}(roots; N=N, periodic=(false, false, false), extents=extents)
    refine!(forest, [first(forest.leaves)])
    balance!(forest)
    return forest
end

# The largest difference between the vertex metric and the analytic one at every stored
# point of every centering the kernels read (`P`, `B[d]`, `A[j]`, `Φ̃`), and of the
# derivatives and `K` at the cells, by region: `region(x)` names each point's.
function vm_errors(forest, provider; region=x -> :all)
    f = mhd_fields(Float64, forest)
    vm, prov = vm_filled(forest, provider)
    err = Dict{Tuple{Symbol,Symbol},Float64}()
    put!(k, e) = (err[k] = max(get(err, k, 0.0), e))
    for (name, sets) in ((:cell, (f.P,)), (:face, f.B), (:edge, f.A), (:vertex, (f.Φ,)))
        for fs in sets
            C = staggers(fs)
            G = fs.G
            w = fs.work
            for b in axes(w, 5), k in axes(w, 3), j in axes(w, 2), i in axes(w, 1)
                n = (i - G[1], j - G[2], k - G[3])
                x = VM.point_x(vm.origins[b], vm.spacings[b], n, Val(C))
                r = region(x)
                m1 = point_metric(prov, 0.0, x, Val(C), n, b)
                m0 = point_metric(provider, 0.0, x)
                put!((name, r), max(abs(m1.α - m0.α), maximum(abs, m1.β - m0.β),
                                    maximum(abs, m1.γ - m0.γ)))
                if name === :cell
                    d1 = metric_gradient(prov, 0.0, x, Val(C), n, b)
                    d0 = metric_gradient(provider, 0.0, x)
                    put!((:dg, r), maximum(abs, d1.dg - d0.dg))
                    put!((:K, r), maximum(abs, d1.K - d0.K))
                end
            end
        end
    end
    return err, metric_fallbacks(prov, f), prov, f
end

vm_rates(e1, e2) = Dict(k => log2(e1[k] / e2[k]) for k in keys(e1) if e1[k] > 0)
vm_fmt(d; f=v -> round(v; digits=2)) =
    join(("$(k[1])$(k[2] === :all ? "" : "/$(k[2])") $(f(v))" for (k, v) in sort(collect(d))),
         ", ")

@testset "The vertex metric converges at fourth order at every centering on a smooth metric" begin
    # Guards `VertexMetric`'s interpolation (`src/vertex_metric.jl`): a stencil off by one
    # vertex, a weight with the wrong sign, a derivative not divided by the block's own
    # `h`, a wrong product rule in `gradient_from_adm`, or `Mv`'s prolongation too narrow
    # at a coarse-fine face would cost the order somewhere. On Kerr-Schild at `a = M/2`
    # over `[2, 6]³`, away from the hole, the first root refined, at `N = 12` and `24`
    # (spacings 1/3 and 1/6 on the coarse level), at every stored point of the cells,
    # faces, edges and vertices the kernels read, ghosts included. Measured 2026-11-08:
    # rates 4.5–4.7 at the cells, faces, edges, for `∂_j g_ab` and `K`; 5.9 at the
    # vertices (only the prolongated ghosts differ). The owned vertices are the analytic
    # provider's metric bit for bit (the same `adm_decompose`, the same `Metric3`).
    provider = Analytic(VM.SpacetimeMetrics.KerrSchild(1.0, 0.5))
    ext = ((2.0, 6.0), (2.0, 6.0), (2.0, 6.0))
    e1, fb1, _, _ = vm_errors(vm_forest((2, 2, 2), ext, 12), provider)
    e2, fb2, prov, f = vm_errors(vm_forest((2, 2, 2), ext, 24), provider)
    rates = vm_rates(e1, e2)
    @info "Vertex metric on Kerr-Schild a = M/2: errors at N = 24 " *
          "$(vm_fmt(e2; f=tv3)); rates $(vm_fmt(rates))"
    @test all(k -> rates[k] > 4.2, ((:cell, :all), (:face, :all), (:edge, :all), (:dg, :all),
                                    (:K, :all)))
    @test rates[(:vertex, :all)] > 5.5
    @test fb1 == fb2 == (; cells=0, faces=0, edges=0, vertices=0)
    # The owned vertices, bit for bit.
    Φ = f.Φ
    G = Φ.G
    N = Φ.forest.N
    o, h = block_origins(Φ.forest, Float64), block_spacings(Φ.forest, Float64)
    same = all(Iterators.product(1:N, 1:N, 1:N, axes(Φ.work, 5))) do (i, j, k, b)
        x = VM.point_x(o[b], h[b], (i, j, k), Val((1, 1, 1)))
        m1 = point_metric(prov, 0.0, x, Val((1, 1, 1)), (i, j, k), b)
        m0 = point_metric(provider, 0.0, x)
        m1.α == m0.α && m1.β == m0.β && m1.γ == m0.γ && m1.γu == m0.γu && m1.sqrtγ == m0.sqrtγ
    end
    @test same
end

@testset "On the star the vertex metric is fourth order off the surface, lower in its layer" begin
    # Guards the same on the Cowling star's `TOVMetric`, the metric step 11 freezes, with
    # its coarse-fine faces through the star (the first root, `[−16, 0]³`, refined): what
    # the interpolation gives where the metric is smooth, and how much the surface costs.
    # The density has a kink at the surface (`ρ ∝ R̄ − r` for Γ = 2), so the metric's
    # third derivative jumps there and a stencil across it loses order. Measured
    # 2026-11-08, at `N = 12` and `24`, the layer `|r − R̄| < 4`: `∂_j g_ab` at rate 3.7
    # inside, 4.0 outside, 1.7 in the layer; the values 3.7 at the cells, faces and
    # edges; no fallback anywhere.
    c = TV_CASE
    R̄ = c.fields.w.metric.R̄
    region(x) = (r = sqrt(sum(abs2, x)); abs(r - R̄) < 4 ? :surface : r < R̄ ? :inside : :outside)
    e1, fb1, _, _ = vm_errors(vm_forest(c.roots, c.extents, 12), c.provider; region)
    e2, fb2, _, _ = vm_errors(vm_forest(c.roots, c.extents, 24), c.provider; region)
    rates = vm_rates(e1, e2)
    @info "Vertex metric on the TOV star: errors at N = 24 " *
          "$(vm_fmt(e2; f=tv3)); rates $(vm_fmt(rates))"
    @test rates[(:dg, :inside)] > 3.4 && rates[(:dg, :outside)] > 3.7
    @test rates[(:dg, :surface)] > 1.4
    @test all(k -> rates[(k, :inside)] > 3.4 && rates[(k, :outside)] > 3.4,
              (:cell, :face, :edge))
    @test fb1 == fb2 == (; cells=0, faces=0, edges=0, vertices=0)
end

@testset "The vertex metric falls back to multilinear weights where Lagrange loses the signature" begin
    # Guards the fallback: a 4-point Lagrange interpolation overshoots below a dip, so a
    # lapse or a `γ_ii` that is small between two large ones comes out negative, and a
    # metric with `α ≤ 0` or an indefinite `γ` would stop the recovery (or worse, run on).
    # On a flat `Mv`, the planes `i = 3 … 6` of block 1 set to `(1, ε, ε, 1)` in `α`, and
    # of block 2 in `γ_xx`: Lagrange gives `(9ε − 1)/8 < 0` at the cells `i = 4` between the
    # two `ε` planes, which fall back and hold `ε`, the multilinear mean, exactly; the
    # faces and edges that are cell-like along `x` there fall back too, and no vertex.
    forest = Forest{Float64}((2, 2, 2); N=10, periodic=(true, true, true),
                             extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    f = mhd_fields(Float64, forest)
    vm = VertexMetricSet(Float64, forest; boundary=nothing)
    flat = SampledADM(VM.SpacetimeMetrics.Minkowski(), vm, 0.0)
    prov = VM.vertex_metric!(vm, flat, 0.0)
    @test metric_fallbacks(prov, f) == (; cells=0, faces=0, edges=0, vertices=0)
    ε = 1e-3
    G = vm.M.G
    for (slot, b) in ((VM.M_ALPHA, 1), (VM.M_GAMMA, 2)), (i, v) in zip(3:6, (1.0, ε, ε, 1.0))
        vm.M.work[i + G[1], :, :, slot, b] .= v
    end
    x = SVector(0.0, 0.0, 0.0)                       # ignored by the indexed call
    m = point_metric(prov, 0.0, x, Val((0, 0, 0)), (4, 3, 3), 1)
    @test m.α == ε && m.γ[1, 1] == 1
    m2 = point_metric(prov, 0.0, x, Val((0, 0, 0)), (4, 3, 3), 2)
    @test m2.γ[1, 1] == ε && m2.α == 1
    # The neighbouring cells keep their Lagrange values, `(8 + 8ε)/16`.
    @test point_metric(prov, 0.0, x, Val((0, 0, 0)), (3, 3, 3), 1).α == (8 + 8ε) / 16
    dm = metric_gradient(prov, 0.0, x, Val((0, 0, 0)), (4, 3, 3), 1)
    @test all(isfinite, dm.dg) && VM.d_lapse(m, dm)[1] == 0
    fb = metric_fallbacks(prov, f)
    @info "Fallbacks with the two dips: $fb"
    @test fb.cells == 2 * (10 + 2 * 3)^2 && fb.faces > 0 && fb.edges > 0 && fb.vertices == 0
end

@testset "The vertex metric refuses what it cannot serve" begin
    # Guards the arguments: `G = 4` needs `N ≥ 10` (TreeAMR's vertex invariant), a
    # non-periodic forest needs the outer faces' metric, and the provider is read at an
    # index — the coordinate calls of the initial data, the hooks and the interior, and a
    # derivative away from the cells, are refused with the reason.
    small = Forest{Float64}((2, 2, 2); N=8, periodic=(true, true, true),
                            extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    @test_throws ArgumentError VertexMetricSet(Float64, small; boundary=nothing)
    open = Forest{Float64}((2, 2, 2); N=10, periodic=(false, false, false),
                           extents=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))
    @test_throws ArgumentError VertexMetricSet(Float64, open; boundary=nothing)
    @test_throws ArgumentError VertexMetricSet(Float64, open; boundary=Flat(), G=3)
    vm, prov = vm_filled(open, Analytic(VM.SpacetimeMetrics.Minkowski()))
    x = SVector(0.5, 0.5, 0.5)
    @test_throws ArgumentError point_metric(prov, 0.0, x)
    @test_throws ArgumentError metric_gradient(prov, 0.0, x)
    @test_throws ArgumentError metric_gradient(prov, 0.0, x, Val((1, 0, 0)), (1, 1, 1), 1)
    @test_throws ArgumentError SampledADM(VM.SpacetimeMetrics.Minkowski(), vm, 0.0)(
        mhd_fields(Float64, open).P)
end

@testset "A regrid carries Mv as scratch, and a refill is the fresh set's, bit for bit" begin
    # Guards `regrid_pairs(vm)` and `rebuild(vm)`: `Mv` must follow the forest through a
    # regrid (TreeAMR replaces every set's storage it is given), and the rebuilt schedule
    # and spacings must be the new mesh's, or the next fill reads the old geometry.
    provider = Analytic(VM.SpacetimeMetrics.KerrSchild(1.0, 0.5))
    ext = ((2.0, 6.0), (2.0, 6.0), (2.0, 6.0))
    forest = Forest{Float64}((2, 2, 2); N=10, periodic=(false, false, false), extents=ext)
    vm, _ = vm_filled(forest, provider)
    regrid!(forest, VM.regrid_pairs(vm); flags=[i == 1 ? Refine : Keep
                                                   for i in 1:nleaves(forest)])
    vm2 = VM.rebuild(vm)
    @test all(isnan, vm2.M.work)
    prov2 = VM.vertex_metric!(vm2, SampledADM(provider.metric, vm2, 0.0), 0.0)
    _, prov3 = vm_filled(forest, provider)
    @test nleaves(forest) == 15
    @test isequal(prov2.w, prov3.w)
end

# Four steps of the same integrator, with the step cache, through `Analytic(TOVMetric)` and
# through the vertex metric filled once from the sampled adapter (the Cowling metric is
# frozen), at `N` on `h = 8/N`: the largest matter difference relative to the change in the
# four steps, and the fallbacks. Made once per `N` by the first testset that asks.
function vm_cowling_diff(N)
    return computed_once((:vm_cowling, N)) do
        c = TV_CASE
        forest = Forest{Float64}(c.roots; N=N, periodic=c.periodic, extents=c.extents)
        tv_boxes(forest)
        p = MHDProblem(mhd_fields(Float64, forest), c; limiter=:mc)
        _, prov = vm_filled(forest, c.provider)
        q = VM.rebuild(p; provider=prov)
        u0 = exact_state(p, c, 0.0)
        dt = 1 / N
        run(r) = (integ = mhd_integrator(r, copy(u0), 0.0, dt, 4);
                  with_step_cache(() -> VM.IRK.solve!(integ), r); integ.u)
        a, b = run(p), run(q)
        r = staterange(p.sets, 1)
        (maximum(abs, a[r] - b[r]) / maximum(abs, a[r] - u0[r]), metric_fallbacks(prov, p.f))
    end
end

@testset "The Cowling star through the vertex metric is its analytic run, to truncation" begin
    # Guards the kernels' use of the vertex metric (the indexed calls of step 10): a kernel
    # that passes another centering's staggers or index (the vertex metric of `∂ₜA`'s
    # gradient read from an edge kernel, the edge metric of the gauge flux from a vertex
    # kernel, a ghost cell's local index) reads the metric at another point, an `O(h)`
    # difference on the star, not the interpolation's `O(h⁴)`. At `N = 10` (`h = 0.8`).
    # Measured 2026-11-08: 1.2e−3 of the four steps' change, no fallback.
    d, fb = vm_cowling_diff(10)
    @info "The Cowling star through the vertex metric, four steps at N = 10: matter " *
          "difference relative to its change $(tv3(d)); fallbacks $fb"
    @test d < 0.01
    @test fb == (; cells=0, faces=0, edges=0, vertices=0)
end

@full_tier @testset "The Cowling star's difference through the vertex metric converges with h" begin
    # Guards the same as a rate: the difference is the interpolation's, so it shrinks at
    # about its order, and a kernel reading a nearby point's metric would leave a floor.
    # `N = 10` and `20` (`h = 0.8`, `0.4`). Measured 2026-11-08: 1.2e−3 and 4.8e−5, rate
    # 4.67; about 1.5 minutes at `N = 20`.
    d1, _ = vm_cowling_diff(10)
    d2, fb2 = vm_cowling_diff(20)
    @info "The Cowling star through the vertex metric, four steps: $(tv3(d1)) at N = 10, " *
          "$(tv3(d2)) at N = 20, rate $(round(log2(d1 / d2); digits=2)); fallbacks $fb2"
    @test log2(d1 / d2) > 3.5
    @test fb2 == (; cells=0, faces=0, edges=0, vertices=0)
end
