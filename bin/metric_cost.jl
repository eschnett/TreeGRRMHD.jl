# Step 9: what the `Analytic` provider costs ("Metric providers" in `CODE.md`) — whether
# evaluating SpacetimeMetrics pointwise at every centering is affordable, or the metric should
# be cached per stage.
#
#     julia --project=. bin/metric_cost.jl
#
# One thread. Three measurements:
#
# 1. **Per call**: `point_metric` and `metric_gradient` of `Analytic(KerrSchild(1))` and of
#    `Flat()`, in ns, over 10⁶ points of the octant `[0, 4]³` (the result summed, so that
#    nothing is optimized away).
# 2. **Per step, the same physics**: the spherical blast of rung 2a (16³, closed box) with
#    `Flat()` and with `Analytic` of a Kerr-Schild hole of mass 10⁻³⁰⁰ far outside the box —
#    Minkowski to rounding at every point, with Kerr-Schild's whole cost. The difference is
#    the pointwise provider's share of a step; a per-stage cache could save at most that,
#    less what it costs to store and load.
# 3. **Per step, the curved cases**: `wald_case` and `bondi_case` at `N = 8, 16` (16³, 32³).
#
# Wall time per step: the mean over steps 2…n, the best of three runs after a warm-up.

using TreeGRRMHD
using SpacetimeMetrics: KerrSchild, translate
using StaticArrays
using Printf
using Statistics: mean

const TG = TreeGRRMHD

function per_call()
    xs = [SVector(4rand(), 4rand(), 4rand()) for _ in 1:10^6]
    for (name, prov) in (("Flat", Flat()), ("Analytic(KerrSchild(1))", Analytic(KerrSchild(1.0))))
        f1(x) = point_metric(prov, 0.0, x).α
        f2(x) = metric_gradient(prov, 0.0, x).K[1, 1]
        sum(f1, xs); sum(f2, xs)
        t1 = minimum(@elapsed(sum(f1, xs)) for _ in 1:3)
        t2 = minimum(@elapsed(sum(f2, xs)) for _ in 1:3)
        @printf("per call %-26s point_metric %7.1f ns, metric_gradient %7.1f ns\n", name,
                1e9 * t1 / length(xs), 1e9 * t2 / length(xs))
    end
end

function steptime(case, D, kw)
    best = Inf
    nsteps = 0
    for k in 1:4
        stamps = UInt64[time_ns()]
        r = evolve!(case, Val(D); after_step=(p, u, t) -> push!(stamps, time_ns()), kw...)
        k == 1 && continue                                  # compile and warm up
        best = min(best, mean(diff(stamps)[2:end]) / 1e6)
        nsteps = r.nsteps
    end
    return best, nsteps
end

with_provider(c, prov) =
    MHDCase(Float64, Val(3); fields=c.fields, potential=c.potential, eos=c.eos, opts=c.opts,
            proj=c.proj, pol=c.pol, model=c.model, provider=prov, 𝔅̄=c.𝔅̄,
            periodic=c.periodic, extents=c.extents, roots=c.roots)

function main()
    @info "metric cost" Threads.nthreads()
    per_call()
    blast = cylindrical_explosion_case(Float64, Val(3); L=2, roots=2, spherical=true)
    far = Analytic(translate(KerrSchild(1e-300), SVector(0.0, 100.0, 100.0, 100.0)))
    kw = (; N=8, t_end=1 // 8, limiter=:minmod)
    tf, n = steptime(blast, 3, kw)
    ta, _ = steptime(with_provider(blast, far), 3, kw)
    @printf("blast 16³: Flat %.2f ms/step, Analytic (Kerr-Schild, M = 1e-300) %.2f ms/step (%d steps): the provider %.0f%% of a step\n",
            tf, ta, n, 100 * (ta - tf) / ta)
    for (name, ctor) in (("wald", wald_case), ("bondi", bondi_case)), N in (8, 16)
        t, n = steptime(ctor(Float64), 3, (; N=N, t_end=1 // 4, limiter=:mc))
        @printf("%s N=%d (%d³ cells): %.2f ms/step (%d steps), %.2f µs per cell and step\n",
                name, N, 2N, t, n, 1e3 * t / (2N)^3)
    end
end

main()
