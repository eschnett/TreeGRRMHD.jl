# Step 10, round 2: two Cowling-star runs of `bin/tov_study.jl` compared, from their `state`
# files — the star through `Analytic(TOVMetric)` and through the vertex metric (`metric=vertex`)
# at the same settings: the largest difference of their central-density series, relative to
# the initial central density, and the largest and the summed difference of their final
# state vectors, the sum also relative to the analytic run's ("Metric at cells, faces and
# edges" in `CODE.md`, "As built in step 10, round 2").
#
#     julia --project=. bin/tov_compare.jl analytic.jls vertex.jls

using Printf
using Serialization

function compare(fa, fb)
    a, b = deserialize(fa), deserialize(fb)
    a.h == b.h && length(a.series) == length(b.series) && first.(a.series) == first.(b.series) ||
        error("tov_compare.jl: the runs differ in h or in their sampling times")
    ρ0 = last(a.series[1])
    dρ = maximum(abs(last(x) - last(y)) for (x, y) in zip(a.series, b.series)) / ρ0
    return (; h=a.h, t_end=first(a.series[end]), dρ, args_a=a.args, args_b=b.args, ua=a.u,
            ub=b.u)
end

function main(args)
    length(args) == 2 || error("usage: tov_compare.jl analytic.jls vertex.jls")
    r = compare(args...)
    d = abs.(r.ua .- r.ub)
    @printf("TOV compare %s vs %s\n", args...)
    @printf("  h = %.4g, t_end = %.1f\n", r.h, r.t_end)
    @printf("  max |Δρ_c|/ρ_c(0) over the series %.4e\n", r.dρ)
    @printf("  state difference: max %.4e, L1 %.4e, relative to the state's L1 %.4e\n",
            maximum(d), sum(d), sum(d) / sum(abs, r.ua))
    return r
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
