# Step 10 (rung 4): the Cowling star's stationarity as a convergence study ("Validation
# ladder" 4 in `CODE.md`), from the central-density series that `bin/tov_study.jl` writes.
#
#     julia --project=. bin/tov_drift.jl t=200 window=40 SERIES:N[:levels:half] ...
#
# For each series, `ρ_c(0)` is recomputed from the mesh it ran on (`N`, `levels`, `half` as
# `bin/tov_study.jl` takes them, default `1` and `1.25`): the cell average at the centre,
# which the series does not record. The drift is `ρ_c / ρ_c(0) − 1` at `t` and its mean over
# `[t − window, t]` (the radial modes averaged out); the order between successive series is
# `log(e₁/e₂) / log(h₁/h₂)`. A series that has not reached `t` is reported at its last time
# and left out of the orders.

using TreeAMR
using TreeGRRMHD
using Printf
const MHD = TreeGRRMHD

function initial_density(; N, levels=1, half=1.25)
    star = MHD.tov_case(Float64)
    forest = Forest{Float64}(star.roots; N, periodic=star.periodic, extents=star.extents)
    MHD.tov_boxes!(forest, star; levels, half_width=half * star.fields.w.metric.R̄)
    mhd = MHD.MHDProblem(MHD.mhd_fields(Float64, forest), star; limiter=:mc)
    u = statevector(MHD.evolved_sets(mhd))
    MHD.initial_state!(u, mhd, star, 0.0)
    return MHD.central_density(mhd, u, 0.0), minimum_spacing(forest)
end

function read_series(path)
    t, ρ = Float64[], Float64[]
    for line in eachline(path)
        f = split(line)
        length(f) ≥ 2 || continue
        push!(t, parse(Float64, f[1]))
        push!(ρ, parse(Float64, f[2]))
    end
    return t, ρ
end

function drift(specs; t=200.0, window=40.0)
    rows = []
    for spec in specs
        f = split(spec, ':')
        N = parse(Int, f[2])
        levels = length(f) ≥ 3 ? parse(Int, f[3]) : 1
        half = length(f) ≥ 4 ? parse(Float64, f[4]) : 1.25
        ρ0, h = initial_density(; N, levels, half)
        ts, ρs = read_series(f[1])
        t1 = min(t, ts[end])
        i1 = findlast(≤(t1), ts)
        win = findall(s -> t1 - window ≤ s ≤ t1, ts)
        e = ρs[i1] / ρ0 - 1
        ē = sum(ρs[win]) / length(win) / ρ0 - 1
        push!(rows, (; path=f[1], h, ρ0, t1, e, ē, done=t1 == t))
        @printf("h = %-6.4g ρ_c(0) = %.10e  t = %6.1f  drift %+.4e  mean over [%g, %g] %+.4e%s\n",
                h, ρ0, t1, e, t1 - window, t1, ē, t1 == t ? "" : "  (not reached)")
    end
    for (a, b) in zip(rows[1:(end - 1)], rows[2:end])
        (a.done && b.done) || continue
        r(x, y) = log(abs(x) / abs(y)) / log(a.h / b.h)
        @printf("order h = %g → %g: at t %.2f, of the mean %.2f\n", a.h, b.h, r(a.e, b.e),
                r(a.ē, b.ē))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    kw = Dict{Symbol,Float64}()
    specs = String[]
    for a in ARGS
        if occursin('=', a)
            k, v = split(a, '='; limit=2)
            kw[Symbol(k)] = parse(Float64, v)
        else
            push!(specs, a)
        end
    end
    drift(specs; kw...)
end
