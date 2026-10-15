# Step 6's comparison of the two IMEX tableaus, SSP3(4,3,3) and ARS(4,4,3), on
# the discontinuous problems of rung 1b — what decided the default tableau
# (`CODE.md`, "The partition"). Every case at the resolution its test runs,
# once with each tableau, at one thread so that the wall times compare:
#
#     julia --project=. bin/tableau_comparison.jl
#
# Balsara 1–5 (400 cells): the L1 error of `√γ D`, `√γ τ` and `𝔅ʸ` against
# both stored 3200-cell references (`bin/shocktube_reference.jl`); the worst
# plateau error against Giacomazzo & Rezzolla over the windows of
# `BALSARA_PLATEAUS`; the oscillations — the total variation of `√γ D` in
# excess of the reference's on the same mesh, and the largest excursion of
# `√γ D` outside the reference's range over the cell's neighbours (three
# cells either side), as a fraction of the reference's total range; the
# distance of the stored `𝔈` from its ideal projection; the stage outcomes,
# the resets (stage/step), the injected rest mass and energy; and the wall
# time. The resistive tube (200 cells) over σ ∈ {0, 10, 10², 10³, 10⁶, ∞}:
# the L1 distance of `𝔅ʸ` to the ideal run of the same tableau. The current
# sheet (Komissarov's, 200 cells to t = 9): the L1 distance to the erf
# profile. The atmosphere tube (200 cells) in three forms: the smallest
# density, the largest Lorentz factor, the resets and the injection.

using Printf
using TreeGRRMHD
import IMEXRungeKutta as IRK

const TABLEAUS = (IRK.IMEXSSP3433(), IRK.ARS443())
const REF_DIR = joinpath(@__DIR__, "..", "test", "data")

read_ref(file) = Float64.(ltoh.(reinterpret(Float32, read(file)))) |> v -> reshape(v, 400, 12)

l1(a, b, h) = sum(abs, a .- b) * h

tv(a) = sum(abs, diff(a))

function excursion(a, ref; w=3)
    lo, hi = minimum(ref), maximum(ref)
    worst = 0.0
    n = length(a)
    for i in 1:n
        r = ref[max(1, i - w):min(n, i + w)]
        worst = max(worst, a[i] - maximum(r), minimum(r) - a[i])
    end
    return worst / (hi - lo)
end

totals(r) = (; stages=sum(c -> collect(c.tally.bits), r.chunks),
             policy=sum(c -> c.tally.policy, r.chunks),
             resets=sum(c -> c.tally.resets, r.chunks),
             step_resets=sum(c -> c.tally.step_resets, r.chunks),
             injected_D=sum(c -> c.tally.injected_D, r.chunks),
             injected_τ=sum(c -> c.tally.injected_τ, r.chunks),
             maxrec=maximum(c -> c.tally.max_recoveries, r.chunks))

function timed(f)
    f()                                   # compile
    t0 = time()
    r = f()
    return r, time() - t0
end

function stored_vs_projected(r)
    proj = copy(r.u)
    ohm_solve!(proj, r.u, 1.0, r.p, r.t)
    a = profile_1d(r.p, r.u, r.t).U[:, 8:10]
    b = profile_1d(r.p, proj, r.t).U[:, 8:10]
    return maximum(abs, a .- b) / maximum(abs, b)
end

function balsara()
    println("\n## Balsara 1–5, 400 cells")
    for k in 1:5, tab in TABLEAUS
        t_end = Float64(BALSARA[k].t_end)
        r, wall = timed(() -> evolve!(balsara_case(Float64, k; roots=40), Val(1); N=10,
                                      t_end=t_end, chunk=t_end / 4, limiter=:mc,
                                      tableau=tab))
        pr = profile_1d(r.p, r.u, r.t)
        h = 1 / 400
        refs = [read_ref(joinpath(REF_DIR, "balsara$k$s.ref")) for s in ("", "_ars")]
        errs = [(l1(pr.U[:, 1], R[:, 1], h), l1(pr.U[:, 5], R[:, 5], h),
                 l1(pr.B[:, 2], R[:, 11], h)) for R in refs]
        R = refs[1]
        pe = 0.0
        for (region, lo, hi) in BALSARA_PLATEAUS[k]
            cells = findall(x -> lo ≤ x ≤ hi, pr.x)
            mean = ntuple(j -> sum(c -> plateau_state(pr, c)[j], cells) / length(cells), 7)
            pe = max(pe, maximum(plateau_error(k, region, mean)))
        end
        tvx = tv(pr.U[:, 1]) / tv(R[:, 1]) - 1
        exc = excursion(pr.U[:, 1], R[:, 1])
        off = stored_vs_projected(r)
        t = totals(r)
        @printf("B%d %-12s L1(D,τ,𝔅ʸ) vs SSP3 ref %s, vs ARS ref %s; plateau %.4f; TV excess %.4f; excursion %.4f; stored 𝔈 off %.3e; stages %s; policy %d; resets %d/%d; injected %.2e/%.2e; wall %.2f s\n",
                k, tab.name, string(round.(errs[1], sigdigits=3)),
                string(round.(errs[2], sigdigits=3)), pe, tvx, exc, off, t.stages, t.policy,
                t.resets, t.step_resets, t.injected_D, t.injected_τ, wall)
    end
end

function sweep()
    println("\n## The resistive tube, 200 cells: L1(𝔅ʸ) to the ideal run")
    for tab in TABLEAUS
        run(model) = evolve!(resistive_tube_case(Float64, model; roots=20), Val(1); N=10,
                             t_end=0.4, limiter=:mc, tableau=tab)
        ri = run(IdealConductor())
        ideal = profile_1d(ri.p, ri.u, ri.t)
        for σ in (0.0, 10.0, 1e2, 1e3, 1e6, Inf)
            model = σ == 0 ? Vacuum() : isinf(σ) ? IdealConductor() : Constant(σ)
            r, wall = timed(() -> run(model))
            pr = profile_1d(r.p, r.u, r.t)
            t = totals(r)
            @printf("%-12s σ = %-6g L1(𝔅ʸ) %.4e; stages %s; max recoveries %d; resets %d/%d; wall %.2f s\n",
                    tab.name, σ, l1(pr.B[:, 2], ideal.B[:, 2], 1 / 200), t.stages, t.maxrec,
                    t.resets, t.step_resets, wall)
        end
    end
end

function sheet()
    println("\n## The current sheet, 200 cells, t = 1 → 9")
    for tab in TABLEAUS
        case = current_sheet_case(Float64; roots=20)
        r, wall = timed(() -> evolve!(case, Val(1); N=10, t_end=8.0, chunk=2.0, limiter=:mc,
                                      tableau=tab))
        pr = profile_1d(r.p, r.u, r.t)
        ex = profile_1d(r.p, exact_state(r.p, case, r.t), r.t)
        t = totals(r)
        @printf("%-12s L1(Bʸ − erf)/L %.4e; stages %s; max recoveries %d; wall %.2f s\n",
                tab.name, l1(pr.B[:, 2], ex.B[:, 2], 3 / 200) / 3, t.stages, t.maxrec, wall)
    end
end

function atmosphere()
    println("\n## The atmosphere tube, 200 cells, ρ_atm = 1e-8")
    forms = (("hydrodynamic, B = 0", 0.0, IdealConductor()),
             ("ideal, Bʸ = 1e-3", 1e-3, IdealConductor()),
             ("resistive, Bʸ = 1e-3, cutoff 10", 1e-3, MagnetizationCutoff(Constant(1e3), 10)))
    for (name, By, model) in forms, tab in TABLEAUS
        case = atmosphere_tube_case(Float64; By=By, model=model, roots=20)
        r, wall = timed(() -> evolve!(case, Val(1); N=10, t_end=0.4, chunk=0.1, limiter=:mc,
                                      tableau=tab))
        pr = profile_1d(r.p, r.u, r.t)
        W = maximum(1 ./ sqrt.(1 .- sum(pr.v .^ 2; dims=2)))
        t = totals(r)
        @printf("%-34s %-12s min ρ %.3e; max W %.3g; stages %s; policy %d; resets %d/%d; injected D %.3e, τ %.3e; max recoveries %d; non-finite %d; wall %.2f s\n",
                name, tab.name, minimum(pr.ρ), W, t.stages, t.policy, t.resets,
                t.step_resets, t.injected_D, t.injected_τ, t.maxrec, count(!isfinite, r.u),
                wall)
    end
end

@info "Tableau comparison, $(Threads.nthreads()) thread(s)"
balsara()
sweep()
sheet()
atmosphere()
