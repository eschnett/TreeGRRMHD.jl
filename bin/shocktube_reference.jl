# The reference solutions of Balsara's five shock tubes for step 6's
# self-convergence claims (`test/shocktube_tests.jl`): a finer run of the same
# code, eight times the finest tested resolution, stored as cell averages on
# the tested mesh. Generated once on the host and checked in, because a run at
# 3200 cells takes minutes and the suite must not (PLAN.md, "Tests are small").
#
#     julia --project=. -t 4 bin/shocktube_reference.jl [ssp | ars] [k ...]
#
# Each test `k` (all five by default) runs on `[−1/2, 1/2]` at `M_REF = 3200`
# cells (320 root blocks of `N = 10`) to its final time — MC limiter, HLLE,
# `cfl = 2/5`, four chunks, the tableau named (SSP3(4,3,3) by default, ARS(4,4,3)
# with `ars`) — exactly as the tests run at 100, 200 and 400 cells. Its cells,
# ordered by `x`, are averaged in groups of eight onto the 400 cells of the
# finest tested mesh, which is the exact restriction of the cell averages. Per
# cell, twelve numbers: the ten evolved cell variables (`√γ D`, `√γ S_i`,
# `√γ τ`, `√γ D Yₑ`, `√γ D s`, `𝔈ⁱ`; flat space, so densitized and plain
# agree) and the cell averages of `𝔅ʸ` and `𝔅ᶻ` (the curl of `A`; `𝔅ˣ` is the
# background). Written as `test/data/balsara<k>.ref` (or `balsara<k>_ars.ref`):
# little-endian `Float32`, column-major `400 × 12`, nothing else — 19,200 bytes.
# `Float32` keeps 7 digits, four orders of magnitude below the reference's own
# error (an eighth of the 400-cell run's).
#
# The run's diagnostics (steps, wall time, the stage outcomes and the resets)
# are printed, for `CODE.md`.

using Printf
using TreeGRRMHD
import IMEXRungeKutta as IRK

const M_REF = 3200
const M_OUT = 400
const NCOL = 12

function reference(k::Integer, tableau)
    b = BALSARA[k]
    t_end = Float64(b.t_end)
    case = balsara_case(Float64, k; roots=M_REF ÷ 10)
    t0 = time()
    r = evolve!(case, Val(1); N=10, t_end=t_end, chunk=t_end / 4, limiter=:mc,
                tableau=tableau)
    wall = time() - t0
    pr = profile_1d(r.p, r.u, r.t)
    cols = hcat(pr.U, pr.B[:, 2], pr.B[:, 3])
    f = M_REF ÷ M_OUT
    out = [sum(cols[(f * (i - 1) + 1):(f * i), c]) / f for i in 1:M_OUT, c in 1:NCOL]
    tallies = [c.tally for c in r.chunks]
    return out, (; r.nsteps, wall, tallies, plateaus=plateaus(k, pr))
end

# Where each of Giacomazzo & Rezzolla's plateaus lies in the fine run: the
# longest run of cells within 2% of the region's state (`plateau_error`), and
# the fine run's mean over it. What `BALSARA_PLATEAUS` in
# `src/cases/shocktubes.jl` is made from.
function plateaus(k, pr)
    n = length(pr.x)
    h = pr.x[2] - pr.x[1]
    map(1:8) do region
        ok = [maximum(plateau_error(k, region, plateau_state(pr, i))) ≤ 0.02 for i in 1:n]
        best, bi, i = 0, 0, 1
        while i ≤ n
            if ok[i]
                j = i
                while j < n && ok[j + 1]
                    j += 1
                end
                j - i + 1 > best && ((best, bi) = (j - i + 1, i))
                i = j + 1
            else
                i += 1
            end
        end
        best == 0 && return (; region, lo=NaN, hi=NaN, err=NaN)
        cells = bi:(bi + best - 1)
        mean = ntuple(j -> sum(c -> plateau_state(pr, c)[j], cells) / best, 7)
        (; region, lo=pr.x[first(cells)] - h / 2, hi=pr.x[last(cells)] + h / 2,
         err=maximum(plateau_error(k, region, mean)))
    end
end

function main(args)
    ars = "ars" in args
    tableau = ars ? IRK.ARS443() : IRK.IMEXSSP3433()
    ks = [parse(Int, a) for a in args if all(isdigit, a)]
    isempty(ks) && (ks = 1:5)
    dir = joinpath(@__DIR__, "..", "test", "data")
    mkpath(dir)
    @info "Balsara references at $M_REF cells, $(tableau.name), $(Threads.nthreads()) thread(s)"
    for k in ks
        out, d = reference(k, tableau)
        file = joinpath(dir, "balsara$(k)$(ars ? "_ars" : "").ref")
        open(file, "w") do io
            write(io, htol.(Float32.(out)))
        end
        bits = sum(t -> collect(t.bits), d.tallies)
        @printf("Balsara %d: %d steps, %.1f s wall; stage bits %s; policy %d; resets %d/%d; injected D %.3e, τ %.3e; max recoveries %d → %s\n",
                k, d.nsteps, d.wall, bits, sum(t -> t.policy, d.tallies),
                sum(t -> t.resets, d.tallies), sum(t -> t.step_resets, d.tallies),
                sum(t -> t.injected_D, d.tallies), sum(t -> t.injected_τ, d.tallies),
                maximum(t -> t.max_recoveries, d.tallies), file)
        for q in d.plateaus
            @printf("  R%d: [%.5f, %.5f], mean within %.4f of the exact state\n", q.region,
                    q.lo, q.hi, q.err)
        end
    end
end

main(ARGS)
