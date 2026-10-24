# Experiment (branch `claude/step-8-ghost-robustness`): how robust the matter's
# ghost fills are on the magnetized atmosphere tube across a coarse-fine face,
# which is chaotic at rounding level — 48 runs per method from perturbed
# initial states.
#
#     julia --project=. bin/ghost_robustness.jl method out.tsv [runs=48]
#
# The tube: a gas at `ρ = p = 1` into a `10⁻⁸` atmosphere, `Bʸ = 10⁻³`,
# `Constant(10³)` cut off at `μ_max = 10`, 200 cells, `[0, 0.2)` refined (the
# coarse-fine faces at `x = 0` and `x = 0.2`), to `t = 2/5`. The perturbed
# initial states: runs 1–12 are `bin/ghost_entropy.jl`'s (`u .*= 1 .+ a ·
# randn(MersenneTwister(1))` with `a` = 0, ±1, ±2, ±3, ±5, ±10, 20 × 10⁻¹⁵);
# runs 13–48 take `a = 10⁻¹⁵` and seeds 2–37.
#
# Methods: on the package's primitive fill (`ops_W`)
#   `prim-cons3-specific`  W = (ρ★, s, Yₑ, uⁱ, Eⁱ), bounded Conservative width 3 (since
#                          2026-10-23 the package's only fill, `default_ops_W`)
#   `prim-cons1-density`   W = (ρ★, ρ★s, ρ★Yₑ, uⁱ, Eⁱ), the parent copy (width 1): only on
#                          `claude/step-8-ghost-robustness` (6b0a72e), which had `ghost_vars`
#   `uniform`              no refinement
# and on a tree with the conserved fill (`git archive 7605986`):
#   `conserved`            U with Conservative width 3, bounded (the default)
#   `uniform`              no refinement
#
# Per run, one line of `out.tsv`: the injected rest mass and energy (the step
# limiter's writes), the stage solve's OHM_FALLBACK stages, the largest `W` of
# an owned cell over the run, the interior ghosts the policy would repair for
# anything but the atmosphere (the conserved fill: the recovery of the
# conserved ghost; the primitive fills: `check_prim_state` of the ghost's
# primitives), the ghosts whose pressure or entropy lies outside the range of
# the owned cells within 1.5 coarse spacings, the wall time, and the first
# fallback, the first step-limiter injection above 10⁻⁹ in one cell, and the
# first `W > 20`, each as `t@x` (and the cell's level).

# *(Amended 2026-10-24, EntropyEOS 2.0:)* `check_prim_state` takes the projection's options and
# reports a density below `ρ_min` as `ρ_FLOORED` where 1.x reported the atmosphere; such a ghost is
# the atmosphere's here, as before (`as_atmosphere`). Run against an older tree (EntropyEOS 1.1),
# take this file from `claude/recovery-pipeline-1` (d94bcf5).

using Printf, Random
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD
const EOS = TG.EntropyEOS

# A density below `ρ_min` is the atmosphere's (EntropyEOS 2.0's `check_prim_state` reports it as
# `ρ_FLOORED`, 1.x as `ATMOSPHERE`).
as_atmosphere(fl) = fl & EOS.FLAG_POL_ρ_FLOORED != 0 ? EOS.FLAG_POL_ATMOSPHERE : fl
const PRIMITIVES = isdefined(TG, :default_ops_W)

refine_band!(forest) =
    (refine!(forest, [k for k in copy(forest.leaves)
                      if 0 ≤ block_origin(forest, k)[1] < 1 // 5 - 1e-12]);
     balance!(forest); forest)

mutable struct Acc
    repaired::Int
    pout::Int
    sout::Int
    maxW::Float64
    first_fb::String
    first_inj::String
    first_W::String
end

where_(t, x, l) = @sprintf("%.4f@%.5f(L%d)", t, x, l)

function monitor!(acc, p, u, t)
    X = p.f.X
    for b in 1:nblocks(X), i in 1:(X.forest.N)
        x = coordinates(X, b, (i,))[1]
        l = level(blockkey(X, b))
        if isempty(acc.first_fb) && X.work[i, TG.X_OHM + 4, b] > 0
            acc.first_fb = where_(t, x, l)
        end
        if isempty(acc.first_inj) && !(abs(X.work[i, TG.X_INJECTED_D, b]) ≤ 1e-9)
            acc.first_inj = where_(t, x, l)
        end
    end
    P = update_primitives!(p, u, t)
    U = p.f.U
    G, N = P.G[1], P.forest.N
    lo, hi = P.forest.extents[1]
    owned = Tuple{Float64,Float64,Float64}[]
    for b in 1:nblocks(P), i in (G + 1):(G + N)
        x = coordinates(P, b, (i,))[1]
        W = cosh(P.work[i, TG.P_W, b])
        push!(owned, (x, P.work[i, TG.P_PRESS, b], P.work[i, TG.P_S, b]))
        acc.maxW = max(acc.maxW, W)
        if isempty(acc.first_W) && !(W ≤ 20)
            acc.first_W = where_(t, x, level(blockkey(P, b)))
        end
    end
    hc = (hi - lo) / (P.forest.roots[1] * N)
    for b in 1:nblocks(P), i in vcat(1:G, (G + N + 1):(N + 2G))
        x = coordinates(P, b, (i,))[1]
        lo < x < hi || continue
        Pc = ntuple(v -> P.work[i, v, b], TG.NP)
        fl = if PRIMITIVES
            as_atmosphere(EOS.check_prim_state(p.eos,
                                               EOS.PrimState{Float64}(Pc[TG.P_RHO], Pc[TG.P_S],
                                                                      Pc[TG.P_YE], Pc[TG.P_W]),
                                               p.proj))
        else
            cons = ntuple(v -> U.work[i, v, b], TG.NU)
            m = TG.aux_metric(Pc)
            con2prim_hydro(p.eos, p.opts, p.proj, p.pol, m, cons, TG.cons_𝔈(cons) / m.sqrtγ,
                           TG.aux_B(Pc), TG.state_guesses(cons))[2]
        end
        fl & EOS.FLAG_POL_ANY & ~EOS.FLAG_POL_ATMOSPHERE != 0 && (acc.repaired += 1)
        near = filter(o -> abs(o[1] - x) ≤ 1.5 * hc, owned)
        isempty(near) && continue
        pr, s = Pc[TG.P_PRESS], Pc[TG.P_S]
        pmin, pmax = extrema(o -> o[2], near)
        smin, smax = extrema(o -> o[3], near)
        max(pr - pmax, pmin - pr, 0.0) / pmax > 1e-12 && (acc.pout += 1)
        max(s - smax, smin - s, 0.0) / abs(smax) > 1e-12 && (acc.sout += 1)
    end
    return acc
end

function run(case, kw, amp, seed)
    t0 = time()
    r0 = evolve!(case, Val(1); N=10, t_end=1e-12, chunk=1e-12, limiter=:mc, kw...)
    p = r0.p
    u = exact_state(p, case, 0.0)
    amp == 0 || (u .*= 1 .+ amp .* randn(MersenneTwister(seed), length(u)))
    acc = Acc(0, 0, 0, 1.0, "", "", "")
    t, D, τ, fb, integ = 0.0, 0.0, 0.0, 0, nothing
    for c in 1:4
        t1 = c / 10
        TG.clear_tally!(p)
        n = TG.chunk_count((t1 - t) * TG.max_light_rate(p, t), 2 / 5)
        integ = TG.mhd_integrator(p, u, t, t1, n; alias_u0=true, reuse=integ)
        while integ.nstep < integ.nsteps
            TG.IRK.step!(integ)
            monitor!(acc, p, integ.u, integ.t)
        end
        tl = ohm_tally(p)
        D += tl.injected_D
        τ += tl.injected_τ
        fb += tl.bits[5]
        t = t1
    end
    return (; acc, D, τ, fb, wall=time() - t0)
end

method = ARGS[1]
out = ARGS[2]
nruns = let a = findfirst(s -> startswith(s, "runs="), ARGS)
    a === nothing ? 48 : parse(Int, ARGS[a][6:end])
end
cb(p) = Operators(; prolongation=p, restriction=2, family=Conservative, bounded=true)
kw = method == "uniform" ? (;) :
     method == "conserved" ? (PRIMITIVES ? error("conserved needs the conserved tree") :
                              (; refine=refine_band!)) :
     method == "prim-cons3-specific" ? (; refine=refine_band!, ops_W=cb(3)) :
     method == "prim-cons1-density" ?
     error("W in density form was removed on 2026-10-23 (Erik's decision of the " *
           "parent-centred primitive fill): run this method on claude/step-8-ghost-robustness " *
           "(6b0a72e).") :
     error("unknown method $method")
const AMPS12 = (0.0, 1e-15, -1e-15, 2e-15, -2e-15, 3e-15, -3e-15, 5e-15, -5e-15, 1e-14, -1e-14,
                2e-14)
perts = [(k ≤ 12 ? (AMPS12[k], 1) : (1e-15, k - 10)) for k in 1:nruns]
case = atmosphere_tube_case(Float64; By=1e-3, model=MagnetizationCutoff(Constant(1e3), 10))
run(case, kw, 0.0, 1)                                  # compile
open(out, "w") do io
    println(io, "run\tamp\tseed\tD\ttau\tfallbacks\tmaxW\trepaired\tpout\tsout\twall\tfirst_fallback\tfirst_injection\tfirst_W20")
    for (k, (amp, seed)) in enumerate(perts)
        r = run(case, kw, amp, seed)
        a = r.acc
        println(io, join((k, amp, seed, r.D, r.τ, r.fb, a.maxW, a.repaired, a.pout, a.sout,
                          round(r.wall; digits=2), a.first_fb, a.first_inj, a.first_W), '\t'))
        flush(io)
    end
end
