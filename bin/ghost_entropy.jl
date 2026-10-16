# Experiment (branch `claude/step-8-ghost-entropy-density`): the matter's
# primitive ghost fill with the specific `s`, `Yₑ` in `W` (step 8's) against
# the densities `ρ★ s`, `ρ★ Yₑ` (mass-weighted), and those with the velocity
# mass-weighted too (`ρ★ uⁱ`, `ghost_vars = :mass`), on the atmosphere tube across
# a coarse-fine face — a gas at `ρ = p = 1` into a `10⁻⁸` atmosphere, 200
# cells, `[0, 0.2)` refined, to `t = 2/5`.
#
#     julia --project=. bin/ghost_entropy.jl [hydro] [magnetized] [only=label-prefix] [verbose]
#
# Per configuration, over the run (after every step): the interior ghosts
# (not beyond the domain's faces) the policy would repair for anything but
# the atmosphere (primitive fills: `check_prim_state` of the ghost's
# primitives; the conserved fill: the recovery of the conserved ghost), those
# below the atmosphere's density; the ghosts whose pressure, or entropy, lies
# outside the range of the owned cells within 1.5 coarse spacings of it (the
# points any fill reads), with the largest excess relative to that range's
# top; the injected rest mass and energy (the step limiter's writes), the
# OHM_FALLBACK stages, and the largest `W` of an owned cell. The magnetized
# tube is chaotic at rounding level, so it runs from twelve initial states
# perturbed by `pert · randn` (seed 1) and prints each quantity's spread.
# Run against a tree from before AMR-10 (`git archive 7605986`), it measures
# the conserved, bounded fill.

using Printf, Random
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD
const EOS = TG.EntropyEOS
const PRIMITIVES = isdefined(TG, :default_ops_W)
const DENSITY = isdefined(TG, :W_store)

refine_band!(forest) =
    (refine!(forest, [k for k in copy(forest.leaves)
                      if 0 ≤ block_origin(forest, k)[1] < 1 // 5 - 1e-12]);
     balance!(forest); forest)

mutable struct Acc
    repaired::Int
    below::Int
    pout::Int
    pexcess::Float64
    sout::Int
    sexcess::Float64
    maxW::Float64
end

function monitor!(acc, p, u, t)
    P = update_primitives!(p, u, t)
    U = p.f.U
    G, N = P.G[1], P.forest.N
    lo, hi = P.forest.extents[1]
    owned = Tuple{Float64,Float64,Float64}[]          # x, p, s
    for b in 1:nblocks(P), i in (G + 1):(G + N)
        push!(owned, (coordinates(P, b, (i,))[1], P.work[i, TG.P_PRESS, b], P.work[i, TG.P_S, b]))
        acc.maxW = max(acc.maxW, cosh(P.work[i, TG.P_W, b]))
    end
    hc = (hi - lo) / (length(P.forest.roots) == 1 ? P.forest.roots[1] * N : 1)
    for b in 1:nblocks(P), i in vcat(1:G, (G + N + 1):(N + 2G))
        x = coordinates(P, b, (i,))[1]
        lo < x < hi || continue
        Pc = ntuple(v -> P.work[i, v, b], TG.NP)
        ρ = Pc[TG.P_RHO]
        (isfinite(ρ) && ρ ≥ (1 - 1e-12) * 1e-8) || (acc.below += 1)
        fl = if PRIMITIVES
            EOS.check_prim_state(p.eos, EOS.PrimState{Float64}(ρ, Pc[TG.P_S], Pc[TG.P_YE],
                                                                Pc[TG.P_W]), p.pol)
        else
            cons = ntuple(v -> U.work[i, v, b], TG.NU)
            m = TG.aux_metric(Pc)
            con2prim_hydro(p.eos, p.opts, p.pol, m, cons, TG.cons_𝔈(cons) / m.sqrtγ,
                           TG.aux_B(Pc), TG.state_guesses(cons))[2]
        end
        fl & EOS.FLAG_POL_ANY & ~EOS.FLAG_POL_ATMOSPHERE != 0 && (acc.repaired += 1)
        near = filter(o -> abs(o[1] - x) ≤ 1.5 * hc, owned)
        isempty(near) && continue
        pr, s = Pc[TG.P_PRESS], Pc[TG.P_S]
        pmin, pmax = extrema(o -> o[2], near)
        smin, smax = extrema(o -> o[3], near)
        ep = max(pr - pmax, pmin - pr, 0.0) / pmax
        es = max(s - smax, smin - s, 0.0) / abs(smax)
        ep > 1e-12 && (acc.pout += 1; acc.pexcess = max(acc.pexcess, ep))
        es > 1e-12 && (acc.sout += 1; acc.sexcess = max(acc.sexcess, es))
    end
    return acc
end

function run(case, kw, pert)
    T = Float64
    r0 = evolve!(case, Val(1); N=10, t_end=1e-12, chunk=1e-12, limiter=:mc, kw...)
    p = r0.p
    u = exact_state(p, case, 0.0)
    pert == 0 || (u .*= 1 .+ pert .* randn(MersenneTwister(1), length(u)))
    acc = Acc(0, 0, 0, 0.0, 0, 0.0, 1.0)
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
    return (; acc, D, τ, fb)
end

pv(w) = Operators(; prolongation=w, restriction=w, family=PointValue, bounded=true)
configs = if DENSITY
    [("uniform", (;)),
     ("specific s, w = 2", (; refine=refine_band!, ops_W=pv(2), ghost_vars=:specific)),
     ("specific s, w = 4", (; refine=refine_band!, ops_W=pv(4), ghost_vars=:specific)),
     ("specific s, w = 6", (; refine=refine_band!, ops_W=pv(6), ghost_vars=:specific)),
     ("density ρs, w = 2", (; refine=refine_band!, ops_W=pv(2), ghost_vars=:density)),
     ("density ρs, w = 4", (; refine=refine_band!, ops_W=pv(4), ghost_vars=:density)),
     ("density ρs, w = 6", (; refine=refine_band!, ops_W=pv(6), ghost_vars=:density)),
     ("mass ρs, ρu, w = 2", (; refine=refine_band!, ops_W=pv(2), ghost_vars=:mass)),
     ("mass ρs, ρu, w = 4", (; refine=refine_band!, ops_W=pv(4), ghost_vars=:mass)),
     ("mass ρs, ρu, w = 6", (; refine=refine_band!, ops_W=pv(6), ghost_vars=:mass))]
else
    [("uniform (conserved)", (;)),
     ("conserved, bounded", (; refine=refine_band!,
                             ops_U=Operators(; prolongation=3, restriction=2,
                                             family=Conservative, bounded=true)))]
end
const PERTS = (0.0, 1e-15, -1e-15, 2e-15, -2e-15, 3e-15, -3e-15, 5e-15, -5e-15, 1e-14, -1e-14,
               2e-14)
spread(xs) = (minimum(xs), maximum(xs))
fmt(xs) = (a = spread(xs); a[1] == a[2] ? @sprintf("%.2e", a[1]) :
           @sprintf("%.2e…%.2e", a...))
fmti(xs) = (a = spread(xs); a[1] == a[2] ? "$(a[1])" : "$(a[1])…$(a[2])")

# `only=prefix` runs the configurations whose label starts with it.
let o = findfirst(a -> startswith(a, "only="), ARGS)
    o === nothing || filter!(c -> startswith(c[1], ARGS[o][6:end]), configs)
end
which = filter(a -> a in ("hydro", "magnetized"), ARGS)
isempty(which) && (which = ["hydro", "magnetized"])
println(DENSITY ? "primitive fills, specific and density-weighted" :
        "conserved fill (tree before AMR-10)")
for name in which
    hydro = name == "hydro"
    case = hydro ? atmosphere_tube_case(Float64) :
           atmosphere_tube_case(Float64; By=1e-3, model=MagnetizationCutoff(Constant(1e3), 10))
    perts = hydro ? PERTS[1:3] : PERTS
    println(name, ": ", length(perts), " runs (perturbations ", perts, ")")
    for (label, kw) in configs
        t0 = time()
        rs = [run(case, kw, q) for q in perts]
        a(f) = [f(r) for r in rs]
        @printf("  %-22s repaired %s, below ρ_atm %s; p out of range %s (max %s), s out %s (max %s); injected D %s, τ %s; fallbacks %s; max W %s  (%.0f s)\n",
                label, fmti(a(r -> r.acc.repaired)), fmti(a(r -> r.acc.below)),
                fmti(a(r -> r.acc.pout)), fmt(a(r -> r.acc.pexcess)), fmti(a(r -> r.acc.sout)),
                fmt(a(r -> r.acc.sexcess)), fmt(a(r -> r.D)), fmt(a(r -> r.τ)), fmti(a(r -> r.fb)),
                fmt(a(r -> r.acc.maxW)), time() - t0)
        if "verbose" in ARGS
            for (q, r) in zip(perts, rs)
                @printf("      pert %+.0e: injected D %+.2e, τ %+.2e, fallbacks %d, max W %.3g\n",
                        q, r.D, r.τ, r.fb, r.acc.maxW)
            end
        end
        flush(stdout)
    end
end
