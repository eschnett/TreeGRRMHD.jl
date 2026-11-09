# Measurement (2026-10-23, branch `claude/recovery-cost`; not a PLAN.md step): how many
# primitive recoveries the scheme makes per cell per step, how each is seeded, what they
# cost, and what Erik's "step-local" view — the only true input is the state at the start of
# an RK step, so seed every later recovery from the cell's latest one — would save. See
# HISTORY.md, "Primitive recovery", "Measurement (2026-10-23): recovery counts and costs".
#
#     julia --project=. bin/recovery_cost.jl counts CASE [FILL]     # FILL: primitive (default) | conserved | both
#     julia --project=. bin/recovery_cost.jl pointwise
#     julia --project=. bin/recovery_cost.jl identity
#     julia --project=. -t T bin/recovery_cost.jl threads OUTFILE   # then `threadcmp F1 F4`
#     julia --project=TREE bin/recovery_cost.jl steptime [CASE...]   # all of `CASES` by default
#     julia --project=. bin/recovery_cost.jl finish [CASE...]         # all of `CASES` by default
#
# One thread unless said (the counting context is global). `bin/recovery_cost_hooks.jl` is
# evaluated into the TreeGRRMHD module and redefines, in this process only, the recovery's
# two calls of `con2prim_safe` and the passes (see its header); `identity` checks that with
# everything off they compute what `src/` computes, bit for bit.
#
# `counts CASE FILL`: the case (see `CASES`) run once counting (mode 1) — per site, per step
# and owned cell: cold-seeded, warm and no-solve calls, Newton iterations, nested fallbacks,
# in-situ ns per call; the input/output coincidence of each recovery with the cell's previous
# one; with the conserved fill, the same-level ghosts whose `U` and `P` equal their source's
# — and every call of a window of steps recorded and replayed: ns per call of
# `con2prim_safe` as seeded now and as seeded from the cell's latest recovery in the step
# (step-local), and how far the two answers are apart. Then the case run again uninstrumented
# (mode 0) for the wall time per step and the pass breakdown.
# `pointwise`: `con2prim_safe` per call on the ideal gas and on EntropyEOS's synthetic table
# through `GeometricEOS`, cold, cold-seeded (`s` known), warm from the exact answer and from
# a neighbour 10⁻³ and 10⁻² away.
# `identity`: (1) the hooks off and counting are the code bit for bit; (2) the step-local
# view, real (every recovery seeded from the cell's latest), against the 1D/2D, turned and
# chunking bit-identity claims, under both fills; how far it moves results; Orszag–Tang's
# totals. `threads` at `-t 1` and `-t 4`, then `threadcmp`: the same under both fills.
#
# HISTORY.md's table: `counts CASE both` for every name in `CASES` (the conserved fill of a
# case is the second block of its output), `pointwise`, `identity`, `threads`.
#
# *(Added 2026-10-24, stage 3 of the recovery pipeline: the real step cache, "The step cache"
# in CODE.md.)* `steptime`: the code as it is, no hooks — per case, the wall time per step
# of `evolve!` (the mean over steps 2…n, the best of three runs after a warm-up), with the
# step cache and, on a tree that has it, without (`step_cache = false`), and the step
# cache's seed tallies per site (cold, warm, reused per owned cell and step). It runs on any
# tree: `--project` names the tree measured, and a tree without the cache (stage 2,
# a9cd482) gives its plain time; run both trees in one job on one node. The hooks below
# replace the passes with cache-free ones, so `counts`, `identity` and `threads` measure the
# 2026-10-23 emulation, never the real cache: with the hooks loaded every recovery is
# seeded as before the cache, and `identity`'s reference run is `step_cache = false`.
#
# *(Added 2026-10-30, stage 4 of the recovery pipeline: the two-phase recovery, "The
# two-phase recovery" in CODE.md.)* `steptime` on a tree with the two-phase recovery times
# it ("split") against the same tree fused (`two_phase = false`), both with the step cache;
# a tree without it gives "cache" and "nocache" as before. `finish`: per case, the share of
# each site's recoveries that reach `con2prim_safe_finish` — the cells whose
# `con2prim_safe_start` was not done, from the hand-off after each first pass — and of the
# stage solve's iterations (its recoveries after the first, which stay fused), counted by
# redefining, in this process only, the passes' `recovery_start!` and `two_levels` with
# copies that count (`load_finish_hooks!`; one thread, the counters are global). The
# counting copies call EntropyEOS exactly as `src/` does, so the run is the run.

# *(Amended 2026-10-24, EntropyEOS 2.0:)* the replays and the pointwise timings call level 1,
# `con2prim_safe` onto the projection `proj`, alone: level 2 (`apply_policy`) costs nothing to seed.
# The recorded calls are the scheme's own, both levels. The 2026-10-23 numbers were made on
# `claude/recovery-cost` (d456a44), with EntropyEOS 1.1.0.
using Printf, Random, Statistics, Serialization
using StaticArrays
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD
const EOS = TG.EntropyEOS

# --- the cases ------------------------------------------------------------------------------

band!(lo, hi) = forest -> begin
    refine!(forest, [k for k in copy(forest.leaves)
                     if lo - 1e-12 ≤ block_origin(forest, k)[1] < hi - 1e-12])
    balance!(forest)
    forest
end
centre!(forest) = begin
    refine!(forest, [k for k in copy(forest.leaves)
                     if all(e -> -0.25 - 1e-12 ≤ block_origin(forest, k)[e] < 0.25 - 1e-12, 1:2)])
    balance!(forest)
    forest
end

# name => (case, D, keywords of evolve!, first recorded step)
const CASES = Dict(
    "emwave" => () -> (em_wave_case(Float64; roots=10), 1, (; N=10, t_end=1 // 4, limiter=:mc), 30),
    "alfven" => () -> (alfven_wave_case(Float64; roots=10), 1,
                       (; N=10, t_end=alfven_period(alfven_wave_case(Float64)) / 4, limiter=:mc), 80),
    "resist10" => () -> (resistive_tube_case(Float64, Constant(10.0); roots=20), 1,
                         (; N=10, t_end=0.4, limiter=:mc), 100),
    "resist100" => () -> (resistive_tube_case(Float64, Constant(100.0); roots=20), 1,
                          (; N=10, t_end=0.4, limiter=:mc), 100),
    "resist1000" => () -> (resistive_tube_case(Float64, Constant(1000.0); roots=20), 1,
                           (; N=10, t_end=0.4, limiter=:mc), 100),
    "atmosphere" => () -> (atmosphere_tube_case(Float64; By=1e-3,
                                                model=MagnetizationCutoff(Constant(1e3), 10),
                                                roots=20), 1,
                           (; N=10, t_end=0.4, chunk=0.1, limiter=:mc), 100),
    "rotor" => () -> (magnetic_rotor_case(Float64; roots=4), 2,
                      (; N=16, t_end=1 // 10, limiter=:mc), 16),
    "orszagtang" => () -> (orszag_tang_case(Float64; roots=4), 2,
                           (; N=16, t_end=1 // 8, limiter=:mc), 20),
    "blast3d" => () -> (cylindrical_explosion_case(Float64, Val(3); L=2, roots=2, spherical=true),
                        3, (; N=8, t_end=1 // 4, limiter=:minmod), 4),
    "resist100_refined" => () -> (resistive_tube_case(Float64, Constant(100.0); roots=20), 1,
                                  (; N=10, t_end=0.4, limiter=:mc, refine=band!(0.4, 0.6)), 100),
    "rotor_refined" => () -> (magnetic_rotor_case(Float64; roots=4), 2,
                              (; N=16, t_end=1 // 10, limiter=:mc, refine=centre!), 30))
const WINDOW = 3                     # recorded steps

const HOOKS = joinpath(@__DIR__, "recovery_cost_hooks.jl")
hooks_loaded() = isdefined(TG, :RCS)
load_hooks!() = hooks_loaded() || Core.eval(TG, :(include($HOOKS)))

# --- running -----------------------------------------------------------------------------

function run_case(name; fill=:primitive, mode=0, steplocal=false, record=false, kw...)
    case, D, ekw, w0 = CASES[name]()
    rcs = TG.RCS
    rcs.fill = fill
    rcs.mode = mode
    rcs.steplocal = steplocal
    rcs.step = 0
    rcs.meshkey = (0, 0)                     # a fresh cache for the run
    TG.rc_clear_counts!()
    fill!(TG.RCT, 0)
    stamps = UInt64[]
    record && (rcs.record = false)
    function after(p, u, t)
        push!(stamps, time_ns())
        rcs.record = record && w0 ≤ rcs.step < w0 + WINDOW
        return nothing
    end
    rcs.record = false
    t0 = time_ns()
    push!(stamps, t0)
    r = evolve!(case, Val(D); after_step=after, ekw..., kw...)
    rcs.record = false
    rcs.mode = 0
    rcs.steplocal = false
    return (; r, case, D, stamps, w0)
end

ncells_owned(r) = nblocks(r.p.f.U) * r.p.f.U.forest.N^length(r.p.f.U.forest.roots)
nstored(r) = nblocks(r.p.f.P) * prod(size(r.p.f.P.work)[1:(ndims(r.p.f.P.work) - 2)])

# --- the replay ---------------------------------------------------------------------------

@noinline function replay_time(eos, opts, proj, recs, seeds::Symbol)
    n = length(recs)
    n == 0 && return (NaN, 0.0, 0)
    acc = 0.0
    iters = 0
    reps = 0
    t0 = time_ns()
    while true
        for r in recs
            g = seeds === :now ? r.gnow : r.gloc
            cin = EOS.Con2PrimIn{Float64}(r.cin...)
            out = EOS.con2prim_safe(eos, cin, opts, proj, g[1], g[2], g[3])
            acc += out.base.ρ
            reps == 0 && (iters += out.base.iters_newton)
        end
        reps += 1
        time_ns() - t0 > 20_000_000 && reps ≥ 3 && break
    end
    return ((time_ns() - t0) / (n * reps), acc, iters)
end

# How far the step-locally seeded answer is from the answer as seeded now.
function replay_diff(eos, opts, proj, recs)
    worst = 0.0
    wρ = ws = ww = 0.0
    at = nothing
    same = 0
    flagdiff = 0
    for r in recs
        cin = EOS.Con2PrimIn{Float64}(r.cin...)
        a = EOS.con2prim_safe(eos, cin, opts, proj, r.gnow...)
        b = EOS.con2prim_safe(eos, cin, opts, proj, r.gloc...)
        (a.base.ρ, a.base.s, a.base.w) == (b.base.ρ, b.base.s, b.base.w) && (same += 1)
        a.policy_flags == b.policy_flags || (flagdiff += 1)
        dρ = abs(a.base.ρ - b.base.ρ) / abs(a.base.ρ)
        ds = abs(a.base.s - b.base.s) / abs(a.base.s)
        dw = abs(a.base.w - b.base.w)
        d = max(dρ, ds, dw)
        isfinite(d) || continue
        wρ, ws, ww = max(wρ, dρ), max(ws, ds), max(ww, dw)
        if d > worst
            worst = d
            # How much of the energy is thermal: `U ρ W²`-ish against `τ`, the entropy's
            # conditioning ("Primitive recovery" in CODE.md).
            at = (; D=cin.D, τ=cin.τ, ρ=a.base.ρ, W=cosh(a.base.w), s=(a.base.s, b.base.s),
                  flags=a.policy_flags)
        end
    end
    return (; worst, wρ, ws, ww, at, same, flagdiff)
end

isnanseed(g) = isnan(g[1]) | isnan(g[2])      # what makes `con2prim` run its cold seed

# --- counts -------------------------------------------------------------------------------

function counts(name, fill)
    rc = TG.RCS
    @info "counts: $name, $fill fill"
    a = run_case(name; fill=fill, mode=1, record=true)
    r = a.r
    p = r.p
    nst = r.nsteps
    nown = ncells_owned(r)
    nsto = nstored(r)
    perc(x) = x / (nst * nown)
    n, newton, fb, tns = copy(rc.n), copy(rc.newton), copy(rc.fallback), copy(rc.tns)
    coin, cout, locavail = copy(rc.coin), copy(rc.cout), copy(rc.locavail)
    copies = (rc.copies, rc.copies_U, rc.copies_P, rc.copies_Pnum)
    recs = copy(rc.recs)
    tally = (; bits=sum(c -> collect(c.tally.bits), r.chunks),
             resets=sum(c -> c.tally.resets, r.chunks),
             step_resets=sum(c -> c.tally.step_resets, r.chunks))
    @printf("\n## %s, %s fill: %d steps, %d blocks, %d owned cells, %d stored (×%.2f)\n",
            name, fill, nst, nblocks(p.f.U), nown, nsto, nsto / nown)
    @printf("stage branches (σ=0, ideal, fixed point, Newton, fallback): %s; resets %d/%d\n",
            string(tally.bits[1:5]), tally.resets, tally.step_resets)
    kinds = [count(==(k), rc.kind) for k in 0:4]
    @printf("stored cells by kind (owned, copy, prolongated, restricted, boundary): %s\n",
            string(kinds))
    println("per step and owned cell: cold-seeded / warm / no solve; Newton iterations per ",
            "solved call (cold-seeded, warm); nested fallbacks; in-situ ns per call (cold-seeded, warm)")
    tot = zeros(3)
    for s in 1:TG.RC_NS
        sum(n[s, :]) == 0 && continue
        c, w, z = perc(n[s, 1]), perc(n[s, 2]), perc(n[s, 3])
        tot .+= (c, w, z)
        @printf("  %-28s %7.3f %7.3f %7.3f | it %5.2f %5.2f | fb %4d | ns %6.0f %6.0f | step-local seed for %5.1f%% of the cold-seeded\n",
                TG.RC_SITES[s], c, w, z, newton[s, 1] / max(n[s, 1], 1),
                newton[s, 2] / max(n[s, 2], 1), fb[s, 1] + fb[s, 2],
                tns[s, 1] / max(n[s, 1], 1), tns[s, 2] / max(n[s, 2], 1),
                100 * locavail[s] / max(n[s, 1], 1))
    end
    @printf("  %-28s %7.3f %7.3f %7.3f  (total %.2f)\n", "all", tot..., sum(tot))
    @printf("in-situ recovery time per step: %.3f ms\n", sum(tns) / nst / 1e6)
    # Coincidence with the cell's previous recovery, for the pairs that matter.
    println("input (output) against the cell's previous recovery — buckets ==, ≤1e-14, ",
            "≤1e-12, ≤1e-10, ≤1e-8, >, ideal-vs-hydro:")
    for s in 1:TG.RC_NS, l in 1:TG.RC_NS
        sum(coin[s, l, :]) == 0 && continue
        sum(coin[s, l, :]) < 0.02 * nst * nown && s ≤ 5 && continue
        @printf("  %-26s after %-26s in %s out %s\n", TG.RC_SITES[s], TG.RC_SITES[l],
                string(coin[s, l, :]), string(cout[s, l, :]))
    end
    fill === :conserved &&
        @printf("same-level ghost copies: %d, U equal to the source %d (the sign of a zero free), P bit for bit %d (the sign of a zero free %d)\n", copies...)
    # The replay of the window.
    nrec = length(recs)
    @printf("replay of %d recorded calls (steps %d…%d):\n", nrec, a.w0, a.w0 + WINDOW - 1)
    eos, opts, proj = p.eos, p.opts, p.proj
    replay = Dict{Int,Any}()
    for s in 1:TG.RC_NS
        rs = filter(x -> x.site == s && x.solved, recs)     # the policy's answers cost nothing to seed
        isempty(rs) && continue
        cold = filter(x -> isnanseed(x.gnow), rs)
        warm = filter(x -> !isnanseed(x.gnow), rs)
        withloc = filter(x -> isnanseed(x.gnow) && !isnanseed(x.gloc), rs)
        tc, _, ic = replay_time(eos, opts, proj, cold, :now)
        tw, _, iw = replay_time(eos, opts, proj, warm, :now)
        tn, _, in_ = replay_time(eos, opts, proj, withloc, :now)
        tl, _, il = replay_time(eos, opts, proj, withloc, :loc)
        d = isempty(withloc) ? (; worst=NaN, wρ=NaN, ws=NaN, ww=NaN, at=nothing, same=0, flagdiff=0) :
            replay_diff(eos, opts, proj, withloc)
        replay[s] = (; cold=length(cold), warm=length(warm), withloc=length(withloc), tc, tw, tn, tl)
        @printf("  %-28s cold-seeded %6d at %6.0f ns (it %4.2f), warm %6d at %6.0f ns (it %4.2f); step-local %6d: %6.0f → %6.0f ns (it %4.2f → %4.2f), answers equal %5.1f%%, worst %.1e, flags differ %d\n",
                TG.RC_SITES[s], length(cold), tc, ic / max(length(cold), 1), length(warm), tw,
                iw / max(length(warm), 1), length(withloc), tn, tl,
                in_ / max(length(withloc), 1), il / max(length(withloc), 1),
                100 * d.same / max(length(withloc), 1), d.worst, d.flagdiff)
        d.worst > 1e-10 && @printf("      worst apart: ρ %.1e, s %.1e, w %.1e; at %s\n", d.wρ, d.ws, d.ww,
                                   string(d.at))
    end
    # The uninstrumented run: wall per step and the pass breakdown (twice; the second counts).
    run_case(name; fill=fill, mode=0)
    b = run_case(name; fill=fill, mode=0)
    # The first interval holds the setup (forest, problem, initial data, integrator).
    dts = diff(b.stamps)[2:end]
    step_ns = median(dts)
    tot_ns = mean(dts)
    @printf("uninstrumented: %.3f ms per step (mean over steps 2…%d; median %.3f)\n",
            tot_ns / 1e6, b.r.nsteps, step_ns / 1e6)
    rct = TG.RCT ./ b.r.nsteps
    for (k, nm) in enumerate(TG.RCT_NAMES)
        rct[k] == 0 && continue
        @printf("  %-42s %8.3f ms  %5.1f%%\n", nm, rct[k] / 1e6, 100 * rct[k] / tot_ns)
    end
    @printf("  %-42s %8.3f ms  %5.1f%%\n", "integrator arithmetic and the rest",
            (tot_ns - sum(rct)) / 1e6, 100 * (tot_ns - sum(rct)) / tot_ns)
    # Estimates, per step: replayed ns × counts.
    pers(s, k) = n[s, k] / nst
    rec_ns = sum(s -> haskey(replay, s) ?
                      pers(s, 1) * nz(replay[s].tc) + pers(s, 2) * nz(replay[s].tw) : 0.0, 1:TG.RC_NS)
    @printf("recoveries per step, replayed cost × counts: %.3f ms (%.1f%% of the step); in situ %.3f ms (%.1f%%)\n",
            rec_ns / 1e6, 100 * rec_ns / tot_ns, sum(tns) / nst / 1e6, 100 * sum(tns) / nst / tot_ns)
    # (a) step-local seeds for every cold-seeded call that has one.
    sa = sum(s -> haskey(replay, s) ?
                  locavail[s] / nst * (nz(replay[s].tn) - nz(replay[s].tl)) : 0.0, 1:TG.RC_NS)
    # (b) reuse instead of recovery: the RHS's owned recoveries whose input is the stage
    # limiter's bit for bit, and the predictors whose input is the step limiter's.
    rb_rhs = coin[TG.RC_RHS, TG.RC_RESET_STAGE, 1] / nst *
             (haskey(replay, TG.RC_RHS) ? nz(replay[TG.RC_RHS].tc) : 0.0)
    pred_like = haskey(replay, TG.RC_PRED) ? TG.RC_PRED : TG.RC_IDEAL
    rb_pred = (coin[TG.RC_PRED, TG.RC_RESET_STEP, 1]) / nst *
              (haskey(replay, TG.RC_PRED) ? nz(replay[TG.RC_PRED].tc) : 0.0)
    # (c) the same-level ghost copies (conserved fill).
    sc = n[TG.RC_RHS + 1, 1] / nst * (haskey(replay, TG.RC_RHS + 1) ? nz(replay[TG.RC_RHS + 1].tc) : 0.0)
    @printf("estimated saving per step: (a) step-local seeds %.3f ms (%.1f%%); (b) reuse the stage limiter's P in the RHS %.3f ms (%.1f%%), the step limiter's in the first predictor %.3f ms (%.1f%%); (c) exchange same-level ghosts %.3f ms (%.1f%%)\n",
            sa / 1e6, 100 * sa / tot_ns, rb_rhs / 1e6, 100 * rb_rhs / tot_ns, rb_pred / 1e6,
            100 * rb_pred / tot_ns, sc / 1e6, 100 * sc / tot_ns)
    return nothing
end
nz(x) = isfinite(x) ? x : 0.0

# The code as it is, before the hooks are loaded: wall per step, for the hooks' overhead.
function plain_time(name)
    case, D, ekw, _ = CASES[name]()
    evolve!(case, Val(D); ekw...)
    stamps = UInt64[]
    r = evolve!(case, Val(D); after_step=(p, u, t) -> push!(stamps, time_ns()), ekw...)
    dts = diff(stamps)
    @printf("\n%s, the code without the hooks (primitive fill): %.3f ms per step (mean over steps 2…%d; median %.3f)\n",
            name, mean(dts) / 1e6, r.nsteps, median(dts) / 1e6)
end

# --- pointwise: the ideal gas and the synthetic table ----------------------------------------

function pointwise()
    gas = TG.shock_gas(Float64, 5 // 3)
    proj_g = EOS.default_projection(gas, 1e-10)
    table = EOS.build_eos(EOS.make_synthetic_table())
    f = MSUN_DENSITY_CGS
    tab = GeometricEOS(table.view; density=f)
    proj_t = EOS.default_projection(tab, 1e7 / f)
    opts = EOS.Con2PrimOptions{Float64}()
    rng = Xoshiro(23)
    for (label, eos, proj, ρs) in (("ideal gas, Γ = 5/3", gas, proj_g, () -> exp10(-2 + 2 * rand(rng))),
                                  ("synthetic table (GeometricEOS)", tab, proj_t,
                                   () -> exp10(9 + 4 * rand(rng)) / f))
        for W in (1.005, 2.0, 5.0)
            states = map(1:400) do _
                ρ = ρs()
                sr = EOS.srange(eos, ρ, 0.3)
                s = sr.s_min + (0.2 + 0.6 * rand(rng)) * (sr.s_max - sr.s_min)
                (ρ, s, acosh(W))
            end
            cin(ρ, s, w) = (c = EOS.prim2con(eos, ρ, s, 0.3, w, 0.0, 0.0, NaN);
                            EOS.Con2PrimIn{Float64}(c.D, c.τ, c.D_Y, 0.0, c.S_perp, 0.0))
            ins = [cin(x...) for x in states]
            sol = [EOS.con2prim_safe(eos, c, opts, proj) for c in ins]
            near(δ) = map(states) do (ρ, s, w)
                c = cin(ρ * (1 + δ * randn(rng)), s * (1 + δ * randn(rng)), w * (1 + δ * randn(rng)))
                o = EOS.con2prim_safe(eos, c, opts, proj)
                (o.base.s, o.base.w, o.base.eos.u_solved)
            end
            seeds = [("cold", [(NaN, NaN, NaN) for _ in ins]),
                     ("cold-seeded (s exact)", [(x[2], NaN, NaN) for x in states]),
                     ("warm, the answer", [(o.base.s, o.base.w, o.base.eos.u_solved) for o in sol]),
                     ("warm, 1e-3 away", near(1e-3)), ("warm, 1e-2 away", near(1e-2))]
            for (sl, g) in seeds
                t, it = time_calls(eos, opts, proj, ins, g)
                med, nfb = median_call(eos, opts, proj, ins, g)
                @printf("%-32s W = %-5g %-24s %6.0f ns per call (median %6.0f), %4.2f Newton iterations, nested fallback %4.1f%%\n",
                        label, W, sl, t, med, it, 100 * nfb / length(ins))
            end
        end
    end
end

# The median over the states of each call's own time (a few repeats each, the fastest), and
# how many calls reached the nested fallback.
@noinline function median_call(eos, opts, proj, ins, g)
    ts = map(eachindex(ins)) do i
        best = typemax(UInt64)
        for _ in 1:5
            t0 = time_ns()
            o = EOS.con2prim_safe(eos, ins[i], opts, proj, g[i]...)
            best = min(best, time_ns() - t0)
            o.base.ρ == 0.5 && println()
        end
        best
    end
    nfb = count(i -> EOS.con2prim_safe(eos, ins[i], opts, proj, g[i]...).base.result !==
                     EOS.C2PResult.converged_newton, eachindex(ins))
    return median(Float64.(ts)), nfb
end

@noinline function time_calls(eos, opts, proj, ins, g)
    reps = 0
    its = 0
    acc = 0.0
    t0 = time_ns()
    while true
        for i in eachindex(ins)
            o = EOS.con2prim_safe(eos, ins[i], opts, proj, g[i]...)
            acc += o.base.ρ
            reps == 0 && (its += o.base.iters_newton)
        end
        reps += 1
        time_ns() - t0 > 30_000_000 && reps ≥ 3 && break
    end
    acc == 0.5 && println()
    return (time_ns() - t0) / (reps * length(ins)), its / length(ins)
end

# --- identity -----------------------------------------------------------------------------

function md_global(p, u, i)
    s = p.sets[i]
    forest = s.forest
    D = length(forest.roots)
    N = forest.N
    a = statearray(u, p.sets, i)
    h = block_spacings(forest)
    o = block_origins(forest)
    out = zeros(forest.roots .* N..., s.nvars)
    for b in 1:nblocks(s)
        g0 = ntuple(d -> round(Int, (o[b][d] - forest.extents[d][1]) / h[b]), D)
        for I in CartesianIndices(ntuple(_ -> N, D)), v in 1:(s.nvars)
            out[(Tuple(I) .+ g0)..., v] = a[I, v, b]
        end
    end
    return out
end
same_as_1d(r1, rD) = all(1:5) do i
    a1 = md_global(r1.p, r1.u, i)
    aD = md_global(rD.p, rD.u, i)
    all(I -> isequal(aD[I], a1[I[1], I[3]]), CartesianIndices(size(aD)))
end
function turned_equal(rx, ry)
    turnU(q, v) = v in (2, 8) ? q[v + 1] : v in (3, 9) ? q[v - 1] : v in (4, 10) ? -q[v] : q[v]
    seg = (1, 3, 2, 4, 5)
    sgn = (1, 1, 1, -1, 1)
    return all(1:5) do i
        ay = md_global(ry.p, ry.u, i)
        ax = md_global(rx.p, rx.u, seg[i])
        all(CartesianIndices(size(ay)[1:2])) do I
            J = (I[2], I[1])
            all(v -> ay[I, v] == (i == 1 ? turnU(view(ax, J..., :), v) : sgn[i] * ax[J..., v]),
                1:size(ay, 3))
        end
    end
end
reldiff(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

function setmode!(; fill=:primitive, mode=0, steplocal=false)
    rc = TG.RCS
    rc.fill, rc.mode, rc.steplocal, rc.record, rc.step = fill, mode, steplocal, false, 0
    rc.meshkey = (0, 0)
    return nothing
end

# The identity checks' runs. The hooks must be loaded between `identity_plain` and
# `identity_hooks`, at top level: a function already running keeps the methods of the world
# it started in.
const ID_TUBE = resistive_tube_case(Float64, Constant(100.0); roots=4)
const ID_ROTOR = magnetic_rotor_case(Float64; roots=2)
const ID_ATM = atmosphere_tube_case(Float64; By=1e-3, model=MagnetizationCutoff(Constant(1e3), 10),
                                    roots=20)
const ID_KW1 = (; N=10, t_end=1 // 10, limiter=:mc)
const ID_KWR = (; N=8, t_end=1 // 20, limiter=:mc)
const ID_KWA = (; N=10, t_end=0.4, chunk=0.1, limiter=:mc)

# (1) The code as it is, before the hooks are loaded, with the step cache off (the hooks'
# passes do not consult it; 2026-10-24).
const ID_NOCACHE = isdefined(TG, :with_step_cache) ? (; step_cache=false) : (;)
identity_plain() = (evolve!(ID_TUBE, Val(1); ID_KW1..., ID_NOCACHE...),
                    evolve!(ID_ROTOR, Val(2); ID_KWR..., ID_NOCACHE...))

function identity_hooks((plain1, plainr))
    c1, rot, atm = ID_TUBE, ID_ROTOR, ID_ATM
    kw1, kwr, kwa = ID_KW1, ID_KWR, ID_KWA
    for mode in (0, 1)
        setmode!(; mode)
        h1 = evolve!(c1, Val(1); kw1...)
        hr = evolve!(rot, Val(2); kwr...)
        @printf("hooks loaded, mode %d, primitive fill: resistive tube %s, rotor %s the code bit for bit\n",
                mode, isequal(h1.u, plain1.u) ? "is" : "is NOT",
                isequal(hr.u, plainr.u) ? "is" : "is NOT")
    end
    # (2) The step-local view, real.
    for fill in (:primitive, :conserved)
        setmode!(; fill)
        b1 = evolve!(c1, Val(1); kw1...)
        br = evolve!(rot, Val(2); kwr...)
        ba = evolve!(atm, Val(1); kwa...)
        setmode!(; fill, steplocal=true)
        s1 = evolve!(c1, Val(1); kw1...)
        setmode!(; fill, steplocal=true)
        s4 = evolve!(c1, Val(1); kw1..., chunk=1 // 20)
        setmode!(; fill, steplocal=true)
        c2 = extruded_case(c1, Val(2); N=10)
        s2 = evolve!(c2, Val(2); kw1..., cfl=4 // 5)
        setmode!(; fill, steplocal=true)
        sx = evolve!(c2, Val(2); kw1...)
        setmode!(; fill, steplocal=true)
        sy = evolve!(turned_case(c2), Val(2); kw1...)
        setmode!(; fill, steplocal=true)
        sr = evolve!(rot, Val(2); kwr...)
        setmode!(; fill, steplocal=true)
        sa = evolve!(atm, Val(1); kwa...)
        setmode!(; fill)
        @printf("%s fill, step-local seeds: resistive tube (σ = 100, 40 cells, t = 1/10) 1D = 2D %s; turned %s; one chunk = two %s\n",
                fill, same_as_1d(s1, s2), turned_equal(sx, sy), isequal(s1.u, s4.u))
        @printf("  against the run as seeded now: resistive tube %.2e, rotor %.2e, magnetized atmosphere tube %.2e (max |Δu| / max |u|); stage recoveries %d → %d (tube), %d → %d (atmosphere); fallbacks %d → %d (atmosphere); injected D %.2e → %.2e\n",
                reldiff(s1.u, b1.u), reldiff(sr.u, br.u), reldiff(sa.u, ba.u),
                sum(c -> c.tally.recoveries, b1.chunks), sum(c -> c.tally.recoveries, s1.chunks),
                sum(c -> c.tally.recoveries, ba.chunks), sum(c -> c.tally.recoveries, sa.chunks),
                sum(c -> c.tally.bits[5], ba.chunks), sum(c -> c.tally.bits[5], sa.chunks),
                sum(c -> c.tally.injected_D, ba.chunks), sum(c -> c.tally.injected_D, sa.chunks))
        # Conservation on a periodic box: each face's flux is computed by both blocks that
        # share it, from their own `P`; with the conserved fill a ghost's `P` is its own
        # recovery's.
        for sl in (false, true)
            setmode!(; fill, steplocal=sl)
            ot = orszag_tang_case(Float64; roots=4)
            r = evolve!(ot, Val(2); N=8, t_end=1 // 8, limiter=:mc)
            drift = conserved_totals(r.p, r.u) .- r.totals0
            sc = conserved_scales(r.p, r.u)
            worst = maximum(v -> abs(drift[v]) / sc[v] / eps(), (TG.U_D, TG.U_S, TG.U_S + 1, TG.U_TAU))
            @printf("  Orszag–Tang 32², t = 1/8 (%d steps), %s: worst drift of D, S, τ %.1f ulp of the scale\n",
                    r.nsteps, sl ? "step-local seeds" : "seeded now", worst)
        end
        setmode!(; fill)
    end
    return nothing
end

# One-versus-four threads: the same runs written to a file per thread count.
function threads(out)
    c1 = resistive_tube_case(Float64, Constant(100.0); roots=4)
    rot = magnetic_rotor_case(Float64; roots=2)
    res = Dict{String,Any}()
    for fill in (:primitive, :conserved), sl in (false, true)
        setmode!(; fill, steplocal=sl)
        res["tube $fill $sl"] = evolve!(c1, Val(1); N=10, t_end=1 // 10, limiter=:mc).u
        setmode!(; fill, steplocal=sl)
        res["rotor $fill $sl"] = evolve!(rot, Val(2); N=8, t_end=1 // 20, limiter=:mc).u
    end
    serialize(out, res)
    @info "$(Threads.nthreads()) thread(s): wrote $(length(res)) states to $out"
end
function threadcmp(f1, f4)
    a, b = deserialize(f1), deserialize(f4)
    for k in sort(collect(keys(a)))
        @printf("%-28s one thread = four: %s\n", k, isequal(a[k], b[k]))
    end
end

# --- steptime: the real step cache (2026-10-24) -----------------------------------------------

function steptime_run(name; kw...)
    case, D, ekw, _ = CASES[name]()
    stamps = UInt64[]
    push!(stamps, time_ns())
    r = evolve!(case, Val(D); after_step=(p, u, t) -> push!(stamps, time_ns()), ekw..., kw...)
    dts = diff(stamps)[2:end]               # the first interval holds the setup
    return (; r, ms=mean(dts) / 1e6, med=median(dts) / 1e6)
end

function steptime(names)
    cached = isdefined(TG, :with_step_cache)
    split = isdefined(TG, :Handoff)
    variants = split ? (("split", (; two_phase=true)), ("fused", (; two_phase=false))) :
               cached ? (("cache", (; step_cache=true)), ("nocache", (; step_cache=false))) :
               (("plain", (;)),)
    for name in names
        for (label, kw) in variants
            steptime_run(name; kw...)                       # compile and warm up
            runs = [steptime_run(name; kw...) for _ in 1:3]
            best = argmin(x -> x.ms, runs)
            r = best.r
            nown = ncells_owned(r)
            @printf("steptime %-18s %-7s %8.4f ms/step (runs %s; median %.4f), %d steps, %d owned cells, %.1f ns per cell and step
",
                    name, label, best.ms, join([@sprintf("%.4f", x.ms) for x in runs], " "),
                    best.med, r.nsteps, nown, 1e6 * best.ms / nown)
            if cached && label in ("cache", "split")
                per(x) = x / (r.nsteps * nown)
                for k in (:rhs, :stage, :stage_limiter, :step_limiter)
                    c = map(q -> sum(ch -> getproperty(getproperty(ch.tally.seeds, k), q), r.chunks),
                            (:cold, :warm, :reused))
                    @printf("    seeds %-14s cold %6.3f warm %6.3f reused %6.3f per cell and step
",
                            k, per.(c)...)
                end
                @printf("    stage iterations (recoveries after the first) %.3f per cell and step
",
                        per(sum(ch -> ch.tally.recoveries, r.chunks)) -
                        per(sum(ch -> ch.tally.seeds.stage.cold + ch.tally.seeds.stage.warm +
                                      ch.tally.seeds.stage.reused, r.chunks)))
            end
        end
    end
    return nothing
end

# --- finish: the share of recoveries that reach con2prim_safe_finish (2026-10-30) -----------

# The site the integrator's callback in progress recovers for, and per site
# `[recoveries, reached finish]`; for the stage solve's iterations also the
# reused ones.
const FS_SITE = Ref(:none)
const FS_COUNTS = Dict{Symbol,Vector{Int}}()

function load_finish_hooks!()
    @eval TG begin
        # `src/evolution.jl`'s, counting the hand-off's `todo` after the second pass.
        function recovery_start!(p::MHDProblem{T,D}, Uw, GU::Val, model, t) where {T,D}
            sc = p.handoff
            sc === nothing && return nothing
            f = p.f
            map_blocks!(start_kernel!, p.cache, sc, Uw, f.B[1].work, f.B[2].work, f.B[3].work,
                        p.cache.work, p.origins, p.spacings, p.provider, t, p.eos, p.opts, p.proj,
                        model, p.cache_on[], Val(D), GU, Val(GB_of(f)), Val(p.cache.G))
            map_blocks!(finish_kernel!, p.cache, sc, p.eos, p.opts, p.proj, Val(D))
            site = Main.FS_SITE[]
            if site !== :none
                c = get!(() -> zeros(Int, 3), Main.FS_COUNTS, site)
                c[1] += length(sc.todo)
                c[2] += count(sc.todo)
            end
            return nothing
        end
        # `src/con2prim.jl`'s, counting the fused calls inside a stage solve (its
        # iterations: the first recovery is the hand-off's) that would reach finish.
        @inline function two_levels(eos::EntropyEOS.AbstractEOS,
                                    opts::EntropyEOS.Con2PrimOptions{T},
                                    proj::EntropyEOS.ProjectionOptions{T},
                                    pol::EntropyEOS.PolicyOptions{T},
                                    cin::EntropyEOS.Con2PrimIn{T}, guesses::NTuple{3,T},
                                    entry::NTuple{NC,T}, sc=nothing, idx::Tuple=()) where {T}
            count = sc === nothing && Main.FS_SITE[] === :stage
            c = count ? get!(() -> zeros(Int, 3), Main.FS_COUNTS, :iterations) : Int[]
            if entry_hit(entry, cin)
                count && (c[1] += 1; c[3] += 1)
                return entry, true
            end
            if sc === nothing
                o, st = EntropyEOS.con2prim_safe_start(eos, cin, opts, proj, guesses[1],
                                                       guesses[2], guesses[3])
                done = EntropyEOS.is_done(st)
                count && (c[1] += 1; c[2] += !done)
                out = done ? o : EntropyEOS.con2prim_safe_finish(eos, cin, opts, proj, st)
            else
                out = level1(eos, opts, proj, cin, guesses, sc, idx)
            end
            out = EntropyEOS.apply_policy(eos, cin, out, proj, pol)
            return make_entry(cin, out), false
        end
        # `src/driver.jl`'s, with every callback naming its site.
        function mhd_integrator(p::MHDProblem{T}, u, t0, t1, nsteps::Integer;
                                tableau=IRK.IMEXSSP3433(), alias_u0::Bool=false,
                                reuse=nothing) where {T}
            t0, t1 = T(t0), T(t1)
            at(site, f) = (args...) -> (Main.FS_SITE[] = site; f(args...);
                                        Main.FS_SITE[] = :none; nothing)
            prob = IRK.IMEXProblem(at(:rhs, mhd_rhs!), at(:stage, ohm_solve!), u, (t0, t1), p)
            integ = IRK.init(prob, tableau; dt=(t1 - t0) / nsteps,
                             stage_limiter=at(:stage_limiter, reset_atmosphere!),
                             step_limiter=at(:step_limiter, reset_step!),
                             partition=u isa Array ? :even : nothing, alias_u0=alias_u0,
                             reuse=reuse)
            integ.nsteps == nsteps || error("step count")
            return integ
        end
    end
    return nothing
end

function finish_share(names)
    load_finish_hooks!()
    for name in names
        case, D, ekw, _ = CASES[name]()
        empty!(FS_COUNTS)
        r = Base.invokelatest(evolve!, case, Val(D); ekw..., two_phase=true)
        nown = ncells_owned(r)
        seeds(k) = sum(ch -> getproperty(ch.tally.seeds, k).reused, r.chunks)
        @printf("finish %-18s %d steps, %d owned cells\n", name, r.nsteps, nown)
        for (k, sk) in ((:rhs, :rhs), (:stage, :stage), (:stage_limiter, :stage_limiter),
                        (:step_limiter, :step_limiter))
            c = get(FS_COUNTS, k, zeros(Int, 3))
            reused = seeds(sk)
            @printf("    %-14s recoveries %8d, reused %8d, finish %7d: %6.3f%% of all, %6.3f%% of those started\n",
                    k, c[1], reused, c[2], 100 * c[2] / max(c[1], 1),
                    100 * c[2] / max(c[1] - reused, 1))
        end
        c = get(FS_COUNTS, :iterations, zeros(Int, 3))
        @printf("    %-14s recoveries %8d, reused %8d, finish %7d: %6.3f%% of all, %6.3f%% of those started\n",
                "iterations", c[1], c[3], c[2], 100 * c[2] / max(c[1], 1),
                100 * c[2] / max(c[1] - c[3], 1))
    end
    return nothing
end

# --- main -----------------------------------------------------------------------------------

cmd = isempty(ARGS) ? "help" : ARGS[1]
@info "recovery_cost $cmd, $(Threads.nthreads()) thread(s), load $(round.(Sys.loadavg(), digits=1))"
if cmd == "counts"
    plain_time(ARGS[2])
    load_hooks!()
    fills = length(ARGS) ≥ 3 ? (ARGS[3] == "both" ? (:primitive, :conserved) : (Symbol(ARGS[3]),)) :
            (:primitive,)
    for fl in fills
        Base.invokelatest(counts, ARGS[2], fl)
    end
elseif cmd == "pointwise"
    pointwise()
elseif cmd == "identity"
    plain = identity_plain()
    load_hooks!()
    Base.invokelatest(identity_hooks, plain)
elseif cmd == "threads"
    load_hooks!()
    Base.invokelatest(threads, ARGS[2])
elseif cmd == "threadcmp"
    threadcmp(ARGS[2], ARGS[3])
elseif cmd == "finish"
    finish_share(length(ARGS) ≥ 2 ? ARGS[2:end] :
                 ["emwave", "alfven", "resist10", "resist100", "resist1000", "atmosphere", "rotor",
                  "orszagtang", "blast3d", "resist100_refined", "rotor_refined"])
elseif cmd == "steptime"
    steptime(length(ARGS) ≥ 2 ? ARGS[2:end] :
             ["emwave", "alfven", "resist10", "resist100", "resist1000", "atmosphere", "rotor",
              "orszagtang", "blast3d", "resist100_refined", "rotor_refined"])
else
    println("usage: see the header of bin/recovery_cost.jl")
end
