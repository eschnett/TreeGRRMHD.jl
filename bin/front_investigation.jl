# Investigation (2026-10-16, branch `claude/front-investigation`; not a PLAN.md step): why the
# magnetized atmosphere tube is chaotic at rounding level, and what removes it. See CODE.md,
# "Floors, the atmosphere and the magnetosphere", "Investigation (2026-10-16)".
#
#     julia --project=. bin/front_investigation.jl matrix MESH CANDIDATE [runs=48]
#     julia --project=. bin/front_investigation.jl diverge
#     julia --project=. bin/front_investigation.jl precursor
#     julia --project=. bin/front_investigation.jl stageproblems
#     julia --project=. bin/front_investigation.jl emwave
#
# The tube is `bin/ghost_robustness.jl`'s: `atmosphere_tube_case` with `Bʸ = 10⁻³` and
# `MagnetizationCutoff(Constant(10³), 10)`, 200 cells, SSP3(4,3,3), MC, to `t = 2/5` in four
# chunks; the same 48 perturbed initial states (runs 1–12 amplitudes 0, ±1, ±2, ±3, ±5, ±10,
# 20 × 10⁻¹⁵ with seed 1; runs 13–48 amplitude 10⁻¹⁵ with seeds 2–37), and the same numbers for
# the baseline (checked: this script and `bin/ghost_robustness.jl` agree to every digit).
#
# `matrix`: one summary line for 48 runs of a candidate on a mesh — blow-ups (injected `D` above
# 10⁻⁶, `|τ|` above 10⁻³ or non-finite) with a 95% Wilson interval; min / median / max over the
# runs that did not blow up of the injected rest mass and energy (the step limiter's writes),
# OHM_FALLBACK stages, the largest `W` of an owned cell after any step, and the cell-steps with
# `W > 20`; the step-limiter writes the policy flagged `S_FLOORED` and `CEILING`; then the
# hydrodynamic tube (`Bʸ = 0`, `IdealConductor`) with the same candidate, to every digit.
#   MESH: `uniform`; `refined` (`[0, 0.2)` refined; on this tree the parent-centred primitive
#   fill); `conserved` (the same mesh, to be run on the conserved-fill tree, `git archive
#   7605986` with this file copied into its `bin/` and its EntropyEOS at 1.1.0, so that it runs
#   with the wrapper of `src/con2prim.jl` patched in; see CODE.md).
#   CANDIDATE: baseline (the code as it stands), or one of 2026-10-16's alternatives (collapse,
#   wcap10, trigger10 — any `trigger<k>` —, ceil100, vac10, fixeds; see `CANDIDATES` below).
#   Besides the summary line: how far two runs 10⁻¹⁵ apart drift, and the hydrodynamic tube.
#   *(Amended 2026-10-21: the in-process policy emulation `bin/policy_variants.jl` and the
#   reset-side "B" are gone — EntropyEOS 1.1.0 reports `FLAG_POL_ρ_FLOORED` and
#   `src/con2prim.jl` writes the atmosphere at rest on it; the 2026-10-17 sweep's tables were
#   made by commit ec4f044's version of this file.)*
# `diverge`: two runs 10⁻¹⁵ apart, step by step — the largest relative difference of `D` and
# `τ`, and which discrete switches (the recovery's policy flags, the step limiter's writes, the
# stage predictors' policy flags, the stage branch, `σ > 0`) differ.
# `precursor`: the cells ahead of the front (`1.01 ρ_atm < D < 10⁻⁴`) over the run, hydrodynamic
# and magnetized: how close the fluid's share is to the light cone, its entropy, `e_em/τ_f`.
# `stageproblems`: every stage of the unperturbed run that falls back, re-solved under EOS-5, and
# scanned in `u = Wvˣ` for an admissible root.
# `emwave`: a vacuum electromagnetic wave (`σ = 0`) over a tenuous gas at rest, which the
# exact solution leaves at rest: what the scheme does to the fluid.

using Printf, Random, StaticArrays, Statistics
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD
const EOS = TG.EntropyEOS
const IRK = TG.IRK
# *(2026-10-22, branch `claude/no-collapse-to-atmosphere`:)* the entropy-ceiling projection of
# the hot branch, emulated; `HOT_PROJECTION=0` in the environment gives 1.1.0's own policy, 1 the
# first emulation, 2 (the default) EOS-9 as rewritten after Erik's review.
include(joinpath(@__DIR__, "hot_projection.jl"))

# --- the classified step limiter and the monitor ----------------------------------------

const CLASSES = (:atm, :sfl, :sce, :wcap, :ceil, :other)
classify(f) = f & EOS.FLAG_POL_CEILING != 0 ? :ceil :
              f & EOS.FLAG_POL_ATMOSPHERE != 0 ? :atm :
              f & EOS.FLAG_POL_S_FLOORED != 0 ? :sfl :
              f & EOS.FLAG_POL_W_CAPPED != 0 ? :wcap :
              f & EOS.FLAG_POL_S_CEILED != 0 ? :sce : :other

mutable struct Acc
    n::Dict{Symbol,Int}
    D::Dict{Symbol,Float64}
    τ::Dict{Symbol,Float64}
    maxW::Float64
    nW20::Int
end
Acc() = Acc(Dict(c => 0 for c in CLASSES), Dict(c => 0.0 for c in CLASSES),
            Dict(c => 0.0 for c in CLASSES), 1.0, 0)
const ACC = Ref(Acc())

# A cell's stage of a CPU loop over the owned cells: metric, conserved state, `B`, `E`.
function cell_state(p, us, b, i, t)
    f = p.f
    G = f.U.G[1]
    x = coordinates(f.U, b, (i + G,))[1]
    m = TG.point_metric(p.provider, t, SVector(x, 0.0, 0.0))
    cons = ntuple(v -> us[i, v, b], TG.NU)
    B = TG.cell_𝔅((f.B[1].work, f.B[2].work, f.B[3].work), (i,), b, TG.GB_of(f), Val(1)) /
        m.sqrtγ
    return x, m, cons, B, TG.cons_𝔈(cons) / m.sqrtγ
end

# The step limiter, with what each write is classified by the policy's flags.
function classified_reset_step!(u, integ, p, t)
    f = p.f
    TG.update_fields!(p, u, t)
    us = TG.statearray(u, p.sets, TG.SEG_U)
    acc = ACC[]
    for b in 1:nblocks(f.U)
        h = spacing(Float64, f.U.forest, blockkey(f.U, b))
        for i in 1:(f.U.forest.N)
            _, m, cons, B, E = cell_state(p, us, b, i, t)
            prim, fl = con2prim_hydro(p.eos, p.opts, p.pol, m, cons, E, B,
                                      TG.state_guesses(cons))
            fl & EOS.FLAG_POL_ANY == 0 && continue
            c = prim2con(p.eos, m, prim, E, B)
            k = classify(fl)
            acc.n[k] += 1
            acc.D[k] += h * (c[TG.U_D] - cons[TG.U_D])
            acc.τ[k] += h * (c[TG.U_TAU] - cons[TG.U_TAU])
        end
    end
    return TG.reset_step!(u, integ, p, t)
end

function monitor!(acc, p, u, t)
    P = TG.update_primitives!(p, u, t)
    G, N = P.G[1], P.forest.N
    for b in 1:nblocks(P), i in (G + 1):(G + N)
        W = cosh(P.work[i, TG.P_W, b])
        acc.maxW = max(acc.maxW, W)
        W ≤ 20 || (acc.nW20 += 1)
    end
    return acc
end

# --- the candidates' resets ---------------------------------------------------------------

# Experimental changes to the reset (both limiters), on the CPU:
#   μ_ceil   add rest mass at rest in the normal frame (D, D Yₑ, D s scaled; S, τ kept) where
#            B²/D > μ_ceil — the ideal-GRMHD magnetization ceiling;
#   μ_vac    the atmosphere where B²/D > μ_vac (μ_vac = μ_max: every cell the cutoff makes
#            electrovacuum holds the atmosphere);
#   entropy_ratio  where ½(E² + B²) > ratio · τ_f, or τ_f ≤ 0, the fluid recovered at the
#            advected entropy from D and S_f alone (EntropyEOS's internal `inner_solve_w`;
#            EOS-3 would make it public), the energy then rewritten.
Base.@kwdef struct Fix
    μ_ceil::Float64 = 0.0
    μ_vac::Float64 = 0.0
    entropy_ratio::Float64 = 0.0
end

atm_prim(p, m, ye) = let a = EOS.policy_atmosphere(p.eos, p.pol, ye)
    TG.primitive_state(p.eos, m, a.ρ, zero(SVector{3,Float64}), a.s, a.ye)
end

function fixed_s_prim(p, m, D, S_f, τ_f, ye, s)
    Su = TG.raise(m, S_f)
    Smag = sqrt(max(0.0, sum(S_f .* Su)))
    sr = EOS.srange(p.eos, D, ye)
    s = clamp(s, sr.s_min, sr.s_max)
    r, _ = EOS.inner_solve_w(p.eos, D, τ_f, ye, 0.0, Smag, 0.0, s, p.opts.w_max,
                             p.opts.τ_floor_rel, 0.0, NaN, p.opts.max_iter_1d, p.opts.tol)
    w = min(r.w, EOS.pol_w_cap(p.pol))
    D / cosh(w) < EOS.pol_ρ_atm(p.eos, p.pol) && return atm_prim(p, m, ye)
    return TG.primitive_state(p.eos, m, D / cosh(w), TG.velocity_at_rapidity(m, Su, w), s, ye)
end

const MARK = UInt32(1) << 30           # "rewritten by the candidate", beside the policy's flags

function exp_reset!(u, integ, p, t, step::Bool, fix::Fix)
    f = p.f
    TG.update_fields!(p, u, t)
    if p.haswalls
        scatter!(p.sets[TG.SEG_Φ], stateview(u, p.sets, TG.SEG_Φ))
        TG.fill_set!(p, TG.SEG_Φ, t)
        TG.walls_to_state!(p, u)
    end
    us = TG.statearray(u, p.sets, TG.SEG_U)
    X = f.X.work
    GX = f.X.G[1]
    acc = ACC[]
    for b in 1:nblocks(f.U), i in 1:(f.U.forest.N)
        _, m, cons0, B, E = cell_state(p, us, b, i, t)
        cons = cons0
        B² = TG.dot_up(m, B, B)
        D = TG.cons_D(cons) / m.sqrtγ
        ye = TG.cons_DYₑ(cons) / TG.cons_D(cons)
        fl = UInt32(0)
        if fix.μ_ceil > 0 && D > 0 && B² / D > fix.μ_ceil
            k = (B² / fix.μ_ceil) / D
            cons = ntuple(v -> v in (TG.U_D, TG.U_DYE, TG.U_DS) ? cons[v] * k : cons[v], TG.NU)
            fl |= MARK
        end
        prim, pf = con2prim_hydro(p.eos, p.opts, p.pol, m, cons, E, B, TG.state_guesses(cons))
        fl |= pf
        if fix.μ_vac > 0 && (D ≤ 0 || B² / D > fix.μ_vac) && pf & EOS.FLAG_POL_ATMOSPHERE == 0
            prim = atm_prim(p, m, ye)
            fl |= EOS.FLAG_POL_ATMOSPHERE
        elseif fix.entropy_ratio > 0 && pf & EOS.FLAG_POL_ATMOSPHERE == 0
            S_f, τ_f = TG.field_budget(m, TG.cons_S(cons) / m.sqrtγ, TG.cons_τ(cons) / m.sqrtγ, E,
                                       B)
            if !(τ_f > 0) || (TG.dot_up(m, E, E) + B²) / 2 > fix.entropy_ratio * τ_f
                prim2 = fixed_s_prim(p, m, TG.cons_D(cons) / m.sqrtγ, S_f, τ_f, ye,
                                     TG.cons_Ds(cons) / TG.cons_D(cons))
                c2 = prim2con(p.eos, m, prim2, E, B)
                scale = abs(cons[TG.U_TAU])
                if any(k -> abs(c2[k] - cons[k]) > 1e-13 * scale, (TG.U_S:(TG.U_S + 2)..., TG.U_TAU))
                    prim = prim2
                    fl |= MARK
                end
            end
        end
        fl & (EOS.FLAG_POL_ANY | MARK) == 0 && continue
        c = prim2con(p.eos, m, prim, E, B)
        for v in 1:(TG.U_E - 1)
            us[i, v, b] = c[v]
        end
        xi = i + GX
        if step
            X[xi, TG.X_STEP_RESETS, b] += 1
            X[xi, TG.X_INJECTED_D, b] += c[TG.U_D] - cons0[TG.U_D]
            X[xi, TG.X_INJECTED_TAU, b] += c[TG.U_TAU] - cons0[TG.U_TAU]
            h = spacing(Float64, f.U.forest, blockkey(f.U, b))
            k = fl & EOS.FLAG_POL_ANY == 0 ? :other : classify(fl)
            acc.n[k] += 1
            acc.D[k] += h * (c[TG.U_D] - cons0[TG.U_D])
            acc.τ[k] += h * (c[TG.U_TAU] - cons0[TG.U_TAU])
        else
            X[xi, TG.X_RESETS, b] += 1
        end
    end
    return nothing
end

limiters(fix::Fix) = (; stage_limiter=(u, i, p, t) -> exp_reset!(u, i, p, t, false, fix),
                      step_limiter=(u, i, p, t) -> exp_reset!(u, i, p, t, true, fix))

# The case with some of its policy's options replaced.
function with_pol(case; kw...)
    q = case.pol
    nt = merge((; ρ_atm=q.ρ_atm, s_atm=q.s_atm, ye_atm=q.ye_atm, atm_trigger=q.atm_trigger,
                ρ_ceiling=q.ρ_ceiling, w_cap=q.w_cap, collapse_to_atmosphere=q.collapse_to_atmosphere),
               kw)
    pol = EOS.policy_derive_bounds(case.eos, EOS.PolicyOptions{Float64}(; nt...))
    return TG.MHDCase{Float64,1,typeof(case.fields),typeof(case.potential),typeof(case.eos),
                      typeof(case.opts),typeof(pol),typeof(case.model),typeof(case.provider)}(
        case.fields, case.potential, case.eos, case.opts, pol, case.model, case.provider,
        case.𝔅̄, case.periodic, case.extents, case.roots)
end

# name => (label, case transform, Fix or nothing)
const CANDIDATES = Dict(
    "baseline" => ("baseline", identity, nothing),
    "collapse" => ("collapse_to_atmosphere", c -> with_pol(c; collapse_to_atmosphere=true),
                   nothing),
    "wcap10" => ("W cap 10", c -> with_pol(c; w_cap=acosh(10.0)), nothing),
    "ceil100" => ("magnetization ceiling 100", identity, Fix(; μ_ceil=100.0)),
    "vac10" => ("atmosphere where B²/D > 10", identity, Fix(; μ_vac=10.0)),
    "fixeds" => ("fixed-s recovery where e_em > τ_f", identity, Fix(; entropy_ratio=1.0)))

# --- runs -------------------------------------------------------------------------------------

cut(μ) = MagnetizationCutoff(Constant(1e3), μ)
tube(; By=1e-3, model=cut(10), kw...) = atmosphere_tube_case(Float64; By=By, model=model, kw...)
refine_band!(forest) =
    (refine!(forest, [k for k in copy(forest.leaves)
                      if 0 ≤ block_origin(forest, k)[1] < 1 // 5 - 1e-12]);
     balance!(forest); forest)

function run_case(case; amp=0.0, seed=1, t_end=2 // 5, nchunks=4, tableau=IRK.IMEXSSP3433(),
                  step_limiter=classified_reset_step!, stage_limiter=TG.reset_atmosphere!,
                  solve=TG.ohm_solve!, after_step=nothing, kw...)
    t0 = time()
    p = evolve!(case, Val(1); N=10, t_end=1e-12, chunk=1e-12, limiter=:mc, kw...).p
    u = exact_state(p, case, 0.0)
    amp == 0 || (u .*= 1 .+ amp .* randn(MersenneTwister(seed), length(u)))
    ACC[] = Acc()
    t, D, τ = 0.0, 0.0, 0.0
    bits = zeros(Int, TG.N_OHM_BITS)
    for c in 1:nchunks
        t1 = Float64(c * t_end / nchunks)              # c/10, as bin/ghost_robustness.jl
        TG.clear_tally!(p)
        n = TG.chunk_count((t1 - t) * TG.max_light_rate(p, t), 2 / 5)
        prob = IRK.IMEXProblem(TG.mhd_rhs!, solve, u, (t, t1), p)
        integ = IRK.init(prob, tableau; dt=(t1 - t) / n, stage_limiter=stage_limiter,
                         step_limiter=step_limiter, partition=:even, alias_u0=true)
        while integ.nstep < integ.nsteps
            IRK.step!(integ)
            monitor!(ACC[], p, integ.u, integ.t)
            after_step === nothing || after_step(p, integ.u, integ.t)
        end
        u = integ.u
        tl = ohm_tally(p)
        D += tl.injected_D
        τ += tl.injected_τ
        bits .+= collect(tl.bits)
        t = t1
    end
    blowup = !(isfinite(D) && isfinite(τ) && D ≤ 1e-6 && abs(τ) ≤ 1e-3)
    return (; acc=ACC[], D, τ, fb=bits[5], bits, blowup, p, u, wall=time() - t0)
end

const AMPS12 = (0.0, 1e-15, -1e-15, 2e-15, -2e-15, 3e-15, -3e-15, 5e-15, -5e-15, 1e-14, -1e-14,
                2e-14)
perts(nruns) = [(k ≤ 12 ? (AMPS12[k], 1) : (1e-15, k - 10)) for k in 1:nruns]

function wilson(k, n; z=1.959964)
    p = k / n
    d = 1 + z^2 / n
    c = (p + z^2 / (2n)) / d
    w = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / d
    return (max(0, c - w), min(1, c + w))
end

fmt3(x; f="%.2g") = isempty(x) ? "–" :
                    Printf.format(Printf.Format("$f / $f / $f"), minimum(x), median(x), maximum(x))

function candidate_kw(name)
    if startswith(name, "trigger")
        k = parse(Float64, name[8:end])
        return "atm_trigger $k", c -> with_pol(c; atm_trigger=k), (;)
    end
    lab, cf, fix = CANDIDATES[name]
    return lab, cf, fix === nothing ? (;) : limiters(fix)
end

function batch(label, case; nruns=48, kw...)
    rs = [run_case(case; amp=a, seed=s, kw...) for (a, s) in perts(nruns)]
    nb = count(r -> r.blowup, rs)
    lo, hi = wilson(nb, nruns)
    ok = filter(r -> !r.blowup, rs)
    @printf("%-44s | %2d/%d (%.1f–%.1f%%) | D %s | τ %s | fb %s | max W %s | W>20 %s | writes %s | S_FLOORED %s | CEILING %s | %.2f s\n",
            label, nb, nruns, 100lo, 100hi, fmt3([r.D for r in ok]), fmt3([r.τ for r in ok]),
            fmt3([r.fb for r in ok]; f="%.0f"), fmt3([r.acc.maxW for r in ok]; f="%.1f"),
            fmt3([r.acc.nW20 for r in ok]; f="%.0f"),
            fmt3([sum(values(r.acc.n)) for r in rs]; f="%.0f"),
            fmt3([r.acc.n[:sfl] for r in rs]; f="%.0f"),
            fmt3([r.acc.n[:ceil] for r in rs]; f="%.0f"), mean(r.wall for r in rs))
    return rs
end

function matrix(mesh, name; nruns=48)
    kw = mesh == "uniform" ? (;) : (; refine=refine_band!)
    lab, cf, lkw = candidate_kw(name)
    lab = lab * " [policy v$(HOT_PROJECTION[])]"
    HOT_COUNT .= 0
    batch("$mesh, $lab", cf(tube()); nruns=nruns, kw..., lkw...)
    @printf("%-44s   hot branch over the %d runs (every recovery): projections %d, to the atmosphere %d, rapidity capped %d, pol_collapse %d\n",
            "$mesh, $lab", nruns, HOT_COUNT...)
    r1 = run_case(cf(tube()); kw..., lkw...)
    r2 = run_case(cf(tube()); amp=1e-15, seed=7, kw..., lkw...)
    a = TG.statearray(r1.u, r1.p.sets, TG.SEG_U)
    b = TG.statearray(r2.u, r2.p.sets, TG.SEG_U)
    @printf("%-44s   two runs 1e-15 apart: max |ΔD|/D %.3g, max |Δτ|/max |τ| %.3g\n", "$mesh, $lab",
            maximum(abs.(a[:, TG.U_D, :] .- b[:, TG.U_D, :]) ./ abs.(a[:, TG.U_D, :])),
            maximum(abs.(a[:, TG.U_TAU, :] .- b[:, TG.U_TAU, :])) / maximum(abs.(a[:, TG.U_TAU, :])))
    h = run_case(cf(atmosphere_tube_case(Float64; By=0)); kw..., lkw...)
    @printf("%-44s   hydrodynamic: D %.10e, τ %.10e, max W %.10f, fallbacks %d, writes %d\n",
            "$mesh, $lab", h.D, h.τ, h.acc.maxW, h.fb, sum(values(h.acc.n)))
    return nothing
end

# --- two runs 10⁻¹⁵ apart -------------------------------------------------------------------

function step_record(p, u, t)
    P = TG.update_primitives!(p, u, t)
    us = TG.statearray(u, p.sets, TG.SEG_U)
    G, N = P.G[1], P.forest.N
    rows = [(x=coordinates(P, b, (i + G,))[1], D=us[i, TG.U_D, b], τ=us[i, TG.U_TAU, b],
             W=cosh(P.work[i + G, TG.P_W, b]), s=P.work[i + G, TG.P_S, b],
             flags=unsafe_trunc(UInt32, P.work[i + G, TG.P_FLAGS, b]),
             inj=p.f.X.work[i + p.f.X.G[1], TG.X_INJECTED_D, b])
            for b in 1:nblocks(P) for i in 1:N]
    return rows
end

const STAGE_LOG = Ref{Any}(nothing)
function logged_ohm!(U, ustar, γΔt, p, t)
    TG.ohm_solve!(U, ustar, γΔt, p, t)
    us = TG.statearray(ustar, p.sets, TG.SEG_U)
    rec = Tuple{UInt32,UInt32,Bool}[]
    for b in 1:nblocks(p.f.U), i in 1:(p.f.U.forest.N)
        _, m, cons, B, E★ = cell_state(p, us, b, i, t)
        _, prim, r, _ = TG.solve_ohm_stage(p.eos, p.opts, p.pol, p.model, m, cons, E★, B, γΔt,
                                           TG.state_guesses(cons); stage=p.stage)
        S_f, τ_f = TG.field_budget(m, TG.cons_S(cons), TG.cons_τ(cons), E★, B)
        pr, pf, _ = TG.fluid_recovery(p.eos, p.opts, p.pol, m, TG.cons_D(cons), S_f, τ_f,
                                      TG.cons_DYₑ(cons), TG.state_guesses(cons))
        σ = TG.conductivity(p.model, pr, TG.magnetization(m, B, TG.cons_D(cons)))
        push!(rec, (pf, r.flags & 0x1f, σ > 0))
    end
    push!(STAGE_LOG[], rec)
    return nothing
end

function traced(case; kw...)
    steps = []
    STAGE_LOG[] = []
    run_case(case; solve=logged_ohm!, after_step=(p, u, t) -> push!(steps, (t, step_record(p, u, t))),
             kw...)
    return steps, STAGE_LOG[]
end

function diverge(; case=tube(), nshow=60)
    A, sA = traced(case)
    B, sB = traced(case; amp=1e-15, seed=1)
    println("step  t       max|ΔD|/D  max|Δτ|/|τ|  max W (A, B)   discrete switches that differ (cell numbers)")
    for k in eachindex(A)
        a, b = A[k][2], B[k][2]
        dD = maximum(abs(a[j].D - b[j].D) / abs(a[j].D) for j in eachindex(a))
        dτ = maximum(abs(a[j].τ - b[j].τ) / abs(a[j].τ) for j in eachindex(a))
        sw = String[]
        d = findall(j -> a[j].flags != b[j].flags, eachindex(a))
        isempty(d) || push!(sw, "recovery flags $d")
        d = findall(j -> (abs(a[j].inj) > 1e-20) != (abs(b[j].inj) > 1e-20), eachindex(a))
        isempty(d) || push!(sw, "step-limiter writes $d")
        for st in 1:4
            ra, rb = sA[4(k - 1) + st], sB[4(k - 1) + st]
            for (q, name) in ((1, "predictor flags"), (2, "branch"), (3, "σ > 0"))
                d = findall(j -> ra[j][q] != rb[j][q], eachindex(ra))
                isempty(d) || push!(sw, "stage $st $name $d")
            end
        end
        if k ≤ nshow
            @printf("%4d %.4f  %9.2e  %9.2e  %6.1f %6.1f   %s\n", k, A[k][1], dD, dτ,
                    maximum(r.W for r in a), maximum(r.W for r in b), join(sw, "; "))
        end
    end
    return A, B
end

# --- the precursor, the stage problems, the EM wave -------------------------------------------

function precursor(lab, case)
    rat, ss, emr = Float64[], Float64[], Float64[]
    run_case(case; after_step=function (p, u, t)
                 P = TG.update_primitives!(p, u, t)
                 us = TG.statearray(u, p.sets, TG.SEG_U)
                 G = P.G[1]
                 for b in 1:nblocks(P), i in 1:(P.forest.N)
                     D = us[i, TG.U_D, b]
                     1.02e-8 < D < 1e-4 || continue
                     E = SVector(ntuple(k -> P.work[i + G, TG.P_E + k - 1, b], 3))
                     B = SVector(ntuple(k -> P.work[i + G, TG.P_B + k - 1, b], 3))
                     S = SVector(ntuple(k -> us[i, TG.U_S + k - 1, b], 3))
                     m = TG.flat_metric(Float64)
                     S_f, τ_f = TG.field_budget(m, S, us[i, TG.U_TAU, b], E, B)
                     push!(rat, sqrt(sum(S_f .^ 2)) / (τ_f + D))
                     push!(ss, P.work[i + G, TG.P_S, b])
                     push!(emr, (sum(E .^ 2) + sum(B .^ 2)) / 2 / τ_f)
                 end
             end)
    @printf("%-12s ahead of the front (1.02e-8 < D < 1e-4): %d cell-steps; |S_f|/(τ_f + D) median %.3f, max %.4f; s median %.2f, max %.2f (window 1 … 40, atmosphere 20.5); e_em/τ_f median %.3g, max %.3g\n",
            lab, length(rat), median(rat), maximum(rat), median(ss), maximum(ss),
            median(emr), maximum(emr))
end

function stageproblems(; case=tube())
    probs = []
    capture!(U, ustar, γΔt, p, t) = begin
        TG.ohm_solve!(U, ustar, γΔt, p, t)
        us = TG.statearray(ustar, p.sets, TG.SEG_U)
        for b in 1:nblocks(p.f.U), i in 1:(p.f.U.forest.N)
            _, m, cons, B, E★ = cell_state(p, us, b, i, t)
            _, _, r, _ = TG.solve_ohm_stage(p.eos, p.opts, p.pol, p.model, m, cons, E★, B, γΔt,
                                            TG.state_guesses(cons); stage=p.stage)
            r.flags & TG.OHM_FALLBACK != 0 && push!(probs, (p, m, cons, B, E★, γΔt))
        end
    end
    run_case(case; solve=capture!)
    μs, fs = Float64[], Float64[]
    noadm, withroot, noroot = 0, 0, 0
    for (p, m, cons, B, E★, γΔt) in probs
        D, S, τ, DY = TG.cons_D(cons), TG.cons_S(cons), TG.cons_τ(cons), TG.cons_DYₑ(cons)
        S_f, τ_f = TG.field_budget(m, S, τ, E★, B)
        pr, _, _ = TG.fluid_recovery(p.eos, p.opts, p.pol, m, D, S_f, τ_f, DY, TG.state_guesses(cons))
        μ = TG.magnetization(m, B, D)
        push!(μs, μ)
        vB = TG.raise(m, TG.cross_down(m, TG.prim_v(pr), B))
        push!(fs, sqrt(TG.dot_up(m, E★ + vB, E★ + vB) / TG.dot_up(m, B, B)))
        c = γΔt * m.α * TG.conductivity(p.model, pr, μ)
        anyadm, root, prev = false, false, NaN
        for a in range(-6, 6; length=4001)          # u = sinh a, W up to 200
            u = SVector(sinh(a), 0.0, 0.0)
            S_E, τ_E = TG.field_budget(m, S, τ, TG.ohm_E(m, c, u, E★, B), B)
            q, fl, _ = TG.fluid_recovery(p.eos, p.opts, p.pol, m, D, S_E, τ_E, DY,
                                         TG.state_guesses(cons))
            if fl == 0
                anyadm = true
                g = TG.four_velocity(q)[1] - u[1]
                isfinite(prev) && sign(g) != sign(prev) && (root = true)
                prev = g
            else
                prev = NaN
            end
        end
        anyadm ? (root ? (withroot += 1) : (noroot += 1)) : (noadm += 1)
    end
    @printf("stages that fall back: %d, μ = B²/D in [%.3g, %.3g], f = |E★ + v×B|/|B| in [%.3g, %.3g]\n",
            length(probs), extrema(μs)..., extrema(fs)...)
    @printf("  no u gives an admissible fluid share: %d; admissible shares and a root: %d; admissible shares, no root: %d\n",
            noadm, withroot, noroot)
end

function emwave(; ρ, b=1e-3, roots=5, periods=1)
    case = em_wave_case(Float64; a=0, b=b, ρ=ρ, p=ρ^(5 / 3), roots=roots)
    r = evolve!(case, Val(1); N=10, t_end=periods, chunk=periods, limiter=:mc)
    P = TG.update_primitives!(r.p, r.u, r.t)
    G = P.G[1]
    v = [P.work[i, TG.P_V, b] for b in 1:nblocks(P) for i in (G + 1):(G + P.forest.N)]
    s = [P.work[i, TG.P_S, b] for b in 1:nblocks(P) for i in (G + 1):(G + P.forest.N)]
    @printf("ρ = %.0e (s = 1, window 1/2 … 3/2), ½b² = %.1e, %d cells, %d period(s): vˣ in [%.3g, %.3g], s in [%.4g, %.4g], step-limiter writes %d\n",
            ρ, b^2 / 2, roots * 10, periods, extrema(v)..., extrema(s)..., r.chunks[end].tally.step_resets)
end

# --- main ---------------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    mode = ARGS[1]
    nruns = let a = findfirst(s -> startswith(s, "runs="), ARGS)
        a === nothing ? 48 : parse(Int, ARGS[a][6:end])
    end
    if mode == "matrix"
        matrix(ARGS[2], ARGS[3]; nruns=nruns)
    elseif mode == "diverge"
        diverge()
    elseif mode == "precursor"
        precursor("hydrodynamic", atmosphere_tube_case(Float64; By=0))
        precursor("magnetized", tube())
    elseif mode == "stageproblems"
        stageproblems()
    elseif mode == "emwave"
        for (ρ, roots, periods) in ((1.0, 5, 1), (1e-3, 5, 1), (1e-5, 5, 1), (1e-7, 5, 1),
                                    (1e-7, 20, 1), (1e-7, 80, 1), (1e-7, 5, 4))
            emwave(; ρ=ρ, roots=roots, periods=periods)
        end
    else
        error("unknown mode $mode")
    end
end
