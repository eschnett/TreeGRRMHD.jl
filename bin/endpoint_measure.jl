# Experiment (branch `claude/no-collapse-to-atmosphere`, 2026-10-22): measurements behind the
# rewritten EOS-9 (PLAN.md) — `con2prim_safe`'s step 4, the two endpoint branches (cold: τ below
# the coldest state at the momentum, entropy floored; hot: τ above the hottest, today the collapse
# ceiling, under EOS-9 the entropy ceiling), on EntropyEOS's synthetic table (its entropy window
# depends on ρ; the analytic EOSes' does not), in g/cm³ and through `GeometricEOS`, and in the
# atmosphere tubes.
#
#     julia --project=. bin/endpoint_measure.jl table     # A1 and A2 on the table
#     julia --project=. bin/endpoint_measure.jl tubes     # A2 in the tubes
#
# Inputs reaching step 4: random valid states across the box (log ρ uniform, s uniform in the
# window at ρ, Yₑ uniform, w ∈ [0, 4]; hydrodynamic mode, B² = 0, and MHD mode, B² = z·10^U(−2,1)
# with a random angle between v and B), conserved by EntropyEOS's `prim2con`, with τ then scaled
# down (×10^−U(0.05,3), aiming at the cold branch) or up (×10^U(0.3,6), the hot one); kept if the
# solve fails and the input passes step 1. Per branch, four versions of the diagnosis and the
# projection:
#   released  1.1.0: the endpoint solve at `sr_ref = srange(D/cosh w_ref)`'s edge, the output at
#             ρ₁ = D/cosh w₁ with the window re-taken at ρ₁ (cold) or the collapse ceiling (hot);
#   EOS-9 v1  2026-10-22's first emulation (`bin/hot_projection.jl`): the hot output at
#             (D/cosh w₁, sr_ref.s_max) projected; the cold as released;
#   fixed     ρ ↦ s_edge(ρ) ↦ w (the 1D solve) ↦ ρ = D/cosh w, to |Δρ| ≤ 1e-13 ρ, from ρ_ref;
#   edge      one root find of f₁(w) at s = s_edge(D/cosh w) inside the residual (Illinois on
#             [0, w_max]), used for the diagnosis and the projection.
# Measured: how far the edge at ρ₁ is from the edge at ρ_ref (as a fraction of the window), how
# often the diagnosis flips at ρ₁, how far s_out lands from the edge at ρ_out, the momentum error
# |S_out − S_in|/|S_in| of unclamped outputs, passes and iterations; and (A2) how each endpoint
# solve ends — converged (f₁ to tolerance), capped legitimately (w = w_max with f₁ < 0: the root
# lies beyond), unconverged (iterations exhausted or bracket collapsed elsewhere), non-finite — and
# whether the diagnosis flips with a solve 1000× tighter and 10× longer.

using Printf, Random, Statistics
using TreeGRRMHD
const E = TreeGRRMHD.EntropyEOS
include(joinpath(@__DIR__, "hot_projection.jl"))

solve_at(eos, in, ye, B², s, opts; w_start, u=NaN, tol=opts.tol, maxit=opts.max_iter_1d) =
    E.inner_solve_w(eos, in.D, in.τ, ye, in.S_par, in.S_perp, B², s, opts.w_max, opts.τ_floor_rel,
                    w_start, u, maxit, tol)[1]

function solve_class(r, opts)
    isfinite(r.w) || return :nonfinite
    E.f1_converged(r, opts.tol) && return :converged
    (r.w ≥ opts.w_max * (1 - 1e-12) && r.f1 < 0) && return :capped
    return :unconverged
end

edge(b, eos, ρ, ye) = let sr = E.srange(eos, ρ, ye)
    b === :cold ? sr.s_min : sr.s_max
end
width(eos, ρ, ye) = let sr = E.srange(eos, ρ, ye)
    sr.s_max - sr.s_min
end
decides(b, r) = b === :cold ? r.f2 > 0 : r.f2 < 0

# Released 1.1.0's step-4 diagnosis, replicated, or `nothing` if the input does not reach it.
function step4(eos, in, opts, pol)
    B² = max(in.B², zero(in.B²))
    in.D < E.pol_atm_trigger(pol) * E.pol_ρ_atm(eos, pol) && return nothing
    (in.D > pol.D_max || in.τ - B² > pol.τ_max) && return :step1
    base = E.con2prim(eos, in, opts)
    (base.result === E.C2PResult.converged_newton ||
     base.result === E.C2PResult.converged_fallback) && return nothing
    ye_in = in.D_Y / in.D
    y_lo, y_hi = E.yₑ_bounds(eos)
    ye_c = clamp(ye_in, y_lo, y_hi)
    w_ref = isfinite(base.w) ? clamp(base.w, 0.0, opts.w_max) : 0.0
    ρ_ref = in.D / cosh(w_ref)
    sr = E.srange(eos, ρ_ref, ye_c)
    r_lo = solve_at(eos, in, ye_in, B², sr.s_min, opts; w_start=w_ref, u=base.eos.u_solved)
    r_hi = solve_at(eos, in, ye_in, B², sr.s_max, opts; w_start=r_lo.w, u=r_lo.pt.u_solved)
    b = r_lo.f2 > 0 ? :cold : r_hi.f2 < 0 ? :hot : :excise
    return (; b, w_ref, ρ_ref, ye_in, ye_c, B², r_lo, r_hi, u=base.eos.u_solved)
end

# The fixed point ρ ↦ s_edge(ρ) ↦ w ↦ D/cosh w.
function fixed_point(b, eos, in, opts, d)
    ρ, w = d.ρ_ref, d.w_ref
    r = nothing
    for k in 1:100
        r = solve_at(eos, in, d.ye_in, d.B², edge(b, eos, ρ, d.ye_c), opts; w_start=w)
        isfinite(r.w) || return (; ok=false, k, r)
        ρn = in.D / cosh(r.w)
        w = r.w
        abs(ρn - ρ) ≤ 1e-13 * ρ && return (; ok=true, k, r)
        ρ = ρn
    end
    return (; ok=false, k=100, r)
end

# One root find of f₁ with the edge entropy at the trial density (Illinois).
function edge_solve(b, eos, in, opts, d)
    res(w) = E.residuals(eos, in.D, in.τ, d.ye_in, in.S_par, in.S_perp, d.B²,
                         edge(b, eos, in.D / cosh(w), d.ye_c), w, NaN, opts.τ_floor_rel)
    a, c = 0.0, opts.w_max
    ra, rc = res(a), res(c)
    rc.f1 < 0 && return (; ok=true, capped=true, k=0, r=rc)
    ra.f1 ≥ 0 && return (; ok=true, capped=false, k=0, r=ra)
    fa, fc, side = ra.f1, rc.f1, 0
    r = rc
    for k in 1:200
        w = (a * fc - c * fa) / (fc - fa)
        r = res(w)
        E.f1_converged(r, opts.tol) && return (; ok=true, capped=false, k, r)
        if r.f1 < 0
            a, fa = w, r.f1
            side == -1 && (fc /= 2)
            side = -1
        else
            c, fc = w, r.f1
            side == 1 && (fa /= 2)
            side = 1
        end
        c - a ≤ 4eps(c) && return (; ok=false, capped=false, k, r)
    end
    return (; ok=false, capped=false, k=200, r)
end

smag(c) = sqrt(c.S_par^2 + c.S_perp^2)
momentum_error(eos, in, ρ, s, ye, w, B²) =
    abs(smag(E.pol_package(eos, E.PrimState{Float64}(ρ, s, ye, w), in.S_par, in.S_perp, B², NaN)[2]) -
        smag(in)) / smag(in)
const CLAMPS = E.FLAG_POL_ρ_FLOORED | E.FLAG_POL_ρ_CEILED | E.FLAG_POL_W_CAPPED |
               E.FLAG_POL_ATMOSPHERE | E.FLAG_POL_CEILING

q(x) = isempty(x) ? "–" : @sprintf("%.2g / %.2g", median(x), maximum(x))
frac(n, N) = @sprintf("%d/%d", n, N)

function table_study(lab, eos, ρ_atm; n=4000, seed=1)
    opts = E.Con2PrimOptions{Float64}()
    pol = E.default_policy(eos, ρ_atm)
    rng = Xoshiro(seed)
    x_lo, x_hi = E.logρ_bounds(eos)
    y_lo, y_hi = E.yₑ_bounds(eos)
    println("\n## $lab: ρ in 10^[$(round(x_lo; digits=2)), $(round(x_hi; digits=2))], ρ_atm $ρ_atm")
    for mode in (:hydro, :mhd), aim in (:cold, :hot)
        recs = []
        nstep1 = 0
        for _ in 1:n
            ρ = exp10(x_lo + (x_hi - x_lo) * rand(rng))
            ye = y_lo + (y_hi - y_lo) * rand(rng)
            sr = E.srange(eos, ρ, ye)
            s = sr.s_min + (sr.s_max - sr.s_min) * rand(rng)
            w = 4 * rand(rng)
            h = E.evaluate(eos, ρ, s, ye, NaN).h
            z = ρ * h * cosh(w)^2
            B² = mode === :hydro ? 0.0 : z * exp10(-2 + 3 * rand(rng))
            cosvB = mode === :hydro ? 0.0 : 2 * rand(rng) - 1
            c = E.prim2con(eos, ρ, s, ye, w, B², cosvB, NaN)
            fτ = aim === :cold ? exp10(-(0.05 + 2.95 * rand(rng))) : exp10(0.3 + 5.7 * rand(rng))
            in = E.Con2PrimIn{Float64}(c.D, c.τ * fτ, c.D_Y, c.S_par, c.S_perp, c.B²)
            d = step4(eos, in, opts, pol)
            d === :step1 && (nstep1 += 1; continue)
            d === nothing && continue
            push!(recs, (in, d))
        end
        bs = [d.b for (_, d) in recs]
        @printf("%-5s aimed %-4s: %d inputs reach step 4 (cold %d, hot %d, excise %d); %d stopped by step 1\n",
                mode, aim, length(recs), count(==(:cold), bs), count(==(:hot), bs),
                count(==(:excise), bs), nstep1)
        for b in (:cold, :hot)
            R = [(in, d) for (in, d) in recs if d.b === b]
            isempty(R) && continue
            Δedge, flips, gapR, errR, gapE9, errE9, clR, clE9 = Float64[], 0, Float64[], Float64[],
                                                              Float64[], Float64[], 0, 0
            gapV2, errV2, clV2, v2flag = Float64[], Float64[], 0, Dict{UInt32,Int}()
            fpk, fpfail, fperr, esk, esfail, eserr, agree, cls = Int[], 0, Float64[], Int[], 0,
                                                                  Float64[], 0, Dict{Symbol,Int}()
            longflip = 0
            for (in, d) in R
                r = b === :cold ? d.r_lo : d.r_hi
                cls[solve_class(r, opts)] = get(cls, solve_class(r, opts), 0) + 1
                w1 = isfinite(r.w) ? clamp(r.w, 0.0, E.pol_w_cap(pol)) : 0.0
                ρ1 = in.D / cosh(w1)
                wr = width(eos, d.ρ_ref, d.ye_c)
                push!(Δedge, abs(edge(b, eos, ρ1, d.ye_c) - edge(b, eos, d.ρ_ref, d.ye_c)) / wr)
                r1 = solve_at(eos, in, d.ye_in, d.B², edge(b, eos, ρ1, d.ye_c), opts; w_start=w1)
                decides(b, r1) || (flips += 1)
                rl = solve_at(eos, in, d.ye_in, d.B², edge(b, eos, d.ρ_ref, d.ye_c), opts;
                              w_start=0.0, tol=opts.tol / 1000, maxit=10 * opts.max_iter_1d)
                decides(b, rl) || (longflip += 1)
                for (on, gap, err) in ((0, gapR, errR), (1, gapE9, errE9), (2, gapV2, errV2))
                    HOT_PROJECTION[] = on
                    o = E.con2prim_safe(eos, in, opts, pol)
                    HOT_PROJECTION[] = 0
                    on == 2 && (v2flag[o.policy_flags] = get(v2flag, o.policy_flags, 0) + 1)
                    if o.policy_flags & CLAMPS != 0
                        on == 2 ? (clV2 += 1) : on == 1 ? (clE9 += 1) : (clR += 1)
                        continue
                    end
                    ρo = o.base.ρ
                    push!(gap, abs(o.base.s - edge(b, eos, ρo, d.ye_c)) / width(eos, ρo, d.ye_c))
                    push!(err, abs(smag(o.cons) - smag(in)) / smag(in))
                end
                fp = fixed_point(b, eos, in, opts, d)
                push!(fpk, fp.k)
                if fp.ok
                    ρf = in.D / cosh(fp.r.w)
                    push!(fperr, momentum_error(eos, in, ρf, edge(b, eos, ρf, d.ye_c), d.ye_in,
                                                fp.r.w, d.B²))
                else
                    fpfail += 1
                end
                es = edge_solve(b, eos, in, opts, d)
                push!(esk, es.k)
                if es.ok
                    ρe = in.D / cosh(es.r.w)
                    push!(eserr, momentum_error(eos, in, ρe, edge(b, eos, ρe, d.ye_c), d.ye_in,
                                                es.r.w, d.B²))
                    fp.ok && decides(b, es.r) == decides(b, fp.r) &&
                        abs(es.r.w - fp.r.w) ≤ 1e-9 * max(1, fp.r.w) && (agree += 1)
                else
                    esfail += 1
                end
            end
            N = length(R)
            @printf("  %-4s branch, %d inputs; endpoint solve %s\n", b, N,
                    join(["$k $v" for (k, v) in sort(collect(cls))], ", "))
            @printf("    edge at ρ₁ vs at ρ_ref (fraction of the window): %s; differs > 1e-6 in %s; diagnosis flips at ρ₁ in %s; with a 1000× tighter, 10× longer solve in %s\n",
                    q(Δedge), frac(count(>(1e-6), Δedge), N), frac(flips, N), frac(longflip, N))
            @printf("    released: clamped %s; unclamped: |s_out − edge(ρ_out)| %s, |ΔS|/|S| %s\n",
                    frac(clR, N), q(gapR), q(errR))
            @printf("    EOS-9 v1: clamped %s; unclamped: |s_out − edge(ρ_out)| %s, |ΔS|/|S| %s\n",
                    frac(clE9, N), q(gapE9), q(errE9))
            @printf("    EOS-9 v2: clamped %s; unclamped: |s_out − edge(ρ_out)| %s, |ΔS|/|S| %s; flags %s\n",
                    frac(clV2, N), q(gapV2), q(errV2),
                    join(["0x$(string(k; base=16)) ×$v" for (k, v) in sort(collect(v2flag))], ", "))
            @printf("    fixed point: passes %s (max %d), not converged %s; |ΔS|/|S| %s\n",
                    q(Float64.(fpk)), maximum(fpk), frac(fpfail, N), q(fperr))
            @printf("    edge solve: iterations %s (max %d), failed %s; |ΔS|/|S| %s; same diagnosis and w as the fixed point %s\n",
                    q(Float64.(esk)), maximum(esk), frac(esfail, N), q(eserr), frac(agree, N))
        end
    end
end

# A2 in the tubes: every step-4 diagnosis a run makes.
const TUBE_LOG = Dict{Symbol,Int}()
function Main.endpoint_record(eos, in, opts, ye_in, B², sr_ref, w_ref, u_warm, r_lo)
    c = solve_class(r_lo, opts)
    TUBE_LOG[Symbol(:cold_, c)] = get(TUBE_LOG, Symbol(:cold_, c), 0) + 1
    rl = solve_at(eos, in, ye_in, B², sr_ref.s_min, opts; w_start=w_ref, tol=opts.tol / 1000,
                  maxit=10 * opts.max_iter_1d)
    (r_lo.f2 > 0) == (rl.f2 > 0) || (TUBE_LOG[:cold_flip] = get(TUBE_LOG, :cold_flip, 0) + 1)
    r_lo.f2 > 0 && return nothing
    r_hi = solve_at(eos, in, ye_in, B², sr_ref.s_max, opts; w_start=r_lo.w, u=r_lo.pt.u_solved)
    c = solve_class(r_hi, opts)
    TUBE_LOG[Symbol(:hot_, c)] = get(TUBE_LOG, Symbol(:hot_, c), 0) + 1
    rh = solve_at(eos, in, ye_in, B², sr_ref.s_max, opts; w_start=r_lo.w, tol=opts.tol / 1000,
                  maxit=10 * opts.max_iter_1d)
    (r_hi.f2 < 0) == (rh.f2 < 0) || (TUBE_LOG[:hot_flip] = get(TUBE_LOG, :hot_flip, 0) + 1)
    TUBE_LOG[r_hi.f2 < 0 ? :hot : :excise] = get(TUBE_LOG, r_hi.f2 < 0 ? :hot : :excise, 0) + 1
    return nothing
end

function tubes()
    cut = MagnetizationCutoff(Constant(1e3), 10)
    IRK = TreeGRRMHD.IRK
    for (lab, case, kw) in (("cut-off tube", atmosphere_tube_case(Float64; By=1e-3, model=cut), (;)),
                            ("Bʸ = 1e-2", atmosphere_tube_case(Float64; By=1e-2, model=cut), (;)),
                            ("ideal, SSP3", atmosphere_tube_case(Float64; By=1e-3), (;)),
                            ("ideal, ARS", atmosphere_tube_case(Float64; By=1e-3),
                             (; tableau=IRK.ARS443())))
        empty!(TUBE_LOG)
        ENDPOINT_HOOK[] = true
        HOT_PROJECTION[] = 1        # the hook sits in 1.1.0's diagnosis, which v2 replaces
        evolve!(case, Val(1); N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc, kw...)
        ENDPOINT_HOOK[] = false
        println(rpad(lab, 14), join(["$k $v" for (k, v) in sort(collect(TUBE_LOG))], ", "))
    end
end

HOT_PROJECTION[] = 0
if ARGS[1] == "table"
    t = E.build_eos(E.make_synthetic_table())
    x_lo, _ = E.logρ_bounds(t.view)
    table_study("synthetic table, g/cm³", t.view, exp10(x_lo))
    g = GeometricEOS(t.view; density=MSUN_DENSITY_CGS)
    table_study("synthetic table through GeometricEOS", g, exp10(x_lo) / MSUN_DENSITY_CGS)
elseif ARGS[1] == "tubes"
    tubes()
end
