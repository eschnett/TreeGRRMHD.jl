# Experiment (branch `claude/no-collapse-to-atmosphere`), NOT part of the package: EOS-9 (PLAN.md)
# emulated by redefining `EntropyEOS.con2prim_safe` in the running process. The function is
# EntropyEOS 1.1.0's own (origin/main 43923c7, `src/core/state_policy.jl`), copied verbatim but for
# the parts selected by `HOT_PROJECTION[]` (the environment's `HOT_PROJECTION`, default 2):
#   0  1.1.0 bit for bit;
#   1  2026-10-22's first emulation: the hot branch projected onto sr_ref's entropy ceiling, D kept;
#   2  EOS-9 as rewritten 2026-10-22 after Erik's review: both endpoint branches by one edge solve
#      (s = s_edge(D/cosh w) inside the 1D root find) with guards, the density clamped into
#      [ρ_atm, ρ_ceiling] and reported (ρ_FLOORED/ρ_CEILED, never the atmosphere), S_CEILED for the
#      hot branch, the atmosphere's flags without the discarded state's, step 1's τ_max test gone
#      and the D_max collapse reporting ρ_CEILED.
# The sibling checkout is not touched and nothing in `src/` loads this file. `HOT_COUNT` counts over
# every recovery the process makes: hot projections, those floored (v2) or sent to the atmosphere
# (v1), those with the rapidity capped, and calls of `pol_collapse`. For the suite:
#
#     julia --project=. -e 'using Pkg; Pkg.test(; julia_args=["-L", "<abs path>/bin/hot_projection.jl"])'
import TreeGRRMHD
isdefined(Main, :HOT_PROJECTION) ||
    @eval Main const HOT_PROJECTION = Ref(parse(Int, get(ENV, "HOT_PROJECTION", "2")))
isdefined(Main, :HOT_COUNT) || @eval Main const HOT_COUNT = zeros(Int, 4)
# A hook on step 4's diagnosis, for `bin/endpoint_measure.jl` (off by default).
isdefined(Main, :ENDPOINT_HOOK) || @eval Main const ENDPOINT_HOOK = Ref(false)
isdefined(Main, :endpoint_record) || @eval Main endpoint_record(args...) = nothing
@eval TreeGRRMHD.EntropyEOS begin
function con2prim_safe(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                       pol::PolicyOptions{T}, s_guess::T=T(NaN), w_guess::T=T(NaN),
                       u_guess::T=T(NaN)) where {S,T}
    B² = (isfinite(in.B²) && in.B² > zero(T)) ? in.B² : zero(T)
    fin = isfinite(in.D) && isfinite(in.τ) && isfinite(in.D_Y) && isfinite(in.S_par) &&
          isfinite(in.S_perp) && isfinite(in.B²)

    # 0. Garbage in: excise to atmosphere rather than propagate it.
    if !(fin && in.D > zero(T) && in.B² >= zero(T))
        ye_in = (isfinite(in.D) && in.D > zero(T) && isfinite(in.D_Y)) ? in.D_Y / in.D : T(NaN)
        ps = policy_atmosphere(eos, pol, ye_in)
        base, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
        return Con2PrimSafeOut{T}(base, cons, FLAG_POL_NONFINITE | FLAG_POL_ATMOSPHERE, false)
    end

    ye_in = in.D_Y / in.D

    # 1. Vacuum and collapse, both pure arithmetic, before any solve is attempted.
    if in.D < pol_atm_trigger(pol) * pol_ρ_atm(eos, pol)
        ps = policy_atmosphere(eos, pol, ye_in)
        base, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
        return Con2PrimSafeOut{T}(base, cons, FLAG_POL_ATMOSPHERE, false)
    end

    D_over = isfinite(pol.D_max) && pol.D_max > zero(T) && in.D > pol.D_max
    τ_over = isfinite(pol.τ_max) && pol.τ_max > zero(T) && (in.τ - B²) > pol.τ_max
    V2 = Main.HOT_PROJECTION[] == 2
    V2 && (τ_over = false)              # EOS-9 (Q6): only D_over short-circuits
    if D_over || τ_over
        base, cons, flags = pol_collapse(eos, pol, in, ye_in, B²)
        V2 && pol.collapse_to_atmosphere == false && (flags |= FLAG_POL_ρ_CEILED)
        Main.HOT_COUNT[4] += 1
        return Con2PrimSafeOut{T}(base, cons, flags, false)
    end

    # 2. The ordinary path.
    base = con2prim(eos, in, opts, s_guess, w_guess, u_guess)
    res = base.result
    it_n, it_f = base.iters_newton, base.iters_fallback
    u_warm = base.eos.u_solved

    if res === C2PResult.converged_newton || res === C2PResult.converged_fallback
        # 3. Cap the rapidity first, preserving D exactly, then project.
        w_cap = pol_w_cap(pol)
        ps = PrimState{T}(base.ρ, base.s, base.ye, base.w)
        pre = UInt32(0)
        if isfinite(base.w) && base.w > w_cap
            pre = FLAG_POL_W_CAPPED
            ps = PrimState{T}(in.D / cosh(w_cap), base.s, base.ye, w_cap)
        end
        ps, pf = project_prim_state(eos, ps, pol)
        pf |= pre
        # EOS-9 (Q4): a state replaced by the atmosphere reports that, not the discarded clamps.
        V2 && pf & FLAG_POL_ATMOSPHERE != 0 && (pf &= FLAG_POL_ATMOSPHERE | FLAG_POL_NONFINITE)

        if pf == 0
            # Already valid: hand back the input conservatives untouched, rather
            # than a round trip through prim2con that would perturb them.
            return Con2PrimSafeOut{T}(base, Prim2ConOut{T}(in.D, in.τ, in.D_Y, in.S_par, in.S_perp, in.B²),
                                      UInt32(0), true)
        end
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², u_warm)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end

    # 4. The solve failed: diagnose with the outer function's own endpoints.
    w_ref = isfinite(base.w) ? clamp(base.w, zero(T), opts.w_max) : zero(T)
    y_lo, y_hi = yₑ_bounds(eos)
    ye_c = clamp(ye_in, T(y_lo), T(y_hi))
    ρ_ref = in.D / cosh(w_ref)
    sr_ref = srange(eos, ρ_ref, ye_c)
    V2 && return eos9_step4(eos, pol, opts, in, ye_in, ye_c, B², res, it_n, it_f)

    r_lo, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_min, opts.w_max,
                            opts.τ_floor_rel, w_ref, u_warm, opts.max_iter_1d, opts.tol)
    Main.ENDPOINT_HOOK[] && Main.endpoint_record(eos, in, opts, ye_in, B², sr_ref, w_ref, u_warm, r_lo)
    if isfinite(r_lo.f2) && r_lo.f2 > zero(T)
        # τ sits below the coldest state compatible with this momentum: floor
        # the entropy onto the physical minimum.
        #
        # The clamps on w and ρ here leave a state that `project_prim_state`
        # finds already in range, so they have to report themselves. Raising ρ
        # is the one that matters to a caller: ρ·cosh w then exceeds D, and the
        # repair has added rest mass and momentum.
        pf = FLAG_POL_S_FLOORED
        w_cap = pol_w_cap(pol)
        w1 = zero(T)
        if isfinite(r_lo.w)
            w1 = clamp(r_lo.w, zero(T), w_cap)
            w1 == r_lo.w || (pf |= FLAG_POL_W_CAPPED)
        end
        ρ_lo = pol_ρ_atm(eos, pol)
        ρ_hi = pol_ρ_ceiling(eos, pol)
        ρ1 = in.D / cosh(w1)
        if ρ1 < ρ_lo
            ρ1 = ρ_lo
            pf |= FLAG_POL_ρ_FLOORED
        elseif ρ1 > ρ_hi
            ρ1 = ρ_hi
            pf |= FLAG_POL_ρ_CEILED
        end
        sr1 = srange(eos, ρ1, ye_c)
        ps, pf1 = project_prim_state(eos, PrimState{T}(ρ1, sr1.s_min, ye_in, w1), pol)
        pf |= pf1
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_lo.pt.u_solved)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end

    r_hi, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_max, opts.w_max,
                            opts.τ_floor_rel, r_lo.w, r_lo.pt.u_solved, opts.max_iter_1d, opts.tol)
    if isfinite(r_hi.f2) && r_hi.f2 < zero(T)
        if Main.HOT_PROJECTION[] == 1
            # 2026-10-22's first emulation ("EOS-9 v1"): the entropy ceiling of sr_ref, D kept,
            # the s_max rapidity capped, projected at the resulting density.
            pf = FLAG_POL_S_CEILED
            w_cap = pol_w_cap(pol)
            w1 = zero(T)
            if isfinite(r_hi.w)
                w1 = clamp(r_hi.w, zero(T), w_cap)
                w1 == r_hi.w || (pf |= FLAG_POL_W_CAPPED)
            end
            ps, pf1 = project_prim_state(eos, PrimState{T}(in.D / cosh(w1), sr_ref.s_max, ye_in, w1),
                                         pol)
            pf |= pf1
            Main.HOT_COUNT[1] += 1
            pf1 & FLAG_POL_ATMOSPHERE != 0 && (Main.HOT_COUNT[2] += 1)
            pf & FLAG_POL_W_CAPPED != 0 && (Main.HOT_COUNT[3] += 1)
            nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_hi.pt.u_solved)
            return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par,
                                                     nb.v_perp, nb.eos, res, it_n, it_f, nb.flags),
                                      cons, pf, true)
        end
        Main.HOT_COUNT[4] += 1
        base2, cons, flags = pol_collapse(eos, pol, in, ye_in, B²)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(base2.ρ, base2.s, base2.ye, base2.w, base2.W,
                                                 base2.v_par, base2.v_perp, base2.eos, res, it_n, it_f,
                                                 base2.flags), cons, flags, true)
    end

    # Neither diagnosis fits: excise.
    ps = policy_atmosphere(eos, pol, ye_in)
    nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_hi.pt.u_solved)
    return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp, nb.eos,
                                             res, it_n, it_f, nb.flags), cons, FLAG_POL_ATMOSPHERE, true)
end

# --- EOS-9 as rewritten 2026-10-22 (`HOT_PROJECTION = 2`) ------------------------------------

# The window's edge at a density: its floor for the cold branch, its ceiling for the hot one.
@inline function eos9_edge(eos, ρ, ye_c, hot::Bool)
    sr = srange(eos, ρ, ye_c)
    return hot ? sr.s_max : sr.s_min
end

# The edge solve: one root find of f₁(w) with s = s_edge(D/cosh w) inside the residual (Illinois
# on [0, w_max]). Returns the residual at the root, whether it is one (f₁ converged, or its bracket
# collapsed to the precision floor), and whether the root lies beyond w_max (capped legitimately).
function eos9_edge_root(eos, in::Con2PrimIn{T}, ye_in::T, ye_c::T, B²::T, hot::Bool,
                        opts::Con2PrimOptions{T}) where {T}
    g(w) = residuals(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B²,
                     eos9_edge(eos, in.D / cosh(w), ye_c, hot), w, T(NaN), opts.τ_floor_rel)
    a, c = zero(T), opts.w_max
    rc = g(c)
    isfinite(rc.f1) || return rc, false, false
    rc.f1 < zero(T) && return rc, true, true
    ra = g(a)
    isfinite(ra.f1) || return ra, false, false
    ra.f1 >= zero(T) && return ra, true, false
    fa, fc, side, r = ra.f1, rc.f1, 0, rc
    for _ in 1:100
        w = (a * fc - c * fa) / (fc - fa)
        r = g(w)
        isfinite(r.f1) || return r, false, false
        f1_converged(r, opts.tol) && return r, true, false
        if r.f1 < zero(T)
            a, fa = w, r.f1
            side == -1 && (fc /= 2)
            side = -1
        else
            c, fc = w, r.f1
            side == 1 && (fa /= 2)
            side = 1
        end
        c - a <= 4 * eps(c) && return r, true, false
    end
    return r, false, false
end

# Step 4: the cold, then the hot diagnosis, each by its edge solve; a branch only on a
# root (converged, at the precision floor, or beyond w_max) with a finite f₂; the projection at the
# root, the density clamped into [ρ_atm, ρ_ceiling] and reported; otherwise excised.
function eos9_step4(eos, pol::PolicyOptions{T}, opts, in::Con2PrimIn{T}, ye_in::T, ye_c::T, B²::T,
                    res, it_n, it_f) where {T}
    for hot in (false, true)
        r, ok, capped = eos9_edge_root(eos, in, ye_in, ye_c, B², hot, opts)
        (ok && isfinite(r.f2) && isfinite(r.w)) || continue
        (hot ? r.f2 < zero(T) : r.f2 > zero(T)) || continue
        pf = hot ? FLAG_POL_S_CEILED : FLAG_POL_S_FLOORED
        w_cap = pol_w_cap(pol)
        w1 = r.w
        if capped || w1 > w_cap
            w1 = min(w1, w_cap)
            pf |= FLAG_POL_W_CAPPED
        end
        ρ1 = in.D / cosh(w1)
        ρ_lo, ρ_hi = pol_ρ_atm(eos, pol), pol_ρ_ceiling(eos, pol)
        if ρ1 < ρ_lo
            ρ1 = ρ_lo
            pf |= FLAG_POL_ρ_FLOORED
        elseif ρ1 > ρ_hi
            ρ1 = ρ_hi
            pf |= FLAG_POL_ρ_CEILED
        end
        ps, pf1 = project_prim_state(eos, PrimState{T}(ρ1, eos9_edge(eos, ρ1, ye_c, hot), ye_in, w1),
                                     pol)
        pf |= pf1
        if hot
            Main.HOT_COUNT[1] += 1
            pf & FLAG_POL_ρ_FLOORED != 0 && (Main.HOT_COUNT[2] += 1)
            pf & FLAG_POL_W_CAPPED != 0 && (Main.HOT_COUNT[3] += 1)
        end
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r.pt.u_solved)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end
    ps = policy_atmosphere(eos, pol, ye_in)
    nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
    return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp, nb.eos,
                                             res, it_n, it_f, nb.flags), cons, FLAG_POL_ATMOSPHERE, true)
end
end
