# Experiment (branch `claude/no-collapse-to-atmosphere`), NOT part of the package: the
# entropy-ceiling projection of `con2prim_safe`'s "τ above the hottest state" branch (the
# proposed EOS-9, PLAN.md), emulated by redefining `EntropyEOS.con2prim_safe` in the running
# process. The function is EntropyEOS 1.1.0's own (origin/main 43923c7,
# `src/core/state_policy.jl`), copied verbatim but for that branch; the sibling checkout is not
# touched and nothing in `src/` loads this file. `HOT_PROJECTION[] = false` gives 1.1.0's own
# behaviour, bit for bit (the branch calls `pol_collapse`). `HOT_COUNT` counts, over every
# recovery the process makes (the reset's, the right-hand side's, the stage solve's): the
# projections, those whose density at the s_max rapidity fell below `ρ_atm` (so that the
# projection wrote the atmosphere), those whose rapidity was capped, and, with the projection
# off, the calls of `pol_collapse` from that branch. For the suite:
#
#     julia --project=. -e 'using Pkg; Pkg.test(; julia_args=["-L", "<abs path>/bin/hot_projection.jl"])'
import TreeGRRMHD
isdefined(Main, :HOT_PROJECTION) ||
    @eval Main const HOT_PROJECTION = Ref(get(ENV, "HOT_PROJECTION", "1") == "1")
isdefined(Main, :HOT_COUNT) || @eval Main const HOT_COUNT = zeros(Int, 4)
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
    if D_over || τ_over
        base, cons, flags = pol_collapse(eos, pol, in, ye_in, B²)
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

    r_lo, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_min, opts.w_max,
                            opts.τ_floor_rel, w_ref, u_warm, opts.max_iter_1d, opts.tol)
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
        if Main.HOT_PROJECTION[]
            # EXPERIMENT (the proposed EOS-9): τ above the hottest state compatible with this
            # momentum is not a collapse: project the entropy onto the physical maximum, with D
            # kept and the rapidity of the s_max momentum solve (capped as the converged path
            # caps it, D kept), in MHD mode with the same S_par, S_perp and B² as every branch.
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
end
