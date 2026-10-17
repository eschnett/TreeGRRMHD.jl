# Experiment (branch `claude/front-investigation`), NOT part of the package: emulates the
# proposed EntropyEOS policy change EOS-5 (PLAN.md, "Upstream requirements") by redefining
# `EntropyEOS.con2prim_safe` in the running Julia process — the sibling checkout is not touched,
# and nothing in `src/` loads this file. Used by `bin/front_investigation.jl`, and for the suite as
#
#     julia --project=. -e 'using Pkg; Pkg.test(; julia_args=["-L", "bin/eos5_emulation.jl"])'
#
# EOS-5, "no repair creates rest mass", changes two branches of `con2prim_safe`'s step 4 (the
# solve failed and the endpoints diagnose the state) and nothing else:
#   (i)  τ below the coldest state compatible with the momentum: where the density at the
#        recovered rapidity, D/cosh w, is below ρ_atm, the state is the atmosphere — the policy's
#        own rule for ρ < ρ_atm (`pol_project_core`) — rather than ρ clamped *up* to ρ_atm at that
#        rapidity, which multiplies the rest mass by up to W_cap = 100;
#   (ii) τ above the hottest state: the state at s_max with the momentum's rapidity and D kept
#        (FLAG_POL_S_CEILED, what the converged path's projection gives a state just inside the
#        bracket), rather than the collapse ceiling ρ = ρ_ceiling (10⁶ for the shock gas). A
#        collapse is still diagnosed by step 1's arithmetic tests (D > D_max or τ > τ_max).
# `EOS5_ON[] = false` restores the released behaviour bit for bit (checked).
import TreeGRRMHD
isdefined(Main, :EOS5_ON) || @eval Main const EOS5_ON = Ref(true)
isdefined(Main, :EOS5_HOT) || @eval Main const EOS5_HOT = Ref(true)
@eval TreeGRRMHD.EntropyEOS begin
function con2prim_safe(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                       pol::PolicyOptions{T}, s_guess::T=T(NaN), w_guess::T=T(NaN),
                       u_guess::T=T(NaN)) where {S,T}
    B² = (isfinite(in.B²) && in.B² > zero(T)) ? in.B² : zero(T)
    fin = isfinite(in.D) && isfinite(in.τ) && isfinite(in.D_Y) && isfinite(in.S_par) &&
          isfinite(in.S_perp) && isfinite(in.B²)
    if !(fin && in.D > zero(T) && in.B² >= zero(T))
        ye_in = (isfinite(in.D) && in.D > zero(T) && isfinite(in.D_Y)) ? in.D_Y / in.D : T(NaN)
        ps = policy_atmosphere(eos, pol, ye_in)
        base, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
        return Con2PrimSafeOut{T}(base, cons, FLAG_POL_NONFINITE | FLAG_POL_ATMOSPHERE, false)
    end
    ye_in = in.D_Y / in.D
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
    base = con2prim(eos, in, opts, s_guess, w_guess, u_guess)
    res = base.result
    it_n, it_f = base.iters_newton, base.iters_fallback
    u_warm = base.eos.u_solved
    if res === C2PResult.converged_newton || res === C2PResult.converged_fallback
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
            return Con2PrimSafeOut{T}(base, Prim2ConOut{T}(in.D, in.τ, in.D_Y, in.S_par, in.S_perp, in.B²),
                                      UInt32(0), true)
        end
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², u_warm)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end
    w_ref = isfinite(base.w) ? clamp(base.w, zero(T), opts.w_max) : zero(T)
    y_lo, y_hi = yₑ_bounds(eos)
    ye_c = clamp(ye_in, T(y_lo), T(y_hi))
    ρ_ref = in.D / cosh(w_ref)
    sr_ref = srange(eos, ρ_ref, ye_c)
    r_lo, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_min, opts.w_max,
                            opts.τ_floor_rel, w_ref, u_warm, opts.max_iter_1d, opts.tol)
    if isfinite(r_lo.f2) && r_lo.f2 > zero(T)
        w_cap = pol_w_cap(pol)
        w1 = isfinite(r_lo.w) ? clamp(r_lo.w, zero(T), w_cap) : zero(T)
        if Main.EOS5_ON[] && in.D / cosh(w1) < pol_ρ_atm(eos, pol)
            # (i): below the atmosphere at its rapidity — the atmosphere, D not raised.
            ps = policy_atmosphere(eos, pol, ye_in)
            nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², T(NaN))
            return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                     nb.eos, res, it_n, it_f, nb.flags), cons,
                                      FLAG_POL_ATMOSPHERE | FLAG_POL_S_FLOORED, true)
        end
        ρ1 = clamp(in.D / cosh(w1), pol_ρ_atm(eos, pol), pol_ρ_ceiling(eos, pol))
        sr1 = srange(eos, ρ1, ye_c)
        ps, pf = project_prim_state(eos, PrimState{T}(ρ1, sr1.s_min, ye_in, w1), pol)
        pf |= FLAG_POL_S_FLOORED
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_lo.pt.u_solved)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end
    r_hi, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_max, opts.w_max,
                            opts.τ_floor_rel, r_lo.w, r_lo.pt.u_solved, opts.max_iter_1d, opts.tol)
    if isfinite(r_hi.f2) && r_hi.f2 < zero(T)
        if Main.EOS5_ON[] && Main.EOS5_HOT[]
            # (ii): too hot for the window — the entropy ceiling at the momentum's rapidity.
            w_cap = pol_w_cap(pol)
            w1 = isfinite(r_hi.w) ? clamp(r_hi.w, zero(T), w_cap) : zero(T)
            ps, pf = project_prim_state(eos, PrimState{T}(in.D / cosh(w1), sr_ref.s_max, ye_in, w1), pol)
            pf |= FLAG_POL_S_CEILED
            nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_hi.pt.u_solved)
            return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                     nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
        end
        base2, cons, flags = pol_collapse(eos, pol, in, ye_in, B²)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(base2.ρ, base2.s, base2.ye, base2.w, base2.W,
                                                 base2.v_par, base2.v_perp, base2.eos, res, it_n, it_f,
                                                 base2.flags), cons, flags, true)
    end
    ps = policy_atmosphere(eos, pol, ye_in)
    nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_hi.pt.u_solved)
    return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp, nb.eos,
                                             res, it_n, it_f, nb.flags), cons, FLAG_POL_ATMOSPHERE, true)
end
end
