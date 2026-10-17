# Experiment (branch `claude/front-investigation`), NOT part of the package: variants of
# EntropyEOS's atmosphere policy, emulated by redefining `EntropyEOS.default_policy` and
# `EntropyEOS.con2prim_safe` in the running Julia process. The sibling checkout is not touched, and
# nothing in `src/` loads this file. Every recovery the scheme makes goes through
# `con2prim_safe` — the right-hand side's, the stage solve's (its predictor and every iterate) and
# the reset's — so a variant here reaches all of them. Used by `bin/front_investigation.jl`, and for
# the suite as
#
#     FRONT_VARIANT=A3+C julia --project=. -e 'using Pkg;
#         Pkg.test(; julia_args=["-L", "<abs path>/bin/policy_variants.jl"])'
#
# A variant is a `+`-joined list of (`POLICY_VARIANT[]`, or the environment's `FRONT_VARIANT`):
#   `A<k>`  `atm_trigger = k` in `default_policy` (an existing option: the reset to the
#           atmosphere fires at `D < k ρ_atm`; the released default is 1.01);
#   `cold`  `s_atm = s_min` in `default_policy` (an existing option; the released default is NaN,
#           which `policy_atmosphere` takes as the middle of the physical entropy window — 20.5 of
#           the shock gas's 1 … 40, not cold);
#   `B`     the `S_FLOORED` branch of step 4 (τ below the coldest state at the momentum), where the
#           density at the recovered rapidity, `D/cosh w₁`, is below `ρ_atm`: the atmosphere at rest
#           (`policy_atmosphere`, flags ATMOSPHERE | S_FLOORED), where the released policy raises
#           `ρ` to `ρ_atm` and keeps `w₁`, writing `W ≈ |S|/ρ_atm`;
#   `C`     the entropy reset in the atmosphere: a state whose density, after the policy's own
#           projection, lies in the atmosphere band `ρ < atm_trigger · ρ_atm` takes the
#           atmosphere's entropy (`policy_atmosphere(...).s`), with `ρ`, `w`, `Yₑ` kept (flag
#           S_CEILED or S_FLOORED by the direction); implies `cold` (C is the cold atmosphere);
#   `Cmid`  C with the released `s_atm` (the window's middle);
#   `coll`  `collapse_to_atmosphere = true` in `default_policy` (an existing option);
#   `hot`   step 4's "τ above the hottest state": the state at `s_max` with `D` kept and the
#           rapidity of the `s_max` solve (S_CEILED), where the released policy writes the collapse
#           ceiling `ρ_ceiling`;
#   `base`  nothing (the released policy, bit for bit: checked).
import TreeGRRMHD

mutable struct PolicyVariant
    trigger::Float64     # 0: the released 1.01
    cold::Bool
    B::Bool
    C::Bool
    hot::Bool
    coll::Bool
end
function PolicyVariant(name::AbstractString)
    v = PolicyVariant(0.0, false, false, false, false, false)
    for tok in split(name, '+')
        tok == "base" ? nothing :
        startswith(tok, "A") ? (v.trigger = parse(Float64, tok[2:end])) :
        tok == "cold" ? (v.cold = true) :
        tok == "B" ? (v.B = true) :
        tok == "C" ? (v.C = true; v.cold = true) :
        tok == "Cmid" ? (v.C = true) :
        tok == "hot" ? (v.hot = true) :
        tok == "coll" ? (v.coll = true) :
        tok == "Breset" ? nothing :                   # the reset-side B: bin/front_investigation.jl
        error("unknown variant token $tok")
    end
    return v
end
isdefined(Main, :POLICY_VARIANT) ||
    @eval Main const POLICY_VARIANT = Ref(PolicyVariant(get(ENV, "FRONT_VARIANT", "base")))

@eval TreeGRRMHD.EntropyEOS begin
function default_policy(eos::AbstractEOS{S}, ρ_atm::T) where {S,T}
    v = Main.POLICY_VARIANT[]
    y_lo, y_hi = yₑ_bounds(eos)
    s_atm = v.cold ? T(srange(eos, ρ_atm, T(0.5) * (T(y_lo) + T(y_hi))).s_min) : T(NaN)
    trigger = v.trigger > 0 ? T(v.trigger) : T(1.01)
    pol = PolicyOptions{T}(; ρ_atm=ρ_atm, s_atm=s_atm, ye_atm=T(NaN), atm_trigger=trigger,
                           ρ_ceiling=exp10(T(last(logρ_bounds(eos)))), w_cap=acosh(T(100)),
                           collapse_to_atmosphere=v.coll)
    return policy_derive_bounds(eos, pol)
end

# C: the atmosphere's entropy for a state in the atmosphere band.
@inline function variant_entropy(eos, pol::PolicyOptions{T}, ps::PrimState{T}, pf::UInt32) where {T}
    Main.POLICY_VARIANT[].C || return ps, pf
    ps.ρ < pol_atm_trigger(pol) * pol_ρ_atm(eos, pol) || return ps, pf
    s_a = policy_atmosphere(eos, pol, ps.ye).s
    ps.s == s_a && return ps, pf
    return PrimState{T}(ps.ρ, s_a, ps.ye, ps.w), pf | (ps.s > s_a ? FLAG_POL_S_CEILED : FLAG_POL_S_FLOORED)
end

function con2prim_safe(eos::AbstractEOS{S}, in::Con2PrimIn{T}, opts::Con2PrimOptions{T},
                       pol::PolicyOptions{T}, s_guess::T=T(NaN), w_guess::T=T(NaN),
                       u_guess::T=T(NaN)) where {S,T}
    V = Main.POLICY_VARIANT[]
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
        pf & FLAG_POL_ATMOSPHERE == 0 && ((ps, pf) = variant_entropy(eos, pol, ps, pf))
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
        if V.B && in.D / cosh(w1) < pol_ρ_atm(eos, pol)
            # B: the density is raised to the atmosphere's, so the velocity is the atmosphere's.
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
        pf & FLAG_POL_ATMOSPHERE == 0 && ((ps, pf) = variant_entropy(eos, pol, ps, pf))
        nb, cons = pol_package(eos, ps, in.S_par, in.S_perp, B², r_lo.pt.u_solved)
        return Con2PrimSafeOut{T}(Con2PrimOut{T}(nb.ρ, nb.s, nb.ye, nb.w, nb.W, nb.v_par, nb.v_perp,
                                                 nb.eos, res, it_n, it_f, nb.flags), cons, pf, true)
    end
    r_hi, _ = inner_solve_w(eos, in.D, in.τ, ye_in, in.S_par, in.S_perp, B², sr_ref.s_max, opts.w_max,
                            opts.τ_floor_rel, r_lo.w, r_lo.pt.u_solved, opts.max_iter_1d, opts.tol)
    if isfinite(r_hi.f2) && r_hi.f2 < zero(T)
        if V.hot
            w_cap = pol_w_cap(pol)
            w1 = isfinite(r_hi.w) ? clamp(r_hi.w, zero(T), w_cap) : zero(T)
            ps, pf = project_prim_state(eos, PrimState{T}(in.D / cosh(w1), sr_ref.s_max, ye_in, w1), pol)
            pf |= FLAG_POL_S_CEILED
            pf & FLAG_POL_ATMOSPHERE == 0 && ((ps, pf) = variant_entropy(eos, pol, ps, pf))
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
