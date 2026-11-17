# Step 12 (rung 6): the magnetized TOV star in the evolved spacetime, for `CODE.md`'s tables
# ("Validation ladder" 6). `magnetized_tov_case` through `tov_coupled.jl`'s problems and
# callbacks: TreeGH evolves the metric, and the matter's stress-energy, the fields' included,
# crosses back (`ADMMatter`). One run per invocation, sampled every `chunk`, with what
# `bin/magnetized_tov_study.jl` measures on the Cowling star — the field energy inside and
# outside `R̄`, `E·B` and `|E|` inside, the drift of `B` inside `R̄` and in the shell
# `R̄ + 1 ≤ r̄ < L − 1`, the charge, the entropy fallback's trigger and the policy's repairs,
# `max |∇·𝔅| h/max |𝔅|`, the rest mass against the reset's injection, the stage solve's
# outcome map in four radial regions — and TreeGH's matter-corrected constraints inside
# (`r < 6`) and outside (`r ≥ 10`) the star, as `tov_coupled_study.jl` reports them.
#
#     julia --project=coupled -t T coupled/magnetized_tov_study.jl [key=value...]
#
# Keys: `N` (cells per block, 12), `levels` (box levels around the star, 1), `L` (24),
# `roots` (4), `t_end` (50), `chunk` (2), `limiter` (`ppm`), `cfl_gh` (1/2), `γ0M` (1/10),
# `γ2` (0), `order` (the matter's transfer to the vertices, 4), and the case's: `sigma0` (σ₀,
# 1e6), `mumax` (μ_max, 10), `cutoff` (in units of ρ_min, 10⁴), `power` (1; 2 the
# literature's squared profile), `field` (`dipole` or `confined`), `Pcut` (0.04), `Bc`
# (1e-4), `rhomin` (ρ_min, 1e-10). Lines start with `RUN`, `CHUNK` or `MAP`, for grepping.

include(joinpath(@__DIR__, "tov_coupled_study.jl"))  # problems, callbacks, study_norms
using Printf

field_scale(f::MHD.StellarDipole) = f.μ
field_scale(f::MHD.ConfinedField) = f.A_b

const BIT_NAMES = ("σ=0", "ideal", "fixed-point", "Newton", "fallback", "ideal-seed", "switched",
                   "backtracked", "repaired")

function magnetized_study(T=Float64; N=12, levels=1, L=24, roots=4, t_end=50, chunk=2,
                          limiter=:ppm, cfl_gh=T(1 // 2), γ0M=1 // 10, γ2=0, order=4, σ₀=1e6,
                          μ_max=10, cutoff=10_000, power=1, field=:dipole, P_cut=0.04,
                          B_c=1e-4, ρ_min=1e-10)
    case = MHD.magnetized_tov_case(T; σ₀, μ_max, cutoff, power, field, P_cut, B_c, L, roots,
                                   ρ_min)
    (; p, u, forest) = problems(T; N, levels, limiter, γ0M, γ2, order, case)
    R̄ = case.fields.w.star.metric.R̄
    region(x) = (r = sqrt(sum(abs2, x)); r < 4R̄ / 5 ? 1 : r < R̄ ? 2 : r < R̄ + 3 // 2 ? 3 : 4)
    profile(p, u, t) = MHD.cell_profile(p.mhd, view(u, p.s[2]), t; metric=metric_at(p, u, t))
    prof0 = profile(p, u, zero(T))
    M0 = MHD.conserved_totals(p.mhd, view(u, p.s[2]))[1]
    fd0 = MHD.field_diagnostics(prof0; radius=R̄)
    n0 = study_norms(p, u, zero(T))
    Printf.format(stdout, Printf.Format("RUN coupled N=%d levels=%d L=%g roots=%d h=%.5f " *
            "cells=%d σ₀=%g μ_max=%g cutoff=%g power=%d field=%s B_c=%g ρ_min=%g limiter=%s " *
            "order=%d γ0M=%g R̄=%.6f scale=%.6e E_in=%.6e E_out=%.6e divB=%.3e threads=%d " *
            "B_max=%.10e B_center=%.10e ham_in=%.6e mom_in=%.6e ham_out=%.6e mom_out=%.6e " *
            "M0=%.12e\n"),
            N, levels, L, roots, minimum(prof0.h), length(prof0.h), σ₀, μ_max, cutoff, power,
            field, B_c, ρ_min, limiter, order, γ0M, R̄, field_scale(case.fields.w.field),
            fd0.energy_in, fd0.energy_out, MHD.max_div_B(p.mhd, view(u, p.s[2]), zero(T)),
            Threads.nthreads(), fd0.B_max, fd0.B_center, n0.ham_in, n0.mom_in, n0.ham_out,
            n0.mom_out, M0)
    flush(stdout)
    cfl_mhd = T(MHD.default_cfl(limiter))
    t_end, chunk = T(t_end), T(chunk)
    nchunks = MHD.chunk_count(t_end, chunk)
    t, integ, steps, injected = zero(T), nothing, 0, zero(T)
    wall0 = time()
    for c in 1:nchunks
        t1 = c == nchunks ? t_end : c * chunk
        MHD.clear_tally!(p.mhd)
        nsteps = step_count(p, u, t, t1, forest, cfl_gh, cfl_mhd)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=(t1 - t) / nsteps,
                         stage_limiter=stage_limit!, step_limiter=step_limit!,
                         partition=:even, alias_u0=true, reuse=integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
        steps += nsteps
        t = t1
        injected += MHD.ohm_tally(p.mhd).injected_D
        prof = profile(p, u, t)
        n = study_norms(p, u, t)                     # forgets the fill
        fd = MHD.field_diagnostics(prof; radius=R̄)
        dr = MHD.field_drift(prof, prof0; radius=R̄)
        sh = MHD.field_drift(prof, prof0; radius=R̄ + 1, outer=L - 1)
        ΔM = MHD.conserved_totals(p.mhd, view(u, p.s[2]))[1] - M0
        Printf.format(stdout, Printf.Format("CHUNK t=%.3f wall=%.1f steps=%d ρc=%.10e " *
                "E_in=%.6e E_out=%.6e EB_in=%.4e EB_max_in=%.4e |E|/|B|_in=%.4e " *
                "drift_in=%.4e drift_shell=%.4e q_in=%.4e q_out=%.4e q_max=%.4e trigger=%d/%d " *
                "repaired=%d/%d divB=%.3e ΔM=%.6e injected=%.6e budget=%.3e B_max=%.10e " *
                "B_center=%.10e ham_in=%.6e mom_in=%.6e ham_out=%.6e mom_out=%.6e\n"),
                t, time() - wall0, steps, n.ρ_c, fd.energy_in, fd.energy_out, fd.EB_in,
                fd.EB_max_in, fd.E_in, dr.inside, sh.outside, fd.q_in, fd.q_out, fd.q_max,
                fd.trigger_in, fd.trigger_out, fd.repaired_in, fd.repaired_out,
                MHD.max_div_B(p.mhd, view(u, p.s[2]), t), ΔM, injected, ΔM - injected,
                fd.B_max, fd.B_center, n.ham_in, n.mom_in, n.ham_out, n.mom_out)
        for (k, e) in enumerate(MHD.ohm_map(p.mhd, region, 4))
            stages = e.bits[1] + e.bits[2] + e.bits[3] + e.bits[4] + e.bits[5]
            Printf.format(stdout, Printf.Format("MAP t=%.3f region=%d cells=%d stages=%d %s " *
                    "policy=%d resets=%d/%d recoveries=%d\n"), t, k, e.cells, stages,
                    join(("$(BIT_NAMES[j])=$(e.bits[j])" for j in 1:9), " "), e.policy,
                    e.resets, e.step_resets, e.recoveries)
        end
        flush(stdout)
    end
    @printf("RUN done t=%.3f steps=%d wall=%.1f\n", t, steps, time() - wall0)
    return (; p, u, t, steps)
end

function main_study(args)
    kw = Dict(split(a, '='; limit=2) for a in args)
    known = ("N", "levels", "L", "roots", "t_end", "chunk", "limiter", "cfl_gh", "γ0M", "γ2",
             "order", "sigma0", "mumax", "cutoff", "power", "field", "Pcut", "Bc", "rhomin")
    for k in keys(kw)
        k in known || error("magnetized_tov_study.jl: unknown key `$k` ($(join(known, ", ")))")
    end
    num(k, d) = parse(Float64, get(kw, k, string(d)))
    int(k, d) = parse(Int, get(kw, k, string(d)))
    magnetized_study(; N=int("N", 12), levels=int("levels", 1), L=num("L", 24),
                     roots=int("roots", 4), t_end=num("t_end", 50), chunk=num("chunk", 2),
                     limiter=Symbol(get(kw, "limiter", "ppm")), cfl_gh=num("cfl_gh", 0.5),
                     γ0M=num("γ0M", 0.1), γ2=num("γ2", 0), order=int("order", 4),
                     σ₀=num("sigma0", 1e6), μ_max=num("mumax", 10),
                     cutoff=num("cutoff", 10_000), power=int("power", 1),
                     field=Symbol(get(kw, "field", "dipole")), P_cut=num("Pcut", 0.04),
                     B_c=num("Bc", 1e-4), ρ_min=num("rhomin", 1e-10))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_study(ARGS)
end
