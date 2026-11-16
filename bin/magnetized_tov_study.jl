# Step 12 (rung 6): the magnetized TOV star's studies, for `CODE.md`'s tables ("Validation
# ladder" 6): one run of `magnetized_tov_case` per invocation through `evolve!`, sampled every
# `chunk`, with
#
#   * per chunk: the central density; the field energy inside and outside `R̄`; `E·B` and `|E|`
#     inside against `B` (`field_diagnostics`); the drift of `B` from the initial field inside
#     `R̄` and in the shell `R̄ + 1 ≤ r̄ < L − 1` (`field_drift`); the charge; the cells whose
#     fields' energy exceeds the fluid's (the entropy fallback's trigger) and the cells the
#     policy repaired, the atmosphere left out; `max |∇·𝔅| h/max |𝔅|`; the rest mass against
#     what the reset injected;
#   * per chunk, the stage solve's outcome map (`ohm_map`) in four radial regions: the core
#     `r̄ < 0.8 R̄`, the envelope `0.8 R̄ ≤ r̄ < R̄`, the surface band `R̄ ≤ r̄ < R̄ + 1.5` and the
#     exterior.
#
#     julia --project=. -t T bin/magnetized_tov_study.jl [key=value...]
#
# Keys: `N` (cells per block, 8), `levels` (box levels around the star, `tov_boxes!`, 1), `L`
# (24), `roots` (4), `t_end` (10), `chunk` (1), `limiter` (`ppm`), `sigma0` (σ₀, 1e6),
# `mumax` (μ_max, 10), `cutoff` (`DensityProfile`'s, in units of ρ_min, the case's 10⁴),
# `power` (`DensityProfile`'s, 1; 2 the literature's squared profile), `field` (`dipole` or
# `confined`, the interior-only field of Duez et al. 2006), `Pcut` (the confined field's edge,
# a fraction of the central pressure, 0.04), `Bc` (the central field, 1e-4), `rhomin` (ρ_min,
# 1e-10). The `CHUNK` lines end with `B_max` and `B_center`, the mean `|B|` within `r̄ < 1`. The lines
# start with a tag (`CHUNK`, `MAP`, `RUN`) for grepping.

using TreeAMR
using TreeGRRMHD
using Printf
const MHD = TreeGRRMHD

field_scale(f::MHD.StellarDipole) = f.μ
field_scale(f::MHD.ConfinedField) = f.A_b

const BIT_NAMES = ("σ=0", "ideal", "fixed-point", "Newton", "fallback", "ideal-seed", "switched",
                   "backtracked", "repaired")

function study(T=Float64; N=8, levels=1, L=24, roots=4, t_end=10, chunk=1, limiter=:ppm,
               σ₀=1e6, μ_max=10, cutoff=10_000, power=1, field=:dipole, P_cut=0.04, B_c=1e-4,
               ρ_min=1e-10)
    case = magnetized_tov_case(T; σ₀, μ_max, cutoff, power, field, P_cut, B_c, L, roots, ρ_min)
    R̄ = case.fields.w.star.metric.R̄
    region(x) = (r = sqrt(sum(abs2, x)); r < 4R̄ / 5 ? 1 : r < R̄ ? 2 : r < R̄ + 3 // 2 ? 3 : 4)
    prof0 = Ref{Any}(nothing)
    next = Ref(T(chunk))
    injected = Ref(zero(T))
    M0 = Ref(zero(T))
    t0 = time()
    function sample(p, u, t)
        prof = cell_profile(p, u, t)
        prof0[] === nothing && error("no initial profile")
        fd = field_diagnostics(prof; radius=R̄)
        dr = field_drift(prof, prof0[]; radius=R̄)
        sh = field_drift(prof, prof0[]; radius=R̄ + 1, outer=L - 1)
        ρc = maximum(prof.ρ)
        ΔM = conserved_totals(p, u)[1] - M0[]
        injected[] += MHD.ohm_tally(p).injected_D
        Printf.format(stdout, Printf.Format("CHUNK t=%.3f wall=%.1f ρc=%.10e E_in=%.6e E_out=%.6e EB_in=%.4e EB_max_in=%.4e " *
                "|E|/|B|_in=%.4e drift_in=%.4e drift_shell=%.4e q_in=%.4e q_out=%.4e " *
                "q_max=%.4e trigger=%d/%d repaired=%d/%d divB=%.3e ΔM=%.6e injected=%.6e " *
                "budget=%.3e B_max=%.10e B_center=%.10e\n"),
                t, time() - t0, ρc, fd.energy_in, fd.energy_out, fd.EB_in, fd.EB_max_in, fd.E_in,
                dr.inside, sh.outside, fd.q_in, fd.q_out, fd.q_max, fd.trigger_in,
                fd.trigger_out, fd.repaired_in, fd.repaired_out, max_div_B(p, u, t), ΔM,
                injected[], ΔM - injected[], fd.B_max, fd.B_center)
        for (k, e) in enumerate(ohm_map(p, region, 4))
            stages = e.bits[1] + e.bits[2] + e.bits[3] + e.bits[4] + e.bits[5]
            Printf.format(stdout, Printf.Format("MAP t=%.3f region=%d cells=%d stages=%d %s policy=%d resets=%d/%d " *
                    "recoveries=%d\n"), t, k, e.cells, stages,
                    join(("$(BIT_NAMES[j])=$(e.bits[j])" for j in 1:9), " "), e.policy,
                    e.resets, e.step_resets, e.recoveries)
        end
        flush(stdout)
        return nothing
    end
    after_step = function (p, u, t)
        # `evolve!` clears the tally at each chunk's start, so the map is sampled at its end.
        t ≥ next[] - T(1 // 10^9) || return nothing
        next[] += T(chunk)
        sample(p, u, t)
        return nothing
    end
    refine = levels > 0 ? (f -> tov_boxes!(f, case; levels)) : nothing
    # The initial profile, on the run's own mesh: a zero-length run builds it.
    r0 = evolve!(case, Val(3); N, roots, t_end=T(chunk) / 10^6, limiter, refine)
    prof0[] = cell_profile(r0.p, exact_state(r0.p, case, zero(T)), zero(T))
    M0[] = r0.totals0[1]
    fd0 = field_diagnostics(prof0[]; radius=R̄)
    Printf.format(stdout, Printf.Format("RUN N=%d levels=%d L=%g roots=%d h=%.5f cells=%d σ₀=%g μ_max=%g cutoff=%g power=%d field=%s B_c=%g ρ_min=%g " *
            "limiter=%s R̄=%.6f scale=%.6e E_in=%.6e E_out=%.6e trigger=%d/%d repaired=%d/%d " *
            "divB=%.3e threads=%d B_max=%.10e B_center=%.10e\n"),
            N, levels, L, roots, minimum(prof0[].h), length(prof0[].h), σ₀, μ_max, cutoff,
            power, field, B_c, ρ_min,
            limiter, R̄, field_scale(case.fields.w.field), fd0.energy_in, fd0.energy_out, fd0.trigger_in,
            fd0.trigger_out, fd0.repaired_in, fd0.repaired_out,
            max_div_B(r0.p, exact_state(r0.p, case, zero(T)), zero(T)), Threads.nthreads(),
            fd0.B_max, fd0.B_center)
    flush(stdout)
    r = evolve!(case, Val(3); N, roots, t_end=T(t_end), chunk=T(chunk), limiter, refine,
                after_step)
    @printf("RUN done t=%.3f steps=%d wall=%.1f\n", r.t, r.nsteps, time() - t0)
    return r
end

function main(args)
    kw = Dict(split(a, '='; limit=2) for a in args)
    known = ("N", "levels", "L", "roots", "t_end", "chunk", "limiter", "sigma0", "mumax", "Bc",
             "rhomin", "cutoff", "power", "field", "Pcut")
    for k in keys(kw)
        k in known || error("magnetized_tov_study.jl: unknown key `$k` ($(join(known, ", ")))")
    end
    num(k, d) = parse(Float64, get(kw, k, string(d)))
    int(k, d) = parse(Int, get(kw, k, string(d)))
    study(; N=int("N", 8), levels=int("levels", 1), L=num("L", 24), roots=int("roots", 4),
          t_end=num("t_end", 10), chunk=num("chunk", 1),
          limiter=Symbol(get(kw, "limiter", "ppm")), σ₀=num("sigma0", 1e6),
          μ_max=num("mumax", 10), cutoff=num("cutoff", 10_000), power=int("power", 1),
          field=Symbol(get(kw, "field", "dipole")), P_cut=num("Pcut", 0.04),
          B_c=num("Bc", 1e-4), ρ_min=num("rhomin", 1e-10))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
