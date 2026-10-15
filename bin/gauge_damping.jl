# The gauge damping `ξ` against the spurious `B` of a refinement boundary
# (step 8; "Refinement" in `CODE.md`), which decides the (open) `ξ`.
#
#     julia --project=. bin/gauge_damping.jl [w=4]
#
# 1. The pure gauge pulse (`gauge_pulse_case`: `A = 0`, `Φ̃` a Gaussian of
#    width 1/10, no field, gas at rest, `Vacuum`) on the periodic unit square,
#    2 × 2 roots of N = 16 with one root refined, to `t = 2` (two light
#    crossings): `max |𝔅|` over the run and at its end, and `max |Φ̃|` at its
#    end, for `ξ ∈ {0, 1, 4, 16, 64}`; the uniform mesh is the control, where
#    `𝔅` stays zero to rounding.
# 2. The Alfvén wave along the diagonal (`oblique_case`, an eighth of a
#    period, 2 × 2 roots of N = 16, one refined): the L1 errors of `S_x` and
#    `A_z` for the same `ξ` — what the damping does to a physical solution.

using Printf
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD

refine_first!(forest) = (refine!(forest, [first(forest.leaves)]); balance!(forest); forest)

w = let a = findfirst(s -> startswith(s, "w="), ARGS)
    a === nothing ? 4 : parse(Int, ARGS[a][3:end])
end
ops_A = Operators(; prolongation=w, restriction=w)
println("potentials' width w = $w")

maxB(p) = maximum(d -> mesh_mapreduce(abs, max, 0.0, p.f.B[d]; vars=1), 1:3)

println("1. gauge pulse, t = 2: max|𝔅| over the run, at the end; max|Φ̃| at the end")
gp = gauge_pulse_case(Float64; roots=2)
for refine in (nothing, refine_first!), ξ in (0, 1, 4, 16, 64)
    Bs = Float64[]
    mon(p, u, t) = (update_fields!(p, u, t); push!(Bs, maxB(p)))
    t0 = time()
    r = evolve!(gp, Val(2); N=16, t_end=2, limiter=:mc, ξ=ξ, refine=refine, ops_A=ops_A,
                after_step=mon)
    Φmax = mesh_mapreduce(abs, max, 0.0, r.p.sets[TG.SEG_Φ], stateview(r.u, r.p.sets, TG.SEG_Φ);
                          vars=1)
    @printf("  %-10s ξ = %-3d max|𝔅| %.3e, at t = 2 %.3e; max|Φ̃| %.3e  (%d steps, %.0f s)\n",
            refine === nothing ? "uniform" : "two-level", ξ, maximum(Bs), Bs[end], Φmax,
            r.nsteps, time() - t0)
end

println("2. oblique Alfvén wave, an eighth of a period, two-level: L1 of S_x, A_z, max|Φ̃|")
c1 = alfven_wave_case(Float64; roots=1)
c = oblique_case(c1; N=16, roots=2)
for ξ in (0, 1, 4, 16, 64)
    r = evolve!(c, Val(2); N=16, t_end=alfven_period(c1) / 8, limiter=:mc, ξ=ξ,
                refine=refine_first!, ops_A=ops_A)
    e = l1_errors(r.p, r.u, exact_state(r.p, c, r.t))
    Φmax = mesh_mapreduce(abs, max, 0.0, r.p.sets[TG.SEG_Φ], stateview(r.u, r.p.sets, TG.SEG_Φ);
                          vars=1)
    @printf("  ξ = %-3d L1 S_x %.4e, A_z %.4e; max|Φ̃| %.3e\n", ξ, e.U[TG.U_S], e.A[3], Φmax)
end
