# The matter's ghost fill across a coarse-fine face (step 8; "Refinement" in
# `CODE.md`): the atmosphere tube — a gas at `ρ = p = 1` expanding into a
# `10⁻⁸` atmosphere, 200 cells, `[0, 0.2)` refined, which the front crosses —
# hydrodynamic, and magnetized (`Bʸ = 10⁻³`, `MagnetizationCutoff(Constant(10³),
# 10)`). Printed per configuration: the interior ghosts (not beyond the
# domain's faces) whose conserved state the recovery repairs for anything but
# the atmosphere, summed over every step, with the `|` of their flags; and
# the rest mass and energy the reset injected (the step limiter's writes).
#
#     julia --project=. bin/ghost_fill.jl
#
# Since step 8 (AMR-10) the fill prolongates the primitives `W` (PointValue,
# bounded) and rebuilds each conserved ghost by `prim2con` with the ghost's
# own `B`; the configurations are the uniform mesh and the primitives at
# widths 2, 4, 6, and at width 4 unbounded. *(Amended 2026-10-23:)* the
# package's fill is now the parent-centred one (`default_ops_W`, Conservative
# width 3 bounded; decided by Erik), which no longer writes the conserved
# ghosts — `P` in every ghost comes from `W` — and is the first refined row. Run against a tree from before
# (`git archive 7605986`), where `default_ops_W` does not exist, it measures
# the conserved fill instead, bounded and unbounded (Conservative width 3).
# A ghost counts as repaired where the policy would repair it for anything
# but the atmosphere: with conserved ghosts, the recovery of the ghost's
# conserved state with its own `B` (what step 8 first measured); with
# primitive ghosts, EntropyEOS's `check_prim_state` of the prolongated
# primitives `(ρ, s, Yₑ, asinh |u|)` — whether the ghost is an admissible
# state as it stands (the recovery of its `prim2con` would also flag states
# that sit exactly on a cap, the entropy ceiling or the rapidity cap, which
# the owners' own recoveries put there, by rounding).

using Printf
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD
const EOS = TG.EntropyEOS

refine_band!(forest) =
    (refine!(forest, [k for k in copy(forest.leaves)
                      if 0 ≤ block_origin(forest, k)[1] < 1 // 5 - 1e-12]);
     balance!(forest); forest)

const PRIMITIVES = isdefined(TG, :default_ops_W)

mutable struct Ghosts
    repaired::Int
    flags::UInt32
    below::Int          # primitive ghosts below the atmosphere's density, or non-finite
end

function count_ghosts!(acc, p, u, t)
    P = update_primitives!(p, u, t)
    U = p.f.U
    G, N = P.G[1], P.forest.N
    lo, hi = P.forest.extents[1]
    for b in 1:nblocks(P), i in vcat(1:G, (G + N + 1):(N + 2G))
        lo < coordinates(P, b, (i,))[1] < hi || continue
        Pc = ntuple(v -> P.work[i, v, b], TG.NP)
        (isfinite(Pc[TG.P_RHO]) && Pc[TG.P_RHO] ≥ (1 - 1e-12) * 1e-8) || (acc.below += 1)
        fl = if PRIMITIVES
            ps = EOS.PrimState{Float64}(Pc[TG.P_RHO], Pc[TG.P_S], Pc[TG.P_YE], Pc[TG.P_W])
            EOS.check_prim_state(p.eos, ps, p.pol)
        else
            cons = ntuple(v -> U.work[i, v, b], TG.NU)
            m = TG.aux_metric(Pc)
            E = TG.cons_𝔈(cons) / m.sqrtγ
            con2prim_hydro(p.eos, p.opts, p.pol, m, cons, E, TG.aux_B(Pc),
                           TG.state_guesses(cons))[2]
        end
        if fl & EOS.FLAG_POL_ANY & ~EOS.FLAG_POL_ATMOSPHERE != 0
            acc.repaired += 1
            acc.flags |= fl
        end
    end
    return acc
end

function run(case, kw)
    acc = Ghosts(0, 0, 0)
    r = evolve!(case, Val(1); N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc,
                after_step=(p, u, t) -> count_ghosts!(acc, p, u, t), kw...)
    return (; acc, D=sum(c -> c.tally.injected_D, r.chunks),
            τ=sum(c -> c.tally.injected_τ, r.chunks), nblocks=r.nblocks)
end

configs = if PRIMITIVES
    pv(w; bounded=true) = Operators(; prolongation=w, restriction=w, family=PointValue,
                                    bounded=bounded)
    [("uniform", (;)),
     ("primitives, parent-centred", (; refine=refine_band!)),
     ("primitives, w = 2", (; refine=refine_band!, ops_W=pv(2))),
     ("primitives, w = 4", (; refine=refine_band!, ops_W=pv(4))),
     ("primitives, w = 6", (; refine=refine_band!, ops_W=pv(6))),
     ("primitives, w = 4 unbounded", (; refine=refine_band!, ops_W=pv(4; bounded=false)))]
else
    cons(b) = Operators(; prolongation=3, restriction=2, family=Conservative, bounded=b)
    [("uniform", (;)),
     ("conserved, bounded", (; refine=refine_band!, ops_U=cons(true))),
     ("conserved, unbounded", (; refine=refine_band!, ops_U=cons(false)))]
end
println(PRIMITIVES ? "ghost fill in primitives (AMR-10)" : "ghost fill in conserved variables")
for (name, By, model) in (("hydrodynamic", 0.0, IdealConductor()),
                          ("magnetized", 1e-3, MagnetizationCutoff(Constant(1e3), 10)))
    case = atmosphere_tube_case(Float64; By=By, model=model, roots=20)
    for (label, kw) in configs
        t0 = time()
        r = run(case, kw)
        @printf("  %-12s %-28s repaired ghosts %4d (flags %5s), %4d below ρ_atm; injected D %+.2e, τ %+.2e  (%d blocks, %.0f s)\n",
                name, label, r.acc.repaired, string(r.acc.flags; base=16), r.acc.below, r.D,
                r.τ, r.nblocks, time() - t0)
    end
end
