# The interface-order table of step 8: the circularly polarized Alfvén wave
# crossing coarse-fine faces, for the potentials' PointValue widths
# `w ∈ {2, 4, 6}` (prolongation and restriction), which decides the (open)
# width of `A` and `Φ̃` ("Field sets" and "Refinement" in `CODE.md`).
#
#     julia --project=. bin/interface_order.jl [1d] [2d] [matter=unbounded]
#
# 1D: the wave of rung 1a (`alfven_wave_case`, MC, HLLE, SSP3(4,3,3), one
# period) on a static two-level mesh — `M` coarse cells in blocks of `N = 8`
# with the middle half of the roots refined, so the wave crosses two
# coarse-fine faces twice — at M = 32, 64, 128, 256, against the exact
# solution; the controls are the uniform coarse and uniform fine meshes.
# 2D: the wave along the diagonal (`oblique_case`, as `multid_cases_tests.jl`
# runs it, an eighth of a period) on 2 × 2 roots of N = 8, 16, 32 with one
# root refined. The matter's operators are the default (bounded) unless
# `matter=unbounded`. Printed: L1 and L∞ errors of `S_y` and `A_y` (1D), of `S_x`
# and `A_z` (2D), and least-squares rates over the resolutions.

using Printf
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD

rate(hs, es) = TG.convergence_rate(collect(hs), collect(es))

function refine_middle!(forest)
    nr = length(forest.leaves)
    refine!(forest, [k for (i, k) in enumerate(copy(forest.leaves)) if nr ÷ 4 < i ≤ 3nr ÷ 4])
    balance!(forest)
    return forest
end
refine_first!(forest) = (refine!(forest, [first(forest.leaves)]); balance!(forest); forest)

function linf(p, u, uref, i, v)
    d = u .- uref
    return mesh_mapreduce(abs, max, 0.0, p.sets[i], stateview(d, p.sets, i); vars=v)
end

function errors(r, case, Uv, Aj)
    ex = exact_state(r.p, case, r.t)
    e = l1_errors(r.p, r.u, ex)
    return (e.U[Uv], e.A[Aj], linf(r.p, r.u, ex, TG.SEG_U, Uv),
            linf(r.p, r.u, ex, TG.SEG_A + Aj - 1, 1))
end

function study(label, runs, hs)
    names = ("L1 U", "L1 A", "L∞ U", "L∞ A")
    for (k, name) in enumerate(names)
        es = [x[k] for x in runs]
        @printf("  %-28s %-6s %s  rate %.2f\n", label, name,
                join((@sprintf("%.3e", e) for e in es), " "), rate(hs, es))
    end
end

ops_U = Operators(; prolongation=3, restriction=2, family=Conservative,
                  bounded=!any(==("matter=unbounded"), ARGS))
println("matter operators: ", ops_U)

if isempty(ARGS) || "1d" in ARGS
    println("1D Alfvén wave, one period, MC; S_y and A_y")
    Ms = (32, 64, 128, 256)
    hs = 1 ./ Ms
    for (label, kw) in (("uniform coarse", (;)),
                        ("uniform fine (2M)", (; fine=true)),
                        ("two-level, w = 2", (; w=2)), ("two-level, w = 4", (; w=4)),
                        ("two-level, w = 6", (; w=6)))
        t0 = time()
        runs = map(Ms) do M
            fine = get(kw, :fine, false)
            c = alfven_wave_case(Float64; roots=(fine ? 2M : M) ÷ 8)
            w = get(kw, :w, 4)
            r = evolve!(c, Val(1); N=8, t_end=alfven_period(c), limiter=:mc,
                        refine=haskey(kw, :w) ? refine_middle! : nothing, ops_U=ops_U,
                        ops_A=Operators(; prolongation=w, restriction=w))
            errors(r, c, TG.U_S + 1, 2)
        end
        study(label, runs, hs)
        @printf("  (%.0f s)\n", time() - t0)
    end
end

if isempty(ARGS) || "2d" in ARGS
    println("2D Alfvén wave along the diagonal, an eighth of a period, MC; S_x and A_z")
    c1 = alfven_wave_case(Float64; roots=1)
    t_end = alfven_period(c1) / 8
    Ns = (8, 16, 32)
    hs = 1 ./ (2 .* Ns)
    for (label, kw) in (("uniform coarse", (;)), ("uniform fine (2N)", (; fine=true)),
                        ("two-level, w = 2", (; w=2)), ("two-level, w = 4", (; w=4)),
                        ("two-level, w = 6", (; w=6)))
        t0 = time()
        runs = map(Ns) do N
            fine = get(kw, :fine, false)
            c = oblique_case(c1; N=N, roots=fine ? 4 : 2)
            w = get(kw, :w, 4)
            r = evolve!(c, Val(2); N=N, t_end=t_end, limiter=:mc,
                        refine=haskey(kw, :w) ? refine_first! : nothing, ops_U=ops_U,
                        ops_A=Operators(; prolongation=w, restriction=w))
            errors(r, c, TG.U_S, 3)
        end
        study(label, runs, hs)
        @printf("  (%.0f s)\n", time() - t0)
    end
end
