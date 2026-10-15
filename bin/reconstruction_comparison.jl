# Balsara's five shock tubes under every reconstruction (step 7): the table of
# self-convergence rates and plateau errors in `CODE.md`, "Validation ladder"
# 1b and "Reconstruction".
#
#     julia --project=. -t 4 bin/reconstruction_comparison.jl [outdir] [method ...]
#
# For each method (`:mc`, `:ppm`, `:wenoz`, `:mp5` by default) and each test
# `k ∈ 1:5`, the tube runs at 100, 200 and 400 cells exactly as
# `test/shocktube_tests.jl` runs it (HLLE, SSP3(4,3,3), `cfl = 2/5`, four
# chunks), and at 3200 cells — the method's own reference, eight times the
# finest, restricted to 400 cells and written to `outdir` (default
# `bin/output/`, which `.gitignore` keeps out of the repository) as
# `balsara<k>_<method>.ref` in the format of `test/data/` — so that a rerun
# reuses it. Printed per method and test:
#
#   * the L1 rates of `D`, `S_x`, `τ`, `𝔅ʸ` by least squares over the three
#     resolutions, against the method's own 3200-cell run (self-convergence)
#     and against the stored MC reference of `test/data/` (which the suite
#     reads);
#   * the worst plateau error at 400 cells over `BALSARA_PLATEAUS`'s windows,
#     with the region, and Balsara 3's largest density at 400 cells (the
#     exact shell holds 10.1);
#   * the stages' outcomes and the resets, and the wall time.

using Printf
using TreeGRRMHD
const TG = TreeGRRMHD

const MS = (100, 200, 400)
const M_REF = 3200
const M_OUT = 400

function run_tube(k, M, lim)
    t_end = Float64(BALSARA[k].t_end)
    case = balsara_case(Float64, k; roots=M ÷ 10)
    t0 = time()
    r = evolve!(case, Val(1); N=10, t_end=t_end, chunk=t_end / 4, limiter=lim)
    wall = time() - t0
    return (; r, pr=profile_1d(r.p, r.u, r.t), wall, tallies=[c.tally for c in r.chunks])
end

columns(pr) = hcat(pr.U, pr.B[:, 2], pr.B[:, 3])
restrict(a, f) = [sum(a[(f * (i - 1) + 1):(f * i), c]) / f for i in 1:(size(a, 1) ÷ f),
                  c in 1:size(a, 2)]

function read_ref(path)
    bytes = read(path)
    return reshape(Float64.(ltoh.(reinterpret(Float32, bytes))), M_OUT, 12)
end
write_ref(path, a) = write(path, htol.(Float32.(vec(a))))

function own_reference(k, lim, outdir)
    path = joinpath(outdir, "balsara$(k)_$(lim).ref")
    isfile(path) && return read_ref(path), NaN
    run = run_tube(k, M_REF, lim)
    a = restrict(columns(run.pr), M_REF ÷ M_OUT)
    mkpath(outdir)
    write_ref(path, a)
    return read_ref(path), run.wall
end

const VARS = (("D", 1), ("S_x", 2), ("τ", 5), ("𝔅ʸ", 11))

function rates(runs, R)
    map(VARS) do (name, col)
        errs = map(runs) do run
            ref = restrict(R, M_OUT ÷ size(run.pr.U, 1))[:, col]
            sum(abs, columns(run.pr)[:, col] .- ref) / size(run.pr.U, 1)
        end
        name => (TG.convergence_rate(1 ./ collect(MS), errs), errs)
    end
end

function plateau_worst(k, pr)
    worst, where = 0.0, 0
    for (region, lo, hi) in BALSARA_PLATEAUS[k]
        cells = findall(x -> lo ≤ x ≤ hi, pr.x)
        mean = ntuple(j -> sum(c -> plateau_state(pr, c)[j], cells) / length(cells), 7)
        e = maximum(plateau_error(k, region, mean))
        e > worst && ((worst, where) = (e, region))
    end
    return worst, where
end

function main(args)
    outdir = isempty(args) || startswith(args[1], ":") ? joinpath(@__DIR__, "output") : args[1]
    methods = [Symbol(lstrip(a, ':')) for a in args if startswith(a, ":")]
    isempty(methods) && (methods = [:mc, :ppm, :wenoz, :mp5])
    for lim in methods, k in 1:5
        runs = [run_tube(k, M, lim) for M in MS]
        own, wall_ref = own_reference(k, lim, outdir)
        mc = read_ref(joinpath(@__DIR__, "..", "test", "data", "balsara$k.ref"))
        self = rates(runs, own)
        vsmc = rates(runs, mc)
        pw, region = plateau_worst(k, last(runs).pr)
        bits = sum(t -> collect(t.bits), last(runs).tallies)
        resets = sum(run -> sum(t -> t.resets + t.step_resets, run.tallies), runs)
        policy = sum(run -> sum(t -> t.policy, run.tallies), runs)
        @printf("%-6s Balsara %d: self rates", lim, k)
        for (name, (q, _)) in self
            @printf(" %s %.2f", name, q)
        end
        print("; against MC-3200")
        for (name, (q, _)) in vsmc
            @printf(" %s %.2f", name, q)
        end
        @printf("; plateaus worst %.4f (R%d)", pw, region)
        k == 3 && @printf("; max ρ at 400 %.3f", maximum(last(runs).pr.ρ))
        @printf("; resets %d, repaired stages %d, ideal stages %d of %d", resets, policy,
                bits[2], sum(bits))
        @printf("; wall %.1f s (400 cells), reference %.0f s\n", last(runs).wall, wall_ref)
        for (name, (_, errs)) in self
            @printf("        %s self L1 errors %s\n", name, string(round.(errs, sigdigits=3)))
        end
        flush(stdout)
    end
end

main(ARGS)
