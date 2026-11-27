# M0's steady state in the uniform medium at rest at 16³, 32³ and 64³: the L1 error
# inside r ≤ 0.6 against the radial solution and the rates between them, per
# species (step 18; "Validation ladder" 8c in `CODE.md`). The suite runs the
# first two (`test/m0_tests.jl`).
#
#     julia --project=. -t 4 bin/m0_ball.jl            # all three
#     julia --project=. -t 4 bin/m0_ball.jl roots=2,4  # the suite's pair
#     julia --project=. -t 4 bin/m0_ball.jl t_end=1.2  # the front further out
using TreeAMR, TreeGRRMHD, StaticArrays, Printf

const ARGS_KV = Dict(split(a, '=') for a in ARGS)
roots = parse.(Int, split(get(ARGS_KV, "roots", "2,4,8"), ','))
t_end = parse(Float64, get(ARGS_KV, "t_end", "0.7"))

# The test's helpers, without its testsets.
let src = read(joinpath(@__DIR__, "..", "test", "m0_tests.jl"), String)
    stop = findfirst("@testset \"M0's steady", src)
    include_string(Main, "computed_once(f, key) = f()\n" * src[1:(first(stop) - 1)])
end

errs = Dict{Tuple{Int,Int},Float64}()
for r in roots
    t = @elapsed run = evolve!(Float64, m0_ball(; roots=r, ρ_out=1.0), Val(3); N=8, t_end=t_end,
                               limiter=:mc, neutrinos=M0_THIN)
    for i in 1:3
        errs[(r, i)] = m0_error(run, i, 0.6)
    end
    @printf("%3d³: %.1f s, L1 errors %.3e %.3e %.3e\n", 8r, t, errs[(r, 1)], errs[(r, 2)],
            errs[(r, 3)])
end
for k in 2:length(roots), i in 1:3
    @printf("rate %d³ → %d³, species %d: %.3f\n", 8roots[k - 1], 8roots[k], i,
            log2(errs[(roots[k - 1], i)] / errs[(roots[k], i)]))
end
