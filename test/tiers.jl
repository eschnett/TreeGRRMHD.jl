# The suite's tiers (added 2026-11-08, after TreeGeneralizedHarmonic's of
# 2026-11-06), as a helper rather than a test: `runtests.jl` includes it
# first, and every file that marks a testset includes it if it was not.
#
# On GitHub's runners (four shared vCPUs, `--check-bounds=yes`) the suite took
# 14 minutes at one thread and 32 at four (2026-11-05). The four-thread time
# is not the 3D or the curved runs, which run faster threaded, but the 1D
# studies: many short steps on a few hundred cells, 4–8 times slower at four
# threads there than at one (the Alfvén wave under PPM, WENO-Z and MP5: 41 s
# against 278 s). On Symmetry four threads are faster than one throughout. So
# the suite has three tiers, chosen by `TREEGRRMHD_TEST_TIER`:
#
#   * `full`, the default — every testset. This is what `Pkg.test()` runs on a
#     workstation before a pull request and on Symmetry, what the weekly CI
#     run runs, and what `ci/full-suite.sh` runs and reports as a commit
#     status.
#   * `quick` — GitHub CI's one-thread cells: every testset but those marked
#     `@full_tier`, the runs at the finest resolution of a 3D curved study and
#     the 1000-step uniform flows.
#   * `threaded` — GitHub CI's four-thread cell: `quick` less the testsets
#     marked `@serial_tier`, the 1D studies. The kernels are one code for
#     D = 1, 2, 3, so at four threads the 2D and 3D runs, the refinement, the
#     curved backgrounds, the step cache and the two-phase recovery exercise
#     every threaded path; a 1D study at four threads adds the wait, not a
#     path. The 1D runs that hold a 2D or 3D run to the 1D run bit for bit
#     (`multid_tests.jl`) stay.
#
# A testset left out is still a testset, holding one skipped test (per
# iteration of a `for` testset), so the summary counts it as "Broken" and
# `runtests.jl` names it at the end: a green tick never claims more than it
# ran.

const TEST_TIER = let tier = get(ENV, "TREEGRRMHD_TEST_TIER", "full")
    tier in ("full", "quick", "threaded") ||
        error("TREEGRRMHD_TEST_TIER=$tier: the tiers are full (the default), quick " *
              "(GitHub CI at one thread) and threaded (GitHub CI at four)")
    Symbol(tier)
end

# The testsets the tier left out, each with the marker that left it out, which
# `runtests.jl` lists at the end.
const TIER_SKIPPED = Pair{String,String}[]

# `@full_tier @testset "name" … begin … end`: the testset in the full tier
# only; elsewhere a testset of the same name holding one skipped test.
macro full_tier(ex)
    return tier_testset(ex, :(TEST_TIER === :full), "@full_tier")
end

# `@serial_tier @testset "name" … begin … end`: the testset in the full and
# the quick tier, and skipped in the threaded one.
macro serial_tier(ex)
    return tier_testset(ex, :(TEST_TIER !== :threaded), "@serial_tier")
end

# The marked testset, or its skipped twin: the same name, options and loop,
# the body replaced. The name may interpolate the loop variable, since the
# twin keeps the loop.
function tier_testset(ex, runs, marker)
    ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@testset") ||
        error("$marker marks a `@testset`")
    i = findfirst(a -> a isa String || (a isa Expr && a.head === :string), ex.args)
    i === nothing && error("$marker needs a testset with a name")
    name = ex.args[i]
    skip = quote
        push!(TIER_SKIPPED, $name => $marker)
        @test_skip "left out of the $(TEST_TIER) tier by " * $marker
    end
    twin = Expr(:macrocall, ex.args[1:(end - 1)]...)
    body = ex.args[end]
    push!(twin.args, body isa Expr && body.head === :for ? Expr(:for, body.args[1], skip) : skip)
    return esc(:($runs ? $ex : $twin))
end

# A value computed on first use and kept for the testsets after it: the runs
# that several testsets read. A top-level `const` would compute it while the
# file is included, whatever the tier left out.
const COMPUTED_ONCE = Dict{Any,Any}()
computed_once(f, key) = get!(f, COMPUTED_ONCE, key)
