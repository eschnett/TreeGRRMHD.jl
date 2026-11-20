# The coupled suite of step 11: TreeGRRMHD's matter on TreeGeneralizedHarmonic's spacetime,
# through `tov_coupled.jl`, the script. It runs in this directory's environment, which
# loads TreeGH beside this checkout (`Project.toml`), never in the package's own suite:
#
#     julia --project=coupled -e 'using Pkg; Pkg.instantiate()'
#     julia --project=coupled coupled/runtests.jl
#
# with `-t 4` and `--check-bounds=yes` as for the package's suite (`CLAUDE.md`,
# "Commands"); `TREEGRRMHD_TEST_TIER` chooses the tier as there (`test/tiers.jl`).

using Test
using TreeAMR
using StaticArrays

include(joinpath(@__DIR__, "..", "test", "tiers.jl"))
@info "Running the $TEST_TIER tier of the coupled tests on $(Threads.nthreads()) thread(s)"

include(joinpath(@__DIR__, "tov_coupled.jl"))

@testset "TreeGRRMHD coupled to TreeGeneralizedHarmonic" begin
    include("prerequisite_tests.jl")
    include("coupled_tests.jl")
    # Step 13: the tabulated star (DD2) in the evolved spacetime, where
    # `ENTROPYEOS_TABLE_DIR` holds the table (`test/tables.jl`).
    include(joinpath(@__DIR__, "..", "test", "tables.jl"))
    include("tabulated_tests.jl")
end

isempty(TABLE_SKIPPED) ||
    @info "without DD2 under ENTROPYEOS_TABLE_DIR, $(length(TABLE_SKIPPED)) testsets were " *
          "left out:\n  " * join(first.(TABLE_SKIPPED), "\n  ")
isempty(TIER_SKIPPED) ||
    @info "the $TEST_TIER tier left out $(length(TIER_SKIPPED)) testsets " *
          "(TREEGRRMHD_TEST_TIER=full runs them):\n  " *
          join(("$name ($marker)" for (name, marker) in TIER_SKIPPED), "\n  ")
