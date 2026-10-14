using Test
using TreeAMR
using TreeGRRMHD

# One suite, run whole, at every thread count: `Pkg.test` does not inherit
# `-t`, so the thread count has to be passed explicitly — see "Commands" in
# `CLAUDE.md`. Every number the suite asserts is expected to be identical at
# one thread and at four.
@info "Running the tests on $(Threads.nthreads()) thread(s)"

@testset "TreeGRRMHD.jl" begin
    # Every test file below is `include`d into this one module, so a top-level
    # name defined in two of them is one name. Julia before 1.12 refuses to
    # redefine a `const` and 1.12 and later quietly allow it, so a collision
    # passes every local run at the newer version and fails only the floor
    # entries of CI — which is how TreeHydro's `TRACKED_1D` reached its `main`.
    # Checked from the source text, once, before anything is included, so
    # that it fails at every version and not only at the floor.
    @testset "No two test files define the same top-level constant" begin
        files = filter(f -> endswith(f, "_tests.jl"), readdir(@__DIR__))
        defined = Dict{String,Vector{String}}()
        for f in files, line in eachline(joinpath(@__DIR__, f))
            m = match(r"^const\s+([^\s=]+)", line)
            m === nothing && continue
            push!(get!(defined, m.captures[1], String[]), f)
        end
        duplicates = sort([name => fs for (name, fs) in defined if length(fs) > 1])
        @test isempty(duplicates)
        isempty(duplicates) || @info "duplicated top-level constants" duplicates
    end

    include("precision_tests.jl")
    # What the resolved releases provide, and which upstream items have
    # landed: before any physics, so that a release that lost a name is
    # found here rather than by a `MethodError` in the middle of a step.
    include("prerequisite_tests.jl")
    # What this package relies on from upstream items that have landed and
    # are not yet exercised through its own code.
    include("upstream_behaviour_tests.jl")
    # Step 1: the units, the variables and the 3+1 primitive recovery.
    include("units_tests.jl")
    include("recovery_tests.jl")
    # Step 2: the pointwise physics. Uses `recovery_tests.jl`'s state
    # generators, so it comes after it.
    include("pointwise_tests.jl")
    # Step 3: the implicit stage solve. Uses `recovery_tests.jl`'s generators.
    include("stage_solve_tests.jl")
    # Step 5: the staggered scheme on the mesh, and the 1D smooth waves.
    include("scheme_tests.jl")
    include("wave_tests.jl")
end
