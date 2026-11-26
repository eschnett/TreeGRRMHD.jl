using Test
using TreeAMR
using TreeGRRMHD

# One suite at every thread count: `Pkg.test` does not inherit `-t`, so the
# thread count has to be passed explicitly — see "Commands" in `CLAUDE.md`.
# Every number the suite asserts is expected to be identical at one thread and
# at four. The tier (added 2026-11-08): `full` by default, `quick` and
# `threaded` on GitHub CI; see `tiers.jl`.
include("tiers.jl")
@info "Running the $TEST_TIER tier of the tests on $(Threads.nthreads()) thread(s)"

@testset "TreeGRRMHD.jl" begin
    # Every test file below is `include`d into this one module, so a top-level
    # name defined in two of them is one name. Julia 1.12 and later — every
    # version this package supports — quietly redefine a `const`, so a
    # collision would pass every run, the later definition replacing the
    # earlier one. (TreeHydro's `TRACKED_1D` reached its `main` that way,
    # caught there only by CI's 1.11 cell, which refused the redefinition.)
    # Checked from the source text, once, before anything is included; the
    # helpers (`tiers.jl`, `recovery_states.jl`, …) are in the same module.
    @testset "No two test files define the same top-level constant" begin
        files = filter(f -> endswith(f, ".jl") && f != "runtests.jl", readdir(@__DIR__))
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
    # Step 1: the units, the variables and the 3+1 primitive recovery.
    include("units_tests.jl")
    # The helpers (moved out of the test files 2026-11-08) are included before
    # the first file that uses them; each names its users.
    include("recovery_states.jl")
    include("recovery_tests.jl")
    # Step 2: the pointwise physics. Uses `recovery_states.jl`'s generators.
    include("pointwise_tests.jl")
    # Step 3: the implicit stage solve. Uses `recovery_states.jl`'s generators.
    include("stage_solve_tests.jl")
    # Step 5: the staggered scheme on the mesh, and the 1D smooth waves.
    include("scheme_tests.jl")
    include("wave_study.jl")
    include("wave_tests.jl")
    # Step 6: the shock tubes, the resistive tube and the current sheet, and
    # the atmosphere. Use `prerequisite_tests.jl`'s `IRK`.
    include("shocktube_tests.jl")
    include("resistive_tests.jl")
    include("atmosphere_tests.jl")
    # Stage 3 of the recovery pipeline (2026-10-24): the step cache — warm
    # seeds and reuse inside a step, and what they must not change.
    include("step_cache_tests.jl")
    # Stage 4 (2026-10-30): the two-phase recovery — the fused recovery bit
    # for bit.
    include("two_phase_tests.jl")
    # Step 7: the higher-order reconstructions (uses `wave_study.jl`), and the
    # scheme in two and three dimensions.
    include("reconstruction_tests.jl")
    include("multid_runs.jl")
    include("multid_tests.jl")
    include("multid_cases_tests.jl")
    # Step 8: coarse-fine faces and regridding (both use `multid_runs.jl`'s
    # `md_run` and `refinement_meshes.jl`'s meshes).
    include("refinement_meshes.jl")
    include("refinement_tests.jl")
    include("refinement_cases_tests.jl")
    # Step 9: analytic curved backgrounds — the Wald solution and the magnetized Bondi
    # flow on a Kerr-Schild hole, with its interior.
    include("curved_tests.jl")
    # Step 10: the TOV star in the Cowling approximation, on `Analytic(TOVMetric)`; uses
    # `pointwise_tests.jl`'s `ForwardDiff`.
    include("tov_tests.jl")
    # Step 10, round 2: the vertex metric of the coupling contract (redesigned 2026-11-08),
    # through a sampled metric in TreeGH's `ADMMetric` form; uses `tov_tests.jl`'s star.
    include("vertex_metric_tests.jl")
    # Step 11: the matter adapter of the coupling contract (GH-3) and the `flags` verb,
    # without TreeGH; uses `tov_tests.jl`'s star. The coupled runs are `coupled/`'s.
    include("adm_matter_tests.jl")
    # Step 12: the magnetized TOV star on the Cowling metric — the exterior dipole and its
    # continuation, ∇·B, `E·B`, the drift and the stage solve's outcome map.
    include("magnetized_tov_tests.jl")
    # Step 13: the advection of `Yₑ`, and the TOV star on a tabulated EOS (DD2) in cold
    # β-equilibrium; the latter reads a real table, so its testsets run only where
    # `ENTROPYEOS_TABLE_DIR` holds one (`tables.jl`).
    include("ye_advection_tests.jl")
    include("tables.jl")
    include("tabulated_tov_tests.jl")
    # Step 16: the neutrino rates and opacities, pointwise (rung 8a). Uses
    # `recovery_states.jl`'s `rec_allocations`.
    include("neutrino_rates_tests.jl")
    # Step 17: the optical depth on the forest (rung 8b).
    include("optical_depth_tests.jl")
    # Step 18: M0's free-streaming neutrinos, without feedback (rungs 8c, 8d).
    include("m0_tests.jl")
    # Step 19: the neutrinos' feedback on the matter (rungs 8e, 8f). Uses
    # `m0_tests.jl`'s helpers.
    include("neutrino_coupling_tests.jl")
    # Step 14: the run bit for bit at another thread count (a subprocess), and on a device,
    # or on the CPU standing in for one (`TREEGRRMHD_TEST_BACKEND`; the file also runs by
    # itself, which is how a device job runs it).
    include("threading_tests.jl")
    include("device_tests.jl")
end

isempty(TABLE_SKIPPED) ||
    @info "without DD2 under ENTROPYEOS_TABLE_DIR, $(length(TABLE_SKIPPED)) testsets were " *
          "left out:\n  " * join(first.(TABLE_SKIPPED), "\n  ")
isempty(TIER_SKIPPED) ||
    @info "the $TEST_TIER tier left out $(length(TIER_SKIPPED)) testsets " *
          "(TREEGRRMHD_TEST_TIER=full runs them):\n  " *
          join(("$name ($marker)" for (name, marker) in TIER_SKIPPED), "\n  ")
