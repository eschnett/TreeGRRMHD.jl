# Behaviour of an upstream release this package relies on and does not yet
# exercise through its own code. A landed upstream item loses its probe in
# `prerequisite_tests.jl` once `[compat]` requires the release that has it
# (decided by Erik 2026-10-14); what the scheme needs of it is then tested
# here until the step that uses it tests it in place, and moves there.

# Two roots per dimension, one of them refined, 2:1 balanced, so that the
# blocks do not all have the same spacing.
function ub_forest(::Val{D}; N=8) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, [first(forest.leaves)])
    balance!(forest)
    return forest
end

# A stellar surface: a drop over thirteen orders of magnitude.
ub_surface(x) = (x[1] < 0.3 ? 1.0 : 1.0e-13,)

@testset "The bounded conservative prolongation refines a surface without new extrema: D=$D" for D in 1:2
    # Guards what `CODE.md`, "Refinement", builds on (AMR-4, landed in
    # TreeAMR 0.2.1 as `Operators(; …, bounded = true)`, the MC-bounded
    # width-3 conservative prolongation): a surface refined everywhere stays
    # positive and within its range and keeps its total, where the unbounded
    # operator overshoots it — so an unbounded operator, or a release that
    # dropped the bound, would floor cells at a stellar surface. Moves to
    # step 8's refinement tests, which regrid the matter set with it.
    bounded = Operators(; prolongation=3, restriction=2, family=Conservative, bounded=true)
    forest = ub_forest(Val(D))
    fs = FieldSet{Float64}(forest, 1; G=1, centering=cellcentered(D))
    fill_by_coordinates!(AllVariables(ub_surface), fs)
    before = total_mass(fs)
    regrid!(forest, fs => GhostSchedule(fs, bounded); flags=fill(Refine, nleaves(forest)))
    fill_ghosts!(fs, GhostSchedule(fs, bounded))
    lo, hi = extrema(fs.work)
    @test 0 < lo && hi <= 1 + eps()
    @test isapprox(total_mass(fs), before; rtol=1e-13)

    # The control: unbounded, the same refinement overshoots.
    unbounded = Operators(; prolongation=3, restriction=2, family=Conservative)
    forest = ub_forest(Val(D))
    fs = FieldSet{Float64}(forest, 1; G=1, centering=cellcentered(D))
    fill_by_coordinates!(AllVariables(ub_surface), fs)
    regrid!(forest, fs => GhostSchedule(fs, unbounded); flags=fill(Refine, nleaves(forest)))
    lo, hi = extrema(fs.work)
    @test lo < 0 || hi > 1
end
