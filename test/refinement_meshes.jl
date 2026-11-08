# The static two-level meshes of `refinement_tests.jl` and
# `refinement_cases_tests.jl`. A helper, not a test file (moved out of
# `refinement_tests.jl` on 2026-11-08): `runtests.jl` includes it before both.

const RF = TreeGRRMHD

# Two roots per dimension, one of them refined, 2:1 balanced, so that the
# blocks do not all have the same spacing.
function rf_forest(::Val{D}; N=8) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    rf_refine_first!(forest)
    return forest
end

# The static meshes of the tests: the first root refined, or the middle half
# of the roots of a one-dimensional mesh.
rf_refine_first!(forest) = (refine!(forest, [first(forest.leaves)]); balance!(forest); forest)
function rf_refine_middle!(forest)
    nr = length(forest.leaves)
    refine!(forest, [k for (i, k) in enumerate(copy(forest.leaves)) if nr ÷ 4 < i ≤ 3nr ÷ 4])
    balance!(forest)
    return forest
end
