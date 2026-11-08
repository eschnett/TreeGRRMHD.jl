# The seeded state generators of `recovery_tests.jl`, which `pointwise_tests.jl`
# and `stage_solve_tests.jl` draw from too: an ideal gas per Γ in one box, a
# boosted spinning Kerr-Schild hole sampled at random events, random
# directions and fluid states, and the allocation probe. A helper, not a test
# file (moved out of `recovery_tests.jl` on 2026-11-08): `runtests.jl`
# includes it before the files that use it.

using Random: Xoshiro
using StaticArrays: SVector
import SpacetimeMetrics

const REC_EOS = TreeGRRMHD.EntropyEOS
const REC_BOX = (; s_ref=5.0, s_window=(4.0, 6.0), ρ_bounds=(1e-10, 1e-2),
                 yₑ_bounds=(0.05, 0.6))
const REC_ρ_ATM = 1e-10
# Γ = 4/3, 5/3 and 2: EOS-2's downstream check. Two is the TOV polytrope's.
const REC_GAMMAS = (4 // 3, 5 // 3, 2)

rec_gas(Γ, ::Type{T}=Float64) where {T} =
    REC_EOS.IdealGasEOS{T}(; Γ=T(Γ), K_ref=T(100), REC_BOX...)

# A boosted spinning Kerr-Schild hole, sampled at a random event between
# r = 3M and 20M: lapse, shift and a non-diagonal 3-metric all far from
# flat, with `√γ` between about 1 and 2. `rec_event` returns the spacetime
# and the event `(t, x, y, z)`, for the tests that need the metric's
# derivatives there too (step 2); `rec_metric` its 3+1 split.
function rec_event(rng)
    hole = SpacetimeMetrics.KerrSchild{Float64}(1.0, 0.9 * rand(rng))
    vb = SVector{3}(randn(rng, 3))
    m4 = SpacetimeMetrics.boost(hole, (0.6 * rand(rng) / sqrt(sum(abs2, vb))) * vb)
    while true
        x = SVector{3}(randn(rng, 3))
        x = ((3 + 17 * rand(rng)) / sqrt(sum(abs2, x))) * x
        p = SVector(10 * randn(rng), x...)
        α, _, _ = SpacetimeMetrics.adm_decompose(m4, p)
        α > 0 && return m4, p
    end
end

function rec_metric(rng)
    m4, p = rec_event(rng)
    return Metric3(SpacetimeMetrics.adm_decompose(m4, p)...)
end

# A random direction of unit length in the metric `m`.
function rec_direction(rng, m)
    d = SVector{3}(randn(rng, 3))
    return d / sqrt(TreeGRRMHD.dot_up(m, d, d))
end

# A random fluid state, as `primitive_state` builds it.
function rec_prim(rng, eos, m; wmax=acosh(10.0), swindow=(4.05, 5.95), log10ρ=(-9.0, -2.1))
    ρ = exp10(log10ρ[1] + (log10ρ[2] - log10ρ[1]) * rand(rng))
    s = swindow[1] + (swindow[2] - swindow[1]) * rand(rng)
    yₑ = 0.06 + 0.53 * rand(rng)
    w = wmax * rand(rng)
    v = tanh(w) * rec_direction(rng, m)
    return primitive_state(eos, m, ρ, v, s, yₑ)
end

# The bytes one call allocates, after a first call compiles it, measured
# inside a function rather than at the top level of a testset.
function rec_allocations(f, args...)
    f(args...)
    return @allocated f(args...)
end
