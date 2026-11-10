# Step 11: this package's side of the coupling contract's matter direction (GH-3) — the
# stress-energy averaged from the cells to the producer's owned vertices as `(ρ, S_i, S_ij)`
# (`src/adm_matter.jl`; "Matter to the vertices" in `CODE.md`) — and the `flags` verb the
# coupled script's regrid calls. Tested without TreeGH: the adapter is bound to a vertex set
# of this package's own making on the matter's forest, as TreeGH's right-hand side binds it
# to its own. Uses `tov_tests.jl`'s `TV_CASE`.

const AM = TreeGRRMHD

# The Cowling star on `[−16, 16]³`, two roots, no boxes, `N` cells a block, prepared at its
# initial state on its own metric.
function am_star(N)
    forest = Forest{Float64}(TV_CASE.roots; N=N, periodic=TV_CASE.periodic,
                             extents=TV_CASE.extents)
    p = MHDProblem(mhd_fields(Float64, forest), TV_CASE; limiter=:ppm)
    u = statevector(evolved_sets(p))
    AM.initial_state!(u, p, TV_CASE, 0.0)
    AM.prepare!(p, u, 0.0)
    return forest, p, u
end

# The star's exact `(ρ, S_i, S_ij)` at `x`: at rest with no field, `ρ = e = ρ₀ + p/(Γ − 1)`,
# `S_i = 0`, `S_ij = p γ_ij`, on the polytrope (the atmosphere on the same isentrope).
function am_exact(x)
    w = TV_CASE.fields.w
    eq = w.metric.eq
    ρ₀ = max(tov_density(w.metric, sum(abs2, x)), w.ρ_atm)
    p = eq.K * ρ₀^eq.Γ
    γ = point_metric(TV_CASE.provider, 0.0, x).γ
    return ρ₀ + p / (eq.Γ - 1), p * γ
end

# The largest errors of the adapter at the owned vertices with `r ≤ r_in` (inside the
# star, away from its surface) and `r ≥ r_out` (the atmosphere), scaled by the central
# `e_c` and `p_c`; and the adapter bound once.
function am_errors(N; r_in=4.0, r_out=11.0)
    forest, p, _ = am_star(N)
    U = FieldSet{Float64}(forest, 1; G=0, centering=vertexcentered(3))
    at = ADMMatter(p)(U)
    origins, spacings = block_origins(forest, Float64), block_spacings(forest, Float64)
    e_c, p_c = am_exact(SVector(0.0, 0.0, 0.0))
    p_c = p_c[1, 1]
    errs = Dict(:ρ_in => 0.0, :Sij_in => 0.0, :S => 0.0, :ρ_out => 0.0, :sym => 0.0,
                :neg => 0.0)
    for b in 1:nblocks(U), k in 1:forest.N, j in 1:forest.N, i in 1:forest.N
        n = (i, j, k)
        x = AM.point_x(origins[b], spacings[b], n, Val((1, 1, 1)))
        r = sqrt(sum(abs2, x))
        v = at(n, b)
        e, Sij = am_exact(x)
        errs[:S] = max(errs[:S], maximum(abs, v.S) / e_c)
        errs[:sym] = max(errs[:sym], maximum(abs, v.Sij - v.Sij'))
        errs[:neg] = min(errs[:neg], v.ρ)
        if r ≤ r_in
            errs[:ρ_in] = max(errs[:ρ_in], abs(v.ρ - e) / e_c)
            errs[:Sij_in] = max(errs[:Sij_in], maximum(abs, v.Sij - Sij) / p_c)
        elseif r ≥ r_out
            errs[:ρ_out] = max(errs[:ρ_out], abs(v.ρ - e) / e_c)
        end
    end
    return errs, at, U, p, forest
end

@testset "The matter adapter converges at second order to the star's (ρ, S_i, S_ij)" begin
    # Guards `ADMMatter` (`src/adm_matter.jl`): a cell stencil off by one (the vertex `n` is
    # the low corner of cell `n`), a missing lowering of `S^ij` or a lowering with the wrong
    # metric, a weight other than 1/8, or `𝓔` confused with `D` or `τ` would leave an O(1)
    # error or cost the order. On the Cowling star at rest (`ρ = e`, `S_i = 0`,
    # `S_ij = p γ_ij` exactly) at `h = 2` and `1`, inside the star (`r ≤ 4`, away from the
    # surface at `R̄ = 8.1`, where the density's kink makes any mean first order) and in the
    # atmosphere (`r ≥ 11`).
    e1, _, _, _, _ = am_errors(8)
    e2, at, U, p, forest = am_errors(16)
    rate(k) = log2(e1[k] / e2[k])
    @info "ADMMatter on the Cowling star: errors at h = 1 ρ $(tv3(e2[:ρ_in])), " *
          "S_ij $(tv3(e2[:Sij_in])) (rates $(round(rate(:ρ_in); digits=2)), " *
          "$(round(rate(:Sij_in); digits=2))); |S_i| $(tv3(e2[:S])), atmosphere " *
          "$(tv3(e2[:ρ_out])), asymmetry $(e2[:sym])"
    @test 1.7 < rate(:ρ_in) < 2.6
    @test 1.7 < rate(:Sij_in) < 2.6
    # Measured 2026-11-10: 4.74e−2 and 8.85e−2 of `e_c` and `p_c` at `h = 1` (rates 1.82,
    # 1.74): the mean of eight cell values, each a cell average, on a star four cells
    # across its radius at `h = 2`.
    @test e2[:ρ_in] < 6e-2 && e2[:Sij_in] < 1.1e-1
    # At rest with no field: `S_i` from the recovery's velocity alone.
    @test e2[:S] < 1e-12
    @test e2[:ρ_out] < 1e-9
    # A convex mean of nonnegative energies, and symmetric to the bit.
    @test e2[:neg] ≥ 0 && e2[:sym] == 0
end

# A stand-in for a device adaptor: an array becomes its pointer, an `isbits` value.
struct AMPointer end
AM.Adapt.adapt_storage(::AMPointer, a::Array) = pointer(a)

@testset "The bound matter adapter is isbits, allocation-free, and refuses another mesh" begin
    # Guards the contract's form (GH-3): TreeGH calls the bound adapter inside its kernel,
    # once per SIMD lane on the CPU, so it must be `isbits` and must not allocate; a binding
    # to another forest or to a set that is not vertex-centered would read the wrong points
    # and must throw, saying why.
    _, at, U, p, forest = am_errors(8)
    # `isbits` once adapted: every field but the cell array is, and Adapt reaches that.
    @test isbits(AM.Adapt.adapt(AMPointer(), at))
    f(at) = at((3, 4, 5), 2)
    f(at)
    @test (@allocated f(at)) == 0
    @test f(at).Sij isa SMatrix{3,3,Float64,9} && f(at).S isa SVector{3,Float64}
    @test f(at).ρ isa Float64
    other = Forest{Float64}(TV_CASE.roots; N=8, periodic=TV_CASE.periodic,
                            extents=TV_CASE.extents)
    @test_throws ArgumentError ADMMatter(p)(FieldSet{Float64}(other, 1; G=0,
                                                              centering=vertexcentered(3)))
    @test_throws ArgumentError ADMMatter(p)(FieldSet{Float64}(forest, 1; G=0,
                                                              centering=cellcentered(3)))
end

@testset "The flags verb is evolve!'s indicator on a prepared state" begin
    # Guards `flags` (`src/refinement.jl`), which the coupled script's regrid calls with the
    # producer's metric: it must fill `P` at the state it is given before the indicator
    # reads it, as `evolve!` did inline, so that a script's flags are `evolve!`'s.
    forest, p, u = am_star(8)
    kw = (; refine_tol=0.05, coarsen_tol=0.01, maxlevel=1)
    fl = AM.flags(p, u, 0.0; kw...)
    AM.prepare!(p, u, 0.0; metric=TV_CASE.provider)
    @test isequal(fl, mhd_flags(p; kw...))
    @test any(f -> f isa Tuple && first(f) == Refine, fl)
end
