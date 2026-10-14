# The staggered scheme of step 5, on the mesh but before any evolution is
# measured: where the field sets put each quantity, the multi-set state vector
# over them (AMR-1's downstream check), the curl that makes `∇·B` structural,
# and the right-hand side's purity. The waves that measure the scheme are in
# `wave_tests.jl`. See "Where the fields live", "Field sets" and "The
# right-hand side" in `CODE.md`.

using Random: Xoshiro
using StaticArrays: SVector

const SC = TreeGRRMHD

# Two roots per dimension, one refined, 2:1 balanced, so that the blocks do
# not all have one spacing; `N = 8` is the smallest block a `G = 3` staggered
# set allows (`N ≥ 2G + 2`).
function sc_forest(::Val{D}; N=8, periodic=true) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> periodic, D),
                    extents=ntuple(_ -> (0.0, 1.0), D))
    refine!(forest, [first(forest.leaves)])
    balance!(forest)
    return forest
end

# A value per variable that differs at every point and in every set, so that a
# segment in the wrong place or a transposed copy shows.
sc_values(nvars, tag) =
    x -> ntuple(v -> tag * 100 + v + sum(x[d] * 10.0^(d - 1) for d in eachindex(x)), nvars)

@testset "The field sets have the centerings of CODE.md's table: D=$D" for D in 1:3
    # Guards "Where the fields live": `A_j` on the edges along `j`, `𝔅ᵈ` on the
    # faces normal to `d`, `Φ̃` at the vertices, the matter and `P` at the
    # cells — and, along a direction the mesh does not have, the centering
    # that remains (in 1D, `A_y`, `A_z` at the vertices and `𝔅ʸ`, `𝔅ᶻ` at the
    # cells). A centering off by one dimension would put the curl's two
    # differences at the wrong points, which no evolution would report as
    # anything but a wrong answer.
    forest = sc_forest(Val(D))
    f = mhd_fields(Float64, forest)
    for j in 1:3
        @test f.A[j].centering == edge_centering(Val(D), j)
        @test f.B[j].centering == face_centering(Val(D), j)
        @test f.EMF[j].centering == edge_centering(Val(D), j)
        if j ≤ D
            @test edge_centering(Val(D), j) == edgecentered(D, j)
            @test face_centering(Val(D), j) == facecentered(D, j)
        else
            @test edge_centering(Val(D), j) == vertexcentered(D)
            @test face_centering(Val(D), j) == cellcentered(D)
        end
    end
    @test length(f.F) == D
    @test all(d -> f.F[d].centering == facecentered(D, d), 1:D)
    @test f.Φ.centering == vertexcentered(D)
    @test f.U.centering == cellcentered(D) && f.P.centering == cellcentered(D)
    @test (f.U.nvars, f.P.nvars, f.Φ.nvars) == (NU, NP, 1)
    @test all(s -> s.G == ntuple(_ -> 3, D), (evolved_sets(f)..., f.P, f.B...))
    @test all(s -> s.G == ntuple(_ -> 0, D), (f.F..., f.EMF..., f.X))
    # `B` is never state.
    @test length(evolved_sets(f)) == 5
    @test !any(s -> any(b -> b === s, f.B), evolved_sets(f))
    @test_throws "G ≥ 2" mhd_fields(Float64, forest; G=1)
end

@testset "AMR-1 downstream: the state vector round-trips over this package's sets: D=$D" for D in 1:3
    # Guards the multi-set state vector (AMR-1, landed in TreeAMR 0.2.1; its
    # probe retired 2026-10-14, so this is now the check `PLAN.md` names): the
    # matter set, the three edge sets of `A` and the vertex set of `Φ̃` — every
    # centering, mixed `nvars` — concatenated in `evolved_sets` order, each
    # segment bit for bit what that set's own state vector holds; `scatter!`
    # the inverse of `gather!`; `stateview` and `statearray` aliasing the one
    # vector; single-set reductions on a segment equal to those on a separate
    # vector; and a set over another forest refused. A segment in the wrong
    # place would hand the RHS one set's numbers as another's.
    forest = sc_forest(Val(D))
    f = mhd_fields(Float64, forest)
    sets = evolved_sets(f)
    for (i, s) in enumerate(sets)
        fill_by_coordinates!(AllVariables(sc_values(s.nvars, i)), s)
    end
    u = statevector(sets)
    @test length(u) == sum(statelength, sets) == statelength(sets)
    gather!(u, sets)
    for (i, s) in enumerate(sets)
        own = statevector(s)
        gather!(own, s)
        @test isequal(stateview(u, sets, i), own)
        @test staterange(sets, i) == (sum(statelength, sets[1:(i - 1)]; init=0) + 1):sum(statelength, sets[1:i])
        @test block_mapreduce(abs, +, 0.0, s, stateview(u, sets, i)) ==
              block_mapreduce(abs, +, 0.0, s, own)
    end

    # The inverse: a second family of sets over the same forest, scattered
    # from `u`, holds the same owned points.
    g = mhd_fields(Float64, forest)
    scatter!(evolved_sets(g), u)
    for (s, t) in zip(sets, evolved_sets(g))
        @test all(b -> isequal(interiorview(s, b), interiorview(t, b)), 1:nblocks(s))
    end

    # Aliasing: a write through `statearray` is a write to `u`.
    a = statearray(u, sets, SC.SEG_Φ)
    a[1] = -12345.0
    @test u[first(staterange(sets, SC.SEG_Φ))] == -12345.0

    # A set over another forest is not part of this state vector.
    other = mhd_fields(Float64, sc_forest(Val(D)))
    @test_throws ArgumentError statevector((f.U, other.A[1]))
end

@testset "∇·𝔅 vanishes to rounding on every stored cell, whatever A holds: D=$D" for D in 1:3
    # Guards the claim the whole staggering is for ("Where the fields live":
    # `∇·B = 0` per cell, on every level, whatever operator moved `A`): the
    # curl on the faces of `src/curl.jl`, summed over a cell's faces, cancels
    # every edge value. So `A` is filled with random numbers at every stored
    # point, ghosts and shared planes included — no ghost fill, no smoothness —
    # on a two-level forest, with a background field. A face reading the wrong
    # edge, or an edge difference with the wrong sign, leaves an `O(1)`
    # divergence.
    forest = sc_forest(Val(D))
    f = mhd_fields(Float64, forest)
    rng = Xoshiro(500 + D)
    for j in 1:3
        f.A[j].work .= randn(rng, size(f.A[j].work))
    end
    𝔅̄ = SVector(0.3, -0.7, 1.1)
    spacings = block_spacings(forest)
    curl_A!(f, 𝔅̄, spacings)
    worst = 0.0
    scale = maximum(j -> maximum(abs, f.B[j].work), 1:3)
    G = f.B[1].G
    for b in 1:nblocks(f.U), c in CartesianIndices(ntuple(d -> (1 - G[d]):(forest.N + G[d]), D))
        n = Tuple(c)
        h = spacings[b]
        div = sum(1:D) do d
            hi = f.B[d].work[SC.stored(SC.shift(n, d, 1), f.B[d].G)..., 1, b]
            lo = f.B[d].work[SC.stored(n, f.B[d].G)..., 1, b]
            hi - lo
        end
        worst = max(worst, abs(div))
    end
    @info "∇·𝔅 on random A, D = $D: max |Σ_d Δ𝔅ᵈ| / max |𝔅| = $(worst / scale)"
    # Measured 0 in 1D (no difference to cancel), and below 1e-15 in 2D and
    # 3D; the bound is rounding of the face values, with room.
    @test worst ≤ 1e-13 * scale
    # And in 1D, where `𝔅ˣ` has no derivative along a present direction to
    # take, it is the background, untouched: what no translation-invariant
    # `A` can produce (`CODE.md`, "Maxwell").
    D == 1 && @test all(==(𝔅̄[1]), f.B[1].work)
end

@testset "The right-hand side is pure and never mutates u" begin
    # Guards "The right-hand side" (`CODE.md`: "The right-hand side never
    # mutates `u`"; PLAN.md's purity claim): `mhd_rhs!` reads `u` and writes
    # `du` and its own scratch, so `u` comes back bit for bit, and the same
    # `u` gives the same `du` bit for bit whatever was evaluated in between —
    # which is what makes the warm starts come from the state rather than
    # from `P` (proposed in step 5). The same for the stage solve, which may
    # write only `𝔈` into `U`, and for the reset, which writes nothing where
    # the policy repaired nothing.
    for case in (em_wave_case(; roots=3), alfven_wave_case(; roots=3))
        forest = Forest(case.roots; N=10, periodic=case.periodic, extents=case.extents)
        f = mhd_fields(Float64, forest)
        p = MHDProblem(f; eos=case.eos, opts=case.opts, pol=case.pol, model=case.model,
                       𝔅̄=case.𝔅̄, limiter=:mc)
        u1 = exact_state(p, case, 0.0)
        u2 = exact_state(p, case, 0.3)
        u1c = copy(u1)
        du1, du2, du1b = similar(u1), similar(u1), similar(u1)
        mhd_rhs!(du1, u1, p, 0.0)
        @test isequal(u1, u1c)
        mhd_rhs!(du2, u2, p, 0.3)
        mhd_rhs!(du1b, u1, p, 0.0)
        @test isequal(du1, du1b)
        @test !isequal(du1, du2)
        @test all(isfinite, du1)

        U = copy(u1)
        ohm_solve!(U, u1, 0.25, p, 0.0)
        @test isequal(u1, u1c)
        # Only the 𝔈 slots of the matter segment may change.
        uU, UU = statearray(u1, p.sets, 1), statearray(U, p.sets, 1)
        @test isequal(uU[:, 1:(SC.U_E - 1), :], UU[:, 1:(SC.U_E - 1), :])
        for i in 2:5
            @test isequal(stateview(U, p.sets, i), stateview(u1, p.sets, i))
        end
        r = copy(u1)
        reset_atmosphere!(r, nothing, p, 0.0)
        @test isequal(r, u1)
    end
end

@testset "MHDProblem refuses what it cannot run" begin
    # Guards the constructor's checks: no default limiter (the case's choice),
    # a limiter or solver with no method, a non-periodic forest with no hooks,
    # and mismatched ghost widths, each an ArgumentError saying why rather
    # than a MethodError inside a kernel or a silently wrong stencil.
    case = em_wave_case(; roots=2)
    forest = Forest(case.roots; N=10, periodic=case.periodic, extents=case.extents)
    f = mhd_fields(Float64, forest)
    kw = (; eos=case.eos, opts=case.opts, pol=case.pol, model=case.model)
    @test_throws "no default limiter" MHDProblem(f; kw...)
    @test_throws "limiter must be one of" MHDProblem(f; kw..., limiter=:ppm)
    @test_throws "riemann must be one of" MHDProblem(f; kw..., limiter=:mc, riemann=:hllc)
    closed = Forest((2,); N=10, periodic=(false,), extents=((0.0, 1.0),))
    @test_throws "no boundary hooks" MHDProblem(mhd_fields(Float64, closed); kw..., limiter=:mc)
    g = mhd_fields(Float64, forest; G=2)
    mixed = SC.MHDFields(f.U, g.A, f.Φ, f.P, f.B, f.F, f.EMF, f.X, Val(1))
    @test_throws "ghost width" MHDProblem(mixed; kw..., limiter=:mc)
end

@testset "PLM reconstructs linear data exactly and falls back to first order where it must" begin
    # Guards "Reconstruction": the slopes are TreeHydro's (exact on linear
    # data, zero at an extremum under the TVD limiters, MC steeper than
    # minmod), and the positivity fallback replaces a face state's whole
    # fluid group by the cell's where the reconstruction left the EOS's
    # domain — a negative density, or an entropy outside the window — while
    # its fields are reconstructed as usual. Without the fallback the centered
    # slope at a density jump hands the EOS a negative `ρ`.
    for lim in (Val(:none), Val(:minmod), Val(:mc))
        @test SC.slope(lim, 0.25, 0.25) == 0.25
        @test SC.slope(lim, -0.5, 0.5) == 0
        @test SC.slope(lim, 0.1, 0.3) == SC.slope(lim, 0.3, 0.1)
        @test SC.slope(lim, -0.1, -0.3) == -SC.slope(lim, 0.1, 0.3)
    end
    @test SC.slope(Val(:minmod), 0.1, 0.3) == 0.1
    @test SC.slope(Val(:mc), 0.1, 0.3) == 0.2
    @test SC.slope(Val(:minmod), 0.2, -0.1) == 0 && SC.slope(Val(:mc), 0.2, -0.1) == 0

    eos = SC.case_gas(Float64, 5 // 3)
    rec(ρ, s, u, E, B) = (ρ, s, 0.5, u..., E..., B...)
    q₋₁ = rec(1.0, 1.0, (0.1, 0.0, 0.0), (0.0, 0.2, 0.0), (1.0, 0.0, 0.3))
    q₀ = rec(1e-3, 1.0, (0.2, 0.0, 0.0), (0.0, 0.4, 0.0), (1.0, 0.0, 0.6))
    q₊₁ = rec(1e-6, 1.0, (0.3, 0.0, 0.0), (0.0, 0.6, 0.0), (1.0, 0.0, 0.9))
    # The left state of the face above cell 0: the centered slope across the
    # drop, `(q₊₁ − q₋₁)/2 ≈ −1/2`, takes `ρ` to `10⁻³ − 1/4 < 0`.
    r = SC.extrapolate(Val(:none), q₋₁, q₀, q₊₁, 1)
    @test r[SC.R_RHO] < 0
    l = SC.limit_face(eos, r, q₀)
    @test l[1:(SC.R_E - 1)] == q₀[1:(SC.R_E - 1)]        # the fluid group: the cell's
    @test l[SC.R_E:end] == r[SC.R_E:end]                  # the fields: reconstructed
    # An entropy outside the window: the same.
    r2 = Base.setindex(q₀, 2.0, SC.R_S)
    @test SC.limit_face(eos, r2, q₀)[SC.R_S] == 1.0
    # Under MC the face value stays between its neighbours — here at the
    # steeper one, `q₀ + (q₊₁ − q₀)`, to the rounding of that sum (8.4e-20
    # measured against a bound of 1e-18) — and nothing falls back.
    rm = SC.extrapolate(Val(:mc), q₋₁, q₀, q₊₁, 1)
    @test 1e-6 - 1e-15 * 1e-3 ≤ rm[SC.R_RHO] ≤ 1e-3
    @test SC.limit_face(eos, rm, q₀) == rm
    # A non-finite field falls back too, without touching the fluid.
    r3 = Base.setindex(rm, NaN, SC.R_E + 1)
    l3 = SC.limit_face(eos, r3, q₀)
    @test l3[SC.R_E:end] == q₀[SC.R_E:end] && l3[1:(SC.R_E - 1)] == rm[1:(SC.R_E - 1)]
end

@testset "The reset writes back exactly the cells the policy repaired, and is idempotent" begin
    # Guards the atmosphere reset ("Floors, the atmosphere and the
    # magnetosphere"): a cell whose recovery the policy repaired is replaced
    # by the repaired state's conserved variables — `𝔈` untouched — and every
    # other cell is left bit for bit; the replaced state recovers as itself,
    # so a second reset changes nothing; and each written cell is counted. A
    # reset that wrote every cell would move the conserved totals by its
    # rounding at every stage.
    case = em_wave_case(; roots=3)
    forest = Forest(case.roots; N=10, periodic=case.periodic, extents=case.extents)
    f = mhd_fields(Float64, forest)
    p = MHDProblem(f; eos=case.eos, opts=case.opts, pol=case.pol, model=case.model,
                   limiter=:mc)
    u = exact_state(p, case, 0.0)
    uU = statearray(u, p.sets, SC.SEG_U)
    cell, blk = 4, 2
    for v in (SC.U_D, SC.U_DYE, SC.U_DS)
        uU[cell, v, blk] *= 1e-12                    # below the atmosphere
    end
    before = copy(u)
    clear_tally!(p)
    reset_atmosphere!(u, nothing, p, 0.0)
    changed = findall(i -> !isequal(u[i], before[i]), eachindex(u))
    uU = statearray(u, p.sets, SC.SEG_U)
    li = LinearIndices(uU)
    @test !isempty(changed)
    @test all(i -> i in [li[cell, v, blk] for v in 1:(SC.U_E - 1)], changed)
    @test uU[cell, SC.U_D, blk] ≈ 1e-9                # ρ_atm, at rest, flat
    @test all(v -> uU[cell, v, blk] == before[li[cell, v, blk]], SC.U_E:(SC.U_E + 2))
    @test ohm_tally(p).resets == 1
    again = copy(u)
    reset_atmosphere!(u, nothing, p, 0.0)
    @test isequal(u, again)
end

@testset "A run is the same in one chunk or in several" begin
    # Guards the driver's chunking ("Time integration": a fresh integrator
    # per chunk, reusing the last one's scratch). With the light speed the
    # same at every chunk's start, every chunk takes the same `Δt`, and where
    # the chunks hold whole numbers of steps of one dyadic `Δt` (here
    # `cfl h = 1/125` rounded down to `1/128`), the state after four chunks
    # is the state after one, bit for bit. A chunk that lost a step, or an
    # integrator that carried something over, shows.
    case = alfven_wave_case(; roots=5)
    one_ = evolve!(case, Val(1); N=10, t_end=1 // 4, limiter=:mc)
    four = evolve!(case, Val(1); N=10, t_end=1 // 4, chunk=1 // 16, limiter=:mc)
    @test length(four.chunks) == 4
    @test one_.chunks[1].dt == 1 / 128
    @test four.nsteps == one_.nsteps == 32
    @test all(c -> c.dt == 1 / 128 && c.nsteps == 8, four.chunks)
    @test isequal(four.u, one_.u)
end
