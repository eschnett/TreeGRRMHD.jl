# Rung 2b: coarse-fine faces and regridding ("Validation ladder" 2 and
# "Refinement" in `CODE.md`; step 8 of `PLAN.md`). The scheme of steps 5–7 on
# meshes with more than one level:
#
#   * the bounded conservative prolongation keeps a surface in range (AMR-4,
#     moved here from `upstream_behaviour_tests.jl`);
#   * on a static two-level mesh every total is conserved to rounding with
#     `restrict_interfaces!` on the fluxes, and leaks without it;
#   * the magnetic flux through a coarse face changes as the sum through its
#     fine faces with `restrict_interfaces!` on the EMFs (the CT correction),
#     and does not without it;
#   * a shock tube tracked through regrids conserves every total, and `∇·𝔅`
#     stays at rounding on every level and after every regrid.
#
# Uses `multid_tests.jl`'s `md_run` (which asserts `∇·𝔅 ≤ 1e−12` after every
# step) and `md_conserved`, so it is included after it.

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

# A stellar surface: a drop over thirteen orders of magnitude.
rf_surface(x) = (x[1] < 0.3 ? 1.0 : 1.0e-13,)

@testset "The bounded conservative prolongation refines a surface without new extrema: D=$D" for D in 1:2
    # Guards what `CODE.md`, "Refinement", builds on (AMR-4, landed in
    # TreeAMR 0.2.1 as `Operators(; …, bounded = true)`, the MC-bounded
    # width-3 conservative prolongation, which the matter's ghost fill and
    # regrid use since step 8): a surface refined everywhere stays positive
    # and within its range and keeps its total, where the unbounded operator
    # overshoots it — so an unbounded operator, or a release that dropped the
    # bound, would floor cells at a stellar surface. (Moved here from
    # `upstream_behaviour_tests.jl` in step 8.)
    bounded = Operators(; prolongation=3, restriction=2, family=Conservative, bounded=true)
    forest = rf_forest(Val(D))
    fs = FieldSet{Float64}(forest, 1; G=1, centering=cellcentered(D))
    fill_by_coordinates!(AllVariables(rf_surface), fs)
    before = total_mass(fs)
    regrid!(forest, fs => GhostSchedule(fs, bounded); flags=fill(Refine, nleaves(forest)))
    fill_ghosts!(fs, GhostSchedule(fs, bounded))
    lo, hi = extrema(fs.work)
    @test 0 < lo && hi <= 1 + eps()
    @test isapprox(total_mass(fs), before; rtol=1e-13)

    # The control: unbounded, the same refinement overshoots.
    unbounded = Operators(; prolongation=3, restriction=2, family=Conservative)
    forest = rf_forest(Val(D))
    fs = FieldSet{Float64}(forest, 1; G=1, centering=cellcentered(D))
    fill_by_coordinates!(AllVariables(rf_surface), fs)
    regrid!(forest, fs => GhostSchedule(fs, unbounded); flags=fill(Refine, nleaves(forest)))
    lo, hi = extrema(fs.work)
    @test lo < 0 || hi > 1
end

# The largest drift of the conserved totals `D`, `S_i`, `τ`, `D Yₑ`, `D s` over
# a run, in ulp of each total's scale per step (`md_conserved`'s measure,
# without its assertion).
function rf_drift(run)
    worst = 0.0
    for v in (RF.U_D, RF.U_S, RF.U_S + 1, RF.U_S + 2, RF.U_TAU, RF.U_DYE, RF.U_DS)
        run.scales[v] > 0 || continue
        worst = max(worst, abs(run.drift[v]) / (eps() * run.scales[v] * run.r.nsteps))
    end
    return worst
end

@testset "A static two-level mesh conserves every total to rounding with the flux fixup, and leaks without it: D=$D" for D in 1:2
    # Guards the flux correction ("Conservation at coarse-fine faces" in
    # `CODE.md`): with `restrict_interfaces!` on `F_d` a coarse face's flux is
    # the average of its fine faces', and every total — `D`, `S_i`, `τ`,
    # `D Yₑ`, `D s` — changes only by rounding; without it the coarse and the
    # fine side disagree at every coarse-fine face, and mass leaks at the
    # truncation error. The Alfvén wave of rung 1a over a quarter period on
    # 64 cells with the middle half refined (1D), and along the diagonal on
    # 2 × 2 roots of 8² with one refined (2D). Measured (1D): 0.008 ulp of a
    # total's scale a step with the fixup, 3·10⁸ to 1.5·10¹⁰ without.
    c1 = alfven_wave_case(Float64; roots=D == 1 ? 8 : 1)
    case = D == 1 ? c1 : oblique_case(c1; N=8, roots=2)
    t_end = alfven_period(c1) / (D == 1 ? 4 : 8)
    refine = D == 1 ? rf_refine_middle! : rf_refine_first!
    runs = map((true, false)) do fixup
        md_run(case, Val(D); N=8, t_end=t_end, limiter=:mc, refine=refine, fixup=fixup)
    end
    with, without = rf_drift(runs[1]), rf_drift(runs[2])
    @info "Static two-level mesh, D = $D: totals drift $(round(with, sigdigits=3)) ulp of " *
          "their scale a step with the fixup, $(round(without, sigdigits=3)) without " *
          "($(runs[1].r.nsteps) steps, $(runs[1].r.nblocks) blocks)"
    @test runs[1].r.nblocks == (D == 1 ? 12 : 7)
    @test with ≤ 8
    @test without ≥ 1e6
    @test runs[1].tally.policy == 0 && runs[1].tally.resets == 0
end

# --- the magnetic flux at coarse-fine faces ---------------------------------------

# The EMF sets' values on their closed ranges, keyed by `(j, level, position)`
# with the position in half-spacings of that level from the domain's low
# corner (even along a vertex-like dimension, odd along a cell-like one); from
# block `only` alone, if given.
function rf_emf_table(p; only=nothing)
    f = p.f
    forest = f.U.forest
    D = length(forest.roots)
    N = forest.N
    tab = Dict{Tuple{Int,Int,NTuple{D,Int}},Float64}()
    for j in 1:3
        s = f.EMF[j]
        st = staggers(s)
        for b in (only === nothing ? (1:nblocks(s)) : (only:only))
            k = blockkey(s, b)
            h = spacing(forest, k)
            o = block_origin(forest, k)
            g0 = ntuple(e -> round(Int, (o[e] - forest.extents[e][1]) / h), D)
            for I in CartesianIndices(ntuple(e -> N + st[e], D))
                n = Tuple(I)
                pos = ntuple(e -> 2 * (g0[e] + n[e] - 1) + (st[e] == 1 ? 0 : 1), D)
                tab[(j, level(k), pos)] = s.work[(n .+ s.G)..., 1, b]
            end
        end
    end
    return tab
end

# The rate of change of the magnetic flux through the face normal to `d`
# centred at `c` on level `l` (spacing `h`), from the EMFs around it: with
# `(d, a, b)` cyclic, `∂ₜ𝔅ᵈ = −(∂_a EMF_b − ∂_b EMF_a)` (the gauge terms' curl
# vanishes), times the face's area; along a missing direction no difference
# and unit length.
function rf_flux_rate(tab, d, l, c::NTuple{D,Int}, h) where {D}
    a, b = mod1(d + 1, 3), mod1(d + 2, 3)
    shift(c, e, k) = Base.setindex(c, c[e] + k, e)
    La, Lb = a ≤ D ? h : 1.0, b ≤ D ? h : 1.0
    ta = a ≤ D ? (tab[(b, l, shift(c, a, 1))] - tab[(b, l, shift(c, a, -1))]) * Lb : 0.0
    tb = b ≤ D ? (tab[(a, l, shift(c, b, 1))] - tab[(a, l, shift(c, b, -1))]) * La : 0.0
    return -(ta - tb)
end

# Over every face of a level-0 block that lies on a coarse-fine face — each
# boundary plane of the block, where the level-1 table has the fine faces
# covering it — the largest difference between the flux rate from that
# block's own EMFs and the sum of the rates through the `2^(D−1)` fine faces
# from the fine blocks' EMFs, and the largest rate. The block's own: an edge
# on a coarse-fine face is also an edge of a coarse block that touches the
# fine region only along that edge, which `restrict_interfaces!` leaves alone
# (TreeAMR's face directions only), so the coarse value is the one the block
# whose face it is computed and was corrected.
function rf_ct_mismatch(p)
    fine = rf_emf_table(p)
    f = p.f
    forest = f.U.forest
    D = length(forest.roots)
    N = forest.N
    h0 = spacing(forest, 0)
    worst, scale, nfaces = 0.0, 0.0, 0
    for blk in 1:nblocks(f.U)
        k = blockkey(f.U, blk)
        level(k) == 0 || continue
        own = rf_emf_table(p; only=blk)
        o = block_origin(forest, k)
        g0 = ntuple(e -> round(Int, (o[e] - forest.extents[e][1]) / h0), D)
        for d in 1:D, side in (1, N + 1), I in CartesianIndices(ntuple(e -> e == d ? 1 : N, D))
            n = ntuple(e -> e == d ? side : I[e], D)
            c = ntuple(e -> 2 * (g0[e] + n[e] - 1) + (e == d ? 0 : 1), D)
            offsets = Iterators.product(ntuple(e -> e == d ? (0,) : (-1, 1), D)...)
            # Not on a coarse-fine face where the fine table has no faces there.
            rates = try
                [rf_flux_rate(fine, d, 1, ntuple(e -> 2 * c[e] + o2[e], D), h0 / 2)
                 for o2 in offsets]
            catch err
                err isa KeyError || rethrow()
                continue
            end
            coarse = rf_flux_rate(own, d, 0, c, h0)
            fsum = sum(rates)
            nfaces += 1
            worst = max(worst, abs(coarse - fsum))
            scale = max(scale, abs(coarse), abs(fsum))
        end
    end
    return (; worst, scale, nfaces)
end

@testset "The magnetic flux through a coarse face changes as the sum through its fine faces with the EMF correction: D=$D" for D in 2:3
    # Guards the constrained-transport correction ("Conservation at
    # coarse-fine faces" in `CODE.md`): with `restrict_interfaces!` on
    # `EMF_d` a coarse edge on a coarse-fine face carries the average of its
    # fine edges' EMFs, so the rate `−∮ EMF · dl` at which the magnetic flux
    # through a coarse face changes is the sum of the rates through the fine
    # faces that cover it — to rounding (the fine sum telescopes in floating
    # point); without it the two sides' EMFs are two different
    # discretizations and disagree at the truncation error. After one
    # right-hand side, on every face of every coarse block that lies on a
    # coarse-fine face: Orszag–Tang's initial data on 2 × 2 roots of 8² with
    # one refined (2D; `E = −v × B` varies everywhere), and the spherical
    # blast in its closed box after two steps on 2 × 2 × 2 roots of 8³ with
    # one refined (3D, the Dirichlet hooks across the levels too). Measured:
    # 1.7e−18 of a largest rate of 0.016 (2D) and 0.0085 (3D) with the
    # correction, 0.025 of 0.031 and 0.0039 of 0.0092 without.
    case = D == 2 ? orszag_tang_case(Float64; roots=2) :
           cylindrical_explosion_case(Float64, Val(3); L=2, roots=2, spherical=true)
    t_end = D == 2 ? 1 // 1000 : 1 // 32
    run = md_run(case, Val(D); N=8, t_end=t_end, limiter=D == 2 ? :mc : :minmod,
                 refine=rf_refine_first!)
    p = run.r.p
    du = similar(run.r.u)
    with = (mhd_rhs!(du, run.r.u, p, run.r.t); rf_ct_mismatch(p))
    q = RF.rebuild(p; fixup=false)
    without = (mhd_rhs!(du, run.r.u, q, run.r.t); rf_ct_mismatch(q))
    @info "Magnetic flux at coarse-fine faces, D = $D: $(with.nfaces) coarse faces; " *
          "mismatch $(with.worst) of a rate $(with.scale) with the EMF correction, " *
          "$(without.worst) of $(without.scale) without"
    @test with.nfaces == (D == 2 ? 16 : 192) && without.nfaces == with.nfaces
    @test with.worst ≤ 1e-14 * with.scale
    @test without.worst ≥ 1e-2 * without.scale
end

@testset "A shock tube tracked through regrids conserves every total to rounding, and leaks without the fixup" begin
    # Guards the regrid over every field set (`regrid_state!`: `U` with the
    # bounded conservative width-3 operators, `A` and `Φ̃` PointValue width 4,
    # the rest resized) and the flux correction across moving coarse-fine
    # faces: Balsara 1 on its box made periodic (as `shocktube_tests.jl`
    # runs it, so that no flux leaves), 100 coarse cells and two levels more
    # where Löhner on `ρ`, `|B|`, `|E|` fires (`refine_tol = 0.08`,
    # `coarsen_tol = 0.02`, a 6-cell buffer), regridded every 1/80 to
    # `t = 0.4`. The bounded regrid transfer is exactly conservative and the
    # post-regrid reset writes nothing, so every total stays constant to
    # rounding through the regrids: measured 0.013 ulp of a total's scale a
    # step (8 is the bound), and 3·10⁶ to 1.4·10⁷ without the fixup.
    c = balsara_case(Float64, 1; roots=10)
    case = MHDCase(Float64, Val(1); fields=c.fields, potential=c.potential, eos=c.eos,
                   pol=c.pol, model=c.model, 𝔅̄=c.𝔅̄, periodic=true, extents=c.extents,
                   roots=c.roots)
    runs = map((true, false)) do fixup
        r = evolve!(case, Val(1); N=10, t_end=2 // 5, chunk=1 // 80, limiter=:mc, maxlevel=2,
                    refine_tol=0.08, coarsen_tol=0.02, buffer=6, fixup=fixup)
        scales = conserved_scales(r.p, r.u)
        drift = conserved_totals(r.p, r.u) .- r.totals0
        worst = maximum(v -> scales[v] > 0 ? abs(drift[v]) / (eps() * scales[v] * r.nsteps) : 0.0,
                        (RF.U_D, RF.U_S, RF.U_TAU, RF.U_DYE, RF.U_DS))
        (; r, worst)
    end
    r = runs[1].r
    @info "Tracked Balsara 1: $(r.nregrids) regrids, $(r.passes) adaptation passes, " *
          "$(r.nsteps) steps, $(r.nblocks) blocks at the end; drift $(runs[1].worst) ulp a step " *
          "with the fixup, $(runs[2].worst) without"
    @test r.nregrids ≥ 5 && r.nblocks > 10
    @test runs[1].worst ≤ 8
    @test runs[2].worst ≥ 1e4
    @test all(c -> c.tally.policy == 0 && c.tally.resets == 0 && c.tally.step_resets == 0,
              r.chunks)
end

@testset "∇·𝔅 stays at rounding on both levels and after every regrid, and the totals through them" begin
    # Guards the claim of rung 2 ("Where the fields live" in `CODE.md`): `𝔅`
    # is the curl of `A` on every level, whatever moved `A` — the
    # prolongation of a fresh fine block, the restriction into a coarsened
    # one — so `max |∇·𝔅| h / max |𝔅| ≤ 1e−12` after every step (`md_run`)
    # and right after every regrid. GRaM-X's rotor (closed box, the
    # Dirichlet hooks on both levels) on 4 × 4 roots of 8² with one level
    # more where Löhner fires, regridded every 1/40 to `t = 1/10`: the
    # adaptation makes 28 blocks, the regrids 52 and 58 as the torsional
    # waves leave the rotor. And the totals through the regrids, constant to
    # rounding but for the precursors at the faces (`md_conserved`;
    # measured 0.036 ulp of a total's scale a step).
    divs = Float64[]
    run = md_run(magnetic_rotor_case(Float64; roots=4), Val(2); N=8, t_end=1 // 10,
                 chunk=1 // 40, limiter=:mc, maxlevel=1, refine_tol=0.08, coarsen_tol=0.02,
                 buffer=4, after_regrid=(p, u, t) -> push!(divs, max_div_B(p, u, t)))
    worst = md_conserved(run)
    @info "Adaptive rotor: $(run.r.passes) adaptation passes, $(run.r.nregrids) regrids, " *
          "blocks $(map(c -> c.nblocks, run.r.chunks)); ∇·𝔅 at most $(run.divb) after a " *
          "step and $(maximum(divs)) after a regrid; drift $(round(worst, sigdigits=3)) ulp a step"
    @test run.r.nregrids ≥ 2 && length(divs) == run.r.nregrids + 1
    @test maximum(divs) ≤ MD_DIVB
    @test run.r.chunks[1].nblocks < run.r.chunks[end].nblocks < 4^2 * 4
    @test run.tally.policy == 0 && run.tally.resets == 0
end

# `Σ hᴰ 𝔅ᶻ` over the leaves of a two-dimensional state: the magnetic flux
# through the plane, which `∮ A · dl` around the periodic box fixes.
function rf_total_Bz(p, u, t)
    update_fields!(p, u, t)
    B = p.f.B[3]
    return mesh_mapreduce(identity, +, 0.0, B; vars=1,
                          weight=k -> spacing(Float64, B.forest, k)^2),
           mesh_mapreduce(abs, +, 0.0, B; vars=1, weight=k -> spacing(Float64, B.forest, k)^2)
end

@testset "The magnetic flux through the plane drifts at the potentials' interpolation error across levels (AMR-9)" begin
    # Records what the EMF correction does not give (`CODE.md`,
    # "Conservation at coarse-fine faces"; AMR-9 in `PLAN.md`): on a periodic
    # 2D box `Σ hᴰ 𝔅ᶻ` is `∮ A · dl` around the box, constant to rounding on
    # a uniform mesh, but across a coarse-fine face it telescopes only if the
    # coarse `A_y` on the face is the mean of the fine ones beside it — and
    # PointValue operators interpolate along an edge's own direction, where
    # a conservative transfer would average. The oblique Alfvén wave (whose
    # field has a `z` component) on 2 × 2 roots of 8², one refined, an eighth
    # of a period: measured 7.1e−4 of the flux's scale 1.25 with the
    # correction and 7.2e−4 without at width 4, 1.4e−3 at width 2, and
    # 1.5e−16 on the uniform mesh — the operators', not the EMFs'. Bounded
    # here, and to be asserted to rounding when AMR-9 lands.
    c1 = alfven_wave_case(Float64; roots=1)
    case = oblique_case(c1; N=8, roots=2)
    drifts = map((nothing, rf_refine_first!)) do refine
        r = evolve!(case, Val(2); N=8, t_end=alfven_period(c1) / 8, limiter=:mc, refine=refine)
        u0 = exact_state(r.p, case, 0.0)
        low_walls!(r.p, u0, 0.0)
        b0, _ = rf_total_Bz(r.p, u0, 0.0)
        b1, scale = rf_total_Bz(r.p, r.u, r.t)
        abs(b1 - b0) / scale
    end
    @info "Flux of 𝔅ᶻ through the plane, oblique Alfvén wave: drift $(drifts[1]) of its " *
          "scale on the uniform mesh, $(drifts[2]) on the two-level mesh"
    @test drifts[1] ≤ 1e-14
    @test drifts[2] ≤ 1e-2
end

