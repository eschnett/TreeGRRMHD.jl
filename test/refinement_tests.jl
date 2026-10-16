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
#     stays at rounding on every level and after every regrid;
#   * one fill applies the complete outer boundary condition (TreeAMR's
#     relative hook and symmetric walls, AMR-7 and AMR-8), the regrid's
#     included, and the state holds the prescribed low walls;
#   * a closed box refined symmetrically keeps TreeAMR's half-open
#     asymmetry at coarse-fine faces (recorded).
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

const RF_PV = Operators(; prolongation=4, restriction=4)   # the potentials before AMR-9

@testset "The magnetic flux through the plane is constant to rounding across levels and through a regrid (AMR-9)" begin
    # Guards the potentials' conservative transfer (AMR-9, TreeAMR 0.2.2;
    # `CODE.md`, "Conservation at coarse-fine faces"): on a periodic 2D box
    # `Σ hᴰ 𝔅ᶻ` is `∮ A · dl` around the box, and across a coarse-fine face
    # it telescopes only if the coarse `A_y` on the face is the mean of the
    # fine ones beside it — which the conservative family along an edge's
    # own direction makes it, after every fill and every regrid, and the
    # PointValue family, which interpolates there, does not. The oblique
    # Alfvén wave (whose field has a `z` component) on 2 × 2 roots of 8², one
    # refined, an eighth of a period, then a regrid that coarsens the refined
    # root and refines the opposite one. Measured: on the uniform mesh 5.1e−17
    # of the flux's scale; two-level 5.1e−17 after the run and the same after
    # the regrid; with the PointValue operators of step 8 5.8e−4 (and the
    # regrid moves it further).
    c1 = alfven_wave_case(Float64; roots=1)
    case = oblique_case(c1; N=8, roots=2)
    function drift(refine, ops_A)
        r = evolve!(case, Val(2); N=8, t_end=alfven_period(c1) / 8, limiter=:mc, refine=refine,
                    ops_A=ops_A)
        b0, _ = rf_total_Bz(r.p, exact_state(r.p, case, 0.0), 0.0)
        b1, scale = rf_total_Bz(r.p, r.u, r.t)
        refine === nothing && return (abs(b1 - b0) / scale, NaN)
        p = r.p
        flags = [level(blockkey(p.f.U, b)) == 1 ? Coarsen :
                 blockkey(p.f.U, b) == last(p.f.U.forest.leaves) ? Refine : Keep
                 for b in 1:nblocks(p.f.U)]
        g = regrid_state!(p, r.u, r.t; flags=flags)
        @test g.moved && max_div_B(g.p, g.u, r.t) ≤ MD_DIVB
        b2, _ = rf_total_Bz(g.p, g.u, r.t)
        return (abs(b1 - b0) / scale, abs(b2 - b0) / scale)
    end
    uniform = drift(nothing, RF.default_ops_A())
    cons = drift(rf_refine_first!, RF.default_ops_A())
    pv = drift(rf_refine_first!, RF_PV)
    @info "Flux of 𝔅ᶻ through the plane, oblique Alfvén wave: drift $(uniform[1]) of its " *
          "scale on the uniform mesh; two-level $(cons[1]), after a regrid $(cons[2]) " *
          "(PointValue potentials: $(pv[1]), $(pv[2]))"
    @test uniform[1] ≤ 1e-14
    @test cons[1] ≤ 1e-14 && cons[2] ≤ 1e-14
    @test pv[1] ≥ 1e-4 && pv[2] ≥ 1e-4
end

# The face sets' `𝔅` on their closed ranges, keyed by `(d, level, position)`
# with the position in half-spacings of that level from the domain's low
# corner; from block `only` alone, if given.
function rf_B_table(p; only=nothing)
    f = p.f
    forest = f.U.forest
    D = length(forest.roots)
    N = forest.N
    tab = Dict{Tuple{Int,Int,NTuple{D,Int}},Float64}()
    for d in 1:D
        s = f.B[d]
        st = staggers(s)
        for b in (only === nothing ? (1:nblocks(s)) : (only:only))
            k = blockkey(s, b)
            h = spacing(forest, k)
            o = block_origin(forest, k)
            g0 = ntuple(e -> round(Int, (o[e] - forest.extents[e][1]) / h), D)
            for I in CartesianIndices(ntuple(e -> N + st[e], D))
                n = Tuple(I)
                pos = ntuple(e -> 2 * (g0[e] + n[e] - 1) + (st[e] == 1 ? 0 : 1), D)
                tab[(d, level(k), pos)] = s.work[(n .+ s.G)..., 1, b]
            end
        end
    end
    return tab
end

# Over every face of a level-0 block on a coarse-fine face: the magnetic flux
# through it from that block's own (filled) `𝔅`, `𝔅ᵈ H^{D−1}`, against the
# sum through the `2^{D−1}` fine faces covering it, `Σ 𝔅ᵈ h^{D−1}` — the
# state, after a fill.
function rf_flux_state_mismatch(p, u, t)
    update_fields!(p, u, t)
    fine = rf_B_table(p)
    f = p.f
    forest = f.U.forest
    D = length(forest.roots)
    N = forest.N
    H = spacing(forest, 0)
    worst, scale, nfaces = 0.0, 0.0, 0
    for blk in 1:nblocks(f.U)
        k = blockkey(f.U, blk)
        level(k) == 0 || continue
        own = rf_B_table(p; only=blk)
        o = block_origin(forest, k)
        g0 = ntuple(e -> round(Int, (o[e] - forest.extents[e][1]) / H), D)
        for d in 1:D, side in (1, N + 1), I in CartesianIndices(ntuple(e -> e == d ? 1 : N, D))
            n = ntuple(e -> e == d ? side : I[e], D)
            c = ntuple(e -> 2 * (g0[e] + n[e] - 1) + (e == d ? 0 : 1), D)
            offsets = Iterators.product(ntuple(e -> e == d ? (0,) : (-1, 1), D)...)
            keys = [(d, 1, ntuple(e -> 2 * c[e] + o2[e], D)) for o2 in offsets]
            all(kk -> haskey(fine, kk), keys) || continue
            coarse = own[(d, 0, c)] * H^(D - 1)
            fsum = sum(kk -> fine[kk], keys) * (H / 2)^(D - 1)
            nfaces += 1
            worst = max(worst, abs(coarse - fsum))
            scale = max(scale, abs(coarse))
        end
    end
    return (; worst, scale, nfaces)
end

@testset "The magnetic flux through a coarse face is the sum through its fine faces, as a state, after steps and a regrid (AMR-9)" begin
    # Guards PLAN.md's step-8 claim as a state (AMR-9): in 3D the flux through
    # a face on a coarse-fine face is the circulation of the two in-plane
    # potentials around it, and the sum through the fine faces covering it is
    # the circulation around the same perimeter in fine edges — equal when
    # every coarse edge's line integral is the sum of its two fine edges',
    # which the conservative family along an edge's own direction keeps
    # through every fill and regrid. That is the transfer's property, not
    # the EMFs': the EMF correction makes the *rate* agree (above), and
    # without it the state still agrees after every fill, the non-owner's
    # edges being derived from the owner's. The spherical blast on 2 × 2 × 2
    # roots of 8³, one refined, after two steps, then a regrid refining the
    # opposite root (every face of a coarse block on a coarse-fine face, 192
    # and then 384). Measured, in its periodic box: 8.7e−19 of a largest
    # flux of 6.3e−3 with the correction, 1.7e−18 without, the same after the
    # regrid; with the PointValue operators of step 8, 1.4e−5. In the closed
    # box the 147 faces away from the walls agree to 8.7e−19, but the 45
    # whose perimeter lies on an outer face miss by 4.6e−8 (7e−6 of the
    # scale; 1.7e−7 after the regrid): there each level's relative hook
    # writes its own wall edges from its own nearest evolved point, so a
    # coarse wall edge is not the sum of the fine ones (recorded, `CODE.md`;
    # AMR-11 in `PLAN.md`).
    function measure(periodic; kw...)
        case = cylindrical_explosion_case(Float64, Val(3); L=2, roots=2, spherical=true,
                                          periodic=periodic)
        run = md_run(case, Val(3); N=8, t_end=1 // 32, limiter=:minmod, refine=rf_refine_first!,
                     kw...)
        p, u, t = run.r.p, run.r.u, run.r.t
        before = rf_flux_state_mismatch(p, u, t)
        flags = [blockkey(p.f.U, b) == last(p.f.U.forest.leaves) ? Refine : Keep
                 for b in 1:nblocks(p.f.U)]
        g = regrid_state!(p, u, t; flags=flags)
        @test g.moved && max_div_B(g.p, g.u, t) ≤ MD_DIVB
        after = rf_flux_state_mismatch(g.p, g.u, t)
        return (; before, after)
    end
    with = measure(true)
    without = measure(true; fixup=false)
    pv = measure(true; ops_A=RF_PV)
    closed = measure(false)
    rel(m) = m.worst / m.scale
    @info "Magnetic flux as a state, 3D: mismatch of a coarse face against its fine faces " *
          "$(rel(with.before)) of the largest flux ($(with.before.nfaces) faces), after a regrid " *
          "$(rel(with.after)) ($(with.after.nfaces)); without the EMF correction " *
          "$(rel(without.before)), $(rel(without.after)); PointValue potentials " *
          "$(rel(pv.before)), $(rel(pv.after)); closed box $(rel(closed.before)), " *
          "$(rel(closed.after))"
    @test with.before.nfaces == 192 && with.after.nfaces == 384
    @test rel(with.before) ≤ 1e-14 && rel(with.after) ≤ 1e-14
    @test rel(without.before) ≤ 1e-14 && rel(without.after) ≤ 1e-14
    @test rel(pv.before) ≥ 1e-4 && rel(pv.after) ≥ 1e-4
    @test 1e-10 ≤ rel(closed.before) ≤ 1e-4
end

# The explosion in its closed box, `[−2, 2]²` on 4 × 4 roots of 8², and the
# root on the low `y` wall between two coarse roots refined.
const RF_BOX = cylindrical_explosion_case(Float64; L=2, roots=4)
rf_refine_wall!(forest) =
    (refine!(forest, [k for k in forest.leaves if block_origin(forest, k) == (-1.0, -2.0)]);
     balance!(forest); forest)

@testset "One fill applies the complete outer boundary condition, and regrid! with the hooks transfers it" begin
    # Guards Erik's decision of 2026-10-15 ("an outer boundary condition
    # needs to be applied always") as TreeAMR 0.2.2 delivers it: a
    # potential's outer points — its ghosts beyond an outer face, the high
    # wall plane, and under symmetric walls (AMR-8) the owned low wall — are
    # the nearest evolved point plus the case's difference, the wall's row
    # next to another block included, and TreeAMR's relative hook (AMR-7)
    # writes all of them in one fill, each region after the stage that fills
    # what it reads. Until step 8 took up 0.2.2 this needed this package's
    # `outer_points!` after the hook and, on a multilevel mesh, a hookless
    # fill and `outer_points!` once more; a regrid filled completely first
    # and handed `regrid!` no hook for the potentials. On the closed box with
    # a refined wall root, to `t = 1/4`: (1) after one fill a second changes
    # nothing, bit for bit, and one fill of NaN-prefilled working arrays
    # writes the same — every stored point, from the state alone; (2) a
    # hookless fill leaves it bit for bit; (3) `regrid_state!`, whose
    # `regrid!` fills with the hooks, transfers exactly what step 8's first
    # arrangement (a complete fill, then `regrid!` with no hook for the
    # potentials) transfers, bit for bit — a regrid refining the next wall
    # root. Before the complete condition (step 8's first form, the hook
    # alone) the transfer was 1.8e−10 off (of 1.2e−2) in the potentials.
    flags(p) = [block_origin(p.f.U.forest, blockkey(p.f.U, b)) == (0.0, -2.0) ? Refine : Keep
                for b in 1:nblocks(p.f.U)]
    run() = evolve!(RF_BOX, Val(2); N=8, t_end=1 // 4, limiter=:minmod, refine=rf_refine_wall!)
    r = run()
    p, u, t = r.p, r.u, r.t
    scatter!(p.sets, u)
    fixed, fromnan, kept = true, true, true
    for i in RF.SEG_A:RF.SEG_Φ
        s = p.sets[i]
        RF.fill_set!(p, i, t)
        full = copy(s.work)
        RF.fill_set!(p, i, t)
        fixed &= isequal(s.work, full)
        fill!(s.work, NaN)
        scatter!(s, stateview(u, p.sets, i))
        RF.fill_set!(p, i, t)
        fromnan &= isequal(s.work, full)
        fill_ghosts!(s, p.schedules[i]; boundary=nothing)
        kept &= isequal(s.work, full)
    end
    @test maxlevel(p.f.U.forest) == 1 && fixed && fromnan && kept
    # (3) through `regrid_state!`, and through step 8's first arrangement,
    # from the same state.
    new = regrid_state!(p, u, t; flags=flags(p))
    r2 = run()
    q = r2.p
    scatter!(q.sets, r2.u)
    for i in eachindex(q.sets)
        RF.fill_set!(q, i, t)
    end
    old = (ntuple(i -> q.sets[i] => (q.schedules[i],
                                     i == RF.SEG_U ? RF.boundary_at(q.boundary[i], t) : nothing),
                  5)..., RF.regrid_pairs(q, t)[6:end]...)
    regrid!(first(q.sets).forest, old; flags=flags(q))
    q = RF.rebuild(q)
    uold = statevector(q.sets)
    gather!(uold, q.sets)
    clear_tally!(q)
    reset_step!(uold, nothing, q, t)
    @info "Complete boundary condition: one fill is its fixed point $fixed (from NaN $fromnan); " *
          "a regrid with the hooks is the complete fill's $(isequal(new.u, uold)); blocks " *
          "$(r.nblocks) → $(nblocks(new.p.f.U))"
    @test new.moved && nblocks(new.p.f.U) == r.nblocks + 3
    @test isequal(new.u, uold)
end

# The owned low-wall points of the potentials of a state: per set, the state's
# value and the value a complete fill writes there, and `du`'s.
function rf_walls(p, u, du, t)
    scatter!(p.sets, u)
    D = length(p.f.U.forest.roots)
    N = p.f.U.forest.N
    state, filled, rate = Float64[], Float64[], Float64[]
    for i in RF.SEG_A:RF.SEG_Φ
        s = p.sets[i]
        RF.fill_set!(p, i, t)
        lw = RF.low_walls(s.forest, Float64)
        a, d = statearray(u, p.sets, i), statearray(du, p.sets, i)
        for b in 1:nblocks(s), I in CartesianIndices(ntuple(_ -> N, D))
            n = Tuple(I)
            RF.on_low_wall(n, lw[b], Val(staggers(s))) || continue
            push!(state, a[n..., 1, b])
            push!(filled, s.work[(n .+ s.G)..., 1, b])
            push!(rate, d[n..., 1, b])
        end
    end
    return (; state, filled, rate)
end

@testset "The potentials' low walls are prescribed: the state holds them after every step, and nothing reads them unfilled: $(refined ? "refined" : "uniform")" for refined in (false, true)
    # Guards the switch from `low_walls!` to TreeAMR's symmetric walls
    # (AMR-8; decided in step 8, "Boundaries" in `CODE.md`). The low wall
    # plane of a vertex-like dimension at a low outer face is owned, so it is
    # in the state vector, but the fill writes it — into the working array,
    # not into `u`. So the right-hand side writes zero there (the wall is
    # prescribed, not evolved; a branch, not a mask), and the reset, as
    # stage and step limiter, copies the filled walls into `u`
    # (`walls_to_state!`). Claims, on the closed box (uniform, and with the
    # wall root refined), after every step to `t = 1/4`: `u`'s wall entries
    # are what a complete fill prescribes there, bit for bit, and `du` is
    # exactly zero there; and one more step from a state whose wall entries
    # are perturbed by 1 gives the same state as from the true one, bit for
    # bit — they are never read without a fill.
    nwalls, held, still = 0, true, true
    function monitor(p, u, t)
        du = zero(u)
        mhd_rhs!(du, u, p, t)
        w = rf_walls(p, u, du, t)
        nwalls = length(w.state)
        held &= isequal(w.state, w.filled)
        still &= all(iszero, w.rate)
    end
    r = evolve!(RF_BOX, Val(2); N=8, t_end=1 // 4, limiter=:minmod,
                refine=refined ? rf_refine_wall! : nothing, after_step=monitor)
    p, u, t = r.p, r.u, r.t
    dt = r.chunks[end].dt
    step(u0) = (integ = RF.mhd_integrator(p, u0, t, t + dt, 1); RF.IRK.solve!(integ); integ.u)
    w = rf_walls(p, u, zero(u), t)
    bumped = copy(u)
    for i in RF.SEG_A:RF.SEG_Φ
        s = p.sets[i]
        lw = RF.low_walls(s.forest, Float64)
        a = statearray(bumped, p.sets, i)
        for b in 1:nblocks(s), I in CartesianIndices(ntuple(_ -> 8, 2))
            RF.on_low_wall(Tuple(I), lw[b], Val(staggers(s))) && (a[I, 1, b] += 1)
        end
    end
    a, b = step(copy(u)), step(bumped)
    @info "Low walls, $(refined ? "refined" : "uniform") closed box: $nwalls wall points, held " *
          "in the state after every step $held, du zero there $still; a perturbed wall " *
          "changes the next state by $(maximum(abs.(a .- b)))"
    # 190 on the uniform mesh: `A_x` and `A_y` 8 on each of the four roots
    # along their wall, `A_z` and `Φ̃` 8 + 8 − 1 on the corner root and 8 on
    # each of the six others; 24 more with the wall root refined.
    @test nwalls == (refined ? 214 : 190)
    @test r.nsteps ≥ 10 && held && still
    @test count(!=(0.0), bumped .- u) == nwalls
    @test isequal(a, b)
end

# The largest violation of the mirror symmetry along `axis` of a state on a
# refined mesh symmetric about the centre of a box centred at the origin, per
# evolved set, relative to each variable's largest magnitude: every set's
# ghost-filled working array on each block's closed range against the mirror
# block's at the mirrored point (a cell `n` to `N + 1 − n`, a vertex-like
# point to `N + 2 − n`), with `md_mirror_equal`'s parities.
function rf_mirror_error(p, u, t, axis, c)
    update_primitives!(p, u, t)
    pol(j) = j == axis ? -1 : 1
    parity(i, v) = i == 1 ? (v in 2:4 ? pol(v - 1) : v in 8:10 ? c * pol(v - 7) : 1) :
                   i ≤ 4 ? c * pol(i - 1) : c
    forest = p.f.U.forest
    D = length(forest.roots)
    N = forest.N
    blocks = Dict((level(blockkey(p.f.U, b)), block_origin(forest, blockkey(p.f.U, b))) => b
                  for b in 1:nblocks(p.f.U))
    return map(1:5) do i
        s = p.sets[i]
        st = staggers(s)
        worst = 0.0
        for v in 1:s.nvars
            scale = maximum(abs, selectdim(s.work, D + 1, v))
            scale == 0 && continue
            for b in 1:nblocks(s)
                k = blockkey(s, b)
                o = block_origin(forest, k)
                h = spacing(forest, k)
                om = Base.setindex(o, 0.0 - (o[axis] + N * h), axis)
                bm = blocks[(level(k), om)]
                for I in CartesianIndices(ntuple(e -> N + st[e], D))
                    n = Tuple(I)
                    m = Base.setindex(n, (st[axis] == 1 ? N + 2 : N + 1) - n[axis], axis)
                    any(e -> m[e] > N + st[e], 1:D) && continue
                    x = s.work[(n .+ s.G)..., v, b]
                    y = s.work[(m .+ s.G)..., v, bm]
                    worst = max(worst, abs(x - parity(i, v) * y) / scale)
                end
            end
        end
        worst
    end
end

@testset "A closed box refined symmetrically is mirror-symmetric only to the half-open asymmetry at coarse-fine faces (TreeAMR's open item)" begin
    # Records TreeAMR's open item (AMR-8 as landed, its `CODE.md`,
    # "Centerings"): on a refined mesh a vertex-like set is not
    # mirror-symmetric under either kind of wall, because a plane on a
    # coarse-fine face is the fine block's owned plane when the face is its
    # low face and the coarse block's, interpolated at the fine points between
    # the coarse ones, when it is its high face (4.7e−5 upstream, on smooth
    # data, one fill). The explosion in its closed box with the middle 2 × 2
    # roots refined — a mesh symmetric under both reflections, whose
    # coarse-fine faces `|x|, |y| = 1` cut the blast's transition region
    # (`0.8 ≤ r ≤ 1`) — to `t = 1/4`: measured 1.0e−2 (`x → −x`) and 8.8e−3
    # (`y → −y`) of a variable's scale in `U`, 1.1e−2 and 1.3e−2 in `A_z`
    # (3.2e−4 and 6.3e−3 in `A_z` after the first step), against exactly zero
    # on the uniform mesh (`multid_cases_tests.jl`). Asserted as present and
    # bounded, so that the day upstream closes it this testset says so.
    refine_middle!(forest) =
        (refine!(forest, [k for k in forest.leaves
                          if all(x -> -1 ≤ x < 1, block_origin(forest, k))]);
         balance!(forest); forest)
    r = evolve!(RF_BOX, Val(2); N=8, t_end=1 // 4, limiter=:minmod, refine=refine_middle!)
    ex = rf_mirror_error(r.p, r.u, r.t, 1, 1)
    ey = rf_mirror_error(r.p, r.u, r.t, 2, -1)
    @info "Explosion in a closed box, middle refined, t = 1/4: mirror asymmetry per set " *
          "(U, A₁, A₂, A₃, Φ̃) x → −x $ex, y → −y $ey; $(r.nsteps) steps, $(r.nblocks) blocks"
    @test r.nblocks == 16 + 12 && r.nsteps == 20
    @test 1e-4 ≤ maximum(ex) ≤ 5e-2 && 1e-4 ≤ maximum(ey) ≤ 5e-2
end
