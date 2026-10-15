# Rung 2b, continued: what the refinement costs and buys, and the three
# choices step 8 made by measurement ("Refinement" in `CODE.md`; step 8 of
# `PLAN.md`):
#
#   * a field loop across a coarse-fine face lies between the uniform coarse
#     and the uniform fine control, in energy and in shape;
#   * the potentials' PointValue width 4 keeps the Alfvén wave's order across
#     coarse-fine faces, where width 2 loses one in L∞;
#   * the gauge damping clears the `𝔅` a pure gauge pulse makes at a
#     refinement boundary;
#   * the matter's ghost fill, bounded, takes a blast into the atmosphere
#     across a coarse-fine face without a repaired ghost.
#
# Uses `refinement_tests.jl`'s meshes and `multid_tests.jl`'s `md_run` and
# `md_conserved`, so it is included after both; the split keeps each file
# near a minute at one thread.

rf_refine_right!(forest) =
    (refine!(forest, [k for k in copy(forest.leaves) if block_origin(forest, k)[1] ≥ 0]);
     balance!(forest); forest)

# `Σ hᴰ |𝔅|²/2` of the state `u` at `t`.
function rf_magnetic_energy(p, u, t)
    update_fields!(p, u, t)
    forest = p.f.U.forest
    D = length(forest.roots)
    return sum(d -> mesh_mapreduce(x -> x^2 / 2, +, 0.0, p.f.B[d]; vars=1,
                                   weight=k -> spacing(Float64, forest, k)^D), 1:3)
end

@testset "A field loop across a coarse-fine face keeps its energy and shape between the coarse and the fine control" begin
    # Guards the field across a refinement boundary (PLAN.md, step 8: energy
    # and shape against a uniform-fine control): Beckwith & Stone's loop on
    # 32 × 16 cells with the half `x ≥ 0` refined, so the face `x = 0` runs
    # through the loop's centre, against the uniform 32 × 16 and 64 × 32
    # meshes, to `t = 1/2`. A coarse-fine face that reflected, created or
    # dissipated field would put the two-level run outside the two controls.
    # Measured at `t = 1/2` (and at `t = 4` by `bin/field_loop_refined.jl`,
    # where the loop has drifted a third of its radius across the face):
    # the magnetic energy 0.?? of its initial value against 0.?? (coarse) and
    # 0.?? (fine), and the L1 error of `A_z` ??.
    case = field_loop_case(Float64)
    runs = map(((; roots=(4, 2)), (; roots=(8, 4)), (; refine=rf_refine_right!))) do kw
        run = md_run(case, Val(2); N=8, t_end=1 // 2, limiter=:mc, kw...)
        p = run.r.p
        e = rf_magnetic_energy(p, run.r.u, run.r.t) / rf_magnetic_energy(p, run.u0, 0.0)
        err = l1_errors(p, run.r.u, exact_state(p, case, run.r.t)).A[3]
        (; run, e, err)
    end
    coarse, fine, two = runs
    @info "Field loop across a coarse-fine face, t = 1/2: energy $(two.e) of its initial " *
          "value (coarse $(coarse.e), fine $(fine.e)); L1(A_z) $(two.err) (coarse " *
          "$(coarse.err), fine $(fine.err)); ∇·𝔅 $(two.run.divb)"
    @test coarse.e < two.e < fine.e
    @test fine.err < two.err < coarse.err
    @test md_conserved(two.run) ≤ 8
end

@testset "The Alfvén wave keeps its order across coarse-fine faces with the potentials at width 4, and not at width 2" begin
    # Guards the potentials' operators (decided in step 8 by the
    # interface-order table, `bin/interface_order.jl`, "Refinement" in
    # `CODE.md`): the fine ghosts of `A` across a coarse-fine face are
    # interpolated from the coarse level, and their curl is `𝔅` in the fine
    # ghost cells, which the reconstruction at the face reads. At width 2
    # (linear) the ghosts' `𝔅` is first order and so is the error at the
    # face: the circularly polarized wave of rung 1a over one period, the
    # middle half of 32, 64 and 128 cells refined, loses an order in L∞
    # (measured 0.9 in `S_y`, against 2.0 at width 4; L1 1.9 and 2.1). Width
    # 6 measured the same as 4 to three digits, as a second-order scheme
    # should.
    rates = map((4, 2)) do w
        errs = map((32, 64, 128)) do M
            c = alfven_wave_case(Float64; roots=M ÷ 8)
            r = evolve!(c, Val(1); N=8, t_end=alfven_period(c), limiter=:mc,
                        refine=rf_refine_middle!, ops_A=Operators(; prolongation=w, restriction=w))
            ex = exact_state(r.p, c, r.t)
            d = r.u .- ex
            (l1_errors(r.p, r.u, ex).U[RF.U_S + 1],
             mesh_mapreduce(abs, max, 0.0, r.p.sets[RF.SEG_U], stateview(d, r.p.sets, RF.SEG_U);
                            vars=RF.U_S + 1))
        end
        hs = [1 / 32, 1 / 64, 1 / 128]
        (RF.convergence_rate(hs, first.(errs)), RF.convergence_rate(hs, last.(errs)))
    end
    @info "Alfvén wave across coarse-fine faces: S_y L1, L∞ rates $(rates[1]) at width 4, " *
          "$(rates[2]) at width 2"
    @test rates[1][1] ≥ 1.9 && rates[1][2] ≥ 1.8
    @test rates[2][2] ≤ 1.2
end

@testset "A pure gauge pulse makes 𝔅 only at refinement boundaries, and the default damping clears it" begin
    # Guards the gauge damping (decided in step 8, `default_ξ`, "Refinement"
    # in `CODE.md`): `A = 0` and a Gaussian `Φ̃` (`gauge_pulse_case`) carry a
    # pure gauge wave in which `A` stays a gradient, so on a uniform mesh
    # `𝔅` stays zero to rounding; across a coarse-fine face the
    # prolongation of a coarse gradient is not a fine one, and `𝔅` appears.
    # Without damping the gauge pulse keeps crossing the faces of the
    # periodic box and the spurious field grows; with `ξ = 16/L` it decays
    # with the pulse. 2 × 2 roots of 8², one refined, to `t = 2`: measured
    # `max |𝔅|` at the end 0.18 with `ξ = 0` and 1.2e−3 with the default
    # (of a pulse whose `|∇Φ̃|` peaks at 8.6), 4.4e−16 on the uniform mesh.
    gp = gauge_pulse_case(Float64; roots=2)
    maxB(p) = maximum(d -> mesh_mapreduce(abs, max, 0.0, p.f.B[d]; vars=1), 1:3)
    final(r) = (update_fields!(r.p, r.u, r.t); maxB(r.p))
    uniform = evolve!(gp, Val(2); N=8, t_end=2, limiter=:mc)
    undamped = evolve!(gp, Val(2); N=8, t_end=2, limiter=:mc, refine=rf_refine_first!, ξ=0)
    damped = evolve!(gp, Val(2); N=8, t_end=2, limiter=:mc, refine=rf_refine_first!)
    b = (final(uniform), final(undamped), final(damped))
    @info "Gauge pulse, t = 2: max |𝔅| $(b[1]) (uniform), $(b[2]) (two-level, ξ = 0), " *
          "$(b[3]) (two-level, ξ = $(damped.p.ξ))"
    @test damped.p.ξ == 16 && uniform.p.ξ == 0
    @test b[1] ≤ 1e-13
    @test b[2] ≥ 1e-2
    @test b[3] ≤ b[2] / 30
end

# Over the run, the recoveries of the interior ghosts — those not beyond the
# domain's faces — that the policy repaired for anything but the atmosphere,
# their flags, and the change the step limiter's writes made to `√γ D`.
mutable struct RFGhosts
    repaired::Int
    flags::UInt32
end
function rf_ghost_repairs!(acc, p, u, t)
    P = update_primitives!(p, u, t)
    G, N = P.G[1], P.forest.N
    lo, hi = P.forest.extents[1]
    for b in 1:nblocks(P), i in vcat(1:G, (G + N + 1):(N + 2G))
        lo < coordinates(P, b, (i,))[1] < hi || continue
        fl = unsafe_trunc(UInt32, P.work[i, RF.P_FLAGS, b])
        if fl & RF.EntropyEOS.FLAG_POL_ANY & ~RF.EntropyEOS.FLAG_POL_ATMOSPHERE != 0
            acc.repaired += 1
            acc.flags |= fl
        end
    end
    return acc
end

@testset "The matter's ghost fill, bounded, takes a blast into the atmosphere across a coarse-fine face" begin
    # Guards the matter's operators (decided in step 8, "Refinement" in
    # `CODE.md`): the ghost fill prolongates the conserved state, and the
    # unbounded width-3 operator overshoots the front of a gas expanding
    # into a 10⁻⁸ atmosphere — fine ghosts with `D < 0`, which the recovery
    # reports as non-finite and repairs, and a reset that injects 250 times
    # the rest mass it does on the uniform mesh; bounded (TreeAMR's MC-bounded
    # prolongation, AMR-4), no ghost is repaired at all and the injection is
    # the uniform mesh's (−8.4e−12, `atmosphere_tests.jl`). The atmosphere
    # tube on 200 cells with `[0, 0.2)` refined, which the front crosses.
    case = atmosphere_tube_case(Float64)
    refine_band!(forest) =
        (refine!(forest, [k for k in copy(forest.leaves)
                          if 0 ≤ block_origin(forest, k)[1] < 1 // 5 - 1e-12]);
         balance!(forest); forest)
    runs = map((true, false)) do bounded
        acc = RFGhosts(0, 0)
        ops_U = Operators(; prolongation=3, restriction=2, family=Conservative, bounded=bounded)
        r = evolve!(case, Val(1); N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc,
                    refine=refine_band!, ops_U=ops_U,
                    after_step=(p, u, t) -> rf_ghost_repairs!(acc, p, u, t))
        (; r, acc, injected=sum(c -> c.tally.injected_D, r.chunks))
    end
    bounded, unbounded = runs
    @info "Atmosphere tube across a coarse-fine face: repaired interior ghosts " *
          "$(bounded.acc.repaired) bounded, $(unbounded.acc.repaired) unbounded (flags " *
          "$(string(unbounded.acc.flags; base=16))); injected rest mass $(bounded.injected) and " *
          "$(unbounded.injected)"
    @test bounded.r.nblocks == 24
    @test bounded.acc.repaired == 0
    @test abs(bounded.injected) ≤ 1e-10
    @test unbounded.acc.flags & RF.EntropyEOS.FLAG_POL_NONFINITE != 0
    @test unbounded.injected ≥ 1e-9
end
