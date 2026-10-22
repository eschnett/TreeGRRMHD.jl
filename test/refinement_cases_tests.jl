# Rung 2b, continued: what the refinement costs and buys, and the three
# choices step 8 made by measurement ("Refinement" in `CODE.md`; step 8 of
# `PLAN.md`):
#
#   * a field loop across a coarse-fine face lies between the uniform coarse
#     and the uniform fine control, in energy and in shape;
#   * the potentials' width 4 keeps the Alfvén wave's order across
#     coarse-fine faces, where width 2 loses one in L∞;
#   * the gauge damping clears the `𝔅` a pure gauge pulse makes at a
#     refinement boundary;
#   * the matter's ghost fill, in primitives at width 2, takes a blast into
#     the atmosphere across a coarse-fine face with no inadmissible ghost.
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
    # the magnetic energy 0.803 of its initial value against 0.623 (coarse)
    # and 0.844 (fine), and the L1 error of `A_z` 1.56e−6 against 3.0e−6 and
    # 8.7e−7.
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
    # (measured 0.88 in `S_y`, against 1.93 at width 4; L1 1.91 and 2.13,
    # with the matter's ghost fill in primitives at width 2 — 0.93, 2.03,
    # 1.91, 2.16 with step 8's first, conserved fill). Width 6 measured the
    # same as 4 to three digits, as a second-order scheme should. Since step
    # 8 took up AMR-9, `A_j` are conservative along the edge and point
    # values of the width across it, and `Φ̃` PointValue of the width.
    rates = map((4, 2)) do w
        errs = map((32, 64, 128)) do M
            c = alfven_wave_case(Float64; roots=M ÷ 8)
            r = evolve!(c, Val(1); N=8, t_end=alfven_period(c), limiter=:mc,
                        refine=rf_refine_middle!,
                        ops_A=Operators(; prolongation=3, restriction=2, family=Conservative,
                                        vertex_prolongation=w),
                        ops_Φ=Operators(; prolongation=w, restriction=w))
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
    # `max |𝔅|` at the end 0.049 with `ξ = 0` and 1.6e−5 with the default
    # (of a pulse whose `|∇Φ̃|` peaks at 8.6), and on the uniform mesh, with
    # the same default `ξ` (one `ξ` on every mesh, decided by Erik
    # 2026-10-15), 8.7e−19 (4.4e−16 at `ξ = 0`). *(Re-measured in step 8
    # with the potentials' conservative transfer, AMR-9, and the matter's
    # ghost fill in primitives: it was 0.18 and 1.2e−3 under the PointValue
    # operators and the conserved fill.)*
    gp = gauge_pulse_case(Float64; roots=2)
    maxB(p) = maximum(d -> mesh_mapreduce(abs, max, 0.0, p.f.B[d]; vars=1), 1:3)
    final(r) = (update_fields!(r.p, r.u, r.t); maxB(r.p))
    uniform = evolve!(gp, Val(2); N=8, t_end=2, limiter=:mc)
    undamped = evolve!(gp, Val(2); N=8, t_end=2, limiter=:mc, refine=rf_refine_first!, ξ=0)
    damped = evolve!(gp, Val(2); N=8, t_end=2, limiter=:mc, refine=rf_refine_first!)
    b = (final(uniform), final(undamped), final(damped))
    @info "Gauge pulse, t = 2: max |𝔅| $(b[1]) (uniform), $(b[2]) (two-level, ξ = 0), " *
          "$(b[3]) (two-level, ξ = $(damped.p.ξ))"
    @test damped.p.ξ == 16 && uniform.p.ξ == 16
    @test b[1] ≤ 1e-13
    @test b[2] ≥ 1e-2
    @test b[3] ≤ b[2] / 30
end

# Over the run, the interior ghosts — those not beyond the domain's faces —
# whose primitives the policy would repair for anything but the atmosphere
# (EntropyEOS's `check_prim_state` of `(ρ, s, Yₑ, asinh |u|)`: is the ghost an
# admissible state as it stands), with their flags, and those below the
# atmosphere's density or non-finite.
mutable struct RFGhosts
    repaired::Int
    flags::UInt32
    below::Int
end
function rf_ghost_repairs!(acc, p, u, t)
    P = update_primitives!(p, u, t)
    G, N = P.G[1], P.forest.N
    lo, hi = P.forest.extents[1]
    for b in 1:nblocks(P), i in vcat(1:G, (G + N + 1):(N + 2G))
        lo < coordinates(P, b, (i,))[1] < hi || continue
        ρ = P.work[i, RF.P_RHO, b]
        (isfinite(ρ) && ρ ≥ (1 - 1e-12) * 1e-8) || (acc.below += 1)
        ps = RF.EntropyEOS.PrimState{Float64}(ρ, P.work[i, RF.P_S, b], P.work[i, RF.P_YE, b],
                                              P.work[i, RF.P_W, b])
        fl = RF.EntropyEOS.check_prim_state(p.eos, ps, p.pol)
        if fl & RF.EntropyEOS.FLAG_POL_ANY & ~RF.EntropyEOS.FLAG_POL_ATMOSPHERE != 0
            acc.repaired += 1
            acc.flags |= fl
        end
    end
    return acc
end

@testset "The matter's ghost fill in primitives takes a blast into the atmosphere across a coarse-fine face with the parent-centred bounded prolongation, and not with PointValue width 4" begin
    # *(Experiment branch `claude/step-8-ghost-athena`:)* the default fill of
    # `W` is TreeAMR's bounded conservative operator (width 3, MC slopes
    # centred on the parent, scaled to keep every child within the parent and
    # its face neighbours — Athena++'s construction, Stone et al. 2020,
    # §2.1.5, but for MC against minmod), on the specific `s`, `Yₑ`: measured
    # injected D −1.6e−11 and τ −2.6e−10 (the conserved fill's −1.5e−11 and
    # −2.5e−10), and no blow-up in twelve perturbed magnetized runs. The
    # rest of this comment is the PointValue fill's.
    # Guards the matter's ghost fill (decided by Erik 2026-10-15; the width
    # proposed in step 8, "Refinement" in `CODE.md`): the fill prolongates
    # the primitives `ρ, s, Yₑ, uⁱ, Eⁱ` (PointValue, bounded: AMR-10) and
    # builds each conserved ghost by `prim2con` with the ghost's own `B`.
    # The atmosphere tube on 200 cells with `[0, 0.2)` refined, which the
    # front crosses (`bin/ghost_fill.jl` has the whole table). At width 2 —
    # the multilinear interpolation, every variable with the same weights —
    # no interior ghost is inadmissible or below the atmosphere, hydro-
    # dynamically and magnetized (`Bʸ = 10⁻³`, cut off at `μ_max = 10`, where
    # step 8's conserved ghosts had 33 repaired recoveries), and the reset
    # injects 5.6e−9 of rest mass and 2.4e−9 of energy (the uniform mesh
    # −8.4e−12 and −2.2e−10; the conserved fill −1.5e−11 and −2.5e−10). At
    # width 4 the bound clamps each variable on its own, and where the front
    # starts on the coarse-fine face a coarse ghost restricted from a dense
    # and a heated atmosphere cell took the first's density and the second's
    # entropy — a pressure of 5 beside 0.2 — and the reset injects 5.4e−4 of
    # energy; unbounded, ghosts fall below the atmosphere and the energy goes
    # non-finite.
    case = atmosphere_tube_case(Float64)
    mag = atmosphere_tube_case(Float64; By=1e-3, model=MagnetizationCutoff(Constant(1e3), 10))
    refine_band!(forest) =
        (refine!(forest, [k for k in copy(forest.leaves)
                          if 0 ≤ block_origin(forest, k)[1] < 1 // 5 - 1e-12]);
         balance!(forest); forest)
    pv(w; bounded=true) = Operators(; prolongation=w, restriction=w, family=PointValue,
                                    bounded=bounded)
    function run(c, ops_W)
        acc = RFGhosts(0, 0, 0)
        r = evolve!(c, Val(1); N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc,
                    refine=refine_band!, ops_W=ops_W,
                    after_step=(p, u, t) -> rf_ghost_repairs!(acc, p, u, t))
        (; r, acc, D=sum(c -> c.tally.injected_D, r.chunks),
         τ=sum(c -> c.tally.injected_τ, r.chunks), fb=sum(c -> c.tally.bits[5], r.chunks))
    end
    two = run(case, RF.default_ops_W())
    magnetized = run(mag, RF.default_ops_W())
    four = run(case, pv(4))
    unbounded = run(case, pv(4; bounded=false))
    @info "Atmosphere tube across a coarse-fine face, primitive ghosts: width 2 repaired " *
          "$(two.acc.repaired), below ρ_atm $(two.acc.below), injected D $(two.D), τ $(two.τ); " *
          "magnetized $(magnetized.acc.repaired), $(magnetized.acc.below), $(magnetized.D), " *
          "$(magnetized.τ), fallbacks $(magnetized.fb); width 4 injected τ $(four.τ); " *
          "unbounded below ρ_atm " *
          "$(unbounded.acc.below), injected τ $(unbounded.τ)"
    @test two.r.nblocks == 24 && RF.default_ops_W().family == Conservative &&
          RF.default_ops_W().prolongation == 3
    @test two.acc.repaired == 0 && two.acc.below == 0
    @test magnetized.acc.repaired == 0 && magnetized.acc.below == 0
    @test abs(two.D) ≤ 1e-10 && abs(two.τ) ≤ 1e-8
    @test abs(magnetized.D) ≤ 1e-6
    # No OHM_FALLBACK magnetized (decided by Erik 2026-10-22): under the
    # atmosphere-at-rest recovery the tube is deterministic, with no fallback
    # in any of 48 perturbed runs on the uniform mesh and on both refined
    # meshes (`bin/front_investigation.jl`; "Floors, the atmosphere and the
    # magnetosphere" in `CODE.md`).
    @test magnetized.fb == 0
    @test four.τ ≥ 1e-5
    # *(Experiment branch: with `W` holding `ρ★ s`, `ρ★ Yₑ`, a ghost with
    # `ρ★ ≤ 0` takes the policy's atmosphere, so the unbounded control's
    # energy is absurd (−1.1e15) rather than non-finite.)*
    @test unbounded.acc.below > 0 && !(abs(unbounded.τ) < 1)
end
