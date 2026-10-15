# Rung 2a, continued: the solutions that vary in more than one direction
# ("Validation ladder" 2 in `CODE.md`; step 7 of `PLAN.md`) — the Alfvén wave
# along the diagonal, and GRaM-X's and Beckwith & Stone's standard tests:
# symmetry, conservation and `∇·𝔅` at every step. Uses `multid_tests.jl`'s
# `md_run` (which asserts `∇·𝔅 ≤ 1e−12` after every step of every run),
# `md_mirror_error` and `md_conserved`, so it is included after it; the split
# keeps each file near a minute at one thread.

@testset "The Alfvén wave along the diagonal converges at second order: $lim" for lim in (:mc, :wenoz)
    # Guards what the bit-for-bit claims cannot: a solution that varies along
    # both axes, so that both fluxes, the EMF along `z` with both of its
    # transverse reconstructions, and the curl's two differences all carry
    # it. The 1D wave turned to `(1, 1)/√2` on the periodic `[0, √2]²`
    # (`oblique_case`), an eighth of a period at 16², 32² and 64² cells, against
    # the exact solution. Measured (MC): `S_x`, `S_z` 2.21, 2.23, `A_x`, `A_z`
    # 2.77, 2.31; (WENO-Z) 1.97, 1.98, 2.05, 2.01 (2.769 and 2.07 for `A_x`
    # since step 8 damps the gauge, `ξ = 16/L`) — the plan's 1.8 for the
    # 1D wave holds. And the totals are conserved to rounding.
    c1 = alfven_wave_case(Float64; roots=1)
    t_end = alfven_period(c1) / 8
    runs = map((8, 16, 32)) do N
        run = md_run(oblique_case(c1; N=N, roots=2), Val(2); N=N, t_end=t_end, limiter=lim)
        (; run, e=l1_errors(run.r.p, run.r.u, exact_state(run.r.p, run.case, run.r.t)))
    end
    hs = (1 / 16, 1 / 32, 1 / 64)
    for (name, get) in (("S_x", e -> e.U[MD.U_S]), ("S_y", e -> e.U[MD.U_S + 1]),
                        ("S_z", e -> e.U[MD.U_S + 2]), ("A_x", e -> e.A[1]), ("A_y", e -> e.A[2]),
                        ("A_z", e -> e.A[3]))
        errs = [get(x.e) for x in runs]
        rate = MD.convergence_rate(collect(hs), errs)
        @info "Oblique Alfvén wave, $lim: $name L1 errors $(errs), rate $(round(rate, digits=3))"
        @test rate ≥ 1.8
        @test issorted(errs; rev=true)
    end
    worst = maximum(x -> md_conserved(x.run), runs)
    @info "Oblique Alfvén wave, $lim: worst drift $(round(worst, digits=4)) ulp of a total's scale per step"
    @test all(x -> x.run.tally.policy == 0 && x.run.tally.resets == 0, runs)
end

@testset "The cylindrical explosion keeps its mirror symmetries to rounding" begin
    # Guards the scheme on a strong 2D blast (GRaM-X §V B: a pressure ratio of
    # 3·10⁴, a shell at `W ≈ 4`): with the field along `x`, the solution is
    # symmetric under `x → −x`, and under `y → −y` with every field reversed;
    # a scheme that is not — an upwinding that prefers one side, a
    # reconstruction that is not mirror-symmetric — shows as an asymmetry far
    # above rounding long before it shows in a picture. Minmod, as GRaM-X runs
    # it, at 64² on `[−6, 6]²` to `t = 1`: no stage is repaired and no reset
    # fires (under MC the reset floors the entropy of a few cells behind the
    # front — measured in step 7, `CODE.md`). Measured asymmetry 1.6e−18 and
    # 2.5e−17 of each variable's scale; asserted ≤ 1e−12.
    run = md_run(cylindrical_explosion_case(Float64; roots=4), Val(2); N=16, t_end=1,
                 limiter=:minmod)
    ex = md_mirror_error(run.r.p, run.r.u, (1,), 1)
    ey = md_mirror_error(run.r.p, run.r.u, (2,), -1)
    @info "Cylindrical explosion, 64², t = 1: asymmetry x → −x $ex, y → −y (fields reversed) " *
          "$ey; ∇·𝔅 $(run.divb); $(run.r.nsteps) steps"
    @test ex ≤ 1e-12 && ey ≤ 1e-12
    @test run.tally.policy == 0 && run.tally.resets == 0
    @test run.tally.bits[2] == sum(run.tally.bits)        # OHM_IDEAL, every stage
    # The blast did something: the shell moves out at relativistic speed.
    P = hostcopy(update_primitives!(run.r.p, run.r.u, run.r.t))
    @test maximum(P.work[:, :, MD.P_V, :]) > 1 // 2
end

@testset "The magnetic rotor keeps its symmetry under a rotation by π to rounding" begin
    # Guards the scheme on GRaM-X's rotor (§V C: `W ≈ 10` at the cylinder's
    # edge, a density jump of 10 with no smoothing): the rotation by π about
    # the axis maps the rotor to itself with every field reversed. MC at 64²
    # to `t = 0.1`; the fast waves have not reached the faces, so the totals
    # change only by the light-cone fluxes' exponentially small precursors
    # (`shocktube_tests.jl`'s bound, 10⁻¹⁰ of the scale). Measured asymmetry
    # 2.6e−27 in step 7, 0.0 since step 8's walls.
    run = md_run(magnetic_rotor_case(Float64; roots=4), Val(2); N=16, t_end=1 // 10,
                 limiter=:mc)
    e = md_mirror_error(run.r.p, run.r.u, (1, 2), -1)
    worst = maximum(v -> abs(run.drift[v]) / run.scales[v],
                    (MD.U_D, MD.U_S, MD.U_S + 1, MD.U_TAU, MD.U_DYE, MD.U_DS))
    @info "Magnetic rotor, 64², t = 1/10: asymmetry under the rotation by π $e; totals " *
          "drift $worst of their scale; ∇·𝔅 $(run.divb)"
    @test e ≤ 1e-12
    @test worst ≤ 1e-10
    @test run.tally.policy == 0 && run.tally.resets == 0
end

@testset "Orszag–Tang conserves every total to rounding and keeps its symmetry" begin
    # Guards conservation in 2D on a periodic box (PLAN.md: `D`, `S_i`, `τ` to
    # round-off): the fluxes of a face computed once and used by both cells,
    # in both directions, and the explicit sources zero in flat space — so
    # each total drifts by a few ulp of its scale per step at most. And the
    # vortex's symmetry under the rotation by π about the box's centre
    # (Beckwith & Stone's "ability to maintain symmetry"), here without
    # reversing the fields. MC at 64² to `t = 1/8` (40 steps). Not to the last
    # bit: the asymmetry is set by the recovery's tolerance (EntropyEOS's
    # 1e−12), since mirrored states that differ in the last bit can stop at
    # different iterates — measured 9.3e−16 at `t = 0`, then 1.2e−13, 2.5e−13,
    # 1.9e−13, 1.6e−14 at `t` = 1/32 … 1/4, and 5.3e−15 at 1/16 with the
    # tolerance at 1e−14 (step 7). Asserted ≤ 1e−11, ten times the tolerance.
    # The explosion and the rotor, whose recoveries converge alike on both
    # sides, are symmetric far below it.
    run = md_run(orszag_tang_case(Float64; roots=4), Val(2); N=16, t_end=1 // 8, limiter=:mc)
    worst = md_conserved(run)
    e = md_mirror_error(run.r.p, run.r.u, (1, 2), 1)
    @info "Orszag–Tang, 64², t = 1/8: worst drift $(round(worst, digits=4)) ulp of a total's " *
          "scale per step; asymmetry $e; ∇·𝔅 $(run.divb)"
    @test e ≤ 1e-11
    @test run.tally.policy == 0 && run.tally.resets == 0
end

@testset "A field loop is advected conservatively, and Bᶻ stays zero without vᶻ" begin
    # Guards the advection of a weak, discontinuous field (Beckwith & Stone
    # §4.2; Gardiner & Stone 2005): conservation to rounding on the periodic
    # box, and — with `vᶻ = 0` — `Eˣ = Eʸ = 0` exactly, so `A_x`, `A_y` and
    # with them `𝔅ᶻ` stay zero exactly, as Beckwith & Stone found. With
    # `vᶻ ≠ 0` the in-plane EMFs are `vᶻ B` and `𝔅ᶻ` is the discrete curl of
    # their time integral: it stays small (4.6e−4 of the in-plane field,
    # measured; asserted ≤ 1e−2), and its size is recorded. 64 × 32
    # cells to `t = 1/2` (a fiftieth of a crossing, 80 steps).
    flat = md_run(field_loop_case(Float64), Val(2); N=16, t_end=1 // 2, limiter=:mc)
    tilted = md_run(field_loop_case(Float64; vz=1 // 10), Val(2); N=16, t_end=1 // 2,
                    limiter=:mc)
    for run in (flat, tilted)
        worst = md_conserved(run)
        @info "Field loop, vᶻ = $(run.case.fields.w.p₄): worst drift $(round(worst, digits=4)) " *
              "ulp of a total's scale per step; ∇·𝔅 $(run.divb)"
        @test run.tally.policy == 0 && run.tally.resets == 0
    end
    p, u = flat.r.p, flat.r.u
    @test all(iszero, stateview(u, p.sets, MD.SEG_A)) &&
          all(iszero, stateview(u, p.sets, MD.SEG_A + 1))
    uU = statearray(u, p.sets, MD.SEG_U)
    @test all(iszero, uU[:, :, MD.U_E, :]) && all(iszero, uU[:, :, MD.U_E + 1, :])
    # The tilted loop's 𝔅ᶻ, against its in-plane field.
    q = tilted.r
    update_fields!(q.p, q.u, q.t)
    Bz = maximum(abs, q.p.f.B[3].work)
    Bxy = max(maximum(abs, q.p.f.B[1].work), maximum(abs, q.p.f.B[2].work))
    @info "Field loop, vᶻ = 0.1/√6: max |𝔅ᶻ| / max |𝔅ˣʸ| = $(Bz / Bxy) at t = 1/2"
    @test Bz ≤ 1e-2 * Bxy
end

@testset "A closed box is mirror-symmetric bit for bit: the explosion in 2D" begin
    # Guards AMR-8's workaround (step 8; decided by Erik 2026-10-15): TreeAMR
    # fills the high wall plane of a vertex-like dimension at an outer face as a
    # ghost and leaves the low one owned and evolved, so without the
    # prescription of the low walls (`low_walls!`) `𝔅` on a box's high boundary
    # face is the case's and on its low one the evolution's; and the hook,
    # which runs before the prolongations, took a point where a wall meets
    # another block from the block's own last column (`outer_points!`). The
    # explosion in a box small enough that its light-cone precursors reach the
    # faces within a step — `[−2, 2]²` at 32², minmod, to `t = 1/4` (10 steps):
    # 7.1e−6 (`x → −x`) and 6.7e−5 (`y → −y`, fields reversed) of a variable's
    # scale apart before the fix (measured in step 8), and now the mirror image
    # of itself, every evolved set on the closed range, both walls included,
    # bit for bit but the sign of a zero; and the state holds the prescribed
    # low walls.
    run = md_run(cylindrical_explosion_case(Float64; L=2, roots=4), Val(2); N=8, t_end=1 // 4,
                 limiter=:minmod)
    ex = md_mirror_equal(run.r.p, run.r.u, run.r.t, (1,), 1)
    ey = md_mirror_equal(run.r.p, run.r.u, run.r.t, (2,), -1)
    @info "Explosion in a closed box, 32², t = 1/4: mirror-symmetric x → −x $(ex.mirrored), " *
          "y → −y $(ey.mirrored); ∇·𝔅 $(run.divb); $(run.r.nsteps) steps"
    @test run.r.nsteps == 10
    @test ex.mirrored && ey.mirrored && ex.persisted
    @test run.tally.policy == 0 && run.tally.resets == 0
    # The faces were reached: the cells along them changed.
    q, q0 = md_global(run.r.p, run.r.u, 1), md_global(run.r.p, run.u0, 1)
    @test q[1, :, :] != q0[1, :, :] && q[:, 1, :] != q0[:, 1, :]
end

@testset "A spherical blast in 3D runs in a closed box, mirror-symmetric bit for bit, with ∇·𝔅 at rounding" begin
    # The 3D smoke test (PLAN.md, step 7): the explosion with `r` the distance
    # from the origin, on `[−2, 2]³` at 16³ (two roots of `N = 8` a dimension),
    # minmod, to `t = 1/4` — every kernel along every direction, the
    # twelve-edge curl, and the field's three reflections (`x → −x` alone,
    # `y → −y` and `z → −z` with the fields reversed). In a closed box since
    # step 8 (AMR-8's workaround; step 7 had to run it periodic): the
    # light-cone precursors reach every face of so small a box within a step
    # or two, and with the low walls evolved the box was 1.3e−2 (`x → −x`) and
    # 1.0e−1 (`y`, `z`) of a variable's scale away from its mirror image at
    # `t = 1/4` (measured in step 8; 1.3e−4 in step 7's measurement after
    # three steps). Now bit for bit but the sign of a zero, every evolved set
    # on the closed range.
    run = md_run(cylindrical_explosion_case(Float64, Val(3); L=2, roots=2, spherical=true),
                 Val(3); N=8, t_end=1 // 4, limiter=:minmod)
    e = (md_mirror_equal(run.r.p, run.r.u, run.r.t, (1,), 1),
         md_mirror_equal(run.r.p, run.r.u, run.r.t, (2,), -1),
         md_mirror_equal(run.r.p, run.r.u, run.r.t, (3,), -1))
    @info "Spherical blast in a closed box, 16³, t = 1/4: mirror-symmetric " *
          "$(map(x -> x.mirrored, e)); ∇·𝔅 $(run.divb); $(run.r.nsteps) steps"
    @test all(x -> x.mirrored, e) && e[1].persisted
    @test all(isfinite, run.r.u)
    @test run.tally.policy == 0 && run.tally.resets == 0
end
