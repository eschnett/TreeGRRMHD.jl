# The two-phase recovery (stage 4 of the recovery pipeline, 2026-10-30): the
# recoveries outside the stage solve's iterations — the right-hand side's,
# both limiters', the stage predictor's — run as three passes over the owned
# cells: `con2prim_safe_start` into a hand-off, `con2prim_safe_finish` where
# the first stage was not done, and level 2 with the site's own work. What it
# must not change: anything. EntropyEOS defines `con2prim_safe` as its two
# phases, so the split is the fused recovery bit for bit. Fused is the
# default (decided by Erik 2026-11-02), so every split run here asks for
# `two_phase = true`. See "The two-phase recovery" in `CODE.md`.

const TP = TreeGRRMHD

tp_same_tallies(a, b) = length(a.chunks) == length(b.chunks) &&
                        all(i -> isequal(a.chunks[i].tally, b.chunks[i].tally), eachindex(a.chunks))

@testset "The two-phase recovery is the fused recovery bit for bit, in every run" begin
    # Guards the split into passes ("The two-phase recovery" in `CODE.md`):
    # an input the first pass computes with other arithmetic than the third,
    # a seed or a step-cache entry read differently, a hand-off read from
    # another cell or left over from another site, a reused entry started
    # anyway, or a finish skipped where it was needed would move a state or
    # a tally of the run. Nothing may move at all: every state and every
    # tally of every chunk identical. The cases cover every site and both
    # sides of the finish guard — the EM wave (`Vacuum`, the stage at
    # `σ = 0`), the Alfvén wave (`IdealConductor`, the ideal predictor), the
    # resistive tube (Newton), the magnetized atmosphere tube (the faces and
    # the atmosphere: finish reached at the predictor, the right-hand side and
    # the step limiter; also with the step cache off, every recovery cold),
    # the rotor in 2D, and the adaptive rotor, whose regrids rebuild the
    # problem and its hand-off. A monitor counts the cells whose step-limiter
    # recovery reached finish (the step limiter's hand-off is the last a step
    # writes); the atmosphere tube must have some, or the guard's other side
    # is not tested.
    atm = atmosphere_tube_case(Float64; By=1e-3, model=MagnetizationCutoff(Constant(1e3), 10),
                               roots=20)
    cases = [("EM wave", em_wave_case(Float64; roots=10), Val(1),
              (; N=10, t_end=1 // 8, limiter=:mc), false),
             ("Alfvén wave", alfven_wave_case(Float64; roots=10), Val(1),
              (; N=10, t_end=1 // 8, limiter=:mc), false),
             ("resistive tube", resistive_tube_case(Float64, Constant(100.0); roots=20), Val(1),
              (; N=10, t_end=1 // 5, limiter=:mc), false),
             ("magnetized atmosphere tube", atm, Val(1),
              (; N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc), true),
             ("magnetized atmosphere tube, no step cache", atm, Val(1),
              (; N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc, step_cache=false), true),
             ("rotor", magnetic_rotor_case(Float64; roots=2), Val(2),
              (; N=8, t_end=1 // 20, limiter=:mc), false),
             ("adaptive rotor", magnetic_rotor_case(Float64; roots=4), Val(2),
              (; N=8, t_end=1 // 20, chunk=1 // 40, limiter=:mc, maxlevel=1, refine_tol=0.08,
               coarsen_tol=0.02, buffer=4), false)]
    for (name, case, vD, kw, finishes) in cases
        nfinish = Ref(0)
        split = evolve!(case, vD; kw..., two_phase=true,
                        after_step=(p, u, t) -> (nfinish[] += count(p.handoff.todo); nothing))
        fused = evolve!(case, vD; kw..., two_phase=false, after_step=(p, u, t) -> nothing)
        @info "$name, $(split.nsteps) steps, $(split.nregrids) regrids: step-limiter " *
              "recoveries that reached finish $(nfinish[])"
        @test split.p.handoff isa TP.Handoff && fused.p.handoff === nothing
        @test split.nsteps == fused.nsteps && split.nregrids == fused.nregrids
        @test isequal(split.u, fused.u)
        @test tp_same_tallies(split, fused)
        @test !finishes || nfinish[] > 0
    end
end

@testset "Every callback is the fused one bit for bit where the first phase is not done" begin
    # Guards the finish pass and the hand-off on the inputs that reach it, at
    # every site outside a step (the step cache off, every recovery from the
    # state): a cell below the density floor (`ρ_FLOORED`, the atmosphere), a
    # cell whose energy is below the coldest state (the `s_min` face), one
    # whose momentum exceeds the rapidity cap, and a non-finite one — each a
    # branch of `con2prim_safe_finish` — in the magnetized atmosphere tube, for
    # the reset, the right-hand side, and the stage solve under the tube's
    # conductivity and under `IdealConductor` (the ideal predictor). The
    # split's hand-off must hold at least those four cells as needing finish.
    r = evolve!(atmosphere_tube_case(Float64; By=1e-3,
                                     model=MagnetizationCutoff(Constant(1e3), 10), roots=20),
                Val(1); N=10, t_end=1 // 40, limiter=:mc)
    t = r.t
    for model in (r.p.model, IdealConductor())
        ps = TP.rebuild(r.p; model=model, two_phase=true)
        pf = TP.rebuild(r.p; model=model, two_phase=false)
        @test ps.handoff isa TP.Handoff && pf.handoff === nothing
        u = copy(r.u)
        uU = statearray(u, ps.sets, TP.SEG_U)
        uU[3, TP.U_D, 2] *= 1e-6                       # below ρ_min
        uU[5, TP.U_TAU, 4] = -abs(uU[5, TP.U_TAU, 4])  # colder than any state
        # faster than the cap (`W ≈ 700`, at `|S| = (1 − 10⁻⁶)(τ + D)`)
        uU[7, TP.U_S, 6] = (1 - 1e-6) * (uU[7, TP.U_TAU, 6] + uU[7, TP.U_D, 6])
        uU[2, TP.U_TAU, 9] = NaN                       # not a state
        γΔt = 1e-3
        results = map((ps, pf)) do p
            clear_tally!(p)
            v = copy(u)
            reset_atmosphere!(v, nothing, p, t)
            n_reset = p.handoff === nothing ? 0 : count(p.handoff.todo)
            du = similar(u)
            mhd_rhs!(du, u, p, t)
            n_rhs = p.handoff === nothing ? 0 : count(p.handoff.todo)
            U = copy(u)
            ohm_solve!(U, u, γΔt, p, t)
            n_stage = p.handoff === nothing ? 0 : count(p.handoff.todo)
            P = copy(p.f.P.work)
            (; v, du, U, P, tally=ohm_tally(p), n=(n_reset, n_rhs, n_stage))
        end
        s, f = results
        @info "$(nameof(typeof(model))): cells that reached finish (reset, right-hand " *
              "side, stage) $(s.n)"
        @test all(≥(4), s.n)
        @test isequal(s.v, f.v)
        @test isequal(s.du, f.du)
        @test isequal(s.U, f.U)
        @test isequal(s.P, f.P)
        @test isequal(s.tally, f.tally)
    end
end
