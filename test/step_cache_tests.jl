# The step cache (stage 3 of the recovery pipeline; decided by Erik
# 2026-10-23): inside a step of `evolve!` every recovery is seeded from the
# cell's latest recovery in the step, and one whose EntropyEOS input is that
# one's bit for bit reuses its output; the cache is invalidated at the end of
# every step and whenever stepping starts or stops. What it must not change:
# a step is a pure function of `uⁿ`, `t` and `Δt`, so chunks, restarts and
# monitors reproduce the run bit for bit, and outside a step the callbacks are
# today's. What it may change: the answers, by the recovery's tolerance. See
# "The step cache" in `CODE.md`.

const SK = TreeGRRMHD

# The recovered `ρ`, `s` and rapidity `w` at the owned cells of a final state,
# from a right-hand side's recovery outside a step (cold, from the state).
function sk_prims(r)
    P = update_primitives!(r.p, r.u, r.t)
    G, N = r.p.f.P.G, r.p.f.P.forest.N
    D = length(G)
    at(v) = [P.work[ntuple(d -> I[d] + G[d], D)..., v, b]
             for I in CartesianIndices(ntuple(_ -> N, D)), b in 1:nblocks(P)]
    return (; ρ=at(SK.P_RHO), s=at(SK.P_S), w=at(SK.P_W))
end

sk_seeds(r) = map(k -> map(q -> sum(c -> getproperty(getproperty(c.tally.seeds, k), q), r.chunks),
                           (; cold=:cold, warm=:warm, reused=:reused)),
                  (; rhs=:rhs, stage=:stage, stage_limiter=:stage_limiter,
                   step_limiter=:step_limiter))
sk_sum(r, k) = sum(c -> getproperty(c.tally, k), r.chunks)
sk_cells(r) = nblocks(r.p.f.U) * r.p.f.U.forest.N^length(r.p.f.U.forest.roots)

@testset "Warm-seeded runs match the cold runs to the recovery's tolerance" begin
    # Guards the step cache's seeds and reuse ("The step cache" in
    # `CODE.md`): a warm-seeded or reused recovery answers to EntropyEOS's
    # tolerance what the cold one answers, so the run with the cache is the
    # run without it but for that tolerance, carried through the steps: no
    # stage takes another branch, the reset writes the same cells and injects
    # the same mass. A seed taken from the wrong cell or the wrong step, or a
    # cached output reused for another input, moves the state far beyond it.
    # Measured on the laptop and on Symmetry's EPYC 7532 (2026-10-24, the same
    # digits) at the end, `max |Δu|/max |u|` and
    # the relative `ρ`, `s` and absolute `w` of the recovered final state: the
    # EM wave 1.2e-12 (ρ 7.0e-13), Alfvén 1.0e-14, the resistive tube 2.5e-12
    # (ρ 3.0e-12, w 2.3e-12), the magnetized atmosphere tube 7.7e-13 (s
    # 6.3e-12, w 9.9e-12), the rotor 7.8e-13; the bounds hold a margin of 4
    # and 5.
    cases = [("EM wave", em_wave_case(Float64; roots=10), Val(1),
              (; N=10, t_end=1 // 4, limiter=:mc)),
             ("Alfvén wave", alfven_wave_case(Float64; roots=10), Val(1),
              (; N=10, t_end=alfven_period(alfven_wave_case(Float64)) / 4, limiter=:mc)),
             ("resistive tube", resistive_tube_case(Float64, Constant(100.0); roots=20), Val(1),
              (; N=10, t_end=2 // 5, limiter=:mc)),
             ("magnetized atmosphere tube",
              atmosphere_tube_case(Float64; By=1e-3, model=MagnetizationCutoff(Constant(1e3), 10),
                                   roots=20), Val(1),
              (; N=10, t_end=2 // 5, chunk=1 // 10, limiter=:mc)),
             ("rotor", magnetic_rotor_case(Float64; roots=2), Val(2),
              (; N=8, t_end=1 // 20, limiter=:mc))]
    for (name, case, vD, kw) in cases
        cold = evolve!(case, vD; kw..., step_cache=false)
        warm = evolve!(case, vD; kw...)
        du = maximum(abs.(warm.u .- cold.u)) / maximum(abs, cold.u)
        pc, pw = sk_prims(cold), sk_prims(warm)
        dρ = maximum(abs.(pw.ρ .- pc.ρ) ./ pc.ρ)
        ds = maximum(abs.(pw.s .- pc.s) ./ abs.(pc.s))
        dw = maximum(abs.(pw.w .- pc.w))
        r2(x) = round(x; sigdigits=2)
        @info "$name, $(warm.nsteps) steps: the cache moves the state by $(r2(du)), " *
              "ρ $(r2(dρ)), s $(r2(ds)), w $(r2(dw))"
        @test warm.nsteps == cold.nsteps
        @test du ≤ 1e-11
        @test max(dρ, ds, dw) ≤ 5e-11
        @test sum(c -> collect(c.tally.bits), warm.chunks) ==
              sum(c -> collect(c.tally.bits), cold.chunks)
        @test sk_sum(warm, :resets) == sk_sum(cold, :resets)
        @test sk_sum(warm, :step_resets) == sk_sum(cold, :step_resets)
        @test sk_sum(warm, :recoveries) == sk_sum(cold, :recoveries)
        @test isapprox(sk_sum(warm, :injected_D), sk_sum(cold, :injected_D); rtol=1e-6, atol=0)
        # Without the cache nothing is counted as seeded inside a step.
        @test all(s -> s == (; cold=0, warm=0, reused=0), sk_seeds(cold))
    end
end

# The resistive tube on 40 cells to `t = 1/4` in 32 steps of the dyadic
# `Δt = 1/128` (`cfl = 5/16`), so that a chunk of `1/8` holds 16 of them and
# every step's time is the same however the run is cut.
const SK_TUBE = resistive_tube_case(Float64, Constant(100.0); roots=4)
const SK_KW = (; N=10, limiter=:mc, cfl=5 // 16)

@testset "A step is a function of its start: one chunk is two, and a restart is the run" begin
    # Guards rule 1 of the step cache ("The step cache" in `CODE.md`): the
    # cache lives within one step, invalidated at its end and whenever
    # `evolve!` starts or stops stepping. A cache that outlived a step would
    # seed the first step of a chunk from the previous step in one run and
    # from the state in the other, and the runs would part at the recovery's
    # tolerance. The same for a restart from the saved state: a fresh
    # problem (`rebuild`, as after a regrid) and a fresh integrator.
    one_ = evolve!(SK_TUBE, Val(1); SK_KW..., t_end=1 // 4)
    two = evolve!(SK_TUBE, Val(1); SK_KW..., t_end=1 // 4, chunk=1 // 8)
    @test one_.nsteps == two.nsteps == 32
    @test all(c -> c.dt == 1 / 128, two.chunks) && one_.chunks[1].dt == 1 / 128
    @test isequal(two.u, one_.u)

    half = evolve!(SK_TUBE, Val(1); SK_KW..., t_end=1 // 8)
    @test half.nsteps == 16
    p = SK.rebuild(half.p)
    integ = mhd_integrator(p, copy(half.u), 1 // 8, 1 // 4, 16)
    with_step_cache(() -> SK.IRK.solve!(integ), p)
    @test integ.t == 1 / 4
    @test isequal(integ.u, one_.u)
    # And the cache did work in both: one cold recovery per cell and step, at
    # the step's first predictor.
    @test sk_seeds(one_).stage.cold == 32 * sk_cells(one_)
    @test sk_seeds(one_).rhs.reused == 3 * 32 * sk_cells(one_)
end

@testset "Outside a step the cache is off: the callbacks are pure, and monitors change nothing" begin
    # Guards rule 2 of the step cache: only `with_step_cache` turns it on,
    # around `evolve!`'s `step!` and `solve!`, and turns it off and
    # invalidates it in a `finally`. A cache left on would make a direct
    # call of a callback depend on what ran before it, and an `after_step`
    # monitor that recovers (`update_primitives!`, `mhd_rhs!`) would seed the
    # next step; a cache left valid after an exception would seed the next
    # step from the abandoned one.
    plain = evolve!(SK_TUBE, Val(1); SK_KW..., t_end=1 // 8)
    function monitor(p, u, t)
        update_primitives!(p, u, t)
        mhd_rhs!(similar(u), u, p, t)
        return nothing
    end
    watched = evolve!(SK_TUBE, Val(1); SK_KW..., t_end=1 // 8, after_step=monitor)
    @test isequal(watched.u, plain.u)
    @test watched.chunks[1].tally == plain.chunks[1].tally

    p = plain.p
    @test !p.cache_on[]
    @test all(isnan, selectdim(p.cache.work, ndims(p.cache.work) - 1, SK.C_S))
    u = copy(plain.u)
    du1, du2 = similar(u), similar(u)
    mhd_rhs!(du1, u, p, 1 / 8)
    integ = mhd_integrator(p, copy(u), 1 // 8, 1 // 4, 4)
    with_step_cache(() -> SK.IRK.step!(integ), p)
    mhd_rhs!(du2, u, p, 1 / 8)
    @test isequal(du1, du2)

    # An exception inside leaves it off and invalid, and a nested switch is
    # refused.
    @test_throws ErrorException with_step_cache(p) do
        mhd_rhs!(du2, u, p, 1 / 8)
        error("abandoned")
    end
    @test !p.cache_on[]
    @test all(isnan, selectdim(p.cache.work, ndims(p.cache.work) - 1, SK.C_S))
    @test_throws "already active" with_step_cache(() -> with_step_cache(() -> nothing, p), p)
    @test !p.cache_on[]
end

@testset "The right-hand side reuses the stage limiter's recovery wherever the limiter wrote nothing" begin
    # Guards rule (b) and the tallies ("The step cache" in `CODE.md`): the
    # stage limiter and the right-hand side after it recover the same state
    # with the same arithmetic (`B`, `E` and the undensitized variables as
    # `· inv(√γ)` at every site), so the right-hand side's input is the
    # limiter's bit for bit wherever the limiter wrote nothing, and its output
    # is reused; and only the first recovery of a step, the first stage's
    # predictor, is seeded from the state. Per cell and SSP3(4,3,3) step, 11
    # counted recoveries. In vacuum (σ = 0) the stage writes `E★` back, so the
    # limiter reuses the predictor's recovery as well, and the second stage's
    # predictor is `uⁿ` again (the first stage's increment is zero); under
    # `IdealConductor` the second predictor's matter is `uⁿ`'s, which is all
    # the ideal recovery reads. A key computed with other arithmetic at one
    # site, or a cache invalidated between the limiter and the right-hand
    # side, would turn these reuses into warm recoveries.
    kw = (; N=10, t_end=1 // 16, limiter=:mc)
    em = evolve!(em_wave_case(Float64; roots=10), Val(1); kw...)
    al = evolve!(alfven_wave_case(Float64; roots=10), Val(1); kw...)
    for r in (em, al)
        @test r.nsteps == 16
    end
    n = 16 * 100
    @test sk_seeds(em) == (; rhs=(; cold=0, warm=0, reused=3n),
                           stage=(; cold=n, warm=2n, reused=n),
                           stage_limiter=(; cold=0, warm=0, reused=3n),
                           step_limiter=(; cold=0, warm=n, reused=0))
    @test sk_seeds(al) == (; rhs=(; cold=0, warm=0, reused=3n),
                           stage=(; cold=n, warm=2n, reused=n),
                           stage_limiter=(; cold=0, warm=3n, reused=0),
                           step_limiter=(; cold=0, warm=n, reused=0))
    # Where the stage limiter writes (the atmosphere tube's front and its
    # atmosphere), a right-hand side after a write can still find the input
    # unchanged — the atmosphere written over itself — but every one it does
    # not reuse is a cell the limiter wrote.
    atm = evolve!(atmosphere_tube_case(Float64; By=1e-3,
                                       model=MagnetizationCutoff(Constant(1e3), 10),
                                       roots=20), Val(1);
                  N=10, t_end=1 // 10, limiter=:mc)
    s = sk_seeds(atm)
    m = atm.nsteps * sk_cells(atm)
    @test s.rhs.cold == 0 && s.rhs.warm + s.rhs.reused == 3m
    @test 0 < s.rhs.warm ≤ sk_sum(atm, :resets)
    @test s.stage.cold == m && s.stage_limiter.cold == 0 && s.step_limiter.cold == 0
    @test s.stage_limiter.warm + s.stage_limiter.reused == 3m
    @test s.step_limiter.warm + s.step_limiter.reused == m
end
