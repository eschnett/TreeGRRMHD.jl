# The implicit stage solve of step 3: Ohm's conduction current, per cell, per
# implicit stage ("The implicit stage solve" in `CODE.md`).
#
# The stage problems are built backwards from their solution, so that each has
# a known answer: a fluid state and a field `E` in a random boosted Kerr-Schild
# metric are drawn first, and the predictor is what the stage equation says it
# must have been, `E★ = E + c W [E + v × B − (v·E) v]` with the totals
# `D, S_i, τ` of `(prim, E, B)`. `E` is `−v × B` plus a non-ideal part of up to
# `|B|/(1 + cW)` in a random direction, so that `E★` departs from the ideal
# field by up to `|B|` at every `σΔt`. Across the sweep the predictor's own
# fluid share — the totals less the energy and momentum of `E★` — is often
# not a fluid state at all, which is the inadmissible `u★` the IMEX contract
# warns of.
#
# The claims are measured against the stage equation itself, evaluated with
# the public pieces (`prim2con`, `ampere_stiff_source`), and not against the
# solve's own residual. The states come from `recovery_tests.jl`'s generators
# (`rec_metric`, `rec_direction`, `rec_gas`), included before this file.

using KernelAbstractions: @kernel, @index, CPU, synchronize
using Random: Xoshiro
using StaticArrays: SVector, SMatrix
import TreeGRRMHD.IRK

const ST = TreeGRRMHD

st_norm(m, a) = sqrt(ST.dot_up(m, a, a))

# A stage problem with a known solution, at `σΔt` (with `γΔt = 3/10`), the
# magnetization `b²/ρ` (the ideal comoving `b² = B²/W² + (B·v)²`) and the
# Lorentz factor `W`. Returns the solution `(prim, E)`, the predictor `E★`,
# the field `B`, the totals `cons★`, the model and `γΔt`. The model is
# `Constant(σΔt/γΔt)`, or `model_of(ρ)` of the drawn density.
function st_problem(rng, eos, m, σΔt, b²ρ, W; model_of=nothing)
    ρ = exp10(-6 + 3 * rand(rng))
    s = 4.1 + 1.8 * rand(rng)
    yₑ = 0.06 + 0.5 * rand(rng)
    v = sqrt(1 - 1 / W^2) * rec_direction(rng, m)
    prim = primitive_state(eos, m, ρ, v, s, yₑ)
    B̂ = rec_direction(rng, m)
    Bmag = sqrt(b²ρ * ρ / (1 / W^2 + ST.dot_up(m, v, B̂)^2))
    B = Bmag * B̂
    γΔt = 0.3
    model = model_of === nothing ? Constant(σΔt / γΔt) : model_of(ρ)
    c = γΔt * m.α * conductivity(model, prim)
    E = ideal_E(m, v, B) + (rand(rng) * Bmag / (1 + c * W)) * rec_direction(rng, m)
    E★ = E + c * ST.conduction_drift(m, prim, E, B)
    return prim, E, E★, B, prim2con(eos, m, prim, E, B), model, γΔt
end

# The full system at a returned `(E, prim)`: Ohm's law, as `𝔈 − 𝔈★ − γΔt ·
# ampere_stiff_source`, relative to the size of its terms; and the matter,
# `prim2con(prim, E, B)` against the totals, relative to the energy density
# `τ + D` (as `recovery_tests.jl` measures the recovery's round trip).
function st_residual(eos, model, m, cons★, E★, B, γΔt, E, prim)
    src = γΔt * ampere_stiff_source(model, m, prim, E, B) / m.sqrtγ
    cW = γΔt * m.α * conductivity(model, prim) * ST.prim_W(prim)
    r_ohm = st_norm(m, E - E★ - src) /
            (st_norm(m, E★) + st_norm(m, E) + cW * (st_norm(m, E) + st_norm(m, B)))
    c = prim2con(eos, m, prim, E, B)
    sg = m.sqrtγ
    𝓔 = (ST.cons_τ(cons★) + ST.cons_D(cons★)) / sg
    dS = (ST.cons_S(c) - ST.cons_S(cons★)) / sg
    r_matter = max(abs(ST.cons_D(c) - ST.cons_D(cons★)) / sg, sqrt(ST.dot_down(m, dS, dS)),
                   abs(ST.cons_τ(c) - ST.cons_τ(cons★)) / sg) / 𝓔
    return r_ohm, r_matter
end

@testset "Ohm's law at a fixed velocity is solved in closed form, and tends to −v×B" begin
    # Guards the Sherman–Morrison solve every iteration stands on: a sign of
    # the `(E·v) v` term, `v` where `u = W v` is meant, or the metric left
    # out of a dot product would solve a different system — invisible at
    # `v = 0` or in flat space. Checked against a dense solve of
    # `(1 + cW) E − cW (E·v) v = E★ − cW v × B`, and in both limits.
    rng = Xoshiro(51)
    worst = worst_ideal = 0.0
    for _ in 1:300
        m = rec_metric(rng)
        u = 3 * rand(rng) * rec_direction(rng, m)
        W = sqrt(1 + ST.dot_up(m, u, u))
        v = u / W
        E★ = SVector{3}(randn(rng, 3))
        B = SVector{3}(randn(rng, 3))
        c = exp10(-3 + 11 * rand(rng))
        E = ohm_E(m, c, u, E★, B)
        M = (1 + c * W) * one(SMatrix{3,3,Float64}) - c * W * (v * (m.γ * v)')
        Ed = M \ (E★ + c * W * ideal_E(m, v, B))
        worst = max(worst, st_norm(m, E - Ed) / (st_norm(m, Ed) + st_norm(m, E★) / (1 + c * W)))
        # Stiff: `E → −v × B`, off by `M⁻¹(E★ + v × B)`. `M` is self-adjoint
        # in the metric, with the eigenvalues `1 + cW` across `v` and
        # `1 + c/W` along it, so the departure is at most `|E★ + v × B|`
        # divided by `1 + c/W`.
        Eid = ideal_E(m, v, B)
        Es = ohm_E(m, 1e8, u, E★, B)
        worst_ideal = max(worst_ideal,
                          st_norm(m, Es - Eid) * (1 + 1e8 / W) / st_norm(m, E★ - Eid))
        # No conductivity: the predictor, bit for bit, whatever `u` is.
        @test ohm_E(m, 0.0, u, E★, B) === E★
        @test ohm_E(m, 0.0, SVector(NaN, 0.0, 0.0), E★, B) === E★
    end
    @info "ohm_E against a dense solve; the stiff limit's departure from −v×B in units of " *
          "its bound" worst worst_ideal
    # Measured 2.1e-15; and the bound attained, to rounding: 1 + 1.4e-9.
    @test worst < 2e-14
    @test worst_ideal < 1 + 1e-6
end

@testset "The Newton Jacobian is the derivative of the iteration" begin
    # Guards the analytic Jacobian Newton steps with ("The implicit stage
    # solve" in `CODE.md`): a wrong term in the recovery's derivative (EntropyEOS's
    # own `(s, w)` Jacobian, eliminated for `∂u/∂S` and `∂u/∂τ`) or in Ohm's
    # (`∂E/∂u` by implicit differentiation) would leave Newton converging
    # linearly or not at all, while every converged answer stayed right — so
    # only this test would see it. Against central differences: of the
    # recovery at a tight tolerance, with a step of `10⁻⁷ τ` so that the
    # thermal energy of a cold state is not pushed across the entropy window;
    # and of `ohm_E`.
    eos = rec_gas(2)
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    tight = REC_EOS.Con2PrimOptions{Float64}(; tol=1e-14)
    rng = Xoshiro(52)
    e(k) = SVector(ntuple(i -> i == k ? 1.0 : 0.0, 3))
    worst_rec = worst_ohm = 0.0
    for _ in 1:200
        m = rec_metric(rng)
        prim, _, E★, B, cons, _, _ = st_problem(rng, eos, m, exp10(-3 + 11 * rand(rng)),
                                                exp10(-4 + 6 * rand(rng)), 1 + 9 * rand(rng))
        D = ST.cons_D(cons) / m.sqrtγ
        DYₑ = ST.cons_DYₑ(cons) / m.sqrtγ
        S_f, τ_f = ST.fluid_share(eos, m, prim)
        recover(S, τ) = ST.fluid_recovery(eos, tight, pol, m, D, S, τ, DYₑ, cold_guesses(Float64))
        p0, f0, pt = recover(S_f, τ_f)
        @test f0 == 0
        du_dS, du_dτ = ST.recovery_jacobian(m, D, p0, pt)
        h = 1e-7 * τ_f
        u(S, τ) = ST.four_velocity(recover(S, τ)[1])
        J = hcat([(u(S_f + h * e(k), τ_f) - u(S_f - h * e(k), τ_f)) / 2h for k in 1:3]...)
        jτ = (u(S_f, τ_f + h) - u(S_f, τ_f - h)) / 2h
        sc = maximum(abs.(du_dS))
        worst_rec = max(worst_rec, maximum(abs.(J - du_dS)) / sc, maximum(abs.(jτ - du_dτ)) / sc)

        c = m.α * exp10(-3 + 11 * rand(rng))
        u0 = ST.four_velocity(prim)
        E = ohm_E(m, c, u0, E★, B)
        dE_du = ST.ohm_jacobian(m, c, u0, E, B)
        hu = 1e-6 * sqrt(1 + ST.dot_up(m, u0, u0))
        Jo = hcat([(ohm_E(m, c, u0 + hu * e(k), E★, B) - ohm_E(m, c, u0 - hu * e(k), E★, B)) / 2hu
                   for k in 1:3]...)
        worst_ohm = max(worst_ohm, maximum(abs.(Jo - dE_du)) / maximum(abs.(dE_du)))
    end
    @info "Analytic Jacobians against central differences: recovery, Ohm" worst_rec worst_ohm
    # Measured 1.4e-8 and 3.0e-8, the differences' own truncation error.
    @test worst_rec < 1e-6
    @test worst_ohm < 1e-6
    @test ST.ohm_jacobian(flat_metric(), 0.0, SVector(0.3, 0.0, 0.0), SVector(1.0, 2.0, 3.0),
                          SVector(0.0, 1.0, 0.0)) == zero(SMatrix{3,3})
end

@testset "σ = 0: E = E★ exactly, and the primitives are exactly the hydro recovery" begin
    # Guards the acceptance's first item: where there is no conductivity —
    # `Vacuum`, and `DensityProfile` at and below its cutoff — the stage is
    # the identity on `𝔈`, and the recovery is the one every right-hand side
    # makes, bit for bit, flags included. A solve that iterated anyway would
    # move `E` by rounding.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(53)
    for k in 1:100
        m = rec_metric(rng)
        prim, _, E★, B, cons, _, γΔt = st_problem(rng, eos, m, 1.0, exp10(-4 + 8 * rand(rng)),
                                                  1 + 9 * rand(rng))
        # `DensityProfile` vanishes at `ρ ≤ 2ρ_atm`, and no recovered
        # density exceeds `D` (as `W ≥ 1`): at the cutoff and below it.
        D = ST.cons_D(cons) / m.sqrtγ
        g = k % 2 == 0 ? cold_guesses(Float64) : guesses(prim)
        for model in (Vacuum(), DensityProfile(1e6, D / 2), DensityProfile(1e6, D))
            E, q, r, n = solve_ohm_stage(eos, opts, pol, model, m, cons, E★, B, γΔt, g)
            qh, fh = con2prim_hydro(eos, opts, pol, m, cons, E★, B, g)
            @test E === E★
            @test isequal(q, qh)
            @test r == OhmStageResult(OHM_SIGMA_ZERO, fh)
            @test n == 1
        end
    end
end

@testset "IdealConductor: E = −v×B and the ideal recovery's primitives" begin
    # Guards the acceptance's second item: the `σ = ∞` branch is the ideal
    # recovery, not an iteration at a large σ, and its `E` is the ideal field
    # of the recovered velocity — so on a state built with `E = −v × B` it
    # returns that field, to the recovery's accuracy, whatever `E★` is.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(54)
    worst = 0.0
    for _ in 1:200
        m = rec_metric(rng)
        prim, _, _, B, _, _, γΔt = st_problem(rng, eos, m, 1.0, exp10(-4 + 8 * rand(rng)),
                                              1 + 9 * rand(rng))
        Eid = ideal_E(m, ST.prim_v(prim), B)
        cons = prim2con(eos, m, prim, Eid, B)
        E★ = SVector{3}(randn(rng, 3))
        E, q, r, n = solve_ohm_stage(eos, opts, pol, IdealConductor(), m, cons, E★, B, γΔt,
                                     cold_guesses(Float64))
        qi, fi = con2prim_ideal(eos, opts, pol, m, cons, B, cold_guesses(Float64))
        @test isequal(q, qi) && r == OhmStageResult(OHM_IDEAL, fi) && fi == 0 && n == 1
        @test E == ideal_E(m, ST.prim_v(q), B)
        worst = max(worst, st_norm(m, E - Eid) / st_norm(m, B))
    end
    @info "IdealConductor: |E − (−v×B)| / |B| at the true v" worst
    # Measured 3.3e-12: the ideal recovery's velocity error.
    @test worst < 1e-10
end

@testset "A large σΔt approaches the ideal answer at the rate 1/(σΔt)" begin
    # Guards the two branches against each other: `Constant(σ)` at large
    # `σΔt` solves the same stage that `IdealConductor` solves in the limit,
    # so the two answers must part only by the predictor's share
    # `~ |E★ + v×B| / (cW)` — L-stability of the stage, seen pointwise.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(55)
    worst = 0.0
    for _ in 1:100, σΔt in (1e4, 1e6, 1e8)
        m = rec_metric(rng)
        prim, _, _, B, _, _, γΔt = st_problem(rng, eos, m, 1.0, exp10(-4 + 6 * rand(rng)),
                                              1 + 9 * rand(rng))
        Eid = ideal_E(m, ST.prim_v(prim), B)
        cons = prim2con(eos, m, prim, Eid, B)
        E★ = Eid + st_norm(m, B) * rec_direction(rng, m)
        E, q, r, _ = solve_ohm_stage(eos, opts, pol, Constant(σΔt / γΔt), m, cons, E★, B, γΔt,
                                     cold_guesses(Float64))
        @test ST.converged(r) && r.policy_flags == 0
        Ei, _, _, _ = solve_ohm_stage(eos, opts, pol, IdealConductor(), m, cons, E★, B, γΔt,
                                      cold_guesses(Float64))
        # `M⁻¹`'s largest eigenvalue, `1/(1 + c/W)`, as in `ohm_E`'s test;
        # the fluid's response to the departure moves `v` too.
        c_W = σΔt * m.α / ST.prim_W(q)
        worst = max(worst, st_norm(m, E - Ei) * (1 + c_W) / st_norm(m, E★ - Eid))
    end
    @info "|E(σΔt) − E(∞)| · (1 + c/W) / |E★ + v×B|" worst
    # Measured 1.0.
    @test worst < 10
end

# The σΔt × b²/ρ × W sweep, over every Γ: the branch each point takes, the
# recoveries it needs, and the residuals.
function st_sweep(Γ, nper)
    eos = rec_gas(Γ)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(56)
    points = NamedTuple[]
    for lσ in -3:8, lb in -4:4, W in (1.0, 2.0, 10.0), _ in 1:nper
        m = rec_metric(rng)
        prim, Esol, E★, B, cons, model, γΔt = st_problem(rng, eos, m, exp10(lσ), exp10(lb), W)
        E, q, r, n = solve_ohm_stage(eos, opts, pol, model, m, cons, E★, B, γΔt,
                                     cold_guesses(Float64))
        r_ohm, r_matter = st_residual(eos, model, m, cons, E★, B, γΔt, E, q)
        # Whether either seed is a fluid state: the predictor's recovery at
        # `E★` and the ideal recovery.
        _, f★ = con2prim_hydro(eos, opts, pol, m, cons, E★, B, cold_guesses(Float64))
        _, fI = con2prim_ideal(eos, opts, pol, m, cons, B, cold_guesses(Float64))
        push!(points,
              (; lσ, lb, W, flags=r.flags, n, finite=all(isfinite, E) && all(isfinite, q[1:10]),
               r_ohm, r_matter, seeded=f★ == 0 || fI == 0,
               dE=st_norm(m, E - Esol) / (st_norm(m, Esol) + st_norm(m, B))))
    end
    return points
end

st_branch(f) = f & OHM_FALLBACK != 0 ? "fallback" :
               f & OHM_FIXED_POINT != 0 ? "fixed point" :
               f & OHM_IDEAL_SEED != 0 ? "Newton, ideal seed" : "Newton"

@testset "The σΔt × b²/ρ × W sweep: residual ≤ 1e-10 wherever the solve converged" begin
    # Guards the stage solve as a whole, over the grid of the acceptance —
    # `σΔt ∈ 10^{−3…8}`, `b²/ρ ∈ 10^{−4…4}`, `W ∈ {1, 2, 10}` — at every Γ,
    # eight random problems a point: a residual left by a wrong branch, a
    # NaN from a clamped recovery inside Newton, and a change that moves
    # where each branch is taken or what it costs. The branch map and the
    # counts are "(measured in step 3)" in `CODE.md`.
    for Γ in REC_GAMMAS
        pts = st_sweep(Γ, 8)
        conv = filter(p -> p.flags & OHM_FALLBACK == 0, pts)
        fb = filter(p -> p.flags & OHM_FALLBACK != 0, pts)
        @test all(p -> p.finite, pts)
        worst_ohm = maximum(p -> p.r_ohm, conv)
        worst_matter = maximum(p -> p.r_matter, conv)
        # Measured at most 2.7e-16 (Ohm's law holds by construction between
        # the returned E and the velocity it was solved at) and 1.3e-12 (the
        # solve's 1e-12 of τ + D, plus the recovery's own).
        @test worst_ohm <= 1e-10
        @test worst_matter <= 1e-10

        counts = Dict{String,Int}()
        most = Dict{String,Int}()
        for p in pts
            b = st_branch(p.flags)
            counts[b] = get(counts, b, 0) + 1
            most[b] = max(get(most, b, 0), p.n)
        end
        clean = filter(p -> p.flags & (OHM_REPAIRED | OHM_BACKTRACKED | OHM_SWITCHED) == 0, conv)
        most_clean = maximum(p -> p.n, clean)
        fallback_region = sort(unique((p.lσ, p.lb, p.W) for p in fb))
        other_root = count(p -> p.dE > 1e-8, conv)
        seeded_fallbacks = count(p -> p.seeded, fb)
        @info "Stage solve sweep, Γ = $Γ: points and most recoveries per branch, most " *
              "recoveries without a repair or a backtrack, the fallback region " *
              "(log₁₀ σΔt, log₁₀ b²/ρ, W), fallbacks with a fluid seed, converged to another root" seeded_fallbacks string(sort(collect(counts))) string(sort(collect(most))) most_clean string(fallback_region) other_root worst_ohm worst_matter

        # Where the field barely couples to the fluid, the fixed point does
        # it all: b²/ρ ≤ 1e-3 at every σΔt (measured: every point).
        @test all(p -> p.flags & OHM_FIXED_POINT != 0, filter(p -> p.lb <= -3, pts))
        # Without a repair or a backtrack, a few recoveries: measured at most
        # 6 (fixed point) and 10 (Newton, seeds included) over the three Γ.
        @test maximum(p -> p.n, filter(p -> p.flags & OHM_FIXED_POINT != 0, clean)) <= 10
        @test most_clean <= 16
        # The fallback is the strongly magnetized corner at intermediate
        # σΔt, where both seeds needed a repair: measured only at b²/ρ ≥ 10
        # and σΔt ≤ 1e2, on 36, 78 and 103 of the 2592 points (Γ = 4/3, 5/3,
        # 2; at most 4.0%), and on one (Γ = 5/3) where a seed was a fluid
        # state. A few points converge to another root of the stage
        # equations than the one the problem was built from (5, 2 and 2), to
        # the same residual: the stage has more than one solution there.
        @test all(p -> p.lb >= 1 && p.lσ <= 2, fb)
        @test length(fb) <= 0.08 * length(pts)
        @test seeded_fallbacks <= 3
        # And it is flagged as such: no fallback is reported converged.
        @test all(p -> !ST.converged(OhmStageResult(p.flags, 0)), fb)
    end
end

@testset "DensityProfile: the conductivity follows the recovered density" begin
    # Guards the density-dependent model inside the iteration: σ is
    # re-evaluated at each iterate's recovery (it is held fixed only inside
    # one Newton step), so the converged state must satisfy Ohm's law with
    # the σ of its own density — including near the cutoff, where σ is a
    # steep function of ρ, and across it, where the branch is the vacuum's.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(57)
    worst = 0.0
    nσ0 = npts = nfb = 0
    for _ in 1:300
        m = rec_metric(rng)
        # σ₀Δt up to 1e8, with the cutoff density within a factor of three
        # of the state's: σ from zero to σ₀.
        σΔt = exp10(-2 + 10 * rand(rng))
        W = 1 + 3 * rand(rng)
        f = exp10(-0.5 + rand(rng))
        prim, _, E★, B, cons, model, γΔt =
            st_problem(rng, eos, m, σΔt, exp10(-3 + 4 * rand(rng)), W;
                       model_of=ρ -> DensityProfile(σΔt / 0.3, f * ρ / 2))
        E, q, r, _ = solve_ohm_stage(eos, opts, pol, model, m, cons, E★, B, γΔt,
                                     cold_guesses(Float64))
        @test all(isfinite, E)
        r.flags & OHM_SIGMA_ZERO != 0 && (nσ0 += 1; continue)
        npts += 1
        ST.converged(r) || (nfb += 1)
        if ST.converged(r)
            r_ohm, r_matter = st_residual(eos, model, m, cons, E★, B, γΔt, E, q)
            worst = max(worst, r_ohm, r_matter)
        end
    end
    @info "DensityProfile near its cutoff: worst residual; points at σ = 0, with σ > 0, " *
          "and of those not converged" worst nσ0 npts nfb
    @test worst <= 1e-10
    @test nσ0 > 0
    # Measured 1 of 148: a Newton iteration held at the entropy window's edge
    # (see "The implicit stage solve" in `CODE.md`); 3 before the Jacobian
    # carried σ's dependence on ρ.
    @test nfb <= 3
end

@testset "An inadmissible predictor is repaired inside the solve and flagged" begin
    # Guards the IMEX contract's `u★ may be inadmissible`: a predictor with
    # less energy than the field of `E★` carries, or with a non-finite slot,
    # must give a finite `E` and a flagged result, not a NaN or an exception;
    # and the predictor's repair must not leak into the answer, which for a
    # recoverable stage is the true solution.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    F = REC_EOS
    rng = Xoshiro(58)
    worst = 0.0
    n_bad = 0
    for _ in 1:200
        m = rec_metric(rng)
        # Strong Joule heating: `E★` carries much more energy than the fluid
        # has, so the predictor's share is below the coldest state.
        prim, Esol, _, B, cons, model, γΔt = st_problem(rng, eos, m, 10.0, 1.0, 2.0)
        c = γΔt * m.α * conductivity(model, prim)
        E★ = Esol + c * ST.conduction_drift(m, prim, Esol, B)
        τ_f = ST.cons_τ(prim2con(eos, m, prim, zero(Esol), zero(Esol))) / m.sqrtγ
        E★ = E★ + sqrt(10 * τ_f) * rec_direction(rng, m)
        _, f★ = con2prim_hydro(eos, opts, pol, m, cons, E★, B, cold_guesses(Float64))
        f★ == 0 && continue
        n_bad += 1
        E, q, r, _ = solve_ohm_stage(eos, opts, pol, model, m, cons, E★, B, γΔt,
                                     cold_guesses(Float64))
        @test all(isfinite, E) && all(isfinite, q[1:10])
        if ST.converged(r) && r.policy_flags == 0
            worst = max(worst, st_residual(eos, model, m, cons, E★, B, γΔt, E, q)...)
        end
    end
    @info "Inadmissible predictors: how many, worst residual of their solutions" n_bad worst
    @test n_bad >= 100
    @test worst <= 1e-10

    # Non-finite input: no exception, and a finite `E` from a finite `E★`
    # where the conductivity vanishes at the policy's atmosphere.
    m = rec_metric(rng)
    prim, Esol, E★, B, cons, _, γΔt = st_problem(rng, eos, m, 1.0, 1.0, 2.0)
    bad = ntuple(i -> i == ST.U_TAU ? NaN : cons[i], NU)
    for model in (Constant(10.0), DensityProfile(10.0, 1e-9))
        E, q, r, _ = solve_ohm_stage(eos, opts, pol, model, m, bad, E★, B, γΔt,
                                     cold_guesses(Float64))
        @test r.policy_flags & F.FLAG_POL_NONFINITE != 0
        @test all(isfinite, q[1:10])
    end
end

# One cell, resting, with no field: Ohm's law alone decays `E` as
# `exp(−ασt)`, and the field's energy heats the fluid at fixed total `τ`.
# `u` is the cell's ten variables; the explicit part is zero.
struct STCell{M,E,O,P}
    m::M
    eos::E
    opts::O
    pol::P
    model::Constant{Float64}
end

function st_solve_imp!(U, u★, γΔt, p, t)
    cons★ = ntuple(i -> u★[i], NU)
    E★ = ST.cons_𝔈(cons★) / p.m.sqrtγ
    E, _, r, _ = solve_ohm_stage(p.eos, p.opts, p.pol, p.model, p.m, cons★, E★, zero(E★), γΔt,
                                 cold_guesses(Float64))
    ST.converged(r) || error("the stage solve did not converge: $r")
    𝔈 = p.m.sqrtγ * E
    for k in 1:3
        U[ST.U_E + k - 1] = 𝔈[k]                 # only `𝔈`; `U` holds `u★` on entry
    end
    return nothing
end

@testset "Under IMEXSSP3433 a resting cell's E decays as exp(−ασt), its matter bit for bit" begin
    # Guards the stage solve as `solve_imp!` sees it (IMEXRungeKutta's
    # `CODE.md`, "The callback contracts"): `U = u★ + γΔt g(U)` with only
    # `𝔈` written, so the integrator's implicit tendency `(U − u★)/(γΔt)`
    # is zero in every other slot and `D`, `S_i`, `τ`, `D Yₑ` and `D s` come
    # back bit for bit. With no field and no flow, Ampère's source is
    # `−ασ𝔈`, and the step is SSP3(4,3,3)'s L-stable DIRK on it: third
    # order, and after one step at `ασΔt = 10⁶` the field is gone.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    m = rec_metric(Xoshiro(59))
    prim = primitive_state(eos, m, 1e-4, zero(SVector{3,Float64}), 4.5, 0.3)
    τ_f = ST.cons_τ(prim2con(eos, m, prim, zero(SVector{3,Float64}), zero(SVector{3,Float64})))
    E0 = sqrt(0.5 * τ_f / m.sqrtγ) * rec_direction(Xoshiro(60), m)
    u0 = collect(prim2con(eos, m, prim, E0, zero(E0)))
    f_exp! = (du, u, p, t) -> (du .= 0; nothing)
    run(σ, n) = begin
        p = STCell(m, eos, opts, pol, Constant(σ))
        prob = IRK.IMEXProblem(f_exp!, st_solve_imp!, copy(u0), (0.0, 1.0), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=1 / n)
        IRK.solve!(integ)
        integ.u
    end
    σ = 2 / m.α                                   # ασ = 2
    errs = map((10, 20, 40)) do n
        u = run(σ, n)
        @test all(i -> u[i] === u0[i], 1:(ST.U_E - 1))
        𝔈 = SVector(u[ST.U_E], u[ST.U_E + 1], u[ST.U_E + 2])
        st_norm(m, 𝔈 - exp(-2.0) * m.sqrtγ * E0) / st_norm(m, m.sqrtγ * E0)
    end
    rates = log2.(errs[1:2] ./ errs[2:3])
    @info "IMEXSSP3433 on a resting cell: errors at 10, 20, 40 steps, and the orders" errs rates
    # Measured orders 2.9 and 3.0.
    @test all(r -> 2.7 < r < 3.3, rates)
    u = run(1e6 / m.α, 1)
    𝔈 = SVector(u[ST.U_E], u[ST.U_E + 1], u[ST.U_E + 2])
    @test st_norm(m, 𝔈) <= 1e-5 * st_norm(m, m.sqrtγ * E0)
    @test all(i -> u[i] === u0[i], 1:(ST.U_E - 1))
end

@kernel function st_kernel!(Es, prims, results, iters, eos, opts, pol, @Const(σs), @Const(metrics),
                            @Const(conss), @Const(E★s), @Const(Bs), γΔt)
    i = @index(Global)
    E, q, r, n = solve_ohm_stage(eos, opts, pol, Constant(σs[i]), metrics[i], conss[i], E★s[i],
                                 Bs[i], γΔt, cold_guesses(typeof(γΔt)))
    Es[i] = E
    prims[i] = q
    results[i] = r
    iters[i] = n
end

@testset "The stage solve runs inside a CPU kernel, bit for bit: T=$T" for T in (Float64, Float32)
    # Guards the kernel-callability the RHS's `solve_imp!` will rely on
    # (step 5): the solve, its options and its result are `isbits`, nothing
    # throws, allocates or fails to infer, and a kernel computes what the
    # host computes, on every branch — the fixed point, Newton from either
    # seed, and the fallback of the strongly magnetized corner. `Float32` is
    # the leak detector, and a smoke test of the solve itself.
    eos64 = rec_gas(2)
    eos = REC_EOS.narrow(eos64, T)
    opts = REC_EOS.Con2PrimOptions{T}()
    pol = REC_EOS.default_policy(eos, T(REC_ρ_ATM))
    rng = Xoshiro(61)
    σs, metrics, conss, E★s, Bs = T[], Metric3{T}[], NTuple{NU,T}[], SVector{3,T}[], SVector{3,T}[]
    exact = SVector{3,Float64}[]
    for lσ in (-3, 0, 3, 8), lb in (-3, -1, 0, 3), W in (1.0, 2.0, 10.0)
        m64 = rec_metric(rng)
        _, Esol, E★, B, cons, model, γΔt = st_problem(rng, eos64, m64, exp10(lσ), exp10(lb), W)
        push!(σs, T(model.σ₀))
        push!(metrics, Metric3(T(m64.α), T.(m64.β), T.(m64.γ)))
        push!(conss, T.(cons))
        push!(E★s, T.(E★))
        push!(Bs, T.(B))
        push!(exact, Esol)
    end
    γΔt = T(3 // 10)
    n = length(metrics)
    host = [solve_ohm_stage(eos, opts, pol, Constant(σs[i]), metrics[i], conss[i], E★s[i], Bs[i],
                            γΔt, cold_guesses(T)) for i in 1:n]
    Es = Vector{SVector{3,T}}(undef, n)
    prims = Vector{NTuple{NPRIM,T}}(undef, n)
    results = Vector{OhmStageResult}(undef, n)
    iters = Vector{Int}(undef, n)
    backend = CPU()
    st_kernel!(backend)(Es, prims, results, iters, eos, opts, pol, σs, metrics, conss, E★s, Bs, γΔt;
                        ndrange=n)
    synchronize(backend)
    @test isequal(Es, [h[1] for h in host])
    @test isequal(prims, [h[2] for h in host])
    @test results == [h[3] for h in host]
    @test iters == [h[4] for h in host]
    @test eltype(Es) === SVector{3,T}
    @test all(E -> all(isfinite, E), Es)
    branches = reduce(|, r.flags for r in results)
    @test branches & (OHM_FIXED_POINT | OHM_NEWTON | OHM_IDEAL_SEED) ==
          OHM_FIXED_POINT | OHM_NEWTON | OHM_IDEAL_SEED
    @test isbits(OhmStageOptions{T}()) && isbits(results[1])

    # Allocation-free and inferred on the host, at every model.
    m, cons, E★, B = metrics[1], conss[1], E★s[1], Bs[1]
    for mdl in (Vacuum(), Constant(T(10)), DensityProfile(T(10), T(1e-9)), IdealConductor())
        @test (@inferred solve_ohm_stage(eos, opts, pol, mdl, m, cons, E★, B, γΔt,
                                         cold_guesses(T))) isa
              Tuple{SVector{3,T},NTuple{NPRIM,T},OhmStageResult,Int}
        @test rec_allocations(solve_ohm_stage, eos, opts, pol, mdl, m, cons, E★, B, γΔt,
                              cold_guesses(T)) == 0
    end

    # Where it converged to the known answer's neighbourhood, it lands on it
    # to the solve's tolerance; at `Float32` that is `1024 eps = 1.2e-4`.
    conv = [i for i in 1:n if ST.converged(results[i]) && results[i].policy_flags == 0]
    @test length(conv) >= 3n ÷ 4
    worst = maximum(conv) do i
        m64 = Metric3(Float64(metrics[i].α), Float64.(metrics[i].β), Float64.(metrics[i].γ))
        st_norm(m64, Float64.(Es[i]) - exact[i]) /
        (st_norm(m64, exact[i]) + st_norm(m64, Float64.(Bs[i])))
    end
    @info "The stage solve in a kernel, T = $T: converged cells, and their worst error against " *
          "the known answer" length(conv) n worst
    # Measured 6.3e-13 and 1.3e-4.
    @test worst < (T === Float64 ? 1e-10 : 2e-3)
end
