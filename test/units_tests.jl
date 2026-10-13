# The units: the geometric unit system the scheme computes in, and the one
# density factor (and `κ`) between it and an EOS's run units.
#
# A wrong unit is the error that survives every self-consistency test: a
# recovery that round-trips in the wrong units round-trips perfectly. So the
# claims here are against numbers from outside the package — the solar
# density unit everyone quotes — and against the same physics written in two
# unit systems: an ideal gas given in g/cm³ and seen through the adapter
# must be the same gas as the one written in geometric units, and a table
# recovered in geometric units must agree with the same table recovered in
# its own. See "Thermodynamics" in `CODE.md`.

using Random: Xoshiro
using StaticArrays: SVector

const UNITS_EOS = TreeGRRMHD.EntropyEOS

@testset "The geometric units are the solar ones everyone quotes" begin
    # Guards a constant typed wrong or a power of c misplaced. The density
    # unit `c⁶/(G (GM☉)²)` is the one factor between the scheme and a table
    # in g/cm³, so an error in it moves every star's central density. The
    # reference values are the usual ones of the numerical-relativity
    # literature (e.g. the Einstein Toolkit's), 6.176e17 g/cm³, 1.4766 km
    # and 4.9255 μs; the step-1 acceptance asks for the density to 1e-4.
    @test isapprox(MSUN_DENSITY_CGS, 6.176e17; rtol=1e-4)
    @test isapprox(TreeGRRMHD.MSUN_LENGTH_CGS, 1.4766e5; rtol=1e-4)
    @test isapprox(TreeGRRMHD.MSUN_TIME_CGS, 4.9255e-6; rtol=1e-4)
    @test isapprox(TreeGRRMHD.MSUN_MASS_CGS, 1.9884e33; rtol=1e-4)
    # And the three are one unit system: a density is a mass over a length
    # cubed, and c is a length over a time.
    @test isapprox(MSUN_DENSITY_CGS,
                   TreeGRRMHD.MSUN_MASS_CGS / TreeGRRMHD.MSUN_LENGTH_CGS^3; rtol=4eps())
    @test isapprox(TreeGRRMHD.MSUN_LENGTH_CGS / TreeGRRMHD.MSUN_TIME_CGS, TreeGRRMHD.C_CGS;
                   rtol=4eps())
    @info "The solar density unit (measured in step 1)" MSUN_DENSITY_CGS
end

@testset "The units adapter round-trips: T=$T" for T in (Float64, Float32)
    # Guards a conversion and its inverse that disagree — a `to_eos` that
    # multiplies where `from_eos` multiplies too — and the `κ` direction:
    # `ρ★ = κρ`, so the physical density is `ρ★/κ`, never `κρ★`.
    units = EOSUnits{T}(; density=MSUN_DENSITY_CGS, κ=T(0.9999234707050715))
    @test isbits(units)
    rng = Xoshiro(11)
    worst = 0.0
    for _ in 1:1000
        x = T(exp10(-12 + 10 * rand(rng)))
        y = TreeGRRMHD.to_eos(units, x)
        @test y isa T
        worst = max(worst, abs(TreeGRRMHD.from_eos(units, y) - x) / x)
        ρ = TreeGRRMHD.rest_mass_density(units, x)
        worst = max(worst, abs(TreeGRRMHD.rescaled_density(units, ρ) - x) / x)
        @test ρ > x                       # κ < 1: the physical density is larger
    end
    # Two roundings each way, so two ulps.
    @test worst <= 2eps(T)
    @test TreeGRRMHD.to_eos(units, one(T)) == T(MSUN_DENSITY_CGS)

    # The units an EOS needs are its own κ and the caller's density unit;
    # the analytic EOSes have κ = 1.
    ideal = UNITS_EOS.IdealGasEOS{Float64}(; Γ=2.0, K_ref=100.0, s_ref=5.0,
                                           s_window=(4.0, 6.0), ρ_bounds=(1e-10, 1e-2),
                                           yₑ_bounds=(0.05, 0.6))
    @test EOSUnits(ideal; density=1).κ === 1.0

    # Nonsense is refused, saying why.
    @test_throws ArgumentError EOSUnits{T}(; density=0, κ=1)
    @test_throws ArgumentError EOSUnits{T}(; density=-1, κ=1)
    @test_throws ArgumentError EOSUnits{T}(; density=Inf, κ=1)
    @test_throws ArgumentError EOSUnits{T}(; density=1, κ=0)
    @test_throws ArgumentError EOSUnits{T}(; density=1, κ=NaN)
end

# The TOV star's gas, `p = 100 ρ²` in geometric units (K = 100, Γ = 2), and the
# same gas written in g/cm³: `p_cgs/f = K f^{Γ−1} (ρ_cgs/f)^Γ` with `f` the
# density unit, so `K_cgs = K f^{1−Γ}`; `U` is dimensionless and unchanged.
function units_twin_gases(Γ)
    f = MSUN_DENSITY_CGS
    box = (; s_ref=5.0, s_window=(4.0, 6.0), yₑ_bounds=(0.05, 0.6))
    geo = UNITS_EOS.IdealGasEOS{Float64}(; Γ=Γ, K_ref=100.0, ρ_bounds=(1e-10, 1e-2), box...)
    cgs = UNITS_EOS.IdealGasEOS{Float64}(; Γ=Γ, K_ref=100.0 * f^(1 - Γ),
                                         ρ_bounds=(1e-10 * f, 1e-2 * f), box...)
    return geo, GeometricEOS(cgs; density=f)
end

@testset "A gas in g/cm³ is, through the adapter, the gas in geometric units: Γ=$Γ" for
        Γ in (4 // 3, 5 // 3, 2)
    # Guards a derivative scaled by the wrong power of the density unit. The
    # recovery's Jacobian reads `U_ρ`, `U_ρs` and `cs²`, so a `U_ρρ` scaled
    # by `f` instead of `f²` would pass every value test and still slow the
    # Newton down; here every field of the EOS point is compared, at random
    # states, against the gas written directly in geometric units.
    geo, adapted = units_twin_gases(Float64(Γ))
    @test adapted isa UNITS_EOS.AbstractEOS{Float64}
    @test isbits(adapted)
    @test UNITS_EOS.κ(adapted) === 1.0
    @test all(isapprox.(UNITS_EOS.logρ_bounds(adapted), UNITS_EOS.logρ_bounds(geo);
                        atol=8eps()))
    @test UNITS_EOS.yₑ_bounds(adapted) == UNITS_EOS.yₑ_bounds(geo)
    rng = Xoshiro(12)
    worst = 0.0
    for _ in 1:200
        ρ = exp10(-9 + 7 * rand(rng))
        s = 4 + 2 * rand(rng)
        a = UNITS_EOS.evaluate(adapted, ρ, s, 0.3, NaN)
        b = UNITS_EOS.evaluate(geo, ρ, s, 0.3, NaN)
        for name in (:U, :U_ρ, :U_s, :U_ρρ, :U_ρs, :T̂, :p, :h, :cs²)
            x, y = getfield(a, name), getfield(b, name)
            worst = max(worst, abs(x - y) / max(abs(y), floatmin()))
        end
        @test a.flags == b.flags
        @test UNITS_EOS.srange(adapted, ρ, 0.3) == UNITS_EOS.srange(geo, ρ, 0.3)
        @test UNITS_EOS.srange_extended(adapted, ρ, 0.3) ==
              UNITS_EOS.srange_extended(geo, ρ, 0.3)
    end
    # `exp` of a sum of logs, one of which moved by log f ≈ 41: a few tens
    # of ulps (measured 9.1e-15 at Γ = 2 in step 1).
    @test worst < 1e-13
    @info "The adapter against the geometric gas, Γ = $Γ, worst relative EOS error" worst

    # And the policy derived from it is the same policy.
    pa = UNITS_EOS.default_policy(adapted, 1e-10)
    pg = UNITS_EOS.default_policy(geo, 1e-10)
    for name in (:ρ_atm, :ρ_ceiling, :w_cap, :D_max, :τ_max)
        @test isapprox(getfield(pa, name), getfield(pg, name); rtol=1e-13)
    end
end

@testset "A table recovered in geometric units agrees with the table in its own units" begin
    # Guards the adapter on the EOS it exists for: a table, in g/cm³, with
    # `κ ≠ 1`. The same conserved state is recovered twice — in geometric
    # units through `GeometricEOS`, and by EntropyEOS directly on the state
    # multiplied by the density unit — and the two answers must be one
    # state. The synthetic table is EntropyEOS's own (an ideal gas of
    # nucleons, Γ = 5/3, T ≤ 50 MeV), built in a fraction of a second.
    table = UNITS_EOS.build_eos(UNITS_EOS.make_synthetic_table())
    view = table.view
    f = MSUN_DENSITY_CGS
    g = GeometricEOS(view; density=f)
    κ = UNITS_EOS.κ(view)
    @test κ < 1                                   # the synthetic table re-zeroes
    @test EOSUnits(g).κ == κ
    @test EOSUnits(g).density == f

    opts = UNITS_EOS.Con2PrimOptions{Float64}()
    pol_geo = UNITS_EOS.default_policy(g, 1e7 / f)
    pol_cgs = UNITS_EOS.default_policy(view, 1e7)
    m = flat_metric(Float64)
    zero3 = zero(SVector{3,Float64})
    rng = Xoshiro(13)
    worst_eval = worst_ρ = worst_s = 0.0
    bad = 0
    for _ in 1:40
        ρ_cgs = exp10(8 + 6 * rand(rng))          # g/cm³, inside the table
        yₑ = 0.1 + 0.4 * rand(rng)
        sr = UNITS_EOS.srange(view, ρ_cgs, yₑ)
        s = sr.s_min + (0.1 + 0.8 * rand(rng)) * (sr.s_max - sr.s_min)

        # One evaluation, in both units: the density-like fields scale by
        # the density unit and its powers, and the rest are unchanged.
        a = UNITS_EOS.evaluate(g, ρ_cgs / f, s, yₑ, NaN)
        b = UNITS_EOS.evaluate(view, ρ_cgs, s, yₑ, NaN)
        for (x, y) in ((a.U, b.U), (a.U_ρ * f^-1, b.U_ρ), (a.U_s, b.U_s),
                       (a.U_ρρ * f^-2, b.U_ρρ), (a.U_ρs * f^-1, b.U_ρs), (a.p * f, b.p),
                       (a.h, b.h), (a.cs², b.cs²), (a.T_MeV, b.T_MeV))
            worst_eval = max(worst_eval, abs(x - y) / abs(y))
        end
        @test UNITS_EOS.srange(g, ρ_cgs / f, yₑ) == sr

        v = SVector(0.3, -0.2, 0.1) * rand(rng)
        prim = primitive_state(g, m, ρ_cgs / f, v, s, yₑ)
        cons = prim2con(g, m, prim, zero3, zero3)
        p_geo, fl_geo = con2prim_hydro(g, opts, pol_geo, m, cons, zero3, zero3,
                                       cold_guesses(Float64))
        S = TreeGRRMHD.cons_S(cons)
        cin = UNITS_EOS.Con2PrimIn{Float64}(cons[1] * f, cons[5] * f, cons[6] * f, 0.0,
                                            sqrt(sum(abs2, S)) * f, 0.0)
        out = UNITS_EOS.con2prim_safe(view, cin, opts, pol_cgs, NaN, NaN, NaN)
        (fl_geo == 0 && out.policy_flags == 0) || (bad += 1)
        # Two solves to the same tolerance from inputs a rounding apart
        # agree as two round trips do: conditioned, as in EntropyEOS's own
        # test (EOS-2 in `PLAN.md`; `recovery_tests.jl`).
        o = out.base
        W² = cosh(o.w)^2
        κ_ρ = o.eos.h * W²
        κ_s = (1 + cin.τ / (o.ρ * W² * o.eos.U)) * max(1, o.eos.U / (o.s * o.eos.U_s))
        worst_ρ = max(worst_ρ, max(abs(TreeGRRMHD.prim_ρ(p_geo) * f - o.ρ) / o.ρ,
                                   abs(TreeGRRMHD.prim_w(p_geo) - o.w) / max(o.w, 1)) / κ_ρ)
        worst_s = max(worst_s, abs(TreeGRRMHD.prim_s(p_geo) - o.s) / o.s / κ_s)
        # κ: the recovered density is ρ★; the physical one is ρ★/κ.
        @test TreeGRRMHD.rest_mass_density(EOSUnits(g), TreeGRRMHD.prim_ρ(p_geo)) ==
              TreeGRRMHD.prim_ρ(p_geo) / κ
    end
    @test bad == 0
    # The evaluation: the inner temperature solve sees a density one
    # rounding away, and lands within a rounding of the same value
    # (measured 2.0e-16 in step 1; the recovery, conditioned, 3.0e-15 in
    # (ρ, w) and 8.4e-13 in s).
    @test worst_eval < 1e-12
    @test worst_ρ < 3e-11
    @test worst_s < 1e-10
    @info "A table in geometric and in its own units: worst EOS field; conditioned " *
          "recovery (ρ, w), s" worst_eval worst_ρ worst_s
end
