# Step 16: the neutrino rates and opacities, pointwise (rung 8a; "Neutrinos",
# "Leakage" in `CODE.md`). Uses `recovery_states.jl`'s `rec_allocations`.
#
# The composition is `CompositionPoint`, built by hand, or from EntropyEOS's
# composition (EOS-12) on the synthetic table; DD2's is step 20's.

using KernelAbstractions: @kernel, @index, CPU, synchronize

const NR = TreeGRRMHD
const NR_UNITS = EOSUnits{Float64}(density=MSUN_DENSITY_CGS, κ=1.0)
const NR_MODEL = RJS96Rates(NR_UNITS; baryon_mass=NR.AMU_G)

nr_comp(::Type{T}=Float64; μ_e, μ̂, μ_n=-30.0, μ_p=-30.0, X_n=0.5, X_p=0.1, X_α=0.0,
        X_h=0.0, X_l=0.0, Ā=1.0, Z̄=0.0) where {T} =
    CompositionPoint{T}(; μ_e=T(μ_e), μ̂=T(μ̂), μ_n=T(μ_n), μ_p=T(μ_p), X_n=T(X_n), X_p=T(X_p),
                        X_α=T(X_α), X_h=T(X_h), X_l=T(X_l), Ā=T(Ā), Z̄=T(Z̄))

# ∫₀^∞ xᵏ/(exp(x − η) + 1) dx in BigFloat: tanh-sinh on [0, max(η, 0)] and
# exp-sinh above it, so that the step at x = η is a node of neither.
function nr_fermi_quadrature(k, η::BigFloat)
    f(x) = x^k / (exp(x - η) + 1)
    a = max(η, zero(η))
    h = big(1) / 32
    s = zero(η)
    if a > 0
        for i in -200:200
            t = i * h
            u = big(π) / 2 * sinh(t)
            s += h * a / 2 * big(π) / 2 * cosh(t) / cosh(u)^2 * f(a / 2 * (1 + tanh(u)))
        end
    end
    for i in -200:200
        t = i * h
        e = exp(big(π) / 2 * sinh(t))
        s += h * e * big(π) / 2 * cosh(t) * f(a + e)
    end
    return s
end

@testset "The Fermi integrals are within a few ulp of quadrature: T=$T" for T in
                                                                          (Float64, Float32)
    # Guards the accelerated series and the reflection formula (and with them
    # every rate): a wrong weight, sign or polynomial coefficient is off by far
    # more than an ulp somewhere on η ∈ [−60, 120].
    worst = 0.0
    setprecision(BigFloat, 128) do
        for k in 0:5, η in -60:1.37:120
            x = T(η)
            ref = nr_fermi_quadrature(k, big(x))
            worst = max(worst, Float64(abs(big(fermi(Val(k), x)) - ref) / ref / eps(T)))
        end
    end
    @info "Fermi integrals, worst error in ulp" T worst
    @test worst ≤ 4      # measured in step 16: 2.1 (Float64), 3.1 (Float32)
end

@testset "The Fermi integrals reach their limits" begin
    # Guards the two ends: the series' first term for η → −∞ and the
    # polynomial's leading term for η → ∞.
    for k in 0:5
        @test fermi(Val(k), -40.0) ≈ factorial(k) * exp(-40.0) rtol = 1e-15
        @test fermi(Val(k), 1e4) ≈ 1e4^(k + 1) / (k + 1) rtol = 1e-6
        @test fermi(Val(k), -800.0) == 0
    end
    @test fermi(Val(0), 0.0) ≈ log(2) rtol = 1e-15
    @test fermi(Val(1), 0.0) ≈ π^2 / 12 rtol = 1e-15
    @test_throws ArgumentError fermi(Val(6), 0.0)
    # The ratios stay finite where both integrals underflow (step 16: 0/0
    # there gave NaN opacities at η = −3200), with the limit a!/b!.
    @test NR.fermi_ratio(Val(5), Val(4), -3000.0) ≈ 5 rtol = 1e-15
    @test NR.fermi_ratio(Val(4), Val(2), -3000.0f0) ≈ 12 rtol = 1e-6
    for η in (-30.0, -0.5, 0.0, 0.5, 30.0)
        @test NR.fermi_ratio(Val(5), Val(3), η) ≈ fermi(Val(5), η) / fermi(Val(3), η) rtol = 1e-14
    end
end

@testset "RJS96's β-processes obey detailed balance in the non-degenerate limit" begin
    # Guards the capture rate's and the absorption opacity's prefactors,
    # blocking and nucleon fractions against each other: with non-degenerate
    # electrons and nucleons in β-equilibrium, ν_e emission is absorption at
    # the equilibrium density times e^{Q/T} (RJS96 neglects Q in the
    # neutrino's energy) — exactly, once the blocking factors are 1.
    for t in (1.0, 5.0, 20.0)
        X_n, X_p = 0.6, 0.4
        μ̂ = NR.Q_NP_MEV + t * log(X_n / X_p)
        c = nr_comp(; μ_e=-20t, μ̂, X_n, X_p)
        r = neutrino_rates(NR_MODEL, 1e-6, t, 0.5, c)
        Y_np, Y_pn = NR.nucleon_blocking(X_n, X_p, (μ̂ - NR.Q_NP_MEV) / t)
        @test Y_np ≈ X_n rtol = 1e-14
        @test Y_pn ≈ X_p rtol = 1e-14
        ηe, ην = -20.0, (-20t - μ̂) / t
        Rβ = NR_MODEL.c_β * 1e-6 * Y_pn * t^5 * fermi(Val(4), ηe) *
             NR.blocking(ην, fermi(Val(5), ηe) / fermi(Val(4), ηe))
        @test Rβ / (r[1].κa0 * r[1].nβ) ≈ exp(NR.Q_NP_MEV / t) rtol = 1e-9
    end
end

@testset "RJS96's pair emission agrees with BRT06's fit to 1%" begin
    # Guards the pair rate's constants and the unit conversions through an
    # independent formula: Burrows, Reddy & Thompson (2006) Eq. 121,
    # 9.7615e24 (T/MeV)⁹ erg cm⁻³ s⁻¹ for ν_e ν̄_e at η_e = 0, blocking left
    # out. Measured in step 16: 0.9953.
    t = 10.0
    F3, F4 = fermi(Val(3), 0.0), fermi(Val(4), 0.0)
    pairs = 2 * NR_MODEL.C_ee_e * NR_MODEL.c_ee / 36 * t^8 * F3^2 * (F4 / F3) * t
    νb = MSUN_DENSITY_CGS / NR.AMU_G
    Q_cgs = pairs * νb / NR.MSUN_TIME_CGS * NR.MEV_ERG
    @test Q_cgs / (9.7615e24 * t^9) ≈ 0.9953 atol = 0.01
end

@testset "The nucleon blocking reaches X_n, X_p and stays in range" begin
    # Guards RJS96 (A13, A14) with `X_p − X_n` (step 16): the non-degenerate
    # limit is the free fractions even where nuclei hold most of the matter,
    # the degenerate one is below them, and no table state gives a negative
    # or an excess target.
    for (X_n, X_p) in ((0.6, 0.4), (0.05, 0.01), (0.01, 0.05), (0.3, 0.3))
        x = log(X_n / X_p)
        Y_np, Y_pn = NR.nucleon_blocking(X_n, X_p, x)
        @test Y_np ≈ X_n rtol = 1e-7
        @test Y_pn ≈ X_p rtol = 1e-7
        for x in -50:0.5:50
            Y_np, Y_pn = NR.nucleon_blocking(X_n, X_p, x)
            @test 0 ≤ Y_np ≤ X_n && 0 ≤ Y_pn ≤ X_p
        end
    end
    @test NR.nucleon_blocking(0.6, 0.4, 0.0) == (0.6, 0.4)
end

@testset "Coherent scattering on α is BRT06's Eq. 32" begin
    # Guards the coherent term's weak charge and its transport weighting: one
    # α scatters with σ_tr = (2/3) 4 σ₀ sin⁴θ_W (ε/mₑc²)².
    t = 3.0
    c = nr_comp(; μ_e=0.0, μ̂=0.0, X_n=0.0, X_p=0.0, X_α=1.0)
    r = neutrino_rates(NR_MODEL, 1e-6, t, 0.5, c)[3]
    n_α = 1e-6 * MSUN_DENSITY_CGS / NR.AMU_G / 4
    ε² = t^2 * fermi(Val(4), 0.0) / fermi(Val(2), 0.0)
    σ = 2 / 3 * 4 * NR.RJS_σ0 * NR.RJS_SIN2W^2 * ε² / NR.RJS_ME^2
    @test r.κs0 ≈ n_α * σ * NR.MSUN_LENGTH_CGS rtol = 1e-12
end

# RJS96 Appendices A and B, BRT06 Eqs. 25–32 and 142–145, written out directly
# in cgs: the reference for the prefactors folded into `RJS96Rates`.
function nr_cgs_rates(ρ, t, c)
    σ0, α, sin2, me, hc = NR.RJS_σ0, NR.RJS_α, NR.RJS_SIN2W, NR.RJS_ME, NR.RJS_HC
    CV, CA, αf = 0.5 + 2sin2, 0.5, 1 / 137.036
    nb = ρ / NR.AMU_G
    F(k, η) = fermi(Val(k), η)
    B(η, ϵ) = 1 / (1 + exp(η - ϵ))
    ηe = c.μ_e / t
    ην = (c.μ_e - c.μ̂) / t
    x = (c.μ̂ - NR.Q_NP_MEV) / t
    Ynp = clamp((c.X_p - c.X_n) / (exp(-x) - 1), 0, c.X_n)
    Ypn = clamp(exp(-x) * (c.X_p - c.X_n) / (exp(-x) - 1), 0, c.X_p)
    ε(k, η) = 8π / hc^3 * t^(k + 1) * F(k, η)
    pre = σ0 * 2.99792458e10 / me^2
    Rβe = (1 + 3α^2) / 8 * pre * nb * Ypn * ε(4, ηe) * B(ην, F(5, ηe) / F(4, ηe))
    Rβa = (1 + 3α^2) / 8 * pre * nb * Ynp * ε(4, -ηe) * B(-ην, F(5, -ηe) / F(4, -ηe))
    ϵee = (F(4, ηe) / F(3, ηe) + F(4, -ηe) / F(3, -ηe)) / 2
    Ree = ((CV - CA)^2 + (CV + CA)^2) / 36 * pre * ε(3, ηe) * ε(3, -ηe) *
          B(ην, ϵee) * B(-ην, ϵee)
    Reex = ((CV - CA)^2 + (CV + CA - 2)^2) / 9 * pre * ε(3, ηe) * ε(3, -ηe) * B(0, ϵee)^2
    γ = 2sqrt(αf / 3π) * sqrt(π^2 / 3 + ηe^2)   # RJS96 (B12)
    ϵγ = 1 + γ^2 / (2(1 + γ))
    pl = pre * t^8 / hc^6 * γ^6 * exp(-γ) * (1 + γ)
    Rγ = π^3 / (3αf) * CV^2 * pl * B(ην, ϵγ) * B(-ην, ϵγ)
    Rγx = 4π^3 / (3αf) * (CV - 1)^2 * pl * B(0, ϵγ)^2
    Qbr = 1.04e30 * 0.5 * (c.X_n^2 + c.X_p^2 + 28 / 3 * c.X_n * c.X_p) * (ρ / 1e14)^2 *
          t^5.5 / NR.MEV_ERG                       # MeV cm⁻³ s⁻¹ per pair
    Ebr = 2.4 / 1.1 * t
    R = (Rβe + Ree + Rγ + Qbr / 2 / Ebr, Rβa + Ree + Rγ + Qbr / 2 / Ebr,
         Reex + Rγx + 2Qbr / Ebr)
    Q = ((Rβe * F(5, ηe) / F(4, ηe) + Ree * ϵee + Rγ * ϵγ) * t + Qbr / 2,
         (Rβa * F(5, -ηe) / F(4, -ηe) + Ree * ϵee + Rγ * ϵγ) * t + Qbr / 2,
         (Reex * ϵee + Rγx * ϵγ) * t + 2Qbr)
    W = 1 - 2 * c.Z̄ / c.Ā * (1 - 2sin2)
    scat = (1 + 5α^2) / 24 * c.X_n / (1 + 2max(c.μ_n / t, 0) / 3) +
           (4(CV - 1)^2 + 5α^2) / 24 * c.X_p / (1 + 2max(c.μ_p / t, 0) / 3) +
           (c.X_h * c.Ā * W^2 + 4 * c.X_α * (2sin2)^2) / 24
    ηs = (ην, -ην, 0.0)
    ab = ((1 + 3α^2) / 4 * Ynp * B(ηe, F(5, ην) / F(4, ην)),
          (1 + 3α^2) / 4 * Ypn * B(-ηe, F(5, -ην) / F(4, -ην)), 0.0)
    g = (1, 1, 4)
    return ntuple(3) do i
        η = ηs[i]
        k = σ0 * nb * (t / me)^2
        (R=R[i], Q=Q[i], κa0=k * ab[i] * F(4, η) / F(2, η), κs1=k * scat * F(5, η) / F(3, η),
         nβ=g[i] * 4π / hc^3 * t^3 * F(2, η), eβ=g[i] * 4π / hc^3 * t^4 * F(3, η))
    end
end

@testset "The rates in geometric units are RJS96's and BRT06's cgs formulas" begin
    # Guards the folding of every constant and unit conversion into the
    # prefactors: the model against the formulas written out in cgs, at hot,
    # degenerate and nucleus-rich states, all processes and species.
    states = ((1e-4, 8.0, nr_comp(; μ_e=20.0, μ̂=5.0, μ_n=2.0, μ_p=-1.0, X_n=0.5, X_p=0.1,
                                   X_α=0.1, X_h=0.3, Ā=60.0, Z̄=26.0)),
              (1e-3, 30.0, nr_comp(; μ_e=200.0, μ̂=100.0, μ_n=50.0, μ_p=-10.0, X_n=0.5,
                                   X_p=0.05, X_α=0.1, X_h=0.3, X_l=0.05, Ā=90.0, Z̄=35.0)),
              (1e-7, 2.0, nr_comp(; μ_e=1.0, μ̂=1.5, X_n=0.2, X_p=0.3, X_α=0.4, X_h=0.1,
                                  Ā=56.0, Z̄=26.0)))
    νb = MSUN_DENSITY_CGS / NR.AMU_G
    tu, lu, m★ = NR.MSUN_TIME_CGS, NR.MSUN_LENGTH_CGS, NR.AMU_MEV
    for (ρ★, t, c) in states
        model = neutrino_rates(NR_MODEL, ρ★, t, 0.3, c)
        ref = nr_cgs_rates(ρ★ * MSUN_DENSITY_CGS, t, c)
        for i in 1:3
            r, s = model[i], ref[i]
            @test r.R ≈ s.R * tu / νb rtol = 1e-12
            @test r.Q ≈ s.Q * tu / νb / m★ rtol = 1e-12
            @test r.κa0 ≈ s.κa0 * lu rtol = 1e-12 atol = 1e-300
            @test r.κs1 ≈ s.κs1 * lu rtol = 1e-12
            @test r.nβ ≈ s.nβ / νb rtol = 1e-12
            @test r.eβ ≈ s.eβ / νb / m★ rtol = 1e-12
        end
    end
end

@testset "Every rate is finite and non-negative, and Float32 follows Float64" begin
    # Guards the device path: no cgs-sized number reaches a kernel, so a
    # Float32 evaluation stays finite and follows Float64 wherever the value is
    # a normal Float32 number, across degeneracies and temperatures. The
    # rates are exponential in the degeneracies, so rounding `η = μ/T` to
    # Float32 alone moves them by `eps · |η|` (9.3e-4 at μₑ/T = 3000, measured
    # in step 16); the error is asserted in units of `eps(Float32)` times
    # `1 + |ηₑ| + |η_ν| + |(μ̂ − Q)/T|`.
    m32 = RJS96Rates(Float32, EOSUnits{Float32}(density=MSUN_DENSITY_CGS, κ=1);
                     baryon_mass=NR.AMU_G)
    worst = 0.0
    for ρ★ in (1e-12, 1e-8, 1e-5, 1e-3), t in (0.1, 1.0, 10.0, 80.0),
        μ_e in (-5.0, 0.5, 30.0, 300.0), μ̂ in (-20.0, 1.3, 60.0)
        kw = (; μ_e, μ̂, μ_n=μ̂ / 2, μ_p=-μ̂ / 2, X_n=0.4, X_p=0.2, X_α=0.1, X_h=0.25,
              X_l=0.05, Ā=70.0, Z̄=30.0)
        r64 = neutrino_rates(NR_MODEL, ρ★, t, 0.3, nr_comp(; kw...))
        r32 = neutrino_rates(m32, Float32(ρ★), Float32(t), 0.3f0, nr_comp(Float32; kw...))
        cond = eps(Float32) * (1 + abs(μ_e / t) + abs((μ_e - μ̂) / t) +
                               abs((μ̂ - NR.Q_NP_MEV) / t))
        for i in 1:3, f in fieldnames(SpeciesRates)
            a, b = getfield(r64[i], f), getfield(r32[i], f)
            @test isfinite(a) && a ≥ 0
            @test isfinite(b) && b ≥ 0
            a > 1e3 * floatmin(Float32) && a < floatmax(Float32) / 1e3 &&
                (worst = max(worst, abs(b - a) / a / cond))
        end
    end
    @info "RJS96 rates, worst Float32 error in eps · (1 + |η|)" worst
    @test worst ≤ 20
end

@testset "The effective rates have the free and the diffusion limits" begin
    # Guards the leakage interpolation and its branches: the free rate where
    # τ = 0, n^β κ/(𝒟 τ²) deep inside, zero without emission, and zero where
    # a cell that does not interact inherits an optical depth.
    r = neutrino_rates(NR_MODEL, 1e-4, 8.0, 0.3,
                       nr_comp(; μ_e=20.0, μ̂=5.0, μ_n=2.0, μ_p=-1.0, X_h=0.3, Ā=60.0,
                               Z̄=26.0))[1]
    @test effective_rates(r, 0.0, 0.0) == (r.R, r.Q)
    τ = 1e5
    R, Q = effective_rates(r, τ, τ)
    @test R ≈ r.nβ * (r.κa0 + r.κs0) / (6τ^2) rtol = 1e-6
    @test Q ≈ r.eβ * (r.κa1 + r.κs1) / (6τ^2) rtol = 1e-6
    @test effective_rates(SpeciesRates(0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), 2.0, 2.0) ==
          (0.0, 0.0)
    @test effective_rates(SpeciesRates(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0), 2.0, 2.0) ==
          (0.0, 0.0)
    # M0's absorption opacity: κ⁰_a at the equilibrium mean energy and τ = 0,
    # scaling as (E/E^β)² and e^{−τ}.
    Eβ = r.Eβ
    @test Eβ ≈ r.eβ / r.nβ rtol = 1e-14
    @test absorption_opacity(r, 0.0, Eβ) ≈ r.κa0 rtol = 1e-14
    @test absorption_opacity(r, 2.0, 3Eβ) ≈ 9 * exp(-2.0) * r.κa0 rtol = 1e-14
end

@testset "GrayRates: Kirchhoff's equilibrium density and the free limit" begin
    # Guards the test model the transport's studies use (steps 17–19).
    g = GrayRates{Float64}(; κa=(2.0, 1.0, 0.0), κs=(0.5, 0.5, 3.0), η=(4.0, 1.0, 2.0),
                           E=(0.01, 0.02, 0.03))
    r = neutrino_rates(g, 0.1, NaN, 0.3, nothing)
    @test r[1].R ≈ 0.4 && r[1].Q ≈ 0.004 && r[1].κa0 ≈ 0.2 && r[1].κs1 ≈ 0.05
    @test r[1].nβ ≈ 2.0 && r[1].eβ ≈ 0.02 && r[1].Eβ == 0.01
    @test isinf(r[3].nβ) && effective_rates(r[3], 5.0, 5.0) == (r[3].R, r[3].Q)
    @test absorption_opacity(r[3], 0.0, 0.03) == 0
    @test_throws ArgumentError GrayRates{Float64}(; κa=(-1.0, 0.0, 0.0), κs=(0.0, 0.0, 0.0),
                                                  η=(0.0, 0.0, 0.0), E=(0.0, 0.0, 0.0))
    @test_throws ArgumentError RJS96Rates(NR_UNITS; baryon_mass=0.0)
end

@testset "The rates read EntropyEOS's composition and refuse another table's (EOS-12)" begin
    # Guards the composition's way into the rates: `CompositionPoint` takes
    # EntropyEOS's fields as they are (its μ_ν is EntropyEOS's exactly), the
    # rates built from a table take its baryon mass and the composition's Q,
    # and a composition from another file, or units with another κ, are
    # refused rather than converting ρ★ silently wrong. On the synthetic
    # table with its composition datasets; DD2 is step 20's.
    E = TreeGRRMHD.EntropyEOS
    raw = E.make_synthetic_table(E.SyntheticOptions(; with_composition=true))
    eos = E.build_eos(raw, E.BuildOptions(; m_B_table_g=E.M_AMU_G))
    cv = E.CompositionView(E.CompositionTable(raw, eos))
    units = EOSUnits(E.EOSTableView(eos); density=MSUN_DENSITY_CGS)
    m = RJS96Rates(units, eos, cv)
    @test isbits(m) && E.matches(cv, eos)
    @test m.q_np == E.q_np(cv) == NR.Q_NP_MEV
    @test m.inv_m★ ≈ NR.AMU_G / (eos.m_B_star_g * NR.AMU_MEV) rtol = 1e-14
    for (logρ, t, yₑ) in ((12.0, 5.0, 0.3), (8.0, 1.0, 0.45), (14.5, 30.0, 0.1))
        ρ★ = E.κ(eos) * exp10(logρ)              # the EOS's units, g/cm³
        c = E.composition(cv, ρ★, t, yₑ)
        p = CompositionPoint(c; μ_n=0.0, μ_p=0.0)
        @test NR.neutrino_potential(p) == c.μ_ν
        r = neutrino_rates(m, ρ★ / MSUN_DENSITY_CGS, t, yₑ, p)
        @test all(s -> all(x -> isfinite(x) && x ≥ 0, (getfield(s, f) for f in
                                                       fieldnames(SpeciesRates))), r)
    end
    raw2 = E.make_synthetic_table(E.SyntheticOptions(; with_composition=true, nρ=41))
    eos2 = E.build_eos(raw2, E.BuildOptions(; m_B_table_g=E.M_AMU_G))
    cv2 = E.CompositionView(E.CompositionTable(raw2, eos2))
    @test !E.matches(cv2, eos)
    @test_throws ArgumentError RJS96Rates(units, eos, cv2)
    @test_throws ArgumentError RJS96Rates(EOSUnits{Float64}(density=MSUN_DENSITY_CGS, κ=1.0),
                                          eos, cv)
end

@kernel function nr_kernel!(out, m, @Const(ρs), @Const(ts), c)
    i = @index(Global)
    r = neutrino_rates(m, ρs[i], ts[i], ts[i], c)
    R, Q = effective_rates(r[1], ρs[i], ts[i])
    out[i] = R + Q + r[2].κa0 + r[3].κs1 + absorption_opacity(r[2], ρs[i], r[2].Eβ)
end

@testset "The rates run inside a CPU kernel, bit for bit, allocation-free: T=$T" for T in
                                                                                   (Float64,
                                                                                    Float32)
    # Guards kernel-callability: the models and the composition are `isbits`,
    # nothing throws, allocates or fails to infer, and a Float64 constant
    # would widen the Float32 result.
    m = RJS96Rates(T, EOSUnits{T}(density=MSUN_DENSITY_CGS, κ=1); baryon_mass=NR.AMU_G)
    c = nr_comp(T; μ_e=20.0, μ̂=5.0, μ_n=2.0, μ_p=-1.0, X_α=0.1, X_h=0.3, Ā=60.0, Z̄=26.0)
    @test isbits(m) && isbits(c)
    ρs = T.(exp10.(range(-8, -3; length=16)))
    ts = T.(range(1, 40; length=16))
    host = [(r = neutrino_rates(m, ρs[i], ts[i], ts[i], c);
             sum(effective_rates(r[1], ρs[i], ts[i])) + r[2].κa0 + r[3].κs1 +
             absorption_opacity(r[2], ρs[i], r[2].Eβ)) for i in 1:16]
    out = similar(host)
    nr_kernel!(CPU())(out, m, ρs, ts, c; ndrange=16)
    synchronize(CPU())
    @test eltype(out) === T
    @test isequal(out, host)
    @test (@inferred neutrino_rates(m, ρs[1], ts[1], ts[1], c)) isa NTuple{3,SpeciesRates{T}}
    @test rec_allocations(neutrino_rates, m, ρs[1], ts[1], ts[1], c) == 0
    @test rec_allocations(effective_rates, neutrino_rates(m, ρs[1], ts[1], ts[1], c)[1],
                          ρs[1], ts[1]) == 0
    @test rec_allocations(fermi, Val(5), ts[1]) == 0
end
