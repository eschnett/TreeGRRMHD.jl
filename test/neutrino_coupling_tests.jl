# Step 19: the neutrinos' feedback on the matter (rungs 8e and 8f;
# "Neutrinos", "Coupling to the matter" and "In the time integrator" in
# `CODE.md`). Uses `m0_tests.jl`'s `m0_ball`, `m0_cells` and `M0`, and
# `tiers.jl`'s `computed_once`.

const NC_GRAY = GrayRates{Float64}(; κa=(4.0, 1.0, 0.0), κs=(0.0, 0.0, 0.0),
                                   η=(0.05, 0.02, 0.03), E=(1e-3, 1.2e-3, 1.5e-3))
const NC_NU = M0Leakage(NC_GRAY; center=(0.0, 0.0, 0.0), r_core=0.001, feedback=true,
                        optical_depth=false)

# The uniform medium, cold (`p = 10⁻⁴`), in `dim` dimensions. The absorbed
# neutrinos push the gas outward, and the enthalpy flux `(τ + p) v` then
# changes τ by about `τ + p` of what they deposit: at `p = 10⁻²` 4%, which the
# closed form leaves out and no resolution removes (step 19: Δτ's error 3.8%
# on 16³ and 3.9% on 32³); here 0.04%.
nc_medium(; roots=2, dim=3) = m0_ball(; roots, ρ_out=1.0, p=1e-4, dim)

# The outflow through the outer faces (`±L`, `L = 0.8`) of slot `v` of the
# face sets `Fs` the last `rhs!` left: high faces positive, low negative,
# weighted by the face's area.
function nc_outflow(Fs, v; L=0.8)
    D = length(Fs)
    total = 0.0
    for d in 1:D
        F = Fs[d]
        Fw = F.work
        for b in axes(Fw, D + 2)
            h = coordinates(F, b, ntuple(e -> e == d ? 2 : 1, D))[d] -
                coordinates(F, b, ntuple(_ -> 1, D))[d]
            for idx in CartesianIndices(size(Fw)[1:D])
                xd = coordinates(F, b, Tuple(idx))[d]
                sgn = isapprox(xd, L; atol=1e-12) ? 1 : isapprox(xd, -L; atol=1e-12) ? -1 : 0
                total += sgn * Fw[Tuple(idx)..., v, b] * h^(D - 1)
            end
        end
    end
    return total
end

# Σ over the owned cells, volume-weighted, of slot `v` of segment `seg` of `u`.
function nc_sum(p, u, seg, v)
    s = p.sets[seg]
    D = length(s.G)
    T = eltype(s.work)
    return mesh_mapreduce(identity, +, zero(T), s, stateview(u, p.sets, seg); vars=v,
                          weight=key -> spacing(T, s.forest, key)^D)
end

# A problem on a mesh with a refined corner block, its state after a few
# steps (so that N, 𝓔 and the matter's change are nonzero everywhere near the
# centre), and the matter's and neutrinos' flux sets.
function nc_state(dim)
    case = nc_medium(; dim)
    forest = Forest{Float64}(case.roots; N=8, periodic=case.periodic, extents=case.extents)
    refine!(forest, [first(forest.leaves)]); balance!(forest)
    p = MHDProblem(mhd_fields(Float64, forest), case; limiter=:mc, neutrinos=NC_NU)
    u = exact_state(p, case, 0.0)
    initial_depth!(p, u, 0.0)                # `P` before the first implicit stage
    clear_tally!(p)
    integ = mhd_integrator(p, u, 0.0, 0.2, 20; alias_u0=true)
    with_step_cache(() -> M0.IRK.solve!(integ), p)
    return p, integ.u
end

@testset "Lepton number and energy are conserved by the feedback: D=$dim" for dim in (2, 3)
    # Guards the matter's share of every neutrino source against the
    # neutrinos' own, across coarse-fine faces: in a right-hand side,
    # Σ du of √γ D Yₑ + N_{ν_e} − N_{ν̄_e} plus its outflow through the outer
    # faces is zero to round-off, and so, in flat space at rest, is that of
    # √γ τ + Σ 𝓔 (the matter's own fluxes included: the Dirichlet faces
    # dissipate the changed Yₑ and τ); in an implicit stage the matter's
    # increments are the neutrinos' with the opposite sign, cell by cell.
    p, u = nc_state(dim)
    # No cell needed the reset in 20 steps (step 19: a NaN from `P`'s zeros
    # before the first right-hand side sent every cell to the atmosphere).
    @test M0.ohm_tally(p).resets == 0 && M0.ohm_tally(p).step_resets == 0
    du = similar(u)
    M0.prepare!(p, u, 0.2)
    M0.rhs!(du, p, 0.2)
    Fm, Fn = p.f.F, p.neutrinos.Fν
    lepton = nc_sum(p, du, M0.SEG_U, M0.U_DYE) + nc_sum(p, du, M0.SEG_NU, M0.nu_N(1)) -
             nc_sum(p, du, M0.SEG_NU, M0.nu_N(2)) + nc_outflow(Fm, M0.U_DYE) +
             nc_outflow(Fn, M0.nu_N(1)) - nc_outflow(Fn, M0.nu_N(2))
    # The scales: every magnitude that cancels, the matter's included.
    lscale = nc_sum(p, abs.(du), M0.SEG_U, M0.U_DYE) +
             nc_sum(p, abs.(du), M0.SEG_NU, M0.nu_N(1)) + nc_sum(p, abs.(du), M0.SEG_NU, M0.nu_N(2))
    energy = nc_sum(p, du, M0.SEG_U, M0.U_TAU) + nc_outflow(Fm, M0.U_TAU) +
             sum(nc_sum(p, du, M0.SEG_NU, M0.nu_E(i)) + nc_outflow(Fn, M0.nu_E(i)) for i in 1:3)
    escale = nc_sum(p, abs.(du), M0.SEG_U, M0.U_TAU) +
             sum(nc_sum(p, abs.(du), M0.SEG_NU, M0.nu_E(i)) for i in 1:3)
    # One implicit stage, as the integrator calls it: U a copy of u★.
    U = copy(u)
    M0.stage_solve!(U, u, 0.003, p, 0.2)
    ΔL = nc_sum(p, U .- u, M0.SEG_U, M0.U_DYE) + nc_sum(p, U .- u, M0.SEG_NU, M0.nu_N(1)) -
         nc_sum(p, U .- u, M0.SEG_NU, M0.nu_N(2))
    # The stage's increments are as small as 10⁻²⁰ against a matter state of
    # order 10⁻²; their rounding is the state's, ε·|u|.
    ΔLs = nc_sum(p, abs.(u), M0.SEG_U, M0.U_DYE)
    ΔE = nc_sum(p, U .- u, M0.SEG_U, M0.U_TAU) +
         sum(nc_sum(p, U .- u, M0.SEG_NU, M0.nu_E(i)) for i in 1:3)
    ΔEs = nc_sum(p, abs.(u), M0.SEG_U, M0.U_TAU)
    @info "Feedback conservation, relative" dim rhs_lepton = lepton / lscale rhs_energy = energy / escale stage_lepton = ΔL / ΔLs stage_energy = ΔE / ΔEs
    @test abs(lepton) ≤ 1e-13 * lscale
    @test abs(energy) ≤ 1e-13 * escale
    @test abs(ΔL) ≤ 1e-15 * ΔLs
    @test abs(ΔE) ≤ 1e-15 * ΔEs
    # A regrid transfers the lepton number conservatively.
    before = nc_sum(p, u, M0.SEG_U, M0.U_DYE) + nc_sum(p, u, M0.SEG_NU, M0.nu_N(1)) -
             nc_sum(p, u, M0.SEG_NU, M0.nu_N(2))
    g = regrid_state!(p, u, 0.2; flags=fill(Refine, nleaves(p.f.U.forest)))
    after = nc_sum(g.p, g.u, M0.SEG_U, M0.U_DYE) + nc_sum(g.p, g.u, M0.SEG_NU, M0.nu_N(1)) -
            nc_sum(g.p, g.u, M0.SEG_NU, M0.nu_N(2))
    @info "Regrid, lepton number" dim (after - before) / before
    @test abs(after - before) ≤ 1e-14 * abs(before)   # measured in step 19: 2e−16
end

# The coupled exchange in the uniform medium has a closed form while the gas
# stays at rest: along each ray `∂ₜ m + ∂_r m = η r² − κ m`, `m = r² n`, from
# zero, so `m = η e^{−κr} (F(r) − F(a))` with `a = max(0, r − t)` and
# `F(x) = e^{κx}(x²/κ − 2x/κ² + 2/κ³)` (or `η (r³ − a³)/3` without absorption);
# and the matter's `√γ D Yₑ` and `√γ τ` change at each cell by the time
# integrals of what the neutrinos gain and lose there.
function nc_n(η, κ, r, t)
    r == 0 && return 0.0
    a = max(0.0, r - t)
    κ == 0 && return η * (r^3 - a^3) / 3 / r^2
    F(x) = exp(κ * x) * (x^2 / κ - 2x / κ^2 + 2 / κ^3)
    return η * exp(-κ * r) * (F(r) - F(a)) / r^2
end

# ∫₀ᵗ f(t′) dt′ by Simpson's rule on each side of the kink at `t′ = r`.
function nc_integral(f, r, t; k=400)
    simpson(a, b) = a ≥ b ? 0.0 :
                    (h = (b - a) / k; h / 3 * sum((j == 0 || j == k ? 1 : iseven(j) ? 2 : 4) *
                                                  f(a + j * h) for j in 0:k))
    return simpson(0.0, min(r, t)) + simpson(min(r, t), t)
end

nc_exchange_run(roots) = computed_once((:nc_exchange, roots)) do
    case = nc_medium(; roots)
    r0 = evolve!(Float64, case, Val(3); N=8, t_end=0.4, limiter=:mc, neutrinos=NC_NU)
    p, u = r0.p, r0.u
    u0 = exact_state(p, case, 0.0)
    (; p, u, u0, t=r0.t, tally=neutrino_tally(p), resets=M0.ohm_tally(p).resets)
end

# The L1 errors, relative, of N of each species, and of the changes of
# √γ D Yₑ and √γ τ, at the cells inside r ≤ 0.6.
function nc_exchange_errors(roots)
    r0 = nc_exchange_run(roots)
    p, u, u0, t = r0.p, r0.u, r0.u0, r0.t
    uU, uU0 = statearray(u, p.sets, M0.SEG_U), statearray(u0, p.sets, M0.SEG_U)
    uN = statearray(u, p.sets, M0.SEG_NU)
    η, κ, E = NC_GRAY.η, NC_GRAY.κa, NC_GRAY.E
    errs = zeros(5)
    scales = zeros(5)
    for b in axes(uU, 5), idx in CartesianIndices(size(uU)[1:3])
        x = coordinates(p.f.U, b, Tuple(idx) .+ 3)
        ρr = sqrt(sum(abs2, x))
        ρr ≤ 0.6 || continue
        n(i, tt) = nc_n(η[i], κ[i], ρr, tt)
        for i in 1:3
            ref = n(i, t)
            errs[i] += abs(uN[Tuple(idx)..., M0.nu_N(i), b] - ref)
            scales[i] += ref
        end
        dYe = -nc_integral(tt -> (η[1] - κ[1] * n(1, tt)) - (η[2] - κ[2] * n(2, tt)), ρr, t)
        dτ = nc_integral(tt -> sum(E[i] * (κ[i] * n(i, tt) - η[i]) for i in 1:3), ρr, t)
        errs[4] += abs(uU[Tuple(idx)..., M0.U_DYE, b] - uU0[Tuple(idx)..., M0.U_DYE, b] - dYe)
        scales[4] += abs(dYe)
        errs[5] += abs(uU[Tuple(idx)..., M0.U_TAU, b] - uU0[Tuple(idx)..., M0.U_TAU, b] - dτ)
        scales[5] += abs(dτ)
    end
    return errs ./ scales
end

@testset "The coupled exchange against its closed form (8f)" begin
    # Guards emission and absorption on both sides together, through the
    # approach to R^eff = κ^eff n: in the uniform medium the neutrinos' N and
    # the matter's changes of √γ D Yₑ and √γ τ at t = 0.4 against the closed
    # form at every cell inside r ≤ 0.6, with no cell repaired by the reset;
    # and `neutrino_tally`'s largest |d ln(√γ D Yₑ)/dt|, the emission's
    # 0.05 − 0.02 against √γ D Yₑ, which falls from 1/2 to about 0.49.
    # Measured in step 19 on 16³: N 0.0051, 0.0070, 0.0079; ΔDYₑ 0.0085;
    # Δτ 0.0025.
    e = nc_exchange_errors(2)
    @info "Coupled exchange on 16³, relative L1 errors (N₁ N₂ N₃ ΔDYₑ Δτ)" e
    @test all(e[1:3] .≤ 0.01) && e[4] ≤ 0.012 && e[5] ≤ 0.004
    r0 = nc_exchange_run(2)
    @test r0.resets == 0
    @test 0.06 ≤ r0.tally.dYe_rate ≤ 0.062
end

@full_tier @testset "The coupled exchange converges at second order" begin
    # Guards the order of the coupled update from 16³ to 32³. Measured in
    # step 19: N 2.04, 2.11, 2.13, Δτ 2.38, ΔDYₑ 1.37 — the last the matter
    # scheme's, not the coupling's: the same at a quarter of the step, so
    # spatial; the matter's MC clips Yₑ's extrema (rung 7: 1.69–1.86 for a
    # smooth advected profile), and the deposited Yₑ has a cone-shaped one at
    # the centre and a kink at the front.
    e16, e32 = nc_exchange_errors(2), nc_exchange_errors(4)
    rates = log2.(e16 ./ e32)
    @info "Coupled exchange, rates 16³ → 32³ (N₁ N₂ N₃ ΔDYₑ Δτ)" rates
    @test all(rates[1:3] .≥ 1.9) && rates[5] ≥ 2.0 && rates[4] ≥ 1.3
end

@testset "evolve! refuses neutrinos with a checkpoint or a restart" begin
    # Guards a run that could not be continued faithfully: the optical depth
    # lives outside the state vector, so a checkpoint would not hold it and a
    # restart would not be the uninterrupted run (PLAN.md, "Neutrinos with
    # checkpoints and MPI ranks").
    case = nc_medium()
    @test_throws ArgumentError evolve!(Float64, case, Val(3); N=8, t_end=0.1, limiter=:mc,
                                       neutrinos=NC_NU, checkpoint=tempname())
    @test_throws ArgumentError evolve!(Float64, case, Val(3); N=8, t_end=0.1, limiter=:mc,
                                       neutrinos=NC_NU, restart=tempname())
end
