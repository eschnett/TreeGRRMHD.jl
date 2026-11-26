# Step 20: the star with neutrinos (rung 8g; "Neutrinos" in `CODE.md`). First the
# composition's way into the kernels, on the synthetic table (no DD2): the rates a kernel
# evaluates at a cell are the pointwise rates at the cell's state, and the refusals. Uses
# `optical_depth_tests.jl`'s `od_forest` and `m0_tests.jl`'s `m0_ball`.

const NS = TreeGRRMHD
const NS_E = TreeGRRMHD.EntropyEOS

# The synthetic table with its composition, the EOS built from it, the view, the units
# and the rates on them, read once.
ns_synthetic() = computed_once(:ns_synthetic) do
    raw = NS_E.make_synthetic_table(NS_E.SyntheticOptions(; with_composition=true))
    eos = NS_E.build_eos(raw, NS_E.BuildOptions(; m_B_table_g=NS_E.M_AMU_G))
    cv = NS_E.CompositionView(NS_E.CompositionTable(raw, eos))
    units = EOSUnits(NS_E.EOSTableView(eos); density=MSUN_DENSITY_CGS)
    (; raw, eos, cv, units, rates=RJS96Rates(units, eos, cv))
end

# A primitive set holding, at the cell centre `x`, the state `state(x) = (ρ★, u, Yₑ)`
# (geometric `ρ★`, `u = log₁₀ T_MeV`) in a flat metric at rest, ghosts included.
function ns_primitives(forest, state)
    D = length(forest.periodic)
    P = FieldSet{Float64}(forest, NS.NP; G=3, centering=cellcentered(D))
    fill!(P.work, 0)
    for b in 1:size(P.work)[end], idx in CartesianIndices(size(P.work)[1:D])
        s = Tuple(idx)
        ρ, u, Y = state(coordinates(P, b, s))
        P.work[s..., NS.P_RHO, b] = ρ
        P.work[s..., NS.P_UEOS, b] = u
        P.work[s..., NS.P_YE, b] = Y
        P.work[s..., NS.P_GAMMA, b] = 1
        P.work[s..., NS.P_GAMMA + 3, b] = 1
        P.work[s..., NS.P_GAMMA + 5, b] = 1
        P.work[s..., NS.P_ALPHA, b] = 1
        P.work[s..., NS.P_SQRTG, b] = 1
    end
    return P
end

@testset "A kernel's tabulated rates are the pointwise rates at its cell's state" begin
    # Guards the composition's way into the kernels (`TabulatedRates`, `cell_rates`): the
    # opacity kernel's κ at every owned cell equals `neutrino_rates` at the cell's `ρ★`,
    # `T = 10^u` and `Yₑ` with the composition there, bit for bit, across the synthetic
    # table's densities and temperatures; a cell with `ρ★ = 0` or a warm start that is
    # not finite has none, as `P` is before the first fill.
    d = ns_synthetic()
    m, cv = d.rates, d.cv
    f = d.units.density
    # log₁₀ ρ from 6 to 14 (g/cm³) along x, log₁₀ T from −1 to 1.5 along y; the corner
    # cells hold the two empty states.
    function state(x)
        x[1] < -0.9 && x[2] < -0.9 && return (0.0, 0.5, 0.3)
        x[1] > 0.9 && x[2] > 0.9 && return (1e-6, NaN, 0.3)
        return (exp10(10 + 4x[1]) / f, 0.25 + 1.25x[2], 0.2 + 0.1x[1] * x[2])
    end
    forest = od_forest(Val(2), 2)
    P = ns_primitives(forest, state)
    od = OpticalDepth(forest)
    model = TabulatedRates(m, cv)
    opacities!(od, model, P)
    worst, nzero, n = 0.0, 0, 0
    for b in 1:size(P.work)[end], idx in CartesianIndices((8, 8))
        s = Tuple(idx) .+ 3
        ρ, u, Y = P.work[s..., NS.P_RHO, b], P.work[s..., NS.P_UEOS, b],
                  P.work[s..., NS.P_YE, b]
        κ = ntuple(k -> od.Nτ.work[(Tuple(idx) .+ 1)..., NS.TAU_K + k, b], 6)
        n += 1
        if ρ > 0 && isfinite(u)
            t = exp10(u)
            c = CompositionPoint(NS_E.composition(cv, ρ * m.ρ_unit, t, Y))
            r = neutrino_rates(m, ρ, t, Y, c)
            ref = (r[1].κa0 + r[1].κs0, r[1].κa1 + r[1].κs1, r[2].κa0 + r[2].κs0,
                   r[2].κa1 + r[2].κs1, r[3].κa0 + r[3].κs0, r[3].κa1 + r[3].κs1)
            worst = max(worst, maximum(abs.(κ .- ref) ./ ref))
            @test all(>(0), κ)
        else
            nzero += 1
            @test all(iszero, κ)
        end
    end
    @info "Tabulated rates in the opacity kernel against pointwise, worst relative " *
          "difference" worst n nzero
    @test worst == 0
    @test nzero == 2
    # Allocation-free and inferred, as a kernel needs.
    Pc = ntuple(v -> v == NS.P_RHO ? 1e-6 : v == NS.P_UEOS ? 0.5 : v == NS.P_YE ? 0.3 : 0.0,
                NS.NP)
    @inferred NS.cell_rates(model, Pc)
    NS.cell_rates(model, Pc)
    @test (@allocated NS.cell_rates(model, Pc)) == 0
end

@testset "M0Leakage refuses rates without their composition, and a composition of another table" begin
    # Guards the composition's pairing: `RJS96Rates` read the table at every cell, so
    # without the composition they are refused, and `GrayRates` read none, so one with
    # them is refused; a composition that does not `match` the problem's EOS (here the
    # ideal gas of `m0_ball`) is refused when the problem is built, rather than
    # evaluated at densities of another table.
    d = ns_synthetic()
    gray = GrayRates{Float64}(; κa=(1.0, 1.0, 0.0), κs=(0.0, 0.0, 0.0), η=(1.0, 1.0, 1.0),
                              E=(0.01, 0.01, 0.01))
    kw = (; center=(0.0, 0.0, 0.0), r_core=0.01)
    @test_throws ArgumentError M0Leakage(d.rates; kw...)
    @test_throws ArgumentError M0Leakage(gray; composition=d.cv, kw...)
    nu = M0Leakage(d.rates; composition=d.cv, kw...)
    @test NS.rate_model(nu) isa TabulatedRates
    @test NS.rate_model(M0Leakage(gray; kw...)) === gray
    case = m0_ball(; dim=2)
    forest = Forest{Float64}(case.roots; N=8, periodic=case.periodic, extents=case.extents)
    @test_throws ArgumentError MHDProblem(mhd_fields(Float64, forest), case; limiter=:mc,
                                          neutrinos=nu)
end

# The ball of `m0_tests.jl` in 2D with gray rates, feedback and an optical depth, so that
# a restart that lost `Nν`, `Nτ` or the `P` the first implicit stage reads would differ.
const NS_GRAY = GrayRates{Float64}(; κa=(4.0, 1.0, 0.0), κs=(1.0, 1.0, 2.0),
                                   η=(0.05, 0.02, 0.03), E=(1e-3, 1.2e-3, 1.5e-3))
const NS_NU = M0Leakage(NS_GRAY; center=(0.0, 0.0, 0.0), r_core=0.05)

@testset "A run with neutrinos restarted from its checkpoint is the run that never stopped, bit for bit" begin
    # Guards what a checkpoint with neutrinos must hold: `Nν`'s segment, and the optical
    # depth `Nτ` (τ and κ), carried from step to step outside the state vector and not
    # reconverged by the restart; and the `P` the first implicit stage reads, which the
    # restart prepares from the state, as every chunk boundary does. On a static mesh
    # and on an adaptive one (refined from the initial data, and regridded, moving or
    # not, at every chunk boundary, the restart's first included). A restart with
    # another `M0Leakage`, or without neutrinos, is refused. Measured in step 20: before
    # `P` was prepared at every chunk boundary, the static restart differed by 1e−12.
    case = m0_ball(; dim=2)
    kw = (; N=8, chunk=0.05, limiter=:mc, neutrinos=NS_NU)
    for (name, akw) in (("static", (;)),
                        ("adaptive", (; maxlevel=1, refine_tol=0.01, coarsen_tol=0.002,
                                      buffer=2)))
        path = joinpath(mktempdir(), "nu.h5")
        whole = evolve!(case, Val(2); kw..., akw..., t_end=0.15)
        evolve!(case, Val(2); kw..., akw..., t_end=0.05, checkpoint=path)
        rest = evolve!(case, Val(2); kw..., akw..., t_end=0.15, restart=path)
        τw, τr = whole.p.neutrinos.od.Nτ, rest.p.neutrinos.od.Nτ
        own(s) = statevector(s) |> v -> gather!(v, s)
        @info "Neutrino restart, $name mesh: $(whole.nsteps) steps, $(whole.nregrids) " *
              "regrids, Σ|u| $(sum(abs, whole.u))"
        @test rest.u == whole.u
        @test own(τr) == own(τw)
        @test rest.nsteps == whole.nsteps && rest.nregrids == whole.nregrids
        name == "adaptive" && @test whole.passes > 0
    end
    path = joinpath(mktempdir(), "nu.h5")
    evolve!(case, Val(2); kw..., t_end=0.05, checkpoint=path)
    other = M0Leakage(NS_GRAY; center=(0.0, 0.0, 0.0), r_core=0.1)
    @test_throws "neutrinos" evolve!(case, Val(2); kw..., neutrinos=other, t_end=0.1,
                                     restart=path)
    @test_throws "neutrinos" evolve!(case, Val(2); kw..., neutrinos=nothing, t_end=0.1,
                                     restart=path)
end

@testset "evolve! refuses neutrinos on MPI ranks" begin
    # Guards a run nothing has tested yet (PLAN.md, step 21): the neutrinos' fills,
    # reductions and regrid across ranks.
    @test_throws "MPI ranks" evolve!(m0_ball(; dim=2), Val(2); N=8, t_end=0.05, limiter=:mc,
                                     neutrinos=NS_NU, comm=TreeAMR.SerialCommunicator())
end

@testset "The shell's luminosity of a uniform emitting medium is the radial solution's" begin
    # Guards `neutrino_luminosity`: the window's average over the shell of the radial flux,
    # in coordinates and densitized, against `∫ W(r) 4π r² n(r) dr` of the radial solution
    # in the medium's steady state (`m0_tests.jl`'s run on 32³ to t = 0.7, steady for
    # r ≤ 0.7), for an absorbing and a free-streaming species; and `L_𝓔/L_N` the emitted
    # energy, since at rest in flat space every neutrino keeps it. A wrong sign, a missing
    # area or volume factor, or a window not normalized to `∫ W dr = 1` is off by far more.
    r = m0_ball_run(4, 0.7)
    r_L, width = 0.4, 0.15
    W(ρ) = NS.shell_window(ρ, r_L, width)
    L = neutrino_luminosity(r.p, r.u; r_L, width)
    errs = map(1:3) do i
        η, κ = M0_GRAY.η[i], M0_GRAY.κa[i]
        n = 4000
        q = range(r_L - width, r_L + width; length=n + 1)
        exact = sum(W(ρ) * 4π * ρ^2 * m0_exact(η, κ, ρ) for ρ in q[2:(end - 1)]) * step(q)
        @test L[2i] / L[2i - 1] ≈ M0_GRAY.E[i] rtol = 1e-12
        abs(L[2i - 1] - exact) / exact
    end
    @info "Shell luminosity on 32³ against the radial solution, relative error per species" errs
    @test all(<(2e-3), errs)          # 0.009%, 0.04%, 0.08% (step 20)
end

# DD2's 1.4 M☉ star with neutrinos (rung 8g), cold and hot (`s = 1`), on its Cowling metric
# at `h = 1` (the suite's tabulated star, `tabulated_tov_tests.jl`), to `t = 2M` in two
# chunks, so that a chunk boundary prepares `P`. Read only where DD2 is (`tables.jl`).
const NS_STAR = (; ρ_c=9.4e-4, ρ_min=1e-10, L=16, roots=2, t_end=2.0, chunk=1.0)

ns_dd2() = computed_once(:ns_dd2) do
    d = dd2_table()
    cv = NS_E.CompositionView(NS_E.CompositionTable(d.table, d.eos))
    units = EOSUnits(NS_E.EOSTableView(d.eos); density=MSUN_DENSITY_CGS)
    (; d.table, d.eos, cv, rates=RJS96Rates(units, d.eos, cv))
end

ns_star_run(s) = computed_once((:ns_star_run, s)) do
    d = ns_dd2()
    case = tabulated_tov_case(Float64; d.table, d.eos, NS_STAR.ρ_c, NS_STAR.ρ_min, s,
                              NS_STAR.L, NS_STAR.roots)
    nu = M0Leakage(d.rates; composition=d.cv, center=(0.0, 0.0, 0.0), r_core=1.0)
    r = evolve!(case, Val(3); N=8, t_end=NS_STAR.t_end, chunk=NS_STAR.chunk, limiter=:mc,
                refine=f -> tov_boxes!(f, case; levels=1), neutrinos=nu)
    (; r, case)
end

# The lepton number's change net of the reset, its scale, the neutrino totals, the
# matter's `√γ D Yₑ` change, and the temperature at the cell nearest the centre.
function ns_star_numbers(r)
    tot = conserved_totals(r.p, r.u)
    nt = neutrino_totals(r.p, r.u)
    injY = sum(c -> c.tally.injected_DYₑ, r.chunks)
    dDYe = tot[NS.U_DYE] - r.totals0[NS.U_DYE]
    P = NS.hostcopy(NS.update_primitives!(r.p, r.u, r.t))
    best, Tc = Inf, NaN
    for b in 1:size(P.work)[end], idx in CartesianIndices((8, 8, 8))
        s = Tuple(idx) .+ 3
        x = coordinates(P, b, s)
        ρ = sum(abs2, x)
        ρ < best && ((best, Tc) = (ρ, exp10(P.work[s..., NS.P_UEOS, b])))
    end
    return (; net=dDYe + nt[1] - nt[3] - injY, scale=r.totals0[NS.U_DYE], nt, dDYe, Tc,
            E=nt[2] + nt[4] + nt[6], floors=sum(c -> c.tally.s_floors, r.chunks))
end

@dd2_gated @testset "DD2's cold star with neutrinos conserves lepton number, and emits only what the mesh heats" begin
    # Guards the tabulated rates in the scheme on the star: lepton number, matter plus
    # neutrinos, conserved to round-off net of the reset (nothing reaches the outer
    # faces by 2M); and the cold star's emission, which on the mesh is the
    # discretization's: the slice is at 0.01 MeV, but the recovered cell averages of the
    # initial data are 4.4 MeV at the centre and 6.4 MeV where the density falls (step
    # 20), and the rates go as T⁶ to T⁹ there. Measured in step 20 at t = 2M: Σ𝓔 =
    # 5.2e−8, the centre 4.4 MeV; held to 10%, so that a change of the rates or of the
    # matter's heating is seen. The bound over 1000 M and its fall with h are
    # `bin/neutrino_star.jl`'s.
    n = ns_star_numbers(ns_star_run(nothing).r)
    @info "DD2 cold star with neutrinos, h = 1, t = 2M" n
    @test abs(n.net) < 1e-14 * n.scale
    @test isapprox(n.E, 5.2e-8; rtol=0.1)
    @test isapprox(n.Tc, 4.4; rtol=0.1)
    @test n.floors == 0
end

@dd2_gated @testset "DD2's hot star with neutrinos conserves lepton number, cools and gains electrons" begin
    # Guards the hot star (`hot_barotrope`, s = 1: 26.3 MeV at the centre, 1.51 M☉ at the
    # cold star's central density) with the tabulated rates: lepton number conserved to
    # round-off net of the reset; its matter, which holds the cold slice's Yₑ, out of
    # β-equilibrium at its temperature, emits more ν̄_e than ν_e, so its `√γ D Yₑ` rises;
    # its neutrinos' energy at 2M as measured in step 20 (1.33e−7, 2.6 times the cold
    # star's), held to 10%.
    n = ns_star_numbers(ns_star_run(1.0).r)
    @info "DD2 hot star (s = 1) with neutrinos, h = 1, t = 2M" n
    @test abs(n.net) < 1e-14 * n.scale
    @test n.nt[3] > n.nt[1] && n.dDYe > 0
    @test isapprox(n.E, 1.33e-7; rtol=0.1)
    @test isapprox(n.Tc, 26.3; rtol=0.02)
    @test n.floors == 0
end
