# Step 13 (rung 7): the TOV star on a tabulated EOS, DD2, in cold β-equilibrium
# ("Validation ladder" 7 in `CODE.md`; `src/tov_table.jl`, `src/cases/tabulated_tov.jl`).
# Every testset reads the table and runs only where `ENTROPYEOS_TABLE_DIR` holds DD2
# (`tables.jl`); elsewhere each is one skipped test. First the barotrope, the solve and
# the composition, pointwise; then the star on the mesh.

isdefined(@__MODULE__, :DD2_PRESENT) || include("tables.jl")

const TT = TreeGRRMHD
# The suite's star: DD2 at `ρ_c = 9.4·10⁻⁴` (5.8·10¹⁴ g/cm³, `M = 1.404`), the atmosphere
# at `10⁻¹⁰` (6.2·10⁷ g/cm³), on the polytrope's suite mesh (`tov_tests.jl`).
const TT_STAR = (; ρ_c=9.4e-4, ρ_min=1e-10)

tt_barotrope() = computed_once(:tt_barotrope) do
    d = dd2_table()
    slice = TT.EntropyEOS.cold_beta_slice(d.table, d.eos)
    (; slice, b=cold_barotrope(slice; ρ_surface=TT_STAR.ρ_min))
end

@dd2_gated @testset "The cold barotrope is DD2's β-equilibrium slice, with dp/dη = e + p" begin
    # Guards `cold_barotrope`: a unit conversion other than `GeometricEOS`'s, a knot
    # that is not the slice's state at its pressure, or an `η` that is not `∫ dp/(e+p)`
    # would put the star out of the equilibrium the evolution keeps. The knots against
    # the slice's own lookup by pressure (measured 5.9e−13), the interpolant's `dp/dη`
    # against `e + p` by central differences (1.7e−6), and `η`'s top at 8001 against
    # 32001 samples (8.6e−9).
    (; slice, b) = tt_barotrope()
    f = MSUN_DENSITY_CGS
    dev = 0.0
    for k in 1:50:length(b.η)
        st = TT.EntropyEOS.beta_equilibrium(slice; p=exp(b.logp[k]) * f)
        dev = max(dev, abs(st.ρ★ / f / b.ρ[k] - 1), abs(st.s - b.s[k]), abs(st.yₑ - b.Yₑ[k]))
    end
    w = 0.0
    for η in range(1e-4, b.η[end] - 1e-4; length=2000)
        d = (barotrope_state(b, η + 1e-6).p - barotrope_state(b, η - 1e-6).p) / 2e-6
        st = barotrope_state(b, η)
        w = max(w, abs(d / (st.e + st.p) - 1))
    end
    b4 = cold_barotrope(slice; ρ_surface=TT_STAR.ρ_min, samples=32001)
    @info "DD2's cold barotrope: $(length(b.η)) knots, η ∈ [0, $(b.η[end])]; knots against " *
          "the slice $dev; dp/dη against e + p $w; η's top against 32001 samples " *
          "$(b.η[end] - b4.η[end])"
    @test dev < 1e-11
    @test w < 1e-5
    @test abs(b.η[end] - b4.η[end]) < 1e-7
    @test issorted(b.η) && issorted(b.ρ) && isapprox(b.ρ[1], TT_STAR.ρ_min; rtol=1e-9)
end

@dd2_gated @testset "The tabulated star's solve is converged and gives DD2's radius at 1.4 M☉" begin
    # Guards `tov_solve` on a barotrope and the units: the RK4 at the default 4000 steps
    # against 16000 (measured 4.5e−9 in `M`, 3.9e−10 in `R`; the slice's kinks at the
    # table's density nodes end fourth-order convergence near 1e−9), and the radius of
    # the 1.4 M☉ star, found by bisection in `ρ_c`, against the literature's for DD2:
    # about 13.2–13.3 km (e.g. Fischer et al. 2014, EPJA 50, 46). That radius needs the
    # crust, so its barotrope reaches down to 10⁻¹⁴ (6.2·10³ g/cm³; measured 13.225 km,
    # 13.194 km cut at the suite's `ρ_min`). A wrong density unit or a missing rest-mass
    # term moves it by far more.
    (; slice, b) = tt_barotrope()
    b14 = cold_barotrope(slice; ρ_surface=1e-14)
    s4 = tov_solve(Float64, b; ρ_c=TT_STAR.ρ_c)
    s16 = tov_solve(Float64, b; ρ_c=TT_STAR.ρ_c, steps=16000)
    lo, hi = 2e-4, 2e-3
    for _ in 1:60
        mid = sqrt(lo * hi)
        tov_solve(Float64, b14; ρ_c=mid, steps=1000).M < 1.4 ? (lo = mid) : (hi = mid)
    end
    km = 1.4766250614046494                           # G M☉/c² in km
    R14 = tov_solve(Float64, b14; ρ_c=lo).R * km
    @info "DD2's star: ρ_c = $(TT_STAR.ρ_c): M = $(s4.M), R = $(s4.R * km) km; against " *
          "16000 steps ΔM = $(s4.M - s16.M), ΔR = $(s4.R - s16.R); at 1.4 M☉ ρ_c = $lo " *
          "($(lo * MSUN_DENSITY_CGS) g/cm³), R = $R14 km"
    @test abs(s4.M - s16.M) < 1e-8 * s4.M && abs(s4.R - s16.R) < 1e-8 * s4.R
    @test 13.1 < R14 < 13.4
end

@dd2_gated @testset "The composition table holds the solve's matter to 3e−5 in L1" begin
    # Guards `TOVComposition`: knots that miss the crust, or derivatives in the wrong
    # variable, would start the star off its equilibrium. `ρ★`, `s` and `Yₑ` at 40001
    # points uniform in `r̄²` against the barotrope at the solve's `η` there, weighted by
    # the volume; measured at `NC = 129`: `ρ★` 2.3e−5, `Yₑ` 1.8e−4, `s` 7.8e−4, the mass
    # 1.5e−5. Pointwise the error reaches 3% where the slice's `Yₑ` changes steeply in the
    # crust, which no smooth table follows; the L1 error falls at second order with `NC`.
    (; b) = tt_barotrope()
    sol = tov_solve(Float64, b; ρ_c=TT_STAR.ρ_c)
    c = TOVComposition(sol; NC=129)
    us = range(0, sol.R̄^2 * (1 - 1e-12); length=40001)
    ref = [barotrope_state(b, TT.tov_eta(sol, u)[1]) for u in us]
    got = [composition_state(c, u) for u in us]
    w = sqrt.(us)                                      # r̄² dr̄ ∝ √u du
    ρr = [r.ρ for r in ref]
    l1ρ = sum(w .* abs.(first.(got) .- ρr)) / sum(w .* ρr)
    l1Y = sum(w .* abs.(last.(got) .- [r.Yₑ for r in ref])) / sum(w)
    l1s = sum(w .* abs.([g[2] for g in got] .- [r.s for r in ref])) / sum(w)
    @info "DD2's composition table, NC = 129: L1 ρ★ $l1ρ, Yₑ $l1Y, s $l1s"
    @test l1ρ < 3e-5 && l1Y < 3e-4 && l1s < 1e-3
end

tt_case() = computed_once(:tt_case) do
    d = dd2_table()
    tabulated_tov_case(Float64; d.table, d.eos, TT_STAR.ρ_c, TT_STAR.ρ_min, L=16, roots=2)
end
tt_boxes(f) = tov_boxes!(f, tt_case(); levels=1)

@dd2_gated @testset "Every cell of the tabulated star's initial data is a valid state" begin
    # Guards the case's initial data at the cold edge: the star's states lie on the
    # entropy window's minimum, so a cell average that fell below it, or a composition
    # outside the table, would be repaired by the policy from the first recovery. On
    # the suite's mesh every cell above the atmosphere's trigger recovers with no policy
    # flag, and every cell below it goes to the atmosphere (measured 2224 and 30544 of
    # 32768).
    case = tt_case()
    forest = Forest{Float64}(case.roots; N=8, periodic=case.periodic, extents=case.extents)
    tt_boxes(forest)
    p = TT.MHDProblem(TT.mhd_fields(Float64, forest), case; limiter=:mc)
    u = statevector(TT.evolved_sets(p))
    TT.initial_state!(u, p, case, 0.0)
    P = TT.hostcopy(TT.update_primitives!(p, u, 0.0))
    n_star = mesh_mapreduce(x -> x > 1.01 * TT_STAR.ρ_min ? 1 : 0, +, 0, P; vars=TT.P_RHO)
    n_flagged = mesh_mapreduce(x -> unsafe_trunc(UInt32, x) & TT.EntropyEOS.FLAG_POL_ANY != 0 ?
                                    1 : 0, +, 0, P; vars=TT.P_FLAGS)
    n = nblocks(P) * 8^3
    @info "The tabulated star's initial data: $n_star cells in the star, $n_flagged repaired " *
          "by the policy, of $n"
    @test n_star > 1000 && n_star + n_flagged == n
end

@dd2_gated @testset "The tabulated Cowling star holds still and conserves rest mass and lepton number" begin
    # Guards the case as a whole (`tabulated_tov_case`, the barotrope's `η`, the cold
    # atmosphere) and `D Yₑ`'s transport through a star whose `Yₑ` varies: a star out of
    # equilibrium with its metric would move `ρ_c` at first order in `t`, and a leak of
    # `D Yₑ` at a coarse-fine face, a boundary or the reset beyond what the reset's tally
    # says would change the totals. On `h = 1`, MC, to `t = 4M`, as `tov_tests.jl`'s
    # polytrope (−3.3%): measured −0.82% in `ρ_c`; rest mass and `D Yₑ` net of the reset
    # to 6.6e−16 and 2.0e−17; no entropy floor on a state the policy did not send to the
    # atmosphere (the cold edge, measured).
    case = tt_case()
    r = evolve!(case, Val(3); N=8, t_end=4, chunk=1, limiter=:mc, refine=tt_boxes)
    ρ0 = TT.central_density(r.p, exact_state(r.p, case, 0.0), 0.0)
    ρ1 = TT.central_density(r.p, r.u, r.t)
    tot = conserved_totals(r.p, r.u)
    M0, Y0 = r.totals0[TT.U_D], r.totals0[TT.U_DYE]
    injD = sum(c -> c.tally.injected_D, r.chunks)
    injY = sum(c -> c.tally.injected_DYₑ, r.chunks)
    floors = sum(c -> c.tally.s_floors, r.chunks)
    @info "Tabulated Cowling star (DD2), h = 1, t = 4M: ρ_c $ρ0 → $ρ1 " *
          "($((ρ1 - ρ0) / ρ0)); rest mass $M0, change net of the reset " *
          "$(tot[TT.U_D] - M0 - injD); D Yₑ $Y0, change $(tot[TT.U_DYE] - Y0), injected " *
          "$injY, net $(tot[TT.U_DYE] - Y0 - injY); entropy floors $floors; $(r.nsteps) " *
          "steps; checksum $(sum(abs, r.u))"
    @test abs(ρ1 - ρ0) / ρ0 < 0.02
    @test abs(tot[TT.U_D] - M0 - injD) < 1e-14 * M0
    @test abs(tot[TT.U_DYE] - Y0 - injY) < 1e-14 * Y0
    @test abs(injD) < 1e-8 * M0
    @test floors == 0
end
