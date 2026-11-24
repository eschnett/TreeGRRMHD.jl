# Step 13 (rung 7): the TOV star on a tabulated EOS (DD2, in cold β-equilibrium) in
# TreeGeneralizedHarmonic's spacetime, through `tov_coupled.jl`'s `problems(; case)`; the
# checks of `coupled_tests.jl` that hold a star at all, on this one. The star is the
# package suite's (`test/tabulated_tov_tests.jl`): `ρ_c = 9.4·10⁻⁴`, `M = 1.404`,
# `R̄ = 7.47`, its atmosphere at `10⁻¹⁰`, on `[−16, 16]³`, two roots, no boxes. Every
# testset reads the table (`test/tables.jl`).

ct_tab_case() = computed_once(:ct_tab_case) do
    d = dd2_table()
    MHD.tabulated_tov_case(Float64; d.table, d.eos, ρ_c=9.4e-4, ρ_min=1e-10, L=CT_L,
                           roots=CT_ROOTS)
end
ct_tab_problems(N; kw...) = problems(Float64; N, levels=0, limiter=:mc, case=ct_tab_case(),
                                     kw...)

@dd2_gated @testset "The tabulated star on TreeGH's frozen metric is its Cowling star" begin
    # Guards the coupled plumbing for a star whose `Yₑ` and `s` vary (`problems(; case)`):
    # as `coupled_tests.jl`'s polytrope, the script's callbacks with TreeGH's right-hand
    # side zeroed against the Cowling run on the analytic star's vertex metric, `t = 2`,
    # 24 steps, `N = 10`. A composition that reached the matter only on one path would
    # show. Measured 1.2e−13.
    t_end, nsteps = 2.0, 24
    ref = ct_cowling(10, t_end, nsteps; mk=ct_tab_problems)
    u, p = ct_frozen(10, t_end, nsteps; mk=ct_tab_problems)
    d = maximum(abs, view(u, p.s[2]) - ref) / maximum(abs, ref)
    @info "Tabulated star, frozen TreeGH metric against the Cowling run, t = 2, N = 10: " *
          "relative max difference $(ct3(d)); identical: $(isequal(view(u, p.s[2]), ref))"
    @test d < 1e-12
    @test all(isfinite, u)
end

@dd2_gated @testset "The tabulated star's matter-corrected Hamiltonian constraint converges" begin
    # Guards the tabulated star's matter as TreeGH reads it: `ℋ − 16πρ` at the initial
    # state inside the star (`r < 6`, inside the crust at `R̄ = 7.47`) at the transfer's
    # order, second and fourth, as for the polytrope; a barotrope whose `e` or `p` were
    # not the EOS's, or units off, would leave an O(1) residual against the analytic
    # metric solved from them. Measured: 2.93e−3, 8.23e−4 (rate 1.83) at order 2,
    # 1.07e−3, 7.80e−5 (3.78) at order 4; 3.44e−2 without the matter.
    c = Dict(o => (ct_constraints(10; matter=true, order=o, mk=ct_tab_problems),
                   ct_constraints(20; matter=true, order=o, mk=ct_tab_problems)) for o in (2, 4))
    v2 = ct_constraints(20; matter=false, order=2, mk=ct_tab_problems)
    rate(o) = log2(c[o][1].inside / c[o][2].inside)
    @info "Tabulated star, ℋ − 16πρ at t = 0, N = 10, 20: order 2 inside " *
          "$(ct3.((c[2][1].inside, c[2][2].inside))) (rate $(round(rate(2); digits=2))), " *
          "order 4 inside $(ct3.((c[4][1].inside, c[4][2].inside))) (rate " *
          "$(round(rate(4); digits=2))), outside $(ct3.((c[4][1].outside, c[4][2].outside))); " *
          "without the matter $(ct3(v2.inside))"
    @test rate(2) > 1.8
    @test rate(4) > 3.0
    @test v2.inside > 10 * c[2][2].inside
end

@dd2_gated @testset "The coupled tabulated star conserves rest mass and lepton number" begin
    # Guards `main` on the tabulated star: two chunks to `t = 1` at `N = 10`, finite, the
    # rest mass and `D Yₑ` changing by exactly what the reset injects, and no fallback of
    # the vertex metric's interpolation. Measured: net of the injection 1.2e−15 and
    # 2.9e−15 of the totals; no entropy floor outside the atmosphere.
    (; p, u) = ct_tab_problems(10)
    tot0 = MHD.conserved_totals(p.mhd, view(u, p.s[2]))
    r = main(; N=10, levels=0, t_end=1, chunk=1 // 2, limiter=:mc, case=ct_tab_case(),
             record=(t, ρ_c, c) -> nothing)
    um = view(r.u, r.p.s[2])
    tot = MHD.conserved_totals(r.p.mhd, um)
    tl = MHD.ohm_tally(r.p.mhd)                      # the run's: `main` never clears it
    fb = MHD.metric_fallbacks(metric_at(r.p, r.u, r.t), r.p.mhd.f)
    ΔM, ΔY = tot[MHD.U_D] - tot0[MHD.U_D], tot[MHD.U_DYE] - tot0[MHD.U_DYE]
    @info "Coupled tabulated star, N = 10, t = 1: rest mass $(tot0[MHD.U_D]), change $ΔM " *
          "(injected $(tl.injected_D)); D Yₑ $(tot0[MHD.U_DYE]), change $ΔY " *
          "(injected $(tl.injected_DYₑ)); entropy floors $(tl.s_floors); " *
          "fallbacks $(fb)"
    @test all(isfinite, r.u) && r.t == 1
    @test abs(ΔM - tl.injected_D) < 1e-14 * tot0[MHD.U_D]
    @test abs(ΔY - tl.injected_DYₑ) < 1e-14 * tot0[MHD.U_DYE]
    @test all(==(0), values(fb))
end
