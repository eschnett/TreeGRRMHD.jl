# The coupled runs of step 11 ("Validation ladder" 5, "The coupling contract" in `CODE.md`),
# through the script's own functions (`tov_coupled.jl`): `problems`, the four callbacks,
# `metric_at`, `regrid_all` and `main`. The star is step 10's on `[−16, 16]³`, two roots,
# no boxes, at `N = 10` (`h = 1.6`, `Mv`'s smallest `N`) and `N = 20` (`h = 0.8`).

const CT_L, CT_ROOTS = 16, 2
ct_problems(N; kw...) = problems(Float64; N, levels=0, L=CT_L, roots=CT_ROOTS, limiter=:mc,
                                 kw...)
ct3(x) = round(x; sigdigits=3)

# Step 10's Cowling script, for the reference runs on the vertex metric of the analytic
# star (`SampledADM`), with no TreeGH.
module CTCowling
include(joinpath(@__DIR__, "..", "bin", "tov_cowling.jl"))
end

# Every owned vertex of `vm`'s set: `(n, b)`.
ct_vertices(vm) = ((n, b) for b in 1:nblocks(vm.M)
                   for n in Iterators.product(1:vm.M.forest.N, 1:vm.M.forest.N,
                                              1:vm.M.forest.N))

# TreeGH's `ADMMetric` of the star's initial state against the analytic star's
# (`SampledADM`, `K` exact) at every owned vertex: the largest differences.
function ct_adm_diff(N)
    (; p, u, star) = ct_problems(N)
    at = GH.ADMMetric(p.gh, view(u, p.s[1]), 0.0)(p.vm.M)
    ref = MHD.SampledADM(star.provider.metric, p.vm, 0.0)(p.vm.M)
    d = Dict(:α => 0.0, :β => 0.0, :γ => 0.0, :K => 0.0, :Kref => 0.0)
    for (n, b) in ct_vertices(p.vm)
        a, r = at(n, b), ref(n, b)
        d[:α] = max(d[:α], abs(a.α - r.α))
        d[:β] = max(d[:β], maximum(abs, a.β - r.β))
        d[:γ] = max(d[:γ], maximum(abs, a.γ - r.γ))
        d[:K] = max(d[:K], maximum(abs, a.K - r.K))
        d[:Kref] = max(d[:Kref], maximum(abs, r.K))
    end
    return d
end

@testset "TreeGH's metric of the star crosses at its vertices as the analytic star's" begin
    # Guards GH-1 downstream (`PLAN.md`, GH-1): TreeGH's decoded `α`, `β`, `γ` and its `K`,
    # read through `ADMMetric` bound to this package's `Mv`, against the analytic star's at
    # every owned vertex. A wrong index layout, a transposed `γ`, the wrong sign of `K` or a
    # vertex offset by one would show as an O(1) difference. The star is static with zero
    # shift, so its `K` is zero, and TreeGH's, from `Π` and its `q = 4` stencils, must be
    # zero to rounding too.
    d1, d2 = ct_adm_diff(10), ct_adm_diff(20)
    @info "ADMMetric against the analytic star at N = 10, 20: α $(ct3.((d1[:α], d2[:α]))), " *
          "β $(ct3.((d1[:β], d2[:β]))), γ $(ct3.((d1[:γ], d2[:γ]))), " *
          "K $(ct3.((d1[:K], d2[:K]))) (the analytic K: $(d2[:Kref]))"
    @test d2[:Kref] == 0
    for d in (d1, d2)
        @test d[:α] < 1e-14 && d[:β] < 1e-14 && d[:γ] < 1e-14 && d[:K] < 1e-14
    end
end

# The Cowling run on `metric`, step 10's callbacks, `nsteps` steps to `t_end`, from the
# matter segment of a fresh bundle; and the coupled run with TreeGH's right-hand side
# replaced by zero (the frozen metric), the script's callbacks otherwise.
function ct_cowling(N, t_end, nsteps)
    (; p, u, star) = ct_problems(N)
    um = copy(view(u, p.s[2]))
    metric = MHD.vertex_metric!(p.vm, MHD.SampledADM(star.provider.metric, p.vm, 0.0), 0.0)
    pc = (; p.mhd, metric)
    integ = IRK.init(IRK.IMEXProblem(CTCowling.f_exp!, CTCowling.solve_imp!, um,
                                     (0.0, t_end), pc), IRK.IMEXSSP3433();
                     dt=t_end / nsteps, stage_limiter=CTCowling.stage_limit!,
                     step_limiter=CTCowling.step_limit!, partition=:even, alias_u0=true)
    MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
    return um
end
function ct_frozen_f_exp!(du, u, p, t)
    f_exp!(du, u, p, t)
    fill!(view(du, p.s[1]), zero(eltype(du)))
    return nothing
end
function ct_frozen(N, t_end, nsteps)
    (; p, u) = ct_problems(N)
    integ = IRK.init(IRK.IMEXProblem(ct_frozen_f_exp!, solve_imp!, u, (0.0, t_end), p),
                     IRK.IMEXSSP3433(); dt=t_end / nsteps, stage_limiter=stage_limit!,
                     step_limiter=step_limit!, partition=:even, alias_u0=true)
    MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
    return u, p
end

@testset "The star on TreeGH's frozen metric is the Cowling star" begin
    # Guards the coupled plumbing with the spacetime held fixed (`PLAN.md`, step 11, "the
    # frozen metric"): the script's callbacks on a state vector of both packages' sets, the
    # metric refilled from TreeGH at every call, against step 10's Cowling run on the
    # analytic star's vertex metric. A segment offset, a metric read at the wrong state or
    # time, or TreeGH's segment leaking into the matter's would show; TreeGH's metric of
    # the static star is the analytic one to rounding (the testset above), so the runs
    # agree to rounding, well inside the truncation error the plan allows. `t = 2`, 24
    # steps, at `N = 10`.
    t_end, nsteps = 2.0, 24
    ref = ct_cowling(10, t_end, nsteps)
    u, p = ct_frozen(10, t_end, nsteps)
    d = maximum(abs, view(u, p.s[2]) - ref) / maximum(abs, ref)
    @info "Frozen TreeGH metric against the Cowling run, t = 2, N = 10: relative max " *
          "difference $(ct3(d)); identical: $(isequal(view(u, p.s[2]), ref))"
    @test d < 1e-12
    @test all(isfinite, u)
end

# TreeGH alone on the star with the matter held fixed (TreeGH's anti-Cowling run, GH-3's
# static-matter check), with this package's matter or none: the drift of TreeGH's state
# from the static star after `t_end`, its max and RMS over the state.
function ct_static_matter(N, t_end; matter::Bool)
    (; p, u, star, forest) = ct_problems(N)
    ug = copy(view(u, p.s[1]))
    MHD.prepare!(p.mhd, view(u, p.s[2]), 0.0; metric=star.provider)
    λ = GH.max_speed(p.gh, ug, 0.0)
    nsteps = MHD.chunk_count(t_end * λ, minimum_spacing(Float64, forest) / 4)
    u1 = GH.gh_solve(p.gh, ug, (0.0, t_end); dt=t_end / nsteps,
                     matter=matter ? MHD.ADMMatter(p.mhd) : nothing)
    Δ = u1 - ug
    return (; max=maximum(abs, Δ), rms=sqrt(sum(abs2, Δ) / length(Δ)))
end

@testset "TreeGH holds the static star with this package's matter (GH-3's check)" begin
    # Guards GH-3 downstream (`PLAN.md`, GH-3): TreeGH evolves the star's spacetime with
    # the matter fixed at the star's, from `ADMMatter`, and the spacetime must stay static
    # to truncation error, converging; without the matter it must not. A matter term with
    # the wrong units, index position or density (`ρ` for `e`, `S^ij` for `S_ij`) would
    # leave an O(1) drift. `t = 4`, at `N = 10` and `20`.
    w1, w2 = ct_static_matter(10, 4.0; matter=true), ct_static_matter(20, 4.0; matter=true)
    v2 = ct_static_matter(20, 4.0; matter=false)
    rate(k) = log2(w1[k] / w2[k])
    @info "TreeGH on the star with ADMMatter fixed, t = 4: drift at N = 10, 20 max " *
          "$(ct3.((w1.max, w2.max))) (rate $(round(rate(:max); digits=2))), RMS " *
          "$(ct3.((w1.rms, w2.rms))) (rate $(round(rate(:rms); digits=2))); without " *
          "the matter at N = 20: max $(ct3(v2.max)), RMS $(ct3(v2.rms))"
    @test rate(:rms) > 1.5
    @test v2.rms > 10 * w2.rms
end

# The matter-corrected ADM constraints of the initial state, inside the star away from its
# surface and outside it, with this package's matter at the transfer `order` or none.
function ct_constraints(N; matter::Bool, order::Integer)
    (; p, u, star) = ct_problems(N; order)
    MHD.prepare!(p.mhd, view(u, p.s[2]), 0.0; metric=star.provider)
    m = matter ? Main.matter(p) : nothing
    c(lo, hi) = gh_constraint_norms(p.gh, view(u, p.s[1]), 0.0; matter=m,
                                    mask=GH.ShellMask(SVector(0.0, 0.0, 0.0), lo, hi))
    return (; inside=c(0.0, 6.0).ham_l2, outside=c(10.0, 100.0).ham_l2,
            mom=sqrt(sum(abs2, c(0.0, 6.0).mom_l2)))
end

@testset "The star's matter-corrected Hamiltonian constraint converges at second order" begin
    # Guards the contract's matter as TreeGH's monitor reads it (GH-3; "Matter to the
    # vertices" in `CODE.md`): `ℋ − 16πρ` at the initial state, the analytic star's metric
    # and this package's matter averaged to the vertices, must vanish at the matter's
    # second order inside the star (`r < 6`) and stay at TreeGH's own truncation error
    # outside it (`r ≥ 10`); without the matter it is `16πρ` inside. A `ρ` with the wrong
    # factor or the wrong vertex would leave an O(1) residual.
    c1 = ct_constraints(10; matter=true, order=2)
    c2 = ct_constraints(20; matter=true, order=2)
    v2 = ct_constraints(20; matter=false, order=2)
    rate = log2(c1.inside / c2.inside)
    @info "ℋ − 16πρ at t = 0, N = 10, 20: inside $(ct3.((c1.inside, c2.inside))) (rate " *
          "$(round(rate; digits=2))), outside $(ct3.((c1.outside, c2.outside))), ℳ inside " *
          "$(ct3.((c1.mom, c2.mom))); without the matter inside $(ct3(v2.inside))"
    @test rate > 1.8
    @test v2.inside > 10 * c2.inside
end

@testset "With the fourth-order transfer the star's Hamiltonian constraint converges at fourth order" begin
    # Guards the stored adapter at order 4 as TreeGH reads it (`order = 4` in
    # `tov_coupled.jl`; "Matter to the vertices" in `CODE.md`): `ℋ − 16πρ` at the initial
    # state inside the star (`r < 6`) must fall at the transfer's fourth order, far below
    # the mean's, and stay at TreeGH's own truncation error outside (`r ≥ 10`), as the
    # mean's does there. A stored set not filled before TreeGH reads it, the mean's
    # weights at order 4, or a fallback firing inside the star would leave the second
    # order.
    c1 = ct_constraints(10; matter=true, order=4)
    c2 = ct_constraints(20; matter=true, order=4)
    m2 = ct_constraints(20; matter=true, order=2)
    rate = log2(c1.inside / c2.inside)
    @info "ℋ − 16πρ at t = 0 with the 4th-order transfer, N = 10, 20: inside " *
          "$(ct3.((c1.inside, c2.inside))) (rate $(round(rate; digits=2))), outside " *
          "$(ct3.((c1.outside, c2.outside))); the mean's inside at N = 20 $(ct3(m2.inside))"
    # Measured 2026-11-12: 1.62e−4 and 1.30e−5 (rate 3.64), against the mean's 6.17e−4 at
    # `N = 20`; outside 9.19e−8 against the mean's 8.76e−8.
    @test rate > 3.3
    @test m2.inside > 10 * c2.inside
    @test c2.outside < 1.2 * m2.outside
end

@testset "The coupled script runs, conserves rest mass and never falls back" begin
    # Guards `main` as a whole: the callbacks in the order of the data flow, the two step
    # sizes, the record, and the vertex metric of an evolving spacetime. Two chunks to
    # `t = 1` at `N = 10`: the state stays finite, the rest mass changes by exactly what the
    # reset injects (the atmosphere at rest at the Dirichlet faces lets nothing through),
    # and no cell, face or edge needs the multilinear fallback of the interpolation.
    (; p, u) = ct_problems(10)
    M0 = MHD.conserved_totals(p.mhd, view(u, p.s[2]))[1]
    rec = []
    r = main(; N=10, levels=0, L=CT_L, roots=CT_ROOTS, t_end=1, chunk=1 // 2, limiter=:mc,
             record=(t, ρ_c, c) -> push!(rec, (; t, ρ_c, ham=c.ham_l2)))
    um = view(r.u, r.p.s[2])
    ΔM = MHD.conserved_totals(r.p.mhd, um)[1] - M0
    injected = MHD.ohm_tally(r.p.mhd).injected_D
    fb = MHD.metric_fallbacks(metric_at(r.p, r.u, r.t), r.p.mhd.f)
    @info "Coupled script, N = 10, t = 1: record $(rec); rest mass $(M0), change $(ΔM), " *
          "injected $(injected); fallbacks $(fb)"
    @test all(isfinite, r.u) && r.t == 1 && length(rec) == 2
    @test abs(ΔM - injected) < 1e-14 * M0
    @test all(==(0), values(fb))
end

@testset "The fill's key skips only fills that repeat, bit for bit" begin
    # Guards `FillKey` in `tov_coupled.jl`: a skipped fill must be
    # the fill it replaces, so the run with the key and the run that fills at every call
    # end in the same state bit for bit, and the key must skip, or it costs a comparison
    # and a copy for nothing. A key that ignored `t`, missed a change of TreeGH's segment
    # or survived another TreeGH verb would leave the two runs apart.
    run(reuse_fill) = main(; N=10, levels=0, L=CT_L, roots=CT_ROOTS, t_end=1, chunk=1 // 2,
                           limiter=:mc, reuse_fill, record=(t, ρ_c, c) -> nothing)
    a, b = run(true), run(false)
    (; p, u) = ct_problems(10)
    integ = IRK.init(IRK.IMEXProblem(f_exp!, solve_imp!, u, (0.0, 1.0), p),
                     IRK.IMEXSSP3433(); dt=1 / 16, stage_limiter=stage_limit!,
                     step_limiter=step_limit!, partition=:even, alias_u0=true)
    MHD.with_step_cache(() -> IRK.step!(integ), p.mhd)       # the first fill of a step
    c0 = p.key.count[]                                       # finds the key empty
    MHD.with_step_cache(() -> IRK.step!(integ), p.mhd)
    fills = p.key.count[] - c0
    @info "FillKey: $(fills) fills in an SSP3(4,3,3) step of 11 calls; the runs with and " *
          "without the key agree bit for bit: $(isequal(a.u, b.u))"
    @test isequal(a.u, b.u) && a.t == b.t
    @test fills < 11
end

@testset "The script's regrid moves every set together and conserves rest mass" begin
    # Guards `regrid_all` (and through it GH-7's workaround): one `regrid!` over TreeGH's
    # set, the matter's and `Mv`, both problems rebuilt, the state gathered, both limiters
    # run once. A set left out would keep the old mesh's length (the state vector would
    # not match), a stale schedule or a `NaN` `Mv` would poison the next step. From the
    # uniform `N = 10` mesh, the matter's indicator refines the star (TreeGH's thresholds
    # are set not to fire): the rest mass is the same after the regrid, net of the reset's
    # injection, and a further chunk runs finite on the new mesh.
    refinement = GH.Refinement(Float64; refine_tol=0.99, coarsen_tol=0.98, maxlevel_cap=1,
                               floor_margin=0.0)
    mhd_flags = (; refine_tol=0.05, coarsen_tol=0.01, maxlevel=1)
    (; p, u, sets, forest) = ct_problems(10; refinement)
    M0 = MHD.conserved_totals(p.mhd, view(u, p.s[2]))[1]
    n0 = nblocks(p.gh.U)
    r = regrid_all(p, u, sets, forest, 0.0; mhd_flags)
    M1 = MHD.conserved_totals(r.p.mhd, view(r.u, r.p.s[2]))[1]
    injected = MHD.ohm_tally(r.p.mhd).injected_D
    @info "regrid_all at t = 0: moved $(r.moved), blocks $(n0) → $(nblocks(r.p.gh.U)); rest " *
          "mass $(M0) → $(M1), injected $(injected)"
    @test r.moved && nblocks(r.p.gh.U) > n0
    @test length(r.u) == statelength(sets)
    @test abs(M1 - M0 - injected) < 1e-14 * M0
    s = main(; N=10, levels=0, L=CT_L, roots=CT_ROOTS, t_end=1 // 2, chunk=1 // 4,
             limiter=:mc, refinement, mhd_flags, record=(t, ρ_c, c) -> nothing)
    @test all(isfinite, s.u) && nblocks(s.p.gh.U) > n0
end
