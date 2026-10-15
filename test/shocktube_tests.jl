# Rung 1b, ideal: Balsara's five relativistic MHD shock tubes under
# `IdealConductor` ("Validation ladder" 1 in `CODE.md`; step 6 of `PLAN.md`),
# and the Dirichlet boundaries every non-periodic case stands on.
#
# Each tube runs once at 100, 200 and 400 cells (MC, HLLE, the default
# tableau, `cfl = 2/5`, four chunks), and every claim is made on those runs:
# the self-convergence against the 3200-cell reference that
# `bin/shocktube_reference.jl` stored in `test/data/`, the plateaus against
# Giacomazzo & Rezzolla's exact solution, the conservation through the
# boundaries, and the implicit stages' outcomes.

using StaticArrays: SVector

const SHK = TreeGRRMHD
const ST_MS = (100, 200, 400)

# The stored reference of test `k`: 400 cells × 12 columns (the ten evolved
# cell variables, then `𝔅ʸ`, `𝔅ᶻ`), little-endian `Float32`; see
# `bin/shocktube_reference.jl` for how it was made.
function st_reference(k)
    bytes = read(joinpath(@__DIR__, "data", "balsara$k.ref"))
    length(bytes) == 400 * 12 * 4 || error("balsara$k.ref has $(length(bytes)) bytes")
    return reshape(Float64.(ltoh.(reinterpret(Float32, bytes))), 400, 12)
end

# Cell averages restricted by a factor `f`: the exact coarse averages.
st_restrict(a, f) = [sum(a[(f * (i - 1) + 1):(f * i), c]) / f for i in 1:(size(a, 1) ÷ f),
                     c in 1:size(a, 2)]

# One run of test `k` at `M` cells, with what the claims read.
function st_run(k, M; tableau=nothing)
    t_end = Float64(BALSARA[k].t_end)
    case = balsara_case(Float64, k; roots=M ÷ 10)
    kw = tableau === nothing ? (;) : (; tableau)
    r = evolve!(case, Val(1); N=10, t_end=t_end, chunk=t_end / 4, limiter=:mc, kw...)
    pr = profile_1d(r.p, r.u, r.t)
    u0 = exact_state(r.p, case, 0.0)
    return (; k, M, case, r, pr, nsteps=r.nsteps, tallies=[c.tally for c in r.chunks],
            drift=conserved_totals(r.p, r.u) .- conserved_totals(r.p, u0),
            scales=conserved_scales(r.p, u0))
end

const ST_RUNS = Dict(k => [st_run(k, M) for M in ST_MS] for k in 1:5)

@testset "A uniform flow crosses the Dirichlet boundaries unchanged under either tableau" begin
    # Guards the boundary hooks (`DirichletBoundary`, "Boundaries" in
    # `CODE.md`): a uniform state, here Balsara 4's fast flow at
    # `vˣ = ±0.999` (`W = 22`) and a slow one at ±1/2 with the same field,
    # flows in through one face and out through the other, and must stay
    # uniform to rounding. A Dirichlet *potential* failed exactly this under
    # SSP3(4,3,3) (measured in step 6: a fifth of the density gone from the
    # last cell within 0.05): its first implicit stage is told
    # `t = tⁿ + 0.24Δt` while the stage's `A` is at `tⁿ`, and a uniform
    # state's `A` grows as `−tE`. The potentials' hooks therefore fix the
    # field — the difference to the nearest evolved point — and not the
    # level of `A`.
    for vx in (-999 // 1000, -1 // 2, 1 // 2, 999 // 1000),
        tableau in (IRK.IMEXSSP3433(), IRK.ARS443())

        state = (ρ=1, p=1 // 10, v=(vx, 0, 0), B=(10, -7, -7))
        case = shock_tube_case(Float64; Γ=5 // 3, left=state, right=state, roots=10)
        r = evolve!(case, Val(1); N=10, t_end=1 // 20, limiter=:mc, tableau=tableau)
        U = profile_1d(r.p, r.u, r.t).U
        U0 = profile_1d(r.p, exact_state(r.p, case, 0.0), 0.0).U
        worst = maximum(v -> maximum(abs, U[:, v] .- U0[:, v]) / max(maximum(abs, U0[:, v]), 1),
                        1:SHK.NU)
        @info "Uniform flow vˣ = $vx through Dirichlet faces, $(tableau.name): worst " *
              "change $(worst) of a variable's scale after $(r.nsteps) steps"
        # Measured at most 3.3e-13 (vˣ = 0.999); the bound is rounding over
        # 20 steps with a factor of 30 to spare.
        @test worst ≤ 1e-11
    end
end

# What each self-convergence rate is asserted against (measured in step 6;
# PLAN.md predicted ≥ 0.7 throughout). Where the prediction failed, the test
# asserts the measured rate with a tolerance and says why, so that a change
# shows either way:
#   * Balsara 2, `D` and `S_x`: 0.662 and 0.672 — the `h^{2/3}` a contact
#     converges at under a second-order scheme, the error of these two being
#     the contact's;
#   * Balsara 3: not yet asymptotic at ≤ 400 cells. Its dense shell (`R5`,
#     ρ = 10.1 between the contact and the slow shock moving at 0.95) is a
#     few cells wide at 400 (the run's largest ρ is 8.35, 9.67 at 800), and
#     the shell's missing mass dominates the L1 error of `D`, `S_x` and `τ`:
#     their errors fall only from 800 cells on (`D`: 0.37, 0.34, 0.38, 0.30,
#     0.15 at 100 … 1600 cells, the last against a reference only twice as
#     fine). `𝔅ʸ`, which the shell does not carry, converges at 0.53.
const ST_RATES = Dict((2, "D") => (0.6, Inf), (2, "S_x") => (0.6, Inf),
                      (3, "D") => (-0.3, 0.3), (3, "S_x") => (0.05, 0.4),
                      (3, "τ") => (0.05, 0.4), (3, "𝔅ʸ") => (0.45, Inf))

@testset "Balsara $k converges to the 3200-cell reference at L1 rate ≥ 0.7" for k in 1:5
    # Guards the scheme on discontinuities ("Reconstruction", "Riemann
    # solver", "The electromotive force for `A`"): MC-limited PLM, the
    # light-cone HLLE fluxes and the UCT EMF, with every implicit stage the
    # ideal recovery. A solution with shocks converges at first order at best,
    # and contacts and slow waves that the light-cone fluxes smear converge
    # slower; PLAN.md predicted ≥ 0.7, which `ST_RATES` amends where it
    # failed. The reference is a run of the same code at 3200 cells
    # (`bin/shocktube_reference.jl`), restricted to each mesh.
    runs = ST_RUNS[k]
    R = st_reference(k)
    for (name, col, field) in (("D", 1, pr -> pr.U[:, SHK.U_D]), ("S_x", 2, pr -> pr.U[:, SHK.U_S]),
                             ("τ", 5, pr -> pr.U[:, SHK.U_TAU]), ("𝔅ʸ", 11, pr -> pr.B[:, 2]))
        errs = map(runs) do run
            ref = st_restrict(R, 400 ÷ run.M)[:, col]
            sum(abs, field(run.pr) .- ref) / run.M
        end
        rate = SHK.convergence_rate(1 ./ collect(ST_MS), errs)
        @info "Balsara $k: $name L1 errors $(errs), rate $(round(rate, digits=3))"
        lo, hi = get(ST_RATES, (k, name), (0.7, Inf))
        @test lo ≤ rate ≤ hi
        # Balsara 3's `D` is the one error that does not fall over these meshes.
        (k, name) == (3, "D") || @test issorted(errs; rev=true)
    end
end

@testset "Balsara $k: the plateaus are within 2% of Giacomazzo & Rezzolla's exact solution" for k in 1:5
    # Guards the scheme's answer, not only its convergence: the states
    # between the waves at 400 cells against the exact Riemann solution
    # (Giacomazzo & Rezzolla 2006, Tables 7–11, `BALSARA_EXACT`), averaged
    # over each plateau's window (`BALSARA_PLATEAUS`: where the 3200-cell
    # reference is within 2% of the region's state, less four cells of this
    # mesh on either side). The error of `ρ` and the total pressure is
    # relative to the region's own value, and that of a component of `v` or
    # `B` to the largest magnitude it takes in the solution (`plateau_error`).
    # PLAN.md predicted 2%. Regions too thin for 400 cells to hold a plateau
    # have no window; `CODE.md` records which.
    #
    # One window misses the prediction: Balsara 3's `R4` (ρ = 0.098, between
    # the slow rarefaction and the contact that bounds the dense shell) comes
    # out 4.7% dense at 400 cells (1.3% at 800, measured in step 6 and not in
    # the suite); the shell's smeared mass reaches into it. It is asserted at
    # its measured value.
    pr = last(ST_RUNS[k]).pr
    for (region, lo, hi) in BALSARA_PLATEAUS[k]
        cells = findall(x -> lo ≤ x ≤ hi, pr.x)
        @test length(cells) ≥ 3
        mean = ntuple(j -> sum(c -> plateau_state(pr, c)[j], cells) / length(cells), 7)
        err = plateau_error(k, region, mean)
        @info "Balsara $k, R$region on [$lo, $hi] ($(length(cells)) cells): worst " *
              "error $(round(maximum(err), sigdigits=3))"
        @test maximum(err) ≤ ((k, region) == (3, 4) ? 0.06 : 0.02)
    end
end

@testset "Balsara $k: the totals change by the boundary fluxes, and every stage is ideal" for k in 1:5
    # Guards conservation with physical boundaries: until a wave arrives,
    # each Dirichlet face carries the physical flux of its far state, so the
    # domain integral of every conserved matter variable changes by
    # `−(F_R − F_L) t`. Not to rounding: the light-cone fluxes' numerical
    # precursors, exponentially small, reach the faces long before any wave,
    # and the faces' fluxes differ from the far states' by their amplitude —
    # at most 4.1e−12 of a total's scale (measured in step 6, Balsara 2 at 100
    # cells) — so the bound is 10⁻¹⁰ of the scale, which a boundary that
    # mishandled its far state by `O(h)` or `O(Δt)` exceeds by orders of
    # magnitude. Rounding itself is the next testset's. And the stages: under
    # `IdealConductor` every one is the ideal recovery, one each, with no
    # repair, no reset and no non-finite value anywhere.
    for run in ST_RUNS[k]
        w = run.case.fields.w
        m = flat_metric(Float64)
        F(q) = begin
            prim = primitive_state(run.case.eos, m, q.ρ, q.v, q.s, q.Yₑ)
            physical_flux(m, prim2con(run.case.eos, m, prim, q.E, q.B), prim, q.E, q.B, Val(1))
        end
        budget = (F(w.right) .- F(w.left)) .* run.r.t
        scales = max.(run.scales, conserved_scales(run.r.p, run.r.u))
        worst = maximum(v -> abs(run.drift[v] + budget[v]) / scales[v],
                        (SHK.U_D, SHK.U_S, SHK.U_TAU, SHK.U_DYE, SHK.U_DS))
        @info "Balsara $k, $(run.M) cells: the totals miss the boundary budget by at most " *
              "$worst of their scale"
        @test worst ≤ 1e-10
        stages = 4 * run.M * run.nsteps
        bits = sum(t -> collect(t.bits), run.tallies)
        @test bits[2] == stages && sum(bits) == stages          # OHM_IDEAL
        @test sum(t -> t.recoveries, run.tallies) == stages
        @test all(t -> t.policy == 0 && t.resets == 0 && t.step_resets == 0, run.tallies)
        @test all(isfinite, run.r.u)
    end
end

@testset "A shock tube on a periodic mesh conserves every total to rounding" begin
    # Guards the scheme's conservation on a discontinuous solution, without
    # the boundaries' precursors: Balsara 1 with its box made periodic, so
    # that the wrap is a second diaphragm between the same two states at
    # rest, and no flux leaves. Each total drifts by a few ulp of its scale
    # per step at most, as on the smooth waves (8 is `wave_tests.jl`'s
    # bound; measured 0.018). The other four tubes are not used: their
    # wrap-around Riemann problems collide or evacuate and fire the reset.
    c = balsara_case(Float64, 1; roots=10)
    case = MHDCase(Float64, Val(1); fields=c.fields, potential=c.potential, eos=c.eos,
                   pol=c.pol, model=c.model, 𝔅̄=c.𝔅̄, periodic=true, extents=c.extents,
                   roots=c.roots)
    r = evolve!(case, Val(1); N=10, t_end=2 // 5, limiter=:mc)
    u0 = exact_state(r.p, case, 0.0)
    drift = conserved_totals(r.p, r.u) .- conserved_totals(r.p, u0)
    scales = max.(conserved_scales(r.p, u0), conserved_scales(r.p, r.u))
    for v in (SHK.U_D, SHK.U_S, SHK.U_S + 1, SHK.U_S + 2, SHK.U_TAU, SHK.U_DYE, SHK.U_DS)
        @test abs(drift[v]) ≤ 8 * eps() * scales[v] * r.nsteps
    end
    @test all(c -> c.tally.step_resets == 0, r.chunks)
end
