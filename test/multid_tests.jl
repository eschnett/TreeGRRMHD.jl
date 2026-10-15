# Rung 2a: the scheme in two and three dimensions on a uniform mesh
# ("Validation ladder" 2 in `CODE.md`; step 7 of `PLAN.md`). The kernels of
# step 5 were written for D = 1, 2, 3 and validated in 1D; here the 2D and
# 3D paths are held to the 1D run bit for bit where the solution allows it,
# to a turned run bit for bit, to an exact solution crossing the mesh
# obliquely at second order, and to the symmetries and conservation of
# GRaM-X's and Beckwith & Stone's standard tests:
#
#   * a run that is translation-invariant along `y` (and `z`) is the 1D run;
#   * a run along `y` is the run along `x`, turned (`turned_case`);
#   * the Alfvén wave along the diagonal converges at second order;
#   * the cylindrical explosion keeps its mirror symmetries, the rotor and
#     Orszag–Tang their symmetry under a rotation by π, to rounding;
#   * Orszag–Tang and the field loop conserve every total to rounding;
#   * a 3D blast runs, symmetric, with `∇·𝔅` at rounding.
#
# **`∇·𝔅` at every step of every run** (PLAN.md: `max|∇·𝔅| h / max|𝔅| ≤
# 1e−12`): every 2D and 3D run of this file goes through `md_run`, which
# measures it after every step (`max_div_B`, as the driver's `after_step`
# monitor) and asserts the largest. The bound is 10⁴ times the rounding of a
# face value, so a curl with one wrong edge — an `O(1)` divergence — fails it
# by twelve orders of magnitude.
#
# Compiling the 2D and 3D kernels is the expensive part of this file (about
# 12 s for each dimension, and a few more per reconstruction), so the cases
# share their configurations: one gas type, `IdealConductor` throughout.

using StaticArrays: SVector

const MD = TreeGRRMHD
const MD_DIVB = 1e-12

# One run with `∇·𝔅` measured after every step, asserted, and returned.
function md_run(case, ::Val{D}; N, t_end, limiter, kw...) where {D}
    divs = Float64[]
    monitor(p, u, t) = push!(divs, max_div_B(p, u, t))
    r = evolve!(case, Val(D); N=N, t_end=t_end, limiter=limiter, after_step=monitor, kw...)
    @test length(divs) == r.nsteps
    @test maximum(divs) ≤ MD_DIVB
    u0 = exact_state(r.p, case, 0.0)
    return (; r, case, divb=maximum(divs), u0,
            drift=conserved_totals(r.p, r.u) .- conserved_totals(r.p, u0),
            scales=max.(conserved_scales(r.p, u0), conserved_scales(r.p, r.u)),
            tally=(; bits=sum(c -> collect(c.tally.bits), r.chunks),
                   policy=sum(c -> c.tally.policy, r.chunks),
                   resets=sum(c -> c.tally.resets + c.tally.step_resets, r.chunks)))
end

# The owned points of evolved set `i` of a state as one array over the whole
# uniform mesh, `(M₁, …, M_D, nvars)`, each point at its global index
# (staggered sets own `N` points a block along every dimension, as cells do).
function md_global(p, u, i)
    s = p.sets[i]
    forest = s.forest
    D = length(forest.roots)
    N = forest.N
    a = statearray(u, p.sets, i)
    h = block_spacings(forest)
    o = block_origins(forest)
    out = zeros(forest.roots .* N..., s.nvars)
    for b in 1:nblocks(s)
        g0 = ntuple(d -> round(Int, (o[b][d] - forest.extents[d][1]) / h[b]), D)
        for I in CartesianIndices(ntuple(_ -> N, D)), v in 1:(s.nvars)
            out[(Tuple(I) .+ g0)..., v] = a[I, v, b]
        end
    end
    return out
end

# Bit for bit: every evolved set of the D-dimensional run is the 1D run's,
# repeated along the dimensions the solution does not vary in.
function md_same_as_1d(r1, rD)
    D = length(rD.p.f.U.forest.roots)
    return all(1:5) do i
        a1 = md_global(r1.p, r1.u, i)
        aD = md_global(rD.p, rD.u, i)
        all(CartesianIndices(size(aD))) do I
            isequal(aD[I], a1[I[1], I[D + 1]])
        end
    end
end

# Bit for bit: the turned run is the run with `x` and `y` exchanged and the
# `z` components negated — `S_i`, `𝔈ⁱ` and `A_i` turned by
# `R(x, y, z) = (y, x, −z)`, `A_x` the turned `A_y`. Compared with `==`, which
# is bit identity for every nonzero value and leaves only the sign of a zero
# free: a `z` component that is exactly zero is `+0` in both runs, where the
# turn predicts `−0` (`0 − 0 = +0` whichever way the operands came; measured in
# step 7 on Balsara 1, whose `S_z` and `𝔈ᶻ` vanish identically).
function md_turned_equal(rx, ry)
    turnU(q, v) = v in (2, 8) ? q[v + 1] : v in (3, 9) ? q[v - 1] : v in (4, 10) ? -q[v] : q[v]
    seg = (1, 3, 2, 4, 5)                 # set i of the turned run is set seg[i] of the run
    sgn = (1, 1, 1, -1, 1)
    return all(1:5) do i
        ay = md_global(ry.p, ry.u, i)
        ax = md_global(rx.p, rx.u, seg[i])
        all(CartesianIndices(size(ay)[1:2])) do I
            J = (I[2], I[1])
            all(1:size(ay, 3)) do v
                x = i == 1 ? turnU(view(ax, J..., :), v) : sgn[i] * ax[J..., v]
                ay[I, v] == x
            end
        end
    end
end

# The largest violation of a symmetry, relative to each variable's largest
# magnitude: reflections along `axes` (a box symmetric about its centre),
# with every field reversed when `c = −1` (`E`, `B`, `A`, `Φ̃` → their
# negatives, a symmetry of the equations). Under a reflection along `k` a
# scalar is even, a polar vector's `k` component odd and the others even;
# `B` is never state, so only `S_i`, `𝔈ⁱ`, `A_i` and `Φ̃` carry parities. A
# cell `g` mirrors to `M + 1 − g`, a vertex-like point to `M + 2 − g` (the
# box's high boundary vertex is no block's, so the first vertex has no mirror
# on a closed box; on a periodic one it is its own).
function md_mirror_error(p, u, axes, c)
    pol(j) = prod(k -> k == j ? -1 : 1, axes; init=1)
    function parity(i, v)
        i == 1 && v in 2:4 && return pol(v - 1)
        i == 1 && v in 8:10 && return c * pol(v - 7)
        i == 1 && return 1
        return i ≤ 4 ? c * pol(i - 1) : c
    end
    worst = 0.0
    for i in 1:5
        q = md_global(p, u, i)
        st = staggers(p.sets[i])
        D = length(st)
        M = size(q)[1:D]
        per = p.sets[i].forest.periodic
        for v in 1:size(q, D + 1)
            scale = maximum(abs, selectdim(q, D + 1, v))
            scale == 0 && continue
            for I in CartesianIndices(M)
                g = Tuple(I)
                gm = ntuple(D) do d
                    !(d in axes) ? g[d] : st[d] == 0 ? M[d] + 1 - g[d] :
                    per[d] ? mod1(M[d] + 2 - g[d], M[d]) : M[d] + 2 - g[d]
                end
                any(d -> gm[d] > M[d], 1:D) && continue
                worst = max(worst, abs(q[g..., v] - parity(i, v) * q[gm..., v]) / scale)
            end
        end
    end
    return worst
end

# Every total but `𝔈`'s constant to rounding: TreeHydro's 8 ulp of its scale per
# step, as `wave_tests.jl` asserts.
function md_conserved(run)
    worst = 0.0
    for v in (MD.U_D, MD.U_S, MD.U_S + 1, MD.U_S + 2, MD.U_TAU, MD.U_DYE, MD.U_DS)
        run.scales[v] > 0 || continue
        ulps = abs(run.drift[v]) / (eps() * run.scales[v] * run.r.nsteps)
        @test ulps ≤ 8
        worst = max(worst, ulps)
    end
    return worst
end

# --- direction independence -----------------------------------------------------

const MD_ALFVEN = alfven_wave_case(Float64; roots=5)
const MD_T = 1 // 20                 # seven steps at N = 10

@testset "A run translation-invariant along y is the 1D run, bit for bit: $lim" for lim in (:mc, :ppm, :wenoz, :mp5)
    # Guards the 2D paths against the 1D ones (PLAN.md, step 7): the curl, the
    # cell average of `𝔅`, the fluxes in both directions, the three EMFs with
    # their transverse reconstructions, the gauge at vertices — on a solution
    # that does not vary along `y`, every one of them must reduce to the 1D
    # kernel's arithmetic exactly. Step 5's check of the same agreed only to
    # 2.7e−15: the initial data's 3² Gauss weights, multiplied out, rounded a
    # constant (amended in step 7: the average is now the middle sample plus
    # a correction in the differences, `src/driver.jl`). The 2D run's `Δt`
    # is the 1D run's (`cfl` doubled, since `Δt = cfl/(D max λ/h)`).
    cfl = MD.default_cfl(lim)
    r1 = evolve!(MD_ALFVEN, Val(1); N=10, t_end=MD_T, limiter=lim)
    c2 = extruded_case(MD_ALFVEN, Val(2); N=10)
    r2 = md_run(c2, Val(2); N=10, t_end=MD_T, limiter=lim, cfl=2 * cfl).r
    @test r2.nsteps == r1.nsteps == (lim === :mp5 ? 13 : 7)
    @test md_same_as_1d(r1, r2)
    # The initial data alone, which is where step 5's runs parted.
    @test md_same_as_1d((; p=r1.p, u=exact_state(r1.p, MD_ALFVEN, 0.0)),
                        (; p=r2.p, u=exact_state(r2.p, c2, 0.0)))
end

@testset "A run translation-invariant along y and z is the 1D run, bit for bit, in 3D: $lim" for lim in (:mc, :wenoz)
    # The same in three dimensions: four more EMF and flux directions, and the
    # face sets normal to `z`.
    c1 = alfven_wave_case(Float64; roots=5)
    r1 = evolve!(c1, Val(1); N=8, t_end=MD_T, limiter=lim)
    r3 = md_run(extruded_case(c1, Val(3); N=8), Val(3); N=8, t_end=MD_T, limiter=lim,
                cfl=3 * MD.default_cfl(lim)).r
    @test r3.nsteps == r1.nsteps
    @test md_same_as_1d(r1, r3)
end

@testset "A run along y is the run along x, turned, bit for bit: $lim" for lim in (:mc, :ppm, :wenoz, :mp5)
    # Guards the symmetry of the kernels under exchanging the two axes (PLAN.md,
    # step 7): the turned case (`R(x, y, z) = (y, x, −z)`, a rotation, so that
    # `B = curl A` turns as a vector) must give the turned run exactly. A
    # kernel that treats `x` and `y` differently — an EMF that weights its
    # four states asymmetrically, a curl with one sign wrong, a flux
    # difference summed in an order that depends on the direction — breaks
    # it in the last bit or worse.
    cx = extruded_case(MD_ALFVEN, Val(2); N=10)
    kw = (; N=10, t_end=MD_T, limiter=lim)
    rx = md_run(cx, Val(2); kw...).r
    ry = md_run(turned_case(cx), Val(2); kw...).r
    @test md_turned_equal(rx, ry)
    @test !isequal(rx.u, exact_state(rx.p, cx, 0.0))      # something happened
end

@testset "A shock tube with Dirichlet faces: 2D is 1D, and turned is turned, bit for bit" begin
    # The same two claims through the boundary hooks (`DirichletBoundary`):
    # Balsara 1, whose faces along `x` hold the far states and whose
    # potentials grow in time, in 2D with `y` periodic, and turned, with the
    # Dirichlet faces along `y`.
    c1 = balsara_case(Float64, 1; roots=10)
    r1 = evolve!(c1, Val(1); N=10, t_end=1 // 10, limiter=:mc)
    c2 = extruded_case(c1, Val(2); N=10)
    r2 = md_run(c2, Val(2); N=10, t_end=1 // 10, limiter=:mc, cfl=4 // 5).r
    @test md_same_as_1d(r1, r2)
    rx = md_run(c2, Val(2); N=10, t_end=1 // 10, limiter=:mc).r
    ry = md_run(turned_case(c2), Val(2); N=10, t_end=1 // 10, limiter=:mc).r
    @test md_turned_equal(rx, ry)
end

@testset "The Alfvén wave along the diagonal converges at second order: $lim" for lim in (:mc, :wenoz)
    # Guards what the bit-for-bit claims cannot: a solution that varies along
    # both axes, so that both fluxes, the EMF along `z` with both of its
    # transverse reconstructions, and the curl's two differences all carry
    # it. The 1D wave turned to `(1, 1)/√2` on the periodic `[0, √2]²`
    # (`oblique_case`), a quarter period at 16², 32² and 64² cells, against
    # the exact solution. Measured (MC): `S_x`, `S_z` 2.28, 2.36, `A_x`, `A_z`
    # 2.66, 2.51; (WENO-Z) 2.03, 1.92, 2.06, 2.04 — the plan's 1.8 for the
    # 1D wave holds. And the totals are conserved to rounding.
    c1 = alfven_wave_case(Float64; roots=1)
    t_end = alfven_period(c1) / 4
    runs = map((8, 16, 32)) do N
        run = md_run(oblique_case(c1; N=N, roots=2), Val(2); N=N, t_end=t_end, limiter=lim)
        (; run, e=l1_errors(run.r.p, run.r.u, exact_state(run.r.p, run.case, run.r.t)))
    end
    hs = (1 / 16, 1 / 32, 1 / 64)
    for (name, get) in (("S_x", e -> e.U[MD.U_S]), ("S_y", e -> e.U[MD.U_S + 1]),
                        ("S_z", e -> e.U[MD.U_S + 2]), ("A_x", e -> e.A[1]), ("A_y", e -> e.A[2]),
                        ("A_z", e -> e.A[3]))
        errs = [get(x.e) for x in runs]
        rate = MD.convergence_rate(collect(hs), errs)
        @info "Oblique Alfvén wave, $lim: $name L1 errors $(errs), rate $(round(rate, digits=3))"
        @test rate ≥ 1.8
        @test issorted(errs; rev=true)
    end
    worst = maximum(x -> md_conserved(x.run), runs)
    @info "Oblique Alfvén wave, $lim: worst drift $(round(worst, digits=4)) ulp of a total's scale per step"
    @test all(x -> x.run.tally.policy == 0 && x.run.tally.resets == 0, runs)
end

# --- the standard tests -----------------------------------------------------------

@testset "The cylindrical explosion keeps its mirror symmetries to rounding" begin
    # Guards the scheme on a strong 2D blast (GRaM-X §V B: a pressure ratio of
    # 3·10⁴, a shell at `W ≈ 4`): with the field along `x`, the solution is
    # symmetric under `x → −x`, and under `y → −y` with every field reversed;
    # a scheme that is not — an upwinding that prefers one side, a
    # reconstruction that is not mirror-symmetric — shows as an asymmetry far
    # above rounding long before it shows in a picture. Minmod, as GRaM-X runs
    # it, at 64² on `[−6, 6]²` to `t = 1`: no stage is repaired and no reset
    # fires (under MC the reset floors the entropy of a few cells behind the
    # front — measured in step 7, `CODE.md`). Measured asymmetry 1.6e−18 and
    # 2.5e−17 of each variable's scale; asserted ≤ 1e−12.
    run = md_run(cylindrical_explosion_case(Float64; roots=4), Val(2); N=16, t_end=1,
                 limiter=:minmod)
    ex = md_mirror_error(run.r.p, run.r.u, (1,), 1)
    ey = md_mirror_error(run.r.p, run.r.u, (2,), -1)
    @info "Cylindrical explosion, 64², t = 1: asymmetry x → −x $ex, y → −y (fields reversed) " *
          "$ey; ∇·𝔅 $(run.divb); $(run.r.nsteps) steps"
    @test ex ≤ 1e-12 && ey ≤ 1e-12
    @test run.tally.policy == 0 && run.tally.resets == 0
    @test run.tally.bits[2] == sum(run.tally.bits)        # OHM_IDEAL, every stage
    # The blast did something: the shell moves out at relativistic speed.
    P = hostcopy(update_primitives!(run.r.p, run.r.u, run.r.t))
    @test maximum(P.work[:, :, MD.P_V, :]) > 1 // 2
end

@testset "The magnetic rotor keeps its symmetry under a rotation by π to rounding" begin
    # Guards the scheme on GRaM-X's rotor (§V C: `W ≈ 10` at the cylinder's
    # edge, a density jump of 10 with no smoothing): the rotation by π about
    # the axis maps the rotor to itself with every field reversed. MC at 64²
    # to `t = 0.1`; the fast waves have not reached the faces, so the totals
    # change only by the light-cone fluxes' exponentially small precursors
    # (`shocktube_tests.jl`'s bound, 10⁻¹⁰ of the scale). Measured asymmetry
    # 2.6e−27.
    run = md_run(magnetic_rotor_case(Float64; roots=4), Val(2); N=16, t_end=1 // 10,
                 limiter=:mc)
    e = md_mirror_error(run.r.p, run.r.u, (1, 2), -1)
    worst = maximum(v -> abs(run.drift[v]) / run.scales[v],
                    (MD.U_D, MD.U_S, MD.U_S + 1, MD.U_TAU, MD.U_DYE, MD.U_DS))
    @info "Magnetic rotor, 64², t = 1/10: asymmetry under the rotation by π $e; totals " *
          "drift $worst of their scale; ∇·𝔅 $(run.divb)"
    @test e ≤ 1e-12
    @test worst ≤ 1e-10
    @test run.tally.policy == 0 && run.tally.resets == 0
end

@testset "Orszag–Tang conserves every total to rounding and keeps its symmetry" begin
    # Guards conservation in 2D on a periodic box (PLAN.md: `D`, `S_i`, `τ` to
    # round-off): the fluxes of a face computed once and used by both cells,
    # in both directions, and the explicit sources zero in flat space — so
    # each total drifts by a few ulp of its scale per step at most. And the
    # vortex's symmetry under the rotation by π about the box's centre
    # (Beckwith & Stone's "ability to maintain symmetry"), here without
    # reversing the fields. MC at 64² to `t = 1/4`. Measured asymmetry 1.6e−14
    # (the initial data's own rounding, grown over 80 steps).
    run = md_run(orszag_tang_case(Float64; roots=4), Val(2); N=16, t_end=1 // 4, limiter=:mc)
    worst = md_conserved(run)
    e = md_mirror_error(run.r.p, run.r.u, (1, 2), 1)
    @info "Orszag–Tang, 64², t = 1/4: worst drift $(round(worst, digits=4)) ulp of a total's " *
          "scale per step; asymmetry $e; ∇·𝔅 $(run.divb)"
    @test e ≤ 1e-12
    @test run.tally.policy == 0 && run.tally.resets == 0
end

@testset "A field loop is advected conservatively, and Bᶻ stays zero without vᶻ" begin
    # Guards the advection of a weak, discontinuous field (Beckwith & Stone
    # §4.2; Gardiner & Stone 2005): conservation to rounding on the periodic
    # box, and — with `vᶻ = 0` — `Eˣ = Eʸ = 0` exactly, so `A_x`, `A_y` and
    # with them `𝔅ᶻ` stay zero exactly, as Beckwith & Stone found. With
    # `vᶻ ≠ 0` the in-plane EMFs are `vᶻ B` and `𝔅ᶻ` is the discrete curl of
    # their time integral: it stays small, and its size is recorded. 64 × 32
    # cells to `t = 1` (a twenty-fifth of a crossing).
    flat = md_run(field_loop_case(Float64), Val(2); N=16, t_end=1, limiter=:mc)
    tilted = md_run(field_loop_case(Float64; vz=1 // 10), Val(2); N=16, t_end=1, limiter=:mc)
    for run in (flat, tilted)
        worst = md_conserved(run)
        @info "Field loop, vᶻ = $(run.case.fields.w.p₄): worst drift $(round(worst, digits=4)) " *
              "ulp of a total's scale per step; ∇·𝔅 $(run.divb)"
        @test run.tally.policy == 0 && run.tally.resets == 0
    end
    p, u = flat.r.p, flat.r.u
    @test all(iszero, stateview(u, p.sets, MD.SEG_A)) &&
          all(iszero, stateview(u, p.sets, MD.SEG_A + 1))
    uU = statearray(u, p.sets, MD.SEG_U)
    @test all(iszero, uU[:, :, MD.U_E, :]) && all(iszero, uU[:, :, MD.U_E + 1, :])
    # The tilted loop's 𝔅ᶻ, against its in-plane field.
    q = tilted.r
    update_fields!(q.p, q.u, q.t)
    Bz = maximum(abs, q.p.f.B[3].work)
    Bxy = max(maximum(abs, q.p.f.B[1].work), maximum(abs, q.p.f.B[2].work))
    @info "Field loop, vᶻ = 0.1/√6: max |𝔅ᶻ| / max |𝔅ˣʸ| = $(Bz / Bxy) at t = 1"
    @test Bz ≤ 1e-2 * Bxy
end

@testset "A spherical blast in 3D runs, symmetric, with ∇·𝔅 at rounding" begin
    # The 3D smoke test (PLAN.md, step 7): the explosion with `r` the distance
    # from the origin, on the periodic `[−2, 2]³` at 16³ (two roots of `N = 8`
    # a dimension), minmod, to `t = 1/2` — every kernel along every direction,
    # the twelve-edge curl, and the field's three reflections (`x → −x` alone,
    # `y → −y` and `z → −z` with the fields reversed) to rounding. Periodic,
    # because the potentials' Dirichlet hooks are not mirror-symmetric: they
    # evolve the low boundary vertex and fill the high one, and the light-cone
    # fluxes' precursors, which reach every face of so small a box within a
    # step or two, came out 1.3e−4 asymmetric there (measured in step 7; the
    # 2D explosion's faces are 30 cells from the blast, and its precursors
    # are below rounding).
    run = md_run(cylindrical_explosion_case(Float64, Val(3); L=2, roots=2, spherical=true,
                                            periodic=true),
                 Val(3); N=8, t_end=1 // 2, limiter=:minmod)
    e = (md_mirror_error(run.r.p, run.r.u, (1,), 1), md_mirror_error(run.r.p, run.r.u, (2,), -1),
         md_mirror_error(run.r.p, run.r.u, (3,), -1))
    @info "Spherical blast, 16³, t = 1/2: asymmetries $e; ∇·𝔅 $(run.divb); $(run.r.nsteps) steps"
    @test all(≤(1e-12), e)
    @test all(isfinite, run.r.u)
    @test run.tally.policy == 0 && run.tally.resets == 0
end
