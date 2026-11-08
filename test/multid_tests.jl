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
# The first two are this file's; the rest, which vary in more than one
# direction, are `multid_cases_tests.jl`'s. The helpers are `multid_runs.jl`'s.
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
    r1 = evolve!(c1, Val(1); N=10, t_end=1 // 20, limiter=:mc)
    c2 = extruded_case(c1, Val(2); N=10)
    r2 = md_run(c2, Val(2); N=10, t_end=1 // 20, limiter=:mc, cfl=4 // 5).r
    @test md_same_as_1d(r1, r2)
    rx = md_run(c2, Val(2); N=10, t_end=1 // 20, limiter=:mc).r
    ry = md_run(turned_case(c2), Val(2); N=10, t_end=1 // 20, limiter=:mc).r
    @test md_turned_equal(rx, ry)
end
