# Rung 2a, the higher-order reconstructions: PPM, WENO-Z and MP5 beside
# step 5's PLM ("Reconstruction" in `CODE.md`; step 7 of `PLAN.md`). First
# pointwise — the properties the direction-independence claims of
# `multid_tests.jl` rest on, and each method's order on smooth data — then the
# circularly polarized Alfvén wave of `wave_tests.jl` under each method.

using Random: Xoshiro

const RC = TreeGRRMHD
const RC_METHODS = (:mc, :ppm, :wenoz, :mp5)

@testset "Every reconstruction is exact on a constant, odd, and mirror-symmetric, bit for bit" begin
    # Guards the offset form of `src/reconstruction.jl` (amended in step 7):
    # a face value is `q₀ + δ` with `δ` a function of the differences
    # `q_k − q₀`, odd under negation, and the lower face is the upper face of
    # the mirrored stencil. A method written with its weights multiplied out —
    # `(2a − 13b + 47c + 27d − 3e)/60` — rounds a constant to a neighbouring
    # float, and a run that is translation-invariant along `y` would then
    # differ from the 1D run in the last bit: the reason step 5's 2D check
    # agreed only to 2.7e−15. Checked bit for bit (`===`), on random stencils
    # and on constants of every magnitude.
    rng = Xoshiro(70)
    for lim in Val.(RC.MHD_LIMITERS)
        for c in (0.0, 1.0, -3.7, 1e-300, 6.02e23, -eps())
            @test RC.face_value(lim, c, c, c, c, c, 1) === c
            @test RC.face_value(lim, c, c, c, c, c, -1) === c
        end
        nodd = nmirror = 0
        for _ in 1:20_000
            q = ntuple(_ -> randn(rng) * exp(4 * randn(rng)), 5)
            v = RC.face_value(lim, q..., 1)
            RC.face_value(lim, (.-q)..., 1) === -v || (nodd += 1)
            RC.face_value(lim, reverse(q)..., -1) === v || (nmirror += 1)
        end
        @test nodd == 0
        @test nmirror == 0
    end
end

@testset "PPM and MP5 keep a monotone stencil's face value between its neighbours; WENO-Z nearly" begin
    # Guards the limiters of the higher-order methods: on monotone data the
    # face value of PPM (Colella & Woodward's constraints) and of MP5 (Suresh
    # & Huynh's bound, which contains `[q₀, q₊₁]` on monotone data) lies
    # between the two cells that share the face, as PLM's does. WENO-Z has no
    # such bound — it is essentially non-oscillatory, not monotone, and next to
    # a large step its face value can leave the range of two nearly equal
    # cells — so its overshoot is asserted small against the stencil's whole
    # rise instead: measured 5.6% at most on these 20,000 stencils, whose steps
    # range over twelve orders of magnitude; asserted ≤ 10%.
    rng = Xoshiro(71)
    worst_weno = 0.0
    for lim in (Val(:mc), Val(:ppm), Val(:mp5), Val(:wenoz))
        bad = 0
        for _ in 1:20_000
            q = cumsum(ntuple(_ -> rand(rng) < 0.6 ? exp(6 * randn(rng)) : 0.0, 5))
            v = RC.face_value(lim, q..., 1)
            lo, hi = q[3], q[4]
            if lim === Val(:wenoz)
                q[5] > q[1] &&
                    (worst_weno = max(worst_weno, max(lo - v, v - hi, 0) / (q[5] - q[1])))
            else
                lo - 1e-12 * abs(lo) ≤ v ≤ hi + 1e-12 * abs(hi) || (bad += 1)
            end
        end
        lim === Val(:wenoz) || @test bad == 0
    end
    @info "WENO-Z: largest overshoot of a monotone stencil's face value $(worst_weno) of its rise"
    @test worst_weno ≤ 0.1
end

@testset "On smooth data the face values converge at 2 (PLM), 4 (PPM) and 5 (WENO-Z, MP5)" begin
    # Guards each method's order: the value at the upper face from the exact
    # cell averages of `sin 2πx`, at a point that is no extremum, against the
    # exact point value, at four spacings. A wrong coefficient in one of the
    # candidate polynomials, or a limiter that clips smooth data, drops the
    # order. PPM's interface interpolation is fourth order away from extrema
    # (Colella & Woodward); WENO-Z and MP5 reach the fifth order of their
    # underlying polynomial. Measured 2.04, 3.97, 4.99, 5.02.
    hs = (1 / 20, 1 / 40, 1 / 80, 1 / 160)
    expect = Dict(:mc => 2, :ppm => 4, :wenoz => 5, :mp5 => 5)
    for lim in RC_METHODS
        errs = map(hs) do h
            x0 = 0.3
            avg(xc) = (cos(2π * (xc - h / 2)) - cos(2π * (xc + h / 2))) / (2π * h)
            q = ntuple(k -> avg(x0 + (k - 3) * h), 5)
            abs(RC.face_value(Val(lim), q..., 1) - sin(2π * (x0 + h / 2)))
        end
        rate = RC.convergence_rate(collect(hs), collect(errs))
        @info "Face value of sin 2πx, $lim: errors $(errs), rate $(round(rate, digits=3))"
        @test abs(rate - expect[lim]) ≤ 0.2
    end
end

@testset "MHDProblem refuses a five-cell reconstruction on G = 2 sets" begin
    # Guards the ghost width (`CODE.md`, "Field sets": `G = 3` for PPM,
    # WENO-Z and MP5): a face reads the three cells on either side, and with
    # `G = 2` the outermost face of a block would read outside its storage.
    case = em_wave_case(; roots=2)
    forest = Forest(case.roots; N=10, periodic=case.periodic, extents=case.extents)
    kw = (; eos=case.eos, opts=case.opts, pol=case.pol, model=case.model)
    g2 = mhd_fields(Float64, forest; G=2)
    for lim in (:ppm, :wenoz, :mp5)
        @test_throws "needs G ≥ 3" MHDProblem(g2; kw..., limiter=lim)
    end
    @test MHDProblem(g2; kw..., limiter=:mc) isa MHDProblem
    @test RC.default_cfl(:mp5) == 1 // 5 && RC.default_cfl(:wenoz) == 2 // 5
end

# The Alfvén wave of `wave_tests.jl` under each higher-order method, at 50,
# 100 and 200 cells (400 is in `CODE.md`, measured once: it doubles the file's
# time and moves no rate by more than 0.05).
const RC_MS = (50, 100, 200)
const RC_ALFVEN = Dict(lim => wv_study((T, roots) -> alfven_wave_case(T; roots=roots),
                                       alfven_period; limiter=lim, Ms=RC_MS)
                       for lim in (:ppm, :wenoz, :mp5))
const RC_MC = only(wv_study((T, roots) -> alfven_wave_case(T; roots=roots), alfven_period;
                            limiter=:mc, Ms=(last(RC_MS),)))

@testset "The Alfvén wave under $lim converges at second order or better, below MC's error" for lim in (:ppm, :wenoz, :mp5)
    # Guards the higher-order methods inside the whole scheme — the fluxes,
    # the UCT EMF's edge values and the face-normal fields' transverse
    # reconstruction all go through `face_value` — on the exact nonlinear
    # solution. The scheme stays second order however the faces are
    # reconstructed: the primitives are recovered pointwise from cell
    # averages, and `E = −v × B` is imposed at cell centres, both `O(h²)` off
    # the averages a fifth-order reconstruction needs (amended in step 7; see
    # "Reconstruction" in `CODE.md`). So each method must keep the order
    # (≥ 1.8, MC's measured 2.09 being the reference) and must not be worse
    # than MC at the finest mesh: at 200 cells PPM's error is 6.2 times
    # smaller than MC's in `S_y` and 9.1 times in `A_y`, WENO-Z's and MP5's 17.7
    # and 15.2 times (measured in step 7); asserted ≥ 2.
    rs = RC_ALFVEN[lim]
    rates = wv_rates(rs, (WV_MATTER[3:4]..., WV_FIELD[3], WV_FIELD[4]))
    for (name, r) in rates
        @info "Alfvén wave, $lim: $name L1 errors $(r.errs), rate $(round(r.rate, digits=3))"
        @test r.rate ≥ 1.8
        @test r.decreasing
    end
    finest = last(rs).e
    for (name, get) in (WV_MATTER[3], WV_FIELD[3])
        ratio = get(RC_MC.e) / get(finest)
        @info "Alfvén wave at $(last(RC_MS)) cells: $name error MC / $lim = $(round(ratio, digits=2))"
        @test ratio ≥ 2
    end
    worst = wv_conservation(rs)
    @info "Alfvén wave, $lim: worst drift of a conserved total $(round(worst, digits=4)) ulp of its scale per step"
    for r in rs
        stages = 4 * r.ncells * r.nsteps
        bits = sum(t -> collect(t.bits), r.tallies)
        @test bits[2] == stages && sum(bits) == stages      # OHM_IDEAL
        @test all(t -> t.policy == 0 && t.resets == 0 && t.step_resets == 0, r.tallies)
    end
end
