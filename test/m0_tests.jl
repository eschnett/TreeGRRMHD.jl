# Step 18: M0's free-streaming neutrinos on the forest, without feedback on
# the matter (rungs 8c and 8d; "Neutrinos", "M0" and "In the time
# integrator" in `CODE.md`). Uses `tiers.jl`'s `computed_once` and
# `@full_tier`.

const M0 = TreeGRRMHD

# A ball of cold gas at rest in flat space, `ρ` exponential in `r` from 1
# inside `r_in` to 1/4 outside `r_out`, under a uniform pressure: the
# explosion case's profile with equal pressures and no field. The contrast
# is mild because the scheme reconstructs entropy and density apart, and a
# steep ramp at uniform pressure moves the gas (step 18: v ≈ 0.09 at a
# 1000:1 ramp over three cells); here |v| stays below 3e−5.
function m0_ball(; L=0.8, roots=2, r_in=0.2, r_out=0.7, ρ_in=1.0, ρ_out=0.25, p=1e-4,
                 dim=3)
    T = Float64
    eos = M0.explosion_gas(T, 4 // 3)
    s₁ = M0.EntropyEOS.polytropic_entropy(eos, one(T))
    w = M0.MultiDCase{:explosion,T}(r_in, r_out, ρ_in, ρ_out, p, p, 0.0, 4 / 3, s₁, s₁,
                                    0.5, true)
    return MHDCase(T, Val(dim); fields=M0.WaveFields(w), potential=M0.WavePotential(w),
                   eos=eos, M0.recovery_options(eos, 1e-10)..., model=IdealConductor(),
                   𝔅̄=SVector(0.0, 0.0, 0.0), periodic=false,
                   extents=ntuple(_ -> (-L, L), Val(dim)), roots=roots)
end

const M0_GRAY = GrayRates{Float64}(; κa=(4.0, 1.0, 0.0), κs=(0.0, 0.0, 0.0),
                                   η=(1.0, 1.0, 1.0), E=(0.01, 0.02, 0.03))
const M0_THIN = M0Leakage(M0_GRAY; center=(0.0, 0.0, 0.0), r_core=0.001, feedback=false,
                          optical_depth=false)

# The steady radial solution of `(r² n)′ = (η − κ n) r²` in the uniform medium
# (`ρ = 1`): `r² n = (η/κ)(r² − 2r/κ + 2(1 − e^{−κr})/κ²)`, and `η r³/3`
# without absorption. In flat space at rest `N = n`.
m0_exact(η, κ, r) = κ == 0 ? η * r / 3 :
                    η / κ * (r^2 - 2r / κ + 2 / κ^2 * (1 - exp(-κ * r))) / r^2

# The run in the uniform medium (the ball's profile with `ρ_out = ρ_in = 1`)
# on 16³ (`roots = 2`) or 32³ (`roots = 4`), to `t_end`: the radially
# streaming solution at `r ≤ t` has reached its steady state exactly, since
# its past light cone along the ray holds the whole ray. The medium is the
# scheme's exact steady state (|v| ≈ 1e−19): with the ball's ramp the matter
# scheme smears the density's kinks at rest, and the M0 error measured
# against the initial density grew with time (step 18: 1.3e−3 at t = 0.7,
# 2.1e−3 at t = 1.2 on 16³).
m0_ball_run(roots, t_end) = computed_once((:m0_ball, roots, t_end)) do
    evolve!(Float64, m0_ball(; roots, ρ_out=1.0), Val(3); N=8, t_end=t_end, limiter=:mc,
            neutrinos=M0_THIN)
end

# `(x, N, 𝓔)` of species `i` at every owned cell of a run's state.
function m0_cells(r, i)
    p, u = r.p, r.u
    uN = statearray(u, p.sets, M0.SEG_NU)
    out = Tuple{NTuple{3,Float64},Float64,Float64}[]
    for b in axes(uN, 5), idx in CartesianIndices(size(uN)[1:3])
        x = coordinates(p.f.U, b, Tuple(idx) .+ 3)
        push!(out, (x, uN[Tuple(idx)..., M0.nu_N(i), b], uN[Tuple(idx)..., M0.nu_E(i), b]))
    end
    return out
end

function m0_error(r, i, rmax)
    errs = [abs(N - m0_exact(M0_GRAY.η[i], M0_GRAY.κa[i], sqrt(sum(abs2, x))))
            for (x, N, _) in m0_cells(r, i) if sqrt(sum(abs2, x)) ≤ rmax]
    return sum(errs) / length(errs)
end

@testset "M0's steady state in a uniform medium against the radial solution" begin
    # Guards the closure's radial streaming, the emission, the closed-form
    # absorption and the upwind fluxes together: on 16³ the steady state
    # inside r ≤ 0.6 is within a stated L1 error of the radial ODE's for an
    # absorbing species (κ = 4ρ), a weakly absorbing one and ν_x (none), and
    # 𝓔/N is the emitted energy (χ = 1 at rest).
    r = m0_ball_run(2, 0.7)
    for i in 1:3
        err = m0_error(r, i, 0.6)
        scale = maximum(N for (_, N, _) in m0_cells(r, i))
        @info "M0 in the uniform medium on 16³, L1 error / max N" i err / scale
        @test err / scale ≤ 0.004    # measured in step 18: 0.0028, 0.0032, 0.0032
        ratios = [𝓔 / N for (_, N, 𝓔) in m0_cells(r, i) if N > 1e-12]
        @test maximum(abs, ratios .- M0_GRAY.E[i]) ≤ 1e-12 * M0_GRAY.E[i]
        @test all(c -> c[2] ≥ 0 && c[3] ≥ 0, m0_cells(r, i))
    end
end

@full_tier @testset "M0's steady state converges at second order" begin
    # Guards the scheme's order: the L1 error inside r ≤ 0.6 from 16³ to 32³
    # (measured in step 18 with MC: 1.99, 2.18, 2.25; with minmod 1.71, 2.08,
    # 2.25, and 1.78, 1.83, 2.01 from 32³ to 64³, where minmod's clipping of
    # the smooth extremum that every grid line through the radial field has
    # dominated; `bin/m0_ball.jl` runs 64³).
    r16, r32 = m0_ball_run(2, 0.7), m0_ball_run(4, 0.7)
    for i in 1:3
        rate = log2(m0_error(r16, i, 0.6) / m0_error(r32, i, 0.6))
        @info "M0 in the uniform medium, convergence rate 16³ → 32³" i rate
        @test rate ≥ 1.9
    end
end

@testset "The light front: the solution at t = 0.3 is the radial one with its front at r = t" begin
    # Guards the front's position and causality: before the steady state,
    # `r² n` of ν_x (no absorption) is `∫ η r′² dr′` over the ray's past light
    # cone, `max(0, r − t) … r`, whose kink at `r = t` is the front; and with
    # feedback off the matter, the potentials and the gauge of the ball, whose
    # gas moves (|v| ≈ 3e−5), evolve bit for bit as without neutrinos.
    t = 0.3
    r = m0_ball_run(2, t)
    m_t(ρr) = (ρr^3 - max(0.0, ρr - t)^3) / 3        # ∫ r′² dr′, ρ = 1
    cells = [(sqrt(sum(abs2, x)), N) for (x, N, _) in m0_cells(r, 3) if sqrt(sum(abs2, x)) ≤ 0.75]
    errs = [abs(N - M0_GRAY.η[3] * m_t(ρr) / ρr^2) for (ρr, N) in cells]
    scale = maximum(N for (_, N) in cells)
    @info "Light front at t = 0.3, L1 error / max N (16³)" sum(errs) / length(errs) / scale
    @test sum(errs) / length(errs) ≤ 0.004 * scale   # measured in step 18: 0.0031
    ball = m0_ball(; roots=2)
    with = evolve!(Float64, ball, Val(3); N=8, t_end=t, limiter=:mc, neutrinos=M0_THIN)
    plain = evolve!(Float64, ball, Val(3); N=8, t_end=t, limiter=:mc)
    @test plain.nsteps == with.nsteps
    @test plain.u == with.u[1:length(plain.u)]
end

@testset "Across the rays: the error along an axis and along a diagonal" begin
    # Guards (and states) the price of transport on a Cartesian mesh: radial
    # rays not aligned with the grid diffuse across one another, which would
    # make a diagonal worse than an axis. Measured in step 18 on 16³, with MC:
    # 0.027 along an axis and 0.0055 along a diagonal (minmod: 0.023, 0.021) —
    # the diffusion across rays does not show; the axis's larger error is the
    # limiter's, since beside an axis N has an extremum in both transverse
    # directions, where any limiter flattens the slope.
    r = m0_ball_run(2, 0.7)
    ref(ρr) = m0_exact(M0_GRAY.η[3], M0_GRAY.κa[3], ρr)
    h = 1.6 / 16
    axis, diag = Float64[], Float64[]
    for (x, N, _) in m0_cells(r, 3)
        ρr = sqrt(sum(abs2, x))
        0.3 ≤ ρr ≤ 0.6 || continue
        e = abs(N - ref(ρr)) / ref(ρr)
        a = sort(collect(abs.(x)))
        a[1] < h && a[2] < h && push!(axis, e)
        a[3] - a[1] < h && push!(diag, e)
    end
    @info "Across the rays (16³), relative error at r = 0.3 … 0.6" axis = maximum(axis) diagonal = maximum(diag)
    @test maximum(axis) ≤ 0.035 && maximum(diag) ≤ 0.01
end

# The outflow through the outer faces of the fluxes the last `rhs!` left in
# `Fν`: high faces positive, low faces negative, weighted by the face area.
function m0_outflow(p, v)
    s = p.neutrinos
    L = 0.8
    total = 0.0
    for d in 1:3
        F = s.Fν[d]
        Fw = F.work
        for b in axes(Fw, 5)
            h = coordinates(F, b, ntuple(e -> e == d ? 2 : 1, 3))[d] -
                coordinates(F, b, ntuple(_ -> 1, 3))[d]
            for idx in CartesianIndices(size(Fw)[1:3])
                xd = coordinates(F, b, Tuple(idx))[d]
                sgn = isapprox(xd, L; atol=1e-12) ? 1 : isapprox(xd, -L; atol=1e-12) ? -1 : 0
                total += sgn * Fw[Tuple(idx)..., v, b] * h^2
            end
        end
    end
    return total
end

@testset "M0's fluxes are conservative across coarse-fine faces, and so is its regrid" begin
    # Guards the flux correction at coarse-fine faces and Nν's transfer at
    # `regrid!`: on a mesh with a refined octant through the middle of a
    # pulse, Σ du plus the outflow through the outer faces is zero to
    # round-off, before and after the pulse spreads; refining everywhere
    # leaves Σ N and Σ 𝓔 unchanged to round-off; N and 𝓔 stay non-negative.
    # (A periodic box is refused: the radial direction flips across its
    # faces, and step 18 measured the totals drifting by 1e−6 there.)
    case = m0_ball(; roots=2)
    none = M0Leakage(GrayRates{Float64}(; κa=(0.0, 0.0, 0.0), κs=(0.0, 0.0, 0.0),
                                        η=(0.0, 0.0, 0.0), E=(1.0, 1.0, 1.0));
                     center=(0.0, 0.0, 0.0), r_core=0.001, feedback=false,
                     optical_depth=false)
    forest = Forest{Float64}(case.roots; N=8, periodic=case.periodic, extents=case.extents)
    refine!(forest, [first(forest.leaves)]); balance!(forest)
    p = MHDProblem(mhd_fields(Float64, forest), case; limiter=:mc, neutrinos=none)
    u = exact_state(p, case, 0.0)
    uN = statearray(u, p.sets, M0.SEG_NU)
    for b in axes(uN, 5), idx in CartesianIndices(size(uN)[1:3])
        x = coordinates(p.f.U, b, Tuple(idx) .+ 3)
        ρr = sqrt(sum(abs2, x .- (0.05, 0.02, -0.03)))
        bump = ρr < 0.3 ? cos(π * ρr / 0.6)^2 : 0.0
        for i in 1:3
            uN[Tuple(idx)..., M0.nu_N(i), b] = i * bump
            uN[Tuple(idx)..., M0.nu_E(i), b] = 0.1 * i * bump
        end
    end
    function balance(p, u)
        du = similar(u)
        M0.prepare!(p, u, 0.0)
        M0.rhs!(du, p, 0.0)
        sums = neutrino_totals(p, du)
        scale = neutrino_totals(p, abs.(du))
        return maximum(abs(sums[v] + m0_outflow(p, v)) / scale[v] for v in 1:6)
    end
    before = balance(p, u)
    integ = mhd_integrator(p, u, 0.0, 0.2, 30; alias_u0=true)
    with_step_cache(() -> M0.IRK.solve!(integ), p)
    u = integ.u
    after = balance(p, u)
    totals1 = neutrino_totals(p, u)
    @test minimum(statearray(u, p.sets, M0.SEG_NU)) ≥ 0
    g = regrid_state!(p, u, 0.2; flags=fill(Refine, nleaves(forest)))
    totals2 = neutrino_totals(g.p, g.u)
    transfer = maximum(abs(totals2[v] - totals1[v]) / totals1[v] for v in 1:6)
    @info "M0 conservation: Σ du + outflow, before and after spreading; the regrid" before after transfer
    @test before ≤ 1e-14 && after ≤ 1e-14     # measured in step 18: 6e−17, 5e−17
    @test transfer ≤ 1e-14
    # The bounded prolongation can leave rounding-sized negatives (step 18:
    # −7e−21 of a pulse of height 3), where absorption is off.
    uN2 = statearray(g.u, g.p.sets, M0.SEG_NU)
    @test minimum(uN2) ≥ -1e-15 * maximum(uN2)
    pforest = Forest{Float64}((2, 2, 2); N=8, periodic=(true, true, true),
                              extents=ntuple(_ -> (-0.8, 0.8), 3))
    @test_throws ArgumentError M0.m0_state(none, mhd_fields(Float64, pforest), M0.default_ops_U(),
                                           M0.CPU())
end

@testset "On the Cowling star 𝓔/N outside is the emission's mean redshifted energy" begin
    # Guards χ, the metric's redshift (8d): in a static spacetime the energy
    # at infinity is conserved along each ray, so outside a static emitting
    # star 𝓔/N of a non-absorbing species is the same at every radius and on
    # every ray, the emission-weighted mean of E α over the star (χ = α for
    # matter at rest in the TOV metric, whose exterior is Schwarzschild's).
    gray = GrayRates{Float64}(; κa=(0.0, 0.0, 0.0), κs=(0.0, 0.0, 0.0), η=(0.0, 0.0, 1.0),
                              E=(1.0, 1.0, 1.0))
    nu = M0Leakage(gray; center=(0.0, 0.0, 0.0), r_core=0.5, feedback=false,
                   optical_depth=false)
    case = tov_case(Float64; L=16, roots=2)
    r = evolve!(Float64, case, Val(3); N=8, t_end=20, limiter=:mc, neutrinos=nu)
    p, u = r.p, r.u
    M0.prepare!(p, u, r.t)
    P = p.f.P.work
    uN = statearray(u, p.sets, M0.SEG_NU)
    num = den = 0.0
    ratios = Float64[]
    for b in axes(uN, 5), idx in CartesianIndices(size(uN)[1:3])
        s = Tuple(idx) .+ 3
        ρ★, α, sg = P[s..., M0.P_RHO, b], P[s..., M0.P_ALPHA, b], P[s..., M0.P_SQRTG, b]
        num += α * sg * ρ★ * α
        den += α * sg * ρ★
        x = coordinates(p.f.U, b, s)
        ρr = sqrt(sum(abs2, x))
        9 ≤ ρr ≤ 15 && push!(ratios, uN[Tuple(idx)..., M0.nu_E(3), b] / uN[Tuple(idx)..., M0.nu_N(3), b])
    end
    mean_α = num / den
    spread = (maximum(ratios) - minimum(ratios)) / mean_α
    off = maximum(abs, ratios .- mean_α) / mean_α
    @info "Cowling star: 𝓔/N outside" mean_α spread off
    # Measured in step 18 at h = 2 (the star 4 cells in radius): ⟨α⟩ 0.776,
    # 𝓔/N spread by 0.037 of it over 9 ≤ r ≤ 15, at most 0.019 from it
    # (minmod: 0.042, 0.028).
    @test spread ≤ 0.045 && off ≤ 0.025
end

# A ball of gas carried by a smooth jet: v = v₀ exp(−y²/w²) x̂ at uniform
# pressure, ρ = ρ_out + (ρ_in − ρ_out) exp(−r²/R²) advected layer by layer, so
# that ρ(x, y, t) = ρ₀(x − v(y) t, y) is exact.
struct M0Jet{T}
    v₀::T
    w::T
    R::T
    ρ_in::T
    ρ_out::T
    p::T
    Γ::T
    s₁::T
end
@inline function M0.wave_fields(j::M0Jet, x::SVector{3,T}, t::T) where {T}
    vx = j.v₀ * exp(-(x[2] / j.w)^2)
    ξ = x[1] - vx * t
    ρ = j.ρ_out + (j.ρ_in - j.ρ_out) * exp(-(ξ^2 + x[2]^2) / j.R^2)
    s = j.s₁ + log(j.p / ρ^j.Γ) / (j.Γ - 1)
    z = zero(T)
    return M0.ideal_fields(ρ, SVector(vx, z, z), s, T(1 // 2), SVector(z, z, z))
end
@inline M0.wave_potential(j::M0Jet, x::SVector{3,T}, t::T) where {T} =
    zero(SVector{3,T}), zero(T)
function m0_jet_case(; roots, v₀=0.3, w=0.3, R=0.3)
    T = Float64
    eos = M0.explosion_gas(T, 4 // 3)
    s₁ = M0.EntropyEOS.polytropic_entropy(eos, one(T))
    j = M0Jet{T}(v₀, w, R, 1.0, 0.25, 1e-4, 4 / 3, s₁)
    return MHDCase(T, Val(2); fields=M0.WaveFields(j), potential=M0.WavePotential(j),
                   eos=eos, M0.recovery_options(eos, 1e-10)..., model=IdealConductor(),
                   𝔅̄=SVector(0.0, 0.0, 0.0), periodic=false,
                   extents=((-1.0, 1.0), (-1.0, 1.0)), roots=roots)
end

# The moving ball's runs on 32², 64² and 128², to t = 0.6.
m0_jet_run(roots) = computed_once((:m0_jet, roots)) do
    evolve!(Float64, m0_jet_case(; roots), Val(2); N=8, t_end=0.6, limiter=:mc,
            neutrinos=M0_THIN)
end

# `x => (N, 𝓔)` of species `k` at every owned cell of a 2D run.
function m0_cells2(r, k)
    p, u = r.p, r.u
    uN = statearray(u, p.sets, M0.SEG_NU)
    d = Dict{NTuple{2,Float64},Tuple{Float64,Float64}}()
    for b in axes(uN, 4), idx in CartesianIndices(size(uN)[1:2])
        x = round.(coordinates(p.f.U, b, Tuple(idx) .+ 3); digits=10)
        d[x] = (uN[Tuple(idx)..., M0.nu_N(k), b], uN[Tuple(idx)..., M0.nu_E(k), b])
    end
    return d
end

@testset "A ball carried by a jet: the Doppler factor on the axis" begin
    # Guards the closure's dependence on the fluid's motion, which no test
    # at rest sees: on the jet's axis every emission point along a ray moves
    # at v₀ along it, so 𝓔/N of ν_x is E √((1 + v₀)/(1 − v₀)) ahead of the
    # centre and E √((1 − v₀)/(1 + v₀)) behind it. Measured in step 18 at
    # 0.3 ≤ |x| ≤ 0.55, the cells beside the axis, with minmod: 2.3%, 2.8% on
    # 32²; 0.70%, 0.81% on 64²; 0.22%, 0.23% on 128² (MC on 64²: 0.71%,
    # 0.81%).
    v₀ = 0.3
    χ_ahead, χ_behind = sqrt((1 + v₀) / (1 - v₀)), sqrt((1 - v₀) / (1 + v₀))
    h = 2 / 64
    ax = [(x, 𝓔 / N / M0_GRAY.E[3]) for (x, (N, 𝓔)) in m0_cells2(m0_jet_run(8), 3)
          if abs(x[2]) < h && 0.3 ≤ abs(x[1]) ≤ 0.55 && N > 0]
    ahead = maximum(abs(q - χ_ahead) / χ_ahead for (x, q) in ax if x[1] > 0)
    behind = maximum(abs(q - χ_behind) / χ_behind for (x, q) in ax if x[1] < 0)
    @info "Jet axis, 𝓔/N against the Doppler factor (64²)" ahead behind
    @test ahead ≤ 0.01 && behind ≤ 0.01
end

@full_tier @testset "A ball carried by a jet converges at second order behind the front" begin
    # Guards the face velocities' order where the fluid moves: N's
    # self-differences inside r ≤ 0.5, behind the light front at r = t = 0.6,
    # from 32² to 64² to 128². With N, 𝓔 and uⁱ reconstructed by MC (step 18)
    # the rates are 2.13, 2.18, 2.20; by minmod 2.07, 2.07, 2.09; with minmod
    # and the face velocities from the two cells' uⁱ alone 1.90, 1.96, 2.00,
    # the 64²→128² differences 10–19% larger than minmod's. (Including the
    # front, a moving kink, all fall to about 1.6.)
    function selfdiff(rc, rf, k)
        c, f = m0_cells2(m0_jet_run(rc), k), m0_cells2(m0_jet_run(rf), k)
        hf = 2 / (8rf)
        errs = Float64[]
        for (x, (N, _)) in c
            sqrt(sum(abs2, x)) ≤ 0.5 || continue
            fine = sum(f[round.(x .+ (sx * hf / 2, sy * hf / 2); digits=10)][1]
                       for sx in (-1, 1), sy in (-1, 1)) / 4
            push!(errs, abs(N - fine))
        end
        return sum(errs) / length(errs)
    end
    for k in 1:3
        rate = log2(selfdiff(4, 8, k) / selfdiff(8, 16, k))
        @info "Jet, self-convergence behind the front" k rate
        @test rate ≥ 2.03
    end
end
