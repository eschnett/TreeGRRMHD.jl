# Step 12 (rung 6): the magnetized TOV star ("Validation ladder" 6 in `CODE.md`;
# `src/cases/magnetized_tov.jl`, `src/field_diagnostics.jl`). First the field itself,
# pointwise: the exterior dipole on the star's metric and its continuation inside. Then the
# scheme on the star, on `[−12, 12]³` with three roots and no boxes, so `h = 1` at `N = 8`
# (the studies' box is `bin/magnetized_tov_study.jl`'s).

using StaticArrays: SVector

const MT = TreeGRRMHD
# `σ₀ = 10⁶`: σΔt ≈ 10⁵ at `h = 1`, the stiff interior; `μ_max = 10`, the measured safe value
# ("Ohm's law and the conductivity" in `CODE.md`).
const MT_CASE = magnetized_tov_case(Float64; σ₀=1e6, μ_max=10, L=12, roots=3)

mt3(x) = round(x; sigdigits=3)
mt_norm(v) = sqrt(sum(abs2, v))

# `ε_ijk ∂ⱼ(α B_k)`, `B_k = γ_kl Bˡ`, by central differences of step `δ`: the current
# `∇ × (αB)` the field needs to be static with `E = 0`.
function mt_curl_αB(case, 𝔅, x, δ)
    function H(y)
        m = point_metric(case.provider, 0.0, y)
        return m.α * (m.γ * (𝔅(y) / m.sqrtγ))
    end
    e(j) = SVector(ntuple(k -> k == j ? δ : 0.0, 3))
    d(j) = (H(x + e(j)) - H(x - e(j))) / (2δ)
    J = hcat(d(1), d(2), d(3))                     # J[k, j] = ∂ⱼ H_k
    return SVector(J[3, 2] - J[2, 3], J[1, 3] - J[3, 1], J[2, 1] - J[1, 2])
end

@testset "The exterior dipole is a current-free static field on the star's metric" begin
    # Guards `exterior_dipole` (Wasserman & Shapiro's static dipole on Schwarzschild, in
    # isotropic coordinates): a wrong `r(r̄)`, a wrong derivative of `S` or of `a`, or a
    # flat-space dipole would leave a current `∇ × (αB) ≠ 0` outside the star, which
    # Ampère's law turns into `E` and the exterior would not stay the initial field. The
    # flat dipole `μ ẑ × x/r̄³` on the same metric is the negative control: its current is
    # the metric's, independent of the difference step.
    w = MT_CASE.fields.w
    d = w.dipole
    flat(y) = (r² = sum(abs2, y);
               (d.μ / r²^(3 // 2)) * SVector(3 * y[3] * y[1], 3 * y[3] * y[2], 3 * y[3]^2 - r²) /
               r²)
    pts = (SVector(6.0, 4.0, 5.0), SVector(10.0, 0.0, 3.0), SVector(-7.0, 5.0, -4.0),
           SVector(9.0, 1.0, 0.5))
    ratios, worst, controls = Float64[], 0.0, Float64[]
    for x in pts
        @test sum(abs2, x) > d.u_s
        B = mt_norm(dipole_curl(d, x))
        c1 = mt_norm(mt_curl_αB(MT_CASE, y -> dipole_curl(d, y), x, 1e-3))
        c2 = mt_norm(mt_curl_αB(MT_CASE, y -> dipole_curl(d, y), x, 5e-4))
        push!(ratios, c1 / c2)
        worst = max(worst, c2 / B)
        push!(controls, mt_norm(mt_curl_αB(MT_CASE, flat, x, 5e-4)) / mt_norm(flat(x)))
    end
    @info "Exterior dipole: ∇ × (αB)/|𝔅| at δ = 5·10⁻⁴ at most $(mt3(worst)), halving δ " *
          "divides it by $(mt3.(ratios)); the flat dipole's $(mt3.(controls))"
    @test all(r -> 3.5 < r < 4.5, ratios)          # the difference's own error, O(δ²)
    @test worst < 1e-8                             # 3.6e−9: the difference's error
    @test all(>(1e-3), controls)
    # The closed form's curl is the potential's, by differences.
    x = SVector(3.0, -2.0, 5.0)
    e(j) = SVector(ntuple(k -> k == j ? 1e-5 : 0.0, 3))
    J = hcat(((dipole_potential(d, x + e(j)) - dipole_potential(d, x - e(j))) / 2e-5 for j in 1:3)...)
    @test isapprox(SVector(J[3, 2] - J[2, 3], J[1, 3] - J[3, 1], J[2, 1] - J[1, 2]),
                   dipole_curl(d, x); rtol=1e-8)
    # The series below `x = 1/4` and the closed form above meet there, each within its own
    # error of the exact `S`: the closed form's cancellation costs `3ε/x²` (1.1e−14) in `S`.
    for x in (prevfloat(0.25), 0.25, 0.1, 0.5)
        S, dS = MT.dipole_S(x)
        Sx, dSx = setprecision(BigFloat, 256) do
            y = big(x)
            L = log1p(-y) + y + y^2 / 2
            -L / y^3, inv(y * (1 - y)) + 3 * L / y^4
        end
        @test isapprox(S, Sx; rtol=1.2e-14) && isapprox(dS, dSx; rtol=1e-13)
    end
    # Far away it is the flat dipole `μ/r̄³`.
    a, _ = exterior_dipole(d.M, 1e12)
    @test isapprox(a, 1e-18; rtol=1e-5)
    @test isbits(w) && isbits(MT_CASE.model)
end

@testset "The interior continues the dipole to second order at the surface, at the stated B_c" begin
    # Guards `stellar_dipole`: a mismatch of `a` or `da/du` at the surface is a jump of `B`
    # (a surface current no conductor holds there), a mismatch of the second derivative a
    # jump of the current where σ falls to zero; a wrong normalization another field.
    d = MT_CASE.fields.w.dipole
    u = d.u_s
    for δ in (1e-4, 1e-5)
        a₋, da₋ = MT.dipole_a(d, u - δ * u)
        a₊, da₊ = MT.dipole_a(d, u + δ * u)
        # Across `2δu`, a `C²` join differs by `O(δ)` in `a` and `a′`, with no `O(1)` jump in
        # `a″`: the one-sided slopes of `a′` agree to `O(δ)`.
        @test abs(a₊ - a₋) < 3 * δ * u * abs(da₋)
        @test abs(da₊ - da₋) < 3 * δ * u * abs(2 * d.c₂)
    end
    h = 1e-3 * u
    d2(f, v) = (f(v + h) - 2 * f(v) + f(v - h)) / h^2
    a′(v) = MT.dipole_a(d, v)[2]
    slope₋ = (a′(u - h) - a′(u - 2h)) / h
    slope₊ = (a′(u + 2h) - a′(u + h)) / h
    @info "Interior field: c₀, c₁, c₂ = $(mt3.((d.c₀, d.c₁, d.c₂))); a″ at the surface " *
          "$(mt3(slope₋)) inside, $(mt3(slope₊)) outside; μ = $(mt3(d.μ))"
    @test isapprox(slope₋, slope₊; rtol=1e-2)
    @test isapprox(slope₋, 2 * d.c₂; rtol=1e-2)
    m = point_metric(MT_CASE.provider, 0.0, SVector(0.0, 0.0, 0.0))
    @test isapprox(dipole_curl(d, SVector(0.0, 0.0, 0.0)) / m.sqrtγ, SVector(0.0, 0.0, 1e-4);
                   rtol=1e-14)
    @test_throws ArgumentError magnetized_tov_case(; σ₀=0, μ_max=10)
    @test_throws UndefKeywordError magnetized_tov_case(; σ₀=1e6)
end

# The suite's run: `h = 1`, MC, to `t = 2M`, with the initial profile on the same mesh.
mt_run(N) = computed_once((:magnetized_tov, N)) do
    r = evolve!(MT_CASE, Val(3); N, t_end=2, chunk=2, limiter=:mc)
    u0 = exact_state(r.p, MT_CASE, 0.0)
    prof0 = cell_profile(r.p, u0, 0.0)
    prof = cell_profile(r.p, r.u, r.t)
    (; r, u0, prof0, prof, divB0=max_div_B(r.p, u0, 0.0), divB=max_div_B(r.p, r.u, r.t))
end

const MT_R̄ = MT_CASE.fields.w.star.metric.R̄

@testset "The magnetized star keeps ∇·B at round-off, E·B small inside and its rest mass" begin
    # Guards the case on the scheme: `B` from the curl of `A` is divergence-free on every
    # cell whatever `A` holds; inside the conductor the field is frozen into a star at rest,
    # so `E·B` is the truncation error of a poloidal field (zero in the continuum); the
    # rest mass changes by exactly what the reset injects. Measured at `h = 1`, `t = 2M`:
    # `∇·B` 1.9e−16; `Σ|E·B|/ΣB²` inside 9.2e−5, `|E|/|B|` 1.1e−2 (the star's own truncation
    # velocity times `B`); the field outside `R̄` within 1.2% of the initial one.
    run = mt_run(8)
    r = run.r
    fd = field_diagnostics(run.prof; radius=MT_R̄)
    fd0 = field_diagnostics(run.prof0; radius=MT_R̄)
    dr = field_drift(run.prof, run.prof0; radius=MT_R̄)
    M0 = r.totals0[1]
    ΔM = conserved_totals(r.p, r.u)[1] - M0
    injected = sum(c -> c.tally.injected_D, r.chunks)
    @info "Magnetized star, h = 1, t = 2M: ∇·B $(mt3(run.divB0)) → $(mt3(run.divB)); field " *
          "energy inside $(mt3(fd0.energy_in)) → $(mt3(fd.energy_in)), outside " *
          "$(mt3(fd0.energy_out)) → $(mt3(fd.energy_out)); E·B inside $(mt3(fd.EB_in)), " *
          "|E|/|B| $(mt3(fd.E_in)); drift inside $(mt3(dr.inside)), outside " *
          "$(mt3(dr.outside)); charge $(mt3(fd.q_in)) / $(mt3(fd.q_out)); trigger " *
          "$(fd.trigger_in)/$(fd.trigger_out), repaired $(fd.repaired_in)/$(fd.repaired_out); " *
          "rest mass change $(ΔM), injected $(injected); $(r.nsteps) steps"
    @test run.divB0 < 1e-14 && run.divB < 1e-14
    @test fd0.EB_in == 0 && fd0.E_in == 0
    @test 0 < fd.EB_in < 2e-4
    @test fd.E_in < 2e-2
    @test dr.inside < 3e-2 && dr.outside < 2e-2
    @test abs(fd.energy_out - fd0.energy_out) < 2e-2 * fd0.energy_out
    @test abs(ΔM - injected) < 1e-14 * M0
    # Inside the star no recovery is repaired and none is where the fields' energy
    # exceeds the fluid's.
    @test fd.trigger_in == 0 && fd.repaired_in == 0
end

@testset "The stage solve's outcome map: the star by fixed point, fallbacks just outside it" begin
    # Guards the stiff interior and records the map rung 6 asks for (`ohm_map`): inside the
    # surface every stage converges by the fixed point, with no fallback and no repair; the
    # lagged-velocity fallback (flagged, `OHM_FALLBACK`, its state repaired by the policy)
    # appears only outside it, in the cells the star's surface spreads into, cold matter
    # whose energy is a small part of the field's. Measured at `h = 1`, `t ≤ 2M`: inside,
    # 127232 stages, all fixed point; outside, 18920 fallbacks of 646912 stages (2.9%),
    # every one in a cell whose centre lies outside `R̄`; the same within 25% at
    # `σ₀ = 10³` or `μ_max = 1, 0.1` (`CODE.md`, "Validation ladder" 6).
    run = mt_run(8)
    region(x) = sum(abs2, x) < MT_R̄^2 ? 1 : 2
    mp = ohm_map(run.r.p, region, 2)
    stages(e) = sum(e.bits[1:5])
    @info "Outcome map (inside R̄, outside): " *
          join(("$(e.cells) cells, $(stages(e)) stages, bits $(e.bits), policy $(e.policy)"
                for e in mp), "; ")
    inside, outside = mp
    @test inside.bits[3] == stages(inside) > 0 && inside.policy == 0 && inside.resets == 0
    @test all(iszero, inside.bits[[1, 2, 4, 5, 6, 7, 8, 9]])
    @test outside.bits[1] ≥ 0.9 * stages(outside)  # the exterior is electrovacuum
    @test 0 < outside.bits[5] ≤ 0.04 * stages(outside)
    @test outside.bits[9] == outside.bits[5]        # every fallback's state is repaired
end

@full_tier @testset "The exterior dipole stays the initial field, its drift converging at second order" begin
    # Guards the claim of rung 6 (`PLAN.md`, step 12): the field outside the star is a static
    # vacuum solution, so it moves only by the scheme's truncation error, which converges.
    # `h = 1` and `1/2` to `t = 2M`; measured on the laptop: the drift outside `R̄` 1.21e−2,
    # 3.35e−3 (rate 1.85), in the shell `R̄ + 1 … 11` 1.43e−2, 3.87e−3 (1.88), inside
    # 1.69e−2, 4.12e−3 (2.03); `E·B` inside 9.2e−5, 1.6e−5 (2.5); `|E|/|B|` inside 1.1e−2,
    # 5.5e−3 (1.0, the star's own velocity, whose surface error is first order).
    runs = (mt_run(8), mt_run(16))
    drift = [field_drift(x.prof, x.prof0; radius=MT_R̄) for x in runs]
    shell = [field_drift(x.prof, x.prof0; radius=MT_R̄ + 1, outer=11) for x in runs]
    fds = [field_diagnostics(x.prof; radius=MT_R̄) for x in runs]
    rate(a, b) = log2(a / b)
    rates = (; outside=rate(drift[1].outside, drift[2].outside),
             shell=rate(shell[1].outside, shell[2].outside),
             inside=rate(drift[1].inside, drift[2].inside),
             EB=rate(fds[1].EB_in, fds[2].EB_in), E=rate(fds[1].E_in, fds[2].E_in))
    @info "Magnetized star, h = 1 → 1/2 at t = 2M: drift outside " *
          "$(mt3.(getfield.(drift, :outside))), shell $(mt3.(getfield.(shell, :outside))), " *
          "inside $(mt3.(getfield.(drift, :inside))); E·B $(mt3.(getfield.(fds, :EB_in))), " *
          "|E|/|B| $(mt3.(getfield.(fds, :E_in))); rates $(map(x -> round(x; digits=2), rates))"
    @test rates.outside ≥ 1.7 && rates.shell ≥ 1.7 && rates.inside ≥ 1.8
    @test rates.EB ≥ 2.0
    @test runs[2].divB < 1e-14
end
