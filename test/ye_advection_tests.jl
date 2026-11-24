# Step 13: the advection of the electron fraction (`src/cases/ye_advection.jl`). `D Yₑ`
# is a conserved variable of its own, with the fluid's flux and no source; until step 13
# every case carried a constant `Yₑ = 1/2`, so nothing showed that a profile is carried,
# conserved and kept within its bounds. A `Yₑ` profile on a uniform flow of the ideal gas,
# which does not see `Yₑ`, through one period of the periodic box.

isdefined(@__MODULE__, :tier_testset) || include("tiers.jl")

const YA = TreeGRRMHD

@serial_tier @testset "A smooth Yₑ profile is carried at the scheme's order, D Yₑ to rounding" begin
    # Guards `D Yₑ`'s flux (`src/flux.jl`), its reconstruction as the specific `Yₑ`
    # (`R_YE`) and its absence from every source: a flux that used another variable's
    # speed, a reconstruction of `Yₑ` that dropped to first order, or a source would
    # break the order or the total. One period (`t = 2`, `v = 1/2`) on 32, 64 and 128
    # cells. Measured: MC 1.69, 1.86 (clipped at the extrema), PPM 2.05, 2.23; the
    # total `D Yₑ` to 6e−17 of 0.35; `D` uniform to the bit.
    for limiter in (:mc, :ppm)
        errs = Float64[]
        for roots in (4, 8, 16)
            c = ye_advection_case(; profile=:smooth, roots)
            r = evolve!(c, Val(1); N=8, t_end=2, chunk=1, limiter)
            e = l1_errors(r.p, r.u, exact_state(r.p, c, 2.0)).U
            push!(errs, e[YA.U_DYE])
            ΔY = conserved_totals(r.p, r.u)[YA.U_DYE] - r.totals0[YA.U_DYE]
            @test abs(ΔY) ≤ 8 * eps() * r.totals0[YA.U_DYE] * r.nsteps
            @test e[YA.U_D] == 0
        end
        rates = log2.(errs[1:(end - 1)] ./ errs[2:end])
        @info "Yₑ advection, $limiter: L1 errors $(round.(errs; sigdigits=3)), rates " *
              "$(round.(rates; digits=2))"
        @test all(>(1.6), rates)
    end
end

@serial_tier @testset "A Yₑ step stays within its two values" begin
    # Guards the limiter on `Yₑ`: an unlimited reconstruction, or one of `D Yₑ` rather
    # than of the specific `Yₑ`, would overshoot at the step's edges and could carry
    # `Yₑ` out of the EOS's range. One period on 64 cells; measured: within 4e−13 of
    # 0.2 and 0.4 under PPM, to the last ulp under MC.
    for limiter in (:mc, :ppm)
        c = ye_advection_case(; profile=:step, roots=8)
        r = evolve!(c, Val(1); N=8, t_end=2, chunk=1, limiter)
        pr = profile_1d(r.p, r.u, r.t)
        Y = pr.U[:, YA.U_DYE] ./ pr.U[:, YA.U_D]
        ΔY = conserved_totals(r.p, r.u)[YA.U_DYE] - r.totals0[YA.U_DYE]
        @info "Yₑ step, $limiter: Yₑ ∈ [$(minimum(Y)), $(maximum(Y))], ΔDYₑ $ΔY"
        @test 0.2 - 1e-12 ≤ minimum(Y) && maximum(Y) ≤ 0.4 + 1e-12
        @test abs(ΔY) ≤ 8 * eps() * r.totals0[YA.U_DYE] * r.nsteps
        @test maximum(pr.ρ) == minimum(pr.ρ)
    end
end
