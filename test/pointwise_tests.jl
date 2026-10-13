# The pointwise physics of step 2: the metric providers, the stress-energy,
# Ohm's law, the fluxes and the Riemann solvers, the UCT–HLL EMF and the
# sources. Everything a right-hand-side kernel will call at one point, with no
# mesh. See "Maxwell", "Ohm's law and the conductivity", "Stress-energy for
# the spacetime", "Riemann solver", "The electromotive force for `A`",
# "Sources" and "Metric providers" in `CODE.md`.
#
# The claims are made against something outside the formula under test
# wherever there is one: the four-dimensional stress-energy built from the
# four-velocity and the field tensor, the exact upwind solution of the vacuum
# Maxwell equations, an exact electromagnetic plane wave, IllinoisGRMHD's
# Faraday law, Banyuls et al.'s covariant energy source, and SpacetimeMetrics'
# own derivatives through ForwardDiff. The signs are what these tests are for:
# a sign error in a flux, a shift term or a cross product is invisible in
# most self-consistency checks and wrong everywhere else.
#
# The states come from `recovery_tests.jl`'s generators (`rec_event`,
# `rec_metric`, `rec_prim`, `rec_direction`, `rec_gas`), which is included
# before this file: random fluids with W ≤ 10 in a boosted spinning
# Kerr-Schild hole between r = 3M and 20M, with fields carrying 1e-4 to 100
# times the fluid's energy.

using ForwardDiff
using KernelAbstractions: @kernel, @index, CPU, synchronize
using Random: Xoshiro
using StaticArrays: SVector, SMatrix, SArray
import SpacetimeMetrics

const PW = TreeGRRMHD

# The Levi-Civita symbol [ijk].
pw_levi(i, j, k) = (i - j) * (j - k) * (k - i) ÷ 2

# The four-metric `g_ab` of a point metric (index 1 is `t`), built here from
# the 3+1 definitions rather than read from SpacetimeMetrics.
function pw_g4(m::Metric3{T}) where {T}
    βl = m.γ * m.β
    gtt = -m.α^2 + sum(βl .* m.β)
    return SMatrix{4,4,T}(gtt, βl[1], βl[2], βl[3],
                          βl[1], m.γ[1, 1], m.γ[2, 1], m.γ[3, 1],
                          βl[2], m.γ[1, 2], m.γ[2, 2], m.γ[3, 2],
                          βl[3], m.γ[1, 3], m.γ[2, 3], m.γ[3, 3])
end

# The unit normal `nᵃ = (1, −βⁱ)/α`.
pw_n(m::Metric3) = SVector(1, -m.β[1], -m.β[2], -m.β[3]) / m.α

# A random state: fluid, fields, and the conserved variables. The fields
# carry between 1e-4 and 100 times the fluid's `τ` (at most `fmax`), shared
# at random between E and B.
function pw_state(rng, eos, m; fmax=100.0)
    prim = rec_prim(rng, eos, m)
    zero3 = zero(SVector{3,Float64})
    τ_fluid = PW.cons_τ(prim2con(eos, m, prim, zero3, zero3)) / m.sqrtγ
    ratio = exp10(-4 + (4 + log10(fmax)) * rand(rng))
    share = rand(rng)
    E = sqrt(2 * ratio * share * τ_fluid) * rec_direction(rng, m)
    B = sqrt(2 * ratio * (1 - share) * τ_fluid) * rec_direction(rng, m)
    return prim, E, B, prim2con(eos, m, prim, E, B)
end

# A primitive state with no matter, for the field-only tests: zero density
# and pressure, at rest, `h = 1`.
pw_vacuum(::Type{T}=Float64) where {T} =
    PW.pack_prim(zero(T), zero(SVector{3,T}), T(5), T(1 // 2), zero(T), zero(T), one(T),
                 zero(T), T(NaN))

@testset "The metric providers give SpacetimeMetrics' metric, the derivative index converted once" begin
    # Guards the one place the derivative index order is converted
    # ("Sharp edges" in `PLAN.md`): a `dg` read with SpacetimeMetrics'
    # derivative-last order would put a time derivative where a spatial one
    # belongs, and the sources would be wrong by terms that vanish only in a
    # static metric. Also the lapse and shift derivatives derived from it,
    # against ForwardDiff of the 3+1 split itself, and `Flat()`.
    rng = Xoshiro(31)
    worst_α = worst_β = worst_K = 0.0
    for _ in 1:100
        m4, p = rec_event(rng)
        prov = Analytic(m4)
        t, x = p[1], SVector(p[2], p[3], p[4])
        m = point_metric(prov, t, x)
        ref = Metric3(SpacetimeMetrics.adm_decompose(m4, p)...)
        @test m.α == ref.α && m.β == ref.β && m.γ == ref.γ && m.γu == ref.γu &&
              m.sqrtγ == ref.sqrtγ
        dm = metric_gradient(prov, t, x)
        _, sdg = SpacetimeMetrics.dmetric(m4, p)
        @test all(dm.dg[j, a, b] == sdg[a, b, j + 1] for j in 1:3, a in 1:4, b in 1:4)
        # `K_ij` is computed from the same `dmetric` call (SpacetimeMetrics'
        # `ExtrinsicCurvature` differentiates again and allocates), so it
        # agrees with upstream's to rounding.
        K_ref = SpacetimeMetrics.ExtrinsicCurvature(m4, p)
        worst_K = max(worst_K, maximum(abs.(dm.K - K_ref)) / maximum(abs.(K_ref)))
        @test dm.K == dm.K'

        split(y) = SpacetimeMetrics.adm_decompose(m4, SVector(t, y[1], y[2], y[3]))
        dα_ad = ForwardDiff.gradient(y -> split(y)[1], x)
        dβ_ad = ForwardDiff.jacobian(y -> split(y)[2], x)       # [i, j] = ∂_j βⁱ
        dα = PW.d_lapse(m, dm)
        dβ = PW.d_shift(m, dm)                                  # [j, i] = ∂_j βⁱ
        # Relative to the terms of `d_lapse`'s and `d_shift`'s formulas,
        # which cancel: `|∂g_tt| + 2|β||∂β_k| + |β|²|∂γ|` over `2α`, and
        # `|γ^ik|(|∂β_k| + |∂γ||β|)`.
        bmax = maximum(abs.(m.β))
        gmax = maximum(abs.(dm.dg[:, 2:4, 2:4]))
        βmax = maximum(abs.(dm.dg[:, 1, 2:4]))
        α_scale = (maximum(abs.(dm.dg[:, 1, 1])) + 6 * bmax * βmax + 9 * bmax^2 * gmax) / (2 * m.α)
        β_scale = 3 * maximum(abs.(m.γu)) * (βmax + 3 * gmax * bmax)
        worst_α = max(worst_α, maximum(abs.(dα - dα_ad)) / α_scale)
        worst_β = max(worst_β, maximum(abs.(transpose(dβ) - dβ_ad)) / β_scale)
    end
    @info "Lapse and shift derivatives from ∂g against ForwardDiff, K against upstream, relative" worst_α worst_β worst_K
    @test worst_K < 1e-13
    @test worst_α < 1e-14
    @test worst_β < 1e-14

    x = SVector(1.0f0, 2.0f0, 3.0f0)
    m = point_metric(Flat(), 0.5f0, x)
    @test m isa Metric3{Float32} && m.α == 1 && m.β == zero(m.β) && m.γ == m.γu
    dm = metric_gradient(Flat(), 0.5f0, x)
    @test dm isa MetricGradient{Float32} && all(iszero, dm.dg) && all(iszero, dm.K)
    @test isbits(Analytic(SpacetimeMetrics.KerrSchild{Float32}(1, 0.5f0))) && isbits(Flat())
end

@testset "The stress-energy is the fluid's and Maxwell's tensor, and projects back to 𝓔 and S_i" begin
    # Guards the 3+1 projections every flux and source is built from, and the
    # `T_ab` TreeGH will read: a term of `S^ij` dropped or of the wrong sign,
    # a cross product with the symbol instead of the tensor, and `T_ab`
    # assembled with the shift on the wrong index. The fluid part is checked
    # against `ρh uᵃuᵇ + p gᵃᵇ`, the projections against `nᵃnᵇT_ab = 𝓔` and
    # `−γ_iᵃ nᵇ T_ab = S_i` (the step-2 acceptance), and `𝓔`, `S_i` against
    # what `prim2con` packs.
    eos = rec_gas(2)
    rng = Xoshiro(32)
    worst_proj = worst_low = worst_fluid = worst_trace = worst_cons = 0.0
    for _ in 1:300
        m = rec_metric(rng)
        prim, E, B, cons = pw_state(rng, eos, m)
        𝓔, S, Sij = stress_energy(m, prim, E, B)
        @test Sij == Sij'
        Tl = T4_lower(m, 𝓔, S, Sij)
        Tu = PW.T4_upper(m, 𝓔, S, Sij)
        g = pw_g4(m)
        n = pw_n(m)
        scale = maximum(abs.(Tl))
        # nᵃnᵇT_ab = 𝓔 and −nᵇT_ib = S_i, since γ_iᵃ = δ_iᵃ + n_i nᵃ and n_i = 0.
        Tn = Tl * n
        worst_proj = max(worst_proj, abs(sum(n .* Tn) - 𝓔) / scale,
                         maximum(abs.(-SVector(Tn[2], Tn[3], Tn[4]) - S)) / scale)
        worst_low = max(worst_low, maximum(abs.(g * Tu * g - Tl)) / scale)
        # The densitized `τ + D` and `S` that `prim2con` packs.
        sg = m.sqrtγ
        worst_cons = max(worst_cons,
                         abs((PW.cons_τ(cons) + PW.cons_D(cons)) / sg - 𝓔) / 𝓔,
                         sqrt(PW.dot_down(m, PW.cons_S(cons) / sg - S,
                                          PW.cons_S(cons) / sg - S)) / 𝓔)

        # The fluid alone is `ρh uᵃuᵇ + p gᵃᵇ` with `uᵃ = W (nᵃ + vᵃ)`.
        z3 = zero(SVector{3,Float64})
        Tf = PW.T4_upper(m, stress_energy(m, prim, z3, z3)...)
        W = PW.prim_W(prim)
        v = PW.prim_v(prim)
        u = W * (n + SVector(0, v[1], v[2], v[3]))
        Tf_ref = PW.prim_ρ(prim) * PW.prim_h(prim) * (u * u') + PW.prim_p(prim) * inv(g)
        worst_fluid = max(worst_fluid, maximum(abs.(Tf - Tf_ref)) / maximum(abs.(Tf_ref)))
        # The field alone is trace-free.
        Tem = PW.T4_upper(m, stress_energy(m, pw_vacuum(), E, B)...)
        worst_trace = max(worst_trace, abs(sum(g .* Tem)) / maximum(abs.(Tem)))
    end
    @info "Stress-energy: projections, lowering, fluid, EM trace, against prim2con" worst_proj worst_low worst_fluid worst_trace worst_cons
    @test worst_proj < 1e-13
    @test worst_low < 1e-13
    @test worst_fluid < 1e-13
    @test worst_trace < 1e-13
    @test worst_cons < 1e-13

    # The field's tensor against the field tensor itself, in flat space:
    # `F_i0 = E_i` and `F_ij = [ijk] Bᵏ` are `E = −∇Φ − ∂ₜA` and `B = ∇ × A`
    # for `A_a = (−Φ, A_i)`, and `T^ab = F^ac F^b_c − ¼ ηᵃᵇ F_cd F^cd`. This
    # pins the sign of the momentum `ε_ijk Eʲ Bᵏ` (the Poynting vector), which
    # no projection identity can.
    η = SMatrix{4,4,Float64}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    m = flat_metric()
    worst = 0.0
    for _ in 1:20
        E = SVector{3}(randn(rng, 3))
        B = SVector{3}(randn(rng, 3))
        F = SMatrix{4,4,Float64}((a == b ? 0.0 :
                                  a == 1 ? -E[b - 1] :
                                  b == 1 ? E[a - 1] :
                                  sum(pw_levi(a - 1, b - 1, k) * B[k] for k in 1:3))
                                 for a in 1:4, b in 1:4)
        Fu = η * F * η
        F² = sum(F .* Fu)
        T_ref = Fu * η * Fu' - η * F² / 4
        T = PW.T4_upper(m, stress_energy(m, pw_vacuum(), E, B)...)
        worst = max(worst, maximum(abs.(T - T_ref)) / maximum(abs.(T_ref)))
    end
    @info "Flat-space Maxwell stress-energy against the field tensor's" worst
    @test worst < 1e-14
end

@testset "The fluxes transport the stress-energy, and the light cone is null" begin
    # Guards every term of the ten physical fluxes in a curved metric: the
    # momentum flux must be `α√γ Tᵈ_j`, the flux of `τ + D` must be
    # `α²√γ T^{td}`, the baryon flux `α√γ ρ uᵈ`, and the flux of `𝔈` must be
    # Komissarov's `−[kdl] H_l` with `H = αB − β × E`. Each reference is
    # built from the four-tensor or written with explicit symbols, not from
    # the flux's own formulas. And the light cone `−βᵈ ± α√γ^{dd}` must be
    # the null directions of `g^ab`.
    eos = rec_gas(5 // 3)
    rng = Xoshiro(33)
    worst_D = worst_S = worst_τ = worst_E = worst_null = 0.0
    for _ in 1:300
        m = rec_metric(rng)
        prim, E, B, cons = pw_state(rng, eos, m)
        𝓔, S, Sij = stress_energy(m, prim, E, B)
        Tu = PW.T4_upper(m, 𝓔, S, Sij)
        g = pw_g4(m)
        Tmix = Tu * g                                    # [a, b] = Tᵃ_b
        α, β, sg = m.α, m.β, m.sqrtγ
        W = PW.prim_W(prim)
        v = PW.prim_v(prim)
        H = α * PW.lower(m, B) - PW.cross_down(m, β, E)
        for d in 1:3
            F = physical_flux(m, cons, prim, E, B, Val(d))
            ρu = PW.prim_ρ(prim) * W * (v[d] - β[d] / α)
            fscale = α * sg * maximum(abs.(Tmix))
            worst_D = max(worst_D, abs(F[PW.U_D] - α * sg * ρu) / (α * sg * PW.prim_ρ(prim) * W))
            @test isapprox(F[PW.U_DYE], PW.prim_Yₑ(prim) * F[PW.U_D]; rtol=4eps())
            @test isapprox(F[PW.U_DS], PW.prim_s(prim) * F[PW.U_D]; rtol=4eps())
            F_S = SVector(F[PW.U_S], F[PW.U_S + 1], F[PW.U_S + 2])
            worst_S = max(worst_S,
                          maximum(abs.(F_S - α * sg * SVector(ntuple(j -> Tmix[d + 1, j + 1], 3)))) /
                          fscale)
            worst_τ = max(worst_τ,
                          abs(F[PW.U_TAU] + F[PW.U_D] - α^2 * sg * Tu[1, d + 1]) / fscale)
            F_E = SVector(F[PW.U_E], F[PW.U_E + 1], F[PW.U_E + 2])
            F_E_ref = SVector(ntuple(k -> -sum(pw_levi(k, d, l) * H[l] for l in 1:3), 3))
            escale = maximum(abs.(H)) + maximum(abs.(sg * E)) * maximum(abs.(β))
            worst_E = max(worst_E, maximum(abs.(F_E - F_E_ref)) / escale)

            # `k_a = (−λ, e_d)` is null for both speeds.
            gu = inv(g)
            for λ in light_cone(m, Val(d))
                k = SVector(ntuple(a -> a == 1 ? -λ : a == d + 1 ? 1.0 : 0.0, 4))
                worst_null = max(worst_null, abs(k' * gu * k) / (gu[d + 1, d + 1] + λ^2 / α^2))
            end
            λm, λp = light_cone(m, Val(d))
            @test λp - λm > 0
        end
    end
    @info "Fluxes against the stress-energy and Komissarov's H; light-cone nullity" worst_D worst_S worst_τ worst_E worst_null
    @test worst_D < 1e-14
    @test worst_S < 1e-13
    @test worst_τ < 1e-13
    @test worst_E < 1e-14
    @test worst_null < 1e-13
end

# A cyclic relabelling of the axes, `x → y → z → x`: the new components are
# `w′ = (w₃, w₁, w₂)`, so what was along x is along y. It keeps the
# orientation, so cross products and curls keep their sign.
pw_rot(w::SVector{3}) = SVector(w[3], w[1], w[2])
pw_rot(γ::SMatrix{3,3}) = SMatrix{3,3}(γ[mod1(i - 1, 3), mod1(j - 1, 3)] for i in 1:3, j in 1:3)
pw_rot(m::Metric3) = Metric3(m.α, pw_rot(m.β), pw_rot(m.γ))
pw_rot_prim(p) = PW.pack_prim(PW.prim_ρ(p), pw_rot(PW.prim_v(p)), PW.prim_s(p), PW.prim_Yₑ(p),
                              PW.prim_w(p), PW.prim_p(p), PW.prim_h(p), PW.prim_cs²(p),
                              PW.prim_ueos(p))
pw_rot_cons(u) = PW.pack_cons(PW.cons_D(u), pw_rot(PW.cons_S(u)), PW.cons_τ(u), PW.cons_DYₑ(u),
                              PW.cons_Ds(u), pw_rot(PW.cons_𝔈(u)))

@testset "Both Riemann solvers are consistent, and every direction is direction 1 relabelled" begin
    # Guards the two acceptance properties of a flux function: `F(U, U)` is
    # the physical flux (an averaging or upwinding term that does not vanish
    # on equal states), and the direction-`d` flux is the direction-1 flux of
    # the relabelled state (a component or a sign that is right along x and
    # wrong along y or z). In flat space, as the acceptance asks, and in the
    # relabelled Kerr-Schild metrics too, where the shift and the
    # off-diagonal metric are relabelled with the state.
    eos = rec_gas(4 // 3)
    rng = Xoshiro(34)
    worst_llf = worst_hlle = worst_perm_flat = worst_perm = 0.0
    for k in 1:300
        m = k <= 100 ? flat_metric() : rec_metric(rng)
        prim, E, B, cons = pw_state(rng, eos, m)
        s = (cons, prim, E, B)
        for d in (Val(1), Val(2), Val(3))
            F = physical_flux(m, s..., d)
            scale = maximum(abs.(F))
            # `(F + F)/2 − λ·0` is `F` in exact arithmetic and in rounding,
            # but StaticArrays' products are `muladd`s, which LLVM may fuse
            # differently where `physical_flux` is inlined into the solver;
            # so both are equal to rounding, not bit for bit.
            Fl = riemann_flux(Val(:llf), m, s, s, d)
            worst_llf = max(worst_llf, maximum(abs.(Fl .- F)) / scale)
            Fh = riemann_flux(Val(:hlle), m, s, s, d)
            worst_hlle = max(worst_hlle, maximum(abs.(Fh .- F)) / scale)
        end
        # Relabelled once and twice: direction 2 and 3 of the new state.
        m1, s1 = m, s
        F1 = physical_flux(m, s..., Val(1))
        for d in (Val(2), Val(3))
            m1 = pw_rot(m1)
            s1 = (pw_rot_cons(s1[1]), pw_rot_prim(s1[2]), pw_rot(s1[3]), pw_rot(s1[4]))
            F1 = pw_rot_cons(F1)
            for solver in (Val(:llf), Val(:hlle))
                L = s1
                Fd = riemann_flux(solver, m1, L, L, d)
                err = maximum(abs.(Fd .- F1)) / maximum(abs.(F1))
                k <= 100 ? (worst_perm_flat = max(worst_perm_flat, err)) :
                (worst_perm = max(worst_perm, err))
            end
        end
    end
    @info "F(U,U) by LLF and HLLE, relabelled directions (flat, Kerr-Schild), relative" worst_llf worst_hlle worst_perm_flat worst_perm
    @test worst_llf < 1e-14
    @test worst_hlle < 1e-14
    @test worst_perm_flat < 1e-14
    @test worst_perm < 1e-13
end

@testset "HLLE and the UCT–HLL EMF are the exact upwind solver of vacuum Maxwell" begin
    # Guards the signs and the speeds of the field's dissipation, in a
    # curved metric. In a constant metric the vacuum Maxwell equations in 1D
    # are linear, and with `Eˣ` and `Bˣ` continuous every jump is carried by
    # light waves moving at exactly `λ±`. HLL with exactly those speeds is
    # then the exact Godunov solver: its flux is the physical flux of its
    # own middle state. That holds only if the flux of `𝔈` (Riemann solver)
    # and the EMF (`uct_hll_emf`, through `∂ₜ𝔅ʸ = ∂ₓEMF_z` and
    # `∂ₜ𝔅ᶻ = −∂ₓEMF_y`) form one hyperbolic system with those speeds, and
    # the upwind terms have the signs that system needs: a sign error in the
    # EMF's shift term, in its jump terms or in the flux of `𝔈` breaks it.
    # The metrics are Kerr-Schild samples, frozen.
    eos = rec_gas(2)
    rng = Xoshiro(35)
    worst = 0.0
    straddled = 0
    for _ in 1:300
        m = rec_metric(rng)
        prim = rec_prim(rng, eos, m)
        Bx = randn(rng)
        Ex = randn(rng)
        E_L = SVector(Ex, randn(rng), randn(rng))
        E_R = SVector(Ex, randn(rng), randn(rng))
        B_L = SVector(Bx, randn(rng), randn(rng))
        B_R = SVector(Bx, randn(rng), randn(rng))
        L = (prim2con(eos, m, prim, E_L, B_L), prim, E_L, B_L)
        R = (prim2con(eos, m, prim, E_R, B_R), prim, E_R, B_R)
        s_L, s_R = light_cone(m, Val(1))
        s_L < 0 < s_R || continue
        straddled += 1
        sg = m.sqrtγ
        # The numerical fluxes along x: of 𝔈 by the Riemann solver, of 𝔅ʸ
        # (−EMF_z) and 𝔅ᶻ (+EMF_y) by the EMF, the edge states varying along x
        # only. z-edge: (d, a, b) = (3, 1, 2), varying along a; y-edge:
        # (2, 3, 1), varying along b.
        F𝔈 = SVector(riemann_flux(Val(:hlle), m, L, R, Val(1))[PW.U_E .+ (0:2)]...)
        emf_z = uct_hll_emf(m, (E_L, E_L, E_R, E_R), (Bx, Bx), (B_L[2], B_R[2]), Val(3))
        emf_y = uct_hll_emf(m, (E_L, E_R, E_L, E_R), (B_L[3], B_R[3]), (Bx, Bx), Val(2))
        # The physical fluxes of the same variables, and the HLL middle state.
        flux(E, B) = SVector(physical_flux(m, prim2con(eos, m, prim, E, B), prim, E, B,
                                           Val(1))[PW.U_E .+ (0:2)]...,
                             -faraday_emf(m, E, B)[3], faraday_emf(m, E, B)[2])
        U(E, B) = sg * SVector(E..., B[2], B[3])
        U★ = (s_R * U(E_R, B_R) - s_L * U(E_L, B_L) - (flux(E_R, B_R) - flux(E_L, B_L))) /
             (s_R - s_L)
        E★ = SVector(U★[1], U★[2], U★[3]) / sg
        B★ = SVector(Bx, U★[4] / sg, U★[5] / sg)
        numerical = SVector(F𝔈..., -emf_z, emf_y)
        scale = maximum(abs.(flux(E_L, B_L))) + maximum(abs.(flux(E_R, B_R)))
        worst = max(worst, maximum(abs.(numerical - flux(E★, B★))) / scale)
        # And the middle state keeps `Eˣ`: the longitudinal field has no flux.
        @test abs(E★[1] - Ex) <= 1e-13 * maximum(abs.(E★))
    end
    @info "HLL/UCT flux against the physical flux of the HLL middle state" worst straddled
    @test straddled > 200
    @test worst < 1e-13

    # With a superluminal shift both solvers take the upwind state alone.
    m = Metric3(1.0, SVector(-1.5, 0.0, 0.0), SMatrix{3,3}(1.0, 0, 0, 0, 1, 0, 0, 0, 1))
    prim = rec_prim(rng, eos, m; wmax=0.5)
    E_L, B_L, E_R, B_R = (SVector{3}(randn(rng, 3)) for _ in 1:4)
    L = (prim2con(eos, m, prim, E_L, B_L), prim, E_L, B_L)
    R = (prim2con(eos, m, prim, E_R, B_R), prim, E_R, B_R)
    @test riemann_flux(Val(:hlle), m, L, R, Val(1)) == physical_flux(m, L..., Val(1))
    @test uct_hll_emf(m, (E_L, E_L, E_R, E_R), (0.3, 0.3), (B_L[2], B_R[2]), Val(3)) ≈
          faraday_emf(m, E_L, SVector(0.3, B_L[2], B_L[3]))[3]
end

@testset "In flat space ∂ₜE = ∇×B − j, ∂ₜA = −E − ∇Φ, and the gauge is Lorenz" begin
    # Guards the signs of Maxwell's equations as the scheme evaluates them,
    # against an exact solution: a plane wave `A = a cos θ`, `Φ = φ₀ cos θ`,
    # `θ = k·x − ωt`, `ω = |k|`, in the Lorenz gauge `φ₀ = k·a/ω`. In a
    # constant metric the fluxes are linear in the fields, so their
    # divergence is the flux of the fields' gradient, and the right-hand
    # sides are evaluated exactly as the scheme will:
    #     ∂ₜ𝔈 = −∂ⱼ F^j(𝔈) + Ampère's source,
    #     ∂ₜA = −EMF − ∇(αΦ − βʲA_j),
    #     ∂ₜΦ̃ = −∂ⱼ(gauge flux) − ξαΦ̃.
    # Then the current: the source must be `−j`, with `j` from Ohm's law.
    rng = Xoshiro(36)
    m = flat_metric()
    vac = pw_vacuum()
    z10 = ntuple(_ -> 0.0, NU)
    worst_E = worst_A = worst_Φ = worst_j = 0.0
    for _ in 1:50
        k = SVector{3}(randn(rng, 3))
        ω = sqrt(sum(abs2, k))
        a = SVector{3}(randn(rng, 3))
        φ₀ = sum(k .* a) / ω
        x = SVector{3}(randn(rng, 3))
        t = randn(rng)
        θ = sum(k .* x) - ω * t
        e = φ₀ * k - ω * a                     # E = e sin θ
        b = -PW.symbol_cross(k, a)             # B = b sin θ
        E, B = e * sin(θ), b * sin(θ)
        # ∂ⱼ of the fields, by direction.
        dE = ntuple(j -> e * k[j] * cos(θ), 3)
        dB = ntuple(j -> b * k[j] * cos(θ), 3)
        dA = ntuple(j -> -a * k[j] * sin(θ), 3)
        dΦ = ntuple(j -> -φ₀ * k[j] * sin(θ), 3)

        div_F = sum(SVector(physical_flux(m, z10, vac, dE[j], dB[j], Val(j))[PW.U_E .+ (0:2)]...)
                    for j in 1:3)
        rhs_E = -div_F + ampere_source(Vacuum(), m, vac, E, B, 0.0)
        ∂ₜE = -ω * e * cos(θ)
        curl_B = PW.symbol_cross(k, b) * cos(θ)
        worst_E = max(worst_E, maximum(abs.(rhs_E - ∂ₜE)) / maximum(abs.(e)) / ω,
                      maximum(abs.(rhs_E - curl_B)) / maximum(abs.(e)) / ω)

        A, Φ = a * cos(θ), φ₀ * cos(θ)
        grad = SVector(ntuple(j -> gauge_potential(m, dΦ[j], dA[j]), 3))
        rhs_A = -faraday_emf(m, E, B) - grad
        ∂ₜA = ω * a * sin(θ)
        worst_A = max(worst_A, maximum(abs.(rhs_A - ∂ₜA)) / maximum(abs.(a)) / ω)
        @test gauge_potential(m, Φ, A) == Φ

        rhs_Φ = -sum(gauge_flux(m, dA[j], dΦ[j], Val(j)) for j in 1:3) +
                gauge_damping(0.0, m, Φ)
        ∂ₜΦ = φ₀ * ω * sin(θ)
        worst_Φ = max(worst_Φ, abs(rhs_Φ - ∂ₜΦ) / maximum(abs.(a)) / ω)

        # A current: the source is `−j`, `j = q v + σ W [E + v×B − (v·E) v]`.
        v = 0.8 * rec_direction(rng, m)
        fluid = PW.pack_prim(1.0, v, 5.0, 0.3, atanh(sqrt(sum(abs2, v))), 0.1, 1.5, 0.2, NaN)
        q = randn(rng)
        σ = 10 * rand(rng)
        W = 1 / sqrt(1 - sum(abs2, v))
        j = q * v + σ * W * (E + PW.symbol_cross(v, B) - sum(v .* E) * v)
        src = ampere_source(Constant(σ), m, fluid, E, B, q)
        worst_j = max(worst_j, maximum(abs.(src + j)) / maximum(abs.(j)))
    end
    @info "Flat-space Maxwell against the plane wave, and Ampère's source against −j" worst_E worst_A worst_Φ worst_j
    @test worst_E < 1e-14
    @test worst_A < 1e-14
    @test worst_Φ < 1e-14
    @test worst_j < 1e-14
    # The gauge damping damps.
    @test gauge_damping(2.0, Metric3(0.5, zero(SVector{3}), one(SMatrix{3,3,Float64})), 3.0) == -3.0
end

@testset "Faraday's shift terms are IllinoisGRMHD's, with the coordinate velocity" begin
    # Guards the sign of the shift term in the EMF, which CODE.md derives from
    # Komissarov's `E_aux = αD + β × B` and checks against IllinoisGRMHD's
    # ideal `∂ₜA_i = ε_ijk vʲBᵏ − ∂ᵢ(αΦ − βʲA_j)`, whose `vⁱ = uⁱ/u⁰ =
    # α v_Eⁱ − βⁱ` is the coordinate velocity. With `E = −v_E × B` the two
    # must agree in any metric; the gradient term is the same scalar in both
    # (`gauge_potential`). On random states and Kerr-Schild metrics.
    eos = rec_gas(2)
    rng = Xoshiro(37)
    worst = worst_u = 0.0
    for _ in 1:300
        m4, p = rec_event(rng)
        m = Metric3(SpacetimeMetrics.adm_decompose(m4, p)...)
        prim = rec_prim(rng, eos, m)
        v_E = PW.prim_v(prim)
        B = SVector{3}(randn(rng, 3))
        E = ideal_E(m, v_E, B)
        v_coord = m.α * v_E - m.β
        ours = -faraday_emf(m, E, B)
        igm = PW.cross_down(m, v_coord, B)                  # ε_ijk vʲ Bᵏ
        # Relative to the terms, `α E_i` and `ε_ijk βʲ Bᵏ`, which cancel
        # where the coordinate velocity is small.
        scale = maximum(abs.(m.α * PW.lower(m, E))) + maximum(abs.(PW.cross_down(m, m.β, B)))
        worst = max(worst, maximum(abs.(ours - igm)) / scale)
        # And `uⁱ/u⁰` is that velocity: `u = W (n + v_E)` in the four-metric.
        n = pw_n(m)
        W = PW.prim_W(prim)
        u = W * (n + SVector(0, v_E[1], v_E[2], v_E[3]))
        worst_u = max(worst_u, maximum(abs.(SVector(u[2], u[3], u[4]) / u[1] - v_coord)))
        # The 4-velocity is normalized, so `W` and `v_E` are the normal frame's.
        @test isapprox(u' * pw_g4(m) * u, -1; atol=1e-12 * W^2)
    end
    @info "Faraday's EMF against IllinoisGRMHD's ideal form" worst worst_u
    @test worst < 1e-13
    @test worst_u < 1e-14
end

@testset "Ohm's law: no current for E = −v×B, and the fluid-frame field otherwise" begin
    # Guards the conduction current's bracket and the conductivity models: a
    # sign error in `v × B` or the `(v·E) v` term would leave a current in an
    # ideal state (the acceptance, `j = 0` for `E = −v × B` and `q = 0`), and
    # in flat space the bracket must be the fluid frame's electric field
    # `eᵃ = F^ab u_b` projected as `j = σ e + ρ′ u` with the charge `q`
    # measured by the normal observer (Palenzuela et al. 2009).
    eos = rec_gas(2)
    rng = Xoshiro(38)
    models = (Constant(1e6), DensityProfile(1e6, 1e-12))
    worst_ideal = 0.0
    for _ in 1:300
        m = rec_metric(rng)
        prim = rec_prim(rng, eos, m)
        B = SVector{3}(randn(rng, 3))
        E = ideal_E(m, PW.prim_v(prim), B)
        for model in models
            j = ohm_current(model, m, prim, E, B, 0.0)
            σ = conductivity(model, prim)
            worst_ideal = max(worst_ideal,
                              sqrt(PW.dot_up(m, j, j)) /
                              (σ * PW.prim_W(prim) * sqrt(PW.dot_up(m, B, B))))
        end
        @test ohm_current(Vacuum(), m, prim, E, B, 0.0) == zero(SVector{3})
        q = randn(rng)
        @test ohm_current(Vacuum(), m, prim, E + B, B, q) == q * PW.prim_v(prim)
        @test ampere_explicit_source(m, prim, q) + ampere_stiff_source(Constant(3.0), m, prim, E + B, B) ≈
              ampere_source(Constant(3.0), m, prim, E + B, B, q)
    end
    @info "Ohm's law at E = −v×B: |j| / (σ W |B|)" worst_ideal
    @test worst_ideal < 1e-14

    # The fluid-frame field, in flat space, from the field tensor.
    η = SMatrix{4,4,Float64}(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    m = flat_metric()
    worst = 0.0
    for _ in 1:50
        E = SVector{3}(randn(rng, 3))
        B = SVector{3}(randn(rng, 3))
        v = 0.9 * rand(rng) * rec_direction(rng, m)
        W = 1 / sqrt(1 - sum(abs2, v))
        prim = PW.pack_prim(1.0, v, 5.0, 0.3, atanh(sqrt(sum(abs2, v))), 0.1, 1.5, 0.2, NaN)
        q, σ = randn(rng), 5 * rand(rng)
        Fu = SMatrix{4,4,Float64}((a == b ? 0.0 :
                                   a == 1 ? E[b - 1] :
                                   b == 1 ? -E[a - 1] :
                                   sum(pw_levi(a - 1, b - 1, k) * B[k] for k in 1:3))
                                  for a in 1:4, b in 1:4)          # F^ab
        u = W * SVector(1, v[1], v[2], v[3])
        e = Fu * (η * u)                                             # eᵃ = F^ab u_b
        # The comoving charge ρ′ such that the normal observer sees `q = J^t`.
        ρ′ = (q - σ * e[1]) / u[1]
        J = σ * e + ρ′ * u
        j = ohm_current(Constant(σ), m, prim, E, B, q)
        worst = max(worst, maximum(abs.(j - SVector(J[2], J[3], J[4]))) / maximum(abs.(J)))
    end
    @info "Ohm's law against the fluid-frame field, flat space" worst
    @test worst < 1e-14

    # The models.
    p(ρ) = PW.pack_prim(ρ, zero(SVector{3}), 5.0, 0.3, 0.0, 0.1, 1.5, 0.2, NaN)
    @test conductivity(Vacuum(), p(1.0)) === 0.0
    @test conductivity(Constant(7), p(1.0)) === 7.0
    prof = DensityProfile(10.0, 1e-10)
    @test conductivity(prof, p(1e-10)) == 0 && conductivity(prof, p(2e-10)) == 0
    @test conductivity(prof, p(4e-10)) ≈ 5 && conductivity(prof, p(1.0)) ≈ 10
    @test isnan(conductivity(prof, p(NaN)))
    @test conductivity(prof, PW.pack_prim(4f-10, zero(SVector{3,Float32}), 5f0, 0.3f0, 0f0,
                                          0.1f0, 1.5f0, 0.2f0, NaN32)) isa Float32
    @test conductivity(IdealConductor(), p(1.0)) == Inf
    @test_throws ArgumentError ohm_current(IdealConductor(), m, p(1.0), zero(SVector{3}),
                                           zero(SVector{3}), 0.0)
    @test_throws ArgumentError ampere_stiff_source(IdealConductor(), m, p(1.0),
                                                   zero(SVector{3}), zero(SVector{3}))
    # Where σ = 0 the conduction term is not formed: a NaN field stays out.
    nanE = SVector(NaN, 0.0, 0.0)
    @test ohm_current(prof, m, p(1e-10), nanE, zero(SVector{3}), 0.0) == zero(SVector{3})
    @test ampere_stiff_source(prof, m, p(1e-10), nanE, zero(SVector{3})) == zero(SVector{3})
end

@testset "The two momentum-source forms agree, and the energy source is Banyuls'" begin
    # Guards the geometric sources, each against an independent form, on
    # boosted Kerr-Schild at random events (time-dependent, so `K_ij` and
    # `∂ₜ` terms do not vanish): the default `½√−g T^ab ∂ⱼg_ab` against the
    # Valencia `√γ[½αS^ik∂ⱼγ_ik + S_i∂ⱼβⁱ − 𝓔∂ⱼα]` (the acceptance) and
    # against `√−g T^μν(∂_μ g_νj − Γ^δ_νμ g_δj)` with SpacetimeMetrics'
    # Christoffels; and the energy source `√γ[αS^ijK_ij − Sʲ∂ⱼα]` against
    # Banyuls et al. (1997)'s `√−g (T^μt ∂_μα − α T^μν Γᵗ_μν)`, with `∂_μα`
    # by ForwardDiff. Each is measured against the sum of the magnitudes of
    # its terms, which is what rounding is relative to.
    eos = rec_gas(2)
    rng = Xoshiro(39)
    worst_val = worst_chr = worst_τ = 0.0
    for _ in 1:300
        m4, p = rec_event(rng)
        prov = Analytic(m4)
        t, x = p[1], SVector(p[2], p[3], p[4])
        m = point_metric(prov, t, x)
        dm = metric_gradient(prov, t, x)
        prim, E, B, _ = pw_state(rng, eos, m)
        𝓔, S, Sij = stress_energy(m, prim, E, B)
        sc, τc = geometric_source(Val(:covariant), m, dm, 𝓔, S, Sij)
        sv, τv = geometric_source(Val(:valencia), m, dm, 𝓔, S, Sij)
        @test τc == τv
        Tu = PW.T4_upper(m, 𝓔, S, Sij)
        α, sg = m.α, m.sqrtγ
        dα = PW.d_lapse(m, dm)
        dβ = PW.d_shift(m, dm)
        Γ = SpacetimeMetrics.ChristoffelSymbols(m4, p)
        g, sdg = SpacetimeMetrics.dmetric(m4, p)
        for j in 1:3
            mag_c = α * sg / 2 * sum(abs(Tu[a, b] * dm.dg[j, a, b]) for a in 1:4, b in 1:4)
            mag_v = sg * (α / 2 * sum(abs.(Sij .* PW.d_spatial(dm, j))) +
                          sum(abs.(S .* dβ[j, :])) + abs(𝓔 * dα[j]))
            worst_val = max(worst_val, abs(sc[j] - sv[j]) / max(mag_c, mag_v))
            terms = [Tu[μ, ν] * (sdg[ν, j + 1, μ] - sum(Γ[δ, ν, μ] * g[δ, j + 1] for δ in 1:4))
                     for μ in 1:4, ν in 1:4]
            worst_chr = max(worst_chr, abs(sc[j] - α * sg * sum(terms)) / (α * sg * sum(abs.(terms))))
        end
        lapse(q) = SpacetimeMetrics.adm_decompose(m4, q)[1]
        ∂α = ForwardDiff.gradient(lapse, p)                      # (∂ₜα, ∂ᵢα)
        terms = [Tu[μ, 1] * ∂α[μ] for μ in 1:4]
        terms2 = [-α * Tu[μ, ν] * Γ[1, μ, ν] for μ in 1:4, ν in 1:4]
        banyuls = α * sg * (sum(terms) + sum(terms2))
        worst_τ = max(worst_τ, abs(τc - banyuls) / (α * sg * (sum(abs.(terms)) + sum(abs.(terms2)))))
    end
    @info "Geometric sources: covariant vs Valencia, vs Christoffel form; energy vs Banyuls" worst_val worst_chr worst_τ
    @test worst_val < 1e-14
    @test worst_chr < 1e-13
    @test worst_τ < 1e-13

    # Flat space has no geometric source, exactly; and the explicit source
    # of the ten variables is the geometric one plus Ampère's.
    m = flat_metric()
    prim, E, B, _ = pw_state(rng, eos, m)
    src = explicit_source(m, metric_gradient(Flat(), 0.0, zero(SVector{3})), prim, E, B, 0.0)
    @test all(iszero, src)
    m4, p = rec_event(rng)
    prov = Analytic(m4)
    m = point_metric(prov, p[1], SVector(p[2], p[3], p[4]))
    dm = metric_gradient(prov, p[1], SVector(p[2], p[3], p[4]))
    prim, E, B, _ = pw_state(rng, eos, m)
    q = 0.1
    for form in (Val(:covariant), Val(:valencia))
        src = explicit_source(m, dm, prim, E, B, q, form)
        s_S, s_τ = geometric_source(form, m, dm, stress_energy(m, prim, E, B)...)
        @test src[PW.U_D] == src[PW.U_DYE] == src[PW.U_DS] == 0
        @test PW.cons_S(src) == s_S && PW.cons_τ(src) == s_τ
        @test PW.cons_𝔈(src) == ampere_explicit_source(m, prim, q)
    end
    @test explicit_source(m, dm, prim, E, B, q) == explicit_source(m, dm, prim, E, B, q,
                                                                   Val(:covariant))
end

# Everything a right-hand-side kernel calls, at one point: the provider, the
# stress-energy, both Riemann solvers in all three directions, Ohm's law, both
# source forms, the EMF and the gauge terms. Returned as one tuple of numbers.
@inline function pw_pointwise(prov, t, x, eos, prim, E, B, prim2, E2, B2, q)
    m = point_metric(prov, t, x)
    dm = metric_gradient(prov, t, x)
    T = typeof(t)
    cons = prim2con(eos, m, prim, E, B)
    cons2 = prim2con(eos, m, prim2, E2, B2)
    L, R = (cons, prim, E, B), (cons2, prim2, E2, B2)
    𝓔, S, Sij = stress_energy(m, prim, E, B)
    Tl = T4_lower(m, 𝓔, S, Sij)
    f1 = riemann_flux(Val(:hlle), m, L, R, Val(1))
    f2 = riemann_flux(Val(:llf), m, L, R, Val(2))
    f3 = riemann_flux(Val(:hlle), m, R, L, Val(3))
    j = ohm_current(DensityProfile(T(100), T(1e-9)), m, prim, E, B, q)
    sc = explicit_source(m, dm, prim, E, B, q)
    sv = explicit_source(m, dm, prim, E, B, q, Val(:valencia))
    st = ampere_stiff_source(Constant(T(10)), m, prim, E, B)
    emf = uct_hll_emf(m, (E, E2, E2, E), (B[1], B2[1]), (B[2], B2[2]), Val(3))
    A = E + B
    gauge = (gauge_flux(m, A, q, Val(2)), gauge_potential(m, q, A), gauge_damping(T(1), m, q))
    return (Tl..., f1..., f2..., f3..., j..., sc..., sv..., st..., emf, faraday_emf(m, E, B)...,
            gauge...)
end

@kernel function pw_kernel!(out, prov, @Const(ts), @Const(xs), eos, @Const(prims), @Const(Es),
                            @Const(Bs), @Const(qs))
    i = @index(Global)
    n = length(prims)
    i2 = mod1(i + 1, n)
    out[i] = pw_pointwise(prov, ts[i], xs[i], eos, prims[i], Es[i], Bs[i], prims[i2], Es[i2],
                          Bs[i2], qs[i])
end

@testset "The pointwise physics runs inside a CPU kernel, bit for bit: T=$T" for T in
                                                                                 (Float64, Float32)
    # Guards the kernel-callability every right-hand side relies on: the
    # provider (a SpacetimeMetrics metric with its ForwardDiff derivatives),
    # the conductivity model and the EOS are `isbits` arguments, nothing
    # throws, allocates or fails to infer, and the kernel computes exactly
    # what the host does. `Float32` is the leak detector: a `Float64`
    # constant anywhere would widen the result type.
    eos = REC_EOS.narrow(rec_gas(2), T)
    prov = Analytic(SpacetimeMetrics.KerrSchild{T}(1, T(1 // 2)))
    rng = Xoshiro(40)
    n = 32
    ts = T[]
    xs = SVector{3,T}[]
    prims = NTuple{NPRIM,T}[]
    Es = SVector{3,T}[]
    Bs = SVector{3,T}[]
    for _ in 1:n
        x = SVector{3}(randn(rng, 3))
        x = T.(((3 + 10 * rand(rng)) / sqrt(sum(abs2, x))) * x)
        t = T(randn(rng))
        m = point_metric(prov, t, x)
        v = T.(0.5 * rand(rng) * rec_direction(rng, m))
        push!(ts, t)
        push!(xs, x)
        push!(prims, primitive_state(eos, m, T(exp10(-4 + 2 * rand(rng))), v, T(5), T(0.3)))
        push!(Es, T.(1e-3 * SVector{3}(randn(rng, 3))))
        push!(Bs, T.(1e-3 * SVector{3}(randn(rng, 3))))
    end
    qs = T.(1e-4 * randn(rng, n))
    host = [pw_pointwise(prov, ts[i], xs[i], eos, prims[i], Es[i], Bs[i], prims[mod1(i + 1, n)],
                         Es[mod1(i + 1, n)], Bs[mod1(i + 1, n)], qs[i]) for i in 1:n]
    @test eltype(host) <: NTuple{N,T} where {N}
    @test all(h -> all(isfinite, h), host)
    out = similar(host)
    backend = CPU()
    pw_kernel!(backend)(out, prov, ts, xs, eos, prims, Es, Bs, qs; ndrange=n)
    synchronize(backend)
    @test isequal(out, host)

    # Allocation-free and inferred, on the host, call by call.
    t, x, prim, E, B, q = ts[1], xs[1], prims[1], Es[1], Bs[1], qs[1]
    m = point_metric(prov, t, x)
    dm = metric_gradient(prov, t, x)
    cons = prim2con(eos, m, prim, E, B)
    s = (cons, prim, E, B)
    𝓔, S, Sij = stress_energy(m, prim, E, B)
    calls = ((point_metric, (prov, t, x)), (metric_gradient, (prov, t, x)),
             (stress_energy, (m, prim, E, B)), (T4_lower, (m, 𝓔, S, Sij)),
             (physical_flux, (m, cons, prim, E, B, Val(2))),
             (riemann_flux, (Val(:llf), m, s, s, Val(1))),
             (riemann_flux, (Val(:hlle), m, s, s, Val(3))),
             (ohm_current, (Constant(T(2)), m, prim, E, B, q)),
             (geometric_source, (Val(:covariant), m, dm, 𝓔, S, Sij)),
             (geometric_source, (Val(:valencia), m, dm, 𝓔, S, Sij)),
             (explicit_source, (m, dm, prim, E, B, q)),
             (ampere_source, (DensityProfile(T(1), T(1e-9)), m, prim, E, B, q)),
             (uct_hll_emf, (m, (E, E, B, B), (q, q), (q, -q), Val(1))),
             (faraday_emf, (m, E, B)), (gauge_flux, (m, E, q, Val(3))),
             (pw_pointwise, (prov, t, x, eos, prim, E, B, prim, E, B, q)))
    for (f, args) in calls
        r = @inferred f(args...)
        @test all(v -> v isa Union{T,SVector{3,T},SMatrix{3,3,T,9},SMatrix{4,4,T,16},Metric3{T},
                                   MetricGradient{T},Tuple}, (r isa Tuple ? r : (r,)))
        @test rec_allocations(f, args...) == 0
    end
    @test pw_pointwise(prov, t, x, eos, prim, E, B, prim, E, B, q) isa NTuple{N,T} where {N}
end
