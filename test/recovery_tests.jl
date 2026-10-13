# The 3+1 primitive recovery and everything it stands on: the point metric,
# the variable layouts, `prim2con` with the electromagnetic parts, and the
# two calls of EntropyEOS's one recovery — the hydrodynamic mode every
# right-hand side makes, and the ideal mode the implicit stage solve seeds
# from. See "Primitive recovery" and "Conventions" in `CODE.md`.
#
# The states are drawn from a seeded generator; the assertions are not. Every
# claim follows from the ranges the states are drawn from — ρ across the
# EOS's box, s across its window, W ≤ 10, metrics from a boosted spinning
# Kerr-Schild hole at random points, fields carrying up to 100 times the
# fluid's energy — and would hold for any draw.
#
# The accuracy is stated as EntropyEOS states its own (EOS-2 in `PLAN.md`):
# in the primitives, the errors divided by the conditioning factors
# `κ_ρ = hW²` (for ρ, w and v) and `κ_s = (1 + τ/(ρW²U))·max(1, U/(s U_s))`
# (for s), below upstream's 3e-11 and 1e-10; in the conserved variables, 1e-12
# relative. The latter is the recovery's own tolerance: `con2prim` stops when
# its energy residual is below 1e-12 of `τ`, so the conserved round trip
# cannot be tighter than that, and is not measured against anything looser.

using KernelAbstractions: @kernel, @index, CPU, synchronize
using Random: Xoshiro
using StaticArrays: SVector, SMatrix
import SpacetimeMetrics

const REC_EOS = TreeGRRMHD.EntropyEOS
const REC_BOX = (; s_ref=5.0, s_window=(4.0, 6.0), ρ_bounds=(1e-10, 1e-2),
                 yₑ_bounds=(0.05, 0.6))
const REC_ρ_ATM = 1e-10
# Γ = 4/3, 5/3 and 2: EOS-2's downstream check. Two is the TOV polytrope's.
const REC_GAMMAS = (4 // 3, 5 // 3, 2)

rec_gas(Γ, ::Type{T}=Float64) where {T} =
    REC_EOS.IdealGasEOS{T}(; Γ=T(Γ), K_ref=T(100), REC_BOX...)

# A boosted spinning Kerr-Schild hole, sampled at a random event between
# r = 3M and 20M: lapse, shift and a non-diagonal 3-metric all far from
# flat, with `√γ` between about 1 and 2.
function rec_metric(rng)
    hole = SpacetimeMetrics.KerrSchild{Float64}(1.0, 0.9 * rand(rng))
    vb = SVector{3}(randn(rng, 3))
    m4 = SpacetimeMetrics.boost(hole, (0.6 * rand(rng) / sqrt(sum(abs2, vb))) * vb)
    while true
        x = SVector{3}(randn(rng, 3))
        x = ((3 + 17 * rand(rng)) / sqrt(sum(abs2, x))) * x
        α, β, γ = SpacetimeMetrics.adm_decompose(m4, SVector(10 * randn(rng), x...))
        α > 0 && return Metric3(α, β, γ)
    end
end

# A random direction of unit length in the metric `m`.
function rec_direction(rng, m)
    d = SVector{3}(randn(rng, 3))
    return d / sqrt(TreeGRRMHD.dot_up(m, d, d))
end

# A random fluid state, as `primitive_state` builds it.
function rec_prim(rng, eos, m; wmax=acosh(10.0), swindow=(4.05, 5.95), log10ρ=(-9.0, -2.1))
    ρ = exp10(log10ρ[1] + (log10ρ[2] - log10ρ[1]) * rand(rng))
    s = swindow[1] + (swindow[2] - swindow[1]) * rand(rng)
    yₑ = 0.06 + 0.53 * rand(rng)
    w = wmax * rand(rng)
    v = tanh(w) * rec_direction(rng, m)
    return primitive_state(eos, m, ρ, v, s, yₑ)
end

# Two primitive states are the same state: equal slot by slot (so that
# `-0.0` and `0.0` are one velocity), the EOS warm start equal or both NaN
# (an analytic EOS has none), and the velocity to `vtol`.
function rec_same(p, q; vtol=0.0)
    vp, vq = TreeGRRMHD.prim_v(p), TreeGRRMHD.prim_v(q)
    others = [i for i in 1:(NPRIM - 1) if !(i in TreeGRRMHD.P_V .+ (0:2))]
    return all(i -> p[i] == q[i], others) && maximum(abs.(vp - vq)) <= vtol &&
           isequal(TreeGRRMHD.prim_ueos(p), TreeGRRMHD.prim_ueos(q))
end

# The conditioning factors of EntropyEOS's own round-trip test, at the true
# state, with the `τ` the recovery sees.
function rec_conditioning(eos, prim, τ)
    ρ, s, w = TreeGRRMHD.prim_ρ(prim), TreeGRRMHD.prim_s(prim), TreeGRRMHD.prim_w(prim)
    pt = REC_EOS.evaluate(eos, ρ, s, TreeGRRMHD.prim_Yₑ(prim), NaN)
    W² = cosh(w)^2
    κ_ρ = pt.h * W²
    κ_s = (1 + τ / (ρ * W² * pt.U)) * max(1, pt.U / (s * pt.U_s))
    return κ_ρ, κ_s
end

# The primitive errors, conditioned: (ρ, w, v) by κ_ρ and s by κ_s.
function rec_prim_errors(m, p, q, κ_ρ, κ_s)
    dv = TreeGRRMHD.prim_v(q) - TreeGRRMHD.prim_v(p)
    eρ = max(abs(TreeGRRMHD.prim_ρ(q) - TreeGRRMHD.prim_ρ(p)) / TreeGRRMHD.prim_ρ(p),
             abs(TreeGRRMHD.prim_w(q) - TreeGRRMHD.prim_w(p)) / max(TreeGRRMHD.prim_w(p), 1),
             sqrt(TreeGRRMHD.dot_up(m, dv, dv)))
    es = abs(TreeGRRMHD.prim_s(q) - TreeGRRMHD.prim_s(p)) / TreeGRRMHD.prim_s(p)
    return eρ / κ_ρ, es / κ_s, es
end

# The conserved errors. `D`, `D Yₑ` relative to themselves; `S_i` and `τ`,
# the energy-momentum components, relative to the energy density
# `𝓔 = τ + D` they are components of (|S| ≤ 𝓔); `𝔈` relative to itself.
# Also the per-component `τ` error, which is reported.
function rec_cons_errors(m, c, d)
    sg = m.sqrtγ
    𝓔 = (TreeGRRMHD.cons_τ(c) + TreeGRRMHD.cons_D(c)) / sg
    dS = (TreeGRRMHD.cons_S(d) - TreeGRRMHD.cons_S(c)) / sg
    e𝔈 = TreeGRRMHD.cons_𝔈(d) - TreeGRRMHD.cons_𝔈(c)
    n𝔈 = sqrt(sum(abs2, TreeGRRMHD.cons_𝔈(c)))
    e = max(abs(TreeGRRMHD.cons_D(d) - TreeGRRMHD.cons_D(c)) / TreeGRRMHD.cons_D(c),
            abs(TreeGRRMHD.cons_DYₑ(d) - TreeGRRMHD.cons_DYₑ(c)) / TreeGRRMHD.cons_DYₑ(c),
            sqrt(TreeGRRMHD.dot_down(m, dS, dS)) / 𝓔,
            abs(TreeGRRMHD.cons_τ(d) - TreeGRRMHD.cons_τ(c)) / sg / 𝓔,
            n𝔈 > 0 ? sqrt(sum(abs2, e𝔈)) / n𝔈 : 0.0)
    eτ = abs(TreeGRRMHD.cons_τ(d) - TreeGRRMHD.cons_τ(c)) / abs(TreeGRRMHD.cons_τ(c))
    return e, eτ
end

@testset "The point metric inverts, densitizes and crosses consistently" begin
    # Guards the index operations every pointwise function stands on: an
    # inverse that is not one, a `√γ` that is not `√det γ`, and a cross
    # product written with the symbol where the tensor is meant — which is
    # invisible in flat space and wrong everywhere else.
    rng = Xoshiro(21)
    worst_inv = worst_det = worst_cross = 0.0
    for _ in 1:200
        m = rec_metric(rng)
        @test isbits(m)
        I3 = m.γ * m.γu
        worst_inv = max(worst_inv, maximum(abs.(I3 - one(SMatrix{3,3,Float64}))))
        γ = m.γ
        detγ = γ[1, 1] * (γ[2, 2] * γ[3, 3] - γ[2, 3]^2) -
               γ[1, 2] * (γ[1, 2] * γ[3, 3] - γ[2, 3] * γ[1, 3]) +
               γ[1, 3] * (γ[1, 2] * γ[2, 3] - γ[2, 2] * γ[1, 3])
        worst_det = max(worst_det, abs(m.sqrtγ^2 - detγ) / detγ)
        @test m.γu == m.γu'                          # exactly symmetric

        # The ideal field is orthogonal to v and to B, and `ε_ijk vʲ Bᵏ`
        # is antisymmetric: E·v = E·B = 0 in the metric, and swapping
        # v and B flips the sign.
        v = 0.7 * rec_direction(rng, m)
        B = SVector{3}(randn(rng, 3))
        E = ideal_E(m, v, B)
        scale = sqrt(TreeGRRMHD.dot_up(m, E, E) * TreeGRRMHD.dot_up(m, B, B))
        worst_cross = max(worst_cross, abs(TreeGRRMHD.dot_up(m, E, v)) / scale,
                          abs(TreeGRRMHD.dot_up(m, E, B)) / scale)
        @test ideal_E(m, B, v) ≈ -E
        # |v × B|² = v²B² − (v·B)², with the tensor ε and not the symbol.
        vB = TreeGRRMHD.dot_up(m, v, B)
        @test TreeGRRMHD.dot_up(m, E, E) ≈
              TreeGRRMHD.dot_up(m, v, v) * TreeGRRMHD.dot_up(m, B, B) - vB^2
        # `cross_up` of the lowered vectors is the raised `cross_down`.
        @test TreeGRRMHD.cross_up(m, TreeGRRMHD.lower(m, v), TreeGRRMHD.lower(m, B)) ≈
              TreeGRRMHD.raise(m, TreeGRRMHD.cross_down(m, v, B))
    end
    @test worst_inv < 1e-14
    @test worst_det < 1e-14
    @test worst_cross < 1e-14

    m = flat_metric(Float32)
    @test m.α === 1.0f0 && m.sqrtγ === 1.0f0
    @test m.γ == m.γu
    @test Metric3(1, [0, 0, 0], [1 0 0; 0 1 0; 0 0 1]) isa Metric3{Float64}
end

@testset "The variable layouts are a partition read back exactly" begin
    # Guards two slots that overlap — a vector of three written over the
    # scalar after it — and an accessor reading a neighbour's slot. Every
    # slot of `U` and of `P` is written with a distinct value and read back
    # through its accessor.
    U_slots = [TreeGRRMHD.U_D; TreeGRRMHD.U_S .+ (0:2); TreeGRRMHD.U_TAU; TreeGRRMHD.U_DYE;
               TreeGRRMHD.U_DS; TreeGRRMHD.U_E .+ (0:2)]
    @test sort(U_slots) == 1:NU
    P_slots = [TreeGRRMHD.P_RHO; TreeGRRMHD.P_V .+ (0:2); TreeGRRMHD.P_S; TreeGRRMHD.P_YE;
               TreeGRRMHD.P_W; TreeGRRMHD.P_PRESS; TreeGRRMHD.P_H; TreeGRRMHD.P_CS2;
               TreeGRRMHD.P_UEOS; TreeGRRMHD.P_FLAGS; TreeGRRMHD.P_E .+ (0:2);
               TreeGRRMHD.P_B .+ (0:2); TreeGRRMHD.P_Q; TreeGRRMHD.P_ALPHA;
               TreeGRRMHD.P_BETA .+ (0:2); TreeGRRMHD.P_GAMMA .+ (0:5); TreeGRRMHD.P_SQRTG]
    @test sort(P_slots) == 1:NP
    @test maximum(P_slots[1:11]) == NPRIM        # the primitives lead

    u = TreeGRRMHD.pack_cons(1.0, SVector(2.0, 3.0, 4.0), 5.0, 6.0, 7.0,
                             SVector(8.0, 9.0, 10.0))
    @test u == ntuple(Float64, NU)
    @test TreeGRRMHD.cons_D(u) == 1 && TreeGRRMHD.cons_S(u) == SVector(2, 3, 4)
    @test TreeGRRMHD.cons_τ(u) == 5 && TreeGRRMHD.cons_DYₑ(u) == 6
    @test TreeGRRMHD.cons_Ds(u) == 7 && TreeGRRMHD.cons_𝔈(u) == SVector(8, 9, 10)

    prim = TreeGRRMHD.pack_prim(1.0, SVector(2.0, 3.0, 4.0), 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                                11.0)
    @test prim == ntuple(Float64, NPRIM)
    m = rec_metric(Xoshiro(22))
    flags = REC_EOS.FLAG_POL_ATMOSPHERE | REC_EOS.FLAG_POL_W_CAPPED
    P = TreeGRRMHD.pack_P(prim, flags, SVector(13.0, 14.0, 15.0), SVector(16.0, 17.0, 18.0),
                          19.0, m)
    @test P isa NTuple{NP,Float64}
    @test P[1:NPRIM] == prim
    @test TreeGRRMHD.prim_ρ(P) == 1 && TreeGRRMHD.prim_v(P) == SVector(2, 3, 4)
    @test TreeGRRMHD.prim_s(P) == 5 && TreeGRRMHD.prim_Yₑ(P) == 6
    @test TreeGRRMHD.prim_w(P) == 7 && TreeGRRMHD.prim_W(P) == cosh(7.0)
    @test TreeGRRMHD.prim_p(P) == 8 && TreeGRRMHD.prim_h(P) == 9
    @test TreeGRRMHD.prim_cs²(P) == 10 && TreeGRRMHD.prim_ueos(P) == 11
    @test guesses(P) == (5.0, 7.0, 11.0)
    @test TreeGRRMHD.aux_flags(P) === flags
    @test TreeGRRMHD.aux_E(P) == SVector(13, 14, 15)
    @test TreeGRRMHD.aux_B(P) == SVector(16, 17, 18)
    @test TreeGRRMHD.aux_q(P) == 19
    m2 = TreeGRRMHD.aux_metric(P)
    @test m2.α == m.α && m2.β == m.β && m2.γ == m.γ && m2.γu == m.γu && m2.sqrtγ == m.sqrtγ
    @test all(isnan, cold_guesses(Float32)) && cold_guesses(Float32) isa NTuple{3,Float32}
end

@testset "con2prim_hydro inverts prim2con on random states: Γ=$Γ" for Γ in REC_GAMMAS
    # Guards the 3+1 layer around the one recovery: a field energy or
    # momentum subtracted with the wrong sign or the wrong metric, a
    # densitization missed, a projection with `γ_ij` where `γ^ij` is meant,
    # and a velocity rebuilt in the wrong direction or with a magnitude that
    # does not belong to the returned rapidity. Each state is a random fluid
    # (W ≤ 10) in a random boosted Kerr-Schild metric, with random E and B
    # carrying between 1e-4 and 100 times the fluid's `τ`.
    eos = rec_gas(Γ)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(23)
    worst_ρ = worst_s = worst_s_raw = worst_cons = worst_τ = 0.0
    flagged = 0
    for _ in 1:1000
        m = rec_metric(rng)
        prim = rec_prim(rng, eos, m)
        zero3 = zero(SVector{3,Float64})
        τ_fluid = TreeGRRMHD.cons_τ(prim2con(eos, m, prim, zero3, zero3)) / m.sqrtγ
        ratio = exp10(-4 + 6 * rand(rng))
        share = rand(rng)
        E = sqrt(2 * ratio * share * τ_fluid) * rec_direction(rng, m)
        B = sqrt(2 * ratio * (1 - share) * τ_fluid) * rec_direction(rng, m)

        cons = prim2con(eos, m, prim, E, B)
        q, flags = con2prim_hydro(eos, opts, pol, m, cons, E, B, cold_guesses(Float64))
        flags == 0 || (flagged += 1)

        κ_ρ, κ_s = rec_conditioning(eos, prim, τ_fluid)
        eρ, es, es_raw = rec_prim_errors(m, prim, q, κ_ρ, κ_s)
        worst_ρ = max(worst_ρ, eρ)
        worst_s = max(worst_s, es)
        worst_s_raw = max(worst_s_raw, es_raw)
        # Yₑ is D Yₑ / D: exact but for two roundings.
        @test abs(TreeGRRMHD.prim_Yₑ(q) - TreeGRRMHD.prim_Yₑ(prim)) <=
              4eps() * TreeGRRMHD.prim_Yₑ(prim)
        # The velocity's magnitude is the rapidity's, to rounding.
        vq = TreeGRRMHD.prim_v(q)
        @test isapprox(sqrt(TreeGRRMHD.dot_up(m, vq, vq)), tanh(TreeGRRMHD.prim_w(q));
                       rtol=8eps())

        # Back to the conserved variables. `√γ D s` is not an input of the
        # hydrodynamic recovery — `s` comes from `τ`, and `D s` is reset from
        # it ("Matter" in `CODE.md`) — so its slot is checked through `s`
        # above, and not here.
        e, eτ = rec_cons_errors(m, cons, prim2con(eos, m, q, E, B))
        worst_cons = max(worst_cons, e)
        worst_τ = max(worst_τ, eτ)
    end
    @test flagged == 0
    @test worst_ρ < 3e-11
    @test worst_s < 1e-10
    @test worst_cons <= 1e-12
    @info "con2prim_hydro ∘ prim2con, Γ = $Γ: conditioned (ρ,w,v), s; raw s; conserved " *
          "(per energy density); τ per itself" worst_ρ worst_s worst_s_raw worst_cons worst_τ
end

@testset "con2prim_ideal recovers v when E = −v×B: Γ=$Γ" for Γ in REC_GAMMAS
    # Guards the seed of the implicit stage solve, and the exact recovery for
    # `IdealConductor`: the momentum split along and across `B` in the
    # metric, the field's energy left in `τ` for EntropyEOS's magnetized
    # mode, and the velocity reassembled from its two scalar components. A
    # state built with the ideal field must come back with its velocity,
    # whatever the magnetization (`B²` from 1e-4 to 100 times the fluid's
    # `τ`) and the angle between `v` and `B`.
    eos = rec_gas(Γ)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(24)
    worst_ρ = worst_s = worst_v = 0.0
    flagged = 0
    for k in 1:600
        m = rec_metric(rng)
        prim = rec_prim(rng, eos, m)
        zero3 = zero(SVector{3,Float64})
        τ_fluid = TreeGRRMHD.cons_τ(prim2con(eos, m, prim, zero3, zero3)) / m.sqrtγ
        # One state in ten has B along v and one across it: the two ends of
        # the split, where one of `S_par` and `S_perp` vanishes.
        v = TreeGRRMHD.prim_v(prim)
        dir = k % 10 == 0 ? v : k % 10 == 1 ?
              TreeGRRMHD.raise(m, TreeGRRMHD.cross_down(m, v, rec_direction(rng, m))) :
              rec_direction(rng, m)
        nd = sqrt(TreeGRRMHD.dot_up(m, dir, dir))
        nd > 0 || continue
        B = (sqrt(2 * exp10(-4 + 6 * rand(rng)) * τ_fluid) / nd) * dir
        E = ideal_E(m, v, B)
        cons = prim2con(eos, m, prim, E, B)
        q, flags = con2prim_ideal(eos, opts, pol, m, cons, B, cold_guesses(Float64))
        flags == 0 || (flagged += 1)
        κ_ρ, κ_s = rec_conditioning(eos, prim, TreeGRRMHD.cons_τ(cons) / m.sqrtγ)
        eρ, es, _ = rec_prim_errors(m, prim, q, κ_ρ, κ_s)
        worst_ρ = max(worst_ρ, eρ)
        worst_s = max(worst_s, es)
        dv = TreeGRRMHD.prim_v(q) - v
        worst_v = max(worst_v, sqrt(TreeGRRMHD.dot_up(m, dv, dv)))
        # And the ideal field of the recovered velocity is the field the
        # state was built with.
        Eq = ideal_E(m, TreeGRRMHD.prim_v(q), B)
        dE = Eq - E
        @test sqrt(TreeGRRMHD.dot_up(m, dE, dE)) <=
              1e-10 * sqrt(TreeGRRMHD.dot_up(m, B, B))
    end
    @test flagged == 0
    @test worst_ρ < 3e-11
    @test worst_s < 1e-10
    @info "con2prim_ideal ∘ prim2con, Γ = $Γ: conditioned (ρ,w,v), s; raw |Δv|" worst_ρ worst_s worst_v

    # Without a field it is the hydrodynamic recovery: the same call of
    # EntropyEOS, and the velocity assembled along a different path, so equal
    # but for the last bits of v.
    m = rec_metric(rng)
    prim = rec_prim(rng, eos, m)
    zero3 = zero(SVector{3,Float64})
    cons = prim2con(eos, m, prim, zero3, zero3)
    qi, fi = con2prim_ideal(eos, opts, pol, m, cons, zero3, cold_guesses(Float64))
    qh, fh = con2prim_hydro(eos, opts, pol, m, cons, zero3, zero3, cold_guesses(Float64))
    @test fi == fh == 0
    @test rec_same(qi, qh; vtol=4eps())
end

@testset "A warm start recovers the same state" begin
    # Guards the order of the positional guesses: `(s, w, u_eos)`, as
    # `guesses` returns them and as EntropyEOS's `con2prim_safe` takes them.
    # A transposed pair converges anyway — the recovery is robust to a bad
    # seed — so the claim is about the work: started at the answer, Newton
    # has nothing to do, and the state is the cold one to the tolerance.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    rng = Xoshiro(25)
    for _ in 1:50
        m = rec_metric(rng)
        prim = rec_prim(rng, eos, m)
        B = 1e-3 * rec_direction(rng, m)
        E = 2e-3 * rec_direction(rng, m)
        cons = prim2con(eos, m, prim, E, B)
        cold, _ = con2prim_hydro(eos, opts, pol, m, cons, E, B, cold_guesses(Float64))
        warm, flags = con2prim_hydro(eos, opts, pol, m, cons, E, B, guesses(prim))
        @test flags == 0
        κ_ρ, κ_s = rec_conditioning(eos, prim, TreeGRRMHD.cons_τ(cons) / m.sqrtγ)
        eρ, es, _ = rec_prim_errors(m, cold, warm, κ_ρ, κ_s)
        @test eρ < 3e-11 && es < 1e-10
        # Started at the answer, the recovery takes no Newton step.
        sg = m.sqrtγ
        S_f = TreeGRRMHD.cons_S(cons) / sg - TreeGRRMHD.cross_down(m, E, B)
        τ_f = TreeGRRMHD.cons_τ(cons) / sg -
              (TreeGRRMHD.dot_up(m, E, E) + TreeGRRMHD.dot_up(m, B, B)) / 2
        cin = REC_EOS.Con2PrimIn{Float64}(cons[1] / sg, τ_f, cons[6] / sg, 0.0,
                                          sqrt(TreeGRRMHD.dot_down(m, S_f, S_f)), 0.0)
        out = REC_EOS.con2prim_safe(eos, cin, opts, pol, guesses(prim)...)
        @test out.base.iters_newton <= 1
    end
end

@testset "The atmosphere and the other repairs are flagged and invertible" begin
    # Guards the policy's path through the 3+1 layer: a repaired state must
    # carry its flag, be a state the scheme can hold (finite, subluminal,
    # inside the EOS's box), and — what lets the atmosphere reset write
    # `prim2con` of it back — be recovered again as itself. The fields stay
    # what they were: the policy repairs the fluid only.
    eos = rec_gas(2)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(eos, REC_ρ_ATM)
    F = REC_EOS
    rng = Xoshiro(26)
    m = rec_metric(rng)
    E = 1e-6 * rec_direction(rng, m)
    B = 2e-6 * rec_direction(rng, m)
    cold = cold_guesses(Float64)
    recover(c) = con2prim_hydro(eos, opts, pol, m, c, E, B, cold)
    # The EOS's warm start is NaN for an analytic EOS, so it is not read.
    healthy(q) = all(isfinite, q[1:(NPRIM - 1)]) && TreeGRRMHD.prim_w(q) <= acosh(100.0) + 4eps() &&
                 REC_BOX.s_window[1] <= TreeGRRMHD.prim_s(q) <= REC_BOX.s_window[2]

    # Below the atmosphere: the whole state is replaced, at rest at the
    # floor density, and the replacement is a fixed point.
    thin = primitive_state(eos, m, 1e-11, 0.5 * rec_direction(rng, m), 5.0, 0.3)
    q, flags = recover(prim2con(eos, m, thin, E, B))
    @test flags & F.FLAG_POL_ATMOSPHERE != 0
    @test TreeGRRMHD.prim_ρ(q) == REC_ρ_ATM
    @test TreeGRRMHD.prim_v(q) == zero(SVector{3}) && TreeGRRMHD.prim_w(q) == 0
    @test healthy(q)
    q2, flags2 = recover(prim2con(eos, m, q, E, B))
    @test flags2 == F.FLAG_POL_ATMOSPHERE        # at the floor, so flagged again
    @test rec_same(q2, q)                        # and the same state

    # Non-finite input: excised to the atmosphere, and said so.
    c = prim2con(eos, m, rec_prim(rng, eos, m), E, B)
    for slot in (TreeGRRMHD.U_D, TreeGRRMHD.U_TAU, TreeGRRMHD.U_S + 1)
        bad = ntuple(i -> i == slot ? NaN : c[i], NU)
        q, flags = recover(bad)
        @test flags & F.FLAG_POL_NONFINITE != 0
        @test flags & F.FLAG_POL_ATMOSPHERE != 0
        @test healthy(q)
    end
    q, flags = con2prim_hydro(eos, opts, pol, m, c, SVector(NaN, 0, 0), B, cold)
    @test flags & F.FLAG_POL_NONFINITE != 0 && healthy(q)

    # Faster than the cap (W = 200 > 100): the rapidity is capped with `D`
    # kept, and the capped state recovers as itself.
    fast = primitive_state(eos, m, 1e-4, tanh(acosh(200.0)) * rec_direction(rng, m), 5.0, 0.3)
    c = prim2con(eos, m, fast, E, B)
    q, flags = recover(c)
    @test flags & F.FLAG_POL_W_CAPPED != 0
    @test TreeGRRMHD.prim_w(q) == acosh(100.0)
    @test isapprox(TreeGRRMHD.prim_ρ(q) * cosh(TreeGRRMHD.prim_w(q)),
                   TreeGRRMHD.cons_D(c) / m.sqrtγ; rtol=4eps())
    @test healthy(q)
    q2, flags2 = recover(prim2con(eos, m, q, E, B))
    @test flags2 == 0
    κ_ρ, κ_s = rec_conditioning(eos, q, TreeGRRMHD.cons_τ(c) / m.sqrtγ)
    eρ, es, _ = rec_prim_errors(m, q, q2, κ_ρ, κ_s)
    @test eρ < 3e-11 && es < 1e-10

    # Less energy than the coldest state of this density and momentum —
    # here because the field's energy is most of `τ`, and `τ` lost a little
    # more than the fluid had: the entropy is floored onto the window.
    warm = primitive_state(eos, m, 1e-4, 0.3 * rec_direction(rng, m), 5.0, 0.3)
    c = prim2con(eos, m, warm, E, B)
    τ_thermal = m.sqrtγ * 1e-4 * cosh(TreeGRRMHD.prim_w(warm))^2 *
                REC_EOS.evaluate(eos, 1e-4, 5.0, 0.3, NaN).U
    cold_c = ntuple(i -> i == TreeGRRMHD.U_TAU ? c[i] - 1.5 * τ_thermal : c[i], NU)
    q, flags = recover(cold_c)
    @test flags & F.FLAG_POL_S_FLOORED != 0
    @test TreeGRRMHD.prim_s(q) == REC_BOX.s_window[1]
    @test healthy(q)
    q2, flags2 = recover(prim2con(eos, m, q, E, B))
    @test flags2 == 0
    @test isapprox(TreeGRRMHD.prim_s(q2), TreeGRRMHD.prim_s(q); rtol=1e-10)

    # Above the density ceiling: clamped, not excised.
    dense = primitive_state(eos, m, 3e-2, 0.2 * rec_direction(rng, m), 5.0, 0.3)
    q, flags = recover(prim2con(eos, m, dense, E, B))
    @test flags & F.FLAG_POL_ρ_CLAMPED != 0
    @test TreeGRRMHD.prim_ρ(q) == 1e-2
    @test healthy(q)

    # The flags travel in `P` and come back out as EntropyEOS's word.
    P = TreeGRRMHD.pack_P(q, flags, E, B, 0.0, m)
    @test TreeGRRMHD.aux_flags(P) === flags
end

# The three conversions in one kernel, as the right-hand side will call them.
@kernel function rec_kernel!(out_hydro, out_ideal, flags, @Const(prims), @Const(metrics),
                             @Const(Es), @Const(Bs), eos, opts, pol)
    i = @index(Global)
    m = metrics[i]
    prim = prims[i]
    E, B = Es[i], Bs[i]
    T = eltype(E)
    cons = prim2con(eos, m, prim, E, B)
    q, fh = con2prim_hydro(eos, opts, pol, m, cons, E, B, cold_guesses(T))
    cons_ideal = prim2con(eos, m, prim, ideal_E(m, TreeGRRMHD.prim_v(prim), B), B)
    r, fi = con2prim_ideal(eos, opts, pol, m, cons_ideal, B, guesses(q))
    out_hydro[i] = q
    out_ideal[i] = r
    flags[i] = fh | fi
end

# The bytes one call allocates, after a first call compiles it, measured
# inside a function specialized on `f` and every argument: Julia 1.11 does
# not specialize a plain `args...` here, and then boxes the result (96 bytes
# for an `NTuple{10,Float64}`), which 1.12 and later do not.
function rec_allocations(f::F, args::Vararg{Any,N}) where {F,N}
    f(args...)
    return @allocated f(args...)
end

function rec_kernel_run(eos, opts, pol, prims, metrics, Es, Bs)
    n = length(prims)
    T = eltype(eltype(Es))
    out_h = Vector{NTuple{NPRIM,T}}(undef, n)
    out_i = Vector{NTuple{NPRIM,T}}(undef, n)
    flags = Vector{UInt32}(undef, n)
    backend = CPU()
    rec_kernel!(backend)(out_h, out_i, flags, prims, metrics, Es, Bs, eos, opts, pol;
                         ndrange=n)
    synchronize(backend)
    return out_h, out_i, flags
end

function rec_host_run(eos, opts, pol, prims, metrics, Es, Bs)
    T = eltype(eltype(Es))
    out_h = NTuple{NPRIM,T}[]
    out_i = NTuple{NPRIM,T}[]
    flags = UInt32[]
    for i in eachindex(prims)
        m, prim, E, B = metrics[i], prims[i], Es[i], Bs[i]
        q, fh = con2prim_hydro(eos, opts, pol, m, prim2con(eos, m, prim, E, B), E, B,
                               cold_guesses(T))
        cons_ideal = prim2con(eos, m, prim, ideal_E(m, TreeGRRMHD.prim_v(prim), B), B)
        r, fi = con2prim_ideal(eos, opts, pol, m, cons_ideal, B, guesses(q))
        push!(out_h, q)
        push!(out_i, r)
        push!(flags, fh | fi)
    end
    return out_h, out_i, flags
end

function rec_states(rng, eos, ::Type{T}, n) where {T}
    metrics = Metric3{T}[]
    prims = NTuple{NPRIM,T}[]
    Es = SVector{3,T}[]
    Bs = SVector{3,T}[]
    for _ in 1:n
        m64 = rec_metric(rng)
        m = Metric3(T(m64.α), T.(m64.β), T.(m64.γ))
        # States `Float32` can hold: dense enough that the thermal energy is
        # a percent of `τ` or more, and inside the middle of the entropy
        # window. At ρ = 1e-9 the gas's `U` is 1e-7 of its rest mass, `κ_s`
        # is 1e6, and at `Float32` the entropy is not determined at all —
        # such a state is recovered across the window's edge and repaired
        # (`FLAG_POL_S_CEILED`), which is the policy working and the reason
        # physics is validated in `Float64` ("Precision and devices" in
        # `CODE.md`).
        p64 = rec_prim(rng, rec_gas(2), m64; wmax=acosh(3.0), swindow=(4.5, 5.5),
                       log10ρ=(-4.0, -2.1))
        prim = primitive_state(eos, m, T(TreeGRRMHD.prim_ρ(p64)), T.(TreeGRRMHD.prim_v(p64)),
                               T(TreeGRRMHD.prim_s(p64)), T(TreeGRRMHD.prim_Yₑ(p64)))
        # Fields carrying a tenth of the fluid's `τ` each.
        zero3 = zero(SVector{3,Float64})
        τ_fluid = TreeGRRMHD.cons_τ(prim2con(rec_gas(2), m64, p64, zero3, zero3)) / m64.sqrtγ
        push!(metrics, m)
        push!(prims, prim)
        push!(Es, T.(sqrt(τ_fluid / 5) * rec_direction(rng, m64)))
        push!(Bs, T.(sqrt(τ_fluid / 5) * rec_direction(rng, m64)))
    end
    return prims, metrics, Es, Bs
end

@testset "The conversions run inside a CPU kernel, bit for bit: T=$T" for T in (Float64, Float32)
    # Guards the kernel-callability every later step relies on: the metric,
    # the EOS, the options and the policy are `isbits` arguments, nothing
    # throws or allocates, and a kernel computes exactly what the host does.
    # `Float32` is the smoke test and the leak detector — a `Float64`
    # constant anywhere in the 3+1 layer would widen a result and fail the
    # element types here.
    eos = REC_EOS.narrow(rec_gas(2), T)
    opts = REC_EOS.Con2PrimOptions{T}()
    pol = REC_EOS.default_policy(eos, T(REC_ρ_ATM))
    @test isbits(eos) && isbits(opts) && isbits(pol) && isbits(flat_metric(T))
    prims, metrics, Es, Bs = rec_states(Xoshiro(27), eos, T, 64)
    kh, ki, kf = rec_kernel_run(eos, opts, pol, prims, metrics, Es, Bs)
    hh, hi, hf = rec_host_run(eos, opts, pol, prims, metrics, Es, Bs)
    @test isequal(kh, hh)
    @test isequal(ki, hi)
    @test kf == hf
    @test eltype(kh) === NTuple{NPRIM,T}
    @test all(==(0), kf)

    # Allocation-free and inferred, on the host.
    m, prim, E, B = metrics[1], prims[1], Es[1], Bs[1]
    cons = prim2con(eos, m, prim, E, B)
    @test (@inferred prim2con(eos, m, prim, E, B)) isa NTuple{NU,T}
    @test (@inferred con2prim_hydro(eos, opts, pol, m, cons, E, B, cold_guesses(T))) isa
          Tuple{NTuple{NPRIM,T},UInt32}
    @test (@inferred con2prim_ideal(eos, opts, pol, m, cons, B, cold_guesses(T))) isa
          Tuple{NTuple{NPRIM,T},UInt32}
    @test rec_allocations(prim2con, eos, m, prim, E, B) == 0
    @test rec_allocations(con2prim_hydro, eos, opts, pol, m, cons, E, B, cold_guesses(T)) == 0
    @test rec_allocations(con2prim_ideal, eos, opts, pol, m, cons, B, cold_guesses(T)) == 0

    # At `Float32` the recovery is a smoke test: it runs, it is `Float32`,
    # and it lands on the state to EntropyEOS's `Float32` accuracy — its
    # tolerance is 512 eps = 6.1e-5 — conditioned as the `Float64` round trip
    # is. The same claim at `Float64` is the round trip's, much tighter.
    worst = 0.0
    for i in eachindex(prims)
        p, q, m = prims[i], kh[i], metrics[i]
        τ = Float64(TreeGRRMHD.cons_τ(prim2con(eos, m, p, Es[i], Bs[i])) / m.sqrtγ)
        κ_ρ, _ = rec_conditioning(rec_gas(2), map(Float64, p), τ)
        dv = TreeGRRMHD.prim_v(q) - TreeGRRMHD.prim_v(p)
        worst = max(worst,
                    max(abs(TreeGRRMHD.prim_ρ(q) - TreeGRRMHD.prim_ρ(p)) / TreeGRRMHD.prim_ρ(p),
                        sqrt(TreeGRRMHD.dot_up(m, dv, dv))) / κ_ρ)
    end
    T === Float32 && @info "The Float32 smoke run, worst conditioned ρ and v error" worst
    # Twice the tolerance at `Float32` (measured 3.6e-5 in step 1).
    @test worst < (T === Float64 ? 3e-11 : 2 * 512 * Float64(eps(Float32)))
end

@testset "A table in g/cm³ runs the recovery inside a kernel through the adapter" begin
    # Guards the adapter as a kernel argument: a `GeometricEOS` around a
    # table holds arrays, so it is not `isbits` on the host, but on the CPU
    # backend it is passed as it is, and the kernel must compute what the
    # host computes. (On a device it would be `Adapt`ed, step 14.)
    table = REC_EOS.build_eos(REC_EOS.make_synthetic_table())
    g = GeometricEOS(table.view; density=MSUN_DENSITY_CGS)
    opts = REC_EOS.Con2PrimOptions{Float64}()
    pol = REC_EOS.default_policy(g, 1e7 / MSUN_DENSITY_CGS)
    rng = Xoshiro(28)
    m = flat_metric(Float64)
    prims = NTuple{NPRIM,Float64}[]
    for _ in 1:16
        ρ = exp10(9 + 4 * rand(rng)) / MSUN_DENSITY_CGS
        sr = REC_EOS.srange(g, ρ, 0.3)
        s = sr.s_min + (0.2 + 0.6 * rand(rng)) * (sr.s_max - sr.s_min)
        push!(prims, primitive_state(g, m, ρ, 0.3 * rec_direction(rng, m), s, 0.3))
    end
    metrics = fill(m, length(prims))
    Bs = [1e-12 * rec_direction(rng, m) for _ in prims]
    Es = [1e-12 * rec_direction(rng, m) for _ in prims]
    kh, ki, kf = rec_kernel_run(g, opts, pol, prims, metrics, Es, Bs)
    hh, hi, hf = rec_host_run(g, opts, pol, prims, metrics, Es, Bs)
    @test isequal(kh, hh) && isequal(ki, hi) && kf == hf
    @test all(==(0), kf)
end
