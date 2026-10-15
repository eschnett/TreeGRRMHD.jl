# Rung 1b: the discontinuous problems in one dimension ("Validation ladder" 1
# in `CODE.md`; step 6 of `PLAN.md`), each a one-dimensional `MHDCase` with
# Dirichlet boundaries from its own data:
#
#   * Balsara's (2001) five relativistic MHD Riemann problems, under
#     `IdealConductor`, with the exact plateau states of Giacomazzo & Rezzolla
#     (2006) for comparison;
#   * the resistive shock tube swept over the conductivity (Palenzuela et al.
#     2009; Dionysopoulou et al. 2013), from the vacuum's wave solution at
#     σ = 0 to the ideal one;
#   * Komissarov's (2007) self-similar current sheet, against its analytic
#     erf profile;
#   * a shock tube into a near vacuum, a density ratio of 10⁸, which the
#     atmosphere reset has to hold.
#
# **A Riemann problem as a case.** The state is uniform on either side of
# `x₀`, and so is everything the boundary needs: the conserved state of the
# side, and its vector potential. The transverse field is the curl of a
# piecewise linear `A`, continuous at `x₀` — in one dimension
# `𝔅ʸ = −∂ₓA_z` and `𝔅ᶻ = ∂ₓA_y` — and the normal field, which no
# translation-invariant `A` produces, is the background `𝔅̄ˣ` (`src/curl.jl`).
# Each side's electric field makes its potential grow, `∂ₜA = −E` (`Φ̃ = 0`),
# so the potential the boundary is filled with is `A(x, 0) − t E_side`: the
# solution outside the box until a wave arrives (`DirichletBoundary` in
# `src/driver.jl`). The diaphragm sits on a face of the mesh, so the Gauss
# averages of the initial data are each side's state exactly.
#
# **Published values, copied with their citation** (the reviewer's rule:
# nothing invented). The initial data of Balsara 1–5 are GRaM-X's Table 1
# (Shankar et al. 2023, arXiv:2210.17509, which lists `ε`; `p = (Γ − 1)ρε`)
# and Giacomazzo & Rezzolla's Table 4 (arXiv:gr-qc/0507102), which agree.
# The exact plateau states are Giacomazzo & Rezzolla's Tables 7–11.

"""
    shock_gas(T, Γ) -> IdealGasEOS{T}

The ideal gas of the discontinuous cases: `p = K(s) ρ^Γ` with
`K(s) = exp((Γ − 1)(s − 20))`, the entropy window `(1, 40)` — `K` from
`3·10⁻⁶` to `6·10⁵` at `Γ = 5/3` and from `6·10⁻⁹` to `5·10⁸` at `Γ = 2` —
the density box `(10⁻¹⁴, 10⁶)` and `Yₑ ∈ (0.05, 0.6)`. The window is wide
because a shock raises the entropy by orders of magnitude (Balsara 3's two
sides differ by `10⁴` in `K`); `s_ref = 20` keeps `s` of order 1 to 40, as
EntropyEOS's bracket scan wants (proposed in step 6).
"""
shock_gas(::Type{T}, Γ) where {T} =
    EntropyEOS.IdealGasEOS{T}(; Γ=T(Γ), K_ref=one(T), s_ref=T(20), s_window=(one(T), T(40)),
                              ρ_bounds=(T(1 // 10^14), T(10^6)),
                              yₑ_bounds=(T(1 // 20), T(3 // 5)))

"""
    ShockSide{T}

One side of a Riemann problem, `isbits`: the rest-mass density, the
contravariant velocity, the entropy, the electron fraction and the
contravariant fields `E`, `B` the normal observer sees (`B[1]` the normal
field, the same on both sides).
"""
struct ShockSide{T}
    ρ::T
    v::SVector{3,T}
    s::T
    Yₑ::T
    E::SVector{3,T}
    B::SVector{3,T}
end

"""
    ShockTube{T}(left, right, x₀)

A Riemann problem along `x` with its diaphragm at `x₀`, `isbits`: the case
data of [`shock_tube_case`](@ref). Its `fields` and `potential` are the
initial state, and — on either side of every wave — the solution as long as
no wave has reached the point.
"""
struct ShockTube{T}
    left::ShockSide{T}
    right::ShockSide{T}
    x₀::T
end

@inline tube_side(w::ShockTube, x) = x[1] < w.x₀ ? w.left : w.right

@inline function wave_fields(w::ShockTube, x::SVector{3,T}, t::T) where {T}
    q = tube_side(w, x)
    return q.ρ, q.v, q.s, q.Yₑ, q.E, q.B
end

@inline function wave_potential(w::ShockTube, x::SVector{3,T}, t::T) where {T}
    q = tube_side(w, x)
    ξ = x[1] - w.x₀
    A = SVector(zero(T), q.B[3] * ξ, -q.B[2] * ξ) - t * q.E
    return A, zero(T)
end

"""
    shock_tube_case([T]; Γ, left, right, x₀ = 0, extents = ((-1/2, 1/2),),
                    model = IdealConductor(), ideal_E = model isa IdealConductor,
                    roots = 40, ρ_atm = 10⁻¹⁰, eos = shock_gas(T, Γ))

A Riemann problem as a one-dimensional [`MHDCase`](@ref) with Dirichlet
boundaries (see the top of `src/cases/shocktubes.jl`). `left` and `right`
are named tuples `(ρ, p, v, B)` — the rest-mass density, the gas pressure and
the contravariant 3-vectors — with the same normal field `B[1]`, which becomes
the background `𝔅̄ˣ`. The electric field is the ideal `E = −v × B` when
`ideal_E` (the default under [`IdealConductor`](@ref), where every stage
imposes it), and zero otherwise, as the resistive tests start. `ρ_atm` is the
atmosphere of EntropyEOS's default policy.
"""
function shock_tube_case(::Type{T}=Float64; Γ, left, right, x₀=0, extents=((-1 // 2, 1 // 2),),
                         model::Conductivity=IdealConductor(),
                         ideal_E::Bool=model isa IdealConductor, roots=40,
                         ρ_atm=1 // 10^10, eos=shock_gas(T, Γ)) where {T}
    T(left.B[1]) == T(right.B[1]) || throw(ArgumentError(
        "the normal field must be the same on both sides, got $(left.B[1]) and " *
        "$(right.B[1]): ∇·B = 0 across the diaphragm, and in one dimension it is the " *
        "background 𝔅̄ˣ, which no translation-invariant A can change."))
    function side(q)
        ρ, p = T(q.ρ), T(q.p)
        v, B = SVector{3,T}(q.v), SVector{3,T}(q.B)
        s = EntropyEOS.polytropic_entropy(eos, p / ρ^T(Γ))
        E = ideal_E ? TreeGRRMHD.ideal_E(flat_metric(T), v, B) : zero(SVector{3,T})
        return ShockSide{T}(ρ, v, s, T(1 // 2), E, B)
    end
    w = ShockTube{T}(side(left), side(right), T(x₀))
    return MHDCase(T, Val(1); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(ρ_atm)), model=model,
                   𝔅̄=SVector(T(left.B[1]), zero(T), zero(T)), periodic=false,
                   extents=extents, roots=roots)
end

# --- Balsara 1–5 ----------------------------------------------------------------------

"""
    BALSARA

The initial data of Balsara's (2001) five relativistic MHD Riemann problems,
`(Γ, t_end, left, right)` with each state `(ρ, p, v, B)`, as tabulated in
GRaM-X's Table 1 (Shankar et al. 2023, arXiv:2210.17509; it lists the
specific internal energy, `p = (Γ − 1)ρε`) and in Giacomazzo & Rezzolla
(2006, arXiv:gr-qc/0507102), Table 4. On `[−1/2, 1/2]`, the diaphragm at 0.
Exact rationals where the tables print short decimals.
"""
const BALSARA = (
    # 1: the relativistic Brio–Wu tube.
    (Γ=2 // 1, t_end=2 // 5,
     left=(ρ=1 // 1, p=1 // 1, v=(0, 0, 0), B=(1 // 2, 1 // 1, 0)),
     right=(ρ=1 // 8, p=1 // 10, v=(0, 0, 0), B=(1 // 2, -1 // 1, 0))),
    # 2: blast wave 1.
    (Γ=5 // 3, t_end=2 // 5,
     left=(ρ=1 // 1, p=30 // 1, v=(0, 0, 0), B=(5 // 1, 6 // 1, 6 // 1)),
     right=(ρ=1 // 1, p=1 // 1, v=(0, 0, 0), B=(5 // 1, 7 // 10, 7 // 10))),
    # 3: blast wave 2.
    (Γ=5 // 3, t_end=2 // 5,
     left=(ρ=1 // 1, p=1000 // 1, v=(0, 0, 0), B=(10 // 1, 7 // 1, 7 // 1)),
     right=(ρ=1 // 1, p=1 // 10, v=(0, 0, 0), B=(10 // 1, 7 // 10, 7 // 10))),
    # 4: colliding flows.
    (Γ=5 // 3, t_end=2 // 5,
     left=(ρ=1 // 1, p=1 // 10, v=(999 // 1000, 0, 0), B=(10 // 1, 7 // 1, 7 // 1)),
     right=(ρ=1 // 1, p=1 // 10, v=(-999 // 1000, 0, 0), B=(10 // 1, -7 // 1, -7 // 1))),
    # 5: the generic Alfvén test.
    (Γ=5 // 3, t_end=11 // 20,
     left=(ρ=108 // 100, p=95 // 100, v=(4 // 10, 3 // 10, 2 // 10), B=(2 // 1, 3 // 10, 3 // 10)),
     right=(ρ=1 // 1, p=1 // 1, v=(-45 // 100, -2 // 10, 2 // 10), B=(2 // 1, -7 // 10, 5 // 10))),
)

"""
    balsara_case([T], k; roots = 40, model = IdealConductor(), kwargs...)

Balsara's test `k ∈ 1:5` ([`BALSARA`](@ref)) as a [`shock_tube_case`](@ref)
on `[−1/2, 1/2]` with `roots` blocks; run it to `BALSARA[k].t_end` (0.4, and
0.55 for test 5). The fastest wave of each stays inside the box until then.
"""
function balsara_case(::Type{T}, k::Integer; roots=40, model=IdealConductor(),
                      kwargs...) where {T}
    1 ≤ k ≤ 5 || throw(ArgumentError("Balsara's tests are numbered 1 to 5, got $k."))
    b = BALSARA[k]
    return shock_tube_case(T; Γ=b.Γ, left=b.left, right=b.right, model=model, roots=roots,
                           kwargs...)
end
balsara_case(k::Integer; kwargs...) = balsara_case(Float64, k; kwargs...)

"""
    BALSARA_EXACT

The exact solutions of Balsara's tests 1–5 at their final times, Giacomazzo
& Rezzolla (2006, J. Fluid Mech. 562, 223; arXiv:gr-qc/0507102), Tables 7–11:
for the regions `R1` (the left state) to `R8` (the right state) between the
seven waves — fast, Alfvén, slow, contact, slow, Alfvén, fast — the values
they print to four digits, `(ρ, p, vˣ, vʸ, vᶻ, Bʸ, Bᶻ)`. Their `p` is the
**total** pressure `p_gas + b²/2` with the comoving `b²`: `R1` and `R8`
reproduce the initial states' totals (1.625 and 0.725 for test 1). Their
stated accuracy is 10⁻¹⁰, but 10⁻⁷ for test 4 and 3·10⁻⁴ for test 5.
Host data, never in a kernel.
"""
const BALSARA_EXACT = (
    # Table 7, Balsara 1.
    ((1.000, 1.625, 0.0, 0.0, 0.0, 1.000, 0.0),
     (0.6257, 0.6989, 0.3742, -0.3561e-1, 0.0, 0.6594, 0.0),
     (0.6257, 0.6989, 0.3742, -0.3561e-1, 0.0, 0.6594, 0.0),
     (0.7092, 0.7062, 0.2555, -0.6804, 0.0, -0.4285, 0.0),
     (0.2695, 0.7062, 0.2555, -0.6804, 0.0, -0.4285, 0.0),
     (0.1223, 0.6976, -0.2080e-1, -0.3460e-2, 0.0, -0.9769, 0.0),
     (0.1223, 0.6976, -0.2080e-1, -0.3460e-2, 0.0, -0.9769, 0.0),
     (0.1250, 0.7250, 0.0, 0.0, 0.0, -1.000, 0.0)),
    # Table 8, Balsara 2.
    ((1.000, 78.50, 0.0, 0.0, 0.0, 6.000, 6.000),
     (0.4300, 23.21, 0.6344, -0.9981e-1, -0.9981e-1, 3.045, 3.045),
     (0.4300, 23.21, 0.6344, -0.9981e-1, -0.9981e-1, 3.045, 3.045),
     (0.3830, 22.84, 0.6770, -0.5566e-1, -0.5566e-1, 3.205, 3.205),
     (2.828, 22.84, 0.6770, -0.5566e-1, -0.5566e-1, 3.205, 3.205),
     (1.582, 20.72, 0.4688, -0.2538, -0.2538, 3.971, 3.971),
     (1.582, 20.72, 0.4688, -0.2538, -0.2538, 3.971, 3.971),
     (1.000, 13.99, 0.0, 0.0, 0.0, 0.7000, 0.7000)),
    # Table 9, Balsara 3.
    ((1.000, 1099.0, 0.0, 0.0, 0.0, 7.000, 7.000),
     (0.1381, 86.04, 0.9246, -0.3513e-1, -0.3513e-1, 2.238, 2.238),
     (0.1381, 86.04, 0.9246, -0.3513e-1, -0.3513e-1, 2.238, 2.238),
     (0.9798e-1, 76.53, 0.9529, 0.4366e-1, 0.4366e-1, 4.670, 4.670),
     (10.10, 76.53, 0.9529, 0.4366e-1, 0.4366e-1, 4.670, 4.670),
     (1.218, 63.63, 0.4670, -0.4270, -0.4270, 9.408, 9.408),
     (1.218, 63.63, 0.4670, -0.4270, -0.4270, 9.408, 9.408),
     (1.000, 50.59, 0.0, 0.0, 0.0, 0.7000, 0.7000)),
    # Table 10, Balsara 4.
    ((1.000, 50.20, 0.9990, 0.0, 0.0, 7.000, 7.000),
     (51.75, 1184.0, 0.4408e-1, 0.3263e-1, 0.3263e-1, 16.68, 16.68),
     (51.75, 1184.0, 0.4408e-1, 0.3263e-1, 0.3263e-1, 16.68, 16.68),
     (61.48, 1188.0, 0.1086e-7, -0.2877, -0.2877, 0.8042e-9, 0.8036e-9),
     (61.48, 1188.0, -0.1089e-7, -0.2877, -0.2877, 0.8042e-9, 0.8036e-9),
     (51.75, 1184.0, -0.4408e-1, 0.3263e-1, 0.3263e-1, -16.68, -16.68),
     (51.75, 1184.0, -0.4408e-1, 0.3263e-1, 0.3263e-1, -16.68, -16.68),
     (1.000, 50.20, -0.9990, 0.0, 0.0, -7.000, -7.000)),
    # Table 11, Balsara 5.
    ((1.080, 2.885, 0.4000, 0.3000, 0.2000, 0.3000, 0.3000),
     (2.447, 5.908, -0.1331, 0.2111, 0.1751, 0.2662, 0.5076),
     (2.447, 5.908, -0.1215, 0.1264, 0.1158, -0.1182, 0.2302),
     (2.050, 5.616, -0.4547e-1, -0.1463, 0.2146, -1.175, 0.5852),
     (1.884, 5.616, -0.4543e-1, -0.1462, 0.2149, -1.175, 0.5850),
     (1.642, 5.488, -0.1129, -0.4606e-1, 0.1601, -1.429, 0.7320),
     (1.642, 5.488, -0.1155, -0.8536e-1, 0.1027, -1.272, 0.9468),
     (1.000, 2.918, -0.4500, -0.2000, 0.2000, -0.7000, 0.5000)),
)

"""
    BALSARA_PLATEAUS

Where the plateaus of [`BALSARA_EXACT`](@ref) lie at the final time, as
windows `(region, x_lo, x_hi)` per test (measured in step 6): the longest run
of cells of the 3200-cell reference (`bin/shocktube_reference.jl`, either
tableau, the same to the digits printed) within 2% of the region's exact
state ([`plateau_error`](@ref)), less 0.01 — four cells of the 400-cell
mesh — at either end, so that a 400-cell run's smeared waves stay outside
it. A region whose window would be narrower than three cells of that mesh is
left out: test 2's `R6`/`R7` (the reference holds them over 0.023), test 3's
dense shell `R5` and its `R6`/`R7`, and test 5's `R3` and `R6`. Regions with
the same state as their neighbour (`R3` = `R2` and `R7` = `R6` where no
Alfvén wave rotates the field; test 4's `R5`, the mirror of `R4`) appear
once. `R1` and `R8` are the initial states.
"""
const BALSARA_PLATEAUS = (
    ((1, -0.49, -0.35625), (2, -0.24281, 0.00094), (4, 0.02406, 0.08906),
     (5, 0.11438, 0.14156), (6, 0.16375, 0.36813), (8, 0.39406, 0.49)),
    ((1, -0.49, -0.37406), (2, -0.23875, 0.02906), (4, 0.06, 0.25719),
     (5, 0.28687, 0.34625), (8, 0.39438, 0.49)),
    ((1, -0.49, -0.33875), (2, 0.18, 0.24031), (4, 0.29312, 0.36594), (8, 0.41, 0.49)),
    ((1, -0.49, -0.28281), (2, -0.26094, -0.07094), (4, -0.04781, -0.0125),
     (6, 0.07094, 0.26094), (8, 0.28281, 0.49)),
    ((1, -0.49, -0.37594), (2, -0.35469, -0.32594), (4, -0.27719, -0.0375),
     (5, -0.01531, 0.18938), (7, 0.2375, 0.37094), (8, 0.3925, 0.49)),
)

"""
    plateau_state(pr, i) -> NTuple{7}

The state of cell `i` of a one-dimensional profile ([`profile_1d`](@ref)) in
the variables of [`BALSARA_EXACT`](@ref): `(ρ, p + b²/2, vˣ, vʸ, vᶻ, Bʸ, Bᶻ)`,
with the total pressure from the comoving `b²`.
"""
plateau_state(pr, i) =
    (pr.ρ[i], pr.p[i] + pr.b2[i] / 2, pr.v[i, 1], pr.v[i, 2], pr.v[i, 3], pr.B[i, 2], pr.B[i, 3])

"""
    plateau_error(k, region, state) -> NTuple{7}

How far `state` ([`plateau_state`](@ref)) is from the exact plateau `region`
of Balsara's test `k` ([`BALSARA_EXACT`](@ref)), per variable, as a fraction
of that variable's scale (proposed in step 6): `ρ` and the total pressure
against the region's own value, and each component of `v` and `B` against
the largest magnitude that component takes in any region of the test — a
component that is zero, or `10⁻⁸`, in one region has no relative error to
speak of, and its scale is that of the wave that moves it. A component that
is zero in every region (`vᶻ`, `Bᶻ` of the coplanar test 1) is measured
against the largest magnitude of any component of its vector. "Within 2%" is
every entry at most `0.02`.
"""
function plateau_error(k::Integer, region::Integer, state)
    ex = BALSARA_EXACT[k]
    r = ex[region]
    return ntuple(7) do j
        vector = j ≤ 2 ? (j:j) : j ≤ 5 ? (3:5) : (6:7)
        scale = j ≤ 2 ? abs(r[j]) : maximum(q -> abs(q[j]), ex)
        scale > 0 || (scale = maximum(q -> maximum(i -> abs(q[i]), vector), ex))
        abs(state[j] - r[j]) / scale
    end
end

# --- the resistive shock tube ---------------------------------------------------------

"""
    resistive_tube_case([T], model; roots = 20)

The resistive shock tube of Palenzuela et al. (2009, §5.1.3; arXiv:0810.1838)
and Dionysopoulou et al. (2013, arXiv:1208.3487), which sweep it over the
conductivity: a Brio–Wu tube with `(ρ, p, Bʸ) = (1, 1, 1/2)` left of
`x₀ = 1/2` and `(1/8, 1/10, −1/2)` right of it, every other variable zero
(`Bˣ = 0`, `v = 0`, `E = 0`), `Γ = 2`, on `[0, 1]` to `t = 0.4`. At σ = 0 the
field and the fluid decouple and the field's jump leaves as two light
waves; as σ → ∞ the solution is the ideal one, whose three waves (a fast
rarefaction, the contact, a fast shock) Giacomazzo & Rezzolla solve exactly.
`model` is the conductivity: [`Vacuum`](@ref), [`Constant`](@ref) or
[`IdealConductor`](@ref).
"""
function resistive_tube_case(::Type{T}, model::Conductivity; roots=20) where {T}
    return shock_tube_case(T; Γ=2, left=(ρ=1, p=1, v=(0, 0, 0), B=(0, 1 // 2, 0)),
                           right=(ρ=1 // 8, p=1 // 10, v=(0, 0, 0), B=(0, -1 // 2, 0)),
                           x₀=1 // 2, extents=((0, 1),), model=model, ideal_E=false,
                           roots=roots)
end
resistive_tube_case(model::Conductivity; kwargs...) = resistive_tube_case(Float64, model; kwargs...)

# --- the self-similar current sheet ---------------------------------------------------

"""
    erf_series(x)

The error function in pure Julia, generic in the number type and kernel-
callable: `erf x = (2/√π) e^{−x²} Σₙ 2ⁿ x^{2n+1}/(2n+1)!!`, a series of
positive terms (no cancellation), summed until a term no longer changes the
sum; `±1` beyond `|x| = 6`, where `erfc` is below `2·10⁻¹⁷`. Agrees with
SpecialFunctions' `erf` to a few ulp (a test), without the dependency.
"""
@inline function erf_series(x::T) where {T}
    a = abs(x)
    a ≥ 6 && return copysign(one(T), x)
    term = a
    sum = a
    n = 0
    while n < 400
        n += 1
        term *= 2 * a * a / (2 * n + 1)
        sum += term
        term ≤ eps(T) * sum && break
    end
    return copysign(2 / sqrt(T(π)) * exp(-a * a) * sum, x)
end

"""
    CurrentSheet{T}(; B0, σ, t0, ρ, s, Yₑ)

Komissarov's (2007) self-similar current sheet, `isbits`: `Bʸ` diffusing
through a resting fluid of conductivity `σ`,

    Bʸ(x, t) = B₀ erf(a x),   a = ½ √(σ/(t₀ + t)),
    Eᶻ = (1/σ) ∂ₓBʸ,          A_z = −B₀ [x erf(a x) + e^{−a²x²}/(a√π)],

the solution of `∂ₜBʸ = σ⁻¹ ∂ₓ²Bʸ`, which the full system approaches when
the displacement current and the fluid's motion are negligible. The case's
time `t` starts at 0 for the paper's `t₀ = 1`. `Eᶻ` is the quasi-static field
`∇ × B/σ`, which Ampère's law relaxes to in a time `1/σ` (proposed in step
6: the paper starts from `E = 0`), and `∂ₜA_z = −Eᶻ` holds exactly, so the
potential is also the boundary's at every time.
"""
struct CurrentSheet{T}
    B0::T
    σ::T
    t0::T
    ρ::T
    s::T
    Yₑ::T
end
CurrentSheet{T}(; B0, σ, t0, ρ, s, Yₑ) where {T} =
    CurrentSheet{T}(T(B0), T(σ), T(t0), T(ρ), T(s), T(Yₑ))

@inline sheet_a(w::CurrentSheet, t) = sqrt(w.σ / (w.t0 + t)) / 2

@inline function wave_fields(w::CurrentSheet, x::SVector{3,T}, t::T) where {T}
    a = sheet_a(w, t)
    By = w.B0 * erf_series(a * x[1])
    Ez = w.B0 * 2 * a / sqrt(T(π)) * exp(-(a * x[1])^2) / w.σ
    z = zero(T)
    return w.ρ, zero(SVector{3,T}), w.s, w.Yₑ, SVector(z, z, Ez), SVector(z, By, z)
end

@inline function wave_potential(w::CurrentSheet, x::SVector{3,T}, t::T) where {T}
    a = sheet_a(w, t)
    ξ = x[1]
    Az = -w.B0 * (ξ * erf_series(a * ξ) + exp(-(a * ξ)^2) / (a * sqrt(T(π))))
    return SVector(zero(T), zero(T), Az), zero(T)
end

"""
    current_sheet_case([T]; σ = 100, B0 = 1, p = 50, ρ = 1, Γ = 4/3, t0 = 1,
                       L = 3/2, roots = 20)

Komissarov's (2007, §6.1.4; arXiv:0708.0323) self-similar current sheet as a
one-dimensional [`MHDCase`](@ref) on `[−L, L]` with Dirichlet boundaries from
the analytic solution ([`CurrentSheet`](@ref)): `B₀ = 1`, `p = 50`, `ρ = 1`,
`v = 0`, `σ = 100` (`η = 0.01`), `Γ = 4/3`, from the profile at `t₀ = 1`
(the paper runs it to `t = 9`; Palenzuela et al. 2009 to 10). The gas
pressure is a hundred times the field's, so the fluid barely moves and the
field diffuses as in a resting conductor. [`Constant`](@ref)`(σ)`.
"""
function current_sheet_case(::Type{T}=Float64; σ=100, B0=1, p=50, ρ=1, Γ=4 // 3, t0=1,
                            L=3 // 2, roots=20) where {T}
    eos = shock_gas(T, Γ)
    s = EntropyEOS.polytropic_entropy(eos, T(p) / T(ρ)^T(Γ))
    w = CurrentSheet{T}(; B0=B0, σ=σ, t0=t0, ρ=ρ, s=s, Yₑ=T(1 // 2))
    return MHDCase(T, Val(1); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   pol=EntropyEOS.default_policy(eos, T(1 // 10^10)), model=Constant(T(σ)),
                   periodic=false, extents=((-T(L), T(L)),), roots=roots)
end

# --- the atmosphere tube ------------------------------------------------------------

"""
    atmosphere_tube_case([T]; ρ_atm = 10⁻⁸, By = 0, model = IdealConductor(),
                         roots = 20)

A blast into the atmosphere (proposed in step 6): a gas at `ρ = p = 1`,
`Γ = 5/3`, at rest left of `x₀ = 0` on `[−1/2, 1/2]`, and right of it the
atmosphere of EntropyEOS's default policy itself — `ρ_atm`, at rest, at the
entropy `s_atm` the policy resets to (the middle of the window) — so the
density ratio is `1/ρ_atm`, 10⁸ by default. A transverse field `Bʸ`,
uniform across the diaphragm, makes the atmosphere magnetically dominated
(`B²/ρ_atm = 100` at `Bʸ = 10⁻³`). Run to `t = 0.4`: the gas expands into the
atmosphere at up to `W ≈ 5`, and its front reaches `x ≈ 0.39`. Every cell
the expansion thins below the atmosphere's trigger, and every atmosphere
cell the reconstruction disturbs, is the reset's.
"""
function atmosphere_tube_case(::Type{T}=Float64; ρ_atm=1 // 10^8, By=0,
                              model::Conductivity=IdealConductor(), roots=20) where {T}
    Γ = 5 // 3
    eos = shock_gas(T, Γ)
    pol = EntropyEOS.default_policy(eos, T(ρ_atm))
    atm = EntropyEOS.policy_atmosphere(eos, pol, T(1 // 2))
    p_atm = EntropyEOS.evaluate(eos, atm.ρ, atm.s, atm.ye, T(NaN)).p
    return shock_tube_case(T; Γ=Γ, left=(ρ=1, p=1, v=(0, 0, 0), B=(0, By, 0)),
                           right=(ρ=atm.ρ, p=p_atm, v=(0, 0, 0), B=(0, By, 0)), model=model,
                           ρ_atm=ρ_atm, roots=roots, eos=eos)
end
