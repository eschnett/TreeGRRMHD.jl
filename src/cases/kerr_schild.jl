# Rung 3: analytic curved backgrounds ("Validation ladder" 3 in `CODE.md`;
# step 9 of `PLAN.md`). Two stationary solutions on a Schwarzschild hole in
# Kerr-Schild coordinates, each an exact solution of the whole system, matter
# and fields, at every time:
#
#   * **the Wald solution** (`Vacuum`): the hole in an asymptotically uniform
#     magnetic field (Wald 1974, Phys. Rev. D 10, 1680), above Michel's
#     accretion flow, which the field does not touch;
#   * **the magnetized Bondi flow** (`IdealConductor`): Michel's (1972)
#     relativistic accretion flow threaded by a radial monopole field
#     (Gammie, McKinney & Tóth 2003, ApJ 589, 444, §4.2; Porth et al. 2017,
#     ComAC 4, 1, §4.3), along which the fluid falls, so that `E = −v × B = 0`.
#
# **Kerr-Schild coordinates, `a = 0`** (Cook 2000, Living Rev. Relativ. 3, 5,
# §3.3.1; SpacetimeMetrics' `KerrSchild`): with `r = |x|`, `lⁱ = xⁱ/r` and
# `f = 2M/r`, `γ_ij = δ_ij + f l_i l_j`, `β^i = f lⁱ/(1 + f)`,
# `α = 1/√(1 + f)`, `√γ = √(1 + f)`, so `α√γ = 1`. The coordinates are
# horizon-penetrating, and the coordinate speeds of light are at most 1. The
# provider is `Analytic(KerrSchild(M))`, and every case function below takes
# its metric from that provider at the point, so that the case's initial data
# and the scheme see the same metric bit for bit.
#
# **Why `a = 0`** (proposed in step 9). Wald's field exists at any spin, but the
# matter must be stationary too, and Michel's flow is Schwarzschild's; Kerr has
# no closed-form stationary accretion flow. Under `Vacuum` the matter does not
# feel the field, so a spinning Wald case could carry a passive fluid at the
# floor and be measured on `𝔈` and `𝔅` alone — a follow-up for Erik.
#
# **Why the octant** (proposed in step 9). The monopole has no vector potential
# regular around a whole sphere: `A = q (n̂ × x)/(r (r + n̂·x))` is regular
# everywhere but on the Dirac string `x = −s n̂`, which carries the return
# flux. In a box around the hole the string crosses the evolved region and
# carries `4πq` through the faces it pierces. The cases therefore run on the
# octant `[0, L]³` with the hole at its corner and Dirichlet boundaries from
# their own data on every face (`DirichletBoundary`); the string points along
# `−(1, 1, 1)`, out of the octant through its corner, so it reaches only ghosts
# outside all three low faces at once, which nothing owned reads. Both
# solutions are smooth across the octant's faces (but at `r = 0`), so a
# Dirichlet face from the exact solution is a consistent boundary, and the
# octant is a tenth of the cost of the box around the hole. The hole's inside
# near the corner is the interior (`HoleInterior`, `src/interior.jl`).

# --- Michel's accretion flow ---------------------------------------------------------

"""
    MichelFlow{T}

Michel's (1972) transonic relativistic accretion flow onto a Schwarzschild hole
of mass `M`, for the polytrope `p = K ρ^Γ` (`n = 1/(Γ − 1)`), with its sonic
point at `r_c`, where the density is `ρ_c`, `isbits`. `C₁ = r² ρ |uʳ|` is the
mass flux per solid angle and `C₂ = h² (1 − 2M/r + (uʳ)²)` the Bernoulli
constant; `s` and `Yₑ` are the flow's (constant) entropy and electron fraction.
See [`michel_density`](@ref) and the top of `src/cases/kerr_schild.jl`.
"""
struct MichelFlow{T}
    M::T
    n::T
    K::T
    r_c::T
    ρ_c::T
    C₁::T
    C₂::T
    s::T
    Yₑ::T
end

"""
    MichelFlow(eos, M, Γ, r_c, ρ_c)

The flow on the ideal gas `eos` of adiabatic index `Γ`, from its sonic point
(Hawley, Smarr & Wilson 1984, ApJ 277, 296, §III; Rezzolla & Zanotti 2013,
*Relativistic Hydrodynamics*, §11.8): at `r_c`, `(uʳ)² = M/(2r_c)` and
`c_s² = (uʳ)²/(1 − 3(uʳ)²)`, from which the temperature `θ = p/ρ = c_s²/(Γ(1 −
n c_s²))`, `K = θ/ρ_c^{1/n}` and the two constants follow. Needs `r_c > 3M/2`.
"""
function MichelFlow(eos, M::T, Γ::T, r_c::T, ρ_c::T) where {T}
    r_c > 3 * M / 2 || throw(ArgumentError(
        "the sonic radius must lie outside 3M/2 (where (uʳ)² = 1/3), got r_c = $r_c, M = $M."))
    n = inv(Γ - 1)
    u² = M / (2 * r_c)
    c² = u² / (1 - 3 * u²)
    θ = c² / (Γ * (1 - n * c²))
    K = θ / ρ_c^inv(n)
    h = 1 + (n + 1) * θ
    C₁ = r_c^2 * ρ_c * sqrt(u²)
    C₂ = h^2 * (1 - 3 * u²)
    s = EntropyEOS.polytropic_entropy(eos, K)
    return MichelFlow{T}(M, n, K, r_c, ρ_c, C₁, C₂, s, T(1 // 2))
end

# The Bernoulli residual `F = h² (1 − 2M/r + u²) − C₂` and its derivative in
# `y = log ρ` at radius `r`, with `uʳ = C₁/(r² ρ)` from the mass flux.
@inline function michel_residual(f::MichelFlow{T}, r::T, y::T) where {T}
    ρ = exp(y)
    θ = f.K * exp(y / f.n)
    h = 1 + (f.n + 1) * θ
    u = f.C₁ / (r^2 * ρ)
    A = 1 - 2 * f.M / r
    F = h^2 * (A + u^2) - f.C₂
    dF = 2 * h * ((h - 1) * (A + u^2) / f.n - h * u^2)
    return F, dF
end

# The sonic condition `u² (1 − c_s²) − c_s² (1 − 2M/r)`, decreasing in `y`: zero
# where the flow at this density would move at the local sound speed, which is
# where the Bernoulli residual is smallest (its two roots are the subsonic and
# the supersonic flow).
@inline function michel_sonic(f::MichelFlow{T}, r::T, y::T) where {T}
    θ = f.K * exp(y / f.n)
    h = 1 + (f.n + 1) * θ
    c² = (h - 1) / (f.n * h)
    u = f.C₁ / (r^2 * exp(y))
    return u^2 * (1 - c²) - c² * (1 - 2 * f.M / r)
end

"""
    michel_density(f::MichelFlow, r) -> ρ

The rest-mass density of the transonic flow at radius `r`: the root of the
Bernoulli residual on the subsonic side of the sonic density outside `r_c`, on
the supersonic side inside it, and the one root inside the horizon, where the
residual is monotone. Bisection for the sonic density, then Newton safeguarded
by bisection in `log ρ`, to rounding; every loop has a fixed bound, so that a
kernel can call it. Near `r_c` the two roots merge and the root is
ill-conditioned (to about `√ε` at `r = r_c`); the cases keep the sonic point
outside their box.
"""
@inline function michel_density(f::MichelFlow{T}, r::T) where {T}
    y₀ = log(f.ρ_c)
    lo, hi = y₀ - 30, y₀ + 30
    if r > 2 * f.M
        a, b = lo, hi
        for _ in 1:80
            m = (a + b) / 2
            (m == a || m == b) && break
            if michel_sonic(f, r, m) > 0
                a = m
            else
                b = m
            end
        end
        ys = (a + b) / 2
        first(michel_residual(f, r, ys)) ≥ 0 && return exp(ys)    # at the sonic point
        if r > f.r_c
            lo = ys
        else
            hi = ys
        end
    end
    Flo = first(michel_residual(f, r, lo))
    x = (lo + hi) / 2
    for _ in 1:200
        F, dF = michel_residual(f, r, x)
        F == 0 && break
        if (F > 0) == (Flo > 0)
            lo = x
        else
            hi = x
        end
        xn = x - F / dF
        if !(lo < xn < hi)
            xn = (lo + hi) / 2
        end
        done = abs(xn - x) ≤ 2 * eps(T) * max(abs(x), one(T))
        x = xn
        (done || lo == hi) && break
    end
    return exp(x)
end

"""
    michel_state(f::MichelFlow, m::Metric3, x) -> (ρ, v)

The density and the contravariant velocity `vⁱ` the normal observer of the
Kerr-Schild slice (`m`, at `x`) measures. The flow is radial, with
`uʳ = −C₁/(r² ρ)` and `u_t = −√(1 − 2M/r + (uʳ)²)` (`h u_t` is Bernoulli's
constant; `u_t` is the same in Schwarzschild's and Kerr-Schild's time, which
differ by a function of `r`). The covariant radial component in Kerr-Schild,
`u_r = (uʳ + f u_t... )`, is written in the form regular across the horizon,

    u_r = ((uʳ)² (1 + f) − f²) / (uʳ − f E),    E = −u_t,  f = 2M/r,

from `uʳ = g^{rt} u_t + g^{rr} u_r = −f E + (1 − f) u_r`. Then `W v_i = u_i =
u_r l_i`, `W = √(1 + γ^{ij} u_i u_j)` and `vⁱ = γ^{ij} u_j / W`, in the metric
`m` itself.
"""
@inline function michel_state(f::MichelFlow{T}, m::Metric3{T}, x::SVector{3,T}) where {T}
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    ρ = michel_density(f, r)
    u = -f.C₁ / (r^2 * ρ)
    E = sqrt(1 - 2 * f.M / r + u^2)
    fr = 2 * f.M / r
    u_r = (u^2 * (1 + fr) - fr^2) / (u - fr * E)
    u_low = (u_r / r) * x
    W = sqrt(1 + dot_down(m, u_low, u_low))
    return ρ, raise(m, u_low) / W
end

# --- the two cases -------------------------------------------------------------------

"""
    BlackHoleCase{K,T,MET}

The data of a stationary case on a Kerr-Schild hole, `isbits`, tagged by `K`
(`:wald`, `:monopole`): the matter's [`MichelFlow`](@ref), the provider the
metric comes from, the field's strength `b` (Wald's `B₀`, the monopole's
charge `q`) and the monopole's string direction `−n̂`. See
`src/cases/kerr_schild.jl`.
"""
struct BlackHoleCase{K,T,MET}
    flow::MichelFlow{T}
    provider::MET
    b::T
    n̂::SVector{3,T}
end

@inline function wave_fields(w::BlackHoleCase{:wald}, x::SVector{3,T}, t::T) where {T}
    m = point_metric(w.provider, t, x)
    ρ, v = michel_state(w.flow, m, x)
    # Wald's potential at `a = 0` is `(B₀/2) ψ_μ`, `ψ = ∂_φ`: `A_i = (B₀/2)(−y,
    # x, 0)` (`ψ` is orthogonal to `l`, so `f l_i l_j ψʲ` vanishes), `𝒜_t =
    # f l_i ψⁱ = 0`, and so `Φ = 0`. Hence `𝔅 = (0, 0, B₀)`, and stationarity of
    # `A` (`∂ₜA_i = −(αE_i + ε_ijk βʲ Bᵏ) − ∂_i(αΦ − βʲA_j)`, both terms of the
    # gradient zero) gives `E = −(β/α) × B`: the ideal field of the coordinate
    # observer, `ideal_E` at `v = β/α`.
    B = SVector(zero(T), zero(T), w.b) / m.sqrtγ
    E = ideal_E(m, m.β / m.α, B)
    return ρ, v, w.flow.s, w.flow.Yₑ, E, B
end

@inline wave_potential(w::BlackHoleCase{:wald}, x::SVector{3,T}, t::T) where {T} =
    SVector(-x[2], x[1], zero(T)) * (w.b / 2), zero(T)

@inline function wave_fields(w::BlackHoleCase{:monopole}, x::SVector{3,T}, t::T) where {T}
    m = point_metric(w.provider, t, x)
    ρ, v = michel_state(w.flow, m, x)
    # The monopole: `𝔅 = q x/r³`, the flat curl of the potential below, and
    # `Bⁱ = 𝔅ⁱ/√γ`; radial, like `v`, so the ideal `E = −v × B` is zero.
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    B = (w.b / (r^3 * m.sqrtγ)) * x
    return ρ, v, w.flow.s, w.flow.Yₑ, zero(SVector{3,T}), B
end

# `A = q (n̂ × x)/(r (r + n̂·x))`, `q (1 − cos θ) ∇φ` about the axis `n̂`, whose
# curl is the monopole `q x/r³` but on the Dirac string `x = −s n̂`; `Φ = 0`
# (`βʲ A_j ∝ x · (n̂ × x) = 0`, `𝒜_t = 0`).
@inline function wave_potential(w::BlackHoleCase{:monopole}, x::SVector{3,T}, t::T) where {T}
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    n̂ = w.n̂
    c = SVector(n̂[2] * x[3] - n̂[3] * x[2], n̂[3] * x[1] - n̂[1] * x[3], n̂[1] * x[2] - n̂[2] * x[1])
    return (w.b / (r * (r + (n̂[1] * x[1] + n̂[2] * x[2] + n̂[3] * x[3])))) * c, zero(T)
end

"""
    hole_extents(T, L) -> NTuple{3}

The octant `[0, L]³` the cases run on (see the top of
`src/cases/kerr_schild.jl`).
"""
hole_extents(::Type{T}, L) where {T} = ntuple(_ -> (zero(T), T(L)), Val(3))

function hole_case(::Type{T}, kind::Symbol, b; M, L, roots, r_in, Γ, r_c, ρ_c, ρ_min,
                   interior) where {T}
    T(L) > 2 * T(M) || throw(ArgumentError(
        "the box [0, L]³ must reach outside the horizon at 2M; got L = $L, M = $M."))
    0 < T(r_in) < 2 * T(M) || throw(ArgumentError(
        "the interior's radius must lie inside the horizon (0 < r_in < 2M = $(2M)), got " *
        "r_in = $r_in: it replaces the evolution by the analytic solution, which is only " *
        "harmless where nothing can leave (CODE.md, \"The interior\")."))
    eos = shock_gas(T, Γ)
    flow = MichelFlow(eos, T(M), T(Γ), T(r_c), T(ρ_c))
    prov = Analytic(SpacetimeMetrics.KerrSchild{T}(T(M)))
    w = BlackHoleCase{kind,T,typeof(prov)}(flow, prov, T(b), SVector(one(T), one(T), one(T)) /
                                                        sqrt(T(3)))
    hole = interior ? HoleInterior(T(r_in)) : nothing
    return MHDCase(T, Val(3); fields=WaveFields(w), potential=WavePotential(w), eos=eos,
                   recovery_options(eos, T(ρ_min))...,
                   model=kind === :wald ? Vacuum() : IdealConductor(), provider=prov,
                   periodic=false, extents=hole_extents(T, L), roots=roots, interior=hole)
end

"""
    wald_case([T]; M = 1, B₀ = 1, L = 4, roots = 2, r_in = 1, Γ = 4/3, r_c = 8,
              ρ_c = 1, ρ_min = 10⁻⁸, interior = true)

The Wald solution at `a = 0` (Wald 1974) as a three-dimensional
[`MHDCase`](@ref) under `Vacuum`: a Schwarzschild hole of mass `M` in
Kerr-Schild coordinates (`Analytic(KerrSchild(M))`) in the field that is
uniform, `B₀ ẑ`, at infinity — `A = (B₀/2)(−y, x, 0)`, `Φ̃ = 0`, `𝔅 = B₀ ẑ`
and `E = −(β/α) × B` — above Michel's accretion flow ([`MichelFlow`](@ref):
`Γ = 4/3`, the sonic point at `r_c` with density `ρ_c`, on
[`shock_gas`](@ref)), which no current couples to the field. On the octant
`[0, L]³`, the hole at its corner, with Dirichlet boundaries from the solution
on every face and the interior `r < r_in` replaced by the solution
([`HoleInterior`](@ref)) unless `interior = false`. Stationary: the solution
at every `t` is the initial data (proposed in step 9; see the top of
`src/cases/kerr_schild.jl`).
"""
wald_case(::Type{T}=Float64; M=1, B₀=1, L=4, roots=2, r_in=1, Γ=4 // 3, r_c=8, ρ_c=1,
          ρ_min=1 // 10^8, interior::Bool=true) where {T} =
    hole_case(T, :wald, B₀; M, L, roots, r_in, Γ, r_c, ρ_c, ρ_min, interior)

"""
    bondi_case([T]; M = 1, q = 9, L = 4, roots = 2, r_in = 1, Γ = 4/3, r_c = 8,
               ρ_c = 1, ρ_min = 10⁻⁸, interior = true)

The magnetized Bondi flow (Gammie, McKinney & Tóth 2003, §4.2; Porth et al.
2017, §4.3) as a three-dimensional [`MHDCase`](@ref) under `IdealConductor`:
Michel's accretion flow ([`MichelFlow`](@ref), as in [`wald_case`](@ref)) onto
a Schwarzschild hole in Kerr-Schild coordinates, threaded by the radial
monopole `𝔅 = q x/r³` from `A = q (n̂ × x)/(r (r + n̂·x))`, `n̂ = (1, 1, 1)/√3`,
`Φ̃ = 0`. The field lies along the flow, so `E = −v × B = 0`, and the monopole
is a vacuum solution of Maxwell's equations on Schwarzschild, so it exerts no
force: the flow is Michel's exactly. On the octant `[0, L]³` (the string
`x = −s n̂` leaves through the corner), with Dirichlet boundaries and the
interior as for the Wald case. At the default `q`, `b²/ρ` is 0.98 at the
horizon, 6.1 at `r = M` and 0.030 at the far corner of the default box
(`r = 4√3 M`).
"""
bondi_case(::Type{T}=Float64; M=1, q=9, L=4, roots=2, r_in=1, Γ=4 // 3, r_c=8, ρ_c=1,
           ρ_min=1 // 10^8, interior::Bool=true) where {T} =
    hole_case(T, :monopole, q; M, L, roots, r_in, Γ, r_c, ρ_c, ρ_min, interior)
