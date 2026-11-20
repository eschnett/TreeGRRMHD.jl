# The radial pulsation modes of the TOV star (`src/tov.jl`), from linear perturbation theory:
# their frequencies, and their eigenfunctions as the initial velocity that excites one
# mode ("Validation ladder" 5 in `CODE.md`; proposed in step 11, after a sine-shaped
# velocity excited the overtones too weakly).
#
# **The equations** (proposed in step 11). Adiabatic radial perturbations `∝ e^{iωt}` of
# the static star, `ds² = −e^ν dt² + e^Λ dr² + r² dΩ²` (`r` areal), in the Lagrangian
# relative displacement `ξ = Δr/r` and `ζ = Δn/n = ΔP/(Γ P)` (the gas is ideal, so its
# adiabatic index is the polytrope's `Γ`). Two versions:
#
#   * coupled, with the metric perturbed: Chandrasekhar's (1964) equation in the first-order
#     form of Gondek, Haensel & Zdunik (1997, A&A 325, 217), eqs. (13)–(14),
#
#         dξ/dr  = −(3ξ + ζ)/r − ξ P′/(e + P),
#         dΔP/dr = ξ [ω² e^{Λ−ν} (e + P) r − 4P′ + P′² r/(e + P) − 8π e^Λ (e + P) P r]
#                  + ΔP [P′/(e + P) − 4π (e + P) r e^Λ];
#
#   * Cowling, with the metric fixed: baryon number `Δn/n = −(r² e^{Λ/2})⁻¹ d(r³ e^{Λ/2} ξ)/dr`
#     and the Euler equation `(e + P) e^{Λ−ν} ∂ₜ²(rξ) = −∂ᵣ δP − (δe + δP) ν′/2` (Eulerian
#     `δ`), which with `P′ = (e + P) η′` become
#
#         dξ/dr  = −(3ξ + ζ)/r − ξ Λ′/2,
#         dΔP/dr = ω² e^{Λ−ν} (e + P) r ξ + (e + P) d(r ξ/η′)/dr + (e + P) ζ η′ + ΔP η′.
#
# Both are written in the log-enthalpy `η` of `tov_solve`, the same RK4 steps, so the
# background is that solve's, the centre and the surface are grid points, and
# `dP/dη = e + P`, `e^{−ν} = h²/α_s²`. With `ζ` in place of `ΔP` the surface is a regular
# singular point: there `(e + P)/P ∝ 1/η` multiplies a bracket `G`, and the regular
# solution has `G = 0` at `η = 0`, the condition `ΔP = 0`. At the centre `ζ = −3ξ`, and
# `dξ/dη` is the limit `(3ξ + ζ)/q → (3ξ + ζ)′/q′`.
#
# **The eigenvalues.** Shooting outward from `ξ = 1`, `ζ = −3` at the centre to the last
# node before the surface, `η = Δ`; there the residual `G` changes sign at each eigenvalue,
# since a solution that is not regular grows like `η⁻²`. The roots of `G(ω²)` are
# bracketed on a grid in `ω²` and bisected to round-off; the residual's own error in `ω²`
# is of order `Δ³`. `ω` is the frequency in the coordinate time `t`, which is the time at
# infinity, so the frequency in kHz is `ω/(2π)` divided by `G M☉/c³` in ms.
#
# **The velocity.** At the phase of zero displacement the mode's velocity is `ω Δr`, so in
# the isotropic coordinates (`dr̄/r̄ = dr/(r √(1 − 2m/r))`) the Valencia velocity is
# `v^{r̄} = ω r̄ ξ/(α √(1 − 2m/r))` and `vⁱ = g(r̄²) xⁱ` with `g = ω ξ/(α √(1 − 2m/r))`, an
# even and smooth function of `r̄`. A mode's amplitude `a` is that of its oscillation in
# `ρ_c/ρ_c` (`ζ = −3ξ` at the centre, so `ξ = a/3` there); `a > 0` starts the centre
# expanding. The sum `g = Σₙ aₙ gₙ/3` is fitted by `TOV_NCHEB = 12` Chebyshev terms in
# `2r̄²/R̄² − 1` (`tov_mode_velocity`), which `TOVStar` stores as an `SVector`, so it stays
# `isbits`. The fit uses the nodes with `r̄² ≤ 0.99 R̄²`, since the last few nodes before the
# surface carry RK4's error of the stiff `1/η` term (`10⁻⁶` of `g` at 4000 steps), and
# extrapolates over the last 1%; inside it holds `g` to `10⁻⁷` for H1 and H2.

"""
    TOVModes

The star's radial modes ([`tov_radial_modes`](@ref)): `cowling`, whether the metric was
held fixed; `ω`, the angular frequencies in the coordinate time (`1/M☉`); `f_kHz`, the
frequencies in kHz; and for each mode, on `tov_solve`'s nodes in `η` (centre to the last
node before the surface), the isotropic radius `r̄`, the eigenfunction `ξ = Δr/r` (`1` at
the centre), `ζ = Δn/n`, and `g = ω ξ/(α √(1 − 2m/r))`, the Valencia velocity's `vⁱ/xⁱ`
per unit `ξ` at the centre.
"""
struct TOVModes
    cowling::Bool
    ω::Vector{Float64}
    f_kHz::Vector{Float64}
    r̄::Vector{Float64}
    ξ::Vector{Vector{Float64}}
    ζ::Vector{Vector{Float64}}
    g::Vector{Vector{Float64}}
end

# `d(ξ, ζ)/dη` and the surface bracket `G` at a background state `y = (q, μ, λ)`.
function pulsation_rhs(sol::TOVSolution{Float64,Polytrope{Float64}}, cowling::Bool, ω²::Float64, η::Float64,
                       y::SVector{3,Float64}, ξ::Float64, ζ::Float64)
    q, μ, _ = y
    eq = sol.eq
    Γ = eq.Γ
    h = exp(η)
    st = polytrope_state(eq, h)
    P, e = st.p, st.e
    q_η, μ_η, _ = tov_rhs(eq, η, q, μ, y[3])
    s² = 1 - 2 * μ * q
    eΛ = 1 / s²
    eΛν = eΛ * h^2 / sol.α_s^2                       # e^{Λ−ν}
    if !cowling
        B = ω² * eΛν * q_η / 2 - 4 + 2 * q / q_η - 4π * eΛ * P * q_η
        G = ξ * B / Γ - ζ
        ζ_η = (e + P) / P * G + ζ * (1 - 2π * (e + P) * eΛ * q_η)
        ξ_η = q == 0 ? -(ζ_η + 2 * ξ) / 5 : -q_η / (2 * q) * (3 * ξ + ζ) - ξ
    else
        Λ_η = 2 * (μ_η * q + μ * q_η) / s²
        if q == 0
            G = (ω² * eΛν * q_η / 2 * ξ + 2 * ξ + ζ) / Γ - ζ
            ζ_η = (e + P) / P * G + ζ
            ξ_η = -(ζ_η + Λ_η * ξ) / 5
        else
            ξ_η = -q_η / (2 * q) * (3 * ξ + ζ) - Λ_η / 2 * ξ
            ds² = -2 * (μ_η * q + μ * q_η)
            a = μ + 4π * P
            q_ηη = -2 * (ds² * a - s² * (μ_η + 4π * (e + P))) / a^2
            D = 2 * ξ + 2 * q * ξ_η / q_η - 2 * q * ξ * q_ηη / q_η^2   # d(2qξ/q_η)/dη
            G = (ω² * eΛν * q_η / 2 * ξ + D + ζ) / Γ - ζ
            ζ_η = (e + P) / P * G + ζ
        end
    end
    return SVector(ξ_η, ζ_η), G
end

# Shoot from the centre to the node before the surface; return the residual `G` there and,
# if `record`, the nodes' `(ξ, ζ)`.
function pulsation_shoot(sol::TOVSolution{Float64,Polytrope{Float64}}, cowling::Bool, ω²::Float64; record=false)
    η, Y = sol.η, sol.y
    n = length(η) - 1                                # the last node, η = Δ
    z = SVector(1.0, -3.0)
    zs = record ? [z] : SVector{2,Float64}[]
    f(i, ηᵢ, yᵢ, z) = pulsation_rhs(sol, cowling, ω², ηᵢ, yᵢ, z[1], z[2])[1]
    for i in 1:(n - 1)
        Δ = η[i + 1] - η[i]
        ym = tov_state(sol, i, η[i] + Δ / 2)
        k₁ = f(i, η[i], Y[i], z)
        k₂ = f(i, η[i] + Δ / 2, ym, z + Δ / 2 * k₁)
        k₃ = f(i, η[i] + Δ / 2, ym, z + Δ / 2 * k₂)
        k₄ = f(i, η[i + 1], Y[i + 1], z + Δ * k₃)
        z += Δ / 6 * (k₁ + 2 * k₂ + 2 * k₃ + k₄)
        record && push!(zs, z)
    end
    _, G = pulsation_rhs(sol, cowling, ω², η[n], Y[n], z[1], z[2])
    return G, zs
end

"""
    tov_radial_modes(sol::TOVSolution{Float64,Polytrope{Float64}}; cowling::Bool, nmodes = 3,
                     ω²_max = 0.1, nscan = 400) -> TOVModes

The star's lowest `nmodes` radial modes (F, H1, H2, …) from linear perturbation theory,
with the metric perturbed (`cowling = false`, the coupled star) or held fixed
(`cowling = true`): the roots in `(0, ω²_max]` of the shooting residual, bracketed on
`nscan` intervals and bisected (`src/tov_pulsation.jl`). Throws if fewer than `nmodes`
are found.
"""
function tov_radial_modes(sol::TOVSolution{Float64,Polytrope{Float64}}; cowling::Bool, nmodes::Integer=3,
                          ω²_max=0.1, nscan::Integer=400)
    res(ω²) = pulsation_shoot(sol, cowling, ω²)[1]
    grid = range(0, ω²_max; length=nscan + 1)[2:end]
    roots = Float64[]
    a, Ga = grid[1], res(grid[1])
    for b in grid[2:end]
        Gb = res(b)
        if sign(Ga) != sign(Gb)
            lo, hi, Glo = a, b, Ga
            for _ in 1:200
                m = (lo + hi) / 2
                (m == lo || m == hi) && break
                Gm = res(m)
                sign(Gm) == sign(Glo) ? ((lo, Glo) = (m, Gm)) : (hi = m)
            end
            push!(roots, (lo + hi) / 2)
            length(roots) == nmodes && break
        end
        a, Ga = b, Gb
    end
    length(roots) == nmodes || throw(ArgumentError(
        "found $(length(roots)) radial modes below ω² = $ω²_max, not $nmodes; raise ω²_max."))
    ω = sqrt.(roots)
    n = length(sol.η) - 1
    r̄ = [sqrt(tov_u(sol, sol.η[i], sol.y[i])[1]) for i in 1:n]
    ξs, ζs, gs = Vector{Float64}[], Vector{Float64}[], Vector{Float64}[]
    for (k, ω²) in enumerate(roots)
        _, zs = pulsation_shoot(sol, cowling, ω²; record=true)
        ξ = [z[1] for z in zs]
        g = [ω[k] * ξ[i] * exp(sol.η[i]) / sol.α_s / sqrt(1 - 2 * sol.y[i][2] * sol.y[i][1])
             for i in 1:n]
        push!(ξs, ξ)
        push!(ζs, [z[2] for z in zs])
        push!(gs, g)
    end
    return TOVModes(cowling, ω, ω ./ (2π * 1e3 * MSUN_TIME_CGS), r̄, ξs, ζs, gs)
end

# The number of Chebyshev terms of the perturbation's `vⁱ/xⁱ` (top of the file).
const TOV_NCHEB = 12

# `T₀(t), …, T_{K−1}(t)`.
function chebyshev_basis(K::Integer, t)
    b = zeros(typeof(t), K)
    b[1] = 1
    K > 1 && (b[2] = t)
    for k in 3:K
        b[k] = 2 * t * b[k - 1] - b[k - 2]
    end
    return b
end

# `Σₖ c[k] T_{k−1}(t)` by Clenshaw's recurrence.
@inline function chebyshev_sum(c::SVector{K,T}, t::T) where {K,T}
    b₁ = zero(T)
    b₂ = zero(T)
    for k in K:-1:2
        b₁, b₂ = 2 * t * b₁ - b₂ + c[k], b₁
    end
    return t * b₁ - b₂ + c[1]
end

"""
    tov_mode_velocity(sol::TOVSolution{Float64,Polytrope{Float64}}, δv; cowling::Bool) -> SVector{TOV_NCHEB}

The Chebyshev coefficients, in `2r̄²/R̄² − 1`, of `vⁱ/xⁱ` for the star moving in its radial
modes at the phase of zero displacement, mode `n` with amplitude `δv[n]` in `ρ_c/ρ_c`
(`src/tov_pulsation.jl`).
"""
function tov_mode_velocity(sol::TOVSolution{Float64,Polytrope{Float64}}, δv; cowling::Bool)
    m = tov_radial_modes(sol; cowling, nmodes=length(δv))
    u = (m.r̄ ./ sol.R̄) .^ 2
    keep = u .≤ 0.99
    g = sum(δv[n] / 3 .* m.g[n] for n in eachindex(δv))
    A = reduce(hcat, [chebyshev_basis(TOV_NCHEB, 2 * ui - 1) for ui in u[keep]])'
    return SVector{TOV_NCHEB,Float64}(A \ g[keep])
end
