# The bracketed path of the implicit stage solve: a search over the fluid's
# rapidity that does not depend on a seed ("The bracketed path" in `CODE.md`,
# added in step 3 after review).
#
# The iteration on the fluid's share (`stage_solve.jl`) converges from a seed
# within the fluid's own scale of the answer. Where the field's energy dwarfs
# the fluid's (`b²/ρ ≫ 1`) and the predictor departs from the answer by more
# than the fluid's energy (`f b²/ρ ≳ 10`, `f = |E★ + v×B|/|B|`), neither seed
# — the predictor's recovery nor the ideal one — is a fluid state, and the
# iteration has nowhere to start. This path finds the answer instead as a root
# of one function of one variable, the fluid's rapidity `w`.
#
# **Given `w`**, everything but the thermodynamics is algebra. With
# `W = cosh w`, the density `ρ = D/W`, the conductivity `σ(ρ)` and `c = γΔt α σ`
# are known. Ohm's law at the velocity `u = W v` is (`ohm_E`)
#
#     E = [E★ − c u × B + κ (u·E★) u] / (1 + cW),   κ = (c/W)/(1 + c/W),
#
# and the momentum `S_i = z v_i + ε_ijk Eʲ Bᵏ`, with `z v = D h u`, splits
# along and across `B`. Along `B` the field carries no momentum, so
# `D h u_∥ = S_∥`. Across it, `ε (u × B) × B = −B² u_⊥`, and
#
#     A u_⊥ + k (u_⊥ × B) = r,    A = D h + cB²/(1 + cW),   k = κ ζ / (1 + cW),
#     r = (S − S_∥ b̂ − ε E★ × B / (1 + cW))^♯,              ζ = u·E★,
#
# a rotation-dilation in the plane across `B` (`(u × B) × B = −B² u` there),
# inverted in closed form: `u_⊥ = (A r − k r × B)/(A² + k² B²)`. The one
# scalar it needs, `ζ = u·E★`, satisfies a cubic (`momentum_velocity`), whose
# real root nearest the unrotated `ζ∥ + (r·E★)/A` is taken. That gives `u(h)`
# for a trial specific enthalpy `h`, and the `h` with `|u(h)| = sinh w` — the
# speed equation — is a root in a band of `log h` bounded in closed form
# (`inertia_at_rapidity`). It can have several roots; they are counted from
# the largest `h` (the most inertia), and the path is run on the largest and
# then on the second.
#
# **The energy then fixes the thermodynamics.** The field leaves the fluid
# `τ_f = τ − ½(E² + B²)`; with `z = D h W` the fluid needs the pressure
# `p = z − D − τ_f` and the specific internal energy `U = h − 1 − p/ρ`. The
# residual is `φ(w) = p_EOS(ρ, U) − p`: the EOS's pressure at that density
# and energy against the one the budgets require. It is evaluated without the
# policy's clamps, the EOS continued linearly in `U` beyond its extended
# entropy window, so that `φ` is smooth and its roots are only the stage's
# solutions.
#
# **What this calls, and why it is not a second primitive recovery**
# (proposed in step 3). `p_EOS(ρ, U)` inverts `U(ρ, s)` for `s` at fixed `ρ`
# (`pressure_at_energy`): EntropyEOS's `entropy_from_energy`, requested as
# EOS-3, written here as EOS-3's workaround from the forward
# `EntropyEOS.evaluate` and `srange_extended` — a monotone one-dimensional
# solve (`U_s = T > 0`) of the equation of state, with no policy, no repair
# and no state returned to the scheme. The root's field is handed to the
# iteration, whose recoveries are `con2prim_safe`'s: the returned primitive
# state, its policy and its flags are EntropyEOS's, as for every other
# branch. The path does solve the recovery's equations together with Ohm's
# law, to find a seed, and whether that is within "One con2prim" in
# `CLAUDE.md` is Erik's call.
#
# **`φ` has roots close together.** The fluid's admissible states occupy a
# needle in `w` — a velocity error moves the field's energy by `b²/ρ` times
# the fluid's — and inside it `φ` crosses zero at the solution and often
# again nearby (measured: two roots within 1e-3 of each other in `w`, and
# likewise in the speed equation's `h`). So the search scans `w` from 0 to the
# policy's rapidity cap and refines every sign change, and every local
# extremum of `φ` that does not reach zero on the grid but may between its
# points (a golden-section search finds it); cells around the window's
# needle and around roots that did not converge are scanned again, finer.
# The lowest root from which the iteration, seeded there, converges is the
# answer (proposed in step 3).

"""
    OhmMomentum{T}

The parts of the momentum balance across and along `B` that do not depend
on the fluid's specific enthalpy, at a fixed Lorentz factor (see the head of
`src/stage_bracket.jl`): the unit `b̂ = B/|B|`, `S_∥`, `r`, `r × B`, the
field's inertia `cB²/(1 + cW)`, the rotation's coefficient `κ/(1 + cW)`, and
the dot products of `b̂`, `r` and `r × B` with `E★`.
"""
struct OhmMomentum{T}
    b̂::SVector{3,T}
    S_par::T
    r::SVector{3,T}
    rB::SVector{3,T}
    a₀::T
    κ̃::T
    E_par::T
    a₁::T
    a₂::T
    B²::T
end

@inline function ohm_momentum(m::Metric3{T}, S::SVector{3,T}, E★::SVector{3,T},
                              B::SVector{3,T}, c::T, W::T) where {T}
    B² = dot_up(m, B, B)
    b̂ = B² > zero(T) ? B / sqrt(B²) : zero(SVector{3,T})
    S_par = sum(S .* b̂)
    cW = c * W
    cw = c / W
    κ̃ = (cw / (one(T) + cw)) / (one(T) + cW)
    q = cross_down(m, E★, B) / (one(T) + cW)
    r = raise(m, S - S_par * lower(m, b̂) - q)
    rB = raise(m, cross_down(m, r, B))
    return OhmMomentum{T}(b̂, S_par, r, rB, c * B² / (one(T) + cW), κ̃, dot_up(m, b̂, E★),
                          dot_up(m, r, E★), dot_up(m, rB, E★), B²)
end

"""
    momentum_velocity(P::OhmMomentum, D, h) -> u

The velocity `u = W v` at which the momentum balance and Ohm's law hold for
the specific enthalpy `h`: `u_∥ = S_∥/(D h)` along `B`, and across it the
rotation-dilation `(A + k ×B) u_⊥ = r` inverted in closed form, with
`ζ = u·E★` a root of the cubic
`(ζ − ζ_∥)(A² + κ̃² B² ζ²) − A (r·E★) + κ̃ ζ ((r × B)·E★) = 0` — the real root
nearest the unrotated `ζ₀ = ζ_∥ + (r·E★)/A` ([`cubic_root_near`](@ref)).
"""
function momentum_velocity(P::OhmMomentum{T}, D::T, h::T) where {T}
    u_par = P.S_par / (D * h)
    A = D * h + P.a₀
    ζ_par = u_par * P.E_par
    ζ₀ = ζ_par + P.a₁ / A
    κ²B² = P.κ̃ * P.κ̃ * P.B²
    A² = A * A
    # κ²B² ζ³ − κ²B² ζ_∥ ζ² + (A² + κ̃ a₂) ζ − (A² ζ_∥ + A a₁) = 0
    ζ = cubic_root_near(κ²B², -κ²B² * ζ_par, A² + P.κ̃ * P.a₂, -(A² * ζ_par + A * P.a₁), ζ₀)
    k = P.κ̃ * ζ
    return u_par * P.b̂ + (A * P.r - k * P.rB) / (A² + k * k * P.B²)
end

"""
    cubic_root_near(c₃, c₂, c₁, c₀, x₀) -> x

The real root of `c₃x³ + c₂x² + c₁x + c₀` nearest `x₀`: Newton from `x₀`
(bisection on the Cauchy bound if it does not converge), the other two
roots from the deflated quadratic, each polished by Newton, and the nearest
of the real ones. Linear where `c₃ = 0`.
"""
function cubic_root_near(c₃::T, c₂::T, c₁::T, c₀::T, x₀::T) where {T}
    iszero(c₃) && return -c₀ / c₁
    b, c, d = c₂ / c₃, c₁ / c₃, c₀ / c₃
    P(x) = ((x + b) * x + c) * x + d
    dP(x) = (3 * x + 2 * b) * x + c
    function polish(x)
        for _ in 1:20
            δ = P(x) / dP(x)
            isfinite(δ) || break
            x -= δ
            abs(δ) <= 4 * eps(T) * (one(T) + abs(x)) && break
        end
        return x
    end
    # One root: Newton from `x₀`, or bisection on `|x| ≤ 1 + max(|b|, |c|, |d|)`.
    x = x₀
    ok = false
    for _ in 1:40
        δ = P(x) / dP(x)
        isfinite(δ) || break
        x -= δ
        if abs(δ) <= 4 * eps(T) * (one(T) + abs(x))
            ok = true
            break
        end
    end
    if !ok
        R = one(T) + max(abs(b), abs(c), abs(d))
        lo, hi = -R, R                         # P(lo) < 0 < P(hi)
        for _ in 1:200
            x = (lo + hi) / 2
            P(x) < zero(T) ? (lo = x) : (hi = x)
            hi - lo <= 4 * eps(T) * (one(T) + abs(x)) && break
        end
        x = polish((lo + hi) / 2)
    end
    # The others: x² + p x + q from the deflation.
    p = b + x
    q = c + x * p
    disc = p * p - 4 * q
    best = x
    if disc >= zero(T)
        t = -(p + copysign(sqrt(disc), p)) / 2
        for y in (t, iszero(t) ? zero(T) : q / t)
            y = polish(y)
            abs(y - x₀) < abs(best - x₀) && (best = y)
        end
    end
    return best
end

# Illinois steps on `[a, b]` with `F(a)`, `F(b)` of opposite signs: the new
# interior point, and the updated bracket and the side last kept.
@inline function illinois(a::T, Fa::T, b::T, Fb::T, x::T, Fx::T, side::Int) where {T}
    if (Fx < zero(T)) == (Fa < zero(T))
        a, Fa = x, Fx
        side == -1 && (Fb /= 2)
        side = -1
    else
        b, Fb = x, Fx
        side == 1 && (Fa /= 2)
        side = 1
    end
    return a, Fa, b, Fb, side
end

@inline function illinois_point(a::T, Fa::T, b::T, Fb::T) where {T}
    x = (a * Fb - b * Fa) / (Fb - Fa)
    return (isfinite(x) && min(a, b) < x < max(a, b)) ? x : (a + b) / 2
end

"""
    inertia_at_rapidity(m::Metric3, P::OhmMomentum, D, sw, Emag, branch=0; nh=24) -> h

A specific enthalpy `h` for which `|momentum_velocity(P, D, h)| = sinh w = sw`:
the `branch`-th root of this speed equation counted from the largest `h`
(NaN if there is none, except for `branch = 0`, which falls back on the
band's bottom). Across `B`, `|u_⊥|² = r²/(A² + k² B²)` with `A = D h + a₀`
and `|k| ≤ κ̃ sw |E★|` (`Emag = |E★|`), so the roots lie between the `h`
where `S_∥²/(Dh)² + r²/A²` and where
`S_∥²/(Dh)² + r²/(A² + κ̃² sw² E★² B²)` equal `sw²`, both monotone in `h` and
found by bisection. That band is scanned downward in `nh` geometric steps;
every sign change is a root, and a local extremum of one sign whose
golden-section refinement crosses zero is two — the speed equation, too,
has roots closer together than any fixed grid (proposed in step 3).
"""
function inertia_at_rapidity(m::Metric3{T}, P::OhmMomentum{T}, D::T, sw::T, Emag::T,
                                     branch::Int=0; nh::Int=24) where {T}
    sw² = sw * sw
    Sp² = P.S_par * P.S_par
    r² = dot_up(m, P.r, P.r)
    kB² = (P.κ̃ * sw * Emag)^2 * P.B²
    x_hi = band_root(D, P.a₀, Sp², r², sw², zero(T))
    isfinite(x_hi) || return branch == 0 ? floatmin(T) : T(NaN)
    x_lo = band_root(D, P.a₀, Sp², r², sw², kB²)
    isfinite(x_lo) || (x_lo = x_hi - T(40))
    x_lo = min(x_lo, x_hi) - T(1 // 1000)          # a little room below
    x_hi = x_hi + T(1 // 1000)

    # Downward from the top: every sign change is a root, and a local
    # extremum of one sign that crosses zero between the points (golden
    # section finds it) is two; the `branch`-th, counted from the largest.
    # Above the band `|u| < sinh w`: `F₂ = −∞` there lets the first cell
    # hold a local maximum too.
    k = 0
    x₂, F₂ = x_hi, T(-Inf)
    x₁, F₁ = x_hi, speed_excess(m, P, D, sw², x_hi)
    F₁ > zero(T) && branch == 0 && return exp(x₁)
    for i in 1:nh
        x₀ = x_hi - (x_hi - x_lo) * i / nh
        F₀ = speed_excess(m, P, D, sw², x₀)
        if (F₀ > zero(T)) != (F₁ > zero(T))
            k == branch && return exp(speed_root(m, P, D, sw², x₀, F₀, x₁, F₁))
            k += 1
        elseif !isnan(F₂) && (F₂ > zero(T)) == (F₁ > zero(T)) == (F₀ > zero(T))
            sgn = F₁ > zero(T) ? one(T) : -one(T)
            if sgn * F₁ < sgn * F₂ && sgn * F₁ < sgn * F₀
                xm, Fm = speed_extremum(m, P, D, sw², x₀, x₂, sgn)
                if (Fm > zero(T)) != (F₁ > zero(T))
                    k == branch &&
                        return exp(speed_root(m, P, D, sw², xm, Fm, x₂, isfinite(F₂) ? F₂ : F₁))
                    k + 1 == branch && return exp(speed_root(m, P, D, sw², x₀, F₀, xm, Fm))
                    k += 2
                end
            end
        end
        x₂, F₂ = x₁, F₁
        x₁, F₁ = x₀, F₀
    end
    return branch == 0 ? exp(x_lo) : T(NaN)
end

# A root of `speed_excess` in `[a, b]`, by Illinois to `eps^(3/4)` in `log h`:
# the root this `h` leads to is polished by the iteration, which needs it
# only within the fluid's scale.
function speed_root(m::Metric3{T}, P::OhmMomentum{T}, D::T, sw²::T, a::T, Fa::T, b::T,
                            Fb::T) where {T}
    side = 0
    for _ in 1:60
        x = illinois_point(a, Fa, b, Fb)
        Fx = speed_excess(m, P, D, sw², x)
        Fx == zero(T) && return x
        a, Fa, b, Fb, side = illinois(a, Fa, b, Fb, x, Fx, side)
        abs(b - a) <= eps(T)^(3 // 4) * (one(T) + abs(x)) && break
    end
    return (a + b) / 2
end

# Golden section for the extremum of `speed_excess` on `[a, c]` (`sgn = −1`:
# a maximum below zero; `+1`: a minimum above it) — the least `sgn · F` —
# stopping once it crosses zero.
function speed_extremum(m::Metric3{T}, P::OhmMomentum{T}, D::T, sw²::T, a::T, c::T,
                                sgn::T) where {T}
    g = (sqrt(T(5)) - one(T)) / 2
    x1 = c - g * (c - a)
    x2 = a + g * (c - a)
    f1 = sgn * speed_excess(m, P, D, sw², x1)
    f2 = sgn * speed_excess(m, P, D, sw², x2)
    for _ in 1:40
        (f1 < zero(T) || f2 < zero(T)) && break
        c - a <= eps(T)^(3 // 4) * (one(T) + abs(c)) && break
        if f1 < f2
            c, x2, f2 = x2, x1, f1
            x1 = c - g * (c - a)
            f1 = sgn * speed_excess(m, P, D, sw², x1)
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (c - a)
            f2 = sgn * speed_excess(m, P, D, sw², x2)
        end
    end
    return f1 < f2 ? (x1, sgn * f1) : (x2, sgn * f2)
end

# `|u(h)|² − sinh² w` at `x = log h`.
@inline speed_excess(m::Metric3{T}, P::OhmMomentum{T}, D::T, sw²::T, x::T) where {T} =
    (u = momentum_velocity(P, D, exp(x)); dot_up(m, u, u) - sw²)

# The bound on `|u|² − sinh² w` at `x = log h` with the rotation's `k² B²`
# at most `kB²`: decreasing in `x`.
@inline function band_excess(D::T, a₀::T, Sp²::T, r²::T, sw²::T, kB²::T, x::T) where {T}
    Dh = D * exp(x)
    A = Dh + a₀
    return Sp² / (Dh * Dh) + r² / (A * A + kB²) - sw²
end

# Its root in `x`, by bisection from above the closed-form bound
# `√(S_∥² + r²)/(D sinh w)`; `-Inf` if there is none.
function band_root(D::T, a₀::T, Sp²::T, r²::T, sw²::T, kB²::T) where {T}
    hmax = sqrt((Sp² + r²) / sw²) / D
    (hmax > zero(T) && isfinite(hmax)) || return T(-Inf)
    b = log(hmax)
    a = b - one(T)
    k = 0
    while band_excess(D, a₀, Sp², r², sw², kB², a) <= zero(T)
        k += 1
        k > 200 && return T(-Inf)
        a -= T(2)^min(k, 8)
    end
    # To a ten-thousandth in `log h`: the band is widened by ten times that.
    for _ in 1:200
        x = (a + b) / 2
        band_excess(D, a₀, Sp², r², sw², kB², x) > zero(T) ? (a = x) : (b = x)
        b - a <= T(1 // 10000) && break
    end
    return (a + b) / 2
end

"""
    pressure_at_energy(eos, ρ, yₑ, U, s_guess) -> (p, s, side)

The EOS's pressure at the density `ρ★ = ρ` and the specific internal energy
`U`, with the entropy `s` it found: `U(ρ, s) = U` inverted for `s` by
safeguarded Newton on the EOS's extended window
(`EntropyEOS.srange_extended`), where `U_s = T > 0`; beyond the window, the
EOS continued linearly in `U` from its edge, `p + (U − U_edge) p_s/U_s`, so
that the result is smooth and never clamped; `side` is −1 or +1 there (and
`s` the edge), 0 inside.

This is EntropyEOS's `entropy_from_energy`, requested as **EOS-3**, written
here from forward `EntropyEOS.evaluate` calls as EOS-3's workaround (amended
in step 3): when EOS-3 lands, it is called instead.
"""
function pressure_at_energy(eos::EntropyEOS.AbstractEOS, ρ::T, yₑ::T, U::T,
                                    s_guess::T) where {T}
    sr = EntropyEOS.srange_extended(eos, ρ, yₑ)
    lo, hi = T(sr.s_min), T(sr.s_max)
    plo = EntropyEOS.evaluate(eos, ρ, lo, yₑ, T(NaN))
    U <= plo.U && return plo.p + (U - plo.U) * (ρ * ρ * plo.U_ρs / plo.U_s), lo, -1
    phi = EntropyEOS.evaluate(eos, ρ, hi, yₑ, T(NaN))
    U >= phi.U && return phi.p + (U - phi.U) * (ρ * ρ * phi.U_ρs / phi.U_s), hi, 1
    s = (isfinite(s_guess) && lo < s_guess < hi) ? s_guess : (lo + hi) / 2
    u_warm = T(NaN)
    p = plo.p
    for _ in 1:60
        pt = EntropyEOS.evaluate(eos, ρ, s, yₑ, u_warm)
        u_warm = pt.u_solved
        p = pt.p
        F = pt.U - U
        F == zero(T) && break
        F > zero(T) ? (hi = s) : (lo = s)
        sn = s - F / pt.U_s
        (isfinite(sn) && lo < sn < hi) || (sn = (lo + hi) / 2)
        done = abs(sn - s) <= 4 * eps(T) * (one(T) + abs(s))
        s = sn
        done && (p = EntropyEOS.evaluate(eos, ρ, s, yₑ, u_warm).p; break)
    end
    return p, s, 0
end

"""
    rapidity_residual(eos, model, m, D, yₑ, S, τ, E★, B, αΔt, w, prim_ref, scale, s_guess,
                      branch=0) -> (φ, E, s, side)

The bracketed path's function of the fluid's rapidity `w > 0`: the EOS's
pressure at the density and energy the budgets require, less the pressure
they require, relative to `scale` (see the head of `src/stage_bracket.jl`);
with the field `E` at `w`, the entropy found, and which side of the EOS's
extended window the energy fell (−1, +1; 0 inside). `prim_ref` stands in for
the fluid where the conductivity model reads more than `ρ = D/cosh w` (the
models here read only `ρ`). `branch` picks the root of the speed equation
([`inertia_at_rapidity`](@ref)), counted from the largest `h`; where there is
no such root, `φ` is NaN.
"""
function rapidity_residual(eos::EntropyEOS.AbstractEOS, model::Conductivity,
                                   m::Metric3{T}, D::T, yₑ::T, S::SVector{3,T}, τ::T,
                                   E★::SVector{3,T}, B::SVector{3,T}, αΔt::T, w::T,
                                   prim_ref::NTuple{NPRIM,T}, scale::T, s_guess::T,
                                   branch::Int=0) where {T}
    W = cosh(w)
    sw = sinh(w)
    ρ = D / W
    c = αΔt * conductivity(model, at_density(prim_ref, ρ))
    P = ohm_momentum(m, S, E★, B, c, W)
    h = inertia_at_rapidity(m, P, D, sw, sqrt(dot_up(m, E★, E★)), branch)
    isfinite(h) || return T(NaN), E★, T(NaN), 0     # no such branch at this `w`
    u = momentum_velocity(P, D, h)
    E = ohm_E(m, c, u, E★, B)
    τ_f = τ - (dot_up(m, E, E) + dot_up(m, B, B)) / 2
    p_req = D * h * W - D - τ_f
    U_req = h - one(T) - p_req / ρ
    p, s, side = pressure_at_energy(eos, ρ, yₑ, U_req, s_guess)
    return (p - p_req) / scale, E, s, side
end

# A primitive state with its density replaced, for the conductivity.
@inline at_density(prim::NTuple{NPRIM,T}, ρ::T) where {T} =
    ntuple(i -> i == P_RHO ? ρ : prim[i], Val(NPRIM))

# The policy's rapidity cap: `PolicyOptions.w_cap`, or EntropyEOS's default
# `acosh(100)` where it is unset (zero or non-finite), as `con2prim_safe` reads it.
@inline rapidity_cap(pol::EntropyEOS.PolicyOptions{T}) where {T} =
    (isfinite(pol.w_cap) && pol.w_cap > zero(T)) ? pol.w_cap : acosh(T(100))

# Everything the pieces of the bracketed path need besides `w`, so that they
# are functions and not closures (a closure that assigns a name its
# enclosing function also uses boxes it).
struct BracketContext{E,O,P,C,T}
    eos::E
    opts::O
    pol::P
    model::C
    m::Metric3{T}
    D::T
    yₑ::T
    DYₑ::T
    S::SVector{3,T}
    τ::T
    E★::SVector{3,T}
    B::SVector{3,T}
    αΔt::T
    prim_ref::NTuple{NPRIM,T}
    scale::T
    s_guess::T
    stage::OhmStageOptions{T}
    branch::Int
end

"""
    bracketed_stage(eos, opts, pol, model, m, D, S, τ, DYₑ, E★, B, αΔt, scale, prim_ref, stage)
        -> NamedTuple

The bracketed path: the fluid at rest, if the momentum allows it, then a scan
of [`rapidity_residual`](@ref) over `stage.nscan` points from 0 to the
policy's rapidity cap, refining each sign change by Illinois and each local
extremum short of zero by golden section; from each root found, in order of
increasing `w`, the iteration of `share_iteration`, seeded with the root's
field and the recovery of the share it leaves, for at most `stage.polish`
recoveries. Returns `done`, the field and the recovery, the recoveries made,
the modifier bits, and the number of `rapidity_residual` evaluations.
"""
function bracketed_stage(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                 pol::EntropyEOS.PolicyOptions{T}, model::Conductivity,
                                 m::Metric3{T}, D::T, S::SVector{3,T}, τ::T, DYₑ::T,
                                 E★::SVector{3,T}, B::SVector{3,T}, αΔt::T, scale::T,
                                 prim_ref::NTuple{NPRIM,T}, stage::OhmStageOptions{T}) where {T}
    cx = BracketContext(eos, opts, pol, model, m, D, DYₑ / D, DYₑ, S, τ, E★, B, αΔt, prim_ref,
                        scale, prim_s(prim_ref), stage, 0)

    # At rest: `u = 0` needs the field to carry all the momentum.
    c0 = αΔt * conductivity(model, at_density(prim_ref, D))
    E0 = ohm_E(m, c0, zero(SVector{3,T}), E★, B)
    R0 = S - cross_down(m, E0, B)
    nrec = 0
    modifiers = UInt32(0)
    if sqrt(dot_down(m, R0, R0)) <= sqrt(stage.tol) * scale
        it0 = polish_from(cx, E0)
        nrec += 1 + it0.nrec
        modifiers |= it0.modifiers
        it0.done && return (; done=true, E=it0.E, prim=it0.prim, f=it0.f, nrec, modifiers,
                            neval=0)
    end

    # The scan, from one cell in to the cap: `w = 0` itself is the case above.
    # On the speed equation's largest root first, then on its second.
    w_hi = rapidity_cap(pol)
    n = Int(stage.nscan)
    r = scan_interval(cx, w_hi / n, w_hi, n - 1, true, Int(stage.max_eval))
    nrec += r.nrec
    modifiers |= r.modifiers
    neval = r.neval
    r.done && return (; done=true, E=r.E, prim=r.prim, f=r.f, nrec, modifiers, neval)
    cx1 = BracketContext(eos, opts, pol, model, m, D, DYₑ / D, DYₑ, S, τ, E★, B, αΔt, prim_ref,
                         scale, prim_s(prim_ref), stage, 1)
    r = scan_interval(cx1, w_hi / n, w_hi, n - 1, true, Int(stage.max_eval) - neval)
    return (; done=r.done, E=r.E, prim=r.prim, f=r.f, nrec=nrec + r.nrec,
            modifiers=modifiers | r.modifiers, neval=neval + r.neval)
end

"""
    scan_interval(cx, a, b, n, refine, budget) -> NamedTuple

The scan of the rapidity residual over `n` cells of `[a, b]`: in each cell,
a bracket where the residual changes sign, or two where three points show
a local extremum of one sign whose golden-section refinement crosses zero;
each bracket's root, if admissible, polished by the iteration, the first
that converges ending the scan. With `refine`, a cell across which the
energy passes through the EOS's extended window — where the fluid's
admissible states are, a needle in `w` — or which held a root that gave no
answer, and the cell after such a cell, are scanned again in
`stage.nscan ÷ 2` sub-cells (one level, without `refine`). At most `budget`
evaluations of the residual, roughly: a cell under way is finished.
"""
function scan_interval(cx::BracketContext{E,O,P,C,T}, a::T, b::T, n::Int,
                               refine::Bool, budget::Int) where {E,O,P,C,T}
    nrec = 0
    neval = 0
    modifiers = UInt32(0)
    w₂, F₂ = a, T(NaN)                             # two points back
    w₁ = a
    F₁, _, _, side₁ = rapidity_residual(cx, w₁)
    neval += 1
    near = false
    for i in 1:n
        neval >= budget && break
        w₀ = a + (b - a) * i / n
        F₀, _, _, side₀ = rapidity_residual(cx, w₀)
        neval += 1
        # Up to two brackets from this cell: a sign change, or the two sides
        # of a local extremum of one sign that crosses zero between the
        # grid's points — two roots close together.
        a₁, Fa₁, b₁, Fb₁ = w₁, F₁, w₀, F₀
        a₂, Fa₂, b₂, Fb₂ = w₁, F₁, w₀, F₀
        nbr = 0
        if isfinite(F₀) && isfinite(F₁) && (F₀ < zero(T)) != (F₁ < zero(T))
            nbr = 1
        elseif isfinite(F₂) && (F₂ < zero(T)) == (F₁ < zero(T)) == (F₀ < zero(T))
            sgn = F₁ < zero(T) ? -one(T) : one(T)
            if sgn * F₁ < sgn * F₂ && sgn * F₁ < sgn * F₀
                x, Fx, ne = extremum(cx, w₂, w₀, sgn)
                neval += ne
                if (Fx < zero(T)) != (F₁ < zero(T))
                    nbr = 2
                    a₁, Fa₁, b₁, Fb₁ = w₂, F₂, x, Fx
                    a₂, Fa₂, b₂, Fb₂ = x, Fx, w₀, F₀
                end
            end
        end
        for j in 1:nbr
            aj, Faj, bj, Fbj = j == 1 ? (a₁, Fa₁, b₁, Fb₁) : (a₂, Fa₂, b₂, Fb₂)
            Ej, admissible, ne = root_in(cx, aj, Faj, bj, Fbj)
            neval += ne
            admissible || continue
            itr = polish_from(cx, Ej)
            nrec += 1 + itr.nrec
            modifiers |= itr.modifiers
            itr.done && return (; done=true, E=itr.E, prim=itr.prim, f=itr.f, nrec, modifiers,
                                neval)
        end
        # Roots cluster where the fluid's admissible states are: a cell the
        # energy crosses the EOS's window in, one that held a root that did
        # not converge, and the one after such a cell are scanned again.
        if refine && (!(side₀ == side₁ != 0) || nbr > 0 || near)
            r = scan_fine(cx, w₁, w₀, max(Int(cx.stage.nscan) ÷ 2, 2), budget - neval)
            nrec += r.nrec
            neval += r.neval
            modifiers |= r.modifiers
            r.done && return (; done=true, E=r.E, prim=r.prim, f=r.f, nrec, modifiers, neval)
        end
        near = nbr > 0
        w₂, F₂ = w₁, F₁
        w₁, F₁, side₁ = w₀, F₀, side₀
    end
    return (; done=false, E=cx.E★, prim=cx.prim_ref, f=UInt32(0), nrec, modifiers, neval)
end

# `scan_interval` without refinement, within `budget` evaluations: a separate
# function, so that neither is recursive and both infer.
function scan_fine(cx::BracketContext{E,O,P,C,T}, a::T, b::T, n::Int,
                           budget::Int) where {E,O,P,C,T}
    nrec = 0
    neval = 0
    modifiers = UInt32(0)
    w₂, F₂ = a, T(NaN)
    w₁ = a
    F₁ = rapidity_residual(cx, w₁)[1]
    neval += 1
    for i in 1:n
        neval >= budget && break
        w₀ = a + (b - a) * i / n
        F₀ = rapidity_residual(cx, w₀)[1]
        neval += 1
        a₁, Fa₁, b₁, Fb₁ = w₁, F₁, w₀, F₀
        a₂, Fa₂, b₂, Fb₂ = w₁, F₁, w₀, F₀
        nbr = 0
        if isfinite(F₀) && isfinite(F₁) && (F₀ < zero(T)) != (F₁ < zero(T))
            nbr = 1
        elseif isfinite(F₂) && (F₂ < zero(T)) == (F₁ < zero(T)) == (F₀ < zero(T))
            sgn = F₁ < zero(T) ? -one(T) : one(T)
            if sgn * F₁ < sgn * F₂ && sgn * F₁ < sgn * F₀
                x, Fx, ne = extremum(cx, w₂, w₀, sgn)
                neval += ne
                if (Fx < zero(T)) != (F₁ < zero(T))
                    nbr = 2
                    a₁, Fa₁, b₁, Fb₁ = w₂, F₂, x, Fx
                    a₂, Fa₂, b₂, Fb₂ = x, Fx, w₀, F₀
                end
            end
        end
        for j in 1:nbr
            aj, Faj, bj, Fbj = j == 1 ? (a₁, Fa₁, b₁, Fb₁) : (a₂, Fa₂, b₂, Fb₂)
            Ej, admissible, ne = root_in(cx, aj, Faj, bj, Fbj)
            neval += ne
            admissible || continue
            itr = polish_from(cx, Ej)
            nrec += 1 + itr.nrec
            modifiers |= itr.modifiers
            itr.done && return (; done=true, E=itr.E, prim=itr.prim, f=itr.f, nrec, modifiers,
                                neval)
        end
        w₂, F₂ = w₁, F₁
        w₁, F₁ = w₀, F₀
    end
    return (; done=false, E=cx.E★, prim=cx.prim_ref, f=UInt32(0), nrec, modifiers, neval)
end

@inline rapidity_residual(cx::BracketContext, w) =
    rapidity_residual(cx.eos, cx.model, cx.m, cx.D, cx.yₑ, cx.S, cx.τ, cx.E★, cx.B, cx.αΔt, w,
                      cx.prim_ref, cx.scale, cx.s_guess, cx.branch)

# The iteration, seeded with the field `E` and the recovery of its share.
function polish_from(cx::BracketContext{E,O,P,C,T}, Ew::SVector{3,T}) where {E,O,P,C,T}
    S_f, τ_f = field_budget(cx.m, cx.S, cx.τ, Ew, cx.B)
    prim, f, pt = fluid_recovery(cx.eos, cx.opts, cx.pol, cx.m, cx.D, S_f, τ_f, cx.DYₑ,
                                 guesses_of(cx.prim_ref))
    f == 0 || ((S_f, τ_f) = fluid_share(cx.eos, cx.m, prim))
    σ = conductivity(cx.model, prim)
    return share_iteration(cx.eos, cx.opts, cx.pol, cx.model, cx.m, cx.D, cx.S, cx.τ, cx.DYₑ,
                           cx.E★, cx.B, cx.αΔt, cx.scale, S_f, τ_f, prim, f, pt, σ, true,
                           cx.stage.polish, cx.stage)
end

# A root of the rapidity residual in `[a, b]` (either order), the residual of
# opposite signs at the two: by Illinois, its field, whether it is
# admissible, and the evaluations it took. A root whose entropy lies outside
# the EOS's extended window — where `pressure_at_energy` continued the EOS
# linearly — is no fluid state at all, and is skipped without a recovery. (The
# policy's narrower window is not the test: the iteration from a root just
# outside it can still reach the solution.)
function root_in(cx::BracketContext{E,O,P,C,T}, a::T, Fa::T, b::T,
                         Fb::T) where {E,O,P,C,T}
    side = 0
    ne = 0
    for _ in 1:200
        x = illinois_point(a, Fa, b, Fb)
        Fx = rapidity_residual(cx, x)[1]
        ne += 1
        if Fx == zero(T)
            a, b = x, x
            break
        end
        a, Fa, b, Fb, side = illinois(a, Fa, b, Fb, x, Fx, side)
        abs(b - a) <= 4 * eps(T) * (one(T) + abs(x)) && break
    end
    _, Ew, _, wside = rapidity_residual(cx, (a + b) / 2)
    ne += 1
    return Ew, wside == 0, ne
end

# The point of `[a, c]` where `sgn · F` is least, by golden section — stopping
# early once `F` has crossed zero; with the evaluations it took. `sgn = −1`
# looks for a maximum below zero, `+1` for a minimum above it.
function extremum(cx::BracketContext{E,O,P,C,T}, a::T, c::T, sgn::T) where {E,O,P,C,T}
    g = (sqrt(T(5)) - one(T)) / 2
    x1 = c - g * (c - a)
    x2 = a + g * (c - a)
    f1 = sgn * rapidity_residual(cx, x1)[1]
    f2 = sgn * rapidity_residual(cx, x2)[1]
    ne = 2
    for _ in 1:60
        (f1 < zero(T) || f2 < zero(T)) && break       # crossed zero
        c - a <= 4 * eps(T) * (one(T) + abs(c)) && break
        if f1 < f2
            c, x2, f2 = x2, x1, f1
            x1 = c - g * (c - a)
            f1 = sgn * rapidity_residual(cx, x1)[1]
        else
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (c - a)
            f2 = sgn * rapidity_residual(cx, x2)[1]
        end
        ne += 1
    end
    return f1 < f2 ? (x1, sgn * f1, ne) : (x2, sgn * f2, ne)
end
