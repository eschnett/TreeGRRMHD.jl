# The approximate Riemann solvers: a pair of face states to the flux of the
# ten cell variables through the face.
#
# Both solvers use the local light cone `λ± = −βᵈ ± α √γ^{dd}` as the wave
# speeds of every variable (proposed in `CODE.md`, "Riemann solver"): the
# totals `S_i` and `τ` carry the Maxwell stress, which propagates at the
# speed of light, so no slower estimate bounds their signals. Both states sit
# at the same face, in the same metric, so the speeds are the same on both
# sides and depend on the metric alone — and the solvers need no EOS. Split
# speeds and a resistive HLLC are (open).
#
# A face state is the tuple `(cons, prim, E, B)`: the densitized evolved
# state, the primitive state it was built from (`v` and `p` are read), and
# the contravariant fields, as [`physical_flux`](@ref) takes them
# (proposed in step 2). The reconstruction of step 5 builds `prim`, `E` and
# `B` at the face and `cons` from them with `prim2con`.
#
# The solver is a `Val` of a symbol, as in TreeHydro, so a flux kernel
# specializes on it.

"""
    riemann_flux(::Val{:llf},  m::Metric3, L, R, ::Val{d}) -> NTuple{10}
    riemann_flux(::Val{:hlle}, m::Metric3, L, R, ::Val{d}) -> NTuple{10}

The numerical flux in direction `d` through a face with the states
`L = (cons_L, prim_L, E_L, B_L)` on its left and `R` on its right, in the
face metric `m`, in the order of `U_D … U_E`.

**`:llf`** (local Lax–Friedrichs):
`F = (F_L + F_R)/2 − λ (U_R − U_L)/2`, `λ = max(|λ₋|, |λ₊|)`.

**`:hlle`**: with `s_L = λ₋` and `s_R = λ₊` from [`light_cone`](@ref),
`F_L` if `s_L ≥ 0`, `F_R` if `s_R ≤ 0`, and otherwise
`(s_R F_L − s_L F_R + s_L s_R (U_R − U_L))/(s_R − s_L)`. Light-cone speeds
make it the exact upwind flux of the vacuum Maxwell system, whose waves move
at exactly those speeds; its dissipation on `𝔈` matches the one the
UCT–HLL EMF puts on `𝔅` ([`uct_hll_emf`](@ref)).

Both are consistent: with `L == R` each returns [`physical_flux`](@ref) of
that state, exactly for `:llf` and to rounding for `:hlle`. See "Riemann
solver" in `CODE.md`.
"""
@inline function riemann_flux(::Val{:llf}, m::Metric3{T}, L::Tuple, R::Tuple,
                              ::Val{d}) where {T,d}
    λm, λp = light_cone(m, Val(d))
    λ = max(abs(λm), abs(λp))
    F_L = physical_flux(m, L..., Val(d))
    F_R = physical_flux(m, R..., Val(d))
    U_L, U_R = L[1], R[1]
    return ntuple(v -> @inbounds((F_L[v] + F_R[v] - λ * (U_R[v] - U_L[v])) / 2), Val(NU))
end

@inline function riemann_flux(::Val{:hlle}, m::Metric3{T}, L::Tuple, R::Tuple,
                              ::Val{d}) where {T,d}
    s_L, s_R = light_cone(m, Val(d))
    F_L = physical_flux(m, L..., Val(d))
    # A superluminal shift: every wave leaves through one side, and the face
    # sees one state alone.
    s_L ≥ zero(T) && return F_L
    F_R = physical_flux(m, R..., Val(d))
    s_R ≤ zero(T) && return F_R
    U_L, U_R = L[1], R[1]
    return ntuple(v -> @inbounds((s_R * F_L[v] - s_L * F_R[v] +
                                  s_L * s_R * (U_R[v] - U_L[v])) / (s_R - s_L)), Val(NU))
end

"""
    fluid_speeds(m::Metric3, prim, ::Val{d}) -> (λ₋, λ₊)

The fluid's acoustic characteristic speeds along `d` in the 3+1 metric `m`
(Banyuls et al. 1997, ApJ 476, 221; Font 2008, Living Rev. Relativ. 11, 7,
§2.1.3), of a primitive state with velocity `vⁱ` and sound speed `c`:

    λ± = α/(1 − v²c²) [vᵈ(1 − c²) ± c √((1 − v²)(γ^{dd}(1 − v²c²) − vᵈvᵈ(1 − c²)))] − βᵈ

within the light cone, which they reach as `c → 1` (added in step 10).
"""
@inline function fluid_speeds(m::Metric3{T}, prim::Tuple, ::Val{d}) where {T,d}
    v = prim_v(prim)
    c² = clamp(prim_cs²(prim), zero(T), one(T))
    v² = dot_up(m, v, v)
    vd = @inbounds v[d]
    den = 1 - v² * c²
    disc = (1 - v²) * (@inbounds(m.γu[d, d]) * den - vd * vd * (1 - c²))
    root = sqrt(c² * max(disc, zero(T)))
    β = @inbounds m.β[d]
    return m.α * (vd * (1 - c²) - root) / den - β, m.α * (vd * (1 - c²) + root) / den - β
end

# One variable's HLLE flux with the signal speeds `s_L ≤ s_R`.
@inline hlle1(s_L, s_R, F_L, F_R, U_L, U_R) =
    s_L ≥ zero(s_L) ? F_L : s_R ≤ zero(s_R) ? F_R :
    (s_R * F_L - s_L * F_R + s_L * s_R * (U_R - U_L)) / (s_R - s_L)

"""
    riemann_flux(::Val{:hlle_split}, m::Metric3, L, R, ::Val{d}) -> NTuple{10}

HLLE with split signal speeds (added in step 10, for the measurement of
"Riemann solver" in `CODE.md`): the fluid's acoustic speeds
([`fluid_speeds`](@ref), the extremes over both states) for the advected
`D`, `D Yₑ` and `D s`, and the light cone, as `:hlle`, for `S_i`, `τ` and
`𝔈`, whose fluxes carry the Maxwell stress. Less dissipative on the rest
mass where the fluid is slow — at a star's surface and in its atmosphere,
where the light cone's dissipation is largest against the flow's. With
`L == R` it is [`physical_flux`](@ref) to rounding.
"""
@inline function riemann_flux(::Val{:hlle_split}, m::Metric3{T}, L::Tuple, R::Tuple,
                              ::Val{d}) where {T,d}
    c_L, c_R = light_cone(m, Val(d))
    l₋, l₊ = fluid_speeds(m, L[2], Val(d))
    r₋, r₊ = fluid_speeds(m, R[2], Val(d))
    f_L, f_R = min(l₋, r₋), max(l₊, r₊)
    F_L = physical_flux(m, L..., Val(d))
    F_R = physical_flux(m, R..., Val(d))
    U_L, U_R = L[1], R[1]
    return ntuple(Val(NU)) do v
        fluid = v == U_D || v == U_DYE || v == U_DS
        s_L, s_R = fluid ? (f_L, f_R) : (c_L, c_R)
        @inbounds hlle1(s_L, s_R, F_L[v], F_R[v], U_L[v], U_R[v])
    end
end
