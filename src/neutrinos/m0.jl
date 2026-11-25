# M0: the free-streaming neutrinos ("Neutrinos", "M0" in `CODE.md`;
# Radice et al. 2016, Appendix A, as revised in Radice et al. 2018, §II.3).
#
# Per species, two conserved densities move along a fiducial null vector
# `k^α`, radial from a centre, with `k^α u_α = −1`:
#
#     N = √−g n k^t        (the number, in units of the baryon mass m★)
#     𝓔 = √−g n E χ k^t    (the energy at infinity, E in units of m★c²)
#
#     ∂ₜ N + ∂ᵢ (N vⁱ) = √−g R^eff   − (κ^eff/k^t) N
#     ∂ₜ 𝓔 + ∂ᵢ (𝓔 vⁱ) = √−g χ Q^eff − (κ^eff/k^t) 𝓔
#
# with `vⁱ = kⁱ/k^t` and `χ = −k_t`. This file holds the pointwise pieces;
# the kernels that apply them on the forest follow below them.

# The slots of `Nν`: N then 𝓔, for ν_e, ν̄_e, ν_x.
const NNU = 6
@inline nu_N(i) = 2i - 1
@inline nu_E(i) = 2i

"""
    M0Closure(k_t_up, v, χ)

The closure at one point ("Neutrinos", "M0" in `CODE.md`): `kᵗ`, the
neutrinos' coordinate velocity `vⁱ = kⁱ/kᵗ` and the redshift factor
`χ = −k_t` from the fluid frame to infinity. `isbits`.
"""
struct M0Closure{T}
    kt::T
    v::SVector{3,T}
    χ::T
end

"""
    m0_closure(m::Metric3, u, x, center, r_core) -> M0Closure

The fiducial null vector `k = u + f ℓ` at the point `x`: `u` the fluid's
four-velocity from `uⁱ = W vⁱ` (the Eulerian velocity times the Lorentz
factor, `W`'s slots), `ℓ` the unit vector along the normal observer's radial
direction `r̂ = (x − c)/|x − c|_γ` projected orthogonal to `u`, and
`f = min(1, |x − c|_γ / r_core)`, so that inside `r_core` the neutrinos are
dragged with the fluid (`k` timelike) and at the centre `k = u`.

With `a = u·r̂ = W v_i r̂ⁱ`: `|P r̂|² = 1 + a²`, `ℓ^μ = (r̂^μ + a u^μ)/√(1 + a²)`,
so `kᵗ = uᵗ (1 + f a/√(1 + a²))` and `χ = −u_t − f (β_i r̂ⁱ + a u_t)/√(1 + a²)`
with `u_t = W (−α + β_i vⁱ)`.
"""
@inline function m0_closure(m::Metric3{T}, u::SVector{3,T}, x::SVector{3,T},
                            center::SVector{3,T}, r_core::T) where {T}
    ul = lower(m, u)                           # W v_i
    W = sqrt(1 + sum(u .* ul))
    v = u / W                                  # Eulerian vⁱ
    d = x - center
    r = sqrt(dot_up(m, d, d))
    s = iszero(r) ? zero(d) : d / r            # r̂ⁱ, unit in γ
    f = r < r_core ? r / r_core : one(T)
    a = sum(ul .* s)
    q = f / sqrt(1 + a * a)
    ut = W / m.α                               # uᵗ
    ui = W * (v - m.β / m.α)                   # uⁱ (four-velocity)
    βl = lower(m, m.β)
    u_t = W * (sum(βl .* v) - m.α)
    kt = ut * (1 + q * a)
    ki = ui + q * (s + a * ui)
    χ = -u_t - q * (sum(βl .* s) + a * u_t)
    return M0Closure{T}(kt, ki / kt, χ)
end

"""
    m0_mean_energy(N, 𝓔, χ) -> E

The free-streaming neutrinos' mean energy in the fluid frame, `E = 𝓔/(χ N)`,
in units of `m★c²`; zero where `N` or `𝓔` is not positive, a branch.
"""
@inline m0_mean_energy(N::T, 𝓔::T, χ::T) where {T} =
    (N > 0 && 𝓔 > 0) ? 𝓔 / (χ * N) : zero(T)

"""
    m0_absorption(N★, 𝓔★, κ, kt, γΔt) -> (N, 𝓔)

The implicit stage's closed-form solution of `N = N★ − γΔt (κ/kᵗ) N`, and
the same for `𝓔` with the same coefficient ("Neutrinos", "In the time
integrator" in `CODE.md`). `κ` is `κ^eff`, frozen at the cell's latest
recovery; the mean energy `𝓔/(χN)` is the same before and after.
"""
@inline function m0_absorption(N★::T, 𝓔★::T, κ::T, kt::T, γΔt::T) where {T}
    iszero(κ) && return N★, 𝓔★
    c = inv(1 + γΔt * κ / kt)
    return N★ * c, 𝓔★ * c
end

"""
    m0_absorption_opacity(r::SpeciesRates, τ0, N, 𝓔, χ) -> κ^eff

`κ^eff = e^{−τ⁰} (E/E^β)² κ⁰_a` at the free-streaming mean energy `E` of
`(N, 𝓔)`; zero where the species does not absorb or there are no
free-streaming neutrinos to have a mean energy (`N_min` is zero: a branch on
`N, 𝓔 > 0`, decided in step 18).
"""
@inline function m0_absorption_opacity(r::SpeciesRates{T}, τ0::T, N::T, 𝓔::T,
                                       χ::T) where {T}
    E = m0_mean_energy(N, 𝓔, χ)
    iszero(E) && return zero(T)
    return absorption_opacity(r, τ0, E)
end

# The monotonized-central slope (van Leer 1977): the smallest of 2a, (a + b)/2
# and 2b, zero at an extremum. It keeps a face value between its two cells'
# values, as minmod does, but not the factor of two minmod loses at a smooth
# extremum, which every grid line through M0's radial field has (step 18:
# minmod's rates 1.70–1.96 from 32³ to 64³ away from the centre, MC's
# 2.16–2.24).
@inline mc_slope(a::T, b::T) where {T} =
    a * b ≤ 0 ? zero(T) : sign(a) * min(2 * abs(a), abs(a + b) / 2, 2 * abs(b))

"""
    m0_upwind(vL, vR, qL, qR)

The upwind flux of a density `q` carried at the velocities `vL`, `vR` on
the two sides of a face, `max(vL, 0) qL + min(vR, 0) qR`: what leaves the
left cell rightward and the right cell leftward, so that it preserves
positivity under the light-speed CFL ("Neutrinos", "M0" in `CODE.md`).
"""
@inline m0_upwind(vL::T, vR::T, qL::T, qR::T) where {T} =
    max(vL, zero(T)) * qL + min(vR, zero(T)) * qR

# --- on the forest ---------------------------------------------------------------

"""
    M0Leakage(rates; center, r_core, feedback = true, optical_depth = true)

The neutrino scheme of `CODE.md`'s "Neutrinos", for `MHDProblem`'s
`neutrinos` keyword: leakage with the rate model `rates` ([`GrayRates`](@ref),
[`RJS96Rates`](@ref)) and M0's free-streaming neutrinos, radial from
`center` (a 3-vector, no default), dragged with the fluid inside `r_core`
(no default). `feedback = false` keeps the matter's sources off, for the
transport's tests and for a run that switches the heating on later; the
feedback itself is step 19's, and until then `feedback = true` is refused.
`optical_depth = false` takes `τ ≡ 0`, the optically thin limit, in which
the effective rates are the free ones: for the transport's tests, whose
analytic solutions it makes exact. `isbits` where `rates` is.
"""
struct M0Leakage{R,T}
    rates::R
    center::SVector{3,T}
    r_core::T
    feedback::Bool
    optical_depth::Bool
end
function M0Leakage(rates::NeutrinoRates; center, r_core, feedback::Bool=true,
                   optical_depth::Bool=true)
    feedback && throw(ArgumentError(
        "M0Leakage: the neutrinos' feedback on the matter is step 19's and not built; " *
        "pass `feedback = false` (PLAN.md)."))
    T = typeof(float(r_core))
    (isfinite(r_core) && r_core > 0) || throw(ArgumentError(
        "M0Leakage needs a positive r_core, got $r_core: inside it the radial direction " *
        "is scaled down to the fluid's, since at the centre it is undefined."))
    length(center) == 3 || throw(ArgumentError(
        "M0Leakage's center is a point in three dimensions, got $center; a missing " *
        "dimension's coordinate is zero."))
    return M0Leakage{typeof(rates),T}(rates, SVector{3,T}(center), T(r_core), feedback,
                                      optical_depth)
end

"""
    NuEscape()

The outer boundary of `Nν`: `N = 𝓔 = 0` in every ghost beyond an outer face,
so that the upwind flux there is outflow only ("Neutrinos", "M0" in
`CODE.md`). `isbits`.
"""
struct NuEscape end
(::NuEscape)(x, δ) = ntuple(_ -> zero(x[1]), Val(NNU))

# The neutrinos' sets on the forest, built by `MHDProblem` from an
# `M0Leakage` and handed back by `rebuild` after a regrid.
struct M0State{L,FN,FF,IS,OD,H}
    config::L
    Nν::FN                       # N, 𝓔 per species: the state vector's segment SEG_NU
    Fν::FF                       # NTuple{D}: their fluxes at the faces
    ischeds::IS                  # NTuple{D}: their interface schedules
    od::OD                       # the OpticalDepth, or nothing (τ ≡ 0)
    hook::H                      # NuEscape, as TreeAMR's CellBoundary
end

const SEG_NU = 6

m0_state(::Nothing, f, ops, backend) = nothing
function m0_state(c::M0Leakage, f::MHDFields{D}, ops, backend) where {D}
    T = eltype(f.U.work)
    forest = f.U.forest
    any(forest.periodic) && throw(ArgumentError(
        "M0Leakage needs a forest periodic in no direction: the closure's radial " *
        "direction flips across a periodic face, so the blocks on either side would " *
        "carry the same face at different velocities and the neutrinos' totals would " *
        "not be conserved (measured in step 18)."))
    Nν = FieldSet{T}(forest, NNU; G=f.U.G, centering=cellcentered(D), backend=backend)
    Fν = ntuple(d -> FieldSet{T}(forest, NNU; G=0, centering=face_centering(Val(D), d),
                                 backend=backend), D)
    od = c.optical_depth ? OpticalDepth(T, forest; backend=backend, ops=ops) : nothing
    return M0State(c, Nν, Fν, ntuple(d -> InterfaceSchedule(Fν[d]), D), od,
                   CellBoundary(AllVariables(NuEscape())))
end
# After a regrid the sets are the same, resized; the schedules are new.
m0_state(s::M0State, f::MHDFields{D}, ops, backend) where {D} =
    M0State(s.config, s.Nν, s.Fν, ntuple(d -> InterfaceSchedule(s.Fν[d]), D),
            s.od === nothing ? nothing : rebuild(s.od), s.hook)

# The rates at one cell from its row of `P`. `GrayRates` read only `ρ★`; the
# tabulated rates read the composition too, which step 20 adds.
@inline cell_rates(model::GrayRates{T}, Pc::NTuple) where {T} =
    neutrino_rates(model, Pc[P_RHO], T(NaN), T(NaN), nothing)

# τ⁰, τ¹ of species i at a cell, or zero without an optical depth.
@inline cell_depth(::Nothing, t, b, i, ::Type{T}) where {T} = (zero(T), zero(T))
@inline cell_depth(Tw, t, b, i, ::Type{T}) where {T} =
    @inbounds (Tw[t..., 2i - 1, b], Tw[t..., 2i, b])

# The fluid's uⁱ = W vⁱ of a row of `P`, in its metric.
@inline function cell_u(Pc::NTuple, m::Metric3)
    v = prim_v(Pc)
    return v / sqrt(1 - dot_up(m, v, v))
end

# The neutrinos' fluxes at the face `n` normal to `d` (between the cells
# n − e_d and n): N/√γ and 𝓔/√γ reconstructed by MC from the four cells
# around it, densitized with the face's √γ, carried upwind at the closure's
# velocity on each side, from that side's fluid velocity and the face
# metric and position.
@kernel function m0_flux_kernel!(Fw, @Const(Nw), @Const(Pw), @Const(origins), @Const(spacings),
                                 provider, t, center, r_core, ::Val{D}, ::Val{d}, ::Val{GF},
                                 ::Val{GN}, ::Val{GP}) where {D,d,GF,GN,GP}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(face_staggers(Val(D), Val(d))))
    m = point_metric(provider, oftype(h, t), x, Val(face_staggers(Val(D), Val(d))), n, b)
    # The fluid's uⁱ = W vⁱ at the four cells, reconstructed to each side of
    # the face by the same MC as N and 𝓔, so that the neutrinos' velocity is
    # second order where the fluid moves (step 18: from the two cells' values
    # alone the moving ball's N converged at 1.90–2.00 behind the front).
    cu(c) = (Pc = read_vars(Pw, stored(c, GP), b, Val(NP)); cell_u(Pc, aux_metric(Pc)))
    u₋₂, u₋₁, u₀, u₁ = cu(shift(n, d, -2)), cu(shift(n, d, -1)), cu(n), cu(shift(n, d, 1))
    uL = u₋₁ + mc_slope.(u₋₁ - u₋₂, u₀ - u₋₁) / 2
    uR = u₀ - mc_slope.(u₀ - u₋₁, u₁ - u₀) / 2
    vL = m0_closure(m, uL, x, center, r_core).v[d]
    vR = m0_closure(m, uR, x, center, r_core).v[d]
    q(c, v) = @inbounds Nw[stored(c, GN)..., v, b] / Pw[stored(c, GP)..., P_SQRTG, b]
    for v in 1:NNU
        q₋₂, q₋₁, q₀, q₁ = q(shift(n, d, -2), v), q(shift(n, d, -1), v), q(n, v),
                           q(shift(n, d, 1), v)
        qL = q₋₁ + mc_slope(q₋₁ - q₋₂, q₀ - q₋₁) / 2
        qR = q₀ - mc_slope(q₀ - q₋₁, q₁ - q₀) / 2
        @inbounds Fw[stored(n, GF)..., v, b] = m0_upwind(vL, vR, m.sqrtγ * qL, m.sqrtγ * qR)
    end
end

# The neutrinos' tendencies at an owned cell: the flux divergence and the
# emission, `√−g R^eff` and `√−g χ Q^eff`.
@kernel function m0_rhs_kernel!(du, @Const(Pw), Tw, Fs, @Const(origins), @Const(spacings),
                                model, center, r_core, ::Val{D}, ::Val{GP}, ::Val{GT},
                                ::Val{GF}) where {D,GP,GT,GF}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    T = typeof(h)
    Pc = read_vars(Pw, stored(n, GP), b, Val(NP))
    m = aux_metric(Pc)
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    cl = m0_closure(m, cell_u(Pc, m), x, center, r_core)
    r = cell_rates(model, Pc)
    sqrtmg = m.α * m.sqrtγ
    c = stored(n, GF)
    tc = ntuple(e -> n[e] + GT, Val(D))
    for i in 1:3
        τ0, τ1 = cell_depth(Tw, tc, b, i, T)
        R, Q = effective_rates(r[i], τ0, τ1)
        dN = flux_difference_sum(zero(h), Fs, c, nu_N(i), b)
        dE = flux_difference_sum(zero(h), Fs, c, nu_E(i), b)
        @inbounds du[n..., nu_N(i), b] = -dN / h + sqrtmg * R
        @inbounds du[n..., nu_E(i), b] = -dE / h + sqrtmg * cl.χ * Q
    end
end

# The implicit stage's absorption at an owned cell: closed-form, with κ^eff,
# kᵗ and χ from the cell's latest recovery (`P`) and the frozen τ, the mean
# energy from u★ ("Neutrinos", "In the time integrator" in `CODE.md`).
@kernel function m0_absorb_kernel!(Uw, @Const(Sw), @Const(Pw), Tw, @Const(origins),
                                   @Const(spacings), model, center, r_core, γΔt, ::Val{D},
                                   ::Val{GP}, ::Val{GT}) where {D,GP,GT}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    T = typeof(h)
    Pc = read_vars(Pw, stored(n, GP), b, Val(NP))
    m = aux_metric(Pc)
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    cl = m0_closure(m, cell_u(Pc, m), x, center, r_core)
    r = cell_rates(model, Pc)
    tc = ntuple(e -> n[e] + GT, Val(D))
    for i in 1:3
        τ0, _ = cell_depth(Tw, tc, b, i, T)
        N★ = @inbounds Sw[n..., nu_N(i), b]
        𝓔★ = @inbounds Sw[n..., nu_E(i), b]
        κ = m0_absorption_opacity(r[i], τ0, N★, 𝓔★, cl.χ)
        N, 𝓔 = m0_absorption(N★, 𝓔★, κ, cl.kt, oftype(h, γΔt))
        @inbounds Uw[n..., nu_N(i), b] = N
        @inbounds Uw[n..., nu_E(i), b] = 𝓔
    end
end

# The hooks into the scheme (`src/evolution.jl`), each a no-op without
# neutrinos, so that such a run is today's bit for bit.

depth_work(s::M0State) = s.od === nothing ? nothing : s.od.Nτ.work
depth_G(s::M0State) = s.od === nothing ? Val(0) : Val(first(s.od.Nτ.G))

# `prepare!`: Nν's ghosts, with NuEscape at the outer faces.
m0_prepare!(::Nothing, p, t) = nothing
m0_prepare!(::M0State, p, t) = (fill_set!(p, SEG_NU, t); nothing)

# `rhs!`: the fluxes, restricted at coarse-fine faces, and the tendencies.
m0_rhs!(::Nothing, du, p, t; metric=p.provider) = nothing
function m0_rhs!(s::M0State, du, p, t; metric=p.provider)
    f = p.f
    D = length(s.Fν)
    c = s.config
    ntuple(Val(D)) do d
        Fd = s.Fν[d]
        map_blocks!(m0_flux_kernel!, Fd, Fd.work, s.Nν.work, f.P.work, p.origins,
                    p.spacings, metric, t, c.center, c.r_core, Val(D), Val(d), Val(Fd.G),
                    Val(s.Nν.G), Val(f.P.G); closed=true, workgroupsize=p.workgroupsize)
        p.fixup && restrict_interfaces!(Fd, s.ischeds[d])
        nothing
    end
    map_blocks!(m0_rhs_kernel!, s.Nν, statearray(du, p.sets, SEG_NU), f.P.work, depth_work(s),
                map(F -> F.work, s.Fν), p.origins, p.spacings, c.rates, c.center, c.r_core,
                Val(D), Val(f.P.G), depth_G(s), Val(first(s.Fν).G);
                workgroupsize=p.workgroupsize)
    return nothing
end

# The implicit stage, before Ohm's solve.
m0_stage!(::Nothing, U, ustar, γΔt, p, t) = nothing
function m0_stage!(s::M0State, U, ustar, γΔt, p, t)
    f = p.f
    c = s.config
    D = length(s.Fν)
    map_blocks!(m0_absorb_kernel!, s.Nν, statearray(U, p.sets, SEG_NU),
                statearray(ustar, p.sets, SEG_NU), f.P.work, depth_work(s), p.origins,
                p.spacings, c.rates, c.center, c.r_core, γΔt, Val(D), Val(f.P.G), depth_G(s);
                workgroupsize=p.workgroupsize)
    return nothing
end

# The step limiter: one sweep of the optical depth from the step's κ.
m0_step!(::Nothing, p, t) = nothing
function m0_step!(s::M0State, p, t)
    s.od === nothing && return nothing
    opacities!(s.od, s.config.rates, p.f.P)
    sweep!(s.od, p.f.P)
    return nothing
end

# `regrid!`'s further entries: the fluxes resized, the optical depth carried.
m0_regrid_pairs(::Nothing) = ()
m0_regrid_pairs(s::M0State) =
    (map(F -> F => nothing, s.Fν)..., (s.od === nothing ? () : depth_regrid_pairs(s.od))...)

"""
    initial_depth!(p::MHDProblem, u, t) -> sweeps

The optical depth of the state `u` at `t`, converged from zero
(`converge!`): what `evolve!` does once on its initial data, and a script
that owns the loop does likewise. After a regrid the depth is transferred
and the sweeps go on one per step ("Neutrinos", "The optical depth" in
`CODE.md`). Zero sweeps without neutrinos or without an optical depth.
"""
initial_depth!(p, u, t) = m0_initial_depth!(p.neutrinos, p, u, t)
m0_initial_depth!(::Nothing, p, u, t) = 0
function m0_initial_depth!(s::M0State, p, u, t)
    s.od === nothing && return 0
    prepare!(p, u, t)
    reset_depth!(s.od)
    opacities!(s.od, s.config.rates, p.f.P)
    return converge!(s.od, p.f.P)
end

"""
    neutrino_totals(p::MHDProblem, u) -> NTuple{6}

`Σ N` and `Σ 𝓔` of ν_e, ν̄_e, ν_x over the owned cells, volume-weighted (the
`N, 𝓔` order of `Nν`), as `conserved_totals` sums the matter.
"""
function neutrino_totals(p, u)
    s = p.neutrinos
    s === nothing && throw(ArgumentError("neutrino_totals: the problem has no neutrinos."))
    T = eltype(s.Nν.work)
    forest = s.Nν.forest
    D = length(s.Fν)
    uN = stateview(u, p.sets, SEG_NU)
    return ntuple(v -> mesh_mapreduce(identity, +, zero(T), s.Nν, uN; vars=v,
                                      weight=key -> spacing(T, forest, key)^D), Val(NNU))
end
