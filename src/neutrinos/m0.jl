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
    m0_absorbed_fraction(κ, kt, γΔt) -> a

The share of `N★` and of `𝓔★` the implicit stage absorbs, `a = x/(1 + x)`
with `x = γΔt κ/kᵗ`, so that the matter gains `a N★` and `a 𝓔★`: the
same numbers as `γΔt (κ/kᵗ) N` at the solved `N` of [`m0_absorption`](@ref),
written so that it stays in `[0, 1]` for every `κ`, `κ = Inf` included
("In the time integrator" in `CODE.md`: where `N` and `𝓔` both vanish their
ratio `E` is noise, `κ^eff ∝ E²` can overflow, and `κ N` was `Inf·0`). Zero
where `κ` is, a branch.
"""
@inline function m0_absorbed_fraction(κ::T, kt::T, γΔt::T) where {T}
    iszero(κ) && return zero(T)
    return inv(1 + kt / (γΔt * κ))
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
    M0Leakage(rates; center, r_core, composition = nothing, feedback = true,
              optical_depth = true)

The neutrino scheme of `CODE.md`'s "Neutrinos", for `MHDProblem`'s
`neutrinos` keyword: leakage with the rate model `rates` ([`GrayRates`](@ref),
[`RJS96Rates`](@ref)) and M0's free-streaming neutrinos, radial from
`center` (a 3-vector, no default), dragged with the fluid inside `r_core`
(no default). `feedback = false` keeps the matter's sources off, for the
transport's tests and for a run that switches the heating on later.
`optical_depth = false` takes `τ ≡ 0`, the optically thin limit, in which
the effective rates are the free ones: for the transport's tests, whose
analytic solutions it makes exact. `composition` is the table's
(EntropyEOS's `CompositionView`), required with `RJS96Rates` and refused with
`GrayRates`; `MHDProblem` refuses one that does not `match` its EOS and moves
it to the run's device ("Field sets and interface" in `CODE.md`).
"""
struct M0Leakage{R,T,C}
    rates::R
    center::SVector{3,T}
    r_core::T
    feedback::Bool
    optical_depth::Bool
    composition::C
end
function M0Leakage(rates::NeutrinoRates; center, r_core, composition=nothing,
                   feedback::Bool=true, optical_depth::Bool=true)
    T = typeof(float(r_core))
    if rates isa RJS96Rates
        composition isa EntropyEOS.CompositionView || throw(ArgumentError(
            "M0Leakage with RJS96Rates needs the table's composition, an EntropyEOS " *
            "CompositionView (`composition = cv`), got $(typeof(composition)): the rates " *
            "read the chemical potentials and mass fractions at each cell."))
    else
        composition === nothing || throw(ArgumentError(
            "M0Leakage with $(nameof(typeof(rates))) takes no composition: those rates " *
            "read none, and a composition here would suggest they do."))
    end
    (isfinite(r_core) && r_core > 0) || throw(ArgumentError(
        "M0Leakage needs a positive r_core, got $r_core: inside it the radial direction " *
        "is scaled down to the fluid's, since at the centre it is undefined."))
    length(center) == 3 || throw(ArgumentError(
        "M0Leakage's center is a point in three dimensions, got $center; a missing " *
        "dimension's coordinate is zero."))
    return M0Leakage{typeof(rates),T,typeof(composition)}(rates, SVector{3,T}(center),
                                                          T(r_core), feedback, optical_depth,
                                                          composition)
end

# The `M0Leakage` as a checkpoint's recipe holds it: everything that decides the
# numbers, the composition by its table's fingerprint.
neutrino_recipe(c::M0Leakage) =
    (; rates=repr(c.rates), center=Tuple(c.center), r_core=c.r_core, feedback=c.feedback,
     optical_depth=c.optical_depth,
     composition=c.composition === nothing ? "none" : string(c.composition.fingerprint))

# What the kernels receive: the rates alone, or with the composition.
rate_model(c::M0Leakage{R,T,Nothing}) where {R,T} = c.rates
rate_model(c::M0Leakage) = TabulatedRates(c.rates, c.composition)

# The problem's EOS as EntropyEOS's `matches` reads it: the table's view
# under the geometric units.
eos_view(e::GeometricEOS) = e.eos
eos_view(e) = e

# The `M0Leakage` at the problem's type `T` (its centre and `r_core`, which the
# kernels combine with the cell's coordinates), and its composition checked against the
# problem's EOS and moved to its device.
function placed(c::M0Leakage, ::Type{T}, eos, backend) where {T}
    cv = c.composition
    if cv !== nothing
        EntropyEOS.matches(cv, eos_view(eos)) || throw(ArgumentError(
            "M0Leakage's composition was not built from the problem's EOS table " *
            "(EntropyEOS.matches is false): its densities and fractions belong to another " *
            "table."))
        cv = Adapt.adapt(backend, cv)
    end
    return M0Leakage{typeof(c.rates),T,typeof(cv)}(c.rates, SVector{3,T}(c.center),
                                                   T(c.r_core), c.feedback, c.optical_depth,
                                                   cv)
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
struct M0State{L,FN,FF,IS,OD,H,SH}
    config::L
    Nν::FN                       # N, 𝓔 per species: the state vector's segment SEG_NU
    Fν::FF                       # NTuple{D}: their fluxes at the faces
    ischeds::IS                  # NTuple{D}: their interface schedules
    od::OD                       # the OpticalDepth, or nothing (τ ≡ 0)
    hook::H                      # NuEscape, as TreeAMR's CellBoundary
    shifted::SH                  # with feedback: u★'s matter plus the absorbed share
end

const SEG_NU = 6

m0_state(::Nothing, f, ops, backend, eos) = nothing
function m0_state(c0::M0Leakage, f::MHDFields{D}, ops, backend, eos) where {D}
    T = eltype(f.U.work)
    c = placed(c0, T, eos, backend)
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
                   CellBoundary(AllVariables(NuEscape())), shifted_buffer(c, f, backend))
end
# After a regrid the sets are the same, resized; the schedules and the
# buffer are new.
m0_state(s::M0State, f::MHDFields{D}, ops, backend, eos) where {D} =
    M0State(s.config, s.Nν, s.Fν, ntuple(d -> InterfaceSchedule(s.Fν[d]), D),
            s.od === nothing ? nothing : rebuild(s.od), s.hook,
            shifted_buffer(s.config, f, backend))

# The matter segment's shape, owned cells only, for the shifted u★ that Ohm's
# solve reads when the neutrinos feed back ("Neutrinos", "In the time
# integrator" in `CODE.md`): the integrator forbids changing u★ itself, and
# handing Ohm's kernel `U` would alias its input and output.
function shifted_buffer(c::M0Leakage, f::MHDFields{D}, backend) where {D}
    c.feedback || return nothing
    w = f.U.work
    dims = ntuple(e -> size(w, e) - 2 * f.U.G[e], Val(D))
    return allocate(backend, eltype(w), (dims..., NU, size(w, ndims(w))))
end

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

# Whether a cell's neutrino–matter sources are on: off, by a branch, where
# the atmosphere policy flagged the cell's recovery ("Masks branch, never
# multiply").
@inline sources_on(Pc::NTuple) = aux_flags(Pc) & EntropyEOS.FLAG_POL_ATMOSPHERE == 0

# The neutrinos' tendencies at an owned cell: the flux divergence and the
# emission, `√−g R^eff` and `√−g χ Q^eff`; with feedback, the matter's share
# of the emission into the matter's tendencies (`duU`, written by the matter's
# kernel first): `√γ D Yₑ` loses `√−g (R^eff_{ν_e} − R^eff_{ν̄_e})`, and
# `G^μ = −Σ Q^eff k^μ` enters `√γ τ` as `α²√γ G^t` and `√γ S_j` as `α√γ G_j`
# ("Neutrinos", "Coupling to the matter" in `CODE.md`). The cell's rate
# `|d ln(√γ D Yₑ)/dt|` from the emission goes into `X` (its largest in the
# chunk).
@kernel function m0_rhs_kernel!(du, duU, Xw, @Const(Pw), Tw, Fs, @Const(origins),
                                @Const(spacings), model, center, r_core, ::Val{D},
                                ::Val{GP}, ::Val{GT}, ::Val{GF}) where {D,GP,GT,GF}
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
    on = sources_on(Pc)
    sqrtmg = m.α * m.sqrtγ
    c = stored(n, GF)
    tc = ntuple(e -> n[e] + GT, Val(D))
    Rnet = zero(T)                   # R^eff_{ν_e} − R^eff_{ν̄_e}
    Qsum = zero(T)
    for i in 1:3
        τ0, τ1 = cell_depth(Tw, tc, b, i, T)
        R, Q = on ? effective_rates(r[i], τ0, τ1) : (zero(T), zero(T))
        i == 1 && (Rnet += R)
        i == 2 && (Rnet -= R)
        Qsum += Q
        dN = flux_difference_sum(zero(h), Fs, c, nu_N(i), b)
        dE = flux_difference_sum(zero(h), Fs, c, nu_E(i), b)
        @inbounds du[n..., nu_N(i), b] = -dN / h + sqrtmg * R
        @inbounds du[n..., nu_E(i), b] = -dE / h + sqrtmg * cl.χ * Q
    end
    # A branch, not a product with zero (step 19: before the first right-hand
    # side `P` holds zeros, whose closure is not finite).
    if duU !== nothing && !(iszero(Rnet) && iszero(Qsum))
        k = cl.kt * cl.v                         # kⁱ
        kl = lower(m, k) + cl.kt * lower(m, m.β)  # k_j = γ_jk kᵏ + β_j kᵗ
        @inbounds begin
            duU[n..., U_DYE, b] -= sqrtmg * Rnet
            duU[n..., U_TAU, b] -= m.α * sqrtmg * cl.kt * Qsum
            duU[n..., U_S, b] -= sqrtmg * Qsum * kl[1]
            duU[n..., U_S + 1, b] -= sqrtmg * Qsum * kl[2]
            duU[n..., U_S + 2, b] -= sqrtmg * Qsum * kl[3]
        end
        # √γ D Yₑ = √γ ρ★ W Yₑ from the cell's primitives.
        DYe = m.sqrtγ * Pc[P_RHO] * cosh(Pc[P_W]) * Pc[P_YE]
        rate = DYe > 0 ? abs(sqrtmg * Rnet) / DYe : zero(T)
        @inbounds Xw[n..., X_NU_RATE, b] = max(Xw[n..., X_NU_RATE, b], rate)
    end
end

# The implicit stage's absorption at an owned cell: closed-form, with κ^eff,
# kᵗ and χ from the cell's latest recovery (`P`) and the frozen τ, the mean
# energy from u★ ("Neutrinos", "In the time integrator" in `CODE.md`). With
# feedback, the matter's share is what the neutrinos lose, the absorbed
# amounts `a N★` and `a 𝓔★` (`m0_absorbed_fraction`; `γΔt κ N/kᵗ` and
# `γΔt κ 𝓔/kᵗ` at the solved N, 𝓔, but finite for κ = Inf): `√γ D Yₑ` gains
# `a N★` of ν_e less that of ν̄_e, and `G^μ = Σ κ n E k^μ` with
# `√−g n E = 𝓔/(χ kᵗ)` enters `√γ τ` as `α kᵗ Σ a 𝓔★/χ` and `√γ S_j` as
# `Σ a 𝓔★ k_j/χ`, into `U` and into the shifted u★ Ohm's solve then reads
# (`Sh`, a copy of u★'s matter).
@kernel function m0_absorb_kernel!(Uw, @Const(Sw), UU, Sh, @Const(Pw), Tw, @Const(origins),
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
    on = sources_on(Pc)
    tc = ntuple(e -> n[e] + GT, Val(D))
    ΔYe = zero(T)                    # absorbed N of ν_e less ν̄_e
    Δ𝓔 = zero(T)                     # absorbed 𝓔, the three
    for i in 1:3
        τ0, _ = cell_depth(Tw, tc, b, i, T)
        N★ = @inbounds Sw[n..., nu_N(i), b]
        𝓔★ = @inbounds Sw[n..., nu_E(i), b]
        κ = on ? m0_absorption_opacity(r[i], τ0, N★, 𝓔★, cl.χ) : zero(T)
        N, 𝓔 = m0_absorption(N★, 𝓔★, κ, cl.kt, oftype(h, γΔt))
        @inbounds Uw[n..., nu_N(i), b] = N
        @inbounds Uw[n..., nu_E(i), b] = 𝓔
        a = m0_absorbed_fraction(κ, cl.kt, oftype(h, γΔt))
        i == 1 && (ΔYe += a * N★)
        i == 2 && (ΔYe -= a * N★)
        Δ𝓔 += a * 𝓔★
    end
    # A branch, not a product with zero: SSP3(4,3,3)'s first stage is implicit,
    # so a run's first stage solve can come before any right-hand side has
    # filled `P`, whose zeros make the closure not finite; κ is zero there, and
    # the product was NaN (step 19, found by the reset's NONFINITE flags).
    if UU !== nothing && !(iszero(ΔYe) && iszero(Δ𝓔))
        k = cl.kt * cl.v
        kl = lower(m, k) + cl.kt * lower(m, m.β)
        Δ = (ΔYe, m.α * cl.kt * Δ𝓔 / cl.χ, Δ𝓔 * kl[1] / cl.χ, Δ𝓔 * kl[2] / cl.χ,
             Δ𝓔 * kl[3] / cl.χ)
        slots = (U_DYE, U_TAU, U_S, U_S + 1, U_S + 2)
        for j in 1:5
            @inbounds UU[n..., slots[j], b] += Δ[j]
            @inbounds Sh[n..., slots[j], b] += Δ[j]
        end
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
    duU = c.feedback ? statearray(du, p.sets, SEG_U) : nothing
    map_blocks!(m0_rhs_kernel!, s.Nν, statearray(du, p.sets, SEG_NU), duU, f.X.work, f.P.work,
                depth_work(s), map(F -> F.work, s.Fν), p.origins, p.spacings, rate_model(c),
                c.center, c.r_core, Val(D), Val(f.P.G), depth_G(s), Val(first(s.Fν).G);
                workgroupsize=p.workgroupsize)
    return nothing
end

# The implicit stage, before Ohm's solve; returns the matter's conserved
# state Ohm's solve reads: u★'s, or with feedback the shifted copy.
m0_stage!(::Nothing, U, ustar, γΔt, p, t) = statearray(ustar, p.sets, SEG_U)
function m0_stage!(s::M0State, U, ustar, γΔt, p, t)
    f = p.f
    c = s.config
    D = length(s.Fν)
    us = statearray(ustar, p.sets, SEG_U)
    if c.feedback
        copyto!(s.shifted, us)
        UU, Sh = statearray(U, p.sets, SEG_U), s.shifted
    else
        UU, Sh = nothing, nothing
    end
    map_blocks!(m0_absorb_kernel!, s.Nν, statearray(U, p.sets, SEG_NU),
                statearray(ustar, p.sets, SEG_NU), UU, Sh, f.P.work, depth_work(s),
                p.origins, p.spacings, rate_model(c), c.center, c.r_core, γΔt, Val(D), Val(f.P.G),
                depth_G(s); workgroupsize=p.workgroupsize)
    return c.feedback ? s.shifted : us
end

"""
    neutrino_tally(p::MHDProblem) -> (; dYe_rate)

The largest `|d ln(√γ D Yₑ)/dt|` the neutrinos' emission gave any owned cell
in any right-hand side since the last `clear_tally!` (the chunk's, in
`evolve!`): times the step, the change in `Yₑ` per step the explicit emission
makes ("Neutrinos", "In the time integrator" in `CODE.md`). Zero without
feedback.
"""
function neutrino_tally(p)
    T = eltype(p.f.X.work)
    return (; dYe_rate=mesh_mapreduce(identity, max, zero(T), p.f.X; vars=X_NU_RATE))
end

# The step limiter: one sweep of the optical depth from the step's κ.
m0_step!(::Nothing, p, t) = nothing
function m0_step!(s::M0State, p, t)
    s.od === nothing && return nothing
    opacities!(s.od, rate_model(s.config), p.f.P)
    sweep!(s.od, p.f.P)
    return nothing
end

# `regrid!`'s further entries: the fluxes resized, the optical depth carried.
m0_regrid_pairs(::Nothing) = ()
m0_regrid_pairs(s::M0State) =
    (map(F -> F => nothing, s.Fν)..., (s.od === nothing ? () : depth_regrid_pairs(s.od))...)

"""
    initial_depth!(p::MHDProblem, u, t; metric = p.provider) -> sweeps

The neutrinos' start on the state `u` at `t`: `P` prepared, so that the
first implicit stage, which SSP3(4,3,3) makes before any right-hand side,
reads real coefficients; and the optical depth converged from zero
(`converge!`). What `evolve!` does once on its initial data, and a script
that owns the loop does likewise. After a regrid the depth is transferred
and the sweeps go on one per step ("Neutrinos", "The optical depth" in
`CODE.md`). Zero sweeps without neutrinos or without an optical depth.
"""
initial_depth!(p, u, t; metric=p.provider) = m0_initial_depth!(p.neutrinos, p, u, t; metric)
m0_initial_depth!(::Nothing, p, u, t; metric) = 0
function m0_initial_depth!(s::M0State, p, u, t; metric)
    prepare!(p, u, t; metric)
    s.od === nothing && return 0
    reset_depth!(s.od)
    opacities!(s.od, rate_model(s.config), p.f.P)
    return converge!(s.od, p.f.P)
end

"""
    resume_neutrinos!(p::MHDProblem, u, t; metric = p.provider)

`P` prepared from the state `u`, as [`initial_depth!`](@ref) prepares it, so that the next
implicit stage, which reads the cell's latest recovery, reads the state's: what `evolve!`
does at every chunk boundary, so that a restart, which has no `P` of its own and whose
optical depth the checkpoint restored (`restore_run!`), is the run that never stopped. A
script that owns the loop and checkpoints does likewise. Nothing without neutrinos.
"""
resume_neutrinos!(p, u, t; metric=p.provider) =
    (p.neutrinos === nothing || prepare!(p, u, t; metric); nothing)

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

# The window of the shell luminosity, `∫ W dr = 1` over `|r − r_L| < width`.
@inline function shell_window(r::T, r_L::T, width::T) where {T}
    d = r - r_L
    abs(d) < width || return zero(T)
    return (1 + cospi(d / width)) / (2 * width)
end

# `W(r) Fⁱ r̂_i` of N and 𝓔 of each species at an owned cell, the flux `q vⁱ`
# at the closure's velocity from the cell's own `uⁱ`, `r̂` the coordinate radial
# unit vector about the centre.
@kernel function luminosity_kernel!(Lw, @Const(Nw), @Const(Pw), @Const(origins),
                                    @Const(spacings), center, r_core, r_L, width, ::Val{D},
                                    ::Val{GP}) where {D,GP}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    T = typeof(h)
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    d = x - center
    r = sqrt(sum(d .* d))
    w = shell_window(r, r_L, width)
    if iszero(w)
        for v in 1:NNU
            @inbounds Lw[n..., v, b] = zero(T)
        end
    else
        Pc = read_vars(Pw, stored(n, GP), b, Val(NP))
        m = aux_metric(Pc)
        vr = sum(m0_closure(m, cell_u(Pc, m), x, center, r_core).v .* d) / r
        for v in 1:NNU
            @inbounds Lw[n..., v, b] = w * vr * Nw[n..., v, b]
        end
    end
end

"""
    neutrino_luminosity(p::MHDProblem, u; r_L, width) -> NTuple{6}

The number and energy luminosities `(L_N, L_𝓔)` of ν_e, ν̄_e and ν_x (the `N, 𝓔` order of
`Nν`) through the spherical shell of radius `r_L` about the neutrinos' centre: the
coordinate-volume integral of `W(r) Fⁱ r̂_i` with the window `W` of half-width `width`
(both with no default), `Fⁱ = N vⁱ` and `𝓔 vⁱ` at the closure's velocity ("Neutrinos",
"M0" in `CODE.md`). `𝓔` is the energy at infinity, so `L_𝓔` is the luminosity at infinity
and `L_𝓔/L_N` the mean energy at infinity, in units of `m★c²`. Reads `P` as the run left
it, so it changes nothing; the shell must lie outside the matter and inside the domain.
"""
function neutrino_luminosity(p, u; r_L, width)
    s = p.neutrinos
    s === nothing && throw(ArgumentError("neutrino_luminosity: the problem has no neutrinos."))
    T = eltype(s.Nν.work)
    (width > 0 && r_L > width) || throw(ArgumentError(
        "neutrino_luminosity needs 0 < width < r_L, got r_L = $r_L, width = $width: the " *
        "shell is the window |r − r_L| < width."))
    forest = s.Nν.forest
    D = length(s.Fν)
    L = FieldSet{T}(forest, NNU; G=0, centering=cellcentered(D),
                    backend=get_backend(s.Nν.work))
    c = s.config
    map_blocks!(luminosity_kernel!, L, L.work, statearray(u, p.sets, SEG_NU), p.f.P.work,
                p.origins, p.spacings, c.center, c.r_core, T(r_L), T(width), Val(D),
                Val(p.f.P.G); workgroupsize=p.workgroupsize)
    return ntuple(v -> mesh_mapreduce(identity, +, zero(T), L; vars=v,
                                      weight=key -> spacing(T, forest, key)^D), Val(NNU))
end
