# The right-hand side, the implicit stage and the reset: the kernels that
# write the field sets, and the order they run in ("The right-hand side" in
# `CODE.md`).
#
#   1. `scatter!` the multi-set state; `fill_ghosts!` every evolved set.
#   2. `𝔅 = curl A + 𝔅̄` on every stored face (`src/curl.jl`).
#   3. The metric at every stored cell, `B` averaged to it, and the
#      hydrodynamic recovery, into `P` (`stored = true`).
#   4. `q = ∇·E` at the owned cells.
#   5. (`T_ab` for the spacetime: rung 5.)
#   6. Reconstruction and the Riemann fluxes into `F_d` over the closed face
#      ranges; the EMF into `EMF_j` over the closed edge ranges;
#      `restrict_interfaces!` on both.
#   7. `du`: the flux divergence plus the explicit sources for the matter;
#      `−EMF_j − ∂_j(αΦ − βᵏA_k)` for `A_j`; the gauge equation for `Φ̃`.
#
# The implicit stage solve runs on the owned cells between explicit
# evaluations and writes only `𝔈` (step 3; "The implicit stage solve"), and
# the reset runs after it as the stage limiter and once more on each step's
# result as the step limiter ("Floors, the atmosphere and the magnetosphere").
#
# **Index conventions.** Every kernel works with *local* indices `n`: the owned
# points of a set are `1 … N` along each dimension, ghosts lie outside, and a
# set of ghost width `G` stores local `n` at `n + G`. A launch with
# `stored = true` hands the kernel stored indices, which it converts with
# `n = s − G`; the default and the `closed = true` launches hand it local
# indices directly ("Sharp edges" in `PLAN.md`). A cell `n` lies between the
# vertices `n` and `n + 1`, so a face normal to `d` with local index `n` lies
# between the cells `n − e_d` and `n`. Every kernel receives one `Val(G)` per
# set it reads. A state-vector array (`statearray`) has no ghosts, so a local
# owned index is its index.
#
# **Warm starts come from the state** (proposed in step 5, amending
# "Thermodynamics" in `CODE.md`, which carried them in `P`): every recovery
# the scheme makes is seeded with `s = (√γ D s)/(√γ D)`, the advected entropy
# the state carries for this purpose, and a cold rapidity and EOS warm start.
# A warm start carried in `P` from the previous evaluation would make every
# recovery depend, at the solver's tolerance, on what was evaluated before —
# the right-hand side would not be a function of `(u, t)` bit for bit, and a
# restart from a checkpoint (which holds no `P`) would not reproduce the run.

# --- small pointwise helpers ------------------------------------------------------

# The position of the local point `n` of a set whose staggers are `C`
# (`1` vertex-like, `0` cell-like), in a block with origin `o` and spacing
# `h`, as a 3-vector with zero along the missing directions. The same
# arithmetic as TreeAMR's `coordinates`, so the two agree bit for bit.
@inline function point_x(o::NTuple{D,T}, h::T, n::NTuple{D,Int}, ::Val{C}) where {D,T,C}
    return SVector(ntuple(Val(3)) do e
        e ≤ D ? o[e] + (n[e] - (C[e] == 1 ? one(T) : one(T) / 2)) * h : zero(T)
    end)
end

@inline cell_staggers(::Val{D}) where {D} = ntuple(_ -> 0, Val(D))
@inline vertex_staggers(::Val{D}) where {D} = ntuple(_ -> 1, Val(D))
@inline face_staggers(::Val{D}, ::Val{d}) where {D,d} = ntuple(e -> e == d ? 1 : 0, Val(D))
@inline edge_staggers(::Val{D}, ::Val{j}) where {D,j} = ntuple(e -> e == j ? 0 : 1, Val(D))

# The `M` variables of point `idx` (stored) of block `b`.
@inline read_vars(w, idx::NTuple, b::Int, ::Val{M}) where {M} =
    ntuple(v -> @inbounds(w[idx..., v, b]), Val(M))

# The electric field `Eⁱ` held in `P` at a stored index.
@inline read_E(Pw, idx::NTuple, b::Int) =
    @inbounds SVector(Pw[idx..., P_E, b], Pw[idx..., P_E + 1, b], Pw[idx..., P_E + 2, b])

"""
    state_guesses(cons) -> (s, NaN, NaN)

The warm start of a recovery from the conserved state itself: the advected
entropy `s = (√γ D s)/(√γ D)`, or `NaN` where it is not finite, and no
rapidity or EOS warm start (see the top of `src/evolution.jl`). A pure
function of the state, so that every recovery the scheme makes is.
"""
@inline function state_guesses(cons::NTuple{NU,T}) where {T}
    s = cons_Ds(cons) / cons_D(cons)
    return (isfinite(s) ? s : T(NaN), T(NaN), T(NaN))
end

# --- the problem ------------------------------------------------------------------

"""The admissible Riemann solvers (`src/riemann.jl`)."""
const MHD_SOLVERS = (:llf, :hlle)

"""
    MHDProblem(f::MHDFields; eos, opts, pol, model, provider = Flat(), 𝔅̄ = 0,
               ξ = 0, stage = OhmStageOptions{T}(), limiter, riemann = :hlle,
               ops_U = Operators(prolongation=3, restriction=2, family=Conservative),
               ops_A = Operators(prolongation=4, restriction=4),
               boundary = nothing, fixup = true)

Everything a right-hand side, an implicit stage and a reset need, built once
per mesh: the field sets `f` ([`mhd_fields`](@ref)), a ghost schedule per
evolved set, the interface schedules of the fluxes and the EMFs, the per-block
origins and spacings on the sets' backend, and the physics:

- `eos`, `opts`, `pol`: EntropyEOS's `AbstractEOS`, `Con2PrimOptions` and
  `PolicyOptions`, in geometric units ("Thermodynamics" in `CODE.md`);
- `model`: the conductivity ([`Vacuum`](@ref), [`Constant`](@ref),
  [`DensityProfile`](@ref), [`IdealConductor`](@ref),
  [`MagnetizationCutoff`](@ref));
- `provider`: the metric ([`Flat`](@ref), [`Analytic`](@ref));
- `𝔅̄`: the constant densitized background field, an `SVector{3}`
  (`src/curl.jl`);
- `ξ`: the gauge damping (open; step 8 decides it);
- `stage`: the stage solve's [`OhmStageOptions`](@ref).

**`limiter` has no default** (`:none`, `:minmod` or `:mc`), as in TreeHydro:
it changes what is measured. `riemann` is `:hlle` (the light-cone HLL, whose
dissipation on `𝔈` matches the UCT EMF's on `𝔅`) or `:llf`.

`ops_U` and `ops_A` are the operators of the matter set and of `A`, `Φ̃`
(proposed in step 5: Conservative width 3 for the matter, PointValue width 4
for the potentials, which `CODE.md` leaves (open) until step 8 measures it;
nothing in step 5 refines, so they only have to pass TreeAMR's checks).
`boundary` is `nothing` for a periodic domain, or one hook per evolved set in
the order of [`evolved_sets`](@ref) (AMR-2 having landed, each set takes its
own): a hook TreeAMR's `fill_ghosts!` takes, or a time-dependent one that
[`boundary_at`](@ref) turns into such a hook at the time of each fill — the
case's Dirichlet data, [`DirichletBoundary`](@ref) (step 6). `fixup = false`
skips `restrict_interfaces!`, the negative control of step 8.
"""
struct MHDProblem{T,D,FS,ST,SC,ISF,ISE,OR,SP,EOS,OPT,POL,MOD,MET,BC,LIM,RS}
    f::FS
    sets::ST                     # evolved_sets(f): the state vector's segments
    schedules::SC                # one GhostSchedule per evolved set
    ischeds_F::ISF               # NTuple{D,InterfaceSchedule}
    ischeds_EMF::ISE             # NTuple{3}: an InterfaceSchedule, or nothing
    origins::OR                  # per block, on the sets' backend
    spacings::SP
    eos::EOS
    opts::OPT
    pol::POL
    model::MOD
    provider::MET
    𝔅̄::SVector{3,T}
    ξ::T
    stage::OhmStageOptions{T}
    boundary::BC
    fixup::Bool
    limiter::Val{LIM}
    solver::Val{RS}
end

function MHDProblem(f::MHDFields{D}; eos::EntropyEOS.AbstractEOS, opts, pol,
                    model::Conductivity, provider::MetricProvider=Flat(),
                    𝔅̄=nothing, ξ=0, stage=nothing, limiter::Union{Symbol,Nothing}=nothing,
                    riemann::Symbol=:hlle,
                    ops_U::Operators=Operators(; prolongation=3, restriction=2,
                                               family=Conservative),
                    ops_A::Operators=Operators(; prolongation=4, restriction=4),
                    boundary=nothing, fixup::Bool=true) where {D}
    T = eltype(f.U.work)
    limiter === nothing && throw(ArgumentError(
        "MHDProblem has no default limiter: pass `limiter`, one of $(MHD_LIMITERS). " *
        "The choice is the case's — :none keeps a clean O(h²) truncation error for a " *
        "convergence study, :minmod and :mc keep a shock monotone and clip a smooth " *
        "extremum while doing it."))
    limiter in MHD_LIMITERS || throw(ArgumentError(
        "limiter must be one of $(MHD_LIMITERS), got :$limiter: it travels into the " *
        "kernels as a Val, and `slope` has a method for each of those and no others."))
    riemann in MHD_SOLVERS || throw(ArgumentError(
        "riemann must be one of $(MHD_SOLVERS), got :$riemann: `riemann_flux` has a " *
        "method for each of those and no others (CODE.md, \"Riemann solver\")."))
    sets = evolved_sets(f)
    G = f.U.G
    for s in (sets..., f.P, f.B...)
        s.G == G || throw(ArgumentError(
            "every ghosted set must have the matter set's ghost width G=$G, got " *
            "$(s.G) for a $(s.centering) set: the curl reads the edges of every " *
            "stored face, and the recovery reads the faces of every stored cell, at " *
            "the same local index (src/curl.jl)."))
    end
    for s in (f.F..., f.EMF...)
        all(==(0), s.G) || throw(ArgumentError(
            "the fluxes and the EMFs carry no ghosts: they are computed over the " *
            "closed ranges and never exchanged; got G=$(s.G)."))
    end
    boundary === nothing || length(boundary) == length(sets) || throw(ArgumentError(
        "boundary is `nothing` (a periodic domain) or one hook per evolved set, " *
        "$(length(sets)) of them in the order (U, A₁, A₂, A₃, Φ̃); got $(length(boundary))."))
    if boundary === nothing && !all(f.U.forest.periodic)
        throw(ArgumentError(
            "the forest is not periodic in every dimension ($(f.U.forest.periodic)) but " *
            "no boundary hooks were given: the ghosts facing outside the domain would " *
            "hold whatever the allocation left."))
    end
    schedules = ntuple(i -> GhostSchedule(sets[i], i == SEG_U ? ops_U : ops_A),
                       length(sets))
    ischeds_F = ntuple(d -> InterfaceSchedule(f.F[d]), D)
    # An EMF set with no vertex-like dimension (in 1D, the one along x) has no
    # coarse-fine face to restrict, and TreeAMR refuses to build a schedule
    # for it.
    ischeds_EMF = ntuple(j -> any(==(1), staggers(f.EMF[j])) ? InterfaceSchedule(f.EMF[j]) :
                              nothing, 3)
    backend = get_backend(f.U.work)
    forest = f.U.forest
    origins = to_backend(backend, block_origins(forest, T))
    spacings = to_backend(backend, block_spacings(forest, T))
    B̄ = 𝔅̄ === nothing ? zero(SVector{3,T}) : SVector{3,T}(𝔅̄)
    st = stage === nothing ? OhmStageOptions{T}() : stage
    return MHDProblem{T,D,typeof(f),typeof(sets),typeof(schedules),typeof(ischeds_F),
                      typeof(ischeds_EMF),typeof(origins),typeof(spacings),typeof(eos),
                      typeof(opts),typeof(pol),typeof(model),typeof(provider),
                      typeof(boundary),limiter,riemann}(
        f, sets, schedules, ischeds_F, ischeds_EMF, origins, spacings, eos, opts, pol, model,
        provider, B̄, T(ξ), st, boundary, fixup, Val(limiter), Val(riemann))
end

"""
    boundary_at(hook, t)

The hook TreeAMR's `fill_ghosts!` is handed for a fill at time `t`: `hook`
itself, for a hook that does not depend on time, and the time-`t` Dirichlet
hook of a [`DirichletBoundary`](@ref). Every fill knows its time — a
right-hand side's, a stage's, a reset's — and passes it here, so that a
boundary that moves with the solution (the vector potential of a shock tube
whose far states carry an electric field grows linearly in time) is filled
with the value at that time.
"""
boundary_at(hook, t) = hook

# The ghost fill of evolved set `i` at time `t`, with its own boundary hook.
fill_set!(p::MHDProblem, i::Int, t) =
    fill_ghosts!(p.sets[i], p.schedules[i];
                 boundary=p.boundary === nothing ? nothing : boundary_at(p.boundary[i], t))

# --- the kernels ------------------------------------------------------------------

# Step 3: the metric at every stored cell, `B` averaged to it, the
# hydrodynamic recovery, and all of `P`. `q` is written at the owned cells by
# the next kernel; here it is zero.
@kernel function primitive_kernel!(Pw, @Const(Uw), @Const(B1), @Const(B2), @Const(B3),
                                   @Const(origins), @Const(spacings), provider, t, eos, opts,
                                   pol, ::Val{D}, ::Val{GP}, ::Val{GU},
                                   ::Val{GB}) where {D,GP,GU,GB}
    I = @index(Global, NTuple)                       # stored: `stored = true`
    b = I[D + 1]
    s = ntuple(e -> I[e], Val(D))
    n = ntuple(e -> s[e] - GP[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    cons = read_vars(Uw, stored(n, GU), b, Val(NU))
    𝔅 = cell_𝔅((B1, B2, B3), n, b, GB, Val(D))
    B = 𝔅 / m.sqrtγ
    E = cons_𝔈(cons) / m.sqrtγ
    prim, flags = con2prim_hydro(eos, opts, pol, m, cons, E, B, state_guesses(cons))
    Pc = pack_P(prim, flags, E, B, zero(h), m)
    @inbounds for v in 1:NP
        Pw[s..., v, b] = Pc[v]
    end
end

# Step 4: `q = ∇·E = (1/√γ) ∂ᵢ𝔈ⁱ` at the owned cells, by centered differences
# of the ghost-filled cell-centered `𝔈` ("Maxwell" in `CODE.md`: the charge is
# derived, not evolved).
@kernel function charge_kernel!(Pw, @Const(Uw), @Const(spacings), ::Val{D}, ::Val{GP},
                                ::Val{GU}) where {D,GP,GU}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    div = sum(ntuple(Val(D)) do d
        @inbounds (Uw[stored(shift(n, d, 1), GU)..., U_E + d - 1, b] -
                   Uw[stored(shift(n, d, -1), GU)..., U_E + d - 1, b])
    end) / (2 * h)
    @inbounds Pw[stored(n, GP)..., P_Q, b] = div / Pw[stored(n, GP)..., P_SQRTG, b]
end

# Step 6, the fluxes: reconstruction and the Riemann solve at every face
# normal to `d` of the closed range — a block computes the flux on both of its
# own faces, and making the two sides of a coarse-fine face agree is
# `restrict_interfaces!`'s job.
@kernel function flux_kernel!(Fw, @Const(Pw), @Const(Bd), @Const(origins), @Const(spacings),
                              provider, t, eos, ::Val{D}, ::Val{d}, ::Val{GF}, ::Val{GP},
                              ::Val{GB}, lim::Val, solver::Val) where {D,d,GF,GP,GB}
    I = @index(Global, NTuple)                       # local: `closed = true`
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(face_staggers(Val(D), Val(d))))
    m = point_metric(provider, oftype(h, t), x)

    P₋₂ = read_vars(Pw, stored(shift(n, d, -2), GP), b, Val(NP))
    P₋₁ = read_vars(Pw, stored(shift(n, d, -1), GP), b, Val(NP))
    P₀ = read_vars(Pw, stored(n, GP), b, Val(NP))
    P₊₁ = read_vars(Pw, stored(shift(n, d, 1), GP), b, Val(NP))
    r₋₂, r₋₁, r₀, r₊₁ = recon_vars(P₋₂), recon_vars(P₋₁), recon_vars(P₀), recon_vars(P₊₁)
    rL = limit_face(eos, extrapolate(lim, r₋₂, r₋₁, r₀, 1), r₋₁)
    rR = limit_face(eos, extrapolate(lim, r₋₁, r₀, r₊₁, -1), r₀)

    𝔅n = @inbounds Bd[stored(n, GB)..., 1, b]
    L = face_state(eos, m, rL, 𝔅n, prim_ueos(P₋₁), Val(d))
    R = face_state(eos, m, rR, 𝔅n, prim_ueos(P₀), Val(d))
    F = riemann_flux(solver, m, L, R, Val(d))
    f = stored(n, GF)
    @inbounds for v in 1:NU
        Fw[f..., v, b] = F[v]
    end
end

# The electric field of cell `c` extrapolated to its corner on the sides
# `σa`, `σb` (`+1` upper, `−1` lower) along the present directions `a`, `b`.
@inline function corner_E(Pw, c::NTuple{D,Int}, b::Int, ::Val{GP}, lim::Val, ::Val{a}, σa::Int,
                          ::Val{bb}, σb::Int) where {D,GP,a,bb}
    E = read_E(Pw, stored(c, GP), b)
    if a ≤ D
        Em = read_E(Pw, stored(shift(c, a, -1), GP), b)
        Ep = read_E(Pw, stored(shift(c, a, 1), GP), b)
        E = E + σa * map((l, r) -> slope(lim, l, r), E - Em, Ep - E) / 2
    end
    if bb ≤ D
        c0 = read_E(Pw, stored(c, GP), b)
        Em = read_E(Pw, stored(shift(c, bb, -1), GP), b)
        Ep = read_E(Pw, stored(shift(c, bb, 1), GP), b)
        E = E + σb * map((l, r) -> slope(lim, l, r), c0 - Em, Ep - c0) / 2
    end
    return E
end

# A face-normal field reconstructed to the edge along `e` from either side:
# the faces at `n − e_e` (left) and `n` (right) of a set that is cell-like
# along `e`, or the one face value twice where `e` is a missing direction.
@inline function edge_pair(Bw, n::NTuple{D,Int}, b::Int, ::Val{G}, lim::Val,
                           ::Val{e}) where {D,G,e}
    if e ≤ D
        q₋₂ = @inbounds Bw[stored(shift(n, e, -2), G)..., 1, b]
        q₋₁ = @inbounds Bw[stored(shift(n, e, -1), G)..., 1, b]
        q₀ = @inbounds Bw[stored(n, G)..., 1, b]
        q₊₁ = @inbounds Bw[stored(shift(n, e, 1), G)..., 1, b]
        L = q₋₁ + slope(lim, q₋₁ - q₋₂, q₀ - q₋₁) / 2
        R = q₀ - slope(lim, q₀ - q₋₁, q₊₁ - q₀) / 2
        return L, R
    else
        q = @inbounds Bw[stored(n, G)..., 1, b]
        return q, q
    end
end

# The cell on side `σ` (`−1` lower, `+1` upper) of the vertex-like index `n`
# along `e`: `n − e_e` or `n`; `n` itself along a missing direction.
@inline side_cell(n::NTuple{D,Int}, ::Val{e}, σ::Int) where {D,e} =
    e ≤ D && σ < 0 ? shift(n, e, -1) : n

# Step 6, the EMF: the UCT–HLL average of the four states around each edge
# along `j` of the closed range ("The electromotive force for `A`" in
# `CODE.md`, `src/emf.jl`). With `(j, a, b)` cyclic, the four states are the
# cells on either side along `a` and along `b` (one cell along a missing
# direction), each with `E` extrapolated to the edge; `𝔅ᵃ` is reconstructed
# along `b` from the `a`-faces and `𝔅ᵇ` along `a` from the `b`-faces. The
# fields are reconstructed densitized and divided by the edge's `√γ`.
@kernel function emf_kernel!(Ew, @Const(Pw), @Const(Ba), @Const(Bb), @Const(origins),
                             @Const(spacings), provider, t, ::Val{D}, ::Val{j}, ::Val{GE},
                             ::Val{GP}, ::Val{GBa}, ::Val{GBb},
                             lim::Val) where {D,j,GE,GP,GBa,GBb}
    I = @index(Global, NTuple)                       # local: `closed = true`
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    a_ = mod1(j + 1, 3)
    b_ = mod1(j + 2, 3)
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(edge_staggers(Val(D), Val(j))))
    m = point_metric(provider, oftype(h, t), x)

    cell(σa, σb) = side_cell(side_cell(n, Val(a_), σa), Val(b_), σb)
    E_LL = corner_E(Pw, cell(-1, -1), b, Val(GP), lim, Val(a_), 1, Val(b_), 1)
    E_LR = corner_E(Pw, cell(-1, 1), b, Val(GP), lim, Val(a_), 1, Val(b_), -1)
    E_RL = corner_E(Pw, cell(1, -1), b, Val(GP), lim, Val(a_), -1, Val(b_), 1)
    E_RR = corner_E(Pw, cell(1, 1), b, Val(GP), lim, Val(a_), -1, Val(b_), -1)
    𝔅a_L, 𝔅a_R = edge_pair(Ba, n, b, Val(GBa), lim, Val(b_))
    𝔅b_L, 𝔅b_R = edge_pair(Bb, n, b, Val(GBb), lim, Val(a_))
    isg = inv(m.sqrtγ)
    emf = uct_hll_emf(m, (E_LL, E_LR, E_RL, E_RR), (𝔅a_L * isg, 𝔅a_R * isg),
                      (𝔅b_L * isg, 𝔅b_R * isg), Val(j))
    @inbounds Ew[stored(n, GE)..., 1, b] = emf
end

# `−Σ_d (F_d[n + e_d] − F_d[n])`, written out per dimension so that every
# tuple index is a constant (TreeHydro, measured in its step 14).
@inline face_difference(F, c, v, b, ::Val{d}) where {d} =
    @inbounds F[shift(c, d, 1)..., v, b] - F[c..., v, b]
@inline flux_difference_sum(z, F::NTuple{1,Any}, c, v, b) =
    z + face_difference(F[1], c, v, b, Val(1))
@inline flux_difference_sum(z, F::NTuple{2,Any}, c, v, b) =
    z + face_difference(F[1], c, v, b, Val(1)) + face_difference(F[2], c, v, b, Val(2))
@inline flux_difference_sum(z, F::NTuple{3,Any}, c, v, b) =
    z + face_difference(F[1], c, v, b, Val(1)) + face_difference(F[2], c, v, b, Val(2)) +
    face_difference(F[3], c, v, b, Val(3))

# Step 7, the matter: `du = −Σ_d ΔF_d/h + S`, the explicit sources from the
# cell's `P` and the provider's metric derivatives (`src/sources.jl`).
@kernel function matter_rhs_kernel!(du, @Const(Pw), Fs, @Const(origins), @Const(spacings),
                                    provider, t, ::Val{D}, ::Val{GP},
                                    ::Val{GF}) where {D,GP,GF}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    Pc = read_vars(Pw, stored(n, GP), b, Val(NP))
    m = aux_metric(Pc)
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    dm = metric_gradient(provider, oftype(h, t), x)
    S = explicit_source(m, dm, Pc, aux_E(Pc), aux_B(Pc), aux_q(Pc))
    c = stored(n, GF)
    for v in 1:NU
        acc = flux_difference_sum(zero(h), Fs, c, v, b)
        @inbounds du[n..., v, b] = -acc / h + S[v]
    end
end

# The covariant potential averaged to the vertex `v`: `A_k` is cell-like along
# `k`, so along a present `k` it is the mean of the two edges that meet there.
@inline function vertex_A(A::NTuple{3,Any}, v::NTuple{D,Int}, b::Int,
                          GA::NTuple{3,Any}) where {D}
    return SVector(ntuple(Val(3)) do k
        @inbounds begin
            q = A[k][stored(v, GA[k])..., 1, b]
            k ≤ D ? (A[k][stored(shift(v, k, -1), GA[k])..., 1, b] + q) / 2 : q
        end
    end)
end

# The gauge potential `αΦ − βᵏA_k` at the vertex `v` (`src/emf.jl`).
@inline function vertex_ψ(A::NTuple{3,Any}, Φw, v::NTuple{D,Int}, b::Int, GA::NTuple{3,Any},
                          ::Val{GΦ}, o, h, provider, t) where {D,GΦ}
    x = point_x(o, h, v, Val(vertex_staggers(Val(D))))
    m = point_metric(provider, t, x)
    Φ̃ = @inbounds Φw[stored(v, GΦ)..., 1, b]
    return gauge_potential(m, Φ̃, vertex_A(A, v, b, GA))
end

# Step 7, the potential: `∂ₜA_j = −EMF_j − ∂_j(αΦ − βᵏA_k)` at the owned
# edges along `j`. The edge `n` lies between the vertices `n` and `n + e_j`;
# along a missing direction there is no gradient.
@kernel function potential_rhs_kernel!(du, @Const(EMFj), @Const(A1), @Const(A2), @Const(A3),
                                       @Const(Φw), @Const(origins), @Const(spacings), provider,
                                       t, ::Val{D}, ::Val{j}, ::Val{GE}, ::Val{GA},
                                       ::Val{GΦ}) where {D,j,GE,GA,GΦ}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    e = @inbounds EMFj[stored(n, GE)..., 1, b]
    if j ≤ D
        o = @inbounds origins[b]
        tt = oftype(h, t)
        A = (A1, A2, A3)
        ψhi = vertex_ψ(A, Φw, shift(n, j, 1), b, GA, Val(GΦ), o, h, provider, tt)
        ψlo = vertex_ψ(A, Φw, n, b, GA, Val(GΦ), o, h, provider, tt)
        @inbounds du[n..., 1, b] = -e - (ψhi - ψlo) / h
    else
        @inbounds du[n..., 1, b] = -e
    end
end

# The gauge flux `α√γ Aʲ − βʲΦ̃` at the edge `c` along `j`: `A_j` is native
# there; `A_k`, `k ≠ j`, is vertex-like along `j` (mean of `c` and `c + e_j`)
# and cell-like along `k` (mean of `c − e_k` and `c` where `k` is present);
# `Φ̃` is vertex-like along `j`.
@inline function gauge_edge_flux(A::NTuple{3,Any}, Φw, c::NTuple{D,Int}, b::Int,
                                 GA::NTuple{3,Any}, ::Val{GΦ}, o, h, provider, t,
                                 ::Val{j}) where {D,GΦ,j}
    x = point_x(o, h, c, Val(edge_staggers(Val(D), Val(j))))
    m = point_metric(provider, t, x)
    cj = shift(c, j, 1)
    Ae = SVector(ntuple(Val(3)) do k
        @inbounds begin
            if k == j
                A[k][stored(c, GA[k])..., 1, b]
            elseif k ≤ D
                (A[k][stored(shift(c, k, -1), GA[k])..., 1, b] + A[k][stored(c, GA[k])..., 1, b] +
                 A[k][stored(shift(cj, k, -1), GA[k])..., 1, b] +
                 A[k][stored(cj, GA[k])..., 1, b]) / 4
            else
                (A[k][stored(c, GA[k])..., 1, b] + A[k][stored(cj, GA[k])..., 1, b]) / 2
            end
        end
    end)
    Φ̃e = @inbounds (Φw[stored(c, GΦ)..., 1, b] + Φw[stored(cj, GΦ)..., 1, b]) / 2
    return gauge_flux(m, Ae, Φ̃e, Val(j))
end

# Step 7, the gauge: `∂ₜΦ̃ = −∂_j(α√γ Aʲ − βʲΦ̃) − ξαΦ̃` at the owned
# vertices, the flux on the `j`-edges `v − e_j` and `v` on either side.
@kernel function gauge_rhs_kernel!(du, @Const(A1), @Const(A2), @Const(A3), @Const(Φw),
                                   @Const(origins), @Const(spacings), provider, t, ξ,
                                   ::Val{D}, ::Val{GA}, ::Val{GΦ}) where {D,GA,GΦ}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    v = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    o = @inbounds origins[b]
    tt = oftype(h, t)
    A = (A1, A2, A3)
    div = sum(ntuple(Val(D)) do j
        gauge_edge_flux(A, Φw, v, b, GA, Val(GΦ), o, h, provider, tt, Val(j)) -
        gauge_edge_flux(A, Φw, shift(v, j, -1), b, GA, Val(GΦ), o, h, provider, tt, Val(j))
    end)
    x = point_x(o, h, v, Val(vertex_staggers(Val(D))))
    m = point_metric(provider, tt, x)
    Φ̃ = @inbounds Φw[stored(v, GΦ)..., 1, b]
    @inbounds du[v..., 1, b] = -div / h + gauge_damping(oftype(h, ξ), m, Φ̃)
end

# The implicit stage of Ohm's law at the owned cells (step 3's
# `solve_ohm_stage`), writing only `𝔈` into `U`, and tallying the outcome
# into `X`. `U` enters holding `u★`.
@kernel function ohm_kernel!(Uw, @Const(ustar), @Const(B1), @Const(B2), @Const(B3), Xw,
                             @Const(origins), @Const(spacings), provider, t, γΔt, eos, opts,
                             pol, model, stage, ::Val{D}, ::Val{GB}, ::Val{GX}) where {D,GB,GX}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    cons★ = read_vars(ustar, n, b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E★ = cons_𝔈(cons★) * isg
    E, _, r, nrec = solve_ohm_stage(eos, opts, pol, model, m, cons★, E★, B, oftype(h, γΔt),
                                    state_guesses(cons★); stage=stage)
    𝔈 = m.sqrtγ * E
    @inbounds for k in 1:3
        Uw[n..., U_E + k - 1, b] = 𝔈[k]
    end
    xi = stored(n, GX)
    @inbounds begin
        for k in 1:N_OHM_BITS
            Xw[xi..., X_OHM + k - 1, b] += (r.flags >> (k - 1)) & one(UInt32)
        end
        if r.policy_flags != 0
            Xw[xi..., X_POLICY, b] += one(h)
            Xw[xi..., X_POLICY_OR, b] = unsafe_trunc(UInt32, Xw[xi..., X_POLICY_OR, b]) |
                                        r.policy_flags
        end
        Xw[xi..., X_RECOVERIES, b] += nrec
        Xw[xi..., X_MAX_RECOVERIES, b] = max(Xw[xi..., X_MAX_RECOVERIES, b], nrec)
    end
end

# The reset at the owned cells: recover, and where the policy repaired the
# state, write the repaired state's matter variables back — all but `𝔈`,
# which the reset does not touch. Counted into `X`: as the stage limiter
# (`STEP = false`), the cells written; as the step limiter, the cells
# written and the change they made to `√γ D` and `√γ τ`, which is what the
# reset injects into the state (step 6).
@kernel function reset_kernel!(uw, @Const(B1), @Const(B2), @Const(B3), Xw, @Const(origins),
                               @Const(spacings), provider, t, eos, opts, pol, ::Val{D},
                               ::Val{GB}, ::Val{GX}, ::Val{STEP}) where {D,GB,GX,STEP}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    cons = read_vars(uw, n, b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E = cons_𝔈(cons) * isg
    prim, flags = con2prim_hydro(eos, opts, pol, m, cons, E, B, state_guesses(cons))
    if flags & EntropyEOS.FLAG_POL_ANY != 0
        c = prim2con(eos, m, prim, E, B)
        @inbounds for v in 1:(U_E - 1)
            uw[n..., v, b] = c[v]
        end
        xi = stored(n, GX)
        @inbounds if STEP
            Xw[xi..., X_STEP_RESETS, b] += one(h)
            Xw[xi..., X_INJECTED_D, b] += c[U_D] - cons[U_D]
            Xw[xi..., X_INJECTED_TAU, b] += c[U_TAU] - cons[U_TAU]
        else
            Xw[xi..., X_RESETS, b] += one(h)
        end
    end
end

# The fastest coordinate speed of light at the owned cells, over the present
# directions, into `X`: `max_d max |−βᵈ ± α√γ^{dd}|`.
@kernel function light_kernel!(Xw, @Const(origins), @Const(spacings), provider, t, ::Val{D},
                               ::Val{GX}) where {D,GX}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    λ = maximum(ntuple(Val(D)) do d
        λm, λp = light_cone(m, Val(d))
        max(abs(λm), abs(λp))
    end)
    @inbounds Xw[stored(n, GX)..., X_LIGHT, b] = λ
end

# --- the passes -------------------------------------------------------------------

GB_of(f::MHDFields) = (f.B[1].G, f.B[2].G, f.B[3].G)
GA_of(f::MHDFields) = (f.A[1].G, f.A[2].G, f.A[3].G)

"""
    update_fields!(p::MHDProblem, u, t)

`𝔅 = curl A + 𝔅̄` on every stored face, from the potential held in the
state vector `u`: `scatter!` and `fill_ghosts!` of the three `A` sets, then
[`curl_A!`](@ref). What the implicit stage and the reset need of the field —
`B` is explicit in a stage, and the stage's `A` is its predictor's.
"""
function update_fields!(p::MHDProblem, u, t)
    for j in 1:3
        i = SEG_A + j - 1
        scatter!(p.sets[i], stateview(u, p.sets, i))
        fill_set!(p, i, t)
    end
    curl_A!(p.f, p.𝔅̄, p.spacings)
    return p
end

"""
    update_primitives!(p::MHDProblem, u, t) -> P

Steps 1 to 4 of [`mhd_rhs!`](@ref) and no further: scatter `u` into every
evolved set and fill their ghosts, compute `𝔅` on every stored face, recover
`P` in every stored cell, and `q` at the owned cells. Returns the primitive
set `P`.
"""
function update_primitives!(p::MHDProblem{T,D}, u, t) where {T,D}
    f = p.f
    scatter!(p.sets, u)
    for i in eachindex(p.sets)
        fill_set!(p, i, t)
    end
    curl_A!(f, p.𝔅̄, p.spacings)
    map_blocks!(primitive_kernel!, f.P, f.P.work, f.U.work, f.B[1].work, f.B[2].work,
                f.B[3].work, p.origins, p.spacings, p.provider, t, p.eos, p.opts, p.pol,
                Val(D), Val(f.P.G), Val(f.U.G), Val(GB_of(f)); stored=true)
    map_blocks!(charge_kernel!, f.P, f.P.work, f.U.work, p.spacings, Val(D), Val(f.P.G),
                Val(f.U.G))
    return f.P
end

"""
    mhd_rhs!(du, u, p::MHDProblem, t)

The explicit right-hand side of every evolved variable, in the order "The
right-hand side" in `CODE.md` sets out (see the top of `src/evolution.jl`):
the matter's flux divergence and explicit sources, `A`'s EMF and gauge
gradient, and `Φ̃`'s gauge equation. The conduction current, the one
implicit term, is not here: it is [`ohm_solve!`](@ref)'s.

**Pure in `(u, t)`, and it never mutates `u`**: every working array is
scratch, rewritten in full, and every recovery is seeded from `u` itself
([`state_guesses`](@ref)). IMEXRungeKutta's `f_exp!`.
"""
function mhd_rhs!(du, u, p::MHDProblem{T,D}, t) where {T,D}
    f = p.f
    update_primitives!(p, u, t)
    GB = GB_of(f)
    ntuple(Val(D)) do d
        Fd = f.F[d]
        map_blocks!(flux_kernel!, Fd, Fd.work, f.P.work, f.B[d].work, p.origins, p.spacings,
                    p.provider, t, p.eos, Val(D), Val(d), Val(Fd.G), Val(f.P.G), Val(GB[d]),
                    p.limiter, p.solver; closed=true)
        p.fixup && restrict_interfaces!(Fd, p.ischeds_F[d])
        nothing
    end
    ntuple(Val(3)) do j
        a, b = mod1(j + 1, 3), mod1(j + 2, 3)
        Ej = f.EMF[j]
        map_blocks!(emf_kernel!, Ej, Ej.work, f.P.work, f.B[a].work, f.B[b].work, p.origins,
                    p.spacings, p.provider, t, Val(D), Val(j), Val(Ej.G), Val(f.P.G),
                    Val(GB[a]), Val(GB[b]), p.limiter; closed=true)
        (p.fixup && p.ischeds_EMF[j] !== nothing) &&
            restrict_interfaces!(Ej, p.ischeds_EMF[j])
        nothing
    end
    map_blocks!(matter_rhs_kernel!, f.U, statearray(du, p.sets, SEG_U), f.P.work,
                map(F -> F.work, f.F), p.origins, p.spacings, p.provider, t, Val(D),
                Val(f.P.G), Val(first(f.F).G))
    GA = GA_of(f)
    ntuple(Val(3)) do j
        i = SEG_A + j - 1
        map_blocks!(potential_rhs_kernel!, p.sets[i], statearray(du, p.sets, i),
                    f.EMF[j].work, f.A[1].work, f.A[2].work, f.A[3].work, f.Φ.work,
                    p.origins, p.spacings, p.provider, t, Val(D), Val(j), Val(f.EMF[j].G),
                    Val(GA), Val(f.Φ.G))
        nothing
    end
    map_blocks!(gauge_rhs_kernel!, f.Φ, statearray(du, p.sets, SEG_Φ), f.A[1].work,
                f.A[2].work, f.A[3].work, f.Φ.work, p.origins, p.spacings, p.provider, t,
                p.ξ, Val(D), Val(GA), Val(f.Φ.G))
    return nothing
end

"""
    ohm_solve!(U, u★, γΔt, p::MHDProblem, t)

IMEXRungeKutta's `solve_imp!`: `U = u★ + γΔt g(U)` with `g` the conduction
current in Ampère's law, solved per owned cell by
[`solve_ohm_stage`](@ref) with the stage's own `B` (from `u★`'s `A`, which
the implicit part does not change). `U` holds a copy of `u★` on entry, and
only its `𝔈` is written ("The implicit stage solve" in `CODE.md`); the
recovery's repair of an inadmissible `u★` stays inside the solve. Each
cell's outcome is tallied into the diagnostics set ([`ohm_tally`](@ref)).
"""
function ohm_solve!(U, ustar, γΔt, p::MHDProblem{T,D}, t) where {T,D}
    f = p.f
    update_fields!(p, ustar, t)
    map_blocks!(ohm_kernel!, f.U, statearray(U, p.sets, SEG_U),
                statearray(ustar, p.sets, SEG_U), f.B[1].work, f.B[2].work, f.B[3].work,
                f.X.work, p.origins, p.spacings, p.provider, t, γΔt, p.eos, p.opts, p.pol,
                p.model, p.stage, Val(D), Val(GB_of(f)), Val(f.X.G))
    return nothing
end

"""
    reset_atmosphere!(u, integ, p::MHDProblem, t)

The reset, with the limiter signature `(u, integrator, p, t)`: recover every
owned cell of `u` and, where EntropyEOS's policy repaired the state (the
atmosphere, a ceiling, the entropy window, the rapidity cap, non-finite
input), write the repaired state's matter variables back into `u`; `𝔈` is
not touched. Passed to the integrator as both the stage limiter and the step
limiter ("Floors, the atmosphere and the magnetosphere" in `CODE.md`). The
cells it writes are counted into the diagnostics set.

*(Proposed in step 5:)* any policy repair is written back, not the
atmosphere alone: every repaired state is exactly invertible, and is what
the next right-hand side would recover anyway.

This method counts its writes as the stage limiter's; [`reset_step!`](@ref)
is the same reset in the step limiter's role.
"""
reset_atmosphere!(u, integ, p::MHDProblem, t) = reset!(u, p, t, Val(false))

"""
    reset_step!(u, integ, p::MHDProblem, t)

[`reset_atmosphere!`](@ref) as the step limiter: the same writes, counted
apart, with what they change in `√γ D` and `√γ τ` summed per cell — the rest
mass and the energy the reset injects into the state, since only the step
limiter's writes reach `uⁿ⁺¹` ([`ohm_tally`](@ref); added in step 6).
"""
reset_step!(u, integ, p::MHDProblem, t) = reset!(u, p, t, Val(true))

function reset!(u, p::MHDProblem{T,D}, t, step::Val) where {T,D}
    f = p.f
    update_fields!(p, u, t)
    map_blocks!(reset_kernel!, f.U, statearray(u, p.sets, SEG_U), f.B[1].work, f.B[2].work,
                f.B[3].work, f.X.work, p.origins, p.spacings, p.provider, t, p.eos, p.opts,
                p.pol, Val(D), Val(GB_of(f)), Val(f.X.G), step)
    return nothing
end

# --- diagnostics ------------------------------------------------------------------

"""
    clear_tally!(p::MHDProblem)

Zero the per-cell tallies of the diagnostics set, at the start of a chunk.
"""
clear_tally!(p::MHDProblem) = (p.f.X.work .= 0; p)

"""
    ohm_tally(p::MHDProblem) -> NamedTuple

The implicit stages and resets since the last [`clear_tally!`](@ref), summed
over the owned cells ("The implicit stage solve" in `CODE.md`: each outcome
flagged, counted, and reduced per chunk):

- `bits`: for each of the nine `OhmStageResult` flag bits, in bit order
  (`OHM_SIGMA_ZERO` … `OHM_REPAIRED`), the number of cell-stages that set it;
- `flags`: the `|` of every flag bit seen;
- `policy`, `policy_flags`: the cell-stages whose returned state the policy
  repaired, and the `|` of their policy flags;
- `recoveries`, `max_recoveries`: the recoveries the stages made, and the
  most one stage made;
- `resets`: the cells the reset wrote back as the stage limiter;
- `step_resets`: the cells it wrote back as the step limiter (added in step
  6);
- `injected_D`, `injected_τ`: the volume integrals `Σ hᴰ Δ(√γ D)` and
  `Σ hᴰ Δ(√γ τ)` of those step-limiter writes — the rest mass and the
  energy the reset put into the state (negative where it took some out).

The counts are integers carried in the set's float type, exact below `2^24`
in `Float32`.
"""
function ohm_tally(p::MHDProblem{T,D}) where {T,D}
    X = p.f.X
    count(v) = roundint(mesh_mapreduce(identity, +, zero(T), X; vars=v))
    bits = ntuple(k -> count(X_OHM + k - 1), N_OHM_BITS)
    flags = UInt32(0)
    for k in 1:N_OHM_BITS
        bits[k] > 0 && (flags |= UInt32(1) << (k - 1))
    end
    policy_flags = mesh_mapreduce(x -> unsafe_trunc(UInt32, x), |, UInt32(0), X;
                                  vars=X_POLICY_OR)
    forest = X.forest
    volume(v) = mesh_mapreduce(identity, +, zero(T), X; vars=v,
                               weight=key -> spacing(T, forest, key)^D)
    return (; bits, flags, policy=count(X_POLICY), policy_flags,
            recoveries=count(X_RECOVERIES),
            max_recoveries=roundint(mesh_mapreduce(identity, max, zero(T), X;
                                                   vars=X_MAX_RECOVERIES)),
            resets=count(X_RESETS), step_resets=count(X_STEP_RESETS),
            injected_D=volume(X_INJECTED_D), injected_τ=volume(X_INJECTED_TAU))
end

"""
    max_light_rate(p::MHDProblem, t) -> max λ/h

The fastest coordinate speed of light anywhere on the mesh, over the present
directions, divided by the local spacing: the time step is
`cfl / (D · max λ/h)` ("Time integration" in `CODE.md`: `Δt` from the
light-speed CFL on the finest level). Overwrites the light-speed slot of the
diagnostics set.
"""
function max_light_rate(p::MHDProblem{T,D}, t) where {T,D}
    X = p.f.X
    map_blocks!(light_kernel!, X, X.work, p.origins, p.spacings, p.provider, T(t), Val(D),
                Val(X.G))
    forest = X.forest
    return mesh_mapreduce(identity, max, zero(T), X; vars=X_LIGHT,
                          weight=key -> inv(spacing(T, forest, key)))
end

"""
    conserved_totals(p::MHDProblem, u) -> NTuple{10}

The domain integral `Σ hᴰ U_v` of each of the ten cell variables of the
state vector `u`: rest mass `√γ D`, the total momentum `√γ S_i`, the total
energy `√γ τ`, `√γ D Yₑ`, `√γ D s`, and `𝔈`. Through TreeAMR's
volume-weighted `mesh_mapreduce`, combined in block order.
"""
function conserved_totals(p::MHDProblem{T,D}, u) where {T,D}
    U = p.sets[SEG_U]
    forest = U.forest
    uU = stateview(u, p.sets, SEG_U)
    return ntuple(v -> mesh_mapreduce(identity, +, zero(T), U, uU; vars=v,
                                      weight=key -> spacing(T, forest, key)^D), Val(NU))
end

"""
    conserved_scales(p::MHDProblem, u) -> NTuple{10}

`Σ hᴰ |U_v|` for each cell variable: the scale a drift in
[`conserved_totals`](@ref) is rounding against.
"""
function conserved_scales(p::MHDProblem{T,D}, u) where {T,D}
    U = p.sets[SEG_U]
    forest = U.forest
    uU = stateview(u, p.sets, SEG_U)
    return ntuple(v -> mesh_mapreduce(abs, +, zero(T), U, uU; vars=v,
                                      weight=key -> spacing(T, forest, key)^D), Val(NU))
end
