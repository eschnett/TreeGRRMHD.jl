# The right-hand side, the implicit stage and the reset: the kernels that
# write the field sets, and the order they run in ("The right-hand side" in
# `CODE.md`).
#
#   1. `scatter!` the multi-set state; `fill_ghosts!` the potentials.
#   2. `𝔅 = curl A + 𝔅̄` on every stored face (`src/curl.jl`).
#   3. At the owned cells the metric, `B` averaged to the cell, and the
#      hydrodynamic recovery, into `P` and the primitive set `W`; `W`'s
#      ghost fill (copies at same-level faces, the bounded conservative
#      prolongation centred on the parent at refinement boundaries); at every
#      ghost cell `P` from `W` by forward EOS evaluation, with the ghost's own
#      `B` — no ghost is recovered, and the conserved ghosts are not written
#      (amended in step 8, and 2026-10-23).
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
# indices directly ("Sharp edges" in `CODE.md`). A cell `n` lies between the
# vertices `n` and `n + 1`, so a face normal to `d` with local index `n` lies
# between the cells `n − e_d` and `n`. Every kernel receives one `Val(G)` per
# set it reads. A state-vector array (`statearray`) has no ghosts, so a local
# owned index is its index.
#
# **Warm starts come from the state, or from the step** (proposed in step 5,
# amending "Thermodynamics" in `CODE.md`, which carried them in `P`; amended
# 2026-10-24, decided by Erik 2026-10-23): outside a step every recovery the
# scheme makes is seeded with `s = (√γ D s)/(√γ D)`, the advected entropy the
# state carries for this purpose, and a cold rapidity and EOS warm start, so
# that the callbacks are functions of `(u, t)` bit for bit. Inside a step of
# `evolve!` the **step cache** (`MHDProblem`'s `cache`, one entry per owned
# cell) holds the cell's latest recovery in the step: every recovery is
# seeded from it where it holds one, and a recovery whose EntropyEOS input is
# the cached one bit for bit reuses its output. The cache is invalidated at
# the end of every step (the step limiter) and whenever `evolve!` starts or
# stops stepping, so a step is a pure function of `uⁿ`, `t` and `Δt`: one
# chunk or two, a restart from a saved state, one thread or four reproduce
# the run bit for bit ("The step cache" in `CODE.md`). A warm start carried
# across steps, or in `P`, would make a recovery depend, at the solver's
# tolerance, on what was evaluated before, and a restart from a checkpoint
# (which holds neither) would not reproduce the run.

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

# --- the step cache in a kernel -----------------------------------------------------

# The owned cell's step-cache entry where the cache is active, an invalid
# one otherwise: then nothing is reused and every recovery is seeded from the
# state, today's recovery bit for bit.
@inline read_entry(Cw, ci::NTuple, b::Int, active::Bool, ::Type{T}) where {T} =
    active ? read_vars(Cw, ci, b, Val(NC)) : invalid_entry(T)

@inline function write_entry!(Cw, ci::NTuple, b::Int, e::NTuple{NC})
    @inbounds for v in 1:NC
        Cw[ci..., v, b] = e[v]
    end
    return nothing
end

# The seed of a recovery: the entry's, where the cell has recovered in this
# step, the state's otherwise.
@inline seed_of(e::NTuple{NC,T}, cons::NTuple{NU,T}) where {T} =
    entry_valid(e) ? entry_seed(e) : state_guesses(cons)

# Count one recovery at `site` as cold, warm or reused (`X_SEEDS`).
@inline function tally_seed!(Xw, xi::NTuple, b::Int, site::Int, e::NTuple{NC}, hit::Bool)
    k = hit ? 2 : entry_valid(e) ? 1 : 0
    @inbounds Xw[xi..., X_SEEDS + 3 * (site - 1) + k, b] += one(eltype(Xw))
    return nothing
end

# --- the problem ------------------------------------------------------------------

"""The admissible Riemann solvers (`src/riemann.jl`)."""
const MHD_SOLVERS = (:llf, :hlle, :hlle_split)

"""
    MHDProblem(f::MHDFields; eos, opts, proj, pol, model, provider = Flat(), 𝔅̄ = 0,
               ξ = 0, stage = OhmStageOptions{T}(), limiter, riemann = :hlle,
               ops_U = default_ops_U(), ops_A = default_ops_A(), ops_Φ = default_ops_Φ(),
               ops_W = default_ops_W(), boundary = nothing, boundary_W = nothing,
               fixup = true, two_phase = false, interior = nothing)

Everything a right-hand side, an implicit stage and a reset need, built once
per mesh: the field sets `f` ([`mhd_fields`](@ref)), a ghost schedule per
evolved set, the interface schedules of the fluxes and the EMFs, the per-block
origins and spacings on the sets' backend, and the physics:

- `eos`, `opts`, `proj`, `pol`: EntropyEOS's `AbstractEOS`, `Con2PrimOptions`,
  `ProjectionOptions` (the valid range, `ρ_min`, `ρ_max`, `w_cap`) and
  `PolicyOptions` (the atmosphere), in geometric units ("Thermodynamics" and
  "Primitive recovery" in `CODE.md`);
- `model`: the conductivity ([`Vacuum`](@ref), [`Constant`](@ref),
  [`DensityProfile`](@ref), [`IdealConductor`](@ref),
  [`MagnetizationCutoff`](@ref));
- `provider`: the metric ([`Flat`](@ref), [`Analytic`](@ref));
- `𝔅̄`: the constant densitized background field, an `SVector{3}`
  (`src/curl.jl`);
- `ξ`: the gauge damping, `−ξαΦ̃` in the gauge equation (zero by default
  here; [`evolve!`](@ref) chooses it, "Refinement" in `CODE.md`);
- `stage`: the stage solve's [`OhmStageOptions`](@ref).

**`limiter` has no default**, as in TreeHydro: it changes what is measured.
It names the reconstruction ("Reconstruction" in `CODE.md`): PLM with the
slope limiter `:none`, `:minmod` or `:mc` (step 5), or `:ppm`, `:wenoz`,
`:mp5` (step 7), which need `G ≥ 3`. `riemann` is `:hlle` (the light-cone HLL, whose
dissipation on `𝔈` matches the UCT EMF's on `𝔅`) or `:llf`.

`ops_U`, `ops_A`, `ops_Φ` and `ops_W` are the operators of the matter set,
of the three `A_j`, of `Φ̃` and of the primitive set `W` the matter's ghost
fill prolongates ([`default_ops_U`](@ref) and its siblings): Conservative
width 3 for the matter, **bounded** (TreeAMR's MC-bounded prolongation,
AMR-4), which since step 8 acts in its regrid only; the same operator for
`W`, the parent-centred primitive fill (decided by Erik 2026-10-23); for
`A_j` the conservative family on a staggered set (AMR-9), width 3 along the
edge and point values of width 4 across it, so that the magnetic flux
through a coarse face is the sum through its fine faces; and PointValue
width 4 for `Φ̃` (proposed in step 5; decided in step 8 by the
interface-order table, and the matter's bound by the atmosphere across a
coarse-fine face — "Refinement" in `CODE.md`).
`boundary` is `nothing` for a periodic domain, or one hook per evolved set in
the order of [`evolved_sets`](@ref) (AMR-2 having landed, each set takes its
own): a hook TreeAMR's `fill_ghosts!` takes, or a time-dependent one that
[`boundary_at`](@ref) turns into such a hook at the time of each fill — the
case's Dirichlet data, [`DirichletBoundary`](@ref) (step 6). `boundary_W` is
the hook of `W`, by default derived from the matter's
([`primitive_boundary`](@ref)). `fixup = false`
skips `restrict_interfaces!`, the negative control of step 8.

The problem also owns the **step cache** *(added 2026-10-24; "The step
cache" in `CODE.md`)*: a cell set over the owned cells, `G = 0`, of
[`NC`](@ref) slots per cell — EntropyEOS's input to the cell's latest
recovery in the current step and its output — built afresh, invalid, with
the problem (so a [`rebuild`](@ref) after a regrid sizes it to the new mesh,
and nothing is transferred), and a host-side switch that only
[`with_step_cache`](@ref) turns on. While it is off, which is always outside
`evolve!`'s steps, no kernel reads or writes the cache.

`two_phase` *(added 2026-10-30, stage 4 of the recovery pipeline; "The
two-phase recovery" in `CODE.md`)*: the recoveries outside the stage solve's
iterations — the right-hand side's, both limiters', the stage predictor's —
run as three passes over the owned cells, `con2prim_safe_start`,
`con2prim_safe_finish` where the first was not done, and level 2 with the
site's own work, through a per-owned-cell hand-off ([`Handoff`](@ref)) the
problem owns, built and sized with it like the step cache. `two_phase =
false`, the default, runs each of them fused, in one kernel, with no
hand-off: the same results bit for bit (a test). *(Decided by Erik
2026-11-02:)* fused is the default, since on the CPU the split costs 1–10%
a step and only 0–2.4% of the recoveries reach the second phase; the split
stays, tested against fused, for a GPU step to turn on.

`interior` *(added in step 9; "The interior" in `CODE.md`)*: `nothing`, or an
[`InteriorTarget`](@ref) — a black hole's interior and the case's solution —
which the reset pastes into the state inside the hole, first, in the roles
the interior names (stage limiter, step limiter or both). Without one the
reset is exactly what it was.
"""
struct MHDProblem{T,D,FS,ST,SC,OU,OA,OΦ,SW,OW,BW,ISF,ISE,OR,SP,OF,EOS,OPT,PRJ,POL,MOD,MET,BC,
                  LIM,RS,FC,HO,IN}
    f::FS
    sets::ST                     # evolved_sets(f): the state vector's segments
    schedules::SC                # one GhostSchedule per evolved set
    ops_U::OU                    # the matter's operators
    ops_A::OA                    # the operators of A_j
    ops_Φ::OΦ                    # the operators of Φ̃
    schedule_W::SW               # the ghost schedule of the primitive set `W` (step 8)
    ops_W::OW                    # its operators, Conservative width 3 bounded
    boundary_W::BW               # its boundary hook, `nothing` on a periodic domain
    ischeds_F::ISF               # NTuple{D,InterfaceSchedule}
    ischeds_EMF::ISE             # NTuple{3}: an InterfaceSchedule, or nothing
    origins::OR                  # per block, on the sets' backend
    spacings::SP
    lowwall::OF                  # per block, its low domain faces (`low_walls`, AMR-8)
    haswalls::Bool               # whether any block has one
    eos::EOS
    opts::OPT
    proj::PRJ
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
    cache::FC                    # the step cache, NC slots per owned cell (2026-10-24)
    cache_on::Base.RefValue{Bool}  # whether it is active: only inside `with_step_cache`
    handoff::HO                  # the two-phase hand-off, or `nothing`: fused (2026-10-30)
    interior::IN                 # a black hole's interior, or `nothing` (step 9)
end

function MHDProblem(f::MHDFields{D}; eos::EntropyEOS.AbstractEOS, opts, proj, pol,
                    model::Conductivity, provider::MetricProvider=Flat(),
                    𝔅̄=nothing, ξ=0, stage=nothing, limiter::Union{Symbol,Nothing}=nothing,
                    riemann::Symbol=:hlle,
                    ops_U::Operators=default_ops_U(), ops_A::Operators=default_ops_A(),
                    ops_Φ::Operators=default_ops_Φ(), ops_W::Operators=default_ops_W(),
                    boundary=nothing, boundary_W=nothing, fixup::Bool=true,
                    two_phase::Bool=false, interior=nothing) where {D}
    T = eltype(f.U.work)
    limiter === nothing && throw(ArgumentError(
        "MHDProblem has no default limiter: pass `limiter`, one of $(MHD_LIMITERS). " *
        "The choice is the case's — :none keeps a clean O(h²) truncation error for a " *
        "convergence study, :minmod and :mc keep a shock monotone and clip a smooth " *
        "extremum while doing it; :ppm, :wenoz and :mp5 are the higher-order methods."))
    limiter in MHD_LIMITERS || throw(ArgumentError(
        "limiter must be one of $(MHD_LIMITERS), got :$limiter: it travels into the " *
        "kernels as a Val, and `face_offset` has a method for each of those and no others."))
    G = f.U.G
    need = stencil_radius(Val(limiter)) + 1
    all(≥(need), G) || throw(ArgumentError(
        "limiter :$limiter reads $(need - 1) cell(s) either side of a cell, so a face " *
        "needs G ≥ $need ghost cells; the sets have G = $G (mhd_fields' `G`)."))
    riemann in MHD_SOLVERS || throw(ArgumentError(
        "riemann must be one of $(MHD_SOLVERS), got :$riemann: `riemann_flux` has a " *
        "method for each of those and no others (CODE.md, \"Riemann solver\")."))
    sets = evolved_sets(f)
    for s in (sets..., f.P, f.W, f.B...)
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
    schedules = ntuple(i -> GhostSchedule(sets[i], i == SEG_U ? ops_U : i == SEG_Φ ? ops_Φ :
                                                   ops_A), length(sets))
    schedule_W = GhostSchedule(f.W, ops_W)
    bW = boundary === nothing ? nothing :
         boundary_W === nothing ? primitive_boundary(boundary[SEG_U]) : boundary_W
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
    lw_host = low_walls(forest, T)
    haswalls = any(any, lw_host)
    lowwall = to_backend(backend, lw_host)
    B̄ = 𝔅̄ === nothing ? zero(SVector{3,T}) : SVector{3,T}(𝔅̄)
    st = stage === nothing ? OhmStageOptions{T}() : stage
    cache = FieldSet{T}(forest, NC; G=0, centering=cellcentered(D), backend=backend)
    cache.work .= T(NaN)                     # invalid: no recovery in a step yet
    w = cache.work
    handoff = two_phase ? Handoff(T, backend, (ntuple(d -> size(w, d), D)..., size(w, ndims(w)))) :
              nothing
    return MHDProblem{T,D,typeof(f),typeof(sets),typeof(schedules),typeof(ops_U),
                      typeof(ops_A),typeof(ops_Φ),typeof(schedule_W),typeof(ops_W),typeof(bW),
                      typeof(ischeds_F),
                      typeof(ischeds_EMF),typeof(origins),typeof(spacings),typeof(lowwall),
                      typeof(eos),typeof(opts),typeof(proj),typeof(pol),typeof(model),
                      typeof(provider),
                      typeof(boundary),limiter,riemann,typeof(cache),typeof(handoff),
                      typeof(interior)}(
        f, sets, schedules, ops_U, ops_A, ops_Φ, schedule_W, ops_W, bW, ischeds_F, ischeds_EMF, origins, spacings, lowwall,
        haswalls,
        eos, opts, proj, pol, model, provider, B̄, T(ξ), st, boundary, fixup, Val(limiter),
        Val(riemann), cache, Ref(false), handoff, interior)
end

"""
    rebuild(p::MHDProblem; kwargs...) -> MHDProblem

The same problem over the same field sets, its schedules, interface
schedules, per-block geometry and outer faces rebuilt from the forest as it
is now: what a [`regrid!`](@ref) leaves stale (it replaces every set's
storage in place, so `p.f` still names the sets). A keyword of
[`MHDProblem`](@ref) given here replaces `p`'s (`fixup = false`, the
negative control). Added in step 8. The new problem has its own step cache,
sized to the forest and invalid, with the switch off (2026-10-24), and its own
two-phase hand-off where `p` has one (2026-10-30).
"""
rebuild(p::MHDProblem; kwargs...) =
    MHDProblem(p.f; eos=p.eos, opts=p.opts, proj=p.proj, pol=p.pol, model=p.model,
               provider=p.provider,
               𝔅̄=p.𝔅̄, ξ=p.ξ, stage=p.stage, limiter=unval(p.limiter),
               riemann=unval(p.solver), ops_U=p.ops_U, ops_A=p.ops_A, ops_Φ=p.ops_Φ,
               ops_W=p.ops_W, boundary=p.boundary, boundary_W=p.boundary_W, fixup=p.fixup,
               two_phase=p.handoff !== nothing, interior=p.interior, kwargs...)

unval(::Val{X}) where {X} = X

"""
    evolved_sets(p::MHDProblem) -> (U, A₁, A₂, A₃, Φ̃)

The problem's evolved sets, over which its state vector is: what a script
makes the state vector from, `statevector(evolved_sets(p))` (added in step
10).
"""
evolved_sets(p::MHDProblem) = p.sets

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

# The ghost fill of evolved set `i` at time `t`, with its own boundary hook,
# which leaves the set in the state the *complete* outer boundary condition
# defines (decided by Erik 2026-10-15: "an outer boundary condition needs to
# be applied always"). The matter's hook is a function of position; a
# potential's is TreeAMR's relative hook (AMR-7), which takes the nearest
# evolved point across a block face from the block's own filled ghost and
# runs each region after the stage that fills what it reads, and the
# potentials' sets have symmetric walls (AMR-8), so the owned low wall at a
# low outer face is filled by the high wall's rule. One fill is the
# condition's fixed point (a test). *(Amended in step 8: until TreeAMR 0.2.2
# this was the low walls, the hook, `outer_points!`, and on a multilevel
# mesh a hookless fill and `outer_points!` again.)*
function fill_set!(p::MHDProblem, i::Int, t)
    fill_ghosts!(p.sets[i], p.schedules[i];
                 boundary=p.boundary === nothing ? nothing : boundary_at(p.boundary[i], t))
    return nothing
end

# --- the kernels ------------------------------------------------------------------

# The full `P` of a cell from the primitives the ghost fill carries (`W`'s
# slots, `wv`), the cell's `B` and metric, and the recovery's flags: `vⁱ =
# uⁱ/W` with `W = √(1 + γᵢⱼuⁱuʲ)`, the rapidity `asinh |u|`, and `p`, `h`,
# `cs²` and the EOS's warm start from one forward evaluation of the EOS — no
# inversion. Owned and ghost cells alike go through it (proposed in step 8),
# so that where a ghost is a copy of its owner (a uniform mesh, a same-level
# neighbour) the two blocks hold the same `P` bit for bit and compute the
# same flux at their common face. `q` is written by `charge_kernel!`.
# *(Amended 2026-10-23: `W` holds the specific `s` and `Yₑ` only; the density
# forms of the step-8 experiments are on `claude/step-8-ghost-entropy-density`
# and `claude/step-8-ghost-robustness`, "Refinement" in `CODE.md`.)*
@inline function P_from_W(eos, m::Metric3{T}, wv::NTuple{NW,T}, B::SVector{3,T},
                          flags::UInt32) where {T}
    ρ, s, Yₑ, u = wv[W_RHO], wv[W_S], wv[W_YE], vec3(wv, W_U)
    E = vec3(wv, W_E)
    u² = dot_up(m, u, u)
    u² = u² > zero(T) ? u² : zero(T)
    W = sqrt(one(T) + u²)
    v = u / W
    pt = EntropyEOS.evaluate(eos, ρ, s, Yₑ, T(NaN))
    prim = pack_prim(ρ, v, s, Yₑ, asinh(sqrt(u²)), pt.p, pt.h, pt.cs², pt.u_solved)
    return pack_P(prim, flags, E, B, zero(T), m)
end

# The slots `W` stores for the thermodynamic state `ρ, s, Yₑ`, the
# contravariant `uⁱ` and the undensitized `Eⁱ`.
@inline W_store(ρ, s, Yₑ, u, E) = (ρ, s, Yₑ, u[1], u[2], u[3], E[1], E[2], E[3])

# Step 3, at the owned cells: the metric at the cell, `B` averaged to it, the
# hydrodynamic recovery of `U`, and from it the primitives the ghost fill
# carries into `W` and all of `P` (through `P_from_W`, as a ghost's).
# *(Amended in step 8: until AMR-10 this ran over every stored cell after a
# conserved ghost fill.)* *(Amended 2026-10-24:)* through the step cache
# where it is `active` — seeded from the cell's latest recovery in the step,
# reused where its input is that one's (in practice the stage limiter's,
# wherever it wrote nothing) — and `B`, `E` as `· inv(√γ)`, the reset's and
# the stage solve's arithmetic, so that the inputs agree bit for bit (until
# then `/ √γ`; the same in flat space). *(Amended 2026-10-30:)* with a
# hand-off `sc`, the third pass of the two-phase recovery (`start_kernel!`,
# `finish_kernel!` before it): level 1 is the hand-off's; with `nothing`, the
# fused recovery.
@kernel function primitive_kernel!(Pw, Ww, @Const(Uw), @Const(B1), @Const(B2), @Const(B3),
                                   Cw, Xw, sc, @Const(origins), @Const(spacings), provider, t,
                                   eos, opts, proj, pol, active::Bool, ::Val{D}, ::Val{GP},
                                   ::Val{GU}, ::Val{GB}, ::Val{GC},
                                   ::Val{GX}) where {D,GP,GU,GB,GC,GX}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x, Val(cell_staggers(Val(D))), n, b)
    cons = read_vars(Uw, stored(n, GU), b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E = cons_𝔈(cons) * isg
    ci = stored(n, GC)
    entry = read_entry(Cw, ci, b, active, typeof(h))
    prim, flags, _, entry2, hit = hydro_recovery(eos, opts, proj, pol, m, cons, E, B,
                                                 seed_of(entry, cons), entry, sc, (n..., b))
    if active
        write_entry!(Cw, ci, b, entry2)
        tally_seed!(Xw, stored(n, GX), b, SITE_RHS, entry, hit)
    end
    u = prim_W(prim) * prim_v(prim)
    wv = W_store(prim_ρ(prim), prim_s(prim), prim_Yₑ(prim), u, E)
    Pc = P_from_W(eos, m, wv, B, flags)
    sP = stored(n, GP)
    @inbounds for v in 1:NW
        Ww[sP..., v, b] = wv[v]
    end
    @inbounds for v in 1:NP
        Pw[sP..., v, b] = Pc[v]
    end
end

# Step 3, at every ghost cell, after the fill of `W` (AMR-10; step 8): `P`
# from the copied, prolongated, restricted or the hook's primitives and the
# ghost's own `B` — the curl of the filled `A` — by forward EOS evaluation,
# with no recovery flags. No ghost is recovered, and the conserved ghosts are
# not written: nothing the right-hand side, the stage solve, the reset or the
# indicator computes reads them, and `U`'s ghosts hold only what TreeAMR's own
# fills in `regrid!` and `adapt_to_initial_data!` left (amended 2026-10-23,
# decided by Erik: until then this kernel also built each conserved ghost by
# `prim2con`, which nothing read). Owned cells are skipped: `stored = true`
# hands every stored cell.
@kernel function ghost_kernel!(Pw, @Const(Ww), @Const(B1), @Const(B2), @Const(B3),
                               @Const(origins), @Const(spacings), provider, t, eos, N,
                               ::Val{D}, ::Val{GP}, ::Val{GW}, ::Val{GB}) where {D,GP,GW,GB}
    I = @index(Global, NTuple)                       # stored: `stored = true`
    b = I[D + 1]
    sP = ntuple(e -> I[e], Val(D))
    n = ntuple(e -> sP[e] - GP[e], Val(D))
    if !all(ntuple(e -> 1 ≤ n[e] ≤ N, Val(D)))
        h = @inbounds spacings[b]
        x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
        m = point_metric(provider, oftype(h, t), x, Val(cell_staggers(Val(D))), n, b)
        wv = read_vars(Ww, stored(n, GW), b, Val(NW))
        # `· inv(√γ)`, the owned cells' arithmetic (amended 2026-10-24).
        B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * inv(m.sqrtγ)
        Pc = P_from_W(eos, m, wv, B, zero(UInt32))
        @inbounds for v in 1:NP
            Pw[sP..., v, b] = Pc[v]
        end
    end
end

# Step 4: `q = ∇·E = (1/√γ) ∂ᵢ(√γ Eⁱ)` at the owned cells, by centered
# differences of `√γ E` from `P` ("Maxwell" in `CODE.md`: the charge is
# derived, not evolved). *(Amended in step 8: from `P`, which every stored
# cell now holds by the same function, rather than from the conserved
# ghosts; in flat space the same numbers.)*
@kernel function charge_kernel!(Pw, @Const(spacings), ::Val{D}, ::Val{GP}) where {D,GP}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    𝔈(c, d) = @inbounds Pw[stored(c, GP)..., P_SQRTG, b] * Pw[stored(c, GP)..., P_E + d - 1, b]
    div = sum(ntuple(Val(D)) do d
        𝔈(shift(n, d, 1), d) - 𝔈(shift(n, d, -1), d)
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
    m = point_metric(provider, oftype(h, t), x, Val(face_staggers(Val(D), Val(d))), n, b)

    P₋₁ = read_vars(Pw, stored(shift(n, d, -1), GP), b, Val(NP))
    P₀ = read_vars(Pw, stored(n, GP), b, Val(NP))
    rL, rR = face_pair(eos, lim, Pw, n, b, Val(d), Val(GP))

    𝔅n = @inbounds Bd[stored(n, GB)..., 1, b]
    L = face_state(eos, m, rL, 𝔅n, prim_ueos(P₋₁), Val(d))
    R = face_state(eos, m, rR, 𝔅n, prim_ueos(P₀), Val(d))
    F = riemann_flux(solver, m, L, R, Val(d))
    f = stored(n, GF)
    @inbounds for v in 1:NU
        Fw[f..., v, b] = F[v]
    end
end

# The reconstructed variables of the cell `k` cells from `n` along `d`.
@inline recon_at(Pw, n::NTuple, k::Int, b::Int, ::Val{d}, ::Val{GP}) where {d,GP} =
    recon_vars(read_vars(Pw, stored(shift(n, d, k), GP), b, Val(NP)))

# The two reconstructed states at the face `n` normal to `d`, after the
# positivity fallback: the left one from the cell `n − e_d` towards its upper
# face, the right one from the cell `n` towards its lower face. PLM reads the
# four cells `n − 2e_d … n + e_d`; the five-cell methods the six
# `n − 3e_d … n + 2e_d`, which `G = 3` holds for every face of the closed range.
@inline function face_pair(eos, lim::Val, Pw, n, b, vd::Val, vG::Val)
    r₋₂, r₋₁ = recon_at(Pw, n, -2, b, vd, vG), recon_at(Pw, n, -1, b, vd, vG)
    r₀, r₊₁ = recon_at(Pw, n, 0, b, vd, vG), recon_at(Pw, n, 1, b, vd, vG)
    if stencil_radius(lim) == 1
        rL = extrapolate(lim, r₋₂, r₋₁, r₀, 1)
        rR = extrapolate(lim, r₋₁, r₀, r₊₁, -1)
    else
        r₋₃, r₊₂ = recon_at(Pw, n, -3, b, vd, vG), recon_at(Pw, n, 2, b, vd, vG)
        rL = extrapolate(lim, r₋₃, r₋₂, r₋₁, r₀, r₊₁, 1)
        rR = extrapolate(lim, r₋₂, r₋₁, r₀, r₊₁, r₊₂, -1)
    end
    return limit_face(eos, rL, r₋₁), limit_face(eos, rR, r₀)
end

# The offset of the electric field of cell `c` towards its side `σ` along the
# present direction `e`, per component: `face_value − E_c` of the method, read
# from `E` at the cells `c + k e_e`, `|k| ≤ stencil_radius`.
@inline function E_offset(Pw, c::NTuple{D,Int}, b::Int, ::Val{GP}, lim::Val, ::Val{e},
                          σ::Int, E₀) where {D,GP,e}
    Em = read_E(Pw, stored(shift(c, e, -1), GP), b)
    Ep = read_E(Pw, stored(shift(c, e, 1), GP), b)
    if stencil_radius(lim) == 1
        return σ * map((l, r) -> slope(lim, l, r), E₀ - Em, Ep - E₀) / 2
    else
        Emm = read_E(Pw, stored(shift(c, e, -2), GP), b)
        Epp = read_E(Pw, stored(shift(c, e, 2), GP), b)
        return SVector(ntuple(Val(3)) do k
            @inbounds face_value(lim, Emm[k], Em[k], E₀[k], Ep[k], Epp[k], σ) - E₀[k]
        end)
    end
end

# The electric field of cell `c` extrapolated to its corner on the sides
# `σa`, `σb` (`+1` upper, `−1` lower) along the present directions `a`, `b`:
# the cell value plus the offset towards each side, `E + δₐ + δ_b`, each
# offset the one-dimensional reconstruction's along its direction (proposed in
# step 5 for PLM, extended in step 7 to every method). Additive, not a
# tensor product: second order at the corner whatever the method, which is
# the scheme's order in more than one dimension (a face's flux at its centre).
@inline function corner_E(Pw, c::NTuple{D,Int}, b::Int, vG::Val{GP}, lim::Val, va::Val{a},
                          σa::Int, vb::Val{bb}, σb::Int) where {D,GP,a,bb}
    E₀ = read_E(Pw, stored(c, GP), b)
    E = E₀
    a ≤ D && (E = E + E_offset(Pw, c, b, vG, lim, va, σa, E₀))
    bb ≤ D && (E = E + E_offset(Pw, c, b, vG, lim, vb, σb, E₀))
    return E
end

# A face-normal field reconstructed to the edge along `e` from either side:
# the faces at `n − e_e` (left) and `n` (right) of a set that is cell-like
# along `e`, or the one face value twice where `e` is a missing direction.
@inline function edge_pair(Bw, n::NTuple{D,Int}, b::Int, ::Val{G}, lim::Val,
                           ::Val{e}) where {D,G,e}
    if e ≤ D
        q(k) = @inbounds Bw[stored(shift(n, e, k), G)..., 1, b]
        q₋₂, q₋₁, q₀, q₊₁ = q(-2), q(-1), q(0), q(1)
        if stencil_radius(lim) == 1
            L = q₋₁ + slope(lim, q₋₁ - q₋₂, q₀ - q₋₁) / 2
            R = q₀ - slope(lim, q₀ - q₋₁, q₊₁ - q₀) / 2
        else
            q₋₃, q₊₂ = q(-3), q(2)
            L = face_value(lim, q₋₃, q₋₂, q₋₁, q₀, q₊₁, 1)
            R = face_value(lim, q₋₂, q₋₁, q₀, q₊₁, q₊₂, -1)
        end
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
    m = point_metric(provider, oftype(h, t), x, Val(edge_staggers(Val(D), Val(j))), n, b)

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
    dm = metric_gradient(provider, oftype(h, t), x, Val(cell_staggers(Val(D))), n, b)
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
    m = point_metric(provider, t, x, Val(vertex_staggers(Val(D))), v, b)
    Φ̃ = @inbounds Φw[stored(v, GΦ)..., 1, b]
    return gauge_potential(m, Φ̃, vertex_A(A, v, b, GA))
end

# Step 7, the potential: `∂ₜA_j = −EMF_j − ∂_j(αΦ − βᵏA_k)` at the owned
# edges along `j`. The edge `n` lies between the vertices `n` and `n + e_j`;
# along a missing direction there is no gradient. Zero on an owned low wall
# (`lowwall`, AMR-8): the wall is prescribed by the fill, not evolved
# (decided in step 8; a branch, never a multiplication by a mask).
@kernel function potential_rhs_kernel!(du, @Const(EMFj), @Const(A1), @Const(A2), @Const(A3),
                                       @Const(Φw), @Const(origins), @Const(spacings),
                                       @Const(lowwall), provider, t, ::Val{D}, ::Val{j},
                                       ::Val{GE}, ::Val{GA}, ::Val{GΦ}) where {D,j,GE,GA,GΦ}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    e = @inbounds EMFj[stored(n, GE)..., 1, b]
    if on_low_wall(n, @inbounds(lowwall[b]), Val(edge_staggers(Val(D), Val(j))))
        @inbounds du[n..., 1, b] = zero(h)
    elseif j ≤ D
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
    m = point_metric(provider, t, x, Val(edge_staggers(Val(D), Val(j))), c, b)
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
# vertices, the flux on the `j`-edges `v − e_j` and `v` on either side; zero
# on an owned low wall, as for `A_j`.
@kernel function gauge_rhs_kernel!(du, @Const(A1), @Const(A2), @Const(A3), @Const(Φw),
                                   @Const(origins), @Const(spacings), @Const(lowwall),
                                   provider, t, ξ, ::Val{D}, ::Val{GA},
                                   ::Val{GΦ}) where {D,GA,GΦ}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    v = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    if on_low_wall(v, @inbounds(lowwall[b]), Val(vertex_staggers(Val(D))))
        @inbounds du[v..., 1, b] = zero(h)
    else
        o = @inbounds origins[b]
        tt = oftype(h, t)
        A = (A1, A2, A3)
        div = sum(ntuple(Val(D)) do j
            gauge_edge_flux(A, Φw, v, b, GA, Val(GΦ), o, h, provider, tt, Val(j)) -
            gauge_edge_flux(A, Φw, shift(v, j, -1), b, GA, Val(GΦ), o, h, provider, tt,
                            Val(j))
        end)
        x = point_x(o, h, v, Val(vertex_staggers(Val(D))))
        m = point_metric(provider, tt, x, Val(vertex_staggers(Val(D))), v, b)
        Φ̃ = @inbounds Φw[stored(v, GΦ)..., 1, b]
        @inbounds du[v..., 1, b] = -div / h + gauge_damping(oftype(h, ξ), m, Φ̃)
    end
end

# The implicit stage of Ohm's law at the owned cells (step 3's
# `solve_ohm_stage`), writing only `𝔈` into `U`, and tallying the outcome
# into `X`. `U` enters holding `u★`. Through the step cache where it is
# `active` (2026-10-24): the predictor seeded from the cell's latest recovery
# in the step, and the cell left holding the recovery of the state the solve
# returns. *(Amended 2026-10-30:)* with a hand-off `sc`, the third pass of
# the two-phase recovery: the solve's first recovery takes its level 1 from
# it.
@kernel function ohm_kernel!(Uw, @Const(ustar), @Const(B1), @Const(B2), @Const(B3), Xw, Cw, sc,
                             @Const(origins), @Const(spacings), provider, t, γΔt, eos, opts,
                             proj, pol, model, stage, active::Bool, ::Val{D}, ::Val{GB},
                             ::Val{GX}, ::Val{GC}) where {D,GB,GX,GC}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x, Val(cell_staggers(Val(D))), n, b)
    cons★ = read_vars(ustar, n, b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E★ = cons_𝔈(cons★) * isg
    ci = stored(n, GC)
    entry = read_entry(Cw, ci, b, active, typeof(h))
    E, _, r, nrec, entry2, hit = ohm_stage(eos, opts, proj, pol, model, m, cons★, E★, B,
                                           oftype(h, γΔt), seed_of(entry, cons★), entry, sc,
                                           (n..., b); stage=stage)
    if active
        write_entry!(Cw, ci, b, entry2)
        tally_seed!(Xw, stored(n, GX), b, SITE_STAGE, entry, hit)
    end
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
# reset injects into the state (step 6). Through the step cache where it is
# `active` (2026-10-24): seeded from the cell's latest recovery in the step,
# reused where its input is that one's; the stage limiter leaves its recovery
# in the cache, and the step limiter, which ends the step, invalidates it.
# *(Amended 2026-10-30:)* with a hand-off `sc`, the third pass of the
# two-phase recovery.
@kernel function reset_kernel!(uw, @Const(B1), @Const(B2), @Const(B3), Xw, Cw, sc,
                               @Const(origins), @Const(spacings), provider, t, eos, opts, proj,
                               pol, active::Bool, ::Val{D}, ::Val{GB}, ::Val{GX}, ::Val{GC},
                               ::Val{STEP}) where {D,GB,GX,GC,STEP}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x, Val(cell_staggers(Val(D))), n, b)
    cons = read_vars(uw, n, b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E = cons_𝔈(cons) * isg
    ci = stored(n, GC)
    entry = read_entry(Cw, ci, b, active, typeof(h))
    prim, flags, _, entry2, hit = hydro_recovery(eos, opts, proj, pol, m, cons, E, B,
                                                 seed_of(entry, cons), entry, sc, (n..., b))
    if active
        tally_seed!(Xw, stored(n, GX), b, STEP ? SITE_STEP_LIMITER : SITE_STAGE_LIMITER, entry,
                    hit)
        if STEP
            @inbounds Cw[ci..., C_S, b] = oftype(h, NaN)   # the step ends: invalid
        else
            write_entry!(Cw, ci, b, entry2)
        end
    end
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

# The first pass of the two-phase recovery at the owned cells (stage 4 of the
# recovery pipeline, 2026-10-30; "The two-phase recovery" in `CODE.md`): the
# metric, `B` averaged to the cell, `E` from the state `Uw` (ghost width
# `GU`; zero for a state vector), EntropyEOS's input as the site's own kernel
# computes it — the hydrodynamic recovery (`model = nothing`: the right-hand
# side, the limiters), or the stage solve's first recovery
# (`predictor_input(model, …)`) —, the step cache's entry and seed where it is
# `active`, exactly as the third pass reads them; and `con2prim_safe_start`
# into the hand-off `sc`, or nothing where the entry is reused.
@kernel function start_kernel!(sc, @Const(Uw), @Const(B1), @Const(B2), @Const(B3), @Const(Cw),
                               @Const(origins), @Const(spacings), provider, t, eos, opts, proj,
                               model, active::Bool, ::Val{D}, ::Val{GU}, ::Val{GB},
                               ::Val{GC}) where {D,GU,GB,GC}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x, Val(cell_staggers(Val(D))), n, b)
    cons = read_vars(Uw, stored(n, GU), b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E = cons_𝔈(cons) * isg
    entry = read_entry(Cw, stored(n, GC), b, active, typeof(h))
    level1_start!(sc, (n..., b), eos, opts, proj, start_input(model, m, cons, E, B),
                  seed_of(entry, cons), entry)
end

# The input of the first pass: the hydrodynamic recovery's, or the stage
# solve's first recovery's.
@inline start_input(::Nothing, m, cons, E, B) = first(hydro_input(m, cons, E, B))
@inline start_input(model::Conductivity, m, cons, E, B) = predictor_input(model, m, cons, E, B)

# The second pass: `con2prim_safe_finish` at every owned cell whose first
# stage was not done, behind the hand-off's per-cell guard — no compaction:
# on the CPU it would buy nothing, and every cell's work stays independent of
# the others', so one thread and four give the same result.
@kernel function finish_kernel!(sc, eos, opts, proj, ::Val{D}) where {D}
    I = @index(Global, NTuple)                       # local, owned: `(n..., b)`
    level1_finish!(sc, Tuple(I), eos, opts, proj)
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
    m = point_metric(provider, oftype(h, t), x, Val(cell_staggers(Val(D))), n, b)
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
evolved set, fill the potentials' ghosts, compute `𝔅` on every stored face,
recover `P` at the owned cells, fill the primitives' ghosts and build `P`
in every ghost cell from them (no ghost is recovered, and the matter set's
ghosts are not written), and `q` at the owned cells. Returns the primitive
set `P`, filled in every stored cell.
"""
function update_primitives!(p::MHDProblem{T,D}, u, t; metric=p.provider) where {T,D}
    scatter!(p.sets, u)
    for i in SEG_A:SEG_Φ
        fill_set!(p, i, t)
    end
    return primitives_from_sets!(p, t; metric)
end

# Steps 2 to 4 of the right-hand side on sets whose potentials are filled and
# whose matter holds the state on its owned cells (ghosts not needed):
# `𝔅` on every stored face; the recovery at the owned cells into `P` and the
# primitive set `W`; `W`'s ghost fill — copies at same-level faces, the
# bounded conservative prolongation centred on the parent at refinement
# boundaries (`default_ops_W`), with its own hook, which applies the complete
# outer boundary condition; `P` at every ghost cell from `W` and the ghost's
# own `B`; `q` at the owned cells. *(Amended in step 8, decided by Erik
# 2026-10-15: until then the matter's ghost fill was in conserved variables,
# and the recovery ran in every stored cell after it. Amended 2026-10-23,
# decided by Erik: the parent-centred primitive fill is the only one, and the
# conserved ghosts are no longer written.)*
function primitives_from_sets!(p::MHDProblem{T,D}, t; metric=p.provider) where {T,D}
    f = p.f
    curl_A!(f, p.𝔅̄, p.spacings)
    GB = Val(GB_of(f))
    recovery_start!(p, f.U.work, Val(f.U.G), nothing, t; metric)
    map_blocks!(primitive_kernel!, f.P, f.P.work, f.W.work, f.U.work, f.B[1].work,
                f.B[2].work, f.B[3].work, p.cache.work, f.X.work, p.handoff, p.origins,
                p.spacings,
                metric, t, p.eos, p.opts, p.proj, p.pol, p.cache_on[], Val(D), Val(f.P.G),
                Val(f.U.G), GB, Val(p.cache.G), Val(f.X.G))
    fill_ghosts!(f.W, p.schedule_W;
                 boundary=p.boundary_W === nothing ? nothing : boundary_at(p.boundary_W, t))
    map_blocks!(ghost_kernel!, f.P, f.P.work, f.W.work, f.B[1].work, f.B[2].work,
                f.B[3].work, p.origins, p.spacings, metric, t, p.eos, f.P.forest.N,
                Val(D), Val(f.P.G), Val(f.W.G), GB; stored=true)
    map_blocks!(charge_kernel!, f.P, f.P.work, p.spacings, Val(D), Val(f.P.G))
    return f.P
end

"""
    mhd_rhs!(du, u, p::MHDProblem, t)

The explicit right-hand side of every evolved variable, in the order "The
right-hand side" in `CODE.md` sets out (see the top of `src/evolution.jl`):
the matter's flux divergence and explicit sources, `A`'s EMF and gauge
gradient, and `Φ̃`'s gauge equation. The conduction current, the one
implicit term, is not here: it is [`ohm_solve!`](@ref)'s.

**Pure in `(u, t)` outside a step, and it never mutates `u`**: every
working array is scratch, rewritten in full, and every recovery is seeded
from `u` itself ([`state_guesses`](@ref)). Inside a step of
[`evolve!`](@ref) ([`with_step_cache`](@ref)) its recoveries go through the
step cache — seeded from the cell's latest recovery in the step, reused
where the input is that one's — and a step is pure in `uⁿ` (amended
2026-10-24, decided by Erik 2026-10-23; "The step cache" in `CODE.md`).
IMEXRungeKutta's `f_exp!`.
"""
function mhd_rhs!(du, u, p::MHDProblem, t)
    prepare!(p, u, t)
    rhs!(du, p, t)
    return nothing
end

"""
    prepare!(p::MHDProblem, u, t; metric = p.provider) -> p

Make the state `u` at `t` readable: scatter it into every evolved set, fill
the potentials' ghosts, and recover the primitives at the owned cells and
build them in every ghost ([`update_primitives!`](@ref)), in the metric
`metric`. The first half of [`mhd_rhs!`](@ref), whose second is
[`rhs!`](@ref); a script that owns the time loop calls the two in turn
("The coupling contract", "Two scripts" in `CODE.md`; added in step 10).
Returns `p`, whose prepared state the producer's adapters read (step 11's
`T_ab`). `metric` reaches the kernels only: the boundary hooks, the
interior and the initial data keep the case's own provider (proposed in
step 10). Public, not exported.
"""
function prepare!(p::MHDProblem, u, t; metric=p.provider)
    update_primitives!(p, u, t; metric)
    return p
end

"""
    rhs!(du, p::MHDProblem, t; metric = p.provider)

The explicit right-hand side of the state [`prepare!`](@ref) left in `p`
(with the same `metric`), into `du`: the kernel half of
[`mhd_rhs!`](@ref) (added in step 10). Public, not exported.
"""
function rhs!(du, p::MHDProblem{T,D}, t; metric=p.provider) where {T,D}
    f = p.f
    GB = GB_of(f)
    ntuple(Val(D)) do d
        Fd = f.F[d]
        map_blocks!(flux_kernel!, Fd, Fd.work, f.P.work, f.B[d].work, p.origins, p.spacings,
                    metric, t, p.eos, Val(D), Val(d), Val(Fd.G), Val(f.P.G), Val(GB[d]),
                    p.limiter, p.solver; closed=true)
        p.fixup && restrict_interfaces!(Fd, p.ischeds_F[d])
        nothing
    end
    ntuple(Val(3)) do j
        a, b = mod1(j + 1, 3), mod1(j + 2, 3)
        Ej = f.EMF[j]
        map_blocks!(emf_kernel!, Ej, Ej.work, f.P.work, f.B[a].work, f.B[b].work, p.origins,
                    p.spacings, metric, t, Val(D), Val(j), Val(Ej.G), Val(f.P.G),
                    Val(GB[a]), Val(GB[b]), p.limiter; closed=true)
        (p.fixup && p.ischeds_EMF[j] !== nothing) &&
            restrict_interfaces!(Ej, p.ischeds_EMF[j])
        nothing
    end
    map_blocks!(matter_rhs_kernel!, f.U, statearray(du, p.sets, SEG_U), f.P.work,
                map(F -> F.work, f.F), p.origins, p.spacings, metric, t, Val(D),
                Val(f.P.G), Val(first(f.F).G))
    GA = GA_of(f)
    ntuple(Val(3)) do j
        i = SEG_A + j - 1
        map_blocks!(potential_rhs_kernel!, p.sets[i], statearray(du, p.sets, i),
                    f.EMF[j].work, f.A[1].work, f.A[2].work, f.A[3].work, f.Φ.work,
                    p.origins, p.spacings, p.lowwall, metric, t, Val(D), Val(j),
                    Val(f.EMF[j].G),
                    Val(GA), Val(f.Φ.G))
        nothing
    end
    map_blocks!(gauge_rhs_kernel!, f.Φ, statearray(du, p.sets, SEG_Φ), f.A[1].work,
                f.A[2].work, f.A[3].work, f.Φ.work, p.origins, p.spacings, p.lowwall,
                metric, t, p.ξ, Val(D), Val(GA), Val(f.Φ.G))
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
ohm_solve!(U, ustar, γΔt, p::MHDProblem, t) = stage_solve!(U, ustar, γΔt, p, t)

"""
    stage_solve!(U, u★, γΔt, p::MHDProblem, t; metric = p.provider)

[`ohm_solve!`](@ref) in the metric `metric` (added in step 10): the verb a
script's `solve_imp!` calls. Public, not exported.
"""
function stage_solve!(U, ustar, γΔt, p::MHDProblem{T,D}, t; metric=p.provider) where {T,D}
    f = p.f
    update_fields!(p, ustar, t)
    us = statearray(ustar, p.sets, SEG_U)
    recovery_start!(p, us, Val(ntuple(_ -> 0, D)), p.model, t; metric)
    map_blocks!(ohm_kernel!, f.U, statearray(U, p.sets, SEG_U), us, f.B[1].work, f.B[2].work,
                f.B[3].work, f.X.work, p.cache.work, p.handoff, p.origins, p.spacings,
                metric, t, γΔt, p.eos, p.opts, p.proj, p.pol, p.model, p.stage,
                p.cache_on[], Val(D), Val(GB_of(f)), Val(f.X.G), Val(p.cache.G))
    return nothing
end

"""
    reset_atmosphere!(u, integ, p::MHDProblem, t)

The reset, with the limiter signature `(u, integrator, p, t)`: recover every
owned cell of `u` and, where EntropyEOS's two levels repaired the state (a
face of the valid range — the density, `Yₑ` and entropy bounds, the rapidity
cap — or the atmosphere: `FLAG_POL_ANY`), write the repaired state's matter
variables back into `u`; `𝔈` is
not touched. Passed to the integrator as both the stage limiter and the step
limiter ("Floors, the atmosphere and the magnetosphere" in `CODE.md`). The
cells it writes are counted into the diagnostics set.

*(Proposed in step 5:)* any policy repair is written back, not the
atmosphere alone: every repaired state is exactly invertible, and is what
the next right-hand side would recover anyway.

*(Added in step 8; amended in step 8 when AMR-8 landed:)* after its fill of
the potentials, which writes their owned low walls by the outer boundary
condition (TreeAMR's symmetric walls), it copies those walls — and nothing
else — into `u` ([`walls_to_state!`](@ref)), so that the state holds the
boundary condition's walls after every step, where the right-hand side,
writing zero there, would leave them stale ("Boundaries" in `CODE.md`).

This method counts its writes as the stage limiter's; [`reset_step!`](@ref)
is the same reset in the step limiter's role.
"""
reset_atmosphere!(u, integ, p::MHDProblem, t) = stage_limit!(u, p, t)

"""
    stage_limit!(u, p::MHDProblem, t; metric = p.provider)
    step_limit!(u, p::MHDProblem, t; metric = p.provider)

The reset as the stage limiter ([`reset_atmosphere!`](@ref)) and as the
step limiter ([`reset_step!`](@ref)), in the metric `metric`, without the
integrator argument: the verbs a script's limiters call (added in step 10).
Public, not exported.
"""
stage_limit!(u, p::MHDProblem, t; metric=p.provider) = reset!(u, p, t, Val(false); metric)
step_limit!(u, p::MHDProblem, t; metric=p.provider) = reset!(u, p, t, Val(true); metric)

"""
    reset_step!(u, integ, p::MHDProblem, t)

[`reset_atmosphere!`](@ref) as the step limiter: the same writes, counted
apart, with what they change in `√γ D` and `√γ τ` summed per cell — the rest
mass and the energy the reset injects into the state, since only the step
limiter's writes reach `uⁿ⁺¹` ([`ohm_tally`](@ref); added in step 6).
As the end of a step it also invalidates every cell's step-cache entry,
where the cache is active (2026-10-24).
"""
reset_step!(u, integ, p::MHDProblem, t) = step_limit!(u, p, t)

function reset!(u, p::MHDProblem{T,D}, t, step::Val; metric=p.provider) where {T,D}
    f = p.f
    # A black hole's interior first (step 9): the fill and the recovery below
    # see the pasted state.
    paste_in_role!(u, p, p.interior, t, step)
    update_fields!(p, u, t)
    if p.haswalls
        scatter!(p.sets[SEG_Φ], stateview(u, p.sets, SEG_Φ))
        fill_set!(p, SEG_Φ, t)
        walls_to_state!(p, u)
    end
    uU = statearray(u, p.sets, SEG_U)
    recovery_start!(p, uU, Val(ntuple(_ -> 0, D)), nothing, t; metric)
    map_blocks!(reset_kernel!, f.U, uU, f.B[1].work, f.B[2].work,
                f.B[3].work, f.X.work, p.cache.work, p.handoff, p.origins, p.spacings,
                metric, t, p.eos, p.opts, p.proj, p.pol, p.cache_on[], Val(D),
                Val(GB_of(f)), Val(f.X.G), Val(p.cache.G), step)
    return nothing
end

# The first two passes of the two-phase recovery at every owned cell
# (`start_kernel!`, `finish_kernel!`), for the state `Uw` of ghost width `GU`
# and the site whose first recovery `model` names (`nothing`: the
# hydrodynamic recovery), right before the site's own kernel, the third pass;
# nothing where the problem recovers fused (`two_phase = false`).
function recovery_start!(p::MHDProblem{T,D}, Uw, GU::Val, model, t;
                         metric=p.provider) where {T,D}
    sc = p.handoff
    sc === nothing && return nothing
    f = p.f
    map_blocks!(start_kernel!, p.cache, sc, Uw, f.B[1].work, f.B[2].work, f.B[3].work,
                p.cache.work, p.origins, p.spacings, metric, t, p.eos, p.opts, p.proj, model,
                p.cache_on[], Val(D), GU, Val(GB_of(f)), Val(p.cache.G))
    map_blocks!(finish_kernel!, p.cache, sc, p.eos, p.opts, p.proj, Val(D))
    return nothing
end

# --- the step cache ---------------------------------------------------------------

"""
    invalidate_step_cache!(p::MHDProblem)

Mark every cell's step-cache entry invalid (its `s` slot NaN), so that the
next recovery of every cell is seeded from the state. Added 2026-10-24.
"""
function invalidate_step_cache!(p::MHDProblem)
    w = p.cache.work
    selectdim(w, ndims(w) - 1, C_S) .= eltype(w)(NaN)
    return p
end

"""
    with_step_cache(f, p::MHDProblem; active = true)

Call `f()` — IMEXRungeKutta's `step!` or `solve!` on an integrator over `p`
— with `p`'s step cache active, and return what it returns. The cache is
invalidated before, and switched off and invalidated again after, in a
`finally`, so that whatever `f` does, or throws, the first step `f` takes
starts with an invalid cache and nothing outside sees one. Within, the
step limiter ([`reset_step!`](@ref)) invalidates it at the end of every
step. So each step is a pure function of `uⁿ`, `t` and `Δt` (IMEXRungeKutta
1.4 neither retries nor rejects a step, and an exception leaves `integ.u =
uⁿ`), whatever came before it: one chunk or two, a restart, one thread or
four give the same run bit for bit. [`evolve!`](@ref) wraps its stepping in
this; every other caller — a callback called directly, an `after_step`
monitor, an integrator a test builds — runs with the cache off, exactly as
before it existed. `active = false` calls `f()` alone (the negative
control). "The step cache" in `CODE.md`; added 2026-10-24, decided by Erik
2026-10-23.
"""
function with_step_cache(f, p::MHDProblem; active::Bool=true)
    active || return f()
    p.cache_on[] && throw(ArgumentError(
        "with_step_cache: the step cache of this problem is already active; nesting would " *
        "end the outer steps' cache early (CODE.md, \"The step cache\")."))
    invalidate_step_cache!(p)
    p.cache_on[] = true
    try
        return f()
    finally
        p.cache_on[] = false
        invalidate_step_cache!(p)
    end
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
  energy the reset put into the state (negative where it took some out);
- `seeds` *(added 2026-10-24)*: per site — `rhs`, `stage` (the stage
  solve's first recovery), `stage_limiter`, `step_limiter` — the recoveries
  made inside a step as `(; cold, warm, reused)`: seeded from the state,
  seeded from the step cache, or reused from it without calling EntropyEOS
  ("The step cache" in `CODE.md`). Zero for recoveries outside a step.

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
            injected_D=volume(X_INJECTED_D), injected_τ=volume(X_INJECTED_TAU),
            seeds=seed_tally(count))
end

function seed_tally(count)
    site(s) = (; cold=count(X_SEEDS + 3 * (s - 1)), warm=count(X_SEEDS + 3 * (s - 1) + 1),
               reused=count(X_SEEDS + 3 * (s - 1) + 2))
    return (; rhs=site(SITE_RHS), stage=site(SITE_STAGE), stage_limiter=site(SITE_STAGE_LIMITER),
            step_limiter=site(SITE_STEP_LIMITER))
end

"""
    max_light_rate(p::MHDProblem, t) -> max λ/h

The fastest coordinate speed of light anywhere on the mesh, over the present
directions, divided by the local spacing: the time step is
`cfl / (D · max λ/h)` ("Time integration" in `CODE.md`: `Δt` from the
light-speed CFL on the finest level). Overwrites the light-speed slot of the
diagnostics set.
"""
max_light_rate(p::MHDProblem, t) = max_rate(p, nothing, t)

"""
    max_rate(p::MHDProblem, u, t; metric = p.provider) -> max λ/h

[`max_light_rate`](@ref) in the metric `metric` (added in step 10): what a
script's `Δt` reads. The light cone depends on the metric alone, so `u` is
not read today; it is in the signature for a rate that reads the state
(split speeds, the matter's coupled run). Public, not exported.
"""
function max_rate(p::MHDProblem{T,D}, u, t; metric=p.provider) where {T,D}
    X = p.f.X
    map_blocks!(light_kernel!, X, X.work, p.origins, p.spacings, metric, T(t), Val(D),
                Val(X.G))
    forest = X.forest
    return mesh_mapreduce(identity, max, zero(T), X; vars=X_LIGHT,
                          weight=key -> inv(spacing(T, forest, key)))
end

# `h ∇·𝔅 = Σ_d (𝔅ᵈ[n + e_d] − 𝔅ᵈ[n])` at the owned cells, into `X`.
@kernel function divB_kernel!(Xw, @Const(B1), @Const(B2), @Const(B3), ::Val{D}, ::Val{GB},
                              ::Val{GX}) where {D,GB,GX}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    B = (B1, B2, B3)
    div = sum(ntuple(Val(D)) do d
        @inbounds B[d][stored(shift(n, d, 1), GB[d])..., 1, b] - B[d][stored(n, GB[d])..., 1, b]
    end)
    @inbounds Xw[stored(n, GX)..., X_DIVB, b] = div
end

"""
    max_div_B(p::MHDProblem, u, t) -> max |∇·𝔅| h / max |𝔅|

The divergence of the magnetic field of the state `u`, as the plan states its
claim (step 7): `𝔅 = curl A + 𝔅̄` on every face from `u`'s potential
([`update_fields!`](@ref)), the largest `|Σ_d (𝔅ᵈ[n + e_d] − 𝔅ᵈ[n])|` over the
owned cells — `h ∇·𝔅` — over the largest `|𝔅ᵈ|` on any owned face (zero when
there is no field at all). Structurally zero ("Where the fields live" in
`CODE.md`), so this measures rounding. Overwrites the scratch sets `A`, `B`
and the `X_DIVB` slot of `X`.
"""
function max_div_B(p::MHDProblem{T,D}, u, t) where {T,D}
    f = p.f
    update_fields!(p, u, t)
    map_blocks!(divB_kernel!, f.X, f.X.work, f.B[1].work, f.B[2].work, f.B[3].work, Val(D),
                Val(GB_of(f)), Val(f.X.G))
    div = mesh_mapreduce(abs, max, zero(T), f.X; vars=X_DIVB)
    scale = maximum(d -> mesh_mapreduce(abs, max, zero(T), f.B[d]; vars=1), 1:3)
    return scale > 0 ? div / scale : div
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
