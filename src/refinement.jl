# Refinement: the criterion, the regrid of every field set, and the initial
# adaptation ("Refinement" in `CODE.md`; step 8 of `PLAN.md`).
#
# TreeAMR moves the data; what is here is what only the application knows:
#
#   * **The criterion**: Löhner's indicator on `ρ`, `|B|` and `|E|` — the
#     density for the matter's waves, the two field magnitudes for the
#     electromagnetic ones, which in an electrovacuum or a force-free region
#     move where `ρ` does not — reduced per block through TreeAMR's
#     `firing_boxes`, with TreeHydro's four marks and its two noise floors.
#   * **The pairs**: every field set over the forest goes to `regrid!`, the
#     evolved ones with their schedules and their own Dirichlet hooks at the
#     time of the regrid (AMR-2), every other one `=> nothing` — resized, never
#     transferred: `P`, `B_d` (never state), `F_d`, `EMF_d`, `X`.
#   * **What follows a regrid**: the problem rebuilt over the same sets, the
#     state gathered, the potentials' low walls prescribed (AMR-8), and the
#     reset run once as the step limiter, so that a prolongated conserved
#     state the recovery would repair is repaired in the state and its
#     injection is accounted (proposed in step 8, as TreeHydro's post-regrid
#     reset).

"""
    lohner(um, u0, up, u_ref, ε, ε_g)

Löhner's error indicator for three consecutive cell values of one variable
along one dimension, with a local and a global noise floor (TreeHydro's
`lohner`, copied):

    τ = |u₊ − 2u₀ + u₋| /
        (|u₊ − u₀| + |u₀ − u₋| + ε (|u₊| + 2|u₀| + |u₋|) + ε_g |u_ref|)

Undivided differences, so `τ ∈ [0, 1]` measures how well the mesh holds the
data, and refinement of smooth data stops by itself. The local floor keeps
a field with a large dynamic range scale-free; the global one, against the
variable's largest value on the mesh, keeps an atmosphere — or a region
with no field at all — from firing on its rounding. A zero denominator gives
zero.
"""
@inline function lohner(um, u0, up, u_ref, ε, ε_g)
    num = abs(up - 2 * u0 + um)
    den = abs(up - u0) + abs(u0 - um) + ε * (abs(up) + 2 * abs(u0) + abs(um)) +
          ε_g * abs(u_ref)
    return iszero(den) ? zero(num) : num / den
end

# The three indicator variables of the cell at stored index `idx`: `ρ`, and
# the magnitudes `|B| = √(γᵢⱼBⁱBʲ)` and `|E|` in the cell's own metric, from
# the primitive set `P`.
@inline function indicator_values(w, idx::NTuple, b::Integer)
    Pc = read_vars(w, idx, b, Val(NP))
    m = aux_metric(Pc)
    B, E = aux_B(Pc), aux_E(Pc)
    return (prim_ρ(Pc), sqrt(dot_up(m, B, B)), sqrt(dot_up(m, E, E)))
end

# The worst Löhner indicator over the three variables and every dimension at
# the cell `idx` (stored) of block `b`; `refs` the three global references.
@inline function cell_tau(w, idx::NTuple{D,Int}, b::Integer, refs::NTuple{3}, ε, ε_g,
                          ::Val{D}) where {D}
    q₀ = indicator_values(w, idx, b)
    return maximum(ntuple(Val(D)) do d
        q₊ = indicator_values(w, Base.setindex(idx, idx[d] + 1, d), b)
        q₋ = indicator_values(w, Base.setindex(idx, idx[d] - 1, d), b)
        maximum(ntuple(k -> lohner(q₋[k], q₀[k], q₊[k], refs[k], ε, ε_g), Val(3)))
    end)
end

"""
    indicator_scales(P::FieldSet) -> (ρ_ref, B_ref, E_ref)

The global references of [`lohner`](@ref)'s `ε_g` floor: the largest `ρ`,
and the largest component magnitude of `Bⁱ` and of `Eⁱ`, over the owned
cells of the primitive set. Component maxima rather than the largest `|B|`
because a `mesh_mapreduce` reduces one variable at a time; a reference that
is off by at most `√3` only moves a noise floor (proposed in step 8).
"""
function indicator_scales(P::FieldSet{T}) where {T}
    peak(v) = mesh_mapreduce(abs, max, zero(T), P; vars=v)
    return (peak(P_RHO), max(peak(P_B), peak(P_B + 1), peak(P_B + 2)),
            max(peak(P_E), peak(P_E + 1), peak(P_E + 2)))
end

"""
    mhd_flags(P::FieldSet; refine_tol, coarsen_tol, maxlevel, ε = 1//100,
              ε_g = 1//1000, scales = indicator_scales(P))
    mhd_flags(p::MHDProblem; …)

The flag vector `regrid!` takes, one entry per block, from Löhner's
indicator on `ρ`, `|B|` and `|E|` of a current primitive set — `P` must be
recovered in every stored cell, ghosts included ([`update_primitives!`](@ref)),
since the stencil reads one cell past each face. TreeHydro's four marks
(`hydro_flags`, which this follows):

    any cell > refine_tol, level < maxlevel  ->  (Refine, box)
    any cell > coarsen_tol                   ->  (Keep, box)   a travelling margin
    nothing fired, level > 0                 ->  Coarsen
    otherwise                                ->  Keep

with the box of the cells above `coarsen_tol`, which `regrid!` dilates by
its `buffer`. **None of the tolerances and not `maxlevel` has a default**:
each is the case's. `coarsen_tol < refine_tol` is required — the dead band
between them is what keeps a block at the threshold from flipping on every
regrid.
"""
mhd_flags(p::MHDProblem; kwargs...) = mhd_flags(p.f.P; kwargs...)

function mhd_flags(P::FieldSet{T,D}; refine_tol, coarsen_tol, maxlevel::Integer,
                   ε=T(1 // 100), ε_g=T(1 // 1000), scales=indicator_scales(P)) where {T,D}
    rtol, ctol = T(refine_tol), T(coarsen_tol)
    ctol < rtol || throw(ArgumentError(
        "coarsen_tol ($coarsen_tol) must lie strictly below refine_tol ($refine_tol): " *
        "the gap is the hysteresis dead band, without which a block at the threshold " *
        "refines and coarsens on alternate regrids."))
    refs = (T(scales[1]), T(scales[2]), T(scales[3]))
    εT, ε_gT = T(ε), T(ε_g)
    vD = Val(D)
    refires = firing_boxes(P) do w, idx, b, x
        cell_tau(w, idx, b, refs, εT, ε_gT, vD) > rtol
    end
    boxfires = firing_boxes(P) do w, idx, b, x
        cell_tau(w, idx, b, refs, εT, ε_gT, vD) > ctol
    end
    return map(1:nblocks(P)) do b
        lvl = level(blockkey(P, b))
        nrefine, _ = refires[b]
        nbox, box = boxfires[b]
        nrefine > 0 && lvl < maxlevel ? (Refine, box) :
        nbox > 0 ? (Keep, box) :
        lvl > 0 ? Coarsen : Keep
    end
end

"""
    regrid_pairs(p::MHDProblem, t) -> Tuple

Every field set of `p.f` as `regrid!` takes them ("Field sets" in
`CODE.md`): the five evolved sets `=> (schedule, hook)`, each with its own
boundary hook at time `t` (AMR-2; `nothing` on a periodic domain) — but a
potential with the case's Dirichlet hook with none, so that `regrid!`'s fill
keeps the complete condition [`regrid_state!`](@ref) wrote — and `P`,
`B_d`, `F_d`, `EMF_d` and `X` `=> nothing` — resized and never transferred,
since every one of them is recomputed from the state before it is read. `B`
in particular is never state: the regrid moves `A`, and `B` is its curl.
"""
function regrid_pairs(p::MHDProblem, t)
    f = p.f
    # The matter's hook is a function of position alone and complete; a
    # potential's relative hook is not (AMR-7), so a potential with the
    # case's hook goes in without one, its outer points already complete.
    hook(i) = p.boundary === nothing ? nothing :
              i != SEG_U && p.boundary[i] isa DirichletBoundary ? nothing :
              boundary_at(p.boundary[i], t)
    evolved = ntuple(i -> p.sets[i] => (p.schedules[i], hook(i)), length(p.sets))
    scratch = (f.P => nothing, map(s -> s => nothing, f.B)..., map(s -> s => nothing, f.F)...,
               map(s -> s => nothing, f.EMF)..., f.X => nothing)
    return (evolved..., scratch...)
end

"""
    regrid_state!(p::MHDProblem, u, t; flags, buffer = 0) -> (; moved, p, u)

Regrid every field set of `p` as `flags` asks, at time `t`: scatter the
state `u` into the evolved sets, fill them with the complete outer boundary
condition (`fill_set!`), `regrid!` them all ([`regrid_pairs`](@ref):
`regrid!`'s own fill before its transfer then reproduces those ghosts, so the
transfer reads the complete condition), and where the mesh moved, [`rebuild`](@ref) the problem, gather
the state into a new vector, prescribe the potentials' low walls (AMR-8's
workaround), and run the reset once as the step limiter
([`reset_step!`](@ref)) — so that a newly prolongated conserved state the
recovery repairs is repaired in the state, and what that injects is
counted in `X` like any step limiter's write (proposed in step 8, as
TreeHydro's post-regrid reset). Returns the problem and the state that hold
from now on; the ones handed in when nothing moved.
"""
function regrid_state!(p::MHDProblem{T}, u, t; flags, buffer::Integer=0) where {T}
    scatter!(p.sets, u)
    # The complete boundary condition first, every evolved set; `regrid!`'s
    # own fill then has no hook for the potentials (`regrid_pairs`), which
    # leaves their outer points as this wrote them and, the mesh being what
    # this fill saw, recomputes every other ghost bit for bit, so its
    # transfer reads the complete condition.
    for i in eachindex(p.sets)
        fill_set!(p, i, t)
    end
    moved = regrid!(first(p.sets).forest, regrid_pairs(p, t); flags=flags, buffer=buffer)
    moved || return (; moved, p, u)
    p = rebuild(p)
    u = statevector(p.sets)
    gather!(u, p.sets)
    low_walls!(p, u, t)
    # `X` was resized without transfer, so its tallies start again here; the
    # driver regrids at the start of a chunk, after clearing them anyway.
    clear_tally!(p)
    reset_step!(u, nothing, p, t)
    return (; moved, p, u)
end
