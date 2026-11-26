# The neutrino optical depth on the forest ("Neutrinos", "The optical depth"
# in `CODE.md`): Neilsen et al. (2014)'s local estimate, per species and
# weighting,
#
#     τ_c ← min over the 3^D − 1 neighbours n of ( τ_n + ½ (κ_c + κ_n) ℓ_cn ),
#
# with `κ = κ_a + κ_s` and `ℓ_cn` the proper distance between the two cell
# centres, from the two cells' `γ_ij` averaged. It is a Bellman–Ford
# relaxation: from τ = 0 it rises one cell per sweep to the shortest optical
# path to the outer faces, whose ghosts hold τ = κ = 0. Each sweep reads the
# previous τ and writes the new one apart (Jacobi), so that a sweep is the
# same at any thread count and on any backend.
#
# The sets: `Nτ` holds τ⁰, τ¹ and κ⁰, κ¹ of ν_e, ν̄_e, ν_x (12 variables), with
# one ghost layer — the sweep reads one, and three would triple the fill (the
# design's three, changed in step 17, proposed); an owned-only work set
# holds the new τ and each cell's relative change, which a max-reduction
# turns into the sweep's convergence measure.

# The slots of `Nτ`: τ^α of species i (1 ν_e, 2 ν̄_e, 3 ν_x) at 2(i − 1) + α + 1,
# α = 0 (number-weighted) or 1 (energy-weighted); κ^α the same plus `TAU_K`.
const NTAU = 12
const TAU_K = 6
const NTAU_WORK = 7          # the new τ (6) and the cell's relative change
const TAU_CHANGE = 7

"""
    OpacityEscape()

The outer boundary of `Nτ`: τ = κ = 0 in every ghost beyond an outer face,
so that a neutrino at the face has escaped ("Neutrinos", "The optical
depth" in `CODE.md`). `isbits`, for TreeAMR's `CellBoundary(AllVariables(·))`.
"""
struct OpacityEscape end
(::OpacityEscape)(x, δ) = ntuple(_ -> zero(x[1]), Val(NTAU))

"""
    TabulatedRates(rates::RJS96Rates, composition::EntropyEOS.CompositionView)

The rates and the table's composition together, as the kernels receive them
from an [`M0Leakage`](@ref) with a composition ("Neutrinos", "Field sets and
interface" in `CODE.md`): the composition is evaluated at each cell's state.
`isbits` once adapted to a device, as EntropyEOS's views are.
"""
struct TabulatedRates{R,CV}
    rates::R
    composition::CV
end
Adapt.@adapt_structure TabulatedRates

# The rates at one cell from its row of `P` ("Neutrinos", "Leakage",
# "Where the rates are evaluated" in `CODE.md`). `GrayRates` read only `ρ★`;
# the tabulated rates read `ρ★`, `Yₑ` and `T = 10^u_solved`, and the
# composition there, at the EOS's density. A cell with `ρ★ ≤ 0` or a warm
# start that is not finite, as `P` is before the first fill, has none, by a
# branch.
@inline cell_rates(model::GrayRates{T}, Pc::NTuple) where {T} =
    neutrino_rates(model, Pc[P_RHO], T(NaN), T(NaN), nothing)
@inline function cell_rates(model::TabulatedRates, Pc::NTuple{M,T}) where {M,T}
    ρ★ = Pc[P_RHO]
    u = Pc[P_UEOS]
    (ρ★ > 0 && isfinite(u)) || return no_rates(T)
    t = exp10(u)
    yₑ = Pc[P_YE]
    m = model.rates
    c = EntropyEOS.composition(model.composition, ρ★ * m.ρ_unit, t, yₑ)
    return neutrino_rates(m, ρ★, t, yₑ, CompositionPoint(c))
end
@inline function no_rates(::Type{T}) where {T}
    z = zero(T)
    r = SpeciesRates{T}(z, z, z, z, z, z, z, z, z)
    return (r, r, r)
end

# The opacities `κ⁰ = κ⁰_a + κ⁰_s`, `κ¹ = κ¹_a + κ¹_s` of the three species at
# one cell, from its row of `P`.
@inline function cell_opacities(model, Pw, s::NTuple, b::Int)
    r = cell_rates(model, read_vars(Pw, s, b, Val(NP)))
    return (r[1].κa0 + r[1].κs0, r[1].κa1 + r[1].κs1,
            r[2].κa0 + r[2].κs0, r[2].κa1 + r[2].κs1,
            r[3].κa0 + r[3].κs0, r[3].κa1 + r[3].κs1)
end

@kernel function opacity_kernel!(Tw, @Const(Pw), model, ::Val{D}, ::Val{GT},
                                 ::Val{GP}) where {D,GT,GP}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    t = ntuple(e -> n[e] + GT, Val(D))
    κ = cell_opacities(model, Pw, ntuple(e -> n[e] + GP, Val(D)), b)
    @inbounds begin
        Tw[t..., TAU_K + 1, b] = κ[1]
        Tw[t..., TAU_K + 2, b] = κ[2]
        Tw[t..., TAU_K + 3, b] = κ[3]
        Tw[t..., TAU_K + 4, b] = κ[4]
        Tw[t..., TAU_K + 5, b] = κ[5]
        Tw[t..., TAU_K + 6, b] = κ[6]
    end
end

# The proper distance between the centres of the cells `c` and `c + o` of a
# block of spacing `h`, with `γ_ij` the mean of the two cells' (P_GAMMA: xx,
# xy, xz, yy, yz, zz).
@inline function proper_step(Pw, pc::NTuple{D}, pn::NTuple{D}, b::Int, h::T,
                             o::NTuple{3,Int}) where {D,T}
    g(k) = @inbounds (Pw[pc..., P_GAMMA + k, b] + Pw[pn..., P_GAMMA + k, b]) / 2
    Δ1, Δ2, Δ3 = o[1] * h, o[2] * h, o[3] * h
    ℓ² = g(0) * Δ1 * Δ1 + g(3) * Δ2 * Δ2 + g(5) * Δ3 * Δ3 +
         2 * (g(1) * Δ1 * Δ2 + g(2) * Δ1 * Δ3 + g(4) * Δ2 * Δ3)
    return sqrt(ℓ²)
end

# The six τ through the neighbour at `tn`, at the proper distance `ℓ`, where
# lower. A function of its own, so that the closure captures arguments: with
# `best` reassigned in the kernel's loop and captured there, Julia boxed it and
# the sweep allocated at every neighbour (step 20: 7.7 µs a cell, ten times a
# right-hand side's flux phase).
@inline relax_through(best::NTuple{6,T}, Tw, tn, κc::NTuple{6,T}, ℓ::T, b::Int) where {T} =
    ntuple(Val(6)) do k
        @inbounds min(best[k], Tw[tn..., k, b] + (κc[k] + Tw[tn..., TAU_K + k, b]) * ℓ / 2)
    end

# One Jacobi sweep at an owned cell: the new τ into the work set, and the
# largest relative change among the six. A change from or to zero that is
# not zero counts as infinite, so that a sweep converges only once every τ
# is fixed.
@kernel function sweep_kernel!(Sw, @Const(Tw), @Const(Pw), @Const(spacings), ::Val{D},
                               ::Val{GT}, ::Val{GP}, ::Val{GS}) where {D,GT,GP,GS}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    T = typeof(h)
    tc = ntuple(e -> n[e] + GT, Val(D))
    pc = ntuple(e -> n[e] + GP, Val(D))
    sc = ntuple(e -> n[e] + GS, Val(D))
    κc = ntuple(k -> @inbounds(Tw[tc..., TAU_K + k, b]), Val(6))
    best = ntuple(_ -> T(Inf), Val(6))
    r2 = D ≥ 2 ? 1 : 0
    r3 = D ≥ 3 ? 1 : 0
    for o3 in (-r3):r3, o2 in (-r2):r2, o1 in -1:1
        (o1 == 0 && o2 == 0 && o3 == 0) && continue
        o = (o1, o2, o3)
        tn = ntuple(e -> tc[e] + o[e], Val(D))
        pn = ntuple(e -> pc[e] + o[e], Val(D))
        ℓ = proper_step(Pw, pc, pn, b, h, o)
        best = relax_through(best, Tw, tn, κc, ℓ, b)
    end
    change = zero(T)
    for k in 1:6
        old = @inbounds Tw[tc..., k, b]
        new = best[k]
        @inbounds Sw[sc..., k, b] = new
        Δ = abs(new - old)
        rel = iszero(Δ) ? zero(T) : (new > 0 ? Δ / new : T(Inf))
        change = max(change, rel)
    end
    @inbounds Sw[sc..., TAU_CHANGE, b] = change
end

@kernel function sweep_copy_kernel!(Tw, @Const(Sw), ::Val{D}, ::Val{GT},
                                    ::Val{GS}) where {D,GT,GS}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    tc = ntuple(e -> n[e] + GT, Val(D))
    sc = ntuple(e -> n[e] + GS, Val(D))
    for k in 1:6
        @inbounds Tw[tc..., k, b] = Sw[sc..., k, b]
    end
end

"""
    OpticalDepth(T = Float64, forest; backend = CPU(), ops, workgroupsize = nothing)

The neutrino optical depth on `forest` ("Neutrinos", "The optical depth" in
`CODE.md`): the set `Nτ` (τ⁰, τ¹, κ⁰, κ¹ of ν_e, ν̄_e, ν_x, one ghost layer),
an owned-only work set, the ghost schedule and the outer boundary
[`OpacityEscape`](@ref). `ops` are the operators of `Nτ`'s ghost fills and
of its transfer at a regrid, the matter's bounded conservative ones by
default. τ starts at zero.

A forest periodic in every direction has no outer face for τ to be measured
to, and is refused. Faces the forest declares reflecting or outflow are
filled by the schedule, not by the hook: neutrinos escape only through the
other outer faces.

[`opacities!`](@ref) fills κ from `P`, [`sweep!`](@ref) relaxes τ once and
[`converge!`](@ref) until it is fixed. After a regrid that moved the mesh,
[`rebuild`](@ref)`(od)` makes the schedule anew; [`depth_regrid_pairs`](@ref)
are `Nτ`'s and the work set's entries for TreeAMR's `regrid!`.
"""
struct OpticalDepth{FT,FS,S,H,V,WG}
    Nτ::FT
    work::FS
    ops::Operators
    schedule::S
    hook::H
    spacings::V
    workgroupsize::WG
end

function OpticalDepth(::Type{T}, forest::Forest{D}; backend=CPU(),
                      ops=Operators(; prolongation=3, restriction=2, family=Conservative,
                                    bounded=true),
                      workgroupsize=nothing) where {T,D}
    all(forest.periodic) && throw(ArgumentError(
        "OpticalDepth needs an outer face: on a forest periodic in every direction the " *
        "optical depth has no boundary to be measured to, and every sweep would raise it."))
    Nτ = FieldSet{T}(forest, NTAU; G=1, centering=cellcentered(D), backend=backend)
    work = FieldSet{T}(forest, NTAU_WORK; G=0, centering=cellcentered(D), backend=backend)
    return OpticalDepth(Nτ, work, ops, GhostSchedule(Nτ, ops),
                        CellBoundary(AllVariables(OpacityEscape())),
                        to_backend(backend, block_spacings(forest, T)), workgroupsize)
end
OpticalDepth(forest::Forest; kwargs...) = OpticalDepth(Float64, forest; kwargs...)

"""
    rebuild(od::OpticalDepth)

`od` on its forest as the forest now is: a new ghost schedule and the
blocks' spacings, after a regrid that moved the mesh (TreeAMR refuses a
stale schedule). The sets are the same, resized by the regrid.
"""
function rebuild(od::OpticalDepth)
    T = eltype(od.Nτ.work)
    backend = get_backend(od.Nτ.work)
    forest = od.Nτ.forest
    return OpticalDepth(od.Nτ, od.work, od.ops, GhostSchedule(od.Nτ, od.ops), od.hook,
                        to_backend(backend, block_spacings(forest, T)), od.workgroupsize)
end

"""
    depth_regrid_pairs(od) -> Tuple

`Nτ`'s and the work set's entries for TreeAMR's `regrid!`: `Nτ` is filled
with its own schedule and hook and transferred with its operators (τ goes
with the mesh, so that the sweeps after a regrid start near the answer; κ
goes too, and is recomputed before the next sweep); the work set is resized
only.
"""
depth_regrid_pairs(od::OpticalDepth) = (od.Nτ => (od.schedule, od.hook), od.work => nothing)

"""
    reset_depth!(od)

τ and κ to zero everywhere, as at construction.
"""
reset_depth!(od::OpticalDepth) = (fill!(od.Nτ.work, 0); od)

"""
    opacities!(od, model, P)

κ⁰ and κ¹ of the three species at every owned cell, from the primitive set
`P` (a `FieldSet` laid out as `MHDFields`' `P`, filled at the owned cells).
"""
function opacities!(od::OpticalDepth, model, P::FieldSet)
    D = length(od.Nτ.G)
    map_blocks!(opacity_kernel!, od.Nτ, od.Nτ.work, P.work, model, Val(D),
                Val(first(od.Nτ.G)), Val(first(P.G)); workgroupsize=od.workgroupsize)
    return od
end

"""
    sweep!(od, P) -> change

One sweep: `Nτ`'s ghosts filled (with [`OpacityEscape`](@ref) at the outer
faces), τ relaxed at every owned cell from its `3^D − 1` neighbours, and the
largest relative change of any τ returned (infinite where a τ left or reached
zero). `P` supplies the metric at the cells and their ghosts, as
`update_primitives!` leaves it.
"""
function sweep!(od::OpticalDepth, P::FieldSet)
    T = eltype(od.Nτ.work)
    D = length(od.Nτ.G)
    fill_ghosts!(od.Nτ, od.schedule; boundary=od.hook)
    GT, GP, GS = Val(first(od.Nτ.G)), Val(first(P.G)), Val(first(od.work.G))
    map_blocks!(sweep_kernel!, od.work, od.work.work, od.Nτ.work, P.work, od.spacings, Val(D),
                GT, GP, GS; workgroupsize=od.workgroupsize)
    map_blocks!(sweep_copy_kernel!, od.Nτ, od.Nτ.work, od.work.work, Val(D), GT, GS;
                workgroupsize=od.workgroupsize)
    return mesh_mapreduce(identity, max, zero(T), od.work; vars=TAU_CHANGE)
end

"""
    converge!(od, P; tol = 1e-10, maxsweeps = 100_000) -> sweeps

Sweeps until no τ changes by more than `tol` of itself, and the number of
sweeps that took, the last being the one that changed nothing beyond `tol`.
From τ = 0 that is about as many sweeps as the longest optical path is
cells long; after a regrid, which transfers τ, a few. Throws if `maxsweeps`
are not enough.
"""
function converge!(od::OpticalDepth, P::FieldSet; tol=1e-10, maxsweeps::Integer=100_000)
    for k in 1:maxsweeps
        sweep!(od, P) ≤ tol && return k
    end
    throw(ErrorException(
        "converge!: the optical depth still changed by more than $tol after $maxsweeps " *
        "sweeps; a sweep moves τ by one cell, so the mesh is larger than that many cells " *
        "across or τ grows without bound."))
end
