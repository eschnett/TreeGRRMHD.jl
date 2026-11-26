# The driver: a case as data, its initial data on the mesh, and one chunked
# evolution on a uniform mesh.
#
# TreeHydro's discipline, reduced to what step 5 needs: **a case is data** —
# the pointwise solution, the EOS and its recovery options, the
# conductivity, the metric, the background field and the box — and nothing in
# this file knows what an Alfvén wave is. The run is cut into chunks, each a
# fresh integrator (reusing the previous one's scratch) with its own `Δt` from
# the light-speed CFL measured at the chunk's start and its own tally of the
# implicit stages; between chunks is where a regrid will go (step 8).
#
# Initial data and references are **exact**: the matter set holds cell
# averages, so it is filled by a tensor-product three-point Gauss quadrature of
# the conserved state over each cell (exact to `O(h⁶)`), and the potentials
# hold point values, so they are sampled. With `A` exact at the edges, `𝔅` from
# the curl is the exact face average of the field — in 1D the exact cell
# average — so the scheme starts from the solution and nothing else.

"""
    MHDCase(T, Val(D); fields, potential, eos, opts = Con2PrimOptions{T}(), proj, pol,
            model, provider = Flat(), 𝔅̄ = 0, periodic, extents, roots, interior = nothing,
            static_potential = false, gauge_damping = nothing)

A problem, as data:

- `fields(x, t) -> (ρ, v, s, Yₑ, E, B)` — the solution at the point
  `x::SVector{3,T}` (zero along the missing directions) and time `t`: the
  rest-mass density, the contravariant velocity, the entropy, the electron
  fraction, and the contravariant fields `Eⁱ`, `Bⁱ` the normal observer sees;
- `potential(x, t) -> (A, Φ̃)` — the covariant vector potential and the gauge
  scalar there, with `[ijk] ∂ⱼA_k + 𝔅̄ⁱ = √γ Bⁱ`;
- `eos`, `opts`, `proj`, `pol`: EntropyEOS's EOS, `Con2PrimOptions`,
  `ProjectionOptions` (level 1, the valid range) and `PolicyOptions` (level
  2, the atmosphere), as [`recovery_options`](@ref) builds them; `model`: the
  conductivity; `provider`: the metric;
- `𝔅̄`: the constant densitized background field (`src/curl.jl`);
- `periodic`, `extents`, `roots`: the box, per dimension;
- `interior`: `nothing`, or a black hole's [`HoleInterior`](@ref), inside
  which the limiters replace the state by `fields` and `potential` (step 9);
- `static_potential`: whether `potential` does not depend on `t`, which makes
  the potentials' Dirichlet hooks absolute ([`DirichletBoundary`](@ref);
  decided by Erik 2026-11-03). `false` keeps the relative hook (AMR-7), which
  a potential growing in time needs under IMEX;
- `gauge_damping`: the case's own `ξ`, or `nothing` for [`default_ξ`](@ref)'s
  `16/L` — a static potential with `Φ̃ ≠ 0` is a stationary solution of the
  damped gauge only at `ξ = 0` (step 9, round 4).

Both callbacks become kernel arguments, so they must capture `isbits` values
only. Where `fields` is an exact solution at every `t`, it is also the
reference ([`exact_state`](@ref)). A case that is not periodic in every
dimension is given Dirichlet boundaries from the same two callbacks
([`case_boundary`](@ref), [`DirichletBoundary`](@ref); step 6), so there
they must also be the solution outside the box for as long as the run lasts.
"""
struct MHDCase{T,D,FL,PT,EOS,OPT,PRJ,POL,MOD,MET,IN}
    fields::FL
    potential::PT
    eos::EOS
    opts::OPT
    proj::PRJ
    pol::POL
    model::MOD
    provider::MET
    𝔅̄::SVector{3,T}
    periodic::NTuple{D,Bool}
    extents::NTuple{D,Tuple{T,T}}
    roots::NTuple{D,Int}
    interior::IN
    static_potential::Bool
    gauge_damping::Union{Nothing,T}
end

function MHDCase(::Type{T}, ::Val{D}; fields, potential, eos, opts=nothing, proj, pol,
                 model::Conductivity, provider::MetricProvider=Flat(), 𝔅̄=nothing,
                 periodic, extents, roots, interior=nothing,
                 static_potential::Bool=false, gauge_damping=nothing) where {T,D}
    per = periodic isa Tuple ? ntuple(d -> Bool(periodic[d]), D) : ntuple(_ -> Bool(periodic), D)
    length(extents) == D || throw(ArgumentError(
        "a case needs one extent per dimension, got $(length(extents)) for D = $D."))
    ext = ntuple(d -> (T(extents[d][1]), T(extents[d][2])), D)
    all(d -> ext[d][2] > ext[d][1], 1:D) || throw(ArgumentError(
        "each extent must be nonempty and increasing, got $ext."))
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), D) : ntuple(_ -> Int(roots), D)
    all(>(0), rs) || throw(ArgumentError("root counts must be positive, got $rs."))
    o = opts === nothing ? EntropyEOS.Con2PrimOptions{T}() : opts
    B̄ = 𝔅̄ === nothing ? zero(SVector{3,T}) : SVector{3,T}(𝔅̄)
    return MHDCase{T,D,typeof(fields),typeof(potential),typeof(eos),typeof(o),typeof(proj),
                   typeof(pol),typeof(model),typeof(provider),typeof(interior)}(
        fields, potential, eos, o, proj, pol, model, provider, B̄, per, ext, rs, interior,
        static_potential, gauge_damping === nothing ? nothing : T(gauge_damping))
end

"""
    recovery_options(eos, ρ_min) -> (; proj, pol)

The two levels of EntropyEOS's recovery as every case of this package sets
them (amended 2026-10-24, EntropyEOS 2.0): the projection
`default_projection(eos, ρ_min)` — the valid range from `ρ_min` to the top
of the table's density box, the rapidity capped at `W = 100` — and the
atmosphere of `apply_policy` at rest at `ρ_min`, at the trigger
`D < 1.01 ρ_min`, with the electron fraction of the state it replaces, and
at the entropy `s_atm` stated here: the middle of the window at `ρ_min` and
the middle of the `Yₑ` range, which is 1.x's atmosphere for an analytic EOS
(whose window does not depend on `ρ` or `Yₑ`). EntropyEOS 2.0's own default
is the window's minimum ("Floors, the atmosphere and the magnetosphere" in
`CODE.md`). Splat it into [`MHDCase`](@ref).
"""
function recovery_options(eos::EntropyEOS.AbstractEOS, ρ_min::T) where {T}
    proj = EntropyEOS.default_projection(eos, ρ_min)
    lo, hi = EntropyEOS.yₑ_bounds(eos)
    sr = EntropyEOS.srange(eos, ρ_min, (T(lo) + T(hi)) / 2)
    pol = EntropyEOS.PolicyOptions{T}(; s_atm=(sr.s_min + sr.s_max) / 2)
    return (; proj, pol)
end

# --- initial data and references ----------------------------------------------------

# The three-point Gauss–Legendre rule on a cell of width `h` centered at 0:
# nodes `−h√(3/5)/2`, `0`, `h√(3/5)/2`, weights `5/18`, `8/18`, `5/18`.
@inline gauss3_node(h, q::Int) = q == 2 ? zero(h) : (q == 1 ? -one(h) : one(h)) * sqrt(oftype(h, 3) / 5) * h / 2

# The rule's average of three samples, written as the middle sample plus a
# correction in the differences (amended in step 7):
#
#     (5f₋ + 8f₀ + 5f₊)/18 = f₀ + 5((f₋ − f₀) + (f₊ − f₀))/18,
#
# so that three equal samples average to themselves bit for bit. A
# tensor-product rule with the weights multiplied out (`Σ w_q f_q`) rounds a
# constant to a neighbouring float, and a run that is translation-invariant
# along `y` did not start from the 1D run's data (2.7e−15 apart after seven
# steps, measured in step 5). Applied one dimension at a time, the average
# along a direction the solution does not vary in is the identity, and is
# symmetric under reflection (`f₋ ↔ f₊`).
@inline gauss3_average(f₋, f₀, f₊) =
    ntuple(v -> @inbounds(f₀[v] + 5 * ((f₋[v] - f₀[v]) + (f₊[v] - f₀[v])) / 18), Val(length(f₀)))

# The conserved state of the case at the point `x` and time `t`.
@inline function case_cons(fields, eos, provider, x::SVector{3,T}, t::T) where {T}
    m = point_metric(provider, t, x)
    ρ, v, s, Yₑ, E, B = fields(x, t)
    prim = primitive_state(eos, m, ρ, v, s, Yₑ)
    return prim2con(eos, m, prim, E, B)
end

# The Gauss average over the directions `e ≤ E` of the cell centred at `x`,
# one direction at a time, innermost `e = 1`: three averages over the
# directions below `E` at the nodes along `E`, then their average.
@inline cell_average(fields, eos, provider, x::SVector{3}, t, h, ::Val{0}) =
    case_cons(fields, eos, provider, x, t)
@inline function cell_average(fields, eos, provider, x::SVector{3,T}, t, h,
                              ::Val{E}) where {T,E}
    node(q) = x + SVector(ntuple(e -> e == E ? gauss3_node(h, q) : zero(T), Val(3)))
    f₋ = cell_average(fields, eos, provider, node(1), t, h, Val(E - 1))
    f₀ = cell_average(fields, eos, provider, x, t, h, Val(E - 1))
    f₊ = cell_average(fields, eos, provider, node(3), t, h, Val(E - 1))
    return gauss3_average(f₋, f₀, f₊)
end

@kernel function cell_average_kernel!(uU, fields, eos, provider, t, @Const(origins),
                                      @Const(spacings), ::Val{D}) where {D}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    xc = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    acc = cell_average(fields, eos, provider, xc, oftype(h, t), h, Val(D))
    @inbounds for v in 1:NU
        uU[n..., v, b] = acc[v]
    end
end

@kernel function potential_kernel!(uA, potential, t, @Const(origins), @Const(spacings),
                                   ::Val{D}, ::Val{C}, ::Val{j}) where {D,C,j}
    I = @index(Global, NTuple)                       # local, owned
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(C))
    A, Φ̃ = potential(x, oftype(h, t))
    @inbounds uA[n..., 1, b] = j ≤ 3 ? A[j] : Φ̃
end

"""
    case_state!(u, p::MHDProblem, case::MHDCase, t) -> u

The case's solution at time `t` written into the state vector `u` over
`p.sets`: the exact cell averages of the conserved matter variables (Gauss,
three points per dimension), and the exact point values of `A_j` at the edges
and of `Φ̃` at the vertices.
"""
function case_state!(u, p::MHDProblem{T,D}, case::MHDCase, t) where {T,D}
    sets = p.sets
    eos = eos_on(get_backend(sets[SEG_U].work), case.eos)
    map_blocks!(cell_average_kernel!, sets[SEG_U], statearray(u, sets, SEG_U), case.fields,
                eos, case.provider, T(t), p.origins, p.spacings, Val(D))
    for j in 1:3
        i = SEG_A + j - 1
        map_blocks!(potential_kernel!, sets[i], statearray(u, sets, i), case.potential, T(t),
                    p.origins, p.spacings, Val(D), Val(staggers(sets[i])), Val(j))
    end
    map_blocks!(potential_kernel!, sets[SEG_Φ], statearray(u, sets, SEG_Φ), case.potential,
                T(t), p.origins, p.spacings, Val(D), Val(staggers(sets[SEG_Φ])), Val(4))
    return u
end

"""
    exact_state(p::MHDProblem, case::MHDCase, t) -> u

A fresh state vector holding the case's solution at time `t`
([`case_state!`](@ref)): the initial data at `t = 0`, and the reference of an
exact solution at any `t`.
"""
exact_state(p::MHDProblem, case::MHDCase, t) = initial_state!(statevector(p.sets), p, case, t)

"""
    initial_state!(u, p::MHDProblem, case::MHDCase, t) -> u

The case's solution at `t` written into `u` ([`case_state!`](@ref)): the
verb a script that owns the time loop calls for its initial data ("Two
scripts" in `CODE.md`; added in step 10). Public, not exported.
"""
initial_state!(u, p::MHDProblem, case::MHDCase, t) = case_state!(u, p, case, t)

# --- Dirichlet boundaries ----------------------------------------------------------------

"""
    DirichletBoundary{R}(fields, potential, eos, provider; absolute = false)

The case's own solution as the boundary of one evolved set (step 6;
"Boundaries" in `CODE.md`), at the time of each fill. `R` names the set:
`:U`, or `1`, `2`, `3` (`A_R`) or `4` (`Φ̃`). [`boundary_at`](@ref) turns it
into the hook TreeAMR's `fill_ghosts!` takes.

- **The matter**, `R = :U`: every ghost cell facing outside the domain holds
  the conserved state of `fields` at its center, through
  `CellBoundary(AllVariables(…))`, which runs as a kernel on every backend.
  Ghosts of `U` hold cell averages; the point value is the average where the
  solution is uniform (a shock tube's far states) and differs by `O(h²)` of
  the second derivative where it is smooth (the current sheet's tails).
  Only TreeAMR's own fills of `U` use it — `regrid!`'s before its transfer
  and the initial adaptation's; the scheme's ghost fill is the primitives',
  whose hook [`primitive_boundary`](@ref) derives from this one (amended
  2026-10-23).
- **The potentials**, `R = 1 … 4`: Dirichlet in the *field*, not in the
  potential (proposed in step 6). Each ghost point holds the nearest evolved
  point's value plus the case's difference between the two points,
  `A(x_g) = A(x_r) + [A_case(x_g, t) − A_case(x_r, t)]`, so that the curl —
  `𝔅` on the boundary faces and in the ghosts — is the case's, while the
  potential's level, which no physical quantity reads, is the evolved one's.
  A Dirichlet `A` itself is wrong at the first implicit stage of a tableau
  whose two abscissae differ: a uniform state's `A` grows as `−tE`, and the
  stage's own `A` is the explicit combination, at `tⁿ + c̃Δt`, while
  `solve_imp!` is told `tⁿ + cΔt` (SSP3(4,3,3)'s first stage has `c̃ = 0`,
  `c = 0.24`). Measured in step 6: a uniform flow at `vˣ = 0.999` through
  the box lost a fifth of its density in the last cell within 0.05 of a
  crossing that way. The difference is the field's, so it does not depend on
  `t` where the field does not. *(Amended in step 8:)* this is TreeAMR's
  relative hook, `CellBoundary(Relative(f))` with `f` the case's potential
  (AMR-7, landed in TreeAMR 0.2.2): a kernel on every backend, which takes
  `x_r` across a block face from the block's own (filled) ghost and runs each
  region after the stage that fills what it reads, so one fill applies the
  complete condition. A potential's wall planes at an outer face are own
  points (TreeAMR 0.3, closed blocks), which the hook prescribes too
  (`wall = :prescribed`), from the point one plane in. Until step
  8 this was TreeAMR's region form, CPU-only, with this package's own
  `low_walls!` and `outer_points!` kernels around it.
- **The potentials of a case whose potential does not depend on `t`**,
  `absolute = true` (decided by Erik 2026-11-03, "option 1"; the case's
  `static_potential`): the case's `A_j` and `Φ̃` themselves at every ghost
  and on every prescribed wall, `CellBoundary(AllVariables(…); wall =
  :prescribed)`, which writes the wall planes too. The relative rule copies
  the evolved potential's offset from `x_r` to `x_g` unchanged along the
  normal; a gauge drift `∇χ` of the evolved `A` (truncation order, and a
  static `χ` the damped Lorenz gauge does not damp) is then no longer a
  gradient across the face, its curl puts spurious `𝔅` into the ghost and
  wall faces, and the loop boundary → `𝔅` → `E` → `A` → `χ` grows (measured
  in step 9 on the Kerr-Schild cases: `e^{0.11 t/M}` at `h = M/4`; "Boundaries"
  in `CODE.md`). The absolute hook pins the gauge at the boundary. It is
  valid **only** where `A_case` does not depend on `t`: under IMEX the
  stage's `A` is at `tⁿ + c̃Δt` while `solve_imp!` is told `tⁿ + cΔt`, and a
  potential that moves with `t` would be filled at the wrong time (above).
  Ignored for `R = :U` and `:W`.

As in TreeHydro, a Dirichlet boundary is exact only until a wave arrives; a
case's `t_end` is chosen before that.
"""
struct DirichletBoundary{R,FL,PT,EOS,MET}
    fields::FL
    potential::PT
    eos::EOS
    provider::MET
    absolute::Bool
end
DirichletBoundary{R}(fields, potential, eos, provider; absolute::Bool=false) where {R} =
    DirichletBoundary{R,typeof(fields),typeof(potential),typeof(eos),typeof(provider)}(
        fields, potential, eos, provider, absolute)

# The time-`t` callback of a `DirichletBoundary`, `isbits` whenever the hook
# is.
struct DirichletAt{H,T}
    hook::H
    t::T
end

@inline point3(x::NTuple{D,T}) where {D,T} = SVector(ntuple(e -> e ≤ D ? x[e] : zero(T), Val(3)))

# The matter: `g(x, δ) -> vals`, `x` the ghost cell's center.
@inline function (d::DirichletAt{<:DirichletBoundary{:U}})(x::NTuple{D,T}, δ) where {D,T}
    h = d.hook
    return case_cons(h.fields, h.eos, h.provider, point3(x), T(d.t))
end

# The case's potential `R` at the point `x`.
@inline function case_potential(h::DirichletBoundary{R}, x::NTuple{D,T}, t) where {R,D,T}
    A, Φ̃ = h.potential(point3(x), T(t))
    return R ≤ 3 ? A[R] : Φ̃
end

# The primitives of the matter's ghost fill (step 8): `g(x, δ) -> vals`, the
# case's `ρ★, s, Yₑ, uⁱ = W vⁱ, Eⁱ` at the ghost cell's center (`W`'s slots).
@inline function (d::DirichletAt{<:DirichletBoundary{:W}})(x::NTuple{D,T}, δ) where {D,T}
    h = d.hook
    return case_prims(h.fields, h.provider, point3(x), T(d.t))
end

@inline function case_prims(fields, provider, x::SVector{3,T}, t::T) where {T}
    m = point_metric(provider, t, x)
    ρ, v, s, Yₑ, E, B = fields(x, t)
    v3, E3 = SVector{3,T}(v), SVector{3,T}(E)
    W = inv(sqrt(one(T) - dot_up(m, v3, v3)))
    u = W * v3
    return W_store(T(ρ), T(s), T(Yₑ), u, E3)
end

"""
    primitive_boundary(hook) -> DirichletBoundary{:W}

The boundary hook of the primitive set `W` that goes with the matter's hook
`hook` ([`MHDProblem`](@ref)'s `boundary[1]`): for the case's Dirichlet data
([`DirichletBoundary`](@ref)`{:U}`) the same case's primitives at each ghost
cell's center (added in step 8). Any other matter hook has no primitive
counterpart this package can derive; pass `boundary_W` to `MHDProblem`.
"""
primitive_boundary(hook::DirichletBoundary{:U}) =
    DirichletBoundary{:W}(hook.fields, hook.potential, hook.eos, hook.provider;
                          absolute=hook.absolute)
primitive_boundary(hook) = throw(ArgumentError(
    "the matter's boundary hook $(typeof(hook)) has no primitive counterpart: since step 8 " *
    "the matter's ghost fill prolongates the primitives (CODE.md, \"Refinement\"), so a " *
    "non-periodic problem needs a hook for the primitive set `W` too — pass `boundary_W`."))

# A potential: `f(x) -> value`, the case's potential `R` at the point `x`, which
# TreeAMR's relative hook turns into `f(x_g) + (u(x_r) − f(x_r))` (AMR-7).
@inline (d::DirichletAt{<:DirichletBoundary})(x::NTuple{D,T}) where {D,T} =
    case_potential(d.hook, x, d.t)

# A static potential, absolute: `g(x, δ) -> (value,)`, the case's potential at
# the ghost or wall point itself (decided by Erik 2026-11-03).
struct AbsolutePotentialAt{H}
    at::H
end
@inline (a::AbsolutePotentialAt)(x::NTuple{D,T}, δ) where {D,T} = (a.at(x),)

# A kernel's adaptor makes a device-resident table `isbits` only through a rule for every
# struct around it: the hooks carry the EOS ([`eos_on`](@ref) uploads it; these unwrap it
# at the launch). Without them a tabulated case's boundary kernel is refused on a device.
Adapt.adapt_structure(to, h::DirichletBoundary{R}) where {R} =
    DirichletBoundary{R}(Adapt.adapt(to, h.fields), Adapt.adapt(to, h.potential),
                         Adapt.adapt(to, h.eos), Adapt.adapt(to, h.provider);
                         absolute=h.absolute)
Adapt.@adapt_structure DirichletAt
Adapt.@adapt_structure AbsolutePotentialAt

boundary_at(hook::DirichletBoundary{:U}, t) = CellBoundary(AllVariables(DirichletAt(hook, t)))
boundary_at(hook::DirichletBoundary{:W}, t) = CellBoundary(AllVariables(DirichletAt(hook, t)))
# A potential's wall plane at an outer face is an own point of its blocks
# (TreeAMR 0.3, closed blocks); `wall = :prescribed` has every fill write it
# from the hook too, a Dirichlet condition, as AMR-8's symmetric walls did.
boundary_at(hook::DirichletBoundary, t) =
    hook.absolute ?
    CellBoundary(AllVariables(AbsolutePotentialAt(DirichletAt(hook, t))); wall=:prescribed) :
    CellBoundary(Relative(DirichletAt(hook, t)); wall=:prescribed)

"""
    case_boundary(case::MHDCase) -> NTuple{5,DirichletBoundary} or nothing

`nothing` for a case periodic in every dimension, and otherwise one
[`DirichletBoundary`](@ref) per evolved set, in the order `(U, A₁, A₂, A₃,
Φ̃)` that [`MHDProblem`](@ref)'s `boundary` takes: the case's solution as the
boundary of every set, through each set's own hook (AMR-2), its EOS `eos` (the case's,
or the case's uploaded where the hooks run on a device, [`eos_on`](@ref)). Each hook holds
only what it reads — the matter's the fields, `eos` and the provider, a potential's the
potential — since a hook is a kernel argument and CUDA's are limited to 32 KiB. The potentials'
hooks are absolute where the case declares its potential static
(`static_potential`, decided by Erik 2026-11-03), and relative otherwise.
"""
function case_boundary(case::MHDCase; static_potential::Bool=case.static_potential,
                       eos=case.eos)
    all(case.periodic) && return nothing
    # Each hook carries only what it reads: the matter's the fields, the EOS and the
    # provider (and so the primitives' hook made from it), a potential's the potential. A
    # hook is a kernel argument, and CUDA's limit is 32 KiB: the tabulated star's fields
    # and potential are 14.5 KB each, and a potential's hook with everything came to 32.8.
    matter = DirichletBoundary{:U}(case.fields, nothing, eos, case.provider;
                                   absolute=static_potential)
    mk(R) = DirichletBoundary{R}(nothing, case.potential, nothing, nothing;
                                 absolute=static_potential)
    return (matter, mk(1), mk(2), mk(3), mk(4))
end

"""
    l1_errors(p::MHDProblem, u, uref) -> (U = NTuple{10}, A = NTuple{3}, Φ = scalar)

The volume-weighted L1 norm of `u − uref`, per variable of every evolved
set, each normalized by the domain volume: the mean absolute error of each
of the ten cell variables, of each `A_j` and of `Φ̃`. Each point counts
once, at its primary copy, and `u` is unified first (TreeAMR's `unify!`):
a block holds both face planes of a vertex-like dimension (TreeAMR 0.3),
and the integrator carries each copy with its own right-hand side, which
only the next fill makes one number.
"""
function l1_errors(p::MHDProblem{T,D}, u, uref) where {T,D}
    diff = unify!(copy(u), p.sets, p.schedules) .- uref
    forest = p.sets[SEG_U].forest
    vol = prod(e -> forest.extents[e][2] - forest.extents[e][1], 1:D)
    l1(i, v) = mesh_mapreduce(abs, +, zero(T), p.sets[i], stateview(diff, p.sets, i);
                              vars=v, weight=key -> spacing(T, forest, key)^D,
                              primaries=true) / T(vol)
    return (; U=ntuple(v -> l1(SEG_U, v), Val(NU)), A=ntuple(j -> l1(SEG_A + j - 1, 1), 3),
            Φ=l1(SEG_Φ, 1))
end

"""
    central_density(p::MHDProblem, u, t; metric = p.provider) -> ρ

The largest rest-mass density over the owned cells, from one recovery of `u`
at `t` in `metric` ([`update_primitives!`](@ref)): the central density of a
star, whose drift and oscillations rung 4 measures (step 10). A diagnostic,
called outside a step.
"""
function central_density(p::MHDProblem{T}, u, t; metric=p.provider) where {T}
    P = update_primitives!(p, u, T(t); metric)
    return mesh_mapreduce(identity, max, zero(T), P; vars=P_RHO)
end

"""
    profile_1d(p::MHDProblem{T,1}, u, t) -> NamedTuple

The state `u` of a one-dimensional run as host arrays ordered by `x`: the
cell centers `x`; the evolved cell variables `U` (`ncells × 10`); and,
from one recovery of `u` at `t` ([`update_primitives!`](@ref)), the
rest-mass density `ρ`, the gas pressure `p`, the velocity `v`, the fields
`E` and `B` (each `ncells × 3`, contravariant, `B` the cell average of the
curl of `A` plus `𝔅̄`), the comoving `b² = B²/W² + (B·v)²` and the policy
flags. What the shock-tube claims read (step 6).
"""
function profile_1d(p::MHDProblem{T,1}, u, t) where {T}
    P = hostcopy(update_primitives!(p, u, T(t)))
    U = p.sets[SEG_U]
    uU = Array(statearray(u, p.sets, SEG_U))
    N = U.forest.N
    G = P.G[1]
    nb = nblocks(U)
    # This rank's cells in block order, then every rank's in rank order
    # (`gather_cells`): the serial run's cells, whatever the rank count.
    rows = [(n, b) for b in 1:nb for n in 1:N]
    Pv(n, b, v) = P.work[n + G, v, b]
    col(f) = gather_cells(p, T[f(n, b) for (n, b) in rows])
    x = col((n, b) -> coordinates(T, U, b, (n + U.G[1],))[1])
    vec3(slot) = reduce(hcat, [col((n, b) -> Pv(n, b, slot + k - 1)) for k in 1:3])
    ρ = col((n, b) -> Pv(n, b, P_RHO))
    pr = col((n, b) -> Pv(n, b, P_PRESS))
    v, E, B = vec3(P_V), vec3(P_E), vec3(P_B)
    b2 = col() do n, b
        # Flat or not, the comoving field from the cell's own metric.
        m = aux_metric(ntuple(k -> Pv(n, b, k), Val(NP)))
        vv = SVector{3,T}(Pv(n, b, P_V), Pv(n, b, P_V + 1), Pv(n, b, P_V + 2))
        BB = SVector{3,T}(Pv(n, b, P_B), Pv(n, b, P_B + 1), Pv(n, b, P_B + 2))
        W² = inv(one(T) - dot_up(m, vv, vv))
        dot_up(m, BB, BB) / W² + dot_up(m, BB, vv)^2
    end
    flags = gather_cells(p, UInt32[unsafe_trunc(UInt32, Pv(n, b, P_FLAGS)) for (n, b) in rows])
    Um = reduce(hcat, [col((n, b) -> uU[n, k, b]) for k in 1:NU])
    order = sortperm(x)
    return (; x=x[order], U=Um[order, :], ρ=ρ[order], p=pr[order], v=v[order, :],
            E=E[order, :], B=B[order, :], b2=b2[order], flags=flags[order])
end

# --- the integrator and the run --------------------------------------------------------

"""
    mhd_integrator(p::MHDProblem, u, t0, t1, nsteps; tableau = IMEXSSP3433(),
                   alias_u0 = false, reuse = nothing,
                   partition = u isa Array ? :even : nothing)

An IMEXRungeKutta integrator for `nsteps` fixed steps from `t0` to `t1`:
[`mhd_rhs!`](@ref) as the explicit part, [`ohm_solve!`](@ref) as the stage
solve, and the reset as both the stage limiter ([`reset_atmosphere!`](@ref)) and
the step limiter ([`reset_step!`](@ref), which also accounts what it injects)
("The IMEX package" in `CODE.md`). The stage arithmetic is threaded evenly
over a CPU state (`partition = :even`) and broadcast on a device
(`partition = nothing`), which a CPU state may also ask for, to measure it
(step 14). The step count is asserted, as in TreeHydro.
"""
function mhd_integrator(p::MHDProblem{T}, u, t0, t1, nsteps::Integer;
                        tableau=IRK.IMEXSSP3433(), alias_u0::Bool=false,
                        reuse=nothing, partition=u isa Array ? :even : nothing) where {T}
    nsteps ≥ 1 || throw(ArgumentError("nsteps must be at least 1, got $nsteps."))
    t0, t1 = T(t0), T(t1)
    prob = IRK.IMEXProblem(mhd_rhs!, ohm_solve!, u, (t0, t1), p)
    integ = IRK.init(prob, tableau; dt=(t1 - t0) / nsteps,
                     stage_limiter=reset_atmosphere!, step_limiter=reset_step!,
                     partition=partition, alias_u0=alias_u0,
                     reuse=reuse)
    integ.nsteps == nsteps || throw(ErrorException(
        "IMEXRungeKutta derived $(integ.nsteps) steps from dt = (t1 − t0)/$nsteps; the " *
        "driver sized its step for $nsteps."))
    return integ
end

"""
    chunk_count(t_end, chunk)

`⌈t_end / chunk⌉`, except that a quotient within a few ulp of an integer `m`
gives `m`, so that a run meant to be a whole number of chunks (or of steps)
is not given one more by rounding. TreeHydro's, which is IMEXRungeKutta's
`step_count` rule.
"""
function chunk_count(t_end::T, chunk::T) where {T}
    r = t_end / chunk
    m = round(r)
    tol = 4 * (eps(r) + eps(t_end) / chunk)
    return m ≥ 1 && abs(r - m) ≤ tol ? roundint(m) : ceilint(r)
end

"""
    default_cfl(limiter) -> 2/5, or 1/5 for :mp5

The light-speed CFL factor [`evolve!`](@ref) uses unless told otherwise:
TreeHydro's 2/5 under SSP3(4,3,3)'s explicit SSPRK(3,3) part (proposed in step
5), and 1/5 for MP5, whose monotonicity-preserving bound holds for a forward
Euler step of CFL `≤ 1/(1 + α) = 1/5` at its `α = 4` (Suresh & Huynh 1997,
§2.4; proposed in step 7): at 2/5 it fires the reset on Balsara 3 and 4 and
grows Balsara 3's dense shell to `ρ ≈ 600` (measured in step 7).
"""
default_cfl(limiter) = limiter === :mp5 ? 1 // 5 : 2 // 5

"""
    evolve!([T], case::MHDCase, Val(D); N, t_end, chunk = t_end, cfl = default_cfl(limiter), limiter,
            riemann = :hlle, ξ = default_ξ(case), stage = nothing, tableau = IMEXSSP3433(), G = 3,
            backend = CPU(), after_step = nothing, roots = case.roots, ops_U, ops_A, ops_Φ, ops_W,
            fixup = true, refine = nothing, maxlevel = 0, refine_tol, coarsen_tol,
            buffer = 0, ε = 1//100, ε_g = 1//1000, maxpasses = 10,
            after_regrid = nothing, step_cache = true, two_phase = nothing,
            interior = case.interior, workgroupsize = nothing, comm = nothing,
            checkpoint = nothing, checkpoint_every = 1, restart = nothing,
            checkpoint_io = :node) -> NamedTuple

Run `case` from `t = 0` to `t_end` on a mesh of `roots` root blocks (the
case's own by default) of `N` cells per dimension, in chunks of length
`chunk`. Each chunk's `Δt` is `cfl / (D · max λ/h)` from the light cone at
the chunk's start on the finest level ([`max_light_rate`](@ref)), rounded down
to a whole number of steps, and each chunk tallies its implicit stages and
resets ([`ohm_tally`](@ref)).

Returns `(; p, u, t, nsteps, chunks, nregrids, passes, nblocks, totals0)`:
the problem, the final state vector over `p.sets`, the final time, the total
step count, per chunk `(t, nsteps, dt, tally, nblocks)`, the number of
regrids that moved the mesh, the initial adaptation's passes (0 without
one), the final block count (the forest's leaves, `nleaves`, whatever the
rank count), and the [`conserved_totals`](@ref) of the
initial state on the initial mesh (which a regridded run's own initial
data on its final mesh does not reproduce to rounding).

**Running distributed** ("Running distributed" in `CODE.md`): `comm` is the
communicator the forest is built over — an `MPI.Comm` after the caller's
`using MPI; MPI.Init()`, or `nothing` for a serial run — and the call is
then collective, with the same arguments on every rank; the result is the
serial run's, bit for bit but for its sums, which agree to rounding.
`checkpoint` is a path the run writes ([`save_run`](@ref)) after every
`checkpoint_every`-th chunk and after the last, at the chunk boundary
before the next chunk's regrid, with `checkpoint_io` TreeIOHDF5's I/O
groups; `restart` is a path it continues from ([`load_run`](@ref)), at any
rank count, refusing a checkpoint whose recipe differs (every parameter
that decides the numbers; `t_end` may change). The restarted run is the
uninterrupted one bit for bit, its `totals0` to rounding where the
checkpoint was written at another rank count.

`after_step(p, u, t)`, if given, is called after every step with the
problem, the state and its time — a monitor, such as the `∇·𝔅` check of
rung 2a ([`max_div_B`](@ref); added in step 7); it must not change `u`.
It runs outside the step, with the step cache off, so a monitor that calls
[`update_primitives!`](@ref) or [`mhd_rhs!`](@ref) does not change the run.

The steps run with the **step cache** ([`with_step_cache`](@ref); added
2026-10-24, decided by Erik 2026-10-23): inside a step every recovery is
seeded from the cell's latest one in the step, and one whose input is that
one's bit for bit is reused; the cache is invalidated at the end of every
step and whenever stepping starts or stops, so a step is a pure function of
`uⁿ`, `t` and `Δt`. `step_cache = false` runs every recovery from the state,
as before (the negative control; "The step cache" in `CODE.md`).
`two_phase = true` recovers in the three passes of the two-phase recovery
rather than fused ([`MHDProblem`](@ref); added 2026-10-30): the same run bit
for bit. The default, `nothing`, is fused on the CPU (decided by Erik
2026-11-02) and split on a device (decided by Erik 2026-11-20).
`after_regrid(p, u, t)` likewise after every regrid that moved the mesh and
once after the initial adaptation (added in step 8). `interior` is the
case's black-hole interior ([`HoleInterior`](@ref)) unless given — another
radius or role, or `nothing` to evolve the inside too, for a measurement
(added in step 9). `workgroupsize` is the kernels' workgroup on a device
([`MHDProblem`](@ref); added in step 14).

`limiter` has no default (see [`MHDProblem`](@ref)); `cfl = 2/5` is
TreeHydro's, under the explicit part of SSP3(4,3,3) — the SSPRK(3,3)
tableau, whose SSP coefficient is 1 (proposed in step 5) — and 1/5 under
MP5 ([`default_cfl`](@ref), step 7).

**The mesh** *(added in step 8; "Refinement" in `CODE.md`)*:

- `refine(forest)`, if given, refines the root mesh before anything is
  built — a static mesh, fixed for the run (`maxlevel = 0`), such as the two
  levels of the coarse-fine tests. It must leave the forest balanced.
- `maxlevel > 0` adapts the mesh: [`adapt_to_initial_data!`](@ref) over every
  field set on the case's own data (AMR-3), then a regrid at the start of
  every chunk after the first ([`regrid_state!`](@ref)), both with
  [`mhd_flags`](@ref) — Löhner on `ρ`, `|B|`, `|E|` — whose `refine_tol` and
  `coarsen_tol` have no default; `buffer` is the margin in cells `regrid!`
  dilates the firing cells by, the case's (feature speed × `chunk`), and
  `ε`, `ε_g` the indicator's floors.
- `ops_U`, `ops_A`, `ops_Φ` are the operators of the matter, of `A_j` and of
  `Φ̃`, and `ops_W` those of the primitives' ghost fill ([`MHDProblem`](@ref)),
  `fixup = false` skips `unify!` (the
  negative control).
"""
function evolve!(::Type{T}, case::MHDCase{T,D}, ::Val{D}; N::Integer, t_end, chunk=t_end,
                 cfl=nothing, limiter=nothing, riemann::Symbol=:hlle, ξ=nothing,
                 stage=nothing, tableau=IRK.IMEXSSP3433(), G::Integer=3,
                 backend=CPU(), after_step=nothing, roots=case.roots,
                 ops_U::Operators=default_ops_U(), ops_A::Operators=default_ops_A(),
                 ops_Φ::Operators=default_ops_Φ(), ops_W::Operators=default_ops_W(),
                 fixup::Bool=true, refine=nothing,
                 maxlevel::Integer=0, refine_tol=nothing,
                 coarsen_tol=nothing, buffer::Integer=0, ε=T(1 // 100), ε_g=T(1 // 1000),
                 maxpasses::Integer=10, after_regrid=nothing,
                 step_cache::Bool=true, two_phase::Union{Bool,Nothing}=nothing,
                 interior=case.interior, workgroupsize=nothing, comm=nothing,
                 checkpoint=nothing, checkpoint_every::Integer=1, restart=nothing,
                 checkpoint_io=:node) where {T,D}
    t_end, chunk = T(t_end), T(chunk)
    cfl = T(cfl === nothing ? default_cfl(limiter) : cfl)
    t_end > 0 || throw(ArgumentError("t_end must be positive, got $t_end."))
    chunk > 0 || throw(ArgumentError("chunk must be positive, got $chunk."))
    cfl > 0 || throw(ArgumentError("cfl must be positive, got $cfl."))
    maxlevel ≥ 0 || throw(ArgumentError("maxlevel must be non-negative, got $maxlevel."))
    if maxlevel > 0
        (refine_tol === nothing || coarsen_tol === nothing) && throw(ArgumentError(
            "an adaptive run (maxlevel = $maxlevel) needs refine_tol and coarsen_tol: " *
            "they have no default, since what is under-resolved is the case's to say " *
            "(CODE.md, \"Refinement\")."))
        refine === nothing || throw(ArgumentError(
            "pass either a static mesh (`refine`) or an adaptive one (`maxlevel > 0`), not " *
            "both: the first regrid would undo the static refinement."))
    end
    checkpoint_every ≥ 1 || throw(ArgumentError(
        "checkpoint_every must be at least 1, got $checkpoint_every."))
    ξ = resolve_ξ(T, ξ, case)
    rs = roots isa Tuple ? ntuple(d -> Int(roots[d]), D) : ntuple(_ -> Int(roots), D)
    # What decides the numbers, which a restart must not change ("Checkpoints" in
    # `CODE.md`); `t_end` may, which is how a run is continued.
    recipe = checkpoint_recipe(; T, D, N, roots=rs, chunk, cfl, limiter, riemann, ξ, stage,
                               tableau=tableau.name, G, ops_U, ops_A, ops_Φ, ops_W, fixup,
                               maxlevel, refine_tol, coarsen_tol, buffer, ε, ε_g, interior,
                               case=nameof(typeof(case.fields)), extents=case.extents,
                               periodic=case.periodic)
    build(f) = MHDProblem(f, case; ξ=ξ, stage=stage, limiter=limiter, riemann=riemann,
                          ops_U=ops_U, ops_A=ops_A, ops_Φ=ops_Φ, ops_W=ops_W, fixup=fixup,
                          two_phase=two_phase, interior=interior,
                          workgroupsize=workgroupsize)
    flagkw = (; refine_tol, coarsen_tol, maxlevel, ε, ε_g)
    if restart === nothing
        forest = Forest{T}(rs; N=N, periodic=case.periodic, extents=case.extents, comm=comm)
        refine === nothing || refine(forest)
        f = mhd_fields(T, forest; G=G, backend=backend)
        passes = 0
        if maxlevel > 0
            passes = adapt_case!(f, case, () -> build(f); ops_U=ops_U, ops_A=ops_A,
                                 ops_Φ=ops_Φ, buffer=buffer, maxpasses=maxpasses, flagkw...)
        end
        p = build(f)
        # The exact potentials satisfy the walls' condition already: the relative
        # hook gives `f(x_w) + (f(x_r) − f(x_r)) = f(x_w)` bit for bit.
        u = exact_state(p, case, zero(T))
        totals0 = conserved_totals(p, u)
        maxlevel > 0 && after_regrid !== nothing && after_regrid(p, u, zero(T))
        c0, t, total, nregrids, chunks = 0, zero(T), 0, 0, Any[]
    else
        # A restart: the checkpoint's forest, distributed over `comm` whatever the rank
        # count it was written at, and its state; `B`, `P`, `W` and the scratch sets are
        # rebuilt by the first fill. The step cache is not stored: it holds nothing across
        # a step, so the restarted run is the uninterrupted one bit for bit.
        ck = load_run(restart; backend, comm, types=(T,))
        check_recipe(ck.recipe, recipe)
        p = build(mhd_fields(T, ck.forest; G=G, backend=backend))
        u = restore_run!(statevector(p.sets), p, ck)
        run = ck.run
        c0, t, total, nregrids, passes = run.chunk, T(run.t), run.nsteps, run.nregrids,
                                         run.passes
        totals0 = ntuple(v -> T(run.totals0[v]), Val(NU))
        chunks = Any[run.chunks...]
    end
    nchunks = chunk_count(t_end, chunk)
    c0 < nchunks || throw(ArgumentError(
        "the restart is at chunk $c0 (t = $t), and t_end = $t_end is $nchunks chunks: " *
        "nothing is left to run."))
    integ = nothing
    for c in (c0 + 1):nchunks
        t1 = c == nchunks ? t_end : c * chunk
        clear_tally!(p)
        if maxlevel > 0 && c > 1
            r = regrid_state!(p, u, t; flags=flags(p, u, t; flagkw...), buffer=buffer)
            if r.moved
                p, u = r.p, r.u
                integ = nothing                       # the scratch has the old length
                nregrids += 1
                after_regrid === nothing || after_regrid(p, u, t)
            end
        end
        rate = max_rate(p, u, t)
        nsteps = chunk_count((t1 - t) * D * rate, cfl)
        integ = mhd_integrator(p, u, t, t1, nsteps; tableau=tableau, alias_u0=true,
                               reuse=integ)
        if after_step === nothing
            with_step_cache(() -> IRK.solve!(integ), p; active=step_cache)
        else
            while integ.nstep < integ.nsteps
                with_step_cache(() -> IRK.step!(integ), p; active=step_cache)
                after_step(p, integ.u, integ.t)
            end
        end
        push!(chunks, (; t=t1, nsteps, dt=(t1 - t) / nsteps, tally=ohm_tally(p),
                       nblocks=nleaves(p.f.U.forest)))
        total += nsteps
        t = t1
        # At the chunk boundary, before the next chunk's regrid, as TreeHydro and TreeGH
        # write: a restart regrids first, through the loop's own code.
        if checkpoint !== nothing && (c % checkpoint_every == 0 || c == nchunks)
            save_run(checkpoint, p, u; recipe, io=checkpoint_io,
                     run=(; chunk=c, t, nsteps=total, nregrids, passes, totals0,
                          chunks=Tuple(chunks)))
        end
    end
    return (; p, u, t, nsteps=total, chunks, nregrids, passes, nblocks=nleaves(p.f.U.forest),
            totals0)
end

"""
    default_ξ(case)

The gauge damping [`evolve!`](@ref) uses unless told otherwise: `16/L`, `L`
the longest side of the case's box — the damping of a gauge wave over a
sixteenth of a light crossing of the box — on every mesh, uniform or refined
(decided in step 8; amended in step 8, decided by Erik 2026-10-15: "choose
the same ξ on all meshes", where it was zero on a uniform mesh). The longest
side, so that a case and its extrusions and turns (`extruded_case`,
`turned_case`), whose boxes differ only in the directions added or
exchanged, damp alike and stay the same run bit for bit. Across a
coarse-fine face the gauge reaches `𝔅`, since the prolongation of a coarse
gradient is not a fine one: a pure gauge pulse on a two-level unit box left
`max |𝔅|` at 5.3e−2 of the pulse after two light crossings with `ξ = 0`,
6.6e−3 at `ξ = 4`, 1.4e−3 at 16 and 3.3e−4 at 64, while the oblique Alfvén
wave's errors moved in the fifth digit at most (measured in step 8,
`bin/gauge_damping.jl`). On a uniform mesh `−∇(αΦ − β·A)` is a discrete
gradient whose curl vanishes to rounding, so there `ξ` moves `A`'s gauge
part and `Φ̃`, and `𝔅` and the matter only at rounding ("Refinement" in
`CODE.md`). A case's own `gauge_damping` replaces it (step 9, round 4: the
spinning Wald case, whose static potential has `Φ̃ ≠ 0`, takes 0).
"""
default_ξ(case::MHDCase{T,D}) where {T,D} =
    case.gauge_damping === nothing ?
    16 / maximum(e -> case.extents[e][2] - case.extents[e][1], 1:D) : case.gauge_damping

# `evolve!`'s `ξ`: the default, or the caller's in the case's type.
resolve_ξ(::Type{T}, ξ, case) where {T} = ξ === nothing ? T(default_ξ(case)) : T(ξ)

"""
    MHDProblem(f::MHDFields, case::MHDCase; limiter, ξ = default_ξ(case),
               interior = case.interior, kw...)

The problem of `case` over the field sets `f`, as [`evolve!`](@ref) builds
it: the case's EOS, options, conductivity, provider, background field and
Dirichlet hooks ([`case_boundary`](@ref)), its interior, the gauge damping
`ξ`, and every other keyword of [`MHDProblem`](@ref) (`stage`, `riemann`,
the operators, `fixup`, `two_phase`) passed on. What a script that owns the
time loop builds, and rebuilds after a regrid ("Two scripts" in `CODE.md`;
added in step 10).
"""
function MHDProblem(f::MHDFields, case::MHDCase; limiter, ξ=nothing, interior=case.interior,
                    kw...)
    T = eltype(f.U.work)
    backend = get_backend(f.U.work)
    eos = eos_on(backend, case.eos)                  # uploaded once, for all three
    return MHDProblem(f; eos, opts=case.opts, proj=case.proj, pol=case.pol,
                      model=case.model, provider=case.provider, 𝔅̄=case.𝔅̄,
                      ξ=resolve_ξ(T, ξ, case), limiter=limiter,
                      boundary=case_boundary(case; eos),
                      interior=interior_target(interior, case; eos), kw...)
end

"""
    default_ops_U(), default_ops_A(), default_ops_Φ(), default_ops_W()

The operators [`MHDProblem`](@ref) and [`evolve!`](@ref) give the matter set,
the three `A_j` and `Φ̃` unless told otherwise ("Field sets" and
"Refinement" in `CODE.md`):

- the matter: Conservative width 3, **bounded** (decided in step 8), in the
  regrid (since AMR-10 its ghost fill is `W`'s);
- `A_j`: the conservative family on a staggered set (AMR-9, TreeAMR 0.2.2)
  — along the edge's own (cell-like) direction the conservative width-3
  operators, whose restriction is the two-edge mean and whose prolongation's
  two children average to the parent, so a coarse edge's line integral is
  the sum of its fine edges' after every fill and regrid, and with it the
  magnetic flux through a coarse face the sum through its fine faces; across
  it (vertex-like) point values of width 4 (decided in step 8 by the
  interface-order table, re-measured under this family);
- `Φ̃`: PointValue width 4. A vertex set has no cell-like dimension, so the
  conservative family would be this very operator (injection and width-4
  Lagrange); it stays PointValue (decided in step 8).
- `W`, the primitives the matter's ghost fill exchanges: the matter's own
  operator, Conservative width 3, `bounded = true` — the parent-centred
  primitive fill (decided by Erik 2026-10-23; "Refinement" in `CODE.md`). At
  a same-level face a ghost is a copy of its owner, exact; at a refinement
  boundary a fine ghost is its coarse parent plus an MC-limited linear slope
  centred on the parent, scaled so that every child stays within the parent
  and its face neighbours (TreeAMR's bounded prolongation, AMR-4) — Athena++'s
  prolongation of the primitives (Stone et al. 2020, ApJS 249, 4, §2.1.5),
  with MC slopes for its minmod — and a coarse ghost is the `2^D` mean of the
  fine cells it covers. At the foot of a jump a child takes its parent's
  value, so a dense neighbour does not leak into an atmosphere cell's
  children, where the PointValue width-2 fill of step 8 did. Every variable
  has its own slope, so a ghost is not a convex mixture of whole states; the
  48 perturbed magnetized runs saw no blow-up and no repaired ghost with it.
"""
default_ops_U() = Operators(; prolongation=3, restriction=2, family=Conservative, bounded=true)
default_ops_A() = Operators(; prolongation=3, restriction=2, family=Conservative,
                            vertex_prolongation=4)
default_ops_Φ() = Operators(; prolongation=4, restriction=4)
default_ops_W() = Operators(; prolongation=3, restriction=2, family=Conservative, bounded=true)

"""
    adapt_case!(f::MHDFields, case, build; ops_U, ops_A, ops_Φ, refine_tol, coarsen_tol,
                maxlevel, ε, ε_g, buffer, maxpasses) -> passes

The initial adaptation of every field set of `f` to `case`'s data at
`t = 0` (AMR-3, TreeAMR's several-set `adapt_to_initial_data!`): each pass
fills the evolved sets from the case — the conserved state *at* each cell's
centre, a point value standing in for the cell average while the mesh is
being chosen, and the potentials at their points — fills their ghosts with
their own hooks, recovers `P` and flags by [`mhd_flags`](@ref), and regrids
with `transfer = false`; the scratch sets are resized. On the converged mesh
the driver then writes the exact initial data ([`exact_state`](@ref)).
`build()` makes the problem over `f` as the forest stands. Throws if the
cycle has not converged within `maxpasses`.
"""
function adapt_case!(f::MHDFields{D}, case::MHDCase{T,D}, build; ops_U, ops_A, ops_Φ,
                     refine_tol, coarsen_tol, maxlevel, ε, ε_g, buffer,
                     maxpasses) where {T,D}
    eos = eos_on(get_backend(f.U.work), case.eos)
    hooks = case_boundary(case; eos)
    hook(i) = hooks === nothing ? nothing : boundary_at(hooks[i], zero(T))
    fl, pt, prov = case.fields, case.potential, case.provider
    initU = AllVariables(x -> case_cons(fl, eos, prov, point3(x), zero(T)))
    initA(R) = AllVariables(x -> (let (A, Φ̃) = pt(point3(x), zero(T))
                                      R ≤ 3 ? A[R] : Φ̃
                                  end,))
    ev = evolved_sets(f)
    ops = (ops_U, ops_A, ops_A, ops_A, ops_Φ)
    evolved = ntuple(i -> ev[i] => (ops[i], hook(i)), 5)
    scratch = (f.P => nothing, f.W => nothing, map(s -> s => nothing, f.B)...,
               map(s -> s => nothing, f.F)...,
               map(s -> s => nothing, f.EMF)..., f.X => nothing)
    nscratch = length(scratch)
    initial = (initU, initA(1), initA(2), initA(3), initA(4), ntuple(_ -> nothing, nscratch)...)
    criterion(sets) = begin
        p = build()
        # The cycle filled every set with its hook, which applies the complete
        # boundary condition (AMR-7; Erik, 2026-10-15).
        primitives_from_sets!(p, zero(T))
        mhd_flags(p; refine_tol, coarsen_tol, maxlevel, ε, ε_g)
    end
    _, passes, converged = adapt_to_initial_data!((evolved..., scratch...); initial=initial,
                                                  flags=criterion, buffer=buffer,
                                                  maxpasses=maxpasses)
    converged || throw(ErrorException(
        "the initial adaptation had not converged after $passes passes: the criterion " *
        "kept asking for a different mesh. Each pass re-evaluates the case on the new " *
        "mesh, so it stops when nothing new is asked — a refine_tol below what the data " *
        "can reach, or too high a maxlevel for a discontinuity, keeps it going."))
    return passes
end

evolve!(case::MHDCase{T}, valD::Val; kwargs...) where {T} = evolve!(T, case, valD; kwargs...)

"""
    convergence_rate(hs, errs)

Least-squares slope of `log(err)` against `log(h)`: the measured order of a
convergence study, from more than two resolutions at once so that one noisy
point does not become the answer. TreeHydro's (after TreeWave's and
TreeAMR's), copied rather than depended on.
"""
function convergence_rate(hs, errs)
    x = log.(hs)
    y = log.(errs)
    n = length(x)
    x̄, ȳ = sum(x) / n, sum(y) / n
    return sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
end
