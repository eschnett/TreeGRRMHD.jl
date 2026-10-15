# What the resolved releases have to provide before any of the scheme can be
# written, and which of the upstream items in `PLAN.md` have landed.
#
# `Project.toml` takes TreeAMR and SpacetimeMetrics from the General
# registry and EntropyEOS and IMEXRungeKutta from their remotes through
# `[sources]`, so what these tests run against is a release or a pushed
# commit and *not* a sibling checkout. A release that lost a name this
# package calls would otherwise be found by a `MethodError` in the middle of
# a step rather than here; the name lists are the cheapest thing that fails
# when an upstream release renames something.
#
# Then one testset per upstream ID still waiting ("Upstream requirements" in
# `PLAN.md`). Each detects its item by `isdefined`, `hasmethod` or
# `applicable` — never by a version number — and says, in an `@info` naming
# the ID, whether it is waiting and which step it blocks. A waiting item is
# not a failure: the suite stays green, and the log says what the next step
# is waiting for. Once an item has landed and `[compat]` requires a release
# that contains it, its probe is retired: the testset is deleted, the item
# is marked "landed in <package> <version>" in `PLAN.md`, and its names
# join the lists below (decided by Erik 2026-10-14). That retired AMR-1 to
# AMR-6 (TreeAMR 0.2.1), EOS-1 and EOS-2 (EntropyEOS 1.0.0) and SM-1
# (SpacetimeMetrics 1.7.0); the behaviour this package relies on from them
# is tested where it is used (`test/upstream_behaviour_tests.jl` for the
# bounded prolongation, `test/pointwise_tests.jl` for `K_ij`).
#
# Nothing here is magnetohydrodynamics. These are claims about the mesh, the
# equation of state, the analytic metrics, the integrator, and the two
# host-side helpers this package keeps of its own.

using KernelAbstractions: CPU, @kernel, @index
using StaticArrays: SVector
import Pkg
import SpacetimeMetrics

# Reached through the package rather than added to the test environment a
# second time: neither is in the General registry, so the test environment
# would need `[sources]` entries of its own, which could then disagree with
# the package's.
const EOS = TreeGRRMHD.EntropyEOS
const IRK = TreeGRRMHD.IRK

# A forest small enough to check point by point and refined enough that the
# blocks do not all have the same spacing: two roots per dimension, one of
# them refined, 2:1 balanced.
function prereq_forest(::Val{D}; N=8, extents=ntuple(_ -> (0.0, 1.0), D)) where {D}
    forest = Forest(ntuple(_ -> 2, D); N=N, periodic=ntuple(_ -> true, D),
                    extents=extents)
    refine!(forest, [first(forest.leaves)])
    balance!(forest)
    return forest
end

# A value per variable that is different at every point and in every set,
# so that a segment in the wrong place or a transposed copy would show.
prereq_values(nvars, tag) =
    x -> ntuple(v -> tag * 100 + v + sum(x[d] * 10.0^(d - 1) for d in eachindex(x)), nvars)

@testset "The resolved TreeAMR exports every name the scheme calls" begin
    # Guards an upstream rename. TreeAMR distinguishes physics packages,
    # which use the exports only and bound TreeAMR by a minor series, from
    # companion packages, which may use internals and pin exact releases
    # (TreeAMR's `CODE.md`, "Companion interface"). This is a physics
    # package, so the list is checked against `names`, i.e. the exports:
    # an unexported name here would be a promise TreeAMR does not make.
    for name in (:Forest, :refine!, :balance!, :nleaves, :spacing, :block_spacings,
                 :block_origins,
                 :FieldSet, :cellcentered, :vertexcentered, :facecentered, :edgecentered,
                 :staggers, :nblocks, :blockview, :interiorview, :coordinates,
                 :fill_by_coordinates!, :AllVariables,
                 :Operators, :PointValue, :Conservative, :GhostSchedule, :fill_ghosts!,
                 :CellBoundary, :boundary_by_coordinates,
                 :statelength, :statevector, :statearray, :scatter!, :gather!,
                 :staterange, :stateview,
                 :map_blocks!, :block_mapreduce, :mesh_mapreduce, :volume_weighted_norm,
                 :RegridFlag, :Refine, :Coarsen, :Keep, :regrid!, :adapt_to_initial_data!,
                 :total_mass, :firing_boxes, :InterfaceSchedule, :restrict_interfaces!)
        @test name in names(TreeAMR)
    end
end

@testset "The resolved EntropyEOS exports the recovery this package calls" begin
    # Guards an upstream rename of the one primitive recovery ("Primitive
    # recovery" in `CODE.md`: `con2prim_safe` in hydrodynamic mode for every
    # σ, `con2prim` only as the ideal seed), of the policy that wraps it, and
    # of the synthetic table step 1 checks the units adapter on, and of the
    # `AbstractEOS` interface and the analytic EOSes (EOS-1 and EOS-2, landed
    # in EntropyEOS 1.0.0; their probes were retired 2026-10-14). `κ` stays
    # unexported and is called qualified (amended in step 0). (Amended in
    # step 1: the synthetic table was to stand in for the analytic EOSes
    # until EOS-2.)
    for name in (:AbstractEOS, :logρ_bounds, :yₑ_bounds,
                 :IdealGasEOS, :HybridEOS, :polytropic_entropy,
                 :EOSTableView, :EOSPoint, :SRange, :evaluate, :srange, :srange_extended,
                 :narrow, :prim2con, :Prim2ConOut, :con2prim, :Con2PrimIn,
                 :Con2PrimOptions, :Con2PrimOut, :con2prim_safe, :Con2PrimSafeOut,
                 :PolicyOptions, :PrimState, :default_policy, :C2PResult,
                 :FLAG_POL_ATMOSPHERE, :FLAG_POL_ANY,
                 :make_synthetic_table, :build_eos)
        @test name in names(EOS)
    end
    @test isdefined(EOS, :κ) && :κ ∉ names(EOS)
end

@testset "The resolved SpacetimeMetrics exports the backgrounds and the ADM split" begin
    # Guards an upstream rename of what the `Analytic` provider (step 2), the
    # Kerr-Schild tests (step 9) and the vertex-metric check against TreeGH
    # (GH-1) call. `dmetric`'s index order, `dg[a,b,c] = ∂_c g_ab`, is
    # converted once, in `src/metric.jl` (step 2).
    for name in (:AbstractMetric, :metric, :dmetric, :adm_decompose, :ExtrinsicCurvature,
                 :EinsteinTensor, :Minkowski, :KerrSchild, :boost)
        @test name in names(SpacetimeMetrics)
    end
end

@testset "The resolved IMEXRungeKutta exports the integrator and both SSP tableaus" begin
    # Guards an upstream rename of the integrator ("The IMEX package" in
    # `CODE.md`): SSP3(4,3,3) is the default tableau and SSP2(2,2,2) the
    # cheap one ("The partition").
    for name in (:IMEXProblem, :IMEXTableau, :IMEXSSP3433, :IMEXSSP222,
                 :init, :step!, :solve!, :solve)
        @test name in names(IRK)
    end
end

mutable struct PrereqIMEXCounts
    f_exp::Int
    solve_imp::Int
    stage::Int
    step::Int
end

@testset "The integrator calls the stage solve and both limiters as CODE.md assumes" begin
    # Guards the integrator contract the scheme is built on ("The IMEX
    # package" in `CODE.md`): one `solve_imp!(U, u★, γΔt, p, t)` per implicit
    # stage, with `U` holding a copy of `u★` on entry; the reset passed as
    # both the stage and the step limiter, each with the signature
    # `(u, integ, p, t)`; and "three right-hand sides per SSP3(4,3,3) step" —
    # the count the cost model of "The partition" rests on, which a tableau
    # change upstream would silently move. On `u′ = −a u − b u` with the
    # second term implicit, so that the stage solve is a division.
    a, b = 1.0, 10.0
    f_exp! = function (du, u, p, t)
        p.f_exp += 1
        du .= -a .* u
        return nothing
    end
    solve_imp! = function (U, u★, γΔt, p, t)
        p.solve_imp += 1
        U == u★ || error("U must hold a copy of u★ on entry")
        U .= u★ ./ (1 + γΔt * b)
        return nothing
    end
    stage! = (u, integ, p, t) -> (p.stage += 1; nothing)
    step! = (u, integ, p, t) -> (p.step += 1; nothing)
    nsteps = 40
    counts = PrereqIMEXCounts(0, 0, 0, 0)
    prob = IRK.IMEXProblem(f_exp!, solve_imp!, [1.0], (0.0, 1.0), counts)
    integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=1 / nsteps, stage_limiter=stage!,
                     step_limiter=step!)
    IRK.solve!(integ)
    @test counts.f_exp == 3 * nsteps
    @test counts.solve_imp == 4 * nsteps
    @test counts.step == nsteps
    # The stage limiter runs before each explicit evaluation except at a
    # trivial stage, where `f_exp!` reads the state itself. SSP3(4,3,3) has
    # none: its first stage is implicit, so every explicit evaluation is
    # preceded by a reset (measured in step 0).
    @test counts.stage == 3 * nsteps
    # And the step is a step: the result is the exact decay `exp(−11)` to
    # the accuracy of 40 steps at `bΔt = 1/4` — 7.3e-4 relative, and 9.3e-5
    # at 80 steps, a ratio of 7.8 (measured in step 0) — asserted at twice
    # the measured error.
    @test isapprox(integ.u[1], exp(-(a + b)); rtol=1.5e-3)
end

@kernel function prereq_metric_kernel!(out, xs, m, ::Val{T}) where {T}
    I = @index(Global, NTuple)                 # stored indices: `stored = true`
    i, j, k, blk = I
    p = SVector{4,T}(zero(T), xs[i, j, k, 1, blk], xs[i, j, k, 2, blk], xs[i, j, k, 3, blk])
    α, β, γ = SpacetimeMetrics.adm_decompose(m, p)
    _, dg = SpacetimeMetrics.dmetric(m, p)
    K = SpacetimeMetrics.ExtrinsicCurvature(m, p)
    out[i, j, k, 1, blk] = α
    out[i, j, k, 2, blk] = β[1]
    out[i, j, k, 3, blk] = γ[1, 2]
    out[i, j, k, 4, blk] = dg[1, 1, 2]             # ∂_x g_tt, through a dual number
    out[i, j, k, 5, blk] = K[1, 3]
end

function prereq_metric_host(m, x::SVector{4,T}) where {T}
    α, β, γ = SpacetimeMetrics.adm_decompose(m, x)
    _, dg = SpacetimeMetrics.dmetric(m, x)
    K = SpacetimeMetrics.ExtrinsicCurvature(m, x)
    return (α, β[1], γ[1, 2], dg[1, 1, 2], K[1, 3])
end

@testset "A SpacetimeMetrics metric runs as a kernel argument: T=$T" for T in
                                                                          (Float64, Float32)
    # Guards the premise of the `Analytic` provider ("Metric providers" in
    # `CODE.md`): a metric is an `isbits` value handed to a kernel, and
    # inside the kernel it gives the ADM split, the extrinsic curvature and a
    # derivative through ForwardDiff's dual numbers — exactly what it gives
    # on the host, bit for bit. A metric that captured a heap object, or a
    # derivative that fell back to a non-inlined generic path, fails here
    # before step 2 builds on it. `Float32` is the leak detector: a stray
    # `Float64` constant in the metric would widen the result. A spinning
    # Kerr-Schild hole, sampled away from its ring singularity.
    m = SpacetimeMetrics.KerrSchild{T}(1, T(1 // 2))
    @test isbits(m)
    forest = prereq_forest(Val(3); N=4, extents=ntuple(_ -> (1.0, 3.0), 3))
    xs = FieldSet{T}(forest, 3; G=(0, 0, 0), centering=vertexcentered(3))
    out = FieldSet{T}(forest, 5; G=(0, 0, 0), centering=vertexcentered(3))
    extent = ntuple(d -> size(xs.work, d), 3)
    for b in 1:nblocks(xs), I in CartesianIndices(extent)
        x = coordinates(xs, b, Tuple(I))
        for d in 1:3
            xs.work[Tuple(I)..., d, b] = x[d]
        end
    end
    map_blocks!(prereq_metric_kernel!, out, out.work, xs.work, m, Val(T); stored=true)
    @test eltype(out.work) === T
    wrong = 0
    for b in 1:nblocks(xs), I in CartesianIndices(extent)
        x = SVector{4,T}(zero(T), (xs.work[Tuple(I)..., d, b] for d in 1:3)...)
        host = prereq_metric_host(m, x)
        host isa NTuple{5,T} || (wrong += 1)
        all(v -> isequal(out.work[Tuple(I)..., v, b], host[v]), 1:5) || (wrong += 1)
    end
    @test wrong == 0
    @test all(isfinite, out.work)
end

@testset "A field set is copied to the host without changing its layout" begin
    # Guards `hostcopy`'s two halves. On the CPU it returns the field set
    # itself, so its copying path would otherwise be dead code on every
    # machine CI runs on; `hostcopy!` is that path, called here host to host.
    # The layout is the thing to get wrong, and this package mixes all of
    # TreeAMR's centerings: a destination built without `G` and the
    # centering has a differently shaped working array, and a `copyto!`
    # between two arrays of equal length and unequal shape transposes the
    # data instead of failing.
    forest = prereq_forest(Val(3))
    layouts = ((cellcentered(3), (3, 3, 3), 10),        # matter
               (edgecentered(3, 2), (3, 3, 3), 1),      # A_y
               (facecentered(3, 1), (3, 3, 3), 1),      # B_x
               (facecentered(3, 3), (0, 0, 0), 10),     # F_z
               (vertexcentered(3), (0, 0, 0), 10))      # Tsrc
    for (centering, G, nvars) in layouts
        src = FieldSet{Float64}(forest, nvars; G=G, centering=centering)
        fill_by_coordinates!(AllVariables(prereq_values(nvars, 1)), src)
        @test hostcopy(src) === src

        dst = FieldSet{Float64}(src.forest, src.nvars; G=src.G, centering=src.centering)
        @test TreeGRRMHD.hostcopy!(dst, src) === dst
        @test dst.work !== src.work
        @test isequal(dst.work, src.work)
        @test any(!iszero, dst.work)
    end

    # A destination whose ghost width or centering was left at something
    # else is the mistake, and it is an error rather than a transposition.
    src = FieldSet{Float64}(forest, 10; G=3, centering=cellcentered(3))
    thin = FieldSet{Float64}(forest, 10; G=2, centering=cellcentered(3))
    @test_throws "same layout" TreeGRRMHD.hostcopy!(thin, src)
    vert = FieldSet{Float64}(forest, 10; G=3, centering=vertexcentered(3))
    @test_throws "same layout" TreeGRRMHD.hostcopy!(vert, src)
    other = FieldSet{Float64}(prereq_forest(Val(3)), 10; G=3, centering=cellcentered(3))
    @test_throws "same forest" TreeGRRMHD.hostcopy!(other, src)

    # And per-block metadata stays where it is on the host.
    a = rand(3, 7)
    @test TreeGRRMHD.to_backend(CPU(), a) === a
end

# ---------------------------------------------------------------------------
# The upstream items still waiting, one testset per ID. "waiting" says so in
# the log and names the step it blocks; "landed" says the probe is due to be
# retired, once `[compat]` requires the release that has it.

function prereq_report(id, landed::Bool, what)
    if landed
        @info "$id landed: $what"
    else
        @info "$id waiting: $what"
    end
    return landed
end

@testset "AMR-7: a device-capable boundary hook relative to the owned point (when landed)" begin
    # Guards the non-periodic cases on a device (step 14): until TreeAMR has
    # a per-cell hook that receives the nearest owned point, the potentials'
    # Dirichlet-in-the-field hook is TreeAMR's region form, CPU-only (added
    # in step 6).
    landed = isdefined(TreeAMR, :Relative)
    prereq_report("AMR-7", landed,
                  landed ? "`Relative` exists; the region-form hook can become a kernel" :
                  "no per-cell hook relative to the owned point; non-periodic cases " *
                  "run on the CPU only (wanted by step 14)")
end

@testset "AMR-8: symmetric walls, the owned low wall filled like the high one (when landed)" begin
    # Guards the mirror symmetry of a closed box (added in step 8; decided by
    # Erik 2026-10-15): until TreeAMR fills the owned low wall of a vertex-like
    # dimension by the rule of the high one, this package prescribes it itself
    # (`low_walls!`, `outer_points!` in `src/driver.jl`, marked AMR-8).
    # Detected by the keyword, never by a version.
    landed = hasmethod(FieldSet{Float64}, Tuple{Forest,Int}, (:walls,))
    prereq_report("AMR-8", landed,
                  landed ? "`FieldSet(…; walls = :symmetric)` exists; `low_walls!` can go" :
                  "no symmetric walls; the potentials' low walls are prescribed by " *
                  "`low_walls!` (wanted, no step blocked)")
end

@testset "EOS-0: EntropyEOS is registered (reported, never required)" begin
    # Guards nothing the suite needs — `[sources]` finds EntropyEOS either
    # way — but says whether the URL workaround can go. Read from the
    # registries this depot reaches, which may be stale, so this is a
    # report and not an assertion.
    uuid = Base.PkgId(EOS).uuid
    landed = any(reg -> haskey(reg.pkgs, uuid), Pkg.Registry.reachable_registries())
    prereq_report("EOS-0", landed,
                  landed ? "EntropyEOS is in a reachable registry; the `[sources]` " *
                  "entry can go" :
                  "EntropyEOS is in no reachable registry; `[sources]` adds it by URL")
end

@testset "EOS-3: the fixed-entropy inversion is public (when landed)" begin
    # Guards the entropy fallback of step 12 and the entropy-from-energy
    # solve for initial data.
    landed = isdefined(EOS, :con2prim_entropy) && isdefined(EOS, :entropy_from_energy)
    prereq_report("EOS-3", landed,
                  landed ? "`con2prim_entropy`, `entropy_from_energy`" :
                  "no public fixed-entropy inversion; wanted by step 12")
end

@testset "EOS-4: the cold β-equilibrium slice exists (when landed)" begin
    # Guards the tabulated-EOS star's initial data (step 13).
    landed = isdefined(EOS, :beta_equilibrium) && isdefined(EOS, :cold_slice)
    prereq_report("EOS-4", landed,
                  landed ? "`beta_equilibrium`, `cold_slice`" :
                  "no β-equilibrium slice; blocks step 13")
end

@testset "GH-$n: reported once TreeGeneralizedHarmonic is a dependency" for n in 0:6
    # Guards nothing yet. TreeGH is not in this package's environment until
    # step 10 (it is unregistered, and its own unregistered dependencies
    # would each need a `[sources]` entry here — GH-2), so its items cannot
    # be detected from this suite. The step that adds TreeGH replaces this
    # loop with one detecting testset per item (proposed in step 0).
    ghloaded = Base.find_package("TreeGeneralizedHarmonic") !== nothing
    prereq_report("GH-$n", false,
                  ghloaded ? "TreeGH is loadable but not yet checked here" :
                  "TreeGH is not a dependency before step 10, so not detectable")
end
