# Step 17: the neutrino optical depth on the forest (rung 8b; "Neutrinos",
# "The optical depth" in `CODE.md`). `P` is built directly from analytic data
# — a flat metric and a density profile — so that the optical depth is tested
# without running the scheme; `GrayRates` turn the density into opacities.

const OD = TreeGRRMHD
const OD_MODEL = GrayRates{Float64}(; κa=(1.0, 0.5, 0.0), κs=(0.0, 0.0, 2.0),
                                    η=(0.0, 0.0, 0.0), E=(0.0, 0.0, 0.0))

# A primitive set on `forest` holding `ρ★ = ρf(x)` and the flat metric in every
# stored cell, ghosts included, as `update_primitives!` leaves them.
function od_primitives!(P, ρf)
    D = length(P.G)
    fill!(P.work, 0)
    for b in 1:size(P.work)[end], idx in CartesianIndices(size(P.work)[1:D])
        s = Tuple(idx)
        P.work[s..., OD.P_RHO, b] = ρf(coordinates(P, b, s))
        P.work[s..., OD.P_GAMMA, b] = 1
        P.work[s..., OD.P_GAMMA + 3, b] = 1
        P.work[s..., OD.P_GAMMA + 5, b] = 1
        P.work[s..., OD.P_SQRTG, b] = 1
    end
    return P
end

od_forest(::Val{D}, roots) where {D} =
    Forest(ntuple(_ -> roots, D); N=8, periodic=ntuple(_ -> false, D),
           extents=ntuple(_ -> (-1.0, 1.0), D))

function od_setup(forest, ρf)
    D = length(forest.periodic)
    P = od_primitives!(FieldSet{Float64}(forest, OD.NP; G=3, centering=cellcentered(D)), ρf)
    od = OpticalDepth(forest)
    opacities!(od, OD_MODEL, P)
    return od, P
end

# A cell centre as a dictionary key, rounded so that two meshes' centres of
# the same point compare equal.
od_key(x) = round.(x; digits=10)

# τ of slot `k` at every owned cell, keyed by the cell centre.
function od_values(od, k)
    D = length(od.Nτ.G)
    v = Dict{NTuple{D,Float64},Float64}()
    for b in 1:size(od.Nτ.work)[end], idx in CartesianIndices(ntuple(_ -> 8, D))
        s = Tuple(idx) .+ 1
        v[od_key(coordinates(od.Nτ, b, s))] = od.Nτ.work[s..., k, b]
    end
    return v
end

# Bellman–Ford on a uniform array of opacities at the cell centres of a box
# whose outside holds τ = κ = 0: the reference, over the 3^D − 1 neighbours or
# over the 2D across faces only. The same arithmetic as the kernel's.
function od_host_depth(κ::Array{Float64,D}, h; faces=false) where {D}
    offs = [o for o in Iterators.product(ntuple(_ -> -1:1, D)...)
            if any(!=(0), o) && (!faces || count(!=(0), o) == 1)]
    τ = zeros(size(κ))
    for sweeps in 1:10_000
        new = similar(τ)
        for I in CartesianIndices(κ)
            best = Inf
            for o in offs
                J = Tuple(I) .+ o
                inside = all(1 .≤ J .≤ size(κ))
                τn, κn = inside ? (τ[J...], κ[J...]) : (0.0, 0.0)
                o3 = ntuple(e -> e ≤ D ? o[e] : 0, 3)
                Δ = o3 .* h
                ℓ = sqrt(Δ[1] * Δ[1] + Δ[2] * Δ[2] + Δ[3] * Δ[3])
                best = min(best, τn + (κ[I] + κn) * ℓ / 2)
            end
            new[I] = best
        end
        changed = new != τ
        τ = new
        changed || return τ, sweeps
    end
    error("od_host_depth did not converge")
end

@testset "The sweep's fixed point is Bellman–Ford's over the 3^D − 1 neighbours: D=$D" for D in
                                                                                         (2, 3)
    # Guards the kernel's stencil, distances, half-weights, outer ghosts and
    # Jacobi update against the plain reference, bit for bit, on a uniform
    # mesh across many blocks, and its sweep count: from τ = 0 one sweep per
    # cell of the longest optical path, plus the one that changes nothing.
    roots = D == 2 ? 4 : 2
    n = roots * 8
    h = 2 / n
    ρf(x) = 1 + x[1] * x[1] + (sum(x) < 0.3 ? 2.0 : 0.0)
    od, P = od_setup(od_forest(Val(D), roots), ρf)
    sweeps = converge!(od, P)
    xs = [-1 + (i - 0.5) * h for i in 1:n]
    cells = CartesianIndices(ntuple(_ -> n, D))
    vals = od_values(od, 1)
    for k in (1, 3, 5)                  # κ = 1, 0.5, 2 times ρ★
        i = (k + 1) ÷ 2
        κ = [OD_MODEL.κa[i] * ρ + OD_MODEL.κs[i] * ρ
             for ρ in (ρf(ntuple(e -> xs[I[e]], D)) for I in cells)]
        κ = reshape(κ, ntuple(_ -> n, D))
        ref, host_sweeps = od_host_depth(κ, h)
        got = od_values(od, k)
        @test all(I -> got[od_key(ntuple(e -> xs[I[e]], D))] == ref[I], cells)
        @test sweeps == host_sweeps          # both count the sweep that changed nothing
    end
    @info "Optical depth from zero" D cells = n sweeps
    # The weightings agree for GrayRates (κ⁰ = κ¹).
    @test od_values(od, 2) == vals
end

@testset "On an oblique boundary the 26-point stencil is within 8% of the distance, faces alone √3" begin
    # Guards the stencil's reach: face neighbours alone measure the distance
    # to a plane whose normal is (1, 1, 1) in the L1 norm, √3 too far; with
    # edges and corners the overestimate is the chamfer metric's, a few per
    # cent at most, and the staircase of the boundary takes up to 8% off.
    # Measured in step 17 on 32³ (τ / κ d, at depths 4h … 0.35): (1,0,0) 1.000
    # and 1.000, (1,1,0) 0.916 and 1.313, (1,1,1) 0.972 and 1.732, (2,1,0) 0.990
    # and 1.171, (3,2,1) 0.976 and 1.372 (26 points, faces).
    n, h = 32, 2 / 32
    xs = [-1 + (i - 0.5) * h for i in 1:n]
    for normal in ((1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (3, 2, 1))
        nv = normal ./ sqrt(sum(abs2, normal))
        κ = [sum(nv .* (x, y, z)) < 0 ? 1.0 : 0.0 for x in xs, y in xs, z in xs]
        ratio(τ) = maximum(CartesianIndices(κ); init=0.0) do I
            x = (xs[I[1]], xs[I[2]], xs[I[3]])
            d = -sum(nv .* x)
            4h ≤ d ≤ 0.35 && maximum(abs, x) < 0.5 ? τ[I] / d : 0.0
        end
        r26 = ratio(first(od_host_depth(κ, h)))
        rf = ratio(first(od_host_depth(κ, h; faces=true)))
        @test 0.9 ≤ r26 ≤ 1.01
        normal == (1, 1, 1) && @test rf ≈ sqrt(3) rtol = 1e-3
        @test rf ≥ r26
    end
end

@testset "A uniform sphere: τ within κh of κ (R − r) inside" begin
    # Guards the optical depth's scale against the analytic one: inside a
    # uniform sphere τ = κ (R − r). The staircase surface puts the discrete
    # boundary up to a cell's diagonal off, so the error is measured in κh.
    # Measured in step 17: from −0.65 to +0.30 κh on 64², from −0.76 to
    # +0.07 κh on 32³.
    for (D, roots) in ((2, 8), (3, 4))
        h = 2 / (roots * 8)
        R = 0.75
        od, P = od_setup(od_forest(Val(D), roots), x -> sqrt(sum(abs2, x)) < R ? 1.0 : 0.0)
        converge!(od, P)
        lo, hi = 0.0, 0.0
        for (x, τ) in od_values(od, 1)
            r = sqrt(sum(abs2, x))
            r < R - 2h || continue
            err = (τ - (R - r)) / h          # κ = 1
            lo, hi = min(lo, err), max(hi, err)
        end
        @info "Uniform sphere, error in κh" D lo hi
        @test -0.8 ≤ lo && hi ≤ 0.35
    end
end

@testset "One sweep a step follows a moving sphere" begin
    # Guards the design's one sweep per step: τ travels one cell per sweep,
    # 2.5 times the speed of light at cfl = 2/5, so it shows the matter as it
    # was about L h / 2.5 earlier, L the optical path in cells. For a sphere
    # of 13 cells' radius moving at v = 0.1 that is half a cell, about 0.5 κh
    # in τ; measured in step 17, 0.69 κh after 150 steps, without converging
    # again.
    D, roots = 2, 8
    h = 2 / (roots * 8)
    R, v, dt = 0.4, 0.1, 2h / 5
    forest = od_forest(Val(D), roots)
    sphere(t) = x -> sqrt((x[1] + 0.3 - v * t)^2 + x[2]^2) < R ? 1.0 : 0.0
    od, P = od_setup(forest, sphere(0.0))
    converge!(od, P)
    t = 0.0
    for _ in 1:150
        t += dt
        od_primitives!(P, sphere(t))
        opacities!(od, OD_MODEL, P)
        sweep!(od, P)
    end
    lagged = od_values(od, 1)
    ref, Pref = od_setup(forest, sphere(t))
    converge!(ref, Pref)
    exact = od_values(ref, 1)
    worst = maximum(abs(lagged[x] - exact[x]) for x in keys(exact))
    @info "Moving sphere, one sweep a step" moved = v * t worst_in_κh = worst / h
    @test worst / h ≤ 1.0
end

@testset "τ across coarse-fine faces and through a regrid matches the uniform mesh" begin
    # Guards the ghost fills between levels and τ's transfer at `regrid!`: a
    # sphere across a coarse-fine face converges to within the coarse mesh's
    # own resolution of the uniform fine mesh's τ (each coarse cell against
    # the mean of its fine cells; measured in step 17, κ h_fine exactly, one
    # fine step of the staircase), and a coarse mesh refined everywhere by
    # `regrid!` carries τ to within the coarse resolution of the fine mesh's
    # converged τ (measured in step 17: 1.29 κ h_fine), and reconverges to it
    # exactly. Reconverging takes as many
    # sweeps as from zero (20 here, measured in step 17): a Bellman–Ford fixed
    # point is reached only once the boundary's information has crossed the
    # whole path, whatever the start; so after a regrid the sweeps go on one
    # a step from the transferred τ.
    D = 2
    R = 0.6
    ρf(x) = sqrt((x[1] - 0.05)^2 + (x[2] - 0.05)^2) < R ? 1.0 : 0.0
    fine = od_forest(Val(D), 4)                     # uniform, h = 1/16
    refine!(fine, collect(fine.leaves)); balance!(fine)
    odf, Pf = od_setup(fine, ρf)
    converge!(odf, Pf)
    uniform = od_values(odf, 1)
    hf = 2 / 64

    mixed = od_forest(Val(D), 4)                    # h = 1/16 and 1/32
    refine!(mixed, [first(mixed.leaves)]); balance!(mixed)
    odm, Pm = od_setup(mixed, ρf)
    converge!(odm, Pm)
    worst = 0.0
    for (x, τ) in od_values(odm, 1)
        if haskey(uniform, x)
            ref = uniform[x]
        else                                         # a coarse cell: its four fine cells
            ref = sum(uniform[od_key(x .+ (sx * hf / 2, sy * hf / 2))]
                      for sx in (-1, 1), sy in (-1, 1)) / 4
        end
        worst = max(worst, abs(τ - ref))
    end
    @info "Coarse-fine faces against the uniform mesh, in κ h_fine" worst / hf
    @test worst / hf ≤ 1.0 + 1e-12

    coarse = od_forest(Val(D), 4)
    odc, Pc = od_setup(coarse, ρf)
    from_zero = converge!(odc, Pc)
    regrid!(coarse, (depth_regrid_pairs(odc)..., Pc => nothing);
            flags=fill(Refine, nleaves(coarse)))
    odc = OD.rebuild(odc)
    od_primitives!(Pc, ρf)
    opacities!(odc, OD_MODEL, Pc)
    off(o) = (v = od_values(o, 1); maximum(abs(v[x] - uniform[x]) for x in keys(uniform)))
    transferred = off(odc)
    sweep!(odc, Pc)
    one_sweep = off(odc)
    after = converge!(odc, Pc)
    fresh = converge!(od_setup(fine, ρf)...)
    @info "After the regrid, in κ h_fine" transferred / hf one_sweep / hf from_zero after fresh
    @test transferred / hf ≤ 1.5      # measured in step 17: 1.29
    @test one_sweep ≤ transferred
    @test off(odc) ≤ 1e-12
    @test after ≤ fresh
end

@testset "OpticalDepth refuses a box with no outer face" begin
    # Guards against a sweep that never converges: on a fully periodic forest
    # τ has no boundary to be measured to.
    forest = Forest((2, 2); N=8, periodic=(true, true), extents=((0.0, 1.0), (0.0, 1.0)))
    @test_throws ArgumentError OpticalDepth(forest)
    @test isbits(OpacityEscape()) && OpacityEscape()((0.5, 0.5), 0) == ntuple(_ -> 0.0, 12)
end
