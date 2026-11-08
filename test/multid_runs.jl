# The runs and comparisons of the 2D and 3D tests: `md_run` (a run with `∇·𝔅`
# asserted after every step), the owned points as one global array, the 1D,
# turned and mirrored comparisons, and the conservation of the totals. Used by
# `multid_tests.jl`, `multid_cases_tests.jl`, `refinement_tests.jl` and
# `refinement_cases_tests.jl`. A helper, not a test file (moved out of
# `multid_tests.jl` on 2026-11-08): `runtests.jl` includes it before them.

using StaticArrays: SVector

const MD = TreeGRRMHD
const MD_DIVB = 1e-12

# One run with `∇·𝔅` measured after every step, asserted, and returned.
function md_run(case, ::Val{D}; N, t_end, limiter, kw...) where {D}
    divs = Float64[]
    monitor(p, u, t) = push!(divs, max_div_B(p, u, t))
    r = evolve!(case, Val(D); N=N, t_end=t_end, limiter=limiter, after_step=monitor, kw...)
    @test length(divs) == r.nsteps
    @test maximum(divs) ≤ MD_DIVB
    u0 = exact_state(r.p, case, 0.0)
    # The initial totals on the initial mesh, which differ from those of the
    # initial data on a regridded run's final mesh (step 8).
    return (; r, case, divb=maximum(divs), u0,
            drift=conserved_totals(r.p, r.u) .- r.totals0,
            scales=max.(conserved_scales(r.p, u0), conserved_scales(r.p, r.u)),
            tally=(; bits=sum(c -> collect(c.tally.bits), r.chunks),
                   policy=sum(c -> c.tally.policy, r.chunks),
                   resets=sum(c -> c.tally.resets + c.tally.step_resets, r.chunks)))
end

# The owned points of evolved set `i` of a state as one array over the whole
# uniform mesh, `(M₁, …, M_D, nvars)`, each point at its global index
# (staggered sets own `N` points a block along every dimension, as cells do).
function md_global(p, u, i)
    s = p.sets[i]
    forest = s.forest
    D = length(forest.roots)
    N = forest.N
    a = statearray(u, p.sets, i)
    h = block_spacings(forest)
    o = block_origins(forest)
    out = zeros(forest.roots .* N..., s.nvars)
    for b in 1:nblocks(s)
        g0 = ntuple(d -> round(Int, (o[b][d] - forest.extents[d][1]) / h[b]), D)
        for I in CartesianIndices(ntuple(_ -> N, D)), v in 1:(s.nvars)
            out[(Tuple(I) .+ g0)..., v] = a[I, v, b]
        end
    end
    return out
end

# Bit for bit: every evolved set of the D-dimensional run is the 1D run's,
# repeated along the dimensions the solution does not vary in.
function md_same_as_1d(r1, rD)
    D = length(rD.p.f.U.forest.roots)
    return all(1:5) do i
        a1 = md_global(r1.p, r1.u, i)
        aD = md_global(rD.p, rD.u, i)
        all(CartesianIndices(size(aD))) do I
            isequal(aD[I], a1[I[1], I[D + 1]])
        end
    end
end

# Bit for bit: the turned run is the run with `x` and `y` exchanged and the
# `z` components negated — `S_i`, `𝔈ⁱ` and `A_i` turned by
# `R(x, y, z) = (y, x, −z)`, `A_x` the turned `A_y`. Compared with `==`, which
# is bit identity for every nonzero value and leaves only the sign of a zero
# free: a `z` component that is exactly zero is `+0` in both runs, where the
# turn predicts `−0` (`0 − 0 = +0` whichever way the operands came; measured in
# step 7 on Balsara 1, whose `S_z` and `𝔈ᶻ` vanish identically).
function md_turned_equal(rx, ry)
    turnU(q, v) = v in (2, 8) ? q[v + 1] : v in (3, 9) ? q[v - 1] : v in (4, 10) ? -q[v] : q[v]
    seg = (1, 3, 2, 4, 5)                 # set i of the turned run is set seg[i] of the run
    sgn = (1, 1, 1, -1, 1)
    return all(1:5) do i
        ay = md_global(ry.p, ry.u, i)
        ax = md_global(rx.p, rx.u, seg[i])
        all(CartesianIndices(size(ay)[1:2])) do I
            J = (I[2], I[1])
            all(1:size(ay, 3)) do v
                x = i == 1 ? turnU(view(ax, J..., :), v) : sgn[i] * ax[J..., v]
                ay[I, v] == x
            end
        end
    end
end

# The largest violation of a symmetry, relative to each variable's largest
# magnitude: reflections along `axes` (a box symmetric about its centre),
# with every field reversed when `c = −1` (`E`, `B`, `A`, `Φ̃` → their
# negatives, a symmetry of the equations). Under a reflection along `k` a
# scalar is even, a polar vector's `k` component odd and the others even;
# `B` is never state, so only `S_i`, `𝔈ⁱ`, `A_i` and `Φ̃` carry parities. A
# cell `g` mirrors to `M + 1 − g`, a vertex-like point to `M + 2 − g` (the
# box's high boundary vertex is no block's, so the first vertex has no mirror
# on a closed box; on a periodic one it is its own).
function md_mirror_error(p, u, axes, c)
    pol(j) = prod(k -> k == j ? -1 : 1, axes; init=1)
    function parity(i, v)
        i == 1 && v in 2:4 && return pol(v - 1)
        i == 1 && v in 8:10 && return c * pol(v - 7)
        i == 1 && return 1
        return i ≤ 4 ? c * pol(i - 1) : c
    end
    worst = 0.0
    for i in 1:5
        q = md_global(p, u, i)
        st = staggers(p.sets[i])
        D = length(st)
        M = size(q)[1:D]
        per = p.sets[i].forest.periodic
        for v in 1:size(q, D + 1)
            scale = maximum(abs, selectdim(q, D + 1, v))
            scale == 0 && continue
            for I in CartesianIndices(M)
                g = Tuple(I)
                gm = ntuple(D) do d
                    !(d in axes) ? g[d] : st[d] == 0 ? M[d] + 1 - g[d] :
                    per[d] ? mod1(M[d] + 2 - g[d], M[d]) : M[d] + 2 - g[d]
                end
                any(d -> gm[d] > M[d], 1:D) && continue
                worst = max(worst, abs(q[g..., v] - parity(i, v) * q[gm..., v]) / scale)
            end
        end
    end
    return worst
end

# The evolved set `i` over the whole uniform mesh from its ghost-filled working
# array, on the closed range: every owned point and, along a vertex-like
# dimension, the box's high boundary plane too, which no block owns and the
# hook fills (`(M₁ + c₁, …, nvars)`). Where two blocks share a plane, the
# later one's copy is the one kept; after a fill the two are equal.
function md_closed_global(p, i)
    s = p.sets[i]
    forest = s.forest
    D = length(forest.roots)
    N = forest.N
    st = staggers(s)
    h = block_spacings(forest)
    o = block_origins(forest)
    out = fill(NaN, (forest.roots .* N .+ st)..., s.nvars)
    for b in 1:nblocks(s)
        g0 = ntuple(d -> round(Int, (o[b][d] - forest.extents[d][1]) / h[b]), D)
        for I in CartesianIndices(ntuple(d -> N + st[d], D)), v in 1:(s.nvars)
            out[(Tuple(I) .+ g0)..., v] = s.work[(Tuple(I) .+ s.G)..., v, b]
        end
    end
    return out
end

# Bit for bit, the mirror symmetry of a closed box (added in step 8, with
# symmetric walls, AMR-8): every evolved set of the state `u` at `t`, scattered
# and ghost-filled, on the closed range — both boundary planes of a vertex-like
# dimension, the low one owned and prescribed, the high one the hook's — is
# its own reflection along `axes`, with the parities of `md_mirror_error`. A
# cell `g` mirrors to `M + 1 − g`, a vertex-like point to `M + 2 − g`, and now
# every point has its mirror. Compared with `==`: bit identity for every
# nonzero value, and the sign of a zero free, as for `md_turned_equal` (a
# parity of −1 turns `+0` into `−0`, and `0 − 0` is `+0` whichever side it was
# computed on). Also returns whether the state itself holds what the fill
# holds on the owned points — that the prescribed low walls persist in it.
function md_mirror_equal(p, u, t, axes, c)
    update_primitives!(p, u, t)
    pol(j) = prod(k -> k == j ? -1 : 1, axes; init=1)
    function parity(i, v)
        i == 1 && v in 2:4 && return pol(v - 1)
        i == 1 && v in 8:10 && return c * pol(v - 7)
        i == 1 && return 1
        return i ≤ 4 ? c * pol(i - 1) : c
    end
    mirrored = all(1:5) do i
        q = md_closed_global(p, i)
        D = ndims(q) - 1
        M = size(q)[1:D]
        all(Iterators.product(CartesianIndices(M), 1:size(q, D + 1))) do (I, v)
            g = Tuple(I)
            gm = ntuple(d -> d in axes ? M[d] + 1 - g[d] : g[d], D)
            q[g..., v] == parity(i, v) * q[gm..., v]
        end
    end
    persisted = all(1:5) do i
        q = md_closed_global(p, i)
        a = md_global(p, u, i)
        all(I -> isequal(a[I], q[I]), CartesianIndices(a))
    end
    return (; mirrored, persisted)
end

# Every total but `𝔈`'s constant to rounding: TreeHydro's 8 ulp of its scale per
# step, as `wave_tests.jl` asserts.
function md_conserved(run)
    worst = 0.0
    for v in (MD.U_D, MD.U_S, MD.U_S + 1, MD.U_S + 2, MD.U_TAU, MD.U_DYE, MD.U_DS)
        run.scales[v] > 0 || continue
        ulps = abs(run.drift[v]) / (eps() * run.scales[v] * run.r.nsteps)
        @test ulps ≤ 8
        worst = max(worst, ulps)
    end
    return worst
end
