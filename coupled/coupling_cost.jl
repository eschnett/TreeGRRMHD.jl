# Step 11: what the coupling costs, for two decisions ("Two scripts", "Matter to the
# vertices" in `CODE.md`): whether the script should skip TreeGH's repeated fill (written
# plainly, an SSP3(4,3,3) step fills TreeGH's set and `Mv` eleven times; `FillKey` in
# `tov_coupled.jl` skips the repeats),
# and whether `ADMMatter` should store its cell values rather than compute them per call.
# On the coupled star (`problems`, default `N = 16`, one box level), each piece timed
# alone after a warm-up, the median of `reps`:
#
# - one SSP3(4,3,3) step of the script, its four callbacks as written, with the fill's key
#   and with every fill done;
# - one fill (`fill_metric`): TreeGH's `prepare!` (scatter and fill) and `vertex_metric!` (the owned
#   copy and `Mv`'s fill), and TreeGH's `prepare!` alone;
# - TreeGH's `rhs!` without the matter, with `ADMMatter` (per call), and with the stored
#   form (the cells' 10 values filled first by a host loop over the stored cells, then a
#   mean of eight reads per vertex), the fill timed separately;
# - this package's `prepare!` and `rhs!`.
#
#     julia --project=coupled -t T coupled/coupling_cost.jl [N=16] [levels=1] [reps=5]

include(joinpath(@__DIR__, "tov_coupled.jl"))
using Printf
using StaticArrays: SVector, SMatrix
const Adapt = MHD.Adapt                              # as the package loads it

# The stored form: `(𝓔, S_i, S_ij)` per cell in a set of its own, filled from `P`.
struct StoredMatterAt{W,G}
    w::W
    valG::Val{G}
end
Adapt.adapt_structure(to, a::StoredMatterAt) = StoredMatterAt(Adapt.adapt(to, a.w), a.valG)
@inline function (a::StoredMatterAt{W,G})(n::NTuple{3,Int}, b) where {W,G}
    T = eltype(a.w)
    s = (n[1] + G[1], n[2] + G[2], n[3] + G[3])
    acc = zero(SVector{10,T})
    for k in -1:0, j in -1:0, i in -1:0
        c = (s[1] + i, s[2] + j, s[3] + k)
        acc += SVector{10,T}(ntuple(v -> @inbounds(a.w[c..., v, b]), Val(10)))
    end
    w = T(1 // 8)
    return (; ρ=w * acc[1], S=w * SVector(acc[2], acc[3], acc[4]),
            Sij=w * SMatrix{3,3,T,9}(acc[5], acc[6], acc[7], acc[6], acc[8], acc[9],
                                     acc[7], acc[9], acc[10]))
end
struct StoredMatter{W,G}
    w::W
    valG::Val{G}
end
(m::StoredMatter)(U) = StoredMatterAt(m.w, m.valG)

function fill_stored!(w, P)
    Pw = P.work
    Threads.@threads for b in axes(Pw, 5)
        for k in axes(Pw, 3), j in axes(Pw, 2), i in axes(Pw, 1)
            Pc = MHD.read_vars(Pw, (i, j, k), b, Val(MHD.NP))
            e, S, Sl = MHD.cell_adm(Pc)
            vals = (e, S[1], S[2], S[3], Sl[1, 1], Sl[1, 2], Sl[1, 3], Sl[2, 2], Sl[2, 3],
                    Sl[3, 3])
            for v in 1:10
                @inbounds w[i, j, k, v, b] = vals[v]
            end
        end
    end
    return w
end

function timed(f, reps)
    f()
    ts = [(t0 = time_ns(); f(); (time_ns() - t0) / 1e9) for _ in 1:reps]
    return sort(ts)[(reps + 1) ÷ 2]
end

function cost(T=Float64; N=16, levels=1, reps=5)
    (; p, u, forest) = problems(T; N, levels, L=24, roots=4, limiter=:ppm)
    t = zero(T)
    du = similar(u)
    ug, um = view(u, p.s[1]), view(u, p.s[2])
    dug, dum = view(du, p.s[1]), view(du, p.s[2])
    nv = nblocks(p.gh.U) * N^3
    metric = metric_at(p, u, t)
    MHD.prepare!(p.mhd, um, t; metric)
    P = p.mhd.f.P
    w = similar(P.work, T, (size(P.work)[1:3]..., 10, size(P.work, 5)))
    stored = StoredMatter(w, Val(P.G))
    fill_stored!(w, P)
    # The stored form against the per-call one, at every owned vertex: the same numbers.
    U = p.gh.U
    a1, a2 = MHD.ADMMatter(p.mhd)(U), stored(U)
    diff = maximum(abs(a1((i, j, k), b).ρ - a2((i, j, k), b).ρ)
                   for b in 1:nblocks(U), k in 1:N, j in 1:N, i in 1:N)
    GH.prepare!(p.gh, ug, t)
    forget_fill!(p)
    cfl_mhd = T(MHD.default_cfl(:ppm))
    nsteps = step_count(p, u, t, one(T), forest, T(1 // 2), cfl_mhd)
    dt = one(T) / nsteps
    integ = IRK.init(IRK.IMEXProblem(f_exp!, solve_imp!, copy(u), (t, T(1000)), p),
                     IRK.IMEXSSP3433(); dt, stage_limiter=stage_limit!,
                     step_limiter=step_limit!, partition=:even, alias_u0=true)
    # The same step with every fill done (`key = nothing`), on the same state, and the
    # fills a step does with the key.
    pf = merge(p, (; key=nothing))
    integf = IRK.init(IRK.IMEXProblem(f_exp!, solve_imp!, copy(u), (t, T(1000)), pf),
                      IRK.IMEXSSP3433(); dt, stage_limiter=stage_limit!,
                      step_limiter=step_limit!, partition=:even, alias_u0=true)
    c0 = p.key.count[]
    MHD.with_step_cache(() -> IRK.step!(integ), p.mhd)
    fills = p.key.count[] - c0
    r = (;
         step=timed(() -> MHD.with_step_cache(() -> IRK.step!(integ), p.mhd), reps),
         step_every_fill=timed(() -> MHD.with_step_cache(() -> IRK.step!(integf), pf.mhd),
                               reps),
         metric_at=timed(() -> fill_metric(p, u, t), reps),
         gh_prepare=timed(() -> GH.prepare!(p.gh, ug, t), reps),
         gh_rhs_vacuum=timed(() -> (GH.prepare!(p.gh, ug, t);
                                    GH.rhs!(dug, p.gh, t)), reps) -
                       timed(() -> GH.prepare!(p.gh, ug, t), reps),
         gh_rhs_admmatter=timed(() -> GH.rhs!(dug, p.gh, t; matter=MHD.ADMMatter(p.mhd)),
                                reps),
         gh_rhs_stored=timed(() -> GH.rhs!(dug, p.gh, t; matter=stored), reps),
         stored_fill=timed(() -> fill_stored!(w, P), reps),
         mhd_prepare=timed(() -> MHD.prepare!(p.mhd, um, t; metric), reps),
         mhd_rhs=timed(() -> MHD.rhs!(dum, p.mhd, t; metric), reps))
    println("coupled star, N = $N, levels = $levels: $(nblocks(U)) blocks, $nv owned " *
            "vertices, $(Threads.nthreads()) threads; stored − per call, ρ: $diff")
    for (k, v) in pairs(r)
        @printf("  %-18s %10.4f s  %8.2f ns per vertex\n", k, v, 1e9 * v / nv)
    end
    @printf("  fills per step with the key: %d of 11; the step %.1f%% shorter\n", fills,
            100 * (1 - r.step / r.step_every_fill))
    seven = 7 * r.metric_at
    @printf("  seven extra fills (TreeGH's and Mv's) per step: %.4f s, %.1f%% of a step\n",
            seven, 100 * seven / r.step_every_fill)
    return r
end

if abspath(PROGRAM_FILE) == @__FILE__
    kw = Dict{Symbol,Any}()
    for a in ARGS
        k, v = split(a, '='; limit=2)
        k in ("N", "levels", "reps") || error("coupling_cost.jl: unknown key `$k`")
        kw[Symbol(k)] = parse(Int, v)
    end
    cost(; kw...)
end
