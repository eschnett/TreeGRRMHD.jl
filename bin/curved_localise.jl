# Step 9, round 2: where the stationarity error of the Kerr-Schild cases lives, and when it
# appears (CODE.md, "Validation ladder" 3, "Where the error lives"). One run of a case to
# `t_end`, with
#
#   * a time series every `every` (in M): the L1 drift outside `r = 2M` of `D`, `τ`, `𝔈ˣ`,
#     `𝔈ᶻ`, the same with two cells next to every face left out, and the charge `√γ q`;
#   * at the end, the drift and the charge binned by radius (layers of `M/4`), by the distance
#     to the nearest of the three Dirichlet planes through the hole (`x, y, z = 0`) and to the
#     nearest outer face (`4M`), in cells, outside `r = 2M`; and the L1 outside `2M` leaving
#     out `k = 0 … 4` cells next to every face;
#   * the resets and the stage solve's repairs binned by the same radial layers: the stage
#     limiter's and the step limiter's resets split by EntropyEOS's policy bits — level 2's
#     atmosphere (`FLAG_POL_ATMOSPHERE`) against level 1's projection faces — and the stage
#     solve's policy repairs with the `|` of their bits; "deep inside" is `r < (r_in + 2M)/2`
#     (decided by Erik 2026-11-02: resets there are not counted).
#
#     julia --project=. -t T bin/curved_localise.jl CASE N ROOTS T_END EVERY [key=value...]
#
# Keys: `B0` (Wald's field; 0 is Michel's flow alone), `q` (the monopole's charge; 0 likewise),
# `limiter` (default `mc`), `r_in` (default 1), `rho_c` (Michel's sonic density, default 1:
# a heavier fluid feels a spurious charge's force less), `riemann` (`hlle`, `llf`), `cfl`, `xi`
# (the gauge damping; default `evolve!`'s 16/L), `relhook` (`true`: the potentials' relative
# Dirichlet hook, AMR-7, where the cases' own is absolute since 2026-11-03 — the negative
# control, which grows), `interior` (`false`: no interior, the inside evolved), `qsource` (`full`, `beta`, `none`: a
# diagnostic replacement of Ampère's explicit source, `install_charge_source!`), `box` (`full`: Wald in the box `[−3, 3]³`
# shifted by `1/16`, around the hole, instead of the octant `[0, 4]³`). `h` is the box's side
# over `ROOTS N`.
#
# The reset's bits are counted by a copy of `reset_kernel!` evaluated into the module in this
# process only (`install_reset_bits!`), made from `src/evolution.jl`'s own text with one
# statement added where the reset writes a cell, so it is the reset bit for bit; the rest is
# the code as it is.

using TreeGRRMHD
using EntropyEOS
using Printf

const TG = TreeGRRMHD

# Per owned cell, flag bit and role (1 the stage limiter, 2 the step limiter): the resets
# whose policy flags had that bit.
const RESET_BITS = Ref(zeros(Float64, 1, 1, 1, 1, 1, 1))

function install_reset_bits!()
    src = read(joinpath(pkgdir(TreeGRRMHD), "src", "evolution.jl"), String)
    i = findfirst("@kernel function reset_kernel!(", src)
    j = findnext("\nend\n", src, last(i))
    code = src[first(i):first(j) + 3]
    anchor = "    if flags & EntropyEOS.FLAG_POL_ANY != 0\n"
    occursin(anchor, code) || error("the reset kernel changed; update the hook")
    code = replace(code, anchor => anchor * """
        let rb = Main.RESET_BITS[]
            for k in 9:18
                (flags >> (k - 1)) & one(UInt32) != 0 && (rb[n..., k, b, STEP ? 2 : 1] += 1)
            end
        end
""")
    Core.eval(TreeGRRMHD, Meta.parse(code))
    return nothing
end

# A diagnostic, in this process only: Ampère's explicit source `−√γ q (α vⁱ − βⁱ)` as it is
# (`full`), without the convected charge (`beta`: `+√γ q βⁱ` alone), or none (`none`: the
# charge neither convected nor shifted). Not a scheme: the run without it is no longer a
# solution of Maxwell's equations with charge; it says which term the growth needs.
function install_charge_source!(mode)
    mode == "full" && return nothing
    body = mode == "none" ? "zero(SVector{3,T})" :
           mode == "beta" ? "(m.sqrtγ * q) * m.β" : error("qsource is full, beta or none")
    Core.eval(TreeGRRMHD, Meta.parse("""
        @inline ampere_explicit_source(m::Metric3{T}, prim::NTuple{N,T}, q::T) where {N,T} =
            $body
    """))
    @info "Ampère's explicit source replaced" mode
    return nothing
end

const BIT_NAMES = Dict(9 => "ATMOSPHERE", 10 => "NONFINITE", 11 => "NOT_CONVERGED",
                       12 => "ρ_FLOORED", 13 => "ρ_CEILED", 14 => "YE_FLOORED", 15 => "YE_CEILED",
                       16 => "S_FLOORED", 17 => "S_CEILED", 18 => "W_CEILED")
bitnames(f::UInt32) = join([BIT_NAMES[k] for k in 9:18 if (f >> (k - 1)) & 1 != 0], "|")

# Positions, spacing and the per-cell arrays of a state difference, on the host.
function cells(p)
    o, hs = Array(p.origins), Array(p.spacings)
    N = p.f.U.forest.N
    return [(n=Tuple(I), b=b, x=TG.point_x(o[b], hs[b], Tuple(I), Val((0, 0, 0))), h=hs[b])
            for b in eachindex(o) for I in CartesianIndices((N, N, N))]
end

function qcells(p, u, t)
    P = update_primitives!(p, u, t)
    a = Array(P.work)
    G = P.G
    return n -> a[(n.n .+ G)..., TG.P_Q, n.b] * a[(n.n .+ G)..., TG.P_SQRTG, n.b]
end

const VARS = ((TG.U_D, "D"), (TG.U_TAU, "τ"), (TG.U_E, "𝔈ˣ"), (TG.U_E + 2, "𝔈ᶻ"), (TG.U_S, "S_x"))

function main(args)
    name, N, roots = args[1], parse(Int, args[2]), parse(Int, args[3])
    t_end, every = parse(Float64, args[4]), parse(Float64, args[5])
    kw = Dict(split(a, '='; limit=2) for a in args[6:end])
    r_in = parse(Float64, get(kw, "r_in", "1"))
    limiter = Symbol(get(kw, "limiter", "mc"))
    full = get(kw, "box", "octant") == "full"
    # The box around the hole, `[−3, 3]³` shifted by `1/16`, so that no vertex of an `h = M/4`
    # mesh lies at the singularity (at `M/8` a cell centre does: use another shift there).
    ext = ntuple(_ -> (-3 + 1 / 16, 3 + 1 / 16), 3)
    ρ_c = parse(Float64, get(kw, "rho_c", "1"))
    case = name == "wald" ? wald_case(Float64; B₀=parse(Float64, get(kw, "B0", "1")), r_in, ρ_c) :
           bondi_case(Float64; q=parse(Float64, get(kw, "q", "9")), r_in, ρ_c)
    # The same case on another box (Wald only: the monopole's string would cross it), and/or
    # with the potentials' relative Dirichlet hook (`relhook=true`, the negative control of
    # the absolute one; decided by Erik 2026-11-03).
    full && name != "wald" && error("box=full is for the Wald case")
    relhook = get(kw, "relhook", "false") == "true"
    if full || relhook
        c = case
        case = MHDCase(Float64, Val(3); fields=c.fields, potential=c.potential, eos=c.eos,
                       opts=c.opts, proj=c.proj, pol=c.pol, model=c.model, provider=c.provider,
                       𝔅̄=c.𝔅̄, periodic=false, extents=full ? ext : c.extents, roots=c.roots,
                       interior=c.interior, static_potential=!relhook)
    end
    interior = get(kw, "interior", "true") == "true" ? case.interior : nothing
    lo, hi = full ? (-3 + 1 / 16, 3 + 1 / 16) : (0.0, 4.0)
    L = hi - lo
    h = L / (roots * N)
    nb = roots^3
    RESET_BITS[] = zeros(Float64, N, N, N, 18, nb, 2)
    install_reset_bits!()
    install_charge_source!(get(kw, "qsource", "full"))
    @info "curved localise" name N roots h t_end every kw Threads.nthreads()
    uref = Ref{Any}(nothing)
    cl = Ref{Any}(nothing)
    next = Ref(every)
    U(u, p) = Array(TG.statearray(u, p.sets, TG.SEG_U))
    function series(p, u, t)
        uref[] === nothing && (uref[] = Array(TG.statearray(exact_state(p, case, 0.0), p.sets, TG.SEG_U));
                               cl[] = cells(p))
        d = U(u, p) .- uref[]
        q = qcells(p, u, t)
        acc = zeros(length(VARS) + 1, 2)
        vol = zeros(2)
        for c in cl[]
            sum(abs2, c.x) ≥ 4 || continue
            inner = minimum(c.x) ≥ lo + 2h && maximum(c.x) ≤ hi - 2h
            for (k, keep) in ((1, true), (2, inner))
                keep || continue
                vol[k] += 1
                for (i, (v, _)) in enumerate(VARS)
                    acc[i, k] += abs(d[c.n..., v, c.b])
                end
                acc[end, k] += abs(q(c))
            end
        end
        acc[:, 1] ./= vol[1]
        acc[:, 2] ./= vol[2]
        @printf("TS t=%6.2f  r≥2M: D %.4e τ %.4e 𝔈ˣ %.4e 𝔈ᶻ %.4e S_x %.4e q %.4e | 2 cells off faces: D %.4e τ %.4e 𝔈ˣ %.4e 𝔈ᶻ %.4e S_x %.4e q %.4e\n",
                t, acc[:, 1]..., acc[:, 2]...)
        flush(stdout)
    end
    t0 = time()
    # `invokelatest`: the hooks above were evaluated after `main` began, in a newer world.
    ekw = (; riemann=Symbol(get(kw, "riemann", "hlle")),
           cfl=haskey(kw, "cfl") ? parse(Float64, kw["cfl"]) : nothing,
           ξ=haskey(kw, "xi") ? parse(Float64, kw["xi"]) : nothing)
    r = Base.invokelatest(evolve!, case, Val(3); ekw..., N=N, roots=roots, t_end=t_end, limiter=limiter,
                interior=interior,
                after_step=(p, u, t) -> if t ≥ next[] - 1e-9
                    series(p, u, t)
                    next[] += every
                end)
    @printf("RUN %s N=%d roots=%d h=%.5f steps=%d wall=%.1fs\n", name, N, roots, h, r.nsteps,
            time() - t0)
    p, u, t = r.p, r.u, r.t
    uref[] === nothing && series(p, u, t)
    for rmin in (0.0, 1.5, 2.0)
        qn = charge_norms(p, u, t, rmin)
        @printf("CHARGE r ≥ %.1f: √γ q L1 %.4e max %.4e total %.4e\n", rmin, qn.l1, qn.max, qn.total)
    end
    @printf("INJECTED D %.4e τ %.4e (the step limiter's writes, the whole octant)\n",
            sum(c -> c.tally.injected_D, r.chunks), sum(c -> c.tally.injected_τ, r.chunks))
    d = U(u, p) .- uref[]
    q = qcells(p, u, t)
    X = Array(p.f.X.work)
    GX = p.f.X.G
    xv(c, s) = X[(c.n .+ GX)..., s, c.b]
    rb = RESET_BITS[]

    # Radial layers of M/4.
    nl = ceil(Int, sqrt(3) * max(abs(lo), hi) / 0.25)
    rows = [zeros(length(VARS) + 1) for _ in 1:nl]
    cnt = zeros(Int, nl)
    res = zeros(nl, 3)                     # stage resets, step resets, stage-solve repairs
    atm = zeros(nl, 2)                     # resets with the atmosphere bit, per role
    bits = [zeros(18, 2) for _ in 1:nl]
    por = fill(UInt32(0), nl)
    deep = (r_in + 2) / 2
    for c in cl[]
        rr = sqrt(sum(abs2, c.x))
        l = min(nl, floor(Int, rr / 0.25) + 1)
        cnt[l] += 1
        for (i, (v, _)) in enumerate(VARS)
            rows[l][i] += abs(d[c.n..., v, c.b])
        end
        rows[l][end] += abs(q(c))
        res[l, 1] += xv(c, TG.X_RESETS)
        res[l, 2] += xv(c, TG.X_STEP_RESETS)
        res[l, 3] += xv(c, TG.X_POLICY)
        por[l] |= unsafe_trunc(UInt32, xv(c, TG.X_POLICY_OR))
        for role in 1:2, k in 9:18
            bits[l][k, role] += rb[c.n..., k, c.b, role]
        end
        atm[l, 1] += rb[c.n..., 9, c.b, 1]
        atm[l, 2] += rb[c.n..., 9, c.b, 2]
    end
    println("LAYERS (r in M; mean |Δ| per cell; resets stage/step with atmosphere; stage-solve repairs and their bits; deep inside r < $deep not counted by Erik's rule)")
    for l in 1:nl
        cnt[l] == 0 && continue
        m = rows[l] ./ cnt[l]
        l1 = [k for k in 10:18 if bits[l][k, 1] + bits[l][k, 2] > 0]
        @printf("LAYER %.2f–%.2f cells %6d | D %.3e τ %.3e 𝔈ˣ %.3e 𝔈ᶻ %.3e S_x %.3e q %.3e | resets %d/%d (atm %d/%d; level-1 bits %s) | stage repairs %d [%s]\n",
                (l - 1) / 4, l / 4, cnt[l], m..., res[l, 1], res[l, 2], atm[l, 1], atm[l, 2],
                join(["$(BIT_NAMES[k])=$(Int(bits[l][k, 1]))/$(Int(bits[l][k, 2]))" for k in l1], ","),
                res[l, 3], bitnames(por[l]))
    end
    tot(f) = sum(f(l) for l in 1:nl if (l - 0.5) / 4 ≥ deep; init=0.0)
    @printf("COUNTED (r ≥ %.2f): resets %d/%d, stage repairs %d; deep inside: resets %d/%d, stage repairs %d\n",
            deep, tot(l -> res[l, 1]), tot(l -> res[l, 2]), tot(l -> res[l, 3]),
            sum(res[:, 1]) - tot(l -> res[l, 1]), sum(res[:, 2]) - tot(l -> res[l, 2]),
            sum(res[:, 3]) - tot(l -> res[l, 3]))

    # Outside 2M, by the distance to the low planes and to the outer faces, in cells.
    for (label, dist) in (("low faces", c -> minimum(c.x) - lo), ("high faces", c -> hi - maximum(c.x)))
        acc = Dict{Int,Vector{Float64}}()
        for c in cl[]
            sum(abs2, c.x) ≥ 4 || continue
            k = min(8, floor(Int, dist(c) / h))
            a = get!(acc, k, zeros(length(VARS) + 2))
            a[1] += 1
            for (i, (v, _)) in enumerate(VARS)
                a[i + 1] += abs(d[c.n..., v, c.b])
            end
            a[end] += abs(q(c))
        end
        for k in sort(collect(keys(acc)))
            a = acc[k]
            @printf("FACE %-18s cell %s cells %6d | D %.3e τ %.3e 𝔈ˣ %.3e 𝔈ᶻ %.3e S_x %.3e q %.3e | share of D's L1 %.3f\n",
                    label, k == 8 ? "≥8" : string(k), Int(a[1]), (a[2:end] ./ a[1])...,
                    a[2] / sum(v[2] for v in values(acc)))
        end
    end
    for k in 0:4
        acc = zeros(length(VARS) + 1)
        n = 0
        for c in cl[]
            sum(abs2, c.x) ≥ 4 && minimum(c.x) ≥ lo + k * h && maximum(c.x) ≤ hi - k * h || continue
            n += 1
            for (i, (v, _)) in enumerate(VARS)
                acc[i] += abs(d[c.n..., v, c.b])
            end
            acc[end] += abs(q(c))
        end
        @printf("EXCL %d cells off every face, r ≥ 2M: cells %d | D %.4e τ %.4e 𝔈ˣ %.4e 𝔈ᶻ %.4e S_x %.4e q %.4e\n",
                k, n, (acc ./ n)...)
    end
    return nothing
end

main(ARGS)
