# Step 9 (rung 3): the stationarity study of the two Kerr-Schild cases, for `CODE.md`'s
# table ("Validation ladder" 3) — the L1 drift from the exact stationary solution after
# `t_end`, at a list of resolutions, and its order; the charge `√γ q = ∂ᵢ𝔈ⁱ` outside the
# hole; `∇·𝔅`; the stage branches, the resets and what they inject; the wall time.
#
#     julia --project=. -t T bin/curved_stationarity.jl CASE T_END N... [key=value...]
#
# CASE is `wald` or `bondi`; N the cells per block along each dimension, on `roots³` root
# blocks (key `roots`, default the case's 2) of `[0, 4]³`, so `h = 4/(roots N)` — more roots,
# more blocks for the threads to share. Keys: `r_in` (the interior's radius, default the case's 1),
# `stage`, `step` (`true`/`false`: the interior's limiter roles, default both), `chunk` (the
# chunk length, default `t_end`), `limiter` (default `mc`), `xi` (the gauge damping,
# default `evolve!`'s 16/L), `interior` (`false`: none, the inside evolved too). Each
# resolution is one run; the rows are printed as they finish, then the rates by least
# squares over the resolutions given (`convergence_rate`).
#
# Measured on Symmetry (`amdq`, one job per row); the table in `CODE.md`, "Validation
# ladder" 3, is assembled from the logs.

using TreeGRRMHD
using Printf

const TG = TreeGRRMHD

function parse_args(args)
    length(args) ≥ 3 || error("usage: curved_stationarity.jl wald|bondi T_END N... [key=value...]")
    name = args[1]
    t_end = parse(Float64, args[2])
    Ns = Int[]
    kw = Dict{String,String}()
    for a in args[3:end]
        if occursin('=', a)
            k, v = split(a, '='; limit=2)
            kw[k] = v
        else
            push!(Ns, parse(Int, a))
        end
    end
    return name, t_end, Ns, kw
end

function main(args)
    name, t_end, Ns, kw = parse_args(args)
    r_in = parse(Float64, get(kw, "r_in", "1"))
    ctor = name == "wald" ? wald_case : name == "bondi" ? bondi_case :
           error("CASE is wald or bondi, got $name")
    case = ctor(Float64; r_in=r_in)
    interior = get(kw, "interior", "true") == "true" ?
               HoleInterior(r_in; stage=get(kw, "stage", "true") == "true",
                            step=get(kw, "step", "true") == "true") : nothing
    chunk = parse(Float64, get(kw, "chunk", string(t_end)))
    limiter = Symbol(get(kw, "limiter", "mc"))
    ξ = haskey(kw, "xi") ? parse(Float64, kw["xi"]) : nothing
    roots = parse(Int, get(kw, "roots", "2"))
    @info "curved stationarity" name t_end Ns roots r_in interior chunk limiter ξ Threads.nthreads()
    hs, rows = Float64[], []
    for N in Ns
        t0 = time()
        r = evolve!(case, Val(3); N=N, t_end=t_end, chunk=chunk, limiter=limiter, ξ=ξ,
                    interior=interior, roots=roots)
        wall = time() - t0
        uref = TG.exact_state(r.p, case, r.t)
        l = outside_l1(r.p, r.u .- uref, 0.0)
        lout = outside_l1(r.p, r.u .- uref, 2.0)        # outside the horizon
        fl = field_l1(r.p, r.u, uref, r.t, 0.0)
        q = charge_norms(r.p, r.u, r.t, r_in)
        qh = charge_norms(r.p, r.u, r.t, 2.0)
        div = max_div_B(r.p, r.u, r.t)
        bits = sum(c -> collect(c.tally.bits), r.chunks)
        resets = sum(c -> c.tally.resets, r.chunks)
        sresets = sum(c -> c.tally.step_resets, r.chunks)
        policy = sum(c -> c.tally.policy, r.chunks)
        injD = sum(c -> c.tally.injected_D, r.chunks)
        h = 4 / (roots * N)
        push!(hs, h)
        push!(rows, (; N, l, lout, fl, q, qh))
        @printf("ROW %s N=%d h=%.5f steps=%d wall=%.1fs\n", name, N, h, r.nsteps, wall)
        println("  L1 U (all)      = ", map(x -> @sprintf("%.4e", x), l.U))
        println("  L1 U (r ≥ 2M)   = ", map(x -> @sprintf("%.4e", x), lout.U))
        println("  L1 A, Φ̃         = ", map(x -> @sprintf("%.4e", x), l.A), " ", @sprintf("%.4e", l.Φ))
        println("  L1 𝔅            = ", map(x -> @sprintf("%.4e", x), fl))
        @printf("  √γ q (r ≥ r_in): L1 %.4e max %.4e total %.4e; (r ≥ 2M): L1 %.4e max %.4e\n",
                q.l1, q.max, q.total, qh.l1, qh.max)
        @printf("  max ∇·𝔅 h/max|𝔅| = %.3e\n", div)
        println("  stage bits = ", bits, "  resets = $resets/$sresets  policy = $policy  injected D = $injD")
        flush(stdout)
    end
    if length(Ns) ≥ 2
        rate(f) = TG.convergence_rate(hs, map(f, rows))
        names = ("D", "S₁", "S₂", "S₃", "τ", "DYₑ", "Ds", "𝔈¹", "𝔈²", "𝔈³")
        println("RATES (all):    ", join((@sprintf("%s %.2f", names[v], rate(r -> r.l.U[v])) for v in 1:10), ", "))
        println("RATES (r ≥ 2M): ", join((@sprintf("%s %.2f", names[v], rate(r -> r.lout.U[v])) for v in 1:10), ", "))
        println("RATES A: ", join((@sprintf("%.2f", rate(r -> r.l.A[j])) for j in 1:3), ", "),
                @sprintf("  Φ̃: %.2f", rate(r -> r.l.Φ)),
                "  𝔅: ", join((@sprintf("%.2f", rate(r -> r.fl[d])) for d in 1:3), ", "),
                @sprintf("  q L1: %.2f", rate(r -> r.q.l1)))
    end
    return nothing
end

main(ARGS)
