# Step 20 (rung 8g): DD2's star with neutrinos on its Cowling metric ("Validation ladder"
# 8g in `CODE.md`): the cold star in β-equilibrium, or the hot star at constant entropy,
# with `RJS96Rates` on the table's composition, sampled at intervals: the luminosities
# through a shell outside the star (`neutrino_luminosity`), the neutrino totals, and the
# temperature at the centre. One run per invocation.
#
#     julia --project=. -t T bin/neutrino_star.jl [key=value...]
#
# Keys: `table` (`dd2`), `ρ_c` (9.4e−4, the 1.4 M☉ star), `ρ_min` (1e−10), `s` (the hot
# star's entropy per baryon; none, the cold star), `N` (cells per block, 8), `levels` (box
# levels, 1), `L` (24), `roots` (4), `t_end` (100), `chunk` (the chunks of `evolve!`, at
# whose boundaries a checkpoint is written, 10), `sample` (the sampling interval, 1),
# `r_L` (the shell's radius, 16, about twice the star's), `width` (its half-width, 2),
# `r_core` (1), `limiter` (`mc`), `out` (the time series, one row per sample: `t`, the
# number and energy luminosities of ν_e, ν̄_e, ν_x, the totals `N`, `𝓔` of the three, and
# the central temperature in MeV; default stdout), `checkpoint` (a file, or a directory
# holding it as `state.h5`, as SimLaunch's restarts give one: the run saves to it every
# `every` chunks, default 1, and resumes from it if it exists, appending to `out`). The
# summary at the end: lepton number's change net of the reset, steps, wall time per step.
#
# For the cold star's heating ("Open questions" in `CODE.md`): `neutrinos` (`true`; `false`
# evolves the matter alone, its luminosity and totals columns `NaN`), and `profile` (a file
# of radial profiles, none by default) every `profile_every` (10) from the start: one row
# per shell of width `dr` (0.5) about the centre holding a cell, `t`, the shell's middle
# radius, the mean `ρ★`, the `ρ★`-weighted mean temperature in MeV, the hottest cell's, the
# mean `Yₑ` and `s` (`ρ★`-weighted, per baryon), and the number of cells, from `P` as it is.
# `nanprobe` (`false`): after every step, report each cell whose step-limiter tally of the
# chunk has turned non-finite, once, with its place and `P` there and at its neighbours.
#
# The luminosity reads `P` as the last stage left it and does not refresh it, since with
# neutrinos the next implicit stage reads it ("Sharp edges").

using TreeAMR
using TreeGRRMHD
using Printf
const MHD = TreeGRRMHD
const E = MHD.EntropyEOS
include(joinpath(@__DIR__, "tables.jl"))

function parse_args(args)
    kw = Dict{Symbol,Any}(:table => "dd2", :ρ_c => 9.4e-4, :ρ_min => 1e-10, :s => nothing,
                          :N => 8, :levels => 1, :L => 24.0, :roots => 4, :t_end => 100.0,
                          :chunk => 10.0, :sample => 1.0, :r_L => 16.0, :width => 2.0,
                          :r_core => 1.0, :limiter => :mc, :out => nothing,
                          :checkpoint => nothing, :every => 1,
                          :neutrinos => true, :profile => nothing, :profile_every => 10.0,
                          :dr => 0.5, :nanprobe => false)
    for a in args
        k, v = split(a, '='; limit=2)
        key = Symbol(k)
        haskey(kw, key) || error("neutrino_star.jl: unknown key `$k` ($(join(keys(kw), ", ")))")
        kw[key] = key in (:N, :levels, :roots, :every) ? parse(Int, v) :
                  key in (:table, :out, :checkpoint, :profile) ? String(v) :
                  key in (:neutrinos, :nanprobe) ? parse(Bool, v) :
                  key == :limiter ? Symbol(v) : parse(Float64, v)
    end
    return (; kw...)
end

# The temperature at the owned cell nearest the centre, from `P` as it is.
function central_temperature(p)
    P = MHD.hostcopy(p.f.P)
    G = P.G[1]
    best, Tc = Inf, NaN
    for b in 1:nblocks(P), idx in CartesianIndices(ntuple(_ -> P.forest.N, 3))
        s = Tuple(idx) .+ G
        r2 = sum(abs2, coordinates(P, b, s))
        r2 < best && ((best, Tc) = (r2, exp10(P.work[s..., MHD.P_UEOS, b])))
    end
    return Tc
end

# The radial profiles of `P` as it is, in shells of width `dr` (the header's `profile`).
function write_profile(io, p, t, dr)
    P = MHD.hostcopy(p.f.P)
    G = P.G[1]
    acc = Dict{Int,NTuple{7,Float64}}()       # Σρv, Σv, ΣρvT, max T, ΣρvYₑ, Σρvs, cells
    for b in 1:nblocks(P)
        vol = spacing(Float64, P.forest, blockkey(P, b))^3
        for idx in CartesianIndices(ntuple(_ -> P.forest.N, 3))
            s = Tuple(idx) .+ G
            k = floor(Int, sqrt(sum(abs2, coordinates(P, b, s))) / dr)
            q = P.work[s..., MHD.P_RHO, b] * vol
            Tc = exp10(P.work[s..., MHD.P_UEOS, b])
            a = get(acc, k, (0.0, 0.0, 0.0, -Inf, 0.0, 0.0, 0.0))
            acc[k] = (a[1] + q, a[2] + vol, a[3] + q * Tc, max(a[4], Tc),
                      a[5] + q * P.work[s..., MHD.P_YE, b], a[6] + q * P.work[s..., MHD.P_S, b],
                      a[7] + 1)
        end
    end
    for k in sort!(collect(keys(acc)))
        m, v, mT, Tmax, mY, ms, n = acc[k]
        @printf(io, "%.6f %.4f %.6e %.6e %.6e %.6e %.6e %d\n", t, (k + 0.5) * dr, m / v,
                mT / m, Tmax, mY / m, ms / m, Int(n))
    end
    println(io)
    flush(io)
    return nothing
end

# The cells whose step-limiter tally (`X_INJECTED_D`, `_TAU`, `_DYE`) is not finite and
# was not reported in this chunk: their place, the tally, and `P` at them and their
# neighbours (the header's `nanprobe`). `seen` is emptied once the tally is finite again.
function probe_tally(p, t, seen)
    slots = (MHD.X_INJECTED_D, MHD.X_INJECTED_TAU, MHD.X_INJECTED_DYE)
    nonfinite(v) = mesh_mapreduce(x -> isfinite(x) ? 0 : 1, +, 0, p.f.X; vars=v)
    bad = nonfinite(MHD.X_INJECTED_D:MHD.X_INJECTED_TAU) + nonfinite(MHD.X_INJECTED_DYE)
    bad == 0 && (empty!(seen); return nothing)
    X = MHD.hostcopy(p.f.X)
    P = MHD.hostcopy(p.f.P)
    GX, GP = X.G[1], P.G[1]
    state(s, b) = @sprintf("ρ★ %.3e T %.3e Yₑ %.4f s %.3e flags %d", P.work[s..., MHD.P_RHO, b],
                           exp10(P.work[s..., MHD.P_UEOS, b]), P.work[s..., MHD.P_YE, b],
                           P.work[s..., MHD.P_S, b], Int(P.work[s..., MHD.P_FLAGS, b]))
    for b in 1:nblocks(X), idx in CartesianIndices(ntuple(_ -> X.forest.N, 3))
        sx = Tuple(idx) .+ GX
        all(v -> isfinite(X.work[sx..., v, b]), slots) && continue
        (b, Tuple(idx)) in seen && continue
        push!(seen, (b, Tuple(idx)))
        sp = Tuple(idx) .+ GP
        x = coordinates(P, b, sp)
        h = spacing(Float64, P.forest, blockkey(P, b))
        println("nanprobe: t = $t, block $b cell $(Tuple(idx)), h = $h, x = $x, " *
                "r = $(sqrt(sum(abs2, x))); injected D, τ, DYₑ = " *
                "$(map(v -> X.work[sx..., v, b], slots)), step resets " *
                "$(X.work[sx..., MHD.X_STEP_RESETS, b]), ν rate $(X.work[sx..., MHD.X_NU_RATE, b])")
        println("  here: ", state(sp, b))
        for d in 1:3, e in (-1, 1)
            q = ntuple(k -> sp[k] + (k == d ? e : 0), 3)
            println("  $(e < 0 ? "-" : "+")$("xyz"[d]): ", state(q, b))
        end
    end
    flush(stdout)
    return nothing
end

function main(args)
    o = parse_args(args)
    d = load_table(o.table)
    cv = E.CompositionView(E.CompositionTable(d.table, d.eos))
    units = EOSUnits(E.EOSTableView(d.eos); density=MSUN_DENSITY_CGS)
    rates = RJS96Rates(units, d.eos, cv)
    case = MHD.tabulated_tov_case(Float64; d.table, d.eos, o.ρ_c, o.ρ_min, o.s, o.L, o.roots)
    nu = o.neutrinos ? M0Leakage(rates; composition=cv, center=(0.0, 0.0, 0.0),
                                 r_core=o.r_core) : nothing
    ck = o.checkpoint
    ck !== nothing && isdir(ck) && (ck = joinpath(ck, "state.h5"))
    restart = ck !== nothing && isfile(ck) ? ck : nothing
    io = o.out === nothing ? stdout : open(o.out, restart === nothing ? "w" : "a")
    restart === nothing &&
        println(io, "# t  L_N L_E (nu_e, nu_e-bar, nu_x)  N E (the three)  T_c[MeV]")
    # The first sample after the checkpoint's time, on the same grid of times.
    t_ck = restart === nothing ? 0.0 : Float64(MHD.load_run(restart).run.t)
    next = Ref(restart === nothing ? 0.0 : (floor(t_ck / o.sample + 1e-9) + 1) * o.sample)
    pio = o.profile === nothing ? nothing : open(o.profile, restart === nothing ? "w" : "a")
    pio === nothing || restart !== nothing ||
        println(pio, "# t  r  ρ★  T[MeV] (ρ★-weighted)  T_max  Yₑ  s  cells")
    pnext = Ref(restart === nothing ? 0.0 :
                (floor(t_ck / o.profile_every + 1e-9) + 1) * o.profile_every)
    seen = Set{Tuple{Int,NTuple{3,Int}}}()
    function sample(p, u, t)
        o.nanprobe && probe_tally(p, t, seen)
        if pio !== nothing && t + 1e-9 ≥ pnext[]
            write_profile(pio, p, t, o.dr)
            pnext[] += o.profile_every
        end
        t + 1e-9 ≥ next[] || return nothing
        L = o.neutrinos ? neutrino_luminosity(p, u; o.r_L, o.width) : ntuple(_ -> NaN, 6)
        nt = o.neutrinos ? neutrino_totals(p, u) : ntuple(_ -> NaN, 6)
        @printf(io, "%.6f %s %s %.6e\n", t, join((@sprintf("%.6e", x) for x in L), " "),
                join((@sprintf("%.6e", x) for x in nt), " "), central_temperature(p))
        flush(io)
        next[] += o.sample
        return nothing
    end
    t0 = time()
    r = evolve!(case, Val(3); N=o.N, t_end=o.t_end, chunk=o.chunk, limiter=o.limiter,
                refine=f -> MHD.tov_boxes!(f, case; levels=o.levels), neutrinos=nu,
                after_step=sample, checkpoint=ck, checkpoint_every=o.every, restart)
    wall = time() - t0
    tot = conserved_totals(r.p, r.u)
    nt = o.neutrinos ? neutrino_totals(r.p, r.u) : ntuple(_ -> 0.0, 6)
    if pio !== nothing          # the end, unless the last sample was already there
        r.t > pnext[] - o.profile_every + 1e-9 && write_profile(pio, r.p, r.t, o.dr)
        close(pio)
    end
    injY = sum(c -> c.tally.injected_DYₑ, r.chunks)
    net = tot[MHD.U_DYE] - r.totals0[MHD.U_DYE] + nt[1] - nt[3] - injY
    println("neutrino_star: $(o.s === nothing ? "cold" : "hot, s = $(o.s)") star, N = $(o.N), " *
            "$(o.neutrinos ? "" : "no neutrinos, ")" *
            "t = $(r.t), $(r.nsteps) steps, $(nleaves(first(r.p.sets).forest)) blocks")
    println("lepton number net of the reset (√γ D Yₑ + N_e − N_ē − injected): $net " *
            "(of $(r.totals0[MHD.U_DYE])); √γ D Yₑ changed by $(tot[MHD.U_DYE] - r.totals0[MHD.U_DYE])")
    println("wall time $(round(wall; digits=1)) s, $(round(wall / max(r.nsteps, 1); digits=3)) s " *
            "per step over the whole run's steps (a resumed run: its own wall time)")
    io === stdout || close(io)
    return nothing
end

main(ARGS)
