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
                          :checkpoint => nothing, :every => 1)
    for a in args
        k, v = split(a, '='; limit=2)
        key = Symbol(k)
        haskey(kw, key) || error("neutrino_star.jl: unknown key `$k` ($(join(keys(kw), ", ")))")
        kw[key] = key in (:N, :levels, :roots, :every) ? parse(Int, v) :
                  key in (:table, :out, :checkpoint) ? String(v) :
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

function main(args)
    o = parse_args(args)
    d = load_table(o.table)
    cv = E.CompositionView(E.CompositionTable(d.table, d.eos))
    units = EOSUnits(E.EOSTableView(d.eos); density=MSUN_DENSITY_CGS)
    rates = RJS96Rates(units, d.eos, cv)
    case = MHD.tabulated_tov_case(Float64; d.table, d.eos, o.ρ_c, o.ρ_min, o.s, o.L, o.roots)
    nu = M0Leakage(rates; composition=cv, center=(0.0, 0.0, 0.0), r_core=o.r_core)
    ck = o.checkpoint
    ck !== nothing && isdir(ck) && (ck = joinpath(ck, "state.h5"))
    restart = ck !== nothing && isfile(ck) ? ck : nothing
    io = o.out === nothing ? stdout : open(o.out, restart === nothing ? "w" : "a")
    restart === nothing &&
        println(io, "# t  L_N L_E (nu_e, nu_e-bar, nu_x)  N E (the three)  T_c[MeV]")
    # The first sample after the checkpoint's time, on the same grid of times.
    t_ck = restart === nothing ? 0.0 : Float64(MHD.load_run(restart).run.t)
    next = Ref(restart === nothing ? 0.0 : (floor(t_ck / o.sample + 1e-9) + 1) * o.sample)
    function sample(p, u, t)
        t + 1e-9 ≥ next[] || return nothing
        L = neutrino_luminosity(p, u; o.r_L, o.width)
        nt = neutrino_totals(p, u)
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
    nt = neutrino_totals(r.p, r.u)
    injY = sum(c -> c.tally.injected_DYₑ, r.chunks)
    net = tot[MHD.U_DYE] - r.totals0[MHD.U_DYE] + nt[1] - nt[3] - injY
    println("neutrino_star: $(o.s === nothing ? "cold" : "hot, s = $(o.s)") star, N = $(o.N), " *
            "t = $(r.t), $(r.nsteps) steps, $(nleaves(first(r.p.sets).forest)) blocks")
    println("lepton number net of the reset (√γ D Yₑ + N_e − N_ē − injected): $net " *
            "(of $(r.totals0[MHD.U_DYE])); √γ D Yₑ changed by $(tot[MHD.U_DYE] - r.totals0[MHD.U_DYE])")
    println("wall time $(round(wall; digits=1)) s, $(round(wall / max(r.nsteps, 1); digits=3)) s " *
            "per step over the whole run's steps (a resumed run: its own wall time)")
    io === stdout || close(io)
    return nothing
end

main(ARGS)
