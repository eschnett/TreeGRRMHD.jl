# Step 11 (rung 5): the coupled TOV star's studies, for `CODE.md`'s tables ("Validation
# ladder" 5): one run per invocation through `tov_coupled.jl`'s problem and callbacks, every
# `chunk` the central density, TreeGH's matter-corrected constraints inside the star
# (`r < 6`, away from the surface at `R̄ = 8.1`) and outside it (`r ≥ 10`), and what the
# reset injected; at the end the rest mass against the injection. The mode frequencies come
# from the series by `bin/tov_modes.jl ref=coupled`. Checkpointed, for pre-emptible queues.
#
#     julia --project=coupled -t T coupled/tov_coupled_study.jl [key=value...]
#
# Keys: `N` (16), `levels` (1), `L` (24), `roots` (4), `t_end` (200), `chunk` (the sampling
# interval, 1), `limiter` (`ppm`), `cfl_gh` (TreeGH's CFL factor, 1/2), `γ0M` and `γ2`
# (TreeGH's damping, `γ0 = γ0M/M`; 1/10 and 0), `δv` (the star's initial velocity along its
# coupled radial modes, the amplitudes of F, H1 and H2 in `ρ_c/ρ_c`, `δv=a₁,a₂,a₃`;
# `tov_case`; none), `order` (the matter's transfer to the vertices, 2 or 4; 4), `ρ_min`
# (the atmosphere's density; 1e−12), `out` (the series,
# `t ρ_c ‖ℋ‖₂in ‖ℳ‖₂in ‖ℋ‖₂out ‖ℳ‖₂out injected_D steps ΔM resets injected_τ` per chunk:
# the injections, the reset's writes and `steps` since `t = 0`, `ΔM` the rest mass's change),
# `checkpoint` (a file, or a directory holding it as `state.jls`, as SimLaunch's restarts
# give one, saved every `every` chunks, default 20, and resumed from if it exists), and *(added in step 13)* `table` (`dd2`, `sfho`: the star on that table in cold
# β-equilibrium, read from `ENTROPYEOS_TABLE_DIR`) with `ρ_c` (its central density,
# geometric). The series then gains four columns: `ΔDYₑ injected_DYₑ s_floors s_floor_τ`,
# the change of the total `D Yₑ`, the reset's injection into it, and its entropy floors
# outside the atmosphere with their `τ`, since `t = 0`.

include(joinpath(@__DIR__, "tov_coupled.jl"))
include(joinpath(@__DIR__, "..", "bin", "tables.jl"))
using Serialization
using StaticArrays: SVector

function study_norms(p, u, t)
    metric = metric_at(p, u, t)
    ρ_c = MHD.central_density(p.mhd, view(u, p.s[2]), t; metric)   # prepares the matter
    m = matter(p)
    T = eltype(u)
    c(lo, hi) = gh_constraint_norms(p.gh, view(u, p.s[1]), t; matter=m,
                                    mask=GH.ShellMask(SVector{3,T}(0, 0, 0), T(lo), T(hi)))
    ci, co = c(0, 6), c(10, 1000)
    forget_fill!(p)                                  # TreeGH was prepared by another verb
    mom(x) = sqrt(sum(abs2, x.mom_l2))
    return (; ρ_c, ham_in=ci.ham_l2, mom_in=mom(ci), ham_out=co.ham_l2, mom_out=mom(co))
end

function study(T=Float64; N=16, levels=1, L=24, roots=4, t_end=200, chunk=1, limiter=:ppm,
               cfl_gh=T(1 // 2), γ0M=1 // 10, γ2=0, δv=(0, 0, 0), order=4, ρ_min=1e-12,
               out=nothing, checkpoint=nothing, every=20, table=nothing, ρ_c=nothing)
    case = if table === nothing
        nothing
    else
        d = load_table(table)
        MHD.tabulated_tov_case(T; d.table, d.eos, ρ_c, ρ_min, L, roots)
    end
    (; p, u, forest) = problems(T; N, levels, L, roots, limiter, γ0M, γ2, δv, order, ρ_min,
                                case)
    tot0 = MHD.conserved_totals(p.mhd, view(u, p.s[2]))
    M0, Y0 = tot0[MHD.U_D], tot0[MHD.U_DYE]
    cfl_mhd = T(MHD.default_cfl(limiter))
    t_end, chunk = T(t_end), T(chunk)
    nchunks = MHD.chunk_count(t_end, chunk)
    t, integ, steps, injected, c0 = zero(T), nothing, 0, zero(T), 0
    resets, injected_τ = 0, zero(T)
    injY, floors, floorτ = zero(T), 0, zero(T)
    rows = Any[]
    # A directory (SimLaunch's `ck/`, `mutable_restarts`) holds the checkpoint as a file.
    checkpoint !== nothing && isdir(checkpoint) && (checkpoint = joinpath(checkpoint, "state.jls"))
    if checkpoint !== nothing && isfile(checkpoint)
        s = deserialize(checkpoint)
        copyto!(u, s.u)
        c0, t, steps, injected, rows = s.c, s.t, s.steps, s.injected, s.rows
        # Checkpoints from before the reset's counts have none (they start at the resume).
        resets, injected_τ = get(s, :resets, 0), get(s, :injected_τ, zero(T))
        injY, floors, floorτ = get(s, :injY, zero(T)), get(s, :floors, 0), get(s, :floorτ, zero(T))
        @info "resumed from $checkpoint at chunk $c0, t = $t"
    else
        push!(rows, (; t, study_norms(p, u, t)..., injected=zero(T), steps=0,
                     ΔM=zero(T), resets=0, injected_τ=zero(T), ΔY=zero(T), injY=zero(T),
                     floors=0, floorτ=zero(T)))
    end
    # The series is rewritten from the rows the checkpoint holds, so that a resumed run's
    # file has no row twice (a pre-emption after a row and before its checkpoint).
    io = out === nothing ? nothing : open(out, "w")
    put(r) = io === nothing ? nothing :
             (println(io, join((r.t, r.ρ_c, r.ham_in, r.mom_in, r.ham_out, r.mom_out,
                                r.injected, r.steps, get(r, :ΔM, NaN), get(r, :resets, -1),
                                get(r, :injected_τ, NaN), get(r, :ΔY, NaN), get(r, :injY, NaN),
                                get(r, :floors, -1), get(r, :floorτ, NaN)), "  ")); flush(io))
    foreach(put, rows)
    wall0 = time()
    for c in (c0 + 1):nchunks
        t1 = c == nchunks ? t_end : c * chunk
        MHD.clear_tally!(p.mhd)
        nsteps = step_count(p, u, t, t1, forest, cfl_gh, cfl_mhd)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=(t1 - t) / nsteps,
                         stage_limiter=stage_limit!, step_limiter=step_limit!,
                         partition=:even, alias_u0=true, reuse=integ)
        MHD.with_step_cache(() -> IRK.solve!(integ), p.mhd)
        steps += nsteps
        t = t1
        tally = MHD.ohm_tally(p.mhd)
        injected += tally.injected_D
        resets += tally.step_resets
        injected_τ += tally.injected_τ
        injY += tally.injected_DYₑ
        floors += tally.s_floors
        floorτ += tally.s_floor_τ
        tot = MHD.conserved_totals(p.mhd, view(u, p.s[2]))
        ΔM, ΔY = tot[MHD.U_D] - M0, tot[MHD.U_DYE] - Y0
        r = (; t, study_norms(p, u, t)..., injected, steps, ΔM, resets, injected_τ, ΔY, injY,
             floors, floorτ)
        push!(rows, r)
        put(r)
        if checkpoint !== nothing && (c % every == 0 || c == nchunks)
            serialize(checkpoint * ".tmp",
                      (; c, t, steps, injected, resets, injected_τ, injY, floors, floorτ, rows,
                       u=Array(u)))
            mv(checkpoint * ".tmp", checkpoint; force=true)
        end
    end
    io === nothing || close(io)
    tot = MHD.conserved_totals(p.mhd, view(u, p.s[2]))
    ΔM, ΔY = tot[MHD.U_D] - M0, tot[MHD.U_DYE] - Y0
    return (; rows, M0, ΔM, Y0, ΔY, injY, floors, floorτ, injected, resets, injected_τ, steps,
            h=minimum_spacing(forest),
            nblocks=nblocks(p.gh.U), wall=time() - wall0)
end

function study_main(args)
    kw = Dict{Symbol,Any}()
    for a in args
        k, v = split(a, '='; limit=2)
        if k in ("N", "levels", "roots", "every", "order")
            kw[Symbol(k)] = parse(Int, v)
        elseif k in ("L", "t_end", "chunk", "cfl_gh", "γ0M", "γ2", "ρ_min", "ρ_c")
            kw[Symbol(k)] = parse(Float64, v)
        elseif k == "table"
            kw[:table] = String(v)
        elseif k == "δv"
            kw[:δv] = Tuple(parse.(Float64, split(v, ',')))
        elseif k == "limiter"
            kw[:limiter] = Symbol(v)
        elseif k in ("out", "checkpoint")
            kw[Symbol(k)] = String(v)
        else
            error("tov_coupled_study.jl: unknown key `$k`")
        end
    end
    r = study(; kw...)
    first_row, last_row = r.rows[1], r.rows[end]
    println("h = $(r.h), $(r.nblocks) blocks, $(r.steps) steps, wall $(round(r.wall; digits=1)) s")
    println("ρ_c: $(first_row.ρ_c) → $(last_row.ρ_c) (drift $((last_row.ρ_c - first_row.ρ_c) / first_row.ρ_c))")
    println("‖ℋ‖₂ inside: $(first_row.ham_in) → $(last_row.ham_in); outside: $(first_row.ham_out) → $(last_row.ham_out)")
    println("rest mass $(r.M0), change $(r.ΔM), injected $(r.injected) (the whole run), " *
            "difference $(r.ΔM - r.injected); the reset wrote $(r.resets) cells and " *
            "injected $(r.injected_τ) of τ")
    println("D Yₑ $(r.Y0), change $(r.ΔY), injected $(r.injY), difference $(r.ΔY - r.injY); " *
            "entropy floors outside the atmosphere $(r.floors), their τ $(r.floorτ)")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    study_main(ARGS)
end
