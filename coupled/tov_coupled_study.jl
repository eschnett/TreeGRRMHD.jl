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
# `checkpoint` (a file, or a directory holding it as `state.h5`, as SimLaunch's restarts
# give one, saved every `every` chunks, default 20, and resumed from if it exists; TreeIOHDF5's
# checkpoint through `MHD.save_run` with TreeGH's set beside the matter's, readable at any
# rank count), `mpi` (`false`; `true` over `MPI.COMM_WORLD` under `mpiexec`, rank 0
# writing; `bin/mpi.jl`, "Running distributed" in `CODE.md`), and *(added in step 13)* `table` (`dd2`, `sfho`: the star on that table in cold
# β-equilibrium, read from `ENTROPYEOS_TABLE_DIR`) with `ρ_c` (its central density,
# geometric). The series then gains four columns: `ΔDYₑ injected_DYₑ s_floors s_floor_τ`,
# the change of the total `D Yₑ`, the reset's injection into it, and its entropy floors
# outside the atmosphere with their `τ`, since `t = 0`. *(Added in step 20)* `s` (the hot
# star's entropy per baryon, `tabulated_tov_case`; none, the cold star) and `neutrinos`
# (`false`; `true`: `RJS96Rates` on the table's composition, `M0Leakage` about the centre
# with `r_core = 1`, whose stress-energy does not reach TreeGH); the series then gains two
# columns, the neutrinos' total energy `Σ𝓔` and lepton number's change net of the reset,
# `ΔDYₑ + N_e − N_ē − injected_DYₑ`.

include(joinpath(@__DIR__, "tov_coupled.jl"))
include(joinpath(@__DIR__, "..", "bin", "tables.jl"))
include(joinpath(@__DIR__, "..", "bin", "mpi.jl"))
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
               out=nothing, checkpoint=nothing, every=20, table=nothing, ρ_c=nothing,
               s=nothing, neutrinos=false, comm=nothing)
    nu = nothing
    case = if table === nothing
        (s === nothing && !neutrinos) ||
            error("tov_coupled_study.jl: `s` and `neutrinos` need a table")
        nothing
    else
        d = load_table(table)
        if neutrinos
            E = MHD.EntropyEOS
            cv = E.CompositionView(E.CompositionTable(d.table, d.eos))
            units = MHD.EOSUnits(E.EOSTableView(d.eos); density=MHD.MSUN_DENSITY_CGS)
            nu = MHD.M0Leakage(MHD.RJS96Rates(T, units, d.eos, cv); composition=cv,
                               center=(0.0, 0.0, 0.0), r_core=1.0)
        end
        MHD.tabulated_tov_case(T; d.table, d.eos, ρ_c, ρ_min, s, L, roots)
    end
    # A directory (SimLaunch's `ck/`, `mutable_restarts`) holds the checkpoint as a file;
    # whether to resume is rank 0's view of it, the rank that reads its index.
    checkpoint !== nothing && isdir(checkpoint) && (checkpoint = joinpath(checkpoint, "state.h5"))
    resume = checkpoint !== nothing && MHD.rank_reduce((a, b) -> a, comm, isfile(checkpoint))
    recipe = MHD.checkpoint_recipe(; T, N, levels, L, roots, chunk=T(chunk), limiter,
                                   cfl_gh=T(cfl_gh), γ0M, γ2, δv, order, ρ_min, table, ρ_c)
    # The neutrinos' keys only where they are, so that an older checkpoint keeps its recipe.
    s === nothing || (recipe = merge(recipe, MHD.checkpoint_recipe(; s)))
    nu === nothing ||
        (recipe = merge(recipe, MHD.checkpoint_recipe(; neutrinos=MHD.neutrino_recipe(nu))))
    ck = resume ? MHD.load_run(checkpoint; comm) : nothing
    ck === nothing || MHD.check_recipe(ck.recipe, recipe)
    (; p, u, forest) = problems(T; N, levels, L, roots, limiter, γ0M, γ2, δv, order, ρ_min,
                                case, comm, forest=ck === nothing ? nothing : ck.forest,
                                neutrinos=nu)
    root = MHD.isroot(comm)
    tot0 = MHD.conserved_totals(p.mhd, view(u, p.s[2]))
    M0, Y0 = tot0[MHD.U_D], tot0[MHD.U_DYE]
    cfl_mhd = T(MHD.default_cfl(limiter))
    t_end, chunk = T(t_end), T(chunk)
    nchunks = MHD.chunk_count(t_end, chunk)
    t, integ, steps, injected, c0 = zero(T), nothing, 0, zero(T), 0
    resets, injected_τ = 0, zero(T)
    injY, floors, floorτ = zero(T), 0, zero(T)
    rows = Any[]
    if ck !== nothing
        # The matter's five sets and TreeGH's, on the checkpoint's forest.
        MHD.restore_run!(view(u, p.s[2]), p.mhd, ck)
        copyto!(view(u, p.s[1]), ck.states["TreeGH.U"])
        s = ck.run
        c0, t, steps, injected = s.c, T(s.t), s.steps, T(s.injected)
        resets, injected_τ = s.resets, T(s.injected_τ)
        injY, floors, floorτ = T(s.injY), s.floors, T(s.floorτ)
        rows = Any[NamedTuple{keys(s.rows)}(map(v -> v[i], values(s.rows)))
                   for i in eachindex(s.rows.t)]
        root && @info "resumed from $checkpoint at chunk $c0, t = $t"
    else
        push!(rows, (; t, study_norms(p, u, t)..., injected=zero(T), steps=0,
                     ΔM=zero(T), resets=0, injected_τ=zero(T), ΔY=zero(T), injY=zero(T),
                     floors=0, floorτ=zero(T), nuE=nu === nothing ? T(NaN) : zero(T),
                     nuY=nu === nothing ? T(NaN) : zero(T)))
    end
    # The series is rewritten from the rows the checkpoint holds, so that a resumed run's
    # file has no row twice (a pre-emption after a row and before its checkpoint).
    io = out === nothing ? nothing : root ? open(out, "w") : devnull
    put(r) = io === nothing ? nothing :
             (println(io, join((r.t, r.ρ_c, r.ham_in, r.mom_in, r.ham_out, r.mom_out,
                                r.injected, r.steps, get(r, :ΔM, NaN), get(r, :resets, -1),
                                get(r, :injected_τ, NaN), get(r, :ΔY, NaN), get(r, :injY, NaN),
                                get(r, :floors, -1), get(r, :floorτ, NaN), get(r, :nuE, NaN),
                                get(r, :nuY, NaN)), "  ")); flush(io))
    foreach(put, rows)
    wall0 = time()
    for c in (c0 + 1):nchunks
        t1 = c == nchunks ? t_end : c * chunk
        MHD.clear_tally!(p.mhd)
        # The neutrinos' first implicit stage reads `P`: from the state at every chunk
        # boundary, so that a resumed run is the one that never stopped (step 20).
        MHD.resume_neutrinos!(p.mhd, view(u, p.s[2]), t; metric=metric_at(p, u, t))
        nsteps = step_count(p, u, t, t1, forest, cfl_gh, cfl_mhd)
        prob = IRK.IMEXProblem(f_exp!, solve_imp!, u, (t, t1), p)
        integ = IRK.init(prob, IRK.IMEXSSP3433(); dt=(t1 - t) / nsteps,
                         stage_limiter=stage_limit!, step_limiter=step_limit!,
                         partition=u isa Array ? :even : nothing, alias_u0=true,
                         reuse=integ)
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
        nuE, nuY = if nu === nothing
            T(NaN), T(NaN)
        else
            nt = MHD.neutrino_totals(p.mhd, view(u, p.s[2]))
            nt[2] + nt[4] + nt[6], ΔY + nt[1] - nt[3] - injY
        end
        r = (; t, study_norms(p, u, t)..., injected, steps, ΔM, resets, injected_τ, ΔY, injY,
             floors, floorτ, nuE, nuY)
        push!(rows, r)
        put(r)
        if checkpoint !== nothing && (c % every == 0 || c == nchunks)
            # The rows as columns (plain data); a row from before a resume has them all.
            cols = NamedTuple{keys(first(rows))}(Tuple([getfield(r, k) for r in rows]
                                                       for k in keys(first(rows))))
            MHD.save_run(checkpoint, p.mhd, view(u, p.s[2]); recipe,
                         extra=("TreeGH.U" => (p.gh.U, view(u, p.s[1])),),
                         run=(; c, t, steps, injected, resets, injected_τ, injY, floors,
                              floorτ, rows=cols))
        end
    end
    io === nothing || io === devnull || close(io)
    tot = MHD.conserved_totals(p.mhd, view(u, p.s[2]))
    ΔM, ΔY = tot[MHD.U_D] - M0, tot[MHD.U_DYE] - Y0
    return (; rows, M0, ΔM, Y0, ΔY, injY, floors, floorτ, injected, resets, injected_τ, steps,
            h=minimum_spacing(forest),
            nblocks=nleaves(forest), wall=time() - wall0)
end

function study_main(args)
    kw = Dict{Symbol,Any}()
    mpi = false
    for a in args
        k, v = split(a, '='; limit=2)
        if k in ("N", "levels", "roots", "every", "order")
            kw[Symbol(k)] = parse(Int, v)
        elseif k in ("L", "t_end", "chunk", "cfl_gh", "γ0M", "γ2", "ρ_min", "ρ_c", "s")
            kw[Symbol(k)] = parse(Float64, v)
        elseif k == "neutrinos"
            kw[:neutrinos] = parse(Bool, v)
        elseif k == "table"
            kw[:table] = String(v)
        elseif k == "δv"
            kw[:δv] = Tuple(parse.(Float64, split(v, ',')))
        elseif k == "limiter"
            kw[:limiter] = Symbol(v)
        elseif k in ("out", "checkpoint")
            kw[Symbol(k)] = String(v)
        elseif k == "mpi"
            mpi = parse(Bool, v)
        else
            error("tov_coupled_study.jl: unknown key `$k`")
        end
    end
    r, root = withcomm(mpi) do comm
        (study(; kw..., comm), MHD.isroot(comm))
    end
    root || return nothing
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
