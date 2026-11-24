# Step 11: TreeGH's CFL factor under the explicit part of SSP3(4,3,3), the coupled run's
# integrator (the plan's step 11, "Decides"; "Time integration" in `CODE.md`). TreeGH's
# own `cfl = 1/4` was measured with RK4, whose stability region reaches further along the
# imaginary axis. TreeGH runs alone, its implicit part a copy (as in the coupled run's
# `solve_imp!`), at `cfl` on a grid, on two cases: TreeGH's gauge wave (periodic,
# `A = 1/20`, `d = 1`, `q = 4`, `ε_KO = 1/2`, `γ0 = 1`, `γ2 = −1/2`, `h = 1/16`) against its
# exact solution; and the TOV star's spacetime with this package's matter held fixed (the
# static-matter run, `h = 0.5` on the star's box), against its initial state. A run is
# stable if its error stays finite and within `growth` times the error at the smallest
# `cfl` (the truncation error grows with `cfl` too, by `(cfl/cfl₀)³` for SSP3); the result
# is the largest stable `cfl` of the grid, for each tableau (RK4 as the reference).
#
#     julia --project=coupled -t T coupled/gh_cfl.jl [case=wave|star] [t_end=T]

include(joinpath(@__DIR__, "tov_coupled.jl"))
using Printf

copy_imp!(U, u★, γΔt, p, t) = copyto!(U, u★)
gh_f!(du, u, p, t) = GH.gh_rhs!(du, u, p.gh, t; matter=p.matter)
gh_lim!(u, _, p, t) = GH.stage_limit!(u, p.gh, t)

function gh_run(gh, matter, u0, t_end, dt, tableau)
    u = copy(u0)
    p = (; gh, matter)
    integ = IRK.init(IRK.IMEXProblem(gh_f!, copy_imp!, u, (zero(t_end), t_end), p), tableau;
                     dt, stage_limiter=gh_lim!, step_limiter=gh_lim!, alias_u0=true)
    IRK.solve!(integ)
    return u
end

function wave_setup(T)
    case = GH.gauge_wave_case(T; A=T(1 // 20), d=one(T), ε_KO=T(1 // 2), γ0=one(T),
                              γ2=T(-1 // 2))
    forest = GH.gh_forest(T, case; N=16, roots=1)
    U = FieldSet{T}(forest, 20; G=3, centering=vertexcentered(3))
    gh = GH.GHProblem(U, GhostSchedule(U, Operators(; prolongation=6, restriction=6)), case;
                      q=4)
    u0 = statevector(U)
    GH.initial_state!(u0, gh, zero(T))
    exact(t) = (E = FieldSet{T}(forest, 20; G=3, centering=vertexcentered(3));
                GH.fill_exact!(E, case, t); e = statevector(E); gather!(e, E); e)
    return (; gh, matter=nothing, u0, forest, reference=exact)
end

function star_setup(T)
    (; p, u, star, forest) = problems(T; N=12, levels=1, L=24, roots=4, limiter=:ppm)
    MHD.prepare!(p.mhd, view(u, p.s[2]), zero(T); metric=star.provider)
    u0 = copy(view(u, p.s[1]))
    return (; gh=p.gh, matter=MHD.ADMMatter(p.mhd), u0, forest, reference=t -> u0)
end

function scan(T=Float64; case="wave", t_end=nothing, cfls=0.1:0.05:1.5, growth=100)
    s = case == "wave" ? wave_setup(T) : star_setup(T)
    t_end = T(something(t_end, case == "wave" ? 20 : 100))
    λ = GH.max_speed(s.gh, s.u0, zero(T))
    h = minimum_spacing(T, s.forest)
    ref = s.reference(t_end)
    for (name, tab) in (("SSP3(4,3,3)", IRK.IMEXSSP3433()), ("RK4", IRK.RK4()))
        e0, last_stable = nothing, nothing
        for cfl in cfls
            nsteps = MHD.chunk_count(t_end * λ, T(cfl) * h)
            t0 = time()
            u = gh_run(s.gh, s.matter, s.u0, t_end, t_end / nsteps, tab)
            e = sqrt(sum(abs2, u - ref) / length(u))
            e0 === nothing && (e0 = e)
            stable = isfinite(e) && e ≤ growth * e0 * (cfl / first(cfls))^3
            @printf("%-5s %-12s cfl %.2f  steps %6d  RMS error %.4e  %s  (%.1f s)\n", case,
                    name, cfl, nsteps, e, stable ? "stable" : "UNSTABLE", time() - t0)
            flush(stdout)
            stable || break
            last_stable = cfl
        end
        @printf("%-5s %-12s largest stable cfl on the grid: %s\n", case, name, last_stable)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    kw = Dict{Symbol,Any}()
    for a in ARGS
        k, v = split(a, '='; limit=2)
        k == "case" ? (kw[:case] = String(v)) :
        k == "t_end" ? (kw[:t_end] = parse(Float64, v)) :
        error("gh_cfl.jl: unknown key `$k` (case, t_end)")
    end
    scan(; kw...)
end
