# The workload behind the thread-count test (step 14), TreeHydro's.
#
# Run as a standalone script —
#
#     julia -t N --project=. test/thread_workload.jl
#
# — it prints a digest of everything three short runs produce: per chunk the step count,
# `Δt`, the stage solve's and the reset's tally and the block count, and at the end the
# state vector, the conserved totals and `max |∇·𝔅|`. Two runs at different thread counts
# must print the same lines, character for character. That is the property the code has
# ("Precision and devices" in `CODE.md`): every kernel runs through TreeAMR's `map_blocks!`
# and writes its own points, every reduction is a `mesh_mapreduce` combined on the host in
# block order, the integrator's stage arithmetic is elementwise however
# IMEXRungeKutta's `:even` partition cuts the state, and nothing in this package threads a
# loop of its own.
#
# It drives the package's own `evolve!` and may use nothing outside `Base`, TreeAMR and the
# package, because it has to run from the package environment as well as from `test/`.
# Hence `hash` rather than `sha256`, and `repr`, which round-trips a `Float64` exactly, so
# that a difference in the last bit is a difference in the text.

using TreeAMR
using TreeGRRMHD

thread_digest(u::AbstractVector) = string(hash(Array(u)); base=16, pad=16)

"""
The magnetized atmosphere tube in `D = 1`, the adaptive rotor in `D = 2`, and the
magnetized TOV star in `D = 3`, reduced to printed lines.

The three cover everything that threads. The tube is the one where the atmosphere reset
fires at every stage and the magnetization cutoff's stage solve takes its stiff branches,
under the step cache; the rotor adapts to its initial data and regrids every chunk, with
the Dirichlet hooks on every face; and the star is three-dimensional on two static levels,
on `Analytic(TOVMetric)`, with a resistive interior, the cutoff and an atmosphere.
"""
function thread_digests()
    lines = String[]
    function record(tag, r)
        for (c, k) in enumerate(r.chunks)
            push!(lines, string(tag, " chunk ", c, " t=", repr(k.t), " nsteps=", k.nsteps,
                                " dt=", repr(k.dt), " blocks=", k.nblocks, " tally=",
                                repr(k.tally)))
        end
        push!(lines, string(tag, " u=", thread_digest(r.u), " nregrids=", r.nregrids,
                            " passes=", r.passes, " totals=",
                            repr(conserved_totals(r.p, r.u)), " divB=",
                            repr(max_div_B(r.p, r.u, r.t))))
        return nothing
    end

    tube = atmosphere_tube_case(Float64; By=1e-3,
                                model=MagnetizationCutoff(Constant(1e3), 10), roots=20)
    record("tube", evolve!(tube, Val(1); N=10, t_end=1 // 5, chunk=1 // 10, limiter=:mc))

    rotor = magnetic_rotor_case(Float64; roots=4)
    record("rotor", evolve!(rotor, Val(2); N=8, t_end=1 // 20, chunk=1 // 40, limiter=:mc,
                            maxlevel=1, refine_tol=0.08, coarsen_tol=0.02, buffer=4))

    star = magnetized_tov_case(Float64; σ₀=1e6, μ_max=10, L=12, roots=3)
    record("star", evolve!(star, Val(3); N=8, t_end=1, chunk=1 // 2, limiter=:mc,
                           refine=forest -> tov_boxes!(forest, star; levels=1)))
    return lines
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && foreach(println, thread_digests())
