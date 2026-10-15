# The field loop across a coarse-fine face (step 8; "Validation ladder" 2b in
# `CODE.md`): its magnetic energy and its shape against a uniform-fine control.
#
#     julia --project=. bin/field_loop_refined.jl [t_end=4]
#
# Beckwith & Stone's loop (`field_loop_case`, `vᶻ = 0`, MC) on the periodic
# `[−1, 1] × [−1/2, 1/2]`, 4 × 2 roots of N = 8 (32 × 16 coarse cells):
#
#   * uniform coarse (32 × 16) and uniform fine (64 × 32), the controls;
#   * static two-level: the roots with `x ≥ 0` refined, so the face `x = 0`
#     runs through the loop's centre and the loop drifts across it;
#   * adaptive: one level more where Löhner on `ρ`, `|B|`, `|E|` fires
#     (`refine_tol = 0.08`, `coarsen_tol = 0.02`), regridded every `1/4` —
#     which, with its buffer and 2:1 balance, refines every root of so small
#     a box and reproduces the uniform-fine run (measured in step 8).
#
# Printed: the magnetic energy `Σ hᴰ |𝔅|²/2` at `t_end` over its initial value,
# the L1 error of `A_z` against the exact advected `A_z(x − vt)` over the box,
# `max |∇·𝔅| h/max |𝔅|` at the end, and the step and block counts.

using Printf
using TreeAMR
using TreeGRRMHD
const TG = TreeGRRMHD

t_end = let a = findfirst(s -> startswith(s, "t_end="), ARGS)
    a === nothing ? 4 : parse(Float64, ARGS[a][7:end])
end

refine_right!(forest) =
    (refine!(forest, [k for k in copy(forest.leaves) if block_origin(forest, k)[1] ≥ 0]);
     balance!(forest); forest)

function magnetic_energy(p, u, t)
    update_fields!(p, u, t)
    forest = p.f.U.forest
    D = 2
    return sum(1:3) do d
        mesh_mapreduce(x -> x^2 / 2, +, 0.0, p.f.B[d]; vars=1,
                       weight=k -> spacing(Float64, forest, k)^D)
    end
end

case = field_loop_case(Float64)
println("field loop, t_end = $t_end")
for (label, kw) in (("uniform coarse 32×16", (;)), ("uniform fine 64×32", (; roots=(8, 4))),
                    ("two-level, x ≥ 0 fine", (; refine=refine_right!)),
                    ("adaptive, one level", (; maxlevel=1, refine_tol=0.08, coarsen_tol=0.02,
                                             buffer=4, chunk=1 // 4)))
    t0 = time()
    r = evolve!(case, Val(2); N=8, t_end=t_end, limiter=:mc, kw...)
    u0 = exact_state(r.p, case, 0.0)
    e0 = magnetic_energy(r.p, u0, 0.0)
    e1 = magnetic_energy(r.p, r.u, r.t)
    ex = exact_state(r.p, case, r.t)
    err = l1_errors(r.p, r.u, ex).A[3]
    @printf("  %-24s energy %.5f of initial; L1(A_z) %.4e; ∇·𝔅 %.1e; %d steps, %d blocks, %.0f s\n",
            label, e1 / e0, err, max_div_B(r.p, r.u, r.t), r.nsteps, r.nblocks, time() - t0)
end
