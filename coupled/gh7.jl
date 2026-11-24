# GH-7 workaround (GH-7, `CODE.md`, "Requests still waiting"; allowed, decided by Erik 2026-11-10): the verbs TreeGH
# does not build yet — the regrid branch's `flags`, `regrid_pairs` and `rebuild`, and the
# constraint norms of a segment — composed from TreeGH's exports. When TreeGH has them,
# each line below becomes a call to `GH.flags`, `GH.regrid_pairs`, `GH.rebuild` and
# `GH.constraint_norms`, and this file goes. Every definition reads only `GHProblem`'s
# fields `U`, `schedule` and `case`, as `evolve!`'s `regrid_mesh` does.

# The order `q` a problem was built with.
gh_order(::GH.GHProblem{T,G,q}) where {T,G,q} = q

# GH-7 workaround: `TreeGH.flags(p, u, t)`, the indicator on a segment.
gh_flags(p::GH.GHProblem, u, t; buffer=0, travel=zero(eltype(u))) =
    GH.gh_indicator!(p, u, t; buffer, travel)

# GH-7 workaround: `TreeGH.regrid_pairs(p, t)`, the state set with its schedule and its
# Dirichlet hook at `t`. `Hsrc` and `diag` are not carried: `gh_rebuild` makes them anew.
gh_regrid_pairs(p::GH.GHProblem, t) = (p.U => (p.schedule, GH.dirichlet(p.case, t)),)

# GH-7 workaround: `TreeGH.rebuild(p, t)`, the problem on the regridded mesh: a new
# schedule with the same operators, the gauge source resampled, the monitors' set anew.
gh_rebuild(p::GH.GHProblem, t) =
    GH.GHProblem(p.U, GhostSchedule(p.U, p.schedule.operators), p.case; q=gh_order(p),
                 t=eltype(p.U.work)(t))

# GH-7 workaround: `TreeGH.constraint_norms(p, u, t; matter, mask)`, the gauge constraints
# and the matter-corrected ADM constraints `ℋ − 16πρ`, `ℳ_i − 8πS_i` of a segment, as
# `constraint_norms(p)` gives them (`gauge_l2`, `ham_l2`, `mom_l2` and the `linf`s).
function gh_constraint_norms(p::GH.GHProblem, u, t; matter=nothing,
                             mask=GH.interior_mask(p.interior, eltype(u)(t)))
    GH.gh_constraint!(p, u, t; mask)
    GH.adm_constraint!(p, u, t; mask, matter)
    return GH.constraint_norms(p)
end
