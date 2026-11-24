# The TreeGH items of `CODE.md`'s "Upstream prerequisites", detected from the release this
# environment resolved, by name and method, never by version. The package's own suite only
# names the open ones; this suite, which loads TreeGH, says which have landed.

import Pkg

function gh_report(id, landed::Bool, what)
    @info landed ? "$id landed: $what" : "$id waiting: $what"
    return landed
end

@testset "GH-1, GH-3, GH-4: the contract's verbs and adapters are in this TreeGH" begin
    # Guards the script against a TreeGH without the contract: GH-1's `ADMMetric`, GH-3's
    # `matter` keyword on the right-hand side and the ADM monitor, GH-4's public verbs on a
    # segment. All three have landed (TreeGH's `main`, 61bcb75, 79ad132, 0cd5f00), so each
    # is asserted, not only reported.
    P = GH.GHProblem
    @test gh_report("GH-1", isdefined(GH, :ADMMetric) &&
                                hasmethod(GH.ADMMetric, Tuple{P,Any,Any}),
                    "ADMMetric(p, u, t)")
    @test gh_report("GH-3", hasmethod(GH.rhs!, Tuple{Any,P,Any}, (:matter,)) &&
                                hasmethod(GH.adm_constraint!, Tuple{P,Any,Any}, (:matter,)),
                    "rhs! and adm_constraint! take `matter`")
    verbs = (:prepare!, :rhs!, :stage_limit!, :step_limit!, :initial_state!, :evolved_sets)
    @test gh_report("GH-4", all(v -> Base.ispublic(GH, v), verbs) &&
                                hasmethod(GH.max_speed, Tuple{P,Any,Any}),
                    "the public verbs and max_speed(p, u, t)")
end

@testset "GH-2 and GH-7: reported, with their workarounds in place" begin
    # Guards nothing the suite needs — each has a workaround — but says whether it can go:
    # GH-2, TreeGH in a registry this depot reaches (then `[sources]` loses its TreeGH
    # line); GH-7, TreeGH's own regrid verbs and segment norms (then `gh7.jl` goes).
    uuid = Base.PkgId(GH).uuid
    registered = any(r -> haskey(r, uuid), Pkg.Registry.reachable_registries())
    gh_report("GH-2", registered,
              registered ? "TreeGH is registered" : "TreeGH is unregistered; [sources]")
    gh7 = all(v -> isdefined(GH, v), (:flags, :regrid_pairs, :rebuild)) &&
          hasmethod(GH.constraint_norms, Tuple{GH.GHProblem,Any,Any})
    gh_report("GH-7", gh7, gh7 ? "TreeGH has the regrid verbs" :
                               "no flags, regrid_pairs, rebuild; coupled/gh7.jl")
    @test true
end
