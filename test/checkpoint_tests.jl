# Checkpoints and restarts, serially ("Running distributed", "Checkpoints" in `CODE.md`).
# Their restart at another rank count is `mpi_tests.jl`'s.

using TreeIOHDF5: TreeIOHDF5

isdefined(@__MODULE__, :TEST_TIER) || include("tiers.jl")

const CK_ROTOR = (; N=8, chunk=1 // 80, limiter=:mc, maxlevel=1, refine_tol=0.08,
                  coarsen_tol=0.02, buffer=4)

@testset "A run restarted from its checkpoint is the run that never stopped, bit for bit" begin
    # Guards what a checkpoint must hold and what it must not: the five evolved sets on
    # the forest they were written on, and `evolve!`'s run state (`totals0`, the chunk
    # count, the steps, the regrids, every chunk's tally), but not `B`, `P`, `W`, the
    # scratch sets or the step cache, which a restart rebuilds. A restart that missed
    # anything — a set, an accumulator, the regrid at the start of its first chunk, the
    # mesh the state lives on — would continue into a different run, or one whose result
    # reports another history. On the adaptive rotor, which regrids at every chunk
    # boundary, so that the restart's first act is a regrid of the loaded state.
    dir = mktempdir()
    path = joinpath(dir, "rotor.h5")
    rotor = magnetic_rotor_case(Float64; roots=4)
    whole = evolve!(rotor, Val(2); CK_ROTOR..., t_end=3 // 80)
    part = evolve!(rotor, Val(2); CK_ROTOR..., t_end=1 // 80, checkpoint=path)
    rest = evolve!(rotor, Val(2); CK_ROTOR..., t_end=3 // 80, restart=path)
    @test whole.nregrids > 0
    @test rest.u == whole.u
    @test rest.t == whole.t && rest.nsteps == whole.nsteps
    @test rest.nregrids == whole.nregrids && rest.passes == whole.passes
    @test rest.totals0 == whole.totals0
    @test rest.chunks == whole.chunks
    @test rest.nblocks == whole.nblocks
    @test first(rest.p.sets).forest.leaves == first(whole.p.sets).forest.leaves
    # Every few chunks, and at the end: the file holds the last chunk boundary written.
    evolve!(rotor, Val(2); CK_ROTOR..., t_end=3 // 80, checkpoint=path, checkpoint_every=2)
    @test load_run(path).run.chunk == 3
end

@testset "A restart with another recipe, or from another application's file, is refused" begin
    # Guards the recipe: a restart whose parameters differ from the checkpoint's would be
    # another experiment that looks like the old one, and a file another application wrote
    # holds another run state. Both are refused, naming the reason; a restart with nothing
    # left to run is refused too.
    dir = mktempdir()
    path = joinpath(dir, "wave.h5")
    wave = alfven_wave_case(Float64; roots=4)
    kw = (; N=10, chunk=1 // 32, limiter=:mc)
    evolve!(wave, Val(1); kw..., t_end=1 // 32, checkpoint=path)
    @test_throws "cfl: 0.4 in the checkpoint, 0.3 now" evolve!(wave, Val(1); kw...,
                                                               t_end=1 // 16, cfl=0.3,
                                                               restart=path)
    @test_throws "limiter" evolve!(wave, Val(1); kw..., limiter=:ppm, t_end=1 // 16,
                                   restart=path)
    @test_throws "nothing is left to run" evolve!(wave, Val(1); kw..., t_end=1 // 32,
                                                  restart=path)
    # The restart's own forest and a problem built on another are not the same mesh.
    ck = load_run(path)
    r = evolve!(wave, Val(1); kw..., t_end=1 // 32)
    @test_throws ArgumentError restore_run!(similar(r.u), r.p, ck)
    # Another application's checkpoint of the same forest.
    other = joinpath(dir, "other.h5")
    TreeIOHDF5.save_checkpoint(other, first(r.p.sets).forest;
                               fieldsets=("U" => r.p.sets[1],),
                               application="SomethingElse.jl" => 1, data=(;))
    @test_throws "not of TreeGRRMHD.jl" load_run(other)
    @test_throws ArgumentError check_recipe((; a=1, b=2), (; a=1))
    @test check_recipe(checkpoint_recipe(; a=1, b=:x, c=(1, 2.5)),
                       checkpoint_recipe(; c=(1, 2.5), b=:x, a=1)) === nothing
end
