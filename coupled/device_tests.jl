# The coupled run on a device ("Running distributed", "Threads and devices under MPI" in
# `CODE.md`): `problems` with a `backend` puts TreeGH's set, the matter's sets, `Mv` and
# the adapters' sets there, and the script's own host loops (the fill's key) take the
# device path. Unset, the CPU stands in for the device, as in the package's
# `test/device_tests.jl`, and the claims are exact; on a device
# (`TREEGRRMHD_TEST_BACKEND=cuda` in an environment that has CUDA), the run takes the
# host run's steps and its record agrees to the type's rounding. The device branch's
# tolerances are not measured yet: no coupled run has been made on a device (`HISTORY.md`,
# "running distributed: built"); the first one sets them.

using KernelAbstractions: CPU, get_backend

const CD_NAME = lowercase(get(ENV, "TREEGRRMHD_TEST_BACKEND", ""))
const CD_DEVICE = if CD_NAME == "cuda"
    Core.eval(Main, :(using CUDA))
    Base.invokelatest(() -> Core.eval(Main, :(CUDA.functional() ? CUDABackend() : CPU())))
elseif isempty(CD_NAME)
    CPU()
else
    error("TREEGRRMHD_TEST_BACKEND must be \"cuda\" here, got \"$CD_NAME\"")
end

@testset "The coupled star on $(nameof(typeof(CD_DEVICE))) is the host run" begin
    # Guards the coupled script's `backend`: a set left on the host, a host loop over a
    # device array, or the integrator's partition chosen for the wrong array type would fail
    # or diverge. Two chunks to `t = 1` at `N = 10`, as "The coupled script runs".
    run(backend) = begin
        rec = []
        r = Base.invokelatest(main; N=10, levels=0, L=CT_L, roots=CT_ROOTS, t_end=1,
                              chunk=1 // 2, limiter=:mc, backend,
                              record=(t, ρ_c, c) -> push!(rec, (; t, ρ_c, ham=c.ham_l2)))
        (; r, rec)
    end
    host, dev = run(CPU()), run(CD_DEVICE)
    @test get_backend(dev.r.u) == CD_DEVICE
    @test dev.r.t == host.r.t && length(dev.rec) == length(host.rec)
    if CD_DEVICE isa CPU
        @test isequal(dev.r.u, host.r.u) && dev.rec == host.rec
    else
        @test all(isapprox(a.ρ_c, b.ρ_c; rtol=1e-12) && isapprox(a.ham, b.ham; rtol=1e-8)
                  for (a, b) in zip(dev.rec, host.rec))
        @test isapprox(Array(dev.r.u), host.r.u; rtol=1e-10)
    end
end
