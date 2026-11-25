# Running on a device (step 14): every kind of run the suite measures runs on a device and
# answers as the host run does at the same type.
#
# The mesh decides where the work runs from where the storage is, so `evolve!` takes a
# `backend` beside its `T`, and the integrator follows the state: a device state takes
# IMEXRungeKutta's broadcast path. What only a device can check is that a run there takes
# the same steps, builds the same mesh and takes the same branches of the stage solve as the
# host run; a device that did not would still run. That needs a device package, which is
# deliberately not a dependency of this package or of TreeAMR, so this file also runs by
# itself, in an environment that has one (SimLaunch's `env = "test+cuda"`, `jobs/`):
#
#     TREEGRRMHD_TEST_BACKEND=cuda julia --project=<test+cuda> test/device_tests.jl
#
# Unset, inside the suite, the CPU stands in for the device, which is what CI does: the
# claims are then trivially exact, and what is exercised is the plumbing — the backend
# reaching every field set and every array a kernel reads, and the state coming back on the
# backend it was asked for. See "Precision and devices" in `CODE.md`.
#
# Under MPI (`TREEGRRMHD_TEST_DEVICE_MPI=1`, standalone, launched by `mpiexec`; added with
# "Running distributed" in `CODE.md`) every run is distributed over `MPI.COMM_WORLD`, each
# rank on its own device — the one its place on its node picks — and the host runs it is
# compared with are distributed alike: `jobs/device-mpi.toml`, two ranks of one H200 each.

const DEVICE_STANDALONE = !isdefined(@__MODULE__, :TEST_TIER)
if DEVICE_STANDALONE
    using Test
    using TreeAMR
    using TreeGRRMHD
    include(joinpath(@__DIR__, "tiers.jl"))
end
isdefined(@__MODULE__, :DD2_PRESENT) || include(joinpath(@__DIR__, "tables.jl"))
using KernelAbstractions: CPU, Backend, get_backend

const DEVICE_NAME = lowercase(get(ENV, "TREEGRRMHD_TEST_BACKEND", ""))
const DEVICE_MPI = get(ENV, "TREEGRRMHD_TEST_DEVICE_MPI", "") == "1"
DEVICE_MPI && !DEVICE_STANDALONE &&
    error("TREEGRRMHD_TEST_DEVICE_MPI=1 runs this file by itself under mpiexec")
if DEVICE_MPI
    using MPI
    MPI.Init()
end
# Every run's communicator: `MPI.COMM_WORLD` under MPI, else the serial run.
const DEVICE_COMM = DEVICE_MPI ? MPI.COMM_WORLD : nothing

if DEVICE_NAME == "cuda"
    using CUDA
    CUDA.allowscalar(false)
    if DEVICE_MPI && CUDA.functional()
        # One rank per device, by the rank's place on its node, before anything is
        # allocated (`bin/mpi.jl`'s `select_device!`).
        let node = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED,
                                       MPI.Comm_rank(MPI.COMM_WORLD))
            CUDA.device!(MPI.Comm_rank(node) % length(CUDA.devices()))
        end
    end
elseif DEVICE_NAME == "metal"
    using Metal
    Metal.allowscalar(false)
elseif !isempty(DEVICE_NAME)
    error("TREEGRRMHD_TEST_BACKEND must be \"cuda\" or \"metal\", got \"$DEVICE_NAME\"")
end

# The device, or the CPU standing in for it.
const DEVICE = let
    device = if DEVICE_NAME == "cuda"
        CUDA.functional() ? CUDABackend() : nothing
    elseif DEVICE_NAME == "metal"
        Metal.functional() ? MetalBackend() : nothing
    end
    if device === nothing && !isempty(DEVICE_NAME)
        @info "TREEGRRMHD_TEST_BACKEND=$DEVICE_NAME is not functional here; the CPU " *
              "stands in for it"
    end
    device === nothing ? CPU() : device
end

const DEVICE_LABEL = string(nameof(typeof(DEVICE)))
const DEVICE_REAL = !(DEVICE isa CPU)

# `Float32` everywhere: the type every device runs and the only one Metal has. `Float64` as
# well on CUDA, the type the host suite makes its claims at. Not on the CPU stand-in, where
# a second type would double the suite's cost for a comparison that is exact by
# construction.
const DEVICE_TYPES = DEVICE_NAME == "cuda" && DEVICE_REAL ? (Float32, Float64) : (Float32,)

# Both recoveries on a device, the two-phase one first: it is the structure a GPU needs, and
# EntropyEOS records a CUDA fault in a kernel that called both levels ("The two-phase
# recovery" in `CODE.md`). On the stand-in the fused one only, which `two_phase_tests.jl`
# holds bit for bit to the other.
const DEVICE_RECOVERIES = DEVICE_REAL ? (true, false) : (false,)

# The runs, each `(name, case, Val(D), keywords)`: the Alfvén wave (every kernel of the
# scheme on a periodic box, against its exact solution), the magnetized atmosphere tube
# (the reset at every stage and the magnetization cutoff's stiff stage solve, under the step
# cache), the adaptive rotor (the adaptation, a regrid and the Dirichlet hooks on every
# face), the magnetized TOV star (three dimensions, two static levels, the
# `Analytic(TOVMetric)` provider, a resistive interior and an atmosphere), and where DD2 is
# found the tabulated star on the same mesh (the table's upload to the device, `eos_on`,
# and the Dirichlet hooks that carry it; added 2026-11-25, when the first device run of a
# table was refused at its first kernel). The suite's own sizes, made short; the wave on
# 40 cells, where its error is large enough for `Float32`'s rounding (below).
function device_runs(::Type{T}) where {T}
    star = magnetized_tov_case(T; σ₀=1e6, μ_max=10, L=12, roots=3)
    tab = if DD2_PRESENT
        d = dd2_table()
        dd2 = tabulated_tov_case(T; d.table, d.eos, ρ_c=9.4e-4, ρ_min=1e-10, L=12, roots=3)
        (("tabulated TOV star (DD2)", dd2, Val(3),
          (; N=8, t_end=1, chunk=1 // 2, limiter=:mc,
           refine=forest -> tov_boxes!(forest, dd2; levels=1))),)
    else
        ()
    end
    return (("Alfvén wave", alfven_wave_case(T; roots=4), Val(1),
             (; N=10, t_end=1 // 8, limiter=:mc)),
            ("magnetized atmosphere tube",
             atmosphere_tube_case(T; By=1e-3,
                                  model=MagnetizationCutoff(Constant(T(1e3)), T(10)),
                                  roots=20), Val(1),
             (; N=10, t_end=1 // 5, chunk=1 // 10, limiter=:mc)),
            ("adaptive rotor", magnetic_rotor_case(T; roots=4), Val(2),
             (; N=8, t_end=1 // 20, chunk=1 // 40, limiter=:mc, maxlevel=1,
              refine_tol=T(0.08), coarsen_tol=T(0.02), buffer=4)),
            ("magnetized TOV star", star, Val(3),
             (; N=8, t_end=1, chunk=1 // 2, limiter=:mc,
              refine=forest -> tov_boxes!(forest, star; levels=1))), tab...)
end

function device_run(::Type{T}, case, vD, kw, backend, two_phase) where {T}
    r = evolve!(T, case, vD; kw..., backend, two_phase, comm=DEVICE_COMM)
    u0 = exact_state(r.p, case, zero(T))
    return (; r, drift=conserved_totals(r.p, r.u) .- r.totals0,
            scales=max.(conserved_scales(r.p, r.u), conserved_scales(r.p, u0)),
            divB=max_div_B(r.p, r.u, r.t), ρ_max=TreeGRRMHD.central_density(r.p, r.u, r.t),
            l1=l1_errors(r.p, r.u, exact_state(r.p, case, r.t)))
end

# The agreement asked of a continuous result — an error norm, a density, `∇·𝔅`: `≈` at
# `sqrt(eps(T))`, TreeHydro's, because a device contracts `a*b + c` into an `fma` where the
# host may not and its `sin` and `exp` are not Julia's, and an error norm is a difference of
# order-one states whose relative noise is `eps(T)/err`. `max_div_B` is already relative to
# the field's scale, so it gets `eps(T)` absolute as well, where the host has exact zeros.
device_close(dev, host, ::Type{T}) where {T} = isapprox(dev, host; rtol=sqrt(eps(T)))
device_close_divB(dev, host, ::Type{T}) where {T} =
    isapprox(dev, host; rtol=sqrt(eps(T)), atol=eps(T))

# Two drifts of a conserved total agree to the roundoff bound the host suite puts on a
# drift, `8 eps(T) · scale · nsteps`, TreeHydro's; a total whose own scale is zero borrows
# the largest of the state's. The conserved totals are the seven of `U_D … U_DS`, as in the
# host suite's own (`md_conserved`): `𝔈`'s has the current as a source. The atmosphere's
# injection is held to the same yardstick: it is a sum of `c − cons` over the reset cells,
# rounding-dominated where the reset changes little.
function device_drift_bound(r, v, ::Type{T}) where {T}
    scale = r.scales[v] > 0 ? r.scales[v] : maximum(r.scales)
    return 8 * eps(T) * scale * r.r.nsteps
end
device_same_drift(dev, host, ::Type{T}) where {T} =
    all(v -> abs(dev.drift[v] - host.drift[v]) ≤ device_drift_bound(host, v, T),
        TreeGRRMHD.U_D:TreeGRRMHD.U_DS)

# A count of threshold decisions — the policy's and the reset's, each a comparison of a
# recovered density with the atmosphere's trigger — agrees to less than one block's share
# of the chunk's recoveries. What this file guards against is per block: a block a kernel or
# a reduction did not see, or saw twice, moves a count by about that share. A device's
# density differs from the host's in its last bits, so a cell within a few ulp of the
# trigger can go either way, and it then evolves differently; how many do is the case's,
# not `eps`'s. Measured in step 14: on an H200 at `Float64` the atmosphere tube's 17848
# policy hits a chunk differed by 2 and its 13380 resets by 1 (a share is 2200); at
# `Float32` the magnetized star's 1.9·10⁶ resets by 82 and 114 (a share is 1.4·10⁴), on
# Metal by 24 and 56.
device_close_count(dev, host, tally, nblocks) = abs(dev - host) < tally.recoveries / nblocks

# The policy's flag word, the OR of every reason a recovery was repaired, compared exactly,
# except at `Float32` its not-converged bit: EntropyEOS is validated at `Float64` only, and
# at `Float32` whether a solve converges to its tolerance can turn on the last bits (on an
# H200 one cell of the magnetized star did not where the host's did; step 14).
device_policy_mask(::Type{Float32}) = ~TreeGRRMHD.EntropyEOS.FLAG_POL_NOT_CONVERGED
device_policy_mask(::Type) = typemax(UInt32)

# The step cache's seeds: how many recoveries were cold is structural, but which of the rest
# were reused rather than warm is not, since a reuse needs the input bit for bit and a
# device's last bits are its own. So per site the cold count and the sum.
device_same_seeds(dev, host) =
    all(s -> dev[s].cold == host[s].cold &&
             dev[s].warm + dev[s].reused == host[s].warm + host[s].reused, keys(host))

function device_testsets()
    @testset "The recovery's default is the backend's: split on a device, fused on the CPU" begin
        # Guards the default resolved in the wrong place or from the wrong storage: a device
        # problem without a hand-off runs the fused recovery, 14–15% slower a step on an
        # H200, and a CPU problem with one pays 1–10% for nothing ("The two-phase recovery"
        # in `CODE.md`).
        case = alfven_wave_case(Float32; roots=4)
        for backend in (DEVICE_REAL ? (CPU(), DEVICE) : (CPU(),))
            forest = Forest{Float32}(case.roots; N=10, periodic=case.periodic,
                                     extents=case.extents)
            p = TreeGRRMHD.MHDProblem(mhd_fields(Float32, forest; backend), case;
                                      limiter=:mc)
            @test (p.handoff !== nothing) == !(backend isa CPU)
        end
    end
    @testset "A run on $DEVICE_LABEL reproduces the host run at $T, two-phase $tp" for
            T in DEVICE_TYPES, tp in DEVICE_RECOVERIES
        # Guards a device path that runs and answers differently: a kernel that reads a host
        # array (which the CPU forgives), a reduction that sees a different block, a hook or
        # a limiter that does not reach the device state, or a regrid that flags a different
        # cell and so builds a different mesh. Step counts, meshes, regrids and the stage
        # solve's branch counts are compared exactly; the states through their diagnostics,
        # to the tolerances above.
        for (name, case, vD, kw) in device_runs(T)
            host = device_run(T, case, vD, kw, CPU(), tp)
            dev = device_run(T, case, vD, kw, DEVICE, tp)
            @test get_backend(dev.r.u) == DEVICE
            @test (dev.r.nsteps, dev.r.nregrids, dev.r.passes) ==
                  (host.r.nsteps, host.r.nregrids, host.r.passes)
            @test [(c.nsteps, c.dt, c.nblocks) for c in dev.r.chunks] ==
                  [(c.nsteps, c.dt, c.nblocks) for c in host.r.chunks]
            for (d, h) in zip(dev.r.chunks, host.r.chunks)
                dt, ht = d.tally, h.tally
                exact(t) = (t.bits, t.flags, t.policy_flags & device_policy_mask(T),
                            t.recoveries, t.max_recoveries)
                @test exact(dt) == exact(ht)
                @test device_close_count(dt.policy, ht.policy, ht, h.nblocks)
                @test device_close_count(dt.resets, ht.resets, ht, h.nblocks)
                @test device_close_count(dt.step_resets, ht.step_resets, ht, h.nblocks)
                @test abs(dt.injected_D - ht.injected_D) ≤ device_drift_bound(host, 1, T)
                @test abs(dt.injected_τ - ht.injected_τ) ≤ device_drift_bound(host, 5, T)
                @test device_same_seeds(dt.seeds, ht.seeds)
            end
            @test device_same_drift(dev, host, T)
            @test device_close_divB(dev.divB, host.divB, T)
            @test device_close(dev.ρ_max, host.ρ_max, T)
            # The error norms only where there is an exact solution to measure against, and
            # of the components the wave carries — `S_y`, `S_z`, `𝔈ʸ`, `𝔈ᶻ`, `A_y`, `A_z` —
            # whose error is its truncation error; the others' is rounding, which a device
            # rounds its own way (the longitudinal `𝔈ˣ`'s moved by 5·10⁻³ of itself on
            # Metal). On 100 cells even these differed by 6·10⁻⁴ of themselves at
            # `Float32`, above `sqrt(eps)`: an error of 10⁻⁴ of the state carries a relative
            # noise of `eps/10⁻⁴`; on 40 cells by at most 8·10⁻⁵ (measured in step 14).
            if name == "Alfvén wave"
                wave = TreeGRRMHD.U_S .+ (1, 2)
                @test all(v -> device_close(dev.l1.U[v], host.l1.U[v], T), wave)
                @test all(v -> device_close(dev.l1.U[v], host.l1.U[v], T),
                          TreeGRRMHD.U_E .+ (1, 2))
                @test all(j -> device_close(dev.l1.A[j], host.l1.A[j], T), (2, 3))
            end
            @info "$name at $T on $DEVICE_LABEL, two-phase $tp: $(dev.r.nsteps) steps, " *
                  "blocks $(map(c -> c.nblocks, dev.r.chunks)); max ρ $(dev.ρ_max) " *
                  "against the host's $(host.ρ_max), ∇·𝔅 $(dev.divB) against " *
                  "$(host.divB); " *
                  "resets $(map(c -> c.tally.resets, dev.r.chunks)) against " *
                  "$(map(c -> c.tally.resets, host.r.chunks))"
        end
    end
    return nothing
end

if DEVICE_STANDALONE
    ranks = DEVICE_MPI ? " on $(MPI.Comm_size(DEVICE_COMM)) MPI ranks" : ""
    @info "Device tests on $DEVICE_LABEL at $DEVICE_TYPES, $(Threads.nthreads()) " *
          "thread(s)$ranks"
    @testset "TreeGRRMHD.jl" begin
        device_testsets()
    end
else
    device_testsets()
end
