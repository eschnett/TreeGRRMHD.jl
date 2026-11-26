# The field diagnostics of rung 6 ("Validation ladder" 6 in `CODE.md`; step 12): where a
# star's field energy is, how far `E` departs from `E·B = 0`, the charge `q = ∇·E`, how far
# the field has drifted from its initial state, which branches the implicit stage takes where,
# and where the recovery's energy is a small difference of large numbers (the entropy
# fallback's trigger, "Primitive recovery" in `CODE.md`). Host-side reductions over the owned
# cells, from one recovery of the state ([`update_primitives!`](@ref)) or from the
# diagnostics set `X`, like [`charge_norms`](@ref): diagnostics, called outside a step. Over
# MPI ranks a profile holds the rank's own cells, and every reduction sums its numerators
# and denominators per rank and combines them over the ranks before it forms a ratio
# (`src/distributed.jl`; "Running distributed" in `CODE.md`), so the value is the same on
# every rank. They are collective then: every rank calls them.

"""
    cell_profile(p::MHDProblem, u, t; metric = p.provider) -> NamedTuple

The owned cells of a three-dimensional state as host vectors, from one recovery of `u` at
`t` ([`update_primitives!`](@ref)): the cell centres `x` and spacings `h`; from `P`, the
metric `m` (a [`Metric3`](@ref)), `ρ`, the contravariant `E` and `B` (the cell average of
`curl A`), the charge `q` and the policy flags; and the undensitized total energy `τ`
(`𝓔 − D`, the fields' included) from the state itself. Over MPI ranks the cells are the
rank's own and `comm` the communicator the reductions combine over;
[`gather_profile`](@ref) makes every rank's profile the whole mesh's.
"""
function cell_profile(p::MHDProblem{T,3}, u, t; metric=p.provider) where {T}
    P = update_primitives!(p, u, T(t); metric)
    a = Array(P.work)
    uU = Array(statearray(u, p.sets, SEG_U))
    o, hs = Array(p.origins), Array(p.spacings)
    N, G = P.forest.N, P.G
    x, h, ms, ρ, E, B, q, flags, τ = (SVector{3,T}[], T[], Metric3{T}[], T[], SVector{3,T}[],
                                      SVector{3,T}[], T[], UInt32[], T[])
    for b in axes(a, 5), I in CartesianIndices((N, N, N))
        n = Tuple(I)
        k = stored(n, G)
        P_at = ntuple(v -> a[k..., v, b], Val(NP))
        m = aux_metric(P_at)
        push!(x, point_x(o[b], hs[b], n, Val(cell_staggers(Val(3)))))
        push!(h, hs[b])
        push!(ms, m)
        push!(ρ, P_at[P_RHO])
        push!(E, aux_E(P_at))
        push!(B, aux_B(P_at))
        push!(q, P_at[P_Q])
        push!(flags, unsafe_trunc(UInt32, P_at[P_FLAGS]))
        push!(τ, uU[n..., U_TAU, b] / m.sqrtγ)          # `u` holds the owned cells
    end
    return (; x, h, m=ms, ρ, E, B, q, flags, τ, comm=communicator(P))
end

"""
    gather_profile(prof) -> NamedTuple

The [`cell_profile`](@ref) of every rank, concatenated in rank order on every rank: the
serial run's profile, cell for cell, whatever the rank count, with a serial `comm`, so that
a reduction of it is not combined over the ranks a second time. Collective. Serially `prof`
itself. For test-sized meshes and figures; a production run records reductions.
"""
function gather_profile(prof)
    TreeAMR.commsize(prof.comm) == 1 && return prof
    g(v) = gather_cells(prof.comm, v)
    return (; x=g(prof.x), h=g(prof.h), m=g(prof.m), ρ=g(prof.ρ), E=g(prof.E), B=g(prof.B),
            q=g(prof.q), flags=g(prof.flags), τ=g(prof.τ),
            comm=TreeAMR.SerialCommunicator())
end

"""
    field_diagnostics(prof; radius, center = 0) -> NamedTuple

The field of a [`cell_profile`](@ref) inside (`|x − center| < radius`) and outside a
sphere, each sum over cells weighted by the proper volume `√γ h³`:

- `energy_in`, `energy_out`: the field energy `½(E·E + B·B)` the normal observer sees;
- `EB_in`, `EB_out`: `Σ |E·B| / Σ B·B`, the parallel electric field relative to `B`;
  `EB_max_in` is `max |E·B| / max B·B` inside;
- `E_in`: `(Σ E·E / Σ B·B)^{1/2}` inside, the size of `E` against `B`;
- `q_in`, `q_out`: `Σ |q|` (the charge's L1 integral), and `q_max`, the largest `|q|`;
- `trigger_in`, `trigger_out`: the cells, the atmosphere left out, whose fields' energy
  exceeds the fluid's share of the energy, `½(E·E + B·B) > τ − ½(E·E + B·B)`: where an
  entropy fallback with that threshold would invert at the advected `s` ("Primitive
  recovery" in `CODE.md`; its fallback waits for EOS-3);
- `repaired_in`, `repaired_out`: the cells, the atmosphere left out, whose recovery the
  policy repaired (any level-1 face);
- `cells_in`, `cells_out`;
- `B_max`, the largest `|B|` (the metric's norm) on any cell, and `B_center`, the
  volume-weighted mean `|B|` over the cells within `center_radius` of `center`: the
  central field whose decay a resistive star measures (added in step 12, after review).

Over MPI ranks every sum and maximum is combined over `prof.comm` before a ratio is formed.
"""
function field_diagnostics(prof; radius, center=zero(eltype(prof.x)), center_radius=1)
    T = eltype(prof.h)
    e, eb, b², e², qa = (zeros(T, 2) for _ in 1:5)
    trig, rep, cells = zeros(Int, 2), zeros(Int, 2), zeros(Int, 2)
    ebmax, bmax, qmax = zero(T), zero(T), zero(T)
    Bmax, Bc, wc = zero(T), zero(T), zero(T)
    for i in eachindex(prof.x)
        m = prof.m[i]
        k = sum(abs2, prof.x[i] - center) < T(radius)^2 ? 1 : 2
        w = m.sqrtγ * prof.h[i]^3
        E, B = prof.E[i], prof.B[i]
        EE, BB, EB = dot_up(m, E, E), dot_up(m, B, B), dot_up(m, E, B)
        e[k] += w * (EE + BB) / 2
        eb[k] += w * abs(EB)
        b²[k] += w * BB
        e²[k] += w * EE
        qa[k] += w * abs(prof.q[i])
        qmax = max(qmax, abs(prof.q[i]))
        Bmax = max(Bmax, sqrt(BB))
        if sum(abs2, prof.x[i] - center) < T(center_radius)^2
            Bc += w * sqrt(BB)
            wc += w
        end
        cells[k] += 1
        if k == 1
            ebmax = max(ebmax, abs(EB))
            bmax = max(bmax, BB)
        end
        fl = prof.flags[i]
        fl & EntropyEOS.FLAG_POL_ATMOSPHERE != 0 && continue
        fl & EntropyEOS.FLAG_POL_ANY != 0 && (rep[k] += 1)
        (EE + BB) / 2 > prof.τ[i] - (EE + BB) / 2 && (trig[k] += 1)
    end
    comm = get(prof, :comm, nothing)
    e1, e2, eb1, eb2, b1, b2, ee1, ee2, q1, q2, Bc, wc = rank_sum(comm,
        (e[1], e[2], eb[1], eb[2], b²[1], b²[2], e²[1], e²[2], qa[1], qa[2], Bc, wc))
    e, eb, b², e², qa = (e1, e2), (eb1, eb2), (b1, b2), (ee1, ee2), (q1, q2)
    t1, t2, r1, r2, c1, c2 = rank_sum(comm, (trig[1], trig[2], rep[1], rep[2], cells[1],
                                             cells[2]))
    trig, rep, cells = (t1, t2), (r1, r2), (c1, c2)
    ebmax, bmax, qmax, Bmax = rank_max(comm, (ebmax, bmax, qmax, Bmax))
    ratio(a, b) = b > 0 ? a / b : zero(T)
    return (; energy_in=e[1], energy_out=e[2], EB_in=ratio(eb[1], b²[1]),
            EB_out=ratio(eb[2], b²[2]), EB_max_in=ratio(ebmax, bmax),
            E_in=sqrt(ratio(e²[1], b²[1])), q_in=qa[1], q_out=qa[2], q_max=qmax,
            trigger_in=trig[1], trigger_out=trig[2], repaired_in=rep[1], repaired_out=rep[2],
            cells_in=cells[1], cells_out=cells[2], B_max=Bmax, B_center=ratio(Bc, wc))
end

"""
    field_drift(prof, prof0; radius, outer = Inf, center = 0) -> (; inside, outside)

How far the field `B` of the [`cell_profile`](@ref) `prof` has moved from `prof0`'s on the
same mesh: `Σ √γ h³ |B − B₀| / Σ √γ h³ |B₀|`, with `|·|` the metric's norm, inside the
sphere `|x − center| < radius` and in the shell `radius ≤ |x − center| < outer`. Over MPI
ranks the two profiles are the same rank's, and the sums are combined over `prof.comm`.
"""
function field_drift(prof, prof0; radius, outer=Inf, center=zero(eltype(prof.x)))
    T = eltype(prof.h)
    length(prof.x) == length(prof0.x) && all(prof.x .== prof0.x) || throw(ArgumentError(
        "the two profiles are not of the same mesh."))
    num, den = zeros(T, 2), zeros(T, 2)
    for i in eachindex(prof.x)
        r² = sum(abs2, prof.x[i] - center)
        r² < T(outer)^2 || continue
        k = r² < T(radius)^2 ? 1 : 2
        m = prof.m[i]
        w = m.sqrtγ * prof.h[i]^3
        δ = prof.B[i] - prof0.B[i]
        num[k] += w * sqrt(dot_up(m, δ, δ))
        den[k] += w * sqrt(dot_up(m, prof0.B[i], prof0.B[i]))
    end
    n1, n2, d1, d2 = rank_sum(get(prof, :comm, nothing), (num[1], num[2], den[1], den[2]))
    return (; inside=n1 / d1, outside=n2 / d2)
end

"""
    ohm_map(p::MHDProblem, region, nregions) -> Vector{NamedTuple}

The diagnostics set's tallies since the last [`clear_tally!`](@ref) ([`ohm_tally`](@ref)'s
counts), summed per region instead of over the mesh: `region(x)` maps an owned cell's
centre to `1 … nregions`. Each entry has `cells`, `bits` (the nine `OhmStageResult` flag
bits, in bit order), `policy` (stages whose returned state the policy repaired), `resets`,
`step_resets` and `recoveries`: the stage solve's outcome map (rung 6). Over MPI ranks the
tallies are summed over the ranks (collective).
"""
function ohm_map(p::MHDProblem{T,D}, region, nregions::Integer) where {T,D}
    X = Array(p.f.X.work)
    o, hs = Array(p.origins), Array(p.spacings)
    N, GX = p.f.X.forest.N, p.f.X.G
    cells = zeros(Int, nregions)
    acc = zeros(T, nregions, N_OHM_BITS + 4)
    slots = (ntuple(k -> X_OHM + k - 1, N_OHM_BITS)..., X_POLICY, X_RESETS, X_STEP_RESETS,
             X_RECOVERIES)
    for b in axes(X, D + 2), I in CartesianIndices(ntuple(_ -> N, Val(D)))
        n = Tuple(I)
        r = region(point_x(o[b], hs[b], n, Val(cell_staggers(Val(D)))))
        cells[r] += 1
        k = stored(n, GX)
        for (j, s) in enumerate(slots)
            acc[r, j] += X[k..., s, b]
        end
    end
    rank_sum!(cells, p)
    acc = reshape(rank_sum!(vec(acc), p), size(acc))
    c(r, j) = roundint(acc[r, j])
    return [(; cells=cells[r], bits=ntuple(j -> c(r, j), N_OHM_BITS),
             policy=c(r, N_OHM_BITS + 1), resets=c(r, N_OHM_BITS + 2),
             step_resets=c(r, N_OHM_BITS + 3), recoveries=c(r, N_OHM_BITS + 4))
            for r in 1:nregions]
end
