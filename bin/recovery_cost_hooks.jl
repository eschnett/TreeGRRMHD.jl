# Instrumentation for `bin/recovery_cost.jl` (measurement 2026-10-23, branch
# `claude/recovery-cost`; not a PLAN.md step). Evaluated *into* the TreeGRRMHD module
# (`Core.eval(TreeGRRMHD, :(include(path)))`), so that it can redefine, in this process
# only, the two functions that call EntropyEOS's `con2prim_safe` and the passes that call
# the kernels. `src/` is untouched: with `RCS.mode = 0`, `RCS.steplocal = false` and
# `RCS.fill = :primitive` every redefined function computes what `src/` computes, bit for
# bit (checked by `bin/recovery_cost.jl identity`).
#
# What it adds:
#
# - **Call sites.** Each kernel that recovers sets `RCS.site`, `RCS.block`, `RCS.cell`
#   (the stored linear index) before the recovery; inside a stage solve the site is
#   resolved per call (the first hydrodynamic recovery is the predictor, the later ones
#   iterations — the lagged-velocity fallback's last one included —, an ideal recovery
#   after the first is the Newton seed, a first one `IdealConductor`'s whole solve).
# - **Counts and in-situ cost** (`RCS.mode = 1`): per site, the calls that ran the cold
#   seed (`s` or `w` NaN: `con2prim` then runs `cold_seed`), the warm ones, and those
#   `con2prim_safe` answered without a solve (the atmosphere trigger, non-finite input, the
#   collapse branch); the
#   Newton iterations, the calls that reached the nested fallback, and `time_ns` around
#   each call. One thread only: the context is global.
# - **The step-local view, emulated**: `RCS.seeds` holds, per stored cell, `(s, w, u_eos)`
#   of the cell's most recent recovery; invalidated at the start of each step limiter (the
#   step limiter's cold recovery of `uⁿ⁺¹` is then the first entry of the next step's
#   cache). Counted: the cold-seeded solves for which a step-local seed existed; recorded in
#   a window of steps (`RCS.record`): every call's input, its seed now and its step-local
#   seed, for an offline replay that times both.
# - **Coincidence with the cell's previous recovery**: the input `(D, τ, D Yₑ, S_∥, S_⊥, B²)`
#   against the previous call's in the same cell (bitwise, or the largest difference over
#   `τ + D`), and the outputs `(ρ, s, w)`, tallied by (site, previous site).
# - **The conserved ghost fill** (`RCS.fill = :conserved`): step 8's first fill, the tree at
#   7605986 — the matter's conserved state filled by its own schedule (Conservative width 3,
#   bounded, the case's hook) and recovered in every stored cell, `P` straight from the
#   recovery. The default (`:primitive`) is the package's fill of `W` (since 2026-10-23
#   the only one in `src/`, whose ghost pass no longer writes the conserved ghosts; the
#   measurement of 2026-10-23 timed it with that `prim2con`, "RHS: ghost P").
# - **Ghost kinds**: per stored cell, owned / same-level copy / prolongated / restricted /
#   outer boundary, from the geometry; with the conserved fill, whether a copy's `U` and
#   `P` equal its source's bit for bit after the pass.
# - **Pass timers** (`RCT`), around every launch and fill of the right-hand side, the stage
#   solve and the two resets.
# - **The step-local view, real** (`RCS.steplocal = true`): the kernels seed every recovery
#   from `RCS.real` — the cell's most recent recovery since the start of the last step
#   limiter — where it holds one, and the state otherwise. Thread-safe (per cell), used
#   for the determinism checks.

const RC_SITES = ("RHS owned", "RHS ghost: same-level copy", "RHS ghost: prolongated",
                  "RHS ghost: restricted", "RHS ghost: outer boundary",
                  "reset as stage limiter", "reset as step limiter", "stage: predictor",
                  "stage: ideal seed", "stage: iterations", "stage: IdealConductor")
const RC_NS = length(RC_SITES)
const RC_RHS = 1                 # + ghost kind (0 owned … 4 boundary)
const RC_RESET_STAGE = 6
const RC_RESET_STEP = 7
const RC_PRED = 8
const RC_IDEALSEED = 9
const RC_ITER = 10
const RC_IDEAL = 11
const RC_STAGE = 100             # context while a stage solve runs
const RC_NB = 7                  # coincidence buckets: ==, ≤1e-14, ≤1e-12, ≤1e-10, ≤1e-8, >, n/a

struct RCRec
    site::Int8
    step::Int32
    block::Int32
    cell::Int32
    ideal::Bool
    solved::Bool                 # false: `con2prim_safe` answered by its policy, no solve
    cin::NTuple{6,Float64}
    gnow::NTuple{3,Float64}
    gloc::NTuple{3,Float64}
end

mutable struct RCState
    mode::Int
    fill::Symbol
    steplocal::Bool
    record::Bool
    site::Int
    block::Int
    cell::Int
    k::Int
    step::Int
    n::Matrix{Int}               # (site, 1 cold-seeded / 2 warm / 3 no solve)
    newton::Matrix{Int}
    fallback::Matrix{Int}
    tns::Matrix{Int}
    locavail::Vector{Int}        # cold-seeded solves with a step-local seed available
    coin::Array{Int,3}           # (site, previous site, bucket) of the input
    cout::Array{Int,3}           # the same, of the output (ρ, s, w)
    seeds::Array{Float64,3}      # (3, stored, block): emulated step-local cache
    real::Array{Float64,3}       # the same, for the real step-local mode
    lastcin::Array{Float64,3}    # (6, stored, block)
    lastsite::Matrix{Int}
    lastout::Array{Float64,3}    # (3, stored, block): ρ, s, w
    recs::Vector{RCRec}
    kind::Matrix{Int8}           # (stored, block): 0 owned, 1 copy, 2 prolongated, 3 restricted, 4 boundary
    src::Matrix{Int}             # (stored, block): a copy's source, (block − 1)·nstored + cell
    meshkey::Tuple{Int,Int}
    copies::Int                  # conserved fill: ghost copies checked
    copies_U::Int                # … whose U equals the source's (the sign of a zero free)
    copies_P::Int                # … whose P does
    copies_Pnum::Int             # … whose P does with `==` (the sign of a zero free)
end

function RCState()
    z(a...) = zeros(Int, a...)
    e3 = Array{Float64,3}(undef, 0, 0, 0)
    return RCState(0, :primitive, false, false, 0, 1, 1, 0, 0, z(RC_NS, 3), z(RC_NS, 3),
                   z(RC_NS, 3), z(RC_NS, 3), z(RC_NS), z(RC_NS, RC_NS, RC_NB),
                   z(RC_NS, RC_NS, RC_NB), e3, e3, e3, zeros(Int, 0, 0), e3, RCRec[],
                   zeros(Int8, 0, 0), zeros(Int, 0, 0), (0, 0), 0, 0, 0, 0)
end

const RCS = RCState()

function rc_clear_counts!()
    for a in (RCS.n, RCS.newton, RCS.fallback, RCS.tns, RCS.locavail, RCS.coin, RCS.cout)
        fill!(a, 0)
    end
    RCS.copies = RCS.copies_U = RCS.copies_P = RCS.copies_Pnum = 0
    empty!(RCS.recs)
    return nothing
end

# --- the mesh: per stored cell, its kind and a copy's source ----------------------------------

function rc_mesh!(p::MHDProblem{T,D}) where {T,D}
    P = p.f.P
    nb = nblocks(P)
    S = size(P.work)[1:D]
    ns = prod(S)
    key = (nb, ns)
    key == RCS.meshkey && size(RCS.seeds, 3) == nb && return nothing
    RCS.meshkey = key
    RCS.seeds = fill(NaN, 3, ns, nb)
    RCS.real = fill(NaN, 3, ns, nb)
    RCS.lastcin = fill(NaN, 6, ns, nb)
    RCS.lastsite = zeros(Int, ns, nb)
    RCS.lastout = fill(NaN, 3, ns, nb)
    RCS.kind = zeros(Int8, ns, nb)
    RCS.src = zeros(Int, ns, nb)
    forest = P.forest
    N, G = forest.N, P.G
    o = block_origins(forest, T)
    h = block_spacings(forest, T)
    ext = forest.extents
    per = forest.periodic
    for b in 1:nb, (lin, I) in enumerate(CartesianIndices(S))
        n = ntuple(e -> I[e] - G[e], D)
        all(e -> 1 ≤ n[e] ≤ N, 1:D) && continue
        x = ntuple(e -> o[b][e] + (n[e] - one(T) / 2) * h[b], D)
        inside = true
        x = ntuple(D) do e
            lo, hi = ext[e]
            xe = x[e]
            if xe < lo || xe > hi
                per[e] ? (xe = lo + mod(xe - lo, hi - lo)) : (inside = false)
            end
            xe
        end
        if !inside
            RCS.kind[lin, b] = 4
            continue
        end
        found = false
        for b2 in 1:nb
            box = all(e -> o[b2][e] ≤ x[e] < o[b2][e] + N * h[b2], 1:D)
            box || continue
            found = true
            if h[b2] == h[b]
                RCS.kind[lin, b] = 1
                n2 = ntuple(e -> floor(Int, (x[e] - o[b2][e]) / h[b2]) + 1 + G[e], D)
                RCS.src[lin, b] = (b2 - 1) * ns + LinearIndices(S)[n2...]
            else
                RCS.kind[lin, b] = h[b2] > h[b] ? 2 : 3
            end
            break
        end
        found || error("rc_mesh!: no block holds the ghost at $x of block $b")
    end
    return nothing
end

# The stored linear index of a stored index tuple, with stored sizes `N + 2G`.
@inline function rc_lin(s::NTuple{D,Int}, N::Int, G::NTuple{D,Int}) where {D}
    l = 0
    stride = 1
    for e in 1:D
        l += (s[e] - 1) * stride
        stride *= N + 2 * G[e]
    end
    return l + 1
end

@inline function rc_context!(site::Int, b::Int, lin::Int)
    RCS.site = site
    RCS.block = b
    RCS.cell = lin
    RCS.k = 0
    return nothing
end

# The real step-local mode: the cell's latest recovery since the start of the last step
# limiter, where it holds one, and the state's own warm start otherwise.
@inline function rc_guesses(cons::NTuple{NU,T}, b::Int, lin::Int) where {T}
    if RCS.steplocal
        r = RCS.real
        g = (r[1, lin, b], r[2, lin, b], r[3, lin, b])
        isnan(g[1]) | isnan(g[2]) || return g
    end
    return state_guesses(cons)
end
@inline function rc_store!(prim::NTuple, b::Int, lin::Int)
    if RCS.steplocal
        g = guesses(prim)
        RCS.real[1, lin, b] = g[1]
        RCS.real[2, lin, b] = g[2]
        RCS.real[3, lin, b] = g[3]
    end
    return nothing
end

# --- the counted call --------------------------------------------------------------------

@inline function rc_c2p(eos, cin::EntropyEOS.Con2PrimIn{T}, opts, pol, g::NTuple{3,T},
                        ideal::Bool) where {T}
    RCS.mode == 0 && return EntropyEOS.con2prim_safe(eos, cin, opts, pol, g[1], g[2], g[3])
    return rc_c2p_counted(eos, cin, opts, pol, g, ideal)
end

@inline rc_cin(c) = (c.D, c.τ, c.D_Y, c.S_par, c.S_perp, c.B²)

function rc_bucket(δ)
    δ == 0 && return 1
    δ ≤ 1e-14 && return 2
    δ ≤ 1e-12 && return 3
    δ ≤ 1e-10 && return 4
    δ ≤ 1e-8 && return 5
    return 6
end

@noinline function rc_c2p_counted(eos, cin::EntropyEOS.Con2PrimIn{T}, opts, pol,
                                  g::NTuple{3,T}, ideal::Bool) where {T}
    site = RCS.site
    if site == RC_STAGE
        RCS.k += 1
        site = ideal ? (RCS.k == 1 ? RC_IDEAL : RC_IDEALSEED) :
               (RCS.k == 1 ? RC_PRED : RC_ITER)
    end
    b, c = RCS.block, RCS.cell
    # `con2prim` runs its cold seed where `s` or `w` is missing; a missing `u_eos` alone
    # only starts the EOS's own temperature solve cold (the analytic gases return none).
    cold = isnan(g[1]) | isnan(g[2])
    sd = RCS.seeds
    loc = (sd[1, c, b], sd[2, c, b], sd[3, c, b])
    locok = !(isnan(loc[1]) | isnan(loc[2]))
    t0 = time_ns()
    out = EntropyEOS.con2prim_safe(eos, cin, opts, pol, g[1], g[2], g[3])
    dt = time_ns() - t0
    cls = !out.solved ? 3 : cold ? 1 : 2
    RCS.n[site, cls] += 1
    RCS.tns[site, cls] += Int(dt)
    RCS.newton[site, cls] += Int(out.base.iters_newton)
    out.solved && out.base.result !== EntropyEOS.C2PResult.converged_newton &&
        (RCS.fallback[site, cls] += 1)
    cls == 1 && locok && (RCS.locavail[site] += 1)
    x = rc_cin(cin)
    ls = RCS.lastsite[c, b]
    if ls > 0
        lc = ntuple(i -> RCS.lastcin[i, c, b], 6)
        if (x[6] == 0) != (lc[6] == 0)
            bi = RC_NB                                  # an ideal against a hydro input
        elseif x == lc
            bi = 1
        else
            sc = abs(x[1]) + abs(x[2])
            bi = rc_bucket(maximum(i -> abs(x[i] - lc[i]), 1:6) / sc)
        end
        RCS.coin[site, ls, bi] += 1
        ρ, s, w = out.base.ρ, out.base.s, out.base.w
        lρ, ls_, lw = RCS.lastout[1, c, b], RCS.lastout[2, c, b], RCS.lastout[3, c, b]
        bo = (ρ, s, w) == (lρ, ls_, lw) ? 1 :
             rc_bucket(max(abs(ρ - lρ) / abs(ρ), abs(s - ls_) / abs(s), abs(w - lw)))
        RCS.cout[site, ls, bo] += 1
    end
    RCS.record && push!(RCS.recs, RCRec(site, RCS.step, b, c, ideal, out.solved, x, g,
                                        locok ? loc : (NaN, NaN, NaN)))
    sd[1, c, b] = out.base.s
    sd[2, c, b] = out.base.w
    sd[3, c, b] = out.base.eos.u_solved
    for i in 1:6
        RCS.lastcin[i, c, b] = x[i]
    end
    RCS.lastsite[c, b] = site
    RCS.lastout[1, c, b] = out.base.ρ
    RCS.lastout[2, c, b] = out.base.s
    RCS.lastout[3, c, b] = out.base.w
    return out
end

# --- the two functions that call con2prim_safe, redefined to count -------------------------

@inline function fluid_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T}, D::T,
                                S_fluid::SVector{3,T}, τ_fluid::T, DYₑ::T,
                                guesses::NTuple{3,T}) where {T}
    Su = raise(m, S_fluid)
    S² = sum(S_fluid .* Su)
    S_mag = S² < zero(T) ? zero(T) : sqrt(S²)
    cin = EntropyEOS.Con2PrimIn{T}(D, τ_fluid, DYₑ, zero(T), S_mag, zero(T))
    out = rc_c2p(eos, cin, opts, pol, guesses, false)
    floored(out.policy_flags) && return atmosphere_at_rest(eos, pol, D, DYₑ)
    b = out.base
    v = velocity_at_rapidity(m, Su, b.w)
    prim = pack_prim(b.ρ, v, b.s, b.ye, b.w, b.eos.p, b.eos.h, b.eos.cs², b.eos.u_solved)
    return prim, out.policy_flags, b.eos
end

@inline function ideal_recovery(eos::EntropyEOS.AbstractEOS, opts::EntropyEOS.Con2PrimOptions{T},
                                pol::EntropyEOS.PolicyOptions{T}, m::Metric3{T},
                                cons::NTuple{NU,T}, B::SVector{3,T},
                                guesses::NTuple{3,T}) where {T}
    isg = inv(m.sqrtγ)
    D = cons_D(cons) * isg
    S = cons_S(cons) * isg
    τ = cons_τ(cons) * isg
    DYₑ = cons_DYₑ(cons) * isg
    Su = raise(m, S)
    B² = dot_up(m, B, B)
    if B² > zero(T)
        Bmag = sqrt(B²)
        b̂ = B / Bmag
        S_par = sum(S .* b̂)
        Su_perp = Su - S_par * b̂
    else
        b̂ = zero(SVector{3,T})
        S_par = zero(T)
        Su_perp = Su
    end
    S_perp² = dot_up(m, Su_perp, Su_perp)
    S_perp = S_perp² < zero(T) ? zero(T) : sqrt(S_perp²)
    cin = EntropyEOS.Con2PrimIn{T}(D, τ, DYₑ, S_par, S_perp, B²)
    out = rc_c2p(eos, cin, opts, pol, guesses, true)
    floored(out.policy_flags) && return atmosphere_at_rest(eos, pol, D, DYₑ)
    b = out.base
    v_perp = S_perp > zero(T) ? (b.v_perp / S_perp) * Su_perp : zero(SVector{3,T})
    v = velocity_at_rapidity(m, b.v_par * b̂ + v_perp, b.w)
    prim = pack_prim(b.ρ, v, b.s, b.ye, b.w, b.eos.p, b.eos.h, b.eos.cs², b.eos.u_solved)
    return prim, out.policy_flags, b.eos
end

# --- the kernels, with the context ---------------------------------------------------------

# `primitive_kernel!` (this branch's primitive fill, owned cells).
@kernel function rc_primitive_kernel!(Pw, Ww, @Const(Uw), @Const(B1), @Const(B2), @Const(B3),
                                      @Const(origins), @Const(spacings), provider, t, eos, opts,
                                      pol, N, ::Val{D}, ::Val{GP}, ::Val{GU},
                                      ::Val{GB}) where {D,GP,GU,GB}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    cons = read_vars(Uw, stored(n, GU), b, Val(NU))
    𝔅 = cell_𝔅((B1, B2, B3), n, b, GB, Val(D))
    B = 𝔅 / m.sqrtγ
    E = cons_𝔈(cons) / m.sqrtγ
    sP = stored(n, GP)
    lin = rc_lin(sP, N, GP)
    rc_context!(RC_RHS, b, lin)
    prim, flags = con2prim_hydro(eos, opts, pol, m, cons, E, B, rc_guesses(cons, b, lin))
    rc_store!(prim, b, lin)
    u = prim_W(prim) * prim_v(prim)
    wv = W_store(prim_ρ(prim), prim_s(prim), prim_Yₑ(prim), u, E)
    Pc = P_from_W(eos, m, wv, B, flags)
    @inbounds for v in 1:NW
        Ww[sP..., v, b] = wv[v]
    end
    @inbounds for v in 1:NP
        Pw[sP..., v, b] = Pc[v]
    end
end

# The conserved fill's recovery in every stored cell (the tree at 7605986).
@kernel function rc_conserved_kernel!(Pw, @Const(Uw), @Const(B1), @Const(B2), @Const(B3),
                                      @Const(origins), @Const(spacings), @Const(kind), provider,
                                      t, eos, opts, pol, N, ::Val{D}, ::Val{GP}, ::Val{GU},
                                      ::Val{GB}) where {D,GP,GU,GB}
    I = @index(Global, NTuple)                       # stored: `stored = true`
    b = I[D + 1]
    s = ntuple(e -> I[e], Val(D))
    n = ntuple(e -> s[e] - GP[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    cons = read_vars(Uw, stored(n, GU), b, Val(NU))
    𝔅 = cell_𝔅((B1, B2, B3), n, b, GB, Val(D))
    B = 𝔅 / m.sqrtγ
    E = cons_𝔈(cons) / m.sqrtγ
    lin = rc_lin(s, N, GP)
    rc_context!(RC_RHS + Int(@inbounds kind[lin, b]), b, lin)
    prim, flags = con2prim_hydro(eos, opts, pol, m, cons, E, B, rc_guesses(cons, b, lin))
    rc_store!(prim, b, lin)
    Pc = pack_P(prim, flags, E, B, zero(h), m)
    @inbounds for v in 1:NP
        Pw[s..., v, b] = Pc[v]
    end
end

@kernel function rc_ohm_kernel!(Uw, @Const(ustar), @Const(B1), @Const(B2), @Const(B3), Xw,
                                @Const(origins), @Const(spacings), provider, t, γΔt, eos, opts,
                                pol, model, stage, N, ::Val{D}, ::Val{GB}, ::Val{GX},
                                ::Val{GP}) where {D,GB,GX,GP}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    cons★ = read_vars(ustar, n, b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E★ = cons_𝔈(cons★) * isg
    lin = rc_lin(stored(n, GP), N, GP)
    rc_context!(RC_STAGE, b, lin)
    E, prim, r, nrec = solve_ohm_stage(eos, opts, pol, model, m, cons★, E★, B,
                                       oftype(h, γΔt), rc_guesses(cons★, b, lin); stage=stage)
    rc_store!(prim, b, lin)
    𝔈 = m.sqrtγ * E
    @inbounds for k in 1:3
        Uw[n..., U_E + k - 1, b] = 𝔈[k]
    end
    xi = stored(n, GX)
    @inbounds begin
        for k in 1:N_OHM_BITS
            Xw[xi..., X_OHM + k - 1, b] += (r.flags >> (k - 1)) & one(UInt32)
        end
        if r.policy_flags != 0
            Xw[xi..., X_POLICY, b] += one(h)
            Xw[xi..., X_POLICY_OR, b] = unsafe_trunc(UInt32, Xw[xi..., X_POLICY_OR, b]) |
                                        r.policy_flags
        end
        Xw[xi..., X_RECOVERIES, b] += nrec
        Xw[xi..., X_MAX_RECOVERIES, b] = max(Xw[xi..., X_MAX_RECOVERIES, b], nrec)
    end
end

@kernel function rc_reset_kernel!(uw, @Const(B1), @Const(B2), @Const(B3), Xw, @Const(origins),
                                  @Const(spacings), provider, t, eos, opts, pol, N, ::Val{D},
                                  ::Val{GB}, ::Val{GX}, ::Val{GP},
                                  ::Val{STEP}) where {D,GB,GX,GP,STEP}
    I = @index(Global, NTuple)
    b = I[D + 1]
    n = ntuple(e -> I[e], Val(D))
    h = @inbounds spacings[b]
    x = point_x(@inbounds(origins[b]), h, n, Val(cell_staggers(Val(D))))
    m = point_metric(provider, oftype(h, t), x)
    cons = read_vars(uw, n, b, Val(NU))
    isg = inv(m.sqrtγ)
    B = cell_𝔅((B1, B2, B3), n, b, GB, Val(D)) * isg
    E = cons_𝔈(cons) * isg
    lin = rc_lin(stored(n, GP), N, GP)
    rc_context!(STEP ? RC_RESET_STEP : RC_RESET_STAGE, b, lin)
    prim, flags = con2prim_hydro(eos, opts, pol, m, cons, E, B, rc_guesses(cons, b, lin))
    rc_store!(prim, b, lin)
    if flags & EntropyEOS.FLAG_POL_ANY != 0
        c = prim2con(eos, m, prim, E, B)
        @inbounds for v in 1:(U_E - 1)
            uw[n..., v, b] = c[v]
        end
        xi = stored(n, GX)
        @inbounds if STEP
            Xw[xi..., X_STEP_RESETS, b] += one(h)
            Xw[xi..., X_INJECTED_D, b] += c[U_D] - cons[U_D]
            Xw[xi..., X_INJECTED_TAU, b] += c[U_TAU] - cons[U_TAU]
        else
            Xw[xi..., X_RESETS, b] += one(h)
        end
    end
end

# --- the passes, with timers -----------------------------------------------------------------

const RCT_NAMES = ("RHS: scatter", "RHS: fill potentials", "RHS: curl", "RHS: recovery kernel",
                   "RHS: matter ghost fill", "RHS: ghost P (forward EOS)",
                   "RHS: charge", "RHS: fluxes", "RHS: restrict fluxes", "RHS: EMFs",
                   "RHS: restrict EMFs", "RHS: tendencies", "stage: scatter, fill, curl of A",
                   "stage: solve kernel", "stage limiter: scatter, fill, curl of A",
                   "stage limiter: kernel", "step limiter: scatter, fill, curl of A",
                   "step limiter: kernel")
const RCT = zeros(Int, length(RCT_NAMES))
macro rct(k, ex)
    return quote
        local t0 = time_ns()
        $(esc(ex))
        RCT[$(esc(k))] += Int(time_ns() - t0)
        nothing
    end
end

function update_primitives!(p::MHDProblem{T,D}, u, t) where {T,D}
    @rct 1 scatter!(p.sets, u)
    @rct 2 for i in SEG_A:SEG_Φ
        fill_set!(p, i, t)
    end
    return primitives_from_sets!(p, t)
end

function primitives_from_sets!(p::MHDProblem{T,D}, t) where {T,D}
    f = p.f
    rc_mesh!(p)
    @rct 3 curl_A!(f, p.𝔅̄, p.spacings)
    GB = Val(GB_of(f))
    N = f.P.forest.N
    if RCS.fill === :conserved
        @rct 5 fill_set!(p, SEG_U, t)
        @rct 4 map_blocks!(rc_conserved_kernel!, f.P, f.P.work, f.U.work, f.B[1].work,
                           f.B[2].work, f.B[3].work, p.origins, p.spacings, RCS.kind,
                           p.provider, t, p.eos, p.opts, p.pol, N, Val(D), Val(f.P.G),
                           Val(f.U.G), GB; stored=true)
        RCS.mode > 0 && rc_check_copies!(p)
    else
        @rct 4 map_blocks!(rc_primitive_kernel!, f.P, f.P.work, f.W.work, f.U.work,
                           f.B[1].work, f.B[2].work, f.B[3].work, p.origins, p.spacings,
                           p.provider, t, p.eos, p.opts, p.pol, N, Val(D), Val(f.P.G),
                           Val(f.U.G), GB)
        @rct 5 fill_ghosts!(f.W, p.schedule_W;
                            boundary=p.boundary_W === nothing ? nothing :
                                     boundary_at(p.boundary_W, t))
        @rct 6 map_blocks!(ghost_kernel!, f.P, f.P.work, f.W.work, f.B[1].work,
                           f.B[2].work, f.B[3].work, p.origins, p.spacings, p.provider, t,
                           p.eos, N, Val(D), Val(f.P.G), Val(f.W.G), GB; stored=true)
    end
    @rct 7 map_blocks!(charge_kernel!, f.P, f.P.work, p.spacings, Val(D), Val(f.P.G))
    return f.P
end

# With the conserved fill: whether each same-level ghost holds its source's `U` and `P`.
function rc_check_copies!(p::MHDProblem{T,D}) where {T,D}
    f = p.f
    S = size(f.P.work)[1:D]
    ns = prod(S)
    CI = CartesianIndices(S)
    for b in axes(RCS.kind, 2), lin in axes(RCS.kind, 1)
        RCS.kind[lin, b] == 1 || continue
        b2, l2 = divrem(RCS.src[lin, b] - 1, ns)
        b2 += 1
        l2 += 1
        I, I2 = CI[lin], CI[l2]
        RCS.copies += 1
        # `isequal` is bit identity but for NaN (an analytic gas's `u_eos`); `==` leaves
        # the sign of a zero free.
        same(x, y) = isequal(x, y) || x == y
        all(v -> same(f.U.work[I, v, b], f.U.work[I2, v, b2]), 1:NU) && (RCS.copies_U += 1)
        all(v -> isequal(f.P.work[I, v, b], f.P.work[I2, v, b2]), 1:NP) &&
            (RCS.copies_P += 1)
        all(v -> same(f.P.work[I, v, b], f.P.work[I2, v, b2]), 1:NP) && (RCS.copies_Pnum += 1)
    end
    return nothing
end

function mhd_rhs!(du, u, p::MHDProblem{T,D}, t) where {T,D}
    f = p.f
    update_primitives!(p, u, t)
    GB = GB_of(f)
    ntuple(Val(D)) do d
        Fd = f.F[d]
        @rct 8 map_blocks!(flux_kernel!, Fd, Fd.work, f.P.work, f.B[d].work, p.origins,
                           p.spacings, p.provider, t, p.eos, Val(D), Val(d), Val(Fd.G),
                           Val(f.P.G), Val(GB[d]), p.limiter, p.solver; closed=true)
        @rct 9 (p.fixup && restrict_interfaces!(Fd, p.ischeds_F[d]))
        nothing
    end
    ntuple(Val(3)) do j
        a, b = mod1(j + 1, 3), mod1(j + 2, 3)
        Ej = f.EMF[j]
        @rct 10 map_blocks!(emf_kernel!, Ej, Ej.work, f.P.work, f.B[a].work, f.B[b].work,
                            p.origins, p.spacings, p.provider, t, Val(D), Val(j), Val(Ej.G),
                            Val(f.P.G), Val(GB[a]), Val(GB[b]), p.limiter; closed=true)
        @rct 11 ((p.fixup && p.ischeds_EMF[j] !== nothing) &&
                 restrict_interfaces!(Ej, p.ischeds_EMF[j]))
        nothing
    end
    t0 = time_ns()
    map_blocks!(matter_rhs_kernel!, f.U, statearray(du, p.sets, SEG_U), f.P.work,
                map(F -> F.work, f.F), p.origins, p.spacings, p.provider, t, Val(D),
                Val(f.P.G), Val(first(f.F).G))
    GA = GA_of(f)
    ntuple(Val(3)) do j
        i = SEG_A + j - 1
        map_blocks!(potential_rhs_kernel!, p.sets[i], statearray(du, p.sets, i),
                    f.EMF[j].work, f.A[1].work, f.A[2].work, f.A[3].work, f.Φ.work,
                    p.origins, p.spacings, p.lowwall, p.provider, t, Val(D), Val(j),
                    Val(f.EMF[j].G), Val(GA), Val(f.Φ.G))
        nothing
    end
    map_blocks!(gauge_rhs_kernel!, f.Φ, statearray(du, p.sets, SEG_Φ), f.A[1].work,
                f.A[2].work, f.A[3].work, f.Φ.work, p.origins, p.spacings, p.lowwall,
                p.provider, t, p.ξ, Val(D), Val(GA), Val(f.Φ.G))
    RCT[12] += Int(time_ns() - t0)
    return nothing
end

function ohm_solve!(U, ustar, γΔt, p::MHDProblem{T,D}, t) where {T,D}
    f = p.f
    rc_mesh!(p)
    @rct 13 update_fields!(p, ustar, t)
    @rct 14 map_blocks!(rc_ohm_kernel!, f.U, statearray(U, p.sets, SEG_U),
                        statearray(ustar, p.sets, SEG_U), f.B[1].work, f.B[2].work,
                        f.B[3].work, f.X.work, p.origins, p.spacings, p.provider, t, γΔt,
                        p.eos, p.opts, p.pol, p.model, p.stage, f.P.forest.N, Val(D),
                        Val(GB_of(f)), Val(f.X.G), Val(f.P.G))
    return nothing
end

function reset!(u, p::MHDProblem{T,D}, t, step::Val{STEP}) where {T,D,STEP}
    f = p.f
    rc_mesh!(p)
    if STEP
        # The step-local cache starts again at the step limiter's recovery of uⁿ⁺¹.
        fill!(RCS.seeds, NaN)
        RCS.steplocal && fill!(RCS.real, NaN)
    end
    t0 = time_ns()
    update_fields!(p, u, t)
    if p.haswalls
        scatter!(p.sets[SEG_Φ], stateview(u, p.sets, SEG_Φ))
        fill_set!(p, SEG_Φ, t)
        walls_to_state!(p, u)
    end
    RCT[STEP ? 17 : 15] += Int(time_ns() - t0)
    t0 = time_ns()
    map_blocks!(rc_reset_kernel!, f.U, statearray(u, p.sets, SEG_U), f.B[1].work,
                f.B[2].work, f.B[3].work, f.X.work, p.origins, p.spacings, p.provider, t,
                p.eos, p.opts, p.pol, f.P.forest.N, Val(D), Val(GB_of(f)), Val(f.X.G),
                Val(f.P.G), step)
    RCT[STEP ? 18 : 16] += Int(time_ns() - t0)
    STEP && (RCS.step += 1)
    return nothing
end
