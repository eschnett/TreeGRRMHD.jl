# Running over MPI ranks ("Running distributed" in `CODE.md`), on TreeAMR's M7.
#
# TreeAMR keeps the whole forest on every rank, gives each rank a contiguous
# run of the Morton curve's leaves, and does everything that crosses ranks
# itself: the ghost exchange, the interface restriction, the regrid,
# `mesh_mapreduce`, the initial adaptation. What it cannot know is a number
# this package combines on the host from per-block or per-cell values, a
# decision taken from what differs between ranks (a file, a clock), and which
# rank writes. That is what this file is for; none of it is mesh machinery.
#
# The verbs are TreeAMR's public communicator verbs (AMR-13, TreeAMR 0.2.11):
# `commrank`, `commsize`, `allgather` and `allgatherv`, on the communicator
# `communicator(forest)` returns — a `SerialCommunicator` on a forest built
# without one. They are `public`, not exported, and called qualified. This
# package never names MPI: a serial run takes every function here through its
# one-rank branch, which returns its argument untouched and gathers nothing,
# so every serial number is what it was. TreeHydro's `src/distributed.jl` is
# the pattern, copied rather than depended on, as `src/device.jl` is.

# The communicator of whatever a reduction is over: a problem's, a forest's or a field
# set's, or what TreeAMR's `communicator` makes of an application's (an `MPI.Comm`, or
# `nothing` for the serial one).
rank_comm(comm::TreeAMR.Communicator) = comm
rank_comm(p::MHDProblem) = communicator(first(p.sets))
rank_comm(x) = communicator(x)

"""
    rank_reduce(op, over, value; present = true)

`value` combined over every rank of `over` — a forest, a field set, a problem or a
`TreeAMR.Communicator` — with `op`, **in rank order**, the same bits on every rank:
collective. `present = false` says this rank has nothing to contribute (it holds no
blocks), and its `value` is left out rather than folded in, TreeAMR's rule for
`mesh_mapreduce`; where no rank is present the result is `value`. On one rank it returns
`value` itself and gathers nothing, which keeps a serial run bit for bit what it was.
`value` must be `isbits`. TreeHydro's, as `combine_blocks` folds: one `allgather` of
`(present, value)`, folded in rank order.
"""
function rank_reduce(op, over, value; present::Bool=true)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return value
    parts = TreeAMR.allgather(comm, (present, value))::Vector{Tuple{Bool,typeof(value)}}
    acc, found = value, false
    for (has, v) in parts
        has || continue
        acc = found ? op(acc, v) : v
        found = true
    end
    return acc
end

# A sum over the ranks of a tuple (or a number) of partial sums, element by
# element: what a host diagnostic combines its numerators and denominators
# with before it forms a ratio.
rank_sum(over, value::Number) = rank_reduce(+, over, value)
rank_sum(over, value::Tuple) = rank_reduce((a, b) -> map(+, a, b), over, value)
rank_max(over, value::Number) = rank_reduce(max, over, value)
rank_max(over, value::Tuple) = rank_reduce((a, b) -> map(max, a, b), over, value)

"""
    rank_sum!(v::Vector, over) -> v

`v` on every rank summed element by element over the ranks, in rank order: for an array
each rank fills only from its own blocks. The length must be the same on every rank.
Serially `v` itself.
"""
function rank_sum!(v::Vector, over)
    comm = rank_comm(over)
    n = TreeAMR.commsize(comm)
    n == 1 && return v
    gathered = TreeAMR.allgatherv(comm, v)
    m = length(v)
    length(gathered) == n * m || throw(DimensionMismatch(
        "rank_sum! needs the same length on every rank, got $(length(gathered)) " *
        "elements from $n ranks of $m here."))
    v .= view(gathered, 1:m)
    for r in 2:n
        v .+= view(gathered, ((r - 1) * m + 1):(r * m))
    end
    return v
end

"""
    agree_any(over, flags::NTuple{N,Bool}) -> NTuple{N,Bool}

Each decision `true` on every rank where it is `true` on any: for a decision taken from
what differs between ranks — the wall clock, the file system — before it gates a
collective call, so that no rank enters a collective another skipped.
"""
agree_any(over, flags::NTuple{N,Bool}) where {N} =
    rank_reduce((a, b) -> map(|, a, b), over, flags)

"""
    isroot(over) -> Bool

Whether this is rank 0 of `over`, the rank that prints, writes a file a study keeps, and
removes one: serially always.
"""
isroot(over) = TreeAMR.commrank(rank_comm(over)) == 0

"""
    agree_refusal(f, over)

Run the checks `f()` on every rank, and if one throws an `ArgumentError` on any rank,
throw on **every** rank: the rank that refused its own error, the others one naming the
ranks that refused. For checks that can come out differently on different ranks, such as
whether a file exists on a node-local path; a refusal on one rank alone would leave the
others waiting in their next collective. Any other exception is rethrown at once.
Serially this is `f()`.
"""
function agree_refusal(f, over)
    comm = rank_comm(over)
    if TreeAMR.commsize(comm) == 1
        f()
        return nothing
    end
    err = nothing
    try
        f()
    catch e
        e isa ArgumentError || rethrow()
        err = e
    end
    refused = TreeAMR.allgather(comm, err !== nothing)::Vector{Bool}
    err === nothing || throw(err)
    any(refused) && throw(ArgumentError(
        "rank(s) $(join(findall(refused) .- 1, ", ")) refused the arguments, and this " *
        "rank did not: a check that reads the file system can come out differently on " *
        "different nodes, and the run stops on every rank rather than leave the others " *
        "waiting for the one that refused. Its error is printed by that rank."))
    return nothing
end

"""
    gather_cells(over, v::AbstractVector) -> Vector

Every rank's vector `v` of per-cell values, concatenated in rank order, on every rank:
over a distributed forest each rank holds a contiguous run of the curve, so a vector built
over the rank's own blocks in block order becomes the serial run's vector. Elements must
be `isbits`. Serially `v` itself (not a copy). For test-sized meshes and figures: a
production run records reductions.
"""
function gather_cells(over, v::AbstractVector)
    comm = rank_comm(over)
    TreeAMR.commsize(comm) == 1 && return v
    return TreeAMR.allgatherv(comm, v)
end
