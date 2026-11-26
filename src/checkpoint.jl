# Checkpoints ("Running distributed", "Checkpoints" in `CODE.md`).
#
# Everything about the *file* is upstream: TreeIOHDF5's `save_checkpoint` writes the forest,
# the evolved field sets and an application's plain data, atomically, one part file per I/O
# group over MPI ranks, and its `load_checkpoint` reads them back at any rank count, or
# serially, distributed over the reader's communicator. TreeHydro's `src/checkpoint.jl` is
# the model. What is left here is what only this package knows:
#
#   * which sets are state: the five evolved sets `U`, `A₁`, `A₂`, `A₃` and `Φ̃` —
#     never `B`, `P`, `W` or the scratch sets, which a restart rebuilds from the state
#     ("Sharp edges": `B` is never state);
#   * the run state behind a result (`evolve!`'s `totals0`, chunk count and history, or a
#     study's accumulators), which the caller hands in as plain data;
#   * the recipe, every parameter that decides the numbers, so that a restart with another
#     one is refused by name rather than run into a different experiment.
#
# When to write is the caller's: at a chunk boundary, before the next chunk's regrid, as
# TreeHydro and TreeGH write. The step cache is not stored: it is invalidated at the end of
# every step ("The step cache" in `CODE.md`), so a restart at a chunk boundary is the
# uninterrupted run bit for bit (`test/checkpoint_tests.jl`).

const CHECKPOINT_APPLICATION = "TreeGRRMHD.jl"
const CHECKPOINT_VERSION = 1

# The names of the five evolved sets in a checkpoint, in `evolved_sets` order.
const CHECKPOINT_SETS = ("U", "A1", "A2", "A3", "Phi")

"""
    checkpoint_sets(p::MHDProblem, u) -> NTuple{5,Pair}

The five evolved sets of `p` with their segments of the state vector `u`, as TreeIOHDF5's
`save_checkpoint` takes them: `"U"`, `"A1"`, `"A2"`, `"A3"`, `"Phi"`. A script that
checkpoints other sets beside them (TreeGH's, coupled) passes them as `extra` to
[`save_run`](@ref).
"""
checkpoint_sets(p::MHDProblem, u) =
    ntuple(i -> CHECKPOINT_SETS[i] => (p.sets[i], stateview(u, p.sets, i)), 5)

"""
    save_run(path, p::MHDProblem, u; run, recipe, extra = (), io = :node, sync = true) -> path

Write the state `u` of the problem `p` — its five evolved sets, over the forest they live
on — to the checkpoint `path` with TreeIOHDF5's `save_checkpoint`, with `run` (the caller's
run state) and `recipe` (its parameters, [`checkpoint_recipe`](@ref)) as plain data
(TreeIOHDF5's `write_plain`: numbers, strings, symbols, tuples, named tuples, dicts and
arrays of numbers). `extra` are further `name => (set, state)` pairs over the same forest.
Atomic: a failed write leaves the previous checkpoint at `path`.

Over MPI ranks it is collective, with the same `path`, `run` and `recipe` on every rank
(refused on every rank otherwise); `io` is TreeIOHDF5's number of I/O groups (`:node`, one
part file per node; `:all`; or a count).
"""
function save_run(path::AbstractString, p::MHDProblem, u; run, recipe, extra=(), io=:node,
                  sync::Bool=true)
    forest = first(p.sets).forest
    return TreeIOHDF5.save_checkpoint(path, forest;
                                      fieldsets=(checkpoint_sets(p, u)..., extra...),
                                      application=CHECKPOINT_APPLICATION => CHECKPOINT_VERSION,
                                      data=(; recipe, run), io, sync)
end

"""
    load_run(path; backend = CPU(), comm = nothing, types = ()) -> (; forest, states, recipe, run)

Read a checkpoint [`save_run`](@ref) wrote: the forest, distributed over `comm` (anything
`Forest` takes; collective then, and a checkpoint written at any rank count loads at any
other, or serially), every set's state vector by name on `backend` (`states`, a `Dict`),
and the `recipe` and `run` it was written with. A file of another application or format
version is refused with the reason. `types` names element types TreeIOHDF5 cannot name on
its own (a MultiFloats type).

A restart builds its field sets on the returned forest ([`mhd_fields`](@ref)), checks the
recipe ([`check_recipe`](@ref)) and copies the states in ([`restore_run!`](@ref)); `B`,
`P`, `W` and the scratch sets are rebuilt from the state by the first fill.
"""
function load_run(path::AbstractString; backend=CPU(), comm=nothing, types=())
    ck = TreeIOHDF5.load_checkpoint(path; backend, comm, types)
    name, version = ck.application
    written = "It was written by TreeAMR " *
              "$(something(ck.provenance.treeamr_version, "(unknown version)")) on " *
              "$(ck.provenance.created)."
    name == CHECKPOINT_APPLICATION || throw(ArgumentError(
        "$(repr(path)) is a checkpoint of the application $(repr(name)), not of " *
        "$CHECKPOINT_APPLICATION: its run state is that application's. $written"))
    version == CHECKPOINT_VERSION || throw(ArgumentError(
        "$(repr(path)) stores TreeGRRMHD's run state in format version $version, and this " *
        "version reads version $CHECKPOINT_VERSION only; TreeIOHDF5's " *
        "`checkpoint_environment(path, dir)` writes the environment that wrote it. $written"))
    (ck.data isa NamedTuple && haskey(ck.data, :recipe) && haskey(ck.data, :run) &&
     all(n -> haskey(ck.fieldsets, n), CHECKPOINT_SETS)) || throw(ArgumentError(
        "$(repr(path)) names $CHECKPOINT_APPLICATION but holds no recipe, no run state or " *
        "not the five evolved sets $(CHECKPOINT_SETS): it is damaged, or was not written " *
        "by `save_run`. $written"))
    states = Dict{String,Any}(n => fs.state for (n, fs) in ck.fieldsets)
    return (; ck.forest, states, ck.data.recipe, ck.data.run)
end

"""
    restore_run!(u, p::MHDProblem, ck) -> u

Copy the five evolved sets' states of the checkpoint `ck` ([`load_run`](@ref)) into the
state vector `u` of `p`, which must have been built on `ck.forest` (refused otherwise: a
state vector is laid out by its forest's blocks).
"""
function restore_run!(u, p::MHDProblem, ck)
    first(p.sets).forest === ck.forest || throw(ArgumentError(
        "the problem is not built on the checkpoint's forest: build its field sets on " *
        "`ck.forest` (`mhd_fields(T, ck.forest)`), whose leaves and partition the stored " *
        "states are laid out by."))
    for i in 1:5
        copyto!(stateview(u, p.sets, i), ck.states[CHECKPOINT_SETS[i]])
    end
    return u
end

# A value as plain data that compares as the value does: a number, string, symbol, bool
# or nothing as itself, a rational as itself, a tuple or named tuple element by element, a
# type by its name, anything else by `repr` (an `Operators`, a model), which is what a
# reader of the refusal sees anyway.
plain_value(x::Union{Number,AbstractString,Symbol,Nothing}) = x
plain_value(x::Tuple) = map(plain_value, x)
plain_value(x::NamedTuple) = map(plain_value, x)
plain_value(::Type{T}) where {T} = string(T)
plain_value(x) = repr(x)

"""
    checkpoint_recipe(; kw...) -> NamedTuple

The parameters of a run that decide its numbers, as plain data a checkpoint stores and
[`check_recipe`](@ref) compares: each keyword's value as itself where it is a number,
string, symbol, tuple or named tuple of those, a type by its name, and anything else by its
`repr`.
"""
checkpoint_recipe(; kw...) = map(plain_value, NamedTuple(kw))

"""
    check_recipe(saved, now)

Refuse, naming every difference, a restart whose recipe `now` ([`checkpoint_recipe`](@ref))
differs from the one the checkpoint was written with: a key missing on either side or a
value that is not `isequal`. A restarted run with another recipe would be another
experiment that looks like the old one.
"""
function check_recipe(saved, now)
    keys_ = union(keys(saved), keys(now))
    differ = String[]
    for k in keys_
        a = haskey(saved, k) ? saved[k] : :missing
        b = haskey(now, k) ? now[k] : :missing
        isequal(a, b) || push!(differ, "$k: $(repr(a)) in the checkpoint, $(repr(b)) now")
    end
    isempty(differ) || throw(ArgumentError(
        "the restart's recipe differs from the checkpoint's, so it would be another run:\n  " *
        join(differ, "\n  ")))
    return nothing
end
