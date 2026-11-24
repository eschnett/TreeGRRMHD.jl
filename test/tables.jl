# The real tables (added in step 13), as a helper rather than a test:
# `runtests.jl` includes it before its first user, and every file that gates a
# testset on the tables includes it if it was not.
#
# The stellarcollapse tables are hundreds of megabytes and are not in git, so
# the testsets that read one run only where `ENTROPYEOS_TABLE_DIR` names a
# directory holding it, as EntropyEOS's own real-table tests do: on Symmetry
# (`jobs/full-suite.toml` sets it) and on a workstation that has the tables;
# GitHub CI has none. A testset gated here is, without its table, a testset of
# the same name holding one skipped test, counted as "Broken" and named by
# `runtests.jl` at the end, as `tiers.jl`'s are: a green tick never claims more
# than it ran.

isdefined(@__MODULE__, :tier_testset) || include("tiers.jl")

const TABLE_DIR = get(ENV, "ENTROPYEOS_TABLE_DIR", "")
const DD2_FILE = "Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5"
const DD2_PRESENT = !isempty(TABLE_DIR) && isfile(joinpath(TABLE_DIR, DD2_FILE))

# The testsets left out for want of a table, which `runtests.jl` lists.
const TABLE_SKIPPED = Pair{String,String}[]

# `@dd2_gated @testset "name" … begin … end`: the testset where DD2 is found
# under `ENTROPYEOS_TABLE_DIR`; elsewhere a testset of the same name holding one
# skipped test.
macro dd2_gated(ex)
    ex isa Expr && ex.head === :macrocall && ex.args[1] === Symbol("@testset") ||
        error("@dd2_gated marks a `@testset`")
    i = findfirst(a -> a isa String || (a isa Expr && a.head === :string), ex.args)
    i === nothing && error("@dd2_gated needs a testset with a name")
    name = ex.args[i]
    skip = quote
        push!(TABLE_SKIPPED, $name => "@dd2_gated")
        @test_skip "needs DD2 under ENTROPYEOS_TABLE_DIR"
    end
    twin = Expr(:macrocall, ex.args[1:(end - 1)]...)
    push!(twin.args, skip)
    return esc(:(DD2_PRESENT ? $ex : $twin))
end

# The DD2 table and the EOS built from it, read once (a few seconds) on first
# use.
dd2_table() = computed_once(:dd2_table) do
    E = TreeGRRMHD.EntropyEOS
    t = E.read_stellarcollapse(joinpath(TABLE_DIR, DD2_FILE))
    eos = E.build_eos(t, E.BuildOptions(; m_B_table_g=E.M_AMU_G))
    (; table=t, eos)
end
