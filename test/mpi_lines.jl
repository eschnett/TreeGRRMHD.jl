# How two runs' lines are compared ("Running distributed" in `CODE.md`): a helper of
# `mpi_tests.jl` and of the coupled suite's `coupled/mpi_tests.jl`. A line that starts with
# `~` holds sums and is compared number by number to rounding, and its text exactly; every
# other line is compared with `==`.

# The numbers in a line, and the line with each of them replaced by `#`, so that two lines
# of sums are the same line exactly when their skeletons are equal and their numbers agree.
const MPI_NUMBER = r"[-+]?(?:\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+|\d+|NaN|Inf)"

mpi_skeleton(line) = replace(line, MPI_NUMBER => "#")
mpi_numbers(line) = [parse(Float64, m.match) for m in eachmatch(MPI_NUMBER, line)]

# Rounding: the sums are totals and norms over the mesh, each reassociated once per rank
# boundary — a few ulp of the total. `rtol` is a few thousand ulp; `atol` covers a sum whose
# exact value is zero (a total charge, a drift) and is then rounding of its terms' scale.
mpi_agree(a, b) = isapprox(a, b; rtol=1e-12, atol=1e-14)

# Two lines that make the same claim: equal, or for a line of sums (`~`) the same text with
# numbers that agree to rounding.
mpi_same(a, b) = startswith(a, "~") ?
                 mpi_skeleton(a) == mpi_skeleton(b) &&
                 all(map(mpi_agree, mpi_numbers(a), mpi_numbers(b))) : a == b

