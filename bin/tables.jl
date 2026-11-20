# The real tables for the scripts (step 13): `load_table(name)` reads a stellarcollapse
# table from `ENTROPYEOS_TABLE_DIR` and builds its EOS, `(; table, eos)`. The names are
# the tables the step uses; the baryon mass is per family and not in the file (EntropyEOS's
# `test/test_real_tables.jl`), the atomic mass unit for Hempel's.

import TreeGRRMHD

const TABLE_FILES = Dict(
    "dd2" => "Hempel_DD2EOS_rho234_temp180_ye60_version_1.1_20120817.h5",
    "sfho" => "Hempel_SFHoEOS_rho222_temp180_ye60_version_1.1_20120817.h5")

function load_table(name::AbstractString)
    haskey(TABLE_FILES, name) ||
        error("unknown table `$name`; expected one of $(join(sort!(collect(keys(TABLE_FILES))), ", "))")
    dir = get(ENV, "ENTROPYEOS_TABLE_DIR", "")
    path = joinpath(dir, TABLE_FILES[name])
    isfile(path) || error("table `$name` not found at $path; set ENTROPYEOS_TABLE_DIR")
    E = TreeGRRMHD.EntropyEOS
    table = E.read_stellarcollapse(path)
    eos = E.build_eos(table, E.BuildOptions(; m_B_table_g=E.M_AMU_G))
    return (; table, eos)
end
