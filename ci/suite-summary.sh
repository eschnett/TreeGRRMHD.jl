#!/bin/bash
# The suite's summary from a log: one line per log, from Test's table.
#
#   ci/suite-summary.sh LOG|DIR...
#
# A directory stands for its `output.log` (a SimLaunch job directory). Each
# line is tab-separated:
#
#   <label>  <pass|fail>  <counts>  <time>
#
# `<label>` is the run's (`t4` for `…/full-suite-t4/001-<job>/output.log`) or,
# for any other log, its file name. `<counts>` is the suite's own row, as
# "26664 pass of 26664" or "26640 pass, 24 fail of 26664", and `<time>` its
# Time column. A log with no summary row — a run that died before Test printed
# one — is `fail` with "no summary". `pass` means a summary with no failures
# and no errors; whether the process exited 0 is the caller's to check.
#
# Used by `ci/full-suite.sh`, by `jobs/full-suite.toml`'s `[after]` and by
# `ci/full-suite-cluster.sh collect`, so the three read a log the same way.

set -euo pipefail

[ $# -gt 0 ] || { sed -n '2,19p' "$0" >&2; exit 2; }

for arg in "$@"; do
    log=$arg
    [ -d "$log" ] && log="$log/output.log"
    case "$log" in
        */full-suite-*/*/output.log)
            run=$(basename "$(dirname "$(dirname "$log")")")
            label=${run#full-suite-} ;;
        *) label=$(basename "$log") ;;
    esac
    line=""
    [ -f "$log" ] && line=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | awk -F'|' '
        /^Test Summary:/ { n = split($2, names, " ") }
        /^TreeGRRMHD\.jl +\|/ && n {
            m = split($2, v, " "); s = ""; total = ""; time = ""; bad = 0
            for (i = 1; i <= m && i <= n; i++) {
                if (names[i] == "Total") total = v[i]
                else if (names[i] == "Time") time = v[i]
                else {
                    s = s (s == "" ? "" : ", ") v[i] " " tolower(names[i])
                    if ((names[i] == "Fail" || names[i] == "Error") && v[i] + 0 > 0) bad = 1
                }
            }
            line = (bad ? "fail" : "pass") "\t" s " of " total "\t" time
        }
        END { print line }')
    [ -n "$line" ] || line=$'fail\tno summary\t'
    printf '%s\t%s\n' "$label" "$line"
done
