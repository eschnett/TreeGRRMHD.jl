#!/bin/bash
# The full suite on one commit, reported as a GitHub commit status (added
# 2026-11-08, TreeGeneralizedHarmonic's of 2026-11-06 for this package).
#
# GitHub CI runs the quick and threaded tiers (`test/tiers.jl`): the finest
# curved runs, the long uniform flows and, at four threads, the 1D studies are
# left out there. The full tier runs here — on the workstation before
# a pull request, and on Symmetry through `ci/full-suite-symmetry.sh` — and
# this script makes that visible on the pull request: with `--post` it sets the
# commit's status `full-suite/<host>` to pending, then to success or failure
# with the assertion count, the time, the thread count and the CPU.
#
# It tests a clean `git archive` of the commit, not the working tree, so the
# status is about the commit and nothing uncommitted (this is the
# clean-checkout check of `CLAUDE.md`, "Commands"). Posting is a public action:
# nothing is posted without `--post`, and the commit must be on GitHub.
#
#   ci/full-suite.sh [--threads N] [--context NAME] [--post] [REV]
#   ci/full-suite.sh --tree DIR --sha SHA [--result FILE] [--threads N] ...
#
# `--tree` tests an unpacked tree that is already there (Symmetry's job), and
# `--result` writes `state<TAB>description` to a file for a later `--post`
# from a machine that has `gh` (`ci/full-suite-symmetry.sh collect`).

set -euo pipefail

threads=4
context=""
post=0
tree=""
sha=""
result=""
rev="HEAD"
while [ $# -gt 0 ]; do
    case "$1" in
        --threads) threads="$2"; shift 2 ;;
        --context) context="$2"; shift 2 ;;
        --post) post=1; shift ;;
        --tree) tree="$2"; shift 2 ;;
        --sha) sha="$2"; shift 2 ;;
        --result) result="$2"; shift 2 ;;
        -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
        -*) echo "full-suite.sh: unknown option $1" >&2; exit 2 ;;
        *) rev="$1"; shift ;;
    esac
done

host=$(hostname -s)
case "$(uname -s)" in
    Darwin) cpu=$(sysctl -n machdep.cpu.brand_string) ;;
    *) cpu=$(lscpu | sed -n 's/^Model name: *//p' | head -1) ;;
esac
[ -n "$context" ] || context="full-suite/$host"

if [ -z "$tree" ]; then
    sha=$(git rev-parse --verify "$rev^{commit}")
    tree=$(mktemp -d "${TMPDIR:-/tmp}/treegrrmhd-full-suite.XXXXXX")
    trap 'rm -rf "$tree"' EXIT
    git archive "$sha" | tar -x -C "$tree"
fi
[ -n "$sha" ] || { echo "full-suite.sh: --tree needs --sha" >&2; exit 2; }

# The repository and the commit on GitHub, checked before anything runs.
if [ "$post" = 1 ]; then
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    gh api "repos/$repo/commits/$sha" --silent ||
        { echo "full-suite.sh: $sha is not on GitHub; push it first" >&2; exit 1; }
fi

post_status() {  # state description
    [ "$post" = 1 ] || return 0
    gh api "repos/$repo/statuses/$sha" -f state="$1" -f context="$context" \
        -f description="${2:0:140}" --silent
    echo "full-suite.sh: posted $context = $1 on $sha"
}

post_status pending "running the full tier at $threads threads on $host"

log="$tree/full-suite.log"
start=$(date +%s)
set +e
TREEGRRMHD_TEST_TIER=full julia --project="$tree" -e \
    "using Pkg; Pkg.instantiate(); Pkg.test(; julia_args = [\"--threads=$threads\"])" \
    2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e
elapsed=$(( $(date +%s) - start ))
minutes="$((elapsed / 60))m$(printf %02d $((elapsed % 60)))s"

# The suite's summary, from Test's table: the header names the columns
# present (`Pass Fail Error Broken Total Time`), and the suite's own row has
# a value in each of them — e.g. "5713 pass of 5713", "5690 pass, 23 fail of
# 5713".
summary=$(sed 's/\x1b\[[0-9;]*m//g' "$log" | awk -F'|' '
    /^Test Summary:/ { n = split($2, names, " ") }
    /^TreeGRRMHD\.jl +\|/ && n {
        m = split($2, v, " "); s = ""; total = ""
        for (i = 1; i <= m && i <= n; i++) {
            if (names[i] == "Total") total = v[i]
            else if (names[i] != "Time") s = s (s == "" ? "" : ", ") v[i] " " tolower(names[i])
        }
        line = s " of " total
    }
    END { print line }')
if [ "$status" = 0 ]; then
    state=success
    description="$summary in $minutes, $threads threads, $cpu"
else
    state=failure
    description="failed after $minutes ($summary), $threads threads, $cpu"
fi
echo "full-suite.sh: $state: $description"
[ -z "$result" ] || printf '%s\t%s\n' "$state" "$description" > "$result"
post_status "$state" "$description"
exit "$status"
