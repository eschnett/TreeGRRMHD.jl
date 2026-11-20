#!/bin/bash
# The full suite on one commit on a cluster, through SimLaunch (2026-11-18;
# it replaces `ci/full-suite-symmetry.sh`, which packed a fixed tree and wrote
# its own batch script). `jobs/full-suite.toml` is what runs: one thread, four
# threads and bounds checks, side by side on one node. This script takes the
# commit's snapshot and submits the spec, and later collects the result and,
# with `--post`, sets the commit's status, which SimLaunch never does.
#
#   ci/full-suite-cluster.sh submit [--site SITE] [--post] [REV]
#   ci/full-suite-cluster.sh collect [--site SITE] [--post] [REV]
#
# SITE is a SimLaunch site (`simlaunch sites`), `symmetry` by default; the
# status is `full-suite/<SITE>`. Each commit is a study of its own,
# `full-suite-<sha12>`, snapshotted from the commit (`--rev`), not from the
# working tree, and compiled once per site. `submit` snapshots it if it is new,
# prepares it on the site and submits; with `--post` it sets the status to
# pending. `collect` looks once (`simlaunch wait --once`): while the job is
# queued or running it says so and posts nothing; afterwards it fetches the
# runs to `simlaunch-data/<study>/<SITE>/`, prints each run's summary
# (`ci/suite-summary.sh`), and with `--post` sets the status to success, when
# every run finished with no failure, or to failure.
#
# Posting is public: ask each time (`CLAUDE.md`, "Commands"). The commit must
# be on GitHub for a post. Needs `simlaunch` on the PATH
# (`Pkg.Apps.develop(path = "<SimLaunch>")`, or `Pkg.Apps.add`).

set -euo pipefail

action="${1:-}"; shift || true
post=0
site=symmetry
rev="HEAD"
while [ $# -gt 0 ]; do
    case "$1" in
        --post) post=1; shift ;;
        --site) site="$2"; shift 2 ;;
        -*) echo "full-suite-cluster.sh: unknown option $1" >&2; exit 2 ;;
        *) rev="$1"; shift ;;
    esac
done

top=$(git rev-parse --show-toplevel)
sha=$(git rev-parse --verify "$rev^{commit}")
study="full-suite-${sha:0:12}"
spec="$top/jobs/full-suite.toml"
context="full-suite/$site"
cache="${SIMLAUNCH_CACHE:-$HOME/.cache/simlaunch}"
data="$top/simlaunch-data/$study"

post_status() {  # state description
    [ "$post" = 1 ] || return 0
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    gh api "repos/$repo/commits/$sha" --silent ||
        { echo "full-suite-cluster.sh: $sha is not on GitHub; push it first" >&2; exit 1; }
    gh api "repos/$repo/statuses/$sha" -f state="$1" -f context="$context" \
        -f description="${2:0:140}" --silent
    echo "full-suite-cluster.sh: posted $context = $1 on $sha"
}

case "$action" in
submit)
    if [ ! -d "$cache/$study" ]; then
        simlaunch snapshot "$top" --study "$study" --rev "$sha"
    fi
    # The commit's own spec, as the snapshot holds it.
    spec=$(ls "$cache/$study"/src/*/jobs/full-suite.toml)
    simlaunch prepare "$study" --site "$site" --spec "$spec"
    simlaunch submit "$study" "$spec" --site "$site"
    post_status pending "SimLaunch study $study on $site: 1 thread, 4 threads, bounds checks"
    ;;
collect)
    [ -d "$cache/$study" ] ||
        { echo "full-suite-cluster.sh: no study $study; submit it first" >&2; exit 1; }
    spec=$(ls "$cache/$study"/src/*/jobs/full-suite.toml)
    set +e
    simlaunch wait "$study" "$spec" --site "$site" --once
    waited=$?
    set -e
    case "$waited" in
        0) finished=1 ;;
        1) finished=0 ;;
        3) echo "full-suite-cluster.sh: $study is still queued or running on $site" >&2
           exit 1 ;;
        *) echo "full-suite-cluster.sh: simlaunch wait failed ($waited)" >&2
           exit "$waited" ;;
    esac
    simlaunch fetch "$study" --site "$site" --to "$data"
    # Each run's newest job directory.
    dirs=()
    for run in "$data/$site"/full-suite-*/; do
        newest=$(ls -d "$run"*/ 2>/dev/null | sort | tail -1)
        [ -n "$newest" ] && dirs+=("${newest%/}")
    done
    [ ${#dirs[@]} -gt 0 ] ||
        { echo "full-suite-cluster.sh: no runs fetched to $data/$site" >&2; exit 1; }
    summary=$(bash "$top/ci/suite-summary.sh" "${dirs[@]}")
    echo "$summary"
    cpu=$(sed -n 's/^cpu = "\(.*\)".*/\1/p' "${dirs[0]}/simlaunch-job.toml" | head -1)
    # The description: the counts once if every run has the same, else per run.
    counts=$(cut -f3 <<<"$summary" | sort -u)
    times=$(awk -F'\t' '{ printf "%s%s %s", (NR > 1 ? ", " : ""), $1, $4 }' <<<"$summary")
    if [ "$(wc -l <<<"$counts")" = 1 ]; then
        description="$counts ($times), $cpu"
    else
        description="$(awk -F'\t' '{ printf "%s%s: %s", (NR > 1 ? "; " : ""), $1, $3 }' <<<"$summary"), $cpu"
    fi
    if [ "$finished" = 1 ] && ! cut -f2 <<<"$summary" | grep -qv '^pass$'; then
        state=success
    else
        state=failure
        [ "$finished" = 1 ] || description="not every run finished; $description"
    fi
    echo "full-suite-cluster.sh: $state: $description"
    post_status "$state" "$description"
    ;;
*)
    sed -n '2,25p' "$0"
    exit 2
    ;;
esac
