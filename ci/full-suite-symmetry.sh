#!/bin/bash
# The full suite on one commit on Symmetry (added 2026-11-08, after
# TreeGeneralizedHarmonic's): x86-64 EPYC, where the last digits differ from
# the workstation's aarch64.
# `ci/full-suite.sh` is what runs; this submits it as a SLURM job and later
# posts its result, since the cluster has no `gh`.
#
#   ci/full-suite-symmetry.sh submit [--post] [--threads N] [REV]
#   ci/full-suite-symmetry.sh collect [--post] [REV]
#
# `submit` unpacks a clean `git archive` of the commit into one fixed
# directory in the home file system (one compile-cache slot, rather than one
# per commit; `CLAUDE.md`, "Commands") and submits one `amddebugq` job,
# refusing while another of its jobs is queued or running. Its log and its
# result go to `/mnt/beegfs/eschnetter/claude/full-suite/TreeGRRMHD/<sha>/`. With `--post` the commit's
# status `full-suite/symmetry` is set to pending. `collect` reads the result and
# prints it, and with `--post` sets the status to success or failure; before the
# job has finished it says so and posts nothing.
#
# The job's environment is what `CLAUDE.md` asks for there: no
# `JULIA_EXCLUSIVE`, and `amddebugq`, whose hour holds the suite's 10-17
# minutes.

set -euo pipefail

action="${1:-}"; shift || true
post=0
threads=4
rev="HEAD"
while [ $# -gt 0 ]; do
    case "$1" in
        --post) post=1; shift ;;
        --threads) threads="$2"; shift 2 ;;
        -*) echo "full-suite-symmetry.sh: unknown option $1" >&2; exit 2 ;;
        *) rev="$1"; shift ;;
    esac
done

host=symmetry
tree=/home/eschnetter/claude/full-suite/TreeGRRMHD
out_base=/mnt/beegfs/eschnetter/claude/full-suite/TreeGRRMHD
context=full-suite/symmetry
job_name=treegrrmhd-full-suite

sha=$(git rev-parse --verify "$rev^{commit}")
out="$out_base/$sha"

post_status() {  # state description
    [ "$post" = 1 ] || return 0
    repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
    gh api "repos/$repo/commits/$sha" --silent ||
        { echo "full-suite-symmetry.sh: $sha is not on GitHub; push it first" >&2; exit 1; }
    gh api "repos/$repo/statuses/$sha" -f state="$1" -f context="$context" \
        -f description="${2:0:140}" --silent
    echo "full-suite-symmetry.sh: posted $context = $1 on $sha"
}

case "$action" in
submit)
    if ssh -n "$host" "bash -lc 'squeue -h -u \$USER -n $job_name'" | grep -q .; then
        echo "full-suite-symmetry.sh: a $job_name job is queued or running;" \
             "its tree is $tree, so wait for it" >&2
        exit 1
    fi
    local_tree=$(mktemp -d "${TMPDIR:-/tmp}/treegrrmhd-full-suite.XXXXXX")
    trap 'rm -rf "$local_tree"' EXIT
    git archive "$sha" | tar -x -C "$local_tree"
    ssh -n "$host" "mkdir -p $tree $out"
    rsync -a --delete --exclude 'Manifest*.toml' "$local_tree/" "$host:$tree/"
    ssh "$host" "cat > $out/job.sh" <<EOF
#!/bin/bash
#SBATCH --partition=amddebugq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=$job_name
#SBATCH --output=$out/job-%j.out

export PATH="\$HOME/.juliaup/bin:\$PATH"
echo "commit $sha on \$(hostname), \$(lscpu | sed -n 's/^Model name: *//p' | head -1), \$(date)"
julia --project=$tree -e 'using Pkg; Pkg.Registry.update()'
bash $tree/ci/full-suite.sh --tree $tree --sha $sha --threads $threads \\
    --context $context --result $out/result.txt
EOF
    job=$(ssh -n "$host" "bash -lc 'sbatch --parsable $out/job.sh'")
    echo "full-suite-symmetry.sh: job $job, log $out/job-$job.out"
    post_status pending "SLURM job $job on Symmetry, $threads threads"
    ;;
collect)
    if ! result=$(ssh -n "$host" "cat $out/result.txt" 2>/dev/null); then
        echo "full-suite-symmetry.sh: no result for $sha yet;" \
             "its job is $(ssh -n "$host" "bash -lc 'squeue -h -u \$USER -n $job_name -o \"%i %T %M\"'" || true)" >&2
        exit 1
    fi
    state=${result%%$'\t'*}
    description=${result#*$'\t'}
    echo "full-suite-symmetry.sh: $state: $description"
    post_status "$state" "$description"
    ;;
*)
    sed -n '2,22p' "$0"
    exit 2
    ;;
esac
