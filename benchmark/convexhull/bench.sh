#!/bin/bash
# One job per line of a task file of plan.jl: runs its configurations on its
# instance with run.jl, for 300 s each, and writes the traces to the output
# directory. Submit from this directory (see README.md), e.g.
#
#   sbatch --array=1-$(wc -l < tasks/coarse.txt) bench.sh tasks/coarse.txt traces/tune
#
#SBATCH --job-name=chp-bench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=04:00:00
#SBATCH --output=logs/bench_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
tasks=$1
out=$2
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$tasks")
echo "$line"
echo "on $(hostname), $SLURM_CPUS_PER_TASK threads"

# The line is the instance name followed by the runs
# shellcheck disable=SC2086
set -- $line
name=$1
shift
julia -t "$SLURM_CPUS_PER_TASK" --project=. run.jl "$out" "$name" 300 "$@"
