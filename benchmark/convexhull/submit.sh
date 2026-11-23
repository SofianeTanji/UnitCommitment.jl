#!/bin/bash
# One job per line of a list of instances (default instances.txt): solves
# max_π L(π) with dw.jl up to the relative gap <tol> (default 1e-6) and writes
# results/<name>.json. Submit from this directory (see README.md):
#
#   sbatch --array=1-$(wc -l < instances.txt) submit.sh
#   sbatch --array=1-$(wc -l < instances_test.txt) submit.sh instances_test.txt 1e-8
#
# A third argument is a directory of earlier results, used as warm starts.
# Gurobi comes from the module of the cluster, with one environment (license
# token) per thread: at most 7 jobs at a time for 60 tokens (--array=...%7).
# Sort the list first with order.jl, the longest instances first.
#
#SBATCH --job-name=chp-dw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1900M
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
list=${1:-instances.txt}
tol=${2:-1e-6}
warm=${3:-}
set +u # Lmod fails on unset variables
module load releases/2025b Gurobi/13.0.1-GCCcore-14.3.0
set -u
export GUROBI_JL_USE_GUROBI_JLL=false
name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$list")
echo "$name on $(hostname), $SLURM_CPUS_PER_TASK threads"

# The column generation stops one hour before the walltime and saves its bounds
julia -t "$SLURM_CPUS_PER_TASK" --project=. dw.jl \
    "$name" instances results $((2 * 86400 - 3600)) "$tol" $warm
