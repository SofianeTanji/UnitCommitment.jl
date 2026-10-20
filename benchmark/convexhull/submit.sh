#!/bin/bash
# One job per line of instances.txt: solves max_π L(π) with dw.jl and writes
# results/<name>.json. Submit from this directory (see README.md):
#
#   sbatch --array=1-$(wc -l < instances.txt) submit.sh
#
#SBATCH --job-name=chp-dw
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
name=$(sed -n "${SLURM_ARRAY_TASK_ID}p" instances.txt)
echo "$name on $(hostname), $SLURM_CPUS_PER_TASK threads"

# The column generation stops one hour before the walltime and saves its bounds
julia -t "$SLURM_CPUS_PER_TASK" --project=. dw.jl \
    "$name" instances results $((2 * 86400 - 3600))
