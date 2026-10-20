# Convex hull pricing: optimal dual values

Computes `max_π L(π)` (the convex hull pricing dual of
`UnitCommitment.ConvexHullOracle`) for the UnitCommitment.jl benchmark
instances listed in `instances.txt`, by Dantzig–Wolfe column generation with
Wentges smoothing and parallel pricing (`dw.jl`), up to a relative gap of 1e-6.

Not included: RTS-GMLC (time-varying minimum power, not supported by the
default thermal formulation), transmission expansion instances, and MATPOWER
cases above case300 (too large with eager contingency constraints).

## Running on a CECI cluster (NIC6)

On a login node (compute nodes may have no internet access):

```bash
curl -fsSL https://install.julialang.org | sh -s -- -y   # juliaup
juliaup add 1.12 && juliaup default 1.12
git clone -b <branch> <repository> && cd UnitCommitment.jl/benchmark/convexhull
./download.sh                                            # instances/
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
mkdir -p logs
sbatch --array=1-$(wc -l < instances.txt) submit.sh
```

Each job uses 8 cores, 32 GB and at most 2 days. When all jobs are done:

```bash
julia --project=. collect.jl
```

## Output

- `results/<name>.json`, one per instance: status (`optimal`, `time_limit` or
  `error`), lower and upper bounds on `max_π L(π)`, relative gap, LP
  relaxation value, `L` at the LP relaxation duals, iterations, times, the
  coupling row keys, the LP duals, the convex hull prices (the prices of the
  best lower bound) and the bounds at each iteration.
- `results.csv`: one row per instance with the scalar fields.
- `logs/<job>_<index>.out`: Slurm logs.
