# Convex hull pricing: optimal dual values

Computes `max_π L(π)` (the convex hull pricing dual of
`UnitCommitment.ConvexHullOracle`) for the UnitCommitment.jl benchmark
instances listed in `instances.txt`, by Dantzig–Wolfe column generation with
Wentges smoothing and parallel pricing (`dw.jl`), up to a relative gap of 1e-6.

Not included: RTS-GMLC and pglib-uc/ca/Scenario400 (time-varying minimum
power, not supported by the default thermal formulation), transmission
expansion instances, MATPOWER case300, case89pegase and larger cases (too
large with eager contingency constraints), and the MATPOWER instances on which
HiGHS returned OTHER_ERROR (case30 and 46 days of case14). MATPOWER cases are
run for 2017-01-01 only, except case14 (319 days, at the end of
`instances.txt`).

## Running on a CECI cluster (NIC6)

On a login node (compute nodes may have no internet access):

```bash
# Julia, through juliaup (installed in ~/.juliaup, added to the PATH by ~/.bashrc)
curl -fsSL https://install.julialang.org | sh -s -- -y
source ~/.bashrc
juliaup add 1.12 && juliaup default 1.12

git clone -b convex-hull-lmp-benchmark https://github.com/SofianeTanji/UnitCommitment.jl.git
cd UnitCommitment.jl/benchmark/convexhull
./download.sh                                            # instances/
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
mkdir -p logs
sbatch --array=1-$(wc -l < instances.txt) submit.sh
```

Each job uses 8 cores, 1900 MB per core (the NIC6 maximum) and at most 2 days. When all jobs are done:

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

## Benchmark of the methods

`methods/` holds the methods that maximize `L` (subgradient, bundle, smoothing
and column generation), with the interface described in `methods/common.jl`.
`configs.jl` lists the 70 benchmarked configurations, each with at most one
tuned parameter, and the tuning instances: one per dataset (or-lib,
pglib-uc/ca and pglib-uc/ferc), the one with the median gap
`(L* - L(π0)) / |L*|`. The other instances on which `L(π0)` is not within
1e-6 of `L*` (79) are the evaluation instances. Every run starts from the LP
relaxation duals and lasts 300 s, or until `L` is within 1e-6 of `L*`.

The tuning: each tuned configuration runs on each tuning instance for an
11-value coarse grid of its parameter, then for 4 more values around the best
one. Then the evaluation runs with the selected values. After the `results/`
of `submit.sh` are in place:

```bash
mkdir -p logs
julia --project=. plan.jl coarse        # tasks/coarse.txt (268 jobs)
sbatch --array=1-$(wc -l < tasks/coarse.txt) bench.sh tasks/coarse.txt traces/tune
# when they are done:
julia --project=. plan.jl fine          # tasks/fine.txt
sbatch --array=1-$(wc -l < tasks/fine.txt) bench.sh tasks/fine.txt traces/tune
# when they are done:
julia --project=. plan.jl select        # tuned.json, and prints the selected values
julia --project=. plan.jl eval          # tasks/eval.txt (822 jobs of 12 runs)
sbatch --array=1-$(wc -l < tasks/eval.txt) bench.sh tasks/eval.txt traces/eval
# column-and-row generation, added after the evaluation (79 jobs of 1 run):
sbatch --array=1-$(wc -l < tasks/crg.txt) bench.sh tasks/crg.txt traces/eval
```

Each job uses 8 cores and at most 4 hours. The traces are in
`traces/<stage>/<run>/<name>.json`: the parameters, the status (`target`,
`time_limit`, `optimal`, `unbounded`, `iterations` or `error`), and the time,
`L`, the certified bound and the method's upper bound at each evaluation.
